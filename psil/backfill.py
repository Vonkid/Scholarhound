"""Historical focused-journal backfill for ScholarHound.

Backfill is intentionally staged:
1. harvest metadata from Crossref over a historical window
2. prefilter with the current concept/kernel map
3. optionally score and store candidates with the current ranking standard
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import requests

from psil.ingest.crossref import CROSSREF_BASE, USER_AGENT, parse_crossref_work
from psil.ingest.toc import enrich_paper_toc_image
from psil.kernel import (
    apply_paper_type_router,
    check_domain_consistency,
    classify_signal_strength,
    estimate_evidence_strength,
    kernel_classify_paper,
)
from psil.rank.concepts import get_matched_concepts, is_non_research
from psil.rank.scorer import prefilter_papers
from psil.store.models import Paper


FOCUSED_JOURNAL_NAMES = {
    "nature biomedical engineering",
    "nature electronics",
    "nature nanotechnology",
    "nature materials",
    "nature photonics",
    "nature biotechnology",
    "advanced materials",
    "advanced functional materials",
    "acs nano",
    "nano letters",
    "acs sensors",
    "analytical chemistry",
    "jacs",
    "angewandte chemie",
}

DEFAULT_BACKFILL_START = date(2020, 1, 1)
RATE_DELAY = 0.15


@dataclass
class BackfillJournalResult:
    journal: str
    fetched: int = 0
    new: int = 0
    passed: int = 0
    ignored: int = 0
    stored: int = 0
    llm_used: int = 0
    by_tier: dict[str, int] = field(default_factory=dict)


@dataclass
class BackfillResult:
    start_date: date
    end_date: date
    journals: list[BackfillJournalResult] = field(default_factory=list)

    @property
    def fetched(self) -> int:
        return sum(j.fetched for j in self.journals)

    @property
    def new(self) -> int:
        return sum(j.new for j in self.journals)

    @property
    def passed(self) -> int:
        return sum(j.passed for j in self.journals)

    @property
    def stored(self) -> int:
        return sum(j.stored for j in self.journals)

    @property
    def llm_used(self) -> int:
        return sum(j.llm_used for j in self.journals)


def default_backfill_window(years: int, today: date | None = None) -> tuple[date, date]:
    today = today or date.today()
    try:
        start = today.replace(year=today.year - years)
    except ValueError:
        start = today.replace(year=today.year - years, day=28)
    return start, today


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def iter_date_chunks(start: date, end: date, granularity: str = "year"):
    """Yield inclusive date chunks for Crossref range queries."""
    if start > end:
        raise ValueError("start date must be before end date")
    current = start
    while current <= end:
        if granularity == "month":
            if current.month == 12:
                next_start = date(current.year + 1, 1, 1)
            else:
                next_start = date(current.year, current.month + 1, 1)
        else:
            next_start = date(current.year + 1, 1, 1)
        chunk_end = min(end, next_start - timedelta(days=1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def select_backfill_journals(
    journals: list[dict],
    focused: bool = True,
    names: tuple[str, ...] | list[str] = (),
) -> list[dict]:
    wanted = {n.lower().strip() for n in names if n.strip()}
    selected = []
    for journal in journals:
        name = journal.get("name", "")
        key = name.lower().strip()
        if wanted and key not in wanted:
            continue
        if focused and not wanted and key not in FOCUSED_JOURNAL_NAMES:
            continue
        if not journal.get("issn"):
            continue
        selected.append(journal)
    return selected


def fetch_crossref_range(
    issn: str,
    journal_name: str,
    start: date,
    end: date,
    rows: int = 200,
    max_pages: int = 50,
) -> list[Paper]:
    papers: list[Paper] = []
    seen: set[str] = set()
    offset = 0
    for _ in range(max_pages):
        url = f"{CROSSREF_BASE}/{issn}/works"
        params = {
            "filter": f"from-pub-date:{start.isoformat()},until-pub-date:{end.isoformat()}",
            "rows": rows,
            "offset": offset,
        }
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        msg = resp.json().get("message", {})
        items = msg.get("items", [])
        if not items:
            break
        for item in items:
            paper = parse_crossref_work(item, journal_name=journal_name)
            if paper.doi and paper.doi not in seen:
                seen.add(paper.doi)
                papers.append(paper)
        offset += rows
        if offset >= msg.get("total-results", 0):
            break
        time.sleep(RATE_DELAY)
    return papers


def harvest_journal(
    journal: dict,
    start: date,
    end: date,
    chunk: str = "year",
    max_papers: int = 0,
) -> list[Paper]:
    papers: list[Paper] = []
    seen: set[str] = set()
    for chunk_start, chunk_end in iter_date_chunks(start, end, granularity=chunk):
        batch = fetch_crossref_range(
            journal["issn"],
            journal.get("name", ""),
            chunk_start,
            chunk_end,
        )
        for paper in batch:
            if paper.doi in seen:
                continue
            seen.add(paper.doi)
            papers.append(paper)
            if max_papers and len(papers) >= max_papers:
                return papers
    return papers


def summarize_candidates(
    papers: list[Paper],
    db,
    threshold: int = 0,
) -> tuple[list[tuple[Paper, int]], list[Paper], int]:
    new_papers = [p for p in papers if p.doi and not db.doi_exists(p.doi)]
    passed, ignored = prefilter_papers(new_papers, threshold=threshold)
    return passed, ignored, len(new_papers)


def _kernel_concept_support(reasoning: dict, concept_name: str, db) -> int:
    raw = reasoning.get("concept_support", 0)
    evidence = (reasoning.get("evidence_strength", "") or "").strip()
    support_type = (reasoning.get("support_type", "") or "").strip()
    cs_name = (reasoning.get("concept_support_name", "") or "").strip()
    try:
        score = float(raw)
    except (TypeError, ValueError):
        score = 0.0
    if "High" in evidence:
        score = min(10, score + 1)
    elif "Low" in evidence:
        score = max(0, score - 2)
    if "Discovery" in support_type:
        score = min(10, score + 1)
    elif "Validation" in support_type:
        score = min(10, score + 0.5)
    elif "Weak Signal" in support_type:
        score = max(0, score - 1)
    if cs_name and cs_name.lower() != "none":
        existing = db.get_concept(cs_name)
        if existing:
            appearances = existing.get("appearances", 0)
            if appearances >= 5:
                score = min(10, score + 2)
            elif appearances >= 3:
                score = min(10, score + 1)
    return round(max(0, min(10, score)))


def score_and_store_candidate(
    paper: Paper,
    score: int,
    db,
    identity,
    llm=None,
    use_llm: bool = False,
    validate: bool = False,
) -> tuple[str, bool]:
    """Score a candidate and insert it into the main papers DB.

    Returns (tier, used_llm).
    """
    if is_non_research(paper.title, paper.abstract):
        return "IGNORE", False

    combined_text = f"{paper.title} {paper.abstract}"
    matched = get_matched_concepts(combined_text)
    signal_names = [name for name, _ in matched]
    dispatch = classify_signal_strength(matched)
    used_llm = False

    if use_llm and llm and dispatch["tier"] != "LOW":
        try:
            reasoning = llm.rank(
                paper,
                identity,
                matched_signals=", ".join(signal_names) if signal_names else "",
            )
            used_llm = True
        except Exception:
            reasoning = kernel_classify_paper(paper, matched)
    else:
        reasoning = kernel_classify_paper(paper, matched)

    tier = reasoning.get("signal_tier", "LOW_PRIORITY").strip().upper()
    concept_name = reasoning.get("concept_name", "").strip()

    raw_cs = reasoning.get("concept_support", 0)
    kernel_cs = _kernel_concept_support(reasoning, concept_name, db)
    domain_check = check_domain_consistency(
        reasoning.get("problem_class", ""),
        reasoning.get("concept_support_name", ""),
        reasoning.get("strategic_value", ""),
    )
    if not domain_check["consistent"]:
        kernel_cs = max(0, kernel_cs - domain_check["confidence_penalty"])
        reasoning["kernel_domain_flag"] = domain_check["flag"]

    evidence_est = estimate_evidence_strength(
        paper.journal,
        paper.abstract or "",
        reasoning.get("problem_class", ""),
        reasoning.get("novelty_type", ""),
    )
    reasoning["kernel_evidence"] = evidence_est
    if (reasoning.get("evidence_strength", "") or "").strip() == "High" and evidence_est.get("kernel_evidence_strength") == "Low":
        kernel_cs = max(0, kernel_cs - 2)
        reasoning["kernel_evidence_flag"] = "LLM overestimated evidence strength"

    reasoning["concept_support"] = kernel_cs
    reasoning["concept_support_raw"] = raw_cs
    apply_paper_type_router(reasoning, paper.title, paper.abstract or "")

    if validate and used_llm and ("IMPORTANT" in tier or "HIGH" in tier):
        try:
            reasoning["validation"] = llm.validate(paper, reasoning)
        except Exception:
            pass

    reasoning["backfill"] = {
        "source": "historical-focused-journal-backfill",
        "pub_date": paper.pub_date,
        "dispatch": dispatch,
        "matched_signals": signal_names,
    }

    if concept_name:
        traj = reasoning.get("trajectory_influence", 0)
        tw = "high" if traj >= 7 else ("medium" if traj >= 4 else "low")
        db.upsert_concept(
            name=concept_name,
            source_doi=paper.doi,
            why_matters=reasoning.get("concept_why_matters", "").strip(),
            connection=reasoning.get("concept_current_connection", "").strip(),
            missing_link=reasoning.get("concept_missing_link", "").strip(),
            opportunity=reasoning.get("concept_opportunity", "").strip(),
            trajectory_weight=tw,
            seen_date=paper.pub_date,
        )
        if identity:
            identity.update_concept_momentum(concept_name, True)

    csn = reasoning.get("concept_support_name", "").strip()
    if csn and csn.lower() != "none":
        db.insert_justification(
            concept_name=csn,
            paper_doi=paper.doi,
            support_type=reasoning.get("support_type", ""),
            evidence_strength=reasoning.get("evidence_strength", ""),
            justification_text=reasoning.get("why_matters", ""),
        )

    enrich_paper_toc_image(paper)
    db.insert_paper(
        paper,
        signal_score=score,
        signal_tier=tier,
        signal_trajectory=reasoning.get("trajectory_influence", 0),
        signal_action=reasoning.get("action", ""),
        llm_reasoning=json.dumps(reasoning),
        concept_name=concept_name,
        concept_drift=concept_name,
        causal=reasoning.get("causal"),
        problem_class=reasoning.get("problem_class", ""),
        novelty_type=reasoning.get("novelty_type", ""),
        evidence_type=reasoning.get("evidence_type", ""),
        strategic_value=reasoning.get("strategic_value", ""),
        concept_support_name=reasoning.get("concept_support_name", ""),
        support_type=reasoning.get("support_type", ""),
        evidence_strength=reasoning.get("evidence_strength", ""),
        concept_support_score=reasoning.get("concept_support", 0),
    )
    return tier, used_llm
