"""V3 evidence-intake ablation benchmark.

This module measures whether the V3 kernel reduces unstable direct commits from
ranked paper analysis. It writes only to a temporary kernel unless the caller
explicitly passes a kernel_dir.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any

from psil.v3_kernel import (
    EVIDENCE_DELTAS,
    create_belief,
    create_evidence,
    create_evidence_from_parse_candidates,
    export_kernel_state,
    get_contested_evidence_queue,
    get_pending_evidence_queue,
    read_jsonl,
    revise_belief,
    validate_v3_kernel,
)


DEFAULT_TIER_LIMITS = [
    ("HIGH_PRIORITY", 1),
    ("IMPORTANT", 10),
    ("POTENTIAL", 5),
    ("WATCHLIST", 4),
]
DEFAULT_LEGACY_BACKFILL_TIERS = [
    "HIGH_PRIORITY",
    "IMPORTANT",
    "POTENTIAL",
    "WATCHLIST",
    "LOW_PRIORITY",
    "COMMENTARY",
    "IGNORE",
]
DEFAULT_DB_PATH = Path.home() / ".psil" / "psil.db"
INITIAL_CONFIDENCE = 0.5
INITIAL_ENTRENCHMENT = 0.1

BACKFILL_BELIEF_TEMPLATES = {
    "transduction_validity": {
        "title": "Sensing papers should preserve biological meaning through transduction",
        "claim": (
            "A sensing or diagnostic paper supports ScholarHound only when molecular "
            "recognition, biological state, and device/readout state remain traceably coupled."
        ),
        "domain": "legacy-backfill-sensing",
    },
    "mechanism_to_coupling": {
        "title": "Mechanism papers should supply reusable coupling principles",
        "claim": (
            "A mechanism paper supports ScholarHound when it clarifies a reusable physical, "
            "chemical, or biological coupling principle that can later constrain experiments."
        ),
        "domain": "legacy-backfill-mechanism",
    },
    "platform_bridge": {
        "title": "Platform papers should bridge method capability to research questions",
        "claim": (
            "A platform or method paper supports ScholarHound when the workflow can reduce "
            "uncertainty in an active research question rather than merely add instrumentation."
        ),
        "domain": "legacy-backfill-platform",
    },
    "biological_relevance": {
        "title": "Biology papers should sharpen disease-state or living-system models",
        "claim": (
            "A biology-mechanism paper supports ScholarHound when it sharpens a disease-state, "
            "cell-state, or living-system model that a future readout must preserve."
        ),
        "domain": "legacy-backfill-biology",
    },
    "synthesis_prior": {
        "title": "Synthesis papers should alter priors without pretending to be direct evidence",
        "claim": (
            "A review or synthesis supports ScholarHound when it changes search priors, "
            "constraints, or missing-link definitions while staying separate from primary evidence."
        ),
        "domain": "legacy-backfill-synthesis",
    },
    "trajectory_fit": {
        "title": "General papers should clarify research trajectory fit",
        "claim": (
            "A general research paper supports ScholarHound when it clarifies whether an idea "
            "belongs in the active trajectory, pending queue, or rejection path."
        ),
        "domain": "legacy-backfill-general",
    },
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "?"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _reasoning(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(row.get("llm_reasoning") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def _final_score(row: dict[str, Any], reasoning: dict[str, Any]) -> float:
    return _number(
        reasoning.get("final_score"),
        _number(reasoning.get("final_score_llm"), 0.0),
    )


def _classify_type(row: dict[str, Any], reasoning: dict[str, Any]) -> tuple[str, str]:
    raw = " ".join(
        str(value or "")
        for value in [
            reasoning.get("paper_type"),
            reasoning.get("judgment_mode"),
            row.get("problem_class"),
            reasoning.get("problem_class"),
            row.get("novelty_type"),
            reasoning.get("novelty_type"),
            row.get("title"),
            (row.get("abstract") or "")[:500],
        ]
    ).lower()
    if "review" in raw or "perspective" in raw:
        return "synthesis_or_review", "synthesis_prior"
    if any(
        word in raw
        for word in ["sensor", "sensing", "biosensor", "detection", "diagnostic"]
    ) or any(word in raw for word in ["assay", "readout", "phenotyping"]):
        return "sensing_or_diagnostic", "transduction_validity"
    if any(
        word in raw
        for word in ["photophysics", "photochem", "polariton", "mechanism", "fundamental"]
    ):
        return "mechanism_paper", "mechanism_to_coupling"
    if any(word in raw for word in ["platform", "method", "workflow", "engineering"]):
        return "platform_or_method", "platform_bridge"
    if any(word in raw for word in ["biological", "immune", "tissue", "disease", "cell"]):
        return "biology_mechanism", "biological_relevance"
    return "general_research", "trajectory_fit"


def _content_fields(row: dict[str, Any], reasoning: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(row.get("title") or ""),
        "abstract": str(row.get("abstract") or ""),
        "why": str(reasoning.get("why_matters") or reasoning.get("why") or ""),
        "connection": str(
            reasoning.get("connection")
            or reasoning.get("potential_connection")
            or reasoning.get("connections")
            or ""
        ),
        "weakness": str(reasoning.get("weakness") or reasoning.get("gap") or ""),
        "problem_class": str(row.get("problem_class") or reasoning.get("problem_class") or ""),
        "novelty_type": str(row.get("novelty_type") or reasoning.get("novelty_type") or ""),
    }


def _content_text(row: dict[str, Any], reasoning: dict[str, Any]) -> str:
    return " ".join(_content_fields(row, reasoning).values()).lower()


def _evidence_text(row: dict[str, Any], reasoning: dict[str, Any]) -> str:
    fields = _content_fields(row, reasoning)
    return " ".join(
        [
            fields["title"],
            fields["abstract"],
            fields["why"],
            fields["problem_class"],
            fields["novelty_type"],
        ]
    ).lower()


def _primary_evidence_text(row: dict[str, Any], reasoning: dict[str, Any]) -> str:
    fields = _content_fields(row, reasoning)
    return " ".join(
        [
            fields["title"],
            fields["abstract"],
            fields["problem_class"],
            fields["novelty_type"],
        ]
    ).lower()


def _has_term(text: str, term: str) -> bool:
    term = term.lower()
    if re.fullmatch(r"[a-z0-9]{1,4}", term):
        return bool(re.search(rf"\b{re.escape(term)}\b", text))
    return term in text


def _count_terms(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if _has_term(text, term))


SENSING_READOUT_TERMS = [
    "assay",
    "biosensor",
    "bioelectronic",
    "capacitance",
    "diagnostic",
    "detection",
    "electrochemical",
    "electronic signal",
    "impedance",
    "oect",
    "organic electrochemical",
    "readout",
    "sensor",
    "sensing",
    "transduc",
    "transistor",
]

BIOLOGICAL_MEANING_TERMS = [
    "activity-affinity",
    "activity based",
    "activity-based",
    "biological state",
    "bacteria",
    "cell state",
    "disease",
    "ev",
    "evs",
    "extracellular vesicle",
    "functional readout",
    "living",
    "metastatic",
    "metabolism",
    "molecular recognition",
    "organoid",
    "plasma",
    "serum",
    "untreated plasma",
    "whole-cell",
]

BIOLOGICAL_PRESERVATION_TERMS = [
    "complex biological",
    "direct living",
    "directly from",
    "functional",
    "label-free",
    "matrix interference",
    "preserve",
    "real-time",
    "traceably",
    "untreated",
    "without complex purification",
]

MECHANISM_TERMS = [
    "charge transfer",
    "coupling",
    "dispersion",
    "field",
    "mechanism",
    "mechanotransduction",
    "photochem",
    "photophysics",
    "phonon",
    "polariton",
    "reusable",
    "state",
    "topological",
    "transduction mechanism",
]

PLATFORM_TERMS = [
    "assay",
    "method",
    "platform",
    "workflow",
    "single-cell",
    "screening",
    "microfluidic",
    "integration",
]

ACTIVE_RESEARCH_TERMS = [
    "bioelectronic",
    "ev",
    "evs",
    "extracellular vesicle",
    "molecular",
    "nanophotonic",
    "oect",
    "organoid",
    "photochem",
    "sensor",
    "sensing",
    "transduction",
]

STRONG_RESEARCH_ANCHORS = [
    "activity-affinity",
    "bioelectronic",
    "bodipy",
    "direct living transducer",
    "ev",
    "evs",
    "extracellular vesicle",
    "functional ev",
    "nanophotonic",
    "oect",
    "organoid",
    "organic electrochemical",
    "photocleavage",
    "photochem",
    "polariton",
    "untreated plasma",
]

CONNECTION_SUPPORT_TERMS = [
    "aligns with",
    "could be coupled",
    "could be integrated",
    "could inform",
    "directly relevant",
    "directly connects",
    "directly implementable",
    "extends",
    "future oect",
    "integrated into",
    "relevant to your",
    "supports",
]

OFF_TOPIC_CUES = [
    "far outside",
    "fundamentally different",
    "not aligned",
    "orthogonal",
    "weak connection",
    "weak relevance",
    "no direct connection",
    "no direct relevance",
    "not directly relevant",
]


def _has_off_topic_cue(text: str) -> bool:
    return any(cue in text for cue in OFF_TOPIC_CUES)


def _relation_from_belief_content(row: dict[str, Any], reasoning: dict[str, Any]) -> str:
    _paper_type, judgment_mode = _classify_type(row, reasoning)
    text = _content_text(row, reasoning)
    evidence_text = _evidence_text(row, reasoning)
    strong_anchor = _count_terms(_primary_evidence_text(row, reasoning), STRONG_RESEARCH_ANCHORS)

    if _relation_from_explicit_conflict(row, reasoning) == "challenge":
        return "challenge"

    if judgment_mode == "transduction_validity":
        readout = _count_terms(evidence_text, SENSING_READOUT_TERMS)
        biology = _count_terms(evidence_text, BIOLOGICAL_MEANING_TERMS)
        preservation = _count_terms(evidence_text, BIOLOGICAL_PRESERVATION_TERMS)
        if _has_off_topic_cue(text) and strong_anchor < 2:
            return "neutral"
        if any(
            phrase in text
            for phrase in [
                "does not specify the biological state",
                "does not specify biological state",
                "without a specific biological readout",
            ]
        ):
            return "unclear" if readout >= 1 else "neutral"
        if (
            readout >= 2
            and biology >= 1
            and strong_anchor >= 1
            and (preservation >= 1 or "oect" in text or "ev" in text)
        ):
            return "support"
        if readout >= 1 and biology >= 1 and strong_anchor >= 1:
            return "unclear"
        return "neutral"

    if judgment_mode == "mechanism_to_coupling":
        mechanism = _count_terms(evidence_text, MECHANISM_TERMS)
        if _has_off_topic_cue(text) and strong_anchor < 1:
            return "neutral"
        if mechanism >= 2 and strong_anchor >= 1:
            return "support"
        if mechanism >= 1 and strong_anchor >= 1:
            return "unclear"
        return "neutral"

    if judgment_mode == "platform_bridge":
        platform = _count_terms(evidence_text, PLATFORM_TERMS)
        if _has_off_topic_cue(text) and strong_anchor < 1:
            return "neutral"
        if platform >= 1 and strong_anchor >= 1:
            return "support"
        if platform >= 1 and _count_terms(evidence_text, ACTIVE_RESEARCH_TERMS) >= 1:
            return "unclear"
        return "neutral"

    if judgment_mode == "biological_relevance":
        biology = _count_terms(evidence_text, BIOLOGICAL_MEANING_TERMS)
        state_terms = _count_terms(
            evidence_text,
            ["disease state", "cell state", "model", "mechanism", "immune", "tissue"],
        )
        if biology >= 1 and state_terms >= 1:
            return "support"
        if biology >= 1:
            return "unclear"
        return "neutral"

    if judgment_mode == "synthesis_prior":
        if _has_off_topic_cue(text) and strong_anchor < 1:
            return "neutral"
        if (
            _count_terms(evidence_text, ["review", "perspective", "prior", "constraint", "gap"]) >= 2
            and strong_anchor >= 1
        ):
            return "support"
        if ("review" in evidence_text or "perspective" in evidence_text) and strong_anchor >= 1:
            return "unclear"
        return "neutral"

    if _has_off_topic_cue(text) and strong_anchor < 1:
        return "neutral"
    if strong_anchor >= 2:
        return "support"
    if strong_anchor == 1:
        return "unclear"
    return "neutral"


def _relation_from_mechanism_trace(row: dict[str, Any], reasoning: dict[str, Any]) -> str:
    _paper_type, judgment_mode = _classify_type(row, reasoning)
    text = _content_text(row, reasoning)
    evidence_text = _evidence_text(row, reasoning)
    strong_anchor = _count_terms(_primary_evidence_text(row, reasoning), STRONG_RESEARCH_ANCHORS)

    if _relation_from_explicit_conflict(row, reasoning) == "challenge":
        return "challenge"

    if judgment_mode == "transduction_validity":
        readout = _count_terms(evidence_text, SENSING_READOUT_TERMS)
        biology = _count_terms(evidence_text, BIOLOGICAL_MEANING_TERMS)
        mechanism = _count_terms(
            evidence_text,
            [
                "activity",
                "affinity",
                "capacitance",
                "electronic signal",
                "enzymatic product",
                "functional readout",
                "metabolism",
                "modulates",
                "redox",
                "transduced",
                "transduction",
            ],
        )
        if _has_off_topic_cue(text) and strong_anchor < 2:
            return "neutral"
        if readout >= 1 and biology >= 1 and mechanism >= 1 and strong_anchor >= 1:
            return "support"
        if readout >= 1 and mechanism >= 1 and strong_anchor >= 1:
            return "unclear"
        return "neutral"

    if judgment_mode == "mechanism_to_coupling":
        if _count_terms(evidence_text, MECHANISM_TERMS) >= 2 and strong_anchor >= 1:
            return "support"
        if _count_terms(evidence_text, MECHANISM_TERMS) == 1 and strong_anchor >= 1:
            return "unclear"
        return "neutral"

    if judgment_mode == "platform_bridge":
        if (
            _count_terms(evidence_text, PLATFORM_TERMS) >= 1
            and _count_terms(text, CONNECTION_SUPPORT_TERMS) >= 1
            and strong_anchor >= 1
        ):
            return "support"
        if _count_terms(evidence_text, PLATFORM_TERMS) >= 1 and strong_anchor >= 1:
            return "unclear"
        return "neutral"

    if _count_terms(evidence_text, MECHANISM_TERMS) >= 2 and strong_anchor >= 1:
        return "support"
    if _count_terms(evidence_text, MECHANISM_TERMS) == 1 and strong_anchor >= 1:
        return "unclear"
    return "neutral"


def _relation_from_digest_connection(row: dict[str, Any], reasoning: dict[str, Any]) -> str:
    fields = _content_fields(row, reasoning)
    connection_text = " ".join([fields["why"], fields["connection"]]).lower()
    all_text = _content_text(row, reasoning)

    if _relation_from_explicit_conflict(row, reasoning) == "challenge":
        return "challenge"
    if not connection_text.strip():
        return "neutral"

    support_terms = _count_terms(connection_text, CONNECTION_SUPPORT_TERMS)
    evidence_anchor = _count_terms(_primary_evidence_text(row, reasoning), STRONG_RESEARCH_ANCHORS)
    connection_anchor = _count_terms(connection_text, STRONG_RESEARCH_ANCHORS)
    if _has_off_topic_cue(connection_text):
        return "neutral"
    if support_terms >= 1 and evidence_anchor >= 1 and connection_anchor >= 1:
        return "support"
    if support_terms >= 1:
        return "unclear"
    return "neutral"


def _relation_from_explicit_conflict(row: dict[str, Any], reasoning: dict[str, Any]) -> str:
    haystack = " ".join(
        str(value or "")
        for value in [
            row.get("title"),
            row.get("abstract"),
            reasoning.get("why_matters"),
            reasoning.get("connection"),
            reasoning.get("potential_connection"),
            reasoning.get("weakness"),
            reasoning.get("gap"),
            reasoning.get("evidence"),
        ]
    ).lower()
    challenge_patterns = [
        "contradicts the belief",
        "challenges the belief",
        "undermines the belief",
        "undermines the premise",
        "evidence against",
        "fails to preserve biological meaning",
        "fails to preserve biological context",
        "cannot distinguish signal from noise",
        "does not distinguish signal from noise",
    ]
    if any(pattern in haystack for pattern in challenge_patterns):
        return "challenge"
    return "neutral"


def _reports_lod(row: dict[str, Any], reasoning: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(value or "")
        for value in [
            row.get("title"),
            row.get("abstract"),
            reasoning.get("why_matters"),
            reasoning.get("potential_connection"),
            reasoning.get("weakness"),
            reasoning.get("evidence"),
        ]
    ).lower()
    return bool(
        re.search(
            r"\b(lod|limit of detection|sensitivity|attomolar|femtomolar|particles/ml|copies/ml)\b",
            haystack,
        )
    )


def _short(text: str, limit: int = 92) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "..."


def build_parse_candidates(row: dict[str, Any]) -> list[dict[str, str]]:
    reasoning = _reasoning(row)
    # UPGRADE (env-gated): when SCHOLARHOUND_RELATION_READER=llm and a key is present,
    # parse-candidates come from multi-model LLM reads of (belief + abstract) instead of the
    # keyword heuristic, which structurally cannot detect contradiction. Any failure or a
    # missing key falls through to the offline keyword candidates below, so offline tests are
    # unaffected.
    try:
        from psil.relation_llm import llm_parse_candidates, enabled as _llm_enabled

        if _llm_enabled():
            _paper_type, judgment_mode = _classify_type(row, reasoning)
            belief_text = _backfill_belief_template(judgment_mode).get("claim", "")
            abstract_text = "\n".join(
                part
                for part in [
                    str(row.get("title") or ""),
                    str(row.get("abstract") or ""),
                    _primary_evidence_text(row, reasoning),
                ]
                if part
            )
            llm_candidates = llm_parse_candidates(belief_text, abstract_text)
            if llm_candidates:
                return llm_candidates
    except Exception:
        pass  # offline keyword fallback below

    return [
        {
            "parser": "belief_conditioned_content_reader",
            "relation": _relation_from_belief_content(row, reasoning),
        },
        {
            "parser": "mechanism_trace_reader",
            "relation": _relation_from_mechanism_trace(row, reasoning),
        },
        {
            "parser": "digest_connection_reader",
            "relation": _relation_from_digest_connection(row, reasoning),
        },
        {
            "parser": "explicit_belief_conflict_parser",
            "relation": _relation_from_explicit_conflict(row, reasoning),
        },
    ]


def select_ranked_papers(
    db_path: str | Path = DEFAULT_DB_PATH,
    tier_limits: list[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    limits = tier_limits or DEFAULT_TIER_LIMITS
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        for tier, limit in limits:
            rows = conn.execute(
                """
                SELECT * FROM papers
                WHERE signal_tier = ? AND llm_reasoning != ''
                ORDER BY COALESCE(json_extract(llm_reasoning, '$.final_score'), 0) DESC,
                         signal_score DESC,
                         ingested_at DESC
                LIMIT ?
                """,
                (tier, limit),
            ).fetchall()
            selected.extend(dict(row) for row in rows)
    finally:
        conn.close()
    return selected


def select_legacy_digest_papers(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    limit: int = 50,
    offset: int = 0,
    tiers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Select legacy LLM-digested papers for a V3 dry-run backfill."""
    selected_tiers = tiers if tiers is not None else DEFAULT_LEGACY_BACKFILL_TIERS
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        where = [
            "COALESCE(llm_reasoning, '') NOT IN ('', '{}')",
            "COALESCE(signal_tier, '') != 'CURATED_LIBRARY'",
        ]
        params: list[Any] = []
        if selected_tiers:
            placeholders = ", ".join("?" for _ in selected_tiers)
            where.append(f"COALESCE(signal_tier, '') IN ({placeholders})")
            params.extend(selected_tiers)
        query = f"""
            SELECT * FROM papers
            WHERE {' AND '.join(where)}
            ORDER BY datetime(ingested_at) ASC,
                     signal_score DESC,
                     title ASC
        """
        if limit > 0:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, max(0, offset)])
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def select_next_review_paper(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    doi: str = "",
) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if doi:
            row = conn.execute(
                """
                SELECT * FROM papers
                WHERE doi = ? AND llm_reasoning != ''
                LIMIT 1
                """,
                (doi,),
            ).fetchone()
            if not row:
                raise ValueError(f"no ranked paper found for doi: {doi}")
            return dict(row)

        rows = conn.execute(
            """
            SELECT * FROM papers
            WHERE signal_tier IN ('HIGH_PRIORITY', 'IMPORTANT', 'POTENTIAL', 'WATCHLIST')
              AND llm_reasoning != ''
            ORDER BY datetime(ingested_at) DESC,
                     COALESCE(json_extract(llm_reasoning, '$.final_score'), 0) DESC,
                     signal_score DESC
            LIMIT 50
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError("no ranked papers with llm_reasoning found")

    candidates = [dict(row) for row in rows]
    for row in candidates:
        votes = [candidate["relation"] for candidate in build_parse_candidates(row)]
        if "unclear" in votes or len(set(votes)) > 1:
            return row
    return candidates[0]


def _backfill_belief_template(judgment_mode: str) -> dict[str, str]:
    return BACKFILL_BELIEF_TEMPLATES.get(
        judgment_mode,
        BACKFILL_BELIEF_TEMPLATES["trajectory_fit"],
    )


def _single_parser_relation(candidates: list[dict[str, str]]) -> str:
    by_parser = {candidate["parser"]: candidate["relation"] for candidate in candidates}
    for parser in [
        "belief_conditioned_content_reader",
        "mechanism_trace_reader",
        "digest_connection_reader",
        "explicit_belief_conflict_parser",
    ]:
        relation = by_parser.get(parser, "unclear")
        if relation in {"support", "challenge"}:
            return relation
    return "unclear"


def _linear_confidence(results: list[dict[str, Any]]) -> float:
    confidence = INITIAL_CONFIDENCE
    for item in results:
        relation = item["relation"]
        if relation not in {"support", "challenge"}:
            continue
        delta = EVIDENCE_DELTAS.get(item.get("evidence_strength"), 0.10)
        confidence += delta if relation == "support" else -delta
        confidence = max(0.0, min(1.0, confidence))
    return round(confidence, 4)


def _linear_entrenchment(results: list[dict[str, Any]]) -> float:
    entrenchment = INITIAL_ENTRENCHMENT
    for item in results:
        relation = item["relation"]
        if relation not in {"support", "challenge"}:
            continue
        delta = EVIDENCE_DELTAS.get(item.get("evidence_strength"), 0.10)
        entrenchment += delta if relation == "support" else -delta
        entrenchment = max(0.0, min(1.0, entrenchment))
    return round(entrenchment, 4)


def run_v3_intake_ablation(
    rows: list[dict[str, Any]],
    *,
    kernel_dir: str | Path | None = None,
) -> dict[str, Any]:
    if kernel_dir is None:
        kernel_dir = (
            Path(tempfile.mkdtemp(prefix="scholarhound_v3_intake_", dir="/private/tmp"))
            / "kernel"
            / "v3"
        )
    kernel_dir = Path(kernel_dir)

    seed = create_evidence(
        kernel_dir,
        {
            "source_type": "report",
            "title": "V3 intake ablation seed",
            "source_ref": "local-ablation-seed",
            "summary": "Temporary seed evidence for V3 intake ablation.",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Ranked papers should strengthen an action-facing research trajectory",
            "claim": (
                "A durable ScholarHound signal should provide mechanism, bridge, "
                "or next-action evidence for an active research direction."
            ),
            "domain": "kernel-ablation",
            "confidence": INITIAL_CONFIDENCE,
            "entrenchment": INITIAL_ENTRENCHMENT,
        },
        reason="Temporary belief for V3 intake ablation.",
        evidence_ids=[seed["id"]],
    )

    results = []
    for row in rows:
        reasoning = _reasoning(row)
        paper_type, judgment_mode = _classify_type(row, reasoning)
        candidates = build_parse_candidates(row)
        evidence = create_evidence_from_parse_candidates(
            kernel_dir,
            {
                "source_type": "paper",
                "title": row.get("title") or "Untitled",
                "source_ref": row.get("doi") or row.get("title") or "unknown-source",
                "summary": _short(
                    reasoning.get("why_matters") or row.get("abstract") or ""
                ),
                "reports_lod": _reports_lod(row, reasoning),
                "paper_type": paper_type,
                "judgment_mode": judgment_mode,
                "source_tier": row.get("signal_tier") or "",
                "final_score": _final_score(row, reasoning),
                "trajectory_score": _number(
                    reasoning.get("trajectory_influence"),
                    _number(row.get("signal_trajectory"), 0.0),
                ),
                "concept_support_score": _number(
                    reasoning.get("concept_support"),
                    _number(row.get("concept_support_score"), 0.0),
                ),
            },
            belief_id=belief["id"],
            parse_candidates=candidates,
        )
        relation = evidence["evidence_relation_provenance"]["relation"]
        relation_provenance = evidence["evidence_relation_provenance"]
        action = {
            "support": "strengthen",
            "challenge": "challenge",
            "underdetermined": "update",
            "neutral": "update",
        }.get(
            relation,
            "contest",
        )
        updated, revision = revise_belief(
            kernel_dir,
            belief_id=belief["id"],
            evidence_ids=[evidence["id"]],
            action=action,
            reason=f"V3 intake ablation relation={relation}; mode={judgment_mode}.",
        )
        confidence_steps = revision.get("confidence_delta_policy", {}).get("steps", [])
        confidence_step = confidence_steps[0] if confidence_steps else {}
        single_relation = _single_parser_relation(candidates)
        results.append(
            {
                "title": row.get("title") or "",
                "doi": row.get("doi") or "",
                "tier": row.get("signal_tier") or "",
                "final_score": _final_score(row, reasoning),
                "paper_type": paper_type,
                "judgment_mode": judgment_mode,
                "relation": relation,
                "weak_direction": relation_provenance.get("weak_direction", ""),
                "adjudication_type": relation_provenance.get("adjudication_type", ""),
                "single_parser_relation": single_relation,
                "votes": [candidate["relation"] for candidate in candidates],
                "evidence_strength": evidence.get("evidence_strength", ""),
                "confidence_delta": revision.get("confidence_delta", 0.0),
                "confidence_after": updated.get("confidence"),
                "entrenchment_after": updated.get("entrenchment"),
                "entrenchment_resistance_factor": confidence_step.get(
                    "entrenchment_resistance_factor"
                ),
            }
        )

    health, issues = validate_v3_kernel(kernel_dir)
    contested_queue = get_contested_evidence_queue(kernel_dir)
    pending_queue = get_pending_evidence_queue(kernel_dir)
    final_belief = read_jsonl(kernel_dir / "beliefs" / "beliefs.jsonl")[0]
    relation_counts = dict(Counter(item["relation"] for item in results))
    direct_commit_count = sum(
        1 for item in results if item["single_parser_relation"] in {"support", "challenge"}
    )
    unstable_commits_prevented = sum(
        1
        for item in results
        if item["relation"] in {"contest", "underdetermined", "neutral"}
        and item["single_parser_relation"] in {"support", "challenge"}
    )
    linear_confidence = _linear_confidence(results)
    full_confidence = final_belief["confidence"]
    linear_entrenchment = _linear_entrenchment(results)
    full_entrenchment = final_belief["entrenchment"]
    inflation = max(0.0, linear_confidence - INITIAL_CONFIDENCE)
    reduction_pct = (
        round(100 * max(0.0, linear_confidence - full_confidence) / inflation, 1)
        if inflation
        else 0.0
    )
    entrenchment_inflation = max(0.0, linear_entrenchment - INITIAL_ENTRENCHMENT)
    entrenchment_reduction_pct = (
        round(
            100
            * max(0.0, linear_entrenchment - full_entrenchment)
            / entrenchment_inflation,
            1,
        )
        if entrenchment_inflation
        else 0.0
    )
    export_kernel_state(kernel_dir, kernel_dir / "exports" / "kernel_state.json")

    return {
        "kernel_dir": str(kernel_dir),
        "paper_count": len(results),
        "full_kernel": {
            "relation_counts": relation_counts,
            "contested_queue_count": len(contested_queue),
            "pending_queue_count": len(pending_queue),
            "final_confidence": full_confidence,
            "final_entrenchment": full_entrenchment,
            "validation_status": health["validation_status"],
            "validation_issue_count": len(issues),
        },
        "ablations": {
            "without_relation_consensus": {
                "direct_commit_count": direct_commit_count,
                "unstable_commits_prevented": unstable_commits_prevented,
                "unstable_commit_reduction_pct": round(
                    100 * unstable_commits_prevented / max(direct_commit_count, 1),
                    1,
                ),
            },
            "without_confidence_dampening": {
                "final_confidence": linear_confidence,
                "overconfidence_reduction_pct": reduction_pct,
            },
            "without_entrenchment_policy": {
                "final_entrenchment": linear_entrenchment,
                "overentrenchment_reduction_pct": entrenchment_reduction_pct,
            },
        },
        "papers": results,
    }


def run_v3_legacy_digest_backfill(
    rows: list[dict[str, Any]],
    *,
    kernel_dir: str | Path | None = None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dry-run legacy digests through V3 rules without touching durable kernel state."""
    if kernel_dir is None:
        kernel_dir = (
            Path(tempfile.mkdtemp(prefix="scholarhound_v3_backfill_", dir="/private/tmp"))
            / "kernel"
            / "v3"
        )
    kernel_dir = Path(kernel_dir)

    seed = create_evidence(
        kernel_dir,
        {
            "source_type": "report",
            "title": "V3 legacy digest backfill seed",
            "source_ref": "legacy-digest-backfill-dry-run",
            "summary": "Temporary seed evidence for V3-compatible legacy digest backfill.",
        },
    )

    beliefs_by_mode: dict[str, dict[str, Any]] = {}
    results = []
    for row in rows:
        reasoning = _reasoning(row)
        paper_type, judgment_mode = _classify_type(row, reasoning)
        template = _backfill_belief_template(judgment_mode)
        belief = beliefs_by_mode.get(judgment_mode)
        if belief is None:
            belief, _revision = create_belief(
                kernel_dir,
                {
                    **template,
                    "confidence": INITIAL_CONFIDENCE,
                    "entrenchment": INITIAL_ENTRENCHMENT,
                    "provenance": {
                        "method": "legacy_digest_backfill_dry_run_v1",
                        "judgment_mode": judgment_mode,
                    },
                },
                reason=(
                    "Temporary mode-specific belief for legacy digest dry-run backfill; "
                    "no durable kernel state is changed."
                ),
                evidence_ids=[seed["id"]],
            )
            beliefs_by_mode[judgment_mode] = belief

        candidates = build_parse_candidates(row)
        evidence = create_evidence_from_parse_candidates(
            kernel_dir,
            {
                "source_type": "paper",
                "title": row.get("title") or "Untitled",
                "source_ref": row.get("doi") or row.get("title") or "unknown-source",
                "summary": _short(
                    reasoning.get("why_matters") or row.get("abstract") or ""
                ),
                "reports_lod": _reports_lod(row, reasoning),
                "paper_type": paper_type,
                "judgment_mode": judgment_mode,
                "source_tier": row.get("signal_tier") or "",
                "final_score": _final_score(row, reasoning),
                "trajectory_score": _number(
                    reasoning.get("trajectory_influence"),
                    _number(row.get("signal_trajectory"), 0.0),
                ),
                "concept_support_score": _number(
                    reasoning.get("concept_support"),
                    _number(row.get("concept_support_score"), 0.0),
                ),
                "legacy_backfill": True,
            },
            belief_id=belief["id"],
            parse_candidates=candidates,
        )
        relation = evidence["evidence_relation_provenance"]["relation"]
        relation_provenance = evidence["evidence_relation_provenance"]
        action = {
            "support": "strengthen",
            "challenge": "challenge",
            "underdetermined": "update",
            "neutral": "update",
        }.get(relation, "contest")
        updated, revision = revise_belief(
            kernel_dir,
            belief_id=belief["id"],
            evidence_ids=[evidence["id"]],
            action=action,
            reason=f"Legacy digest backfill relation={relation}; mode={judgment_mode}.",
        )
        beliefs_by_mode[judgment_mode] = updated
        confidence_steps = revision.get("confidence_delta_policy", {}).get("steps", [])
        confidence_step = confidence_steps[0] if confidence_steps else {}
        results.append(
            {
                "title": row.get("title") or "",
                "doi": row.get("doi") or "",
                "journal": row.get("journal") or "",
                "tier": row.get("signal_tier") or "",
                "ingested_at": row.get("ingested_at") or "",
                "final_score": _final_score(row, reasoning),
                "paper_type": paper_type,
                "judgment_mode": judgment_mode,
                "belief_id": belief["id"],
                "belief_title": belief.get("title", ""),
                "relation": relation,
                "weak_direction": relation_provenance.get("weak_direction", ""),
                "adjudication_type": relation_provenance.get("adjudication_type", ""),
                "single_parser_relation": _single_parser_relation(candidates),
                "votes": [candidate["relation"] for candidate in candidates],
                "evidence_strength": evidence.get("evidence_strength", ""),
                "confidence_delta": revision.get("confidence_delta", 0.0),
                "confidence_after": updated.get("confidence"),
                "entrenchment_after": updated.get("entrenchment"),
                "entrenchment_resistance_factor": confidence_step.get(
                    "entrenchment_resistance_factor"
                ),
            }
        )

    health, issues = validate_v3_kernel(kernel_dir)
    contested_queue = get_contested_evidence_queue(kernel_dir)
    pending_queue = get_pending_evidence_queue(kernel_dir)
    beliefs = read_jsonl(kernel_dir / "beliefs" / "beliefs.jsonl")

    mode_counts: dict[str, dict[str, int]] = {}
    for item in results:
        mode = item["judgment_mode"]
        mode_counts.setdefault(mode, {"total": 0})
        mode_counts[mode]["total"] += 1
        mode_counts[mode][item["relation"]] = mode_counts[mode].get(item["relation"], 0) + 1

    direct_commit_count = sum(
        1 for item in results if item["single_parser_relation"] in {"support", "challenge"}
    )
    unstable_commits_prevented = sum(
        1
        for item in results
        if item["relation"] in {"contest", "underdetermined", "neutral"}
        and item["single_parser_relation"] in {"support", "challenge"}
    )

    return {
        "kernel_dir": str(kernel_dir),
        "selection": selection or {},
        "durable_kernel_impact": "none",
        "paper_count": len(results),
        "full_kernel": {
            "relation_counts": dict(Counter(item["relation"] for item in results)),
            "tier_counts": dict(Counter(item["tier"] for item in results)),
            "mode_counts": mode_counts,
            "contested_queue_count": len(contested_queue),
            "pending_queue_count": len(pending_queue),
            "validation_status": health["validation_status"],
            "validation_issue_count": len(issues),
        },
        "ablations": {
            "without_relation_consensus": {
                "direct_commit_count": direct_commit_count,
                "unstable_commits_prevented": unstable_commits_prevented,
                "unstable_commit_reduction_pct": round(
                    100 * unstable_commits_prevented / max(direct_commit_count, 1),
                    1,
                ),
            },
        },
        "beliefs": [
            {
                "id": belief["id"],
                "title": belief.get("title", ""),
                "domain": belief.get("domain", ""),
                "confidence": belief.get("confidence"),
                "entrenchment": belief.get("entrenchment"),
                "support": len(
                    [
                        evidence_id
                        for evidence_id in belief.get("evidence_ids", [])
                        if evidence_id != seed["id"]
                    ]
                ),
                "challenge": len(belief.get("contra_evidence_ids", [])),
                "pending": len(belief.get("pending_evidence_ids", [])),
                "neutral": len(belief.get("neutral_evidence_ids", [])),
                "contest": len(belief.get("contested_evidence_ids", [])),
            }
            for belief in beliefs
        ],
        "contested_queue": contested_queue,
        "pending_queue": pending_queue,
        "papers": results,
    }


def run_v3_legacy_digest_backfill_from_db(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    limit: int = 50,
    offset: int = 0,
    tiers: list[str] | None = None,
    kernel_dir: str | Path | None = None,
) -> dict[str, Any]:
    rows = select_legacy_digest_papers(
        db_path,
        limit=limit,
        offset=offset,
        tiers=tiers,
    )
    return run_v3_legacy_digest_backfill(
        rows,
        kernel_dir=kernel_dir,
        selection={
            "db_path": str(db_path),
            "limit": limit,
            "offset": offset,
            "tiers": tiers if tiers is not None else DEFAULT_LEGACY_BACKFILL_TIERS,
        },
    )


def run_v3_intake_ablation_from_db(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    kernel_dir: str | Path | None = None,
) -> dict[str, Any]:
    return run_v3_intake_ablation(
        select_ranked_papers(db_path),
        kernel_dir=kernel_dir,
    )


def run_v3_review_smoke(
    row: dict[str, Any],
    *,
    kernel_dir: str | Path | None = None,
) -> dict[str, Any]:
    if kernel_dir is None:
        kernel_dir = (
            Path(tempfile.mkdtemp(prefix="scholarhound_v3_review_", dir="/private/tmp"))
            / "kernel"
            / "v3"
        )
    kernel_dir = Path(kernel_dir)

    reasoning = _reasoning(row)
    paper_type, judgment_mode = _classify_type(row, reasoning)
    seed = create_evidence(
        kernel_dir,
        {
            "source_type": "report",
            "title": "V3 real review smoke seed",
            "source_ref": "local-real-review-smoke-seed",
            "summary": "Temporary seed evidence for a real-paper V3 review smoke test.",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Real review items must revise explicit belief state only",
            "claim": (
                "A real ScholarHound review item should become typed evidence that "
                "supports, challenges, or contests an explicit belief, with every "
                "durable change recorded as a BeliefRevision."
            ),
            "domain": "kernel-validation",
            "confidence": INITIAL_CONFIDENCE,
            "entrenchment": INITIAL_ENTRENCHMENT,
        },
        reason="Temporary belief for real-paper kernel smoke test.",
        evidence_ids=[seed["id"]],
    )
    candidates = build_parse_candidates(row)
    evidence = create_evidence_from_parse_candidates(
        kernel_dir,
        {
            "source_type": "paper",
            "title": row.get("title") or "Untitled",
            "source_ref": row.get("doi") or row.get("title") or "unknown-source",
            "summary": _short(reasoning.get("why_matters") or row.get("abstract") or ""),
            "reports_lod": _reports_lod(row, reasoning),
            "paper_type": paper_type,
            "judgment_mode": judgment_mode,
            "source_tier": row.get("signal_tier") or "",
            "final_score": _final_score(row, reasoning),
            "trajectory_score": _number(
                reasoning.get("trajectory_influence"),
                _number(row.get("signal_trajectory"), 0.0),
            ),
            "concept_support_score": _number(
                reasoning.get("concept_support"),
                _number(row.get("concept_support_score"), 0.0),
            ),
        },
        belief_id=belief["id"],
        parse_candidates=candidates,
    )
    relation = evidence["evidence_relation_provenance"]["relation"]
    relation_provenance = evidence["evidence_relation_provenance"]
    action = {
        "support": "strengthen",
        "challenge": "challenge",
        "underdetermined": "update",
        "neutral": "update",
    }.get(
        relation,
        "contest",
    )
    updated, revision = revise_belief(
        kernel_dir,
        belief_id=belief["id"],
        evidence_ids=[evidence["id"]],
        action=action,
        reason=f"Real review smoke relation={relation}; mode={judgment_mode}.",
    )
    health, issues = validate_v3_kernel(kernel_dir)
    contested_queue = get_contested_evidence_queue(kernel_dir)
    pending_queue = get_pending_evidence_queue(kernel_dir)

    return {
        "kernel_dir": str(kernel_dir),
        "durable_kernel_impact": "none",
        "paper": {
            "title": row.get("title") or "",
            "doi": row.get("doi") or "",
            "journal": row.get("journal") or "",
            "tier": row.get("signal_tier") or "",
            "final_score": _final_score(row, reasoning),
            "paper_type": paper_type,
            "judgment_mode": judgment_mode,
        },
        "parse_candidates": candidates,
        "evidence": evidence,
        "belief_before": belief,
        "belief_after": updated,
        "revision": revision,
        "action": action,
        "relation": relation,
        "weak_direction": relation_provenance.get("weak_direction", ""),
        "adjudication_type": relation_provenance.get("adjudication_type", ""),
        "single_parser_relation": _single_parser_relation(candidates),
        "contested_queue_count": len(contested_queue),
        "pending_queue_count": len(pending_queue),
        "validation_status": health["validation_status"],
        "validation_issue_count": len(issues),
        "validation_issues": [issue.__dict__ for issue in issues],
    }


def run_v3_review_smoke_from_db(
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    doi: str = "",
    kernel_dir: str | Path | None = None,
) -> dict[str, Any]:
    return run_v3_review_smoke(
        select_next_review_paper(db_path, doi=doi),
        kernel_dir=kernel_dir,
    )


def render_ablation_markdown(result: dict[str, Any]) -> str:
    full = result["full_kernel"]
    no_consensus = result["ablations"]["without_relation_consensus"]
    no_dampening = result["ablations"]["without_confidence_dampening"]
    no_entrenchment_policy = result["ablations"]["without_entrenchment_policy"]
    lines = [
        "# V3 Intake Ablation Benchmark",
        "",
        "Date: 2026-06-11",
        "",
        f"Temporary kernel: `{result['kernel_dir']}`",
        "",
        "Durable kernel impact: none unless caller supplied a durable kernel path.",
        "",
        "## Summary",
        "",
        f"- Papers tested: {result['paper_count']}",
        f"- Full-kernel relation counts: `{json.dumps(full['relation_counts'], sort_keys=True)}`",
        f"- Human contested queue: {full['contested_queue_count']}",
        f"- Pending evidence queue: {full['pending_queue_count']}",
        f"- Full-kernel final confidence: {full['final_confidence']}",
        f"- Full-kernel final entrenchment: {full['final_entrenchment']}",
        f"- Validation: {full['validation_status']} ({full['validation_issue_count']} issues)",
        "",
        "## Ablations",
        "",
        "### Without Relation Consensus",
        "",
        f"- Direct commits: {no_consensus['direct_commit_count']}",
        f"- Unstable commits prevented by consensus: {no_consensus['unstable_commits_prevented']}",
        f"- Unstable commit reduction: {no_consensus['unstable_commit_reduction_pct']}%",
        "",
        "### Without Confidence Dampening",
        "",
        f"- Linear final confidence: {no_dampening['final_confidence']}",
        f"- Overconfidence reduction: {no_dampening['overconfidence_reduction_pct']}%",
        "",
        "### Without Entrenchment Policy",
        "",
        f"- Linear final entrenchment: {no_entrenchment_policy['final_entrenchment']}",
        f"- Over-entrenchment reduction: {no_entrenchment_policy['overentrenchment_reduction_pct']}%",
        "",
        "## Paper-Level Outcomes",
        "",
        "Confidence delta is the actual revision after relation consensus, repeated-evidence "
        "dampening, and entrenchment resistance.",
        "",
        "| # | relation | weak | direct | tier | final | mode | votes | title | confidence_delta |",
        "|---:|---|---|---|---|---:|---|---|---|---:|",
    ]
    for index, item in enumerate(result["papers"], 1):
        votes = ",".join(item["votes"])
        title = _short(item["title"], 72)
        lines.append(
            f"| {index} | {item['relation']} | {item.get('weak_direction', '')} | "
            f"{item['single_parser_relation']} | {item['tier']} | {item['final_score']:.1f} | {item['judgment_mode']} | "
            f"{votes} | {title} | {item['confidence_delta']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The benchmark tests accountability behavior, not scientific truth.",
            "",
            "A good V3 run should prevent unstable direct commits and should avoid "
            "saturating confidence from repeated same-direction papers.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_review_smoke_markdown(result: dict[str, Any]) -> str:
    paper = result["paper"]
    revision = result["revision"]
    evidence = result["evidence"]
    confidence_policy = revision.get("confidence_delta_policy", {})
    entrenchment_policy = revision.get("entrenchment_delta_policy", {})
    votes = ",".join(candidate["relation"] for candidate in result["parse_candidates"])
    lines = [
        "# V3 Real Review Smoke Test",
        "",
        "Date: 2026-06-11",
        "",
        f"Temporary kernel: `{result['kernel_dir']}`",
        "",
        "Durable kernel impact: none.",
        "",
        "## Real Paper",
        "",
        f"- Title: {paper['title']}",
        f"- DOI: {paper['doi']}",
        f"- Journal: {paper['journal']}",
        f"- Tier: {paper['tier']}",
        f"- Final score: {paper['final_score']:.1f}",
        f"- Judgment mode: `{paper['judgment_mode']}`",
        "",
        "## Kernel Path",
        "",
        f"- Evidence id: `{evidence['id']}`",
        f"- Relation: `{result['relation']}`",
        f"- Weak direction: `{result['weak_direction']}`",
        f"- Adjudication type: `{result['adjudication_type']}`",
        f"- Direct parser relation: `{result['single_parser_relation']}`",
        f"- Votes: `{votes}`",
        f"- Action: `{result['action']}`",
        f"- Revision id: `{revision['id']}`",
        f"- Confidence delta: {revision.get('confidence_delta', 0.0):.4f}",
        f"- Confidence delta policy: `{confidence_policy.get('method', '')}`",
        f"- Entrenchment delta: {revision.get('entrenchment_delta', 0.0):.4f}",
        f"- Entrenchment delta policy: `{entrenchment_policy.get('method', '')}`",
        f"- Human contested queue count: {result['contested_queue_count']}",
        f"- Pending evidence queue count: {result['pending_queue_count']}",
        f"- Validation: {result['validation_status']} ({result['validation_issue_count']} issues)",
        "",
        "## Parse Boundary",
        "",
        f"- Verified low-inference fields: `{json.dumps(evidence.get('parse_boundary', {}).get('verified_low_inference_fields', []))}`",
        f"- Judgment-heavy fields: `{json.dumps(evidence.get('parse_boundary', {}).get('judgment_heavy_fields', []))}`",
        "",
        "## Interpretation",
        "",
        "This test validates the kernel path, not scientific truth. A passing run means "
        "a real reviewed paper became typed evidence, relation uncertainty stayed explicit, "
        "and any belief projection change was backed by an append-only BeliefRevision.",
    ]
    return "\n".join(lines) + "\n"


def render_legacy_backfill_markdown(result: dict[str, Any]) -> str:
    full = result["full_kernel"]
    no_consensus = result["ablations"]["without_relation_consensus"]
    selection = result.get("selection", {})
    lines = [
        "# V3 Legacy Digest Backfill Dry Run",
        "",
        "Date: 2026-06-11",
        "",
        f"Temporary kernel: `{result['kernel_dir']}`",
        "",
        "Durable kernel impact: none.",
        "",
        "## Selection",
        "",
        f"- Database: `{selection.get('db_path', '')}`",
        f"- Limit: {selection.get('limit', '')}",
        f"- Offset: {selection.get('offset', '')}",
        f"- Tiers: `{json.dumps(selection.get('tiers', []))}`",
        "",
        "## Summary",
        "",
        f"- Papers tested: {result['paper_count']}",
        f"- Relation counts: `{json.dumps(full['relation_counts'], sort_keys=True)}`",
        f"- Tier counts: `{json.dumps(full['tier_counts'], sort_keys=True)}`",
        f"- Human contested queue: {full['contested_queue_count']}",
        f"- Pending evidence queue: {full['pending_queue_count']}",
        f"- Neutral/off-topic evidence: {full['relation_counts'].get('neutral', 0)}",
        f"- Direct commits without consensus: {no_consensus['direct_commit_count']}",
        f"- Unstable commits prevented: {no_consensus['unstable_commits_prevented']} ({no_consensus['unstable_commit_reduction_pct']}%)",
        f"- Validation: {full['validation_status']} ({full['validation_issue_count']} issues)",
        "",
        "## Mode Breakdown",
        "",
        "| mode | total | support | underdetermined | neutral | contest | challenge |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, counts in sorted(full["mode_counts"].items()):
        lines.append(
            f"| {mode} | {counts.get('total', 0)} | {counts.get('support', 0)} | "
            f"{counts.get('underdetermined', 0)} | {counts.get('neutral', 0)} | {counts.get('contest', 0)} | "
            f"{counts.get('challenge', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Temporary Beliefs",
            "",
            "| belief | confidence | entrenchment | support | pending | neutral | contest | challenge |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for belief in result["beliefs"]:
        lines.append(
            f"| {belief['title']} | {belief['confidence']} | {belief['entrenchment']} | "
            f"{belief['support']} | {belief['pending']} | {belief['neutral']} | {belief['contest']} | {belief['challenge']} |"
        )

    lines.extend(
        [
            "",
            "## Human Queue Preview",
            "",
        ]
    )
    if result["contested_queue"]:
        for item in result["contested_queue"][:10]:
            lines.append(
                f"- `{item['evidence_id']}` {item['evidence_title']} -> {item['belief_title']}"
            )
    else:
        lines.append("- Empty")

    lines.extend(
        [
            "",
            "## Pending Queue Preview",
            "",
        ]
    )
    if result["pending_queue"]:
        for item in result["pending_queue"][:10]:
            weak = item.get("relation_provenance", {}).get("weak_direction", "")
            lines.append(
                f"- `{item['evidence_id']}` [{weak}] {item['evidence_title']} -> {item['belief_title']}"
            )
    else:
        lines.append("- Empty")

    lines.extend(
        [
            "",
            "## Paper-Level Outcomes",
            "",
            "| # | relation | weak | direct | tier | mode | final | title | confidence_delta |",
            "|---:|---|---|---|---|---|---:|---|---:|",
        ]
    )
    for index, item in enumerate(result["papers"], 1):
        title = _short(item["title"], 72)
        lines.append(
            f"| {index} | {item['relation']} | {item.get('weak_direction', '')} | "
            f"{item['single_parser_relation']} | {item['tier']} | {item['judgment_mode']} | "
            f"{item['final_score']:.1f} | {title} | {item['confidence_delta']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This dry run backfills legacy LLM digests into V3 accountability semantics. "
            "It does not rewrite old digest text and does not change durable belief state.",
        ]
    )
    return "\n".join(lines) + "\n"


def default_report_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "kernel" / "v3" / "exports" / "20_paper_intake_ablation_2026-06-11.md"


def default_review_smoke_report_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "kernel" / "v3" / "exports" / "real_item_review_smoke_2026-06-11.md"


def default_legacy_backfill_report_path() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "kernel" / "v3" / "backfills" / "legacy_digest_backfill_2026-06-11.md"
