"""
ScholarHound API + HTML5 frontend server.
Launch: psil serve
"""

import base64, errno, hashlib, hmac, json, os, re, socket
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

from psil.config import load_config
from psil.judgment import (
    apply_kernel_revision,
    build_judgment_kernel_summary,
    materialize_kernel_objects,
    materialize_kernel_tasks,
)
from psil.store.db import Database
from psil.rank.identity import load_identity
from psil.state_change import STATE_FIELDS, default_event_log_path, validate_event_log

CONFIG = load_config()
VAULT_PATH = os.getenv("SCHOLARHOUND_VAULT_PATH") or CONFIG.get("vault_path", "")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PACKET_PATH = REPO_ROOT / "g2_packet_v4.json"
DEFAULT_BENCHMARK_SELECTION_LOG_PATH = REPO_ROOT / "g2_packet_v4.selection_log.json"
BENCHMARK_PACKET_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_PACKET_PATH")
    or DEFAULT_BENCHMARK_PACKET_PATH
)
BENCHMARK_SELECTION_LOG_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_SELECTION_LOG_PATH")
    or DEFAULT_BENCHMARK_SELECTION_LOG_PATH
)
BENCHMARK_BOUNDARY_PACKET_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_BOUNDARY_PACKET_PATH")
    or BENCHMARK_PACKET_PATH.with_name("scifact_commitment_boundary_blinded_4_packet.json")
)
BENCHMARK_BOUNDARY_SELECTION_LOG_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_BOUNDARY_SELECTION_LOG_PATH")
    or BENCHMARK_SELECTION_LOG_PATH.with_name("scifact_commitment_boundary_blinded_4.selection_log.json")
)
BENCHMARK_CALIBRATION_PACKET_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_CALIBRATION_PACKET_PATH")
    or BENCHMARK_PACKET_PATH.with_name("g2_calibration_24_v1.json")
)
BENCHMARK_CALIBRATION_SELECTION_LOG_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_CALIBRATION_SELECTION_LOG_PATH")
    or BENCHMARK_SELECTION_LOG_PATH.with_name("g2_calibration_24_v1.selection_log.json")
)
BENCHMARK_DISPUTE_PACKET_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_DISPUTE_PACKET_PATH")
    or BENCHMARK_PACKET_PATH.with_name("g2_dispute_gateflip_28_v1.json")
)
BENCHMARK_DISPUTE_SELECTION_LOG_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_DISPUTE_SELECTION_LOG_PATH")
    or BENCHMARK_SELECTION_LOG_PATH.with_name("g2_dispute_gateflip_28_v1.selection_log.json")
)
BENCHMARK_FEEDBACK_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_FEEDBACK_PATH")
    or REPO_ROOT / "kernel" / "v3" / "goldsets" / "human_feedback" / "feedback.jsonl"
)
BENCHMARK_PROGRESS_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_PROGRESS_PATH")
    or REPO_ROOT / "kernel" / "v3" / "goldsets" / "human_feedback" / "progress.jsonl"
)
BENCHMARK_TEST_FEEDBACK_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_TEST_FEEDBACK_PATH")
    or REPO_ROOT / "kernel" / "v3" / "goldsets" / "human_feedback" / "test_feedback.jsonl"
)
BENCHMARK_TEST_PROGRESS_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_TEST_PROGRESS_PATH")
    or REPO_ROOT / "kernel" / "v3" / "goldsets" / "human_feedback" / "test_progress.jsonl"
)
BENCHMARK_REVIEWER_POLICY_PATH = Path(
    os.getenv("SCHOLARHOUND_BENCHMARK_REVIEWER_POLICY_PATH")
    or REPO_ROOT / "kernel" / "v3" / "goldsets" / "human_feedback" / "reviewer_policy.json"
)
BENCHMARK_SESSION_ID = "g2_v4_blind_feedback"
BENCHMARK_DEFAULT_PACKET_KEY = "full_72"
BENCHMARK_AUTH_COOKIE = "sh_benchmark_session"
BENCHMARK_AUTH_MAX_AGE = 60 * 60 * 24 * 30
BENCHMARK_RELATION_OPTIONS = {
    "support",
    "challenge",
    "neutral",
    "underdetermined",
    "contested",
    "skip",
}
BENCHMARK_CONFIDENCE_OPTIONS = {"low", "medium", "high"}
NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, max-age=0, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}

app = FastAPI(title="ScholarHound API")

ACCESS_PROTECTED_HOST = os.getenv(
    "SCHOLARHOUND_CANONICAL_HOST", "scholarhound.academy"
).strip().lower()
ACCESS_ALIAS_HOSTS = {
    host.strip().lower()
    for host in os.getenv(
        "SCHOLARHOUND_ALIAS_HOSTS", "www.scholarhound.academy"
    ).split(",")
    if host.strip()
}
_TRAJECTORY_MAP_CACHE: dict = {"key": None, "data": None}


@app.middleware("http")
async def redirect_public_aliases_to_access_host(request: Request, call_next):
    host = (request.headers.get("host") or "").split(":", 1)[0].lower()
    if ACCESS_PROTECTED_HOST and host in ACCESS_ALIAS_HOSTS:
        target = f"https://{ACCESS_PROTECTED_HOST}{request.url.path}"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return RedirectResponse(target, status_code=307)
    return await call_next(request)


# ── helpers ──────────────────────────────────────────────────────────────────
def get_db():
    db_path = str(Path.home() / ".psil" / "psil.db")
    db = Database(db_path)
    db.create_tables()
    return db


def list_digests():
    dailydir = os.path.join(VAULT_PATH, "daily")
    if not os.path.isdir(dailydir):
        return []
    files = sorted([f for f in os.listdir(dailydir) if f.endswith("-signals.md")], reverse=True)
    return [f.replace("-signals.md", "") for f in files]


def _state_change_log_path() -> Path:
    base_dir = Path(VAULT_PATH) if VAULT_PATH else Path.cwd()
    return default_event_log_path(base_dir)


def _v3_kernel_dir() -> Path:
    return REPO_ROOT / "kernel" / "v3"


def _state_change_status_counts(events: list[dict]) -> dict:
    counts = {
        state: {status: 0 for status in sorted(values)}
        for state, values in {
            "evidence": {"strengthened", "weakened", "unchanged"},
            "concept": {"new", "refinement", "contradiction", "unchanged"},
            "trajectory": {"reinforce", "branch", "terminate", "unchanged"},
            "constraint": {"reproducibility", "privacy", "translation", "interpretation", "unchanged"},
            "uncertainty": {"reduced", "increased", "unchanged"},
            "action": {"read", "verify", "synthesize", "ignore", "experiment", "none"},
        }.items()
    }
    for event in events:
        changed = event.get("changed_states") or {}
        for state in STATE_FIELDS:
            status = ((changed.get(state) or {}).get("status") or "unchanged").lower()
            counts[state][status] = counts[state].get(status, 0) + 1
    return counts


def _state_change_action_queue(events: list[dict], limit: int = 8) -> list[dict]:
    actions = []
    for event in reversed(events):
        payload = (event.get("changed_states") or {}).get("action") or {}
        status = (payload.get("status") or "none").lower()
        next_action = (payload.get("next_action") or "").strip()
        if status != "none" and next_action:
            actions.append({
                "event_id": event.get("event_id", ""),
                "created_at": event.get("created_at", ""),
                "status": status,
                "next_action": next_action,
                "source": event.get("source", {}),
            })
        if len(actions) >= limit:
            break
    return actions


def _latest_digest_with_ranked_papers(digests: list[str]) -> str:
    for digest_date in digests:
        data = parse_digest(digest_date)
        if data and data.get("papers"):
            return digest_date
    return digests[0] if digests else ""


def _trajectory_map_cache_key(digests: list[str]) -> tuple:
    db_path = Path.home() / ".psil" / "psil.db"
    db_mtime = db_path.stat().st_mtime if db_path.exists() else 0
    daily_dir = Path(VAULT_PATH) / "daily" if VAULT_PATH else Path()
    digest_mtime = 0
    if daily_dir.is_dir():
        digest_mtime = max(
            (path.stat().st_mtime for path in daily_dir.glob("*-signals.md")),
            default=0,
        )
    return (round(db_mtime, 3), round(digest_mtime, 3), tuple(digests[:5]))


def _digest_paper_dates(digests: list[str]) -> dict[str, str]:
    by_doi = {}
    for digest_date in digests:
        data = parse_digest(digest_date)
        if not data:
            continue
        for paper in data.get("papers", []):
            doi = (paper.get("doi") or "").strip()
            if doi and doi not in by_doi:
                by_doi[doi] = digest_date
    return by_doi


def _strip_markup(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _first_sentence_matching(sentences: list[str], patterns: list[str]) -> str:
    for pattern in patterns:
        rx = re.compile(pattern, re.I)
        for sentence in sentences:
            if rx.search(sentence):
                return sentence
    return ""


def _paper_reasoning(row: dict | None) -> dict:
    if not row:
        return {}
    try:
        return json.loads(row.get("llm_reasoning") or "{}")
    except Exception:
        return {}


def _score_number(value, default=0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _paper_scores(row: dict, reasoning: dict) -> dict:
    return {
        "relevance": reasoning.get("relevance", "?"),
        "novelty": reasoning.get("novelty", "?"),
        "bridge": reasoning.get("bridge", "?"),
        "trajectory": reasoning.get("trajectory_influence", row.get("signal_trajectory") or "?"),
        "concept_support": reasoning.get("concept_support", row.get("concept_support_score") or "?"),
        "final": reasoning.get("final_score", "?"),
    }


def _paper_workflow(reasoning: dict) -> dict:
    workflow = reasoning.get("workflow") or {}
    return {
        "question": workflow.get("question") or workflow.get("research_question") or "",
        "hypothesis": workflow.get("hypothesis") or "",
        "design": workflow.get("design") or workflow.get("experimental_design") or "",
        "key_method": workflow.get("key_method") or "",
        "key_result": workflow.get("key_result") or "",
        "gap": workflow.get("gap") or workflow.get("workflow_gap") or "",
    }


def _paper_payload(row: dict, digest_date: str = "") -> dict:
    reasoning = _paper_reasoning(row)
    toc_img = (row.get("toc_image_url") or "").strip()
    return {
        "title": row.get("title", ""),
        "journal": row.get("journal", ""),
        "doi": row.get("doi", ""),
        "pub_date": row.get("pub_date", ""),
        "ingested_at": row.get("ingested_at", ""),
        "tier": row.get("signal_tier", ""),
        "signal_score": row.get("signal_score", 0),
        "scores": _paper_scores(row, reasoning),
        "concept_name": row.get("concept_name") or reasoning.get("concept_name", ""),
        "paper_type": reasoning.get("paper_type", ""),
        "judgment_mode": reasoning.get("judgment_mode", ""),
        "problem_class": row.get("problem_class") or reasoning.get("problem_class", ""),
        "novelty_type": row.get("novelty_type") or reasoning.get("novelty_type", ""),
        "strategic_value": row.get("strategic_value") or reasoning.get("strategic_value", ""),
        "concept_support": row.get("concept_support_name") or reasoning.get("concept_support_name", ""),
        "why": reasoning.get("why_matters", ""),
        "connection": reasoning.get("potential_connection") or reasoning.get("concept_current_connection", ""),
        "weakness": reasoning.get("weakness", ""),
        "action": reasoning.get("action") or row.get("signal_action", ""),
        "workflow": _paper_workflow(reasoning),
        "abstract": _strip_markup(row.get("abstract", "")),
        "abstract_graph": build_abstract_graph(row, _paper_workflow(reasoning)),
        "toc_image": toc_img,
        "toc_graph": build_toc_graph(toc_img, "paper-record" if toc_img else ""),
        "digest_date": digest_date,
        "source": "digest" if digest_date else "database",
    }


_DASHBOARD_TIER_PREVIEW_LIMIT = 5


def _tier_papers(rows: list[dict], tier: str, digest_dates: dict[str, str], limit: int = 8) -> list[dict]:
    tier_rows = [row for row in rows if (row.get("signal_tier") or "") == tier]
    tier_rows.sort(
        key=lambda row: (
            _score_number(_paper_reasoning(row).get("final_score"), row.get("signal_score") or 0),
            row.get("ingested_at") or "",
        ),
        reverse=True,
    )
    return [
        _paper_payload(row, digest_dates.get((row.get("doi") or "").strip(), ""))
        for row in tier_rows[:limit]
    ]


_TRAJECTORY_KEYWORD_HINTS = {
    "molecular bioelectronics": [
        "oect",
        "organic electrochemical",
        "bioelectronic",
        "ionic-electronic",
        "mixed ionic-electronic",
        "iontronic",
        "iontronics",
        "electric double layer",
        "contact injection",
        "ion transport",
        "polyelectrolyte",
        "single-ion conductive",
        "ionogel",
        "ionogel-gated",
        "synaptic transistor",
        "mechanoreceptor",
        "triboelectric-capacitive",
        "mixed electron-ion",
        "foreign-body response",
        "immune-compatible",
        "semiconducting polymer",
        "organic semiconductor phase",
        "electrochemical organic light-emitting",
        "electrochemical",
        "small-molecule",
        "rna",
        "aptamer",
        "nanobody",
    ],
    "organoid + ev + sensing platforms": [
        "organoid",
        "extracellular vesicle",
        "exosome",
        "ev sensing",
        "secretome",
        "microphysiological",
        "organ-on-chip",
    ],
    "alzheimer's disease diagnostic-therapeutic systems": [
        "alzheimer",
        "amyloid",
        "tau",
        "neurodegenerative",
        "blood-brain barrier",
        "bbb",
    ],
    "adaptive biointerfaces": [
        "adaptive biointerface",
        "biointerface",
        "implant",
        "hydrogel",
        "stretchable",
        "soft bioelectronics",
        "foreign-body",
        "foreign-body response",
        "immune-compatible",
        "macrophage",
        "fibrotic",
        "dynamic interface",
    ],
    "nanophotonic field control": [
        "nanophotonic",
        "polariton",
        "plasmonic",
        "structured light",
        "nonlinear optics",
        "lithium niobate",
        "microresonator",
        "optical vortex",
        "skyrmion",
        "microcomb",
        "local optical environment",
        "optical confinement",
        "field-enhanced",
    ],
    "intelligent sensing platforms": [
        "intelligent sensing",
        "multimodal",
        "multiplex",
        "machine learning",
        "closed-loop",
        "programmable",
    ],
    "mechanobiology-enabled diagnostics": [
        "mechanobiology",
        "mechanotransduction",
        "mechanical",
        "force-coupled",
        "strain",
        "mechanically gated",
    ],
}

_RESEARCH_ARCS = [
    {
        "id": "arc-molecular-recognition-bioelectronic-transduction",
        "name": "Molecular Recognition -> Bioelectronic Transduction",
        "source_terms": [
            "molecular",
            "bioelectronic",
            "bioelectronics",
            "oect",
            "nanobody",
            "aptamer",
            "rna",
            "electrochemical",
        ],
        "keywords": [
            "molecular recognition",
            "bioelectronic",
            "bioelectronics",
            "oect",
            "organic electrochemical",
            "electrochemical transistor",
            "electrochemical organic light-emitting",
            "organic light-emitting transistor",
            "electric double layer",
            "contact injection",
            "ion transport",
            "iontronic",
            "polyelectrolyte",
            "single-ion conductive",
            "ionogel",
            "ionogel-gated",
            "synaptic transistor",
            "mechanoreceptor",
            "triboelectric-capacitive",
            "mixed electron-ion",
            "foreign-body response",
            "immune-compatible",
            "semiconducting polymer",
            "organic semiconductor phase",
            "ionic-electronic",
            "mixed ionic-electronic",
            "small molecule",
            "small-molecule",
            "rna",
            "rna biomarker",
            "aptamer",
            "nanobody",
        ],
    },
    {
        "id": "arc-organoid-ev-disease-state-readouts",
        "name": "Organoid / EV Disease-State Readouts",
        "source_terms": [
            "organoid",
            "ev",
            "exosome",
            "alzheimer",
            "disease",
            "secretome",
            "microphysiological",
        ],
        "keywords": [
            "organoid",
            "extracellular vesicle",
            "exosome",
            "ev sensing",
            "secretome",
            "microphysiological",
            "organ-on-chip",
            "alzheimer",
            "amyloid",
            "tau",
            "blood-brain barrier",
            "bbb",
            "disease state",
        ],
    },
    {
        "id": "arc-adaptive-living-interfaces",
        "name": "Adaptive Living Interfaces",
        "source_terms": [
            "adaptive",
            "biointerface",
            "mechanobiology",
            "mechanotransduction",
            "force",
            "implant",
            "stretchable",
        ],
        "keywords": [
            "adaptive biointerface",
            "biointerface",
            "implant",
            "foreign-body",
            "foreign-body response",
            "immune-compatible",
            "macrophage",
            "fibrotic",
            "hydrogel",
            "stretchable",
            "soft bioelectronics",
            "mechanobiology",
            "mechanotransduction",
            "mechanical",
            "force-coupled",
            "strain",
            "mechanically gated",
        ],
    },
    {
        "id": "arc-photonic-control-chemistry-biointerfaces",
        "name": "Nanophotonic Field Control",
        "source_terms": [
            "nanophotonic",
            "nanophotonics",
            "photonic",
            "photon",
            "polariton",
            "plasmonic",
            "structured light",
            "nonlinear optics",
            "lithium niobate",
            "microresonator",
            "optical vortex",
            "skyrmion",
            "microcomb",
            "structured light",
            "nonlinear optics",
            "lithium niobate",
            "microresonator",
            "optical vortex",
            "skyrmion",
            "microcomb",
        ],
        "keywords": [
            "nanophotonic",
            "nanophotonics",
            "polariton",
            "plasmonic",
            "structured light",
            "nonlinear optics",
            "lithium niobate",
            "microresonator",
            "optical vortex",
            "skyrmion",
            "microcomb",
            "structured light",
            "nonlinear optics",
            "integrated nonlinear optics",
            "nonlinear photonics",
            "lithium niobate",
            "microresonator",
            "microring resonator",
            "optical vortex",
            "structured optical vortex",
            "optical skyrmion",
            "microcomb",
            "topological charge",
            "spin-orbit coupling",
            "local optical environment",
            "optical confinement",
            "field-enhanced",
            "photothermal",
        ],
    },
]

_GENERIC_TRAJECTORY_TERMS = {
    "based",
    "diagnostic",
    "diagnostics",
    "disease",
    "enabled",
    "intelligent",
    "molecular",
    "platform",
    "platforms",
    "sensing",
    "systems",
    "therapeutic",
}

_STRONG_SINGLE_KEYWORDS = {
    "alzheimer",
    "amyloid",
    "aptamer",
    "bbb",
    "bioelectronic",
    "bioelectronics",
    "bodipy",
    "closed-loop",
    "exosome",
    "hydrogel",
    "implant",
    "mechanobiology",
    "mechanotransduction",
    "microphysiological",
    "multiplex",
    "nanobody",
    "nanophotonic",
    "nanophotonics",
    "oect",
    "organoid",
    "photochemistry",
    "photocleavage",
    "plasmonic",
    "polariton",
    "programmable",
    "secretome",
    "stretchable",
    "tau",
}


def _trajectory_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug or "trajectory"


def _trajectory_title(name: str) -> str:
    replacements = {
        "ev": "EV",
        "oect": "OECT",
        "nir": "NIR",
        "bodipy": "BODIPY",
        "bbb": "BBB",
        "rna": "RNA",
        "gpcr": "GPCR",
        "and": "and",
        "for": "for",
        "of": "of",
        "to": "to",
        "via": "via",
        "with": "with",
    }
    words = re.split(r"(\W+)", name or "")
    title = "".join(replacements.get(w.lower(), w.capitalize()) if re.match(r"\w+", w) else w for w in words)
    return title.replace("'S", "'s")


def _keywords_for_trajectory(name: str, ident) -> list[str]:
    lower = (name or "").lower()
    keywords = set()
    for key, hints in _TRAJECTORY_KEYWORD_HINTS.items():
        if key in lower or lower in key:
            keywords.update(hints)

    for raw in re.split(r"[\s+/(),;-]+", lower):
        token = raw.strip("'\"")
        if len(token) >= 4 and token not in _GENERIC_TRAJECTORY_TERMS:
            keywords.add(token)

    for topic in ident.trajectory_influence_topics:
        topic_l = (topic or "").lower()
        if any(k in topic_l or topic_l in k for k in keywords):
            keywords.add(topic_l)

    return sorted(keywords, key=len, reverse=True)


def _paper_search_text(row: dict, reasoning: dict) -> str:
    workflow = _paper_workflow(reasoning)
    causal = reasoning.get("causal") or {}
    fields = [
        row.get("title", ""),
        row.get("abstract", ""),
        row.get("journal", ""),
        row.get("concept_name", ""),
        row.get("concept_support_name", ""),
        row.get("problem_class", ""),
        row.get("novelty_type", ""),
        row.get("strategic_value", ""),
        row.get("evidence_type", ""),
        reasoning.get("concept_name", ""),
        reasoning.get("concept_support_name", ""),
        reasoning.get("problem_class", ""),
        reasoning.get("novelty_type", ""),
        reasoning.get("strategic_value", ""),
        causal.get("question", ""),
        causal.get("constraint", ""),
        causal.get("input_state", ""),
        causal.get("transformation", ""),
        causal.get("output_state", ""),
        causal.get("outcome", ""),
        workflow.get("question", ""),
        workflow.get("hypothesis", ""),
        workflow.get("design", ""),
        workflow.get("key_method", ""),
        workflow.get("key_result", ""),
    ]
    return " ".join(_strip_markup(str(field)) for field in fields if field).lower()


def _concept_search_text(concept: dict) -> str:
    fields = [
        concept.get("name", ""),
        concept.get("why_matters", ""),
        concept.get("definition", ""),
        concept.get("logic_pattern", ""),
        concept.get("state_transition", ""),
    ]
    return " ".join(_strip_markup(str(field)) for field in fields if field).lower()


def _keyword_matches(text: str, keywords: list[str]) -> bool:
    phrase_hits = 0
    token_hits = 0
    for keyword in keywords:
        kw = (keyword or "").lower().strip()
        if not kw:
            continue
        if " " in kw or "-" in kw:
            if kw in text:
                phrase_hits += 1
        elif kw in _STRONG_SINGLE_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return True
        elif kw not in _GENERIC_TRAJECTORY_TERMS and re.search(rf"\b{re.escape(kw)}\b", text):
            token_hits += 1
    return phrase_hits > 0 or token_hits >= 2


def _trajectory_priority(row: dict) -> tuple:
    tier_rank = {
        "HIGH_PRIORITY": 5,
        "CURATED_LIBRARY": 4.5,
        "IMPORTANT": 4,
        "POTENTIAL": 3,
        "WATCHLIST": 2,
        "BLIND_SPOT": 1,
    }
    reasoning = _paper_reasoning(row)
    text = _paper_search_text(row, reasoning)
    curated_focus = 0
    if row.get("signal_tier") == "CURATED_LIBRARY":
        if "organic electrochemical" in text or "oect" in text:
            curated_focus += 2
        if "rna" in text or "rna biomarkers" in text:
            curated_focus += 2
        if "small molecule" in text or "small-molecule" in text:
            curated_focus += 1
    return (
        tier_rank.get(row.get("signal_tier", ""), 0),
        curated_focus,
        _score_number(reasoning.get("final_score"), row.get("signal_score") or 0),
        row.get("ingested_at") or "",
    )


def _trajectory_paper_payload(row: dict, digest_date: str = "") -> dict:
    payload = _paper_payload(row, digest_date)
    return {
        "title": payload["title"],
        "journal": payload["journal"],
        "doi": payload["doi"],
        "pub_date": payload["pub_date"],
        "tier": payload["tier"],
        "scores": {"final": payload.get("scores", {}).get("final", "?")},
        "concept_name": payload.get("concept_name", ""),
        "concept_support": payload.get("concept_support", ""),
        "digest_date": payload.get("digest_date", ""),
        "toc_image": payload.get("toc_image", ""),
    }


def _trajectory_status(papers: list[dict], concepts: list[dict], trajectory_row: dict | None) -> str:
    high_count = sum(1 for paper in papers if paper.get("signal_tier") == "HIGH_PRIORITY")
    important_count = sum(1 for paper in papers if paper.get("signal_tier") == "IMPORTANT")
    evidence_count = int((trajectory_row or {}).get("evidence_count") or 0)
    if high_count or evidence_count >= 6:
        return "rising"
    if important_count or evidence_count >= 2 or len(concepts) >= 3:
        return "active"
    if papers or concepts:
        return "watching"
    return "seed"


def _trajectory_next_move(papers: list[dict], concepts: list[dict]) -> str:
    if any(p.get("signal_tier") == "HIGH_PRIORITY" for p in papers):
        return "Read now and decide whether it changes the working map."
    if any(p.get("signal_tier") == "IMPORTANT" for p in papers):
        return "Review this week; it may sharpen an existing direction."
    if concepts:
        return "Watch for a second independent paper before escalating."
    return "Keep as an identity anchor; current corpus evidence is still thin."


def _trajectory_missing_link(papers: list[dict], concepts: list[dict]) -> str:
    for concept in concepts:
        missing = (concept.get("missing_link") or concept.get("opportunity") or "").strip()
        if missing:
            return _strip_markup(missing)[:260]
    for paper in papers:
        gap = _paper_workflow(_paper_reasoning(paper)).get("gap", "").strip()
        if gap:
            return _strip_markup(gap)[:260]
    return ""


def _trajectory_judgment(name: str, papers: list[dict], concepts: list[dict]) -> str:
    if papers:
        top = papers[0]
        concept_names = [c.get("name", "") for c in concepts[:2] if c.get("name")]
        concept_part = ", ".join(concept_names) if concept_names else (top.get("concept_support_name") or top.get("concept_name") or "recent paper evidence")
        return f"Evidence is clustering around {concept_part}; the current lead paper is \"{_strip_markup(top.get('title', ''))[:120]}\"."
    if concepts:
        names = ", ".join(c.get("name", "") for c in concepts[:3] if c.get("name"))
        return f"The corpus has concept momentum around {names}, but no high-priority lead paper has landed yet."
    return f"{_trajectory_title(name)} remains part of the research identity, but it has not accumulated fresh evidence in the current corpus."


def _first_nonempty(values: list[str], default: str = "") -> str:
    for value in values:
        cleaned = _strip_markup(value or "")
        if cleaned:
            return cleaned
    return default


def _top_names(items: list[dict], limit: int = 3) -> list[str]:
    return [item.get("name", "") for item in items[:limit] if item.get("name")]


def _related_trajectory_names(items: list[dict], limit: int = 4) -> list[str]:
    names: list[str] = []
    for item in items:
        related = item.get("source_trajectories") or [item.get("name", "")]
        for name in related:
            if name and name not in names:
                names.append(name)
            if len(names) >= limit:
                return names
    return names


def _merge_concept_payloads(items: list[dict], limit: int = 7) -> list[dict]:
    merged: dict[str, dict] = {}
    for concept in items:
        name = (concept.get("name") or "").strip()
        if not name:
            continue
        current = merged.setdefault(name, dict(concept))
        current["appearances"] = max(
            int(current.get("appearances") or 0),
            int(concept.get("appearances") or 0),
        )
        current.setdefault("status", concept.get("status", ""))
        current.setdefault("weight", concept.get("trajectory_weight") or concept.get("weight", "medium"))
    return sorted(
        merged.values(),
        key=lambda c: (int(c.get("appearances") or 0), c.get("name", "")),
        reverse=True,
    )[:limit]


def _merge_paper_payloads(items: list[dict], limit: int = 12) -> list[dict]:
    merged: dict[str, dict] = {}
    for paper in items:
        key = (paper.get("doi") or paper.get("title") or "").strip().lower()
        if key and key not in merged:
            merged[key] = paper
    return list(merged.values())[:limit]


def _trajectory_name_matches_arc(trajectory: dict, arc: dict) -> bool:
    name = (trajectory.get("name") or "").lower()
    return any(term in name for term in arc.get("source_terms", []))


def _strongest_trajectory_status(statuses: list[str]) -> str:
    rank = {"rising": 3, "active": 2, "watching": 1, "seed": 0}
    return max(statuses or ["seed"], key=lambda status: rank.get(status, 0))


def _build_research_arcs(
    trajectories: list[dict],
    papers: list[dict],
    concepts: list[dict],
    digest_dates: dict[str, str],
) -> list[dict]:
    arcs = []
    for arc in _RESEARCH_ARCS:
        keywords = arc["keywords"]
        source_trajectories = [
            trajectory for trajectory in trajectories
            if _trajectory_name_matches_arc(trajectory, arc)
        ]

        matched_papers = []
        for paper in papers:
            reasoning = _paper_reasoning(paper)
            if _keyword_matches(_paper_search_text(paper, reasoning), keywords):
                matched_papers.append(paper)
        matched_papers.sort(key=_trajectory_priority, reverse=True)

        matched_concepts = [
            concept for concept in concepts
            if _keyword_matches(_concept_search_text(concept), keywords)
        ]
        matched_concepts.sort(
            key=lambda c: (int(c.get("appearances") or 0), c.get("last_seen") or ""),
            reverse=True,
        )

        direct_papers = [
            _trajectory_paper_payload(row, digest_dates.get((row.get("doi") or "").strip(), ""))
            for row in matched_papers[:12]
        ]
        source_papers = [
            paper
            for trajectory in source_trajectories
            for paper in trajectory.get("papers", [])
        ]
        paper_payloads = _merge_paper_payloads([*direct_papers, *source_papers])

        source_concepts = [
            concept
            for trajectory in source_trajectories
            for concept in trajectory.get("concepts", [])
        ]
        concept_payloads = _merge_concept_payloads([*matched_concepts, *source_concepts])

        evidence_count = max(
            sum(int(trajectory.get("evidence_count") or 0) for trajectory in source_trajectories),
            len(paper_payloads) + sum(int(concept.get("appearances") or 0) for concept in concept_payloads[:5]),
        )
        status = _strongest_trajectory_status([
            _trajectory_status(matched_papers, matched_concepts, {"evidence_count": evidence_count}),
            *[trajectory.get("status", "seed") for trajectory in source_trajectories],
        ])
        missing_link = _first_nonempty([
            _trajectory_missing_link(matched_papers, matched_concepts),
            *[trajectory.get("missing_link", "") for trajectory in source_trajectories],
        ])
        next_move = _first_nonempty([
            _trajectory_next_move(matched_papers, matched_concepts),
            *[trajectory.get("next_move", "") for trajectory in source_trajectories],
        ])

        arcs.append({
            "id": arc["id"],
            "name": arc["name"],
            "status": status,
            "confidence": "Rising" if status == "rising" else "Stable",
            "evidence_count": evidence_count,
            "paper_count": len(paper_payloads),
            "concept_count": len(concept_payloads),
            "concepts": concept_payloads[:5],
            "papers": paper_payloads[:8],
            "judgment": _trajectory_judgment(arc["name"], matched_papers, matched_concepts),
            "missing_link": missing_link,
            "next_move": next_move,
            "source_trajectories": [trajectory.get("name", "") for trajectory in source_trajectories if trajectory.get("name")],
        })
    return arcs


def _merge_story_concepts(trajectories: list[dict], limit: int = 5) -> list[dict]:
    merged: dict[str, dict] = {}
    for trajectory in trajectories:
        for concept in trajectory.get("concepts", []):
            name = (concept.get("name") or "").strip()
            if not name:
                continue
            current = merged.setdefault(name, dict(concept))
            current["appearances"] = max(int(current.get("appearances") or 0), int(concept.get("appearances") or 0))
    return sorted(merged.values(), key=lambda c: int(c.get("appearances") or 0), reverse=True)[:limit]


def _merge_story_papers(trajectories: list[dict], limit: int = 5) -> list[dict]:
    by_doi: dict[str, dict] = {}
    for trajectory in trajectories:
        for paper in trajectory.get("papers", []):
            key = (paper.get("doi") or paper.get("title") or "").strip()
            if key and key not in by_doi:
                by_doi[key] = paper
    return list(by_doi.values())[:limit]


def _story_node(
    node_id: str,
    node_type: str,
    title: str,
    summary: str,
    trajectories: list[dict],
    concepts: list[dict],
    papers: list[dict],
    missing_link: str = "",
    next_move: str = "",
    status: str = "active",
    type_label: str | None = None,
) -> dict:
    return {
        "id": node_id,
        "node_type": node_type,
        "type_label": type_label or node_type.replace("_", " ").title(),
        "title": title,
        "summary": summary,
        "status": status,
        "evidence_count": sum(int(t.get("evidence_count") or 0) for t in trajectories),
        "paper_count": max(
            len(papers),
            sum(int(t.get("paper_count") or 0) for t in trajectories),
        ),
        "concept_count": max(
            len(concepts),
            sum(int(t.get("concept_count") or 0) for t in trajectories),
        ),
        "related_trajectories": _related_trajectory_names(trajectories, 4),
        "concepts": concepts[:5],
        "papers": papers[:8],
        "missing_link": missing_link,
        "next_move": next_move,
    }


def _story_direction_question(name: str) -> str:
    title = _trajectory_title(name)
    lower = title.lower()
    if any(k in lower for k in ["molecular", "oect", "bioelectronic"]):
        return "Can molecular recognition switch a bioelectronic state?"
    if any(k in lower for k in ["nanophotonic", "photon", "nonlinear optics", "structured light", "lithium niobate", "microresonator", "optical vortex", "skyrmion", "microcomb"]):
        return "Can programmable optical field states become a control layer?"
    if any(k in lower for k in ["organoid", "ev", "secretome", "microphysiological"]):
        return "Can organoid and EV signals become disease-state readouts?"
    if any(k in lower for k in ["adaptive", "stretchable", "biointerface"]):
        return "Can biointerfaces adapt to tissue force instead of tolerating it?"
    if "mechanobiology" in lower or "force" in lower:
        return "Can mechanical state become a diagnostic signal?"
    if any(k in lower for k in ["alzheimer", "bbb", "barrier"]):
        return "Can barrier and vesicle signals bridge diagnosis and therapy?"
    if "intelligent" in lower or "closed-loop" in lower:
        return "Can sensing become a closed-loop decision system?"
    return f"What mechanism would make {title} change the model?"


def _story_direction_hypothesis(name: str, concepts: list[dict]) -> str:
    title = _trajectory_title(name)
    lower = title.lower()
    if any(k in lower for k in ["molecular", "oect", "bioelectronic"]):
        return "Binding only matters if it changes ionic-electronic state."
    if any(k in lower for k in ["nanophotonic", "photon", "nonlinear optics", "structured light", "lithium niobate", "microresonator", "optical vortex", "skyrmion", "microcomb"]):
        return "Optical structure matters only if it changes a controllable field state."
    if any(k in lower for k in ["organoid", "ev", "secretome", "microphysiological"]):
        return "A readout matters only if it preserves disease context."
    if any(k in lower for k in ["adaptive", "stretchable", "biointerface", "mechanobiology", "force"]):
        return "Interface mechanics matter only if they change the sensing question."
    if "intelligent" in lower or "closed-loop" in lower:
        return "A sensor matters only if it changes the next decision."
    names = _top_names(concepts, 1)
    if names:
        return f"{names[0]} matters only if it changes the working model."
    return f"{title} matters only if it produces a testable mechanism."


def _story_paper_anchor(title: str) -> str:
    cleaned = _strip_markup(title or "")
    if not cleaned:
        return "No lead paper yet"
    words = cleaned.split()
    anchor = " ".join(words[:6])
    return anchor + ("..." if len(words) > 6 else "")


def _story_direction_turning_point(papers: list[dict]) -> str:
    if papers:
        return f"{_story_paper_anchor(papers[0].get('title', ''))} becomes the test case."
    return "The turning point is still waiting for a test case."


def _story_relevance_terms(name: str, concepts: list[dict], papers: list[dict]) -> set[str]:
    stopwords = {
        "paper", "papers", "sensor", "sensors", "sensing", "biosensor", "biosensors",
        "biosensing", "platform", "platforms", "system", "systems", "diagnostic", "diagnostics",
        "enabled", "based", "using", "with", "from",
        "molecular",
    }
    text = _trajectory_title(name).lower()
    terms = {
        token
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) >= 4 and token not in stopwords
    }
    if "bioelectronics" in text:
        terms.add("bioelectronic")
    if "nanophotonics" in text:
        terms.add("nanophotonic")
    if "biointerfaces" in text:
        terms.add("biointerface")
    for phrase in ["oect", "ev", "nir", "bbb", "rna"]:
        if phrase in text:
            terms.add(phrase)
    return terms


def _story_relevant_delta(
    deltas: list[dict],
    name: str,
    concepts: list[dict],
    papers: list[dict],
) -> dict:
    terms = _story_relevance_terms(name, concepts, papers)
    paper_dois = {
        (paper.get("doi") or "").strip().lower()
        for paper in papers
        if (paper.get("doi") or "").strip()
    }
    for delta in deltas:
        source_dois = (delta.get("source_dois") or "").lower()
        if paper_dois and any(doi in source_dois for doi in paper_dois):
            return delta
        text = " ".join([
            delta.get("previous_assumption", "") or "",
            delta.get("new_assumption", "") or "",
            delta.get("delta", "") or "",
        ]).lower()
        if terms and any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms):
            return delta
    return {}


def _story_direction_shift(name: str, deltas: list[dict], concepts: list[dict], papers: list[dict]) -> str:
    relevant = _story_relevant_delta(deltas, name, concepts, papers)
    if relevant:
        previous = _strip_markup(relevant.get("previous_assumption", ""))
        new = _strip_markup(relevant.get("new_assumption", ""))
        if previous and new:
            return f"From {previous} to {new}."
        if new:
            return f"Toward {new}."
    title = _trajectory_title(name)
    return f"From collecting {title} papers to testing a model-changing mechanism."


def _story_current_model_title(name: str, concepts: list[dict]) -> str:
    title = _trajectory_title(name)
    lower = title.lower()
    if any(k in lower for k in ["molecular", "oect", "bioelectronic"]):
        return "Current model: recognition -> ionic-electronic state"
    if any(k in lower for k in ["nanophotonic", "photon", "nonlinear optics", "structured light", "lithium niobate", "microresonator", "optical vortex", "skyrmion", "microcomb"]):
        return "Current model: photonic structure -> programmable field state"
    if any(k in lower for k in ["organoid", "ev", "secretome", "microphysiological"]):
        return "Current model: disease state -> organoid/EV readout"
    if any(k in lower for k in ["adaptive", "stretchable", "biointerface"]):
        return "Current model: tissue force -> adaptive interface"
    if "mechanobiology" in lower or "force" in lower:
        return "Current model: mechanical state -> diagnostic signal"
    if "intelligent" in lower or "closed-loop" in lower:
        return "Current model: sensing result -> next decision"
    names = _top_names(concepts, 1)
    return f"Current model: {names[0]} -> research decision" if names else f"Current model: {title} -> research decision"


def _story_next_question(name: str) -> str:
    title = _trajectory_title(name)
    lower = title.lower()
    if any(k in lower for k in ["molecular", "oect", "bioelectronic"]):
        return "Which validation would make it diagnostic?"
    if any(k in lower for k in ["nanophotonic", "photon", "nonlinear optics", "structured light", "lithium niobate", "microresonator", "optical vortex", "skyrmion", "microcomb"]):
        return "Which field variable proves nanophotonic control?"
    if any(k in lower for k in ["organoid", "ev", "secretome", "microphysiological"]):
        return "Which benchmark separates disease signal from noise?"
    if any(k in lower for k in ["adaptive", "stretchable", "biointerface", "mechanobiology", "force"]):
        return "Which force-coupled experiment would change the model?"
    if "intelligent" in lower or "closed-loop" in lower:
        return "Which decision loop would prove the platform?"
    return "Which result would force the question to change?"


def build_evolution_story(trajectories: list[dict], deltas: list[dict] | None = None) -> dict:
    """Build the trajectory story as question evolution, not topic taxonomy."""
    deltas = deltas or []
    top = trajectories[:3]
    lead = top[0] if top else {}
    story_trajectories = [lead] if lead else []
    direction = lead.get("name") or "Current Corpus"
    concepts = lead.get("concepts", []) or _merge_story_concepts(story_trajectories or trajectories)
    papers = lead.get("papers", []) or _merge_story_papers(story_trajectories or trajectories)
    lead_concepts = concepts
    lead_papers = papers
    question = _story_direction_question(direction)
    hypothesis = _story_direction_hypothesis(direction, lead_concepts)
    turning_point = _story_direction_turning_point(lead_papers)
    shift = _story_direction_shift(direction, deltas, lead_concepts, lead_papers)
    current_model_title = _story_current_model_title(direction, lead_concepts)
    next_question = _story_next_question(direction)
    missing = _first_nonempty([t.get("missing_link", "") for t in story_trajectories], "The next discriminating gap has not been extracted yet.")
    next_move = _first_nonempty([t.get("next_move", "") for t in story_trajectories], "Select the next paper by whether it can change the current model.")
    lead_paper_title = lead_papers[0].get("title", "") if lead_papers else ""
    evidence_count = int(lead.get("evidence_count") or 0)
    if not evidence_count:
        evidence_count = len(papers) + sum(int(concept.get("appearances") or 0) for concept in concepts[:5])
    paper_count = int(lead.get("paper_count") or len(papers))
    concept_count = int(lead.get("concept_count") or len(concepts))
    evidence_phrase = f"{evidence_count} evidence point{'s' if evidence_count != 1 else ''}"
    paper_phrase = f"{paper_count} supporting paper{'s' if paper_count != 1 else ''}"
    concept_phrase = f"{concept_count} related concept signal{'s' if concept_count != 1 else ''}"
    evidence_state_summary = (
        f"{evidence_phrase}, {paper_phrase} and {concept_phrase} currently support this node. "
        "Treat this as evidence coverage, not a confidence percentage."
    )
    current_model_summary = _first_nonempty([
        (lead.get("judgment", "") if lead else ""),
        f"The working model for {_trajectory_title(direction)} is not a topic label; it is the claim that future papers must strengthen, revise, or break.",
    ])
    lead_summary = (
        f"The current test paper is \"{_strip_markup(lead_paper_title)[:150]}\"."
        if lead_paper_title
        else "No single paper has become the test case yet."
    )

    nodes = [
        _story_node(
            "story-question",
            "question",
            question,
            f"{evidence_state_summary} The useful question is whether new evidence changes the working model: {hypothesis}",
            story_trajectories,
            concepts,
            papers,
            missing,
            "Keep the map centered on the question that could change the research model.",
            "active",
            "Core Question",
        ),
        _story_node(
            "story-working-hypothesis",
            "working_hypothesis",
            hypothesis,
            f"This is the working claim underneath the direction. Papers are useful when they make it more precise or easier to falsify.",
            story_trajectories,
            concepts,
            papers,
            missing,
            next_move,
            "watching",
            "Working Hypothesis",
        ),
        _story_node(
            "story-turning-point",
            "turning_point",
            turning_point,
            f"{lead_summary} It matters because it turns the question from an interesting direction into something that can be argued over with evidence.",
            story_trajectories,
            concepts,
            papers,
            missing,
            "Use supporting papers only after the turning point is clear.",
            "rising" if papers else "watching",
            "Turning Point",
        ),
        _story_node(
            "story-conceptual-shift",
            "conceptual_shift",
            shift,
            f"{shift} Interpret this as a trajectory-state change, with domain labels only as context.",
            story_trajectories,
            concepts,
            papers[:2],
            "",
            "Keep domain labels as context and make the conceptual shift explicit.",
            "seed",
            "Conceptual Shift",
        ),
        _story_node(
            "story-current-model",
            "current_model",
            current_model_title,
            current_model_summary,
            story_trajectories,
            concepts,
            papers,
            missing,
            next_move,
            "active",
            "Current Model",
        ),
        _story_node(
            "story-next-question",
            "next_question",
            next_question,
            missing,
            story_trajectories,
            concepts,
            papers,
            missing,
            next_move,
            "rising",
            "Next Question",
        ),
    ]
    return {
        "id": lead.get("id") or f"story-{_trajectory_slug(direction)}",
        "direction": _trajectory_title(direction),
        "status": lead.get("status", "seed"),
        "evidence_count": int(lead.get("evidence_count") or 0),
        "nodes": nodes,
        "edges": [
            {"source": nodes[i]["id"], "target": nodes[i + 1]["id"], "type": "evolves_to"}
            for i in range(len(nodes) - 1)
        ],
    }


def build_trajectory_map(
    papers: list[dict],
    concepts: list[dict],
    digest_dates: dict[str, str] | None = None,
    trajectory_rows: list[dict] | None = None,
    deltas: list[dict] | None = None,
) -> dict:
    """Public-facing trajectory layer over the private judgment kernel."""
    ident = load_identity()
    digest_dates = digest_dates or {}
    trajectory_by_name = {(row.get("name") or "").lower(): row for row in (trajectory_rows or [])}
    raw_names: list[str] = []
    for name in [*ident.long_term_vision, *ident.current_core, *ident.emerging_directions]:
        if name and name.lower() not in {n.lower() for n in raw_names}:
            raw_names.append(name)

    trajectories = []
    edges = []
    center_id = "kernel"

    for name in raw_names[:8]:
        keywords = _keywords_for_trajectory(name, ident)
        matched_papers = []
        for paper in papers:
            reasoning = _paper_reasoning(paper)
            if _keyword_matches(_paper_search_text(paper, reasoning), keywords):
                matched_papers.append(paper)
        matched_papers.sort(key=_trajectory_priority, reverse=True)

        matched_concepts = [
            concept for concept in concepts
            if _keyword_matches(_concept_search_text(concept), keywords)
        ]
        matched_concepts.sort(
            key=lambda c: (int(c.get("appearances") or 0), c.get("last_seen") or ""),
            reverse=True,
        )

        row = trajectory_by_name.get(name.lower())
        node_id = f"trajectory-{_trajectory_slug(name)}"
        status = _trajectory_status(matched_papers, matched_concepts, row)
        paper_count = len(matched_papers)
        concept_count = len(matched_concepts)
        evidence_count = max(
            int((row or {}).get("evidence_count") or 0),
            paper_count + sum(int(c.get("appearances") or 0) for c in matched_concepts[:5]),
        )

        trajectories.append({
            "id": node_id,
            "name": _trajectory_title(name),
            "status": status,
            "confidence": (row or {}).get("confidence") or ("Rising" if status == "rising" else "Stable"),
            "evidence_count": evidence_count,
            "paper_count": paper_count,
            "concept_count": concept_count,
            "concepts": [
                {
                    "name": concept.get("name", ""),
                    "appearances": concept.get("appearances", 0),
                    "status": concept.get("status", ""),
                    "weight": concept.get("trajectory_weight", "medium"),
                }
                for concept in matched_concepts[:5]
            ],
            "papers": [
                _trajectory_paper_payload(row, digest_dates.get((row.get("doi") or "").strip(), ""))
                for row in matched_papers[:8]
            ],
            "judgment": _trajectory_judgment(name, matched_papers, matched_concepts),
            "missing_link": _trajectory_missing_link(matched_papers, matched_concepts),
            "next_move": _trajectory_next_move(matched_papers, matched_concepts),
        })
        edges.append({"source": center_id, "target": node_id, "type": status})
        for concept in matched_concepts[:3]:
            edges.append({
                "source": node_id,
                "target": f"concept-{_trajectory_slug(concept.get('name', ''))}",
                "type": "concept",
            })
        for paper in matched_papers[:2]:
            edges.append({
                "source": node_id,
                "target": f"paper-{_trajectory_slug(paper.get('doi') or paper.get('title', ''))}",
                "type": "paper",
            })

    trajectories.sort(key=lambda item: (item["status"] != "rising", -item["evidence_count"], item["name"]))
    research_arcs = _build_research_arcs(trajectories, papers, concepts, digest_dates)
    story_groups = [
        build_evolution_story([arc], deltas=deltas)
        for arc in research_arcs
    ]
    story = story_groups[0] if story_groups else build_evolution_story([], deltas=deltas)
    return {
        "center": {
            "id": center_id,
            "name": "ScholarHound",
            "subtitle": "Research-question evolution over the private judgment kernel",
        },
        "updated_at": date.today().isoformat(),
        "story_direction_id": story["id"],
        "story_direction": story["direction"],
        "story_groups": story_groups,
        "story_nodes": story["nodes"],
        "story_edges": story["edges"],
        "trajectories": research_arcs,
        "topic_trajectories": trajectories,
        "edges": edges,
    }


def build_abstract_graph(row: dict | None, workflow: dict | None = None) -> dict:
    """Build a small graph from causal/workflow/abstract text for the modal."""
    if not row:
        return {"nodes": [], "edges": []}

    reasoning = _paper_reasoning(row)
    causal = reasoning.get("causal") or {}
    nodes = []

    def add(node_id: str, label: str, text: str):
        text = _strip_markup(text)
        if text and text.lower() not in {"none", "not stated"}:
            nodes.append({"id": node_id, "label": label, "text": text[:220]})

    add("question", "Question", causal.get("question", ""))
    add("constraint", "Constraint", causal.get("constraint", ""))
    add("input", "Input State", causal.get("input_state", ""))
    add("transformation", "Transformation", causal.get("transformation", ""))
    add("output", "Output State", causal.get("output_state", ""))
    add("outcome", "Outcome", causal.get("outcome", ""))

    if not nodes and workflow:
        add("question", "Question", workflow.get("question", ""))
        add("hypothesis", "Hypothesis", workflow.get("hypothesis", ""))
        add("method", "Method", workflow.get("key_method") or workflow.get("design", ""))
        add("result", "Result", workflow.get("key_result", ""))
        add("gap", "Gap", workflow.get("gap", ""))

    abstract = _strip_markup(row.get("abstract") or "")
    if not nodes and abstract:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if s.strip()]
        add("context", "Context", sentences[0] if sentences else "")
        add("approach", "Approach", _first_sentence_matching(
            sentences,
            [r"\bwe (report|develop|demonstrate|show|present|use|designed?)\b", r"\busing\b|\bvia\b|\bthrough\b|\bby\b"],
        ))
        add("finding", "Finding", _first_sentence_matching(
            sentences,
            [r"\bshow(?:s|ed)?\b|\bdemonstrat(?:e|ed|es)\b|\breveal(?:s|ed)?\b|\bachiev(?:e|ed|es)\b|\benabl(?:e|ed|es)\b"],
        ))
        add("implication", "Implication", _first_sentence_matching(
            sentences,
            [r"\bprovid(?:e|es|ed)\b|\bsuggest(?:s|ed)?\b|\bcould\b|\bwill\b|\bmay\b"],
        ))

    seen = set()
    unique_nodes = []
    for node in nodes:
        key = (node["label"], node["text"])
        if key not in seen:
            unique_nodes.append(node)
            seen.add(key)

    edges = [
        {"source": unique_nodes[i]["id"], "target": unique_nodes[i + 1]["id"]}
        for i in range(len(unique_nodes) - 1)
    ]
    return {"nodes": unique_nodes, "edges": edges}


def build_toc_graph(image_url: str = "", source: str = "") -> dict:
    """Normalize TOC graph metadata for the digest modal."""
    image_url = (image_url or "").strip()
    if not image_url:
        return {"status": "missing", "image_url": "", "source": ""}
    return {"status": "available", "image_url": image_url, "source": source or "paper-record"}


def _clean_digest_inline(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text or "")
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    return _strip_markup(text)


def _parse_score_fields(line: str) -> dict:
    scores = {}
    for k, key in [("R:", "relevance"), ("N:", "novelty"), ("B:", "bridge"), ("T:", "trajectory")]:
        m = re.search(re.escape(k) + r"(\d+)/10", line)
        if m:
            scores[key] = int(m.group(1))
    m = re.search(r"→\s*\*?\*?([\d.]+)/10", line)
    if m:
        scores["final"] = float(m.group(1))
    return scores


def _score_value(line: str) -> int | None:
    m = re.search(r"(\d+)/10", line)
    return int(m.group(1)) if m else None


def _parse_flat_digest_entries(body: str, tier: str, db_papers: dict) -> list[dict]:
    entries = []
    current = []
    for line in body.splitlines():
        if line.startswith("- ") and current:
            entries.append("\n".join(current))
            current = [line]
        elif line.strip():
            current.append(line)
    if current:
        entries.append("\n".join(current))

    papers = []
    for entry in entries:
        lines = [line.rstrip() for line in entry.splitlines() if line.strip()]
        if not lines:
            continue
        metadata_start = next(
            (i for i, line in enumerate(lines) if re.match(r"\s*(DOI|Scores|Why|Action):", line.strip())),
            len(lines),
        )
        title_line = " ".join(line.strip() for line in lines[:metadata_start]).lstrip("- ").strip()
        if not title_line or title_line.startswith("No "):
            continue

        title_raw = title_line
        journal = ""
        if " — *" in title_line:
            title_raw, journal_raw = title_line.split(" — *", 1)
            journal = _clean_digest_inline(journal_raw.rstrip("*").strip())
        title = _clean_digest_inline(title_raw)

        doi = ""
        scores = {}
        why = ""
        action = ""
        for line in lines[metadata_start:]:
            clean = line.strip()
            if clean.startswith("DOI:"):
                doi_match = re.search(r"\[(10\.\S+?)\]", clean)
                doi = doi_match.group(1) if doi_match else clean.replace("DOI:", "").strip()
            elif clean.startswith("Scores:"):
                scores.update(_parse_score_fields(clean))
            elif clean.startswith("Why:"):
                why = clean.replace("Why:", "", 1).strip()
            elif clean.startswith("Action:"):
                action = clean.replace("Action:", "", 1).strip()

        db_row = db_papers.get(doi)
        toc_img = ((db_row or {}).get("toc_image_url") or "").strip()
        papers.append({
            "title": title, "journal": journal, "doi": doi,
            "scores": scores, "tier": tier,
            "why": why, "connection": "", "weakness": "", "action": action,
            "paper_type": "", "judgment_mode": "",
            "problem_class": "", "novelty_type": "",
            "strategic_value": "", "concept_support": "",
            "workflow": {},
            "abstract": _strip_markup((db_row or {}).get("abstract", "")),
            "abstract_graph": build_abstract_graph(db_row),
            "toc_image": toc_img,
            "toc_graph": build_toc_graph(toc_img),
        })
    return papers


def parse_digest(date_str):
    path = os.path.join(VAULT_PATH, "daily", f"{date_str}-signals.md")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read()

    sections = {}
    current_section = None
    current_body = []

    for line in content.split("\n"):
        if line.startswith("## "):
            if current_section:
                sections[current_section] = "\n".join(current_body).strip()
            current_section = line[3:].strip()
            current_body = []
        elif current_section:
            current_body.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_body).strip()

    db_papers = {}
    try:
        db = get_db()
        db_papers = {p.get("doi"): p for p in db.get_all_papers() if p.get("doi")}
    except Exception:
        db_papers = {}

    # Extract papers
    papers = []
    for sec_name in ["HIGH PRIORITY", "IMPORTANT", "POTENTIAL", "WATCHLIST", "LOW PRIORITY"]:
        tier_name = sec_name.replace(" ", "_")
        body = sections.get(sec_name, "")
        if not body or body.lower().startswith("no "):
            continue

        if sec_name == "LOW PRIORITY" or not re.search(r"^###\s+", body, re.MULTILINE):
            papers.extend(_parse_flat_digest_entries(body, tier_name, db_papers))
            continue  # skip normal ### parsing for flat sections

        # Split into individual papers — all other tiers use "### Title" format
        paper_blocks = re.split(r"\n### ", body)
        for block in paper_blocks:
            if not block.strip():
                continue
            if not block.startswith("###"):
                block = "### " + block

            lines = block.strip().split("\n")
            title = lines[0].replace("### ", "").strip()

            # Skip non-paper blocks
            if not title or title.startswith("No ") or "papers" in title.lower():
                continue

            journal = doi = ""
            scores = {}
            why = conn = action = weakness = ""
            concept_name = ""
            paper_type = judgment_mode = ""
            problem_class = novelty_type = strategic_value = concept_support = ""

            for i, line in enumerate(lines[1:]):
                line = line.strip()
                if line.startswith("- **Journal:**"):
                    journal = line.replace("- **Journal:**", "").strip()
                elif line.startswith("- **Paper Type:**"):
                    paper_type = line.replace("- **Paper Type:**", "").strip()
                elif line.startswith("- **Judgment Mode:**"):
                    judgment_mode = line.replace("- **Judgment Mode:**", "").strip()
                elif line.startswith("- **Problem Class:**"):
                    problem_class = line.replace("- **Problem Class:**", "").strip()
                elif line.startswith("- **Novelty Type:**"):
                    novelty_type = line.replace("- **Novelty Type:**", "").strip()
                elif line.startswith("- **Strategic Value:**"):
                    strategic_value = line.replace("- **Strategic Value:**", "").strip()
                elif line.startswith("- **Supports:**"):
                    concept_support = line.replace("- **Supports:**", "").strip()
                elif line.startswith("- **DOI:**"):
                    doi_match = re.search(r"\[(10\.\S+)\]", line)
                    if doi_match: doi = doi_match.group(1)
                    else: doi = line.replace("- **DOI:**", "").strip()
                elif line.startswith("- **Relevance:**"):
                    m = re.search(r"(\d+)/10", line)
                    if m: scores["relevance"] = int(m.group(1))
                elif line.startswith("- **Novelty:**"):
                    m = re.search(r"(\d+)/10", line)
                    if m: scores["novelty"] = int(m.group(1))
                elif line.startswith("- **Bridge:**"):
                    m = re.search(r"(\d+)/10", line)
                    if m: scores["bridge"] = int(m.group(1))
                elif line.startswith("- **Trajectory Influence:**") or line.startswith("- **Trajectory:**"):
                    score = _score_value(line)
                    if score is not None: scores["trajectory"] = score
                elif line.startswith("- **Concept Support:**"):
                    score = _score_value(line)
                    if score is not None: scores["concept_support"] = score
                elif line.startswith("- **Final Score:**"):
                    m = re.search(r"([\d.]+)/10", line)
                    if m: scores["final"] = float(m.group(1))
                elif line.startswith("- **Scores:**"):
                    # Condensed format: "Scores: R:x/10 N:x/10 B:x/10 T:x/10 → x.x/10"
                    for k, key in [("R:", "relevance"), ("N:", "novelty"), ("B:", "bridge"), ("T:", "trajectory")]:
                        m = re.search(re.escape(k) + r"(\d+)/10", line)
                        if m: scores[key] = int(m.group(1))
                    m = re.search(r"→\s*\*?\*?([\d.]+)/10", line)
                    if m: scores["final"] = float(m.group(1))

            # Extract rich content sections by re-parsing the full block
            full_block = "\n".join(lines)
            for label, field in [("Why it matters", "why"), ("Potential connection to my work", "conn"), ("Connection", "conn"), ("Weakness / caution", "weakness"), ("Weakness", "weakness")]:
                pattern = rf"\*\*{label}:\*\*\s*\n?(.*?)(?=\n\*\*|\n\n\*\*|\n- \*\*|$)"
                m = re.search(pattern, full_block, re.DOTALL)
                if m:
                    val = m.group(1).strip()
                    # Clean bullets: remove leading "- " from each line
                    val = re.sub(r'^\s*-\s+', '', val, flags=re.MULTILINE).strip()
                    if field == "why": why = val
                    elif field == "conn": conn = val
                    elif field == "weakness": weakness = val

            # Action
            m_action = re.search(r"\*\*Action:\*\*\s*(.+)", full_block)
            if m_action: action = m_action.group(1).strip()

            # Extract paper workflow from markdown. Older digests used the
            # "Experimental Workflow" heading, so keep that parser compatible.
            workflow = {}
            wf_match = re.search(r'\*\*📋 (?:Paper|Experimental) Workflow:\*\*\s*\n(.*?)(?=\n\n|$)', full_block, re.DOTALL)
            if wf_match:
                wf_text = wf_match.group(1)
                for fname in ['Question', 'Hypothesis', 'Design', 'Key Method', 'Key Result', 'Gap']:
                    m = re.search(rf'-\s+\*\*{fname}:\*\*\s*(.*)', wf_text)
                    if m:
                        wf_key = fname.lower().replace(' ', '_')
                        workflow[wf_key] = m.group(1).strip()

            # TOC graph URL — from digest markdown or DB lookup
            toc_img = ""
            toc_source = ""
            db_row = db_papers.get(doi)
            img_match = re.search(r'!\[.*?\]\((https?://\S+)\)', full_block)
            if img_match:
                toc_img = img_match.group(1)
                toc_source = "digest"
            elif db_row:
                toc_img = (db_row.get("toc_image_url") or "").strip()
                if toc_img:
                    toc_source = "paper-record"

            papers.append({
                "title": title,
                "journal": journal,
                "doi": doi,
                "scores": scores,
                "tier": tier_name,
                "why": why,
                "connection": conn,
                "weakness": weakness,
                "action": action,
                "paper_type": paper_type,
                "judgment_mode": judgment_mode,
                "problem_class": problem_class,
                "novelty_type": novelty_type,
                "strategic_value": strategic_value,
                "concept_support": concept_support,
                "workflow": workflow,
                "abstract": _strip_markup((db_row or {}).get("abstract", "")),
                "abstract_graph": build_abstract_graph(db_row, workflow),
                "toc_image": toc_img,
                "toc_graph": build_toc_graph(toc_img, toc_source),
            })

    # Extract concepts
    concepts = []
    concept_body = sections.get("CONCEPT FEED", "")
    if concept_body:
        concept_blocks = concept_body.split("### Concept:")
        for block in concept_blocks[1:]:
            c = {"name": block.strip().split("\n")[0].strip()}
            concepts.append(c)

    # Summary
    summary = {}
    summary_body = sections.get("DAILY SUMMARY", "")
    if summary_body:
        for line in summary_body.split("\n"):
            m = re.match(r"- \*\*(.+?):\*\*\s*(.+)", line)
            if m:
                summary[m.group(1).strip()] = m.group(2).strip()

    return {
        "date": date_str,
        "papers": papers,
        "concepts": concepts,
        "summary": summary,
    }


def _safe_db_call(db, method: str, default, *args, **kwargs):
    fn = getattr(db, method, None)
    if not fn:
        return default
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _build_judgment_kernel_summary(
    db,
    all_papers: list[dict] | None = None,
    concepts: list[dict] | None = None,
    digest_dates: dict[str, str] | None = None,
    include_story: bool = True,
) -> dict:
    """Compose persisted DB state into the internal judgment-kernel summary."""
    all_papers = all_papers if all_papers is not None else _safe_db_call(db, "get_all_papers", [])
    concepts = concepts if concepts is not None else _safe_db_call(db, "get_concept_momentum", [], min_appearances=1)
    digest_dates = digest_dates or {}
    frameworks = [
        fw for fw in _safe_db_call(db, "get_frameworks", [])
        if _is_displayable_framework(fw)
    ]
    constraints = _safe_db_call(db, "get_constraints", [])
    deltas = _safe_db_call(db, "get_deltas", [])
    verifications = _safe_db_call(db, "get_verifications", [])
    experiments = _safe_db_call(db, "get_experiments", [])
    trajectories = _safe_db_call(db, "get_trajectories", [])
    kernel_objects = _safe_db_call(db, "get_kernel_objects", [])
    revision_events = _safe_db_call(db, "get_kernel_object_events", [], limit=25)
    kernel_tasks = _safe_db_call(db, "get_kernel_tasks", [], limit=50)
    memory_summary = _safe_db_call(db, "get_memory_summary", {
        "beliefs": [],
        "rejected": [],
        "contradictions": [],
        "decisions": [],
        "next_actions": [],
    })
    story_groups = []
    if include_story:
        try:
            story_groups = build_trajectory_map(
                all_papers,
                concepts,
                digest_dates=digest_dates,
                trajectory_rows=trajectories,
                deltas=deltas,
            ).get("story_groups", [])
        except Exception:
            story_groups = []
    return build_judgment_kernel_summary(
        papers=all_papers,
        concepts=concepts,
        frameworks=frameworks,
        constraints=constraints,
        deltas=deltas,
        verifications=verifications,
        experiments=experiments,
        trajectories=trajectories,
        memory_summary=memory_summary,
        story_groups=story_groups,
        kernel_objects=kernel_objects,
        revision_events=revision_events,
        kernel_tasks=kernel_tasks,
    )


# ── API routes ──────────────────────────────────────────────────────────────
@app.get("/api/dashboard")
def api_dashboard():
    db = get_db()
    all_papers = db.get_all_papers()
    tiers = {}
    for p in all_papers:
        t = p.get("signal_tier", "LOW_PRIORITY")
        tiers[t] = tiers.get(t, 0) + 1

    concepts = db.get_concept_momentum(min_appearances=1)
    gaining = [c for c in concepts if c["appearances"] >= 2]
    local_sources = db.get_local_sources()

    digests = list_digests()
    latest_ranked_digest = _latest_digest_with_ranked_papers(digests)
    digest_dates = _digest_paper_dates(digests)

    return {
        "total_papers": len(all_papers),
        "high_priority": tiers.get("HIGH_PRIORITY", 0),
        "important": tiers.get("IMPORTANT", 0),
        "potential": tiers.get("POTENTIAL", 0),
        "watchlist": tiers.get("WATCHLIST", 0),
        "concepts_tracked": len(concepts),
        "concepts_gaining": len(gaining),
        "local_sources": len(local_sources),
        "digest_count": len(digests),
        "journals": len(CONFIG.get("journals", [])),
        "latest_digest": digests[0] if digests else None,
        "latest_ranked_digest": latest_ranked_digest,
        "latest_digest_has_ranked_papers": bool(digests and latest_ranked_digest == digests[0]),
        "tier_lists": {
            "HIGH_PRIORITY": _tier_papers(all_papers, "HIGH_PRIORITY", digest_dates, _DASHBOARD_TIER_PREVIEW_LIMIT),
            "IMPORTANT": _tier_papers(all_papers, "IMPORTANT", digest_dates, _DASHBOARD_TIER_PREVIEW_LIMIT),
            "POTENTIAL": _tier_papers(all_papers, "POTENTIAL", digest_dates, _DASHBOARD_TIER_PREVIEW_LIMIT),
            "WATCHLIST": _tier_papers(all_papers, "WATCHLIST", digest_dates, _DASHBOARD_TIER_PREVIEW_LIMIT),
        },
        "gaining_concepts": [
            {"name": c["name"], "appearances": c["appearances"], "weight": c.get("trajectory_weight", "medium")}
            for c in gaining[:6]
        ],
    }


@app.get("/api/trajectory-map")
def api_trajectory_map():
    digests = list_digests()
    cache_key = _trajectory_map_cache_key(digests)
    if _TRAJECTORY_MAP_CACHE.get("key") == cache_key and _TRAJECTORY_MAP_CACHE.get("data"):
        return _TRAJECTORY_MAP_CACHE["data"]

    db = get_db()
    all_papers = db.get_all_papers()
    concepts = db.get_concept_momentum(min_appearances=1)
    digest_dates = _digest_paper_dates(digests)
    data = build_trajectory_map(
        all_papers,
        concepts,
        digest_dates=digest_dates,
        trajectory_rows=db.get_trajectories(),
        deltas=db.get_deltas(),
    )
    _TRAJECTORY_MAP_CACHE["key"] = cache_key
    _TRAJECTORY_MAP_CACHE["data"] = data
    return data


@app.get("/api/state-changes")
def api_state_changes(limit: int = 50):
    path = _state_change_log_path()
    events, issues = validate_event_log(path)
    if issues:
        return {
            "ok": False,
            "path": str(path),
            "event_count": 0,
            "issues": [issue.format() for issue in issues],
            "events": [],
            "state_counts": {},
            "action_queue": [],
            "parser_boundary": {
                "parser_layer": "LLM candidate JSON",
                "acceptance_layer": "psil.state_change validator",
                "source_of_truth": "kernel/state_changes.jsonl",
                "rendered_view": "kernel/state_changes.md",
            },
        }

    requested_limit = max(1, min(int(limit or 50), 200))
    recent_events = list(reversed(events))[:requested_limit]
    confidence_counts: dict[str, int] = {}
    privacy_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for event in events:
        confidence = event.get("confidence", "unknown")
        privacy = event.get("privacy_sensitivity", "unknown")
        source_kind = (event.get("source") or {}).get("kind", "unknown")
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        privacy_counts[privacy] = privacy_counts.get(privacy, 0) + 1
        source_counts[source_kind] = source_counts.get(source_kind, 0) + 1

    return {
        "ok": True,
        "path": str(path),
        "event_count": len(events),
        "issues": [],
        "events": recent_events,
        "latest_event": recent_events[0] if recent_events else None,
        "state_counts": _state_change_status_counts(events),
        "action_queue": _state_change_action_queue(events),
        "confidence_counts": confidence_counts,
        "privacy_counts": privacy_counts,
        "source_counts": source_counts,
        "state_layers": list(STATE_FIELDS),
        "parser_boundary": {
            "parser_layer": "LLM candidate JSON",
            "acceptance_layer": "psil.state_change validator",
            "source_of_truth": "kernel/state_changes.jsonl",
            "rendered_view": "kernel/state_changes.md",
        },
    }


@app.get("/api/judgment-kernel")
def api_judgment_kernel():
    db = get_db()
    return _build_judgment_kernel_summary(
        db,
        digest_dates=_digest_paper_dates(list_digests()),
        include_story=True,
    )


@app.post("/api/judgment-kernel/sync")
def api_judgment_kernel_sync():
    db = get_db()
    summary = _build_judgment_kernel_summary(
        db,
        digest_dates=_digest_paper_dates(list_digests()),
        include_story=True,
    )
    result = materialize_kernel_objects(db, summary)
    object_refreshed = _build_judgment_kernel_summary(
        db,
        digest_dates=_digest_paper_dates(list_digests()),
        include_story=True,
    )
    task_result = materialize_kernel_tasks(db, object_refreshed)
    refreshed = _build_judgment_kernel_summary(
        db,
        digest_dates=_digest_paper_dates(list_digests()),
        include_story=True,
    )
    return {"ok": True, "materialized": result, "tasks": task_result, "summary": refreshed}


@app.get("/api/kernel/objects")
def api_kernel_objects(object_type: str = None, status: str = None):
    db = get_db()
    return {"objects": db.get_kernel_objects(object_type=object_type, status=status)}


def _kernel_task_queue_rank(task: dict) -> tuple[float, float]:
    try:
        metadata = json.loads(task.get("metadata") or "{}")
    except Exception:
        metadata = {}
    rank = _score_number(metadata.get("queue_rank"), 999999)
    priority = _score_number(task.get("priority"), 0)
    return (rank, -priority)


def _get_kernel_task_queue(db, status: str = None, task_type: str = None, limit: int = 100):
    requested_limit = int(limit or 100)
    raw_limit = max(requested_limit, 500)
    tasks = db.get_kernel_tasks(status=status, task_type=task_type, limit=raw_limit)
    return sorted(tasks, key=_kernel_task_queue_rank)[:requested_limit]


@app.get("/api/kernel/tasks")
def api_kernel_tasks(status: str = None, task_type: str = None, limit: int = 100):
    db = get_db()
    return {"tasks": _get_kernel_task_queue(db, status=status, task_type=task_type, limit=limit)}


@app.get("/api/kernel/v3/contested-evidence")
def api_v3_contested_evidence():
    from psil.v3_kernel import get_contested_evidence_queue

    items = get_contested_evidence_queue(_v3_kernel_dir())
    return {"items": items, "count": len(items)}


@app.get("/api/kernel/v3/pending-evidence")
def api_v3_pending_evidence():
    from psil.v3_kernel import get_pending_evidence_queue

    items = get_pending_evidence_queue(_v3_kernel_dir())
    return {"items": items, "count": len(items)}


def _v3_belief_display_id(belief: dict) -> str:
    concepts = {str(item).strip().lower() for item in belief.get("linked_concepts", [])}
    domain = str(belief.get("domain") or "").lower()
    if {"oect", "ev"}.issubset(concepts):
        return "B-OECT"
    if "research-os" in domain or "belief-kernel" in concepts:
        return "B-KERNEL"
    suffix = str(belief.get("id") or "belief").rsplit("_", 1)[-1][:6].upper()
    return f"B-{suffix}"


def _v3_evidence_projection(evidence: dict, relation: str) -> dict:
    return {
        "id": evidence.get("id", ""),
        "relation": relation,
        "title": evidence.get("title", ""),
        "summary": evidence.get("summary", ""),
        "strength": evidence.get("evidence_strength", "unknown"),
        "reliability": evidence.get("reliability", ""),
        "source_type": evidence.get("source_type", ""),
        "source_ref": evidence.get("source_ref", ""),
        "created_at": evidence.get("created_at", ""),
    }


def _v3_relation_ids(belief: dict, evidence: list[dict], relation: str) -> list[str]:
    belief_id = str(belief.get("id") or "")
    field_map = {
        "support": ("evidence_ids", "supports_beliefs"),
        "challenge": ("contra_evidence_ids", "challenges_beliefs"),
        "contest": ("contested_evidence_ids", "contests_beliefs"),
        "pending": ("pending_evidence_ids", "pending_beliefs"),
        "neutral": ("neutral_evidence_ids", "neutral_beliefs"),
    }
    belief_field, evidence_field = field_map[relation]
    ids = [str(item) for item in belief.get(belief_field, []) if item]
    ids.extend(
        str(item.get("id"))
        for item in evidence
        if belief_id in (item.get(evidence_field) or []) and item.get("id")
    )
    return list(dict.fromkeys(ids))


@app.get("/api/kernel/v3/belief-map")
def api_v3_belief_map():
    """Project append-only V3 state into a read-only ScholarHound view."""
    from psil.v3_kernel import object_path, read_jsonl

    kernel_dir = _v3_kernel_dir()
    beliefs = read_jsonl(object_path(kernel_dir, "beliefs"))
    evidence = read_jsonl(object_path(kernel_dir, "evidence"))
    revisions = read_jsonl(object_path(kernel_dir, "revisions"))
    review_requests = read_jsonl(object_path(kernel_dir, "human_review_requests"))
    evidence_by_id = {str(item.get("id")): item for item in evidence if item.get("id")}

    belief_payloads = []
    for belief in sorted(
        beliefs,
        key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
        reverse=True,
    ):
        relation_payloads = {}
        for relation in ("support", "challenge", "contest", "pending", "neutral"):
            relation_payloads[relation] = [
                _v3_evidence_projection(evidence_by_id[evidence_id], relation)
                for evidence_id in _v3_relation_ids(belief, evidence, relation)
                if evidence_id in evidence_by_id
            ]
        belief_revisions = sorted(
            [item for item in revisions if item.get("belief_id") == belief.get("id")],
            key=lambda item: str(item.get("created_at") or ""),
            reverse=True,
        )
        belief_payloads.append({
            "id": belief.get("id", ""),
            "display_id": _v3_belief_display_id(belief),
            "title": belief.get("title", ""),
            "claim": belief.get("claim", ""),
            "domain": belief.get("domain", ""),
            "status": belief.get("status", ""),
            "confidence": belief.get("confidence", 0),
            "entrenchment": belief.get("entrenchment", 0),
            "updated_at": belief.get("updated_at", ""),
            "created_at": belief.get("created_at", ""),
            "linked_concepts": belief.get("linked_concepts", []),
            "linked_constraints": belief.get("linked_constraints", []),
            "questions": belief.get("linked_questions", []),
            "provenance": belief.get("provenance", {}),
            "last_revision_id": belief.get("last_revision_id", ""),
            "evidence": relation_payloads,
            "evidence_counts": {
                relation: len(items) for relation, items in relation_payloads.items()
            },
            "revision_history": [
                {
                    "id": item.get("id", ""),
                    "action": item.get("action", ""),
                    "reason": item.get("reason", ""),
                    "old_confidence": item.get("old_confidence"),
                    "new_confidence": item.get("new_confidence"),
                    "old_entrenchment": item.get("old_entrenchment"),
                    "new_entrenchment": item.get("new_entrenchment"),
                    "triggering_evidence_ids": item.get("triggering_evidence_ids", []),
                    "human_override_id": item.get("human_override_id", ""),
                    "created_at": item.get("created_at", ""),
                }
                for item in belief_revisions
            ],
        })

    safe_review_requests = []
    for item in review_requests:
        if str(item.get("status") or "").lower() not in {"open", "pending"}:
            continue
        reviewer_payload = item.get("reviewer_payload") or {}
        safe_review_requests.append({
            "id": item.get("id") or item.get("request_id") or "",
            "status": item.get("status", ""),
            "priority": item.get("priority", ""),
            "request_type": item.get("request_type", ""),
            "question": reviewer_payload.get("question") or item.get("question", ""),
            "belief_ref": reviewer_payload.get("belief_ref") or item.get("belief_ref", ""),
            "source_ref": reviewer_payload.get("source_ref") or item.get("source_ref", ""),
            "target_state_layer": item.get("target_state_layer", ""),
            "created_at": item.get("created_at", ""),
        })

    unresolved = sum(
        belief["evidence_counts"]["contest"] + belief["evidence_counts"]["pending"]
        for belief in belief_payloads
    )
    return {
        "schema_id": "scholarhound_belief_map_view_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kernel_version": "V3",
        "semantics": "literature state, not probability of scientific truth",
        "counts": {
            "beliefs": len(belief_payloads),
            "evidence": len(evidence),
            "revisions": len(revisions),
            "unresolved_evidence": unresolved,
            "open_review_requests": len(safe_review_requests),
        },
        "beliefs": belief_payloads,
        "review_requests": safe_review_requests,
    }


@app.post("/api/kernel/v3/pending-evidence/reclassify")
async def api_v3_reclassify_pending_evidence(request: Request):
    from psil.v3_kernel import reclassify_pending_evidence

    data = await request.json()
    belief, revision = reclassify_pending_evidence(
        _v3_kernel_dir(),
        belief_id=data.get("belief_id", ""),
        evidence_id=data.get("evidence_id", ""),
        relation=data.get("relation", ""),
        reason=data.get("reason", "Pending evidence reclassified through V3 API."),
        human_override_id=data.get("human_override_id", ""),
    )
    return {"belief": belief, "revision": revision}


@app.post("/api/kernel/tasks/sync")
def api_kernel_tasks_sync():
    db = get_db()
    summary = _build_judgment_kernel_summary(
        db,
        digest_dates=_digest_paper_dates(list_digests()),
        include_story=True,
    )
    result = materialize_kernel_tasks(db, summary)
    return {"ok": True, "materialized": result, "tasks": _get_kernel_task_queue(db, limit=100)}


@app.post("/api/kernel/tasks/{task_key}/status")
async def api_revise_kernel_task(task_key: str, request: Request):
    data = await request.json()
    db = get_db()
    task = db.revise_kernel_task(
        task_key,
        status=data.get("status", "done"),
        reason=data.get("reason", ""),
        actor=data.get("actor", "human"),
    )
    if not task:
        return JSONResponse({"ok": False, "error": "not_found", "task_key": task_key}, status_code=404)
    return {"ok": True, "task": task}


@app.post("/api/kernel/objects")
async def api_create_kernel_object(request: Request):
    data = await request.json()
    db = get_db()
    obj = db.upsert_kernel_object(
        object_type=data.get("object_type", "claim"),
        title=data.get("title", ""),
        statement=data.get("statement", ""),
        status=data.get("status", "candidate"),
        confidence=data.get("confidence", 0),
        entrenchment=data.get("entrenchment", 0),
        source_type=data.get("source_type", "manual"),
        source_ref=data.get("source_ref", ""),
        evidence=data.get("evidence", {}),
        metadata=data.get("metadata", {}),
        object_key=data.get("object_key", ""),
    )
    return {"ok": True, "object": obj}


@app.post("/api/kernel/objects/{object_key}/revise")
async def api_revise_kernel_object(object_key: str, request: Request):
    data = await request.json()
    db = get_db()
    result = apply_kernel_revision(
        db,
        object_key=object_key,
        action=data.get("action", ""),
        reason=data.get("reason", ""),
        evidence_delta=data.get("evidence_delta", {}),
        actor=data.get("actor", "human"),
    )
    if not result.get("ok"):
        return JSONResponse(result, status_code=404 if result.get("error") == "not_found" else 400)
    return result


@app.get("/api/digests")
def api_digests():
    digests = list_digests()
    selected = _latest_digest_with_ranked_papers(digests)
    return {
        "digests": digests,
        "latest_digest": digests[0] if digests else "",
        "selected_digest": selected,
        "latest_digest_has_ranked_papers": bool(digests and selected == digests[0]),
    }


@app.get("/api/digest/{date_str}")
def api_digest(date_str: str):
    data = parse_digest(date_str)
    if data is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return data


@app.get("/api/paper")
def api_paper(doi: str = Query(...)):
    db = get_db()
    target = doi.strip()
    for row in db.get_all_papers():
        if (row.get("doi") or "").strip() == target:
            digest_date = _digest_paper_dates(list_digests()).get(target, "")
            return _paper_payload(row, digest_date)
    return JSONResponse({"error": "Not found"}, status_code=404)


@app.get("/api/concepts")
def api_concepts():
    db = get_db()
    concepts = db.get_concept_momentum(min_appearances=1)
    return {
        "concepts": [
            {
                "name": c["name"],
                "appearances": c["appearances"],
                "status": c["status"],
                "weight": c.get("trajectory_weight", "medium"),
                "connection": c.get("connection", ""),
                "missing_link": c.get("missing_link", ""),
                "opportunity": c.get("opportunity", ""),
            }
            for c in concepts
        ]
    }


@app.get("/api/identity")
def api_identity():
    ident = load_identity()
    return {
        "current_core": ident.current_core,
        "emerging_directions": ident.emerging_directions,
        "long_term_vision": ident.long_term_vision,
        "trajectory_topics": ident.trajectory_influence_topics,
        "last_updated": ident.last_updated,
        "concepts_tracked": len(ident.concept_momentum),
    }


@app.get("/api/history")
def api_history():
    db = get_db()
    logs = db.get_recent_logs(limit=60)
    return {"logs": [dict(l) for l in logs]}


@app.get("/api/patterns")
def api_patterns():
    db = get_db()
    patterns = db.get_logic_patterns()
    return {"patterns": patterns}


@app.get("/api/frameworks")
def api_frameworks(v2: bool = False):
    db = get_db()
    frameworks = [fw for fw in db.get_frameworks() if _is_displayable_framework(fw)]
    if v2:
        from psil.compress import score_framework_v2
        for fw in frameworks:
            constraints = db.get_constraints(framework_name=fw["framework_name"])
            v2 = score_framework_v2(fw, len(constraints))
            fw.update(v2)
        frameworks.sort(key=lambda f: f.get("v2_score", 0), reverse=True)
    return {"frameworks": frameworks}


@app.get("/api/constraints")
def api_constraints(framework: str = None):
    db = get_db()
    constraints = db.get_constraints(framework_name=framework)
    # Attach predictions to each constraint
    for c in constraints:
        c["predictions"] = db.get_predictions(constraint_id=c["id"])
    return {"constraints": constraints}


@app.get("/api/local-sources")
def api_local_sources(bucket: str = None):
    db = get_db()
    sources = db.get_local_sources(bucket=bucket)
    buckets = {}
    for source in sources:
        b = source.get("bucket") or "uncategorized"
        buckets[b] = buckets.get(b, 0) + 1
    return {"sources": sources, "buckets": buckets, "total": len(sources)}


@app.get("/api/gaps")
def api_gaps():
    db = get_db()
    from psil.compress import run_gap_audit
    return run_gap_audit(db)


@app.get("/api/connections")
def api_connections():
    db = get_db()
    from psil.compress import discover_connections
    connections = discover_connections(db)
    return {"connections": connections}


@app.get("/api/verification")
def api_verification():
    db = get_db()
    summary = db.get_verification_summary()
    verifications = db.get_verifications()
    return {"summary": summary, "verifications": verifications}


@app.get("/api/experiments")
def api_experiments(framework: str = None):
    db = get_db()
    experiments = db.get_experiments(framework_name=framework)
    return {"experiments": experiments}


@app.get("/api/constraint-radar")
def api_constraint_radar():
    """Full constraint radar: frameworks → constraints → predictions → experiments."""
    db = get_db()
    from psil.compress import score_framework_v2

    frameworks = [fw for fw in db.get_frameworks() if _is_displayable_framework(fw)]
    radar = []
    for fw in frameworks:
        constraints = db.get_constraints(framework_name=fw["framework_name"])
        for c in constraints:
            c["predictions"] = db.get_predictions(constraint_id=c["id"])
        experiments = db.get_experiments(framework_name=fw["framework_name"])
        v2 = score_framework_v2(fw, len(constraints))
        radar.append({
            "framework": fw,
            "v2_score": v2,
            "constraints": constraints,
            "experiments": experiments,
        })
    radar.sort(key=lambda r: r["v2_score"]["v2_score"], reverse=True)
    return {"radar": radar}


def _is_displayable_framework(fw: dict) -> bool:
    text_fields = [
        "description",
        "covered_patterns",
        "core_logic",
        "worldview_shift",
        "suggested_experiment",
    ]
    score_fields = [
        "compression_score",
        "novelty_score",
        "predictive_power",
        "falsifiability",
        "actionability",
        "transferability",
        "taste_fit",
    ]
    has_text = any((fw.get(field) or "").strip() for field in text_fields)
    has_score = any(float(fw.get(field) or 0) > 0 for field in score_fields)
    return bool((fw.get("framework_name") or "").strip() and (has_text or has_score))


@app.get("/api/deltas")
def api_deltas():
    db = get_db()
    deltas = db.get_deltas()
    return {"deltas": deltas}


@app.get("/api/causal")
def api_causal():
    db = get_db()
    papers = db.get_papers_with_causal(days_back=7)
    return {"papers": papers}


@app.get("/api/memory")
def api_memory(item_type: str = None, status: str = None):
    db = get_db()
    items = db.get_memory(item_type=item_type, status=status)
    return {"items": items}


@app.get("/api/memory/summary")
def api_memory_summary():
    db = get_db()
    return db.get_memory_summary()


@app.post("/api/memory/approve")
async def api_approve(request: Request):
    """Human gate: approve or reject a concept/framework/pattern."""
    data = await request.json()
    item_type = data.get("type", "")
    item_name = data.get("name", "")
    status = data.get("status", "approved")
    reason = data.get("reason", "")
    evidence = data.get("evidence_strength", "")
    projects = data.get("affected_projects", "")

    db = get_db()

    # Kernel learning: if user overrides a kernel decision, learn from it
    paper_doi = data.get("paper_doi", "")
    kernel_decision = data.get("kernel_decision", "")
    if paper_doi and kernel_decision:
        from psil.kernel import learn_from_override
        # Get paper reasoning from DB
        all_p = db.get_all_papers()
        reasoning = {}
        for p in all_p:
            if p.get("doi") == paper_doi:
                import json
                reasoning = json.loads(p.get("llm_reasoning", "{}")) if p.get("llm_reasoning") else {}
                break
        if reasoning:
            learn_from_override(paper_doi, status, kernel_decision, reasoning, db)

    db.upsert_memory(item_type, item_name, status=status,
                     reason=reason, evidence_strength=evidence,
                     affected_projects=projects)

    # Also update the source object's status
    if item_type == "framework":
        db.set_framework_status(item_name, status)
    elif item_type == "concept":
        db.set_concept_status(item_name, status)
        # AGM: user approval → boost epistemic entrenchment
        if status == "approved":
            current = db.get_entrenchment(item_name)
            db.set_entrenchment(item_name, min(10, current + 3))
        elif status == "rejected":
            db.set_entrenchment(item_name, max(0, db.get_entrenchment(item_name) - 2))

    return {"ok": True, "type": item_type, "name": item_name, "status": status}


@app.get("/api/graph")
def api_graph():
    """Return paper-signal connection graph from all digests."""
    digests = list_digests()
    nodes = []
    links = []
    paper_ids = set()
    signal_ids = {}  # signal_name → cid
    signal_counts = {}  # signal_name → count

    for d in reversed(digests[-14:]):
        path = os.path.join(VAULT_PATH, "daily", f"{d}-signals.md")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            content = f.read()

        sections = re.split(r"\n## ", content)
        for sec in sections:
            paper_blocks = re.split(r"\n### ", sec)
            if not paper_blocks:
                continue
            for block in paper_blocks[1:]:  # skip section header
                lines = block.strip().split("\n")
                title = lines[0].strip()
                if not title or title.startswith("No "):
                    continue

                signals = []
                for line in lines:
                    m = re.search(r"\*?\*?Matched(?: Signals)?:\s*\*?\*?(.+)", line)
                    if m:
                        signals = [s.strip().lower().replace('-', ' ').replace('<i>','').replace('</i>','') for s in m.group(1).split(",")]
                        break

                if not signals:
                    continue

                doi_match = re.search(r"10\.\S+", "\n".join(lines))
                pid = f"p_{doi_match.group(0)}" if doi_match else f"p_{title[:40].replace(' ','_')}"

                if pid in paper_ids:
                    continue
                paper_ids.add(pid)

                nodes.append({
                    "id": pid,
                    "name": title[:60],
                    "type": "paper",
                    "tier": sec.split("\n")[0] if "HIGH" in sec[:30] or "IMPORTANT" in sec[:30] or "POTENTIAL" in sec[:30] or "WATCHLIST" in sec[:30] else "OTHER",
                    "score": 0,
                    "journal": "",
                })

                for sig in signals:
                    if not sig: continue
                    if sig not in signal_ids:
                        cid = f"s_{sig.replace(' ', '_')[:40]}"
                        signal_ids[sig] = cid
                        signal_counts[sig] = 0
                    signal_counts[sig] += 1
                    links.append({"source": pid, "target": signal_ids[sig], "type": "match"})

    # Add signal nodes (only those with 2+ appearances to reduce clutter)
    for sig, cid in signal_ids.items():
        if signal_counts[sig] >= 1:
            nodes.append({
                "id": cid,
                "name": sig,
                "type": "signal",
                "weight": "high" if signal_counts[sig] >= 3 else ("medium" if signal_counts[sig] >= 2 else "low"),
                "appearances": signal_counts[sig],
            })

    return {"nodes": nodes, "links": links}


# ── Journal management ──────────────────────────────────────────────────────
@app.get("/api/journals")
def api_journals():
    return {"journals": CONFIG.get("journals", [])}

@app.post("/api/journals")
async def api_add_journal(request: Request):
    body = await request.json()
    name = body.get("name", "").strip()
    issn = body.get("issn", "").strip()
    rss = body.get("rss", "").strip()
    if not name or not issn:
        return JSONResponse({"error": "name and issn required"}, status_code=400)

    journals = CONFIG.get("journals", [])
    if any(j["name"].lower() == name.lower() for j in journals):
        return JSONResponse({"error": "Journal already exists"}, status_code=409)

    entry = {"name": name, "issn": issn}
    if rss: entry["rss"] = rss
    journals.append(entry)
    _save_config()
    return {"ok": True, "journals": journals}

@app.delete("/api/journals/{name}")
def api_delete_journal(name: str):
    journals = CONFIG.get("journals", [])
    new_list = [j for j in journals if j["name"].lower() != name.lower()]
    if len(new_list) == len(journals):
        return JSONResponse({"error": "Not found"}, status_code=404)
    CONFIG["journals"] = new_list
    _save_config()
    return {"ok": True, "journals": new_list}


def _save_config():
    import yaml
    config_path = Path(
        os.getenv("SCHOLARHOUND_CONFIG") or Path.cwd() / "config.yaml"
    )
    with open(config_path, "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ── Benchmark API ───────────────────────────────────────────────────────────
def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _full_72_replacement_paths() -> tuple[Path, Path]:
    if (
        BENCHMARK_PACKET_PATH != DEFAULT_BENCHMARK_PACKET_PATH
        or BENCHMARK_SELECTION_LOG_PATH != DEFAULT_BENCHMARK_SELECTION_LOG_PATH
    ):
        return BENCHMARK_PACKET_PATH, BENCHMARK_SELECTION_LOG_PATH
    return BENCHMARK_BOUNDARY_PACKET_PATH, BENCHMARK_BOUNDARY_SELECTION_LOG_PATH


def _benchmark_packet_configs() -> dict[str, dict]:
    full_packet_path, full_selection_log_path = _full_72_replacement_paths()
    return {
        "full_72": {
            "label": "Review set B",
            "description": "Claim and abstract pairs for relation review.",
            "path": full_packet_path,
            "selection_log_path": full_selection_log_path,
        },
        "calibration_24": {
            "label": "Review set A",
            "description": "Claim and abstract pairs for relation review.",
            "path": BENCHMARK_CALIBRATION_PACKET_PATH,
            "selection_log_path": BENCHMARK_CALIBRATION_SELECTION_LOG_PATH,
        },
        "dispute_gateflip_28": {
            "label": "Review set C",
            "description": "Claim and abstract pairs for relation review.",
            "path": BENCHMARK_DISPUTE_PACKET_PATH,
            "selection_log_path": BENCHMARK_DISPUTE_SELECTION_LOG_PATH,
        },
    }


def _normalize_benchmark_packet_key(value: str | None = "") -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return BENCHMARK_DEFAULT_PACKET_KEY
    alias_map = {
        "full": "full_72",
        "full-72": "full_72",
        "full_72": "full_72",
        "benchmark": "full_72",
        "boundary": "full_72",
        "boundary-4": "full_72",
        "boundary_4": "full_72",
        "claim-boundary": "full_72",
        "claim_boundary": "full_72",
        "scifact-boundary": "full_72",
        "scifact_boundary": "full_72",
        "calibration": "calibration_24",
        "calibration-24": "calibration_24",
        "calibration_24": "calibration_24",
        "expert-calibration": "calibration_24",
        "expert_calibration": "calibration_24",
        "dispute": "dispute_gateflip_28",
        "dispute-28": "dispute_gateflip_28",
        "dispute_28": "dispute_gateflip_28",
        "gateflip": "dispute_gateflip_28",
        "gate-flip": "dispute_gateflip_28",
        "dispute-gateflip": "dispute_gateflip_28",
        "dispute_gateflip": "dispute_gateflip_28",
        "dispute_gateflip_28": "dispute_gateflip_28",
    }
    for key, config in _benchmark_packet_configs().items():
        alias_map[key.lower()] = key
        alias_map[str(config["path"].stem).lower()] = key
    return alias_map.get(raw)


def _benchmark_packet_key_from_query(request: Request) -> str | None:
    return _normalize_benchmark_packet_key(
        request.query_params.get("packet_key")
        or request.query_params.get("packet")
    )


def _benchmark_packet_key_from_body(body: dict | None) -> str | None:
    payload = body or {}
    return _normalize_benchmark_packet_key(
        payload.get("packet_key")
        or payload.get("packet")
    )


def _benchmark_packet_config(
    packet_key: str | None = BENCHMARK_DEFAULT_PACKET_KEY,
) -> dict:
    normalized = _normalize_benchmark_packet_key(packet_key)
    configs = _benchmark_packet_configs()
    if not normalized or normalized not in configs:
        raise ValueError("unknown benchmark packet")
    return configs[normalized]


def _load_benchmark_items(
    packet_key: str | None = BENCHMARK_DEFAULT_PACKET_KEY,
) -> list[dict]:
    packet_path = _benchmark_packet_config(packet_key)["path"]
    if not packet_path.exists():
        return []
    data = json.loads(packet_path.read_text(encoding="utf-8"))
    items = data.get("items", []) if isinstance(data, dict) else data
    safe_items = []
    raw_items = items if isinstance(items, list) else []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        safe_items.append({
            "id": str(item.get("id") or ""),
            "belief_id": str(item.get("belief_id") or ""),
            "belief": str(item.get("belief") or ""),
            "title": _strip_markup(str(item.get("title") or "")),
            "abstract": _strip_markup(str(item.get("abstract") or "")),
            "doi": str(item.get("doi") or ""),
            "journal": str(item.get("journal") or ""),
            "pub_date": str(item.get("pub_date") or ""),
        })
    return [item for item in safe_items if item["id"] and item["belief"] and item["title"]]


def _benchmark_item_type(
    packet_key: str | None,
    item_id: str,
) -> str:
    packet_path = _benchmark_packet_config(packet_key)["path"]
    if not packet_path.exists():
        return ""
    try:
        data = json.loads(packet_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    items = data.get("items", []) if isinstance(data, dict) else data
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        if str(item.get("id") or "") == item_id:
            return str(item.get("item_type") or "").strip()[:80]
    return ""


def _benchmark_packet_id(
    packet_key: str | None = BENCHMARK_DEFAULT_PACKET_KEY,
) -> str:
    return _benchmark_packet_config(packet_key)["path"].stem


def _load_benchmark_selection_log(
    packet_key: str | None = BENCHMARK_DEFAULT_PACKET_KEY,
) -> dict:
    selection_log_path = _benchmark_packet_config(packet_key)["selection_log_path"]
    if not selection_log_path.exists():
        return {}
    try:
        data = json.loads(selection_log_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _append_jsonl_record(path: Path, record: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record


def _read_feedback_records() -> list[dict]:
    return _read_jsonl_records(BENCHMARK_FEEDBACK_PATH)


def _read_progress_records() -> list[dict]:
    return _read_jsonl_records(BENCHMARK_PROGRESS_PATH)


def _read_reviewer_feedback_records(identity: dict) -> list[dict]:
    records = _read_feedback_records()
    if not identity.get("benchmark_eligible", True):
        records.extend(_read_jsonl_records(BENCHMARK_TEST_FEEDBACK_PATH))
    return records


def _read_reviewer_progress_records(identity: dict) -> list[dict]:
    records = _read_progress_records()
    if not identity.get("benchmark_eligible", True):
        records.extend(_read_jsonl_records(BENCHMARK_TEST_PROGRESS_PATH))
    return records


def _read_benchmark_eligible_feedback_records() -> list[dict]:
    return [
        record
        for record in _read_feedback_records()
        if _record_is_benchmark_eligible(record)
    ]


def _feedback_id(record: dict) -> str:
    digest = hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"hfb_{digest[:16]}"


def _progress_event_id(record: dict) -> str:
    digest = hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"hbp_{digest[:16]}"


def _safe_elapsed_ms(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _normalize_reviewer_identity(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _reviewer_id(value: str) -> str:
    normalized = _normalize_reviewer_identity(value)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"reviewer_{digest[:16]}"


def _load_benchmark_reviewer_policies() -> dict[str, dict]:
    if not BENCHMARK_REVIEWER_POLICY_PATH.exists():
        return {}
    try:
        payload = json.loads(BENCHMARK_REVIEWER_POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    reviewers = payload.get("reviewers", []) if isinstance(payload, dict) else []
    policies: dict[str, dict] = {}
    for reviewer in reviewers:
        if not isinstance(reviewer, dict):
            continue
        identity = _normalize_reviewer_identity(reviewer.get("identity", ""))
        if not identity:
            continue
        policies[identity] = {
            "reviewer_role": str(reviewer.get("role") or "participant"),
            "benchmark_eligible": bool(reviewer.get("benchmark_eligible", True)),
            "benchmark_exclusion_reason": str(
                reviewer.get("exclusion_reason") or ""
            ),
        }
    return policies


def _apply_benchmark_reviewer_policy(identity: dict) -> dict:
    normalized = _normalize_reviewer_identity(identity.get("reviewer", ""))
    policy = _load_benchmark_reviewer_policies().get(normalized, {})
    return {
        **identity,
        "reviewer_role": str(policy.get("reviewer_role") or "participant"),
        "benchmark_eligible": bool(policy.get("benchmark_eligible", True)),
        "benchmark_exclusion_reason": str(
            policy.get("benchmark_exclusion_reason") or ""
        ),
    }


def _record_is_benchmark_eligible(record: dict) -> bool:
    identity = _apply_benchmark_reviewer_policy({
        "reviewer": str(record.get("reviewer") or ""),
    })
    return bool(
        record.get("benchmark_eligible", True)
        and identity["benchmark_eligible"]
    )


def _record_reviewer_id(record: dict) -> str:
    explicit = str(record.get("reviewer_id") or "").strip()
    if explicit:
        return explicit
    reviewer = str(record.get("reviewer") or "").strip()
    return _reviewer_id(reviewer) if reviewer else ""


def _request_is_local(request: Request) -> bool:
    host = (request.headers.get("host") or "").split(":", 1)[0].lower()
    client_host = str(request.client.host if request.client else "").lower()
    return host in {"localhost", "127.0.0.1", "::1"} or client_host in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


def _request_is_secure(request: Request) -> bool:
    forwarded = (request.headers.get("x-forwarded-proto") or "").split(",", 1)[0].strip()
    return forwarded == "https" or request.url.scheme == "https"


def _benchmark_session_secret(request: Request) -> str:
    configured = os.getenv("SCHOLARHOUND_BENCHMARK_SESSION_SECRET", "").strip()
    if configured:
        return configured
    if _request_is_local(request):
        return "scholarhound-local-benchmark-session-v1"
    return ""


def _encode_benchmark_session(identity: dict, secret: str) -> str:
    payload = {
        "reviewer_id": identity["reviewer_id"],
        "reviewer": identity["reviewer"],
        "expertise": str(identity.get("expertise") or ""),
        "auth_source": identity["auth_source"],
        "expires_at": int(datetime.now(timezone.utc).timestamp()) + BENCHMARK_AUTH_MAX_AGE,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    signature = hmac.new(secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def _decode_benchmark_session(token: str, secret: str) -> dict | None:
    if not token or not secret or "." not in token:
        return None
    encoded, signature = token.rsplit(".", 1)
    expected = hmac.new(
        secret.encode("utf-8"),
        encoded.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        padding = "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(encoded + padding).decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        expires_at = int(payload.get("expires_at") or 0)
    except (TypeError, ValueError):
        return None
    if expires_at <= int(datetime.now(timezone.utc).timestamp()):
        return None
    reviewer = str(payload.get("reviewer") or "").strip()
    reviewer_id = str(payload.get("reviewer_id") or "").strip()
    if not reviewer or not reviewer_id:
        return None
    return {
        "reviewer_id": reviewer_id,
        "reviewer": reviewer,
        "expertise": str(payload.get("expertise") or ""),
        "auth_source": str(payload.get("auth_source") or "access_code"),
    }


def _cloudflare_benchmark_identity(request: Request) -> dict | None:
    email = str(
        request.headers.get("cf-access-authenticated-user-email")
        or request.headers.get("Cf-Access-Authenticated-User-Email")
        or ""
    ).strip()
    if not email:
        return None
    return _apply_benchmark_reviewer_policy({
        "reviewer_id": _reviewer_id(email),
        "reviewer": email,
        "expertise": "",
        "auth_source": "cloudflare_access",
    })


def _benchmark_identity(request: Request) -> dict | None:
    cloudflare_identity = _cloudflare_benchmark_identity(request)
    if cloudflare_identity:
        return cloudflare_identity
    secret = _benchmark_session_secret(request)
    token = request.cookies.get(BENCHMARK_AUTH_COOKIE, "")
    identity = _decode_benchmark_session(token, secret)
    return _apply_benchmark_reviewer_policy(identity) if identity else None


def _benchmark_auth_error(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "authenticated": False,
            "login_required": True,
            "cloudflare_access_supported": True,
            "access_code_login_available": bool(
                os.getenv("SCHOLARHOUND_BENCHMARK_ACCESS_CODE", "").strip()
                or _request_is_local(request)
            ),
        },
        status_code=401,
        headers=NO_STORE_HEADERS,
    )


def _benchmark_packet_sha256(
    packet_key: str | None = BENCHMARK_DEFAULT_PACKET_KEY,
) -> str:
    packet_path = _benchmark_packet_config(packet_key)["path"]
    if not packet_path.exists():
        return ""
    return hashlib.sha256(packet_path.read_bytes()).hexdigest()


def _ordered_benchmark_items(
    items: list[dict],
    *,
    packet_id: str,
    reviewer_id: str,
) -> list[dict]:
    return sorted(
        items,
        key=lambda item: hashlib.sha256(
            f"{packet_id}|{reviewer_id}|{item['id']}".encode("utf-8")
        ).hexdigest(),
    )


def _feedback_for_reviewer(
    records: list[dict],
    *,
    packet_id: str,
    reviewer_id: str,
) -> list[dict]:
    return [
        record
        for record in records
        if record.get("packet") == packet_id
        and _record_reviewer_id(record) == reviewer_id
    ]


def _progress_for_reviewer(
    records: list[dict],
    *,
    packet_id: str,
    reviewer_id: str,
) -> list[dict]:
    return [
        record
        for record in records
        if record.get("packet") == packet_id
        and str(record.get("reviewer_id") or "") == reviewer_id
    ]


def _reviewer_progress_projection(
    items: list[dict],
    feedback: list[dict],
    progress_events: list[dict],
) -> dict:
    item_ids = [item["id"] for item in items]
    latest_by_item: dict[str, dict] = {}
    for record in feedback:
        item_id = str(record.get("item_id") or "")
        if item_id in item_ids:
            latest_by_item[item_id] = record

    processed_ids = {
        item_id for item_id, record in latest_by_item.items()
        if str(record.get("relation") or "")
    }
    completed_ids = {
        item_id for item_id, record in latest_by_item.items()
        if record.get("relation") != "skip"
    }
    skipped_ids = {
        item_id for item_id, record in latest_by_item.items()
        if record.get("relation") == "skip"
    }

    last_item_id = ""
    for event in reversed(progress_events):
        candidate = str(event.get("item_id") or "")
        if candidate in item_ids:
            last_item_id = candidate
            break

    resume_item_id = ""
    if last_item_id and last_item_id not in processed_ids:
        resume_item_id = last_item_id
    elif len(processed_ids) < len(item_ids):
        start = item_ids.index(last_item_id) + 1 if last_item_id in item_ids else 0
        for offset in range(len(item_ids)):
            candidate = item_ids[(start + offset) % len(item_ids)]
            if candidate not in processed_ids:
                resume_item_id = candidate
                break

    return {
        "submitted_count": len(feedback),
        "processed_item_count": len(processed_ids),
        "completed_item_count": len(completed_ids),
        "skipped_item_count": len(skipped_ids),
        "processed_ids": sorted(processed_ids),
        "completed_ids": sorted(completed_ids),
        "skipped_ids": sorted(skipped_ids),
        "last_item_id": last_item_id,
        "resume_item_id": resume_item_id,
        "packet_complete": len(processed_ids) == len(item_ids) and bool(item_ids),
    }


def _append_progress_event(
    identity: dict,
    *,
    packet_id: str,
    packet_sha256: str,
    event_type: str,
    item_id: str,
    session_run_id: str,
    projection: dict,
) -> dict:
    record = {
        "schema_id": "human_benchmark_progress_event_v1",
        "created_at": _now_utc(),
        "session_id": BENCHMARK_SESSION_ID,
        "session_run_id": session_run_id[:120],
        "packet": packet_id,
        "packet_sha256": packet_sha256,
        "reviewer_id": identity["reviewer_id"],
        "reviewer": identity["reviewer"],
        "auth_source": identity["auth_source"],
        "reviewer_role": identity["reviewer_role"],
        "benchmark_eligible": identity["benchmark_eligible"],
        "benchmark_exclusion_reason": identity["benchmark_exclusion_reason"],
        "event_type": event_type,
        "item_id": item_id,
        "processed_item_count": projection.get("processed_item_count", 0),
        "completed_item_count": projection.get("completed_item_count", 0),
        "skipped_item_count": projection.get("skipped_item_count", 0),
    }
    record["event_id"] = _progress_event_id(record)
    path = (
        BENCHMARK_PROGRESS_PATH
        if identity["benchmark_eligible"]
        else BENCHMARK_TEST_PROGRESS_PATH
    )
    return _append_jsonl_record(path, record)


class BenchmarkFeedbackConflict(Exception):
    def __init__(self, existing: dict):
        super().__init__("benchmark item already has feedback for this reviewer")
        self.existing = existing


def _feedback_payload_matches(existing: dict, record: dict) -> bool:
    text_fields = (
        "relation",
        "confidence",
        "assertions",
        "covered",
        "gap",
        "reason",
        "expertise",
    )
    if any(
        str(existing.get(field) or "") != str(record.get(field) or "")
        for field in text_fields
    ):
        return False
    return (existing.get("flags") or []) == (record.get("flags") or [])


def _append_feedback_record_once(record: dict) -> tuple[dict, bool]:
    path = (
        BENCHMARK_FEEDBACK_PATH
        if record.get("benchmark_eligible", True)
        else BENCHMARK_TEST_FEEDBACK_PATH
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing_records = []
        for line in handle:
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                existing_records.append(parsed)
        for existing in existing_records:
            if (
                existing.get("packet") == record.get("packet")
                and str(existing.get("item_id") or "") == record.get("item_id")
                and _record_reviewer_id(existing) == record.get("reviewer_id")
            ):
                if _feedback_payload_matches(existing, record):
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    return existing, False
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                raise BenchmarkFeedbackConflict(existing)
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record, True


@app.get("/api/benchmark/auth")
def api_benchmark_auth(request: Request):
    identity = _benchmark_identity(request)
    if not identity:
        return _benchmark_auth_error(request)
    return JSONResponse(
        {
            "authenticated": True,
            "reviewer": identity,
            "externally_managed": identity["auth_source"] == "cloudflare_access",
        },
        headers=NO_STORE_HEADERS,
    )


@app.post("/api/benchmark/login")
def api_benchmark_login(request: Request, body: dict):
    identity = _cloudflare_benchmark_identity(request)
    response: JSONResponse
    if identity:
        response = JSONResponse(
            {
                "authenticated": True,
                "reviewer": identity,
                "externally_managed": True,
            },
            headers=NO_STORE_HEADERS,
        )
        return response

    reviewer = str((body or {}).get("reviewer") or "").strip()[:120]
    expertise = str((body or {}).get("expertise") or "").strip()[:120]
    access_code = str((body or {}).get("access_code") or "")
    expected_code = os.getenv("SCHOLARHOUND_BENCHMARK_ACCESS_CODE", "").strip()
    if not reviewer:
        return JSONResponse({"error": "reviewer identity required"}, status_code=400)
    if expected_code:
        if not hmac.compare_digest(access_code, expected_code):
            return JSONResponse({"error": "invalid access code"}, status_code=401)
    elif not _request_is_local(request):
        return JSONResponse(
            {"error": "access-code login is not configured"},
            status_code=503,
        )

    secret = _benchmark_session_secret(request)
    if not secret:
        return JSONResponse(
            {"error": "benchmark session secret is not configured"},
            status_code=503,
        )
    identity = _apply_benchmark_reviewer_policy({
        "reviewer_id": _reviewer_id(reviewer),
        "reviewer": reviewer,
        "expertise": expertise,
        "auth_source": "access_code",
    })
    response = JSONResponse(
        {
            "authenticated": True,
            "reviewer": identity,
            "externally_managed": False,
        },
        headers=NO_STORE_HEADERS,
    )
    response.set_cookie(
        BENCHMARK_AUTH_COOKIE,
        _encode_benchmark_session(identity, secret),
        max_age=BENCHMARK_AUTH_MAX_AGE,
        httponly=True,
        secure=_request_is_secure(request),
        samesite="lax",
        path="/",
    )
    return response


@app.post("/api/benchmark/logout")
def api_benchmark_logout(request: Request):
    identity = _benchmark_identity(request)
    response = JSONResponse(
        {
            "ok": True,
            "externally_managed": bool(
                identity and identity.get("auth_source") == "cloudflare_access"
            ),
        },
        headers=NO_STORE_HEADERS,
    )
    response.delete_cookie(BENCHMARK_AUTH_COOKIE, path="/")
    return response


@app.get("/api/benchmark/session")
def api_benchmark_session(request: Request):
    identity = _benchmark_identity(request)
    if not identity:
        return _benchmark_auth_error(request)

    packet_key = _benchmark_packet_key_from_query(request)
    if not packet_key:
        return JSONResponse(
            {"error": "unknown benchmark packet"},
            status_code=400,
            headers=NO_STORE_HEADERS,
        )
    packet_config = _benchmark_packet_config(packet_key)
    packet_id = _benchmark_packet_id(packet_key)
    packet_sha256 = _benchmark_packet_sha256(packet_key)
    items = _ordered_benchmark_items(
        _load_benchmark_items(packet_key),
        packet_id=packet_id,
        reviewer_id=identity["reviewer_id"],
    )
    items = [
        {**item, "presentation_index": index}
        for index, item in enumerate(items, start=1)
    ]
    feedback = _feedback_for_reviewer(
        _read_reviewer_feedback_records(identity),
        packet_id=packet_id,
        reviewer_id=identity["reviewer_id"],
    )
    progress_events = _progress_for_reviewer(
        _read_reviewer_progress_records(identity),
        packet_id=packet_id,
        reviewer_id=identity["reviewer_id"],
    )
    progress = _reviewer_progress_projection(items, feedback, progress_events)
    belief_counts: dict[str, int] = {}
    for item in items:
        belief_id = item["belief_id"]
        belief_counts[belief_id] = belief_counts.get(belief_id, 0) + 1
    selection_log = _load_benchmark_selection_log(packet_key)
    order_sha256 = hashlib.sha256(
        "|".join(item["id"] for item in items).encode("utf-8")
    ).hexdigest()
    payload = {
        "session": {
            "id": BENCHMARK_SESSION_ID,
            "title": "ScholarHound Human Benchmark Feedback",
            "mode": "blind_human_feedback",
            "packet_key": packet_key,
            "packet": packet_id,
            "packet_label": packet_config["label"],
            "packet_description": packet_config["description"],
            "packet_sha256": packet_sha256,
            "kernel_prediction_visibility": "withheld",
            "item_count": len(items),
            "belief_counts": belief_counts,
            "relation_options": sorted(BENCHMARK_RELATION_OPTIONS),
            "confidence_options": sorted(BENCHMARK_CONFIDENCE_OPTIONS),
            "seed": selection_log.get("base_seed", ""),
            "presentation_order": "reviewer_deterministic_shuffle_v1",
            "presentation_order_sha256": order_sha256,
        },
        "reviewer": identity,
        "progress": progress,
        "items": items,
    }
    return JSONResponse(payload, headers=NO_STORE_HEADERS)


@app.post("/api/benchmark/progress")
def api_benchmark_progress(request: Request, body: dict):
    identity = _benchmark_identity(request)
    if not identity:
        return _benchmark_auth_error(request)

    packet_key = _benchmark_packet_key_from_body(body)
    if not packet_key:
        return JSONResponse(
            {"error": "unknown benchmark packet"},
            status_code=400,
            headers=NO_STORE_HEADERS,
        )
    packet_id = _benchmark_packet_id(packet_key)
    packet_sha256 = _benchmark_packet_sha256(packet_key)
    event_type = str((body or {}).get("event_type") or "").strip().lower()
    if event_type not in {"session_open", "cursor", "session_close"}:
        return JSONResponse({"error": "invalid progress event"}, status_code=400)
    item_id = str((body or {}).get("item_id") or "").strip()
    items = _ordered_benchmark_items(
        _load_benchmark_items(packet_key),
        packet_id=packet_id,
        reviewer_id=identity["reviewer_id"],
    )
    item_ids = {item["id"] for item in items}
    if event_type == "cursor" and item_id not in item_ids:
        return JSONResponse({"error": "unknown benchmark item"}, status_code=404)
    if item_id and item_id not in item_ids:
        item_id = ""

    feedback = _feedback_for_reviewer(
        _read_reviewer_feedback_records(identity),
        packet_id=packet_id,
        reviewer_id=identity["reviewer_id"],
    )
    progress_events = _progress_for_reviewer(
        _read_reviewer_progress_records(identity),
        packet_id=packet_id,
        reviewer_id=identity["reviewer_id"],
    )
    projection = _reviewer_progress_projection(items, feedback, progress_events)
    event = _append_progress_event(
        identity,
        packet_id=packet_id,
        packet_sha256=packet_sha256,
        event_type=event_type,
        item_id=item_id,
        session_run_id=str((body or {}).get("session_run_id") or ""),
        projection=projection,
    )
    return JSONResponse(
        {"ok": True, "event_id": event["event_id"], "progress": projection},
        headers=NO_STORE_HEADERS,
    )


@app.post("/api/benchmark/feedback")
def api_benchmark_feedback(request: Request, body: dict):
    identity = _benchmark_identity(request)
    if not identity:
        return _benchmark_auth_error(request)

    packet_key = _benchmark_packet_key_from_body(body)
    if not packet_key:
        return JSONResponse(
            {"error": "unknown benchmark packet"},
            status_code=400,
            headers=NO_STORE_HEADERS,
        )
    packet_id = _benchmark_packet_id(packet_key)
    packet_sha256 = _benchmark_packet_sha256(packet_key)
    items = _ordered_benchmark_items(
        _load_benchmark_items(packet_key),
        packet_id=packet_id,
        reviewer_id=identity["reviewer_id"],
    )
    item_ids = {item["id"] for item in items}
    item_id = str((body or {}).get("item_id") or "").strip()
    if item_id not in item_ids:
        return JSONResponse({"error": "unknown benchmark item"}, status_code=404)

    relation = str((body or {}).get("relation") or "").strip().lower()
    confidence = str((body or {}).get("confidence") or "").strip().lower()
    if relation not in BENCHMARK_RELATION_OPTIONS:
        return JSONResponse({"error": "invalid relation"}, status_code=400)
    if relation != "skip" and confidence not in BENCHMARK_CONFIDENCE_OPTIONS:
        return JSONResponse({"error": "invalid confidence"}, status_code=400)

    item = next(item for item in items if item["id"] == item_id)
    created_at = _now_utc()
    record = {
        "schema_id": "human_benchmark_feedback_v2",
        "created_at": created_at,
        "timestamp": created_at,
        "session_id": BENCHMARK_SESSION_ID,
        "session_run_id": str((body or {}).get("session_run_id") or "").strip()[:120],
        "packet": packet_id,
        "packet_sha256": packet_sha256,
        "presentation_order": "reviewer_deterministic_shuffle_v1",
        "item_position": next(
            index for index, candidate in enumerate(items, start=1)
            if candidate["id"] == item_id
        ),
        "item_id": item_id,
        "item_type": _benchmark_item_type(packet_key, item_id),
        "belief_id": item["belief_id"],
        "relation": relation,
        "confidence": confidence if relation != "skip" else "",
        "assertions": str((body or {}).get("assertions") or "").strip()[:2000],
        "covered": str((body or {}).get("covered") or "").strip()[:2000],
        "gap": str((body or {}).get("gap") or "").strip()[:2000],
        "reason": str((body or {}).get("reason") or "").strip()[:1000],
        "annotator_id": identity["reviewer_id"],
        "reviewer_id": identity["reviewer_id"],
        "reviewer": identity["reviewer"],
        "auth_source": identity["auth_source"],
        "reviewer_role": identity["reviewer_role"],
        "benchmark_eligible": identity["benchmark_eligible"],
        "benchmark_exclusion_reason": identity["benchmark_exclusion_reason"],
        "expertise": str(
            (body or {}).get("expertise")
            or identity.get("expertise")
            or ""
        ).strip()[:120],
        "client_started_at": str((body or {}).get("client_started_at") or "").strip()[:40],
        "elapsed_ms": _safe_elapsed_ms((body or {}).get("elapsed_ms")),
        "flags": [
            str(flag)
            for flag in (
                (body or {}).get("flags", [])
                if isinstance((body or {}).get("flags", []), list)
                else []
            )
            if str(flag).strip()
        ][:8],
        "kernel_prediction_visibility": "withheld",
    }
    record["feedback_id"] = _feedback_id(record)
    try:
        saved_record, created = _append_feedback_record_once(record)
    except BenchmarkFeedbackConflict as exc:
        return JSONResponse(
            {
                "error": "feedback already recorded for this reviewer and item",
                "feedback_id": exc.existing.get("feedback_id", ""),
                "item_id": item_id,
            },
            status_code=409,
            headers=NO_STORE_HEADERS,
        )

    feedback = _feedback_for_reviewer(
        _read_reviewer_feedback_records(identity),
        packet_id=packet_id,
        reviewer_id=identity["reviewer_id"],
    )
    progress_events = _progress_for_reviewer(
        _read_reviewer_progress_records(identity),
        packet_id=packet_id,
        reviewer_id=identity["reviewer_id"],
    )
    projection = _reviewer_progress_projection(items, feedback, progress_events)
    if created:
        _append_progress_event(
            identity,
            packet_id=packet_id,
            packet_sha256=packet_sha256,
            event_type="feedback_recorded",
            item_id=item_id,
            session_run_id=record["session_run_id"],
            projection=projection,
        )
    return JSONResponse({
        "ok": True,
        "feedback_id": saved_record["feedback_id"],
        "duplicate": not created,
        "item_id": item_id,
        "relation": relation,
        "confidence": record["confidence"],
        "progress": projection,
    }, headers=NO_STORE_HEADERS)


@app.get("/api/benchmark/stats")
def api_benchmark_stats():
    from psil.benchmark.analyze import get_stats
    return get_stats()

@app.get("/api/benchmark/timeline")
def api_benchmark_timeline():
    from psil.benchmark.analyze import get_concept_timeline
    return get_concept_timeline()

@app.get("/api/benchmark/blindspots")
def api_benchmark_blindspots():
    from psil.benchmark.analyze import get_blind_spots
    return {"blind_spots": get_blind_spots()}


# ── HTML5 frontend ───────────────────────────────────────────────────────────
FROZEN_FRONTEND_SNAPSHOT_PATH = (
    REPO_ROOT
    / "ui_snapshots"
    / "2026-06-13-pre-benchmark-console"
    / "frontend.html"
)
FROZEN_FRONTEND_DEPLOY_PATH = Path(__file__).parent / "frozen_frontend.html"
BELIEF_FRONTEND_PATH = Path(__file__).parent / "scholarhound.html"
REVIEW_FRONTEND_PATH = Path(__file__).parent / "frontend.html"
TRAJECTORY_LOGIC_CSS_PATH = Path(__file__).parent / "trajectory_logic.css"
TRAJECTORY_LOGIC_JS_PATH = Path(__file__).parent / "trajectory_logic.js"


def _frozen_frontend_path() -> Path:
    if FROZEN_FRONTEND_SNAPSHOT_PATH.exists():
        return FROZEN_FRONTEND_SNAPSHOT_PATH
    return FROZEN_FRONTEND_DEPLOY_PATH


def _product_frontend_html() -> str:
    frontend_path = _frozen_frontend_path()
    if not frontend_path.exists():
        return ""
    html = frontend_path.read_text(encoding="utf-8")
    if TRAJECTORY_LOGIC_CSS_PATH.exists():
        css = TRAJECTORY_LOGIC_CSS_PATH.read_text(encoding="utf-8")
        html = html.replace("</head>", f"<style id=\"trajectory-logic-css\">{css}</style></head>")
    if TRAJECTORY_LOGIC_JS_PATH.exists():
        script = TRAJECTORY_LOGIC_JS_PATH.read_text(encoding="utf-8")
        html = html.replace("</body>", f"<script id=\"trajectory-logic-js\">{script}</script></body>")
    return html


@app.get("/", response_class=HTMLResponse)
def index():
    html = _product_frontend_html()
    if html:
        return HTMLResponse(html, headers=NO_STORE_HEADERS)
    return HTMLResponse(
        "<h1>Frozen ScholarHound frontend not found.</h1>",
        headers=NO_STORE_HEADERS,
    )


@app.get("/beliefs", response_class=HTMLResponse)
def belief_console():
    if BELIEF_FRONTEND_PATH.exists():
        return HTMLResponse(BELIEF_FRONTEND_PATH.read_text(), headers=NO_STORE_HEADERS)
    return HTMLResponse(
        "<h1>Belief-state view not found. Ensure psil/scholarhound.html exists.</h1>",
        headers=NO_STORE_HEADERS,
    )


@app.get("/review", response_class=HTMLResponse)
def review_console():
    if REVIEW_FRONTEND_PATH.exists():
        return HTMLResponse(REVIEW_FRONTEND_PATH.read_text(), headers=NO_STORE_HEADERS)
    return HTMLResponse(
        "<h1>Review console not found. Ensure psil/frontend.html exists.</h1>",
        headers=NO_STORE_HEADERS,
    )


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return False
            raise
    return True


def find_available_port(port: int = 8501, host: str = "127.0.0.1",
                        attempts: int = 20) -> int:
    for candidate in range(port, port + attempts):
        if is_port_available(candidate, host=host):
            return candidate
    raise RuntimeError(f"No available port found from {port} to {port + attempts - 1}")


def start(port=8501, auto_port: bool = True):
    import uvicorn
    actual_port = find_available_port(port) if auto_port else port
    if actual_port != port:
        print(f"Port {port} is busy; using {actual_port}.")
    uvicorn.run(app, host="0.0.0.0", port=actual_port)
