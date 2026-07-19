"""ScholarHound V3 belief-centered kernel.

V3 treats papers, reports, and notes as evidence. The durable kernel object is
the belief revision history; current belief state is only a projection of that
history. LLMs may propose records, but these helpers validate, append, and
project state without relying on an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ALLOWED_BELIEF_STATUS = {
    "proposed",
    "active",
    "contested",
    "challenged",
    "contradicted",
    "resolved",
    "archived",
}
ALLOWED_EVIDENCE_SOURCE_TYPE = {
    "paper",
    "experiment",
    "user_observation",
    "report",
    "benchmark",
    "external",
}
ALLOWED_EVIDENCE_STRENGTH = {"weak", "moderate", "strong", "decisive"}
ALLOWED_REVISION_ACTION = {
    "create",
    "update",
    "strengthen",
    "weaken",
    "contest",
    "challenge",
    "contradict",
    "resolve",
    "archive",
    "reopen",
}
ALLOWED_OVERRIDE_TARGET = {
    "paper",
    "belief",
    "evidence",
    "constraint",
    "judgment",
    "trajectory",
}
ALLOWED_OVERRIDE_TYPE = {
    "approve",
    "reject",
    "upgrade",
    "downgrade",
    "merge",
    "split",
    "reframe",
    "ignore",
}
ALLOWED_JUDGMENT_RISK = {"low", "medium", "high", "unknown"}
ALLOWED_CONSTRAINT_TYPE = {
    "scientific",
    "strategic",
    "methodological",
    "publication",
    "product",
}
ALLOWED_CONSTRAINT_STATUS = {
    "proposed",
    "testable",
    "tested",
    "failed",
    "survived",
    "retired",
}
ALLOWED_INTAKE_DECISION = {
    "accept",
    "partial_accept",
    "defer",
    "reject",
    "escalate",
    "request_refresh",
}
ALLOWED_INTAKE_AUTHORIZATION = {
    "not_authorized",
    "intake_record_only",
}
ALLOWED_RESEARCH_DECISION = {
    "no_change",
    "request_action_or_read",
    "open_question",
    "propose_belief_revision",
    "propose_trajectory_update",
}
ALLOWED_RESEARCH_DECISION_LAYER = {
    "source",
    "evidence",
    "belief",
    "question",
    "trajectory",
    "attention",
    "constraint",
    "action",
    "uncertainty",
}
ALLOWED_UNCERTAINTY_LEVEL = {"low", "medium", "high", "unknown"}
ALLOWED_QUESTION_STATUS = {
    "proposed",
    "active",
    "answered",
    "parked",
    "rejected",
    "archived",
}
ALLOWED_QUESTION_REVISION_ACTION = {
    "create",
    "update",
    "activate",
    "answer",
    "park",
    "reject",
    "archive",
    "reopen",
}
ALLOWED_TRAJECTORY_STATUS = {
    "proposed",
    "active",
    "revised",
    "parked",
    "archived",
}
ALLOWED_TRAJECTORY_REVISION_ACTION = {
    "create",
    "update",
    "reinforce",
    "shift",
    "split",
    "merge",
    "park",
    "archive",
}
ALLOWED_ACTION_RECORD_TYPE = {
    "read",
    "refresh",
    "human_review",
    "synthesize",
    "experiment",
    "verify",
    "ignore",
}
ALLOWED_ACTION_RECORD_STATUS = {
    "open",
    "in_progress",
    "done",
    "blocked",
    "cancelled",
}
ALLOWED_ACTION_PRIORITY = {"low", "medium", "high", "urgent"}
ALLOWED_HUMAN_REVIEW_REQUEST_TYPE = {
    "human_read",
    "full_text_review",
    "adjudicate_conflict",
    "grounding_check",
    "relation_review",
}
ALLOWED_HUMAN_REVIEW_STATUS = {
    "open",
    "in_progress",
    "answered",
    "closed",
    "blocked",
    "cancelled",
}
ALLOWED_HUMAN_RESPONSE_OPTION = {
    "support",
    "challenge",
    "neutral",
    "underdetermined",
    "contest",
    "needs_full_text",
    "not_my_expertise",
    "abstract_insufficient",
    "belief_too_broad",
    "paper_ambiguous",
}
MEASURED_OR_TESTED_CLAIM_EVIDENCE = {
    "measured",
    "tested",
    "controlled_experiment",
    "direct_experiment",
    "quantitative_observation",
    "quantitative_readout",
    "benchmark",
    "experiment",
    "experimental_result",
    "measurement",
    "validated",
    "validation",
}
NON_TOUCH_CLAIM_EVIDENCE = {
    "asserted",
    "assumed",
    "speculated",
    "background",
    "reviewed",
    "absent",
    "author_interpretation",
    "hypothesis",
}
OVERCLAIM_FLAG_WEIGHTS = {
    "scope_overclaim": 0.25,
    "causal_overclaim": 0.3,
    "mechanism_overclaim": 0.3,
    "translational_overclaim": 0.35,
    "platform_overclaim": 0.25,
    "sensitivity_overclaim": 0.2,
    "robustness_overclaim": 0.2,
    "benchmark_overclaim": 0.2,
    "unsupported_generalization": 0.25,
    "missing_controls": 0.2,
    "claim_design_mismatch": 0.25,
}
HIGH_OVERCLAIM_RISK = {"high", "severe", "critical"}
MEDIUM_OVERCLAIM_RISK = {"medium", "moderate"}
TRANSLATIONAL_CLAIM_SCOPES = {
    "clinical",
    "diagnostic",
    "diagnostics",
    "disease_diagnosis",
    "disease_readout",
    "translational",
    "patient",
    "in_vivo",
    "real_sample",
}
PROTOTYPE_EVIDENCE_SCOPES = {
    "buffer",
    "spiked",
    "spiked_sample",
    "synthetic_sample",
    "model_sample",
    "prototype",
    "in_vitro",
    "cell_line",
    "single_cell_line",
    "proof_of_concept",
}
PLATFORM_CLAIM_SCOPES = {"platform", "general", "generalizable", "universal", "broad"}
NARROW_EVIDENCE_SCOPES = {
    "single_target",
    "single_analyte",
    "single_material",
    "single_model",
    "one_cell_line",
    "one_disease_model",
    "one_sample_type",
}
INTAKE_CONTEXT_MODULES = (
    "source_provenance",
    "object_grounding",
    "confidence_dynamics",
    "task_reliability_envelope",
    "epistemic_task_probe",
    "semantic_audit",
    "calibration_status",
    "trajectory_proposal",
    "human_feedback_brief",
    "knowledge_transfer",
    "personalization_boundary",
    "closed_loop_validation",
    "cognitive_tooling",
    "prior_trace",
    "active_curriculum",
)
EVIDENCE_DELTAS = {
    "weak": 0.05,
    "moderate": 0.10,
    "strong": 0.20,
    "decisive": 0.35,
}
CONFIDENCE_SOFT_CAP = 0.92
CONFIDENCE_DECISIVE_CAP = 0.97
CONFIDENCE_SOFT_FLOOR = 0.08
CONFIDENCE_DECISIVE_FLOOR = 0.03
ENTRENCHMENT_SOURCE_DELTAS = {
    "experiment": 0.035,
    "benchmark": 0.025,
    "paper": 0.006,
    "report": 0.006,
    "user_observation": 0.008,
    "external": 0.003,
}
ENTRENCHMENT_HUMAN_OVERRIDE_DELTA = 0.08
ENTRENCHMENT_CONFIDENCE_RESISTANCE_K = 2.0

SOURCE_TYPE_STRENGTH_POINTS = {
    "experiment": 2.25,
    "benchmark": 2.25,
    "paper": 1.5,
    "report": 1.0,
    "user_observation": 0.75,
    "external": 0.5,
}
PARSE_BOUNDARY_FIELDS = {
    "source_type": {
        "status": "verified_low_inference",
        "evidence": "parse-boundary H1/H2: 5/5 cross-model agreement",
    },
    "reports_lod": {
        "status": "verified_low_inference",
        "evidence": "parse-boundary H1/H2: 4/5 cross-model agreement",
    },
    "venue_tier": {
        "status": "unverified_candidate",
        "evidence": "not yet admitted to deterministic scoring",
    },
    "primary_research": {
        "status": "unverified_candidate",
        "evidence": "not yet admitted to deterministic scoring",
    },
    "replicated": {
        "status": "unverified_candidate",
        "evidence": "not yet admitted to deterministic scoring",
    },
    "independent_validation": {
        "status": "unverified_candidate",
        "evidence": "not yet admitted to deterministic scoring",
    },
    "has_benchmark": {
        "status": "unverified_candidate",
        "evidence": "not yet admitted to deterministic scoring",
    },
    "has_controls": {
        "status": "unverified_candidate",
        "evidence": "not yet admitted to deterministic scoring",
    },
    "has_quantitative_readout": {
        "status": "unverified_candidate",
        "evidence": "not yet admitted to deterministic scoring",
    },
    "sample_size": {
        "status": "unverified_candidate",
        "evidence": "not yet admitted to deterministic scoring",
    },
    "is_review": {
        "status": "unverified_candidate",
        "evidence": "not yet admitted to deterministic scoring",
    },
    "unvalidated_claim": {
        "status": "unverified_candidate",
        "evidence": "not yet admitted to deterministic scoring",
    },
    "direction": {
        "status": "judgment_heavy",
        "evidence": "parse-boundary H1/H2: 0/5 cross-model agreement",
    },
    "relation": {
        "status": "judgment_heavy",
        "evidence": "requires interpretation of whether evidence supports or challenges a belief",
    },
    "support_type": {
        "status": "judgment_heavy",
        "evidence": "requires interpretation of whether evidence supports or challenges a belief",
    },
    "strategic_value": {
        "status": "judgment_heavy",
        "evidence": "requires project-level judgment beyond objective extraction",
    },
}
RELATION_ALIASES = {
    "support": "support",
    "supports": "support",
    "supported": "support",
    "strengthen": "support",
    "strengthens": "support",
    "positive": "support",
    "for": "support",
    "challenge": "challenge",
    "challenges": "challenge",
    "challenged": "challenge",
    "weaken": "challenge",
    "weakens": "challenge",
    "negative": "challenge",
    "against": "challenge",
    "contradict": "challenge",
    "contradicts": "challenge",
    "contest": "contest",
    "contests": "contest",
    "contested": "contest",
    "mixed": "contest",
    "two-edged": "contest",
    "ambiguous": "underdetermined",
    "under-determined": "underdetermined",
    "underdetermined": "underdetermined",
    "insufficient": "underdetermined",
    "pending": "underdetermined",
    "needs-evidence": "underdetermined",
    "unclear": "unclear",
    "unknown": "unclear",
    "neutral": "neutral",
    "unrelated": "neutral",
    "irrelevant": "neutral",
    "off-topic": "neutral",
    "off_topic": "neutral",
}
RELATION_VALUES = {"support", "challenge", "contest", "underdetermined", "neutral", "unclear"}
OBJECT_FILES = {
    "beliefs": ("beliefs", "beliefs.jsonl"),
    "evidence": ("evidence", "evidence.jsonl"),
    "revisions": ("revisions", "revisions.jsonl"),
    "judgments": ("judgments", "judgments.jsonl"),
    "overrides": ("overrides", "overrides.jsonl"),
    "constraints": ("constraints", "constraints.jsonl"),
    "assessments": ("assessments", "assessments.jsonl"),
    "decisions": ("decisions", "research_judgment_decisions.jsonl"),
    "questions": ("questions", "questions.jsonl"),
    "question_revisions": ("questions", "question_revisions.jsonl"),
    "trajectories": ("trajectories", "trajectories.jsonl"),
    "trajectory_revisions": ("trajectories", "trajectory_revisions.jsonl"),
    "actions": ("actions", "actions.jsonl"),
    "human_review_requests": ("human_review_requests", "human_review_requests.jsonl"),
}


@dataclass
class V3Issue:
    path: str
    message: str

    def format(self) -> str:
        return f"{self.path}: {self.message}"


class V3KernelValidationError(ValueError):
    def __init__(self, issues: list[V3Issue]):
        self.issues = issues
        super().__init__("; ".join(issue.format() for issue in issues))


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_v3_kernel_dir(base_dir: str | Path = ".") -> Path:
    return Path(base_dir) / "kernel" / "v3"


def object_path(kernel_dir: str | Path, object_name: str) -> Path:
    directory, filename = OBJECT_FILES[object_name]
    return Path(kernel_dir) / directory / filename


def ensure_kernel_dirs(kernel_dir: str | Path) -> None:
    kernel_dir = Path(kernel_dir)
    for directory, _filename in OBJECT_FILES.values():
        (kernel_dir / directory).mkdir(parents=True, exist_ok=True)
    (kernel_dir / "schemas").mkdir(parents=True, exist_ok=True)
    (kernel_dir / "exports").mkdir(parents=True, exist_ok=True)


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"{prefix}_{digest[:16]}"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _clamp01(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return round(max(0.0, min(1.0, numeric)), 4)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_boundary_status(field: str) -> dict[str, str]:
    status = PARSE_BOUNDARY_FIELDS.get(field)
    if status:
        return {"field": field, **status}
    return {
        "field": field,
        "status": "unclassified",
        "evidence": "field has not been reviewed by parse-boundary tests",
    }


def attach_parse_boundary_provenance(record: dict[str, Any]) -> dict[str, Any]:
    """Label extracted fields by how much judgment they require."""
    normalized = dict(record)
    field_statuses = {
        field: parse_boundary_status(field)
        for field in sorted(record)
        if field in PARSE_BOUNDARY_FIELDS
    }
    normalized["parse_boundary"] = {
        "method": "parse_boundary_field_provenance_v1",
        "verified_low_inference_fields": sorted(
            field
            for field, status in field_statuses.items()
            if status["status"] == "verified_low_inference"
        ),
        "unverified_fields": sorted(
            field
            for field, status in field_statuses.items()
            if status["status"] == "unverified_candidate"
        ),
        "judgment_heavy_fields": sorted(
            field
            for field, status in field_statuses.items()
            if status["status"] == "judgment_heavy"
        ),
        "fields": field_statuses,
    }
    return normalized


def _normalize_relation(value: Any) -> str:
    if value is None:
        return "unclear"
    normalized = str(value).strip().lower().replace("_", "-")
    if not normalized:
        return "unclear"
    return RELATION_ALIASES.get(normalized, normalized if normalized in RELATION_VALUES else "unclear")


def consensus_evidence_relation(parse_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve candidate evidence directions without conflating conflict and underdetermination."""
    votes = []
    for index, candidate in enumerate(parse_candidates or [], 1):
        raw_relation = (
            candidate.get("relation")
            or candidate.get("direction")
            or candidate.get("support_direction")
            or candidate.get("support_type")
        )
        relation = _normalize_relation(raw_relation)
        votes.append(
            {
                "parser": str(
                    candidate.get("model")
                    or candidate.get("parser")
                    or candidate.get("source")
                    or f"parser_{index}"
                ),
                "raw_relation": raw_relation,
                "relation": relation,
            }
        )

    relations = {vote["relation"] for vote in votes}
    directional_relations = {value for value in relations if value in {"support", "challenge"}}
    has_conflict_vote = any(vote["relation"] == "contest" for vote in votes)
    has_neutral_vote = any(vote["relation"] == "neutral" for vote in votes)
    has_unresolved_vote = any(
        vote["relation"] in {"underdetermined", "unclear"} for vote in votes
    )

    if not votes:
        relation = "underdetermined"
        weak_direction = "unclear"
        adjudication_type = "more_evidence"
        reason = "No parser candidates were supplied; evidence needs more evidence before revision."
    elif directional_relations == {"support"} and not has_unresolved_vote:
        relation = "support"
        weak_direction = "support"
        adjudication_type = "none"
        reason = "All parser candidates agreed this evidence supports the belief."
    elif directional_relations == {"challenge"} and not has_unresolved_vote:
        relation = "challenge"
        weak_direction = "challenge"
        adjudication_type = "none"
        reason = "All parser candidates agreed this evidence challenges the belief."
    elif directional_relations == {"support", "challenge"} or has_conflict_vote:
        relation = "contest"
        weak_direction = "unclear"
        adjudication_type = "human"
        reason = "Parser candidates made conflicting support/challenge claims requiring human adjudication."
    elif not directional_relations and has_neutral_vote:
        relation = "neutral"
        weak_direction = "neutral"
        adjudication_type = "none"
        reason = "Parser candidates found the paper neutral or off-topic for this belief."
    else:
        relation = "underdetermined"
        weak_direction = next(iter(directional_relations), "unclear")
        adjudication_type = "more_evidence"
        reason = "Parser candidates gave an incomplete but non-conflicting signal; mark pending evidence instead of routing to human adjudication."

    return {
        "method": "multi_parser_relation_consensus_v2",
        "relation": relation,
        "contested": relation == "contest",
        "underdetermined": relation == "underdetermined",
        "neutral": relation == "neutral",
        "needs_human": adjudication_type == "human",
        "needs_more_evidence": adjudication_type == "more_evidence",
        "adjudication_type": adjudication_type,
        "weak_direction": weak_direction,
        "votes": votes,
        "reason": reason,
    }


def build_evidence_from_parse_candidates(
    record: dict[str, Any],
    *,
    belief_id: str,
    parse_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build accountable evidence from one or more parser proposals."""
    relation = consensus_evidence_relation(parse_candidates)
    evidence = dict(record)
    evidence["evidence_relation_provenance"] = relation
    evidence = attach_parse_boundary_provenance(evidence)

    if belief_id:
        if relation["relation"] == "support":
            evidence["supports_beliefs"] = _dedupe(
                _as_list(evidence.get("supports_beliefs")) + [belief_id]
            )
        elif relation["relation"] == "challenge":
            evidence["challenges_beliefs"] = _dedupe(
                _as_list(evidence.get("challenges_beliefs")) + [belief_id]
            )
        elif relation["relation"] == "underdetermined":
            evidence["pending_beliefs"] = _dedupe(
                _as_list(evidence.get("pending_beliefs")) + [belief_id]
            )
        elif relation["relation"] == "neutral":
            evidence["neutral_beliefs"] = _dedupe(
                _as_list(evidence.get("neutral_beliefs")) + [belief_id]
            )
        else:
            evidence["contests_beliefs"] = _dedupe(
                _as_list(evidence.get("contests_beliefs")) + [belief_id]
            )
    return normalize_evidence(evidence)


def create_evidence_from_parse_candidates(
    kernel_dir: str | Path,
    record: dict[str, Any],
    *,
    belief_id: str,
    parse_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence = build_evidence_from_parse_candidates(
        record,
        belief_id=belief_id,
        parse_candidates=parse_candidates,
    )
    return create_evidence(kernel_dir, evidence)


def infer_evidence_strength_from_facts(record: dict[str, Any]) -> dict[str, Any]:
    """Infer strength from parse-boundary-tested low-inference fields.

    V3-alpha intentionally uses only fields already supported by the
    parse-boundary test: source_type and reports_lod. Other tempting fields
    such as independent_validation or has_controls remain excluded until they
    pass the same cross-model stability test.
    """
    score = 0.0
    factors: list[str] = []
    source_type = str(record.get("source_type", "external"))
    source_points = SOURCE_TYPE_STRENGTH_POINTS.get(source_type, 0.5)
    score += source_points
    factors.append(f"source_type={source_type} (+{source_points:g})")

    if _as_bool(record.get("reports_lod")):
        score += 0.25
        factors.append("reports_lod (+0.25)")

    # NOTE (v3-alpha): with only the two parse-boundary-verified fields admitted
    # (source_type max 2.0 + reports_lod 0.25 = 2.25 max), `decisive` (>=3.25) is
    # intentionally UNREACHABLE and `strong` is reachable only for experiment/benchmark
    # + reports_lod. By design: two objective facts should not license high strength.
    # Higher tiers reactivate automatically once more fields pass the parse-boundary
    # test and move out of `excluded_until_parse_boundary_tested` into scoring.
    if score >= 3.25:
        strength = "decisive"
    elif score >= 2.25:
        strength = "strong"
    elif score >= 1.5:
        strength = "moderate"
    else:
        strength = "weak"

    return {
        "evidence_strength": strength,
        "score": round(max(0.0, score), 2),
        "factors": factors,
        "method": "deterministic_from_verified_low_inference_fields",
        "accepted_input_fields": ["source_type", "reports_lod"],
        "excluded_until_parse_boundary_tested": [
            "venue_tier",
            "primary_research",
            "replicated",
            "independent_validation",
            "has_benchmark",
            "has_controls",
            "has_quantitative_readout",
            "sample_size",
            "is_review",
            "unvalidated_claim",
        ],
        "calibration_status": "uncalibrated_v3_alpha",
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise V3KernelValidationError(
                    [V3Issue(f"{path}:{line_number}", f"invalid JSON: {exc}")]
                ) from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def _require(record: dict[str, Any], field: str, issues: list[V3Issue], path: str) -> None:
    if record.get(field) in (None, "", []):
        issues.append(V3Issue(f"{path}.{field}", "is required"))


def normalize_evidence(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("summary", "")
    normalized.setdefault("supports_beliefs", [])
    normalized.setdefault("challenges_beliefs", [])
    normalized.setdefault("pending_beliefs", [])
    normalized.setdefault("neutral_beliefs", [])
    normalized.setdefault("contests_beliefs", [])
    normalized.setdefault("reliability", "unknown")
    if normalized.get("evidence_strength") in (None, ""):
        inferred = infer_evidence_strength_from_facts(normalized)
        normalized["evidence_strength"] = inferred["evidence_strength"]
        normalized["evidence_strength_provenance"] = inferred
    else:
        provenance = dict(normalized.get("evidence_strength_provenance") or {})
        provenance.setdefault("method", "asserted_by_caller")
        provenance.setdefault("asserted_strength", normalized["evidence_strength"])
        provenance.setdefault(
            "accountability_note",
            "Strength was supplied by the caller and was not recomputed by the V3-alpha kernel.",
        )
        normalized["evidence_strength_provenance"] = provenance
    normalized["source_type"] = str(normalized.get("source_type", "external"))
    normalized["supports_beliefs"] = _dedupe(_as_list(normalized["supports_beliefs"]))
    normalized["challenges_beliefs"] = _dedupe(_as_list(normalized["challenges_beliefs"]))
    normalized["pending_beliefs"] = _dedupe(_as_list(normalized["pending_beliefs"]))
    normalized["neutral_beliefs"] = _dedupe(_as_list(normalized["neutral_beliefs"]))
    normalized["contests_beliefs"] = _dedupe(_as_list(normalized["contests_beliefs"]))
    normalized.setdefault("id", _stable_id("ev", normalized))
    return normalized


def normalize_belief(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("updated_at", normalized["created_at"])
    normalized.setdefault("status", "proposed")
    normalized.setdefault("domain", "general")
    normalized.setdefault("confidence", 0.5)
    normalized.setdefault("entrenchment", 0.1)
    normalized.setdefault("evidence_ids", [])
    normalized.setdefault("contra_evidence_ids", [])
    normalized.setdefault("pending_evidence_ids", [])
    normalized.setdefault("neutral_evidence_ids", [])
    normalized.setdefault("contested_evidence_ids", [])
    normalized.setdefault("linked_concepts", [])
    normalized.setdefault("linked_constraints", [])
    normalized.setdefault("linked_questions", [])
    normalized.setdefault("last_revision_id", "")
    normalized.setdefault("provenance", {})
    normalized["confidence"] = _clamp01(normalized["confidence"], default=0.5)
    normalized["entrenchment"] = _clamp01(normalized["entrenchment"], default=0.1)
    for field in (
        "evidence_ids",
        "contra_evidence_ids",
        "pending_evidence_ids",
        "neutral_evidence_ids",
        "contested_evidence_ids",
        "linked_concepts",
        "linked_constraints",
        "linked_questions",
    ):
        normalized[field] = _dedupe(_as_list(normalized[field]))
    normalized.setdefault("id", _stable_id("belief", normalized))
    return normalized


def normalize_revision(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("triggering_evidence_ids", [])
    normalized.setdefault("human_override_id", "")
    normalized["triggering_evidence_ids"] = _dedupe(
        _as_list(normalized["triggering_evidence_ids"])
    )
    normalized.setdefault("id", _stable_id("rev", normalized))
    return normalized


def normalize_judgment(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("updated_at", normalized["created_at"])
    normalized.setdefault("confidence", 0.5)
    normalized.setdefault("risk_level", "unknown")
    normalized.setdefault("linked_beliefs", [])
    normalized.setdefault("linked_constraints", [])
    normalized.setdefault("linked_evidence", [])
    normalized.setdefault("next_actions", [])
    normalized["confidence"] = _clamp01(normalized["confidence"], default=0.5)
    for field in ("linked_beliefs", "linked_constraints", "linked_evidence", "next_actions"):
        normalized[field] = _dedupe(_as_list(normalized[field]))
    normalized.setdefault("id", _stable_id("judgment", normalized))
    return normalized


def normalize_constraint(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("updated_at", normalized["created_at"])
    normalized.setdefault("type", "scientific")
    normalized.setdefault("status", "proposed")
    normalized.setdefault("confidence", 0.5)
    normalized.setdefault("falsifiability", "")
    normalized.setdefault("predicted_observations", [])
    normalized.setdefault("failure_conditions", [])
    normalized.setdefault("linked_beliefs", [])
    normalized.setdefault("linked_evidence", [])
    normalized["confidence"] = _clamp01(normalized["confidence"], default=0.5)
    for field in (
        "predicted_observations",
        "failure_conditions",
        "linked_beliefs",
        "linked_evidence",
    ):
        normalized[field] = _dedupe(_as_list(normalized[field]))
    normalized.setdefault("id", _stable_id("constraint", normalized))
    return normalized


def normalize_override(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("extracted_learning", "")
    normalized.setdefault("future_rule_candidate", "")
    normalized.setdefault("id", _stable_id("override", normalized))
    return normalized


def normalize_kernel_intake_assessment(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize the kernel's intake judgment over one secretary briefing.

    This object is deliberately not a BeliefRevision. It records whether the
    kernel will accept a briefing as admissible input, needs a better briefing,
    or must escalate to human review before any durable state can change.
    """
    normalized = dict(record)
    normalized.setdefault("schema_id", "kernel_intake_assessment_v1")
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("briefing_id", "")
    normalized.setdefault("source_ref", "")
    normalized.setdefault("decision", "defer")
    normalized.setdefault("durable_change_authorization", "not_authorized")
    normalized.setdefault("admissibility", {})
    normalized.setdefault("boundary_checks", [])
    normalized.setdefault("missing_modules", [])
    normalized.setdefault("accepted_candidates", [])
    normalized.setdefault("deferred_candidates", [])
    normalized.setdefault("rejected_candidates", [])
    normalized.setdefault("reasons", [])
    normalized.setdefault("next_actions", [])
    normalized.setdefault("refresh_request", {})
    normalized.setdefault("escalation", {})
    normalized.setdefault("provenance", {})
    normalized["boundary_checks"] = _as_dict_list(normalized["boundary_checks"])
    normalized["accepted_candidates"] = _dedupe(_as_list(normalized["accepted_candidates"]))
    normalized["deferred_candidates"] = _dedupe(_as_list(normalized["deferred_candidates"]))
    normalized["rejected_candidates"] = _dedupe(_as_list(normalized["rejected_candidates"]))
    normalized["missing_modules"] = _dedupe(_as_list(normalized["missing_modules"]))
    normalized["reasons"] = _dedupe(_as_list(normalized["reasons"]))
    normalized["next_actions"] = _dedupe(_as_list(normalized["next_actions"]))
    if "id" not in normalized and "assessment_id" in normalized:
        normalized["id"] = str(normalized["assessment_id"])
    normalized.setdefault("id", _stable_id("kia", normalized))
    normalized["assessment_id"] = normalized["id"]
    return normalized


def normalize_research_judgment_decision(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize the kernel's first durable research-judgment object.

    A ResearchJudgmentDecision is downstream of intake. It may propose which
    research-state layer should change, but it does not itself mutate belief
    projection or create BeliefRevision records.
    """
    normalized = dict(record)
    normalized.setdefault("schema_id", "research_judgment_decision_v1")
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("assessment_id", "")
    normalized.setdefault("briefing_id", "")
    normalized.setdefault("source_ref", "")
    normalized.setdefault("decision", "no_change")
    normalized.setdefault("affected_objects", {})
    normalized.setdefault("candidate_state_changes", [])
    normalized.setdefault("rejected_alternatives", [])
    normalized.setdefault("rationale", [])
    normalized.setdefault("judgment_frame", {})
    normalized.setdefault("uncertainty", {})
    normalized.setdefault("required_actions", [])
    normalized.setdefault("human_review_required", False)
    normalized.setdefault("applied_revision_ids", [])
    normalized.setdefault("provenance", {})
    normalized["candidate_state_changes"] = _as_dict_list(
        normalized["candidate_state_changes"]
    )
    normalized["affected_objects"] = (
        normalized["affected_objects"]
        if isinstance(normalized["affected_objects"], dict)
        else {}
    )
    normalized["uncertainty"] = (
        normalized["uncertainty"] if isinstance(normalized["uncertainty"], dict) else {}
    )
    normalized["judgment_frame"] = (
        normalized["judgment_frame"]
        if isinstance(normalized["judgment_frame"], dict)
        else {}
    )
    normalized["provenance"] = (
        normalized["provenance"] if isinstance(normalized["provenance"], dict) else {}
    )
    for field in ("rejected_alternatives", "rationale", "required_actions", "applied_revision_ids"):
        normalized[field] = _dedupe(_as_list(normalized[field]))
    normalized["human_review_required"] = _as_bool(normalized["human_review_required"])
    if "id" not in normalized and "decision_id" in normalized:
        normalized["id"] = str(normalized["decision_id"])
    normalized.setdefault("id", _stable_id("rjd", normalized))
    normalized["decision_id"] = normalized["id"]
    return normalized


def normalize_question(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("schema_id", "research_question_v1")
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("updated_at", normalized["created_at"])
    normalized.setdefault("status", "proposed")
    normalized.setdefault("source_decision_id", "")
    normalized.setdefault("source_ref", "")
    normalized.setdefault("linked_beliefs", [])
    normalized.setdefault("linked_evidence", [])
    normalized.setdefault("rationale", [])
    normalized.setdefault("uncertainty_level", "unknown")
    normalized.setdefault("last_revision_id", "")
    normalized.setdefault("provenance", {})
    normalized["linked_beliefs"] = _dedupe(_as_list(normalized["linked_beliefs"]))
    normalized["linked_evidence"] = _dedupe(_as_list(normalized["linked_evidence"]))
    normalized["rationale"] = _dedupe(_as_list(normalized["rationale"]))
    normalized["provenance"] = (
        normalized["provenance"] if isinstance(normalized["provenance"], dict) else {}
    )
    if "id" not in normalized and "question_id" in normalized:
        normalized["id"] = str(normalized["question_id"])
    normalized.setdefault("id", _stable_id("q", normalized))
    normalized["question_id"] = normalized["id"]
    return normalized


def normalize_question_revision(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("schema_id", "research_question_revision_v1")
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("action", "create")
    normalized.setdefault("old_status", "")
    normalized.setdefault("new_status", "proposed")
    normalized.setdefault("reason", "")
    normalized.setdefault("source_decision_id", "")
    normalized.setdefault("provenance", {})
    normalized["provenance"] = (
        normalized["provenance"] if isinstance(normalized["provenance"], dict) else {}
    )
    if "id" not in normalized and "revision_id" in normalized:
        normalized["id"] = str(normalized["revision_id"])
    normalized.setdefault("id", _stable_id("qrev", normalized))
    normalized["revision_id"] = normalized["id"]
    return normalized


def normalize_trajectory(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("schema_id", "research_trajectory_v1")
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("updated_at", normalized["created_at"])
    normalized.setdefault("title", "")
    normalized.setdefault("status", "proposed")
    normalized.setdefault("source_decision_id", "")
    normalized.setdefault("source_ref", "")
    normalized.setdefault("linked_questions", [])
    normalized.setdefault("linked_beliefs", [])
    normalized.setdefault("rationale", [])
    normalized.setdefault("last_revision_id", "")
    normalized.setdefault("provenance", {})
    normalized["linked_questions"] = _dedupe(_as_list(normalized["linked_questions"]))
    normalized["linked_beliefs"] = _dedupe(_as_list(normalized["linked_beliefs"]))
    normalized["rationale"] = _dedupe(_as_list(normalized["rationale"]))
    normalized["provenance"] = (
        normalized["provenance"] if isinstance(normalized["provenance"], dict) else {}
    )
    if "id" not in normalized and "trajectory_id" in normalized:
        normalized["id"] = str(normalized["trajectory_id"])
    normalized.setdefault("id", _stable_id("traj", normalized))
    normalized["trajectory_id"] = normalized["id"]
    return normalized


def normalize_trajectory_revision(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("schema_id", "research_trajectory_revision_v1")
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("action", "create")
    normalized.setdefault("old_status", "")
    normalized.setdefault("new_status", "proposed")
    normalized.setdefault("reason", "")
    normalized.setdefault("source_decision_id", "")
    normalized.setdefault("provenance", {})
    normalized["provenance"] = (
        normalized["provenance"] if isinstance(normalized["provenance"], dict) else {}
    )
    if "id" not in normalized and "revision_id" in normalized:
        normalized["id"] = str(normalized["revision_id"])
    normalized.setdefault("id", _stable_id("trev", normalized))
    normalized["revision_id"] = normalized["id"]
    return normalized


def normalize_action_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("schema_id", "kernel_action_record_v1")
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("updated_at", normalized["created_at"])
    normalized.setdefault("decision_id", "")
    normalized.setdefault("action_type", "read")
    normalized.setdefault("description", "")
    normalized.setdefault("status", "open")
    normalized.setdefault("priority", "medium")
    normalized.setdefault("source_ref", "")
    normalized.setdefault("linked_assessment_id", "")
    normalized.setdefault("linked_briefing_id", "")
    normalized.setdefault("result_ref", "")
    normalized.setdefault("provenance", {})
    normalized["provenance"] = (
        normalized["provenance"] if isinstance(normalized["provenance"], dict) else {}
    )
    if "id" not in normalized and "action_id" in normalized:
        normalized["id"] = str(normalized["action_id"])
    normalized.setdefault("id", _stable_id("act", normalized))
    normalized["action_id"] = normalized["id"]
    return normalized


def normalize_human_review_request(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("schema_id", "human_review_request_v1")
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("updated_at", normalized["created_at"])
    normalized.setdefault("decision_id", "")
    normalized.setdefault("assessment_id", "")
    normalized.setdefault("briefing_id", "")
    normalized.setdefault("source_ref", "")
    normalized.setdefault("belief_ref", "")
    normalized.setdefault("request_type", "human_read")
    normalized.setdefault("target_state_layer", "attention")
    normalized.setdefault("question", "")
    normalized.setdefault("status", "open")
    normalized.setdefault("priority", "medium")
    normalized.setdefault("linked_action_ids", [])
    normalized.setdefault("allowed_responses", [])
    normalized.setdefault("reviewer_payload", {})
    normalized.setdefault("kernel_context", {})
    normalized.setdefault("anti_anchoring", {})
    normalized.setdefault("response_ref", "")
    normalized.setdefault("provenance", {})
    normalized["linked_action_ids"] = _dedupe(_as_list(normalized["linked_action_ids"]))
    normalized["allowed_responses"] = _as_dict_list(normalized["allowed_responses"])
    for field in ("reviewer_payload", "kernel_context", "anti_anchoring", "provenance"):
        normalized[field] = normalized[field] if isinstance(normalized[field], dict) else {}
    if "id" not in normalized and "request_id" in normalized:
        normalized["id"] = str(normalized["request_id"])
    normalized.setdefault("id", _stable_id("hrr", normalized))
    normalized["request_id"] = normalized["id"]
    return normalized


def _briefing_source_ref(briefing: dict[str, Any]) -> str:
    source = briefing.get("source") if isinstance(briefing.get("source"), dict) else {}
    for field in ("doi", "source_id", "paper_id", "title"):
        value = str(source.get(field) or "").strip()
        if value:
            return value
    return str(briefing.get("briefing_id") or "").strip()


def _candidate_ids(candidates: list[dict[str, Any]]) -> list[str]:
    ids = []
    for index, candidate in enumerate(candidates, 1):
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        ids.append(candidate_id or f"candidate_{index:02d}")
    return _dedupe(ids)


def _boundary_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    severity: str,
    message: str,
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "severity": severity,
            "message": message,
        }
    )


def assess_secretary_briefing(briefing: dict[str, Any]) -> dict[str, Any]:
    """Assess an Audit Secretary briefing without committing durable revisions."""
    if not isinstance(briefing, dict):
        briefing = {}

    source = briefing.get("source") if isinstance(briefing.get("source"), dict) else {}
    relation_scope = (
        briefing.get("relation_scope")
        if isinstance(briefing.get("relation_scope"), dict)
        else {}
    )
    evidence_map = (
        briefing.get("evidence_map")
        if isinstance(briefing.get("evidence_map"), dict)
        else {}
    )
    source_provenance = (
        briefing.get("source_provenance")
        if isinstance(briefing.get("source_provenance"), dict)
        else {}
    )
    object_grounding = (
        briefing.get("object_grounding")
        if isinstance(briefing.get("object_grounding"), dict)
        else {}
    )
    confidence_dynamics = (
        briefing.get("confidence_dynamics")
        if isinstance(briefing.get("confidence_dynamics"), dict)
        else {}
    )
    task_reliability = (
        briefing.get("task_reliability_envelope")
        if isinstance(briefing.get("task_reliability_envelope"), dict)
        else {}
    )
    epistemic_probe = (
        briefing.get("epistemic_task_probe")
        if isinstance(briefing.get("epistemic_task_probe"), dict)
        else {}
    )
    semantic_audit = (
        briefing.get("semantic_audit")
        if isinstance(briefing.get("semantic_audit"), dict)
        else {}
    )
    calibration = (
        briefing.get("calibration_status")
        if isinstance(briefing.get("calibration_status"), dict)
        else {}
    )
    personalization = (
        briefing.get("personalization_boundary")
        if isinstance(briefing.get("personalization_boundary"), dict)
        else {}
    )
    closed_loop = (
        briefing.get("closed_loop_validation")
        if isinstance(briefing.get("closed_loop_validation"), dict)
        else {}
    )
    strength_inputs = (
        evidence_map.get("evidence_strength_inputs")
        if isinstance(evidence_map.get("evidence_strength_inputs"), dict)
        else {}
    )
    attention = (
        briefing.get("attention_brief")
        if isinstance(briefing.get("attention_brief"), dict)
        else {}
    )
    provenance = (
        briefing.get("provenance")
        if isinstance(briefing.get("provenance"), dict)
        else {}
    )
    claims = _as_dict_list(briefing.get("claim_map"))
    disagreements = _as_dict_list(briefing.get("disagreement_map"))
    uncertainties = _as_dict_list(briefing.get("uncertainty_map"))
    candidates = _as_dict_list(briefing.get("candidate_kernel_updates"))
    candidate_ids = _candidate_ids(candidates)
    boundary_checks: list[dict[str, Any]] = []
    reasons: list[str] = []
    next_actions: list[str] = []
    blockers: list[str] = []

    has_source = any(
        str(source.get(field) or "").strip()
        for field in ("doi", "source_id", "paper_id", "title")
    )
    _boundary_check(
        boundary_checks,
        name="source_present",
        passed=has_source,
        severity="blocker",
        message="Briefing must identify a source before kernel intake.",
    )
    if not has_source:
        blockers.append("missing_source")

    schema_ok = briefing.get("schema_id") == "audit_secretary_briefing_schema_v1"
    _boundary_check(
        boundary_checks,
        name="secretary_schema",
        passed=schema_ok,
        severity="blocker",
        message="Kernel intake currently accepts Audit Secretary briefing schema v1 only.",
    )
    if not schema_ok:
        blockers.append("unknown_secretary_schema")

    bad_permissions = [
        candidate.get("candidate_id") or f"candidate_{idx:02d}"
        for idx, candidate in enumerate(candidates, 1)
        if candidate.get("commit_permission") != "kernel_only"
    ]
    _boundary_check(
        boundary_checks,
        name="candidate_commit_permission",
        passed=not bad_permissions,
        severity="blocker",
        message="Secretary candidates must be kernel_only recommendations.",
    )
    if bad_permissions:
        blockers.append("secretary_attempted_commit")

    projection = relation_scope.get("whole_belief_projection_candidate", "")
    projection_ok = projection in {"", "not_computed", "not_supplied"}
    _boundary_check(
        boundary_checks,
        name="no_secretary_whole_belief_projection",
        passed=projection_ok,
        severity="blocker",
        message="Secretary may report relation_scope but must not compute durable belief projection.",
    )
    if not projection_ok:
        blockers.append("secretary_attempted_projection")

    input_level = str(source.get("input_text_level") or "").strip() or (
        "abstract" if source.get("abstract") else "unknown"
    )
    abstract_level_only = (
        input_level in {"title_only", "abstract"}
        or _as_bool(strength_inputs.get("abstract_level_only"))
    )
    has_claim_map = bool(claims)
    has_directional_conflict = any(
        item.get("suggested_resolution") == "human_read"
        for item in disagreements
    )
    has_high_uncertainty = any(
        str(item.get("severity") or "").lower() == "high"
        for item in uncertainties
    )
    legacy_reader = (
        relation_scope.get("legacy_whole_belief_reader")
        if isinstance(relation_scope.get("legacy_whole_belief_reader"), dict)
        else {}
    )
    legacy_relation = str(legacy_reader.get("kernel_relation") or "").strip()
    attention_recommendation = str(attention.get("attention_recommendation") or "").strip()

    _boundary_check(
        boundary_checks,
        name="claim_map_present",
        passed=has_claim_map,
        severity="warning",
        message="Claim map is required before the kernel can evaluate candidate revisions.",
    )
    _boundary_check(
        boundary_checks,
        name="abstract_level_boundary",
        passed=not abstract_level_only,
        severity="warning",
        message="Abstract/title-only briefings can be retained, but cannot authorize durable revision.",
    )

    missing_modules = [module for module in INTAKE_CONTEXT_MODULES if module not in briefing]
    _boundary_check(
        boundary_checks,
        name="secretary_context_modules_optional",
        passed=True,
        severity="info",
        message=(
            "Secretary context modules are recorded for downstream review, "
            "but missing optional modules do not gate intake."
        ),
    )

    source_provenance_ok = any(
        str(value or "").strip()
        for value in (
            source_provenance.get("source_text_hash"),
            source_provenance.get("doi"),
            provenance.get("source_text_hash"),
            source.get("doi"),
            source.get("source_id"),
        )
    )
    _boundary_check(
        boundary_checks,
        name="source_provenance_present",
        passed=source_provenance_ok,
        severity="warning",
        message="Source provenance should carry DOI/source hash/retrieval metadata.",
    )
    grounding_risk = str(
        object_grounding.get("hallucination_risk_from_language_prior") or "not_assessed"
    ).lower()
    _boundary_check(
        boundary_checks,
        name="object_grounding_acceptable",
        passed=grounding_risk != "high",
        severity="warning",
        message="Object grounding should expose whether the briefing is text-only or under-grounded.",
    )
    reliability_admissibility = str(task_reliability.get("admissibility") or "not_assessed")
    _boundary_check(
        boundary_checks,
        name="task_reliability_envelope_admissible",
        passed=reliability_admissibility != "not_admissible",
        severity="warning",
        message="LLM reader tasks need a declared reliability envelope before kernel review.",
    )
    epistemic_risk = str(epistemic_probe.get("epistemic_operator_risk") or "not_assessed")
    _boundary_check(
        boundary_checks,
        name="epistemic_operator_risk_bounded",
        passed=epistemic_risk != "high",
        severity="warning",
        message="Belief/knowledge/fact language must be protected before truth-facing interpretation.",
    )
    confidence_usable = _as_bool(confidence_dynamics.get("usable_for_kernel_confidence"))
    calibration_basis = str(calibration.get("confidence_basis") or "not_assessed")
    _boundary_check(
        boundary_checks,
        name="reader_confidence_not_direct_kernel_confidence",
        passed=(not confidence_usable) or calibration_basis not in {"uncalibrated", "unknown"},
        severity="warning",
        message="Reader confidence may not directly revise kernel confidence without calibration basis.",
    )
    semantic_risk = str(semantic_audit.get("behavioral_risk") or "not_assessed")
    _boundary_check(
        boundary_checks,
        name="semantic_audit_risk_bounded",
        passed=semantic_risk != "high",
        severity="warning",
        message="Unexpected/spurious concept reliance should be exposed before durable revision.",
    )
    personalization_safe = personalization.get("truth_or_judgment_effect", "none") == "none"
    _boundary_check(
        boundary_checks,
        name="personalization_does_not_change_truth",
        passed=personalization_safe,
        severity="blocker",
        message="Preferences may shape attention or display, not truth or kernel judgment.",
    )
    if not personalization_safe:
        blockers.append("personalization_attempted_truth_effect")
    closed_loop_safe = not _as_bool(closed_loop.get("kernel_commit_allowed"))
    _boundary_check(
        boundary_checks,
        name="closed_loop_validation_not_overclaimed",
        passed=closed_loop_safe,
        severity="blocker",
        message="Unvalidated oracle-like outputs may not authorize kernel commits.",
    )
    if not closed_loop_safe:
        blockers.append("closed_loop_attempted_commit")

    if blockers:
        decision = "reject"
        reasons.extend(blockers)
        rejected_candidates = candidate_ids
        deferred_candidates: list[str] = []
        accepted_candidates: list[str] = []
        next_actions.append("regenerate_secretary_briefing")
    elif has_directional_conflict or legacy_relation == "contest":
        decision = "escalate"
        reasons.append("directional_reader_conflict")
        if has_high_uncertainty:
            reasons.append("high_uncertainty")
        rejected_candidates = []
        accepted_candidates = []
        deferred_candidates = candidate_ids
        next_actions.extend(["human_read", "request_full_text_or_atom_level_read"])
    elif not has_claim_map or input_level == "title_only":
        decision = "request_refresh"
        reasons.append("insufficient_source_text_for_intake")
        rejected_candidates = []
        accepted_candidates = []
        deferred_candidates = candidate_ids
        next_actions.append("request_abstract_or_full_text_briefing")
    elif reliability_admissibility == "not_admissible":
        decision = "request_refresh"
        reasons.append("task_reliability_not_admissible")
        rejected_candidates = []
        accepted_candidates = []
        deferred_candidates = candidate_ids
        next_actions.append("request_reliable_reader_task_or_full_text")
    elif semantic_risk == "high" or epistemic_risk == "high":
        decision = "escalate"
        reasons.append("methodology_risk_requires_human_review")
        rejected_candidates = []
        accepted_candidates = []
        deferred_candidates = candidate_ids
        next_actions.append("human_review_methodology_risk")
    elif abstract_level_only:
        decision = "partial_accept"
        reasons.append("admissible_as_intake_only")
        rejected_candidates = []
        accepted_candidates = [
            candidate.get("candidate_id") or f"candidate_{idx:02d}"
            for idx, candidate in enumerate(candidates, 1)
            if candidate.get("target_layer") == "source" and candidate.get("risk") == "low"
        ]
        deferred_candidates = [
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id not in accepted_candidates
        ]
        next_actions.append("request_full_text_before_revision")
    else:
        decision = "accept" if candidates else "partial_accept"
        reasons.append("briefing_admissible_for_kernel_review")
        rejected_candidates = []
        accepted_candidates = []
        deferred_candidates = candidate_ids
        next_actions.append("kernel_review_candidate_updates")

    refresh_needed = (
        decision in {"defer", "request_refresh", "partial_accept"}
        or abstract_level_only
        or reliability_admissibility == "not_admissible"
    )
    assessment = normalize_kernel_intake_assessment(
        {
            "briefing_id": str(briefing.get("briefing_id") or ""),
            "source_ref": _briefing_source_ref(briefing),
            "decision": decision,
            "durable_change_authorization": "not_authorized",
            "admissibility": {
                "has_source": has_source,
                "has_claim_map": has_claim_map,
                "input_text_level": input_level,
                "abstract_level_only": abstract_level_only,
                "candidate_count": len(candidates),
                "bad_commit_permission_candidates": bad_permissions,
                "secretary_projection": projection or "not_supplied",
                "directional_conflict": has_directional_conflict,
                "high_uncertainty": has_high_uncertainty,
                "legacy_relation": legacy_relation or "not_supplied",
                "attention_recommendation": attention_recommendation or "not_supplied",
                "source_provenance_ok": source_provenance_ok,
                "object_grounding_risk": grounding_risk,
                "task_reliability_admissibility": reliability_admissibility,
                "epistemic_operator_risk": epistemic_risk,
                "confidence_usable_for_kernel": confidence_usable,
                "calibration_basis": calibration_basis,
                "semantic_audit_risk": semantic_risk,
                "personalization_safe": personalization_safe,
                "closed_loop_commit_allowed": not closed_loop_safe,
            },
            "boundary_checks": boundary_checks,
            "missing_modules": missing_modules,
            "accepted_candidates": accepted_candidates,
            "deferred_candidates": deferred_candidates,
            "rejected_candidates": rejected_candidates,
            "reasons": reasons,
            "next_actions": next_actions,
            "refresh_request": {
                "needed": refresh_needed,
                "reasons": (
                    (["full_text_required_before_revision"] if abstract_level_only else [])
                    + (
                        ["task_reliability_not_admissible"]
                        if reliability_admissibility == "not_admissible"
                        else []
                    )
                ),
            },
            "escalation": {
                "needed": decision == "escalate",
                "reason": "directional reader conflict" if decision == "escalate" else "",
            },
            "provenance": {
                "method": "kernel_intake_assessment_v1_thin_contract",
                "secretary_schema_id": briefing.get("schema_id", ""),
                "secretary_created_at": provenance.get("created_at", ""),
                "source_text_hash": provenance.get("source_text_hash", ""),
                "note": (
                    "Assessment is a thin admissibility, authority, and contamination "
                    "gate only; durable belief projection still requires explicit "
                    "kernel revision."
                ),
            },
        }
    )
    return assessment


def _briefing_relation(briefing: dict[str, Any]) -> str:
    relation_scope = (
        briefing.get("relation_scope")
        if isinstance(briefing.get("relation_scope"), dict)
        else {}
    )
    legacy_reader = (
        relation_scope.get("legacy_whole_belief_reader")
        if isinstance(relation_scope.get("legacy_whole_belief_reader"), dict)
        else {}
    )
    relation = str(legacy_reader.get("kernel_relation") or "").strip().lower()
    return RELATION_ALIASES.get(relation, relation or "not_supplied")


def _briefing_belief_ref(briefing: dict[str, Any]) -> str:
    relation_scope = (
        briefing.get("relation_scope")
        if isinstance(briefing.get("relation_scope"), dict)
        else {}
    )
    return str(relation_scope.get("belief_id") or "").strip()


def _briefing_candidate_question(briefing: dict[str, Any]) -> str:
    trajectory = (
        briefing.get("trajectory_proposal")
        if isinstance(briefing.get("trajectory_proposal"), dict)
        else {}
    )
    return str(trajectory.get("candidate_question") or "").strip()


def _briefing_read_level(briefing: dict[str, Any]) -> str:
    source = briefing.get("source") if isinstance(briefing.get("source"), dict) else {}
    provenance = (
        briefing.get("source_provenance")
        if isinstance(briefing.get("source_provenance"), dict)
        else {}
    )
    return str(
        source.get("input_text_level")
        or provenance.get("read_level")
        or "unknown"
    ).strip().lower()


def _candidate_layers(briefing: dict[str, Any]) -> list[str]:
    return _dedupe(
        str(candidate.get("target_layer") or "").strip()
        for candidate in _as_dict_list(briefing.get("candidate_kernel_updates"))
        if str(candidate.get("target_layer") or "").strip()
    )


def _max_uncertainty_severity(briefing: dict[str, Any]) -> str:
    rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    highest = "none"
    for item in _as_dict_list(briefing.get("uncertainty_map")):
        severity = str(item.get("severity") or "none").strip().lower()
        if rank.get(severity, 0) > rank.get(highest, 0):
            highest = severity
    return highest


def _clean_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _extract_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        for field in ("id", "atom_id", "key_claim_id", "claim_id", "ref"):
            if str(value.get(field) or "").strip():
                return [str(value.get(field)).strip()]
        return []
    if isinstance(value, list):
        ids: list[str] = []
        for item in value:
            ids.extend(_extract_id_list(item))
        return _dedupe(ids)
    return _as_list(value)


def _key_claim_id(record: dict[str, Any], index: int) -> str:
    for field in ("key_claim_id", "atom_id", "id", "claim_id"):
        value = str(record.get(field) or "").strip()
        if value:
            return value
    return f"key_claim_{index:02d}"


def _normalize_key_claims(raw_claims: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_claims, 1):
        key_id = _key_claim_id(item, index)
        if key_id in seen:
            continue
        seen.add(key_id)
        normalized.append(
            {
                "id": key_id,
                "claim": str(
                    item.get("claim")
                    or item.get("claim_text")
                    or item.get("text")
                    or ""
                ).strip(),
                "role": _clean_token(item.get("role") or item.get("claim_type") or "claim"),
                "criticality": _clean_token(item.get("criticality") or "required"),
            }
        )
    return normalized


def _frozen_key_claims(briefing: dict[str, Any]) -> list[dict[str, str]]:
    relation_scope = (
        briefing.get("relation_scope")
        if isinstance(briefing.get("relation_scope"), dict)
        else {}
    )
    raw_claims: list[dict[str, Any]] = []
    for field in (
        "frozen_key_claims",
        "key_claims",
        "belief_key_claims",
        "key_mechanisms",
    ):
        raw_claims.extend(_as_dict_list(relation_scope.get(field)))
        raw_claims.extend(_as_dict_list(briefing.get(field)))
    atoms = [
        atom
        for atom in _as_dict_list(relation_scope.get("atoms"))
        if _clean_token(atom.get("criticality") or "") == "required"
    ]
    raw_claims.extend(atoms)
    return _normalize_key_claims(raw_claims)


def _claim_key_refs(claim: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for field in (
        "matched_key_claim_ids",
        "key_claim_ids",
        "key_claim_refs",
        "touches_key_claims",
        "matched_atoms",
        "matched_atom_ids",
        "atom_ids",
        "relation_scope_atom_ids",
        "mapped_key_claims",
        "key_claim_id",
        "atom_id",
    ):
        refs.extend(_extract_id_list(claim.get(field)))
    return _dedupe(refs)


def _claim_evidence_types(claim: dict[str, Any]) -> list[str]:
    types: list[str] = []
    for field in ("evidence_type", "evidence_basis", "basis", "support_type"):
        types.extend(_clean_token(item) for item in _as_list(claim.get(field)))
    return _dedupe(types)


def _claim_is_measured_or_tested(claim: dict[str, Any]) -> bool:
    evidence_types = _claim_evidence_types(claim)
    if not evidence_types:
        return False
    if any(item in MEASURED_OR_TESTED_CLAIM_EVIDENCE for item in evidence_types):
        return True
    if any(item in NON_TOUCH_CLAIM_EVIDENCE for item in evidence_types):
        return False
    return False


def _claim_type_matches_key_role(claim: dict[str, Any], key_claim: dict[str, str]) -> bool:
    role = key_claim.get("role", "")
    if role not in {"mechanism", "boundary"}:
        return True
    claim_type = _clean_token(claim.get("claim_type") or "")
    if role == "mechanism":
        return claim_type in {"mechanism", "mechanistic", "mechanism_test"}
    if role == "boundary":
        return claim_type in {
            "mechanism",
            "mechanistic",
            "boundary",
            "control",
            "artifact",
            "negative_control",
        }
    return True


def _token_list_from_fields(record: dict[str, Any], fields: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for field in fields:
        value = record.get(field)
        if isinstance(value, dict):
            tokens.extend(_clean_token(key) for key, enabled in value.items() if _as_bool(enabled))
        elif isinstance(value, list):
            tokens.extend(_clean_token(item) for item in value if str(item or "").strip())
        elif str(value or "").strip():
            tokens.append(_clean_token(value))
    return _dedupe(token for token in tokens if token)


def _scope_tokens(record: dict[str, Any], fields: Iterable[str]) -> list[str]:
    return _token_list_from_fields(record, fields)


def _claim_design_negative_weight(
    *,
    flags: list[str],
    risk: str,
    distance: str,
    explicit_weight: Any,
) -> float:
    weight = 0.0
    for flag in flags:
        weight += OVERCLAIM_FLAG_WEIGHTS.get(flag, 0.15)
    if risk in HIGH_OVERCLAIM_RISK:
        weight = max(weight, 0.45)
    elif risk in MEDIUM_OVERCLAIM_RISK:
        weight = max(weight, 0.2)
    if distance in {"high", "far", "large"}:
        weight = max(weight, 0.35)
    elif distance in {"medium", "moderate"}:
        weight = max(weight, 0.18)
    if explicit_weight not in (None, ""):
        weight = max(weight, _clamp01(explicit_weight, default=0.0))
    return _clamp01(weight, default=0.0)


def _claim_design_alignment_audit(
    briefing: dict[str, Any],
    *,
    relation: str,
) -> dict[str, Any]:
    """Audit whether claim scope is entitled by experimental design scope.

    This is not a paper-quality verdict. It records negative boundary pressure:
    when a source may support a narrow prototype/design claim, but should not be
    used as support for a broader diagnostic, causal, platform, robustness, or
    translational claim.
    """
    briefing_audit = (
        briefing.get("claim_design_audit")
        if isinstance(briefing.get("claim_design_audit"), dict)
        else {}
    )
    claim_evaluations: list[dict[str, Any]] = []
    all_flags: list[str] = []
    allowed_interpretations: list[str] = []
    disallowed_interpretations: list[str] = []
    shortages: list[str] = []
    max_weight = 0.0
    has_signal = bool(briefing_audit)

    for claim in _as_dict_list(briefing.get("claim_map")):
        claim_scopes = _scope_tokens(
            claim,
            (
                "claim_scope",
                "claimed_scope",
                "interpretation_scope",
                "conclusion_scope",
            ),
        )
        evidence_scopes = _scope_tokens(
            claim,
            (
                "evidence_scope",
                "design_scope",
                "experimental_scope",
                "supported_scope",
                "sample_context",
                "model_system",
            ),
        )
        flags = _token_list_from_fields(
            claim,
            (
                "overclaim_flags",
                "overclaim_flag",
                "support_gap_flags",
                "claim_design_flags",
            ),
        )
        if set(claim_scopes) & TRANSLATIONAL_CLAIM_SCOPES and set(evidence_scopes) & PROTOTYPE_EVIDENCE_SCOPES:
            flags.append("translational_overclaim")
        if set(claim_scopes) & PLATFORM_CLAIM_SCOPES and set(evidence_scopes) & NARROW_EVIDENCE_SCOPES:
            flags.append("platform_overclaim")
        evidence_types = _claim_evidence_types(claim)
        claim_type = _clean_token(claim.get("claim_type") or "")
        if claim_type in {"causal", "causality"} and any(
            evidence_type in {"correlation", "association", "observational"}
            for evidence_type in evidence_types
        ):
            flags.append("causal_overclaim")
        if claim_type in {"mechanism", "mechanistic"} and any(
            evidence_type in NON_TOUCH_CLAIM_EVIDENCE
            for evidence_type in evidence_types
        ):
            flags.append("mechanism_overclaim")
        flags = _dedupe(flags)
        risk = _clean_token(claim.get("overclaim_risk") or claim.get("claim_design_risk") or "")
        distance = _clean_token(
            claim.get("evidence_claim_distance")
            or claim.get("claim_design_distance")
            or claim.get("support_gap")
            or ""
        )
        explicit_weight = claim.get("negative_weight")
        weight = _claim_design_negative_weight(
            flags=flags,
            risk=risk,
            distance=distance,
            explicit_weight=explicit_weight,
        )
        has_claim_signal = bool(flags or risk or distance or claim_scopes or evidence_scopes)
        has_signal = has_signal or has_claim_signal
        max_weight = max(max_weight, weight)
        all_flags.extend(flags)
        allowed_interpretations.extend(
            _as_list(claim.get("allowed_interpretation"))
            + _as_list(claim.get("allowed_interpretations"))
        )
        disallowed_interpretations.extend(
            _as_list(claim.get("disallowed_interpretation"))
            + _as_list(claim.get("disallowed_interpretations"))
        )
        shortages.extend(
            _as_list(claim.get("design_shortage"))
            + _as_list(claim.get("design_shortages"))
            + _as_list(claim.get("controls_missing"))
            + _as_list(claim.get("missing_controls"))
        )
        if has_claim_signal:
            claim_evaluations.append(
                {
                    "claim_id": str(claim.get("claim_id") or ""),
                    "claim_type": claim_type,
                    "claim_scopes": claim_scopes,
                    "evidence_scopes": evidence_scopes,
                    "overclaim_flags": flags,
                    "overclaim_risk": risk or "not_supplied",
                    "evidence_claim_distance": distance or "not_supplied",
                    "negative_weight": weight,
                    "allowed_interpretations": _dedupe(
                        _as_list(claim.get("allowed_interpretation"))
                        + _as_list(claim.get("allowed_interpretations"))
                    ),
                    "disallowed_interpretations": _dedupe(
                        _as_list(claim.get("disallowed_interpretation"))
                        + _as_list(claim.get("disallowed_interpretations"))
                    ),
                }
            )

    briefing_flags = _token_list_from_fields(
        briefing_audit,
        (
            "overclaim_flags",
            "overclaim_flag",
            "support_gap_flags",
            "claim_design_flags",
        ),
    )
    all_flags.extend(briefing_flags)
    briefing_risk = _clean_token(
        briefing_audit.get("overclaim_risk")
        or briefing_audit.get("claim_design_risk")
        or ""
    )
    briefing_distance = _clean_token(
        briefing_audit.get("evidence_claim_distance")
        or briefing_audit.get("claim_design_distance")
        or briefing_audit.get("support_gap")
        or ""
    )
    max_weight = max(
        max_weight,
        _claim_design_negative_weight(
            flags=briefing_flags,
            risk=briefing_risk,
            distance=briefing_distance,
            explicit_weight=briefing_audit.get("negative_weight"),
        ),
    )
    allowed_interpretations.extend(
        _as_list(briefing_audit.get("allowed_interpretation"))
        + _as_list(briefing_audit.get("allowed_interpretations"))
    )
    disallowed_interpretations.extend(
        _as_list(briefing_audit.get("disallowed_interpretation"))
        + _as_list(briefing_audit.get("disallowed_interpretations"))
    )
    shortages.extend(
        _as_list(briefing_audit.get("design_shortage"))
        + _as_list(briefing_audit.get("design_shortages"))
        + _as_list(briefing_audit.get("controls_missing"))
        + _as_list(briefing_audit.get("missing_controls"))
    )

    flags = _dedupe(all_flags)
    blocks_revision = relation in {"support", "challenge"} and max_weight >= 0.35
    if not has_signal:
        status = "not_supplied"
    elif blocks_revision:
        status = "overclaim_boundary"
    elif max_weight > 0:
        status = "bounded_interpretation"
    else:
        status = "aligned_or_no_overclaim_detected"

    return {
        "schema_id": "claim_design_alignment_audit_v1",
        "status": status,
        "passed": not blocks_revision,
        "relation": relation,
        "rule": (
            "Audit the distance between paper claim scope and experimental design "
            "scope. Negative weight is boundary correction, not a paper-quality "
            "verdict. High claim-design mismatch blocks belief_revision_pressure."
        ),
        "negative_weight": max_weight,
        "blocks_revision": blocks_revision,
        "overclaim_flags": flags,
        "allowed_interpretations": _dedupe(allowed_interpretations),
        "disallowed_interpretations": _dedupe(disallowed_interpretations),
        "shortages": _dedupe(shortages),
        "claim_evaluations": claim_evaluations,
    }


def _key_claim_touch_gate(briefing: dict[str, Any], *, relation: str) -> dict[str, Any]:
    key_claims = _frozen_key_claims(briefing)
    required = [
        item
        for item in key_claims
        if item.get("criticality", "required") == "required"
    ]
    required_ids = [item["id"] for item in required]
    required_by_id = {item["id"]: item for item in required}
    claim_evaluations: list[dict[str, Any]] = []
    touched: set[str] = set()
    measured: set[str] = set()

    for claim in _as_dict_list(briefing.get("claim_map")):
        refs = [ref for ref in _claim_key_refs(claim) if ref in required_by_id]
        if not refs:
            continue
        evidence_types = _claim_evidence_types(claim)
        measured_or_tested = _claim_is_measured_or_tested(claim)
        role_matches = [
            ref
            for ref in refs
            if _claim_type_matches_key_role(claim, required_by_id[ref])
        ]
        touched.update(refs)
        if measured_or_tested:
            measured.update(role_matches)
        claim_evaluations.append(
            {
                "claim_id": str(claim.get("claim_id") or ""),
                "mapped_key_claim_ids": refs,
                "role_matched_key_claim_ids": role_matches,
                "claim_type": _clean_token(claim.get("claim_type") or ""),
                "evidence_types": evidence_types,
                "measured_or_tested": measured_or_tested,
            }
        )

    measured_ids = [item for item in required_ids if item in measured]
    touched_ids = [item for item in required_ids if item in touched]
    missing_measured = [item for item in required_ids if item not in measured]
    if not required:
        status = "no_frozen_key_claims"
        passed = False
    elif relation == "challenge":
        status = "passed" if measured_ids else "failed"
        passed = bool(measured_ids)
    elif relation == "support":
        status = "passed" if not missing_measured else "failed"
        passed = not missing_measured
    else:
        status = "not_directional"
        passed = False

    return {
        "schema_id": "key_claim_touch_gate_v1",
        "status": status,
        "passed": passed,
        "relation": relation,
        "rule": (
            "belief_revision_pressure requires frozen required key claims to be "
            "mapped from claim_map and measured/tested; support must measure all "
            "required key claims, challenge must measure at least one required key claim."
        ),
        "required_key_claims": required,
        "required_key_claim_ids": required_ids,
        "touched_key_claim_ids": touched_ids,
        "measured_key_claim_ids": measured_ids,
        "missing_measured_key_claim_ids": missing_measured,
        "claim_evaluations": claim_evaluations,
    }


def _research_judgment_frame(
    assessment: dict[str, Any],
    briefing: dict[str, Any],
    *,
    relation: str,
    belief_ref: str,
    candidate_question: str,
) -> dict[str, Any]:
    """Build the kernel's deliberative frame before choosing a decision type.

    Relation labels are preserved as secretary signals, but the frame decides
    whether the signal is actionable, merely opens a question, or should be
    downgraded to a read/action request because grounding is insufficient.
    """
    admissibility = (
        assessment.get("admissibility")
        if isinstance(assessment.get("admissibility"), dict)
        else {}
    )
    refresh = (
        assessment.get("refresh_request")
        if isinstance(assessment.get("refresh_request"), dict)
        else {}
    )
    object_grounding = (
        briefing.get("object_grounding")
        if isinstance(briefing.get("object_grounding"), dict)
        else {}
    )
    semantic_audit = (
        briefing.get("semantic_audit")
        if isinstance(briefing.get("semantic_audit"), dict)
        else {}
    )
    calibration = (
        briefing.get("calibration_status")
        if isinstance(briefing.get("calibration_status"), dict)
        else {}
    )
    human_feedback = (
        briefing.get("human_feedback_brief")
        if isinstance(briefing.get("human_feedback_brief"), dict)
        else {}
    )
    attention = (
        briefing.get("attention_brief")
        if isinstance(briefing.get("attention_brief"), dict)
        else {}
    )
    closed_loop = (
        briefing.get("closed_loop_validation")
        if isinstance(briefing.get("closed_loop_validation"), dict)
        else {}
    )
    claims = _as_dict_list(briefing.get("claim_map"))
    disagreements = _as_dict_list(briefing.get("disagreement_map"))
    candidate_layers = _candidate_layers(briefing)
    key_claim_gate = _key_claim_touch_gate(briefing, relation=relation)
    key_claim_gate_passed = _as_bool(key_claim_gate.get("passed"))
    key_claim_gate_blocks_revision = relation in {"support", "challenge"} and not key_claim_gate_passed
    claim_design_audit = _claim_design_alignment_audit(briefing, relation=relation)
    claim_design_blocks_revision = _as_bool(claim_design_audit.get("blocks_revision"))

    read_level = _briefing_read_level(briefing)
    abstract_level_only = _as_bool(admissibility.get("abstract_level_only")) or read_level == "abstract"
    refresh_needed = _as_bool(refresh.get("needed"))
    high_uncertainty = _as_bool(admissibility.get("high_uncertainty"))
    uncertainty_severity = _max_uncertainty_severity(briefing)
    relation_directional = relation in {"support", "challenge"}
    relation_contested = relation == "contest" or bool(disagreements)
    relation_underdetermined = relation in {"underdetermined", "unclear"}

    object_alignment = str(object_grounding.get("object_text_alignment") or "unknown").strip().lower()
    text_only = _as_bool(object_grounding.get("text_only"))
    confidence_downweighted = _as_bool(calibration.get("should_downweight_confidence"))
    semantic_risk = str(semantic_audit.get("behavioral_risk") or "unknown").strip().lower()
    human_reduce_overclaim = (
        str(human_feedback.get("feedback_type") or "").strip().lower()
        in {"reduce_overclaim", "downgrade", "request_caution"}
    )
    closed_loop_commit_allowed = _as_bool(closed_loop.get("kernel_commit_allowed"))

    blockers: list[str] = []
    if abstract_level_only:
        blockers.append("abstract_level_only")
    if refresh_needed:
        blockers.append("refresh_required")
    if relation_contested:
        blockers.append("reader_or_relation_conflict")
    if high_uncertainty or uncertainty_severity == "high":
        blockers.append("high_uncertainty")
    if confidence_downweighted:
        blockers.append("uncalibrated_or_downweighted_confidence")
    if human_reduce_overclaim:
        blockers.append("human_feedback_reduce_overclaim")
    if semantic_risk in {"medium", "high"}:
        blockers.append("semantic_audit_risk")
    if object_alignment in {"none", "weak"}:
        blockers.append("weak_object_grounding")
    if text_only and read_level != "full_text":
        blockers.append("text_only_abstract_grounding")
    if key_claim_gate.get("status") == "no_frozen_key_claims" and relation in {"support", "challenge"}:
        blockers.append("no_frozen_key_claims")
    elif key_claim_gate_blocks_revision:
        blockers.append("key_claim_not_measured_or_tested")
    if claim_design_blocks_revision:
        blockers.append("claim_design_overclaim_boundary")

    source_sufficient = (
        not abstract_level_only
        and not refresh_needed
        and read_level in {"full_text", "paper", "full"}
    )
    conflict_pressure = relation_contested
    source_grounding_pressure = bool(
        abstract_level_only
        or refresh_needed
        or "weak_object_grounding" in blockers
        or "text_only_abstract_grounding" in blockers
    )
    open_question_pressure = bool(
        candidate_question
        and (
            relation_underdetermined
            or high_uncertainty
            or uncertainty_severity in {"medium", "high"}
            or source_grounding_pressure
            or key_claim_gate_blocks_revision
            or claim_design_blocks_revision
        )
    )
    trajectory_pressure = "trajectory" in candidate_layers and not conflict_pressure
    belief_revision_pressure = bool(
        belief_ref
        and relation_directional
        and source_sufficient
        and key_claim_gate_passed
        and not claim_design_blocks_revision
        and not conflict_pressure
        and not high_uncertainty
    )
    action_or_read_pressure = bool(
        conflict_pressure
        or source_grounding_pressure
        or refresh_needed
        or (key_claim_gate_blocks_revision and not candidate_question)
        or (claim_design_blocks_revision and not candidate_question)
        or (human_reduce_overclaim and not source_sufficient)
    )

    if conflict_pressure:
        primary_pressure = "conflict_resolution"
    elif source_grounding_pressure:
        primary_pressure = "source_grounding"
    elif belief_revision_pressure:
        primary_pressure = "belief_revision_pressure"
    elif open_question_pressure:
        primary_pressure = "open_question_pressure"
    elif trajectory_pressure:
        primary_pressure = "trajectory_pressure"
    else:
        primary_pressure = "background_context"

    return {
        "schema_id": "research_judgment_frame_v1",
        "primary_pressure": primary_pressure,
        "source_grounding": {
            "read_level": read_level,
            "abstract_level_only": abstract_level_only,
            "object_text_alignment": object_alignment,
            "text_only": text_only,
            "source_sufficient_for_revision": source_sufficient,
        },
        "relation_signal": {
            "relation": relation,
            "role": "secretary_signal_not_kernel_decision",
            "directional": relation_directional,
            "contested": relation_contested,
            "underdetermined": relation_underdetermined,
        },
        "claim_pressure": {
            "claim_count": len(claims),
            "has_claims": bool(claims),
            "candidate_layers": candidate_layers,
            "belief_ref": belief_ref,
            "candidate_question": candidate_question,
            "key_claim_touch_gate": key_claim_gate,
            "claim_design_alignment_audit": claim_design_audit,
        },
        "reliability_pressure": {
            "uncertainty_severity": uncertainty_severity,
            "high_uncertainty": high_uncertainty,
            "confidence_downweighted": confidence_downweighted,
            "semantic_risk": semantic_risk,
            "closed_loop_commit_allowed": closed_loop_commit_allowed,
        },
        "human_signal": {
            "present": bool(human_feedback),
            "feedback_type": str(human_feedback.get("feedback_type") or ""),
            "passed": _as_bool(human_feedback.get("passed")),
            "attention_recommendation": str(attention.get("attention_recommendation") or ""),
        },
        "warrants": {
            "action_or_read": action_or_read_pressure,
            "belief_revision_candidate": belief_revision_pressure,
            "open_question": open_question_pressure,
            "trajectory_update": trajectory_pressure and not belief_revision_pressure,
        },
        "blockers": _dedupe(blockers),
    }


def _decision_candidate(
    *,
    target_layer: str,
    change_type: str,
    description: str,
    source_candidate_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "target_layer": target_layer,
        "change_type": change_type,
        "description": description,
        "source_candidate_ids": _dedupe(_as_list(source_candidate_ids)),
        "commit_permission": "kernel_revision_api_required",
    }


def decide_research_judgment(
    assessment: dict[str, Any],
    briefing: dict[str, Any],
) -> dict[str, Any]:
    """Create a downstream research-state decision from intake plus briefing.

    This is the first true-kernel handoff object: it decides what kind of
    research-state movement is warranted, while still requiring explicit
    revision APIs for durable projection changes.
    """
    if not isinstance(assessment, dict):
        assessment = {}
    if not isinstance(briefing, dict):
        briefing = {}

    intake_decision = str(assessment.get("decision") or "defer")
    briefing_id = str(
        assessment.get("briefing_id") or briefing.get("briefing_id") or ""
    )
    assessment_id = str(
        assessment.get("assessment_id") or assessment.get("id") or ""
    )
    source_ref = str(assessment.get("source_ref") or _briefing_source_ref(briefing))
    relation = _briefing_relation(briefing)
    belief_ref = _briefing_belief_ref(briefing)
    candidate_question = _briefing_candidate_question(briefing)
    judgment_frame = _research_judgment_frame(
        assessment,
        briefing,
        relation=relation,
        belief_ref=belief_ref,
        candidate_question=candidate_question,
    )
    frame_warrants = (
        judgment_frame.get("warrants")
        if isinstance(judgment_frame.get("warrants"), dict)
        else {}
    )
    candidate_ids = _candidate_ids(_as_dict_list(briefing.get("candidate_kernel_updates")))
    accepted_candidate_ids = _as_list(assessment.get("accepted_candidates"))
    deferred_candidate_ids = _as_list(assessment.get("deferred_candidates"))
    rejected_candidate_ids = _as_list(assessment.get("rejected_candidates"))
    next_actions = _as_list(assessment.get("next_actions"))
    reasons = _as_list(assessment.get("reasons"))
    refresh = (
        assessment.get("refresh_request")
        if isinstance(assessment.get("refresh_request"), dict)
        else {}
    )
    admissibility = (
        assessment.get("admissibility")
        if isinstance(assessment.get("admissibility"), dict)
        else {}
    )
    abstract_level_only = _as_bool(admissibility.get("abstract_level_only"))
    high_uncertainty = _as_bool(admissibility.get("high_uncertainty"))
    refresh_needed = _as_bool(refresh.get("needed"))
    trajectory_candidates = [
        candidate
        for candidate in _as_dict_list(briefing.get("candidate_kernel_updates"))
        if candidate.get("target_layer") == "trajectory"
    ]

    decision = "no_change"
    rationale: list[str] = []
    required_actions: list[str] = []
    rejected_alternatives: list[str] = []
    candidate_state_changes: list[dict[str, Any]] = []
    human_review_required = False

    if assessment_id:
        rationale.append(f"intake_assessment:{assessment_id}")
    rationale.append(f"judgment_frame:{judgment_frame.get('primary_pressure', 'unknown')}")
    if relation and relation != "not_supplied":
        rationale.append(f"secretary_relation_signal:{relation}")
    frame_blockers = _as_list(judgment_frame.get("blockers"))

    if intake_decision == "reject":
        decision = "no_change"
        rationale.append("intake_rejected_packet")
        rejected_alternatives.extend(["propose_belief_revision", "propose_trajectory_update"])
        required_actions.append("regenerate_secretary_briefing")
    elif intake_decision == "escalate":
        decision = "request_action_or_read"
        rationale.extend(reasons or ["intake_escalation_required"])
        required_actions.extend(next_actions or ["human_read"])
        human_review_required = True
        candidate_state_changes.append(
            _decision_candidate(
                target_layer="action",
                change_type="human_or_full_text_review",
                description="Resolve intake-level conflict before any durable kernel revision.",
                source_candidate_ids=deferred_candidate_ids or candidate_ids,
            )
        )
    elif intake_decision in {"request_refresh", "defer"}:
        decision = "request_action_or_read"
        rationale.extend(reasons or [f"intake_{intake_decision}"])
        required_actions.extend(next_actions or ["refresh_secretary_briefing"])
        candidate_state_changes.append(
            _decision_candidate(
                target_layer="action",
                change_type="refresh_required",
                description="Refresh the secretary briefing before research-state judgment.",
                source_candidate_ids=deferred_candidate_ids or candidate_ids,
            )
        )
    elif frame_warrants.get("action_or_read"):
        decision = "request_action_or_read"
        rationale.append("judgment_frame_requires_read_or_action")
        if next_actions:
            required_actions.extend(next_actions)
        elif "no_frozen_key_claims" in frame_blockers:
            required_actions.append("request_belief_key_claims")
        elif "key_claim_not_measured_or_tested" in frame_blockers:
            required_actions.append("request_key_claim_touch_review")
        elif "claim_design_overclaim_boundary" in frame_blockers:
            required_actions.append("request_claim_design_alignment_review")
        else:
            required_actions.append("request_full_text_before_revision")
        candidate_state_changes.append(
            _decision_candidate(
                target_layer="action",
                change_type="full_text_required",
                description="Keep the briefing as intake material, but request full text before belief or trajectory revision.",
                source_candidate_ids=deferred_candidate_ids or candidate_ids,
            )
        )
    elif frame_warrants.get("belief_revision_candidate"):
        decision = "propose_belief_revision"
        rationale.append("judgment_frame_warrants_belief_revision_candidate")
        candidate_state_changes.append(
            _decision_candidate(
                target_layer="belief",
                change_type=f"{relation}_candidate",
                description=(
                    f"Consider a {relation} BeliefRevision candidate for {belief_ref}; "
                    "do not apply without explicit revision API."
                ),
                source_candidate_ids=accepted_candidate_ids or candidate_ids,
            )
        )
    elif relation in {"contest"}:
        decision = "request_action_or_read"
        rationale.append("contested_relation_requires_human_judgment")
        required_actions.extend(next_actions or ["human_read"])
        human_review_required = True
        candidate_state_changes.append(
            _decision_candidate(
                target_layer="action",
                change_type="human_adjudication_required",
                description="The secretary relation is contested; route to human/kernel review before revision.",
                source_candidate_ids=deferred_candidate_ids or candidate_ids,
            )
        )
    elif frame_warrants.get("open_question") and candidate_question:
        decision = "open_question"
        rationale.append("judgment_frame_opens_question")
        candidate_state_changes.append(
            _decision_candidate(
                target_layer="question",
                change_type="open_question_candidate",
                description=candidate_question,
                source_candidate_ids=accepted_candidate_ids or candidate_ids,
            )
        )
    elif frame_warrants.get("trajectory_update") and trajectory_candidates:
        decision = "propose_trajectory_update"
        rationale.append("judgment_frame_warrants_trajectory_update")
        candidate_state_changes.append(
            _decision_candidate(
                target_layer="trajectory",
                change_type="trajectory_update_candidate",
                description=str(
                    trajectory_candidates[0].get("candidate_change")
                    or "Consider trajectory update from secretary proposal."
                ),
                source_candidate_ids=[
                    str(trajectory_candidates[0].get("candidate_id") or "")
                ],
            )
        )
    else:
        decision = "no_change"
        rationale.append("no_research_state_change_warranted")
        rejected_alternatives.extend(["propose_belief_revision", "propose_trajectory_update"])

    if decision != "propose_belief_revision":
        rejected_alternatives.append("direct_belief_revision")
    if decision != "propose_trajectory_update":
        rejected_alternatives.append("direct_trajectory_update")

    uncertainty_drivers = _dedupe(
        reasons
        + _as_list(refresh.get("reasons"))
        + (["abstract_level_only"] if abstract_level_only else [])
        + ([f"relation:{relation}"] if relation in {"contest", "underdetermined", "unclear"} else [])
    )
    if high_uncertainty or human_review_required or relation == "contest":
        uncertainty_level = "high"
    elif abstract_level_only or refresh_needed or relation in {"underdetermined", "unclear"}:
        uncertainty_level = "medium"
    else:
        uncertainty_level = "low"

    affected_objects = {
        "assessment_ids": _dedupe([assessment_id] if assessment_id else []),
        "briefing_ids": _dedupe([briefing_id] if briefing_id else []),
        "source_refs": _dedupe([source_ref] if source_ref else []),
        "belief_refs": _dedupe([belief_ref] if belief_ref else []),
    }

    return normalize_research_judgment_decision(
        {
            "assessment_id": assessment_id,
            "briefing_id": briefing_id,
            "source_ref": source_ref,
            "decision": decision,
            "affected_objects": affected_objects,
            "candidate_state_changes": candidate_state_changes,
            "rejected_alternatives": rejected_alternatives,
            "rationale": rationale,
            "judgment_frame": judgment_frame,
            "uncertainty": {
                "level": uncertainty_level,
                "drivers": uncertainty_drivers,
            },
            "required_actions": required_actions,
            "human_review_required": human_review_required,
            "applied_revision_ids": [],
            "provenance": {
                "method": "research_judgment_decision_v2_frame_first",
                "intake_decision": intake_decision,
                "secretary_relation_signal": relation,
                "note": (
                    "Decision is frame-first: relation labels are secretary signals, "
                    "while judgment_frame determines warranted research-state movement. "
                    "Belief projection still requires explicit BeliefRevision."
                ),
            },
        }
    )


def _validate_float01(record: dict[str, Any], field: str, issues: list[V3Issue], path: str) -> None:
    value = record.get(field)
    if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        issues.append(V3Issue(f"{path}.{field}", "must be a number from 0 to 1"))


def validate_belief(record: dict[str, Any], *, path: str = "$.belief") -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in ("id", "title", "claim", "domain", "status", "created_at", "updated_at"):
        _require(record, field, issues, path)
    if record.get("status") not in ALLOWED_BELIEF_STATUS:
        issues.append(V3Issue(f"{path}.status", f"must be one of {sorted(ALLOWED_BELIEF_STATUS)}"))
    _validate_float01(record, "confidence", issues, path)
    _validate_float01(record, "entrenchment", issues, path)
    for field in (
        "evidence_ids",
        "contra_evidence_ids",
        "pending_evidence_ids",
        "neutral_evidence_ids",
        "contested_evidence_ids",
        "linked_concepts",
        "linked_constraints",
        "linked_questions",
    ):
        if record.get(field, []) is not None and not isinstance(record.get(field, []), list):
            issues.append(V3Issue(f"{path}.{field}", "must be a list"))
    if not isinstance(record.get("provenance"), dict):
        issues.append(V3Issue(f"{path}.provenance", "must be an object"))
    return issues


def validate_evidence(record: dict[str, Any], *, path: str = "$.evidence") -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in ("id", "source_type", "title", "source_ref", "created_at"):
        _require(record, field, issues, path)
    if record.get("source_type") not in ALLOWED_EVIDENCE_SOURCE_TYPE:
        issues.append(
            V3Issue(
                f"{path}.source_type",
                f"must be one of {sorted(ALLOWED_EVIDENCE_SOURCE_TYPE)}",
            )
        )
    if record.get("evidence_strength") not in ALLOWED_EVIDENCE_STRENGTH:
        issues.append(
            V3Issue(
                f"{path}.evidence_strength",
                f"must be one of {sorted(ALLOWED_EVIDENCE_STRENGTH)}",
            )
        )
    for field in (
        "supports_beliefs",
        "challenges_beliefs",
        "pending_beliefs",
        "neutral_beliefs",
        "contests_beliefs",
    ):
        if record.get(field, []) is not None and not isinstance(record.get(field, []), list):
            issues.append(V3Issue(f"{path}.{field}", "must be a list"))
    for field in (
        "parse_boundary",
        "evidence_relation_provenance",
        "evidence_strength_provenance",
    ):
        if record.get(field) is not None and not isinstance(record.get(field), dict):
            issues.append(V3Issue(f"{path}.{field}", "must be an object"))
    return issues


def validate_revision(record: dict[str, Any], *, path: str = "$.revision") -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in (
        "id",
        "belief_id",
        "action",
        "new_confidence",
        "new_entrenchment",
        "reason",
        "created_at",
    ):
        _require(record, field, issues, path)
    if record.get("action") not in ALLOWED_REVISION_ACTION:
        issues.append(
            V3Issue(
                f"{path}.action",
                f"must be one of {sorted(ALLOWED_REVISION_ACTION)}",
            )
        )
    for field in ("new_confidence", "new_entrenchment"):
        _validate_float01(record, field, issues, path)
    for field in ("old_confidence", "old_entrenchment"):
        if record.get(field) is not None:
            _validate_float01(record, field, issues, path)
    if not isinstance(record.get("triggering_evidence_ids"), list):
        issues.append(V3Issue(f"{path}.triggering_evidence_ids", "must be a list"))
    return issues


def validate_judgment(record: dict[str, Any], *, path: str = "$.judgment") -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in ("id", "question", "judgment", "recommendation", "created_at", "updated_at"):
        _require(record, field, issues, path)
    _validate_float01(record, "confidence", issues, path)
    if record.get("risk_level") not in ALLOWED_JUDGMENT_RISK:
        issues.append(V3Issue(f"{path}.risk_level", "must be low, medium, high or unknown"))
    for field in ("linked_beliefs", "linked_constraints", "linked_evidence", "next_actions"):
        if not isinstance(record.get(field), list):
            issues.append(V3Issue(f"{path}.{field}", "must be a list"))
    return issues


def validate_constraint(record: dict[str, Any], *, path: str = "$.constraint") -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in ("id", "statement", "type", "status", "created_at", "updated_at"):
        _require(record, field, issues, path)
    if record.get("type") not in ALLOWED_CONSTRAINT_TYPE:
        issues.append(V3Issue(f"{path}.type", "invalid constraint type"))
    if record.get("status") not in ALLOWED_CONSTRAINT_STATUS:
        issues.append(V3Issue(f"{path}.status", "invalid constraint status"))
    _validate_float01(record, "confidence", issues, path)
    for field in (
        "predicted_observations",
        "failure_conditions",
        "linked_beliefs",
        "linked_evidence",
    ):
        if not isinstance(record.get(field), list):
            issues.append(V3Issue(f"{path}.{field}", "must be a list"))
    return issues


def validate_override(record: dict[str, Any], *, path: str = "$.override") -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in ("id", "target_type", "target_id", "override_type", "user_reason", "created_at"):
        _require(record, field, issues, path)
    if record.get("target_type") not in ALLOWED_OVERRIDE_TARGET:
        issues.append(V3Issue(f"{path}.target_type", "invalid override target type"))
    if record.get("override_type") not in ALLOWED_OVERRIDE_TYPE:
        issues.append(V3Issue(f"{path}.override_type", "invalid override type"))
    return issues


def validate_kernel_intake_assessment(
    record: dict[str, Any],
    *,
    path: str = "$.assessment",
) -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in (
        "id",
        "assessment_id",
        "schema_id",
        "briefing_id",
        "decision",
        "durable_change_authorization",
        "created_at",
    ):
        _require(record, field, issues, path)
    if record.get("schema_id") != "kernel_intake_assessment_v1":
        issues.append(V3Issue(f"{path}.schema_id", "must be kernel_intake_assessment_v1"))
    if record.get("decision") not in ALLOWED_INTAKE_DECISION:
        issues.append(
            V3Issue(
                f"{path}.decision",
                f"must be one of {sorted(ALLOWED_INTAKE_DECISION)}",
            )
        )
    if record.get("durable_change_authorization") not in ALLOWED_INTAKE_AUTHORIZATION:
        issues.append(
            V3Issue(
                f"{path}.durable_change_authorization",
                f"must be one of {sorted(ALLOWED_INTAKE_AUTHORIZATION)}",
            )
        )
    for field in ("admissibility", "refresh_request", "escalation", "provenance"):
        if not isinstance(record.get(field), dict):
            issues.append(V3Issue(f"{path}.{field}", "must be an object"))
    if not isinstance(record.get("boundary_checks"), list):
        issues.append(V3Issue(f"{path}.boundary_checks", "must be a list"))
    else:
        for idx, check in enumerate(record.get("boundary_checks"), 1):
            if not isinstance(check, dict):
                issues.append(V3Issue(f"{path}.boundary_checks:{idx}", "must be an object"))
                continue
            for field in ("check", "passed", "severity", "message"):
                _require(check, field, issues, f"{path}.boundary_checks:{idx}")
            if not isinstance(check.get("passed"), bool):
                issues.append(V3Issue(f"{path}.boundary_checks:{idx}.passed", "must be boolean"))
    for field in (
        "missing_modules",
        "accepted_candidates",
        "deferred_candidates",
        "rejected_candidates",
        "reasons",
        "next_actions",
    ):
        if not isinstance(record.get(field), list):
            issues.append(V3Issue(f"{path}.{field}", "must be a list"))
    if record.get("durable_change_authorization") != "not_authorized":
        issues.append(
            V3Issue(
                f"{path}.durable_change_authorization",
                "V3-alpha assessment cannot authorize durable state changes",
            )
        )
    return issues


def validate_research_judgment_decision(
    record: dict[str, Any],
    *,
    path: str = "$.decision",
) -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in (
        "id",
        "decision_id",
        "schema_id",
        "assessment_id",
        "decision",
        "created_at",
    ):
        _require(record, field, issues, path)
    if record.get("schema_id") != "research_judgment_decision_v1":
        issues.append(V3Issue(f"{path}.schema_id", "must be research_judgment_decision_v1"))
    if record.get("decision") not in ALLOWED_RESEARCH_DECISION:
        issues.append(
            V3Issue(
                f"{path}.decision",
                f"must be one of {sorted(ALLOWED_RESEARCH_DECISION)}",
            )
        )
    if not isinstance(record.get("human_review_required"), bool):
        issues.append(V3Issue(f"{path}.human_review_required", "must be boolean"))
    for field in (
        "candidate_state_changes",
        "rejected_alternatives",
        "rationale",
        "required_actions",
        "applied_revision_ids",
    ):
        if not isinstance(record.get(field), list):
            issues.append(V3Issue(f"{path}.{field}", "must be a list"))
    for field in ("affected_objects", "uncertainty", "provenance"):
        if not isinstance(record.get(field), dict):
            issues.append(V3Issue(f"{path}.{field}", "must be an object"))
    if "judgment_frame" in record and not isinstance(record.get("judgment_frame"), dict):
        issues.append(V3Issue(f"{path}.judgment_frame", "must be an object"))
    uncertainty = record.get("uncertainty") if isinstance(record.get("uncertainty"), dict) else {}
    if uncertainty.get("level", "unknown") not in ALLOWED_UNCERTAINTY_LEVEL:
        issues.append(
            V3Issue(
                f"{path}.uncertainty.level",
                f"must be one of {sorted(ALLOWED_UNCERTAINTY_LEVEL)}",
            )
        )
    if uncertainty.get("drivers") is not None and not isinstance(uncertainty.get("drivers"), list):
        issues.append(V3Issue(f"{path}.uncertainty.drivers", "must be a list"))
    for idx, candidate in enumerate(record.get("candidate_state_changes", []), 1):
        candidate_path = f"{path}.candidate_state_changes:{idx}"
        if not isinstance(candidate, dict):
            issues.append(V3Issue(candidate_path, "must be an object"))
            continue
        for field in ("target_layer", "change_type", "description", "commit_permission"):
            _require(candidate, field, issues, candidate_path)
        if candidate.get("target_layer") not in ALLOWED_RESEARCH_DECISION_LAYER:
            issues.append(
                V3Issue(
                    f"{candidate_path}.target_layer",
                    f"must be one of {sorted(ALLOWED_RESEARCH_DECISION_LAYER)}",
                )
            )
        if candidate.get("commit_permission") != "kernel_revision_api_required":
            issues.append(
                V3Issue(
                    f"{candidate_path}.commit_permission",
                    "must be kernel_revision_api_required",
                )
            )
        if not isinstance(candidate.get("source_candidate_ids", []), list):
            issues.append(V3Issue(f"{candidate_path}.source_candidate_ids", "must be a list"))
    return issues


def validate_question(record: dict[str, Any], *, path: str = "$.question") -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in (
        "id",
        "question_id",
        "schema_id",
        "question",
        "status",
        "source_decision_id",
        "created_at",
        "updated_at",
    ):
        _require(record, field, issues, path)
    if record.get("schema_id") != "research_question_v1":
        issues.append(V3Issue(f"{path}.schema_id", "must be research_question_v1"))
    if record.get("status") not in ALLOWED_QUESTION_STATUS:
        issues.append(
            V3Issue(
                f"{path}.status",
                f"must be one of {sorted(ALLOWED_QUESTION_STATUS)}",
            )
        )
    if record.get("uncertainty_level") not in ALLOWED_UNCERTAINTY_LEVEL:
        issues.append(
            V3Issue(
                f"{path}.uncertainty_level",
                f"must be one of {sorted(ALLOWED_UNCERTAINTY_LEVEL)}",
            )
        )
    for field in ("linked_beliefs", "linked_evidence", "rationale"):
        if not isinstance(record.get(field), list):
            issues.append(V3Issue(f"{path}.{field}", "must be a list"))
    if not isinstance(record.get("provenance"), dict):
        issues.append(V3Issue(f"{path}.provenance", "must be an object"))
    return issues


def validate_question_revision(
    record: dict[str, Any],
    *,
    path: str = "$.question_revision",
) -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in (
        "id",
        "revision_id",
        "schema_id",
        "question_id",
        "action",
        "new_status",
        "reason",
        "source_decision_id",
        "created_at",
    ):
        _require(record, field, issues, path)
    if record.get("schema_id") != "research_question_revision_v1":
        issues.append(
            V3Issue(f"{path}.schema_id", "must be research_question_revision_v1")
        )
    if record.get("action") not in ALLOWED_QUESTION_REVISION_ACTION:
        issues.append(
            V3Issue(
                f"{path}.action",
                f"must be one of {sorted(ALLOWED_QUESTION_REVISION_ACTION)}",
            )
        )
    if record.get("new_status") not in ALLOWED_QUESTION_STATUS:
        issues.append(
            V3Issue(
                f"{path}.new_status",
                f"must be one of {sorted(ALLOWED_QUESTION_STATUS)}",
            )
        )
    if record.get("old_status") not in ("", None) and record.get("old_status") not in ALLOWED_QUESTION_STATUS:
        issues.append(
            V3Issue(
                f"{path}.old_status",
                f"must be blank or one of {sorted(ALLOWED_QUESTION_STATUS)}",
            )
        )
    if not isinstance(record.get("provenance"), dict):
        issues.append(V3Issue(f"{path}.provenance", "must be an object"))
    return issues


def validate_trajectory(
    record: dict[str, Any],
    *,
    path: str = "$.trajectory",
) -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in (
        "id",
        "trajectory_id",
        "schema_id",
        "title",
        "statement",
        "status",
        "source_decision_id",
        "created_at",
        "updated_at",
    ):
        _require(record, field, issues, path)
    if record.get("schema_id") != "research_trajectory_v1":
        issues.append(V3Issue(f"{path}.schema_id", "must be research_trajectory_v1"))
    if record.get("status") not in ALLOWED_TRAJECTORY_STATUS:
        issues.append(
            V3Issue(
                f"{path}.status",
                f"must be one of {sorted(ALLOWED_TRAJECTORY_STATUS)}",
            )
        )
    for field in ("linked_questions", "linked_beliefs", "rationale"):
        if not isinstance(record.get(field), list):
            issues.append(V3Issue(f"{path}.{field}", "must be a list"))
    if not isinstance(record.get("provenance"), dict):
        issues.append(V3Issue(f"{path}.provenance", "must be an object"))
    return issues


def validate_trajectory_revision(
    record: dict[str, Any],
    *,
    path: str = "$.trajectory_revision",
) -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in (
        "id",
        "revision_id",
        "schema_id",
        "trajectory_id",
        "action",
        "new_status",
        "reason",
        "source_decision_id",
        "created_at",
    ):
        _require(record, field, issues, path)
    if record.get("schema_id") != "research_trajectory_revision_v1":
        issues.append(
            V3Issue(f"{path}.schema_id", "must be research_trajectory_revision_v1")
        )
    if record.get("action") not in ALLOWED_TRAJECTORY_REVISION_ACTION:
        issues.append(
            V3Issue(
                f"{path}.action",
                f"must be one of {sorted(ALLOWED_TRAJECTORY_REVISION_ACTION)}",
            )
        )
    if record.get("new_status") not in ALLOWED_TRAJECTORY_STATUS:
        issues.append(
            V3Issue(
                f"{path}.new_status",
                f"must be one of {sorted(ALLOWED_TRAJECTORY_STATUS)}",
            )
        )
    if record.get("old_status") not in ("", None) and record.get("old_status") not in ALLOWED_TRAJECTORY_STATUS:
        issues.append(
            V3Issue(
                f"{path}.old_status",
                f"must be blank or one of {sorted(ALLOWED_TRAJECTORY_STATUS)}",
            )
        )
    if not isinstance(record.get("provenance"), dict):
        issues.append(V3Issue(f"{path}.provenance", "must be an object"))
    return issues


def validate_action_record(
    record: dict[str, Any],
    *,
    path: str = "$.action",
) -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in (
        "id",
        "action_id",
        "schema_id",
        "decision_id",
        "action_type",
        "description",
        "status",
        "priority",
        "created_at",
        "updated_at",
    ):
        _require(record, field, issues, path)
    if record.get("schema_id") != "kernel_action_record_v1":
        issues.append(V3Issue(f"{path}.schema_id", "must be kernel_action_record_v1"))
    if record.get("action_type") not in ALLOWED_ACTION_RECORD_TYPE:
        issues.append(
            V3Issue(
                f"{path}.action_type",
                f"must be one of {sorted(ALLOWED_ACTION_RECORD_TYPE)}",
            )
        )
    if record.get("status") not in ALLOWED_ACTION_RECORD_STATUS:
        issues.append(
            V3Issue(
                f"{path}.status",
                f"must be one of {sorted(ALLOWED_ACTION_RECORD_STATUS)}",
            )
        )
    if record.get("priority") not in ALLOWED_ACTION_PRIORITY:
        issues.append(
            V3Issue(
                f"{path}.priority",
                f"must be one of {sorted(ALLOWED_ACTION_PRIORITY)}",
            )
        )
    if not isinstance(record.get("provenance"), dict):
        issues.append(V3Issue(f"{path}.provenance", "must be an object"))
    return issues


def validate_human_review_request(
    record: dict[str, Any],
    *,
    path: str = "$.human_review_request",
) -> list[V3Issue]:
    issues: list[V3Issue] = []
    for field in (
        "id",
        "request_id",
        "schema_id",
        "decision_id",
        "request_type",
        "target_state_layer",
        "question",
        "status",
        "priority",
        "created_at",
        "updated_at",
    ):
        _require(record, field, issues, path)
    if record.get("schema_id") != "human_review_request_v1":
        issues.append(V3Issue(f"{path}.schema_id", "must be human_review_request_v1"))
    if record.get("request_type") not in ALLOWED_HUMAN_REVIEW_REQUEST_TYPE:
        issues.append(
            V3Issue(
                f"{path}.request_type",
                f"must be one of {sorted(ALLOWED_HUMAN_REVIEW_REQUEST_TYPE)}",
            )
        )
    if record.get("target_state_layer") not in ALLOWED_RESEARCH_DECISION_LAYER:
        issues.append(
            V3Issue(
                f"{path}.target_state_layer",
                f"must be one of {sorted(ALLOWED_RESEARCH_DECISION_LAYER)}",
            )
        )
    if record.get("status") not in ALLOWED_HUMAN_REVIEW_STATUS:
        issues.append(
            V3Issue(
                f"{path}.status",
                f"must be one of {sorted(ALLOWED_HUMAN_REVIEW_STATUS)}",
            )
        )
    if record.get("priority") not in ALLOWED_ACTION_PRIORITY:
        issues.append(
            V3Issue(
                f"{path}.priority",
                f"must be one of {sorted(ALLOWED_ACTION_PRIORITY)}",
            )
        )
    if not isinstance(record.get("linked_action_ids"), list):
        issues.append(V3Issue(f"{path}.linked_action_ids", "must be a list"))
    if not isinstance(record.get("allowed_responses"), list):
        issues.append(V3Issue(f"{path}.allowed_responses", "must be a list"))
    else:
        for idx, option in enumerate(record.get("allowed_responses", []), 1):
            option_path = f"{path}.allowed_responses:{idx}"
            if not isinstance(option, dict):
                issues.append(V3Issue(option_path, "must be an object"))
                continue
            for field in ("value", "label"):
                _require(option, field, issues, option_path)
            if option.get("value") not in ALLOWED_HUMAN_RESPONSE_OPTION:
                issues.append(
                    V3Issue(
                        f"{option_path}.value",
                        f"must be one of {sorted(ALLOWED_HUMAN_RESPONSE_OPTION)}",
                    )
                )
    for field in ("reviewer_payload", "kernel_context", "anti_anchoring", "provenance"):
        if not isinstance(record.get(field), dict):
            issues.append(V3Issue(f"{path}.{field}", "must be an object"))
    return issues


def create_evidence(kernel_dir: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    ensure_kernel_dirs(kernel_dir)
    evidence = normalize_evidence(record)
    issues = validate_evidence(evidence)
    if issues:
        raise V3KernelValidationError(issues)
    existing = {item["id"] for item in read_jsonl(object_path(kernel_dir, "evidence"))}
    if evidence["id"] not in existing:
        append_jsonl(object_path(kernel_dir, "evidence"), evidence)
    return evidence


def create_belief(
    kernel_dir: str | Path,
    record: dict[str, Any],
    *,
    reason: str,
    evidence_ids: list[str] | None = None,
    human_override_id: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_kernel_dirs(kernel_dir)
    belief = normalize_belief(record)
    belief["evidence_ids"] = _dedupe(belief["evidence_ids"] + _as_list(evidence_ids))
    belief["updated_at"] = now_iso()
    revision = normalize_revision(
        {
            "belief_id": belief["id"],
            "action": "create",
            "old_confidence": None,
            "new_confidence": belief["confidence"],
            "old_entrenchment": None,
            "new_entrenchment": belief["entrenchment"],
            "reason": reason,
            "triggering_evidence_ids": belief["evidence_ids"],
            "human_override_id": human_override_id,
        }
    )
    belief["last_revision_id"] = revision["id"]

    issues = validate_belief(belief) + validate_revision(revision)
    if issues:
        raise V3KernelValidationError(issues)

    beliefs = read_jsonl(object_path(kernel_dir, "beliefs"))
    if any(item["id"] == belief["id"] for item in beliefs):
        raise V3KernelValidationError([V3Issue("$.belief.id", f"already exists: {belief['id']}")])
    beliefs.append(belief)
    write_jsonl(object_path(kernel_dir, "beliefs"), beliefs)
    append_jsonl(object_path(kernel_dir, "revisions"), revision)
    return belief, revision


def _action_status(current: str, action: str) -> str:
    if action in {"contest"}:
        return "contested"
    if action in {"challenge"}:
        return "challenged"
    if action in {"contradict"}:
        return "contradicted"
    if action in {"resolve"}:
        return "resolved"
    if action in {"archive"}:
        return "archived"
    if action in {"reopen", "strengthen"}:
        return "active"
    return current


def _evidence_direction(item: dict[str, Any], belief_id: str, action: str) -> str:
    if belief_id in item.get("contests_beliefs", []):
        return "contest"
    if belief_id in item.get("pending_beliefs", []):
        return "underdetermined"
    if belief_id in item.get("neutral_beliefs", []):
        return "neutral"
    if belief_id in item.get("challenges_beliefs", []):
        return "challenge"
    if belief_id in item.get("supports_beliefs", []):
        return "support"
    if action in {"weaken", "challenge", "contradict"}:
        return "challenge"
    if action == "contest":
        return "contest"
    if (
        action == "update"
        and item.get("evidence_relation_provenance", {}).get("relation") == "underdetermined"
    ):
        return "underdetermined"
    if (
        action == "update"
        and item.get("evidence_relation_provenance", {}).get("relation") == "neutral"
    ):
        return "neutral"
    return "support"


def _evidence_delta(
    evidence: list[dict[str, Any]],
    belief_id: str,
    action: str,
    current_belief: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    if action in {"archive", "resolve", "reopen", "update", "contest"}:
        return 0.0, {
            "method": "no_confidence_delta_for_state_action",
            "action": action,
            "steps": [],
        }
    existing_counts = {
        "support": len(_as_list((current_belief or {}).get("evidence_ids"))),
        "challenge": len(_as_list((current_belief or {}).get("contra_evidence_ids"))),
    }
    entrenchment = _clamp01((current_belief or {}).get("entrenchment", 0.0), default=0.0)
    entrenchment_resistance_factor = round(
        1 / (1 + ENTRENCHMENT_CONFIDENCE_RESISTANCE_K * entrenchment),
        4,
    )
    batch_counts = {"support": 0, "challenge": 0}
    delta = 0.0
    steps = []
    for item in evidence:
        direction = _evidence_direction(item, belief_id, action)
        if direction in {"contest", "underdetermined", "neutral"}:
            steps.append(
                {
                    "evidence_id": item.get("id", ""),
                    "direction": direction,
                    "raw_delta": 0.0,
                    "damped_delta": 0.0,
                    "reason": f"{direction} evidence does not move confidence",
                }
            )
            continue
        strength_delta = EVIDENCE_DELTAS.get(item.get("evidence_strength"), 0.10)
        repeated_count = existing_counts[direction] + batch_counts[direction]
        damping_factor = 1 / ((repeated_count + 1) ** 0.5)
        damped = round(strength_delta * damping_factor * entrenchment_resistance_factor, 4)
        signed_delta = -damped if direction == "challenge" else damped
        delta += signed_delta
        batch_counts[direction] += 1
        steps.append(
            {
                "evidence_id": item.get("id", ""),
                "direction": direction,
                "strength": item.get("evidence_strength", ""),
                "raw_delta": strength_delta,
                "repeat_count_before": repeated_count,
                "damping_factor": round(damping_factor, 4),
                "entrenchment": entrenchment,
                "entrenchment_resistance_k": ENTRENCHMENT_CONFIDENCE_RESISTANCE_K,
                "entrenchment_resistance_factor": entrenchment_resistance_factor,
                "damped_delta": signed_delta,
            }
        )
    return round(delta, 4), {
        "method": "diminishing_returns_with_entrenchment_resistance_v1",
        "existing_counts": existing_counts,
        "batch_counts": batch_counts,
        "entrenchment": entrenchment,
        "entrenchment_resistance_k": ENTRENCHMENT_CONFIDENCE_RESISTANCE_K,
        "entrenchment_resistance_factor": entrenchment_resistance_factor,
        "steps": steps,
    }


def _confidence_cap(
    trigger_evidence: list[dict[str, Any]],
    *,
    human_override_id: str = "",
) -> dict[str, Any]:
    if human_override_id:
        return {
            "upper": 1.0,
            "lower": 0.0,
            "reason": "human_override_allows_full_range",
        }
    has_decisive = any(
        item.get("evidence_strength") == "decisive" for item in trigger_evidence
    )
    return {
        "upper": CONFIDENCE_DECISIVE_CAP if has_decisive else CONFIDENCE_SOFT_CAP,
        "lower": CONFIDENCE_DECISIVE_FLOOR if has_decisive else CONFIDENCE_SOFT_FLOOR,
        "reason": "decisive_evidence_cap" if has_decisive else "unvalidated_soft_cap",
    }


def _apply_confidence_policy(
    old_confidence: float,
    raw_confidence: float,
    trigger_evidence: list[dict[str, Any]],
    *,
    human_override_id: str = "",
) -> tuple[float, dict[str, Any]]:
    cap = _confidence_cap(trigger_evidence, human_override_id=human_override_id)
    capped = max(cap["lower"], min(cap["upper"], raw_confidence))
    return _clamp01(capped, default=old_confidence), {
        "method": "soft_cap_without_human_or_decisive_validation_v1",
        "raw_confidence": round(raw_confidence, 4),
        "capped_confidence": round(capped, 4),
        **cap,
    }


def _dependency_count(belief: dict[str, Any]) -> int:
    return (
        len(_as_list(belief.get("linked_concepts")))
        + len(_as_list(belief.get("linked_constraints")))
        + len(_as_list(belief.get("linked_questions")))
    )


def _existing_source_type_counts(
    current_belief: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for evidence_id in _as_list(current_belief.get("evidence_ids")):
        source_type = evidence_by_id.get(evidence_id, {}).get("source_type", "external")
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts


def _entrenchment_delta(
    trigger_evidence: list[dict[str, Any]],
    belief_id: str,
    action: str,
    *,
    current_belief: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    human_override_id: str = "",
) -> tuple[float, dict[str, Any]]:
    if action in {"archive", "resolve", "reopen", "update", "contest"}:
        return 0.0, {
            "method": "no_entrenchment_delta_for_state_action",
            "action": action,
            "steps": [],
        }

    source_type_counts = _existing_source_type_counts(current_belief, evidence_by_id)
    batch_source_counts: dict[str, int] = {}
    dependency_count = _dependency_count(current_belief)
    dependency_multiplier = 1 + min(dependency_count, 5) * 0.08
    delta = 0.0
    steps = []

    for item in trigger_evidence:
        direction = _evidence_direction(item, belief_id, action)
        if direction in {"contest", "underdetermined", "neutral"}:
            steps.append(
                {
                    "evidence_id": item.get("id", ""),
                    "direction": direction,
                    "damped_delta": 0.0,
                    "reason": f"{direction} evidence does not alter entrenchment",
                }
            )
            continue

        source_type = item.get("source_type", "external")
        base_delta = ENTRENCHMENT_SOURCE_DELTAS.get(source_type, 0.003)
        repeated_count = source_type_counts.get(source_type, 0) + batch_source_counts.get(
            source_type,
            0,
        )
        damping_factor = 1 / ((repeated_count + 1) ** 0.5)
        damped = base_delta * damping_factor
        if direction == "support":
            signed_delta = damped * dependency_multiplier
        else:
            signed_delta = -(damped / dependency_multiplier)
        signed_delta = round(signed_delta, 4)
        delta += signed_delta
        batch_source_counts[source_type] = batch_source_counts.get(source_type, 0) + 1
        steps.append(
            {
                "evidence_id": item.get("id", ""),
                "direction": direction,
                "source_type": source_type,
                "base_delta": base_delta,
                "repeat_count_before": repeated_count,
                "damping_factor": round(damping_factor, 4),
                "dependency_count": dependency_count,
                "dependency_multiplier": round(dependency_multiplier, 4),
                "damped_delta": signed_delta,
            }
        )

    if human_override_id:
        delta += ENTRENCHMENT_HUMAN_OVERRIDE_DELTA
        steps.append(
            {
                "evidence_id": "",
                "direction": "human_override",
                "damped_delta": ENTRENCHMENT_HUMAN_OVERRIDE_DELTA,
                "reason": "human override increases belief revision cost",
            }
        )

    return round(delta, 4), {
        "method": "dependency_weighted_entrenchment_v1",
        "dependency_count": dependency_count,
        "dependency_multiplier": round(dependency_multiplier, 4),
        "existing_source_type_counts": source_type_counts,
        "batch_source_type_counts": batch_source_counts,
        "steps": steps,
    }


def _apply_entrenchment_policy(
    old_entrenchment: float,
    raw_entrenchment: float,
    *,
    human_override_id: str = "",
) -> tuple[float, dict[str, Any]]:
    capped = max(0.0, min(1.0, raw_entrenchment))
    return _clamp01(capped, default=old_entrenchment), {
        "method": "bounded_revision_cost_projection_v1",
        "raw_entrenchment": round(raw_entrenchment, 4),
        "capped_entrenchment": round(capped, 4),
        "human_override": bool(human_override_id),
    }


def revise_belief(
    kernel_dir: str | Path,
    *,
    belief_id: str,
    evidence_ids: list[str] | None,
    action: str,
    reason: str,
    confidence_delta: float | None = None,
    entrenchment_delta: float | None = None,
    human_override_id: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revise a belief and always append a BeliefRevision."""
    ensure_kernel_dirs(kernel_dir)
    if action not in ALLOWED_REVISION_ACTION or action == "create":
        raise V3KernelValidationError([V3Issue("$.revision.action", "invalid revise action")])

    beliefs = read_jsonl(object_path(kernel_dir, "beliefs"))
    belief_index = next((idx for idx, item in enumerate(beliefs) if item["id"] == belief_id), None)
    if belief_index is None:
        raise V3KernelValidationError([V3Issue("$.belief_id", f"unknown belief: {belief_id}")])

    evidence_by_id = {
        item["id"]: item for item in read_jsonl(object_path(kernel_dir, "evidence"))
    }
    trigger_ids = _as_list(evidence_ids)
    trigger_evidence = []
    missing = []
    for evidence_id in trigger_ids:
        item = evidence_by_id.get(evidence_id)
        if item:
            trigger_evidence.append(item)
        else:
            missing.append(evidence_id)
    if missing:
        raise V3KernelValidationError(
            [V3Issue("$.triggering_evidence_ids", f"unknown evidence: {', '.join(missing)}")]
        )

    old = dict(beliefs[belief_index])
    delta = confidence_delta
    delta_policy: dict[str, Any] = {"method": "caller_supplied_confidence_delta"}
    if delta is None:
        delta, delta_policy = _evidence_delta(
            trigger_evidence,
            belief_id,
            action,
            current_belief=old,
        )
    raw_confidence = _clamp01(old["confidence"] + delta, default=old["confidence"])
    new_confidence, confidence_policy = _apply_confidence_policy(
        old["confidence"],
        raw_confidence,
        trigger_evidence,
        human_override_id=human_override_id,
    )
    entrenchment_delta_value = entrenchment_delta
    entrenchment_delta_policy: dict[str, Any] = {
        "method": "caller_supplied_entrenchment_delta",
    }
    if entrenchment_delta_value is None:
        entrenchment_delta_value, entrenchment_delta_policy = _entrenchment_delta(
            trigger_evidence,
            belief_id,
            action,
            current_belief=old,
            evidence_by_id=evidence_by_id,
            human_override_id=human_override_id,
        )
    raw_entrenchment = _clamp01(
        old["entrenchment"] + entrenchment_delta_value,
        default=old["entrenchment"],
    )
    new_entrenchment, entrenchment_policy = _apply_entrenchment_policy(
        old["entrenchment"],
        raw_entrenchment,
        human_override_id=human_override_id,
    )

    updated = dict(old)
    updated["confidence"] = new_confidence
    updated["entrenchment"] = new_entrenchment
    updated["status"] = _action_status(old["status"], action)
    updated["updated_at"] = now_iso()
    trigger_direction_by_id = {
        item["id"]: _evidence_direction(item, belief_id, action)
        for item in trigger_evidence
    }
    trigger_id_set = set(trigger_direction_by_id)

    def _retain_non_trigger(field: str) -> list[str]:
        return [
            evidence_id
            for evidence_id in updated.get(field, [])
            if evidence_id not in trigger_id_set
        ]

    updated["evidence_ids"] = _dedupe(
        _retain_non_trigger("evidence_ids")
        + [
            evidence_id
            for evidence_id, direction in trigger_direction_by_id.items()
            if direction == "support"
        ]
    )
    updated["contra_evidence_ids"] = _dedupe(
        _retain_non_trigger("contra_evidence_ids")
        + [
            evidence_id
            for evidence_id, direction in trigger_direction_by_id.items()
            if direction == "challenge"
        ]
    )
    updated["pending_evidence_ids"] = _dedupe(
        _retain_non_trigger("pending_evidence_ids")
        + [
            evidence_id
            for evidence_id, direction in trigger_direction_by_id.items()
            if direction == "underdetermined"
        ]
    )
    updated["neutral_evidence_ids"] = _dedupe(
        _retain_non_trigger("neutral_evidence_ids")
        + [
            evidence_id
            for evidence_id, direction in trigger_direction_by_id.items()
            if direction == "neutral"
        ]
    )
    updated["contested_evidence_ids"] = _dedupe(
        _retain_non_trigger("contested_evidence_ids")
        + [
            evidence_id
            for evidence_id, direction in trigger_direction_by_id.items()
            if direction == "contest"
        ]
    )
    revision = normalize_revision(
        {
            "belief_id": belief_id,
            "action": action,
            "old_confidence": old["confidence"],
            "new_confidence": new_confidence,
            "old_entrenchment": old["entrenchment"],
            "new_entrenchment": new_entrenchment,
            "reason": reason,
            "triggering_evidence_ids": trigger_ids,
            "human_override_id": human_override_id,
            "confidence_delta": round(new_confidence - old["confidence"], 4),
            "confidence_delta_policy": delta_policy,
            "confidence_policy": confidence_policy,
            "entrenchment_delta": round(new_entrenchment - old["entrenchment"], 4),
            "entrenchment_delta_policy": entrenchment_delta_policy,
            "entrenchment_policy": entrenchment_policy,
        }
    )
    updated["last_revision_id"] = revision["id"]

    issues = validate_belief(updated) + validate_revision(revision)
    if issues:
        raise V3KernelValidationError(issues)

    beliefs[belief_index] = updated
    write_jsonl(object_path(kernel_dir, "beliefs"), beliefs)
    append_jsonl(object_path(kernel_dir, "revisions"), revision)
    return updated, revision


def _evidence_queue_item(belief: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "belief_id": belief["id"],
        "belief_title": belief.get("title", ""),
        "belief_claim": belief.get("claim", ""),
        "evidence_id": evidence["id"],
        "evidence_title": evidence.get("title", ""),
        "source_ref": evidence.get("source_ref", ""),
        "summary": evidence.get("summary", ""),
        "evidence_strength": evidence.get("evidence_strength", ""),
        "evidence_strength_provenance": evidence.get(
            "evidence_strength_provenance",
            {},
        ),
        "relation_provenance": evidence.get(
            "evidence_relation_provenance",
            {},
        ),
        "parse_boundary": evidence.get("parse_boundary", {}),
        "created_at": evidence.get("created_at", ""),
    }


def _get_evidence_queue(kernel_dir: str | Path, belief_field: str) -> list[dict[str, Any]]:
    ensure_kernel_dirs(kernel_dir)
    beliefs = read_jsonl(object_path(kernel_dir, "beliefs"))
    evidence_by_id = {
        item["id"]: item for item in read_jsonl(object_path(kernel_dir, "evidence"))
    }
    queue = []
    for belief in beliefs:
        for evidence_id in belief.get(belief_field, []):
            evidence = evidence_by_id.get(evidence_id)
            if not evidence:
                continue
            queue.append(_evidence_queue_item(belief, evidence))
    return sorted(queue, key=lambda item: item.get("created_at", ""), reverse=True)


def get_contested_evidence_queue(kernel_dir: str | Path) -> list[dict[str, Any]]:
    """Return evidence whose sign needs human adjudication."""
    return _get_evidence_queue(kernel_dir, "contested_evidence_ids")


def get_pending_evidence_queue(kernel_dir: str | Path) -> list[dict[str, Any]]:
    """Return underdetermined evidence waiting for more evidence, not human adjudication."""
    return _get_evidence_queue(kernel_dir, "pending_evidence_ids")


def reclassify_pending_evidence(
    kernel_dir: str | Path,
    *,
    belief_id: str,
    evidence_id: str,
    relation: str,
    reason: str,
    human_override_id: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move pending evidence to a resolved relation and append a belief revision."""
    ensure_kernel_dirs(kernel_dir)
    relation = _normalize_relation(relation)
    if relation == "unclear":
        relation = "underdetermined"
    if relation not in {"support", "challenge", "contest", "underdetermined", "neutral"}:
        raise V3KernelValidationError(
            [V3Issue("$.relation", "must be support, challenge, contest, underdetermined, or neutral")]
        )

    beliefs = read_jsonl(object_path(kernel_dir, "beliefs"))
    belief = next((item for item in beliefs if item["id"] == belief_id), None)
    if not belief:
        raise V3KernelValidationError([V3Issue("$.belief_id", f"unknown belief: {belief_id}")])

    evidence_items = read_jsonl(object_path(kernel_dir, "evidence"))
    evidence_index = next(
        (idx for idx, item in enumerate(evidence_items) if item["id"] == evidence_id),
        None,
    )
    if evidence_index is None:
        raise V3KernelValidationError(
            [V3Issue("$.evidence_id", f"unknown evidence: {evidence_id}")]
        )

    evidence = dict(evidence_items[evidence_index])
    if (
        evidence_id not in belief.get("pending_evidence_ids", [])
        and belief_id not in evidence.get("pending_beliefs", [])
    ):
        raise V3KernelValidationError(
            [V3Issue("$.evidence_id", "evidence is not pending for this belief")]
        )

    for field in (
        "supports_beliefs",
        "challenges_beliefs",
        "pending_beliefs",
        "neutral_beliefs",
        "contests_beliefs",
    ):
        evidence[field] = [item for item in _as_list(evidence.get(field)) if item != belief_id]

    if relation == "support":
        evidence["supports_beliefs"].append(belief_id)
        action = "strengthen"
        weak_direction = "support"
        adjudication_type = "none"
    elif relation == "challenge":
        evidence["challenges_beliefs"].append(belief_id)
        action = "challenge"
        weak_direction = "challenge"
        adjudication_type = "none"
    elif relation == "contest":
        evidence["contests_beliefs"].append(belief_id)
        action = "contest"
        weak_direction = "unclear"
        adjudication_type = "human"
    elif relation == "neutral":
        evidence["neutral_beliefs"].append(belief_id)
        action = "update"
        weak_direction = "neutral"
        adjudication_type = "none"
    else:
        evidence["pending_beliefs"].append(belief_id)
        action = "update"
        weak_direction = (
            evidence.get("evidence_relation_provenance", {}).get("weak_direction")
            or "unclear"
        )
        adjudication_type = "more_evidence"

    previous_provenance = dict(evidence.get("evidence_relation_provenance") or {})
    evidence["evidence_relation_provenance"] = {
        "method": "pending_reclassification_v1",
        "previous_method": previous_provenance.get("method", ""),
        "previous_relation": previous_provenance.get("relation", ""),
        "relation": relation,
        "contested": relation == "contest",
        "underdetermined": relation == "underdetermined",
        "neutral": relation == "neutral",
        "needs_human": adjudication_type == "human",
        "needs_more_evidence": adjudication_type == "more_evidence",
        "adjudication_type": adjudication_type,
        "weak_direction": weak_direction,
        "reason": reason,
        "human_override_id": human_override_id,
        "reclassified_at": now_iso(),
        "previous": previous_provenance,
    }
    evidence_items[evidence_index] = normalize_evidence(evidence)
    write_jsonl(object_path(kernel_dir, "evidence"), evidence_items)

    return revise_belief(
        kernel_dir,
        belief_id=belief_id,
        evidence_ids=[evidence_id],
        action=action,
        reason=reason,
        human_override_id=human_override_id,
    )


def create_judgment(kernel_dir: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    ensure_kernel_dirs(kernel_dir)
    judgment = normalize_judgment(record)
    issues = validate_judgment(judgment)
    if issues:
        raise V3KernelValidationError(issues)
    append_jsonl(object_path(kernel_dir, "judgments"), judgment)
    return judgment


def create_constraint(kernel_dir: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    ensure_kernel_dirs(kernel_dir)
    constraint = normalize_constraint(record)
    issues = validate_constraint(constraint)
    if issues:
        raise V3KernelValidationError(issues)
    append_jsonl(object_path(kernel_dir, "constraints"), constraint)
    return constraint


def record_human_override(kernel_dir: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    ensure_kernel_dirs(kernel_dir)
    override = normalize_override(record)
    issues = validate_override(override)
    if issues:
        raise V3KernelValidationError(issues)
    append_jsonl(object_path(kernel_dir, "overrides"), override)
    return override


def create_kernel_intake_assessment(
    kernel_dir: str | Path,
    briefing: dict[str, Any],
) -> dict[str, Any]:
    """Append a kernel intake assessment for a secretary briefing.

    The assessment is durable audit state, but it is not a BeliefRevision and
    cannot change belief confidence, entrenchment, trajectory, or judgments.
    """
    ensure_kernel_dirs(kernel_dir)
    assessment = assess_secretary_briefing(briefing)
    issues = validate_kernel_intake_assessment(assessment)
    if issues:
        raise V3KernelValidationError(issues)
    append_jsonl(object_path(kernel_dir, "assessments"), assessment)
    return assessment


def create_research_judgment_decision(
    kernel_dir: str | Path,
    assessment: dict[str, Any],
    briefing: dict[str, Any],
) -> dict[str, Any]:
    """Append a research judgment decision downstream of intake.

    The decision may propose state changes or request action, but it never
    creates BeliefRevision records or mutates belief projections.
    """
    ensure_kernel_dirs(kernel_dir)
    assessment_id = str(assessment.get("assessment_id") or assessment.get("id") or "")
    known_assessments = {
        item.get("assessment_id") or item.get("id")
        for item in read_jsonl(object_path(kernel_dir, "assessments"))
    }
    if assessment_id and assessment_id not in known_assessments:
        raise V3KernelValidationError(
            [V3Issue("$.assessment_id", f"unknown assessment: {assessment_id}")]
        )
    decision = decide_research_judgment(assessment, briefing)
    issues = validate_research_judgment_decision(decision)
    if issues:
        raise V3KernelValidationError(issues)
    append_jsonl(object_path(kernel_dir, "decisions"), decision)
    return decision


def create_question(
    kernel_dir: str | Path,
    record: dict[str, Any],
    *,
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_kernel_dirs(kernel_dir)
    question = normalize_question(record)
    existing_questions = {
        item["id"]: item for item in read_jsonl(object_path(kernel_dir, "questions"))
    }
    if question["id"] in existing_questions:
        revisions = read_jsonl(object_path(kernel_dir, "question_revisions"))
        revision = next(
            (
                item
                for item in revisions
                if item.get("id") == existing_questions[question["id"]].get("last_revision_id")
            ),
            {},
        )
        return existing_questions[question["id"]], revision

    revision = normalize_question_revision(
        {
            "id": _stable_id(
                "qrev",
                {
                    "question_id": question["id"],
                    "source_decision_id": question["source_decision_id"],
                    "action": "create",
                },
            ),
            "question_id": question["id"],
            "action": "create",
            "old_status": "",
            "new_status": question["status"],
            "reason": reason,
            "source_decision_id": question["source_decision_id"],
            "provenance": {
                "method": "apply_research_judgment_decision_v1",
            },
        }
    )
    question["last_revision_id"] = revision["id"]
    issues = validate_question(question) + validate_question_revision(revision)
    if issues:
        raise V3KernelValidationError(issues)
    append_jsonl(object_path(kernel_dir, "questions"), question)
    append_jsonl(object_path(kernel_dir, "question_revisions"), revision)
    return question, revision


def create_trajectory(
    kernel_dir: str | Path,
    record: dict[str, Any],
    *,
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_kernel_dirs(kernel_dir)
    trajectory = normalize_trajectory(record)
    existing_trajectories = {
        item["id"]: item for item in read_jsonl(object_path(kernel_dir, "trajectories"))
    }
    if trajectory["id"] in existing_trajectories:
        revisions = read_jsonl(object_path(kernel_dir, "trajectory_revisions"))
        revision = next(
            (
                item
                for item in revisions
                if item.get("id")
                == existing_trajectories[trajectory["id"]].get("last_revision_id")
            ),
            {},
        )
        return existing_trajectories[trajectory["id"]], revision

    revision = normalize_trajectory_revision(
        {
            "id": _stable_id(
                "trev",
                {
                    "trajectory_id": trajectory["id"],
                    "source_decision_id": trajectory["source_decision_id"],
                    "action": "create",
                },
            ),
            "trajectory_id": trajectory["id"],
            "action": "create",
            "old_status": "",
            "new_status": trajectory["status"],
            "reason": reason,
            "source_decision_id": trajectory["source_decision_id"],
            "provenance": {
                "method": "apply_research_judgment_decision_v1",
            },
        }
    )
    trajectory["last_revision_id"] = revision["id"]
    issues = validate_trajectory(trajectory) + validate_trajectory_revision(revision)
    if issues:
        raise V3KernelValidationError(issues)
    append_jsonl(object_path(kernel_dir, "trajectories"), trajectory)
    append_jsonl(object_path(kernel_dir, "trajectory_revisions"), revision)
    return trajectory, revision


def create_action_record(
    kernel_dir: str | Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    ensure_kernel_dirs(kernel_dir)
    action = normalize_action_record(record)
    issues = validate_action_record(action)
    if issues:
        raise V3KernelValidationError(issues)
    existing_actions = {
        item["id"]: item for item in read_jsonl(object_path(kernel_dir, "actions"))
    }
    if action["id"] not in existing_actions:
        append_jsonl(object_path(kernel_dir, "actions"), action)
    return existing_actions.get(action["id"], action)


def create_human_review_request(
    kernel_dir: str | Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    ensure_kernel_dirs(kernel_dir)
    request = normalize_human_review_request(record)
    decision_id = str(request.get("decision_id") or "")
    if decision_id:
        _decision_by_id(kernel_dir, decision_id)
    issues = validate_human_review_request(request)
    if issues:
        raise V3KernelValidationError(issues)
    existing_requests = {
        item["id"]: item
        for item in read_jsonl(object_path(kernel_dir, "human_review_requests"))
    }
    if request["id"] not in existing_requests:
        append_jsonl(object_path(kernel_dir, "human_review_requests"), request)
    return existing_requests.get(request["id"], request)


def _decision_by_id(kernel_dir: str | Path, decision_id: str) -> dict[str, Any]:
    decisions = read_jsonl(object_path(kernel_dir, "decisions"))
    decision = next(
        (
            item
            for item in decisions
            if item.get("decision_id") == decision_id or item.get("id") == decision_id
        ),
        None,
    )
    if not decision:
        raise V3KernelValidationError(
            [V3Issue("$.decision_id", f"unknown decision: {decision_id}")]
        )
    return decision


def _first_candidate_for_layer(
    decision: dict[str, Any],
    layer: str,
) -> dict[str, Any]:
    return next(
        (
            candidate
            for candidate in _as_dict_list(decision.get("candidate_state_changes"))
            if candidate.get("target_layer") == layer
        ),
        {},
    )


def _decision_belief_refs(decision: dict[str, Any]) -> list[str]:
    affected = (
        decision.get("affected_objects")
        if isinstance(decision.get("affected_objects"), dict)
        else {}
    )
    return _dedupe(_as_list(affected.get("belief_refs")))


def _action_type_from_text(action_text: str) -> str:
    lowered = action_text.lower()
    if "human" in lowered or "adjudicat" in lowered:
        return "human_review"
    if "refresh" in lowered or "regenerate" in lowered:
        return "refresh"
    if "verify" in lowered or "validat" in lowered or "check" in lowered:
        return "verify"
    if "synth" in lowered:
        return "synthesize"
    if "experiment" in lowered:
        return "experiment"
    if "ignore" in lowered:
        return "ignore"
    return "read"


def _decision_priority(decision: dict[str, Any], action_type: str) -> str:
    uncertainty = (
        decision.get("uncertainty") if isinstance(decision.get("uncertainty"), dict) else {}
    )
    if _as_bool(decision.get("human_review_required")) and action_type == "human_review":
        return "urgent"
    if uncertainty.get("level") == "high" or action_type == "human_review":
        return "high"
    if uncertainty.get("level") == "low":
        return "low"
    return "medium"


def _short_title(text: str, *, limit: int = 84) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _human_review_required_by_action(decision: dict[str, Any]) -> bool:
    if _as_bool(decision.get("human_review_required")):
        return True
    actions = " ".join(_as_list(decision.get("required_actions"))).lower()
    return any(
        marker in actions
        for marker in (
            "human",
            "read",
            "review",
            "adjudicat",
            "full_text",
            "full text",
        )
    )


def _should_create_human_review_request(decision: dict[str, Any]) -> bool:
    if decision.get("decision") != "request_action_or_read":
        return False
    frame = decision.get("judgment_frame") if isinstance(decision.get("judgment_frame"), dict) else {}
    primary_pressure = str(frame.get("primary_pressure") or "")
    return _human_review_required_by_action(decision) or primary_pressure in {
        "conflict_resolution",
        "source_grounding",
    }


def _human_review_request_type(decision: dict[str, Any]) -> str:
    frame = decision.get("judgment_frame") if isinstance(decision.get("judgment_frame"), dict) else {}
    primary_pressure = str(frame.get("primary_pressure") or "")
    blockers = _as_list(frame.get("blockers")) if isinstance(frame.get("blockers"), list) else []
    if primary_pressure == "conflict_resolution":
        return "adjudicate_conflict"
    if "abstract_level_only" in blockers or "text_only_abstract_grounding" in blockers:
        return "full_text_review"
    if primary_pressure == "source_grounding":
        return "grounding_check"
    relation_signal = (
        frame.get("relation_signal")
        if isinstance(frame.get("relation_signal"), dict)
        else {}
    )
    if relation_signal.get("contested"):
        return "relation_review"
    return "human_read"


def _human_review_target_layer(decision: dict[str, Any]) -> str:
    for candidate in _as_dict_list(decision.get("candidate_state_changes")):
        layer = str(candidate.get("target_layer") or "").strip()
        if layer in ALLOWED_RESEARCH_DECISION_LAYER and layer != "action":
            return layer
    affected = decision.get("affected_objects") if isinstance(decision.get("affected_objects"), dict) else {}
    if _as_list(affected.get("belief_refs")):
        return "belief"
    return "attention"


def _default_human_response_options(request_type: str) -> list[dict[str, str]]:
    options = [
        ("support", "Supports", "The paper provides evidence for the stated belief."),
        ("challenge", "Challenges", "The paper weakens or argues against the stated belief."),
        ("neutral", "Neutral / Off-topic", "The paper does not bear on this belief."),
        ("underdetermined", "Underdetermined", "The paper is relevant but insufficient to move the belief."),
        ("contest", "Truly contested", "The paper creates a real conflict requiring adjudication."),
    ]
    if request_type in {"full_text_review", "grounding_check"}:
        options.append(
            (
                "abstract_insufficient",
                "Abstract insufficient",
                "The visible text is not enough; full text or figures are required.",
            )
        )
    options.extend(
        [
            ("not_my_expertise", "Not my expertise", "The reviewer should not be used as a gold label."),
            ("paper_ambiguous", "Paper ambiguous", "The source itself is unclear or two-edged."),
            ("belief_too_broad", "Belief too broad", "The belief statement needs narrowing before judgment."),
        ]
    )
    return [
        {"value": value, "label": label, "meaning": meaning}
        for value, label, meaning in options
    ]


def _human_review_question(decision: dict[str, Any], request_type: str) -> str:
    affected = decision.get("affected_objects") if isinstance(decision.get("affected_objects"), dict) else {}
    belief_ref = ", ".join(_as_list(affected.get("belief_refs"))) or "the stated belief"
    if request_type == "adjudicate_conflict":
        return (
            f"Read the source against {belief_ref}: does it support, challenge, "
            "remain neutral to, or leave this belief underdetermined?"
        )
    if request_type == "full_text_review":
        return (
            f"After checking the best available text for {belief_ref}, is there "
            "enough grounded evidence to move kernel state, or should it remain pending?"
        )
    if request_type == "grounding_check":
        return (
            f"Is the evidence for {belief_ref} grounded enough for kernel use, "
            "or is the paper only a read/action item?"
        )
    return (
        f"Review this source against {belief_ref} and record the evidence basis, "
        "caveat and recommended state movement."
    )


def _human_review_request_from_decision(
    decision: dict[str, Any],
    *,
    linked_action_ids: list[str],
) -> dict[str, Any]:
    request_type = _human_review_request_type(decision)
    target_layer = _human_review_target_layer(decision)
    affected = decision.get("affected_objects") if isinstance(decision.get("affected_objects"), dict) else {}
    belief_refs = _as_list(affected.get("belief_refs"))
    source_refs = _as_list(affected.get("source_refs"))
    frame = decision.get("judgment_frame") if isinstance(decision.get("judgment_frame"), dict) else {}
    question = _human_review_question(decision, request_type)
    priority = _decision_priority(decision, "human_review")
    return normalize_human_review_request(
        {
            "id": _stable_id(
                "hrr",
                {
                    "decision_id": decision.get("decision_id") or decision.get("id"),
                    "request_type": request_type,
                    "target_state_layer": target_layer,
                },
            ),
            "decision_id": str(decision.get("decision_id") or decision.get("id") or ""),
            "assessment_id": str(decision.get("assessment_id") or ""),
            "briefing_id": str(decision.get("briefing_id") or ""),
            "source_ref": str(decision.get("source_ref") or (source_refs[0] if source_refs else "")),
            "belief_ref": belief_refs[0] if belief_refs else "",
            "request_type": request_type,
            "target_state_layer": target_layer,
            "question": question,
            "status": "open",
            "priority": priority,
            "linked_action_ids": linked_action_ids,
            "allowed_responses": _default_human_response_options(request_type),
            "reviewer_payload": {
                "source_ref": str(decision.get("source_ref") or (source_refs[0] if source_refs else "")),
                "belief_ref": belief_refs[0] if belief_refs else "",
                "target_state_layer": target_layer,
                "question": question,
                "instructions": (
                    "Judge the source against the belief. Give an evidence basis "
                    "and caveat; do not infer the kernel prediction."
                ),
            },
            "kernel_context": {
                "decision": decision.get("decision"),
                "decision_id": decision.get("decision_id") or decision.get("id"),
                "primary_pressure": frame.get("primary_pressure", "unknown"),
                "blockers": _as_list(frame.get("blockers")) if isinstance(frame.get("blockers"), list) else [],
                "uncertainty": decision.get("uncertainty") if isinstance(decision.get("uncertainty"), dict) else {},
                "required_actions": _as_list(decision.get("required_actions")),
                "candidate_state_changes": _as_dict_list(decision.get("candidate_state_changes")),
            },
            "anti_anchoring": {
                "kernel_prediction_withheld_from_reviewer": True,
                "model_votes_withheld_from_reviewer": True,
                "hidden_from_reviewer": [
                    "secretary_relation_signal",
                    "model_votes",
                    "kernel_decision",
                    "judgment_frame.warrants",
                    "kernel_context",
                ],
                "reviewer_payload_is_blind": True,
            },
            "provenance": {
                "method": "human_review_request_v1_from_research_judgment_decision",
                "decision_method": (
                    decision.get("provenance", {}).get("method")
                    if isinstance(decision.get("provenance"), dict)
                    else ""
                ),
                "note": (
                    "This request prepares human input without exposing kernel "
                    "predictions or model votes to the reviewer payload."
                ),
            },
        }
    )


def apply_research_judgment_decision(
    kernel_dir: str | Path,
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Apply a recorded ResearchJudgmentDecision to downstream state objects.

    This API intentionally does not apply BeliefRevision records. Belief
    projection is still controlled by revise_belief/create_belief so that
    direction-heavy judgment remains explicit and separately auditable.
    """
    ensure_kernel_dirs(kernel_dir)
    decision_id = str(decision.get("decision_id") or decision.get("id") or "")
    recorded_decision = _decision_by_id(kernel_dir, decision_id)
    issues = validate_research_judgment_decision(recorded_decision)
    if issues:
        raise V3KernelValidationError(issues)

    applied: dict[str, list[str]] = {
        "actions": [],
        "human_review_requests": [],
        "questions": [],
        "question_revisions": [],
        "trajectories": [],
        "trajectory_revisions": [],
    }
    skipped: list[dict[str, str]] = []
    affected = (
        recorded_decision.get("affected_objects")
        if isinstance(recorded_decision.get("affected_objects"), dict)
        else {}
    )
    decision_type = recorded_decision.get("decision")
    source_ref = str(recorded_decision.get("source_ref") or "")
    linked_assessment_id = str(recorded_decision.get("assessment_id") or "")
    linked_briefing_id = str(recorded_decision.get("briefing_id") or "")

    if decision_type == "request_action_or_read":
        actions = _as_list(recorded_decision.get("required_actions"))
        action_candidate = _first_candidate_for_layer(recorded_decision, "action")
        if not actions and action_candidate:
            actions = [str(action_candidate.get("description") or "read_source")]
        for action_text in actions:
            action_type = _action_type_from_text(action_text)
            action = create_action_record(
                kernel_dir,
                {
                    "id": _stable_id(
                        "act",
                        {
                            "decision_id": decision_id,
                            "action": action_text,
                        },
                    ),
                    "decision_id": decision_id,
                    "action_type": action_type,
                    "description": action_text,
                    "status": "open",
                    "priority": _decision_priority(recorded_decision, action_type),
                    "source_ref": source_ref,
                    "linked_assessment_id": linked_assessment_id,
                    "linked_briefing_id": linked_briefing_id,
                    "provenance": {
                        "method": "apply_research_judgment_decision_v1",
                        "decision_type": decision_type,
                    },
                },
            )
            applied["actions"].append(action["id"])
        if _should_create_human_review_request(recorded_decision):
            request = create_human_review_request(
                kernel_dir,
                _human_review_request_from_decision(
                    recorded_decision,
                    linked_action_ids=applied["actions"],
                ),
            )
            applied["human_review_requests"].append(request["id"])

    elif decision_type == "open_question":
        question_candidate = _first_candidate_for_layer(recorded_decision, "question")
        question_text = str(question_candidate.get("description") or "").strip()
        if not question_text:
            skipped.append(
                {
                    "target": "question",
                    "reason": "no_question_candidate_supplied",
                }
            )
        else:
            question, revision = create_question(
                kernel_dir,
                {
                    "id": _stable_id(
                        "q",
                        {
                            "decision_id": decision_id,
                            "question": question_text,
                        },
                    ),
                    "question": question_text,
                    "status": "proposed",
                    "source_decision_id": decision_id,
                    "source_ref": source_ref,
                    "linked_beliefs": _decision_belief_refs(recorded_decision),
                    "linked_evidence": _as_list(affected.get("evidence_ids")),
                    "rationale": _as_list(recorded_decision.get("rationale")),
                    "uncertainty_level": recorded_decision.get("uncertainty", {}).get(
                        "level", "unknown"
                    )
                    if isinstance(recorded_decision.get("uncertainty"), dict)
                    else "unknown",
                    "provenance": {
                        "method": "apply_research_judgment_decision_v1",
                        "source_candidate_ids": _as_list(
                            question_candidate.get("source_candidate_ids")
                        ),
                    },
                },
                reason="Opened from ResearchJudgmentDecision.",
            )
            applied["questions"].append(question["id"])
            if revision:
                applied["question_revisions"].append(revision["id"])

    elif decision_type == "propose_trajectory_update":
        trajectory_candidate = _first_candidate_for_layer(recorded_decision, "trajectory")
        statement = str(trajectory_candidate.get("description") or "").strip()
        if not statement:
            skipped.append(
                {
                    "target": "trajectory",
                    "reason": "no_trajectory_candidate_supplied",
                }
            )
        else:
            trajectory, revision = create_trajectory(
                kernel_dir,
                {
                    "id": _stable_id(
                        "traj",
                        {
                            "decision_id": decision_id,
                            "statement": statement,
                        },
                    ),
                    "title": _short_title(statement),
                    "statement": statement,
                    "status": "proposed",
                    "source_decision_id": decision_id,
                    "source_ref": source_ref,
                    "linked_beliefs": _decision_belief_refs(recorded_decision),
                    "rationale": _as_list(recorded_decision.get("rationale")),
                    "provenance": {
                        "method": "apply_research_judgment_decision_v1",
                        "source_candidate_ids": _as_list(
                            trajectory_candidate.get("source_candidate_ids")
                        ),
                    },
                },
                reason="Opened from ResearchJudgmentDecision.",
            )
            applied["trajectories"].append(trajectory["id"])
            if revision:
                applied["trajectory_revisions"].append(revision["id"])

    elif decision_type == "propose_belief_revision":
        skipped.append(
            {
                "target": "belief",
                "reason": "explicit_belief_revision_api_required",
            }
        )
    else:
        skipped.append(
            {
                "target": "decision",
                "reason": f"no_apply_step_for_{decision_type}",
            }
        )

    return {
        "schema_id": "research_judgment_application_v1",
        "created_at": now_iso(),
        "decision_id": decision_id,
        "decision": decision_type,
        "applied": applied,
        "skipped": skipped,
    }


def apply_override_learning(kernel_dir: str | Path, override_id: str) -> dict[str, Any]:
    overrides = read_jsonl(object_path(kernel_dir, "overrides"))
    override = next((item for item in overrides if item["id"] == override_id), None)
    if not override:
        raise V3KernelValidationError([V3Issue("$.override_id", f"unknown override: {override_id}")])
    rule_path = Path(kernel_dir) / "overrides" / "rule_candidates.json"
    rules = json.loads(rule_path.read_text(encoding="utf-8")) if rule_path.exists() else []
    rule = {
        "id": _stable_id("rule", override),
        "source_override_id": override_id,
        "status": "candidate",
        "rule": override.get("future_rule_candidate", ""),
        "extracted_learning": override.get("extracted_learning", ""),
        "created_at": now_iso(),
    }
    if not any(item["id"] == rule["id"] for item in rules):
        rules.append(rule)
        rule_path.parent.mkdir(parents=True, exist_ok=True)
        rule_path.write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding="utf-8")
    return rule


def validate_v3_kernel(kernel_dir: str | Path) -> tuple[dict[str, Any], list[V3Issue]]:
    ensure_kernel_dirs(kernel_dir)
    kernel_dir = Path(kernel_dir)
    issues: list[V3Issue] = []
    beliefs = read_jsonl(object_path(kernel_dir, "beliefs"))
    evidence = read_jsonl(object_path(kernel_dir, "evidence"))
    revisions = read_jsonl(object_path(kernel_dir, "revisions"))
    judgments = read_jsonl(object_path(kernel_dir, "judgments"))
    overrides = read_jsonl(object_path(kernel_dir, "overrides"))
    constraints = read_jsonl(object_path(kernel_dir, "constraints"))
    assessments = read_jsonl(object_path(kernel_dir, "assessments"))
    decisions = read_jsonl(object_path(kernel_dir, "decisions"))
    questions = read_jsonl(object_path(kernel_dir, "questions"))
    question_revisions = read_jsonl(object_path(kernel_dir, "question_revisions"))
    trajectories = read_jsonl(object_path(kernel_dir, "trajectories"))
    trajectory_revisions = read_jsonl(object_path(kernel_dir, "trajectory_revisions"))
    actions = read_jsonl(object_path(kernel_dir, "actions"))
    human_review_requests = read_jsonl(object_path(kernel_dir, "human_review_requests"))

    for idx, record in enumerate(beliefs, 1):
        issues.extend(validate_belief(record, path=f"beliefs:{idx}"))
    for idx, record in enumerate(evidence, 1):
        issues.extend(validate_evidence(record, path=f"evidence:{idx}"))
    for idx, record in enumerate(revisions, 1):
        issues.extend(validate_revision(record, path=f"revisions:{idx}"))
    for idx, record in enumerate(judgments, 1):
        issues.extend(validate_judgment(record, path=f"judgments:{idx}"))
    for idx, record in enumerate(overrides, 1):
        issues.extend(validate_override(record, path=f"overrides:{idx}"))
    for idx, record in enumerate(constraints, 1):
        issues.extend(validate_constraint(record, path=f"constraints:{idx}"))
    for idx, record in enumerate(assessments, 1):
        issues.extend(validate_kernel_intake_assessment(record, path=f"assessments:{idx}"))
    for idx, record in enumerate(decisions, 1):
        issues.extend(validate_research_judgment_decision(record, path=f"decisions:{idx}"))
    for idx, record in enumerate(questions, 1):
        issues.extend(validate_question(record, path=f"questions:{idx}"))
    for idx, record in enumerate(question_revisions, 1):
        issues.extend(
            validate_question_revision(record, path=f"question_revisions:{idx}")
        )
    for idx, record in enumerate(trajectories, 1):
        issues.extend(validate_trajectory(record, path=f"trajectories:{idx}"))
    for idx, record in enumerate(trajectory_revisions, 1):
        issues.extend(
            validate_trajectory_revision(record, path=f"trajectory_revisions:{idx}")
        )
    for idx, record in enumerate(actions, 1):
        issues.extend(validate_action_record(record, path=f"actions:{idx}"))
    for idx, record in enumerate(human_review_requests, 1):
        issues.extend(
            validate_human_review_request(record, path=f"human_review_requests:{idx}")
        )

    belief_ids = {item["id"] for item in beliefs}
    evidence_ids = {item["id"] for item in evidence}
    constraint_ids = {item.get("id") for item in constraints}
    judgment_ids = {item["id"] for item in judgments}
    assessment_ids = {
        item.get("assessment_id") or item.get("id")
        for item in assessments
    }
    decision_ids = {item.get("decision_id") or item.get("id") for item in decisions}
    question_ids = {item.get("question_id") or item.get("id") for item in questions}
    question_revision_ids = {
        item.get("revision_id") or item.get("id") for item in question_revisions
    }
    trajectory_ids = {
        item.get("trajectory_id") or item.get("id") for item in trajectories
    }
    trajectory_revision_ids = {
        item.get("revision_id") or item.get("id") for item in trajectory_revisions
    }
    action_ids = {item.get("action_id") or item.get("id") for item in actions}
    revision_ids = {item["id"] for item in revisions}
    override_target_ids = {
        "belief": belief_ids,
        "evidence": evidence_ids,
        "constraint": constraint_ids,
        "judgment": judgment_ids,
    }
    evidence_referenced_by_revision = set()

    for belief in beliefs:
        for evidence_id in (
            belief.get("evidence_ids", [])
            + belief.get("contra_evidence_ids", [])
            + belief.get("pending_evidence_ids", [])
            + belief.get("neutral_evidence_ids", [])
            + belief.get("contested_evidence_ids", [])
        ):
            if evidence_id not in evidence_ids:
                issues.append(V3Issue(f"beliefs.{belief['id']}.evidence_ids", f"unknown evidence: {evidence_id}"))
        for constraint_id in belief.get("linked_constraints", []):
            if constraint_id not in constraint_ids:
                issues.append(V3Issue(f"beliefs.{belief['id']}.linked_constraints", f"unknown constraint: {constraint_id}"))
        if (
            not belief.get("evidence_ids")
            and not belief.get("contra_evidence_ids")
            and not belief.get("pending_evidence_ids")
            and not belief.get("neutral_evidence_ids")
            and not belief.get("contested_evidence_ids")
        ):
            issues.append(V3Issue(f"beliefs.{belief['id']}", "belief has no evidence"))

    for revision in revisions:
        if revision.get("belief_id") not in belief_ids:
            issues.append(V3Issue(f"revisions.{revision.get('id')}.belief_id", "unknown belief"))
        for evidence_id in revision.get("triggering_evidence_ids", []):
            evidence_referenced_by_revision.add(evidence_id)
            if evidence_id not in evidence_ids:
                issues.append(V3Issue(f"revisions.{revision.get('id')}.triggering_evidence_ids", f"unknown evidence: {evidence_id}"))

    for item in evidence:
        for belief_id in (
            item.get("supports_beliefs", [])
            + item.get("challenges_beliefs", [])
            + item.get("pending_beliefs", [])
            + item.get("neutral_beliefs", [])
            + item.get("contests_beliefs", [])
        ):
            if belief_id not in belief_ids:
                issues.append(V3Issue(f"evidence.{item['id']}", f"unknown linked belief: {belief_id}"))

    for judgment in judgments:
        for belief_id in judgment.get("linked_beliefs", []):
            if belief_id not in belief_ids:
                issues.append(V3Issue(f"judgments.{judgment['id']}.linked_beliefs", f"unknown belief: {belief_id}"))
        for constraint_id in judgment.get("linked_constraints", []):
            if constraint_id not in constraint_ids:
                issues.append(V3Issue(f"judgments.{judgment['id']}.linked_constraints", f"unknown constraint: {constraint_id}"))
        for evidence_id in judgment.get("linked_evidence", []):
            if evidence_id not in evidence_ids:
                issues.append(V3Issue(f"judgments.{judgment['id']}.linked_evidence", f"unknown evidence: {evidence_id}"))
        if not judgment.get("linked_beliefs"):
            issues.append(V3Issue(f"judgments.{judgment['id']}", "judgment has no linked beliefs"))

    for constraint in constraints:
        for belief_id in constraint.get("linked_beliefs", []):
            if belief_id not in belief_ids:
                issues.append(V3Issue(f"constraints.{constraint['id']}.linked_beliefs", f"unknown belief: {belief_id}"))
        for evidence_id in constraint.get("linked_evidence", []):
            if evidence_id not in evidence_ids:
                issues.append(V3Issue(f"constraints.{constraint['id']}.linked_evidence", f"unknown evidence: {evidence_id}"))

    for override in overrides:
        target_type = override.get("target_type")
        target_id = override.get("target_id")
        known_ids = override_target_ids.get(target_type)
        if known_ids is not None and target_id not in known_ids:
            issues.append(V3Issue(f"overrides.{override['id']}.target_id", "unknown target"))

    for decision in decisions:
        assessment_id = decision.get("assessment_id")
        if assessment_id not in assessment_ids:
            issues.append(
                V3Issue(
                    f"decisions.{decision.get('id')}.assessment_id",
                    f"unknown assessment: {assessment_id}",
                )
            )
        for revision_id in decision.get("applied_revision_ids", []):
            if revision_id not in revision_ids:
                issues.append(
                    V3Issue(
                        f"decisions.{decision.get('id')}.applied_revision_ids",
                        f"unknown revision: {revision_id}",
                )
            )

    for question in questions:
        decision_id = question.get("source_decision_id")
        if decision_id not in decision_ids:
            issues.append(
                V3Issue(
                    f"questions.{question.get('id')}.source_decision_id",
                    f"unknown decision: {decision_id}",
                )
            )
        last_revision_id = question.get("last_revision_id")
        if last_revision_id and last_revision_id not in question_revision_ids:
            issues.append(
                V3Issue(
                    f"questions.{question.get('id')}.last_revision_id",
                    f"unknown question revision: {last_revision_id}",
                )
            )
        for evidence_id in question.get("linked_evidence", []):
            if evidence_id not in evidence_ids:
                issues.append(
                    V3Issue(
                        f"questions.{question.get('id')}.linked_evidence",
                        f"unknown evidence: {evidence_id}",
                    )
                )

    for question_revision in question_revisions:
        question_id = question_revision.get("question_id")
        if question_id not in question_ids:
            issues.append(
                V3Issue(
                    f"question_revisions.{question_revision.get('id')}.question_id",
                    f"unknown question: {question_id}",
                )
            )
        decision_id = question_revision.get("source_decision_id")
        if decision_id not in decision_ids:
            issues.append(
                V3Issue(
                    f"question_revisions.{question_revision.get('id')}.source_decision_id",
                    f"unknown decision: {decision_id}",
                )
            )

    for trajectory in trajectories:
        decision_id = trajectory.get("source_decision_id")
        if decision_id not in decision_ids:
            issues.append(
                V3Issue(
                    f"trajectories.{trajectory.get('id')}.source_decision_id",
                    f"unknown decision: {decision_id}",
                )
            )
        last_revision_id = trajectory.get("last_revision_id")
        if last_revision_id and last_revision_id not in trajectory_revision_ids:
            issues.append(
                V3Issue(
                    f"trajectories.{trajectory.get('id')}.last_revision_id",
                    f"unknown trajectory revision: {last_revision_id}",
                )
            )
        for question_id in trajectory.get("linked_questions", []):
            if question_id not in question_ids:
                issues.append(
                    V3Issue(
                        f"trajectories.{trajectory.get('id')}.linked_questions",
                        f"unknown question: {question_id}",
                    )
                )

    for trajectory_revision in trajectory_revisions:
        trajectory_id = trajectory_revision.get("trajectory_id")
        if trajectory_id not in trajectory_ids:
            issues.append(
                V3Issue(
                    f"trajectory_revisions.{trajectory_revision.get('id')}.trajectory_id",
                    f"unknown trajectory: {trajectory_id}",
                )
            )
        decision_id = trajectory_revision.get("source_decision_id")
        if decision_id not in decision_ids:
            issues.append(
                V3Issue(
                    f"trajectory_revisions.{trajectory_revision.get('id')}.source_decision_id",
                    f"unknown decision: {decision_id}",
                )
            )

    for action in actions:
        decision_id = action.get("decision_id")
        if decision_id not in decision_ids:
            issues.append(
                V3Issue(
                    f"actions.{action.get('id')}.decision_id",
                    f"unknown decision: {decision_id}",
                )
            )

    for request in human_review_requests:
        decision_id = request.get("decision_id")
        if decision_id not in decision_ids:
            issues.append(
                V3Issue(
                    f"human_review_requests.{request.get('id')}.decision_id",
                    f"unknown decision: {decision_id}",
                )
            )
        for action_id in request.get("linked_action_ids", []):
            if action_id not in action_ids:
                issues.append(
                    V3Issue(
                        f"human_review_requests.{request.get('id')}.linked_action_ids",
                        f"unknown action: {action_id}",
                    )
                )

    orphan_evidence = [
        item["id"]
        for item in evidence
        if not item.get("supports_beliefs")
        and not item.get("challenges_beliefs")
        and not item.get("pending_beliefs")
        and not item.get("neutral_beliefs")
        and not item.get("contests_beliefs")
        and item["id"] not in evidence_referenced_by_revision
    ]
    status_counts: dict[str, int] = {}
    for belief in beliefs:
        status_counts[belief["status"]] = status_counts.get(belief["status"], 0) + 1
    health = {
        "kernel_dir": str(kernel_dir),
        "belief_count": len(beliefs),
        "belief_status_counts": status_counts,
        "evidence_count": len(evidence),
        "revision_count": len(revisions),
        "judgment_count": len(judgments),
        "override_count": len(overrides),
        "constraint_count": len(constraints),
        "assessment_count": len(assessments),
        "decision_count": len(decisions),
        "question_count": len(questions),
        "question_revision_count": len(question_revisions),
        "trajectory_count": len(trajectories),
        "trajectory_revision_count": len(trajectory_revisions),
        "action_count": len(actions),
        "open_action_count": sum(1 for action in actions if action.get("status") == "open"),
        "human_review_request_count": len(human_review_requests),
        "open_human_review_request_count": sum(
            1
            for request in human_review_requests
            if request.get("status") == "open"
        ),
        "orphan_evidence_count": len(orphan_evidence),
        "pending_evidence_count": sum(
            1
            for belief in beliefs
            for _ in belief.get("pending_evidence_ids", [])
        ),
        "neutral_evidence_count": sum(
            1
            for belief in beliefs
            for _ in belief.get("neutral_evidence_ids", [])
        ),
        "contested_evidence_count": sum(
            1
            for belief in beliefs
            for _ in belief.get("contested_evidence_ids", [])
        ),
        "beliefs_without_evidence": sum(
            1
            for belief in beliefs
            if not belief.get("evidence_ids")
            and not belief.get("contra_evidence_ids")
            and not belief.get("pending_evidence_ids")
            and not belief.get("neutral_evidence_ids")
            and not belief.get("contested_evidence_ids")
        ),
        "judgments_without_beliefs": sum(1 for judgment in judgments if not judgment.get("linked_beliefs")),
        "assessments_requiring_human": sum(
            1 for assessment in assessments if assessment.get("decision") == "escalate"
        ),
        "assessments_deferred": sum(
            1
            for assessment in assessments
            if assessment.get("decision") in {"defer", "request_refresh"}
        ),
        "decisions_requiring_human": sum(
            1 for decision in decisions if decision.get("human_review_required")
        ),
        "decisions_requesting_action": sum(
            1
            for decision in decisions
            if decision.get("decision") == "request_action_or_read"
        ),
        "validation_issue_count": len(issues),
        "validation_status": "ok" if not issues else "failed",
    }
    return health, issues


def export_kernel_state(kernel_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    health, issues = validate_v3_kernel(kernel_dir)
    state = {
        "generated_at": now_iso(),
        "health": health,
        "issues": [issue.format() for issue in issues],
        "beliefs": read_jsonl(object_path(kernel_dir, "beliefs")),
        "evidence": read_jsonl(object_path(kernel_dir, "evidence")),
        "revisions": read_jsonl(object_path(kernel_dir, "revisions")),
        "judgments": read_jsonl(object_path(kernel_dir, "judgments")),
        "overrides": read_jsonl(object_path(kernel_dir, "overrides")),
        "constraints": read_jsonl(object_path(kernel_dir, "constraints")),
        "assessments": read_jsonl(object_path(kernel_dir, "assessments")),
        "decisions": read_jsonl(object_path(kernel_dir, "decisions")),
        "questions": read_jsonl(object_path(kernel_dir, "questions")),
        "question_revisions": read_jsonl(object_path(kernel_dir, "question_revisions")),
        "trajectories": read_jsonl(object_path(kernel_dir, "trajectories")),
        "trajectory_revisions": read_jsonl(object_path(kernel_dir, "trajectory_revisions")),
        "actions": read_jsonl(object_path(kernel_dir, "actions")),
        "human_review_requests": read_jsonl(object_path(kernel_dir, "human_review_requests")),
    }
    if output_path is None:
        output_path = Path(kernel_dir) / "exports" / "kernel_state.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return state


def seed_minimal_v3_kernel(kernel_dir: str | Path) -> dict[str, Any]:
    """Seed the smallest useful V3 loop from the novelty stress-test decision."""
    ensure_kernel_dirs(kernel_dir)
    evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "report",
            "title": "ScholarHound Hostile Novelty Stress Test",
            "source_ref": "ScholarHound_novelty_stress_test.html",
            "summary": (
                "Hostile review argues ScholarHound should stop claiming generic "
                "persistent-state novelty and center the architecture on belief as "
                "a first-class scientific kernel object."
            ),
            "evidence_strength": "strong",
            "reliability": "local report; needs external prior-art verification",
        },
    )
    belief, revision = create_belief(
        kernel_dir,
        {
            "title": "ScholarHound V3 should be belief-centered",
            "claim": (
                "ScholarHound's V3 kernel should treat belief, not paper or digest, "
                "as the first-class persistent state object."
            ),
            "domain": "research-os-architecture",
            "status": "active",
            "confidence": 0.82,
            "entrenchment": 0.35,
            "linked_concepts": ["belief-kernel", "scientific-judgment", "human-override-learning"],
            "provenance": {
                "source": "user-approved V3 minimal upgrade",
                "report": "ScholarHound_novelty_stress_test.html",
            },
        },
        reason=(
            "Novelty stress test and user confirmation both support reframing V3 "
            "around belief revision rather than paper management or UI polish."
        ),
        evidence_ids=[evidence["id"]],
    )
    evidence["supports_beliefs"] = [belief["id"]]
    evidence_records = read_jsonl(object_path(kernel_dir, "evidence"))
    evidence_records = [
        evidence if item["id"] == evidence["id"] else item for item in evidence_records
    ]
    write_jsonl(object_path(kernel_dir, "evidence"), evidence_records)
    constraint = create_constraint(
        kernel_dir,
        {
            "statement": (
                "A V3 architecture change is not valid unless it can show an "
                "Evidence -> Belief -> BeliefRevision path."
            ),
            "type": "methodological",
            "status": "testable",
            "falsifiability": (
                "Fails if a kernel update can change a belief projection without "
                "creating a BeliefRevision."
            ),
            "predicted_observations": [
                "Every belief create/update action has a linked revision row",
                "Kernel validation reports orphan or broken belief/evidence links",
            ],
            "failure_conditions": [
                "Belief confidence changes without a revision",
                "Judgment exists without linked beliefs",
            ],
            "linked_beliefs": [belief["id"]],
            "linked_evidence": [evidence["id"]],
            "confidence": 0.88,
        },
    )
    judgment = create_judgment(
        kernel_dir,
        {
            "question": "What is the next safe V3 implementation step?",
            "judgment": "Implement the smallest belief revision loop before UI expansion.",
            "recommendation": (
                "Keep V3 focused on Evidence -> Belief -> BeliefRevision -> validation; "
                "defer route-heavy UI work until the kernel is stable."
            ),
            "confidence": 0.84,
            "risk_level": "medium",
            "linked_beliefs": [belief["id"]],
            "linked_constraints": [constraint["id"]],
            "linked_evidence": [evidence["id"]],
            "next_actions": [
                "Add deterministic belief revision API",
                "Validate kernel links and orphan objects",
                "Only then expose belief views in UI",
            ],
        },
    )
    override = record_human_override(
        kernel_dir,
        {
            "target_type": "belief",
            "target_id": belief["id"],
            "override_type": "approve",
            "user_reason": "User agreed to begin with the minimal V3 kernel upgrade.",
            "extracted_learning": (
                "When architecture scope is large, prioritize the kernel object model "
                "and revision mechanics before UI or broad migration."
            ),
            "future_rule_candidate": (
                "For V3 tasks, require a belief revision path before accepting UI-only work."
            ),
        },
    )
    rule = apply_override_learning(kernel_dir, override["id"])
    health, issues = validate_v3_kernel(kernel_dir)
    return {
        "evidence": evidence,
        "belief": belief,
        "revision": revision,
        "constraint": constraint,
        "judgment": judgment,
        "override": override,
        "rule_candidate": rule,
        "health": health,
        "issues": [issue.format() for issue in issues],
    }
