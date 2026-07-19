"""Structured State Change Log for ScholarHound.

LLMs may propose these events, but this module is the non-LLM validation,
append-only storage, and Markdown rendering layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


STATE_FIELDS = (
    "evidence",
    "concept",
    "trajectory",
    "constraint",
    "uncertainty",
    "action",
)

ALLOWED_STATE_VALUES = {
    "evidence": {"strengthened", "weakened", "unchanged"},
    "concept": {"new", "refinement", "contradiction", "unchanged"},
    "trajectory": {"reinforce", "branch", "terminate", "unchanged"},
    "constraint": {
        "reproducibility",
        "privacy",
        "translation",
        "interpretation",
        "unchanged",
    },
    "uncertainty": {"reduced", "increased", "unchanged"},
    "action": {"read", "verify", "synthesize", "ignore", "experiment", "none"},
}

STATE_DETAIL_FIELDS = {
    "evidence": "claim",
    "concept": "concept",
    "trajectory": "trajectory",
    "constraint": "constraint",
    "uncertainty": "uncertainty",
    "action": "next_action",
}

ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_PRIVACY = {"none", "local-only", "sensitive"}


@dataclass
class ValidationIssue:
    path: str
    message: str

    def format(self) -> str:
        return f"{self.path}: {self.message}"


class StateChangeValidationError(ValueError):
    """Raised when a state change event fails schema validation."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__("; ".join(issue.format() for issue in issues))


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_event_log_path(base_dir: str | Path = ".") -> Path:
    return Path(base_dir) / "kernel" / "state_changes.jsonl"


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _stable_event_id(event: dict[str, Any]) -> str:
    payload = {
        "source": event.get("source", {}),
        "evidence_item": event.get("evidence_item", ""),
        "changed_states": event.get("changed_states", {}),
        "reasoning": event.get("reasoning", ""),
        "created_at": event.get("created_at", ""),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"scr_{digest[:16]}"


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized copy, filling non-semantic defaults."""
    normalized = dict(event)
    normalized.setdefault("created_at", now_iso())
    normalized.setdefault("links", [])

    source = normalized.get("source")
    if isinstance(source, str):
        normalized["source"] = {"kind": "note", "ref": source}

    changed = dict(normalized.get("changed_states") or {})
    for state in STATE_FIELDS:
        value = changed.get(state)
        if isinstance(value, str):
            value = {"status": value}
        if not isinstance(value, dict):
            value = {}
        value.setdefault("status", "unchanged")
        value.setdefault(STATE_DETAIL_FIELDS[state], "")
        changed[state] = value
    normalized["changed_states"] = changed

    normalized.setdefault("event_id", _stable_event_id(normalized))
    return normalized


def validate_event(event: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if not isinstance(event, dict):
        return [ValidationIssue("$", "event must be a JSON object")]

    source = event.get("source")
    if not isinstance(source, dict):
        issues.append(ValidationIssue("$.source", "must be an object"))
    else:
        if not _as_str(source.get("kind")).strip():
            issues.append(ValidationIssue("$.source.kind", "is required"))
        if not _as_str(source.get("ref")).strip():
            issues.append(ValidationIssue("$.source.ref", "is required"))

    for field in ("event_id", "created_at", "evidence_item", "reasoning"):
        if not _as_str(event.get(field)).strip():
            issues.append(ValidationIssue(f"$.{field}", "is required"))

    confidence = _as_str(event.get("confidence")).lower()
    if confidence not in ALLOWED_CONFIDENCE:
        issues.append(
            ValidationIssue(
                "$.confidence",
                f"must be one of {sorted(ALLOWED_CONFIDENCE)}",
            )
        )

    privacy = _as_str(event.get("privacy_sensitivity")).lower()
    if privacy not in ALLOWED_PRIVACY:
        issues.append(
            ValidationIssue(
                "$.privacy_sensitivity",
                f"must be one of {sorted(ALLOWED_PRIVACY)}",
            )
        )

    changed = event.get("changed_states")
    if not isinstance(changed, dict):
        issues.append(ValidationIssue("$.changed_states", "must be an object"))
        return issues

    for state in STATE_FIELDS:
        state_path = f"$.changed_states.{state}"
        payload = changed.get(state)
        if not isinstance(payload, dict):
            issues.append(ValidationIssue(state_path, "is required and must be an object"))
            continue
        status = _as_str(payload.get("status")).lower()
        allowed = ALLOWED_STATE_VALUES[state]
        if status not in allowed:
            issues.append(
                ValidationIssue(
                    f"{state_path}.status",
                    f"must be one of {sorted(allowed)}",
                )
            )
        detail_field = STATE_DETAIL_FIELDS[state]
        if detail_field not in payload:
            issues.append(ValidationIssue(f"{state_path}.{detail_field}", "is required"))

    links = event.get("links", [])
    if not isinstance(links, list):
        issues.append(ValidationIssue("$.links", "must be a list when provided"))

    return issues


def ensure_valid_event(event: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_event(event)
    issues = validate_event(normalized)
    if issues:
        raise StateChangeValidationError(issues)
    return normalized


def read_events(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []

    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StateChangeValidationError(
                    [ValidationIssue(f"{path}:{line_number}", f"invalid JSON: {exc}")]
                ) from exc
            try:
                events.append(ensure_valid_event(event))
            except StateChangeValidationError as exc:
                issues = [
                    ValidationIssue(f"{path}:{line_number}{issue.path}", issue.message)
                    for issue in exc.issues
                ]
                raise StateChangeValidationError(issues) from exc
    return events


def append_event(path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    normalized = ensure_valid_event(event)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalized, sort_keys=True, ensure_ascii=False) + "\n")
    return normalized


def validate_event_log(path: str | Path) -> tuple[list[dict[str, Any]], list[ValidationIssue]]:
    try:
        events = read_events(path)
    except StateChangeValidationError as exc:
        return [], exc.issues
    return events, []


def render_event_markdown(event: dict[str, Any]) -> str:
    event = ensure_valid_event(event)
    source = event["source"]
    lines = [
        f"## {event['event_id']}",
        "",
        f"- Created: `{event['created_at']}`",
        f"- Source: `{source.get('kind')}` / `{source.get('ref')}`",
        f"- Evidence item: {event['evidence_item']}",
        f"- Confidence: `{event['confidence']}`",
        f"- Privacy sensitivity: `{event['privacy_sensitivity']}`",
        "",
        "### Changed States",
        "",
    ]
    for state in STATE_FIELDS:
        payload = event["changed_states"][state]
        detail_field = STATE_DETAIL_FIELDS[state]
        detail = payload.get(detail_field, "")
        detail_text = f" - {detail}" if detail else ""
        lines.append(f"- **{state.title()}**: `{payload['status']}`{detail_text}")
    lines.extend(["", "### Reasoning", "", event["reasoning"]])
    links = event.get("links") or []
    if links:
        lines.extend(["", "### Links", ""])
        for link in links:
            lines.append(f"- {link}")
    return "\n".join(lines).rstrip() + "\n"


def render_events_markdown(events: Iterable[dict[str, Any]], title: str) -> str:
    rendered = [f"# {title}", ""]
    for event in events:
        rendered.append(render_event_markdown(event).rstrip())
        rendered.append("")
    return "\n".join(rendered).rstrip() + "\n"


def sample_event() -> dict[str, Any]:
    return normalize_event(
        {
            "source": {
                "kind": "wiki",
                "ref": "wiki/state-change-log-engine.md",
                "title": "State Change Log Engine",
            },
            "evidence_item": "Ingest should produce structured state transitions, not only paper summaries.",
            "confidence": "high",
            "privacy_sensitivity": "none",
            "changed_states": {
                "evidence": {
                    "status": "strengthened",
                    "claim": "ScholarHound needs a non-LLM-readable state layer.",
                },
                "concept": {
                    "status": "new",
                    "concept": "State Change Log Engine",
                },
                "trajectory": {
                    "status": "reinforce",
                    "trajectory": "Research OS with LLM as parsing layer",
                },
                "constraint": {
                    "status": "interpretation",
                    "constraint": "Do not treat summaries as kernel state.",
                },
                "uncertainty": {
                    "status": "reduced",
                    "uncertainty": "Whether the next OS step should be structured and validator-backed.",
                },
                "action": {
                    "status": "verify",
                    "next_action": "Validate JSONL events and render Markdown views from them.",
                },
            },
            "reasoning": (
                "This event makes the state-change layer explicit and gives non-LLM "
                "code a validated object to read, append and render."
            ),
            "links": [
                "[[state-change-log-engine]]",
                "[[scholarhound-research-os]]",
            ],
        }
    )
