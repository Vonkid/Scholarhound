import json

import pytest
from click.testing import CliRunner

from psil.cli import main
from psil.state_change import (
    StateChangeValidationError,
    append_event,
    ensure_valid_event,
    read_events,
    render_events_markdown,
    sample_event,
    validate_event_log,
)


def test_sample_event_is_valid_and_complete():
    event = ensure_valid_event(sample_event())

    assert event["event_id"].startswith("scr_")
    assert event["source"]["kind"] == "wiki"
    assert event["confidence"] == "high"
    assert event["privacy_sensitivity"] == "none"
    assert set(event["changed_states"]) == {
        "evidence",
        "concept",
        "trajectory",
        "constraint",
        "uncertainty",
        "action",
    }


def test_append_and_read_events_roundtrip(tmp_path):
    path = tmp_path / "kernel" / "state_changes.jsonl"

    stored = append_event(path, sample_event())
    events = read_events(path)

    assert len(events) == 1
    assert events[0]["event_id"] == stored["event_id"]
    assert events[0]["changed_states"]["trajectory"]["status"] == "reinforce"

    raw_event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert raw_event["event_id"] == stored["event_id"]


def test_invalid_state_status_is_rejected():
    event = sample_event()
    event["changed_states"]["evidence"]["status"] = "amplified"

    with pytest.raises(StateChangeValidationError) as exc_info:
        ensure_valid_event(event)

    assert "$.changed_states.evidence.status" in str(exc_info.value)


def test_validate_event_log_reports_jsonl_errors(tmp_path):
    path = tmp_path / "state_changes.jsonl"
    path.write_text('{"source": "missing required fields"}\n', encoding="utf-8")

    events, issues = validate_event_log(path)

    assert events == []
    assert issues
    assert str(path) in issues[0].path


def test_render_markdown_from_validated_events():
    rendered = render_events_markdown([sample_event()], title="Kernel View")

    assert rendered.startswith("# Kernel View\n")
    assert "## scr_" in rendered
    assert "**Evidence**: `strengthened`" in rendered
    assert "**Action**: `verify`" in rendered


def test_cli_appends_candidate_event_file(tmp_path):
    event_file = tmp_path / "candidate_event.json"
    log_path = tmp_path / "state_changes.jsonl"
    event_file.write_text(json.dumps(sample_event()), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["state-append", str(event_file), "--path", str(log_path)],
    )

    assert result.exit_code == 0, result.output
    assert "Appended scr_" in result.output
    assert len(read_events(log_path)) == 1
