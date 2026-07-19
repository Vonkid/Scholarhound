import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


def _frontend_html(name: str) -> str:
    html_path = Path(__file__).resolve().parents[1] / "psil" / name
    return html_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["scholarhound.html", "frontend.html"])
def test_frontend_inline_scripts_parse(name):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    html = _frontend_html(name)
    scripts = re.findall(
        r"<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)</script>",
        html,
        flags=re.IGNORECASE,
    )

    runner = """
const vm = require('vm');
const scripts = JSON.parse(process.argv[1]);
scripts.forEach((script, index) => {
  new vm.Script(script, { filename: `frontend-inline-${index}.js` });
});
"""
    subprocess.run([node, "-e", runner, json.dumps(scripts)], check=True)


def test_trajectory_logic_overlay_parses():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    script_path = Path(__file__).resolve().parents[1] / "psil" / "trajectory_logic.js"
    subprocess.run([node, "--check", str(script_path)], check=True)


def test_belief_state_is_a_bounded_scholarhound_function():
    html = _frontend_html("scholarhound.html")

    assert "Persistent belief kernel" in html
    assert "Belief state" in html
    assert "Evidence ledger" in html
    assert "Revision history" in html
    assert "Flagged for your judgment" in html
    assert "literature signal" in html
    assert "/api/kernel/v3/belief-map" in html
    assert 'href="/review"' in html
    assert "/api/benchmark/auth" not in html
    assert "Human Benchmark Console" not in html


def test_scholarhound_keeps_judgment_bounded_and_auditable():
    html = _frontend_html("scholarhound.html")

    assert "not probability of truth" in html
    assert "State connected" in html
    assert "provenance on every node" in html
    assert "Human judgment boundary" in html
    assert "support" in html
    assert "pressure" in html
    assert "unresolved" in html


def test_frozen_product_frontend_remains_unchanged():
    snapshot = Path(__file__).resolve().parents[1] / "psil" / "frozen_frontend.html"
    html = snapshot.read_text(encoding="utf-8")

    assert "Dashboard" in html
    assert "Digest" in html
    assert "Trajectory Map" in html
    assert "/api/dashboard" in html
    assert "/api/digests" in html
    assert "/api/trajectory-map" in html
    assert "trajectory-logic-js" not in html


def test_review_console_uses_blind_feedback_api():
    html = _frontend_html("frontend.html")

    assert "Claim-Abstract Review" in html
    assert "/api/benchmark/auth" in html
    assert "/api/benchmark/login" in html
    assert "/api/benchmark/logout" in html
    assert "/api/benchmark/progress" in html
    assert "/api/benchmark/session" in html
    assert "/api/benchmark/feedback" in html
    assert "/api/benchmark/session?packet=" in html
    assert "packet_key" in html
    assert "https://doi.org/" in html
    assert 'credentials: "same-origin"' in html
    assert 'cache: "no-store"' in html
    assert "Base your answer only on the claim and abstract shown here." in html
    assert "Kernel prediction withheld" not in html
    assert "Kernel Calibration Trace" not in html
    assert "/api/dashboard" not in html


def test_review_console_requires_session_and_offers_distinct_sets():
    html = _frontend_html("frontend.html")

    assert "Reviewer authentication" in html
    assert "Choose review set." in html
    assert 'id="login-form"' in html
    assert 'id="mode-gate"' in html
    assert 'data-packet-choice="calibration_24"' in html
    assert 'data-packet-choice="full_72"' in html
    assert 'id="login-reviewer"' in html
    assert 'id="login-code"' in html
    assert 'id="identity-chip"' in html
    assert 'id="reviewer-label"' in html
    assert "resume_item_id" in html
    assert "processed_ids" in html
    assert "session_run_id" in html
    assert "client_started_at" in html


def test_review_console_collects_neutral_breakdown_without_defaults():
    html = _frontend_html("frontend.html")

    assert "Supports" in html
    assert "Challenges" in html
    assert "Neutral / Off-topic" in html
    assert "Defer / Underdetermined" in html
    assert "Contested" in html
    assert 'id="assertions"' in html
    assert 'id="covered"' in html
    assert 'id="gap"' in html
    assert 'data-confidence="low"' in html
    assert 'data-confidence="medium"' in html
    assert 'data-confidence="high"' in html
    assert 'state.confidence = ""' in html
    assert 'button.classList.remove("active")' in html
    assert 'id="submit-btn"' in html
    assert 'id="skip-btn"' in html
