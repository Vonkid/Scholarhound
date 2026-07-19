import json
from pathlib import Path
import subprocess
import sys

import pytest
from click.testing import CliRunner

from psil.v3_kernel import (
    CONFIDENCE_SOFT_CAP,
    V3KernelValidationError,
    apply_research_judgment_decision,
    attach_parse_boundary_provenance,
    consensus_evidence_relation,
    create_belief,
    create_evidence,
    create_evidence_from_parse_candidates,
    create_kernel_intake_assessment,
    create_research_judgment_decision,
    get_contested_evidence_queue,
    get_pending_evidence_queue,
    infer_evidence_strength_from_facts,
    object_path,
    parse_boundary_status,
    read_jsonl,
    reclassify_pending_evidence,
    revise_belief,
    validate_v3_kernel,
)


def test_evidence_creates_belief_and_revision(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "report",
            "title": "Novelty stress test",
            "source_ref": "ScholarHound_novelty_stress_test.html",
            "summary": "Belief should become the kernel object.",
            "evidence_strength": "strong",
        },
    )

    belief, revision = create_belief(
        kernel_dir,
        {
            "title": "Belief-centered V3",
            "claim": "V3 should treat belief as first-class kernel state.",
            "domain": "architecture",
            "status": "active",
            "confidence": 0.7,
            "entrenchment": 0.3,
        },
        reason="The report supports a belief-centered architecture.",
        evidence_ids=[evidence["id"]],
    )

    assert belief["last_revision_id"] == revision["id"]
    assert revision["action"] == "create"
    assert revision["old_confidence"] is None
    assert revision["new_confidence"] == 0.7
    assert read_jsonl(object_path(kernel_dir, "revisions")) == [revision]

    health, issues = validate_v3_kernel(kernel_dir)

    assert issues == []
    assert health["belief_count"] == 1
    assert health["revision_count"] == 1


def test_revise_belief_is_append_only_and_projection_updates(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    supporting = create_evidence(
        kernel_dir,
        {
            "source_type": "report",
            "title": "Architecture report",
            "source_ref": "report.html",
            "summary": "Supports V3 belief kernel.",
            "evidence_strength": "moderate",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Belief-centered V3",
            "claim": "V3 should treat belief as first-class kernel state.",
            "domain": "architecture",
            "confidence": 0.7,
            "entrenchment": 0.3,
        },
        reason="Initial support.",
        evidence_ids=[supporting["id"]],
    )
    challenge = create_evidence(
        kernel_dir,
        {
            "source_type": "external",
            "title": "BEWA prior art",
            "source_ref": "arxiv:2506.16015",
            "summary": "Challenges novelty of belief updating.",
            "evidence_strength": "strong",
            "challenges_beliefs": [belief["id"]],
        },
    )

    updated, revision = revise_belief(
        kernel_dir,
        belief_id=belief["id"],
        evidence_ids=[challenge["id"]],
        action="challenge",
        reason="BEWA makes pure belief-updating novelty dangerous.",
    )

    assert updated["status"] == "challenged"
    assert updated["confidence"] < belief["confidence"]
    assert challenge["id"] in updated["contra_evidence_ids"]
    assert revision["old_confidence"] == belief["confidence"]
    assert revision["new_confidence"] == updated["confidence"]
    assert len(read_jsonl(object_path(kernel_dir, "revisions"))) == 2

    health, issues = validate_v3_kernel(kernel_dir)

    assert issues == []
    assert health["belief_status_counts"]["challenged"] == 1


def test_revise_belief_rejects_missing_evidence(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "report",
            "title": "Report",
            "source_ref": "report.html",
            "summary": "Supports initial belief.",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Belief-centered V3",
            "claim": "V3 should treat belief as first-class kernel state.",
            "domain": "architecture",
        },
        reason="Initial support.",
        evidence_ids=[evidence["id"]],
    )

    with pytest.raises(V3KernelValidationError) as exc_info:
        revise_belief(
            kernel_dir,
            belief_id=belief["id"],
            evidence_ids=["missing"],
            action="strengthen",
            reason="Should fail.",
        )

    assert "unknown evidence" in str(exc_info.value)
    assert len(read_jsonl(object_path(kernel_dir, "revisions"))) == 1


def test_contested_evidence_is_logged_without_confidence_update(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    seed_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Initial paper",
            "source_ref": "doi:10.0/initial",
            "summary": "Initial support for the belief.",
            "evidence_strength": "moderate",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "OECT value is integration",
            "claim": "OECT value for EV work is integration rather than raw sensitivity.",
            "domain": "ev-biosensing",
            "confidence": 0.6,
            "entrenchment": 0.25,
        },
        reason="Initial support.",
        evidence_ids=[seed_evidence["id"]],
    )
    contested = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Ambiguous OECT evidence",
            "source_ref": "doi:10.0/ambiguous",
            "summary": "Evidence has a strong LOD but the signal depends on an enrichment front-end.",
            "evidence_strength": "strong",
            "contests_beliefs": [belief["id"]],
        },
    )

    updated, revision = revise_belief(
        kernel_dir,
        belief_id=belief["id"],
        evidence_ids=[contested["id"]],
        action="contest",
        reason="The evidence direction is ambiguous and needs human adjudication.",
    )

    assert updated["status"] == "contested"
    assert updated["confidence"] == belief["confidence"]
    assert contested["id"] in updated["contested_evidence_ids"]
    assert contested["id"] not in updated["evidence_ids"]
    assert contested["id"] not in updated["contra_evidence_ids"]
    assert revision["new_confidence"] == revision["old_confidence"]

    health, issues = validate_v3_kernel(kernel_dir)

    assert issues == []
    assert health["belief_status_counts"]["contested"] == 1
    assert health["contested_evidence_count"] == 1


def test_contested_evidence_does_not_move_confidence_even_if_action_is_directional(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    seed_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Initial paper",
            "source_ref": "doi:10.0/initial",
            "summary": "Initial support for the belief.",
            "evidence_strength": "moderate",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "OECT value is integration",
            "claim": "OECT value for EV work is integration rather than raw sensitivity.",
            "domain": "ev-biosensing",
            "confidence": 0.6,
        },
        reason="Initial support.",
        evidence_ids=[seed_evidence["id"]],
    )
    contested = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Ambiguous OECT evidence",
            "source_ref": "doi:10.0/ambiguous",
            "summary": "Evidence could support or challenge the belief depending on framing.",
            "evidence_strength": "decisive",
            "contests_beliefs": [belief["id"]],
        },
    )

    updated, _revision = revise_belief(
        kernel_dir,
        belief_id=belief["id"],
        evidence_ids=[contested["id"]],
        action="strengthen",
        reason="This should not auto-commit a contested sign.",
    )

    assert updated["confidence"] == belief["confidence"]
    assert contested["id"] in updated["contested_evidence_ids"]
    assert contested["id"] not in updated["evidence_ids"]
    assert contested["id"] not in updated["contra_evidence_ids"]


def test_repeated_supporting_evidence_has_diminishing_confidence_returns(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    seed_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Initial paper",
            "source_ref": "doi:10.0/initial",
            "summary": "Initial support for the belief.",
            "evidence_strength": "moderate",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Repeated papers should not saturate confidence",
            "claim": "Same-direction paper evidence should have diminishing returns.",
            "domain": "kernel-calibration",
            "confidence": 0.5,
        },
        reason="Initial support.",
        evidence_ids=[seed_evidence["id"]],
    )
    updated = belief
    first_delta = None

    for index in range(20):
        evidence = create_evidence(
            kernel_dir,
            {
                "source_type": "paper",
                "title": f"Supporting paper {index}",
                "source_ref": f"doi:10.0/support-{index}",
                "summary": "Additional same-direction support.",
                "evidence_strength": "moderate",
                "supports_beliefs": [belief["id"]],
            },
        )
        updated, revision = revise_belief(
            kernel_dir,
            belief_id=belief["id"],
            evidence_ids=[evidence["id"]],
            action="strengthen",
            reason="Repeated supporting paper.",
        )
        if first_delta is None:
            first_delta = revision["confidence_delta"]

    assert updated["confidence"] == CONFIDENCE_SOFT_CAP
    assert revision["confidence_delta"] < first_delta
    assert updated["entrenchment"] < 0.2
    assert revision["confidence_policy"]["reason"] == "unvalidated_soft_cap"
    assert (
        revision["confidence_delta_policy"]["method"]
        == "diminishing_returns_with_entrenchment_resistance_v1"
    )
    assert revision["entrenchment_delta_policy"]["method"] == "dependency_weighted_entrenchment_v1"


def test_entrenchment_resists_confidence_revision(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    low_belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Low entrenchment belief",
            "claim": "A weakly entrenched belief should move more easily.",
            "domain": "kernel-calibration",
            "confidence": 0.5,
            "entrenchment": 0.0,
        },
        reason="Initial low-entrenchment belief.",
        evidence_ids=[],
    )
    high_belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "High entrenchment belief",
            "claim": "A strongly entrenched belief should resist revision.",
            "domain": "kernel-calibration",
            "confidence": 0.5,
            "entrenchment": 0.8,
        },
        reason="Initial high-entrenchment belief.",
        evidence_ids=[],
    )
    low_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Same support low",
            "source_ref": "doi:10.0/same-support-low",
            "summary": "Same-strength supporting evidence.",
            "evidence_strength": "moderate",
            "supports_beliefs": [low_belief["id"]],
        },
    )
    high_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Same support high",
            "source_ref": "doi:10.0/same-support-high",
            "summary": "Same-strength supporting evidence.",
            "evidence_strength": "moderate",
            "supports_beliefs": [high_belief["id"]],
        },
    )

    _low_updated, low_revision = revise_belief(
        kernel_dir,
        belief_id=low_belief["id"],
        evidence_ids=[low_evidence["id"]],
        action="strengthen",
        reason="Same evidence against weak entrenchment.",
    )
    _high_updated, high_revision = revise_belief(
        kernel_dir,
        belief_id=high_belief["id"],
        evidence_ids=[high_evidence["id"]],
        action="strengthen",
        reason="Same evidence against strong entrenchment.",
    )

    high_step = high_revision["confidence_delta_policy"]["steps"][0]

    assert high_revision["confidence_delta"] < low_revision["confidence_delta"]
    assert low_revision["confidence_delta"] == 0.1
    assert high_step["entrenchment"] == 0.8
    assert high_step["entrenchment_resistance_factor"] < 0.5


def test_human_override_can_exceed_soft_confidence_cap(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "report",
            "title": "Human-validated result",
            "source_ref": "local:validated-result",
            "summary": "Human validated a decisive project result.",
            "evidence_strength": "strong",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Human override keeps truth-evaluation outside the kernel",
            "claim": "Human adjudication can override soft calibration limits.",
            "domain": "kernel-calibration",
            "confidence": 0.9,
        },
        reason="Initial support.",
        evidence_ids=[evidence["id"]],
    )

    updated, revision = revise_belief(
        kernel_dir,
        belief_id=belief["id"],
        evidence_ids=[evidence["id"]],
        action="strengthen",
        reason="Human explicitly validated this belief.",
        confidence_delta=0.2,
        human_override_id="override_manual_validation",
    )

    assert updated["confidence"] == 1.0
    assert updated["entrenchment"] > belief["entrenchment"] + 0.07
    assert revision["confidence_policy"]["reason"] == "human_override_allows_full_range"
    assert revision["entrenchment_policy"]["human_override"] is True


def test_entrenchment_is_dependency_weighted(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    seed_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "report",
            "title": "Seed",
            "source_ref": "local:seed",
            "summary": "Seed evidence.",
        },
    )
    plain_belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Plain belief",
            "claim": "A belief with no downstream dependency.",
            "domain": "kernel-calibration",
            "confidence": 0.5,
            "entrenchment": 0.1,
        },
        reason="Seed plain belief.",
        evidence_ids=[seed_evidence["id"]],
    )
    dependent_belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Dependent belief",
            "claim": "A belief reused by concepts and questions is harder to revise.",
            "domain": "kernel-calibration",
            "confidence": 0.5,
            "entrenchment": 0.1,
            "linked_concepts": ["concept-a", "concept-b", "concept-c"],
            "linked_questions": ["question-a", "question-b"],
        },
        reason="Seed dependent belief.",
        evidence_ids=[seed_evidence["id"]],
    )
    plain_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "benchmark",
            "title": "Plain benchmark",
            "source_ref": "local:plain-benchmark",
            "summary": "Benchmark supports plain belief.",
            "supports_beliefs": [plain_belief["id"]],
        },
    )
    dependent_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "benchmark",
            "title": "Dependent benchmark",
            "source_ref": "local:dependent-benchmark",
            "summary": "Benchmark supports dependent belief.",
            "supports_beliefs": [dependent_belief["id"]],
        },
    )

    plain_updated, plain_revision = revise_belief(
        kernel_dir,
        belief_id=plain_belief["id"],
        evidence_ids=[plain_evidence["id"]],
        action="strengthen",
        reason="Benchmark support.",
    )
    dependent_updated, dependent_revision = revise_belief(
        kernel_dir,
        belief_id=dependent_belief["id"],
        evidence_ids=[dependent_evidence["id"]],
        action="strengthen",
        reason="Benchmark support.",
    )

    plain_delta = plain_updated["entrenchment"] - plain_belief["entrenchment"]
    dependent_delta = dependent_updated["entrenchment"] - dependent_belief["entrenchment"]

    assert dependent_delta > plain_delta
    assert plain_revision["entrenchment_delta_policy"]["dependency_count"] == 0
    assert dependent_revision["entrenchment_delta_policy"]["dependency_count"] == 5


def test_relation_consensus_commits_only_unanimous_direction(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    seed_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Initial paper",
            "source_ref": "doi:10.0/initial",
            "summary": "Initial support for the belief.",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Recognition must switch state",
            "claim": "Bioelectronic recognition only matters if binding changes device state.",
            "domain": "molecular-recognition",
        },
        reason="Initial support.",
        evidence_ids=[seed_evidence["id"]],
    )

    evidence = create_evidence_from_parse_candidates(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "State-switching sensor",
            "source_ref": "doi:10.0/state-switch",
            "summary": "Parser candidates agree this supports the belief.",
            "reports_lod": True,
            "direction": "support",
        },
        belief_id=belief["id"],
        parse_candidates=[
            {"model": "parser_a", "relation": "support"},
            {"model": "parser_b", "direction": "supports"},
        ],
    )

    assert evidence["supports_beliefs"] == [belief["id"]]
    assert evidence["challenges_beliefs"] == []
    assert evidence["contests_beliefs"] == []
    assert evidence["evidence_relation_provenance"]["relation"] == "support"
    assert evidence["evidence_relation_provenance"]["contested"] is False
    assert evidence["parse_boundary"]["verified_low_inference_fields"] == [
        "reports_lod",
        "source_type",
    ]
    assert evidence["parse_boundary"]["judgment_heavy_fields"] == ["direction"]


def test_relation_consensus_routes_disagreement_to_contested_queue(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    seed_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Initial paper",
            "source_ref": "doi:10.0/initial",
            "summary": "Initial support for the belief.",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Recognition must switch state",
            "claim": "Bioelectronic recognition only matters if binding changes device state.",
            "domain": "molecular-recognition",
            "confidence": 0.55,
        },
        reason="Initial support.",
        evidence_ids=[seed_evidence["id"]],
    )

    evidence = create_evidence_from_parse_candidates(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Ambiguous binding paper",
            "source_ref": "doi:10.0/ambiguous-binding",
            "summary": "The paper reports binding, but device-state impact is unclear.",
            "relation": "support",
        },
        belief_id=belief["id"],
        parse_candidates=[
            {"model": "parser_a", "relation": "support"},
            {"model": "parser_b", "relation": "challenge"},
            {"model": "parser_c", "relation": "unclear"},
        ],
    )
    updated, revision = revise_belief(
        kernel_dir,
        belief_id=belief["id"],
        evidence_ids=[evidence["id"]],
        action="contest",
        reason="Parser candidates disagree on evidence direction.",
    )
    queue = get_contested_evidence_queue(kernel_dir)

    assert evidence["supports_beliefs"] == []
    assert evidence["challenges_beliefs"] == []
    assert evidence["contests_beliefs"] == [belief["id"]]
    assert evidence["evidence_relation_provenance"]["relation"] == "contest"
    assert evidence["evidence_relation_provenance"]["contested"] is True
    assert evidence["evidence_relation_provenance"]["needs_human"] is True
    assert updated["confidence"] == belief["confidence"]
    assert revision["action"] == "contest"
    assert queue[0]["evidence_id"] == evidence["id"]
    assert queue[0]["relation_provenance"]["reason"] == (
        "Parser candidates made conflicting support/challenge claims requiring human adjudication."
    )


def test_relation_consensus_marks_underdetermined_without_human_queue(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    seed_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Initial paper",
            "source_ref": "doi:10.0/initial",
            "summary": "Initial support for the belief.",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Recognition must switch state",
            "claim": "Bioelectronic recognition only matters if binding changes device state.",
            "domain": "molecular-recognition",
            "confidence": 0.55,
        },
        reason="Initial support.",
        evidence_ids=[seed_evidence["id"]],
    )

    evidence = create_evidence_from_parse_candidates(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Incomplete binding paper",
            "source_ref": "doi:10.0/incomplete-binding",
            "summary": "The paper may support the belief, but parser coverage is incomplete.",
        },
        belief_id=belief["id"],
        parse_candidates=[
            {"model": "parser_a", "relation": "support"},
            {"model": "parser_b", "relation": "unclear"},
        ],
    )
    updated, revision = revise_belief(
        kernel_dir,
        belief_id=belief["id"],
        evidence_ids=[evidence["id"]],
        action="update",
        reason="Evidence direction is underdetermined, not conflicting.",
    )
    contested_queue = get_contested_evidence_queue(kernel_dir)
    pending_queue = get_pending_evidence_queue(kernel_dir)

    assert evidence["supports_beliefs"] == []
    assert evidence["challenges_beliefs"] == []
    assert evidence["pending_beliefs"] == [belief["id"]]
    assert evidence["contests_beliefs"] == []
    assert evidence["evidence_relation_provenance"]["relation"] == "underdetermined"
    assert evidence["evidence_relation_provenance"]["contested"] is False
    assert evidence["evidence_relation_provenance"]["needs_human"] is False
    assert evidence["evidence_relation_provenance"]["needs_more_evidence"] is True
    assert evidence["evidence_relation_provenance"]["weak_direction"] == "support"
    assert updated["confidence"] == belief["confidence"]
    assert evidence["id"] in updated["pending_evidence_ids"]
    assert evidence["id"] not in updated["contested_evidence_ids"]
    assert revision["action"] == "update"
    assert contested_queue == []
    assert pending_queue[0]["evidence_id"] == evidence["id"]
    assert pending_queue[0]["relation_provenance"]["weak_direction"] == "support"


def test_pending_evidence_can_be_reclassified_to_support(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    seed_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Initial paper",
            "source_ref": "doi:10.0/initial",
            "summary": "Initial support for the belief.",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Recognition must switch state",
            "claim": "Bioelectronic recognition only matters if binding changes device state.",
            "domain": "molecular-recognition",
            "confidence": 0.55,
        },
        reason="Initial support.",
        evidence_ids=[seed_evidence["id"]],
    )
    evidence = create_evidence_from_parse_candidates(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Incomplete binding paper",
            "source_ref": "doi:10.0/incomplete-binding",
            "summary": "The paper may support the belief, but parser coverage is incomplete.",
        },
        belief_id=belief["id"],
        parse_candidates=[
            {"model": "parser_a", "relation": "support"},
            {"model": "parser_b", "relation": "unclear"},
        ],
    )
    pending_belief, _pending_revision = revise_belief(
        kernel_dir,
        belief_id=belief["id"],
        evidence_ids=[evidence["id"]],
        action="update",
        reason="Evidence direction is underdetermined, not conflicting.",
    )

    updated, revision = reclassify_pending_evidence(
        kernel_dir,
        belief_id=belief["id"],
        evidence_id=evidence["id"],
        relation="support",
        reason="Later evidence resolves the pending item as support.",
    )
    evidence_after = read_jsonl(object_path(kernel_dir, "evidence"))[-1]
    health, issues = validate_v3_kernel(kernel_dir)

    assert issues == []
    assert health["validation_status"] == "ok"
    assert health["pending_evidence_count"] == 0
    assert evidence["id"] not in updated["pending_evidence_ids"]
    assert evidence["id"] in updated["evidence_ids"]
    assert evidence["id"] not in updated["contested_evidence_ids"]
    assert evidence_after["pending_beliefs"] == []
    assert evidence_after["supports_beliefs"] == [belief["id"]]
    assert evidence_after["evidence_relation_provenance"]["method"] == (
        "pending_reclassification_v1"
    )
    assert evidence_after["evidence_relation_provenance"]["previous_relation"] == (
        "underdetermined"
    )
    assert updated["confidence"] > pending_belief["confidence"]
    assert revision["action"] == "strengthen"
    assert revision["confidence_delta"] > 0
    assert get_pending_evidence_queue(kernel_dir) == []


def test_pending_evidence_can_be_reclassified_to_contest(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    seed_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Initial paper",
            "source_ref": "doi:10.0/initial",
            "summary": "Initial support for the belief.",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Recognition must switch state",
            "claim": "Bioelectronic recognition only matters if binding changes device state.",
            "domain": "molecular-recognition",
        },
        reason="Initial support.",
        evidence_ids=[seed_evidence["id"]],
    )
    evidence = create_evidence_from_parse_candidates(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Incomplete binding paper",
            "source_ref": "doi:10.0/incomplete-binding",
            "summary": "The paper may support the belief, but parser coverage is incomplete.",
        },
        belief_id=belief["id"],
        parse_candidates=[
            {"model": "parser_a", "relation": "support"},
            {"model": "parser_b", "relation": "unclear"},
        ],
    )
    pending_belief, _pending_revision = revise_belief(
        kernel_dir,
        belief_id=belief["id"],
        evidence_ids=[evidence["id"]],
        action="update",
        reason="Evidence direction is underdetermined, not conflicting.",
    )

    updated, revision = reclassify_pending_evidence(
        kernel_dir,
        belief_id=belief["id"],
        evidence_id=evidence["id"],
        relation="contest",
        reason="Later evidence shows true support/challenge conflict.",
    )
    queue = get_contested_evidence_queue(kernel_dir)

    assert updated["confidence"] == pending_belief["confidence"]
    assert evidence["id"] not in updated["pending_evidence_ids"]
    assert evidence["id"] in updated["contested_evidence_ids"]
    assert revision["action"] == "contest"
    assert queue[0]["evidence_id"] == evidence["id"]
    assert queue[0]["relation_provenance"]["needs_human"] is True


def test_parse_boundary_provenance_marks_objective_and_judgment_fields():
    assert parse_boundary_status("source_type")["status"] == "verified_low_inference"
    assert parse_boundary_status("direction")["status"] == "judgment_heavy"
    assert parse_boundary_status("novel_field")["status"] == "unclassified"

    provenanced = attach_parse_boundary_provenance(
        {
            "source_type": "paper",
            "reports_lod": True,
            "independent_validation": True,
            "direction": "support",
        }
    )

    assert provenanced["parse_boundary"]["verified_low_inference_fields"] == [
        "reports_lod",
        "source_type",
    ]
    assert provenanced["parse_boundary"]["unverified_fields"] == [
        "independent_validation",
    ]
    assert provenanced["parse_boundary"]["judgment_heavy_fields"] == ["direction"]


def test_consensus_relation_without_candidates_is_underdetermined():
    relation = consensus_evidence_relation([])

    assert relation["relation"] == "underdetermined"
    assert relation["contested"] is False
    assert relation["needs_more_evidence"] is True
    assert relation["votes"] == []


def test_neutral_relation_is_tracked_without_queue_or_confidence_delta(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    seed_evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Initial paper",
            "source_ref": "doi:10.0/initial",
            "summary": "Initial support for the belief.",
        },
    )
    belief, _revision = create_belief(
        kernel_dir,
        {
            "title": "Recognition must switch state",
            "claim": "Binding should change a device state.",
            "domain": "molecular-recognition",
            "confidence": 0.55,
        },
        reason="Initial support.",
        evidence_ids=[seed_evidence["id"]],
    )

    neutral = create_evidence_from_parse_candidates(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Off-topic paper",
            "source_ref": "doi:10.0/off-topic",
            "summary": "This paper is unrelated to the belief.",
        },
        belief_id=belief["id"],
        parse_candidates=[
            {"model": "parser_a", "relation": "neutral"},
            {"model": "parser_b", "relation": "off-topic"},
        ],
    )
    updated, revision = revise_belief(
        kernel_dir,
        belief_id=belief["id"],
        evidence_ids=[neutral["id"]],
        action="update",
        reason="Evidence is neutral for this belief.",
    )
    health, issues = validate_v3_kernel(kernel_dir)

    assert issues == []
    assert neutral["neutral_beliefs"] == [belief["id"]]
    assert neutral["evidence_relation_provenance"]["relation"] == "neutral"
    assert neutral["evidence_relation_provenance"]["needs_human"] is False
    assert neutral["evidence_relation_provenance"]["needs_more_evidence"] is False
    assert updated["confidence"] == belief["confidence"]
    assert revision["confidence_delta"] == 0
    assert neutral["id"] in updated["neutral_evidence_ids"]
    assert neutral["id"] not in updated["pending_evidence_ids"]
    assert neutral["id"] not in updated["contested_evidence_ids"]
    assert get_pending_evidence_queue(kernel_dir) == []
    assert get_contested_evidence_queue(kernel_dir) == []
    assert health["neutral_evidence_count"] == 1


def test_evidence_strength_can_be_inferred_from_low_inference_fields():
    inferred = infer_evidence_strength_from_facts(
        {
            "source_type": "paper",
            "venue_tier": 1,
            "primary_research": True,
            "independent_validation": True,
            "has_benchmark": True,
            "has_controls": True,
            "reports_lod": True,
            "sample_size": 120,
        }
    )

    assert inferred["evidence_strength"] == "moderate"
    assert inferred["method"] == "deterministic_from_verified_low_inference_fields"
    assert inferred["accepted_input_fields"] == ["source_type", "reports_lod"]
    assert "reports_lod (+0.25)" in inferred["factors"]
    assert "sample_size=120 (+1)" not in inferred["factors"]
    assert "independent_validation" in inferred["excluded_until_parse_boundary_tested"]
    assert inferred["calibration_status"] == "uncalibrated_v3_alpha"


def test_missing_evidence_strength_is_normalized_with_provenance(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "benchmark",
            "title": "Primary sensor paper",
            "source_ref": "doi:10.0/sensor",
            "summary": "Reports a quantitative sensor benchmark.",
            "reports_lod": True,
        },
    )

    assert evidence["evidence_strength"] == "strong"
    assert evidence["evidence_strength_provenance"]["method"] == "deterministic_from_verified_low_inference_fields"


def test_asserted_evidence_strength_is_labeled_with_provenance(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    evidence = create_evidence(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Caller-scored paper",
            "source_ref": "doi:10.0/asserted",
            "summary": "Caller supplied a strength label.",
            "evidence_strength": "decisive",
            "reports_lod": True,
        },
    )

    assert evidence["evidence_strength"] == "decisive"
    assert evidence["evidence_strength_provenance"]["method"] == "asserted_by_caller"
    assert evidence["evidence_strength_provenance"]["asserted_strength"] == "decisive"


def _sample_secretary_briefing(**overrides):
    briefing = {
        "schema_id": "audit_secretary_briefing_schema_v1",
        "briefing_id": "brief_test_001",
        "source": {
            "paper_id": "paper_001",
            "title": "State-switching bioelectronic sensor",
            "doi": "10.1000/test",
            "abstract": (
                "We report a bioelectronic sensor that changes device state after "
                "molecular recognition. The response is quantified in buffer and "
                "in plasma. Controls are discussed in the abstract."
            ),
            "input_text_level": "abstract",
        },
        "claim_map": [
            {
                "claim_id": "claim_01",
                "claim_text": "Recognition changes device state.",
                "claim_type": "result",
                "evidence_basis": "direct_experiment",
            }
        ],
        "evidence_map": {
            "source_type": "paper",
            "evidence_strength_inputs": {
                "abstract_level_only": True,
                "reports_lod": False,
            },
        },
        "relation_scope": {
            "belief_id": "B-SENSE",
            "belief": "Recognition must switch a measurable state.",
            "reader_relation_by_atom": [],
            "atom_consensus": [],
            "whole_belief_projection_candidate": "not_computed",
            "legacy_whole_belief_reader": {
                "votes": {"model_a": "support", "model_b": "support"},
                "kernel_relation": "support",
            },
        },
        "uncertainty_map": [
            {
                "uncertainty_id": "uncertainty_01",
                "uncertainty_type": "insufficient_input_text",
                "severity": "medium",
            }
        ],
        "disagreement_map": [],
        "candidate_kernel_updates": [
            {
                "candidate_id": "candidate_01",
                "target_layer": "source",
                "candidate_change": "Register briefing for kernel review.",
                "risk": "low",
                "commit_permission": "kernel_only",
            },
            {
                "candidate_id": "candidate_02",
                "target_layer": "uncertainty",
                "candidate_change": "Consider uncertainty state.",
                "risk": "medium",
                "commit_permission": "kernel_only",
            },
        ],
        "attention_brief": {
            "attention_recommendation": "skim",
            "reason": "Legacy relation was support; kernel review still required.",
        },
        "provenance": {
            "created_at": "2026-06-12T00:00:00+00:00",
            "source_text_hash": "abc123",
        },
    }
    briefing.update(overrides)
    return briefing


def _methodology_modules():
    return {
        "source_provenance": {
            "doi": "10.1000/test",
            "source_text_hash": "abc123",
            "retrieval_route": "database",
            "read_level": "abstract",
        },
        "object_grounding": {
            "grounded_objects": ["source", "claim", "mechanism"],
            "text_only": True,
            "object_text_alignment": "partial",
            "hallucination_risk_from_language_prior": "medium",
        },
        "confidence_dynamics": {
            "usable_for_kernel_confidence": False,
            "calibration_status": "unknown",
            "confidence_shift_direction": "not_measured",
        },
        "task_reliability_envelope": {
            "admissibility": "weak_signal",
            "task_type": "relation_scope",
            "subjectivity_level": "high",
        },
        "epistemic_task_probe": {
            "epistemic_operator_risk": "medium",
            "requires_epistemic_boundary_review": True,
        },
        "semantic_audit": {
            "expected_concepts": ["sensor"],
            "spurious_or_invalid_concepts": [],
            "unexpected_concepts": [],
            "behavioral_risk": "low",
        },
        "calibration_status": {
            "confidence_basis": "reader_agreement",
            "unknown_or_ood_risk": "high",
            "should_downweight_confidence": True,
        },
        "trajectory_proposal": {
            "source": "semantic_link",
            "candidate_question": "Does this sharpen an active question?",
            "linked_concepts": ["sensor"],
            "expert_review_needed": True,
        },
        "human_feedback_brief": {
            "target": "kernel_decision",
            "feedback_type": "reduce_overclaim",
            "actionable_suggestion": "Keep as intake material.",
            "reliability_tests": ["specific_action_present"],
            "passed": True,
        },
        "knowledge_transfer": {
            "packet_type": "claim",
            "transfer_scope": "belief",
            "integration_policy": "kernel_may_consider",
        },
        "personalization_boundary": {
            "setting_source": "none",
            "profile_inference_allowed": False,
            "preference_effect": "none",
            "truth_or_judgment_effect": "none",
        },
        "closed_loop_validation": {
            "validation_state": "unvalidated",
            "ground_truth_available": False,
            "kernel_commit_allowed": False,
        },
        "cognitive_tooling": {
            "active_question": "What should this paper change?",
            "working_hypothesis": "Recognition changes device state.",
            "task_decomposition": ["check claims"],
            "tool_status": "draft",
        },
        "prior_trace": {
            "prior_type": "project_priority",
            "prior_effect": "interpretation",
            "critique_required": False,
            "counter_prior_needed": False,
        },
        "active_curriculum": {
            "learning_need": "weak_grounding",
            "expected_learning_gain": "medium",
            "next_source_type": "paper",
            "avoid_more_of_same": True,
        },
    }


def _key_claim_relation_scope(*, relation="support"):
    return {
        "belief_id": "B-SENSE",
        "belief": "Recognition must switch a measurable state.",
        "atoms": [
            {
                "atom_id": "state_switching",
                "claim": "Molecular recognition switches a measurable device state.",
                "role": "mechanism",
                "criticality": "required",
            }
        ],
        "reader_relation_by_atom": [],
        "atom_consensus": [],
        "whole_belief_projection_candidate": "not_computed",
        "legacy_whole_belief_reader": {
            "votes": {"model_a": relation, "model_b": relation},
            "kernel_relation": relation,
        },
    }


def _measured_key_claim_map():
    return [
        {
            "claim_id": "claim_01",
            "claim_text": "Recognition changes a measurable device state.",
            "claim_type": "mechanism",
            "evidence_type": "measured",
            "matched_key_claim_ids": ["state_switching"],
        }
    ]


def test_kernel_intake_assessment_keeps_missing_secretary_modules_non_gating(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"

    assessment = create_kernel_intake_assessment(kernel_dir, _sample_secretary_briefing())
    health, issues = validate_v3_kernel(kernel_dir)

    assert issues == []
    assert assessment["decision"] == "partial_accept"
    assert assessment["durable_change_authorization"] == "not_authorized"
    assert "semantic_audit" in assessment["missing_modules"]
    assert "calibration_status" in assessment["missing_modules"]
    assert assessment["accepted_candidates"] == ["candidate_01"]
    assert assessment["deferred_candidates"] == ["candidate_02"]
    assert "missing_secretary_reliability_modules" not in assessment["reasons"]
    assert "add_methodology_modules" not in assessment["next_actions"]
    assert assessment["provenance"]["method"] == "kernel_intake_assessment_v1_thin_contract"
    assert read_jsonl(object_path(kernel_dir, "revisions")) == []
    assert health["assessment_count"] == 1
    assert health["assessments_deferred"] == 0


def test_kernel_intake_assessment_partially_accepts_full_methodology_briefing(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    briefing = _sample_secretary_briefing(**_methodology_modules())

    assessment = create_kernel_intake_assessment(kernel_dir, briefing)

    assert assessment["decision"] == "partial_accept"
    assert assessment["missing_modules"] == []
    assert assessment["durable_change_authorization"] == "not_authorized"
    assert assessment["accepted_candidates"] == ["candidate_01"]
    assert assessment["deferred_candidates"] == ["candidate_02"]
    assert assessment["admissibility"]["task_reliability_admissibility"] == "weak_signal"
    assert assessment["admissibility"]["confidence_usable_for_kernel"] is False
    assert "request_full_text_before_revision" in assessment["next_actions"]
    assert read_jsonl(object_path(kernel_dir, "revisions")) == []


def test_kernel_intake_assessment_rejects_personalization_truth_effect(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    modules = _methodology_modules()
    modules["personalization_boundary"] = {
        "setting_source": "project_config",
        "profile_inference_allowed": False,
        "preference_effect": "ranking",
        "truth_or_judgment_effect": "change_kernel_judgment",
    }
    briefing = _sample_secretary_briefing(**modules)

    assessment = create_kernel_intake_assessment(kernel_dir, briefing)

    assert assessment["decision"] == "reject"
    assert "personalization_attempted_truth_effect" in assessment["reasons"]
    assert assessment["durable_change_authorization"] == "not_authorized"
    assert read_jsonl(object_path(kernel_dir, "revisions")) == []


def test_kernel_intake_assessment_escalates_conflicted_briefing_without_revision(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    briefing = _sample_secretary_briefing(
        relation_scope={
            "belief_id": "B-RFAU",
            "belief": "Gold nanoparticles intrinsically absorb RF.",
            "whole_belief_projection_candidate": "not_computed",
            "legacy_whole_belief_reader": {
                "votes": {"model_a": "support", "model_b": "challenge"},
                "kernel_relation": "contest",
            },
        },
        disagreement_map=[
            {
                "disagreement_id": "disagreement_01",
                "field": "legacy_whole_belief_relation",
                "suggested_resolution": "human_read",
            }
        ],
        uncertainty_map=[
            {
                "uncertainty_id": "uncertainty_01",
                "uncertainty_type": "reader_uncertainty",
                "severity": "high",
            }
        ],
    )

    assessment = create_kernel_intake_assessment(kernel_dir, briefing)

    assert assessment["decision"] == "escalate"
    assert assessment["escalation"]["needed"] is True
    assert "human_read" in assessment["next_actions"]
    assert assessment["deferred_candidates"] == ["candidate_01", "candidate_02"]
    assert read_jsonl(object_path(kernel_dir, "revisions")) == []


def test_kernel_intake_assessment_rejects_secretary_commit_attempt(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    briefing = _sample_secretary_briefing(
        candidate_kernel_updates=[
            {
                "candidate_id": "candidate_bad",
                "target_layer": "belief",
                "candidate_change": "Change belief confidence directly.",
                "risk": "high",
                "commit_permission": "secretary_can_commit",
            }
        ]
    )

    assessment = create_kernel_intake_assessment(kernel_dir, briefing)

    assert assessment["decision"] == "reject"
    assert "secretary_attempted_commit" in assessment["reasons"]
    assert assessment["rejected_candidates"] == ["candidate_bad"]
    assert any(
        check["check"] == "candidate_commit_permission" and not check["passed"]
        for check in assessment["boundary_checks"]
    )
    assert read_jsonl(object_path(kernel_dir, "revisions")) == []


def test_research_judgment_decision_escalation_requests_human_action(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    briefing = _sample_secretary_briefing(
        relation_scope={
            "belief_id": "B-RFAU",
            "belief": "Gold nanoparticles intrinsically absorb RF.",
            "whole_belief_projection_candidate": "not_computed",
            "legacy_whole_belief_reader": {
                "votes": {"model_a": "support", "model_b": "challenge"},
                "kernel_relation": "contest",
            },
        },
        disagreement_map=[
            {
                "disagreement_id": "disagreement_01",
                "field": "legacy_whole_belief_relation",
                "suggested_resolution": "human_read",
            }
        ],
        uncertainty_map=[
            {
                "uncertainty_id": "uncertainty_01",
                "uncertainty_type": "reader_uncertainty",
                "severity": "high",
            }
        ],
    )
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)

    decision = create_research_judgment_decision(kernel_dir, assessment, briefing)
    health, issues = validate_v3_kernel(kernel_dir)

    assert issues == []
    assert decision["decision"] == "request_action_or_read"
    assert decision["human_review_required"] is True
    assert "human_read" in decision["required_actions"]
    assert decision["candidate_state_changes"][0]["target_layer"] == "action"
    assert "direct_belief_revision" in decision["rejected_alternatives"]
    assert read_jsonl(object_path(kernel_dir, "revisions")) == []
    assert health["decision_count"] == 1
    assert health["decisions_requiring_human"] == 1


def test_research_judgment_decision_full_text_support_is_candidate_revision(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    modules = _methodology_modules()
    modules["source_provenance"]["read_level"] = "full_text"
    briefing = _sample_secretary_briefing(
        **modules,
        source={
            "paper_id": "paper_001",
            "title": "State-switching bioelectronic sensor",
            "doi": "10.1000/test",
            "abstract": "Abstract retained for provenance.",
            "input_text_level": "full_text",
        },
        evidence_map={
            "source_type": "paper",
            "evidence_strength_inputs": {
                "abstract_level_only": False,
                "reports_lod": False,
            },
        },
        claim_map=_measured_key_claim_map(),
        relation_scope=_key_claim_relation_scope(),
        uncertainty_map=[],
    )
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)

    decision = create_research_judgment_decision(kernel_dir, assessment, briefing)

    assert assessment["decision"] == "accept"
    assert decision["decision"] == "propose_belief_revision"
    assert decision["human_review_required"] is False
    assert decision["judgment_frame"]["primary_pressure"] == "belief_revision_pressure"
    assert decision["judgment_frame"]["relation_signal"]["role"] == "secretary_signal_not_kernel_decision"
    assert decision["judgment_frame"]["claim_pressure"]["key_claim_touch_gate"]["passed"] is True
    assert decision["judgment_frame"]["claim_pressure"]["key_claim_touch_gate"]["measured_key_claim_ids"] == ["state_switching"]
    assert decision["judgment_frame"]["claim_pressure"]["claim_design_alignment_audit"]["passed"] is True
    assert decision["judgment_frame"]["claim_pressure"]["claim_design_alignment_audit"]["status"] == "not_supplied"
    assert decision["judgment_frame"]["warrants"]["belief_revision_candidate"] is True
    assert decision["candidate_state_changes"][0]["target_layer"] == "belief"
    assert decision["candidate_state_changes"][0]["change_type"] == "support_candidate"
    assert decision["applied_revision_ids"] == []
    assert read_jsonl(object_path(kernel_dir, "revisions")) == []


def test_research_judgment_decision_overclaim_boundary_blocks_revision(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    modules = _methodology_modules()
    modules["source_provenance"]["read_level"] = "full_text"
    modules["trajectory_proposal"]["candidate_question"] = (
        "What patient-cohort design would justify diagnostic-readiness claims?"
    )
    briefing = _sample_secretary_briefing(
        **modules,
        source={
            "paper_id": "paper_overclaim",
            "title": "Prototype sensor with diagnostic-ready conclusion",
            "doi": "10.1000/overclaim",
            "abstract": "Full text retained for provenance.",
            "input_text_level": "full_text",
        },
        evidence_map={
            "source_type": "paper",
            "evidence_strength_inputs": {
                "abstract_level_only": False,
                "reports_lod": True,
            },
        },
        claim_map=[
            {
                "claim_id": "claim_01",
                "claim_text": "Recognition changes a measurable device state.",
                "claim_type": "mechanism",
                "evidence_type": "measured",
                "matched_key_claim_ids": ["state_switching"],
                "claim_scope": "diagnostic",
                "evidence_scope": "spiked_sample",
                "overclaim_risk": "high",
                "allowed_interpretation": "prototype-level sensing feasibility",
                "disallowed_interpretation": "disease diagnostic readiness",
                "design_shortages": ["no patient cohort", "no clinical matrix"],
            }
        ],
        relation_scope=_key_claim_relation_scope(),
        uncertainty_map=[],
    )
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)

    decision = create_research_judgment_decision(kernel_dir, assessment, briefing)

    audit = decision["judgment_frame"]["claim_pressure"]["claim_design_alignment_audit"]
    assert assessment["decision"] == "accept"
    assert decision["decision"] == "open_question"
    assert decision["judgment_frame"]["primary_pressure"] == "open_question_pressure"
    assert decision["judgment_frame"]["claim_pressure"]["key_claim_touch_gate"]["passed"] is True
    assert audit["status"] == "overclaim_boundary"
    assert audit["passed"] is False
    assert audit["blocks_revision"] is True
    assert audit["negative_weight"] >= 0.45
    assert "translational_overclaim" in audit["overclaim_flags"]
    assert "prototype-level sensing feasibility" in audit["allowed_interpretations"]
    assert "disease diagnostic readiness" in audit["disallowed_interpretations"]
    assert "claim_design_overclaim_boundary" in decision["judgment_frame"]["blockers"]
    assert decision["judgment_frame"]["warrants"]["belief_revision_candidate"] is False
    assert decision["candidate_state_changes"][0]["target_layer"] == "question"
    assert read_jsonl(object_path(kernel_dir, "revisions")) == []


def test_research_judgment_decision_key_claim_not_measured_opens_question(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    modules = _methodology_modules()
    modules["source_provenance"]["read_level"] = "full_text"
    modules["trajectory_proposal"]["candidate_question"] = (
        "What evidence directly measures the key mechanism?"
    )
    briefing = _sample_secretary_briefing(
        **modules,
        source={
            "paper_id": "paper_x1",
            "title": "Mechanism-adjacent support without key measurement",
            "doi": "10.1000/x1",
            "abstract": "Full text retained for provenance.",
            "input_text_level": "full_text",
        },
        claim_map=[
            {
                "claim_id": "claim_01",
                "claim_text": "The mechanism is discussed as a possible explanation.",
                "claim_type": "mechanism",
                "evidence_basis": "author_interpretation",
                "matched_key_claim_ids": ["state_switching"],
            }
        ],
        evidence_map={
            "source_type": "paper",
            "evidence_strength_inputs": {
                "abstract_level_only": False,
                "reports_lod": False,
            },
        },
        relation_scope=_key_claim_relation_scope(),
        uncertainty_map=[],
    )
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)

    decision = create_research_judgment_decision(kernel_dir, assessment, briefing)

    gate = decision["judgment_frame"]["claim_pressure"]["key_claim_touch_gate"]
    assert decision["decision"] == "open_question"
    assert decision["judgment_frame"]["primary_pressure"] == "open_question_pressure"
    assert decision["judgment_frame"]["warrants"]["belief_revision_candidate"] is False
    assert gate["passed"] is False
    assert gate["missing_measured_key_claim_ids"] == ["state_switching"]
    assert "key_claim_not_measured_or_tested" in decision["judgment_frame"]["blockers"]
    assert decision["candidate_state_changes"][0]["target_layer"] == "question"
    assert read_jsonl(object_path(kernel_dir, "revisions")) == []


def test_research_judgment_decision_support_label_without_grounding_requests_read(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    briefing = _sample_secretary_briefing()
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)

    decision = create_research_judgment_decision(kernel_dir, assessment, briefing)

    assert assessment["decision"] == "partial_accept"
    assert decision["decision"] == "request_action_or_read"
    assert decision["judgment_frame"]["primary_pressure"] == "source_grounding"
    assert decision["judgment_frame"]["relation_signal"]["relation"] == "support"
    assert decision["judgment_frame"]["warrants"]["belief_revision_candidate"] is False
    assert decision["judgment_frame"]["warrants"]["action_or_read"] is True
    assert "judgment_frame_requires_read_or_action" in decision["rationale"]
    assert "admissible_directional_belief_signal" not in decision["rationale"]
    assert decision["candidate_state_changes"][0]["target_layer"] == "action"
    assert read_jsonl(object_path(kernel_dir, "revisions")) == []


def test_apply_support_without_grounding_creates_blind_human_review_request(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    briefing = _sample_secretary_briefing()
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)
    decision = create_research_judgment_decision(kernel_dir, assessment, briefing)

    result = apply_research_judgment_decision(kernel_dir, decision)
    health, issues = validate_v3_kernel(kernel_dir)
    request = read_jsonl(object_path(kernel_dir, "human_review_requests"))[0]

    assert issues == []
    assert result["applied"]["human_review_requests"] == [request["id"]]
    assert health["human_review_request_count"] == 1
    assert health["open_human_review_request_count"] == 1
    assert request["request_type"] == "full_text_review"
    assert request["target_state_layer"] == "belief"
    assert request["anti_anchoring"]["kernel_prediction_withheld_from_reviewer"] is True
    assert "kernel_context" not in request["reviewer_payload"]
    assert read_jsonl(object_path(kernel_dir, "revisions")) == []


def test_research_judgment_decision_requires_recorded_assessment(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    briefing = _sample_secretary_briefing()
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)

    with pytest.raises(V3KernelValidationError) as exc_info:
        create_research_judgment_decision(
            kernel_dir,
            {**assessment, "assessment_id": "missing"},
            briefing,
        )

    assert "unknown assessment" in str(exc_info.value)
    assert read_jsonl(object_path(kernel_dir, "decisions")) == []


def test_apply_research_judgment_decision_creates_action_records(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    briefing = _sample_secretary_briefing(
        relation_scope={
            "belief_id": "B-RFAU",
            "belief": "Gold nanoparticles intrinsically absorb RF.",
            "whole_belief_projection_candidate": "not_computed",
            "legacy_whole_belief_reader": {
                "votes": {"model_a": "support", "model_b": "challenge"},
                "kernel_relation": "contest",
            },
        },
        disagreement_map=[
            {
                "disagreement_id": "disagreement_01",
                "field": "legacy_whole_belief_relation",
                "suggested_resolution": "human_read",
            }
        ],
    )
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)
    decision = create_research_judgment_decision(kernel_dir, assessment, briefing)

    result = apply_research_judgment_decision(kernel_dir, decision)
    again = apply_research_judgment_decision(kernel_dir, decision)
    health, issues = validate_v3_kernel(kernel_dir)

    assert issues == []
    assert result["decision"] == "request_action_or_read"
    assert len(result["applied"]["actions"]) == 2
    assert len(result["applied"]["human_review_requests"]) == 1
    assert health["action_count"] == 2
    assert health["open_action_count"] == 2
    assert health["human_review_request_count"] == 1
    assert health["open_human_review_request_count"] == 1
    assert read_jsonl(object_path(kernel_dir, "revisions")) == []
    assert len(again["applied"]["actions"]) == 2
    assert len(again["applied"]["human_review_requests"]) == 1
    assert health["action_count"] == 2


def test_apply_research_judgment_decision_does_not_apply_belief_revision(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    modules = _methodology_modules()
    modules["source_provenance"]["read_level"] = "full_text"
    briefing = _sample_secretary_briefing(
        **modules,
        source={
            "paper_id": "paper_001",
            "title": "State-switching bioelectronic sensor",
            "doi": "10.1000/test",
            "abstract": "Abstract retained for provenance.",
            "input_text_level": "full_text",
        },
        evidence_map={
            "source_type": "paper",
            "evidence_strength_inputs": {
                "abstract_level_only": False,
                "reports_lod": False,
            },
        },
        claim_map=_measured_key_claim_map(),
        relation_scope=_key_claim_relation_scope(),
        uncertainty_map=[],
    )
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)
    decision = create_research_judgment_decision(kernel_dir, assessment, briefing)

    result = apply_research_judgment_decision(kernel_dir, decision)
    health, issues = validate_v3_kernel(kernel_dir)

    assert issues == []
    assert decision["decision"] == "propose_belief_revision"
    assert result["skipped"] == [
        {
            "target": "belief",
            "reason": "explicit_belief_revision_api_required",
        }
    ]
    assert health["revision_count"] == 0
    assert health["action_count"] == 0
    assert health["human_review_request_count"] == 0
    assert health["question_count"] == 0
    assert health["trajectory_count"] == 0


def test_apply_research_judgment_decision_opens_question(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    modules = _methodology_modules()
    modules["source_provenance"]["read_level"] = "full_text"
    modules["trajectory_proposal"]["candidate_question"] = (
        "What benchmark separates disease signal from noise?"
    )
    briefing = _sample_secretary_briefing(
        **modules,
        source={
            "paper_id": "paper_002",
            "title": "Ambiguous disease readout",
            "doi": "10.1000/question",
            "abstract": "Full text retained for provenance.",
            "input_text_level": "full_text",
        },
        evidence_map={
            "source_type": "paper",
            "evidence_strength_inputs": {
                "abstract_level_only": False,
                "reports_lod": False,
            },
        },
        relation_scope={
            "belief_id": "B-SENSE",
            "belief": "Recognition must switch a measurable state.",
            "whole_belief_projection_candidate": "not_computed",
            "legacy_whole_belief_reader": {
                "votes": {"model_a": "underdetermined", "model_b": "neutral"},
                "kernel_relation": "underdetermined",
            },
        },
        uncertainty_map=[],
    )
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)
    decision = create_research_judgment_decision(kernel_dir, assessment, briefing)

    result = apply_research_judgment_decision(kernel_dir, decision)
    health, issues = validate_v3_kernel(kernel_dir)
    question = read_jsonl(object_path(kernel_dir, "questions"))[0]

    assert issues == []
    assert decision["decision"] == "open_question"
    assert result["applied"]["questions"] == [question["id"]]
    assert health["question_count"] == 1
    assert health["question_revision_count"] == 1
    assert question["question"] == "What benchmark separates disease signal from noise?"
    assert question["source_decision_id"] == decision["id"]


def test_apply_research_judgment_decision_opens_trajectory(tmp_path):
    kernel_dir = tmp_path / "kernel" / "v3"
    modules = _methodology_modules()
    modules["source_provenance"]["read_level"] = "full_text"
    briefing = _sample_secretary_briefing(
        **modules,
        source={
            "paper_id": "paper_003",
            "title": "Trajectory candidate paper",
            "doi": "10.1000/trajectory",
            "abstract": "Full text retained for provenance.",
            "input_text_level": "full_text",
        },
        evidence_map={
            "source_type": "paper",
            "evidence_strength_inputs": {
                "abstract_level_only": False,
                "reports_lod": False,
            },
        },
        relation_scope={
            "belief_id": "B-EV",
            "belief": "EV signals must preserve disease context.",
            "whole_belief_projection_candidate": "not_computed",
            "legacy_whole_belief_reader": {
                "votes": {"model_a": "neutral", "model_b": "neutral"},
                "kernel_relation": "neutral",
            },
        },
        candidate_kernel_updates=[
            {
                "candidate_id": "candidate_traj",
                "target_layer": "trajectory",
                "candidate_change": (
                    "Shift EV sensing from collection efficiency toward preserved "
                    "disease-state readout."
                ),
                "risk": "medium",
                "commit_permission": "kernel_only",
            }
        ],
        uncertainty_map=[],
    )
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)
    decision = create_research_judgment_decision(kernel_dir, assessment, briefing)

    result = apply_research_judgment_decision(kernel_dir, decision)
    health, issues = validate_v3_kernel(kernel_dir)
    trajectory = read_jsonl(object_path(kernel_dir, "trajectories"))[0]

    assert issues == []
    assert decision["decision"] == "propose_trajectory_update"
    assert result["applied"]["trajectories"] == [trajectory["id"]]
    assert health["trajectory_count"] == 1
    assert health["trajectory_revision_count"] == 1
    assert "disease-state readout" in trajectory["statement"]
    assert trajectory["source_decision_id"] == decision["id"]


def test_cli_v3_assess_briefing_writes_assessment(tmp_path):
    from psil.cli import main

    kernel_dir = tmp_path / "kernel" / "v3"
    briefing_path = tmp_path / "briefing.json"
    output_path = tmp_path / "assessment.json"
    briefing_path.write_text(
        json.dumps(_sample_secretary_briefing(), ensure_ascii=False),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "v3-assess-briefing",
            str(briefing_path),
            "--path",
            str(kernel_dir),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "partial_accept:" in result.output
    assessment = json.loads(output_path.read_text(encoding="utf-8"))
    assert assessment["decision"] == "partial_accept"
    assert read_jsonl(object_path(kernel_dir, "assessments"))[0]["id"] == assessment["id"]


def test_cli_v3_decide_briefing_writes_decision(tmp_path):
    from psil.cli import main

    kernel_dir = tmp_path / "kernel" / "v3"
    briefing = _sample_secretary_briefing()
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)
    briefing_path = tmp_path / "briefing.json"
    assessment_path = tmp_path / "assessment.json"
    output_path = tmp_path / "decision.json"
    briefing_path.write_text(json.dumps(briefing, ensure_ascii=False), encoding="utf-8")
    assessment_path.write_text(json.dumps(assessment, ensure_ascii=False), encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "v3-decide-briefing",
            str(assessment_path),
            str(briefing_path),
            "--path",
            str(kernel_dir),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "request_action_or_read:" in result.output
    decision = json.loads(output_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "request_action_or_read"
    assert read_jsonl(object_path(kernel_dir, "decisions"))[0]["id"] == decision["id"]


def test_cli_v3_apply_decision_writes_application(tmp_path):
    from psil.cli import main

    kernel_dir = tmp_path / "kernel" / "v3"
    briefing = _sample_secretary_briefing(
        relation_scope={
            "belief_id": "B-RFAU",
            "belief": "Gold nanoparticles intrinsically absorb RF.",
            "whole_belief_projection_candidate": "not_computed",
            "legacy_whole_belief_reader": {
                "votes": {"model_a": "support", "model_b": "challenge"},
                "kernel_relation": "contest",
            },
        },
        disagreement_map=[
            {
                "disagreement_id": "disagreement_01",
                "field": "legacy_whole_belief_relation",
                "suggested_resolution": "human_read",
            }
        ],
    )
    assessment = create_kernel_intake_assessment(kernel_dir, briefing)
    decision = create_research_judgment_decision(kernel_dir, assessment, briefing)
    decision_path = tmp_path / "decision.json"
    output_path = tmp_path / "application.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False), encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "v3-apply-decision",
            str(decision_path),
            "--path",
            str(kernel_dir),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "applied request_action_or_read:" in result.output
    application = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(application["applied"]["actions"]) == 2
    assert len(read_jsonl(object_path(kernel_dir, "actions"))) == 2


def test_cli_v3_validate_does_not_import_ingest_stack(tmp_path):
    script = f"""
import builtins
from pathlib import Path

blocked = ("feedparser", "psil.ingest.orchestrator", "psil.ingest.rss")
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name in blocked or any(name.startswith(prefix + ".") for prefix in blocked):
        raise RuntimeError("blocked import: " + name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from click.testing import CliRunner
from psil.cli import main

path = Path({str(tmp_path / "kernel" / "v3")!r})
result = CliRunner().invoke(main, ["v3-validate", "--path", str(path)])
if result.exit_code != 0:
    raise SystemExit(result.output)
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_cli_seed_validate_and_export(tmp_path):
    from psil.cli import main

    kernel_dir = tmp_path / "kernel" / "v3"
    export_path = tmp_path / "kernel_state.json"
    runner = CliRunner()

    seed = runner.invoke(main, ["v3-seed-minimal", "--path", str(kernel_dir)])

    assert seed.exit_code == 0, seed.output
    assert "Seeded V3 kernel" in seed.output

    validate = runner.invoke(main, ["v3-validate", "--path", str(kernel_dir)])

    assert validate.exit_code == 0, validate.output
    assert "OK: V3 kernel valid" in validate.output

    export = runner.invoke(
        main,
        ["v3-export", "--path", str(kernel_dir), "--output", str(export_path)],
    )

    assert export.exit_code == 0, export.output
    state = json.loads(export_path.read_text(encoding="utf-8"))
    assert state["health"]["belief_count"] == 1
    assert state["health"]["constraint_count"] == 1
    assert state["beliefs"][0]["title"] == "ScholarHound V3 should be belief-centered"
    assert state["judgments"][0]["linked_constraints"] == [state["constraints"][0]["id"]]
