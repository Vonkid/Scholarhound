import importlib.util
from pathlib import Path


def load_tool():
    tool_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "audit_secretary_briefing.py"
    )
    spec = importlib.util.spec_from_file_location("audit_secretary_briefing", tool_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_item():
    return {
        "id": "paper_001",
        "belief_id": "B-RFAU",
        "belief": (
            "Radiofrequency fields heat gold nanoparticles through intrinsic absorption "
            "by the nanoparticles themselves, enabling targeted non-invasive tumor hyperthermia."
        ),
        "title": "Electrophoretic Mechanism of Au25 Heating in Radiofrequency Fields",
        "doi": "10.1021/example",
        "abstract": (
            "Gold nanoparticles in radiofrequency fields have been observed to heat. "
            "There is debate over the mechanism of heating. "
            "Au25 clusters are studied for mechanistic insights obtainable from precise synthetic control over charge and size. "
            "An electrophoretic mechanism can adequately account for the observed heat. "
            "This study connects theoretical and experimentally observed heating rates."
        ),
    }


def test_briefing_preserves_legacy_relation_without_kernel_projection():
    tool = load_tool()
    relation_row = {
        "id": "paper_001",
        "belief_id": "B-RFAU",
        "votes": {
            "deepseek": "support",
            "claude": "challenge",
            "gpt": "challenge",
            "llama": "challenge",
        },
        "kernel_relation": "contest",
    }
    atoms_by_belief = {
        "B-RFAU": {
            "atoms": [
                {
                    "atom_id": "intrinsic_rf_absorption_by_gold",
                    "claim": "Gold nanoparticles intrinsically absorb RF energy.",
                    "role": "mechanism",
                    "criticality": "required",
                }
            ]
        }
    }

    briefing = tool.build_briefing(sample_item(), relation_row, atoms_by_belief)

    assert briefing["schema_id"] == "audit_secretary_briefing_schema_v1"
    assert briefing["relation_scope"]["whole_belief_projection_candidate"] == "not_computed"
    assert briefing["relation_scope"]["legacy_whole_belief_reader"]["kernel_relation"] == "contest"
    assert briefing["relation_scope"]["reader_relation_by_atom"] == []
    assert briefing["disagreement_map"][0]["suggested_resolution"] == "human_read"
    assert briefing["attention_brief"]["attention_recommendation"] == "read"
    assert briefing["source_provenance"]["doi"] == "10.1021/example"
    assert briefing["object_grounding"]["object_text_alignment"] == "partial"
    assert briefing["confidence_dynamics"]["usable_for_kernel_confidence"] is False
    assert briefing["task_reliability_envelope"]["admissibility"] == "weak_signal"
    assert briefing["epistemic_task_probe"]["requires_epistemic_boundary_review"] is True
    assert briefing["semantic_audit"]["behavioral_risk"] in {"low", "medium"}
    assert briefing["calibration_status"]["should_downweight_confidence"] is True
    assert briefing["trajectory_proposal"]["expert_review_needed"] is True
    assert briefing["human_feedback_brief"]["human_control"] == "optional_uptake"
    assert briefing["knowledge_transfer"]["integration_policy"] in {
        "kernel_may_consider",
        "human_review_first",
    }
    assert briefing["personalization_boundary"]["truth_or_judgment_effect"] == "none"
    assert briefing["closed_loop_validation"]["kernel_commit_allowed"] is False
    assert briefing["cognitive_tooling"]["tool_status"] == "draft"
    assert briefing["prior_trace"]["prior_type"] == "project_priority"
    assert briefing["active_curriculum"]["expected_learning_gain"] == "high"
    assert all(
        candidate["commit_permission"] == "kernel_only"
        for candidate in briefing["candidate_kernel_updates"]
    )


def test_briefing_shape_validator_rejects_non_kernel_commit_permission(tmp_path):
    tool = load_tool()
    schema = tmp_path / "schema.json"
    schema.write_text(
        tool.json.dumps(
            {
                "required_top_level_fields": [
                    "schema_id",
                    "candidate_kernel_updates",
                    "relation_scope",
                ]
            }
        )
    )
    briefing = {
        "schema_id": "audit_secretary_briefing_schema_v1",
        "candidate_kernel_updates": [
            {"candidate_id": "bad", "commit_permission": "secretary_can_commit"}
        ],
        "relation_scope": {"whole_belief_projection_candidate": "not_computed"},
    }

    issues = tool.validate_briefing_shape(briefing, schema)

    assert any("non-kernel commit permission" in issue for issue in issues)


def test_controls_detection_does_not_treat_synthetic_control_as_experimental_control():
    tool = load_tool()
    evidence = tool.build_evidence_map(
        sample_item()["title"],
        sample_item()["abstract"],
    )

    assert evidence["controls_reported"] == "unclear"
    assert evidence["primary_or_secondary"] == "primary"
