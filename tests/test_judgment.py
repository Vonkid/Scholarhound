import os
import tempfile

from psil.judgment import (
    apply_kernel_revision,
    build_judgment_kernel_summary,
    generate_kernel_tasks,
    materialize_kernel_objects,
    materialize_kernel_tasks,
)
from psil.store.db import Database


def test_build_judgment_kernel_summary_compacts_memory_palace_state():
    papers = [{
        "doi": "10.1234/high",
        "title": "OECT RNA sensing paper needing a judgment commitment",
        "signal_tier": "IMPORTANT",
        "signal_score": 8,
        "llm_reasoning": '{"final_score":8.2}',
    }]
    concepts = [{
        "name": "oect biosensing",
        "appearances": 3,
        "missing_link": "Disease-relevant biomarker validation is still missing.",
    }]
    frameworks = [{
        "framework_name": "recognition-to-state transduction",
        "core_logic": "Molecular recognition matters only if it changes ionic-electronic state.",
        "compression_score": 8,
        "predictive_power": 7,
        "falsifiability": 6,
        "actionability": 6,
        "status": "candidate",
    }]
    constraints = [{
        "name": "binding must modulate channel state",
        "framework_name": "recognition-to-state transduction",
        "statement": "Binding events must be converted into a measurable channel-state change.",
        "prediction_power": 8,
        "confidence": 6,
        "status": "candidate",
    }]
    memory = {
        "beliefs": [{
            "item_type": "framework",
            "item_name": "disease context matters",
            "reason": "Approved after repeated organoid/EV readout papers.",
        }],
        "rejected": [],
        "contradictions": [{
            "item_type": "contradiction",
            "item_name": "same concept used as sensing claim and therapy claim",
            "reason": "Needs semantic drift resolution.",
        }],
        "decisions": [],
        "next_actions": [],
    }
    story_groups = [{
        "direction": "Molecular Recognition -> Bioelectronic Transduction",
        "evidence_count": 12,
        "nodes": [{
            "node_type": "next_question",
            "title": "Which validation would make it diagnostic?",
            "missing_link": "Disease benchmark is missing.",
            "next_move": "Pick a discriminating validation paper.",
        }],
    }]

    summary = build_judgment_kernel_summary(
        papers=papers,
        concepts=concepts,
        frameworks=frameworks,
        constraints=constraints,
        memory_summary=memory,
        story_groups=story_groups,
    )

    assert summary["name"] == "ScholarHound Judgment Kernel"
    assert "intelligent memory palace" in summary["definition"]
    assert summary["status"] == "unstable"
    assert summary["counts"]["beliefs"] == 1
    assert summary["counts"]["open_questions"] >= 1
    assert summary["pulse"]["top_question"] == "Which validation would make it diagnostic?"
    assert summary["memory_palace"]["active_beliefs"][0]["title"] == "disease context matters"
    assert summary["memory_palace"]["candidate_claims"][0]["type"] == "framework_claim"
    assert {p["type"] for p in summary["memory_palace"]["pressure_points"]} >= {
        "contradiction",
        "untested_constraint",
        "high_score_needs_judgment",
    }
    assert summary["memory_palace"]["next_moves"][0]["type"] == "resolve"


def test_materialize_and_revise_kernel_objects_without_llm():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()
    summary = build_judgment_kernel_summary(
        memory_summary={
            "beliefs": [{
                "item_type": "framework",
                "item_name": "disease context matters",
                "reason": "Repeated evidence preserves this claim.",
            }],
            "rejected": [],
            "contradictions": [],
            "decisions": [],
            "next_actions": [],
        },
        frameworks=[{
            "framework_name": "recognition-to-state transduction",
            "core_logic": "Binding must change device state.",
            "compression_score": 6,
            "predictive_power": 6,
            "falsifiability": 6,
            "actionability": 6,
        }],
    )

    sync = materialize_kernel_objects(db, summary)
    objects = db.get_kernel_objects()
    claim = next(obj for obj in objects if obj["object_type"] == "claim")

    assert sync["counts"]["belief"] == 1
    assert sync["counts"]["claim"] == 1

    committed = apply_kernel_revision(
        db,
        claim["object_key"],
        "commit",
        reason="This is now a working model.",
        actor="human",
    )
    challenged = apply_kernel_revision(
        db,
        claim["object_key"],
        "challenge",
        reason="New paper weakens the mechanism.",
        evidence_delta={"paper": "10.1234/challenge"},
    )
    unknown = apply_kernel_revision(db, claim["object_key"], "freeform-think")

    assert committed["ok"] is True
    assert committed["transition"]["status"] == ["candidate", "approved"]
    assert committed["transition"]["confidence"][1] > committed["transition"]["confidence"][0]
    assert challenged["transition"]["status"] == ["approved", "contested"]
    assert db.get_kernel_object(claim["object_key"])["status"] == "contested"
    assert unknown == {"ok": False, "error": "unknown_action", "action": "freeform-think"}
    assert [event["event_type"] for event in db.get_kernel_object_events(claim["object_key"])] == [
        "challenge",
        "commit",
    ]


def test_kernel_tasks_are_generated_from_objects_and_pressure_points():
    summary = build_judgment_kernel_summary(
        papers=[{
            "doi": "10.1234/high",
            "title": "High score paper without final commitment",
            "signal_score": 8,
            "llm_reasoning": '{"final_score":8.2}',
        }],
        constraints=[{
            "name": "binding must modulate channel state",
            "statement": "Binding must change the channel.",
            "prediction_power": 8,
            "confidence": 6,
            "status": "candidate",
        }],
        memory_summary={
            "beliefs": [],
            "rejected": [],
            "contradictions": [],
            "decisions": [],
            "next_actions": [],
        },
        kernel_objects=[{
            "object_key": "claim-recognition",
            "object_type": "claim",
            "title": "Recognition must change channel state",
            "statement": "Binding only matters if it changes device state.",
            "status": "candidate",
            "confidence": 6,
            "entrenchment": 2,
        }, {
            "object_key": "question-validation",
            "object_type": "open_question",
            "title": "Which validation would make this diagnostic?",
            "statement": "Needs a disease benchmark.",
            "status": "open",
            "confidence": 5,
            "entrenchment": 2,
        }],
    )

    tasks = generate_kernel_tasks(summary)

    assert tasks[0]["priority"] >= tasks[-1]["priority"]
    assert {task["task_type"] for task in tasks} >= {
        "commit_or_reject_object",
        "sharpen_open_question",
        "verify_pressure_constraint",
        "judge_high_score_paper",
    }


def test_kernel_task_priority_prefers_taste_aligned_questions_over_generic_pressure():
    summary = build_judgment_kernel_summary(
        constraints=[{
            "name": "output independent of trigger magnitude",
            "statement": "Generic constraint topology pressure point.",
            "prediction_power": 10,
            "confidence": 10,
            "status": "candidate",
        }],
        memory_summary={
            "beliefs": [],
            "rejected": [],
            "contradictions": [],
            "decisions": [],
            "next_actions": [],
        },
        kernel_objects=[{
            "object_key": "claim-rna-oect",
            "object_type": "claim",
            "title": "RNA RIBOTAC recognition must change OECT channel state",
            "statement": "RNA warhead binding only matters if it changes ionic-electronic state.",
            "status": "candidate",
            "confidence": 5,
            "entrenchment": 1,
        }],
    )

    tasks = generate_kernel_tasks(summary)
    top = tasks[0]
    generic = next(task for task in tasks if task["task_type"] == "verify_pressure_constraint")

    assert top["task_type"] == "commit_or_reject_object"
    assert top["object_key"] == "claim-rna-oect"
    assert top["priority"] > generic["priority"]
    assert top["metadata"]["priority_calibration"]["taste_score"] > 0
    assert generic["metadata"]["priority_calibration"]["base_priority"] > generic["priority"]


def test_kernel_task_queue_keeps_multiple_frontiers_visible():
    summary = build_judgment_kernel_summary(
        kernel_objects=[{
            "object_key": "question-rna-oect",
            "object_type": "open_question",
            "title": "Can RNA recognition switch an OECT bioelectronic state?",
            "statement": "RNA binding needs to modulate the ionic-electronic channel state.",
            "status": "open",
            "confidence": 9,
            "entrenchment": 3,
            "source_type": "open_question",
            "source_ref": "Molecular Recognition -> Bioelectronic Transduction",
        }, {
            "object_key": "question-diagnostic-validation",
            "object_type": "open_question",
            "title": "Which validation would make molecular recognition diagnostic?",
            "statement": "Disease benchmark validation is still missing.",
            "status": "open",
            "confidence": 9,
            "entrenchment": 3,
            "source_type": "open_question",
            "source_ref": "Molecular Recognition -> Bioelectronic Transduction",
        }, {
            "object_key": "question-organoid-ev",
            "object_type": "open_question",
            "title": "Can organoid and EV signals become disease-state readouts?",
            "statement": "Organoid EV signals need disease-state separation.",
            "status": "open",
            "confidence": 5,
            "entrenchment": 2,
            "source_type": "open_question",
            "source_ref": "Organoid / EV Disease-State Readouts",
        }, {
            "object_key": "question-photonic",
            "object_type": "open_question",
            "title": "Can programmable optical field states become a control layer?",
            "statement": "Photonic states need a decisive field-state control experiment.",
            "status": "open",
            "confidence": 5,
            "entrenchment": 2,
            "source_type": "open_question",
            "source_ref": "Nanophotonic Field Control",
        }, {
            "object_key": "question-adaptive",
            "object_type": "open_question",
            "title": "Can biointerfaces adapt to tissue force instead of tolerating it?",
            "statement": "Adaptive biointerfaces need a force-coupled experiment.",
            "status": "open",
            "confidence": 5,
            "entrenchment": 2,
            "source_type": "open_question",
            "source_ref": "Adaptive Living Interfaces",
        }],
    )

    tasks = generate_kernel_tasks(summary)
    frontiers = [task["metadata"]["frontier"] for task in tasks[:4]]

    assert len(set(frontiers)) == 4
    assert tasks[0]["metadata"]["frontier"] == "Molecular Recognition -> Bioelectronic Transduction"
    assert tasks[1]["metadata"]["frontier"] != tasks[0]["metadata"]["frontier"]


def test_materialize_kernel_tasks_persists_generated_queue():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()
    obj = db.upsert_kernel_object(
        object_type="claim",
        title="Recognition must change channel state",
        statement="Binding only matters if it changes device state.",
        status="candidate",
        confidence=6,
        entrenchment=2,
    )
    summary = build_judgment_kernel_summary(kernel_objects=[obj])

    result = materialize_kernel_tasks(db, summary)

    assert result["counts"]["commit_or_reject_object"] == 1
    assert db.get_kernel_tasks()[0]["object_key"] == obj["object_key"]
