from psil.serve import (
    app,
    api_dashboard,
    build_abstract_graph,
    build_toc_graph,
    build_trajectory_map,
    _is_displayable_framework,
    _parse_flat_digest_entries,
    _score_value,
)
from fastapi.testclient import TestClient
from collections import Counter
import hashlib
import json
import os
import tempfile
from pathlib import Path

import psil.serve as serve_module
from psil.store.db import Database
from psil.state_change import append_event, sample_event


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_frozen_primary_beliefs_and_review_have_separate_routes():
    client = TestClient(app)

    primary = client.get("/")
    beliefs = client.get("/beliefs")
    review = client.get("/review")

    assert primary.status_code == 200
    assert "Dashboard" in primary.text
    assert "Digest" in primary.text
    assert "Trajectory Map" in primary.text
    assert 'id="trajectory-logic-css"' in primary.text
    assert 'id="trajectory-logic-js"' in primary.text
    assert "Persistent belief kernel" not in primary.text
    assert "Claim-Abstract Review" not in primary.text
    assert beliefs.status_code == 200
    assert "Persistent belief kernel" in beliefs.text
    assert review.status_code == 200
    assert "Claim-Abstract Review" in review.text
    assert "no-store" in primary.headers["cache-control"]
    assert "no-store" in beliefs.headers["cache-control"]
    assert "no-store" in review.headers["cache-control"]


def test_belief_map_projects_v3_state_without_hidden_review_fields(
    monkeypatch,
    tmp_path,
):
    kernel_dir = tmp_path / "kernel" / "v3"
    belief_id = "belief_oect"
    evidence_id = "evidence_support"
    _write_jsonl(kernel_dir / "beliefs" / "beliefs.jsonl", [{
        "id": belief_id,
        "title": "OECT value is integration rather than raw sensitivity",
        "claim": "Integration is the stronger literature signal.",
        "domain": "OECT EV sensing",
        "status": "active",
        "confidence": 0.65,
        "entrenchment": 0.4,
        "linked_concepts": ["OECT", "EV"],
        "evidence_ids": [evidence_id],
        "contra_evidence_ids": [],
        "updated_at": "2026-07-17T00:00:00+00:00",
    }])
    _write_jsonl(kernel_dir / "evidence" / "evidence.jsonl", [{
        "id": evidence_id,
        "title": "Integrated OECT platform",
        "summary": "One device combines capture, sensing and stimulation.",
        "evidence_strength": "moderate",
        "source_type": "primary-research",
        "source_ref": "10.1234/oect",
        "supports_beliefs": [belief_id],
        "created_at": "2026-07-16T00:00:00+00:00",
    }])
    _write_jsonl(kernel_dir / "revisions" / "revisions.jsonl", [{
        "id": "revision_1",
        "belief_id": belief_id,
        "action": "strengthen",
        "reason": "Integrated evidence increased the literature signal.",
        "old_confidence": 0.5,
        "new_confidence": 0.65,
        "triggering_evidence_ids": [evidence_id],
        "created_at": "2026-07-17T00:00:00+00:00",
    }])
    _write_jsonl(
        kernel_dir / "human_review_requests" / "human_review_requests.jsonl",
        [{
            "id": "review_1",
            "status": "open",
            "priority": "high",
            "request_type": "direction",
            "reviewer_payload": {
                "question": "Does this source change the belief direction?",
                "belief_ref": "B-OECT",
            },
            "hidden_kernel_assessment": "must never be exposed",
            "created_at": "2026-07-17T00:00:00+00:00",
        }],
    )
    monkeypatch.setattr(serve_module, "_v3_kernel_dir", lambda: kernel_dir)
    client = TestClient(app)

    response = client.get("/api/kernel/v3/belief-map")
    data = response.json()

    assert response.status_code == 200
    assert data["schema_id"] == "scholarhound_belief_map_view_v1"
    assert data["semantics"] == "literature state, not probability of scientific truth"
    assert data["counts"] == {
        "beliefs": 1,
        "evidence": 1,
        "revisions": 1,
        "unresolved_evidence": 0,
        "open_review_requests": 1,
    }
    belief = data["beliefs"][0]
    assert belief["display_id"] == "B-OECT"
    assert belief["evidence_counts"]["support"] == 1
    assert belief["evidence"]["support"][0]["source_ref"] == "10.1234/oect"
    assert belief["revision_history"][0]["action"] == "strengthen"
    assert data["review_requests"][0]["question"].startswith("Does this source")
    assert "hidden_kernel_assessment" not in data["review_requests"][0]


def test_private_benchmark_material_is_not_required_by_public_distribution():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "kernel" / "v3" / "goldsets" / "human_feedback").exists()


def test_build_abstract_graph_prefers_causal_reasoning():
    row = {
        "abstract": "We develop a sensor and show robust detection.",
        "llm_reasoning": (
            '{"causal":{"question":"What is measured?",'
            '"constraint":"Low signal.",'
            '"input_state":"Weak ionic signal.",'
            '"transformation":"OECT amplification.",'
            '"output_state":"Readable current.",'
            '"outcome":"Enables low-power sensing."}}'
        ),
    }

    graph = build_abstract_graph(row)

    assert [node["label"] for node in graph["nodes"]] == [
        "Question",
        "Constraint",
        "Input State",
        "Transformation",
        "Output State",
        "Outcome",
    ]
    assert len(graph["edges"]) == 5


def test_build_abstract_graph_falls_back_to_abstract():
    row = {
        "abstract": (
            "Bioelectronic sensing needs stable interfaces. "
            "We develop an OECT platform using mixed ionic-electronic conduction. "
            "The platform shows multimodal integration."
        ),
        "llm_reasoning": "{}",
    }

    graph = build_abstract_graph(row)

    assert graph["nodes"]
    assert graph["nodes"][0]["label"] == "Context"
    assert any(node["label"] == "Finding" for node in graph["nodes"])


def test_build_toc_graph_normalizes_available_image():
    graph = build_toc_graph(" https://example.com/toc.png ", "digest")

    assert graph == {
        "status": "available",
        "image_url": "https://example.com/toc.png",
        "source": "digest",
    }


def test_build_toc_graph_reports_missing_image():
    graph = build_toc_graph("")

    assert graph == {"status": "missing", "image_url": "", "source": ""}


def test_build_trajectory_map_returns_public_summary_layer(monkeypatch):
    class Identity:
        current_core = []
        emerging_directions = []
        long_term_vision = ["molecular bioelectronics", "nanophotonic field control"]
        trajectory_influence_topics = []

    monkeypatch.setattr("psil.serve.load_identity", lambda: Identity())
    paper = {
        "doi": "10.1234/oect",
        "title": "OECT bioelectronic sensor for molecular recognition",
        "abstract": "Organic electrochemical transistors amplify molecular sensing.",
        "journal": "Test Journal",
        "signal_tier": "HIGH_PRIORITY",
        "signal_score": 8,
        "llm_reasoning": '{"final_score":8.1,"workflow":{"gap":"Needs disease validation."}}',
    }
    decoy_paper = {
        "doi": "10.5678/nir",
        "title": "NIR photoelectrochemical molecular biosensor using photothermal losses",
        "abstract": "A molecular bioelectronics-adjacent biosensor with photothermal conversion.",
        "journal": "Test Journal",
        "signal_tier": "WATCHLIST",
        "signal_score": 3,
        "llm_reasoning": '{"final_score":3.0}',
    }
    curated_rna_oect = {
        "doi": "10.1002/adma.202515338",
        "title": "Small molecule-driven organic electrochemical transistors for rapid, ultrasensitive and amplification-free detection of RNA biomarkers",
        "abstract": "A local curated seed source for RNA biomarker sensing with organic electrochemical transistors.",
        "journal": "Advanced Materials",
        "signal_tier": "CURATED_LIBRARY",
        "signal_score": 0,
        "llm_reasoning": '{"origin":"local_manifest","bucket":"04_EV_sensing_ready_surface_and_inner_markers"}',
    }
    photochemistry_paper = {
        "doi": "10.9999/polariton",
        "title": "Nanophotonics-enabled polariton control of photochemistry",
        "abstract": "Local optical states redirect photochemical reactions.",
        "journal": "Test Journal",
        "signal_tier": "IMPORTANT",
        "signal_score": 7,
        "llm_reasoning": '{"final_score":7.0}',
    }
    concept = {
        "name": "oect biosensing",
        "appearances": 3,
        "status": "gaining momentum",
        "missing_link": "Disease-relevant validation is still missing.",
    }

    data = build_trajectory_map(
        [paper, decoy_paper, curated_rna_oect, photochemistry_paper],
        [concept],
        {"10.1234/oect": "2026-06-09"},
        [],
        [{
            "previous_assumption": "Huckel aromatic units are required for molecular design rules.",
            "new_assumption": "Anti-Huckel aromatic units can red-shift narrowband emitters.",
            "delta": "unrelated molecular materials shift",
        }, {
            "previous_assumption": "NIR biosensors need external bias.",
            "new_assumption": "Photothermal losses can be harvested for self-powered biosensing.",
            "delta": "unrelated nanophotonic shift",
        }],
    )

    trajectory = data["trajectories"][0]
    assert data["center"]["name"] == "ScholarHound"
    assert data["center"]["subtitle"] == "Research-question evolution over the private judgment kernel"
    assert data["story_direction"] == "Molecular Recognition -> Bioelectronic Transduction"
    assert len(data["story_groups"]) == 4
    assert {group["direction"] for group in data["story_groups"]} == {
        "Molecular Recognition -> Bioelectronic Transduction",
        "Organoid / EV Disease-State Readouts",
        "Adaptive Living Interfaces",
        "Nanophotonic Field Control",
    }
    assert [node["type_label"] for node in data["story_nodes"]] == [
        "Core Question",
        "Working Hypothesis",
        "Turning Point",
        "Conceptual Shift",
        "Current Model",
        "Next Question",
    ]
    titles = [node["title"] for node in data["story_nodes"]]
    assert titles[0] == "Can molecular recognition switch a bioelectronic state?"
    assert titles[1] == "Binding only matters if it changes ionic-electronic state."
    assert "OECT bioelectronic sensor" in titles[2]
    assert titles[3] == "From collecting Molecular Recognition -> Bioelectronic Transduction papers to testing a model-changing mechanism."
    question_summary = data["story_nodes"][0]["summary"]
    assert "evidence points" in question_summary
    assert "confidence percentage" in question_summary
    assert "public graph" not in question_summary.lower()
    assert "Observation" not in " ".join(titles)
    assert data["story_nodes"][1]["related_trajectories"] == ["Molecular Bioelectronics"]
    assert "10.1002/adma.202515338" in [paper["doi"] for paper in data["story_nodes"][1]["papers"]]
    assert data["story_edges"][0] == {"source": "story-question", "target": "story-working-hypothesis", "type": "evolves_to"}
    assert trajectory["name"] == "Molecular Recognition -> Bioelectronic Transduction"
    assert trajectory["status"] == "rising"
    assert trajectory["papers"][0]["doi"] == "10.1234/oect"
    assert len(data["topic_trajectories"]) == 2
    assert "constraints" not in trajectory
    assert "framework" not in trajectory


def test_parse_flat_digest_entries_strips_markdown_and_keeps_metadata():
    body = """- **Artificial Gap Junctional Channel** — *Angewandte Chemie*
  DOI: [10.1002/anie.9685133](https://doi.org/10.1002/anie.9685133)
  Scores: R:2/10 N:6/10 B:4/10 T:3/10 → 3.1/10
  Why: Demonstrates a synthetic approach to functional channels.
  Action: Skip
"""
    papers = _parse_flat_digest_entries(
        body,
        "WATCHLIST",
        {"10.1002/anie.9685133": {"abstract": "", "toc_image_url": "https://example.com/toc.png"}},
    )

    assert len(papers) == 1
    paper = papers[0]
    assert paper["title"] == "Artificial Gap Junctional Channel"
    assert paper["journal"] == "Angewandte Chemie"
    assert paper["doi"] == "10.1002/anie.9685133"
    assert paper["scores"]["final"] == 3.1
    assert paper["why"] == "Demonstrates a synthetic approach to functional channels."
    assert paper["action"] == "Skip"
    assert paper["toc_graph"]["status"] == "available"


def test_trajectory_map_endpoint_reuses_cache(monkeypatch):
    calls = {"build": 0}

    class FakeDb:
        def get_all_papers(self):
            return []

        def get_concept_momentum(self, min_appearances=1):
            return []

        def get_trajectories(self):
            return []

        def get_deltas(self):
            return []

    def fake_build(*args, **kwargs):
        calls["build"] += 1
        return {
            "center": {"name": "ScholarHound"},
            "story_groups": [],
            "story_nodes": [],
            "story_edges": [],
            "trajectories": [],
            "topic_trajectories": [],
            "edges": [],
        }

    serve_module._TRAJECTORY_MAP_CACHE["key"] = None
    serve_module._TRAJECTORY_MAP_CACHE["data"] = None
    monkeypatch.setattr("psil.serve.get_db", lambda: FakeDb())
    monkeypatch.setattr("psil.serve.list_digests", lambda: [])
    monkeypatch.setattr("psil.serve._digest_paper_dates", lambda digests: {})
    monkeypatch.setattr("psil.serve._trajectory_map_cache_key", lambda digests: ("stable",))
    monkeypatch.setattr("psil.serve.build_trajectory_map", fake_build)
    client = TestClient(app)

    assert client.get("/api/trajectory-map").status_code == 200
    assert client.get("/api/trajectory-map").status_code == 200
    assert calls["build"] == 1


def test_dashboard_tier_lists_are_top_score_preview(monkeypatch):
    important_rows = [
        {
            "doi": f"10.1234/important-{idx}",
            "title": f"Important paper {idx}",
            "journal": "Test Journal",
            "signal_tier": "IMPORTANT",
            "signal_score": score,
            "llm_reasoning": f'{{"final_score":{score}}}',
            "ingested_at": f"2026-06-0{idx}",
        }
        for idx, score in enumerate([6.1, 8.4, 7.0, 6.8, 7.9, 5.5, 7.2], start=1)
    ]

    class FakeDb:
        def get_all_papers(self):
            return important_rows

        def get_concept_momentum(self, min_appearances=1):
            return []

        def get_local_sources(self):
            return []

    monkeypatch.setattr("psil.serve.get_db", lambda: FakeDb())
    monkeypatch.setattr("psil.serve.list_digests", lambda: [])

    data = api_dashboard()
    important = data["tier_lists"]["IMPORTANT"]

    assert data["important"] == 7
    assert "kernel_pulse" not in data
    assert len(important) == 5
    assert [paper["scores"]["final"] for paper in important] == [8.4, 7.9, 7.2, 7.0, 6.8]


def test_state_changes_endpoint_returns_validated_kernel_ledger(monkeypatch, tmp_path):
    log_path = tmp_path / "kernel" / "state_changes.jsonl"
    stored = append_event(log_path, sample_event())
    monkeypatch.setattr("psil.serve._state_change_log_path", lambda: log_path)
    client = TestClient(app)

    response = client.get("/api/state-changes")
    data = response.json()

    assert response.status_code == 200
    assert data["ok"] is True
    assert data["event_count"] == 1
    assert data["latest_event"]["event_id"] == stored["event_id"]
    assert data["state_counts"]["evidence"]["strengthened"] == 1
    assert data["action_queue"][0]["status"] == "verify"
    assert data["parser_boundary"]["source_of_truth"] == "kernel/state_changes.jsonl"


def test_state_changes_endpoint_reports_validation_errors(monkeypatch, tmp_path):
    log_path = tmp_path / "state_changes.jsonl"
    log_path.write_text('{"source": "bad"}\n', encoding="utf-8")
    monkeypatch.setattr("psil.serve._state_change_log_path", lambda: log_path)
    client = TestClient(app)

    data = client.get("/api/state-changes").json()

    assert data["ok"] is False
    assert data["event_count"] == 0
    assert data["issues"]
    assert data["parser_boundary"]["acceptance_layer"] == "psil.state_change validator"


def test_judgment_kernel_endpoint_returns_memory_palace(monkeypatch):
    class FakeDb:
        def get_all_papers(self):
            return []

        def get_concept_momentum(self, min_appearances=1):
            return []

        def get_frameworks(self):
            return []

        def get_constraints(self):
            return []

        def get_deltas(self):
            return []

        def get_verifications(self):
            return []

        def get_experiments(self):
            return []

        def get_trajectories(self):
            return []

        def get_memory_summary(self):
            return {
                "beliefs": [{
                    "item_type": "framework",
                    "item_name": "disease context matters",
                    "reason": "Approved by user.",
                }],
                "rejected": [],
                "contradictions": [],
                "decisions": [],
                "next_actions": [],
            }

    monkeypatch.setattr("psil.serve.get_db", lambda: FakeDb())
    monkeypatch.setattr("psil.serve.list_digests", lambda: [])
    client = TestClient(app)

    response = client.get("/api/judgment-kernel")
    data = response.json()

    assert response.status_code == 200
    assert data["name"] == "ScholarHound Judgment Kernel"
    assert data["pulse"]["beliefs"] == 1
    assert data["memory_palace"]["active_beliefs"][0]["title"] == "disease context matters"


def test_private_kernel_object_api_creates_and_revises_objects(monkeypatch):
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()
    monkeypatch.setattr("psil.serve.get_db", lambda: db)
    client = TestClient(app)

    created = client.post("/api/kernel/objects", json={
        "object_type": "claim",
        "title": "Recognition must change channel state",
        "statement": "Binding is only useful if it changes an ionic-electronic state.",
        "confidence": 4,
        "entrenchment": 2,
    }).json()
    object_key = created["object"]["object_key"]

    revised = client.post(f"/api/kernel/objects/{object_key}/revise", json={
        "action": "commit",
        "reason": "Commit as a working model.",
    }).json()
    objects = client.get("/api/kernel/objects").json()["objects"]

    assert created["ok"] is True
    assert revised["transition"]["status"] == ["candidate", "approved"]
    assert objects[0]["object_key"] == object_key
    assert objects[0]["status"] == "approved"


def test_private_kernel_task_api_syncs_and_updates_tasks(monkeypatch):
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()
    db.upsert_kernel_object(
        object_type="claim",
        title="Recognition must change channel state",
        statement="Binding only matters if it changes device state.",
        status="candidate",
        confidence=6,
        entrenchment=2,
    )
    monkeypatch.setattr("psil.serve.get_db", lambda: db)
    monkeypatch.setattr("psil.serve.list_digests", lambda: [])
    client = TestClient(app)

    sync = client.post("/api/kernel/tasks/sync").json()
    tasks = client.get("/api/kernel/tasks").json()["tasks"]
    task_key = tasks[0]["task_key"]
    revised = client.post(f"/api/kernel/tasks/{task_key}/status", json={
        "status": "done",
        "reason": "Handled in kernel review.",
    }).json()

    assert sync["ok"] is True
    assert sync["materialized"]["counts"]["commit_or_reject_object"] == 1
    assert tasks[0]["task_type"] == "commit_or_reject_object"
    assert revised["task"]["status"] == "done"


def test_private_v3_contested_evidence_endpoint(monkeypatch, tmp_path):
    from psil.v3_kernel import (
        create_belief,
        create_evidence,
        create_evidence_from_parse_candidates,
        revise_belief,
    )

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
        },
        reason="Initial support.",
        evidence_ids=[seed_evidence["id"]],
    )
    contested = create_evidence_from_parse_candidates(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Ambiguous paper",
            "source_ref": "doi:10.0/ambiguous",
            "summary": "Direction is unstable across parser candidates.",
        },
        belief_id=belief["id"],
        parse_candidates=[
            {"model": "parser_a", "relation": "support"},
            {"model": "parser_b", "relation": "challenge"},
        ],
    )
    revise_belief(
        kernel_dir,
        belief_id=belief["id"],
        evidence_ids=[contested["id"]],
        action="contest",
        reason="Direction is unstable.",
    )
    monkeypatch.setattr("psil.serve._v3_kernel_dir", lambda: kernel_dir)
    client = TestClient(app)

    response = client.get("/api/kernel/v3/contested-evidence")
    data = response.json()

    assert response.status_code == 200
    assert data["count"] == 1
    assert data["items"][0]["evidence_id"] == contested["id"]
    assert data["items"][0]["relation_provenance"]["contested"] is True


def test_private_v3_pending_evidence_endpoint_and_reclassify(monkeypatch, tmp_path):
    from psil.v3_kernel import (
        create_belief,
        create_evidence,
        create_evidence_from_parse_candidates,
        revise_belief,
    )

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
        },
        reason="Initial support.",
        evidence_ids=[seed_evidence["id"]],
    )
    pending = create_evidence_from_parse_candidates(
        kernel_dir,
        {
            "source_type": "paper",
            "title": "Incomplete paper",
            "source_ref": "doi:10.0/incomplete",
            "summary": "Direction is incomplete but not conflicting.",
        },
        belief_id=belief["id"],
        parse_candidates=[
            {"model": "parser_a", "relation": "support"},
            {"model": "parser_b", "relation": "unclear"},
        ],
    )
    revise_belief(
        kernel_dir,
        belief_id=belief["id"],
        evidence_ids=[pending["id"]],
        action="update",
        reason="Direction needs more evidence.",
    )
    monkeypatch.setattr("psil.serve._v3_kernel_dir", lambda: kernel_dir)
    client = TestClient(app)

    pending_response = client.get("/api/kernel/v3/pending-evidence")
    pending_data = pending_response.json()

    assert pending_response.status_code == 200
    assert pending_data["count"] == 1
    assert pending_data["items"][0]["evidence_id"] == pending["id"]
    assert pending_data["items"][0]["relation_provenance"]["needs_more_evidence"] is True

    reclassify_response = client.post(
        "/api/kernel/v3/pending-evidence/reclassify",
        json={
            "belief_id": belief["id"],
            "evidence_id": pending["id"],
            "relation": "support",
            "reason": "New evidence resolves pending direction.",
        },
    )
    reclassify_data = reclassify_response.json()

    assert reclassify_response.status_code == 200
    assert reclassify_data["revision"]["action"] == "strengthen"
    assert pending["id"] in reclassify_data["belief"]["evidence_ids"]
    assert pending["id"] not in reclassify_data["belief"]["pending_evidence_ids"]
    assert client.get("/api/kernel/v3/pending-evidence").json()["count"] == 0


def test_private_kernel_task_api_returns_frontier_diverse_queue(monkeypatch):
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()
    objects = [{
        "title": "Can RNA recognition switch an OECT bioelectronic state?",
        "statement": "RNA binding needs to modulate the ionic-electronic channel state.",
        "confidence": 9,
        "entrenchment": 3,
        "source_ref": "Molecular Recognition -> Bioelectronic Transduction",
    }, {
        "title": "Which validation would make molecular recognition diagnostic?",
        "statement": "Disease benchmark validation is still missing.",
        "confidence": 9,
        "entrenchment": 3,
        "source_ref": "Molecular Recognition -> Bioelectronic Transduction",
    }, {
        "title": "Can organoid and EV signals become disease-state readouts?",
        "statement": "Organoid EV signals need disease-state separation.",
        "confidence": 5,
        "entrenchment": 2,
        "source_ref": "Organoid / EV Disease-State Readouts",
    }, {
        "title": "Can programmable optical field states become a control layer?",
        "statement": "Photonic states need a decisive field-state control experiment.",
        "confidence": 5,
        "entrenchment": 2,
        "source_ref": "Nanophotonic Field Control",
    }, {
        "title": "Can biointerfaces adapt to tissue force instead of tolerating it?",
        "statement": "Adaptive biointerfaces need a force-coupled experiment.",
        "confidence": 5,
        "entrenchment": 2,
        "source_ref": "Adaptive Living Interfaces",
    }]
    for obj in objects:
        db.upsert_kernel_object(
            object_type="open_question",
            title=obj["title"],
            statement=obj["statement"],
            status="open",
            confidence=obj["confidence"],
            entrenchment=obj["entrenchment"],
            source_type="open_question",
            source_ref=obj["source_ref"],
        )
    monkeypatch.setattr("psil.serve.get_db", lambda: db)
    monkeypatch.setattr("psil.serve.list_digests", lambda: [])
    client = TestClient(app)

    client.post("/api/kernel/tasks/sync")
    tasks = client.get("/api/kernel/tasks?limit=4").json()["tasks"]
    frontiers = [json.loads(task["metadata"])["frontier"] for task in tasks]

    assert len(set(frontiers)) == 4
    assert frontiers[0] == "Molecular Recognition -> Bioelectronic Transduction"


def test_score_value_handles_detailed_score_labels():
    assert _score_value("- **Trajectory Influence:** 6/10") == 6
    assert _score_value("- **Concept Support:** 6/10") == 6


def test_is_displayable_framework_filters_empty_zero_score_records():
    assert not _is_displayable_framework({
        "framework_name": "empty framework",
        "description": "",
        "core_logic": "",
        "worldview_shift": "",
        "compression_score": 0,
        "novelty_score": 0,
    })

    assert _is_displayable_framework({
        "framework_name": "threshold framework",
        "description": "A real framework.",
        "compression_score": 0,
        "novelty_score": 0,
    })


def test_www_public_host_redirects_to_access_protected_host():
    client = TestClient(app, follow_redirects=False)

    response = client.get(
        "/api/constraint-radar?x=1",
        headers={"host": "www.scholarhound.academy"},
    )

    assert response.status_code == 307
    assert response.headers["location"] == "https://scholarhound.academy/api/constraint-radar?x=1"


def test_benchmark_session_serves_blind_packet(monkeypatch, tmp_path):
    packet_path = tmp_path / "g2_packet_v3.json"
    selection_path = tmp_path / "g2_packet_v3.selection_log.json"
    feedback_path = tmp_path / "feedback.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    packet_path.write_text(json.dumps([
            {
                "id": "item_001",
                "item_type": "error",
                "belief_id": "B-OECT",
                "belief": "OECT value is integration.",
            "title": "OECT <sub>8</sub> integration paper",
            "doi": "10.1234/oect-test",
            "journal": "Test Journal",
            "pub_date": "2026-06-13",
            "abstract": "This paper reports an integrated sensing platform.",
            "kernel_relation": "support",
            "votes": {"model": "support"},
        }
    ]), encoding="utf-8")
    selection_path.write_text(json.dumps({"base_seed": 20260612}), encoding="utf-8")
    monkeypatch.setattr(serve_module, "BENCHMARK_PACKET_PATH", packet_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_SELECTION_LOG_PATH", selection_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_FEEDBACK_PATH", feedback_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_PROGRESS_PATH", progress_path)
    client = TestClient(app)

    response = client.get(
        "/api/benchmark/session",
        headers={"Cf-Access-Authenticated-User-Email": "expert@example.org"},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"].startswith("no-store")
    data = response.json()
    assert data["session"]["mode"] == "blind_human_feedback"
    assert data["session"]["kernel_prediction_visibility"] == "withheld"
    assert data["session"]["packet"] == "g2_packet_v3"
    assert data["session"]["seed"] == 20260612
    assert data["session"]["packet_sha256"]
    assert data["session"]["presentation_order"] == "reviewer_deterministic_shuffle_v1"
    assert data["reviewer"]["reviewer"] == "expert@example.org"
    assert data["reviewer"]["auth_source"] == "cloudflare_access"
    assert data["items"][0]["id"] == "item_001"
    assert data["items"][0]["title"] == "OECT 8 integration paper"
    assert data["items"][0]["doi"] == "10.1234/oect-test"
    assert data["items"][0]["journal"] == "Test Journal"
    assert data["items"][0]["pub_date"] == "2026-06-13"
    assert "kernel_relation" not in data["items"][0]
    assert "votes" not in data["items"][0]
    assert "item_type" not in data["items"][0]


def test_benchmark_session_and_feedback_can_select_calibration_packet(
    monkeypatch,
    tmp_path,
):
    packet_path = tmp_path / "g2_packet_v3.json"
    selection_path = tmp_path / "g2_packet_v3.selection_log.json"
    calibration_path = tmp_path / "g2_calibration_24_v1.json"
    calibration_selection_path = tmp_path / "g2_calibration_24_v1.selection_log.json"
    feedback_path = tmp_path / "feedback.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    packet_path.write_text(json.dumps([{
        "id": "full_001",
        "belief_id": "B-OECT",
        "belief": "Full packet belief.",
        "title": "Full packet paper",
        "abstract": "Full abstract.",
    }]), encoding="utf-8")
    selection_path.write_text(json.dumps({"base_seed": 20260612}), encoding="utf-8")
    calibration_path.write_text(json.dumps([
        {
            "id": "cal_001",
            "belief_id": "B-SENSE",
            "belief": "Calibration packet belief.",
            "title": "Calibration sensing paper",
            "abstract": "Calibration abstract one.",
        },
        {
            "id": "cal_002",
            "belief_id": "B-EV",
            "belief": "Calibration packet belief two.",
            "title": "Calibration EV paper",
            "abstract": "Calibration abstract two.",
        },
    ]), encoding="utf-8")
    calibration_selection_path.write_text(
        json.dumps({"base_seed": 20260626}),
        encoding="utf-8",
    )
    monkeypatch.setattr(serve_module, "BENCHMARK_PACKET_PATH", packet_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_SELECTION_LOG_PATH", selection_path)
    monkeypatch.setattr(
        serve_module,
        "BENCHMARK_CALIBRATION_PACKET_PATH",
        calibration_path,
    )
    monkeypatch.setattr(
        serve_module,
        "BENCHMARK_CALIBRATION_SELECTION_LOG_PATH",
        calibration_selection_path,
    )
    monkeypatch.setattr(serve_module, "BENCHMARK_FEEDBACK_PATH", feedback_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_PROGRESS_PATH", progress_path)
    client = TestClient(app)
    headers = {"Cf-Access-Authenticated-User-Email": "expert@example.org"}

    default_response = client.get("/api/benchmark/session", headers=headers)
    calibration_response = client.get(
        "/api/benchmark/session?packet=calibration_24",
        headers=headers,
    )

    assert default_response.status_code == 200
    assert default_response.json()["session"]["packet"] == "g2_packet_v3"
    assert calibration_response.status_code == 200
    calibration_data = calibration_response.json()
    assert calibration_data["session"]["packet_key"] == "calibration_24"
    assert calibration_data["session"]["packet"] == "g2_calibration_24_v1"
    assert calibration_data["session"]["packet_label"] == "Review set A"
    assert calibration_data["session"]["item_count"] == 2
    assert calibration_data["session"]["seed"] == 20260626
    item_id = calibration_data["items"][0]["id"]

    cursor = client.post("/api/benchmark/progress", json={
        "packet_key": "calibration_24",
        "event_type": "cursor",
        "item_id": item_id,
        "session_run_id": "run-calibration",
    }, headers=headers)
    feedback = client.post("/api/benchmark/feedback", json={
        "packet_key": "calibration_24",
        "item_id": item_id,
        "relation": "support",
        "confidence": "medium",
        "reason": "Calibration packet signal.",
        "flags": [],
        "session_run_id": "run-calibration",
    }, headers=headers)

    assert cursor.status_code == 200
    assert feedback.status_code == 200
    feedback_record = json.loads(
        feedback_path.read_text(encoding="utf-8").splitlines()[0]
    )
    progress_records = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert feedback_record["packet"] == "g2_calibration_24_v1"
    assert feedback_record["packet_sha256"] == serve_module._benchmark_packet_sha256(
        "calibration_24"
    )
    assert {record["packet"] for record in progress_records} == {
        "g2_calibration_24_v1"
    }


def test_benchmark_feedback_appends_human_signal(monkeypatch, tmp_path):
    packet_path = tmp_path / "g2_packet_v3.json"
    selection_path = tmp_path / "g2_packet_v3.selection_log.json"
    feedback_path = tmp_path / "feedback.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    packet_path.write_text(json.dumps([
        {
            "id": "item_001",
            "item_type": "error",
            "belief_id": "B-OECT",
            "belief": "OECT value is integration.",
            "title": "OECT integration paper",
            "abstract": "This paper reports an integrated sensing platform.",
        }
    ]), encoding="utf-8")
    selection_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(serve_module, "BENCHMARK_PACKET_PATH", packet_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_SELECTION_LOG_PATH", selection_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_FEEDBACK_PATH", feedback_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_PROGRESS_PATH", progress_path)
    client = TestClient(app)

    response = client.post("/api/benchmark/feedback", json={
        "item_id": "item_001",
        "relation": "support",
        "confidence": "high",
        "assertions": "The claim asserts integration.",
        "covered": "Sentence 0 states integration.",
        "gap": "No missing assertion.",
        "reason": "The paper directly supports integration.",
        "expertise": "bioelectronics",
        "flags": ["abstract_insufficient"],
        "elapsed_ms": 1234,
        "client_started_at": "2026-06-14T01:02:03Z",
        "session_run_id": "run-test-001",
    }, headers={"Cf-Access-Authenticated-User-Email": "expert-a@example.org"})

    assert response.status_code == 200
    assert response.headers["cache-control"].startswith("no-store")
    payload = response.json()
    assert payload["ok"] is True
    records = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["schema_id"] == "human_benchmark_feedback_v2"
    assert records[0]["packet"] == "g2_packet_v3"
    assert records[0]["packet_sha256"]
    assert records[0]["item_id"] == "item_001"
    assert records[0]["item_type"] == "error"
    assert records[0]["item_position"] == 1
    assert records[0]["relation"] == "support"
    assert records[0]["confidence"] == "high"
    assert records[0]["assertions"] == "The claim asserts integration."
    assert records[0]["covered"] == "Sentence 0 states integration."
    assert records[0]["gap"] == "No missing assertion."
    assert records[0]["annotator_id"] == records[0]["reviewer_id"]
    assert records[0]["timestamp"] == records[0]["created_at"]
    assert records[0]["reviewer"] == "expert-a@example.org"
    assert records[0]["reviewer_id"].startswith("reviewer_")
    assert records[0]["auth_source"] == "cloudflare_access"
    assert records[0]["session_run_id"] == "run-test-001"
    assert records[0]["kernel_prediction_visibility"] == "withheld"
    progress_records = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert progress_records[0]["schema_id"] == "human_benchmark_progress_event_v1"
    assert progress_records[0]["event_type"] == "feedback_recorded"
    assert progress_records[0]["processed_item_count"] == 1


def test_benchmark_admin_test_account_is_segregated_and_excluded(
    monkeypatch,
    tmp_path,
):
    packet_path = tmp_path / "g2_packet_v4.json"
    selection_path = tmp_path / "g2_packet_v4.selection_log.json"
    feedback_path = tmp_path / "feedback.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    test_feedback_path = tmp_path / "test_feedback.jsonl"
    test_progress_path = tmp_path / "test_progress.jsonl"
    reviewer_policy_path = tmp_path / "reviewer_policy.json"
    admin_email = "admin@example.org"
    admin_reviewer_id = serve_module._reviewer_id(admin_email)
    packet_path.write_text(json.dumps([{
        "id": "item_001",
        "belief_id": "B-OECT",
        "belief": "OECT value is integration.",
        "title": "OECT integration paper",
        "abstract": "This paper reports an integrated sensing platform.",
    }]), encoding="utf-8")
    selection_path.write_text("{}", encoding="utf-8")
    reviewer_policy_path.write_text(json.dumps({
        "schema_id": "human_benchmark_reviewer_policy_v1",
        "reviewers": [{
            "identity": admin_email,
            "reviewer_id": admin_reviewer_id,
            "role": "administrator_test",
            "benchmark_eligible": False,
            "exclusion_reason": "Deployment testing only.",
        }],
    }), encoding="utf-8")
    feedback_path.write_text(json.dumps({
        "packet": "g2_packet_v4",
        "item_id": "legacy_item",
        "relation": "neutral",
        "reviewer_id": admin_reviewer_id,
        "reviewer": admin_email,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(serve_module, "BENCHMARK_PACKET_PATH", packet_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_SELECTION_LOG_PATH", selection_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_FEEDBACK_PATH", feedback_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_PROGRESS_PATH", progress_path)
    monkeypatch.setattr(
        serve_module,
        "BENCHMARK_TEST_FEEDBACK_PATH",
        test_feedback_path,
    )
    monkeypatch.setattr(
        serve_module,
        "BENCHMARK_TEST_PROGRESS_PATH",
        test_progress_path,
    )
    monkeypatch.setattr(
        serve_module,
        "BENCHMARK_REVIEWER_POLICY_PATH",
        reviewer_policy_path,
    )
    client = TestClient(app)
    headers = {"Cf-Access-Authenticated-User-Email": admin_email}

    auth = client.get("/api/benchmark/auth", headers=headers)
    response = client.post("/api/benchmark/feedback", json={
        "item_id": "item_001",
        "relation": "support",
        "confidence": "medium",
        "reason": "Administrator workflow test.",
        "flags": [],
        "session_run_id": "admin-test-run",
    }, headers=headers)

    assert auth.status_code == 200
    assert auth.json()["reviewer"]["reviewer_role"] == "administrator_test"
    assert auth.json()["reviewer"]["benchmark_eligible"] is False
    assert response.status_code == 200
    assert len(feedback_path.read_text(encoding="utf-8").splitlines()) == 1
    test_record = json.loads(
        test_feedback_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert test_record["reviewer_role"] == "administrator_test"
    assert test_record["benchmark_eligible"] is False
    assert test_progress_path.exists()
    assert serve_module._read_benchmark_eligible_feedback_records() == []


def test_benchmark_api_requires_authenticated_reviewer(monkeypatch, tmp_path):
    packet_path = tmp_path / "g2_packet_v3.json"
    packet_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(serve_module, "BENCHMARK_PACKET_PATH", packet_path)
    client = TestClient(app)

    session = client.get("/api/benchmark/session")
    feedback = client.post("/api/benchmark/feedback", json={})

    assert session.status_code == 401
    assert session.json()["login_required"] is True
    assert feedback.status_code == 401


def test_benchmark_local_login_cookie_and_cursor_progress(monkeypatch, tmp_path):
    packet_path = tmp_path / "g2_packet_v3.json"
    selection_path = tmp_path / "g2_packet_v3.selection_log.json"
    feedback_path = tmp_path / "feedback.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    packet_path.write_text(json.dumps([{
        "id": "item_001",
        "belief_id": "B-OECT",
        "belief": "OECT value is integration.",
        "title": "OECT integration paper",
        "abstract": "This paper reports an integrated sensing platform.",
    }]), encoding="utf-8")
    selection_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(serve_module, "BENCHMARK_PACKET_PATH", packet_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_SELECTION_LOG_PATH", selection_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_FEEDBACK_PATH", feedback_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_PROGRESS_PATH", progress_path)
    monkeypatch.setenv("SCHOLARHOUND_BENCHMARK_ACCESS_CODE", "test-code")
    monkeypatch.setenv("SCHOLARHOUND_BENCHMARK_SESSION_SECRET", "test-secret")
    client = TestClient(app, base_url="http://localhost")

    login = client.post("/api/benchmark/login", json={
        "reviewer": "local-expert",
        "expertise": "bioelectronics",
        "access_code": "test-code",
    })
    auth = client.get("/api/benchmark/auth")
    cursor = client.post("/api/benchmark/progress", json={
        "event_type": "cursor",
        "item_id": "item_001",
        "session_run_id": "run-local",
    })

    assert login.status_code == 200
    assert login.json()["reviewer"]["auth_source"] == "access_code"
    assert auth.status_code == 200
    assert auth.json()["reviewer"]["reviewer"] == "local-expert"
    assert cursor.status_code == 200
    progress_record = json.loads(progress_path.read_text(encoding="utf-8").splitlines()[0])
    assert progress_record["event_type"] == "cursor"
    assert progress_record["item_id"] == "item_001"
    assert progress_record["session_run_id"] == "run-local"


def test_benchmark_progress_is_scoped_and_resumes_cursor(monkeypatch, tmp_path):
    packet_path = tmp_path / "g2_packet_v3.json"
    selection_path = tmp_path / "g2_packet_v3.selection_log.json"
    feedback_path = tmp_path / "feedback.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    packet_path.write_text(json.dumps([
        {
            "id": "item_001",
            "belief_id": "B-OECT",
            "belief": "Belief one.",
            "title": "Paper one",
            "abstract": "Abstract one.",
        },
        {
            "id": "item_002",
            "belief_id": "B-SENSE",
            "belief": "Belief two.",
            "title": "Paper two",
            "abstract": "Abstract two.",
        },
        {
            "id": "item_003",
            "belief_id": "B-EV",
            "belief": "Belief three.",
            "title": "Paper three",
            "abstract": "Abstract three.",
        },
    ]), encoding="utf-8")
    selection_path.write_text("{}", encoding="utf-8")
    reviewer_a = serve_module._reviewer_id("a@example.org")
    reviewer_b = serve_module._reviewer_id("b@example.org")
    feedback_path.write_text("\n".join([
        json.dumps({
            "packet": "g2_packet_v3",
            "item_id": "item_001",
            "relation": "support",
            "reviewer_id": reviewer_a,
            "reviewer": "a@example.org",
        }),
        json.dumps({
            "packet": "g2_packet_v3",
            "item_id": "item_002",
            "relation": "challenge",
            "reviewer_id": reviewer_b,
            "reviewer": "b@example.org",
        }),
    ]) + "\n", encoding="utf-8")
    progress_path.write_text(json.dumps({
        "schema_id": "human_benchmark_progress_event_v1",
        "packet": "g2_packet_v3",
        "reviewer_id": reviewer_a,
        "event_type": "cursor",
        "item_id": "item_003",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(serve_module, "BENCHMARK_PACKET_PATH", packet_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_SELECTION_LOG_PATH", selection_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_FEEDBACK_PATH", feedback_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_PROGRESS_PATH", progress_path)
    client = TestClient(app)

    response_a = client.get(
        "/api/benchmark/session",
        headers={"Cf-Access-Authenticated-User-Email": "a@example.org"},
    )
    response_b = client.get(
        "/api/benchmark/session",
        headers={"Cf-Access-Authenticated-User-Email": "b@example.org"},
    )

    progress_a = response_a.json()["progress"]
    progress_b = response_b.json()["progress"]
    assert progress_a["processed_ids"] == ["item_001"]
    assert progress_a["resume_item_id"] == "item_003"
    assert progress_b["processed_ids"] == ["item_002"]
    assert "item_001" not in progress_b["processed_ids"]


def test_benchmark_duplicate_feedback_is_idempotent_but_not_overwritable(
    monkeypatch,
    tmp_path,
):
    packet_path = tmp_path / "g2_packet_v3.json"
    selection_path = tmp_path / "g2_packet_v3.selection_log.json"
    feedback_path = tmp_path / "feedback.jsonl"
    progress_path = tmp_path / "progress.jsonl"
    packet_path.write_text(json.dumps([{
        "id": "item_001",
        "belief_id": "B-OECT",
        "belief": "OECT value is integration.",
        "title": "OECT integration paper",
        "abstract": "This paper reports an integrated sensing platform.",
    }]), encoding="utf-8")
    selection_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(serve_module, "BENCHMARK_PACKET_PATH", packet_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_SELECTION_LOG_PATH", selection_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_FEEDBACK_PATH", feedback_path)
    monkeypatch.setattr(serve_module, "BENCHMARK_PROGRESS_PATH", progress_path)
    client = TestClient(app)
    headers = {"Cf-Access-Authenticated-User-Email": "expert@example.org"}
    payload = {
        "item_id": "item_001",
        "relation": "support",
        "confidence": "high",
        "reason": "Direct support.",
        "expertise": "bioelectronics",
        "flags": [],
        "elapsed_ms": 1234,
        "session_run_id": "run-test",
    }

    first = client.post("/api/benchmark/feedback", json=payload, headers=headers)
    duplicate = client.post("/api/benchmark/feedback", json=payload, headers=headers)
    changed = client.post(
        "/api/benchmark/feedback",
        json={**payload, "relation": "challenge"},
        headers=headers,
    )

    assert first.status_code == 200
    assert first.json()["duplicate"] is False
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert changed.status_code == 409
    assert len(feedback_path.read_text(encoding="utf-8").splitlines()) == 1
