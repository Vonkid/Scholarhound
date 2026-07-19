from psil.kernel import (
    apply_paper_type_router,
    audit_judgment_mode,
    infer_paper_type_route,
    kernel_classify_paper,
)
from psil.store.models import Paper


def test_paper_type_router_selects_validation_mode_before_scoring():
    route = infer_paper_type_route(
        problem_class="Sensing",
        novelty_type="New Validation",
        evidence_type="Benchmark Evidence",
        strategic_value="Disease-Relevant Functional Readout",
    )

    assert route["paper_type"] == "validation_or_benchmark_paper"
    assert route["judgment_mode"] == "validation_readiness"
    assert route["judgment_weights"]["concept_support"] == 0.30


def test_paper_type_router_changes_score_formula_by_mode():
    reasoning = {
        "paper_type": "Mechanism Paper",
        "relevance": 4,
        "novelty": 10,
        "bridge": 8,
        "trajectory_influence": 9,
        "concept_support": 3,
    }

    routed = apply_paper_type_router(reasoning)

    assert routed["paper_type"] == "mechanism_paper"
    assert routed["judgment_mode"] == "mechanism_shift"
    assert routed["final_score"] == 7.8


def test_kernel_only_classification_still_has_paper_type_route():
    paper = Paper(
        doi="10.0/weak",
        title="Regional geology formation mapping",
        abstract="Sedimentary rock layer analysis.",
    )

    reasoning = kernel_classify_paper(paper, [])

    assert reasoning["paper_type"] == "other_research"
    assert reasoning["judgment_mode"] == "general_judgment"
    assert reasoning["paper_type_router"]["stage"] == "paper_type_first"


def test_mode_audit_passes_transduction_route_with_coupling_evidence():
    reasoning = {
        "paper_type": "Transduction",
        "relevance": 8,
        "novelty": 7,
        "bridge": 8,
        "trajectory_influence": 7,
        "concept_support": 6,
        "why_matters": "The sensor transduces molecular recognition into an OECT readout.",
        "potential_connection": "Surface charge and ionic capacitance modulate channel conductance.",
    }

    routed = apply_paper_type_router(reasoning)

    assert routed["judgment_mode"] == "transduction_route"
    assert routed["mode_audit"]["route_confidence"] == 1.0
    assert routed["mode_audit"]["missing_checks"] == []


def test_mode_audit_flags_missing_transduction_coupling_route():
    reasoning = {
        "paper_type": "Transduction",
        "relevance": 8,
        "novelty": 7,
        "bridge": 8,
        "trajectory_influence": 7,
        "concept_support": 6,
        "why_matters": "The paper claims a useful biosensing signal.",
        "potential_connection": "Connection is conceptual rather than physically specified.",
    }

    routed = apply_paper_type_router(reasoning)

    assert routed["judgment_mode"] == "transduction_route"
    assert "recognition_to_output" in routed["mode_audit"]["passed_checks"]
    assert "coupling_route" in routed["mode_audit"]["missing_checks"]
    assert routed["mode_audit"]["route_confidence"] == 0.5


def test_validation_mode_audit_requires_benchmark_and_sample_context():
    reasoning = {
        "paper_type": "Validation",
        "relevance": 8,
        "novelty": 4,
        "bridge": 6,
        "trajectory_influence": 7,
        "concept_support": 9,
        "evidence_type": "Benchmark Evidence",
        "why_matters": "The assay is validated against a clinical baseline control.",
        "potential_connection": "Patient samples define the disease context.",
    }

    audit = apply_paper_type_router(reasoning)["mode_audit"]

    assert audit["judgment_mode"] == "validation_readiness"
    assert audit["route_confidence"] == 1.0
    assert audit["passed_checks"] == ["benchmark_or_control", "sample_context"]


def test_direct_mode_audit_is_stable_for_general_mode():
    audit = audit_judgment_mode({"judgment_mode": "general_judgment", "why_matters": "General note."})

    assert audit["judgment_mode"] == "general_judgment"
    assert audit["route_confidence"] == 1.0
    assert audit["checks"] == []
