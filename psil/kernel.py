"""
Scientific Reasoning Kernel.

Independent of LLM. No API calls. Pure symbolic logic.

Kernel responsibilities:
0. Tiered dispatch
1. Domain consistency
2. Constraint → Trajectory feedback
3. Semantic drift detection
4. Independent evidence estimation
5. Blind spot detection + human-in-the-loop learning

LLM generates candidates. Kernel makes commitments.
"""

import re
import json


# ═══════════════════════════════════════════════════════════════════════════════
# -1. Paper Type Router → Judgment Mode
# ═══════════════════════════════════════════════════════════════════════════════

PAPER_TYPE_TO_MODE = {
    "mechanism_paper": "mechanism_shift",
    "transduction_or_device_paper": "transduction_route",
    "platform_or_method_paper": "platform_readiness",
    "validation_or_benchmark_paper": "validation_readiness",
    "clinical_or_translation_paper": "translation_readiness",
    "review_or_synthesis_paper": "synthesis_map",
    "computational_or_agent_paper": "agent_architecture",
    "fundamental_phenomenon_paper": "phenomenon_watch",
    "commentary_or_noise": "screen",
    "other_research": "general_judgment",
}

JUDGMENT_MODE_WEIGHTS = {
    "general_judgment": {
        "relevance": 0.25,
        "novelty": 0.20,
        "bridge": 0.20,
        "trajectory_influence": 0.20,
        "concept_support": 0.15,
    },
    "mechanism_shift": {
        "relevance": 0.15,
        "novelty": 0.30,
        "bridge": 0.20,
        "trajectory_influence": 0.25,
        "concept_support": 0.10,
    },
    "transduction_route": {
        "relevance": 0.25,
        "novelty": 0.20,
        "bridge": 0.25,
        "trajectory_influence": 0.20,
        "concept_support": 0.10,
    },
    "platform_readiness": {
        "relevance": 0.20,
        "novelty": 0.20,
        "bridge": 0.20,
        "trajectory_influence": 0.25,
        "concept_support": 0.15,
    },
    "validation_readiness": {
        "relevance": 0.25,
        "novelty": 0.10,
        "bridge": 0.15,
        "trajectory_influence": 0.20,
        "concept_support": 0.30,
    },
    "translation_readiness": {
        "relevance": 0.30,
        "novelty": 0.10,
        "bridge": 0.20,
        "trajectory_influence": 0.20,
        "concept_support": 0.20,
    },
    "synthesis_map": {
        "relevance": 0.20,
        "novelty": 0.10,
        "bridge": 0.25,
        "trajectory_influence": 0.30,
        "concept_support": 0.15,
    },
    "agent_architecture": {
        "relevance": 0.25,
        "novelty": 0.20,
        "bridge": 0.20,
        "trajectory_influence": 0.25,
        "concept_support": 0.10,
    },
    "phenomenon_watch": {
        "relevance": 0.10,
        "novelty": 0.30,
        "bridge": 0.25,
        "trajectory_influence": 0.25,
        "concept_support": 0.10,
    },
    "screen": {
        "relevance": 0.20,
        "novelty": 0.05,
        "bridge": 0.15,
        "trajectory_influence": 0.15,
        "concept_support": 0.45,
    },
}

JUDGMENT_MODE_AUDIT_RULES = {
    "mechanism_shift": [
        ("causal_mechanism", ["mechanism", "pathway", "causal", "drives", "modulates", "controls", "because"]),
        ("model_shift", ["reframe", "challenges", "model", "principle", "explains"]),
    ],
    "transduction_route": [
        ("recognition_to_output", ["transduce", "transduction", "modulate", "readout", "signal", "output", "channel", "state"]),
        ("coupling_route", [
            "redox", "ionic", "surface charge", "capacitance", "impedance",
            "refractive index", "q-factor", "fluorescence", "mechanical deformation",
            "enzymatic product", "electrochemical", "conductance",
        ]),
    ],
    "platform_readiness": [
        ("reusable_platform", ["platform", "method", "workflow", "scaffold", "fabrication", "throughput", "scalable"]),
        ("application_scope", ["sample", "device", "integration", "robust", "repeatable", "reproducible"]),
    ],
    "validation_readiness": [
        ("benchmark_or_control", ["benchmark", "control", "comparison", "validated", "validation", "standard", "baseline"]),
        ("sample_context", ["sample", "patient", "disease", "cohort", "organoid", "clinical", "in vivo"]),
    ],
    "translation_readiness": [
        ("disease_context", ["disease", "patient", "clinical", "in vivo", "therapeutic", "diagnostic"]),
        ("failure_or_constraint", ["failure", "limitation", "toxicity", "specificity", "sensitivity", "off-target", "safety"]),
    ],
    "synthesis_map": [
        ("map_shift", ["review", "synthesis", "landscape", "framework", "taxonomy", "missing", "gap"]),
        ("actionable_gap", ["open question", "missing link", "opportunity", "future direction", "needs"]),
    ],
    "agent_architecture": [
        ("system_architecture", ["agent", "harness", "workflow", "tool", "memory", "kernel", "evaluation"]),
        ("verifiable_state", ["state", "trace", "benchmark", "test", "audit", "reproducible"]),
    ],
    "phenomenon_watch": [
        ("physical_phenomenon", ["photophysics", "photochemistry", "polariton", "energy transfer", "excited state", "coupling"]),
        ("possible_route", ["biosensing", "bioelectronic", "readout", "transduction", "control"]),
    ],
}


def _kernel_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _score(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iter_reasoning_strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_reasoning_strings(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _iter_reasoning_strings(child)
    elif value is not None:
        text = str(value).strip()
        if text:
            yield text


def audit_judgment_mode(reasoning: dict, title: str = "", abstract: str = "") -> dict:
    """Check whether the selected judgment mode has its expected evidence shape."""
    mode = _kernel_label(reasoning.get("judgment_mode", "general_judgment"))
    rules = JUDGMENT_MODE_AUDIT_RULES.get(mode, [])
    text = " ".join([title or "", abstract or "", *_iter_reasoning_strings(reasoning)]).lower()

    checks = []
    for check_name, terms in rules:
        matched_terms = [term for term in terms if term in text]
        checks.append({
            "check": check_name,
            "passed": bool(matched_terms),
            "matched_terms": matched_terms[:5],
        })

    passed_checks = [check["check"] for check in checks if check["passed"]]
    missing_checks = [check["check"] for check in checks if not check["passed"]]
    confidence = len(passed_checks) / len(checks) if checks else 1.0
    return {
        "judgment_mode": mode,
        "route_confidence": round(confidence, 2),
        "passed_checks": passed_checks,
        "missing_checks": missing_checks,
        "checks": checks,
    }


def infer_paper_type_route(
    problem_class: str = "",
    novelty_type: str = "",
    evidence_type: str = "",
    strategic_value: str = "",
    paper_type: str = "",
    judgment_mode: str = "",
    title: str = "",
    abstract: str = "",
) -> dict:
    """First classify the paper type, then select the corresponding judgment mode."""
    raw_type = _kernel_label(paper_type)
    raw_mode = _kernel_label(judgment_mode)
    text = " ".join([
        problem_class or "",
        novelty_type or "",
        evidence_type or "",
        strategic_value or "",
        title or "",
        abstract or "",
    ]).lower()

    aliases = {
        "mechanism": "mechanism_paper",
        "mechanistic": "mechanism_paper",
        "mechanism_paper": "mechanism_paper",
        "transduction": "transduction_or_device_paper",
        "device": "transduction_or_device_paper",
        "device_paper": "transduction_or_device_paper",
        "transduction_or_device_paper": "transduction_or_device_paper",
        "platform": "platform_or_method_paper",
        "method": "platform_or_method_paper",
        "platform_or_method_paper": "platform_or_method_paper",
        "validation": "validation_or_benchmark_paper",
        "benchmark": "validation_or_benchmark_paper",
        "validation_or_benchmark_paper": "validation_or_benchmark_paper",
        "clinical": "clinical_or_translation_paper",
        "translation": "clinical_or_translation_paper",
        "clinical_or_translation_paper": "clinical_or_translation_paper",
        "review": "review_or_synthesis_paper",
        "synthesis": "review_or_synthesis_paper",
        "review_or_synthesis_paper": "review_or_synthesis_paper",
        "computational": "computational_or_agent_paper",
        "agent": "computational_or_agent_paper",
        "computational_or_agent_paper": "computational_or_agent_paper",
        "fundamental": "fundamental_phenomenon_paper",
        "fundamental_phenomenon_paper": "fundamental_phenomenon_paper",
        "commentary": "commentary_or_noise",
        "ignore": "commentary_or_noise",
        "commentary_or_noise": "commentary_or_noise",
    }
    canonical_type = aliases.get(raw_type, "")

    if not canonical_type:
        if _contains_any(text, ["review/perspective/commentary", "mostly review", "review", "perspective", "commentary", "editorial"]):
            canonical_type = "review_or_synthesis_paper"
        elif _contains_any(text, ["clinical evidence", "patient", "human data", "clinical", "translation/clinical readiness"]):
            canonical_type = "clinical_or_translation_paper"
        elif _contains_any(text, ["validation evidence", "benchmark evidence", "new validation", "benchmark"]):
            canonical_type = "validation_or_benchmark_paper"
        elif _contains_any(text, ["new transduction principle", "device/electronics", "sensing", "transistor", "oect", "bioelectronic"]):
            canonical_type = "transduction_or_device_paper"
        elif _contains_any(text, ["biological mechanism", "mechanistic evidence", "new mechanism", "pathway"]):
            canonical_type = "mechanism_paper"
        elif _contains_any(text, ["fundamental photophysics", "fundamental chemistry", "photophysics", "photochemistry", "polariton"]):
            canonical_type = "fundamental_phenomenon_paper"
        elif _contains_any(text, ["material platform", "manufacturing/fabrication", "new method", "new platform", "throughput"]):
            canonical_type = "platform_or_method_paper"
        elif _contains_any(text, ["data/ai/computational", "agent", "llm", "algorithm", "computational"]):
            canonical_type = "computational_or_agent_paper"
        else:
            canonical_type = "other_research"

    canonical_mode = raw_mode if raw_mode in JUDGMENT_MODE_WEIGHTS else PAPER_TYPE_TO_MODE.get(canonical_type, "general_judgment")
    weights = JUDGMENT_MODE_WEIGHTS.get(canonical_mode, JUDGMENT_MODE_WEIGHTS["general_judgment"])
    return {
        "paper_type": canonical_type,
        "judgment_mode": canonical_mode,
        "judgment_weights": weights,
    }


def score_with_judgment_mode(reasoning: dict, judgment_mode: str = "") -> float:
    mode = _kernel_label(judgment_mode or reasoning.get("judgment_mode", "general_judgment"))
    weights = JUDGMENT_MODE_WEIGHTS.get(mode, JUDGMENT_MODE_WEIGHTS["general_judgment"])
    total = sum(_score(reasoning.get(key)) * weight for key, weight in weights.items())
    return round(total, 1)


def apply_paper_type_router(reasoning: dict, title: str = "", abstract: str = "") -> dict:
    """Attach paper type, judgment mode, mode weights, and recomputed score."""
    route = infer_paper_type_route(
        problem_class=reasoning.get("problem_class", ""),
        novelty_type=reasoning.get("novelty_type", ""),
        evidence_type=reasoning.get("evidence_type", ""),
        strategic_value=reasoning.get("strategic_value", ""),
        paper_type=reasoning.get("paper_type", ""),
        judgment_mode=reasoning.get("judgment_mode", ""),
        title=title,
        abstract=abstract,
    )
    reasoning["paper_type"] = route["paper_type"]
    reasoning["judgment_mode"] = route["judgment_mode"]
    reasoning["judgment_weights"] = route["judgment_weights"]
    reasoning["final_score"] = score_with_judgment_mode(reasoning, route["judgment_mode"])
    reasoning["mode_audit"] = audit_judgment_mode(reasoning, title, abstract)
    reasoning["paper_type_router"] = {
        "stage": "paper_type_first",
        "paper_type": route["paper_type"],
        "judgment_mode": route["judgment_mode"],
        "weights": route["judgment_weights"],
        "mode_audit": reasoning["mode_audit"],
    }
    return reasoning


# ═══════════════════════════════════════════════════════════════════════════════
# 0. Tiered Analysis Dispatch (bMAS-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_signal_strength(matched_signals: list[tuple[str, int]]) -> dict:
    """Classify pre-filter signal strength: FULL, LITE, or SKIP LLM analysis.

    HIGH (FULL LLM + Validator): 2+ Tier 1 matches, or 1 Tier 1 + 2+ Tier 2
    MEDIUM (LLM only): 1 Tier 1, or 2+ Tier 2, or 3+ Tier 3
    LOW (Kernel-only, no LLM): anything weaker — skip ranking, mark LOW_PRIORITY
    """
    tier1_count = sum(1 for _, w in matched_signals if w == 5)
    tier2_count = sum(1 for _, w in matched_signals if w == 3)
    tier3_count = sum(1 for _, w in matched_signals if w == 1)

    if tier1_count >= 2:
        return {"tier": "FULL", "reason": f"{tier1_count} T1 + {tier2_count} T2 signals"}
    if tier1_count == 1 and tier2_count >= 2:
        return {"tier": "FULL", "reason": f"{tier1_count} T1 + {tier2_count} T2 signals"}
    if tier1_count >= 1:
        return {"tier": "MEDIUM", "reason": f"{tier1_count} T1 signal"}
    if tier2_count >= 2:
        return {"tier": "MEDIUM", "reason": f"{tier2_count} T2 signals"}
    if tier3_count >= 3:
        return {"tier": "MEDIUM", "reason": f"{tier3_count} T3 signals"}
    return {"tier": "LOW", "reason": f"only {tier1_count}T1/{tier2_count}T2/{tier3_count}T3 — kernel-only classification"}


def kernel_classify_paper(paper, matched_signals: list[tuple[str, int]]) -> dict:
    """Kernel-only paper classification when LLM dispatch is not warranted.

    Returns a minimal reasoning dict comparable to LLM output,
    so the downstream pipeline works without modification.
    """
    signal_names = [s for s, _ in matched_signals]
    reasoning = {
        "signal_tier": "LOW_PRIORITY",
        "relevance": 0,
        "novelty": 0,
        "bridge": 0,
        "trajectory_influence": 0,
        "concept_support": 0,
        "final_score": 0.0,
        "problem_class": "",
        "novelty_type": "Incremental Optimization",
        "evidence_type": "",
        "strategic_value": "Low Strategic Value",
        "concept_support_name": "",
        "support_type": "",
        "evidence_strength": "",
        "why_matters": f"Kernel classified: {', '.join(signal_names[:3]) if signal_names else 'weak signals'}",
        "potential_connection": "No specific connection to research map.",
        "weakness": "",
        "action": "Skip",
        "concept_name": "",
        "causal": {"question": "", "constraint": "", "input_state": "",
                    "transformation": "", "output_state": "", "outcome": ""},
        "kernel_classified": True,
    }
    return apply_paper_type_router(reasoning, getattr(paper, "title", ""), getattr(paper, "abstract", ""))

import re


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Domain Consistency Check
# ═══════════════════════════════════════════════════════════════════════════════

# Problem class → expected concept domains
# Treatment papers should NOT claim Sensing concepts, and vice versa
DOMAIN_RULES = {
    "Sensing": {"expected_concepts": [
        "sensing", "biosensing", "detection", "diagnostic", "monitoring",
        "transduction", "readout", "signal", "recognition", "biomarker",
        "ev phenotyping", "ev sensing", "oect", "electrochemical",
        "nanophotonic", "plasmonic", "optical", "fluorescence",
    ]},
    "Treatment": {"expected_concepts": [
        "treatment", "therapy", "drug", "delivery", "release", "regeneration",
        "wound", "healing", "therapeutic", "photodynamic", "photothermal",
    ]},
    "Device/Electronics": {"expected_concepts": [
        "device", "fabrication", "electronics", "transistor", "electrode",
        "biointerface", "stretchable", "wearable", "oect", "flexible",
    ]},
    "Material Platform": {"expected_concepts": [
        "material", "hydrogel", "nanoparticle", "polymer", "scaffold",
        "nanomaterial", "fabrication", "synthesis",
    ]},
    "Fundamental Photophysics": {"expected_concepts": [
        "photophysics", "photon", "optical", "excited", "energy transfer",
        "photochemistry", "plasmonic", "nanophotonic", "photoluminescence",
    ]},
    "Biological Mechanism": {"expected_concepts": [
        "mechanism", "pathway", "signaling", "biological", "cellular",
        "molecular", "genetic", "protein", "receptor",
    ]},
    "Disease Model": {"expected_concepts": [
        "disease", "model", "organoid", "animal", "patient", "clinical",
        "pathology", "biomarker",
    ]},
    "Data/AI/Computational": {"expected_concepts": [
        "computational", "machine learning", "ai", "modeling", "simulation",
        "prediction", "algorithm", "data",
    ]},
}


def check_domain_consistency(problem_class: str, concept_name: str,
                              strategic_value: str = "") -> dict:
    """Check if the claimed concept domain is consistent with the paper's problem class.

    Returns: {consistent: bool, flag: str or None, confidence_penalty: int}
    """
    if not problem_class or not concept_name:
        return {"consistent": True, "flag": None, "confidence_penalty": 0}

    pc = problem_class.strip()
    cn = concept_name.lower().strip()

    # Find the domain rules for this problem class
    rule = None
    for key, val in DOMAIN_RULES.items():
        if key.lower() in pc.lower():
            rule = val
            break

    if not rule:
        return {"consistent": True, "flag": "Unknown problem class", "confidence_penalty": 0}

    # Check if concept matches expected domain
    expected = rule["expected_concepts"]
    matches = any(exp in cn for exp in expected)

    if matches:
        return {"consistent": True, "flag": None, "confidence_penalty": 0}

    # Mismatch detected — but check for known cross-domain exceptions
    # E.g., "Adaptive Biointerfaces" can appear in both Device and Sensing
    cross_domain = {"adaptive biointerface", "mechanobiology-enabled sensing",
                     "molecular bioelectronics", "platform engineering"}
    if cn in cross_domain:
        return {"consistent": True, "flag": "Cross-domain concept (allowed)", "confidence_penalty": 0}

    # Genuine mismatch
    return {
        "consistent": False,
        "flag": f"Domain mismatch: {pc} paper claims {concept_name}",
        "confidence_penalty": 3,  # penalize CS by 3 points
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Constraint → Trajectory Feedback
# ═══════════════════════════════════════════════════════════════════════════════

def constraint_trajectory_feedback(db, trajectory_map: dict) -> dict:
    """Wire constraint verification results into trajectory confidence.

    If constraints linked to a trajectory are violated, decrease confidence.
    If supported, increase.
    """
    verifications = db.get_verifications()
    trajectories = db.get_trajectories()
    constraints = {c.get("name", ""): c for c in db.get_constraints()}

    feedback = {}
    for traj in trajectories:
        name = traj.get("name", "")
        # Find constraints linked to this trajectory's concepts
        concept_keys = trajectory_map.get(name, [])
        if not concept_keys:
            continue

        violations = 0
        supports = 0
        for v in verifications:
            cname = v.get("constraint_name", "")
            c = constraints.get(cname, {})
            c_fw = c.get("framework_name", "") if c else ""
            # Check if this constraint relates to any trajectory concept
            related = any(ck.lower() in c_fw.lower() for ck in concept_keys)
            if not related:
                continue
            if v.get("result") == "violated":
                violations += 1
            elif v.get("result") == "supported":
                supports += 1

        current_conf = traj.get("confidence", "Stable")
        if violations > supports:
            new_conf = "Decreasing"
        elif supports > 0 and violations == 0:
            new_conf = "Increasing"
        else:
            new_conf = current_conf

        if new_conf != current_conf:
            db.update_trajectory(name=name, confidence=new_conf)
            feedback[name] = {
                "old_confidence": current_conf,
                "new_confidence": new_conf,
                "violations": violations,
                "supports": supports,
            }

    return feedback


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Concept Semantic Drift Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_semantic_drift(db, concept_name: str) -> dict:
    """Check if a concept is being used consistently across papers.

    Compares causal transformations across papers supporting the same concept.
    If transformations are fundamentally different, flag as potential drift.
    """
    papers = db.get_papers_with_causal(days_back=60)
    if not papers:
        return {"drift_detected": False, "drift_score": 0}

    # Find papers with this concept
    relevant = []
    for p in papers:
        t = (p.get("causal_transformation", "") or "").lower()
        if not t:
            continue
        # Check if paper's DB record matches this concept
        concept_col = p.get("concept_support_name", "") or p.get("concept_name", "")
        if concept_name.lower() in concept_col.lower():
            relevant.append(t)

    if len(relevant) < 2:
        return {"drift_detected": False, "drift_score": 0, "reason": "Not enough papers"}

    # Cluster transformations by type keywords
    clusters = {}
    for t in relevant:
        # Classify transformation type
        if any(w in t for w in ["cataly", "enzym", "amplif", "cascade"]):
            cluster = "catalytic/amplification"
        elif any(w in t for w in ["energy", "transduc", "convert", "harvest"]):
            cluster = "energy transduction"
        elif any(w in t for w in ["bind", "affin", "recogn", "capture"]):
            cluster = "binding/recognition"
        elif any(w in t for w in ["structur", "assembl", "configur", "reconfig"]):
            cluster = "structural/assembly"
        elif any(w in t for w in ["switch", "gating", "select", "modulat"]):
            cluster = "switching/gating"
        elif any(w in t for w in ["light", "photon", "optic", "fluoresc"]):
            cluster = "optical/photonic"
        elif any(w in t for w in ["electron", "charge", "current", "electrochem"]):
            cluster = "electronic/electrochemical"
        else:
            cluster = "other"
        clusters[cluster] = clusters.get(cluster, 0) + 1

    if len(clusters) <= 1:
        return {"drift_detected": False, "drift_score": 0, "clusters": clusters}

    # Drift score: proportion of papers NOT in the dominant cluster
    total = len(relevant)
    dominant_count = max(clusters.values())
    drift_score = round(1 - (dominant_count / total), 2)

    return {
        "drift_detected": drift_score >= 0.35,
        "drift_score": drift_score,
        "clusters": clusters,
        "dominant_cluster": max(clusters, key=clusters.get),
        "reason": f"Concept used across {len(clusters)} different mechanism types"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Independent Evidence Strength Estimation
# ═══════════════════════════════════════════════════════════════════════════════

# Journal tiers based on impact and rigor
JOURNAL_TIERS = {
    1: ["nature", "science", "cell", "new england journal", "lancet"],
    2: ["nature materials", "nature nanotechnology", "nature photonics",
        "nature biotechnology", "nature biomedical engineering",
        "nature electronics", "nature chemistry", "nature communications",
        "science advances", "science translational medicine",
        "cell reports medicine", "neuron", "cancer cell", "immunity",
        "advanced materials", "advanced functional materials",
        "acs nano", "nano letters", "angewandte chemie", "jacs"],
    3: ["biosensors and bioelectronics", "acs sensors", "analytical chemistry",
        "chemical society reviews", "acs applied materials"],
}


def estimate_evidence_strength(journal: str, abstract: str = "",
                                problem_class: str = "",
                                novelty_type: str = "") -> dict:
    """Kernel-side evidence strength estimation. No LLM.

    Uses: journal tier, sample size indicators, validation language,
    independent confirmation markers, limitation disclosure.
    """
    score = 5.0  # neutral starting point
    factors = []

    # 1. Journal tier (0-10)
    jl = (journal or "").lower()
    tier = 3
    for t, names in JOURNAL_TIERS.items():
        if any(n in jl for n in names):
            tier = t
            break
    if tier == 1:
        score += 2
        factors.append("top-tier journal (+2)")
    elif tier == 2:
        score += 1
        factors.append("high-impact journal (+1)")

    # 2. Sample size / statistical rigor signals
    if abstract:
        al = abstract.lower()
        # Large samples
        if any(w in al for w in ["n =", "n="]):
            m = re.search(r'n\s*=\s*(\d+)', al)
            if m:
                n_val = int(m.group(1))
                if n_val >= 100:
                    score += 1
                    factors.append(f"large sample (n={n_val}, +1)")

        # Validation language
        if any(w in al for w in ["validated", "independently confirmed",
                                   "consistent with", "reproduc"]):
            score += 0.5
            factors.append("validation language (+0.5)")

        # Limitation disclosure (honesty signal)
        if any(w in al for w in ["limitation", "caveat", "further study needed"]):
            score += 0.5
            factors.append("limitation disclosure (+0.5)")

    # 3. Novelty type affects evidence interpretation
    nt = (novelty_type or "").lower()
    if "validation" in nt:
        score += 1
        factors.append("validation paper (+1)")
    elif "new mechanism" in nt or "new transduction" in nt:
        # New mechanisms need more independent validation
        if tier >= 2:
            score += 0.5
            factors.append("new mechanism in reputable journal (+0.5)")

    # 4. Review papers carry less direct evidence weight
    pc = (problem_class or "").lower()
    if "review" in pc or "commentary" in pc or "perspective" in pc:
        score -= 1.5
        factors.append("review/commentary (-1.5)")

    kernel_strength = "High" if score >= 7 else ("Medium" if score >= 4.5 else "Low")

    return {
        "kernel_evidence_score": round(max(0, min(10, score)), 1),
        "kernel_evidence_strength": kernel_strength,
        "factors": factors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Blind Spot Detection + Human-in-the-Loop Learning
# ═══════════════════════════════════════════════════════════════════════════════

# Problem class → research domain overlap (0-10)
# High overlap means "this problem class is IN our research area"
DOMAIN_OVERLAP = {
    "Sensing": 10,
    "Device/Electronics": 7,  # OECT IS electronics
    "Biological Mechanism": 6,
    "Material Platform": 5,   # materials for bioelectronics
    "Fundamental Photophysics": 8,  # core to NIR/BODIPY
    "Fundamental Chemistry": 4,  # molecular motors etc. — adjacent
    "Disease Model": 5,
    "Treatment": 2,
    "Delivery": 2,
    "Manufacturing/Fabrication": 4,
    "Data/AI/Computational": 3,
    "Review/Perspective/Commentary": 4,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Similarity Engine (sentence embeddings, no LLM)
# ═══════════════════════════════════════════════════════════════════════════════

_embedding_model = None
_cached_ref_embeddings = None
_cached_ref_dois = None
_cached_ref_texts = None


def _get_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def _build_reference_corpus(db) -> tuple:
    """Build sentence embedding matrix from approved/high-relevance papers."""
    global _cached_ref_embeddings, _cached_ref_dois, _cached_ref_texts

    all_papers = db.get_all_papers()
    ref_texts = []
    ref_dois = []

    try:
        approved = db.get_memory(status="approved")
    except Exception:
        approved = []

    for p in all_papers:
        tier = p.get("signal_tier", "")
        is_ref = "IMPORTANT" in tier or "HIGH" in tier or "POTENTIAL" in tier
        is_approved = any(a.get("item_name", "").lower() in
                          ((p.get("concept_support_name") or "") + (p.get("title") or "")).lower()
                          for a in approved)
        if is_ref or is_approved:
            text = ((p.get("title") or "") + " " + (p.get("abstract") or ""))[:800]
            if len(text) > 30:
                ref_texts.append(text)
                ref_dois.append(p.get("doi", ""))

    # Add seed texts if corpus is too small
    seed_texts = [
        "organic electrochemical transistor OECT biosensing molecular recognition small molecule sensing electric double layer iontronic mixed ionic-electronic conduction ionogel synaptic transistor mechanoreceptor immune-compatible semiconducting polymer organic semiconductor phase behavior",
        "extracellular vesicle EV diagnostics functional phenotyping activity-based biomarker",
        "nanophotonics photonic crystal metasurface biosensor optical resonance Q-factor sensing",
        "structured light nonlinear optics lithium niobate microresonator optical vortex optical skyrmion microcomb",
        "NIR photocleavage BODIPY photochemistry light-triggered release molecular photochemistry",
        "mechanobiology stretchable biointerface strain sensor adaptive bioelectronics",
        "organoid on chip microphysiological system non-destructive monitoring secretome analysis",
        "Alzheimer's disease diagnostics blood brain barrier delivery neurodegeneration biomarker",
    ]
    for st in seed_texts:
        ref_texts.append(st)
        ref_dois.append("seed")

    if len(ref_texts) < 2:
        return None, None, None

    model = _get_model()
    embeddings = model.encode(ref_texts, show_progress_bar=False)

    _cached_ref_embeddings = embeddings
    _cached_ref_dois = ref_dois
    _cached_ref_texts = ref_texts
    return embeddings, ref_dois, ref_texts


def compute_semantic_similarity(paper, db) -> dict:
    """Compute sentence embedding cosine similarity against reference corpus.

    True semantic similarity — not keyword overlap.
    Pure inference. No LLM. No API calls.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    global _cached_ref_embeddings

    ref_embs, ref_dois, ref_texts = _cached_ref_embeddings, _cached_ref_dois, _cached_ref_texts
    if ref_embs is None:
        ref_embs, ref_dois, ref_texts = _build_reference_corpus(db)
    if ref_embs is None or len(ref_embs) < 2:
        return {"similarity": 0, "max_similar_text": "", "verdict": "no_reference_corpus"}

    text = ((paper.title or "") + " " + (paper.abstract or ""))[:800]
    try:
        model = _get_model()
        paper_emb = model.encode([text], show_progress_bar=False)
        sims = cosine_similarity(paper_emb, ref_embs)[0]
        max_idx = int(np.argmax(sims))
        max_sim = float(sims[max_idx])
        max_text = ref_texts[max_idx] if max_idx < len(ref_texts) else ""

        verdict = "high" if max_sim >= 0.50 else ("medium" if max_sim >= 0.35 else "low")
        return {
            "similarity": round(max_sim, 3),
            "max_similar_text": max_text[:120],
            "verdict": verdict,
        }
    except Exception:
        return {"similarity": 0, "max_similar_text": "", "verdict": "error"}


def detect_blind_spot(paper, matched_signals: list[tuple[str, int]],
                       journal: str, db=None) -> dict:
    """Detect papers that the keyword filter would reject but that
    might actually be relevant based on journal tier + domain overlap.

    Returns: {is_blind_spot: bool, reason: str, confidence: float, action: str}
    """
    score = sum(w for _, w in matched_signals)
    jl = (journal or "").lower()
    signal_names = [s for s, _ in matched_signals]

    # Only flag papers with weak keyword signals from high-tier journals
    is_high_journal = any(n in jl for n in [
        "nature", "science", "cell", "lancet", "pnas",
        "advanced materials", "advanced functional", "acs nano", "nano letters",
    ])

    # Check if keywords hint at domain overlap even if score is low
    domain_hint_words = {
        "transistor", "gate", "channel", "conductivity", "ionic", "dielectric",
        "photochemistry", "photochemical", "rotor", "motor", "light", "photon",
        "molecular", "self-assemb", "nanoparticle", "elastomer", "stretchable",
        "flexible", "biointerface", "thin film", "nanoribbon", "2d",
        "sensor", "biosensor", "transduction", "readout", "detection",
        "organic", "polymer", "hydrogel", "electrolyte", "ion"
    }
    combined = ((paper.title or "") + " " + (paper.abstract or "")).lower()
    hint_count = sum(1 for w in domain_hint_words if w in combined)

    # Blind spot criteria: three independent signals
    # Signal 1: weak keyword score
    weak_keywords = score <= 2
    # Signal 2: high-tier journal
    # Signal 3: domain hints from vocabulary
    has_hints = hint_count >= 2
    # Signal 4: semantic similarity to approved reference corpus
    sim_result = compute_semantic_similarity(paper, db)
    sem_high = sim_result["verdict"] == "high"
    sem_medium = sim_result["verdict"] in ("high", "medium")

    # Blind spot: 2+ signals suggest relevance despite keyword gap
    signals = sum([weak_keywords, is_high_journal, has_hints, sem_medium])
    if signals >= 3:
        return {
            "is_blind_spot": True,
            "reason": (f"{signals}/4 signals: keyword_wk={weak_keywords} "
                       f"journal={is_high_journal} hints={has_hints} "
                       f"semantic={sim_result['verdict']}({sim_result['similarity']})"),
            "confidence": min(0.9, 0.3 + 0.15 * signals),
            "action": "FLAG_FOR_REVIEW",
        }

    return {"is_blind_spot": False, "reason": "", "confidence": 0, "action": "AUTO_CLASSIFY"}


def learn_from_override(paper_doi: str, user_decision: str, kernel_decision: str,
                         reasoning: dict, db) -> dict:
    """Kernel learns when user overrides its classification.

    User approved a paper kernel marked LOW → add domain vocabulary to learned signals
    User rejected a paper kernel marked HIGH → record negative pattern
    """
    override = {
        "paper_doi": paper_doi,
        "user_decision": user_decision,
        "kernel_decision": kernel_decision,
        "learned": [],
    }

    if user_decision == "approved" and kernel_decision == "LOW":
        # Kernel was too strict — learn new vocabulary
        pc = reasoning.get("problem_class", "")
        csn = reasoning.get("concept_support_name", "")
        sv = reasoning.get("strategic_value", "")

        # Record the successful pattern for future pre-filter boosting
        pattern = json.dumps({
            "problem_class": pc,
            "concept": csn,
            "strategic_value": sv,
        })
        existing = db.get_kernel_state("override_boost_patterns") or "[]"
        patterns = json.loads(existing)
        patterns.append(pattern)
        db.set_kernel_state("override_boost_patterns", json.dumps(patterns), "learning")
        override["learned"].append(f"Boost pattern: {pc} + {csn}")

    elif user_decision == "rejected" and kernel_decision in ("HIGH", "IMPORTANT"):
        # Kernel was too generous — record negative pattern
        pc = reasoning.get("problem_class", "")
        existing = db.get_kernel_state("override_penalty_patterns") or "[]"
        patterns = json.loads(existing)
        patterns.append(json.dumps({"problem_class": pc}))
        db.set_kernel_state("override_penalty_patterns", json.dumps(patterns), "learning")
        override["learned"].append(f"Penalty pattern: {pc}")

    return override


def get_learned_boost(db) -> dict:
    """Get accumulated learned signals for pre-filter boosting."""
    patterns_str = db.get_kernel_state("override_boost_patterns") or "[]"
    try:
        patterns = json.loads(patterns_str)
    except json.JSONDecodeError:
        patterns = []
    # Count pattern frequency
    freq = {}
    for p in patterns:
        freq[p] = freq.get(p, 0) + 1
    return {"patterns": [{"pattern": k, "frequency": v} for k, v in freq.items() if v >= 2]}


def learn_from_review(paper, db):
    """When a human marks a blind-spot paper as relevant,
    extract new concept keywords from its title + abstract and
    store them for future matching.

    Returns the new keywords learned.
    """
    text = ((paper.title or "") + " " + (paper.abstract or "")).lower()
    # Extract noun phrases that don't match existing concepts
    from psil.rank.concepts import CONCEPTS
    known = set(CONCEPTS.keys())

    # Extract potential new keywords: 2-5 word phrases with domain vocabulary
    domain_core = {"transistor", "gate", "channel", "ionic", "dielectric",
                   "photochemical", "rotor", "motor", "photon", "elastomer",
                   "conductivity", "polyelectrolyte", "nanoribbon",
                   "condensate", "molecular motor", "stretchable",
                   "biointerface", "thin film", "organic electrochemical",
                   "electric double layer", "contact injection", "iontronic",
                   "single ion", "light emitting transistor", "ionogel",
                   "mechanoreceptor", "synaptic transistor",
                   "semiconducting polymer", "foreign body",
                   "immune compatible", "phase behavior"}
    import re
    words = re.findall(r'[a-z]+', text)
    phrases_2 = set(" ".join(words[i:i+2]) for i in range(len(words)-1))
    phrases_3 = set(" ".join(words[i:i+3]) for i in range(len(words)-2))

    new_keywords = set()
    for phrase in list(phrases_2) + list(phrases_3):
        if phrase in known:
            continue
        if any(core in phrase for core in domain_core):
            new_keywords.add(phrase)

    # Store learned keywords in kernel_state
    existing = db.get_kernel_state("learned_keywords") or "{}"
    learned = json.loads(existing) if False else {}
    for kw in new_keywords:
        learned[kw] = learned.get(kw, 0) + 1

    db.set_kernel_state("learned_keywords", json.dumps(learned), "learning")

    return {"new_keywords": list(new_keywords), "total_learned": len(learned)}
