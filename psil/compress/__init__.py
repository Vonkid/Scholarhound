"""
Scientific Concept Compression Engine.

Layer 2: Logic Pattern Discovery — cluster papers by causal logic, not by topic.
Layer 3: Framework Discovery — compress multiple patterns into higher-order frameworks.
Delta Detection — identify shifts from old to new worldviews.
"""

import json

PATTERN_DISCOVERY_PROMPT = """You are a scientific logic analyst. Your task is to discover recurring LOGIC PATTERNS from a set of papers.

Each paper below has been causally extracted:
QUESTION | CONSTRAINT | INPUT_STATE | TRANSFORMATION | OUTPUT_STATE | OUTCOME

IMPORTANT: Ignore implementation details (materials, diseases, instruments). Focus on the LOGIC STRUCTURE.

## Step 1: Group papers by similar TRANSFORMATION patterns

Group papers whose transformations solve the same TYPE of problem, regardless of what materials or diseases are involved.

Examples of logic patterns:
- Signal Amplification: Input → Amplification mechanism → Amplified output
- Energy Routing: Excitation → Energy redistribution → Chemical outcome
- Static-to-Functional: Static state → Perturbation → Functional readout
- Passive-to-Active: Passive component → Functional activation → System effect
- State Transition Sensing: Initial state → State change detection → Readout signal

## Step 2: For each pattern, output

PATTERN_NAME: <short label>
PATTERN_TYPE: <Signal Amplification | Energy Routing | Static-to-Functional | Passive-to-Active | State Transition | Other>
DESCRIPTION: <one sentence describing the logic>
CAUSAL_TEMPLATE: <INPUT → TRANSFORMATION → OUTPUT in generic terms>
PAPERS: <DOI list of papers that fit this pattern>

## Step 3: Identify DELTAS — worldview shifts

For any paper or group that challenges a previous assumption:

PREVIOUS_ASSUMPTION: <old belief>
NEW_ASSUMPTION: <new belief>
DELTA: <direction of shift, e.g. "Passive → Active Nanostructure">
SOURCE_DOIS: <supporting paper DOIs>

---

PAPERS:
{causal_papers}

---

Output ALL patterns found. If no clear pattern, say "No patterns discovered."
"""

FRAMEWORK_PROMPT = """You are a scientific framework discoverer. Your task is to find HIGHER-ORDER frameworks that compress multiple logic patterns.

A valid framework must:
1. Explain ALL included patterns
2. Increase explanatory power (explain more with less)
3. Remove redundancy
4. Predict at least one testable new experiment or research direction

IMPORTANT: A framework that says everything is "information processing" or "input→output" has zero value. Reject frameworks that are too generic to make predictions.

## Logic Patterns Found:
{patterns}

## Task

1. Try to compress patterns into fewer frameworks. A framework is only valuable if it predicts something non-obvious.

2. For each framework, output ALL of these fields:

FRAMEWORK_NAME: <short label>
DESCRIPTION: <how this framework explains multiple patterns>
CORE_LOGIC: <the central causal-logic that unifies the patterns>
COVERED_PATTERNS: <which patterns it covers>
WORLDVIEW_SHIFT: <what old assumption does this framework challenge? or "None">
COMPRESSION_SCORE: <0-10, how many concepts unified into how much explanatory power>
NOVELTY_SCORE: <0-10, is this non-obvious and under-explored?>
PREDICTIVE_POWER: <0-10, how many new experiments or predictions does it generate?>
FALSIFIABILITY: <0-10, can it be tested? would a null result disprove it?>
ACTIONABILITY: <0-10, does it give concrete next steps for our research?>
TRANSFERABILITY: <0-10, can it move between fields (e.g., photonics→biology)?>
TASTE_FIT: <0-10, does it match research direction: molecular bioelectronics, EV/organoid sensing, nanophotonics, Alzheimer's diagnostics, adaptive biointerfaces?>
SUGGESTED_EXPERIMENT: <one concrete testable experiment>

Scoring guide:
- COMPRESSION: 10 = one principle explains 5+ patterns with no hand-waving
- NOVELTY: 10 = surprisingly non-obvious, challenges conventional wisdom
- PREDICTIVE_POWER: 10 = generates multiple specific, testable predictions
- FALSIFIABILITY: 10 = crystal-clear null hypothesis, easy to disprove if wrong
- ACTIONABILITY: 10 = directly suggests experiments we can do now
- TRANSFERABILITY: 10 = works across photonics, electronics, biology, chemistry
- TASTE_FIT: 10 = directly advances our long-term research vision

3. Identify UNRESOLVED CONTRADICTIONS: patterns that resist unification

4. Identify POTENTIAL RESEARCH PROGRAMS (actionable directions)

Output all frameworks. If no valid framework can be formed, explain why.
"""


def run_compress(db, llm_client, days_back: int = 7) -> dict:
    """Run the full compression pipeline (Layer 2 + 3).

    Returns a dict with: patterns, frameworks, deltas, stats
    """
    papers = db.get_papers_with_causal(days_back=days_back)
    if len(papers) < 3:
        return {
            "patterns": [], "frameworks": [], "deltas": [],
            "stats": {"papers_with_causal": len(papers),
                       "message": "Need at least 3 papers with causal extraction"}
        }

    # Layer 2: Discover logic patterns
    patterns = _discover_patterns(papers, llm_client, db)

    # Layer 3: Discover frameworks from patterns
    frameworks = []
    if len(patterns) >= 2:
        frameworks = _discover_frameworks(patterns, llm_client, db)

    # Layer 4: Constraint Discovery from frameworks
    constraints = []
    if frameworks:
        from psil.compress.constraints import run_constraint_discovery
        result = run_constraint_discovery(frameworks, patterns, llm_client, db)
        constraints = result.get("constraints", [])

    # Layer 5: Constraint Verification (pure symbolic kernel, no LLM)
    verification = run_constraint_verification(db, days_back=days_back)

    return {
        "patterns": patterns,
        "frameworks": frameworks,
        "constraints": constraints,
        "verification": verification,
        "deltas": db.get_deltas(),
        "experiments": db.get_experiments(),
        "stats": {
            "papers_with_causal": len(papers),
            "patterns_found": len(patterns),
            "frameworks_found": len(frameworks),
            "constraints_found": len(constraints),
            "constraints_supported": verification["supported"],
            "constraints_violated": verification["violated"],
            "deltas_found": len(db.get_deltas()),
        }
    }


def _discover_patterns(papers: list[dict], llm_client, db) -> list[dict]:
    """Layer 2: Discover logic patterns across papers."""
    # Build causal paper summaries
    lines = []
    for p in papers:
        lines.append(
            f"DOI: {p['doi']} | {p.get('causal_question', '')} | "
            f"Constraint: {p.get('causal_constraint', '')} | "
            f"Input: {p.get('causal_input', '')} | "
            f"Transform: {p.get('causal_transformation', '')} | "
            f"Output: {p.get('causal_output', '')} | "
            f"Outcome: {p.get('causal_outcome', '')}"
        )
    causal_text = "\n".join(lines)

    prompt = PATTERN_DISCOVERY_PROMPT.format(causal_papers=causal_text)

    try:
        resp = llm_client.client.chat.completions.create(
            model=llm_client.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  Pattern discovery LLM error: {e}")
        return []

    patterns = _parse_patterns(content, db)
    return patterns


def _parse_patterns(text: str, db) -> list[dict]:
    """Parse LLM pattern discovery output."""
    patterns = []
    current = {}
    current_delta = {}

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("---"):
            continue

        if ":" in line and not line.startswith("-"):
            key, _, value = line.partition(":")
            key = key.strip().lower().replace(" ", "_").replace("#", "").strip("*").strip("_")
            value = value.strip().strip("*").strip()

            if key == "pattern_name":
                if current and current.get("pattern_name"):
                    patterns.append(current)
                current = {"pattern_name": value, "pattern_type": "", "description": "",
                           "causal_template": "", "papers": ""}

            elif key == "previous_assumption":
                if current_delta and current_delta.get("previous"):
                    db.insert_delta(current_delta.get("previous", ""),
                                    current_delta.get("new", ""),
                                    current_delta.get("delta", ""),
                                    current_delta.get("source_dois", ""))
                current_delta = {"previous": value}

            elif current and key in ("pattern_type", "description", "causal_template", "papers"):
                current[key] = value

            elif current_delta and key == "new_assumption":
                current_delta["new"] = value
            elif current_delta and key == "delta":
                current_delta["delta"] = value
            elif current_delta and key == "source_dois":
                current_delta["source_dois"] = value

    if current and current.get("pattern_name"):
        patterns.append(current)

    # Save to DB
    for p in patterns:
        score = _score_pattern(p)
        db.upsert_logic_pattern(
            pattern_name=p.get("pattern_name", ""),
            pattern_type=p.get("pattern_type", ""),
            description=p.get("description", ""),
            causal_template=p.get("causal_template", ""),
            sample_dois=p.get("papers", ""),
            score=score,
        )

    # Save final delta
    if current_delta and current_delta.get("previous"):
        db.insert_delta(current_delta.get("previous", ""),
                        current_delta.get("new", ""),
                        current_delta.get("delta", ""),
                        current_delta.get("source_dois", ""))

    return patterns


def _score_pattern(pattern: dict) -> float:
    """Score a logic pattern based on abstraction level and explanatory power."""
    score = 5.0
    desc = pattern.get("description", "").lower()
    # Prefer abstract over material-specific
    material_words = ["nanozyme", "gold", "silver", "polymer", "silicon", "carbon"]
    if any(w in desc for w in material_words):
        score -= 1.0
    # Prefer novel over routine
    novel_words = ["non-hermitian", "exceptional point", "polariton", "strong coupling",
                   "q-factor", "quantum", "topological"]
    if any(w in desc for w in novel_words):
        score += 2.0
    # Prefer cross-domain
    bridge_words = ["photonic", "electronic", "biological", "mechanical", "chemical"]
    domain_count = sum(1 for w in bridge_words if w in desc)
    if domain_count >= 2:
        score += 1.5
    return min(10.0, max(0.0, score))


def _discover_frameworks(patterns: list[dict], llm_client, db) -> list[dict]:
    """Layer 3: Compress logic patterns into higher-order frameworks."""
    patterns_text = "\n".join(
        f"- {p['pattern_name']}: {p.get('description', '')} "
        f"[{p.get('causal_template', '')}]"
        for p in patterns
    )
    prompt = FRAMEWORK_PROMPT.format(patterns=patterns_text)

    try:
        resp = llm_client.client.chat.completions.create(
            model=llm_client.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  Framework discovery LLM error: {e}")
        return []

    frameworks = _parse_frameworks(content, db)
    return frameworks


def _parse_frameworks(text: str, db) -> list[dict]:
    """Parse LLM framework discovery output."""
    frameworks = []
    current = {}

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if ":" in line and not line.startswith("-"):
            key, _, value = line.partition(":")
            key = key.strip().lower().replace(" ", "_").replace("#", "").strip("*").strip("_")
            value = value.strip().strip("*").strip()

            if key == "framework_name" or key.startswith("framework"):
                if current and current.get("framework_name"):
                    frameworks.append(current)
                current = {"framework_name": value}

            elif current and key in ("description", "covered_patterns",
                                      "core_logic", "worldview_shift",
                                      "compression_score", "novelty_score",
                                      "predictive_power", "falsifiability",
                                      "actionability", "transferability",
                                      "taste_fit", "suggested_experiment"):
                current[key] = value

    if current and current.get("framework_name"):
        frameworks.append(current)

    prepared = []
    best_index_by_name = {}
    for fw in frameworks:
        scores = _parse_framework_scores(fw)
        if not _framework_has_substance(fw, scores):
            continue

        key = fw.get("framework_name", "").lower().strip()
        quality = _framework_quality(fw, scores)
        item = (fw, scores, quality)
        if key in best_index_by_name:
            existing_index = best_index_by_name[key]
            if quality > prepared[existing_index][2]:
                prepared[existing_index] = item
        else:
            best_index_by_name[key] = len(prepared)
            prepared.append(item)

    for fw, scores, _ in prepared:
        db.insert_framework(
            framework_name=fw.get("framework_name", ""),
            description=fw.get("description", ""),
            covered_patterns=fw.get("covered_patterns", ""),
            compression_score=scores["compression_score"],
            novelty_score=scores["novelty_score"],
            core_logic=fw.get("core_logic", ""),
            worldview_shift=fw.get("worldview_shift", ""),
            predictive_power=scores["predictive_power"],
            falsifiability=scores["falsifiability"],
            actionability=scores["actionability"],
            transferability=scores["transferability"],
            taste_fit=scores["taste_fit"],
            suggested_experiment=fw.get("suggested_experiment", ""),
        )

    return [fw for fw, _, _ in prepared]


def _parse_framework_scores(fw: dict) -> dict[str, float]:
    scores = {}
    for dim in ["compression_score", "novelty_score", "predictive_power",
                "falsifiability", "actionability", "transferability", "taste_fit"]:
        try:
            s = fw.get(dim, "0")
            scores[dim] = float(str(s).split("/")[0].split()[0])
        except (ValueError, TypeError, IndexError):
            scores[dim] = 0
    return scores


def _framework_has_substance(fw: dict, scores: dict[str, float]) -> bool:
    text_fields = [
        "description",
        "covered_patterns",
        "core_logic",
        "worldview_shift",
        "suggested_experiment",
    ]
    has_text = any((fw.get(field) or "").strip() for field in text_fields)
    has_score = any(value > 0 for value in scores.values())
    return bool((fw.get("framework_name") or "").strip() and (has_text or has_score))


def _framework_quality(fw: dict, scores: dict[str, float]) -> float:
    text_len = sum(len((fw.get(field) or "").strip()) for field in [
        "description",
        "covered_patterns",
        "core_logic",
        "worldview_shift",
        "suggested_experiment",
    ])
    return text_len + sum(scores.values()) * 10


def score_framework_v2(fw: dict, constraint_count: int = 0) -> dict:
    """Score a framework using Constraint Discovery v2 weights.

    New ranking priority:
    1. Prediction Power (25%)
    2. Constraint Strength (25%)
    3. Falsifiability (20%)
    4. Actionability (15%)
    5. Compression (10%)
    6. Novelty (5%)
    """
    pp = float(fw.get("predictive_power", 0) or 0)
    fl = float(fw.get("falsifiability", 0) or 0)
    ac = float(fw.get("actionability", 0) or 0)
    co = float(fw.get("compression_score", 0) or 0)
    no = float(fw.get("novelty_score", 0) or 0)

    cs = min(10.0, constraint_count * 2.5 + 2)

    v2_score = (
        0.25 * pp +
        0.25 * cs +
        0.20 * fl +
        0.15 * ac +
        0.10 * co +
        0.05 * no
    )
    if co >= 7 and pp <= 3:
        v2_score *= 0.5  # penalize "explains everything, predicts nothing"

    return {
        "v2_score": round(v2_score, 1),
        "prediction_power": pp,
        "constraint_strength": round(cs, 1),
        "falsifiability": fl,
        "actionability": ac,
        "compression": co,
        "novelty": no,
        "constraint_count": constraint_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Constraint Verification Engine (pure symbolic kernel, zero LLM calls)
# ═══════════════════════════════════════════════════════════════════════════════

def run_constraint_verification(db, days_back: int = 7) -> dict:
    """Cross-check new papers' causal extractions against existing constraints.

    For each constraint, check whether newly ingested papers SUPport,
    VIOLate, or have NO relationship to the constraint.
    """
    papers = db.get_papers_with_causal(days_back=days_back)
    constraints = db.get_constraints()

    if not constraints or not papers:
        return {"verified": 0, "supported": 0, "violated": 0,
                "details": [], "message": "Need both constraints and causal papers"}

    verifications = []
    for c in constraints:
        statement = (c.get("statement") or "").lower()
        if not statement:
            continue
        for p in papers:
            result = _check_constraint_match(statement, p)
            if result:
                db.upsert_verification(
                    constraint_name=c.get("name", ""),
                    paper_doi=p.get("doi", ""),
                    result=result["result"],
                    confidence=result["confidence"],
                    evidence=result["evidence"],
                )
                verifications.append({
                    "constraint": c.get("name", ""),
                    "paper_doi": p.get("doi", ""),
                    "paper_title": p.get("title", ""),
                    **result,
                })

    supported = sum(1 for v in verifications if v["result"] == "supported")
    violated = sum(1 for v in verifications if v["result"] == "violated")

    return {
        "verified": len(verifications),
        "supported": supported,
        "violated": violated,
        "details": verifications,
    }


def _check_constraint_match(constraint_stmt: str, paper: dict) -> dict | None:
    """Check if a paper's causal extraction supports or violates a constraint.

    Uses keyword overlap between constraint statement and the paper's
    causal fields (transformation, input, output, constraint).
    No LLM — pure symbolic matching on causal vocabulary.
    """
    causal_text = " ".join([
        paper.get("causal_transformation", "") or "",
        paper.get("causal_input", "") or "",
        paper.get("causal_output", "") or "",
        paper.get("causal_constraint", "") or "",
    ]).lower().strip()

    if not causal_text:
        return None

    # Extract key terms from constraint
    constraint_words = set(constraint_stmt.lower().split())
    # Filter stop words
    stop = {"a", "an", "the", "is", "are", "be", "to", "of", "in", "for",
            "on", "with", "that", "this", "if", "or", "and", "not", "no",
            "should", "can", "cannot", "must", "requires", "require", "required"}
    constraint_terms = constraint_words - stop

    # Count matches
    causal_words = set(causal_text.split())
    matches = constraint_terms & causal_words
    match_ratio = len(matches) / max(1, len(constraint_terms))

    if match_ratio >= 0.4:
        return {
            "result": "supported",
            "confidence": round(match_ratio, 2),
            "evidence": f"Matched terms: {', '.join(sorted(matches)[:10])}",
        }
    elif match_ratio >= 0.15:
        return {
            "result": "partial",
            "confidence": round(match_ratio, 2),
            "evidence": f"Weak match: {', '.join(sorted(matches)[:10])}",
        }

    # Check for negation/violation patterns
    negation_patterns = ["no ", "not ", "absence ", "without ", "lack of "]
    for neg in negation_patterns:
        if neg + " ".join(sorted(constraint_terms)[:3]) in causal_text:
            return {
                "result": "violated",
                "confidence": 0.5,
                "evidence": f"Negation detected: {neg}",
            }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Undiscovered Connection Discovery (Swanson LBD, pure symbolic kernel)
# ═══════════════════════════════════════════════════════════════════════════════

def discover_connections(db, min_similarity: float = 0.35) -> list[dict]:
    """Find undiscovered cross-domain connections via causal chain matching.

    Paper A's OUTPUT matches Paper B's INPUT, but they come from
    different problem classes — suggesting an unexplored causal link.
    """
    papers = db.get_papers_with_causal(days_back=60)
    if len(papers) < 5:
        return []

    connections = []
    for i, a in enumerate(papers):
        a_output = (a.get("causal_output", "") or "").lower().strip()
        a_class = (a.get("signal_tier", "") or "").strip()
        if not a_output or len(a_output) < 10:
            continue

        for j, b in enumerate(papers):
            if i >= j:
                continue
            b_input = (b.get("causal_input", "") or "").lower().strip()
            b_class = (b.get("signal_tier", "") or "").strip()
            if not b_input or len(b_input) < 10:
                continue

            # Skip same-tier papers (less interesting)
            if a_class == b_class:
                continue

            a_words = set(a_output.split())
            b_words = set(b_input.split())
            overlap = a_words & b_words
            if len(overlap) < 3:
                continue

            similarity = len(overlap) / max(1, min(len(a_words), len(b_words)))
            if similarity < min_similarity:
                continue

            connections.append({
                "paper_a_doi": a["doi"],
                "paper_a_title": a["title"],
                "paper_a_output": a_output[:150],
                "paper_b_doi": b["doi"],
                "paper_b_title": b["title"],
                "paper_b_input": b_input[:150],
                "shared_terms": ", ".join(sorted(overlap)[:8]),
                "similarity": round(similarity, 3),
            })

    connections.sort(key=lambda x: x["similarity"], reverse=True)
    seen = set()
    unique = []
    for c in connections:
        key = (c["paper_a_doi"], c["paper_b_doi"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:15]


# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge Gap Audit (pure symbolic kernel)
# ═══════════════════════════════════════════════════════════════════════════════

def run_gap_audit(db) -> dict:
    """Audit the research map: what do we NOT know?

    Returns gaps across four dimensions:
    - concepts lacking evidence
    - constraints never verified
    - framework predictions untested
    - research map blind spots
    """
    # Concepts without supporting papers
    concepts = db.get_concept_momentum(min_appearances=1)
    weak_concepts = [c for c in concepts if c["appearances"] <= 1]
    orphan_concepts = [c for c in concepts
                       if not c.get("opportunity") or not c.get("connection")]

    # Untested constraints
    constraints = db.get_constraints()
    verification_summary = db.get_verification_summary()
    untested = [c for c in constraints
                if not any(v.get("constraint_name") == c.get("name")
                           for v in db.get_verifications())]

    # Framework predictions without experiments
    frameworks = db.get_frameworks()
    untested_frameworks = []
    for fw in frameworks:
        experiments = db.get_experiments(framework_name=fw.get("framework_name", ""))
        if not experiments:
            untested_frameworks.append({
                "framework": fw.get("framework_name", ""),
                "prediction_power": fw.get("predictive_power", 0),
                "actionability": fw.get("actionability", 0),
            })

    # Missing research map coverage
    all_causal = db.get_papers_with_causal(days_back=30)
    covered_patterns = set()
    for p in all_causal:
        t = (p.get("causal_transformation") or "").lower()
        if "amplif" in t: covered_patterns.add("Signal Amplification")
        if "energy" in t or "transduc" in t: covered_patterns.add("Energy Transduction")
        if "structur" in t or "assembl" in t or "configur" in t: covered_patterns.add("Structural Reconfiguration")
        if "switch" in t or "select" in t or "gating" in t: covered_patterns.add("Pathway Switching/Gating")

    all_patterns = {"Signal Amplification", "Energy Transduction",
                    "Structural Reconfiguration", "Pathway Switching/Gating"}
    missing_patterns = all_patterns - covered_patterns

    return {
        "concept_gaps": {
            "weak_concepts": [{"name": c["name"], "appearances": c["appearances"]}
                              for c in weak_concepts[:5]],
            "orphan_concepts": [{"name": c["name"]}
                                for c in orphan_concepts[:5]],
        },
        "constraint_gaps": {
            "total": len(constraints),
            "untested": len(untested),
            "verified_supported": verification_summary["supported"],
            "verified_violated": verification_summary["violated"],
        },
        "framework_gaps": {
            "untested_predictions": len(untested_frameworks),
            "frameworks_needing_experiments": untested_frameworks[:5],
        },
        "coverage_gaps": {
            "missing_logic_patterns": list(missing_patterns),
        },
    }
