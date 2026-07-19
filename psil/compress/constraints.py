"""
Constraint Discovery Engine.

Core principle:
- Patterns describe. Constraints explain. Predictions validate.
- A framework that explains everything but predicts nothing is weak.

Pipeline: Framework → Constraints → Predictions → Experiments
"""

CONSTRAINT_DISCOVERY_PROMPT = """You are a scientific constraint discoverer. Your job is NOT to describe what a framework explains. Your job is to find what a framework FORBIDS or REQUIRES.

Scientific theories are valuable because they introduce CONSTRAINTS — they say "X requires Y" or "X cannot happen without Y."

## Framework to analyze:
NAME: {framework_name}
DESCRIPTION: {description}
CORE LOGIC: {core_logic}
COVERED PATTERNS: {covered_patterns}

## Task

### Step 1: Find Constraints

For every condition shared by ALL supporting evidence, generate a constraint.

Formats to use:
- X requires Y
- X cannot occur without Y
- If Y increases, X should increase
- If Y decreases, X should decrease

CRITICAL: A constraint must be FALSIFIABLE. If you cannot imagine an experiment that would disprove it, the constraint is too weak.

For each constraint, output:

CONSTRAINT_NAME: <short label, max 8 words>
CONSTRAINT_TYPE: <requires | cannot_occur_without | scales_with | threshold | other>
STATEMENT: <one clear sentence stating the constraint>
SUPPORTING_EVIDENCE: <which papers/patterns support this?>
VIOLATING_EXAMPLES: <any known counterexamples? or "None yet">
CONFIDENCE: <0-10, how well-supported is this?>
PREDICTION_POWER: <0-10, how many testable predictions does this generate?>
ACTIONABILITY: <0-10, can we directly test this?>

### Step 2: Generate Predictions from Each Constraint

For each constraint, generate:

POSITIVE_PREDICTION: <if constraint is correct, what MUST happen?>
NEGATIVE_PREDICTION: <if constraint is violated, what MUST happen?>
BOUNDARY_PREDICTION: <where does the constraint FAIL?>
EXPECTED_RELATIONSHIP: <quantitative relationship, e.g. "bell-shaped curve", "linear increase", "threshold behavior">
FAILURE_CONDITION: <what experimental result would FALSIFY this framework?>

### Step 3: Design Experiments

For the strongest predictions, design experiments:

EXPERIMENT_NAME: <short label>
MANIPULATE_VARIABLE: <independent variable to change>
MEASURE_VARIABLE: <dependent variable to measure>
EXPECTED_RESULT: <quantitative relationship expected>
FAILURE_CONDITION: <result that falsifies the framework>
PRIORITY: <high | medium | low>

---

IMPORTANT RULES:
1. A constraint saying "everything is information processing" is WORTHLESS. Be specific.
2. Every constraint MUST be falsifiable.
3. If a framework cannot generate at least one falsifiable constraint, say so honestly.
4. Do not invent constraints without evidence in the supporting papers.
"""


def discover_constraints(framework: dict, patterns: list[dict], llm_client, db) -> list[dict]:
    """Run constraint discovery for a single framework."""
    prompt = CONSTRAINT_DISCOVERY_PROMPT.format(
        framework_name=framework.get("framework_name", ""),
        description=framework.get("description", ""),
        core_logic=framework.get("core_logic", ""),
        covered_patterns=framework.get("covered_patterns", ""),
    )

    try:
        resp = llm_client.client.chat.completions.create(
            model=llm_client.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2500,
        )
        content = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  Constraint discovery LLM error: {e}")
        return []

    constraints = _parse_constraints(content, framework.get("framework_name", ""), db)
    return constraints


def _parse_constraints(text: str, framework_name: str, db) -> list[dict]:
    """Parse LLM constraint discovery output and store in DB."""
    constraints = []
    current_constraint = None
    current_prediction = {}
    prediction_type = ""

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if ":" in line and not line.startswith("-"):
            key, _, value = line.partition(":")
            key = key.strip().lower().replace(" ", "_").strip("*").strip("_")
            value = value.strip().strip("*").strip()

            if key == "constraint_name":
                if current_constraint and current_constraint.get("statement"):
                    _save_constraint(current_constraint, db)
                    constraints.append(current_constraint)
                current_constraint = {
                    "name": value, "framework_name": framework_name,
                    "statement": "", "constraint_type": "requires",
                    "confidence": 0, "prediction_power": 0, "actionability": 0,
                    "predictions": []
                }

            elif key == "positive_prediction":
                prediction_type = "positive"
                current_prediction = {"type": "positive", "statement": value}
            elif key == "negative_prediction":
                prediction_type = "negative"
                current_prediction = {"type": "negative", "statement": value}
            elif key == "boundary_prediction":
                prediction_type = "boundary"
                current_prediction = {"type": "boundary", "statement": value}

            elif current_constraint and key in ("constraint_type", "statement",
                                                  "supporting_evidence", "violating_examples"):
                current_constraint[key] = value
            elif current_constraint and key == "confidence":
                try: current_constraint["confidence"] = float(value.split("/")[0].split()[0])
                except: pass
            elif current_constraint and key == "prediction_power":
                try: current_constraint["prediction_power"] = float(value.split("/")[0].split()[0])
                except: pass
            elif current_constraint and key == "actionability":
                try: current_constraint["actionability"] = float(value.split("/")[0].split()[0])
                except: pass

            elif current_prediction and key == "expected_relationship":
                current_prediction["expected_relationship"] = value
            elif current_prediction and key == "failure_condition":
                current_prediction["failure_condition"] = value

            elif key == "experiment_name":
                if current_constraint and current_prediction.get("statement"):
                    current_constraint.setdefault("predictions", []).append(current_prediction)
                    current_prediction = {"type": prediction_type, "statement": ""}
                # Store experiment directly
                exp_failure = ""
                if current_prediction.get("failure_condition"):
                    exp_failure = current_prediction["failure_condition"]
                db.insert_experiment(
                    prediction_id=0, framework_name=framework_name,
                    manipulate="", measure="", expected=value, failure=exp_failure,
                )
                current_prediction = {}
                prediction_type = ""

            elif key in ("manipulate_variable", "measure_variable", "expected_result",
                         "failure_condition", "priority"):
                # Accumulate experiment fields
                if not hasattr(_parse_constraints, '_exp_fields'):
                    _parse_constraints._exp_fields = {}
                _parse_constraints._exp_fields[key] = value

    # Save last constraint
    if current_constraint and current_constraint.get("statement"):
        if current_prediction.get("statement"):
            current_constraint.setdefault("predictions", []).append(current_prediction)
        _save_constraint(current_constraint, db)
        constraints.append(current_constraint)

    return constraints


def _save_constraint(c: dict, db):
    """Save constraint + its predictions to DB."""
    constraint_id = db.insert_constraint(
        name=c.get("name", ""),
        framework_name=c.get("framework_name", ""),
        statement=c.get("statement", ""),
        constraint_type=c.get("constraint_type", "requires"),
        supporting_evidence=c.get("supporting_evidence", ""),
        violating_examples=c.get("violating_examples", ""),
        confidence=c.get("confidence", 0),
        prediction_power=c.get("prediction_power", 0),
        actionability=c.get("actionability", 0),
    )
    for pred in c.get("predictions", []):
        if pred.get("statement"):
            db.insert_prediction(
                constraint_id=0,  # We don't have the real ID after INSERT
                prediction_type=pred.get("type", ""),
                statement=pred.get("statement", ""),
                expected_relationship=pred.get("expected_relationship", ""),
                failure_condition=pred.get("failure_condition", ""),
            )


def run_constraint_discovery(frameworks: list[dict], patterns: list[dict],
                              llm_client, db) -> dict:
    """Run constraint discovery on all frameworks."""
    all_constraints = []
    for fw in frameworks:
        constraints = discover_constraints(fw, patterns, llm_client, db)
        all_constraints.extend(constraints)
    return {
        "constraints": all_constraints,
        "total": len(all_constraints),
    }
