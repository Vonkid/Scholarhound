import re

from openai import OpenAI

from psil.store.models import Paper
from psil.rank.identity import load_identity, ResearchIdentity
from psil.rank.concepts import (
    check_personal_legacy,
    check_future_trajectory,
    PERSONAL_LEGACY_SET,
    FUTURE_TRAJECTORY_SET,
)
from psil.kernel import apply_paper_type_router

RANKING_PROMPT = """You are a scientific reasoning assistant. Classify papers the way a principal investigator reads literature: by problem class, novelty type, strategic value, and concept support. Do NOT start from keyword matching.

FORMAT INSTRUCTION: Use PLAIN TEXT only. No markdown. Field names followed by colon and value. Bullet points prefixed with - .

{identity_context}

---
PAPER:
Title: {title}
Journal: {journal}
DOI: {doi}
Abstract: {abstract}

MATCHED SIGNALS: {matched_signals}
PERSONAL LEGACY MATCHES: {personal_legacy}
FUTURE TRAJECTORY MATCHES: {future_trajectory}

---

## Step 1: Paper Type Router

First decide what kind of paper this is. This determines how the paper should be judged.

PAPER_TYPE: <Mechanism Paper | Transduction or Device Paper | Platform or Method Paper | Validation or Benchmark Paper | Clinical or Translation Paper | Review or Synthesis Paper | Computational or Agent Paper | Fundamental Phenomenon Paper | Commentary or Noise | Other Research>

JUDGMENT_MODE: <mechanism_shift | transduction_route | platform_readiness | validation_readiness | translation_readiness | synthesis_map | agent_architecture | phenomenon_watch | screen | general_judgment>

Mode meaning:
- mechanism_shift: judge whether the paper changes a causal/mechanistic model.
- transduction_route: judge whether it creates a believable coupling route from recognition/input to measurable output.
- platform_readiness: judge whether the method/platform can become a reusable experimental scaffold.
- validation_readiness: judge benchmark quality, independent support, and whether the evidence actually strengthens a concept.
- translation_readiness: judge disease relevance, biological context, clinical/sample realism, and failure modes.
- synthesis_map: judge whether it reorganizes the literature map or exposes a missing question.
- agent_architecture: judge whether it improves code/kernel/agent harness design.
- phenomenon_watch: judge whether a fundamental physical/chemical phenomenon may become useful later.
- screen: judge only whether the item should be ignored, filed, or watched.

## Step 2: Problem Class

Classify what type of scientific problem this paper solves:

PROBLEM_CLASS: <Sensing | Treatment | Delivery | Disease Model | Biological Mechanism | Material Platform | Device/Electronics | Manufacturing/Fabrication | Fundamental Photophysics | Fundamental Chemistry | Data/AI/Computational | Review/Perspective/Commentary | Other>

## Step 3: Novelty Type

What is actually new here?

NOVELTY_TYPE: <New Mechanism | New Method | New Transduction Principle | New Platform | New Combination | New Application | New Validation | New Scale/Throughput | Incremental Optimization | Mostly Review/Commentary>

## Step 4: Evidence Type

What TYPE of evidence does this paper contribute? Pick the best label.

EVIDENCE_TYPE: <Mechanistic Evidence | Validation Evidence | Clinical Evidence | Engineering Evidence | Translational Evidence | Constraint Evidence | Failure Evidence | Benchmark Evidence | Emerging Signal>

Guide:
- Mechanistic = reveals how something works at molecular/physical level
- Validation = independently confirms an existing finding
- Clinical = demonstrates performance in patient samples or human data
- Engineering = improves device/platform performance or scalability
- Translational = bridges lab finding toward clinical application
- Constraint = provides evidence for or against a scientific constraint
- Failure = negative result or falsification, valuable for ruling out hypotheses
- Benchmark = establishes a performance standard or comparison baseline
- Emerging Signal = early hint, not yet conclusive

## Step 5: Strategic Value

What higher-level scientific idea does this paper support? Pick ONE best label.

STRATEGIC_VALUE: <New Transduction Principle | Functional EV Phenotyping | Single-Entity Resolution | Photon Utilization | Nanostructure as Active Optical Participant | Mechanobiology-Enabled Sensing | Molecular Bioelectronics | Organoid-Derived Readout | Adaptive Biointerface | Activity-Based Biomarker | Disease-Relevant Functional Readout | Platform Engineering | Translation/Clinical Readiness | Useful Experimental Template | Low Strategic Value>

## Step 6: Concept Support

Which existing or emerging concept does this paper support?

Existing concepts in researcher's map:
Functional EV Phenotyping, Photon Utilization, Nanostructure as Active Optical Participant, Molecular Bioelectronics, Mechanobiology-Enabled Sensing, Organoid+EV+Sensing, Nanophotonics-Enabled Photochemistry, Activity-Based Biomarkers, Small-Molecule Bioelectronic Recognition, Adaptive Biointerfaces

Emerging concepts:
Radiative Q-factor Modulation, Exceptional Point Sensing, Digital Droplet EV Decoding, Living Bioelectronic Transducer, Single-EV Multi-Biomarker Decoding, Activity-Affinity EV Assay, BIC Metasurface Biosensing

CONCEPT_SUPPORT_NAME: <concept name, or "None">
SUPPORT_TYPE: <Discovery | Validation | Extension | Weak Signal>
EVIDENCE_STRENGTH: <Low | Medium | High>

Rules:
- Discovery = genuinely new concept not in researcher's map
- Validation = independently supports an existing concept
- Extension = expands existing concept to new application/modality
- Weak Signal = lightly touches a concept without strong evidence

## Step 7: Scoring

RELEVANCE: <0-10>
How close to researcher's CORE? 9-10: OECT+organoid/EV/mechanobiology/photochemistry. 7-8: OECT+biosensing, nanophotonics+biosensing. 5-6: general OECT, bioelectronics. 3-4: passive nanomaterials, generic biosensing. 1-2: generic PDT/PTT, routine delivery. 0: no connection. Distinguish active mediators (nanozyme, plasmonic catalyst, energy converter) from passive carriers.

NOVELTY: <0-10>
9-10: new physical principle, paradigm-shifting. 7-8: new modality combination, field-crossing. 5-6: clever engineering. 3-4: incremental. 1-2: routine. 0: nothing new.

BRIDGE: <0-10>
9-10: photochemistry+nanophotonics, organoids+sensing, EVs+bioelectronics, physics+medicine. 7-8: materials+disease, photonics+biosensing. 5-6: adjacent fields. 3-4: single field, cross-cutting. 1-2: siloed. 0: completely siloed.

TRAJECTORY_INFLUENCE: <0-10>
Could this shape future direction? 9-10: reshape trajectory. 7-8: strong future direction. 5-6: incremental influence. 3-4: weak. 1-2: minimal. 0: none.

CONCEPT_SUPPORT: <0-10>
0-2: no support for research map. 3-5: weak support for adjacent concept. 6-7: good support for existing/emerging concept. 8-10: strong validation of core concept or discovery of trajectory-shaping concept.

Estimate FINAL_SCORE using the selected JUDGMENT_MODE. The kernel will recompute it deterministically.

Mode weights:
- general_judgment: 0.25 R + 0.20 N + 0.20 B + 0.20 T + 0.15 CS
- mechanism_shift: 0.15 R + 0.30 N + 0.20 B + 0.25 T + 0.10 CS
- transduction_route: 0.25 R + 0.20 N + 0.25 B + 0.20 T + 0.10 CS
- platform_readiness: 0.20 R + 0.20 N + 0.20 B + 0.25 T + 0.15 CS
- validation_readiness: 0.25 R + 0.10 N + 0.15 B + 0.20 T + 0.30 CS
- translation_readiness: 0.30 R + 0.10 N + 0.20 B + 0.20 T + 0.20 CS
- synthesis_map: 0.20 R + 0.10 N + 0.25 B + 0.30 T + 0.15 CS
- agent_architecture: 0.25 R + 0.20 N + 0.20 B + 0.25 T + 0.10 CS
- phenomenon_watch: 0.10 R + 0.30 N + 0.25 B + 0.25 T + 0.10 CS
- screen: 0.20 R + 0.05 N + 0.15 B + 0.15 T + 0.45 CS

FINAL_SCORE: <0-10, one decimal>

## Step 8: Classification

Classify using these rules (check in order):

HIGH_PRIORITY: FINAL_SCORE ≥ 8.0 OR (NOVELTY ≥ 9 AND BRIDGE ≥ 8) OR (BRIDGE ≥ 9 AND TRAJECTORY_INFLUENCE ≥ 8)
IMPORTANT: FINAL_SCORE 6.0-7.9 OR (TRAJECTORY_INFLUENCE ≥ 8 AND FINAL_SCORE ≥ 5.5) OR (NOVELTY ≥ 8 AND BRIDGE ≥ 7)
POTENTIAL: FINAL_SCORE 4.0-5.9
WATCHLIST: FINAL_SCORE < 4.0 BUT matches Personal Legacy or Future Trajectory topic
LOW_PRIORITY: FINAL_SCORE < 4.0, no legacy/trajectory match
COMMENTARY: Review, News & Views, Perspective, Editorial (primary research = false)
IGNORE: corrections, errata, non-research, outside science/engineering

SIGNAL_TIER: <HIGH_PRIORITY | IMPORTANT | POTENTIAL | WATCHLIST | LOW_PRIORITY | COMMENTARY | IGNORE>

## Step 9: Analysis

WHY_MATTERS: (3-5 bullet points, each prefixed with - )
POTENTIAL_CONNECTION: (2-5 bullet points, each prefixed with - . IMPORTANT: when proposing integration, explicitly state the coupling route — redox-active product, ionic modulation, surface charge modulation, capacitance modulation, impedance modulation, refractive index modulation, Q-factor modulation, fluorescence/optical coupling, mechanical deformation, enzymatic product transduction. If no specific mechanism, write "Connection is conceptual rather than directly implementable.")
WEAKNESS: (1-2 sentences)
ACTION: (one of: "Read immediately", "Review this week", "Archive and revisit", "Watch for developments", "File only", or "Skip")

## Step 10: Paper Workflow Extraction

For HIGH/IMPORTANT papers, extract the full paper logic chain:

RESEARCH_QUESTION: (the specific question this paper tries to answer, one sentence)
HYPOTHESIS: (what did the authors expect to find? one sentence, or "Not stated")
EXPERIMENTAL_DESIGN: (how was the study structured? e.g. \"case-control\", \"dose-response\", \"comparative analysis\", one phrase)
KEY_METHOD: (the single most critical technique or method used, one phrase)
KEY_RESULT: (the headline quantitative or qualitative finding, one sentence)
WORKFLOW_GAP: (what is still missing before this can be applied? one sentence, or "None")

## Step 11: Causal Extraction

QUESTION: (what problem? one sentence)
CONSTRAINT: (what limitation? one sentence)
INPUT_STATE: (initial state, one phrase)
MODIFIER: (qualifier on input, e.g. \"longitudinal\", \"in vivo\", or \"None\")
TRANSFORMATION: (mechanism, one phrase)
OUTPUT_STATE: (resulting state, one phrase)
CONTEXT: (experimental context on output, e.g. \"in mouse model\", or \"None\")
OUTCOME: (practical consequence, one sentence)

## Step 12: Concept Extraction

CONCEPT_NAME: (short label, max 8 words, or "None")
CONCEPT_WHY_MATTERS: (1-2 sentences)
CONCEPT_CURRENT_CONNECTION: (connection to researcher's current work?)
CONCEPT_POTENTIAL_CONNECTION: (future connection?)
CONCEPT_MISSING_LINK: (what capability/method/knowledge bridges the gap?)
CONCEPT_OPPORTUNITY: (what new research direction/experiment?)
CONCEPT_ACTION: ("Add to dictionary" | "Watch only" | "Ignore for now")

Format each field EXACTLY as: FIELD_NAME: value
For bullet-point fields, start each bullet with - on a new line."""


VALIDATE_PROMPT = """You are a scientific critic. Check the Extractor's analysis for errors, overconfidence, and missed connections.

PAPER: {title} ({journal}, {doi})
ABSTRACT: {abstract}

EXTRACTOR OUTPUT:
Paper Type: {paper_type}
Judgment Mode: {judgment_mode}
Problem Class: {problem_class}
Novelty Type: {novelty_type}
Strategic Value: {strategic_value}
Concept Support: {concept_support} ({support_type}, {evidence})
Scores: R={relevance} N={novelty} B={bridge} T={trajectory} CS={concept_support_score}
Why it matters: {why_matters}

TASKS:
1. Flag any score that seems too high or too low
2. Check if Paper Type and Judgment Mode are appropriate
3. Check if Strategic Value label is appropriate
4. Identify missed concept connections
5. Rate analysis quality

VALIDATION_RESULT: <Good | Acceptable | Needs Revision>
CRITIQUE: <1-3 issues or confirmations>
ADJUSTED_PAPER_TYPE: <same or corrected>
ADJUSTED_JUDGMENT_MODE: <same or corrected>
ADJUSTED_RELEVANCE: <same or corrected>
ADJUSTED_NOVELTY: <same or corrected>
ADJUSTED_TRAJECTORY: <same or corrected>
ADJUSTED_CS: <same or corrected>
MISSED_CONNECTION: <concept missed, or None>
FINAL_STRATEGIC_VALUE: <confirm or suggest better>
"""


def build_ranking_prompt(paper: Paper, identity: ResearchIdentity,
                          matched_signals: str = "",
                          personal_legacy: str = "",
                          future_trajectory: str = "") -> str:
    return RANKING_PROMPT.format(
        identity_context=identity.to_prompt_context(),
        title=paper.title,
        journal=paper.journal,
        doi=paper.doi,
        abstract=paper.abstract,
        matched_signals=matched_signals or "None",
        personal_legacy=personal_legacy or "None",
        future_trajectory=future_trajectory or "None",
    )


def parse_llm_response(text: str) -> dict:
    fields = {}
    current_key = None
    for line in text.strip().split("\n"):
        line_stripped = line.strip()
        # Strip markdown bold markers
        if line_stripped.startswith("**") and line_stripped.endswith("**"):
            line_stripped = line_stripped[2:-2]
        elif line_stripped.startswith("**"):
            line_stripped = line_stripped[2:]
        # Handle bold closing on same line
        line_stripped = line_stripped.rstrip("*")

        # Skip empty lines and lines without colon or equals
        if not line_stripped:
            continue

        # Check if this is a key-value line (contains : or = with a field name)
        colon_pos = line_stripped.find(":")
        equals_pos = line_stripped.find("=")

        # Determine separator position: use colon if present, otherwise equals
        sep_pos = colon_pos if colon_pos >= 0 else equals_pos

        if sep_pos >= 0 and not line_stripped.startswith("-"):
            key = line_stripped[:sep_pos].strip().lower().replace(" ", "_").replace("*", "")
            value = line_stripped[sep_pos+1:].strip().rstrip("*")
            # Skip if key doesn't look like a field name
            if key and not key.startswith("-") and len(key) < 60:
                if value:
                    fields[key] = value
                    current_key = key
                else:
                    current_key = key
                    fields[current_key] = ""
        elif line_stripped.startswith("-") and current_key:
            fields[current_key] = fields.get(current_key, "") + "\n" + line_stripped

    def parse_score(key: str, default: int = 0) -> int:
        val = fields.get(key, str(default))
        try:
            return int(float(val.split("/")[0]))
        except (ValueError, AttributeError):
            return default

    def parse_final_score(key: str) -> float:
        val = fields.get(key, "0")
        try:
            return float(val.split("/")[0])
        except (ValueError, AttributeError):
            return 0.0

    relevance = parse_score("relevance")
    novelty = parse_score("novelty")
    bridge = parse_score("bridge")
    trajectory = parse_score("trajectory_influence")
    if trajectory == 0:
        trajectory = parse_score("trajectory")
    concept_support = parse_score("concept_support")
    signal_tier = fields.get("signal_tier", "LOW_PRIORITY").strip().rstrip("*")
    signal_tier = signal_tier.upper().replace(" ", "_").replace("*", "")
    if "HIGH" in signal_tier:
        signal_tier = "HIGH_PRIORITY"
    elif "IMPORTANT" in signal_tier:
        signal_tier = "IMPORTANT"
    elif "POTENTIAL" in signal_tier:
        signal_tier = "POTENTIAL"
    elif "WATCHLIST" in signal_tier:
        signal_tier = "WATCHLIST"
    elif "LOW" in signal_tier:
        signal_tier = "LOW_PRIORITY"
    elif "COMMENTARY" in signal_tier:
        signal_tier = "COMMENTARY"
    elif "IGNORE" in signal_tier:
        signal_tier = "IGNORE"

    # Classification fields
    paper_type = fields.get("paper_type", "").strip().rstrip("*")
    judgment_mode = fields.get("judgment_mode", "").strip().rstrip("*")
    problem_class = fields.get("problem_class", "").strip().rstrip("*")
    novelty_type = fields.get("novelty_type", "").strip().rstrip("*")
    evidence_type = fields.get("evidence_type", "").strip().rstrip("*")
    strategic_value = fields.get("strategic_value", "").strip().rstrip("*")
    concept_support_name = fields.get("concept_support_name", "").strip().rstrip("*")
    support_type = fields.get("support_type", "").strip().rstrip("*")
    evidence_strength = fields.get("evidence_strength", "").strip().rstrip("*")

    # Preserve the extractor's final score estimate for audit; kernel recomputes below.
    # Pattern: "FINAL_SCORE = ... = X.X" or "FINAL_SCORE: X.X"
    final_score_from_text = 0.0
    final_score_key = fields.get("final_score", "")
    if not final_score_key:
        # Try the raw text for patterns like "= 8.2**"
        import re as _re
        score_match = _re.search(r'(?:FINAL_SCORE|final_score)\s*[=:]\s*[\d.+×\s]+=\s*([\d.]+)', text)
        if score_match:
            try:
                final_score_from_text = float(score_match.group(1))
            except ValueError:
                pass
    else:
        try:
            final_score_from_text = float(str(final_score_key).split("/")[0].rstrip("*"))
        except (ValueError, AttributeError):
            pass

    concept_name = fields.get("concept_name", "").strip().rstrip("*")

    # Causal extraction
    causal = {
        "question": fields.get("question", "").strip().rstrip("*"),
        "constraint": fields.get("constraint", "").strip().rstrip("*"),
        "input_state": fields.get("input_state", "").strip().rstrip("*"),
        "modifier": fields.get("modifier", "").strip().rstrip("*"),
        "transformation": fields.get("transformation", "").strip().rstrip("*"),
        "output_state": fields.get("output_state", "").strip().rstrip("*"),
        "context": fields.get("context", "").strip().rstrip("*"),
        "outcome": fields.get("outcome", "").strip().rstrip("*"),
    }

    # Workflow extraction
    workflow = {
        "research_question": fields.get("research_question", "").strip().rstrip("*"),
        "hypothesis": fields.get("hypothesis", "").strip().rstrip("*"),
        "experimental_design": fields.get("experimental_design", "").strip().rstrip("*"),
        "key_method": fields.get("key_method", "").strip().rstrip("*"),
        "key_result": fields.get("key_result", "").strip().rstrip("*"),
        "workflow_gap": fields.get("workflow_gap", "").strip().rstrip("*"),
    }

    result = {
        "signal_tier": signal_tier,
        "paper_type": paper_type,
        "judgment_mode": judgment_mode,
        "relevance": relevance,
        "novelty": novelty,
        "bridge": bridge,
        "trajectory_influence": trajectory,
        "concept_support": concept_support,
        "final_score_llm": round(final_score_from_text, 1),
        "problem_class": problem_class,
        "novelty_type": novelty_type,
        "evidence_type": evidence_type,
        "strategic_value": strategic_value,
        "concept_support_name": concept_support_name,
        "support_type": support_type,
        "evidence_strength": evidence_strength,
        "workflow": workflow,
        "why_matters": fields.get("why_matters", "").strip().rstrip("*"),
        "potential_connection": fields.get("potential_connection", "").strip().rstrip("*"),
        "weakness": fields.get("weakness", "").strip().rstrip("*"),
        "action": fields.get("action", "").strip().rstrip("*"),
        "concept_name": concept_name,
        "concept_why_matters": fields.get("concept_why_matters", "").strip().rstrip("*"),
        "concept_current_connection": fields.get("concept_current_connection", "").strip().rstrip("*"),
        "concept_potential_connection": fields.get("concept_potential_connection", "").strip().rstrip("*"),
        "concept_missing_link": fields.get("concept_missing_link", "").strip().rstrip("*"),
        "concept_opportunity": fields.get("concept_opportunity", "").strip().rstrip("*"),
        "concept_action": fields.get("concept_action", "").strip().rstrip("*"),
        "causal": causal,
    }
    return apply_paper_type_router(result)


class LLMClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def rank(self, paper: Paper, identity: ResearchIdentity,
             matched_signals: str = "") -> dict:
        combined_text = f"{paper.title} {paper.abstract}"
        legacy = check_personal_legacy(combined_text)
        trajectory = check_future_trajectory(combined_text)

        prompt = build_ranking_prompt(
            paper,
            identity,
            matched_signals=matched_signals,
            personal_legacy=", ".join(legacy) if legacy else "",
            future_trajectory=", ".join(trajectory) if trajectory else "",
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200,
        )
        content = resp.choices[0].message.content or ""
        result = parse_llm_response(content)

        # Enforce WATCHLIST floor
        tier = result.get("signal_tier", "").strip().upper()
        if tier in ("LOW_PRIORITY", "IGNORE") and (legacy or trajectory):
            result["signal_tier"] = "WATCHLIST"

        # Enforce concept evidence: discard concepts with no name or flagged None
        cn = result.get("concept_name", "").strip()
        if not cn or cn.lower() == "none":
            result["concept_name"] = ""

        return result

    def validate(self, paper, extractor_result: dict) -> dict:
        """Stage 2: Validator+Critic — checks the Extractor's output.

        Only runs for IMPORTANT+ papers. Returns adjusted scores and critique.
        """
        prompt = VALIDATE_PROMPT.format(
            title=paper.title,
            journal=paper.journal,
            doi=paper.doi,
            abstract=(paper.abstract or "")[:800],
            paper_type=extractor_result.get("paper_type", ""),
            judgment_mode=extractor_result.get("judgment_mode", ""),
            problem_class=extractor_result.get("problem_class", ""),
            novelty_type=extractor_result.get("novelty_type", ""),
            strategic_value=extractor_result.get("strategic_value", ""),
            concept_support=extractor_result.get("concept_support_name", ""),
            support_type=extractor_result.get("support_type", ""),
            evidence=extractor_result.get("evidence_strength", ""),
            relevance=extractor_result.get("relevance", ""),
            novelty=extractor_result.get("novelty", ""),
            bridge=extractor_result.get("bridge", ""),
            trajectory=extractor_result.get("trajectory_influence", ""),
            concept_support_score=extractor_result.get("concept_support", ""),
            why_matters=(extractor_result.get("why_matters", "") or "")[:300],
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600,
            )
            content = resp.choices[0].message.content or ""
        except Exception:
            return {"validation_result": "Acceptable", "critique": "Validator unavailable",
                    "adjustments": {}}

        # Parse validator response
        fields = {}
        for line in content.strip().split("\n"):
            if ":" in line:
                k, _, v = line.partition(":")
                k = k.strip().lower().replace(" ", "_")
                v = v.strip()
                fields[k] = v

        adjustments = {}
        for key in ["adjusted_paper_type", "adjusted_judgment_mode"]:
            if key in fields and fields[key] and fields[key].lower() != "same":
                adjustments[key] = fields[key]
        for key in ["adjusted_relevance", "adjusted_novelty",
                     "adjusted_trajectory", "adjusted_cs"]:
            if key in fields:
                try:
                    adjustments[key] = int(float(fields[key].split("/")[0]))
                except (ValueError, TypeError):
                    pass

        return {
            "validation_result": fields.get("validation_result", "Acceptable"),
            "critique": fields.get("critique", ""),
            "missed_connection": fields.get("missed_connection", ""),
            "final_strategic_value": fields.get("final_strategic_value", ""),
            "adjustments": adjustments,
        }
