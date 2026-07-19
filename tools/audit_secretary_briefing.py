#!/usr/bin/env python3
"""Build Audit Secretary briefing packets from paper items and reader snapshots.

This is an intake/briefing tool. It does not revise beliefs, update confidence,
or commit kernel state. Existing whole-belief relation snapshots are preserved
as legacy reader input; they are not treated as atom-level projections.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_ID = "audit_secretary_briefing_schema_v1"
SCHEMA_VERSION = "1.1.0-draft"
SCRIPT_VERSION = "audit_secretary_briefing.py:0.2.0"


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_items(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.get("items") or data.get("data") or []
        if not items and data.get("title"):
            items = [data]
    else:
        items = data
    if not isinstance(items, list):
        raise ValueError(f"{path} does not contain a list of items")
    return [item for item in items if isinstance(item, dict)]


def load_relation_rows(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("rows") or data.get("items") or data.get("data") or []
    else:
        rows = data
    return {
        str(row.get("id")): row
        for row in rows
        if isinstance(row, dict) and row.get("id")
    }


def load_atoms(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(belief.get("belief_id")): belief
        for belief in data.get("beliefs", [])
        if isinstance(belief, dict) and belief.get("belief_id")
    }


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [part.strip() for part in parts if len(part.strip()) > 20]


def first_sentence(text: str, fallback: str = "") -> str:
    sentences = split_sentences(text)
    return sentences[0] if sentences else fallback


def contains_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(term in low for term in terms)


def infer_study_type(title: str, abstract: str) -> str:
    blob = f"{title} {abstract}".lower()
    if contains_any(blob, ["protocol", "assessing"]):
        return "protocol"
    if contains_any(blob, ["review", "recent progress", "overview", "perspective"]):
        return "review"
    if contains_any(blob, ["finite element", "computer model", "modeling", "simulation", "comsol"]):
        return "modeling"
    if contains_any(blob, ["dataset", "database"]):
        return "dataset"
    if contains_any(blob, ["we report", "we show", "we demonstrate", "we found", "we investigated", "we propose", "herein", "this study", "studied"]):
        return "primary_research"
    return "unknown"


def choose_sentence(sentences: list[str], terms: list[str], fallback: str = "") -> str:
    for sentence in sentences:
        if contains_any(sentence, terms):
            return sentence
    return fallback or (sentences[0] if sentences else "")


def classify_claim(sentence: str) -> tuple[str, str]:
    low = sentence.lower()
    if contains_any(low, ["limitation", "challenge", "however", "limited", "remains"]):
        claim_type = "limitation"
    elif contains_any(low, ["mechanism", "via", "through", "attributed", "pathway", "absorption", "transduction"]):
        claim_type = "mechanism"
    elif contains_any(low, ["method", "protocol", "model", "developed", "fabricated", "synthesized"]):
        claim_type = "method"
    elif contains_any(low, ["therapy", "diagnosis", "hyperthermia", "sensing", "application", "treatment", "tumor", "cancer"]):
        claim_type = "application"
    elif contains_any(low, ["compared", "relative to", "whereas", "than "]):
        claim_type = "comparison"
    elif contains_any(low, ["may", "might", "could", "potential"]):
        claim_type = "speculation"
    else:
        claim_type = "result"

    if contains_any(low, ["finite element", "model", "simulation", "comsol"]):
        basis = "modeling"
    elif contains_any(low, ["review", "summarize", "overview"]):
        basis = "reviewed_literature"
    elif contains_any(low, ["we show", "we demonstrate", "we found", "measured", "experiment", "in vitro", "ex vivo", "in vivo"]):
        basis = "direct_experiment"
    elif contains_any(low, ["suggest", "propose", "attribute", "hypothesize"]):
        basis = "author_interpretation"
    elif claim_type == "speculation":
        basis = "inference"
    else:
        basis = "author_interpretation"
    return claim_type, basis


def build_claim_map(abstract: str, max_claims: int = 5) -> list[dict[str, Any]]:
    sentences = split_sentences(abstract)
    scored: list[tuple[int, str]] = []
    for idx, sentence in enumerate(sentences):
        score = 0
        if contains_any(sentence, ["we show", "we demonstrate", "we found", "we report", "we propose"]):
            score += 3
        if contains_any(sentence, ["mechanism", "via", "through", "attributed", "absorption"]):
            score += 2
        if contains_any(sentence, ["however", "limited", "challenge", "remains"]):
            score += 2
        if re.search(r"\d", sentence):
            score += 1
        scored.append((score * 1000 - idx, sentence))
    selected = [sentence for _score, sentence in sorted(scored, reverse=True)[:max_claims]]
    claims = []
    for idx, sentence in enumerate(selected, 1):
        claim_type, basis = classify_claim(sentence)
        claims.append(
            {
                "claim_id": f"claim_{idx:02d}",
                "claim_text": sentence,
                "claim_type": claim_type,
                "evidence_basis": basis,
                "evidence_spans": [sentence[:300]],
                "scope": infer_scope(sentence),
                "caveat": infer_caveat(sentence),
                "reader_agreement": "unclear",
            }
        )
    return claims


def infer_scope(sentence: str) -> str:
    low = sentence.lower()
    if contains_any(low, ["cell", "cells", "in vitro"]):
        return "cell-level experiment"
    if contains_any(low, ["xenograft", "tumor", "mouse", "mice", "in vivo"]):
        return "in vivo or tumor model"
    if contains_any(low, ["model", "simulation", "finite element"]):
        return "modeling or simulation"
    if contains_any(low, ["review", "summarize"]):
        return "reviewed literature"
    return "abstract-level claim"


def infer_caveat(sentence: str) -> str:
    low = sentence.lower()
    if contains_any(low, ["may", "might", "could", "potential"]):
        return "Speculative wording in abstract."
    if contains_any(low, ["however", "limited", "challenge", "remains"]):
        return "Limitation or uncertainty is explicit in the sentence."
    return ""


def build_mechanism_map(abstract: str) -> dict[str, Any]:
    sentences = split_sentences(abstract)
    mechanism_sentences = [
        sentence
        for sentence in sentences
        if contains_any(
            sentence,
            ["mechanism", "via", "through", "attributed", "absorption", "joule", "electrophoretic", "transduction", "pathway"],
        )
    ]
    if not mechanism_sentences:
        return {
            "mechanism_status": "not_addressed",
            "input": "",
            "intermediate_steps": [],
            "output": "",
            "entities": infer_entities(abstract),
            "transduction_chain": [],
            "alternative_explanations": [],
            "mechanism_confidence": "low",
            "mechanism_caveat": "No explicit mechanism sentence was detected in the abstract-level input.",
        }

    first = mechanism_sentences[0]
    low = first.lower()
    status = "proposed"
    if contains_any(low, ["we show", "we demonstrate", "measured", "conclusive", "found"]):
        status = "demonstrated"
    if contains_any(low, ["model", "simulation", "finite element"]):
        status = "modeled"
    if contains_any(low, ["not", "negligible", "solely attributable", "artifact"]):
        status = "challenged"

    alternatives = [
        sentence
        for sentence in mechanism_sentences
        if contains_any(sentence, ["joule", "electrophoretic", "artifact", "medium", "not to", "negligible"])
    ]
    return {
        "mechanism_status": status,
        "input": infer_input(first),
        "intermediate_steps": mechanism_sentences[:3],
        "output": infer_output(first),
        "entities": infer_entities(abstract),
        "transduction_chain": mechanism_sentences[:3],
        "alternative_explanations": alternatives[:3],
        "mechanism_confidence": "medium" if status in {"demonstrated", "modeled", "challenged"} else "low",
        "mechanism_caveat": "Mechanism extracted from abstract-level sentences; full text may change scope.",
    }


def infer_input(sentence: str) -> str:
    if contains_any(sentence, ["radiofrequency", "rf"]):
        return "radiofrequency field"
    if contains_any(sentence, ["near-infrared", "nir", "laser"]):
        return "optical or near-infrared stimulus"
    if contains_any(sentence, ["gate", "electrochemical"]):
        return "electrochemical or ionic input"
    return ""


def infer_output(sentence: str) -> str:
    if contains_any(sentence, ["heat", "thermal", "hyperthermia", "temperature"]):
        return "heat or temperature change"
    if contains_any(sentence, ["current", "signal", "readout"]):
        return "electronic or analytical signal"
    if contains_any(sentence, ["cell death", "cytotoxic", "therapy"]):
        return "biological or therapeutic effect"
    return ""


def infer_entities(text: str) -> list[str]:
    entities = []
    patterns = [
        ("gold nanoparticles", ["gold nanoparticle", "gold nanoparticles", "aunps", "gnps"]),
        ("radiofrequency field", ["radiofrequency", "radio-frequency", "rf field", "rf "]),
        ("OECT", ["oect", "organic electrochemical transistor"]),
        ("extracellular vesicles", ["extracellular vesicle", "extracellular vesicles", " ev ", "evs"]),
        ("organoid", ["organoid", "organoids"]),
        ("sensor", ["sensor", "biosensor", "sensing"]),
    ]
    low = f" {text.lower()} "
    for label, keys in patterns:
        if any(key in low for key in keys):
            entities.append(label)
    return entities


def build_evidence_map(title: str, abstract: str) -> dict[str, Any]:
    study_type = infer_study_type(title, abstract)
    low = f"{title} {abstract}".lower()
    primary = "secondary" if study_type == "review" else "primary" if study_type in {"primary_research", "method", "modeling", "protocol"} else "unknown"
    strength_inputs = {
        "study_type": study_type,
        "abstract_level_only": True,
        "has_quantitative_readout": bool(re.search(r"\d", abstract)),
        "reports_lod": contains_any(low, ["limit of detection", "lod", "detection limit"]),
        "controls_mentioned": controls_are_reported(low),
    }
    return {
        "source_type": "review" if study_type == "review" else "paper",
        "primary_or_secondary": primary,
        "controls_reported": yes_no_unclear(strength_inputs["controls_mentioned"]),
        "quantitative_readout": yes_no_unclear(strength_inputs["has_quantitative_readout"]),
        "reports_lod": yes_no_unclear(strength_inputs["reports_lod"]),
        "sample_context": infer_sample_context(abstract),
        "validation_context": infer_validation_context(abstract),
        "replication_or_independent_validation": yes_no_unclear(contains_any(low, ["validated", "independent", "replicated"])),
        "evidence_strength_inputs": strength_inputs,
        "evidence_strength_rule_result": "not_computed",
    }


def yes_no_unclear(value: bool) -> str:
    return "yes" if value else "unclear"


def controls_are_reported(text: str) -> bool:
    return bool(
        re.search(
            r"\b(control group|controlled experiment|negative control|positive control|blank control|without [^.]{0,80}(nanoparticles|treatment|target|analyte)|untreated|sham)\b",
            text,
            flags=re.I,
        )
    )


def infer_sample_context(abstract: str) -> str:
    low = abstract.lower()
    if contains_any(low, ["clinical", "patient", "plasma", "serum", "blood"]):
        return "clinical or biofluid samples mentioned"
    if contains_any(low, ["cell", "in vitro"]):
        return "cell or in vitro context mentioned"
    if contains_any(low, ["xenograft", "mouse", "mice", "in vivo"]):
        return "animal or in vivo context mentioned"
    return "not clear from abstract"


def infer_validation_context(abstract: str) -> str:
    low = abstract.lower()
    if contains_any(low, ["validated", "validation", "clinical", "patient"]):
        return "validation mentioned in abstract"
    if contains_any(low, ["model", "simulation"]):
        return "modeling validation unclear"
    return "not clear from abstract"


def build_uncertainty_map(abstract: str, relation_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    uncertainties = [
        {
            "uncertainty_id": "uncertainty_01",
            "uncertainty_type": "insufficient_input_text",
            "description": "Briefing was generated from abstract/title-level input, so full methods and controls may be missing.",
            "severity": "medium",
            "what_would_reduce_it": "Read the full text and methods.",
        }
    ]
    mechanism = build_mechanism_map(abstract)
    if mechanism["mechanism_status"] == "not_addressed":
        uncertainties.append(
            {
                "uncertainty_id": f"uncertainty_{len(uncertainties)+1:02d}",
                "uncertainty_type": "mechanism_gap",
                "description": "No explicit mechanism was detected in the abstract-level input.",
                "severity": "medium",
                "what_would_reduce_it": "Check full text for mechanism experiments or alternative explanations.",
            }
        )
    if relation_row and len(set((relation_row.get("votes") or {}).values())) > 1:
        uncertainties.append(
            {
                "uncertainty_id": f"uncertainty_{len(uncertainties)+1:02d}",
                "uncertainty_type": "reader_uncertainty",
                "description": "LLM readers did not agree on the legacy whole-belief relation.",
                "severity": "high" if has_directional_conflict(relation_row.get("votes") or {}) else "medium",
                "what_would_reduce_it": "Ask for full-text read or atom-level relation read.",
            }
        )
    return uncertainties


def has_directional_conflict(votes: dict[str, str]) -> bool:
    values = set(votes.values())
    return "support" in values and "challenge" in values


def build_disagreement_map(relation_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not relation_row:
        return []
    votes = relation_row.get("votes") or {}
    if len(set(votes.values())) <= 1:
        return []
    counts = Counter(votes.values())
    return [
        {
            "disagreement_id": "disagreement_01",
            "field": "legacy_whole_belief_relation",
            "models_involved": sorted(votes.keys()),
            "positions": dict(counts),
            "why_it_matters": "Reader disagreement means the secretary should not present the relation label as a settled kernel judgment.",
            "suggested_resolution": "human_read" if has_directional_conflict(votes) else "leave_pending",
        }
    ]


def atoms_for_item(item: dict[str, Any], atoms_by_belief: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    belief = atoms_by_belief.get(str(item.get("belief_id", "")))
    if not belief:
        return []
    return [
        {
            "atom_id": atom.get("atom_id", ""),
            "claim": atom.get("claim", ""),
            "role": atom.get("role", "other"),
            "criticality": atom.get("criticality", "optional"),
        }
        for atom in belief.get("atoms", [])
    ]


def build_relation_scope(
    item: dict[str, Any],
    relation_row: dict[str, Any] | None,
    atoms_by_belief: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    votes = (relation_row or {}).get("votes") or {}
    legacy_relation = (relation_row or {}).get("kernel_relation", "not_supplied")
    return {
        "belief_id": item.get("belief_id", ""),
        "belief": item.get("belief", ""),
        "atoms": atoms_for_item(item, atoms_by_belief),
        "reader_relation_by_atom": [],
        "atom_consensus": [],
        "whole_belief_projection_candidate": "not_computed",
        "projection_reason": (
            "No atom-level reader output was supplied. Existing whole-belief relation votes "
            "are preserved as legacy secretary input and must not be treated as kernel judgment."
        ),
        "legacy_whole_belief_reader": {
            "votes": votes,
            "kernel_relation": legacy_relation,
            "note": "Legacy whole-belief relation snapshot; relation_scope briefing only.",
        },
    }


def build_reader_runs(relation_row: dict[str, Any] | None) -> list[dict[str, Any]]:
    votes = (relation_row or {}).get("votes") or {}
    if not votes:
        return []
    runs = []
    for idx, model in enumerate(sorted(votes), 1):
        runs.append(
            {
                "reader_id": f"reader_{idx:02d}",
                "model": model,
                "provider": infer_provider(model),
                "prompt_version": "legacy_whole_belief_relation_reader",
                "temperature": 0.0,
                "input_text_level": "abstract",
                "raw_response_ref": "",
                "parse_status": "parsed" if votes[model] != "ERR" else "failed",
                "failure_reason": "" if votes[model] != "ERR" else "reader returned ERR",
            }
        )
    return runs


def infer_provider(model: str) -> str:
    low = model.lower()
    if "deepseek" in low:
        return "deepseek"
    if "llama" in low or "ollama" in low:
        return "llama_or_ollama"
    if "claude" in low or "anthropic" in low:
        return "anthropic"
    if "gpt" in low or "openai" in low:
        return "openai"
    return "unknown"


def build_attention_brief(relation_row: dict[str, Any] | None, disagreement_map: list[dict[str, Any]]) -> dict[str, Any]:
    relation = (relation_row or {}).get("kernel_relation", "not_supplied")
    if disagreement_map and any(d["suggested_resolution"] == "human_read" for d in disagreement_map):
        recommendation = "read"
        reason = "Directional reader disagreement was detected; kernel may need a higher-quality briefing."
    elif relation in {"contest", "challenge"}:
        recommendation = "read"
        reason = f"Legacy relation snapshot was {relation}; secretary recommends kernel review, not automatic update."
    elif relation in {"support", "underdetermined"}:
        recommendation = "skim"
        reason = f"Legacy relation snapshot was {relation}; useful but not sufficient for durable judgment."
    elif relation == "neutral":
        recommendation = "archive"
        reason = "Legacy relation snapshot was neutral; preserve briefing but do not spend attention unless the kernel has a separate reason."
    else:
        recommendation = "skim"
        reason = "No legacy relation snapshot was supplied."
    return {
        "attention_recommendation": recommendation,
        "reason": reason,
        "time_sensitivity": "none",
        "suggested_next_action": "Kernel should decide whether to request full-text reading or ignore the candidate.",
    }


def build_candidate_updates(claims: list[dict[str, Any]], uncertainties: list[dict[str, Any]], attention: dict[str, Any]) -> list[dict[str, Any]]:
    updates = [
        {
            "candidate_id": "candidate_01",
            "target_layer": "source",
            "candidate_change": "Register this briefing as source-level intake material for kernel review.",
            "why_candidate": "A secretary briefing was generated with provenance and claim extraction.",
            "supporting_claim_ids": [claim["claim_id"] for claim in claims[:2]],
            "uncertainty_ids": [],
            "risk": "low",
            "commit_permission": "kernel_only",
        }
    ]
    if uncertainties:
        updates.append(
            {
                "candidate_id": "candidate_02",
                "target_layer": "uncertainty",
                "candidate_change": "Consider whether the briefing uncertainty should enter kernel uncertainty state.",
                "why_candidate": "The secretary identified uncertainty that may affect interpretation.",
                "supporting_claim_ids": [],
                "uncertainty_ids": [u["uncertainty_id"] for u in uncertainties],
                "risk": "medium",
                "commit_permission": "kernel_only",
            }
        )
    if attention["attention_recommendation"] in {"read", "urgent"}:
        updates.append(
            {
                "candidate_id": "candidate_03",
                "target_layer": "action",
                "candidate_change": "Consider a human or full-text read before any durable judgment.",
                "why_candidate": attention["reason"],
                "supporting_claim_ids": [claim["claim_id"] for claim in claims[:1]],
                "uncertainty_ids": [u["uncertainty_id"] for u in uncertainties],
                "risk": "medium",
                "commit_permission": "kernel_only",
            }
        )
    return updates


def build_paper_brief(item: dict[str, Any]) -> dict[str, Any]:
    title = clean_text(item.get("title") or item.get("basename") or "")
    abstract = clean_text(item.get("abstract") or "")
    sentences = split_sentences(abstract)
    return {
        "one_sentence_takeaway": first_sentence(abstract, fallback=title),
        "study_type": infer_study_type(title, abstract),
        "system": ", ".join(infer_entities(f"{title} {abstract}")) or "not clear from abstract",
        "intervention_or_method": choose_sentence(sentences, ["method", "developed", "model", "synthesized", "exposed", "using"], fallback="not clear from abstract"),
        "readout": choose_sentence(sentences, ["readout", "signal", "temperature", "heating", "detection", "viability", "current"], fallback="not clear from abstract"),
        "main_result": choose_sentence(sentences, ["we show", "we demonstrate", "we found", "achieved", "increased", "decreased", "produced", "enabled"], fallback=first_sentence(abstract, fallback="not clear from abstract")),
        "main_limitation": choose_sentence(sentences, ["however", "limited", "limitation", "challenge", "remains"], fallback="No explicit limitation was detected in the abstract-level input."),
        "why_it_might_matter": "This source may matter if the kernel finds that its claims affect an existing belief, trajectory, uncertainty, or action state.",
    }


def build_source_provenance(item: dict[str, Any], text_hash: str) -> dict[str, Any]:
    abstract = clean_text(item.get("abstract") or "")
    locator = clean_text(item.get("physical_path") or item.get("path") or item.get("url") or "")
    doi = clean_text(item.get("doi"))
    if locator.lower().endswith(".pdf"):
        route = "local_pdf"
    elif doi:
        route = "database"
    else:
        route = "manual"
    return {
        "source_id": clean_text(item.get("source_id")),
        "doi": doi,
        "locator": locator,
        "retrieval_route": route,
        "read_level": "abstract" if abstract else "metadata",
        "extraction_method": "deterministic_abstract_level_briefing",
        "source_span_available": bool(abstract),
        "citation_support_verified": False,
        "reuse_or_access_note": clean_text(item.get("access_note")),
        "source_text_hash": text_hash,
    }


def build_object_grounding(
    source: dict[str, Any],
    claims: list[dict[str, Any]],
    mechanism: dict[str, Any],
) -> dict[str, Any]:
    grounded = ["source"]
    if claims:
        grounded.append("claim")
    if mechanism.get("mechanism_status") != "not_addressed":
        grounded.append("mechanism")
    abstract = source.get("abstract", "")
    if contains_any(abstract, ["workflow", "pipeline", "protocol"]):
        grounded.append("workflow")
    if contains_any(abstract, ["figure", "fig.", "table", "dataset"]):
        grounded.append("figure_or_table_or_dataset")
    text_only = not any(item in grounded for item in ["workflow", "figure_or_table_or_dataset"])
    input_level = source.get("input_text_level", "unknown")
    if input_level == "title_only":
        alignment = "not_checked"
        risk = "high"
    elif input_level == "abstract":
        alignment = "partial"
        risk = "medium" if text_only else "low"
    else:
        alignment = "checked"
        risk = "low"
    return {
        "grounded_objects": grounded,
        "text_only": text_only,
        "object_text_alignment": alignment,
        "alignment_caveat": "Generated from abstract/title-level input; figures, tables and full methods were not inspected.",
        "hallucination_risk_from_language_prior": risk,
    }


def build_confidence_dynamics(relation_row: dict[str, Any] | None) -> dict[str, Any]:
    votes = (relation_row or {}).get("votes") or {}
    disagreement = len(set(votes.values())) > 1 if votes else False
    return {
        "reader_confidence_reported": False,
        "reader_confidence_value": None,
        "calibration_status": "unknown",
        "initial_answer_visible": False,
        "prior_answer_attributed_to_self": False,
        "contradictory_advice_visible": False,
        "supportive_advice_visible": False,
        "reader_saw_other_reader_outputs": False,
        "confidence_shift_direction": "not_measured",
        "confidence_shift_basis": "unknown",
        "bias_risks": ["reader_disagreement"] if disagreement else [],
        "usable_for_kernel_confidence": False,
        "kernel_caveat": "Reader confidence was not calibrated and must not directly revise kernel belief confidence.",
    }


def build_task_reliability_envelope(
    source: dict[str, Any],
    relation_row: dict[str, Any] | None,
) -> dict[str, Any]:
    votes = (relation_row or {}).get("votes") or {}
    input_level = source.get("input_text_level", "unknown")
    if not votes:
        admissibility = "unknown"
        reason = "No reader relation task was supplied."
    elif input_level in {"title_only", "metadata"}:
        admissibility = "not_admissible"
        reason = "Title/metadata-only input is not enough for relation-scope judgment."
    else:
        admissibility = "weak_signal"
        reason = "Legacy reader votes are abstract-level and lack expert agreement calibration."
    return {
        "task_type": "relation_scope",
        "construct_definition": "Whether source claims affect a scoped belief or question; not a truth judgment.",
        "observable_cues": ["claim_map", "mechanism_map", "evidence_map", "relation_scope"],
        "subjectivity_level": "high",
        "expert_agreement_known": False,
        "expert_agreement_metric": "",
        "expert_agreement_value": None,
        "model_expert_reliability_known": False,
        "model_expert_reliability_value": None,
        "validated_prompt_available": False,
        "few_shot_expert_examples_used": False,
        "admissibility": admissibility,
        "admissibility_reason": reason,
    }


def build_epistemic_task_probe(title: str, abstract: str) -> dict[str, Any]:
    blob = f"{title} {abstract}"
    low = blob.lower()
    belief_terms = ["believe", "belief", "assume", "hypothesize", "hypothesis"]
    knowledge_terms = ["know", "knowledge", "known"]
    report_terms = [
        "report",
        "reported",
        "observed",
        "account",
        "suggest",
        "claim",
        "interpret",
        "indicate",
        "propose",
        "studied",
    ]
    contains_belief = contains_any(low, belief_terms)
    contains_knowledge = contains_any(low, knowledge_terms)
    contains_hypothesis = contains_any(low, ["hypothesize", "hypothesis"])
    contains_reported = contains_any(low, report_terms)
    requires_review = contains_belief or contains_knowledge or contains_hypothesis or contains_reported
    risk = "high" if contains_belief or contains_knowledge else "medium" if requires_review else "low"
    return {
        "contains_belief_statement": contains_belief,
        "contains_knowledge_statement": contains_knowledge,
        "contains_hypothesis_statement": contains_hypothesis,
        "contains_reported_perspective": contains_reported,
        "source_truth_value_known": "unknown" if requires_review else "not_applicable",
        "model_may_fact_check_instead_of_attribute": requires_review,
        "epistemic_operator_risk": risk,
        "requires_epistemic_boundary_review": requires_review,
        "notes": "Preserve source claim, reader position, kernel belief and truth status separately.",
    }


def build_semantic_audit(
    item: dict[str, Any],
    claims: list[dict[str, Any]],
    relation_scope: dict[str, Any],
) -> dict[str, Any]:
    text = " ".join(
        [
            clean_text(item.get("belief")),
            clean_text(item.get("title") or item.get("basename")),
            clean_text(item.get("abstract")),
        ]
    )
    concepts = infer_entities(text)
    expected = infer_entities(clean_text(item.get("belief")))
    unexpected = [concept for concept in concepts if concept not in expected]
    if not expected:
        expected = concepts[:3]
        unexpected = []
    return {
        "expected_concepts": expected,
        "spurious_or_invalid_concepts": [],
        "unexpected_concepts": unexpected,
        "concept_to_source_links": [
            {"concept": concept, "source_field": "title_or_abstract"}
            for concept in concepts
        ],
        "concept_to_claim_links": [
            {"concept": concept, "claim_id": claim.get("claim_id", "")}
            for concept in concepts
            for claim in claims[:1]
        ],
        "behavioral_risk": "medium" if unexpected else "low",
        "relation_scope_belief_id": relation_scope.get("belief_id", ""),
    }


def build_calibration_status(
    source: dict[str, Any],
    relation_row: dict[str, Any] | None,
) -> dict[str, Any]:
    votes = (relation_row or {}).get("votes") or {}
    values = set(votes.values())
    if votes and len(values) == 1:
        basis = "reader_agreement"
        unknown_risk = "medium"
    elif votes:
        basis = "uncalibrated"
        unknown_risk = "high"
    else:
        basis = "uncalibrated"
        unknown_risk = "unknown"
    if source.get("input_text_level") in {"title_only", "abstract"}:
        unknown_risk = "high" if unknown_risk != "unknown" else "unknown"
    return {
        "confidence_basis": basis,
        "calibration_reference": "No expert-agreement or task-calibration benchmark attached to this briefing.",
        "unknown_or_ood_risk": unknown_risk,
        "should_downweight_confidence": basis == "uncalibrated" or source.get("input_text_level") != "full_text",
    }


def build_trajectory_proposal(
    item: dict[str, Any],
    claims: list[dict[str, Any]],
    uncertainty: list[dict[str, Any]],
) -> dict[str, Any]:
    concepts = infer_entities(f"{item.get('title', '')} {item.get('abstract', '')} {item.get('belief', '')}")
    has_gap = any(u.get("uncertainty_type") in {"mechanism_gap", "reader_uncertainty"} for u in uncertainty)
    title = clean_text(item.get("title") or item.get("basename"))
    candidate = (
        "What evidence would resolve the mechanism or reader conflict?"
        if has_gap
        else f"Does {title[:80]} sharpen an active research question?"
    )
    return {
        "source": "kernel_gap" if has_gap else "semantic_link",
        "candidate_question": candidate,
        "linked_concepts": concepts,
        "why_now": "Generated from secretary uncertainty and concept links; not a durable trajectory update.",
        "novelty_basis": claims[0].get("claim_text", "")[:220] if claims else "",
        "expert_review_needed": True,
    }


def build_human_feedback_brief(
    attention: dict[str, Any],
    uncertainty: list[dict[str, Any]],
) -> dict[str, Any]:
    high_uncertainty = any(u.get("severity") == "high" for u in uncertainty)
    if high_uncertainty:
        feedback_type = "request_evidence"
        suggestion = "Read full text or request atom-level relation read before kernel revision."
    elif attention.get("attention_recommendation") in {"read", "urgent"}:
        feedback_type = "make_specific"
        suggestion = attention.get("suggested_next_action", "")
    else:
        feedback_type = "reduce_overclaim"
        suggestion = "Keep as intake material unless the kernel identifies a specific belief, question or action affected."
    passed = bool(suggestion)
    return {
        "target": "kernel_decision",
        "feedback_type": feedback_type,
        "actionable_suggestion": suggestion,
        "reliability_tests": [
            "specific_action_present",
            "human_retains_control",
            "no_durable_state_change_authorized",
        ],
        "passed": passed,
        "human_control": "optional_uptake",
    }


def build_knowledge_transfer(
    relation_scope: dict[str, Any],
    uncertainty: list[dict[str, Any]],
    attention: dict[str, Any],
) -> dict[str, Any]:
    if any(u.get("uncertainty_type") == "mechanism_gap" for u in uncertainty):
        packet_type = "gap"
    elif attention.get("attention_recommendation") in {"read", "urgent"}:
        packet_type = "refresh_request"
    else:
        packet_type = "claim"
    return {
        "packet_type": packet_type,
        "transfer_scope": "belief" if relation_scope.get("belief_id") else "paper",
        "share_timing": "on_conflict" if packet_type == "refresh_request" else "on_ingest",
        "integration_policy": "human_review_first" if packet_type == "refresh_request" else "kernel_may_consider",
        "forgetting_or_overwrite_risk": "medium" if packet_type == "gap" else "low",
    }


def build_personalization_boundary() -> dict[str, Any]:
    return {
        "setting_source": "none",
        "profile_inference_allowed": False,
        "preference_used": "",
        "preference_effect": "none",
        "truth_or_judgment_effect": "none",
    }


def build_closed_loop_validation(relation_row: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "proposal_source": "legacy_digest" if relation_row else "retrieval_rule",
        "oracle_like_score": clean_text((relation_row or {}).get("kernel_relation", "")),
        "validation_state": "unvalidated",
        "ground_truth_available": False,
        "kernel_commit_allowed": False,
        "next_validation_step": "Human/full-text validation required before any truth-facing claim.",
    }


def build_cognitive_tooling(
    item: dict[str, Any],
    claims: list[dict[str, Any]],
    uncertainty: list[dict[str, Any]],
) -> dict[str, Any]:
    title = clean_text(item.get("title") or item.get("basename"))
    first_claim = claims[0].get("claim_text", "") if claims else ""
    counter = next(
        (u.get("description", "") for u in uncertainty if u.get("severity") in {"medium", "high"}),
        "",
    )
    return {
        "active_question": f"What, if anything, should {title[:80]} change in current research state?",
        "working_hypothesis": first_claim[:240],
        "task_decomposition": [
            "check source provenance",
            "inspect claim/mechanism/evidence maps",
            "decide whether refresh, human read or no durable change is needed",
        ],
        "counterargument": counter,
        "future_simulation": "If accepted prematurely, this briefing could overstate abstract-level evidence.",
        "memory_anchor": title,
        "next_goal": "Keep as draft cognitive object until kernel review.",
        "tool_status": "draft",
    }


def build_prior_trace(
    item: dict[str, Any],
    relation_row: dict[str, Any] | None,
) -> dict[str, Any]:
    prior_type = "project_priority" if item.get("belief_id") or item.get("belief") else "unknown"
    votes = (relation_row or {}).get("votes") or {}
    return {
        "prior_type": prior_type,
        "prior_effect": "interpretation" if prior_type != "unknown" else "ranking",
        "critique_required": bool(votes and len(set(votes.values())) > 1),
        "counter_prior_needed": bool(votes and len(set(votes.values())) > 1),
    }


def build_active_curriculum(
    uncertainty: list[dict[str, Any]],
    disagreement: list[dict[str, Any]],
) -> dict[str, Any]:
    uncertainty_types = {u.get("uncertainty_type") for u in uncertainty}
    if disagreement:
        need = "conflict"
        gain = "high"
        source_type = "expert_label"
    elif "mechanism_gap" in uncertainty_types:
        need = "mechanism_gap"
        gain = "medium"
        source_type = "paper"
    elif "insufficient_input_text" in uncertainty_types:
        need = "weak_grounding"
        gain = "medium"
        source_type = "paper"
    else:
        need = "new_direction"
        gain = "low"
        source_type = "review"
    return {
        "learning_need": need,
        "expected_learning_gain": gain,
        "next_source_type": source_type,
        "selection_reason": "Derived from uncertainty and disagreement maps.",
        "avoid_more_of_same": need in {"conflict", "weak_grounding"},
    }


def source_hash(item: dict[str, Any]) -> str:
    blob = "\n".join(
        [
            clean_text(item.get("id")),
            clean_text(item.get("title") or item.get("basename")),
            clean_text(item.get("doi")),
            clean_text(item.get("abstract")),
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_briefing(
    item: dict[str, Any],
    relation_row: dict[str, Any] | None,
    atoms_by_belief: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    title = clean_text(item.get("title") or item.get("basename") or "")
    abstract = clean_text(item.get("abstract") or "")
    claims = build_claim_map(abstract)
    text_hash = source_hash(item)
    source = {
        "source_id": clean_text(item.get("source_id")),
        "paper_id": clean_text(item.get("id")),
        "title": title,
        "doi": clean_text(item.get("doi")),
        "journal": clean_text(item.get("journal")),
        "year": item.get("year") or item.get("publication_year"),
        "abstract": abstract,
        "input_text_level": "abstract" if abstract else "title_only",
    }
    mechanism = build_mechanism_map(abstract)
    evidence = build_evidence_map(title, abstract)
    uncertainty = build_uncertainty_map(abstract, relation_row)
    disagreement = build_disagreement_map(relation_row)
    relation_scope = build_relation_scope(item, relation_row, atoms_by_belief)
    attention = build_attention_brief(relation_row, disagreement)
    source_provenance = build_source_provenance(item, text_hash)
    models = sorted(((relation_row or {}).get("votes") or {}).keys())
    return {
        "schema_id": SCHEMA_ID,
        "briefing_id": "brief_" + hashlib.sha256(f"{item.get('id','')}:{text_hash}".encode("utf-8")).hexdigest()[:16],
        "source": source,
        "source_provenance": source_provenance,
        "reader_runs": build_reader_runs(relation_row),
        "paper_brief": build_paper_brief(item),
        "claim_map": claims,
        "object_grounding": build_object_grounding(source, claims, mechanism),
        "mechanism_map": mechanism,
        "evidence_map": evidence,
        "relation_scope": relation_scope,
        "epistemic_task_probe": build_epistemic_task_probe(title, abstract),
        "confidence_dynamics": build_confidence_dynamics(relation_row),
        "task_reliability_envelope": build_task_reliability_envelope(source, relation_row),
        "semantic_audit": build_semantic_audit(item, claims, relation_scope),
        "calibration_status": build_calibration_status(source, relation_row),
        "trajectory_proposal": build_trajectory_proposal(item, claims, uncertainty),
        "human_feedback_brief": build_human_feedback_brief(attention, uncertainty),
        "knowledge_transfer": build_knowledge_transfer(relation_scope, uncertainty, attention),
        "personalization_boundary": build_personalization_boundary(),
        "closed_loop_validation": build_closed_loop_validation(relation_row),
        "cognitive_tooling": build_cognitive_tooling(item, claims, uncertainty),
        "prior_trace": build_prior_trace(item, relation_row),
        "active_curriculum": build_active_curriculum(uncertainty, disagreement),
        "uncertainty_map": uncertainty,
        "disagreement_map": disagreement,
        "candidate_kernel_updates": build_candidate_updates(claims, uncertainty, attention),
        "attention_brief": attention,
        "provenance": {
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "briefing_schema_version": SCHEMA_VERSION,
            "prompt_versions": ["legacy_whole_belief_relation_reader"] if models else [],
            "source_text_hash": text_hash,
            "models": models,
            "tools": [SCRIPT_VERSION],
            "known_failures": [
                "No new LLM calls were made by this generator.",
                "Legacy whole-belief relation snapshots are preserved but not treated as atom-level relation_scope.",
                "Methodology modules are deterministic intake diagnostics, not proof of full-text validation.",
            ],
        },
    }


def validate_briefing_shape(briefing: dict[str, Any], schema_path: str | Path) -> list[str]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    required = schema.get("required_top_level_fields", [])
    issues = [f"missing top-level field: {field}" for field in required if field not in briefing]
    if briefing.get("schema_id") != SCHEMA_ID:
        issues.append("schema_id mismatch")
    for update in briefing.get("candidate_kernel_updates", []):
        if update.get("commit_permission") != "kernel_only":
            issues.append(f"{update.get('candidate_id', 'candidate')} has non-kernel commit permission")
    if briefing.get("relation_scope", {}).get("whole_belief_projection_candidate") != "not_computed":
        issues.append("minimal generator should not compute whole-belief projection from legacy votes")
    return issues


def write_output(briefings: list[dict[str, Any]], output: str | Path, single: bool) -> None:
    if single:
        payload: Any = briefings[0] if briefings else {}
    else:
        payload = {
            "schema_id": "audit_secretary_briefing_batch_v1",
            "briefing_schema_id": SCHEMA_ID,
            "count": len(briefings),
            "briefings": briefings,
        }
    Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, help="Paper packet JSON list or {'items': [...]} packet.")
    parser.add_argument("--relations", default="", help="Optional legacy relation snapshot JSON.")
    parser.add_argument("--atoms", default="kernel/v3/schemas/atomized_relation_schema_v2.json")
    parser.add_argument("--schema", default="kernel/v3/schemas/audit_secretary_briefing_schema_v1.json")
    parser.add_argument("--item-id", default="", help="Build only one briefing.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="audit_secretary_briefings.json")
    args = parser.parse_args()

    items = load_items(args.items)
    if args.item_id:
        items = [item for item in items if str(item.get("id")) == args.item_id]
    if args.limit:
        items = items[: args.limit]
    relation_rows = load_relation_rows(args.relations)
    atoms_by_belief = load_atoms(args.atoms)
    briefings = [
        build_briefing(item, relation_rows.get(str(item.get("id"))), atoms_by_belief)
        for item in items
    ]
    all_issues = []
    for briefing in briefings:
        for issue in validate_briefing_shape(briefing, args.schema):
            all_issues.append(f"{briefing.get('briefing_id')}: {issue}")
    if all_issues:
        raise SystemExit("\n".join(all_issues))
    write_output(briefings, args.output, single=bool(args.item_id) or len(briefings) == 1)
    print(f"built {len(briefings)} briefing(s) -> {args.output}")
    if briefings:
        attention_counts = Counter(b["attention_brief"]["attention_recommendation"] for b in briefings)
        disagreement_count = sum(1 for b in briefings if b.get("disagreement_map"))
        print(f"attention: {dict(attention_counts)}")
        print(f"with_reader_disagreement: {disagreement_count}")


if __name__ == "__main__":
    main()
