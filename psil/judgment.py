"""Judgment-kernel summary layer.

This module composes ScholarHound's persisted state into a compact,
queryable memory-palace view. It does not call an LLM and it does not make
new commitments; it surfaces what the existing kernel state is already
trying to say.
"""

import json
import re

REVISION_ACTIONS = {
    "commit": {"status": "approved", "confidence_delta": 1.5, "entrenchment_delta": 2.0},
    "approve": {"alias": "commit"},
    "reject": {"status": "rejected", "confidence_delta": -3.0, "entrenchment_delta": -2.0},
    "challenge": {"status": "contested", "confidence_delta": -1.0, "entrenchment_delta": -0.5},
    "contradict": {"alias": "challenge"},
    "resolve": {"status": "resolved", "confidence_delta": 0.5, "entrenchment_delta": 1.0},
    "reopen": {"status": "candidate", "confidence_delta": 0.0, "entrenchment_delta": -1.0},
}

KERNEL_TASTE_TERMS = [
    ("ribotac", 7),
    ("rna", 7),
    ("warhead", 5),
    ("oect", 6),
    ("organic electrochemical", 6),
    ("channel state", 5),
    ("ionic-electronic", 5),
    ("electric double layer", 5),
    ("iontronic", 5),
    ("polyelectrolyte elastomer", 4),
    ("single-ion conductive", 4),
    ("electrochemical organic light-emitting", 4),
    ("ionogel", 4),
    ("synaptic transistor", 3),
    ("mechanoreceptor", 2),
    ("mixed electron-ion", 5),
    ("foreign-body response", 4),
    ("immune-compatible", 4),
    ("semiconducting polymer", 3),
    ("organic semiconductor phase", 3),
    ("molecular recognition", 5),
    ("bioelectronic", 5),
    ("extracellular vesicle", 5),
    ("exosome", 5),
    ("ev", 5),
    ("organoid", 5),
    ("disease", 4),
    ("diagnostic", 4),
    ("alzheimer", 3),
    ("nanobody", 3),
    ("aptamer", 3),
    ("biomarker", 3),
    ("photochemistry", 3),
    ("photonic", 3),
    ("polariton", 3),
    ("structured light", 4),
    ("nonlinear optics", 4),
    ("lithium niobate", 4),
    ("microresonator", 4),
    ("optical vortex", 4),
    ("optical skyrmion", 4),
    ("microcomb", 4),
    ("bodipy", 3),
    ("nir", 3),
    ("photocleavage", 3),
    ("mechanobiology", 2),
    ("biointerface", 2),
    ("theranostic", 2),
]

GENERIC_TASK_PENALTIES = [
    ("output independent of trigger magnitude", -16),
    ("threshold energy requirement", -14),
    ("reservoir depletion", -14),
    ("disproportionate output requirement", -13),
    ("constraint topology", -10),
    ("generic", -5),
]

TASK_TYPE_PRIORITY_MULTIPLIERS = {
    "resolve_contested_object": 1.3,
    "commit_or_reject_object": 1.25,
    "sharpen_open_question": 1.2,
    "verify_constraint": 1.1,
    "judge_high_score_paper": 1.05,
    "next_kernel_move": 0.9,
    "sharpen_pressure_question": 0.75,
    "verify_pressure_constraint": 0.2,
}

TASK_TYPE_PRIORITY_ADJUSTMENTS = {
    "resolve_contested_object": 6,
    "commit_or_reject_object": 5,
    "sharpen_open_question": 4,
    "verify_constraint": 3,
    "judge_high_score_paper": 2,
    "next_kernel_move": 0,
    "sharpen_pressure_question": 0,
    "verify_pressure_constraint": -5,
}

FRONTIER_RULES = [
    ("Molecular Recognition -> Bioelectronic Transduction", [
        "molecular recognition",
        "rna",
        "ribotac",
        "oect",
        "channel state",
        "bioelectronic state",
        "ionic-electronic",
        "electric double layer",
        "iontronic",
        "contact injection",
        "single-ion conductive",
        "ionogel",
        "synaptic transistor",
        "mechanoreceptor",
        "mixed electron-ion",
        "immune-compatible",
        "organic semiconductor phase",
    ]),
    ("Organoid / EV Disease-State Readouts", [
        "organoid",
        "extracellular vesicle",
        "exosome",
        " ev ",
        "disease-state",
        "disease signal",
    ]),
    ("Energy Landscape Transduction", [
        "energy landscape",
        "metastability",
        "latent potential",
        "threshold-triggered",
        "catalytic amplification",
        "asymmetric barrier",
        "amplification needs energy",
        "free energy",
    ]),
    ("Nanophotonic Field Control", [
        "photonic",
        "polariton",
        "optical-state",
        "chiroptical",
        "structured light",
        "nonlinear optics",
        "lithium niobate",
        "microresonator",
        "optical vortex",
        "optical skyrmion",
        "microcomb",
    ]),
    ("Adaptive Living Interfaces", [
        "biointerface",
        "biointerfaces",
        "tissue force",
        "force-coupled",
        "mechanobiology",
        "deforming tissue",
    ]),
    ("Activity-Based Biomarker Transduction", [
        "activity-based biomarker",
        "crispr",
        "reporter",
        "biomarker",
    ]),
]


def _clean_text(value, limit: int = 240) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if limit and len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def _score_number(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _paper_reasoning(row: dict) -> dict:
    try:
        return json.loads(row.get("llm_reasoning") or "{}")
    except Exception:
        return {}


def _json_mapping(value) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _paper_final_score(row: dict) -> float:
    reasoning = _paper_reasoning(row)
    return _score_number(reasoning.get("final_score"), row.get("signal_score") or 0)


def _entry(kind: str, title: str, detail: str = "", **extra) -> dict:
    item = {
        "type": kind,
        "title": _clean_text(title, 180),
        "detail": _clean_text(detail, 320),
    }
    item.update({k: v for k, v in extra.items() if v not in (None, "", [])})
    return item


def _clamp_score(value, low: float = 0, high: float = 10) -> float:
    return round(max(low, min(high, _score_number(value))), 2)


def _clamp_priority(value, low: float = 0, high: float = 60) -> float:
    return round(max(low, min(high, _score_number(value))), 2)


def _memory_items(memory_summary: dict, key: str, limit: int = 5) -> list[dict]:
    items = []
    for item in (memory_summary or {}).get(key, [])[:limit]:
        items.append(_entry(
            item.get("item_type") or key.rstrip("s"),
            item.get("item_name", ""),
            item.get("reason", ""),
            evidence_strength=item.get("evidence_strength", ""),
            affected_projects=item.get("affected_projects", ""),
        ))
    return items


def _story_open_questions(story_groups: list[dict], limit: int = 8) -> list[dict]:
    questions = []
    seen = set()
    for story in story_groups or []:
        direction = story.get("direction", "")
        evidence_count = int(story.get("evidence_count") or 0)
        for node in story.get("nodes", []):
            if node.get("node_type") not in {"next_question", "question"}:
                continue
            title = _clean_text(node.get("title", ""), 180)
            if not title or title in seen:
                continue
            seen.add(title)
            questions.append(_entry(
                "open_question",
                title,
                node.get("missing_link") or node.get("summary") or "",
                direction=direction,
                evidence_count=evidence_count,
                next_move=node.get("next_move", ""),
            ))
            if len(questions) >= limit:
                return questions
    return questions


def _concept_open_questions(concepts: list[dict], limit: int = 8) -> list[dict]:
    questions = []
    seen = set()
    ranked = sorted(
        concepts or [],
        key=lambda item: (int(item.get("appearances") or 0), item.get("last_seen") or ""),
        reverse=True,
    )
    for concept in ranked:
        gap = _clean_text(concept.get("missing_link") or concept.get("opportunity") or "", 220)
        name = _clean_text(concept.get("name", ""), 120)
        if not gap or not name or name in seen:
            continue
        seen.add(name)
        questions.append(_entry(
            "concept_gap",
            f"What would make {name} decisive?",
            gap,
            concept=name,
            evidence_count=int(concept.get("appearances") or 0),
        ))
        if len(questions) >= limit:
            break
    return questions


def _candidate_claims(
    frameworks: list[dict],
    constraints: list[dict],
    deltas: list[dict],
    limit: int = 8,
) -> list[dict]:
    claims = []
    for fw in frameworks or []:
        statement = _clean_text(
            fw.get("core_logic") or fw.get("worldview_shift") or fw.get("description") or "",
            260,
        )
        name = fw.get("framework_name", "")
        if not name or not statement:
            continue
        score = (
            _score_number(fw.get("compression_score"))
            + _score_number(fw.get("predictive_power"))
            + _score_number(fw.get("falsifiability"))
            + _score_number(fw.get("actionability"))
        )
        claims.append(_entry(
            "framework_claim",
            name,
            statement,
            score=round(score, 2),
            status=fw.get("status", "candidate"),
        ))

    for constraint in constraints or []:
        statement = constraint.get("statement") or constraint.get("name", "")
        if not statement:
            continue
        score = (
            _score_number(constraint.get("prediction_power"))
            + _score_number(constraint.get("confidence"))
            + _score_number(constraint.get("actionability"))
        )
        claims.append(_entry(
            "constraint_claim",
            constraint.get("name", "constraint"),
            statement,
            score=round(score, 2),
            status=constraint.get("status", "candidate"),
            framework=constraint.get("framework_name", ""),
        ))

    for delta in deltas or []:
        previous = _clean_text(delta.get("previous_assumption", ""), 120)
        new = _clean_text(delta.get("new_assumption", ""), 160)
        if not previous and not new:
            continue
        title = delta.get("delta") or "worldview shift"
        detail = f"From {previous} to {new}." if previous and new else new or previous
        claims.append(_entry(
            "worldview_shift",
            title,
            detail,
            status=delta.get("status", "candidate"),
        ))

    claims.sort(key=lambda item: _score_number(item.get("score")), reverse=True)
    return claims[:limit]


def _constraint_verification_names(verifications: list[dict], result: str | None = None) -> set[str]:
    names = set()
    for verification in verifications or []:
        if result and verification.get("result") != result:
            continue
        name = (verification.get("constraint_name") or "").strip().lower()
        if name:
            names.add(name)
    return names


def _pressure_points(
    papers: list[dict],
    concepts: list[dict],
    constraints: list[dict],
    verifications: list[dict],
    memory_summary: dict,
    open_questions: list[dict],
    limit: int = 8,
) -> list[dict]:
    pressures = []

    for contradiction in (memory_summary or {}).get("contradictions", [])[:3]:
        pressures.append(_entry(
            "contradiction",
            contradiction.get("item_name", "Unresolved contradiction"),
            contradiction.get("reason", ""),
            priority=3,
        ))

    tested = _constraint_verification_names(verifications)
    for constraint in constraints or []:
        name = (constraint.get("name") or "").strip()
        if not name or name.lower() in tested:
            continue
        if constraint.get("status") in {"rejected", "resolved"}:
            continue
        score = _score_number(constraint.get("prediction_power")) + _score_number(constraint.get("confidence"))
        pressures.append(_entry(
            "untested_constraint",
            name,
            constraint.get("statement", ""),
            priority=2,
            score=round(score, 2),
        ))

    high_score_papers = sorted(papers or [], key=_paper_final_score, reverse=True)
    for paper in high_score_papers:
        score = _paper_final_score(paper)
        if score < 7:
            break
        concept = paper.get("concept_support_name") or paper.get("concept_name") or ""
        evidence_strength = paper.get("evidence_strength") or _paper_reasoning(paper).get("evidence_strength") or ""
        if concept and evidence_strength:
            continue
        pressures.append(_entry(
            "high_score_needs_judgment",
            paper.get("title", "High-scoring paper"),
            "High score but incomplete concept or evidence-strength commitment.",
            priority=1,
            score=round(score, 2),
            doi=paper.get("doi", ""),
        ))
        if len([p for p in pressures if p["type"] == "high_score_needs_judgment"]) >= 2:
            break

    for question in open_questions[:3]:
        pressures.append(_entry(
            "open_question",
            question.get("title", ""),
            question.get("detail", ""),
            priority=1,
            direction=question.get("direction", ""),
        ))

    for concept in concepts or []:
        gap = concept.get("missing_link") or concept.get("opportunity")
        if not gap:
            continue
        pressures.append(_entry(
            "concept_gap",
            concept.get("name", "Concept gap"),
            gap,
            priority=1,
            evidence_count=int(concept.get("appearances") or 0),
        ))
        if len([p for p in pressures if p["type"] == "concept_gap"]) >= 2:
            break

    pressures.sort(key=lambda item: (_score_number(item.get("priority")), _score_number(item.get("score"))), reverse=True)
    return pressures[:limit]


def _next_moves(
    pressure_points: list[dict],
    open_questions: list[dict],
    candidate_claims: list[dict],
    limit: int = 4,
) -> list[dict]:
    moves = []
    for pressure in pressure_points:
        if pressure.get("type") == "contradiction":
            moves.append(_entry("resolve", f"Resolve: {pressure.get('title')}", pressure.get("detail", "")))
        elif pressure.get("type") == "untested_constraint":
            moves.append(_entry("verify", f"Verify constraint: {pressure.get('title')}", pressure.get("detail", "")))
        if len(moves) >= limit:
            return moves

    for question in open_questions:
        moves.append(_entry(
            "sharpen_question",
            f"Sharpen: {question.get('title')}",
            question.get("next_move") or question.get("detail", ""),
        ))
        if len(moves) >= limit:
            return moves

    for claim in candidate_claims:
        moves.append(_entry(
            "commit_or_reject",
            f"Commit or reject: {claim.get('title')}",
            claim.get("detail", ""),
        ))
        if len(moves) >= limit:
            return moves
    return moves


def _materialize_item(db, object_type: str, title: str, statement: str = "",
                      status: str = "candidate", confidence: float = 0,
                      entrenchment: float = 0, source_type: str = "",
                      source_ref: str = "", evidence=None, metadata=None) -> dict:
    return db.upsert_kernel_object(
        object_type=object_type,
        title=title,
        statement=statement,
        status=status,
        confidence=_clamp_score(confidence),
        entrenchment=_clamp_score(entrenchment),
        source_type=source_type,
        source_ref=source_ref,
        evidence=evidence,
        metadata=metadata,
    )


def materialize_kernel_objects(db, summary: dict) -> dict:
    """Persist summary-derived kernel objects into first-class storage.

    This is a deterministic bridge from legacy memory/framework/constraint state
    into the v0.2 object model. It intentionally does not call an LLM.
    """
    palace = (summary or {}).get("memory_palace", {})
    materialized = []

    for item in palace.get("active_beliefs", []):
        materialized.append(_materialize_item(
            db,
            "belief",
            item.get("title", ""),
            item.get("detail", ""),
            status="approved",
            confidence=7,
            entrenchment=7,
            source_type="research_memory",
            source_ref=item.get("title", ""),
            evidence=item,
        ))

    for item in palace.get("rejected", []):
        materialized.append(_materialize_item(
            db,
            "rejected_idea",
            item.get("title", ""),
            item.get("detail", ""),
            status="rejected",
            confidence=0,
            entrenchment=1,
            source_type="research_memory",
            source_ref=item.get("title", ""),
            evidence=item,
        ))

    for item in palace.get("contradictions", []):
        materialized.append(_materialize_item(
            db,
            "contradiction",
            item.get("title", ""),
            item.get("detail", ""),
            status="contested",
            confidence=5,
            entrenchment=4,
            source_type="research_memory",
            source_ref=item.get("title", ""),
            evidence=item,
        ))

    for item in palace.get("decisions", []):
        materialized.append(_materialize_item(
            db,
            "decision",
            item.get("title", ""),
            item.get("detail", ""),
            status="approved",
            confidence=7,
            entrenchment=6,
            source_type="research_memory",
            source_ref=item.get("title", ""),
            evidence=item,
        ))

    for item in palace.get("open_questions", []):
        confidence = 2 + min(6, _score_number(item.get("evidence_count")) / 12)
        materialized.append(_materialize_item(
            db,
            "open_question",
            item.get("title", ""),
            item.get("detail", ""),
            status="open",
            confidence=confidence,
            entrenchment=2,
            source_type=item.get("type", "trajectory"),
            source_ref=item.get("direction") or item.get("concept") or item.get("title", ""),
            evidence=item,
        ))

    for item in palace.get("candidate_claims", []):
        source_kind = item.get("type", "")
        object_type = "constraint" if source_kind == "constraint_claim" else "claim"
        score = min(10, _score_number(item.get("score")) / 3 if item.get("score") is not None else 3)
        materialized.append(_materialize_item(
            db,
            object_type,
            item.get("title", ""),
            item.get("detail", ""),
            status=item.get("status", "candidate") or "candidate",
            confidence=score,
            entrenchment=max(1, score / 2),
            source_type=source_kind or "candidate_claim",
            source_ref=item.get("framework") or item.get("title", ""),
            evidence=item,
        ))

    counts: dict[str, int] = {}
    keys = []
    for obj in materialized:
        if not obj:
            continue
        keys.append(obj.get("object_key", ""))
        obj_type = obj.get("object_type", "object")
        counts[obj_type] = counts.get(obj_type, 0) + 1

    return {
        "total": len([key for key in keys if key]),
        "counts": counts,
        "object_keys": [key for key in keys if key],
    }


def apply_kernel_revision(db, object_key: str, action: str,
                          reason: str = "", evidence_delta=None,
                          actor: str = "human") -> dict:
    """Apply a deterministic, non-LLM revision to a kernel object."""
    obj = db.get_kernel_object(object_key)
    if not obj:
        return {"ok": False, "error": "not_found", "object_key": object_key}

    normalized = (action or "").strip().lower()
    rule = REVISION_ACTIONS.get(normalized)
    if rule and rule.get("alias"):
        normalized = rule["alias"]
        rule = REVISION_ACTIONS.get(normalized)
    if not rule:
        return {"ok": False, "error": "unknown_action", "action": action}

    previous_status = obj.get("status", "candidate")
    previous_confidence = _score_number(obj.get("confidence"))
    previous_entrenchment = _score_number(obj.get("entrenchment"))
    new_status = rule["status"]
    new_confidence = _clamp_score(previous_confidence + rule["confidence_delta"])
    new_entrenchment = _clamp_score(previous_entrenchment + rule["entrenchment_delta"])

    revised = db.revise_kernel_object(
        object_key,
        status=new_status,
        confidence=new_confidence,
        entrenchment=new_entrenchment,
    )
    event = db.add_kernel_object_event(
        object_key=object_key,
        event_type=normalized,
        previous_status=previous_status,
        new_status=new_status,
        previous_confidence=previous_confidence,
        new_confidence=new_confidence,
        previous_entrenchment=previous_entrenchment,
        new_entrenchment=new_entrenchment,
        reason=reason,
        evidence_delta=evidence_delta,
        actor=actor,
    )
    return {
        "ok": True,
        "action": normalized,
        "object": revised,
        "event": event,
        "transition": {
            "status": [previous_status, new_status],
            "confidence": [previous_confidence, new_confidence],
            "entrenchment": [previous_entrenchment, new_entrenchment],
        },
    }


def _task_entry(task_type: str, title: str, description: str = "",
                priority: float = 0, action_hint: str = "",
                object_key: str = "", source_type: str = "",
                source_ref: str = "", metadata=None) -> dict:
    return {
        "task_type": task_type,
        "title": _clean_text(title, 180),
        "description": _clean_text(description, 360),
        "priority": round(float(priority or 0), 2),
        "action_hint": action_hint,
        "object_key": object_key,
        "source_type": source_type,
        "source_ref": source_ref,
        "metadata": metadata or {},
    }


def _object_priority(obj: dict) -> float:
    confidence = _score_number(obj.get("confidence"))
    entrenchment = _score_number(obj.get("entrenchment"))
    status_boost = {
        "contested": 4,
        "candidate": 2,
        "open": 1.5,
        "approved": 0.5,
    }.get(obj.get("status", ""), 0)
    return confidence + entrenchment * 0.6 + status_boost


def _object_task_metadata(obj: dict, obj_type: str, status: str) -> dict:
    evidence = _json_mapping(obj.get("evidence"))
    source_type = obj.get("source_type", "")
    source_ref = obj.get("source_ref", "")
    metadata = {
        "object_type": obj_type,
        "status": status,
    }
    for key in ("direction", "concept", "evidence_count", "next_move"):
        if evidence.get(key) not in (None, "", []):
            metadata[key] = evidence.get(key)
    if not metadata.get("direction") and source_type in {"open_question", "trajectory"} and source_ref:
        metadata["direction"] = source_ref
    if not metadata.get("concept") and source_type == "concept_gap" and source_ref:
        metadata["concept"] = source_ref
    return metadata


def _task_text(task: dict) -> str:
    metadata = task.get("metadata") or {}
    try:
        metadata_text = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    except Exception:
        metadata_text = str(metadata)
    parts = [
        task.get("task_type", ""),
        task.get("title", ""),
        task.get("description", ""),
        task.get("action_hint", ""),
        task.get("source_type", ""),
        task.get("source_ref", ""),
        metadata_text,
    ]
    return " ".join(str(part or "") for part in parts).lower()


def _term_matches(text: str, term: str) -> bool:
    term = (term or "").lower().strip()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9]+", term):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
    return term in text


def _calibrate_kernel_task(task: dict) -> dict:
    """Apply the non-LLM ScholarHound taste profile to task priority."""
    task_type = task.get("task_type", "")
    base_priority = _score_number(task.get("priority"))
    multiplier = TASK_TYPE_PRIORITY_MULTIPLIERS.get(task_type, 1.0)
    type_adjustment = TASK_TYPE_PRIORITY_ADJUSTMENTS.get(task_type, 0)
    text = _task_text(task)
    taste_score = 0.0
    reasons = []

    for term, weight in KERNEL_TASTE_TERMS:
        if _term_matches(text, term):
            taste_score += weight
            reasons.append({"term": term, "weight": weight})

    for term, weight in GENERIC_TASK_PENALTIES:
        if _term_matches(text, term):
            taste_score += weight
            reasons.append({"term": term, "weight": weight})

    calibrated_priority = _clamp_priority(base_priority * multiplier + type_adjustment + taste_score)
    metadata = dict(task.get("metadata") or {})
    metadata["priority_calibration"] = {
        "base_priority": round(base_priority, 2),
        "calibrated_priority": calibrated_priority,
        "task_type_multiplier": multiplier,
        "task_type_adjustment": type_adjustment,
        "taste_score": round(taste_score, 2),
        "reasons": reasons[:8],
    }
    metadata["frontier"] = _task_frontier({**task, "metadata": metadata})
    return {
        **task,
        "priority": calibrated_priority,
        "metadata": metadata,
    }


def _task_frontier(task: dict) -> str:
    metadata = task.get("metadata") or {}
    if metadata.get("direction"):
        return _clean_text(metadata.get("direction"), 120)
    if metadata.get("concept"):
        return f"Concept: {_clean_text(metadata.get('concept'), 96)}"

    text = f" {_task_text(task)} "
    for frontier, terms in FRONTIER_RULES:
        if any(term in text for term in terms):
            return frontier

    if task.get("task_type") in {"verify_pressure_constraint", "next_kernel_move"}:
        return "Abstract Kernel Constraints"
    return _clean_text(task.get("task_type") or "General Kernel Work", 120)


def _rank_diverse_kernel_tasks(tasks: list[dict], limit: int) -> list[dict]:
    """Keep the top queue from collapsing into one research direction."""
    ranked = sorted(tasks, key=lambda item: item.get("priority", 0), reverse=True)
    buckets: dict[str, list[dict]] = {}
    frontier_order = []

    for task in ranked:
        metadata = task.get("metadata") or {}
        frontier = metadata.get("frontier") or _task_frontier(task)
        if frontier not in buckets:
            buckets[frontier] = []
            frontier_order.append(frontier)
        buckets[frontier].append(task)

    deferred = {"Abstract Kernel Constraints"}
    primary_order = [frontier for frontier in frontier_order if frontier not in deferred]
    deferred_order = [frontier for frontier in frontier_order if frontier in deferred]
    selected = []

    def drain_round_robin(order: list[str]) -> list[str]:
        next_order = []
        for frontier in order:
            bucket = buckets.get(frontier) or []
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            if bucket:
                next_order.append(frontier)
            if len(selected) >= limit:
                break
        return next_order

    while len(selected) < limit and primary_order:
        primary_order = drain_round_robin(primary_order)
    while len(selected) < limit and deferred_order:
        deferred_order = drain_round_robin(deferred_order)
    return selected


def generate_kernel_tasks(summary: dict, limit: int = 24) -> list[dict]:
    """Generate deterministic judgment tasks from kernel state."""
    palace = (summary or {}).get("memory_palace", {})
    objects = palace.get("kernel_objects", []) or []
    pressure_points = palace.get("pressure_points", []) or []
    next_moves = palace.get("next_moves", []) or []
    tasks = []
    seen = set()

    def add(task: dict):
        key = (task.get("task_type"), task.get("object_key"), task.get("title"))
        if key in seen or not task.get("title"):
            return
        seen.add(key)
        tasks.append(task)

    for obj in objects:
        obj_type = obj.get("object_type", "")
        status = obj.get("status", "")
        key = obj.get("object_key", "")
        title = obj.get("title", "")
        statement = obj.get("statement") or obj.get("description") or ""
        priority = _object_priority(obj)
        metadata = _object_task_metadata(obj, obj_type, status)
        if status == "contested" or obj_type == "contradiction":
            add(_task_entry(
                "resolve_contested_object",
                f"Resolve: {title}",
                statement or "Object is contested and needs a human/kernel resolution.",
                priority=priority + 4,
                action_hint="resolve_or_reopen",
                object_key=key,
                source_type="kernel_object",
                source_ref=title,
                metadata=metadata,
            ))
        elif status == "candidate" and obj_type in {"claim", "belief", "rejected_idea"}:
            add(_task_entry(
                "commit_or_reject_object",
                f"Commit or reject: {title}",
                statement or "Candidate object needs a commitment decision.",
                priority=priority + 2,
                action_hint="commit_or_reject",
                object_key=key,
                source_type="kernel_object",
                source_ref=title,
                metadata=metadata,
            ))
        elif status == "candidate" and obj_type == "constraint":
            add(_task_entry(
                "verify_constraint",
                f"Verify constraint: {title}",
                statement or "Candidate constraint needs support, violation, or dismissal.",
                priority=priority + 3,
                action_hint="verify_or_challenge",
                object_key=key,
                source_type="kernel_object",
                source_ref=title,
                metadata=metadata,
            ))
        elif status == "open" and obj_type == "open_question":
            add(_task_entry(
                "sharpen_open_question",
                f"Sharpen question: {title}",
                statement or "Open question needs a sharper discriminating test.",
                priority=priority + 1,
                action_hint="sharpen_or_link_evidence",
                object_key=key,
                source_type="kernel_object",
                source_ref=title,
                metadata=metadata,
            ))

    for pressure in pressure_points:
        p_type = pressure.get("type", "")
        priority = _score_number(pressure.get("priority")) * 3 + _score_number(pressure.get("score"))
        title = pressure.get("title", "")
        if p_type == "untested_constraint":
            add(_task_entry(
                "verify_pressure_constraint",
                f"Verify pressure point: {title}",
                pressure.get("detail", ""),
                priority=priority + 5,
                action_hint="verify_constraint",
                source_type="pressure_point",
                source_ref=title,
                metadata=pressure,
            ))
        elif p_type == "high_score_needs_judgment":
            add(_task_entry(
                "judge_high_score_paper",
                f"Judge high-score paper: {title}",
                pressure.get("detail", ""),
                priority=priority + 2,
                action_hint="extract_commitment",
                source_type="pressure_point",
                source_ref=pressure.get("doi") or title,
                metadata=pressure,
            ))
        elif p_type in {"open_question", "concept_gap"}:
            add(_task_entry(
                "sharpen_pressure_question",
                f"Sharpen pressure question: {title}",
                pressure.get("detail", ""),
                priority=priority + 1,
                action_hint="sharpen_question",
                source_type="pressure_point",
                source_ref=pressure.get("direction") or title,
                metadata=pressure,
            ))

    for move in next_moves:
        add(_task_entry(
            "next_kernel_move",
            move.get("title", ""),
            move.get("detail", ""),
            priority=4,
            action_hint=move.get("type", ""),
            source_type="next_move",
            source_ref=move.get("title", ""),
            metadata=move,
        ))

    tasks = [_calibrate_kernel_task(task) for task in tasks]
    return _rank_diverse_kernel_tasks(tasks, limit)


def materialize_kernel_tasks(db, summary: dict) -> dict:
    """Persist deterministic kernel tasks generated from current state."""
    generated = generate_kernel_tasks(summary)
    rows = []
    for rank, task in enumerate(generated, start=1):
        metadata = dict(task.get("metadata") or {})
        metadata["queue_rank"] = rank
        rows.append(db.upsert_kernel_task(
            task_type=task.get("task_type", "review"),
            title=task.get("title", ""),
            description=task.get("description", ""),
            priority=task.get("priority", 0),
            action_hint=task.get("action_hint", ""),
            object_key=task.get("object_key", ""),
            source_type=task.get("source_type", ""),
            source_ref=task.get("source_ref", ""),
            metadata=metadata,
        ))
    counts: dict[str, int] = {}
    for row in rows:
        task_type = row.get("task_type", "review")
        counts[task_type] = counts.get(task_type, 0) + 1
    return {
        "total": len(rows),
        "counts": counts,
        "task_keys": [row.get("task_key", "") for row in rows if row.get("task_key")],
    }


def build_judgment_kernel_summary(
    papers: list[dict] | None = None,
    concepts: list[dict] | None = None,
    frameworks: list[dict] | None = None,
    constraints: list[dict] | None = None,
    deltas: list[dict] | None = None,
    verifications: list[dict] | None = None,
    experiments: list[dict] | None = None,
    trajectories: list[dict] | None = None,
    memory_summary: dict | None = None,
    story_groups: list[dict] | None = None,
    kernel_objects: list[dict] | None = None,
    revision_events: list[dict] | None = None,
    kernel_tasks: list[dict] | None = None,
) -> dict:
    """Build a compact judgment-kernel state from persisted ScholarHound data."""
    papers = papers or []
    concepts = concepts or []
    frameworks = frameworks or []
    constraints = constraints or []
    deltas = deltas or []
    verifications = verifications or []
    experiments = experiments or []
    trajectories = trajectories or []
    memory_summary = memory_summary or {}
    kernel_objects = kernel_objects or []
    revision_events = revision_events or []
    kernel_tasks = kernel_tasks or []

    beliefs = _memory_items(memory_summary, "beliefs", 6)
    rejected = _memory_items(memory_summary, "rejected", 4)
    contradictions = _memory_items(memory_summary, "contradictions", 5)
    decisions = _memory_items(memory_summary, "decisions", 4)
    open_questions = _story_open_questions(story_groups, 8)
    concept_questions = _concept_open_questions(concepts, 8)
    for question in concept_questions:
        if question["title"] not in {q["title"] for q in open_questions}:
            open_questions.append(question)
        if len(open_questions) >= 8:
            break

    candidate_claims = _candidate_claims(frameworks, constraints, deltas, 8)
    pressure_points = _pressure_points(
        papers,
        concepts,
        constraints,
        verifications,
        memory_summary,
        open_questions,
        8,
    )
    next_moves = _next_moves(pressure_points, open_questions, candidate_claims, 4)

    status = "forming"
    if contradictions:
        status = "unstable"
    elif pressure_points:
        status = "learning"
    elif beliefs:
        status = "stable"

    counts = {
        "papers": len(papers),
        "concepts": len(concepts),
        "frameworks": len(frameworks),
        "constraints": len(constraints),
        "experiments": len(experiments),
        "trajectories": len(trajectories),
        "beliefs": len((memory_summary or {}).get("beliefs", [])),
        "contradictions": len((memory_summary or {}).get("contradictions", [])),
        "open_questions": len(open_questions),
        "pressure_points": len(pressure_points),
        "candidate_claims": len(candidate_claims),
        "kernel_objects": len(kernel_objects),
        "revision_events": len(revision_events),
        "kernel_tasks": len(kernel_tasks),
        "open_kernel_tasks": len([task for task in kernel_tasks if task.get("status") == "open"]),
    }

    top_question = open_questions[0]["title"] if open_questions else ""
    top_pressure = pressure_points[0]["title"] if pressure_points else ""
    summary_bits = [
        f"{counts['beliefs']} beliefs",
        f"{counts['open_questions']} open questions",
        f"{counts['pressure_points']} pressure points",
    ]
    if top_question:
        summary_bits.append(f"next: {top_question}")

    return {
        "name": "ScholarHound Judgment Kernel",
        "definition": (
            "An intelligent memory palace for scientific judgment: LLMs widen input "
            "bandwidth, while the kernel preserves beliefs, contradictions, open "
            "questions, and model-changing pressure."
        ),
        "status": status,
        "counts": counts,
        "pulse": {
            "has_signal": bool(papers or concepts or beliefs or open_questions or pressure_points),
            "status": status,
            "summary": " · ".join(summary_bits),
            "beliefs": counts["beliefs"],
            "open_questions": counts["open_questions"],
            "pressure_points": counts["pressure_points"],
            "candidate_claims": counts["candidate_claims"],
            "top_question": top_question,
            "top_pressure": top_pressure,
        },
        "memory_palace": {
            "active_beliefs": beliefs,
            "rejected": rejected,
            "contradictions": contradictions,
            "decisions": decisions,
            "open_questions": open_questions,
            "candidate_claims": candidate_claims,
            "pressure_points": pressure_points,
            "next_moves": next_moves,
            "kernel_objects": kernel_objects,
            "revision_events": revision_events,
            "kernel_tasks": kernel_tasks,
        },
    }
