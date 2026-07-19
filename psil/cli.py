import json
from datetime import date, timedelta
from pathlib import Path

import click

from psil.state_change import (
    StateChangeValidationError,
    append_event,
    default_event_log_path,
    read_events,
    render_events_markdown,
    sample_event,
    validate_event_log,
)
from psil.v3_kernel import (
    V3KernelValidationError,
    apply_research_judgment_decision,
    create_kernel_intake_assessment,
    create_research_judgment_decision,
    default_v3_kernel_dir,
    export_kernel_state,
    seed_minimal_v3_kernel,
    validate_v3_kernel,
)


def get_db_path() -> str:
    from pathlib import Path
    db_dir = Path.home() / ".psil"
    db_dir.mkdir(exist_ok=True)
    return str(db_dir / "psil.db")


@click.group()
def main():
    """PSIL — Personal Scientific Intelligence Layer (Research Radar v3)"""
    pass


@main.command()
@click.option("--config", "-c", default=None, help="Path to config.yaml")
@click.option("--mode", "-m", default="daily", type=click.Choice(["daily", "weekly"]))
@click.option("--dry-run", is_flag=True, help="Print digest to stdout instead of writing to vault")
def scan(config, mode, dry_run):
    """Ingest papers, score, rank, and generate digest."""
    from psil.config import load_config
    from psil.digest.render import render_digest
    from psil.digest.vault import write_digest
    from psil.compress import run_compress
    from psil.ingest.orchestrator import ingest_all_journals
    from psil.ingest.toc import enrich_paper_toc_image
    from psil.rank.concepts import get_matched_concepts, is_non_research
    from psil.rank.identity import load_identity, save_identity
    from psil.rank.llm import LLMClient
    from psil.rank.scorer import prefilter_papers
    from psil.store.db import Database
    from psil.kernel import (
        apply_paper_type_router,
        check_domain_consistency,
        classify_signal_strength,
        constraint_trajectory_feedback,
        detect_blind_spot,
        detect_semantic_drift,
        estimate_evidence_strength,
        kernel_classify_paper,
    )

    cfg = load_config(config)
    db = Database(get_db_path())
    db.create_tables()

    # Load research identity
    identity = load_identity()

    query_date = date.today()
    days_back = 1 if mode == "daily" else 7

    click.echo(f"Ingesting papers from {len(cfg['journals'])} journals ({mode} mode)...")
    papers = ingest_all_journals(cfg["journals"], query_date, days_back=days_back)
    fetched = len(papers)

    new_papers = [p for p in papers if not db.doi_exists(p.doi)]
    click.echo(f"  Fetched: {fetched}, New: {len(new_papers)}")

    threshold = cfg.get("prefilter_threshold", 0)
    passed, ignored_pre = prefilter_papers(new_papers, threshold=threshold)
    click.echo(f"  Pre-filter passed: {len(passed)}, Ignored: {len(ignored_pre)}")

    llm_cfg = cfg["llm"]
    llm = LLMClient(
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
        model=llm_cfg.get("model", "deepseek-chat"),
    )

    high_priority = []
    important = []
    potential = []
    watchlist = []
    commentary = []
    llm_used = 0
    kernel_only = 0
    low_priority = []
    ignored_content = []
    concept_feed_entries = []
    concept_gap_entries = []

    for paper, score in passed:
        combined_text = f"{paper.title} {paper.abstract}"

        # Check for non-research content first
        if is_non_research(paper.title, paper.abstract):
            ignored_content.append(paper)
            click.echo(f"  IGNORE (non-research): {paper.title[:80]}...")
            continue

        matched = get_matched_concepts(combined_text)
        signal_names = [name for name, _ in matched]

        # Kernel tiered dispatch: FULL / MEDIUM / LOW
        dispatch = classify_signal_strength(matched)

        if dispatch["tier"] == "LOW":
            # Check for blind spots before killing
            blind = detect_blind_spot(paper, matched, paper.journal, db)
            if blind["is_blind_spot"]:
                # This paper might be relevant despite weak keywords
                click.echo(f"  Blind spot: {paper.title[:80]}... "
                            f"({blind['reason'][:60]})")
                reasoning = kernel_classify_paper(paper, matched)
                reasoning["signal_tier"] = "BLIND_SPOT"
                reasoning["signal_score"] = score
                reasoning["matched_signals"] = signal_names
                reasoning["blind_spot"] = blind
            else:
                click.echo(f"  Kernel: {paper.title[:80]}...")
                reasoning = kernel_classify_paper(paper, matched)
                reasoning["signal_score"] = score
                reasoning["matched_signals"] = signal_names
                kernel_only += 1
        else:
            click.echo(f"  Ranking: {paper.title[:80]}...")
            llm_used += 1
            try:
                reasoning = llm.rank(
                    paper, identity,
                    matched_signals=", ".join(signal_names) if signal_names else ""
                )
            except Exception as e:
                click.echo(f"    LLM error: {e}, treating as LOW PRIORITY")
                low_priority.append((paper, score, signal_names, {}))
                continue

        tier = reasoning.get("signal_tier", "LOW_PRIORITY").strip().upper()
        concept_name = reasoning.get("concept_name", "").strip()

        # Kernel-level Concept Support recalibration
        raw_cs = reasoning.get("concept_support", 0)
        kernel_cs = _kernel_concept_support(reasoning, concept_name, db)

        # Kernel check 1: Domain consistency
        domain_check = check_domain_consistency(
            reasoning.get("problem_class", ""),
            reasoning.get("concept_support_name", ""),
            reasoning.get("strategic_value", ""),
        )
        if not domain_check["consistent"]:
            kernel_cs = max(0, kernel_cs - domain_check["confidence_penalty"])
            reasoning["kernel_domain_flag"] = domain_check["flag"]

        # Kernel check 2: Independent evidence strength
        evidence_est = estimate_evidence_strength(
            paper.journal, paper.abstract or "",
            reasoning.get("problem_class", ""),
            reasoning.get("novelty_type", ""),
        )
        reasoning["kernel_evidence"] = evidence_est
        # If LLM says High but kernel says Low, adjust CS down
        llm_evidence = (reasoning.get("evidence_strength", "") or "").strip()
        kernel_evidence = evidence_est.get("kernel_evidence_strength", "Medium")
        if llm_evidence == "High" and kernel_evidence == "Low":
            kernel_cs = max(0, kernel_cs - 2)
            reasoning["kernel_evidence_flag"] = "LLM overestimated evidence strength"

        reasoning["concept_support"] = kernel_cs
        reasoning["concept_support_raw"] = raw_cs

        # Recompute final score with kernel-adjusted CS and paper-type route.
        apply_paper_type_router(reasoning, paper.title, paper.abstract or "")

        # Stage 2: Validator+Critic for IMPORTANT+ papers (blackboard pattern)
        if "IMPORTANT" in tier or "HIGH" in tier:
            try:
                validation = llm.validate(paper, reasoning)
                reasoning["validation"] = validation
                # Apply score adjustments if validator suggests changes
                adj = validation.get("adjustments", {})
                if adj.get("adjusted_relevance"):
                    reasoning["relevance_raw"] = reasoning["relevance"]
                    reasoning["relevance"] = adj["adjusted_relevance"]
                if adj.get("adjusted_novelty"):
                    reasoning["novelty_raw"] = reasoning["novelty"]
                    reasoning["novelty"] = adj["adjusted_novelty"]
                if adj.get("adjusted_trajectory"):
                    reasoning["trajectory_raw"] = reasoning["trajectory_influence"]
                    reasoning["trajectory_influence"] = adj["adjusted_trajectory"]
                if adj.get("adjusted_cs"):
                    reasoning["concept_support_raw_2"] = reasoning["concept_support"]
                    reasoning["concept_support"] = adj["adjusted_cs"]
                if adj.get("adjusted_paper_type"):
                    reasoning["paper_type_raw"] = reasoning.get("paper_type", "")
                    reasoning["paper_type"] = adj["adjusted_paper_type"]
                if adj.get("adjusted_judgment_mode"):
                    reasoning["judgment_mode_raw"] = reasoning.get("judgment_mode", "")
                    reasoning["judgment_mode"] = adj["adjusted_judgment_mode"]
                # Recompute final score with adjusted values and judgment mode.
                apply_paper_type_router(reasoning, paper.title, paper.abstract or "")
                if validation.get("final_strategic_value"):
                    reasoning["strategic_value_validated"] = validation["final_strategic_value"]
            except Exception:
                pass  # validator is best-effort, not required

        # Extract concept feed entry
        # Extract concept feed entry (concept_name already set above)
        if concept_name:
            concept_feed_entries.append({
                "name": concept_name,
                "source": paper.title,
                "why_matters": reasoning.get("concept_why_matters", "").strip(),
                "connection": reasoning.get("concept_current_connection", "").strip(),
                "dictionary_update": "",
                "action": reasoning.get("concept_action", "").strip(),
            })

            # Build concept gap map entry
            gap_entry = {
                "concept": concept_name,
                "current_connection": reasoning.get("concept_current_connection", "").strip(),
                "potential_connection": reasoning.get("concept_potential_connection", "").strip(),
                "missing_link": reasoning.get("concept_missing_link", "").strip(),
                "opportunity": reasoning.get("concept_opportunity", "").strip(),
            }
            # Only include if at least one gap field is populated
            if any(v for v in gap_entry.values() if v and v != concept_name):
                concept_gap_entries.append(gap_entry)

            # Track in concept accumulation DB
            traj = reasoning.get("trajectory_influence", 0)
            tw = "high" if traj >= 7 else ("medium" if traj >= 4 else "low")
            db.upsert_concept(
                name=concept_name,
                source_doi=paper.doi,
                why_matters=reasoning.get("concept_why_matters", "").strip(),
                connection=reasoning.get("concept_current_connection", "").strip(),
                missing_link=reasoning.get("concept_missing_link", "").strip(),
                opportunity=reasoning.get("concept_opportunity", "").strip(),
                trajectory_weight=tw,
            )

            # Update identity concept momentum
            identity.update_concept_momentum(concept_name, True)

        # Doyle TMS: record justification for every concept-paper link
        csn = reasoning.get("concept_support_name", "").strip()
        if csn and csn.lower() != "none":
            db.insert_justification(
                concept_name=csn,
                paper_doi=paper.doi,
                support_type=reasoning.get("support_type", ""),
                evidence_strength=reasoning.get("evidence_strength", ""),
                justification_text=reasoning.get("why_matters", ""),
            )
            # Boost entrenchment for validated concepts
            current_ent = db.get_entrenchment(csn)
            st = reasoning.get("support_type", "")
            if "Validation" in st:
                db.set_entrenchment(csn, min(10, current_ent + 1))
            elif "Discovery" in st:
                db.set_entrenchment(csn, min(10, current_ent + 2))

        # Store in DB
        enrich_paper_toc_image(paper)
        db.insert_paper(
            paper,
            signal_score=score,
            signal_tier=tier,
            signal_trajectory=reasoning.get("trajectory_influence", 0),
            signal_action=reasoning.get("action", ""),
            llm_reasoning=json.dumps(reasoning),
            concept_name=concept_name,
            concept_drift=concept_name,
            causal=reasoning.get("causal"),
            problem_class=reasoning.get("problem_class", ""),
            novelty_type=reasoning.get("novelty_type", ""),
            evidence_type=reasoning.get("evidence_type", ""),
            strategic_value=reasoning.get("strategic_value", ""),
            concept_support_name=reasoning.get("concept_support_name", ""),
            support_type=reasoning.get("support_type", ""),
            evidence_strength=reasoning.get("evidence_strength", ""),
            concept_support_score=reasoning.get("concept_support", 0),
        )

        entry = (paper, score, signal_names, reasoning)

        if "HIGH_PRIORITY" in tier:
            high_priority.append(entry)
        elif "IMPORTANT" in tier:
            important.append(entry)
        elif "POTENTIAL" in tier:
            potential.append(entry)
        elif "WATCHLIST" in tier:
            watchlist.append(entry)
        elif "BLIND_SPOT" in tier:
            # These go to watchlist for human review
            watchlist.append(entry)
        elif "COMMENTARY" in tier:
            commentary.append(entry)
        elif "IGNORE" in tier:
            ignored_content.append(paper)
        else:
            low_priority.append(entry)

    # Save updated identity
    save_identity(identity)

    # ── Concept Validation & Repeated Opportunity Tracker ──
    ranked_all = high_priority + important + potential + watchlist + commentary + low_priority
    concept_validations = []
    # Group entries by concept_support_name (existing concepts getting new support)
    concept_groups = {}
    for entry in ranked_all:
        r = entry[3] if len(entry) > 3 else {}
        csn = (r.get("concept_support_name", "") or "").strip()
        st = (r.get("support_type", "") or "").strip()
        if csn and csn.lower() != "none" and st in ("Validation", "Extension"):
            if csn not in concept_groups:
                concept_groups[csn] = []
            concept_groups[csn].append(entry)

    for concept_name, entries in concept_groups.items():
        # Check cross-digest momentum
        momentum_count = len(entries)
        try:
            existing = db.get_concept(concept_name)
            if existing:
                momentum_count = existing.get("appearances", len(entries))
        except Exception:
            pass
        entry = entries[0]
        r = entry[3] if len(entry) > 3 else {}
        concept_validations.append({
            "concept": concept_name,
            "origin": "Existing Research Map",
            "validation_type": r.get("support_type", "Validation"),
            "supporting_papers": [e[0].title for e in entries],
            "evidence_strength": r.get("evidence_strength", "Medium"),
            "why_matters": r.get("why_matters", "")[:200],
            "trend": "Increasing" if momentum_count >= 3 else ("Stable" if momentum_count >= 2 else "One-off"),
            "momentum_count": momentum_count,
        })

    # Repeated opportunities: concepts appearing 2+ times across digests
    repeated_opportunities = []
    try:
        momentum_concepts = db.get_concept_momentum(min_appearances=2)
        for c in momentum_concepts:
            if c["appearances"] >= 2:
                repeated_opportunities.append({
                    "opportunity": c["name"],
                    "appearances": c["appearances"],
                    "sources": c.get("source_doi", ""),
                    "related": c.get("connection", ""),
                    "trend": "Increasing" if c["appearances"] >= 3 else "Stable",
                    "interpretation": c.get("why_matters", ""),
                    "action": "Promote weight" if c["appearances"] >= 3 else "Keep watching",
                })
    except Exception:
        pass

    # ── Trajectory Layer ──
    # Seed trajectories from identity
    trajectory_map = {
        "Molecular Bioelectronics": ["Molecular Bioelectronics", "Small-Molecule Bioelectronic Recognition", "OECT biosensing", "Adaptive Biointerfaces"],
        "Organoid + EV + Sensing": ["Organoid+EV+Sensing", "Functional EV Phenotyping", "Organoid-Derived Readout", "Activity-Based Biomarkers"],
        "Functional EV Diagnostics": ["Functional EV Phenotyping", "Single-Entity Resolution", "Activity-Based Biomarkers"],
        "Nanophotonics-Enabled Photochemistry": ["Nanophotonics-Enabled Photochemistry", "Photon Utilization", "Nanostructure as Active Optical Participant", "Radiative Q-factor Modulation"],
        "Mechanobiology-Enabled Sensing": ["Mechanobiology-Enabled Sensing", "Adaptive Biointerfaces"],
        "Alzheimer's Diagnostic-Therapeutic Systems": ["Organoid+EV+Sensing", "Activity-Based Biomarkers", "Disease-Relevant Functional Readout"],
    }
    db.init_trajectories(list(trajectory_map.keys()))

    # Update trajectory confidence from today's concept validations and discoveries
    trajectory_updates = []
    for traj_name, concept_keys in trajectory_map.items():
        evidence_added = 0
        discovery_added = 0
        validation_added = 0
        supporting = set()

        for entry in ranked_all:
            r = entry[3] if len(entry) > 3 else {}
            csn = (r.get("concept_support_name", "") or "").strip()
            st = (r.get("support_type", "") or "").strip()
            if not csn:
                continue
            # Check if this concept maps to this trajectory
            matched = False
            for ck in concept_keys:
                if ck.lower() in csn.lower() or csn.lower() in ck.lower():
                    matched = True
                    break
            if not matched:
                continue

            evidence_added += 1
            supporting.add(csn)
            if "Discovery" in st:
                discovery_added += 1
            elif "Validation" in st or "Extension" in st:
                validation_added += 1

        if evidence_added > 0:
            new_confidence = "Increasing" if evidence_added >= 2 else "Stable"
            db.update_trajectory(
                name=traj_name,
                confidence=new_confidence,
                evidence_delta=evidence_added,
                discovery_delta=discovery_added,
                validation_delta=validation_added,
                supporting_concepts=", ".join(supporting),
            )
            trajectory_updates.append({
                "name": traj_name,
                "confidence": new_confidence,
                "evidence_added": evidence_added,
                "supporting_concepts": list(supporting),
            })

    # Kernel: Constraint → Trajectory feedback
    constraint_feedback = constraint_trajectory_feedback(db, trajectory_map)
    for tname, fb in constraint_feedback.items():
        click.echo(f"  Trajectory '{tname}': {fb['old_confidence']} → {fb['new_confidence']} "
                    f"(violations: {fb['violations']}, supports: {fb['supports']})")

    # AGM: auto-contract weakest concept when contradiction detected
    verifications = db.get_verifications()
    contracted = 0
    for v in verifications:
        if v.get("result") == "violated":
            cname = v.get("constraint_name", "")
            # Find concepts linked to this constraint and contract the weakest
            linked_concepts = []
            for entry in ranked_all:
                r = entry[3] if len(entry) > 3 else {}
                csn = (r.get("concept_support_name") or "").strip()
                if csn and csn.lower() != "none":
                    linked_concepts.append(csn)
            if linked_concepts:
                # Find the weakest among them
                weakest = min(linked_concepts or ["unknown"],
                              key=lambda cn: db.get_entrenchment(cn))
                result = db.contract_concept(
                    weakest,
                    reason=f"Constraint violation: {cname}",
                )
                contracted += 1

    if contracted:
        click.echo(f"  AGM: contracted {contracted} concepts due to constraint violations")

    # Kernel: Semantic drift detection for concepts with 3+ appearances
    drift_alerts = []
    momentum_concepts = db.get_concept_momentum(min_appearances=3)
    for mc in momentum_concepts[:5]:
        drift = detect_semantic_drift(db, mc["name"])
        if drift.get("drift_detected"):
            drift_alerts.append({"concept": mc["name"], **drift})
            click.echo(f"  Drift alert: '{mc['name']}' — {drift.get('reason', '')}")

    # Run compression BEFORE digest so results can be included
    ranked_entries = high_priority + important + potential + watchlist + low_priority
    papers_with_causal = 0
    for entry in ranked_entries:
        reasoning = entry[3] if len(entry) > 3 else {}
        if reasoning.get("causal", {}).get("transformation", ""):
            papers_with_causal += 1

    compress_result = None
    if papers_with_causal >= 3:
        click.echo("  Running concept compression...")
        compress_result = run_compress(db, llm, days_back=7)
        click.echo(f"  Patterns: {compress_result['stats']['patterns_found']}, "
                    f"Frameworks: {compress_result['stats']['frameworks_found']}")

    # Get momentum data for emerging direction detection
    momentum = db.get_emerging_concepts(threshold=2)

    # Generate daily summary
    daily_summary = _generate_summary(
        high_priority, important, potential, watchlist,
        concept_feed_entries, momentum
    )

    click.echo(f"  LLM-ranked: {llm_used}, Kernel-classified: {kernel_only}")

    # ── Pipeline Gates (AutoResearchClaw-inspired) ──
    gate_flags = []

    # Gate α: Kernel-LLM disagreement on evidence strength
    evidence_disagreements = 0
    for entry in ranked_all:
        r = entry[3] if len(entry) > 3 else {}
        ke = r.get("kernel_evidence", {})
        llm_es = (r.get("evidence_strength", "") or "").strip()
        kernel_es = ke.get("kernel_evidence_strength", "Medium")
        if kernel_es == "Low" and llm_es == "High":
            evidence_disagreements += 1
    if evidence_disagreements >= 3:
        gate_flags.append(f"⚠️ Gate α: {evidence_disagreements} papers with kernel-LLM evidence disagreement")

    # Gate β: Blind spot papers routing to human review
    blind_spot_count = len([e for e in ranked_all if len(e) > 3 and e[3].get("blind_spot", {}).get("is_blind_spot")])
    if blind_spot_count > 0:
        gate_flags.append(f"🔍 Gate β: {blind_spot_count} blind spot papers flagged for review")

    # Gate γ: Trajectory confidence decreasing
    decreasing_trajs = [tname for tname, fb in constraint_feedback.items() if fb.get("new_confidence") == "Decreasing"]
    if decreasing_trajs:
        gate_flags.append(f"🔴 Gate γ: Trajectory confidence DECREASING for: {', '.join(decreasing_trajs)}")

    if gate_flags:
        click.echo("  ═══ Pipeline Gates ═══")
        for flag in gate_flags:
            click.echo(f"  {flag}")

    # Persist kernel state
    db.set_kernel_state("last_scan_papers", str(fetched), "metrics")
    db.set_kernel_state("last_scan_llm_used", str(llm_used), "metrics")
    db.set_kernel_state("last_scan_kernel_only", str(kernel_only), "metrics")
    db.set_kernel_state("last_scan_high", str(len(high_priority)), "metrics")
    db.set_kernel_state("last_scan_concepts", str(len(concept_feed_entries)), "metrics")
    click.echo(f"  HIGH PRIORITY: {len(high_priority)}, IMPORTANT: {len(important)}, "
               f"POTENTIAL: {len(potential)}, WATCHLIST: {len(watchlist)}, "
               f"LOW: {len(low_priority)}, IGNORE: {len(ignored_content)}")
    click.echo(f"  Concepts tracked: {len(concept_feed_entries)}, "
               f"Gap mapped: {len(concept_gap_entries)}, "
               f"Momentum concepts: {len(momentum)}")

    digest_content = render_digest(
        query_date, high_priority, important, potential, watchlist,
        commentary, low_priority, ignored_content, concept_feed_entries,
        concept_gap_entries, concept_validations, repeated_opportunities,
        trajectory_updates, daily_summary, compress_result
    )

    if dry_run:
        click.echo("\n" + digest_content)
    else:
        path = write_digest(cfg["vault_path"], query_date, digest_content)
        click.echo(f"  Digest written to: {path}")

    db.insert_log(
        fetched=fetched,
        new=len(new_papers),
        high=len(high_priority),
        maybe=len(important) + len(potential) + len(watchlist),
        ignored=len(low_priority) + len(ignored_content),
    )


def _kernel_concept_support(reasoning: dict, concept_name: str, db) -> int:
    """Kernel-level Concept Support scoring.

    Takes the LLM's raw CS as a starting point, then adjusts based on
    objective evidence: support type, evidence strength, concept map match,
    and cross-paper momentum.
    """
    raw = reasoning.get("concept_support", 0)
    evidence = (reasoning.get("evidence_strength", "") or "").strip()
    support_type = (reasoning.get("support_type", "") or "").strip()
    cs_name = (reasoning.get("concept_support_name", "") or "").strip()

    # Start from LLM score
    score = float(raw)

    # Evidence strength adjustment
    if "High" in evidence:
        score = min(10, score + 1)
    elif "Low" in evidence:
        score = max(0, score - 2)

    # Support type adjustment
    if "Discovery" in support_type:
        score = min(10, score + 1)  # new concepts are valuable
    elif "Validation" in support_type:
        score = min(10, score + 0.5)  # independent validation is good
    elif "Weak Signal" in support_type:
        score = max(0, score - 1)

    # Check if concept exists in research map (existing concept gets bonus)
    if cs_name and cs_name.lower() != "none":
        try:
            existing = db.get_concept(cs_name)
            if existing:
                appearances = existing.get("appearances", 0)
                if appearances >= 3:
                    score = min(10, score + 1)  # momentum bonus
                elif appearances >= 5:
                    score = min(10, score + 2)  # established concept
        except Exception:
            pass

    return round(max(0, min(10, score)))


def _generate_summary(high_priority, important, potential, watchlist,
                       concept_feed, momentum_concepts) -> dict:
    """Generate the daily summary with v3 fields."""

    def best_by_key(entries, key, default=""):
        best = None
        best_score = -1
        for entry in entries:
            r = entry[3] if len(entry) > 3 else {}
            s = r.get(key, 0)
            if isinstance(s, str):
                try:
                    s = float(s)
                except ValueError:
                    s = 0
            if s > best_score:
                best_score = s
                best = entry
        if best:
            return best[0].title
        return default

    # Collect all ranked papers across tiers for summary fallback
    ranked_all = high_priority + important + potential + watchlist

    # Best paper: highest final score across ALL tiers
    best_paper = best_by_key(ranked_all, "final_score")

    # Best concept
    best_concept = concept_feed[0].get("name", "") if concept_feed else ""

    # Most relevant to current: highest relevance across all tiers
    most_relevant = best_by_key(ranked_all, "relevance")

    # Most relevant to future: highest trajectory_influence across all tiers
    most_future = best_by_key(ranked_all, "trajectory_influence")

    # Read first: highest NOVELTY (different metric from best_paper)
    read_first = best_by_key(ranked_all, "novelty")

    # Concept to add: first with "Add to dictionary" action
    concept_to_add = ""
    for c in concept_feed:
        if "add to dictionary" in c.get("action", "").lower():
            concept_to_add = c.get("name", "")
            break

    # Emerging direction from momentum
    emerging_direction = ""
    confidence = ""
    if momentum_concepts:
        top = momentum_concepts[0]
        emerging_direction = top.get("name", "")
        apps = top.get("appearances", 0)
        if apps >= 5:
            confidence = "High"
        elif apps >= 3:
            confidence = "Medium"
        else:
            confidence = "Low — early signal"

    return {
        "best_paper": best_paper,
        "best_concept": best_concept,
        "most_relevant_current": most_relevant,
        "most_relevant_future": most_future,
        "read_first": read_first,
        "concept_to_add": concept_to_add,
        "emerging_direction": emerging_direction,
        "confidence": confidence,
    }


@main.command()
@click.option("--config", "-c", default=None, help="Path to config.yaml")
@click.option("--days", "-d", default=7, help="Look back N days for causal data")
def compress(config, days):
    """Run concept compression pipeline: discover logic patterns and frameworks."""
    from psil.config import load_config
    from psil.compress import run_compress
    from psil.rank.llm import LLMClient
    from psil.store.db import Database

    cfg = load_config(config)
    db = Database(get_db_path())

    llm_cfg = cfg.get("llm", {})
    api_key = llm_cfg.get("api_key", "")
    base_url = llm_cfg.get("base_url", "https://api.deepseek.com/v1")
    model = llm_cfg.get("model", "deepseek-chat")
    llm = LLMClient(api_key=api_key, base_url=base_url, model=model)

    click.echo(f"Running compression pipeline over {days} days...")
    result = run_compress(db, llm, days_back=days)

    stats = result["stats"]
    click.echo(f"Papers with causal data: {stats['papers_with_causal']}")
    click.echo(f"Logic patterns found: {stats['patterns_found']}")
    click.echo(f"Frameworks discovered: {stats['frameworks_found']}")
    click.echo("")

    if result["patterns"]:
        click.echo("## Logic Patterns")
        for p in result["patterns"]:
            click.echo(f"  - **{p['pattern_name']}** [{p.get('pattern_type', '')}]")
            click.echo(f"    {p.get('description', '')}")
            click.echo(f"    Template: {p.get('causal_template', '')}")
            click.echo("")

    if result["frameworks"]:
        click.echo("## Frameworks")
        for fw in result["frameworks"]:
            click.echo(f"  - **{fw['framework_name']}** "
                       f"(compression: {fw.get('compression_score', '?')}, "
                       f"novelty: {fw.get('novelty_score', '?')})")
            click.echo(f"    {fw.get('description', '')}")
            if fw.get("suggested_experiment"):
                click.echo(f"    → Experiment: {fw['suggested_experiment']}")
            click.echo("")


@main.command()
@click.option("--config", "-c", default=None, help="Path to config.yaml")
@click.option("--years", default=0, help="Rolling years to backfill; 0 uses 2020-01-01")
@click.option("--from-date", default=None, help="Start date YYYY-MM-DD")
@click.option("--to-date", default=None, help="End date YYYY-MM-DD")
@click.option("--focused/--all-journals", default=True, help="Use focused journal subset")
@click.option("--journal", multiple=True, help="Specific journal name; can repeat")
@click.option("--chunk", default="year", type=click.Choice(["year", "month"]), help="Crossref query chunk size")
@click.option("--threshold", default=None, type=int, help="Prefilter threshold")
@click.option("--max-papers-per-journal", default=0, help="Safety cap per journal; 0 means unlimited")
@click.option("--max-llm", default=0, help="Maximum LLM-scored papers; 0 means no cap")
@click.option("--score", is_flag=True, help="Score candidates with the current standard")
@click.option("--validate", is_flag=True, help="Run validator for high/important scored papers")
@click.option("--dry-run/--write", default=True, help="Dry run by default; use --write to store")
def backfill(config, years, from_date, to_date, focused, journal, chunk,
             threshold, max_papers_per_journal, max_llm, score, validate,
             dry_run):
    """Backfill focused journals over historical windows."""
    from psil.config import load_config
    from psil.rank.identity import load_identity, save_identity
    from psil.rank.llm import LLMClient
    from psil.store.db import Database

    from psil.backfill import (
        BackfillJournalResult,
        BackfillResult,
        DEFAULT_BACKFILL_START,
        default_backfill_window,
        harvest_journal,
        parse_date,
        score_and_store_candidate,
        select_backfill_journals,
        summarize_candidates,
    )

    cfg = load_config(config)
    db = Database(get_db_path())
    db.create_tables()

    end = parse_date(to_date) or date.today()
    explicit_start = parse_date(from_date)
    if explicit_start:
        start = explicit_start
    elif years and years > 0:
        start = default_backfill_window(years, today=end)[0]
    else:
        start = DEFAULT_BACKFILL_START
    threshold = cfg.get("prefilter_threshold", 0) if threshold is None else threshold

    journals = select_backfill_journals(
        cfg.get("journals", []),
        focused=focused,
        names=journal,
    )
    click.echo(f"Backfill window: {start} → {end}")
    click.echo(f"Journals: {len(journals)} ({'focused' if focused and not journal else 'selected/all'})")
    click.echo(f"Mode: {'DRY-RUN' if dry_run else 'WRITE'} · score={'yes' if score else 'no'} · threshold={threshold}")
    if score and dry_run:
        click.echo("  Note: --score with dry-run reports candidates only; use --write to store.")
    click.echo("")

    identity = load_identity()
    llm = None
    if score and not dry_run:
        llm_cfg = cfg.get("llm", {})
        llm = LLMClient(
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
            model=llm_cfg.get("model", "deepseek-chat"),
        )

    result = BackfillResult(start_date=start, end_date=end)
    total_llm = 0
    for j in journals:
        name = j.get("name", "")
        click.echo(f"→ {name}")
        jr = BackfillJournalResult(journal=name)
        try:
            papers = harvest_journal(
                j,
                start,
                end,
                chunk=chunk,
                max_papers=max_papers_per_journal,
            )
        except Exception as exc:
            click.echo(f"  fetch error: {exc}")
            result.journals.append(jr)
            continue

        passed, ignored, new_count = summarize_candidates(papers, db, threshold=threshold)
        jr.fetched = len(papers)
        jr.new = new_count
        jr.passed = len(passed)
        jr.ignored = len(ignored)
        click.echo(f"  fetched={jr.fetched} new={jr.new} prefilter_passed={jr.passed} ignored={jr.ignored}")

        if score and not dry_run:
            for paper, signal_score in passed:
                use_llm = True
                if max_llm and total_llm >= max_llm:
                    use_llm = False
                tier, used_llm = score_and_store_candidate(
                    paper,
                    signal_score,
                    db,
                    identity,
                    llm=llm,
                    use_llm=use_llm,
                    validate=validate,
                )
                if used_llm:
                    total_llm += 1
                    jr.llm_used += 1
                jr.stored += 1
                jr.by_tier[tier] = jr.by_tier.get(tier, 0) + 1
            click.echo(f"  stored={jr.stored} llm_used={jr.llm_used} tiers={jr.by_tier}")

        result.journals.append(jr)

    if score and not dry_run:
        save_identity(identity)
        high = sum(j.by_tier.get("HIGH_PRIORITY", 0) for j in result.journals)
        maybe = sum(
            j.by_tier.get(t, 0)
            for j in result.journals
            for t in ("IMPORTANT", "POTENTIAL", "WATCHLIST", "BLIND_SPOT")
        )
        ignored = sum(
            j.ignored + j.by_tier.get("IGNORE", 0) + j.by_tier.get("LOW_PRIORITY", 0)
            for j in result.journals
        )
        db.insert_log(
            fetched=result.fetched,
            new=result.new,
            high=high,
            maybe=maybe,
            ignored=ignored,
        )

    click.echo("")
    click.echo("Backfill summary")
    click.echo(f"  fetched={result.fetched}")
    click.echo(f"  new={result.new}")
    click.echo(f"  prefilter_passed={result.passed}")
    click.echo(f"  stored={result.stored}")
    click.echo(f"  llm_used={result.llm_used}")


@main.command("refresh-toc")
@click.option("--limit", default=25, type=int, help="Rows to try; 0 means no limit")
@click.option("--tier", multiple=True, help="Signal tier to refresh; defaults to HIGH_PRIORITY and IMPORTANT")
@click.option("--journal", multiple=True, help="Restrict to journal name; can be passed multiple times")
@click.option("--all-tiers", is_flag=True, help="Try every tier instead of the default high/important subset")
@click.option("--dry-run", is_flag=True, help="Report found image URLs without writing them")
def refresh_toc(limit, tier, journal, all_tiers, dry_run):
    """Backfill missing TOC/article image URLs for stored papers."""
    from psil.ingest.toc import enrich_paper_toc_image
    from psil.store.db import Database
    from psil.store.models import Paper

    db = Database(get_db_path())
    db.create_tables()

    rows = [
        row for row in db.get_all_papers()
        if not (row.get("toc_image_url") or "").strip() and row.get("doi")
    ]
    if not all_tiers:
        allowed_tiers = {t.upper().strip() for t in tier if t.strip()}
        if not allowed_tiers:
            allowed_tiers = {"HIGH_PRIORITY", "IMPORTANT"}
        rows = [
            row for row in rows
            if (row.get("signal_tier") or "").upper() in allowed_tiers
        ]
    if journal:
        allowed_journals = {j.lower().strip() for j in journal if j.strip()}
        rows = [
            row for row in rows
            if (row.get("journal") or "").lower().strip() in allowed_journals
        ]
    if limit and limit > 0:
        rows = rows[:limit]

    click.echo(f"TOC refresh candidates: {len(rows)}")
    click.echo(f"Mode: {'DRY-RUN' if dry_run else 'WRITE'}")

    found = 0
    updated = 0
    missed = 0
    for row in rows:
        paper = Paper.from_dict(row)
        image_url = enrich_paper_toc_image(paper)
        if image_url:
            found += 1
            if not dry_run:
                updated += db.update_paper_toc_image_url(paper.doi, image_url)
            click.echo(f"  found: {paper.doi} -> {image_url}")
        else:
            missed += 1
            click.echo(f"  missing: {paper.doi} ({paper.journal})")

    click.echo("")
    click.echo("TOC refresh summary")
    click.echo(f"  found={found}")
    click.echo(f"  updated={updated}")
    click.echo(f"  missed={missed}")


@main.command("import-local")
@click.option("--config", "-c", default=None, help="Path to config.yaml")
@click.option(
    "--corpus-root", "--nrb-root", "corpus_root",
    default=None,
    help="Path to copied seed-corpus folder; defaults to <vault_path>/NRB",
)
@click.option("--manifests/--no-manifests", default=True, help="Import curated local DOI manifests")
@click.option("--audits/--no-audits", default=True, help="Import fact-check and terminology audit guardrails")
@click.option("--dry-run", is_flag=True, help="Parse and report without writing to the database")
def import_local(config, corpus_root, manifests, audits, dry_run):
    """Import local seed-corpus metadata and review-audit guardrails."""
    from psil.config import load_config
    from psil.local_import import run_local_import
    from psil.store.db import Database

    cfg = load_config(config)
    db = Database(":memory:" if dry_run else get_db_path())
    stats = run_local_import(
        db,
        cfg.get("vault_path", "."),
        corpus_root=corpus_root,
        include_manifests=manifests,
        include_audits=audits,
        dry_run=dry_run,
    )

    click.echo(f"Vault: {stats['vault_path']}")
    click.echo(f"Corpus root: {stats['corpus_root']}")
    if "manifests" in stats:
        m = stats["manifests"]
        click.echo(
            f"Manifests: {m['rows_seen']} rows, {m['unique_sources']} unique DOI sources, "
            f"{m['papers_inserted']} new paper rows"
        )
        if m["missing_manifests"]:
            click.echo(f"Missing manifests: {', '.join(m['missing_manifests'])}")
    if "audits" in stats:
        fact = stats["audits"]["fact_check"]
        term = stats["audits"]["terminology"]
        click.echo(
            f"Fact-check: {fact['claims']} claims, {fact['constraints']} constraints "
            f"(found={fact['report_found']})"
        )
        click.echo(
            f"Terminology: {term['rules']} rules, {term['constraints']} constraints "
            f"(found={term['report_found']})"
        )
    if dry_run:
        click.echo("Dry run only; no database changes written.")


@main.command()
@click.option("--config", "-c", default=None, help="Path to config.yaml")
def install(config):
    """Install launchd agent for daily scheduling."""
    from psil.config import load_config

    cfg = load_config(config)
    schedule = cfg.get("schedule", {})
    run_time = schedule.get("time", "07:00")
    mode = schedule.get("mode", "daily")

    hour, minute = run_time.split(":")

    import plistlib
    from pathlib import Path

    plist = {
        "Label": "com.psil.daily-scan",
        "ProgramArguments": ["/usr/bin/env", "psil", "scan", "--mode", mode],
        "StartCalendarInterval": {
            "Hour": int(hour),
            "Minute": int(minute),
        },
        "StandardOutPath": f"{Path.home()}/.psil/launchd.log",
        "StandardErrorPath": f"{Path.home()}/.psil/launchd.log",
        "EnvironmentVariables": {
            "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin",
            "DEEPSEEK_API_KEY": cfg["llm"]["api_key"],
        },
    }

    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / "com.psil.daily-scan.plist"

    with open(plist_path, "wb") as f:
        plistlib.dump(plist, f)

    click.echo(f"launchd agent installed to: {plist_path}")
    click.echo(f"Runs daily at {hour}:{minute}")
    click.echo("To activate: launchctl load ~/Library/LaunchAgents/com.psil.daily-scan.plist")


@main.command()
def config_path():
    """Print the default config path."""
    from psil.config import DEFAULT_CONFIG_PATH

    click.echo(str(DEFAULT_CONFIG_PATH))


@main.command()
def drift():
    """Show concept momentum and research drift report."""
    from psil.rank.identity import load_identity
    from psil.store.db import Database

    db = Database(get_db_path())
    db.create_tables()
    identity = load_identity()

    momentum = db.get_concept_momentum(min_appearances=1)
    click.echo("## Concept Momentum Report")
    click.echo("")
    if momentum:
        click.echo("### Concepts Gaining Momentum (appearances ≥ 3)")
        for c in momentum:
            if c["appearances"] >= 3:
                click.echo(f"  - **{c['name']}** — {c['appearances']} appearances, "
                           f"status: {c['status']}, weight: {c['trajectory_weight']}")
        click.echo("")
        click.echo("### All Tracked Concepts")
        for c in momentum:
            click.echo(f"  - {c['name']} ({c['appearances']}×, {c['status']})")
    else:
        click.echo("No concepts tracked yet. Run `psil scan` first.")
        return

    identity_report = identity.get_momentum_report()
    if identity_report["increasing"]:
        click.echo("")
        click.echo("### Increasing")
        for c, data in identity_report["increasing"][:10]:
            click.echo(f"  - {c} ({data['appearances']}×)")

    click.echo("")
    click.echo(f"Tracked concepts (identity): {len(identity.concept_momentum)}")


@main.command()
def identity():
    """Show current research identity and suggest updates."""
    from psil.rank.identity import load_identity

    ident = load_identity()
    click.echo("## Current Research Identity")
    click.echo("")
    click.echo("### Current Core")
    for item in ident.current_core:
        click.echo(f"  - {item}")
    click.echo("")
    click.echo("### Emerging Directions")
    for item in ident.emerging_directions:
        click.echo(f"  - {item}")
    click.echo("")
    click.echo("### Long-Term Vision")
    for item in ident.long_term_vision:
        click.echo(f"  - {item}")
    click.echo("")
    click.echo(f"Last updated: {ident.last_updated}")
    click.echo(f"Concepts tracked: {len(ident.concept_momentum)}")


@main.command()
def benchmark():
    """Run the 15-year benchmark ingestion."""
    from psil.benchmark.ingest import create_tables, fetch_journal, get_progress, REVIEW_JOURNALS, NATURE_SUB_JOURNALS

    create_tables()
    all_journals = REVIEW_JOURNALS + NATURE_SUB_JOURNALS
    click.echo(f"Ingesting {len(all_journals)} journals (2011–2026)...")
    click.echo("This will take ~2 hours for the full corpus.")
    click.echo("Press Ctrl+C to stop at any time (progress is saved).\n")

    total = 0
    for name, issn in all_journals:
        source = "nature_reviews" if (name, issn) in REVIEW_JOURNALS else "nature_sub"
        added = fetch_journal(issn, name, source=source)
        total += added
        click.echo(f"  → {name}: +{added} new papers")

    prog = get_progress()
    click.echo(f"\nDone. Total: {prog['total_papers']} papers in benchmark DB.")


@main.command("state-validate")
@click.option(
    "--path",
    "event_log_path",
    default=None,
    help="Path to state_changes.jsonl. Defaults to ./kernel/state_changes.jsonl.",
)
def state_validate(event_log_path):
    """Validate the structured State Change JSONL log."""
    path = Path(event_log_path) if event_log_path else default_event_log_path(Path.cwd())
    events, issues = validate_event_log(path)
    if issues:
        for issue in issues:
            click.echo(issue.format(), err=True)
        raise click.ClickException(f"state change log failed validation: {path}")
    click.echo(f"OK: {len(events)} state change event(s) valid in {path}")


@main.command("state-render")
@click.option(
    "--path",
    "event_log_path",
    default=None,
    help="Path to state_changes.jsonl. Defaults to ./kernel/state_changes.jsonl.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Optional Markdown output path. Prints to stdout when omitted.",
)
@click.option("--title", default="State Change Log", help="Rendered Markdown title.")
def state_render(event_log_path, output, title):
    """Render validated State Change JSONL events as Markdown."""
    path = Path(event_log_path) if event_log_path else default_event_log_path(Path.cwd())
    events = read_events(path)
    rendered = render_events_markdown(events, title=title)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        click.echo(f"Rendered {len(events)} event(s) to {output_path}")
    else:
        click.echo(rendered)


@main.command("state-append")
@click.argument("event_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--path",
    "event_log_path",
    default=None,
    help="Path to state_changes.jsonl. Defaults to ./kernel/state_changes.jsonl.",
)
def state_append(event_file, event_log_path):
    """Validate and append one candidate State Change JSON event."""
    path = Path(event_log_path) if event_log_path else default_event_log_path(Path.cwd())
    event_path = Path(event_file)
    try:
        event = json.loads(event_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"invalid JSON in {event_path}: {exc}") from exc

    try:
        event = append_event(path, event)
    except StateChangeValidationError as exc:
        for issue in exc.issues:
            click.echo(issue.format(), err=True)
        raise click.ClickException(f"state change event failed validation: {event_path}") from exc

    click.echo(f"Appended {event['event_id']} to {path}")


@main.command("state-append-sample")
@click.option(
    "--path",
    "event_log_path",
    default=None,
    help="Path to state_changes.jsonl. Defaults to ./kernel/state_changes.jsonl.",
)
def state_append_sample(event_log_path):
    """Append a sample State Change event for bootstrapping and tests."""
    path = Path(event_log_path) if event_log_path else default_event_log_path(Path.cwd())
    event = append_event(path, sample_event())
    click.echo(f"Appended {event['event_id']} to {path}")


@main.command("v3-seed-minimal")
@click.option(
    "--path",
    "kernel_path",
    default=None,
    help="Path to V3 kernel directory. Defaults to ./kernel/v3.",
)
def v3_seed_minimal(kernel_path):
    """Seed the smallest V3 Evidence -> Belief -> Revision loop."""
    path = Path(kernel_path) if kernel_path else default_v3_kernel_dir(Path.cwd())
    try:
        result = seed_minimal_v3_kernel(path)
    except V3KernelValidationError as exc:
        for issue in exc.issues:
            click.echo(issue.format(), err=True)
        raise click.ClickException(f"V3 seed failed validation: {path}") from exc
    health = result["health"]
    click.echo(
        f"Seeded V3 kernel at {path}: "
        f"{health['belief_count']} belief(s), "
        f"{health['evidence_count']} evidence record(s), "
        f"{health['revision_count']} revision(s)"
    )


@main.command("v3-validate")
@click.option(
    "--path",
    "kernel_path",
    default=None,
    help="Path to V3 kernel directory. Defaults to ./kernel/v3.",
)
def v3_validate(kernel_path):
    """Validate the V3 belief kernel."""
    path = Path(kernel_path) if kernel_path else default_v3_kernel_dir(Path.cwd())
    try:
        health, issues = validate_v3_kernel(path)
    except V3KernelValidationError as exc:
        for issue in exc.issues:
            click.echo(issue.format(), err=True)
        raise click.ClickException(f"V3 kernel failed validation: {path}") from exc
    if issues:
        for issue in issues:
            click.echo(issue.format(), err=True)
        raise click.ClickException(f"V3 kernel failed validation: {path}")
    click.echo(
        f"OK: V3 kernel valid in {path} "
        f"({health['belief_count']} beliefs, "
        f"{health['evidence_count']} evidence, "
        f"{health['revision_count']} revisions)"
    )


@main.command("v3-export")
@click.option(
    "--path",
    "kernel_path",
    default=None,
    help="Path to V3 kernel directory. Defaults to ./kernel/v3.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Optional JSON output path. Defaults to ./kernel/v3/exports/kernel_state.json.",
)
def v3_export(kernel_path, output):
    """Export V3 kernel state and health as JSON."""
    path = Path(kernel_path) if kernel_path else default_v3_kernel_dir(Path.cwd())
    output_path = Path(output) if output else None
    state = export_kernel_state(path, output_path)
    if state["health"]["validation_status"] != "ok":
        for issue in state["issues"]:
            click.echo(issue, err=True)
        raise click.ClickException(f"V3 kernel export has validation issues: {path}")
    target = output_path or (path / "exports" / "kernel_state.json")
    click.echo(f"Exported V3 kernel state to {target}")


@main.command("v3-assess-briefing")
@click.argument("briefing_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--path",
    "kernel_path",
    default=None,
    help="Path to V3 kernel directory. Defaults to ./kernel/v3.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Optional JSON copy of the assessment. The JSONL ledger is always appended.",
)
def v3_assess_briefing(briefing_file, kernel_path, output):
    """Assess one Audit Secretary briefing without committing belief revisions."""
    path = Path(kernel_path) if kernel_path else default_v3_kernel_dir(Path.cwd())
    briefing_path = Path(briefing_file)
    try:
        briefing = json.loads(briefing_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"invalid JSON in {briefing_path}: {exc}") from exc
    try:
        assessment = create_kernel_intake_assessment(path, briefing)
    except V3KernelValidationError as exc:
        for issue in exc.issues:
            click.echo(issue.format(), err=True)
        raise click.ClickException(f"kernel intake assessment failed: {briefing_path}") from exc

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(assessment, indent=2, ensure_ascii=False), encoding="utf-8")

    click.echo(
        f"{assessment['decision']}: {assessment['assessment_id']} "
        f"(briefing={assessment['briefing_id'] or briefing_path.name}, "
        f"durable_change={assessment['durable_change_authorization']})"
    )


@main.command("v3-decide-briefing")
@click.argument("assessment_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("briefing_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--path",
    "kernel_path",
    default=None,
    help="Path to V3 kernel directory. Defaults to ./kernel/v3.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Optional JSON copy of the decision. The JSONL ledger is always appended.",
)
def v3_decide_briefing(assessment_file, briefing_file, kernel_path, output):
    """Create one ResearchJudgmentDecision downstream of intake."""
    path = Path(kernel_path) if kernel_path else default_v3_kernel_dir(Path.cwd())
    assessment_path = Path(assessment_file)
    briefing_path = Path(briefing_file)
    try:
        assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"invalid JSON in {assessment_path}: {exc}") from exc
    try:
        briefing = json.loads(briefing_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"invalid JSON in {briefing_path}: {exc}") from exc
    try:
        decision = create_research_judgment_decision(path, assessment, briefing)
    except V3KernelValidationError as exc:
        for issue in exc.issues:
            click.echo(issue.format(), err=True)
        raise click.ClickException(
            f"research judgment decision failed: {assessment_path}"
        ) from exc

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    click.echo(
        f"{decision['decision']}: {decision['decision_id']} "
        f"(assessment={decision['assessment_id']}, "
        f"human_review={decision['human_review_required']})"
    )


@main.command("v3-apply-decision")
@click.argument("decision_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--path",
    "kernel_path",
    default=None,
    help="Path to V3 kernel directory. Defaults to ./kernel/v3.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Optional JSON copy of the application result.",
)
def v3_apply_decision(decision_file, kernel_path, output):
    """Apply a recorded ResearchJudgmentDecision to downstream kernel objects."""
    path = Path(kernel_path) if kernel_path else default_v3_kernel_dir(Path.cwd())
    decision_path = Path(decision_file)
    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"invalid JSON in {decision_path}: {exc}") from exc
    try:
        result = apply_research_judgment_decision(path, decision)
    except V3KernelValidationError as exc:
        for issue in exc.issues:
            click.echo(issue.format(), err=True)
        raise click.ClickException(
            f"research judgment application failed: {decision_path}"
        ) from exc

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    applied_counts = {
        key: len(value)
        for key, value in result["applied"].items()
        if value
    }
    click.echo(
        f"applied {result['decision']}: {result['decision_id']} "
        f"(objects={applied_counts or {}}, skipped={len(result['skipped'])})"
    )


@main.command("v3-intake-ablation")
@click.option(
    "--db-path",
    default=None,
    help="Path to ScholarHound sqlite database. Defaults to ~/.psil/psil.db.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Markdown report path. Defaults to kernel/v3/exports/20_paper_intake_ablation_2026-06-11.md.",
)
def v3_intake_ablation(db_path, output):
    """Run the 20-paper V3 evidence-intake ablation benchmark."""
    from psil.benchmark.v3_intake import (
        DEFAULT_DB_PATH,
        default_report_path,
        render_ablation_markdown,
        run_v3_intake_ablation_from_db,
    )

    source_db = Path(db_path) if db_path else DEFAULT_DB_PATH
    result = run_v3_intake_ablation_from_db(source_db)
    target = Path(output) if output else default_report_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_ablation_markdown(result), encoding="utf-8")

    full = result["full_kernel"]
    no_consensus = result["ablations"]["without_relation_consensus"]
    no_dampening = result["ablations"]["without_confidence_dampening"]
    no_entrenchment = result["ablations"]["without_entrenchment_policy"]
    click.echo(f"Wrote V3 intake ablation report to {target}")
    click.echo(
        "Full kernel: "
        f"{result['paper_count']} papers, "
        f"relations={full['relation_counts']}, "
        f"confidence={full['final_confidence']}, "
        f"entrenchment={full['final_entrenchment']}"
    )
    click.echo(
        "Ablations: "
        f"unstable commits prevented={no_consensus['unstable_commits_prevented']} "
        f"({no_consensus['unstable_commit_reduction_pct']}%), "
        f"overconfidence reduction={no_dampening['overconfidence_reduction_pct']}%, "
        f"over-entrenchment reduction={no_entrenchment['overentrenchment_reduction_pct']}%"
    )


@main.command("v3-review-smoke")
@click.option(
    "--db-path",
    default=None,
    help="Path to ScholarHound sqlite database. Defaults to ~/.psil/psil.db.",
)
@click.option(
    "--doi",
    default="",
    help="Optional DOI to test. Defaults to the next real ranked review item.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Markdown report path. Defaults to kernel/v3/exports/real_item_review_smoke_2026-06-11.md.",
)
def v3_review_smoke(db_path, doi, output):
    """Run one real ranked paper through the V3 kernel review path without durable writes."""
    from psil.benchmark.v3_intake import (
        DEFAULT_DB_PATH,
        default_review_smoke_report_path,
        render_review_smoke_markdown,
        run_v3_review_smoke_from_db,
    )

    source_db = Path(db_path) if db_path else DEFAULT_DB_PATH
    result = run_v3_review_smoke_from_db(source_db, doi=doi)
    target = Path(output) if output else default_review_smoke_report_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_review_smoke_markdown(result), encoding="utf-8")

    revision = result["revision"]
    click.echo(f"Wrote V3 real review smoke report to {target}")
    click.echo(
        "Kernel path: "
        f"relation={result['relation']}, "
        f"action={result['action']}, "
        f"confidence_delta={revision.get('confidence_delta', 0.0)}, "
        f"entrenchment_delta={revision.get('entrenchment_delta', 0.0)}, "
        f"validation={result['validation_status']}"
    )


@main.command("v3-backfill-digest")
@click.option(
    "--db-path",
    default=None,
    help="Path to ScholarHound sqlite database. Defaults to ~/.psil/psil.db.",
)
@click.option(
    "--limit",
    default=50,
    show_default=True,
    help="Number of legacy digested papers to dry-run. Use 0 for all matching papers.",
)
@click.option(
    "--offset",
    default=0,
    show_default=True,
    help="Offset into the legacy digested paper selection.",
)
@click.option(
    "--tiers",
    default="",
    help="Comma-separated signal tiers. Defaults to all legacy ranked tiers except CURATED_LIBRARY.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Markdown report path. Defaults to kernel/v3/backfills/legacy_digest_backfill_2026-06-11.md.",
)
def v3_backfill_digest(db_path, limit, offset, tiers, output):
    """Dry-run legacy LLM digests through the V3 accountability kernel."""
    from psil.benchmark.v3_intake import (
        DEFAULT_DB_PATH,
        DEFAULT_LEGACY_BACKFILL_TIERS,
        default_legacy_backfill_report_path,
        render_legacy_backfill_markdown,
        run_v3_legacy_digest_backfill_from_db,
    )

    source_db = Path(db_path) if db_path else DEFAULT_DB_PATH
    selected_tiers = (
        [item.strip().upper() for item in tiers.split(",") if item.strip()]
        if tiers
        else DEFAULT_LEGACY_BACKFILL_TIERS
    )
    result = run_v3_legacy_digest_backfill_from_db(
        source_db,
        limit=limit,
        offset=offset,
        tiers=selected_tiers,
    )
    target = Path(output) if output else default_legacy_backfill_report_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_legacy_backfill_markdown(result), encoding="utf-8")

    full = result["full_kernel"]
    no_consensus = result["ablations"]["without_relation_consensus"]
    click.echo(f"Wrote V3 legacy digest backfill dry-run report to {target}")
    click.echo(
        "Backfill dry-run: "
        f"{result['paper_count']} papers, "
        f"relations={full['relation_counts']}, "
        f"human_queue={full['contested_queue_count']}, "
        f"pending_queue={full['pending_queue_count']}, "
        f"unstable prevented={no_consensus['unstable_commits_prevented']} "
        f"({no_consensus['unstable_commit_reduction_pct']}%), "
        f"validation={full['validation_status']}"
    )


@main.command()
@click.option("--port", "-p", default=8501, help="Port to listen on")
def serve(port):
    """Launch the HTML5 WebUI server."""
    from psil.serve import find_available_port, start
    actual_port = find_available_port(port)
    if actual_port != port:
        click.echo(f"Port {port} is busy; using {actual_port}.")
    click.echo(f"ScholarHound WebUI -> http://localhost:{actual_port}")
    start(port=actual_port, auto_port=False)


if __name__ == "__main__":
    main()
