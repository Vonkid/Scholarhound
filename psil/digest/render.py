from datetime import date

from psil.store.models import Paper


def render_digest(digest_date: date,
                  high_priority: list[tuple],
                  important: list[tuple],
                  potential: list[tuple],
                  watchlist: list[tuple],
                  commentary: list[tuple],
                  low_priority: list[tuple],
                  ignored: list,
                  concept_feed: list[dict],
                  concept_gap_map: list[dict],
                  concept_validations: list[dict],
                  repeated_opportunities: list[dict],
                  trajectory_updates: list[dict],
                  daily_summary: dict,
                  compress_result: dict | None = None) -> str:
    lines = [
        f"# Daily Scientific Signals — {digest_date.strftime('%Y-%m-%d')}",
        "",
        "> **Scoring:** Relevance / Novelty / Bridge / Trajectory Influence / Concept Support → Final Score",
        "> **Formula:** 0.25×R + 0.20×N + 0.20×B + 0.20×T + 0.15×CS",
        "> **Classification:** HIGH PRIORITY (≥8.0 or N≥9+B≥8 or B≥9+T≥8) · "
        "IMPORTANT (6.0-7.9 or T≥8+FS≥5.5 or N≥8+B≥7) · "
        "POTENTIAL (4.0-5.9) · WATCHLIST (<4.0+Legacy/Trajectory) · "
        "LOW PRIORITY (<4.0) · IGNORE (non-research only)",
        "",
    ]

    lines.append("## HIGH PRIORITY")
    lines.append("")
    if high_priority:
        for entry in high_priority:
            lines.extend(_render_detail(entry))
    else:
        lines.append("No high-priority papers today.")
        lines.append("")

    lines.append("## IMPORTANT")
    lines.append("")
    if important:
        for entry in important:
            lines.extend(_render_detail(entry))
    else:
        lines.append("No important papers today.")
        lines.append("")

    lines.append("## POTENTIAL")
    lines.append("")
    if potential:
        for entry in potential:
            lines.extend(_render_condensed(entry))
    else:
        lines.append("No potential papers today.")
        lines.append("")

    lines.append("## COMMENTARY / REVIEW WATCH")
    lines.append("")
    if commentary:
        for entry in commentary:
            lines.extend(_render_detail(entry))
    else:
        lines.append("No commentary items today.")
        lines.append("")

    lines.append("## WATCHLIST")
    lines.append("")
    if watchlist:
        for entry in watchlist:
            lines.extend(_render_watchlist(entry))
    else:
        lines.append("No watchlist papers today.")
        lines.append("")

    lines.append("## LOW PRIORITY")
    lines.append("")
    if low_priority:
        for entry in low_priority:
            paper = entry[0]
            reasoning = entry[3] if len(entry) > 3 else {}
            lines.append(f"- {paper.title} — *{paper.journal}*")
            lines.append(f"  DOI: [{paper.doi}](https://doi.org/{paper.doi})")
            lines.append(f"  Scores: R:{reasoning.get('relevance', '?')}/10 "
                         f"N:{reasoning.get('novelty', '?')}/10 "
                         f"B:{reasoning.get('bridge', '?')}/10 "
                         f"T:{reasoning.get('trajectory_influence', '?')}/10 "
                         f"→ {reasoning.get('final_score', '?')}/10")
        lines.append("")
    else:
        lines.append("No low-priority papers today.")
        lines.append("")

    lines.append("## IGNORE")
    lines.append("")
    if ignored:
        for item in ignored:
            if isinstance(item, Paper):
                lines.append(f"- {item.title} ({item.journal})")
            elif isinstance(item, tuple):
                paper = item[0]
                lines.append(f"- {paper.title} ({paper.journal})")
            else:
                lines.append(f"- {item}")
        lines.append("")
    else:
        lines.append("No ignored items today.")
        lines.append("")

    # CONCEPT FEED
    lines.append("## CONCEPT FEED")
    lines.append("")
    if concept_feed:
        for concept in concept_feed:
            lines.extend(_render_concept(concept))
    else:
        lines.append("No new concepts detected today.")
        lines.append("")

    # CONCEPT GAP MAP
    lines.append("## CONCEPT GAP MAP")
    lines.append("")
    if concept_gap_map:
        for gap in concept_gap_map:
            lines.extend(_render_concept_gap(gap))
    else:
        lines.append("No concept gaps mapped today.")
        lines.append("")

    # CONCEPT VALIDATION
    lines.append("## CONCEPT VALIDATION")
    lines.append("")
    if concept_validations:
        for cv in concept_validations:
            lines.append(f"### {cv.get('concept', '?')}")
            lines.append("")
            lines.append(f"- **Origin:** {cv.get('origin', 'Existing Research Map')}")
            lines.append(f"- **Validation Type:** {cv.get('validation_type', 'Validation')}")
            papers = cv.get('supporting_papers', [])
            if papers:
                lines.append(f"- **New Supporting Papers:** {len(papers)}")
                for p in papers[:3]:
                    lines.append(f"  - {p[:100]}")
            lines.append(f"- **Evidence Strength:** {cv.get('evidence_strength', 'Medium')}")
            lines.append(f"- **Trend:** {cv.get('trend', 'Stable')} ({cv.get('momentum_count', 0)} appearances)")
            why = cv.get('why_matters', '')
            if why:
                lines.append(f"- **Why this matters:** {why[:300]}")
            lines.append("")
    else:
        lines.append("No existing concepts received new validation today.")
        lines.append("")

    # REPEATED OPPORTUNITY TRACKER
    lines.append("## REPEATED OPPORTUNITY TRACKER")
    lines.append("")
    if repeated_opportunities:
        for ro in repeated_opportunities[:6]:
            lines.append(f"### {ro.get('opportunity', '?')}")
            lines.append("")
            lines.append(f"- **Appearances:** {ro.get('appearances', 0)}")
            sources = ro.get('sources', '')
            if sources:
                lines.append(f"- **Sources:** {sources[:200]}")
            lines.append(f"- **Related Concepts:** {ro.get('related', '—')}")
            lines.append(f"- **Trend:** {ro.get('trend', 'Stable')}")
            interp = ro.get('interpretation', '')
            if interp:
                lines.append(f"- **Interpretation:** {interp[:200]}")
            lines.append(f"- **Action:** {ro.get('action', 'Keep watching')}")
            lines.append("")
    else:
        lines.append("No repeated opportunities detected yet (need 2+ appearances across digests).")
        lines.append("")

    # TRAJECTORY UPDATE
    lines.append("## TRAJECTORY UPDATE")
    lines.append("")
    if trajectory_updates:
        for tu in trajectory_updates:
            conf_icon = {"Increasing": "📈", "Stable": "➡️", "Decreasing": "📉"}
            lines.append(f"### {tu.get('name', '?')} {conf_icon.get(tu.get('confidence', ''), '')}")
            lines.append("")
            lines.append(f"- **Confidence:** {tu.get('confidence', 'Stable')}")
            lines.append(f"- **New Evidence:** {tu.get('evidence_added', 0)} papers")
            concepts = tu.get('supporting_concepts', [])
            if concepts:
                lines.append(f"- **Supporting Concepts:** {', '.join(concepts[:5])}")
            lines.append("")
    else:
        lines.append("No trajectory updates today.")
        lines.append("")

    # DAILY SUMMARY
    lines.append("## DAILY SUMMARY")
    lines.append("")
    if daily_summary:
        for key, label in [
            ("best_paper", "Best paper today"),
            ("best_concept", "Best concept today"),
            ("most_relevant_current", "Most relevant to current projects"),
            ("most_relevant_future", "Most relevant to future trajectory"),
            ("read_first", "One paper I should actually read first"),
            ("concept_to_add", "One concept to add to the dictionary"),
            ("emerging_direction", "Potential new direction emerging"),
            ("confidence", "Confidence"),
        ]:
            val = daily_summary.get(key, "")
            if val:
                lines.append(f"- **{label}:** {val}")
        lines.append("")
    else:
        lines.append("Summary not available.")
        lines.append("")

    # COMPRESSION (Layer 2+3 output)
    if compress_result:
        lines.append("## CONCEPT COMPRESSION")
        lines.append("")

        patterns = compress_result.get("patterns", [])
        if patterns:
            lines.append("### Logic Patterns")
            lines.append("")
            for p in patterns:
                ptype = p.get("pattern_type", "Other")
                lines.append(f"- **{p.get('pattern_name', '?')}** [{ptype}]")
                desc = p.get("description", "")
                if desc:
                    lines.append(f"  {desc}")
                tmpl = p.get("causal_template", "")
                if tmpl:
                    lines.append(f"  Template: `{tmpl}`")
                lines.append("")

        frameworks = compress_result.get("frameworks", [])
        if frameworks:
            lines.append("### Frameworks")
            lines.append("")
            for fw in frameworks:
                lines.append(f"- **{fw.get('framework_name', '?')}** "
                             f"(compression: {fw.get('compression_score', '?')}, "
                             f"novelty: {fw.get('novelty_score', '?')})")
                desc = fw.get("description", "")
                if desc:
                    lines.append(f"  {desc}")
                exp = fw.get("suggested_experiment", "")
                if exp:
                    lines.append(f"  → Suggested experiment: {exp}")
                lines.append("")

        deltas = compress_result.get("deltas", [])
        if deltas:
            lines.append("### Worldview Shifts (Deltas)")
            lines.append("")
            for d in deltas:
                lines.append(f"- ~~{d.get('previous', '?')}~~ → **{d.get('new', '?')}**")
                dtext = d.get("delta", "")
                if dtext:
                    lines.append(f"  Shift: {dtext}")
                lines.append("")

    return "\n".join(lines)


def _render_detail(entry: tuple) -> list[str]:
    paper = entry[0]
    signals = entry[2] if len(entry) > 2 else []
    reasoning = entry[3] if len(entry) > 3 else {}

    lines = []
    lines.append(f"### {paper.title}")
    lines.append("")
    lines.append(f"- **Journal:** {paper.journal}")
    lines.append(f"- **DOI:** [{paper.doi}](https://doi.org/{paper.doi})")
    if signals:
        lines.append(f"- **Matched Signals:** {', '.join(signals)}")
    pc = reasoning.get('problem_class', '')
    pt = reasoning.get('paper_type', '')
    jm = reasoning.get('judgment_mode', '')
    nt = reasoning.get('novelty_type', '')
    et = reasoning.get('evidence_type', '')
    sv = reasoning.get('strategic_value', '')
    if pt: lines.append(f"- **Paper Type:** {pt}")
    if jm: lines.append(f"- **Judgment Mode:** {jm}")
    if pc: lines.append(f"- **Problem Class:** {pc}")
    if nt: lines.append(f"- **Novelty Type:** {nt}")
    if et: lines.append(f"- **Evidence Type:** {et}")
    if sv: lines.append(f"- **Strategic Value:** {sv}")
    csn = reasoning.get('concept_support_name', '')
    st = reasoning.get('support_type', '')
    es = reasoning.get('evidence_strength', '')
    if csn: lines.append(f"- **Supports:** {csn} ({st or 'N/A'}, {es or 'N/A'})")
    lines.append(f"- **Relevance:** {reasoning.get('relevance', '?')}/10")
    lines.append(f"- **Novelty:** {reasoning.get('novelty', '?')}/10")
    lines.append(f"- **Bridge:** {reasoning.get('bridge', '?')}/10")
    lines.append(f"- **Trajectory Influence:** {reasoning.get('trajectory_influence', '?')}/10")
    cs_score = reasoning.get('concept_support', '?')
    if cs_score != '?' and cs_score != 0: lines.append(f"- **Concept Support:** {cs_score}/10")
    lines.append(f"- **Final Score:** {reasoning.get('final_score', '?')}/10")
    lines.append("")

    why = reasoning.get('why_matters', '').strip()
    if why and why.lower() != 'none':
        lines.append("**Why it matters:**")
        lines.append(why)
        lines.append("")

    connection = reasoning.get('potential_connection', '').strip()
    if connection and connection.lower() != 'none':
        lines.append("**Potential connection to my work:**")
        lines.append(connection)
        lines.append("")

    weakness = reasoning.get('weakness', '').strip()
    if weakness and weakness.lower() not in ('none', ''):
        lines.append(f"**Weakness / caution:** {weakness}")
        lines.append("")

    action = reasoning.get('action', '').strip()
    if action:
        lines.append(f"**Action:** {action}")
        lines.append("")

    # Paper Workflow
    wf = reasoning.get('workflow', {})
    if wf and wf.get('research_question'):
        lines.append("**📋 Paper Workflow:**")
        lines.append("")
        rq = wf.get('research_question', '')
        if rq: lines.append(f"- **Question:** {rq}")
        hyp = wf.get('hypothesis', '')
        if hyp and hyp.lower() != 'not stated': lines.append(f"- **Hypothesis:** {hyp}")
        ed = wf.get('experimental_design', '')
        if ed: lines.append(f"- **Design:** {ed}")
        km = wf.get('key_method', '')
        if km: lines.append(f"- **Key Method:** {km}")
        kr = wf.get('key_result', '')
        if kr: lines.append(f"- **Key Result:** {kr}")
        wg = wf.get('workflow_gap', '')
        if wg and wg.lower() != 'none': lines.append(f"- **Gap:** {wg}")
        lines.append("")

    return lines


def _render_condensed(entry: tuple) -> list[str]:
    paper = entry[0]
    signals = entry[2] if len(entry) > 2 else []
    reasoning = entry[3] if len(entry) > 3 else {}

    lines = []
    lines.append(f"### {paper.title}")
    lines.append("")
    lines.append(f"- **Journal:** {paper.journal}")
    lines.append(f"- **DOI:** [{paper.doi}](https://doi.org/{paper.doi})")
    if signals:
        lines.append(f"- **Matched Signals:** {', '.join(signals)}")
    lines.append(f"- **Scores:** R:{reasoning.get('relevance', '?')}/10 "
                 f"N:{reasoning.get('novelty', '?')}/10 "
                 f"B:{reasoning.get('bridge', '?')}/10 "
                 f"T:{reasoning.get('trajectory_influence', '?')}/10 "
                 f"→ **{reasoning.get('final_score', '?')}/10**")
    lines.append("")

    why = reasoning.get('why_matters', '').strip()
    if why and why.lower() != 'none':
        first_line = why.split('\n')[0].lstrip('- ')
        lines.append(f"**Why it matters:** {first_line}")
        lines.append("")

    connection = reasoning.get('potential_connection', '').strip()
    if connection and connection.lower() != 'none':
        first_line = connection.split('\n')[0].lstrip('- ')
        lines.append(f"**Connection:** {first_line}")
        lines.append("")

    action = reasoning.get('action', '').strip()
    if action:
        lines.append(f"**Action:** {action}")
        lines.append("")

    return lines


def _render_watchlist(entry: tuple) -> list[str]:
    paper = entry[0]
    signals = entry[2] if len(entry) > 2 else []
    reasoning = entry[3] if len(entry) > 3 else {}

    lines = []
    lines.append(f"- **{paper.title}** — *{paper.journal}*")
    lines.append(f"  DOI: [{paper.doi}](https://doi.org/{paper.doi})")
    if signals:
        lines.append(f"  Matched: {', '.join(signals)}")
    t_score = reasoning.get('trajectory_influence', reasoning.get('trajectory', '?'))
    lines.append(f"  Scores: R:{reasoning.get('relevance', '?')}/10 "
                 f"N:{reasoning.get('novelty', '?')}/10 "
                 f"B:{reasoning.get('bridge', '?')}/10 "
                 f"T:{t_score}/10 → {reasoning.get('final_score', '?')}/10")

    why = reasoning.get('why_matters', '').strip()
    if why and why.lower() != 'none':
        first_line = why.split('\n')[0].lstrip('- ')
        lines.append(f"  Why: {first_line}")

    action = reasoning.get('action', '').strip()
    if action:
        lines.append(f"  Action: {action}")
    lines.append("")
    return lines


def _render_concept(concept: dict) -> list[str]:
    lines = []
    name = concept.get("name", "Unnamed Concept")
    lines.append(f"### Concept: {name}")
    lines.append("")
    source = concept.get("source", "")
    if source:
        lines.append(f"- **Source paper:** {source}")
    why = concept.get("why_matters", "")
    if why:
        lines.append(f"- **Why it matters:** {why}")
    connection = concept.get("connection", "")
    if connection:
        lines.append(f"- **Connection to my map:** {connection}")
    suggestion = concept.get("dictionary_update", "")
    if suggestion:
        lines.append(f"- **Suggested dictionary update:** {suggestion}")
    action = concept.get("action", "")
    if action:
        lines.append(f"- **Action:** {action}")
    lines.append("")
    return lines


def _render_concept_gap(gap: dict) -> list[str]:
    lines = []
    name = gap.get("concept", "Unnamed")
    lines.append(f"### Concept: {name}")
    lines.append("")
    current = gap.get("current_connection", "")
    if current:
        lines.append(f"- **Current Connection:** {current}")
    potential = gap.get("potential_connection", "")
    if potential:
        lines.append(f"- **Potential Connection:** {potential}")
    missing = gap.get("missing_link", "")
    if missing:
        lines.append(f"- **Missing Link:** {missing}")
    opportunity = gap.get("opportunity", "")
    if opportunity:
        lines.append(f"- **Opportunity:** {opportunity}")
    lines.append("")
    return lines
