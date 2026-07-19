"""
ScholarHound — Research Radar WebUI.
Launch: streamlit run psil/webui.py
"""

import os, re
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

from psil.config import load_config
from psil.store.db import Database

VAULT_PATH = load_config().get("vault_path", "")

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScholarHound",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── themed CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* hide streamlit junk */
    #MainMenu, footer, header[data-testid="stHeader"] { display: none; }

    /* brand colors */
    :root {
        --amber: #d97706;
        --gold: #b45309;
        --slate: #1e293b;
        --navy: #0f172a;
    }

    .sh-brand {
        font-size: 1.8rem; font-weight: 700;
        background: linear-gradient(135deg, #f59e0b, #d97706);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .sh-subtitle { color: #64748b; font-size: 0.9rem; margin-top: -0.5rem; }

    /* paper card */
    .paper-card {
        border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.25rem; margin-bottom: 0.75rem;
        transition: box-shadow 0.2s;
        background: #ffffff;
    }
    .paper-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.06); }

    .paper-tier-dot {
        display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; margin-right: 6px;
    }
    .paper-title { font-weight: 600; font-size: 1.05rem; color: #0f172a; }
    .paper-journal { color: #64748b; font-size: 0.85rem; }
    .paper-scores { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #475569; }
    .paper-takeaway { color: #334155; font-size: 0.9rem; line-height: 1.5; margin-top: 0.5rem; }
    .paper-action {
        display: inline-block; padding: 3px 10px;
        border-radius: 16px; font-size: 0.78rem; font-weight: 500;
        margin-top: 0.75rem;
    }
    .action-read     { background: #fef2f2; color: #dc2626; }
    .action-review   { background: #fff7ed; color: #ea580c; }
    .action-archive  { background: #fefce8; color: #ca8a04; }
    .action-watch    { background: #eff6ff; color: #2563eb; }

    /* tier badges */
    .tier-badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 12px; font-size: 0.75rem; font-weight: 600;
        letter-spacing: 0.03em;
    }
    .tier-HIGH { background: #fef2f2; color: #dc2626; }
    .tier-IMPORTANT { background: #fff7ed; color: #ea580c; }
    .tier-POTENTIAL { background: #fefce8; color: #ca8a04; }
    .tier-WATCHLIST { background: #eff6ff; color: #2563eb; }

    /* concept card */
    .concept-card {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1rem; margin-bottom: 0.5rem;
    }
    .concept-name { font-weight: 600; color: #0f172a; }
    .concept-why { color: #475569; font-size: 0.85rem; }

    /* stat number */
    .stat-num { font-size: 2rem; font-weight: 700; color: #0f172a; }
    .stat-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }

    /* sidebar nav */
    [data-testid="stSidebar"] { background: #fafafa; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; padding: 0.4rem 0; }
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "HIGH_PRIORITY": "#dc2626",
    "IMPORTANT": "#ea580c",
    "POTENTIAL": "#ca8a04",
    "WATCHLIST": "#2563eb",
    "LOW_PRIORITY": "#6b7280",
}
ACTION_CLASS = {
    "Read immediately": "action-read",
    "Review this week": "action-review",
    "Archive and revisit": "action-archive",
    "Watch for developments": "action-watch",
}

def get_db():
    return Database(str(Path.home() / ".psil" / "psil.db"))

def get_db_path():
    p = Path.home() / ".psil"; p.mkdir(exist_ok=True)
    return str(p / "psil.db")

def list_digests():
    dailydir = os.path.join(VAULT_PATH, "daily")
    if not os.path.isdir(dailydir):
        return []
    files = sorted([f for f in os.listdir(dailydir) if f.endswith("-signals.md")], reverse=True)
    return [f.replace("-signals.md", "") for f in files]

def load_digest(date_str):
    path = os.path.join(VAULT_PATH, "daily", f"{date_str}-signals.md")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sh-brand">🐕 ScholarHound</p>', unsafe_allow_html=True)
    st.markdown('<p class="sh-subtitle">Research Radar</p>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("", ["🏠 Dashboard", "📄 Digest", "🧠 Concept Radar", "🪪 Identity", "📜 History"])

try:
    db = get_db()
    db.create_tables()
except Exception:
    db = None

digests = list_digests()
today_str = date.today().strftime("%Y-%m-%d")

# ── Dashboard ────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown(f'<p class="sh-brand" style="font-size:2.2rem;">🐕 ScholarHound</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;font-size:1rem;margin-top:-0.5rem;">嗅探学术前沿 · 追踪研究信号 · 发现未来方向</p>', unsafe_allow_html=True)
    st.divider()

    # stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    if db:
        papers = db.get_all_papers()
        tiers = {}
        for p in papers:
            t = p.get("signal_tier", "LOW_PRIORITY")
            tiers[t] = tiers.get(t, 0) + 1
        concepts = db.get_concept_momentum(min_appearances=1)

        with c1: st.markdown(f'<p class="stat-num">{len(papers)}</p><p class="stat-label">Papers Indexed</p>', unsafe_allow_html=True)
        with c2: st.markdown(f'<p class="stat-num">{tiers.get("HIGH_PRIORITY",0)}</p><p class="stat-label">High Priority</p>', unsafe_allow_html=True)
        with c3: st.markdown(f'<p class="stat-num">{len(concepts)}</p><p class="stat-label">Concepts Tracked</p>', unsafe_allow_html=True)
        with c4: st.markdown(f'<p class="stat-num">{len(digests)}</p><p class="stat-label">Digests</p>', unsafe_allow_html=True)
        with c5:
            config = load_config()
            st.markdown(f'<p class="stat-num">{len(config.get("journals",[]))}</p><p class="stat-label">Journals</p>', unsafe_allow_html=True)
    else:
        for c in [c1,c2,c3,c4,c5]:
            with c: st.markdown('<p class="stat-num">—</p>', unsafe_allow_html=True)

    st.divider()

    # latest digest preview
    st.subheader("📄 Latest Digest")
    if digests:
        latest = digests[0]
        content = load_digest(latest)
        if content:
            # quick highlight: find HIGH PRIORITY and IMPORTANT papers
            sections = re.split(r"\n## ", content)
            shown = 0
            for sec in sections:
                if "HIGH PRIORITY" in sec[:30] or "IMPORTANT" in sec[:30]:
                    lines = sec.strip().split("\n")
                    header = lines[0].strip()
                    # find paper title
                    for line in lines[1:]:
                        if line.startswith("### "):
                            title = line.replace("### ", "")
                            journal = ""
                            tier = "HIGH_PRIORITY" if "HIGH" in header else "IMPORTANT"
                            color = TIER_COLORS.get(tier, "#6b7280")
                            st.markdown(
                                f'<div class="paper-card">'
                                f'<span class="tier-badge tier-{"HIGH" if "HIGH" in tier else "IMPORTANT"}">{tier.replace("_"," ").title()}</span>'
                                f'<p class="paper-title" style="margin-top:0.5rem;">{title}</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            shown += 1
                            break
                if shown >= 5:
                    break
            if shown == 0:
                st.info("No HIGH PRIORITY or IMPORTANT papers in latest digest")
            st.caption(f"Digest date: {latest} · [View full digest](/?page=Digest)")
    else:
        st.info("No digests yet. Run `psil scan` to generate your first digest.")

    # concept momentum quick view
    st.divider()
    st.subheader("🔭 Emerging Concepts")
    if db:
        gaining = [c for c in db.get_concept_momentum(min_appearances=1) if c["appearances"] >= 2]
        if gaining:
            cols = st.columns(min(len(gaining), 3))
            for i, c in enumerate(gaining[:6]):
                with cols[i % 3]:
                    apps = c["appearances"]
                    tw = c.get("trajectory_weight", "medium")
                    icons = {1: "🌱", 2: "📈", 3: "🔥"}.get(min(apps, 3), "🔭")
                    st.markdown(
                        f'<div class="concept-card">'
                        f'<p>{icons} <span class="concept-name">{c["name"]}</span></p>'
                        f'<p class="concept-why">{apps} appearances · {tw} weight</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("No concepts gaining momentum yet — keep scanning daily.")

# ── Digest Browser ───────────────────────────────────────────────────────────
elif page == "📄 Digest":
    st.markdown('<p class="sh-brand">📄 Digest</p>', unsafe_allow_html=True)

    available_dates = {}
    daily_dir = os.path.join(VAULT_PATH, "daily")
    if os.path.isdir(daily_dir):
        for f in os.listdir(daily_dir):
            if f.endswith("-signals.md"):
                try:
                    d = date.fromisoformat(f.replace("-signals.md", ""))
                    available_dates[d] = f
                except ValueError:
                    pass
    sorted_dates = sorted(available_dates.keys(), reverse=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        picked = st.date_input(
            "Select date",
            value=sorted_dates[0] if sorted_dates else date.today(),
            min_value=min(sorted_dates) if sorted_dates else None,
            max_value=max(sorted_dates) if sorted_dates else None,
            format="YYYY-MM-DD",
        )
    with c2:
        if sorted_dates:
            st.caption(f"{len(sorted_dates)} digests available · {min(sorted_dates)} → {max(sorted_dates)}")

    digest_str = picked.strftime("%Y-%m-%d") if isinstance(picked, date) else str(picked)
    content = load_digest(digest_str)

    if content:
        st.divider()
        sections = re.split(r"\n## ", content)
        for sec in sections:
            sec = sec.strip()
            if not sec: continue
            lines = sec.split("\n")
            header = lines[0].strip()
            body = "\n".join(lines[1:]).strip()

            if any(t in header for t in ["HIGH PRIORITY", "IMPORTANT", "POTENTIAL", "WATCHLIST", "LOW PRIORITY"]):
                tier_key = header.replace(" ", "_").upper()
                badge = ""
                if "HIGH" in tier_key: badge = "tier-HIGH"
                elif "IMPORTANT" in tier_key: badge = "tier-IMPORTANT"
                elif "POTENTIAL" in tier_key: badge = "tier-POTENTIAL"
                elif "WATCHLIST" in tier_key: badge = "tier-WATCHLIST"

                with st.expander(f"{header}", expanded=("HIGH" in header or "IMPORTANT" in header)):
                    if any(t in header for t in ["LOW PRIORITY", "IGNORE"]):
                        st.text(body[:2000])
                    else:
                        st.markdown(body[:5000])
            elif "CONCEPT FEED" in header:
                with st.expander("💡 CONCEPT FEED", expanded=True):
                    st.markdown(body[:5000])
            elif "DAILY SUMMARY" in header:
                with st.expander("📊 DAILY SUMMARY", expanded=True):
                    st.markdown(body[:3000])
            elif "CONCEPT GAP MAP" in header:
                with st.expander("🗺️ CONCEPT GAP MAP", expanded=False):
                    st.markdown(body[:5000])
            elif "IGNORE" in header:
                with st.expander("🗑️ IGNORE", expanded=False):
                    st.text(body[:1000])
    else:
        st.warning(f"No digest found for {digest_str}")

# ── Concept Radar ────────────────────────────────────────────────────────────
elif page == "🧠 Concept Radar":
    st.markdown('<p class="sh-brand">🧠 Concept Radar</p>', unsafe_allow_html=True)

    if not db:
        st.warning("Database not available")
    else:
        concepts = db.get_concept_momentum(min_appearances=1)
        c1, c2 = st.columns([3, 2])

        with c1:
            st.subheader("📈 Momentum")
            if concepts:
                for c in concepts:
                    apps = c["appearances"]
                    status = c["status"]
                    tw = c.get("trajectory_weight", "medium")
                    icons = {"established": "🔥", "gaining momentum": "📈", "emerging": "🌱"}
                    icon = icons.get(status, "🌱")
                    max_apps = max(5, apps)
                    pct = min(apps / max_apps, 1.0)

                    with st.container():
                        st.markdown(f"{icon} **{c['name']}** — *{status}* ({tw} weight)")
                        st.progress(pct, text=f"{apps} appearances")

                    if c.get("opportunity"):
                        with st.expander("🗺️ Gap Map"):
                            st.markdown(f"**Current:** {c.get('connection', '—')}")
                            st.markdown(f"**Missing Link:** {c.get('missing_link', '—')}")
                            st.markdown(f"**Opportunity:** {c.get('opportunity', '—')}")
            else:
                st.info("No concepts tracked yet.")

        with c2:
            st.subheader("🔭 Emerging Direction")
            gaining = [c for c in concepts if c["appearances"] >= 2]
            if gaining:
                top = sorted(gaining, key=lambda x: x["appearances"], reverse=True)[:5]
                for c in top:
                    tw = c.get("trajectory_weight", "medium")
                    apps = c["appearances"]
                    if tw == "high":
                        emoji = "🎯"
                    elif tw == "medium":
                        emoji = "📌"
                    else:
                        emoji = "👀"
                    st.markdown(f"{emoji} **{c['name']}**")
                    st.caption(f"{apps}× · {tw} trajectory weight")
            else:
                st.info("Need 2+ appearances to detect emerging direction.")

            st.divider()
            st.subheader("📊 Trajectory Distribution")
            weights = {"high": 0, "medium": 0, "low": 0}
            for c in concepts:
                tw = c.get("trajectory_weight", "medium")
                weights[tw] = weights.get(tw, 0) + 1
            if sum(weights.values()) > 0:
                for tw, count in weights.items():
                    if count > 0:
                        colors = {"high": "#dc2626", "medium": "#d97706", "low": "#64748b"}
                        st.markdown(f"{tw.title()} Weight — {count} concepts")
                        st.progress(count / max(weights.values(), 1), text="")

# ── Identity ─────────────────────────────────────────────────────────────────
elif page == "🪪 Identity":
    st.markdown('<p class="sh-brand">🪪 Research Identity</p>', unsafe_allow_html=True)

    from psil.rank.identity import load_identity
    ident = load_identity()

    tabs = st.tabs(["Current Core", "Emerging Directions", "Long-Term Vision"])
    with tabs[0]:
        for item in ident.current_core:
            st.markdown(f"- {item}")
    with tabs[1]:
        for item in ident.emerging_directions:
            st.markdown(f"- {item}")
    with tabs[2]:
        for item in ident.long_term_vision:
            st.markdown(f"- {item}")

    st.divider()
    st.subheader("Trajectory Influence Topics")
    search = st.text_input("Filter", "", placeholder="Search topics...")
    topics = ident.trajectory_influence_topics
    filtered = [t for t in topics if search.lower() in t.lower()] if search else topics
    st.caption(f"{len(filtered)}/{len(topics)} topics")
    if filtered:
        cols = st.columns(3)
        for i, t in enumerate(filtered[:60]):
            with cols[i % 3]:
                st.markdown(f"- {t}")

    st.caption(f"Last updated: {ident.last_updated} · {len(ident.concept_momentum)} tracked concepts")

# ── History ──────────────────────────────────────────────────────────────────
elif page == "📜 History":
    st.markdown('<p class="sh-brand">📜 Scan History</p>', unsafe_allow_html=True)

    if db:
        logs = db.get_recent_logs(limit=60)
        if logs:
            st.subheader("Recent Scans")
            cols = st.columns([2, 1, 1, 1, 1])
            for i, h in enumerate(["Date", "Fetched", "New", "High", "Maybe"]):
                cols[i].markdown(f"**{h}**")
            for log in logs[:20]:
                c = st.columns([2, 1, 1, 1, 1])
                c[0].text(log.get("run_at", "-")[:16])
                c[1].text(str(log.get("papers_fetched", "-")))
                c[2].text(str(log.get("papers_new", "-")))
                c[3].text(str(log.get("papers_high_signal", "-")))
                c[4].text(str(log.get("papers_maybe_signal", "-")))

    st.divider()
    st.subheader("Browse Past Digests")
    if digests:
        selected = st.selectbox("Select date", digests)
        if selected:
            content = load_digest(selected)
            if content:
                with st.expander(f"Digest — {selected}", expanded=True):
                    st.markdown(content[:10000])
                    if len(content) > 10000:
                        st.caption("... truncated. Open full file for complete digest.")
    else:
        st.info("No digests available.")

# ── footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("🐕 ScholarHound · sniffing the research frontier · [github](https://github.com)")
