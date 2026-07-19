"""
Notification module — sends daily digest summary to WeChat via Server酱 (ServerChan).
Each notable paper gets: TOC image (if available) + one-line takeaway + action hint.
"""

import requests

SERVERCHAN_URL = "https://sctapi.ftqq.com/{send_key}.send"

TIER_ICON = {
    "HIGH_PRIORITY": "🔴",
    "IMPORTANT": "🟠",
    "POTENTIAL": "🟡",
    "WATCHLIST": "🔵",
}


def send_scan_summary(send_key: str, date_str: str,
                      papers: list[dict],
                      concepts: list[dict],
                      paper_count: dict,
                      emerging: str = "",
                      confidence: str = "") -> bool:
    """
    Send a rich daily digest to WeChat.

    Each paper dict: {title, journal, tier, takeaway, image_url, action}
    Each concept dict: {name, why_matters}
    """

    title = f"PSIL Daily — {date_str}"
    lines = []

    # ---- Top papers with images and takeaways ----
    top_papers = [p for p in papers if p.get("tier") in ("HIGH_PRIORITY", "IMPORTANT", "POTENTIAL")]
    if not top_papers:
        top_papers = papers[:2]  # fallback: show first 2 WATCHLIST

    for i, p in enumerate(top_papers[:5]):
        tier = p.get("tier", "")
        icon = TIER_ICON.get(tier, "⚪")

        # TOC image
        img = p.get("image_url", "")
        if img:
            lines.append(f"![fig]({img})")

        # Title + journal
        journal = p.get("journal", "")
        lines.append(f"{icon} **{p.get('title', '')}**")
        if journal:
            lines.append(f"*{journal}*")

        # One-line takeaway (first sentence of why_matters)
        takeaway = p.get("takeaway", "")
        if takeaway:
            lines.append(f"> {takeaway}")

        # Action
        action = p.get("action", "")
        if action:
            lines.append(f"→ {action}")

        lines.append("")

    # ---- Stats bar ----
    lines.append(
        f"📊 HIGH:{paper_count.get('high',0)} IMPORTANT:{paper_count.get('important',0)} "
        f"POTENTIAL:{paper_count.get('potential',0)} WATCH:{paper_count.get('watchlist',0)} "
        f"概念:{len(concepts)}"
    )

    # ---- Concepts ----
    if concepts:
        lines.append("")
        lines.append("💡 **新概念**")
        for c in concepts[:3]:
            name = c.get("name", "")
            why = c.get("why_matters", "")
            if why:
                lines.append(f"• **{name}**: {why}")
            else:
                lines.append(f"• **{name}**")

    # ---- Emerging direction ----
    if emerging:
        lines.append("")
        lines.append(f"🔭 **新兴方向:** {emerging} ({confidence})" if confidence else f"🔭 **新兴方向:** {emerging}")

    # ---- Footer ----
    lines.append("")
    lines.append(f"[查看完整 digest](obsidian://open?vault=Auto-daily%20paper%20updates&file=daily/{date_str}-signals.md)")

    desk = "\n".join(lines)

    try:
        resp = requests.post(
            SERVERCHAN_URL.format(send_key=send_key),
            data={"title": title, "desp": desk},
            timeout=10,
        )
        data = resp.json()
        if data.get("code") == 0:
            return True
        else:
            print(f"  Server酱 error: {data.get('message', 'unknown')}")
            return False
    except Exception as e:
        print(f"  Notification failed: {e}")
        return False


def send_drift_report(send_key: str, momentum_data: list[dict], date_str: str = "") -> bool:
    """Send a weekly concept drift report."""
    title = f"PSIL Drift Report {date_str}"

    lines = []
    gaining = [c for c in momentum_data if c.get("appearances", 0) >= 3]
    if gaining:
        lines.append("📈 上升概念:")
        for c in gaining[:5]:
            lines.append(f"  • {c['name']} ({c['appearances']}次)")

    emerging = [c for c in momentum_data if c.get("appearances", 0) < 3]
    if emerging:
        lines.append(f"\n🌱 新兴概念 ({len(emerging)}个):")
        for c in emerging[:8]:
            lines.append(f"  • {c['name']}")

    if not lines:
        lines.append("本周暂无新概念动量数据。")

    desk = "\n".join(lines)

    try:
        resp = requests.post(
            SERVERCHAN_URL.format(send_key=send_key),
            data={"title": title, "desp": desk},
            timeout=10,
        )
        return resp.json().get("code") == 0
    except Exception as e:
        print(f"  Drift notification failed: {e}")
        return False
