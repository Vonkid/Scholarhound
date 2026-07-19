"""
Benchmark analysis — calibrated against OUR research radar.

Purpose: Use 15 years of Nature Reviews to validate and calibrate the
ScholarHound signal dictionary and Trajectory Influence scoring.

Key questions answered:
1. Which of OUR concepts are rising, stable, or declining in the review literature?
2. What adjacent concepts co-occur with our signals but are missing from our dictionary?
3. When did each of our concepts emerge and peak?
"""

import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from psil.rank.concepts import CONCEPTS, TIER1_CORE_SIGNALS, TIER2_ADJACENT_SIGNALS, TIER3_WEAK_SIGNALS

DB_PATH = Path.home() / ".psil" / "benchmark.db"


def get_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    return db


def _normalize(text):
    return text.lower().replace("-", " ").replace("<i>", "").replace("</i>", "")


def get_concept_timeline(top_n=30):
    """
    Timeline of OUR signal keywords in the review corpus.
    Shows emergence, peak, and decline of each concept in our radar.
    """
    db = get_db()
    rows = db.execute(
        "SELECT pub_year, title, abstract FROM reviews WHERE pub_year BETWEEN 2011 AND 2026"
    ).fetchall()
    db.close()

    # Count OUR concepts per year
    year_concepts = defaultdict(Counter)
    for row in rows:
        year = row["pub_year"]
        text = _normalize((row["title"] or "") + " " + (row["abstract"] or ""))

        for phrase in CONCEPTS:
            if re.search(re.escape(phrase), text, re.IGNORECASE):
                year_concepts[year][phrase] += 1

    # Rank concepts by total appearances
    total = Counter()
    for yc in year_concepts.values():
        total.update(yc)

    top_concepts = [c for c, _ in total.most_common(top_n)]

    # Build matrix + classify trajectory
    years = sorted(year_concepts.keys())
    matrix = []
    for year in years:
        row = {"year": year}
        for c in top_concepts:
            row[c] = year_concepts[year].get(c, 0)
        matrix.append(row)

    # Classify each concept: rising / stable / declining
    trajectories = {}
    for c in top_concepts:
        recent = [year_concepts.get(y, Counter()).get(c, 0) for y in range(2021, 2027)]
        earlier = [year_concepts.get(y, Counter()).get(c, 0) for y in range(2015, 2021)]
        avg_recent = sum(recent) / max(len(recent), 1)
        avg_earlier = sum(earlier) / max(len(earlier), 1)

        if avg_recent > avg_earlier * 1.3:
            trajectories[c] = "rising"
        elif avg_recent < avg_earlier * 0.7:
            trajectories[c] = "declining"
        else:
            trajectories[c] = "stable"

        # New concept: <5 total in early years but rising
        if avg_earlier < 1 and avg_recent >= 2:
            trajectories[c] = "emerging"

    return {
        "years": years,
        "concepts": top_concepts,
        "matrix": matrix,
        "trajectories": trajectories,
        "total_papers": len(rows),
        "dictionary_coverage": f"{len(total)}/{len(CONCEPTS)} signals appear in reviews",
    }


def get_blind_spots(top_n=40):
    """
    Find concepts that co-occur with OUR signals in reviews but are NOT
    in our dictionary. Strictly filtered to remove noise.
    """
    db = get_db()
    rows = db.execute(
        "SELECT title, abstract, pub_year FROM reviews WHERE pub_year >= 2020"
    ).fetchall()
    db.close()

    dict_words = set()
    for c in CONCEPTS:
        for w in _normalize(c).split():
            if len(w) >= 4:
                dict_words.add(w)

    # Aggressive generic filter — academic boilerplate
    generic = {
        # Boilerplate phrases
        "based on", "has been", "such as", "can be", "review of",
        "how the", "in this", "of the", "will be", "this review",
        "provide an", "overview of", "discuss the", "recent advances",
        "recent progress", "current state", "past decade", "last decade",
        "present review", "we discuss", "we review", "this article",
        "review highlights", "review summarizes", "review focuses",
        "review provides", "we summarize", "we highlight", "we describe",
        "insights into", "lessons from", "past present", "present future",
        "next generation", "opportunities challenges", "associated with",
        "patients with", "cells with", "diseases such", "disorders such",
        "pathogenesis and", "mechanisms underlying", "role the",
        "review the", "field has", "years have", "here review",
        "paper reviews", "article reviews", "chapter reviews",
        "this paper", "this perspective", "this commentary",
        "clinical translation", "therapeutic potential",
        "the development", "development new", "new approaches",
        "approaches for", "strategies for", "challenges and",
        "challenges opportunities", "future directions",
        "future perspectives", "emerging role", "emerging roles",
        "recent developments", "latest advances", "key challenges",
        "key players", "the emerging", "the current", "the recent",
        "the field", "the past", "the future",
        # Too broad/generic concepts
        "author correction", "publisher correction", "correction author",
        # Prepositions/connectors
        "that the", "which are", "been shown", "have been",
        "may provide", "could provide", "might provide",
    }

    # Only analyze reviews that match at least one of our signals
    co_occurring = Counter()
    for row in rows:
        title = row["title"] or ""
        # Skip corrections and editorials
        if any(w in title.lower() for w in ("correction", "errata", "erratum",
                "editorial", "announcements", "issue publication")):
            continue

        text = _normalize(title + " " + (row["abstract"] or ""))

        # Check if review matches any of our concepts
        matched = False
        for phrase in CONCEPTS:
            if re.search(re.escape(phrase), text, re.IGNORECASE):
                matched = True
                break

        if not matched:
            continue

        # Extract bigrams from this matched review
        words = re.findall(r"[a-zA-Z]{5,}", text)  # min 5 chars per word
        seen = set()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in seen or bigram in generic:
                continue
            seen.add(bigram)

            # Skip if contains dict words
            if set(bigram.split()) & dict_words:
                continue

            co_occurring[bigram] += 1

    return [
        {"phrase": phrase, "frequency": count}
        for phrase, count in co_occurring.most_common(top_n)
    ]


def get_stats():
    """Benchmark stats — corpus size and journal coverage."""
    db = get_db()
    total = db.execute("SELECT COUNT(*) as n FROM reviews").fetchone()["n"]
    years = db.execute(
        "SELECT pub_year, COUNT(*) as n FROM reviews GROUP BY pub_year ORDER BY pub_year"
    ).fetchall()
    journals = db.execute(
        "SELECT journal, COUNT(*) as n FROM reviews GROUP BY journal ORDER BY n DESC"
    ).fetchall()
    db.close()
    return {
        "total": total,
        "years": [{"year": r["pub_year"], "count": r["n"]} for r in years],
        "journals": [{"name": r["journal"], "count": r["n"]} for r in journals],
    }
