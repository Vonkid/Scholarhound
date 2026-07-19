import os
from datetime import date

from psil.store.db import Database
from psil.store.models import Paper
from psil.rank.scorer import prefilter_papers
from psil.rank.concepts import get_matched_concepts
from psil.digest.render import render_digest
from psil.digest.vault import write_digest


def test_full_pipeline_dry_run(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()

    papers = [
        Paper(doi="10.1038/high1", title="OECT sensing with dynamic biointerfaces",
              abstract="We demonstrate a new mechano-electrochemical sensing paradigm.",
              journal="Nature Materials", pub_date="2026-05-27"),
        Paper(doi="10.1126/mid1", title="EV sensing with hydrogel electronics",
              abstract="Improved hydrogel-electronic coupling for wearable biosensing.",
              journal="Science Advances", pub_date="2026-05-27"),
        Paper(doi="10.1021/geology1", title="A study of regional geology formations",
              abstract="Sedimentary rock layer analysis.",
              journal="Geology", pub_date="2026-05-27"),
    ]

    new_papers = [p for p in papers if not db.doi_exists(p.doi)]
    assert len(new_papers) == 3

    passed, ignored = prefilter_papers(new_papers, threshold=0)
    assert len(passed) >= 1
    assert len(ignored) >= 1

    ignored_dois = {p.doi for p in ignored}
    assert "10.1021/geology1" in ignored_dois

    passed_dois = {p[0].doi for p in passed}
    assert "10.1038/high1" in passed_dois
    assert "10.1126/mid1" in passed_dois

    for paper, score in passed:
        db.insert_paper(paper, signal_score=score,
                        signal_tier="HIGH_PRIORITY" if score >= 5 else "IMPORTANT",
                        signal_trajectory=5.0)

    stored = db.get_all_papers()
    assert len(stored) == len(passed)

    high = []
    important = []
    for p, s in passed:
        signals = [name for name, _ in get_matched_concepts(f"{p.title} {p.abstract}")]
        entry = (p, s, signals, {
            "relevance": 8 if s >= 5 else 6,
            "novelty": 7, "bridge": 6, "trajectory_influence": 5,
            "final_score": 6.6 if s >= 5 else 6.0,
            "why_matters": "- Test.", "potential_connection": "- Test.",
            "weakness": "None.", "action": "Review this week",
            "signal_tier": "HIGH_PRIORITY" if s >= 5 else "IMPORTANT",
        })
        if s >= 5:
            high.append(entry)
        else:
            important.append(entry)

    content = render_digest(date.today(), high, important, [], [], [],
                            [], ignored, [], [], [], [], [], {})
    assert "HIGH PRIORITY" in content
    assert "IGNORE" in content
    assert "CONCEPT GAP MAP" in content

    vault = str(tmp_path / "vault")
    filepath = write_digest(vault, date.today(), content)
    assert os.path.isfile(filepath)

    db.insert_log(fetched=3, new=3, high=len(high), maybe=len(important), ignored=len(ignored))
    logs = db.get_recent_logs()
    assert len(logs) == 1
    assert logs[0]["papers_fetched"] == 3
