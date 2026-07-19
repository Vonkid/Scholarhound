import tempfile
import os
from psil.store.db import Database
from psil.store.models import Paper


def test_create_tables():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()

    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "papers" in tables
    assert "ingest_log" in tables
    assert "local_sources" in tables
    assert "kernel_objects" in tables
    assert "kernel_object_events" in tables
    assert "kernel_tasks" in tables
    assert "kernel_task_events" in tables
    conn.close()


def test_insert_paper():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()

    paper = Paper(
        doi="10.1038/test",
        title="Test Paper",
        abstract="An abstract",
        journal="Nature Materials",
        authors=["A. Author"],
        affiliations=["Univ"],
        pub_date="2026-05-27",
    )
    db.insert_paper(paper, signal_score=3, signal_tier="MAYBE")

    rows = db.get_all_papers()
    assert len(rows) == 1
    assert rows[0]["doi"] == "10.1038/test"
    assert rows[0]["signal_tier"] == "MAYBE"


def test_deduplication():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()

    paper = Paper(doi="10.1038/dup", title="Dup Paper")
    db.insert_paper(paper, signal_score=5, signal_tier="HIGH")
    db.insert_paper(paper, signal_score=5, signal_tier="HIGH")

    rows = db.get_all_papers()
    assert len(rows) == 1


def test_update_paper_toc_image_url_only_fills_missing_value():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()

    paper = Paper(doi="10.1038/toc", title="TOC Paper")
    db.insert_paper(paper)

    assert db.update_paper_toc_image_url("10.1038/toc", "https://example.com/toc.png") == 1
    rows = db.get_all_papers()
    assert rows[0]["toc_image_url"] == "https://example.com/toc.png"

    assert db.update_paper_toc_image_url("10.1038/toc", "https://example.com/other.png") == 0
    rows = db.get_all_papers()
    assert rows[0]["toc_image_url"] == "https://example.com/toc.png"


def test_doi_exists():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()

    paper = Paper(doi="10.1038/exists", title="Exists")
    db.insert_paper(paper, signal_score=0, signal_tier="MAYBE")

    assert db.doi_exists("10.1038/exists") is True
    assert db.doi_exists("10.1038/nonexistent") is False


def test_insert_log():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()

    db.insert_log(fetched=50, new=10, high=2, maybe=5, ignored=3)
    log = db.get_recent_logs(limit=1)
    assert len(log) == 1
    assert log[0]["papers_fetched"] == 50
    assert log[0]["papers_high_signal"] == 2


def test_upsert_local_source():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()

    db.upsert_local_source(
        doi="10.1038/local",
        title="Local Source",
        journal="Nature",
        pub_year="2026",
        bucket="EV sensing",
        status="local_pdf",
        local_path="/tmp/source.pdf",
        note="Curated local source.",
        source_manifest="manifest.csv",
    )
    db.upsert_local_source(
        doi="10.1038/local",
        title="Local Source Updated",
        journal="Nature",
        pub_year="2026",
        bucket="EV sensing",
        status="local_pdf",
        local_path="/tmp/source.pdf",
        note="Updated note.",
        source_manifest="manifest.csv",
    )

    rows = db.get_local_sources()
    assert len(rows) == 1
    assert rows[0]["title"] == "Local Source Updated"
    assert rows[0]["note"] == "Updated note."


def test_upsert_concept_uses_historical_seen_date():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()

    db.upsert_concept("OECT biosensing", source_doi="10.0/old", seen_date="2021-02-03")
    db.upsert_concept("OECT biosensing", source_doi="10.0/new", seen_date="2025-06-01")

    concept = db.get_concept("OECT biosensing")
    assert concept["first_seen"] == "2021-02-03"
    assert concept["last_seen"] == "2025-06-01"
    assert concept["appearances"] == 2


def test_kernel_objects_and_revision_events_are_persisted():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()

    obj = db.upsert_kernel_object(
        object_type="claim",
        title="Recognition must change channel state",
        statement="Binding is only useful if it changes an ionic-electronic state.",
        confidence=4,
        entrenchment=2,
        evidence={"doi": "10.1234/test"},
    )

    assert obj["object_type"] == "claim"
    assert obj["status"] == "candidate"
    assert obj["object_key"].startswith("claim-recognition-must-change-channel-state")

    revised = db.revise_kernel_object(obj["object_key"], status="approved", confidence=5.5, entrenchment=4)
    event = db.add_kernel_object_event(
        object_key=obj["object_key"],
        event_type="commit",
        previous_status="candidate",
        new_status="approved",
        previous_confidence=4,
        new_confidence=5.5,
        previous_entrenchment=2,
        new_entrenchment=4,
        reason="User committed this as a working claim.",
    )

    assert revised["status"] == "approved"
    assert revised["confidence"] == 5.5
    assert event["event_type"] == "commit"
    assert db.get_kernel_object_events(obj["object_key"])[0]["reason"].startswith("User committed")


def test_kernel_tasks_are_persisted_and_status_changes_are_logged():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    db = Database(db_path)
    db.create_tables()

    task = db.upsert_kernel_task(
        task_type="commit_or_reject_object",
        title="Commit or reject: recognition-state claim",
        description="Candidate claim needs a decision.",
        priority=9,
        action_hint="commit_or_reject",
        object_key="claim-123",
    )
    revised = db.revise_kernel_task(
        task["task_key"],
        status="done",
        reason="Committed during review.",
        actor="human",
    )

    assert task["status"] == "open"
    assert revised["status"] == "done"
    assert db.get_kernel_tasks(status="done")[0]["task_key"] == task["task_key"]
    assert db.get_kernel_task_events(task["task_key"])[0]["new_status"] == "done"
