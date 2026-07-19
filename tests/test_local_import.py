from psil.local_import import (
    import_fact_check_report,
    import_local_manifests,
    import_terminology_report,
)
from psil.store.db import Database


def test_import_local_manifests(tmp_path):
    manifest_dir = tmp_path / "NRB" / "Evs based sensing"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "EV_sensing_and_limitations_manifest.csv"
    manifest.write_text(
        "bucket,title,journal,year,doi,local_status,local_path_or_source,limitation_note\n"
        "EV sensing,Local EV Paper,Nature,2026,https://doi.org/10.1038/test,local_pdf,paper.pdf,Useful note\n",
        encoding="utf-8",
    )
    manifest2_dir = tmp_path / "NRB" / "OEV readiness literature no MDPI"
    pdf2 = manifest2_dir / "01_organoids_ready" / "paper.pdf"
    pdf2.parent.mkdir(parents=True)
    pdf2.write_text("pdf placeholder", encoding="utf-8")
    manifest2 = manifest2_dir / "manifest.csv"
    manifest2.write_text(
        "cat,title,journal,year,doi,status,message,file,note\n"
        "01_organoids_ready,Second Paper,Nature Reviews,2025,10.1038/second,copied_local_pdf,"
        "copied,OEV readiness literature no MDPI/01_organoids_ready/paper.pdf,Ready note\n",
        encoding="utf-8",
    )

    db = Database(str(tmp_path / "test.db"))
    db.create_tables()

    stats = import_local_manifests(db, tmp_path)
    assert stats["rows_seen"] == 2
    assert stats["unique_sources"] == 2
    assert stats["papers_inserted"] == 2

    sources = db.get_local_sources()
    by_doi = {source["doi"]: source for source in sources}
    assert by_doi["10.1038/test"]["bucket"] == "EV sensing"
    assert by_doi["10.1038/second"]["local_path"] == str(pdf2.resolve())
    assert db.doi_exists("10.1038/test")


def test_import_fact_check_report(tmp_path):
    report = tmp_path / "fact_check_report.md"
    report.write_text(
        "# Report\n\n"
        "## A.1 - Precise unsupported claim\n\n"
        "**Status: Unsupported**\n\n"
        "**Problem:** Too precise for the evidence.\n\n"
        "**Safer wording:** Use a wider range and cite the method.\n",
        encoding="utf-8",
    )

    db = Database(str(tmp_path / "test.db"))
    db.create_tables()

    stats = import_fact_check_report(db, report)
    assert stats["claims"] == 1
    assert stats["constraints"] == 1
    assert db.get_memory(item_type="claim_audit")[0]["status"] == "rejected"
    assert db.get_constraints()[0]["constraint_type"] == "requires_verification"


def test_import_terminology_report(tmp_path):
    report = tmp_path / "terminology_audit_report_latest.md"
    report.write_text(
        "# Report\n\n"
        "## \u4e5d\u3001\u6c47\u603b\uff1a\u5168\u5c40\u66ff\u6362\u5efa\u8bae\n\n"
        "| \u81ea\u521b\u8bcd | \u5168\u6587\u5efa\u8bae\u7edf\u4e00\u66ff\u6362\u4e3a |\n"
        "|---|---|\n"
        "| **interpretability / interpretable** | biological fidelity |\n",
        encoding="utf-8",
    )

    db = Database(str(tmp_path / "test.db"))
    db.create_tables()

    stats = import_terminology_report(db, report)
    assert stats["rules"] == 1
    assert stats["constraints"] == 1
    memory = db.get_memory(item_type="terminology_rule")
    assert "biological fidelity" in memory[0]["reason"]
