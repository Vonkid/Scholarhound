import importlib.util
from pathlib import Path


def load_registry_tool():
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "build_source_registry.py"
    spec = importlib.util.spec_from_file_location("build_source_registry", tool_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_migration_tool():
    tool_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "migrate_project_pdfs_to_source_store.py"
    )
    spec = importlib.util.spec_from_file_location("migrate_project_pdfs", tool_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_registry_separates_project_legacy_and_external_sources(tmp_path):
    tool = load_registry_tool()
    project = tmp_path / "project"
    external = tmp_path / "Documents"
    (project / "wiki").mkdir(parents=True)
    external.mkdir()

    project_pdf = project / "OECT sensor paper.pdf"
    project_pdf.write_bytes(b"%PDF-1.4 oect sensor")
    project_private = project / "Payment Received.pdf"
    project_private.write_bytes(b"%PDF-1.4 receipt")
    external_candidate = external / "mitochondria mtDNA sensing.pdf"
    external_candidate.write_bytes(b"%PDF-1.4 mitochondria")
    source_store_unknown = (
        external
        / "ScholarHound Sources"
        / "legacy-project-import"
        / "uncategorized"
        / "legacy.pdf"
    )
    source_store_unknown.parent.mkdir(parents=True)
    source_store_unknown.write_bytes(b"%PDF-1.4 legacy source")
    external_private = external / "invoice payment.pdf"
    external_private.write_bytes(b"%PDF-1.4 invoice")
    (project / "wiki" / "oect.md").write_text(
        '---\nsource_file: "OECT sensor paper.pdf"\n---\n',
        encoding="utf-8",
    )

    summary = tool.build_registry(
        project_root=project,
        external_roots=[external],
        source_store=external / "ScholarHound Sources",
        output_jsonl=project / "kernel" / "source_registry.jsonl",
        summary_json=project / "kernel" / "source_registry_summary.json",
        audit_md=project / "daily" / "source_registry_audit.md",
        max_hash_bytes=1024 * 1024,
    )

    rows = [
        tool.json.loads(line)
        for line in (project / "kernel" / "source_registry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert summary["counts"]["project_pdfs_scanned"] == 2
    assert summary["counts"]["external_pdfs_scanned_metadata_only"] == 3
    assert summary["counts"]["external_records_included"] == 2
    assert summary["external_excluded_counts"]["excluded_private_or_admin"] == 1
    assert any(row["identity"]["basename"] == project_pdf.name for row in rows)
    assert any(row["identity"]["basename"] == external_candidate.name for row in rows)
    assert any(row["identity"]["basename"] == source_store_unknown.name for row in rows)
    assert not any(row["identity"]["basename"] == external_private.name for row in rows)
    assert any(row["references"]["wiki_refs"] for row in rows)
    assert all(row["source_id"].startswith(("sha256:", "stat:")) for row in rows)


def test_migration_tool_moves_only_ready_externalization_records(tmp_path):
    registry_tool = load_registry_tool()
    migration_tool = load_migration_tool()
    project = tmp_path / "project"
    external = tmp_path / "Documents"
    project.mkdir()
    external.mkdir()

    source = project / "OECT sensor paper.pdf"
    source.write_bytes(b"%PDF-1.4 oect sensor")
    private_source = project / "Payment Received.pdf"
    private_source.write_bytes(b"%PDF-1.4 payment")
    registry_path = project / "kernel" / "source_registry.jsonl"

    registry_tool.build_registry(
        project_root=project,
        external_roots=[external],
        source_store=external / "ScholarHound Sources",
        output_jsonl=registry_path,
        summary_json=project / "kernel" / "source_registry_summary.json",
        audit_md=project / "daily" / "source_registry_audit.md",
        max_hash_bytes=1024 * 1024,
    )

    dry_plan = migration_tool.build_plan(
        registry_path,
        action="externalize_legacy_project_pdf",
    )
    assert len(dry_plan) == 1
    assert dry_plan[0]["status"] == "ready"
    assert source.exists()

    applied = migration_tool.apply_plan(dry_plan)

    assert applied[0]["apply_status"] == "moved"
    assert not source.exists()
    assert Path(applied[0]["destination"]).exists()
    assert private_source.exists()
