"""Import curated local seed-corpus sources and review-audit guardrails."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from psil.store.models import Paper


DEFAULT_MANIFESTS = (
    Path("NRB") / "Evs based sensing" / "EV_sensing_and_limitations_manifest.csv",
    Path("NRB") / "OEV readiness literature no MDPI" / "manifest.csv",
)

DEFAULT_FACT_CHECK = Path("fact_check_report.md")
DEFAULT_TERMINOLOGY = Path("terminology_audit_report_latest.md")
PROJECT_LABEL = "ScholarHound seed corpus"


@dataclass
class LocalSource:
    doi: str
    title: str
    journal: str = ""
    pub_year: str = ""
    bucket: str = ""
    status: str = ""
    local_path: str = ""
    note: str = ""
    source_manifest: str = ""


def normalize_doi(value: str) -> str:
    doi = _clean(value)
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = doi.strip().rstrip(".")
    return doi.lower()


def import_local_manifests(db, vault_path: str | Path,
                           manifest_paths: list[str | Path] | None = None,
                           dry_run: bool = False) -> dict:
    """Import local curated DOI manifests into local_sources and papers."""
    vault = Path(vault_path).expanduser().resolve()
    paths = tuple(Path(p) for p in (manifest_paths or DEFAULT_MANIFESTS))
    sources: dict[str, LocalSource] = {}
    rows_seen = 0
    missing: list[str] = []

    for manifest in paths:
        path = _resolve_path(vault, manifest)
        if not path.exists():
            missing.append(str(manifest))
            continue

        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows_seen += 1
                source = _source_from_manifest_row(row, path, vault)
                if not source:
                    continue
                if source.doi in sources:
                    sources[source.doi] = _merge_sources(sources[source.doi], source)
                else:
                    sources[source.doi] = source

    papers_inserted = 0
    if not dry_run:
        for source in sources.values():
            if not db.doi_exists(source.doi):
                papers_inserted += 1
            db.upsert_local_source(
                doi=source.doi,
                title=source.title,
                journal=source.journal,
                pub_year=source.pub_year,
                bucket=source.bucket,
                status=source.status,
                local_path=source.local_path,
                note=source.note,
                source_manifest=source.source_manifest,
            )
            db.insert_paper(
                Paper(
                    doi=source.doi,
                    title=source.title,
                    journal=source.journal,
                    pub_date=source.pub_year,
                    abstract=source.note,
                ),
                signal_score=0,
                signal_tier="CURATED_LIBRARY",
                signal_action="Local curated seed source",
                llm_reasoning=json.dumps({
                    "origin": "local_manifest",
                    "bucket": source.bucket,
                    "status": source.status,
                    "local_path": source.local_path,
                    "source_manifest": source.source_manifest,
                    "note": source.note,
                }),
                concept_name=source.bucket,
                problem_class="Curated Local Source",
                strategic_value=source.bucket,
            )

        db.set_kernel_state("local_import_last_sources", str(len(sources)), "local_import")
        db.set_kernel_state(
            "local_import_last_manifests",
            json.dumps([str(p) for p in paths]),
            "local_import",
        )

    return {
        "rows_seen": rows_seen,
        "unique_sources": len(sources),
        "papers_inserted": papers_inserted,
        "missing_manifests": missing,
    }


def import_audits(db, corpus_root: str | Path, dry_run: bool = False) -> dict:
    """Import fact-check and terminology reports as kernel memory/constraints."""
    root = Path(corpus_root).expanduser().resolve()
    fact_path = root / DEFAULT_FACT_CHECK
    terminology_path = root / DEFAULT_TERMINOLOGY
    if not terminology_path.exists():
        terminology_path = root / "terminology_audit_report.md"

    fact_stats = import_fact_check_report(db, fact_path, dry_run=dry_run)
    terminology_stats = import_terminology_report(db, terminology_path, dry_run=dry_run)

    if not dry_run:
        db.set_kernel_state("local_import_fact_check_report", str(fact_path), "local_import")
        db.set_kernel_state("local_import_terminology_report", str(terminology_path), "local_import")

    return {
        "fact_check": fact_stats,
        "terminology": terminology_stats,
    }


def import_fact_check_report(db, report_path: str | Path,
                             dry_run: bool = False) -> dict:
    path = Path(report_path).expanduser()
    if not path.exists():
        return {"report_found": False, "claims": 0, "constraints": 0}

    text = path.read_text(encoding="utf-8")
    sections = re.split(r"\n(?=## A\.\d+\s+[—-]\s+)", text)
    existing_constraints = _existing_constraint_names(db) if not dry_run else set()
    claims = 0
    constraints = 0

    for section in sections:
        title_match = re.match(r"## A\.\d+\s+[—-]\s+(.+)", section.strip())
        if not title_match:
            continue

        claims += 1
        title = _collapse(title_match.group(1))
        status = _extract_bold_field(section, "Status")
        problem = _extract_bold_field(section, "Problem")
        safer = _extract_bold_field(section, "Safer wording")
        evidence = _extract_bold_field(section, "Evidence") or _extract_bold_field(
            section, "Evidence from source"
        )
        memory_status = _classify_claim_status(status)
        reason = _collapse(f"Status: {status}. {problem or safer or evidence}")[:1200]

        if not dry_run:
            db.upsert_memory(
                "claim_audit",
                title,
                status=memory_status,
                reason=reason,
                evidence_strength=status,
                affected_projects=PROJECT_LABEL,
            )

        if memory_status != "approved":
            name = f"fact-check: {title}"[:120]
            if not dry_run and name.lower() not in existing_constraints:
                db.insert_constraint(
                    name=name,
                    framework_name="seed corpus claim discipline",
                    statement=safer or f"Verify before using claim: {title}",
                    constraint_type="requires_verification",
                    supporting_evidence=status,
                    violating_examples=problem,
                    confidence=0.8,
                    prediction_power=0.3,
                    actionability=0.8,
                )
                existing_constraints.add(name.lower())
            constraints += 1

    return {"report_found": True, "claims": claims, "constraints": constraints}


def import_terminology_report(db, report_path: str | Path,
                              dry_run: bool = False) -> dict:
    path = Path(report_path).expanduser()
    if not path.exists():
        return {"report_found": False, "rules": 0, "constraints": 0}

    existing_constraints = _existing_constraint_names(db) if not dry_run else set()
    rules = list(_iter_terminology_rules(path.read_text(encoding="utf-8")))
    constraints = 0

    for term, replacement in rules:
        item_name = f"{term} -> {replacement}"
        if not dry_run:
            db.upsert_memory(
                "terminology_rule",
                item_name,
                status="approved",
                reason=f"Prefer: {replacement}",
                evidence_strength="Terminology audit",
                affected_projects=PROJECT_LABEL,
            )

        name = f"terminology: {term}"[:120]
        if not dry_run and name.lower() not in existing_constraints:
            db.insert_constraint(
                name=name,
                framework_name="seed corpus terminology discipline",
                statement=(
                    f'Avoid slogan-like use of "{term}"; prefer "{replacement}" '
                    "when writing or synthesizing this topic area."
                ),
                constraint_type="terminology_preference",
                supporting_evidence="terminology audit",
                violating_examples=term,
                confidence=0.9,
                prediction_power=0.2,
                actionability=0.9,
            )
            existing_constraints.add(name.lower())
        constraints += 1

    return {"report_found": True, "rules": len(rules), "constraints": constraints}


def run_local_import(db, vault_path: str | Path, corpus_root: str | Path | None = None,
                     include_manifests: bool = True, include_audits: bool = True,
                     dry_run: bool = False) -> dict:
    vault = Path(vault_path).expanduser().resolve()
    corpus = _resolve_corpus_root(vault, corpus_root)
    db.create_tables()

    stats = {
        "vault_path": str(vault),
        "corpus_root": str(corpus),
        "dry_run": dry_run,
    }

    if include_manifests:
        stats["manifests"] = import_local_manifests(db, vault, dry_run=dry_run)
    if include_audits:
        stats["audits"] = import_audits(db, corpus, dry_run=dry_run)

    if not dry_run:
        db.set_kernel_state("local_import_last_root", str(corpus), "local_import")
    return stats


def _source_from_manifest_row(row: dict, manifest_path: Path, vault: Path) -> LocalSource | None:
    doi = normalize_doi(row.get("doi", ""))
    title = _clean(row.get("title", ""))
    if not doi or not title:
        return None

    bucket = _clean(row.get("bucket") or row.get("cat"))
    message = _clean(row.get("message"))
    raw_path = _clean(row.get("local_path_or_source") or row.get("file"))
    if not raw_path and _looks_like_path(message):
        raw_path = message
    local_path = _resolve_local_reference(raw_path, manifest_path, vault)
    note = _clean(row.get("limitation_note") or row.get("note"))
    if message and message != raw_path and not _looks_like_path(message):
        note = _join_unique(note, message)
    source_manifest = _relative_to(manifest_path, vault)

    return LocalSource(
        doi=doi,
        title=title,
        journal=_clean(row.get("journal", "")),
        pub_year=_clean(row.get("year", "")),
        bucket=bucket,
        status=_clean(row.get("local_status") or row.get("status")),
        local_path=local_path,
        note=note,
        source_manifest=source_manifest,
    )


def _merge_sources(left: LocalSource, right: LocalSource) -> LocalSource:
    return LocalSource(
        doi=left.doi,
        title=left.title or right.title,
        journal=left.journal or right.journal,
        pub_year=left.pub_year or right.pub_year,
        bucket=_join_unique(left.bucket, right.bucket),
        status=_join_unique(left.status, right.status),
        local_path=_join_unique(left.local_path, right.local_path),
        note=_join_unique(left.note, right.note),
        source_manifest=_join_unique(left.source_manifest, right.source_manifest),
    )


def _iter_terminology_rules(text: str):
    in_summary = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## 九"):
            in_summary = True
            continue
        if in_summary and stripped.startswith("## "):
            break
        if not in_summary or not stripped.startswith("|") or "---" in stripped:
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        term = _strip_markdown(cells[0])
        replacement = _strip_markdown(cells[1])
        if not term or not replacement:
            continue
        if term in {"自创词", "原文（定位句子）"}:
            continue
        yield term, replacement


def _extract_bold_field(text: str, field: str) -> str:
    pattern = rf"\*\*{re.escape(field)}:\s*(.*?)\*\*"
    inline = re.search(pattern, text, re.S)
    if inline:
        return _collapse(inline.group(1))

    block_pattern = rf"\*\*{re.escape(field)}:\*\*\s*(.*?)(?=\n\n\*\*|\n---|\n## |$)"
    block = re.search(block_pattern, text, re.S)
    return _collapse(block.group(1)) if block else ""


def _classify_claim_status(status: str) -> str:
    value = status.lower()
    if "unsupported" in value:
        return "rejected"
    if "supported" in value and "partially" not in value and "caveat" not in value:
        return "approved"
    return "candidate"


def _existing_constraint_names(db) -> set[str]:
    return {c.get("name", "").lower() for c in db.get_constraints()}


def _resolve_corpus_root(vault: Path, corpus_root: str | Path | None) -> Path:
    if corpus_root is None:
        return vault / "NRB"
    path = Path(corpus_root).expanduser()
    if path.is_absolute():
        return path
    under_vault = vault / path
    if under_vault.exists():
        return under_vault.resolve()
    return path.resolve()


def _resolve_path(vault: Path, path: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else vault / path


def _resolve_local_reference(raw_path: str, manifest_path: Path, vault: Path) -> str:
    raw_path = _clean(raw_path)
    if not raw_path:
        return ""
    if raw_path.startswith(("http://", "https://")):
        return raw_path

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)

    candidates = (
        manifest_path.parent / path,
        vault / "NRB" / path,
        vault / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((manifest_path.parent / path).resolve()) if _looks_like_path(raw_path) else raw_path


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _join_unique(*values: str) -> str:
    seen = []
    for value in values:
        for part in _clean(value).split(" | "):
            part = part.strip()
            if part and part not in seen:
                seen.append(part)
    return " | ".join(seen)


def _strip_markdown(value: str) -> str:
    value = value.replace("**", "").replace("`", "")
    value = re.sub(r"<[^>]+>", "", value)
    return _collapse(value)


def _looks_like_path(value: str) -> bool:
    value = _clean(value)
    if not value:
        return False
    return (
        "/" in value
        or "\\" in value
        or value.lower().endswith((".pdf", ".html", ".md", ".docx", ".csv"))
    )


def _collapse(value: str) -> str:
    return re.sub(r"\s+", " ", _clean(value))


def _clean(value: str | None) -> str:
    return (value or "").strip()
