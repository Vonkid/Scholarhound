#!/usr/bin/env python3
"""Build a local source registry for ScholarHound PDF locators.

This script is intentionally metadata-first. It walks configured filesystem
roots, records physical paths and safe filesystem metadata, and only hashes
files that are eligible for source identity. It does not extract PDF text,
does not summarize papers, and does not call the network.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_local_pdf_corpus import classify, is_low_signal


REGISTRY_SCHEMA_VERSION = 1
DEFAULT_SOURCE_STORE = Path.home() / "Documents" / "ScholarHound Sources"
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    ".Trash",
    "Trash",
    "Backups.backupdb",
}
URL_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def iter_pdf_paths(root: Path, *, skip_roots: Iterable[Path] = ()) -> Iterable[Path]:
    resolved_skip_roots = [skip.resolve() for skip in skip_roots if skip.exists()]
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        if any(is_relative_to(current, skip) for skip in resolved_skip_roots):
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                yield current / filename


def hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_stat_id(path: Path, stat: os.stat_result) -> str:
    payload = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return "stat:" + hashlib.sha256(encoded).hexdigest()[:24]


def name_size_key(path: Path, stat: os.stat_result) -> str:
    return f"{path.name.casefold()}::{stat.st_size}"


def classify_privacy(topics: list[str], *, non_research: bool, low_signal: bool) -> str:
    if non_research:
        return "excluded_private_or_admin"
    if low_signal:
        return "excluded_low_signal"
    if topics:
        return "research_candidate"
    return "unknown"


def proposed_external_path(
    path: Path,
    *,
    source_store: Path,
    topics: list[str],
    digest: str | None,
) -> str:
    topic = topics[0] if topics else "uncategorized"
    safe_topic = re.sub(r"[^A-Za-z0-9_.-]+", "-", topic).strip("-") or "uncategorized"
    short = (digest or hashlib.sha256(str(path).encode("utf-8")).hexdigest())[:12]
    path_short = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8]
    stem = re.sub(r"[/:]+", "-", path.stem).strip() or "source"
    stem = stem[:120].rstrip()
    filename = f"{stem}__{short}__p{path_short}{path.suffix.lower()}"
    return str(source_store / "legacy-project-import" / safe_topic / filename)


def parse_wiki_source_refs(wiki_dir: Path) -> dict[str, list[dict[str, str]]]:
    refs: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not wiki_dir.exists():
        return refs

    pattern = re.compile(r'^source_file:\s*["\']?(.*?)["\']?\s*$')
    for page in sorted(wiki_dir.glob("*.md")):
        try:
            text = page.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line in text.splitlines()[:30]:
            match = pattern.match(line.strip())
            if not match:
                continue
            ref = match.group(1).strip()
            if not ref or URL_RE.match(ref):
                continue
            key = Path(ref).name.casefold()
            refs[key].append({"wiki_page": str(page), "source_file": ref})
    return refs


def make_record(
    path: Path,
    *,
    project_root: Path,
    scope: str,
    source_store: Path,
    generated_at: str,
    wiki_refs_by_basename: dict[str, list[dict[str, str]]],
    duplicate_name_size_counts: Counter[str],
    duplicate_sha_counts: Counter[str],
    max_hash_bytes: int,
) -> tuple[dict[str, Any], str | None]:
    stat = path.stat()
    topics, non_research, curated = classify(path)
    low_signal = is_low_signal(path)
    privacy_class = classify_privacy(
        topics,
        non_research=non_research,
        low_signal=low_signal,
    )

    hash_status = "not_computed"
    sha256: str | None = None
    should_hash = scope == "project" or privacy_class == "research_candidate" or curated
    if should_hash and stat.st_size <= max_hash_bytes:
        sha256 = hash_file(path)
        hash_status = "computed"
    elif should_hash:
        hash_status = "skipped_too_large"
    elif privacy_class.startswith("excluded_"):
        hash_status = "skipped_privacy_excluded"

    source_id = f"sha256:{sha256}" if sha256 else stable_stat_id(path, stat)
    rel_project_path = ""
    if is_relative_to(path, project_root):
        rel_project_path = str(path.resolve().relative_to(project_root.resolve()))

    wiki_refs = wiki_refs_by_basename.get(path.name.casefold(), [])
    name_key = name_size_key(path, stat)
    sha_count = duplicate_sha_counts.get(sha256 or "", 0) if sha256 else 0

    if scope == "project":
        if privacy_class.startswith("excluded_"):
            recommended_action = "review_then_remove_from_project"
            storage_role = "legacy_nonresearch_pdf_in_project"
        else:
            recommended_action = "externalize_legacy_project_pdf"
            storage_role = "legacy_embedded_source_pdf"
    elif privacy_class == "research_candidate" or curated:
        recommended_action = "keep_as_external_source_locator"
        storage_role = "external_source_pdf"
    else:
        recommended_action = "exclude_from_research_registry"
        storage_role = "external_nonresearch_pdf"

    record = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_id": source_id,
        "source_kind": "local_pdf",
        "locator": {
            "physical_path": str(path),
            "project_relative_path": rel_project_path,
            "storage_scope": scope,
            "storage_role": storage_role,
        },
        "identity": {
            "sha256": sha256 or "",
            "hash_status": hash_status,
            "size_bytes": stat.st_size,
            "mtime": dt.datetime.fromtimestamp(
                stat.st_mtime, tz=dt.timezone.utc
            ).isoformat(),
            "mtime_ns": stat.st_mtime_ns,
            "basename": path.name,
        },
        "classification": {
            "topic_hints": topics,
            "privacy_class": privacy_class,
            "curated_hint": bool(curated),
        },
        "references": {
            "wiki_refs": [item["wiki_page"] for item in wiki_refs],
            "source_file_refs": [item["source_file"] for item in wiki_refs],
        },
        "duplicates": {
            "name_size_key": name_key,
            "name_size_count": duplicate_name_size_counts.get(name_key, 0),
            "sha256_count": sha_count,
        },
        "recommended_action": recommended_action,
        "proposed_external_path": (
            proposed_external_path(
                path,
                source_store=source_store,
                topics=topics,
                digest=sha256,
            )
            if scope == "project"
            else ""
        ),
    }
    return record, sha256


def collect_paths(project_root: Path, external_roots: list[Path]) -> tuple[list[Path], list[Path]]:
    project_paths = sorted(iter_pdf_paths(project_root))
    external_paths: list[Path] = []
    for root in external_roots:
        if root.resolve() == project_root.resolve():
            continue
        external_paths.extend(iter_pdf_paths(root, skip_roots=[project_root]))
    return project_paths, sorted(set(external_paths))


def include_external_path(path: Path, project_name_size: set[str]) -> tuple[bool, str]:
    try:
        stat = path.stat()
    except OSError:
        return False, "stat_error"
    topics, non_research, curated = classify(path)
    low_signal = is_low_signal(path)
    privacy_class = classify_privacy(
        topics,
        non_research=non_research,
        low_signal=low_signal,
    )
    if privacy_class == "research_candidate" or curated:
        return True, privacy_class
    if name_size_key(path, stat) in project_name_size and not privacy_class.startswith("excluded_"):
        return True, "project_duplicate_candidate"
    return False, privacy_class


def build_registry(
    *,
    project_root: Path,
    external_roots: list[Path],
    source_store: Path,
    output_jsonl: Path,
    summary_json: Path,
    audit_md: Path,
    max_hash_bytes: int,
) -> dict[str, Any]:
    generated_at = now_iso()
    project_paths, external_paths = collect_paths(project_root, external_roots)

    name_size_counts: Counter[str] = Counter()
    for path in project_paths + external_paths:
        try:
            name_size_counts[name_size_key(path, path.stat())] += 1
        except OSError:
            continue
    project_name_size = {
        name_size_key(path, path.stat()) for path in project_paths if path.exists()
    }

    included_external: list[Path] = []
    excluded_external_counts: Counter[str] = Counter()
    for path in external_paths:
        include, reason = include_external_path(path, project_name_size)
        if include:
            included_external.append(path)
        else:
            excluded_external_counts[reason] += 1

    wiki_refs = parse_wiki_source_refs(project_root / "wiki")

    records: list[dict[str, Any]] = []
    sha_values: list[str] = []
    pending: list[tuple[Path, str]] = [(path, "project") for path in project_paths]
    pending.extend((path, "documents") for path in included_external)

    provisional: list[tuple[dict[str, Any], str | None]] = []
    for path, scope in pending:
        try:
            record, sha256 = make_record(
                path,
                project_root=project_root,
                scope=scope,
                source_store=source_store,
                generated_at=generated_at,
                wiki_refs_by_basename=wiki_refs,
                duplicate_name_size_counts=name_size_counts,
                duplicate_sha_counts=Counter(),
                max_hash_bytes=max_hash_bytes,
            )
        except OSError:
            continue
        provisional.append((record, sha256))
        if sha256:
            sha_values.append(sha256)

    sha_counts = Counter(sha_values)
    for record, sha256 in provisional:
        if sha256:
            record["duplicates"]["sha256_count"] = sha_counts[sha256]
        records.append(record)

    records.sort(
        key=lambda item: (
            item["locator"]["storage_scope"],
            item["classification"]["privacy_class"],
            item["identity"]["basename"].casefold(),
            item["locator"]["physical_path"],
        )
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    topic_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    privacy_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    duplicate_sha_groups = 0
    for record in records:
        action_counts[record["recommended_action"]] += 1
        privacy_counts[record["classification"]["privacy_class"]] += 1
        scope_counts[record["locator"]["storage_scope"]] += 1
        for topic in record["classification"]["topic_hints"]:
            topic_counts[topic] += 1
        if record["duplicates"]["sha256_count"] > 1 and record["identity"]["sha256"]:
            duplicate_sha_groups += 1

    summary = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "method": (
            "filesystem_metadata_plus_optional_sha256_for_project_and_research_candidates;"
            "no_pdf_text_extraction;no_network"
        ),
        "roots": {
            "project_root": str(project_root),
            "external_roots": [str(root) for root in external_roots],
            "source_store_target": str(source_store),
        },
        "counts": {
            "project_pdfs_scanned": len(project_paths),
            "external_pdfs_scanned_metadata_only": len(external_paths),
            "registry_records": len(records),
            "external_records_included": len(included_external),
            "external_records_excluded_from_registry": sum(excluded_external_counts.values()),
            "project_legacy_pdfs_to_externalize": action_counts[
                "externalize_legacy_project_pdf"
            ],
            "project_nonresearch_pdfs_to_review": action_counts[
                "review_then_remove_from_project"
            ],
            "wiki_source_file_refs_matched_by_basename": sum(
                1 for record in records if record["references"]["wiki_refs"]
            ),
            "records_with_sha256": sum(
                1 for record in records if record["identity"]["sha256"]
            ),
            "duplicate_sha_record_members": duplicate_sha_groups,
        },
        "action_counts": dict(action_counts.most_common()),
        "privacy_counts": dict(privacy_counts.most_common()),
        "scope_counts": dict(scope_counts.most_common()),
        "topic_counts": dict(topic_counts.most_common()),
        "external_excluded_counts": dict(excluded_external_counts.most_common()),
        "outputs": {
            "registry": str(output_jsonl),
            "summary": str(summary_json),
            "audit": str(audit_md),
        },
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_md.parent.mkdir(parents=True, exist_ok=True)
    audit_md.write_text(render_audit(summary), encoding="utf-8")
    return summary


def render_audit(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# Source Registry Audit",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Method: `{summary['method']}`",
        f"- Project root: `{summary['roots']['project_root']}`",
        f"- External roots: {', '.join(f'`{root}`' for root in summary['roots']['external_roots'])}",
        f"- Proposed source store: `{summary['roots']['source_store_target']}`",
        "",
        "## What Changed",
        "",
        "- Raw PDFs are now represented as local source locators instead of being treated as the project itself.",
        "- `kernel/source_registry.jsonl` is the machine-readable map from `source_id` to physical path.",
        "- Project-folder PDFs are marked as legacy embedded sources and queued for a separate, gated externalization step.",
        "- External Documents PDFs are included only when they look like research/source candidates; likely administrative or private PDFs are counted but not listed as registry rows.",
        "",
        "## Counts",
        "",
    ]
    for key, value in counts.items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Actions", ""])
    for key, value in summary["action_counts"].items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Topic Hints In Registry", ""])
    for key, value in summary["topic_counts"].items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Privacy Boundary", ""])
    for key, value in summary["privacy_counts"].items():
        lines.append(f"- `{key}`: {value}")
    if counts["project_legacy_pdfs_to_externalize"]:
        lines.extend(
            [
                "",
                "## Next Gated Migration",
                "",
                "1. Create the proposed source store directory outside the project folder.",
                "2. Move only `externalize_legacy_project_pdf` records after reviewing the registry.",
                "3. Keep `review_then_remove_from_project` records out of research ingest unless explicitly confirmed.",
                "4. Regenerate this registry after the move so physical paths become canonical.",
                "",
                "No PDF is moved or deleted by the registry build itself.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Remaining Work",
                "",
                "- No project research PDFs remain queued for externalization.",
                "- Review `review_then_remove_from_project` records manually; they are not research-source inputs.",
                "- Keep regenerating this registry after future source-store moves or new PDF downloads.",
                "",
                "No PDF is moved or deleted by the registry build itself.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--external-root",
        action="append",
        type=Path,
        default=[],
        help="External root to scan. Can be repeated.",
    )
    parser.add_argument("--source-store", type=Path, default=DEFAULT_SOURCE_STORE)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("kernel/source_registry.jsonl"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("kernel/source_registry_summary.json"),
    )
    parser.add_argument(
        "--audit-md",
        type=Path,
        default=Path("daily/source_registry_audit_2026-06-11.md"),
    )
    parser.add_argument(
        "--max-hash-mb",
        type=int,
        default=256,
        help="Maximum size to hash per eligible file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    external_roots = [root.resolve() for root in args.external_root] or [
        (Path.home() / "Documents").resolve()
    ]
    summary = build_registry(
        project_root=project_root,
        external_roots=external_roots,
        source_store=args.source_store.resolve(),
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        audit_md=args.audit_md,
        max_hash_bytes=args.max_hash_mb * 1024 * 1024,
    )
    print(
        json.dumps(
            {
                "registry_records": summary["counts"]["registry_records"],
                "project_pdfs_scanned": summary["counts"]["project_pdfs_scanned"],
                "external_pdfs_scanned_metadata_only": summary["counts"][
                    "external_pdfs_scanned_metadata_only"
                ],
                "project_legacy_pdfs_to_externalize": summary["counts"][
                    "project_legacy_pdfs_to_externalize"
                ],
                "project_nonresearch_pdfs_to_review": summary["counts"][
                    "project_nonresearch_pdfs_to_review"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
