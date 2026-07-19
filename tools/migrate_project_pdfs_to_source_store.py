#!/usr/bin/env python3
"""Move legacy project PDFs into the external ScholarHound source store.

Default mode is a dry run. The script reads `kernel/source_registry.jsonl`,
selects records whose recommended action is `externalize_legacy_project_pdf`,
and writes a migration manifest. Use `--apply` only after reviewing the plan.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path("kernel/source_registry.jsonl")
DEFAULT_MANIFEST = Path("kernel/source_migration_plan_2026-06-11.jsonl")
DEFAULT_AUDIT = Path("daily/source_migration_plan_2026-06-11.md")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_registry(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(registry_path: Path, *, action: str) -> list[dict[str, Any]]:
    records = read_registry(registry_path)
    plan = []
    for record in records:
        if record.get("recommended_action") != action:
            continue
        source = Path(record["locator"]["physical_path"])
        destination_text = record.get("proposed_external_path", "")
        destination = Path(destination_text) if destination_text else None
        issues = []
        if not source.exists():
            issues.append("missing_source")
        if destination is None:
            issues.append("missing_destination")
        elif destination.exists():
            issues.append("destination_exists")
        if source.exists() and destination and source.resolve() == destination.resolve():
            issues.append("source_equals_destination")

        plan.append(
            {
                "source_id": record["source_id"],
                "source": str(source),
                "destination": str(destination) if destination else "",
                "sha256": record["identity"].get("sha256", ""),
                "size_bytes": record["identity"].get("size_bytes", 0),
                "topic_hints": record["classification"].get("topic_hints", []),
                "project_relative_path": record["locator"].get("project_relative_path", ""),
                "status": "blocked" if issues else "ready",
                "issues": issues,
            }
        )
    return plan


def apply_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for item in plan:
        result = dict(item)
        if item["status"] != "ready":
            result["applied"] = False
            result["apply_status"] = "skipped_blocked"
            results.append(result)
            continue

        source = Path(item["source"])
        destination = Path(item["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_sha = item.get("sha256", "")
        if expected_sha:
            actual_sha = hash_file(source)
            if actual_sha != expected_sha:
                result["applied"] = False
                result["apply_status"] = "skipped_sha256_mismatch"
                result["actual_sha256"] = actual_sha
                results.append(result)
                continue

        shutil.move(str(source), str(destination))
        result["applied"] = True
        result["apply_status"] = "moved"
        results.append(result)
    return results


def write_manifest(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in items),
        encoding="utf-8",
    )


def render_audit(items: list[dict[str, Any]], *, apply: bool, generated_at: str) -> str:
    statuses = Counter(item.get("status", "unknown") for item in items)
    apply_statuses = Counter(item.get("apply_status", "not_applied") for item in items)
    topic_counts = Counter()
    bytes_total = 0
    for item in items:
        bytes_total += int(item.get("size_bytes") or 0)
        for topic in item.get("topic_hints", []):
            topic_counts[topic] += 1

    lines = [
        "# Source Migration Plan",
        "",
        f"- Generated: `{generated_at}`",
        f"- Mode: `{'apply' if apply else 'dry-run'}`",
        f"- Planned records: {len(items)}",
        f"- Approx size: {round(bytes_total / (1024 ** 3), 2)} GB",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in statuses.most_common():
        lines.append(f"- `{key}`: {value}")
    if apply:
        lines.extend(["", "## Apply Counts", ""])
        for key, value in apply_statuses.most_common():
            lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Topic Hints", ""])
    for key, value in topic_counts.most_common():
        lines.append(f"- `{key}`: {value}")

    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This plan includes only `externalize_legacy_project_pdf` records.",
            "- It does not move `review_then_remove_from_project` records.",
            "- After an applied migration, regenerate `kernel/source_registry.jsonl` so physical paths become canonical.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--action",
        default="externalize_legacy_project_pdf",
        choices=["externalize_legacy_project_pdf"],
    )
    parser.add_argument("--apply", action="store_true", help="Move files. Omit for dry run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated_at = now_iso()
    plan = build_plan(args.registry, action=args.action)
    items = apply_plan(plan) if args.apply else plan
    write_manifest(items, args.manifest)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        render_audit(items, apply=args.apply, generated_at=generated_at),
        encoding="utf-8",
    )
    statuses = Counter(item.get("status", "unknown") for item in items)
    apply_statuses = Counter(item.get("apply_status", "not_applied") for item in items)
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "records": len(items),
                "statuses": dict(statuses),
                "apply_statuses": dict(apply_statuses),
                "manifest": str(args.manifest),
                "audit": str(args.audit),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
