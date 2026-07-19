#!/usr/bin/env python3
"""Create a privacy-conscious local PDF corpus map.

The scanner uses filesystem metadata only: path, filename, size, and mtime.
It does not open PDF contents and does not call network services.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


TOPIC_PATTERNS = {
    "sensing": [
        r"\bsensing\b",
        r"\bsensor[s]?\b",
        r"\bbiosensor[s]?\b",
        r"electrochemical",
        r"\boect\b",
        r"organic electrochemical transistor",
        r"\btransistor[s]?\b",
        r"nanopore",
        r"nanophotonic",
        r"plasmon",
        r"\bsers\b",
        r"aptamer",
        r"immunoassay",
        r"microfluidic",
        r"lab[- ]on[- ]a[- ]chip",
        r"optofluidic",
    ],
    "photochemistry": [
        r"photochem",
        r"photocatal",
        r"photodynamic",
        r"photorelease",
        r"photo[- ]?uncag",
        r"photocage",
        r"light[- ]?(trigger|responsive|activated|activation|mediated)",
        r"near[- ]?infrared",
        r"\bnir\b",
        r"bodipy",
        r"cyanine",
        r"rhodamine",
        r"photosensiti",
        r"singlet oxygen",
        r"\bros\b",
        r"phosphorescen",
        r"upconversion",
    ],
    "mitochondria": [
        r"mitochond",
        r"\bmtdna\b",
        r"mitophagy",
        r"oxidative phosphorylation",
        r"respiration",
        r"atp synthase",
    ],
    "EV_OEV": [
        r"extracellular vesicle",
        r"extracellular vesicles",
        r"\bexosome[s]?\b",
        r"\bsev[s]?\b",
        r"\boev\b",
        r"organoid[- ]derived extracellular vesicle",
    ],
    "organoids": [
        r"\borganoid[s]?\b",
        r"microphysiological",
        r"organ[- ]on[- ]a[- ]chip",
    ],
    "bioelectronics": [
        r"bioelectronic",
        r"\boect\b",
        r"organic mixed ionic",
        r"\bpedot\b",
        r"mixed ionic[- ]electronic",
        r"hydrogel",
        r"implant",
        r"microneedle",
    ],
    "calcium_GPCR": [
        r"\bcalcium\b",
        r"\bca2\+",
        r"\bgpcr\b",
        r"g[- ]?protein",
        r"calcium[- ]sensing receptor",
    ],
    "drug_delivery_nanomedicine": [
        r"drug delivery",
        r"prodrug",
        r"nanomedicine",
        r"nanoparticle",
        r"nanovesicle",
        r"doxorubicin",
        r"paclitaxel",
        r"ferroptosis",
        r"immunotherapy",
    ],
}

NON_RESEARCH_PATTERNS = [
    r"\bvisa\b",
    r"\binvoice\b",
    r"\breceipt\b",
    r"payment",
    r"\bcv\b",
    r"resume",
    r"admit",
    r"confirmation",
    r"\bclaim\b",
    r"passport",
    r"quotation",
    r"\bcontract\b",
    r"合同",
    r"知情同意书",
    r"社保",
    r"insurance",
    r"\bsds\b",
    r"safety data",
    r"hazardous",
    r"ethics files",
    r"training[- ]record",
    r"monitoring sheet",
    r"naplan",
    r"homework",
    r"vocabulary",
    r"violin",
    r"membership card",
    r"user guide",
    r"handbook",
]

LOW_SIGNAL_PATH_PARTS = {
    ".seagate",
    ".claude",
    ".hermes",
    ".codex",
    "node_modules",
    "__pycache__",
}

CURATED_HINTS = [
    "auto-daily paper updates",
    "scholarhound sources",
    "legacy-project-import",
    "zotero/storage",
    "oev readiness literature",
    "evs based sensing",
    "evs isolation and sensing",
    "oev review",
    "nrb new",
]


def compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


COMPILED_TOPICS = {
    topic: compile_patterns(patterns) for topic, patterns in TOPIC_PATTERNS.items()
}
COMPILED_NON_RESEARCH = compile_patterns(NON_RESEARCH_PATTERNS)


def normalize_path_for_match(path: Path) -> str:
    return str(path).lower().replace("_", " ").replace("-", " ")


def top_bucket(path: Path, home: Path) -> str:
    try:
        rel = path.relative_to(home)
    except ValueError:
        parts = path.parts
        return "/" + parts[1] if len(parts) > 1 else str(path)
    if not rel.parts:
        return str(home)
    return rel.parts[0]


def parent_alias(path: Path, home: Path) -> str:
    try:
        rel = path.parent.relative_to(home)
        parts = rel.parts
        if len(parts) >= 3 and parts[0] == "Library":
            return "/".join(parts[:3]) + "/..."
        return "/".join(parts[:4]) if parts else "."
    except ValueError:
        return str(path.parent)


def is_low_signal(path: Path) -> bool:
    lowered = [part.lower() for part in path.parts]
    return any(part in LOW_SIGNAL_PATH_PARTS for part in lowered)


def is_curated(path_text: str) -> bool:
    return any(hint in path_text for hint in CURATED_HINTS)


def classify(path: Path) -> tuple[list[str], bool, bool]:
    text = normalize_path_for_match(path)
    topics = []
    for topic, patterns in COMPILED_TOPICS.items():
        if any(pattern.search(text) for pattern in patterns):
            topics.append(topic)
    non_research = any(pattern.search(text) for pattern in COMPILED_NON_RESEARCH)
    curated = is_curated(text)
    return topics, non_research, curated


def iter_pdfs(root: Path, inaccessible: Counter, home: Path):
    def onerror(error: OSError) -> None:
        filename = getattr(error, "filename", None)
        area = parent_alias(Path(filename), home) if filename else str(root)
        inaccessible[area] += 1

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=onerror):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {".Trash", "Trash", "Backups.backupdb"}
        ]
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                yield Path(dirpath) / filename


def scan(roots: list[Path]) -> dict:
    home = Path.home().resolve()
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    topic_counts = Counter()
    topic_examples: dict[str, list[dict]] = defaultdict(list)
    bucket_counts = Counter()
    bucket_research_counts = Counter()
    curated_counts = Counter()
    cooccurrence = Counter()
    duplicate_keys = Counter()
    inaccessible = Counter()
    total = 0
    candidate_count = 0
    nonresearch_count = 0
    low_signal_count = 0
    total_bytes = 0
    newest_candidates = []

    for root in roots:
        try:
            root = root.resolve()
        except FileNotFoundError:
            inaccessible[str(root)] += 1
            continue
        for path in iter_pdfs(root, inaccessible, home):
            try:
                stat = path.stat()
            except (OSError, PermissionError):
                inaccessible[parent_alias(path, home)] += 1
                continue

            total += 1
            total_bytes += stat.st_size
            topics, non_research, curated = classify(path)
            low_signal = is_low_signal(path)
            bucket = top_bucket(path, home)
            bucket_counts[bucket] += 1
            digest_key = f"{path.name.lower()}::{stat.st_size}"
            duplicate_keys[digest_key] += 1

            if non_research:
                nonresearch_count += 1
            if low_signal:
                low_signal_count += 1

            candidate = bool(topics) and not non_research and not low_signal
            if curated:
                curated_counts["all_curated_pdfs"] += 1
            if candidate:
                candidate_count += 1
                bucket_research_counts[bucket] += 1
                newest_candidates.append(
                    {
                        "name": path.name,
                        "parent": parent_alias(path, home),
                        "mtime": dt.datetime.fromtimestamp(stat.st_mtime).date().isoformat(),
                        "topics": topics,
                        "curated": curated,
                    }
                )

            for topic in topics:
                topic_counts[topic] += 1
                if candidate and len(topic_examples[topic]) < 12:
                    topic_examples[topic].append(
                        {
                            "name": path.name,
                            "parent": parent_alias(path, home),
                            "curated": curated,
                        }
                    )
            for i, left in enumerate(sorted(topics)):
                for right in sorted(topics)[i + 1 :]:
                    cooccurrence[f"{left} + {right}"] += 1

    newest_candidates.sort(key=lambda item: item["mtime"], reverse=True)
    duplicates = sum(count - 1 for count in duplicate_keys.values() if count > 1)

    return {
        "generated_at": now,
        "roots": [str(root) for root in roots],
        "method": "metadata_only_path_filename_size_mtime_no_pdf_text_no_network",
        "totals": {
            "pdfs": total,
            "approx_gb": round(total_bytes / (1024**3), 2),
            "topic_matched_pdfs_without_admin_or_tool_noise": candidate_count,
            "admin_or_private_likely_nonresearch": nonresearch_count,
            "low_signal_software_or_vendor_pdfs": low_signal_count,
            "duplicate_name_size_extra_copies": duplicates,
        },
        "topic_counts_all_path_matches": dict(topic_counts.most_common()),
        "top_buckets_all_pdfs": dict(bucket_counts.most_common(20)),
        "top_buckets_research_candidates": dict(bucket_research_counts.most_common(20)),
        "curated_signal_counts": dict(curated_counts),
        "topic_cooccurrence_all_path_matches": dict(cooccurrence.most_common(25)),
        "topic_examples_research_candidates": topic_examples,
        "newest_research_candidates": newest_candidates[:40],
        "inaccessible_or_stat_errors_by_area": dict(inaccessible.most_common(30)),
    }


def write_markdown(summary: dict, output: Path) -> None:
    totals = summary["totals"]
    lines = [
        "# Local PDF Corpus Scan",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Method: `{summary['method']}`",
        f"- Roots: {', '.join(f'`{root}`' for root in summary['roots'])}",
        "",
        "## Totals",
        "",
    ]
    for key, value in totals.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Topic Matches", ""])
    for topic, count in summary["topic_counts_all_path_matches"].items():
        lines.append(f"- `{topic}`: {count}")
    lines.extend(["", "## Research Candidate Locations", ""])
    for bucket, count in summary["top_buckets_research_candidates"].items():
        lines.append(f"- `{bucket}`: {count}")
    lines.extend(["", "## Strong Topic Co-occurrences", ""])
    for pair, count in summary["topic_cooccurrence_all_path_matches"].items():
        lines.append(f"- `{pair}`: {count}")
    lines.extend(["", "## Topic Examples", ""])
    for topic, examples in summary["topic_examples_research_candidates"].items():
        lines.append(f"### {topic}")
        for example in examples[:8]:
            curated = " curated" if example["curated"] else ""
            lines.append(f"- {example['name']} ({example['parent']}{curated})")
        lines.append("")
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--md", required=True, type=Path)
    args = parser.parse_args()

    summary = scan(args.roots)
    args.json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(summary, args.md)
    print(
        json.dumps(
            {
                "pdfs": summary["totals"]["pdfs"],
                "research_candidates": summary["totals"][
                    "topic_matched_pdfs_without_admin_or_tool_noise"
                ],
                "topics": summary["topic_counts_all_path_matches"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
