#!/usr/bin/env python3
"""Prepare a deterministic semantic-audit sample for E1.

The script does not judge records. It creates a compact JSONL for reviewer
inspection and a CSV template that agents/humans can fill.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from pathlib import Path


FIG_RAG_DIR = Path(os.environ.get("FIG_RAG_DIR", Path.home() / "fig-rag-dir"))
# The 50 reviewed records were drawn from an earlier metadata snapshot of the
# corpus; all of them are in the 57,100-record release. Running on the release
# data.jsonl draws a different sample with the same procedure.
DATA_PATH = FIG_RAG_DIR / "data.jsonl"
IMAGE_BASE = FIG_RAG_DIR
OUT_DIR = Path(__file__).resolve().parent
SAMPLE_JSONL = OUT_DIR / "e1_semantic_sample.jsonl"
TEMPLATE_CSV = OUT_DIR / "e1_semantic_annotations_template.csv"
REPORT_MD = OUT_DIR / "e1_semantic_sample_report.md"

SEED = 20260709
TOTAL_SAMPLE = 50
NO_CONTEXT_TARGET = 10


def squash(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def load_records() -> list[dict]:
    records = []
    with DATA_PATH.open() as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            row["_row_id"] = idx
            records.append(row)
    return records


def stratified_sample(records: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    with_context = [r for r in records if (r.get("figure_context") or "").strip()]
    no_context = [r for r in records if not (r.get("figure_context") or "").strip()]

    sample: list[dict] = []
    sample.extend(rng.sample(no_context, min(NO_CONTEXT_TARGET, len(no_context))))

    remaining = TOTAL_SAMPLE - len(sample)
    venues = sorted({r.get("venue", "UNKNOWN") for r in with_context})
    by_venue = {venue: [r for r in with_context if r.get("venue", "UNKNOWN") == venue] for venue in venues}

    per_venue = max(1, remaining // max(1, len(venues)))
    for venue in venues:
        take = min(per_venue, len(by_venue[venue]), remaining - len(sample))
        sample.extend(rng.sample(by_venue[venue], take))

    if len(sample) < TOTAL_SAMPLE:
        picked = {r["_row_id"] for r in sample}
        pool = [r for r in with_context if r["_row_id"] not in picked]
        sample.extend(rng.sample(pool, TOTAL_SAMPLE - len(sample)))

    rng.shuffle(sample)
    return sample


def compact_record(row: dict, sample_id: int) -> dict:
    rel_image = row.get("figure_path", "")
    abs_image = IMAGE_BASE / rel_image
    context = row.get("figure_context", "")
    platform_id = row.get("platform_id", "")
    expected_url = f"https://openreview.net/forum?id={platform_id}" if platform_id else ""

    return {
        "sample_id": f"E1-{sample_id:03d}",
        "row_id": row["_row_id"],
        "venue": row.get("venue"),
        "year": row.get("year"),
        "platform_id": platform_id,
        "url": row.get("url"),
        "url_matches_platform_id": row.get("url") == expected_url,
        "title": squash(row.get("title", ""), 260),
        "abstract": squash(row.get("abstract", ""), 600),
        "figure_number": row.get("figure_number"),
        "figure_caption": squash(row.get("figure_caption", ""), 500),
        "figure_context": squash(context, 900),
        "has_figure_context": bool(context.strip()),
        "figure_type": row.get("figure_type"),
        "confidence": row.get("confidence"),
        "figure_path": rel_image,
        "image_abs_path": str(abs_image),
        "image_exists": abs_image.exists(),
    }


def write_outputs(sample: list[dict]) -> None:
    compact = [compact_record(row, i + 1) for i, row in enumerate(sample)]

    with SAMPLE_JSONL.open("w") as f:
        for row in compact:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "sample_id",
        "row_id",
        "venue",
        "year",
        "platform_id",
        "figure_number",
        "image_exists",
        "caption_image_match",
        "context_alignment",
        "same_paper_provenance",
        "usable_for_rebuttal",
        "notes",
    ]
    with TEMPLATE_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in compact:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "row_id": row["row_id"],
                    "venue": row["venue"],
                    "year": row["year"],
                    "platform_id": row["platform_id"],
                    "figure_number": row["figure_number"],
                    "image_exists": row["image_exists"],
                    "caption_image_match": "",
                    "context_alignment": "",
                    "same_paper_provenance": "",
                    "usable_for_rebuttal": "",
                    "notes": "",
                }
            )

    venue_counts: dict[str, int] = {}
    context_count = 0
    image_count = 0
    for row in compact:
        venue_counts[row["venue"]] = venue_counts.get(row["venue"], 0) + 1
        context_count += int(row["has_figure_context"])
        image_count += int(row["image_exists"])

    lines = [
        "# E1 Semantic Sample Report",
        "",
        f"- Source: `{DATA_PATH.name}`",
        f"- Sample size: {len(compact)}",
        f"- Random seed: {SEED}",
        f"- Records with `figure_context`: {context_count}",
        f"- Records without `figure_context`: {len(compact) - context_count}",
        f"- Records with local image file present: {image_count}",
        "",
        "## Venue Counts",
        "",
        "| Venue | Count |",
        "|---|---:|",
    ]
    for venue, count in sorted(venue_counts.items()):
        lines.append(f"| {venue} | {count} |")
    lines.extend(
        [
            "",
            "## Label Set",
            "",
            "- `caption_image_match`: `pass`, `minor_issue`, `fail`, `uncertain`, or `unassessable_missing_image`.",
            "- `context_alignment`: `2_explains`, `1_references_only`, `0_wrong`, `NA_missing`, or `uncertain`.",
            "- `same_paper_provenance`: `pass`, `fail`, or `uncertain`.",
            "- `usable_for_rebuttal`: `yes`, `partial`, or `no`.",
            "",
            "Because the current checkout does not include the release figure image files, visual image-caption",
            "matching must stay `unassessable_missing_image` unless an image exists locally or a separate",
            "image package is provided.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    records = load_records()
    sample = stratified_sample(records)
    write_outputs(sample)
    print(f"wrote {SAMPLE_JSONL}")
    print(f"wrote {TEMPLATE_CSV}")
    print(f"wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
