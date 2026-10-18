#!/usr/bin/env python3
"""Freeze 20 diverse targets for no-retrieval versus retrieval generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


VENUES = ("ICLR", "ICML", "NeurIPS", "TMLR")
BIN_QUOTAS = {
    "short_1_12": 1,
    "medium_13_40": 2,
    "long_41_plus": 2,
}
INCLUDE_RE = re.compile(
    r"\b(?:architecture|framework|pipeline|workflow|overview|schematic|diagram|"
    r"process|system|module|mechanism|procedure|network|policy|model)\b",
    re.IGNORECASE,
)
EXCLUDE_RE = re.compile(
    r"\b(?:qualitative|visualization|comparison|results?|samples?|screenshots?|"
    r"histogram|heatmap|plot|curves?|accuracy|performance)\b",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def order_key(row: dict, seed: int) -> str:
    payload = f"{seed}|{row['query_id']}|{row['platform_id']}|{row['figure_number']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generation_eligible(row: dict) -> bool:
    caption = str(row["figure_caption"])
    return bool(INCLUDE_RE.search(caption)) and not bool(EXCLUDE_RE.search(caption))


def select_for_venue(rows: list[dict], seed: int) -> list[dict]:
    by_bin: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_bin[row["caption_length_bin"]].append(row)
    for values in by_bin.values():
        values.sort(key=lambda row: order_key(row, seed))

    selected = []
    used_years: set[str] = set()
    for length_bin, quota in BIN_QUOTAS.items():
        candidates = by_bin[length_bin]
        for _ in range(quota):
            unused_year = next(
                (row for row in candidates if row not in selected and str(row["year"]) not in used_years),
                None,
            )
            choice = unused_year or next(row for row in candidates if row not in selected)
            selected.append(choice)
            used_years.add(str(choice["year"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    targets = load_jsonl(args.targets)
    eligible = [row for row in targets if generation_eligible(row)]
    selected = []
    for venue in VENUES:
        selected.extend(select_for_venue([row for row in eligible if row["venue"] == venue], args.seed))

    selected.sort(key=lambda row: (VENUES.index(row["venue"]), order_key(row, args.seed)))
    output = []
    for index, row in enumerate(selected, start=1):
        output.append(
            {
                "case_id": f"E4-GEN-{index:02d}",
                "source_query_id": row["query_id"],
                "platform_id": row["platform_id"],
                "figure_number": row["figure_number"],
                "figure_path": row["figure_path"],
                "venue": row["venue"],
                "year": row["year"],
                "caption_length_bin": row["caption_length_bin"],
                "title": row["title"],
                "abstract": row["abstract"],
                "figure_caption": row["figure_caption"],
                "figure_context": row["figure_context"],
                "source_sha256": row["source_sha256"],
            }
        )

    if len(output) != 20 or len({row["platform_id"] for row in output}) != 20:
        raise RuntimeError("generation case selection did not produce 20 unique papers")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "case_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "seed": args.seed,
        "selection_rule": {
            "caption_must_match": INCLUDE_RE.pattern,
            "caption_must_not_match": EXCLUDE_RE.pattern,
        },
        "source_targets": len(targets),
        "generation_eligible_targets": len(eligible),
        "cases": len(output),
        "unique_papers": len({row["platform_id"] for row in output}),
        "venue_counts": dict(sorted(Counter(row["venue"] for row in output).items())),
        "caption_length_counts": dict(
            sorted(Counter(row["caption_length_bin"] for row in output).items())
        ),
        "year_counts": dict(sorted(Counter(str(row["year"]) for row in output).items())),
    }
    (args.output_dir / "case_selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
