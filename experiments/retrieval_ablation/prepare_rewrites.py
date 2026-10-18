#!/usr/bin/env python3
"""Prepare source-rich shards for queries rejected by automatic validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=60)
    args = parser.parse_args()

    sources = {row["query_id"]: row for row in load_jsonl(args.manifest)}
    rejected = [
        row for row in load_jsonl(args.rejected) if "missing_output" not in row["reasons"]
    ]
    rows = []
    for rejection in rejected:
        rows.append(
            {
                **sources[rejection["query_id"]],
                "previous_query": rejection.get("query", ""),
                "rejection_reasons": rejection["reasons"],
                "overlap_examples": rejection.get("overlap_examples", []),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), args.shard_size):
        index = start // args.shard_size
        path = args.output_dir / f"rewrite_source_{index:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows[start : start + args.shard_size]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rejected_queries": len(rows), "shards": (len(rows) + args.shard_size - 1) // args.shard_size}))


if __name__ == "__main__":
    main()
