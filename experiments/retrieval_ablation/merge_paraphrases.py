#!/usr/bin/env python3
"""Merge original agent rewrites with validated replacement rewrites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_directory(directory: Path) -> dict[str, dict]:
    output = {}
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in output:
                raise RuntimeError(f"duplicate query_id in {directory}: {query_id}")
            output[query_id] = {"query_id": query_id, "query": str(row["query"]).strip()}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--replacement-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_order = [
        json.loads(line)["query_id"]
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    original = load_directory(args.original_dir)
    replacement = load_directory(args.replacement_dir)
    merged = {**original, **replacement}
    missing = [query_id for query_id in source_order if query_id not in merged]
    unexpected = sorted(set(merged) - set(source_order))
    if missing or unexpected:
        raise RuntimeError(f"merge coverage error: missing={len(missing)} unexpected={len(unexpected)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for query_id in source_order:
            handle.write(json.dumps(merged[query_id], ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(source_order), "replacements": len(replacement)}))


if __name__ == "__main__":
    main()
