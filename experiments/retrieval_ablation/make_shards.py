#!/usr/bin/env python3
"""Split the frozen target manifest into deterministic agent-owned shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=100)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), args.shard_size):
        shard_index = start // args.shard_size
        path = args.output_dir / f"source_{shard_index:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows[start : start + args.shard_size]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(rows), "shards": (len(rows) + args.shard_size - 1) // args.shard_size}))


if __name__ == "__main__":
    main()
