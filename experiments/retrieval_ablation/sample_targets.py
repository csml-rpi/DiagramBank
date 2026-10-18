#!/usr/bin/env python3
"""Freeze the complete-record universe and a stratified known-item query sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import duckdb


FIELDS = (
    "platform_id",
    "figure_number",
    "figure_path",
    "venue",
    "year",
    "title",
    "abstract",
    "figure_caption",
    "figure_context",
)


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def caption_bin(caption: str) -> str:
    words = len(clean(caption).split())
    if words <= 12:
        return "short_1_12"
    if words <= 40:
        return "medium_13_40"
    return "long_41_plus"


def record_hash(record: dict) -> str:
    payload = "\n".join(
        clean(record[field])
        for field in ("platform_id", "figure_number", "title", "abstract", "figure_caption", "figure_context")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_complete_records(db_path: Path) -> list[dict]:
    connection = duckdb.connect(str(db_path), read_only=True)
    rows = connection.execute(
        """
        SELECT
            platform_id, figure_number, figure_path, venue, year,
            title, abstract, figure_caption, figure_context
        FROM Data
        WHERE trim(coalesce(title, '')) <> ''
          AND trim(coalesce(abstract, '')) <> ''
          AND trim(coalesce(figure_caption, '')) <> ''
          AND trim(coalesce(figure_context, '')) <> ''
        ORDER BY platform_id, figure_number, figure_path
        """
    ).fetchall()
    connection.close()
    return [dict(zip(FIELDS, row)) for row in rows]


def select_one_per_paper(records: list[dict], seed: int) -> list[dict]:
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_paper[str(record["platform_id"])].append(record)

    selected = []
    for platform_id in sorted(by_paper):
        paper_seed = int.from_bytes(
            hashlib.sha256(f"{seed}|{platform_id}".encode("utf-8")).digest()[:8],
            "big",
        )
        paper_rng = random.Random(paper_seed)
        selected.append(paper_rng.choice(by_paper[platform_id]))
    return selected


def proportional_quotas(strata: dict[tuple, list[dict]], sample_size: int) -> dict[tuple, int]:
    total = sum(len(records) for records in strata.values())
    raw = {key: sample_size * len(records) / total for key, records in strata.items()}
    quotas = {key: min(len(strata[key]), math.floor(value)) for key, value in raw.items()}
    remaining = sample_size - sum(quotas.values())

    order = sorted(
        strata,
        key=lambda key: (raw[key] - math.floor(raw[key]), len(strata[key]), key),
        reverse=True,
    )
    while remaining:
        changed = False
        for key in order:
            if quotas[key] < len(strata[key]):
                quotas[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise RuntimeError("unable to allocate requested sample across strata")
    return quotas


def stratified_sample(records: list[dict], sample_size: int, seed: int) -> tuple[list[dict], dict]:
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (record["venue"], int(record["year"]), caption_bin(record["figure_caption"]))
        strata[key].append(record)

    quotas = proportional_quotas(strata, sample_size)
    rng = random.Random(seed)
    sampled = []
    stratum_rows = []
    for key in sorted(strata):
        candidates = sorted(
            strata[key],
            key=lambda row: (str(row["platform_id"]), str(row["figure_number"]), str(row["figure_path"])),
        )
        picked = rng.sample(candidates, quotas[key])
        sampled.extend(picked)
        stratum_rows.append(
            {
                "venue": key[0],
                "year": key[1],
                "caption_length_bin": key[2],
                "eligible_papers": len(candidates),
                "sampled_queries": quotas[key],
            }
        )

    rng.shuffle(sampled)
    return sampled, {"strata": stratum_rows}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    complete = load_complete_records(args.db)
    paper_representatives = select_one_per_paper(complete, args.seed)
    if args.sample_size > len(paper_representatives):
        raise ValueError(
            f"sample size {args.sample_size} exceeds {len(paper_representatives)} eligible papers"
        )

    sampled, sampling = stratified_sample(paper_representatives, args.sample_size, args.seed)
    rows = []
    for index, record in enumerate(sampled, start=1):
        normalized = {field: clean(record[field]) for field in FIELDS}
        rows.append(
            {
                "query_id": f"E4-KIR-{index:05d}",
                **normalized,
                "caption_length_bin": caption_bin(normalized["figure_caption"]),
                "source_sha256": record_hash(normalized),
            }
        )

    target_path = args.output_dir / "target_manifest.jsonl"
    write_jsonl(target_path, rows)

    summary = {
        "database": str(args.db),
        "database_sha256": sha256_file(args.db),
        "complete_record_count": len(complete),
        "complete_paper_count": len(paper_representatives),
        "sample_size": len(rows),
        "unique_sampled_papers": len({row["platform_id"] for row in rows}),
        "seed": args.seed,
        "target_manifest": str(target_path),
        "target_manifest_sha256": sha256_file(target_path),
        **sampling,
    }
    summary_path = args.output_dir / "sampling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "strata"}, indent=2))


if __name__ == "__main__":
    main()
