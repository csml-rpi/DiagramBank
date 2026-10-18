#!/usr/bin/env python3
"""Retrieve full-field reference candidates for the 20 frozen generation cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np


FIELDS = ("title", "abstract", "caption", "context")


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def fingerprint(title: object, abstract: object) -> str:
    return f"{clean(title).casefold()}\n{clean(abstract).casefold()}"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_candidates(db_path: Path) -> list[dict]:
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
    columns = [description[0] for description in connection.description]
    connection.close()
    return [dict(zip(columns, row)) for row in rows]


def load_embeddings(directory: Path) -> dict[str, np.ndarray]:
    paths = {
        "title": directory / "candidate_title.f16.npy",
        "abstract": directory / "candidate_abstract.f16.npy",
        "caption": directory / "candidate_caption.f16.npy",
        "context": directory / "candidate_context.f32.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing embedding files: {missing}")
    return {field: np.load(path, mmap_mode="r") for field, path in paths.items()}


def rank_case(
    target_index: int,
    candidates: list[dict],
    embeddings: dict[str, np.ndarray],
    top_k: int,
    chunk_size: int,
) -> list[tuple[int, float]]:
    target = candidates[target_index]
    target_fingerprint = fingerprint(target["title"], target["abstract"])
    scores = np.zeros(len(candidates), dtype=np.float32)
    for field in FIELDS:
        query = np.asarray(embeddings[field][target_index], dtype=np.float32)
        for start in range(0, len(candidates), chunk_size):
            stop = min(start + chunk_size, len(candidates))
            block = np.asarray(embeddings[field][start:stop], dtype=np.float32)
            scores[start:stop] += block @ query
    scores /= float(len(FIELDS))

    for index, candidate in enumerate(candidates):
        if str(candidate["platform_id"]) == str(target["platform_id"]):
            scores[index] = -np.inf
        elif fingerprint(candidate["title"], candidate["abstract"]) == target_fingerprint:
            scores[index] = -np.inf

    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [(int(index), float(scores[index])) for index in top_indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()

    candidates = load_candidates(args.db)
    manifest = load_jsonl(args.embedding_dir / "candidate_manifest.jsonl")
    keys = [(str(row["platform_id"]), str(row["figure_number"])) for row in candidates]
    manifest_keys = [
        (str(row["platform_id"]), str(row["figure_number"])) for row in manifest
    ]
    if keys != manifest_keys:
        raise RuntimeError("database candidate order does not match embedding manifest")
    index_by_key = {key: index for index, key in enumerate(keys)}
    embeddings = load_embeddings(args.embedding_dir)
    cases = load_jsonl(args.cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            target_key = (str(case["platform_id"]), str(case["figure_number"]))
            target_index = index_by_key[target_key]
            ranked = rank_case(
                target_index,
                candidates,
                embeddings,
                args.top_k,
                args.chunk_size,
            )
            for rank, (candidate_index, score) in enumerate(ranked, start=1):
                candidate = candidates[candidate_index]
                handle.write(
                    json.dumps(
                        {
                            "case_id": case["case_id"],
                            "rank": rank,
                            "score": score,
                            "retrieved_platform_id": str(candidate["platform_id"]),
                            "retrieved_figure_number": str(candidate["figure_number"]),
                            "retrieved_figure_path": str(candidate["figure_path"]),
                            "retrieved_title": str(candidate["title"]),
                            "retrieved_caption": str(candidate["figure_caption"]),
                            "retrieved_venue": str(candidate["venue"]),
                            "retrieved_year": int(candidate["year"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"{case['case_id']}: retrieved {len(ranked)} candidates", flush=True)


if __name__ == "__main__":
    main()
