#!/usr/bin/env python3
"""Select top-three generation references after exact and perceptual deduplication."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perceptual_hash(path: Path, hash_size: int = 8, high_frequency_factor: int = 4) -> str:
    size = hash_size * high_frequency_factor
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.asarray(grayscale, dtype=np.float32)
    coefficients = dctn(pixels, type=2, norm="ortho")[:hash_size, :hash_size]
    values = coefficients.flatten()
    median = float(np.median(values[1:]))
    bits = values > median
    encoded = 0
    for bit in bits:
        encoded = (encoded << 1) | int(bit)
    return f"{encoded:0{hash_size * hash_size // 4}x}"


def hamming_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def image_identity(path: Path) -> dict:
    with Image.open(path) as image:
        image.verify()
    return {
        "local_image_path": str(path.resolve()),
        "image_sha256": sha256_file(path),
        "image_phash": perceptual_hash(path),
    }


def duplicate_reason(
    candidate: dict,
    reference: dict,
    threshold: int,
    prefix: str,
) -> str | None:
    if candidate["image_sha256"] == reference["image_sha256"]:
        return f"exact_duplicate_of_{prefix}"
    distance = hamming_distance(candidate["image_phash"], reference["image_phash"])
    if distance <= threshold:
        return f"near_duplicate_of_{prefix}_phash_distance_{distance}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--references-per-case", type=int, default=3)
    parser.add_argument("--phash-threshold", type=int, default=5)
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    rankings_by_case: dict[str, list[dict]] = defaultdict(list)
    for row in load_jsonl(args.rankings):
        rankings_by_case[row["case_id"]].append(row)
    for rows in rankings_by_case.values():
        rows.sort(key=lambda row: int(row["rank"]))

    selected_rows = []
    exclusion_rows = []
    for case in cases:
        target_path = args.image_root / case["figure_path"]
        if not target_path.is_file():
            raise FileNotFoundError(f"missing target image: {target_path}")
        target_identity = image_identity(target_path)
        selected = []
        for candidate in rankings_by_case[case["case_id"]]:
            candidate_path = args.image_root / candidate["retrieved_figure_path"]
            if not candidate_path.is_file():
                exclusion_rows.append(
                    {
                        "case_id": case["case_id"],
                        "original_rank": candidate["rank"],
                        "retrieved_figure_path": candidate["retrieved_figure_path"],
                        "reason": "missing_image",
                    }
                )
                continue
            identity = image_identity(candidate_path)
            enriched = {**candidate, **identity}
            reason = duplicate_reason(
                enriched,
                target_identity,
                args.phash_threshold,
                "target",
            )
            if reason is None:
                for selected_index, existing in enumerate(selected, start=1):
                    reason = duplicate_reason(
                        enriched,
                        existing,
                        args.phash_threshold,
                        f"selected_reference_{selected_index}",
                    )
                    if reason is not None:
                        break
            if reason is not None:
                exclusion_rows.append(
                    {
                        "case_id": case["case_id"],
                        "original_rank": candidate["rank"],
                        "retrieved_figure_path": candidate["retrieved_figure_path"],
                        "reason": reason,
                    }
                )
                continue
            selected.append(enriched)
            if len(selected) == args.references_per_case:
                break

        if len(selected) != args.references_per_case:
            raise RuntimeError(
                f"{case['case_id']} has only {len(selected)} non-duplicate references"
            )
        for final_rank, row in enumerate(selected, start=1):
            selected_rows.append(
                {
                    **row,
                    "rank": final_rank,
                    "original_retrieval_rank": row["rank"],
                    "target_image_sha256": target_identity["image_sha256"],
                    "target_image_phash": target_identity["image_phash"],
                    "phash_threshold": args.phash_threshold,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.exclusions.parent.mkdir(parents=True, exist_ok=True)
    with args.exclusions.open("w", encoding="utf-8") as handle:
        for row in exclusion_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "selected_references": len(selected_rows),
                "excluded_candidates": len(exclusion_rows),
                "phash_algorithm": "32x32 grayscale DCT, median-thresholded top-left 8x8",
                "phash_threshold": args.phash_threshold,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
