#!/usr/bin/env python3
"""
Step 3: Aggregate three judges via median consensus.

Loads labels from Haiku 4.5, GPT-5.4-mini, and GPT-5.4, joins by
judgment_key, computes the median score, and reports agreement statistics.

Output: consensus.jsonl

Usage:
    python consensus.py
"""

import json
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
POOL_PATH = SCRIPT_DIR / "pool.jsonl"
LABELS_HAIKU = SCRIPT_DIR / "labels_haiku.jsonl"
LABELS_GPT_MINI = SCRIPT_DIR / "labels_gpt_mini.jsonl"
LABELS_GPT = SCRIPT_DIR / "labels_gpt.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "consensus.jsonl"


def load_labels(path: Path, score_field: str) -> dict:
    """Load label file into {judgment_key: score} dict."""
    labels = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                labels[rec["judgment_key"]] = rec[score_field]
    return labels


def fleiss_kappa(ratings_matrix: list[list[int]], n_categories: int) -> float:
    """Compute Fleiss' kappa for inter-rater reliability.

    ratings_matrix: list of [count_cat_0, count_cat_1, ..., count_cat_c] per item
    n_categories: number of rating categories
    """
    N = len(ratings_matrix)  # number of items
    n = sum(ratings_matrix[0])  # number of raters per item

    # Proportion of all ratings in each category
    p = [0.0] * n_categories
    for row in ratings_matrix:
        for j in range(n_categories):
            p[j] += row[j]
    total_ratings = N * n
    p = [x / total_ratings for x in p]

    # Per-item agreement
    P_i = []
    for row in ratings_matrix:
        s = sum(r * (r - 1) for r in row)
        P_i.append(s / (n * (n - 1)) if n > 1 else 1.0)

    P_bar = sum(P_i) / N
    P_e = sum(pj ** 2 for pj in p)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def main():
    # Load pool for metadata
    pool = {}
    with open(POOL_PATH) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                pool[rec["judgment_key"]] = rec

    # Load all three judge label files
    haiku = load_labels(LABELS_HAIKU, "score_haiku")
    gpt_mini = load_labels(LABELS_GPT_MINI, "score_gpt_mini")
    gpt = load_labels(LABELS_GPT, "score_gpt")

    print(f"Pool:      {len(pool):,} items")
    print(f"Haiku:     {len(haiku):,} labels")
    print(f"GPT-mini:  {len(gpt_mini):,} labels")
    print(f"GPT:       {len(gpt):,} labels")

    # Join and compute consensus
    results = []
    missing = 0
    error_count = 0

    for key, pool_rec in pool.items():
        s_h = haiku.get(key)
        s_m = gpt_mini.get(key)
        s_g = gpt.get(key)

        if s_h is None or s_m is None or s_g is None:
            missing += 1
            continue

        # Treat parse errors (-1) as 0 for consensus
        scores = []
        has_error = False
        for s in (s_h, s_m, s_g):
            if s == -1:
                scores.append(0)
                has_error = True
            else:
                scores.append(s)

        if has_error:
            error_count += 1

        median_score = int(statistics.median(scores))

        results.append({
            "judgment_key": key,
            "query_name": pool_rec["query_name"],
            "retrieved_platform_id": pool_rec["retrieved_platform_id"],
            "retrieved_figure_number": pool_rec["retrieved_figure_number"],
            "score_haiku": s_h,
            "score_gpt_mini": s_m,
            "score_gpt": s_g,
            "consensus_score": median_score,
            "unanimous": (s_h == s_m == s_g and s_h != -1),
        })

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    print(f"\nConsensus: {len(results):,} items written to {OUTPUT_PATH}")
    if missing:
        print(f"  Missing from one or more judges: {missing}")
    if error_count:
        print(f"  Items with at least one parse error (-1 -> 0): {error_count}")

    # ── Agreement Statistics ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGREEMENT STATISTICS")
    print(f"{'='*60}")

    # Score distribution
    consensus_dist = Counter(r["consensus_score"] for r in results)
    print(f"\nConsensus score distribution:")
    for s in (0, 1, 2):
        cnt = consensus_dist.get(s, 0)
        pct = 100 * cnt / len(results) if results else 0
        print(f"  {s}: {cnt:>5d} ({pct:5.1f}%)")

    # Per-judge distributions
    for judge_name, field in [("Haiku", "score_haiku"), ("GPT-mini", "score_gpt_mini"), ("GPT", "score_gpt")]:
        dist = Counter(r[field] for r in results)
        print(f"\n{judge_name} distribution:")
        for s in (-1, 0, 1, 2):
            cnt = dist.get(s, 0)
            if cnt > 0:
                pct = 100 * cnt / len(results)
                print(f"  {s}: {cnt:>5d} ({pct:5.1f}%)")

    # Unanimous agreement
    unanimous = sum(1 for r in results if r["unanimous"])
    print(f"\nUnanimous agreement: {unanimous}/{len(results)} ({100*unanimous/len(results):.1f}%)")

    # Pairwise agreement
    judge_pairs = [
        ("Haiku", "score_haiku", "GPT-mini", "score_gpt_mini"),
        ("Haiku", "score_haiku", "GPT", "score_gpt"),
        ("GPT-mini", "score_gpt_mini", "GPT", "score_gpt"),
    ]
    print(f"\nPairwise agreement:")
    for name_a, field_a, name_b, field_b in judge_pairs:
        agree = sum(1 for r in results
                    if r[field_a] == r[field_b] and r[field_a] != -1 and r[field_b] != -1)
        valid = sum(1 for r in results if r[field_a] != -1 and r[field_b] != -1)
        pct = 100 * agree / valid if valid else 0
        print(f"  {name_a} vs {name_b}: {agree}/{valid} ({pct:.1f}%)")

    # Fleiss' kappa (3 raters, 3 categories: 0, 1, 2)
    ratings_matrix = []
    for r in results:
        scores = []
        for field in ("score_haiku", "score_gpt_mini", "score_gpt"):
            s = r[field]
            scores.append(max(s, 0))  # clamp -1 to 0
        counts = [0, 0, 0]
        for s in scores:
            if 0 <= s <= 2:
                counts[s] += 1
        ratings_matrix.append(counts)

    kappa = fleiss_kappa(ratings_matrix, 3)
    print(f"\nFleiss' kappa (3 raters, 3 categories): {kappa:.3f}")
    if kappa < 0.2:
        interp = "slight"
    elif kappa < 0.4:
        interp = "fair"
    elif kappa < 0.6:
        interp = "moderate"
    elif kappa < 0.8:
        interp = "substantial"
    else:
        interp = "almost perfect"
    print(f"  Interpretation: {interp} agreement")


if __name__ == "__main__":
    main()
