"""
[A1] Stratified sampling of figures for manual annotation study.

Samples 25 figures per (CLIP category x confidence bin) cell = 500 total.
Within each cell, samples are drawn uniformly at random with venue-proportional
allocation. Fixed seed for reproducibility.

Usage:
    python sample.py

Output:
    sample.jsonl — one JSON object per line with fields:
        venue, platform_id, figure_number, figure_path, figure_type,
        confidence, confidence_bin
"""

import duckdb
import json
import os
import random

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
SAMPLES_PER_CELL = 25
CATEGORIES = ["diagram", "plot", "photo", "other"]
BINS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.001),  # upper bound slightly above 1.0 to include 1.0
]
BIN_LABELS = ["[0.50,0.60)", "[0.60,0.70)", "[0.70,0.80)", "[0.80,0.90)", "[0.90,1.00]"]

FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR", os.path.expanduser("~/fig-rag-dir"))
VENUES = {
    "ICLR":    os.path.join(FIG_RAG_DIR, "OpenReview", "ICLR",    "research.db"),
    "ICML":    os.path.join(FIG_RAG_DIR, "OpenReview", "ICML",    "research.db"),
    "NeurIPS": os.path.join(FIG_RAG_DIR, "OpenReview", "NeurIPS", "research.db"),
    "TMLR":    os.path.join(FIG_RAG_DIR, "OpenReview", "TMLR",    "research.db"),
}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sample.jsonl")

# ── Main ────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)

    con = duckdb.connect(":memory:")

    # Attach all venue databases
    for venue, db_path in VENUES.items():
        if os.path.exists(db_path):
            con.execute(f"ATTACH '{db_path}' AS {venue.lower()}")
            print(f"Attached {venue}: {db_path}")
        else:
            print(f"WARNING: missing {db_path}")

    # Build union query across all venues
    parts = []
    for venue in VENUES:
        if os.path.exists(VENUES[venue]):
            parts.append(
                f"""SELECT figure_type, confidence, figure_path, figure_number,
                           platform_id, '{venue}' as venue
                    FROM {venue.lower()}.Figures
                    WHERE figure_type IS NOT NULL
                      AND confidence IS NOT NULL
                      AND confidence >= 0.50"""
            )
    union_query = " UNION ALL ".join(parts)

    samples = []
    stats = {}

    for cat in CATEGORIES:
        for (lo, hi), bin_label in zip(BINS, BIN_LABELS):
            # Fetch all candidates in this cell
            rows = con.execute(
                f"""SELECT venue, platform_id, figure_number, figure_path,
                           figure_type, confidence
                    FROM ({union_query})
                    WHERE figure_type = ?
                      AND confidence >= ? AND confidence < ?""",
                [cat, lo, hi],
            ).fetchall()

            cell_key = f"{cat}|{bin_label}"
            population = len(rows)

            # Sample up to SAMPLES_PER_CELL
            n = min(SAMPLES_PER_CELL, population)
            chosen = random.sample(rows, n)

            stats[cell_key] = {"population": population, "sampled": n}

            for row in chosen:
                samples.append({
                    "venue": row[0],
                    "platform_id": row[1],
                    "figure_number": row[2],
                    "figure_path": row[3],
                    "figure_type": row[4],        # CLIP-assigned label
                    "confidence": round(row[5], 4),
                    "confidence_bin": bin_label,
                })

    # Shuffle to avoid ordering bias during annotation
    random.shuffle(samples)

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Print summary
    print(f"\nSampled {len(samples)} figures → {OUTPUT_PATH}")
    print(f"\n{'Category':>12s} | {'Bin':>12s} | {'Population':>10s} | {'Sampled':>7s}")
    print("-" * 52)
    for cat in CATEGORIES:
        for bin_label in BIN_LABELS:
            cell_key = f"{cat}|{bin_label}"
            s = stats[cell_key]
            print(f"{cat:>12s} | {bin_label:>12s} | {s['population']:>10d} | {s['sampled']:>7d}")

    # Venue distribution of samples
    venue_counts = {}
    for s in samples:
        venue_counts[s["venue"]] = venue_counts.get(s["venue"], 0) + 1
    print(f"\nVenue distribution: {venue_counts}")


if __name__ == "__main__":
    main()
