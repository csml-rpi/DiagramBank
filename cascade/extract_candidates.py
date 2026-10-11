"""
Stage 1: Extract two-tier candidates from all venue databases.

Tier 1: All figures where CLIP predicts "diagram" (any confidence).
        ~91,696 figures. Rule: majority vote (>=2/3 say diagram).

Tier 2: Non-diagram figures with CLIP confidence < 0.70.
        ~50,331 figures. Rule: VLM consensus (both VLMs say diagram).

Output:
    candidates.jsonl — all candidates (Tier 1 + Tier 2) with tier label

Usage:
    python extract_candidates.py [--tier2-conf 0.70]
"""

import argparse
import duckdb
import json
import os
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR", os.path.expanduser("~/fig-rag-dir"))
VENUES = {
    "ICLR":    os.path.join(FIG_RAG_DIR, "OpenReview", "ICLR",    "research.db"),
    "ICML":    os.path.join(FIG_RAG_DIR, "OpenReview", "ICML",    "research.db"),
    "NeurIPS": os.path.join(FIG_RAG_DIR, "OpenReview", "NeurIPS", "research.db"),
    "TMLR":    os.path.join(FIG_RAG_DIR, "OpenReview", "TMLR",    "research.db"),
}
SCRIPT_DIR = Path(__file__).parent
OUTPUT_PATH = SCRIPT_DIR / "candidates.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Extract two-tier cascade candidates")
    parser.add_argument("--tier2-conf", type=float, default=0.70,
                        help="Max CLIP confidence for Tier 2 (default: 0.70)")
    parser.add_argument("--accepted-only", action="store_true",
                        help="Only include figures from accepted papers")
    args = parser.parse_args()

    con = duckdb.connect(":memory:")

    # Attach all venue databases
    for venue, db_path in VENUES.items():
        if os.path.exists(db_path):
            con.execute(f"ATTACH '{db_path}' AS {venue.lower()} (READ_ONLY)")
            print(f"Attached {venue}: {db_path}")
        else:
            print(f"WARNING: missing {db_path}")

    # ── Tier 1: CLIP = diagram (any confidence) ─────────────────────────
    tier1_parts = []
    for venue in VENUES:
        if not os.path.exists(VENUES[venue]):
            continue
        accepted_clause = _accepted_clause(venue, args.accepted_only)
        tier1_parts.append(f"""
            SELECT '{venue}' AS venue,
                   f.platform_id,
                   f.figure_number,
                   f.figure_path,
                   f.figure_type AS clip_type,
                   f.confidence AS clip_confidence,
                   'tier1' AS tier
            FROM {venue.lower()}.Figures f
            WHERE f.figure_type = 'diagram'
              {accepted_clause}
        """)

    # ── Tier 2: Non-diagram, confidence < threshold ─────────────────────
    tier2_parts = []
    for venue in VENUES:
        if not os.path.exists(VENUES[venue]):
            continue
        accepted_clause = _accepted_clause(venue, args.accepted_only)
        tier2_parts.append(f"""
            SELECT '{venue}' AS venue,
                   f.platform_id,
                   f.figure_number,
                   f.figure_path,
                   f.figure_type AS clip_type,
                   f.confidence AS clip_confidence,
                   'tier2' AS tier
            FROM {venue.lower()}.Figures f
            WHERE f.figure_type != 'diagram'
              AND f.confidence < {args.tier2_conf}
              {accepted_clause}
        """)

    if not tier1_parts:
        print("ERROR: No venue databases found")
        return

    # Execute both queries
    tier1_query = " UNION ALL ".join(tier1_parts)
    tier2_query = " UNION ALL ".join(tier2_parts)
    full_query = f"({tier1_query}) UNION ALL ({tier2_query})"

    rows = con.execute(f"""
        SELECT venue, platform_id, figure_number, figure_path,
               clip_type, clip_confidence, tier
        FROM ({full_query})
        ORDER BY tier, venue, clip_confidence DESC
    """).fetchall()

    # Write output and collect stats
    tier_counts = {"tier1": {}, "tier2": {}}
    tier2_by_type = {}
    with open(OUTPUT_PATH, "w") as f:
        for row in rows:
            rec = {
                "venue": row[0],
                "platform_id": row[1],
                "figure_number": row[2],
                "figure_path": row[3],
                "clip_type": row[4],
                "clip_confidence": round(row[5], 4),
                "tier": row[6],
            }
            f.write(json.dumps(rec) + "\n")
            tier = row[6]
            venue = row[0]
            tier_counts[tier][venue] = tier_counts[tier].get(venue, 0) + 1
            if tier == "tier2":
                clip_type = row[4]
                tier2_by_type[clip_type] = tier2_by_type.get(clip_type, 0) + 1

    # ── Print summary ───────────────────────────────────────────────────
    t1_total = sum(tier_counts["tier1"].values())
    t2_total = sum(tier_counts["tier2"].values())
    grand_total = t1_total + t2_total

    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"\nTier 1 — CLIP=diagram (any conf): {t1_total:,}")
    print(f"  {'Venue':<10s} {'Count':>8s}")
    print(f"  {'-'*20}")
    for venue in sorted(tier_counts["tier1"]):
        print(f"  {venue:<10s} {tier_counts['tier1'][venue]:>8,d}")

    print(f"\nTier 2 — Non-diagram, conf < {args.tier2_conf}: {t2_total:,}")
    print(f"  {'Venue':<10s} {'Count':>8s}")
    print(f"  {'-'*20}")
    for venue in sorted(tier_counts["tier2"]):
        print(f"  {venue:<10s} {tier_counts['tier2'][venue]:>8,d}")
    print(f"\n  By CLIP class:")
    for clip_type in sorted(tier2_by_type):
        print(f"    {clip_type:<10s} {tier2_by_type[clip_type]:>8,d}")

    print(f"\nTotal candidates: {grand_total:,}")
    print(f"Output: {OUTPUT_PATH}")

    # Confidence distribution for Tier 1
    conf_bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.001)]
    bin_labels = ["[0.50,0.60)", "[0.60,0.70)", "[0.70,0.80)", "[0.80,0.90)", "[0.90,1.00]"]
    print(f"\nTier 1 confidence distribution:")
    for (lo, hi), label in zip(conf_bins, bin_labels):
        n = con.execute(f"""
            SELECT COUNT(*) FROM ({tier1_query})
            WHERE clip_confidence >= {lo} AND clip_confidence < {hi}
        """).fetchone()[0]
        pct = 100 * n / t1_total if t1_total > 0 else 0
        print(f"  {label}: {n:>8,d} ({pct:>5.1f}%)")

    con.close()


def _accepted_clause(venue: str, accepted_only: bool) -> str:
    if not accepted_only:
        return ""
    return """
        AND f.platform_id IN (
            SELECT platform_id FROM {v}.Papers
            WHERE lower(decision) LIKE '%accept%'
               OR lower(decision) LIKE '%spotlight%'
               OR lower(decision) LIKE '%poster%'
               OR lower(decision) LIKE '%oral%'
        )
    """.format(v=venue.lower())


if __name__ == "__main__":
    main()
