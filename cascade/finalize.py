"""
Stage 4: Finalize the cascade-filtered dataset.

Merges all voting results + tiebreak labels to produce the final set of
verified diagrams. Compares against the old τ=0.85 threshold.

Outputs:
  - final_diagrams.jsonl:  Final set of verified diagrams
  - final_stats.json:      Comprehensive statistics
  - comparison.txt:        Human-readable old vs. new comparison

Usage:
    python finalize.py [--update-db]

Flags:
    --update-db   Update the venue DuckDB databases with cascade labels
                  (adds cascade_verified column to Figures table)
"""

import argparse
import duckdb
import hashlib
import json
import os
from collections import Counter
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
VOTED_PATH = SCRIPT_DIR / "voted.jsonl"
TIEBREAK_PATH = SCRIPT_DIR / "labels_gpt_tiebreak.jsonl"
OUTPUT_FINAL = SCRIPT_DIR / "final_diagrams.jsonl"
OUTPUT_STATS = SCRIPT_DIR / "final_stats.json"
OUTPUT_COMPARISON = SCRIPT_DIR / "comparison.txt"

OLD_THRESHOLD = 0.85


def final_keep_path(record, gpt_label=None):
    """Apply the documented cascade to one completed voting record."""
    category = record["vote_category"]
    if category in {"t1_unanimous_diagram", "t1_majority_diagram"}:
        return category
    if category in {"t1_minority_diagram", "t2_vlm_consensus"}:
        if gpt_label not in {"diagram", "plot", "photo", "other"}:
            raise ValueError(f"Missing or invalid GPT confirmation for {category}")
        if gpt_label == "diagram":
            return category + "_gpt_confirmed"
        return None
    if category in {"t1_unanimous_reject", "t2_one_vlm_diagram", "t2_no_vlm_diagram"}:
        return None
    raise ValueError(f"Incomplete or unknown vote category: {category}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Finalize cascade-filtered dataset")
    parser.add_argument("--update-db", action="store_true",
                        help="Update venue DuckDB databases with cascade labels")
    parser.add_argument("--voted", type=Path, default=VOTED_PATH)
    parser.add_argument("--tiebreak", type=Path, default=TIEBREAK_PATH)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--verify-release-db", type=Path,
                        help="Require exact key equality with a read-only Data table")
    args = parser.parse_args()

    # ── Load voted results ──────────────────────────────────────────────
    print("Loading voted results...")
    voted = {}
    with open(args.voted) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                key = f"{r['platform_id']}|{r['figure_number']}"
                if key in voted:
                    raise ValueError(f"Duplicate voting record: {key}")
                voted[key] = r

    # ── Load tiebreak results ───────────────────────────────────────────
    tiebreak = {}
    if args.tiebreak.exists():
        print("Loading tiebreak results...")
        with open(args.tiebreak) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    key = f"{r['platform_id']}|{r['figure_number']}"
                    if key in tiebreak:
                        raise ValueError(f"Duplicate tiebreak record: {key}")
                    tiebreak[key] = r.get("label_gpt_tiebreak", "MISSING")
        print(f"  Tiebreak labels: {len(tiebreak):,}")

    # ── Apply final decision ────────────────────────────────────────────
    print("\nApplying final decisions...")
    final = []
    decision_counts = Counter()
    keep_paths = Counter()
    venue_counts = Counter()
    conf_dist = {"above_085": 0, "below_085": 0}

    for key, rec in voted.items():
        clip_conf = rec["clip_confidence"]
        if key in tiebreak:
            rec["label_gpt_tiebreak"] = tiebreak[key]
        keep_path = final_keep_path(rec, tiebreak.get(key))
        keep = keep_path is not None

        # Final decision
        decision = "keep" if keep else "reject"
        rec["final_decision"] = decision
        rec["keep_path"] = keep_path
        decision_counts[decision] += 1

        if keep:
            final.append(rec)
            keep_paths[keep_path] += 1
            venue_counts[rec["venue"]] += 1
            if clip_conf >= OLD_THRESHOLD:
                conf_dist["above_085"] += 1
            else:
                conf_dist["below_085"] += 1

    verification = None
    if args.verify_release_db:
        with duckdb.connect(str(args.verify_release_db), read_only=True) as con:
            release_rows = con.execute("SELECT platform_id, figure_number FROM Data").fetchall()
        release_keys = {f"{pid}|{number}" for pid, number in release_rows}
        final_keys = {f"{r['platform_id']}|{r['figure_number']}" for r in final}
        if len(release_rows) != len(release_keys) or final_keys != release_keys:
            raise ValueError(f"Release mismatch: {len(final_keys - release_keys)} extra, "
                             f"{len(release_keys - final_keys)} missing; "
                             f"{len(release_rows) - len(release_keys)} duplicate database keys")
        verification = {"database": str(args.verify_release_db), "sha256": sha256_file(args.verify_release_db),
                        "exact_key_match": True, "records": len(release_keys)}

    # ── Write final dataset ─────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting final dataset...")
    with open(args.output_dir / OUTPUT_FINAL.name, "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")
    print(f"  final_diagrams.jsonl: {len(final):,} diagrams")

    # ── Count old threshold baseline ────────────────────────────────────
    old_count = sum(1 for r in voted.values()
                    if r["clip_type"] == "diagram" and r["clip_confidence"] >= OLD_THRESHOLD)

    # ── Statistics ──────────────────────────────────────────────────────
    total_candidates = len(voted)
    stats = {
        "total_candidates": total_candidates,
        "tier_candidates": dict(Counter(r["tier"] for r in voted.values())),
        "final_diagrams_cascade": len(final),
        "final_diagrams_old_threshold": old_count,
        "decisions": dict(decision_counts),
        "kept_paths": dict(keep_paths),
        "release_verification": verification,
        "input_sha256": {str(p): sha256_file(p) for p in (args.voted, args.tiebreak) if p.exists()},
        "per_venue": dict(venue_counts),
        "confidence_distribution": conf_dist,
        "new_diagrams_below_085": conf_dist["below_085"],
        "retention_rate_cascade": round(len(final) / total_candidates, 4),
        "retention_rate_old": round(old_count / total_candidates, 4),
        "improvement_count": len(final) - old_count,
        "improvement_pct": round(100 * (len(final) - old_count) / old_count, 1) if old_count > 0 else 0,
    }

    with open(args.output_dir / OUTPUT_STATS.name, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Comparison report ───────────────────────────────────────────────
    report_lines = [
        "=" * 60,
        "CASCADE FILTER vs. OLD THRESHOLD COMPARISON",
        "=" * 60,
        "",
        f"Total accepted-paper cascade candidates: {total_candidates:>10,d}",
        "",
        "STRICT CLIP VIEW IN THIS CANDIDATE POOL (diagram label and confidence >= 0.85):",
        f"  Diagrams retained:    {old_count:>10,d}",
        f"  Retention rate:       {100*old_count/total_candidates:>9.1f}%",
        "",
        "NEW APPROACH (Multi-judge cascade):",
        f"  Diagrams retained:    {len(final):>10,d}",
        f"  Retention rate:       {100*len(final)/total_candidates:>9.1f}%",
        "",
        "COMPARISON:",
        f"  Additional diagrams:  {len(final) - old_count:>10,d}",
        f"  Retained-count change: {stats['improvement_pct']:>8.1f}%",
        f"  New diagrams (conf < 0.85): {conf_dist['below_085']:>6,d}",
        "",
        "Per-venue breakdown:",
    ]
    for venue in sorted(venue_counts):
        report_lines.append(f"  {venue:<10s}: {venue_counts[venue]:>8,d}")

    report = "\n".join(report_lines)
    with open(args.output_dir / OUTPUT_COMPARISON.name, "w") as f:
        f.write(report + "\n")
    print(f"\n{report}")

    # ── Optionally update DuckDB ────────────────────────────────────────
    if args.update_db:
        print("\nUpdating venue databases...")
        final_keys = {f"{r['platform_id']}|{r['figure_number']}" for r in final}

        for venue, db_path in VENUES.items():
            if not os.path.exists(db_path):
                continue

            con = duckdb.connect(db_path)

            # Add column if needed
            try:
                con.execute("ALTER TABLE Figures ADD COLUMN cascade_verified BOOLEAN DEFAULT FALSE")
                print(f"  {venue}: Added cascade_verified column")
            except Exception:
                # Column already exists
                con.execute("UPDATE Figures SET cascade_verified = FALSE")

            # Mark verified diagrams
            venue_finals = [
                r for r in final if r["venue"] == venue
            ]
            for r in venue_finals:
                con.execute(
                    "UPDATE Figures SET cascade_verified = TRUE "
                    "WHERE platform_id = ? AND figure_number = ?",
                    [r["platform_id"], r["figure_number"]]
                )

            n_updated = con.execute(
                "SELECT COUNT(*) FROM Figures WHERE cascade_verified = TRUE"
            ).fetchone()[0]
            print(f"  {venue}: {n_updated:,d} diagrams marked as cascade_verified")
            con.close()

        print("Database update complete.")


if __name__ == "__main__":
    main()
