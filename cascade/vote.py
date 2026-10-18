"""
Stage 2 merge: Apply majority vote across CLIP + Haiku + GPT-mini.

Merges labels from all three judges and applies voting rules:
  - unanimous_diagram: all 3 say diagram → keep (high confidence)
  - majority_diagram:  ≥2 of 3 say diagram → keep
  - minority_diagram:  only 1 says diagram (CLIP only, both VLMs disagree)
  - unanimous_reject:  all 3 say non-diagram → reject

Outputs:
  - voted.jsonl:         All candidates with merged votes and decision
  - keep_majority.jsonl: Figures passing majority vote (≥2 of 3)
  - disagreements.jsonl: Figures where judges disagree (for Stage 3 tiebreak)
  - vote_stats.json:     Summary statistics

Usage:
    python vote.py
"""

import json
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CANDIDATES_PATH = SCRIPT_DIR / "candidates.jsonl"
HAIKU_PATH = SCRIPT_DIR / "labels_haiku.jsonl"
GPT_MINI_PATH = SCRIPT_DIR / "labels_gpt_mini.jsonl"
OUTPUT_ALL = SCRIPT_DIR / "voted.jsonl"
OUTPUT_KEEP = SCRIPT_DIR / "keep_majority.jsonl"
OUTPUT_DISAGREE = SCRIPT_DIR / "disagreements.jsonl"
OUTPUT_STATS = SCRIPT_DIR / "vote_stats.json"


def main():
    # ── Load all data ───────────────────────────────────────────────────
    print("Loading data...")

    with open(CANDIDATES_PATH) as f:
        candidates = {
            f"{r['platform_id']}|{r['figure_number']}": r
            for line in f if line.strip()
            for r in [json.loads(line)]
        }
    print(f"  Candidates:  {len(candidates):,}")

    haiku_labels = {}
    if HAIKU_PATH.exists():
        with open(HAIKU_PATH) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    key = f"{r['platform_id']}|{r['figure_number']}"
                    haiku_labels[key] = r.get("label_haiku", "MISSING")
        print(f"  Haiku labels: {len(haiku_labels):,}")
    else:
        print("  WARNING: labels_haiku.jsonl not found")

    gpt_labels = {}
    if GPT_MINI_PATH.exists():
        with open(GPT_MINI_PATH) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    key = f"{r['platform_id']}|{r['figure_number']}"
                    gpt_labels[key] = r.get("label_gpt_mini", "MISSING")
        print(f"  GPT-mini labels: {len(gpt_labels):,}")
    else:
        print("  WARNING: labels_gpt_mini.jsonl not found")

    # ── Merge and vote ──────────────────────────────────────────────────
    print("\nApplying tier-aware voting...")

    results = []
    categories = Counter()
    venue_counts = Counter()
    missing_count = 0

    for key, cand in candidates.items():
        tier = cand.get("tier", "tier1")
        clip_vote = cand["clip_type"]  # "diagram" for tier1, other for tier2
        haiku_vote = haiku_labels.get(key, "PENDING")
        gpt_vote = gpt_labels.get(key, "PENDING")

        # Count diagram votes
        votes = [clip_vote, haiku_vote, gpt_vote]
        diagram_votes = sum(1 for v in votes if v == "diagram")

        # Determine category (tier-aware)
        if haiku_vote == "PENDING" or gpt_vote == "PENDING":
            category = "pending"
            missing_count += 1
        elif tier == "tier1":
            # Tier 1: CLIP=diagram. Majority vote (>=2/3).
            if diagram_votes == 3:
                category = "t1_unanimous_diagram"
            elif diagram_votes >= 2:
                category = "t1_majority_diagram"
            elif diagram_votes == 1:
                category = "t1_minority_diagram"  # CLIP only, both VLMs disagree
            else:
                category = "t1_unanimous_reject"
        else:
            # Tier 2: CLIP!=diagram. VLM consensus required (both VLMs = diagram).
            both_vlms_diagram = (haiku_vote == "diagram" and gpt_vote == "diagram")
            if both_vlms_diagram:
                category = "t2_vlm_consensus"  # Both VLMs override CLIP → send to GPT-5.4
            elif haiku_vote == "diagram" or gpt_vote == "diagram":
                category = "t2_one_vlm_diagram"  # Only 1 VLM says diagram → reject
            else:
                category = "t2_no_vlm_diagram"  # Neither VLM says diagram → reject

        # Decision logic
        if tier == "tier1":
            keep = category in ("t1_unanimous_diagram", "t1_majority_diagram")
            disagree = category in ("t1_majority_diagram", "t1_minority_diagram")
        else:
            # Tier 2: VLM consensus matches go to GPT-5.4 verification
            keep = False  # Tier 2 keeps are deferred to Stage 3 (GPT-5.4)
            disagree = (category == "t2_vlm_consensus")  # Send to tiebreak

        rec = {
            **cand,
            "label_haiku": haiku_vote,
            "label_gpt_mini": gpt_vote,
            "diagram_votes": diagram_votes,
            "vote_category": category,
            "keep_majority": keep,
            "judges_disagree": disagree,
        }
        results.append(rec)
        categories[category] += 1
        if keep:
            venue_counts[cand["venue"]] += 1

    # ── Write outputs ───────────────────────────────────────────────────
    print("\nWriting outputs...")

    # All results
    with open(OUTPUT_ALL, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  voted.jsonl: {len(results):,} records")

    # Majority-vote keeps
    keeps = [r for r in results if r["keep_majority"]]
    with open(OUTPUT_KEEP, "w") as f:
        for r in keeps:
            f.write(json.dumps(r) + "\n")
    print(f"  keep_majority.jsonl: {len(keeps):,} records")

    # Disagreements (for Stage 3)
    disagrees = [r for r in results if r["judges_disagree"]]
    with open(OUTPUT_DISAGREE, "w") as f:
        for r in disagrees:
            f.write(json.dumps(r) + "\n")
    print(f"  disagreements.jsonl: {len(disagrees):,} records")

    # ── Statistics ──────────────────────────────────────────────────────
    total = len(results)
    t1_total = sum(1 for r in results if r.get("tier") == "tier1")
    t2_total = sum(1 for r in results if r.get("tier") == "tier2")
    stats = {
        "total_candidates": total,
        "tier1_candidates": t1_total,
        "tier2_candidates": t2_total,
        "vote_categories": dict(categories),
        "tier1_kept": len(keeps),
        "tier2_vlm_consensus": categories.get("t2_vlm_consensus", 0),
        "sent_to_tiebreak": len(disagrees),
        "pending_labels": missing_count,
        "kept_per_venue": dict(venue_counts),
        "note": (
            "Tier 1 (CLIP=diagram): majority vote (>=2/3). "
            "Tier 2 (non-diagram, low conf): VLM consensus (both VLMs=diagram) → sent to GPT-5.4."
        ),
    }

    with open(OUTPUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TWO-TIER VOTE SUMMARY")
    print(f"{'='*60}")
    print(f"Total candidates:    {total:>8,d}  (Tier 1: {t1_total:,}, Tier 2: {t2_total:,})")
    print(f"")
    print(f"Tier 1 (CLIP=diagram, majority vote):")
    for cat in ["t1_unanimous_diagram", "t1_majority_diagram", "t1_minority_diagram", "t1_unanimous_reject"]:
        n = categories.get(cat, 0)
        pct = 100 * n / t1_total if t1_total > 0 else 0
        marker = " ← keep" if cat in ("t1_unanimous_diagram", "t1_majority_diagram") else ""
        print(f"  {cat:<26s}: {n:>8,d} ({pct:5.1f}%){marker}")
    print(f"")
    print(f"Tier 2 (non-diagram conf<0.70, VLM consensus):")
    for cat in ["t2_vlm_consensus", "t2_one_vlm_diagram", "t2_no_vlm_diagram"]:
        n = categories.get(cat, 0)
        pct = 100 * n / t2_total if t2_total > 0 else 0
        marker = " → GPT-5.4" if cat == "t2_vlm_consensus" else ""
        print(f"  {cat:<26s}: {n:>8,d} ({pct:5.1f}%){marker}")
    if categories.get("pending", 0) > 0:
        print(f"\n  pending:                    {categories['pending']:>8,d}")
    print(f"")
    print(f"Tier 1 kept (majority):  {len(keeps):>8,d}")
    print(f"Sent to GPT-5.4:        {len(disagrees):>8,d}")
    print(f"  (Tier 1 disagreements + Tier 2 VLM consensus)")
    print(f"")
    print(f"Per-venue Tier 1 keeps:")
    for venue in sorted(venue_counts):
        print(f"  {venue:<10s}: {venue_counts[venue]:>8,d}")

    # ── Cost estimate for Stage 3 ──────────────────────────────────────
    n_tiebreak = len(disagrees)
    if n_tiebreak > 0:
        # GPT-5.4 pricing: ~$2.50/M input, ~$10.00/M output
        est_input = n_tiebreak * 335 * 2.50 / 1_000_000
        est_output = n_tiebreak * 2 * 10.00 / 1_000_000
        est_total = est_input + est_output
        print(f"\nStage 3 cost estimate (GPT-5.4):")
        print(f"  Figures: {n_tiebreak:,}")
        print(f"  Est. cost: ${est_total:,.2f}")


if __name__ == "__main__":
    main()
