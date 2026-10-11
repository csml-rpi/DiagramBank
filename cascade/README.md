# Two-Tier Filtering Cascade

Replaces the single CLIP confidence threshold (τ=0.85) with a multi-judge,
two-tier cascade that separates recall from precision.

## Pipeline

```
475,616 total figures
    │
    ▼ Stage 1: CLIP (already computed, free)
    │
    ├── Tier 1: CLIP=diagram, any conf (91,696)
    │   Rule: majority vote (≥2/3 say diagram)
    │
    └── Tier 2: non-diagram, conf < 0.70 (50,331)
        Rule: VLM consensus (both VLMs say diagram)
    │
    ▼ Stage 2: Haiku 4.5 + GPT-5.4-mini (~$151)
    │   Run on all 142,027 candidates in parallel
    │
    ▼ Stage 3: GPT-5.4 tiebreaker (~$51)
    │   Tier 1 disagreements + Tier 2 VLM-consensus matches
    │
    ▼ Stage 4: Finalize + human spot-check
```

## A1 Evidence (`experiments/clip_router_audit/`, 500 figures, 90 true diagrams)

| Rule | TP | FP | FN | Prec | Recall | F1 |
|---|---|---|---|---|---|---|
| Old: CLIP τ≥0.85 | 34 | 2 | 56 | 94.4% | 37.8% | 0.540 |
| Tier 1 majority | 78 | 16 | 12 | 83.0% | 86.7% | 0.848 |
| + Tier 2 VLM consensus | 88 | 34 | 2 | 72.1% | 97.8% | 0.830 |
| + GPT-5.4 verification | TBD | TBD | TBD | TBD | TBD | TBD |

## Usage

```bash
# Dry run (cost estimates only)
bash run_cascade.sh --dry-run

# Full run
bash run_cascade.sh

# Or run stages individually:
python extract_candidates.py                       # Stage 1 (both tiers)
python judge_haiku.py --resume --workers 8         # Stage 2a
python judge_gpt_mini.py --resume --workers 8      # Stage 2b (parallel with 2a)
python vote.py                                     # Merge votes (tier-aware)
python judge_gpt_tiebreak.py --resume --workers 4  # Stage 3
python finalize.py                                 # Stage 4
python finalize.py --update-db                     # Also update DuckDB databases
```

## Cost Estimate (~$202 total)

| Stage | Model | Figures | Est. Cost |
|---|---|---|---|
| 2 | Haiku 4.5 | 142,027 | ~$128 |
| 2 | GPT-5.4-mini | 142,027 | ~$23 |
| 3 | GPT-5.4 | ~18K+ | ~$51 |

## Tier 2 Breakdown (non-diagram, conf < 0.70)

| CLIP class | Count | Why included |
|---|---|---|
| plot | 28,777 | 9/10 CLIP misses were plot |
| photo | 13,029 | 1/10 CLIP misses was photo |
| other | 8,525 | Catch-all for remaining low-conf |
| **Total** | **50,331** | |
# Replaying the final release

`finalize.py` reads the saved voting and GPT confirmation files. To preserve
historical outputs and verify membership against a release database, run:

```sh
python finalize.py --output-dir /path/to/new-run \
  --verify-release-db /path/to/faiss/research.db
```

The comparison database is opened read-only. The replay requires exact record-key
equality and rejects missing confirmation labels or duplicate input keys. Tier 1
unanimous and majority decisions are retained; Tier 1 minority and Tier 2 VLM
consensus decisions require the GPT label `diagram`. The retained-count comparison
uses the same candidate pool and is not a precision or recall estimate. Venue
databases are modified only when `--update-db` is explicitly supplied.
