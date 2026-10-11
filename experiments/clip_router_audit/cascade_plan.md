# Two-Tier Filtering Cascade Plan

## Goal
High recall (don't miss diagrams) + high precision (don't include non-diagrams).
Replace the single CLIP threshold (τ=0.85) with a multi-judge, two-tier cascade.

## Key Insight
A single-tier cascade (only CLIP=diagram figures) caps recall at 88.9%.
True diagrams CLIP misclassified as plot/photo are never reconsidered.
A second tier targeting low-confidence non-diagram predictions recovers these.

## Full Dataset Statistics (queried from DuckDB)

| CLIP class | ICLR | ICML | NeurIPS | TMLR | Total |
|---|---|---|---|---|---|
| diagram | 48,103 | 12,267 | 21,569 | 9,757 | 91,696 (19.3%) |
| plot | 160,983 | 46,558 | 64,307 | 37,074 | 308,922 (65.0%) |
| photo | 28,938 | 5,724 | 13,584 | 4,928 | 53,174 (11.2%) |
| other | 15,055 | 1,919 | 3,254 | 1,596 | 21,824 (4.6%) |
| **Total** | **253,079** | **66,468** | **102,714** | **53,355** | **475,616** |

## Pipeline

```
475,616 total figures
    │
    ▼ Stage 1: CLIP (already computed, free)
    │
    ├── Tier 1: CLIP=diagram (any confidence)
    │   91,696 figures
    │
    └── Tier 2: non-diagram, confidence < 0.70
        50,331 figures (plot: 28,777, photo: 13,029, other: 8,525)
    │
    ▼ Stage 2: Haiku 4.5 + GPT-5.4-mini (cheap VLMs)
    │   Process all 142,027 candidates in parallel
    │   Tier 1 rule: majority vote (≥2 of 3 say diagram)
    │   Tier 2 rule: VLM consensus (both VLMs must say diagram)
    │
    ▼ Stage 3: GPT-5.4 (expensive VLM)
    │   Input: Tier 1 disagreements (~20% of 91K ≈ 18K)
    │        + Tier 2 VLM-consensus matches (expected small)
    │   Rule: GPT-5.4 as tiebreaker/verifier
    │
    ▼ Stage 4: Human spot-check
    │
  Final dataset (high precision + high recall)
```

## Cost Estimate

| Stage | Model | Figures | Est. Cost |
|---|---|---|---|
| 2 (Tier 1) | Haiku 4.5 | 91,696 | ~$83 |
| 2 (Tier 1) | GPT-5.4-mini | 91,696 | ~$15 |
| 2 (Tier 2) | Haiku 4.5 | 50,331 | ~$45 |
| 2 (Tier 2) | GPT-5.4-mini | 50,331 | ~$8 |
| 3 | GPT-5.4 | ~18K + small | ~$51 |
| **Total** | | | **~$202** |

## Tier 2 Breakdown (non-diagram, confidence < 0.70)

| CLIP class | ICLR | ICML | NeurIPS | TMLR | Total |
|---|---|---|---|---|---|
| plot | 15,403 | 3,824 | 6,620 | 2,930 | 28,777 |
| photo | 7,333 | 1,367 | 3,154 | 1,175 | 13,029 |
| other | 5,318 | 894 | 1,619 | 694 | 8,525 |
| **Total** | **28,054** | **6,085** | **11,393** | **4,799** | **50,331** |

Why confidence < 0.70? All 10 CLIP-missed diagrams in A1 had top-1 confidence < 0.68.

## Evidence from A1 Experiment (500 figures, 90 true diagrams, post-reclassification)

| Rule | TP | FP | FN | Prec | Recall | F1 | Included |
|---|---|---|---|---|---|---|---|
| Old: CLIP τ≥0.85 | 34 | 2 | 56 | 94.4% | 37.8% | 0.540 | 36 |
| CLIP=diag (any conf) | 80 | 45 | 10 | 64.0% | 88.9% | 0.744 | 125 |
| CLIP + ≥1 VLM=diag | 78 | 16 | 12 | 83.0% | 86.7% | 0.848 | 94 |
| CLIP + both VLMs=diag | 75 | 8 | 15 | 90.4% | 83.3% | 0.867 | 83 |
| **≥2 of 3 say diag (majority)** | **88** | **34** | **2** | **72.1%** | **97.8%** | **0.830** | **122** |
| ≥1 of 3 say diag (union) | 90 | 79 | 0 | 53.3% | 100% | 0.695 | 169 |

### Tier breakdown of majority vote (TP=88, FP=34):
- **Tier 1 contribution** (CLIP=diagram + ≥1 VLM): TP=78, FP=16, precision=83.0%
- **Tier 2 contribution** (CLIP≠diagram, both VLMs=diagram): TP=10, FP=18, precision=35.7%
  - All 10 CLIP-missed diagrams were recovered
  - 18 FP → mitigated by GPT-5.4 verification in Stage 3
- **2 FN** (both from Tier 1): CLIP=diagram but both VLMs=other
  - z8sxoCYgmd Fig38 (CLIP conf 0.568)
  - 3zWSmhNSa7 Fig2 (CLIP conf 0.938)

## Precision Analysis

Tier 1 (majority vote): 83.0% precision before tiebreak.
Stage 3 GPT-5.4 further raises precision on disagreement cases.

Tier 2 (VLM consensus alone): 35.7% precision — too low.
**Solution**: GPT-5.4 verification. Only include Tier 2 figures where
BOTH cheap VLMs AND GPT-5.4 all say diagram (3 VLMs overriding CLIP).
This should substantially raise Tier 2 precision.

## Analysis of CLIP-Missed Diagrams

10 true diagrams where CLIP predicted non-diagram. All had confidence < 0.68:

| Paper | CLIP top-1 | Conf | diagram softmax | plot softmax |
|---|---|---|---|---|
| 9UExQpH078 | plot | 0.535 | **0.664** | 0.336 |
| P6uZ7agiyCT | plot | 0.504 | **0.674** | 0.325 |
| qE3knKF1rz | plot | 0.617 | **0.538** | 0.450 |
| fM9xTkpAdu | plot | 0.587 | 0.485 | 0.514 |
| V2bw5SmpF6 | plot | 0.579 | 0.475 | 0.509 |
| lEXrEcrbmV | plot | 0.532 | 0.465 | 0.532 |
| 1qP3lsatCR | plot | 0.538 | 0.458 | 0.538 |
| 2v91xhNdsz | plot | 0.618 | 0.370 | 0.615 |
| zVrQeoPIoQ | plot | 0.671 | 0.243 | 0.717 |
| iooQFm2qHf | photo | 0.628 | 0.492 | 0.038 |

Key: 9/10 are plot, 1/10 is photo. 9/10 have diagram softmax ≥ 0.37.
All have top-1 confidence < 0.68 → safely captured by conf < 0.70 cutoff.

## Why Two Tiers Are Better Than τ=0.85

The old approach uses a single confidence threshold: high precision but low recall.
The two-tier cascade separates the two goals:
- **Recall** from Tier 1 (broad CLIP=diagram) + Tier 2 (VLM recovery of CLIP misses)
- **Precision** from multi-judge agreement (majority vote + VLM consensus + GPT-5.4)
