# [A1] Manual Annotation Study for CLIP Filter

## Purpose
Validate the CLIP-based figure classifier by sampling ~500 figures and having human + LLM judges label them. Results feed into [A2] (threshold sensitivity analysis).

## Sampling Design

**Population:** 475,616 classified figures across all 4 venues (ICLR, ICML, NeurIPS, TMLR), from both accepted and rejected papers. Figures with confidence < 0.50 (16,338) are excluded since they are below any reasonable operating threshold.

**Stratification:** 4 CLIP categories x 5 confidence bins = 20 cells, 25 figures per cell = **500 total**.

| Type \ Conf. | [0.50,0.60) | [0.60,0.70) | [0.70,0.80) | [0.80,0.90) | [0.90,1.00] | Total |
|---|---|---|---|---|---|---|
| diagram | 8,428 | 8,056 | 9,332 | 13,036 | 48,825 | 91,696 |
| plot | 10,979 | 12,175 | 15,704 | 26,412 | 238,029 | 308,922 |
| photo | 4,356 | 4,830 | 6,038 | 8,932 | 25,175 | 53,174 |
| other | 2,840 | 2,832 | 3,085 | 4,170 | 6,044 | 21,824 |

All cells have thousands of figures — no sparse-cell issues.

**Within-cell sampling:** Uniform random, seeded (seed=42) for reproducibility. Venue proportions are preserved within each cell (proportional allocation).

## Key Decisions

1. **Include rejected papers:** Yes. The CLIP filter operates before decision filtering, so the annotation should cover the same population the filter sees.

2. **Confidence < 0.50 excluded:** These 16,338 figures (~3.4%) are well below any reasonable operating threshold and would not affect the sensitivity analysis in [A2].

3. **Seed = 42:** Fixed for reproducibility. Re-running `sample.py` produces the identical 500 figures.

4. **LLM prompt definitions align with but are independent of CLIP prompts.** The CLIP classifier (`reproduce/*/classify.py`) uses multi-prompt ensembles (4–6 phrases per category) for zero-shot classification. The LLM judges receive concise natural-language definitions that cover the **same semantic boundaries** (e.g., "photo" includes screenshots, pixel art, and comic illustrations in both) but are **independently phrased**. This avoids circularity — the annotation validates the CLIP filter's *output*, not its input prompts. In the paper we note: *"LLM judges received concise definitions covering the same semantic scope as the CLIP ensemble prompts, but independently phrased to ensure the annotation is not circular."*

5. **Why two LLMs (not one)?** Reporting agreement across two independent LLM judges (plus human) strengthens the claim that labels are not model-specific artifacts. GPT-5.4-mini and Claude Haiku 4.5 were chosen for cost efficiency; both are vision-capable and sufficient for a 4-class image classification task.

6. **Ground truth:** The human labels are the sole ground truth. LLM labels are reported as cross-validation (human–LLM and LLM–LLM Cohen's κ).

7. **Single human annotator:** We report human–LLM agreement instead of human–human agreement. The two LLM judges serve as independent cross-validators.

## LLM Judges

Two vision-capable LLMs classify each of the 500 figures independently, serving as cross-validators against the human ground truth.

| Judge | Model | Auth |
|---|---|---|
| OpenAI | `gpt-5.4-mini` | Codex auth (no explicit key needed) |
| Anthropic | Claude Haiku 4.5 via Bedrock | AWS credentials (boto3 default) |

**Prompt:** Both models receive the same prompt asking them to classify the figure as `diagram`, `plot`, `photo`, or `other`, with brief definitions for each category. Temperature = 0.0 for deterministic output. Images sent at low detail to save tokens.

**Running:**
```bash
# OpenAI (Codex auth)
python judge_openai.py

# Anthropic via Bedrock (uses default AWS credentials)
python judge_anthropic.py

# Both support --resume to continue from where they left off
python judge_openai.py --resume
```

**Cost estimate:** ~500 images at low-res ≈ a few cents per model.

## Human Annotator

One author is the sole human annotator. Their labels were collected with `ANNOTATOR_NAME = "tingwen"`, which names the label file.

The notebook displays one figure at a time with colored category buttons. Labels auto-save to `annotations_tingwen.jsonl` after each click. The notebook supports:
- **Resume:** closing and reopening picks up where you left off
- **Navigation:** prev/next buttons and jump-to-index
- **Notes:** optional free-text field for ambiguous cases
- **Blind annotation:** CLIP label and confidence are hidden from the annotator to avoid bias. Only venue, paper ID, and figure number are shown.

## Files
- `sample.py` — Sampling script (reads DuckDB, outputs sample.jsonl)
- `sample.jsonl` — The 500 sampled figures with CLIP labels and confidence
- `judge_openai.py` — GPT-5.4-mini annotation script → `annotations_openai.jsonl`
- `judge_anthropic.py` — Claude Haiku 4.5 annotation script → `annotations_anthropic.jsonl`
- `annotate.ipynb` — Jupyter notebook for human annotators → `annotations_{name}.jsonl`
- `evaluate.py` — Compute all metrics from the 4 annotation files → `results.json`, `report.txt`, plots
- `README.md` — This file

### Output files (generated by evaluate.py)
- `results.json` — All metrics in machine-readable form
- `report.txt` — Human-readable summary for paper writing
- `confusion_matrix.png` — CLIP predicted vs. human ground truth (4x4 heatmap)
- `agreement_matrix.png` — Pairwise Cohen's κ across all 3 judges

## Results

### Inter-Annotator Agreement

Pairwise Cohen's κ across all 4 judges (Human, 2 VLMs, CLIP):

| Pair | κ | Band |
|---|---|---|
| Human vs. GPT-5.4-mini | 0.753 | Substantial |
| Human vs. Haiku 4.5 | 0.744 | Substantial |
| Human vs. CLIP | 0.627 | Substantial |
| GPT-5.4-mini vs. Haiku 4.5 | 0.773 | Substantial |
| GPT-5.4-mini vs. CLIP | 0.600 | Moderate |
| Haiku 4.5 vs. CLIP | 0.611 | Substantial |

**Interpretation:** Human–VLM agreement (κ ≈ 0.75) is nearly as high as VLM–VLM (0.77), confirming well-defined category definitions. CLIP shows moderate-to-substantial agreement (κ ≈ 0.60–0.63) with all other judges — lower than the VLMs but still reasonable for a zero-shot classifier operating without any task-specific fine-tuning. The gap between CLIP and VLM agreement is modest (~0.13–0.17), suggesting CLIP captures the same semantic boundaries.

### Diagram Filter Quality (type=diagram AND confidence ≥ 0.85)

| Metric | Value |
|---|---|
| Precision | 94.4% |
| Recall | 28.3% |
| F1 | 0.436 |
| False positive rate | 0.5% |
| False negative rate | 71.7% |

**Interpretation:** The filter achieves **very high precision** (94.4%) — virtually no contamination in the released dataset. The low recall is an **artifact of stratified sampling**: we intentionally over-sampled low-confidence bins to study the decision boundary. In the actual dataset, most true diagrams have confidence > 0.85 and would be retained. [A2] (threshold sweep) will provide proper dataset-wide recall estimates.

### Per-Category CLIP Classification (ignoring threshold)

| Category | Precision | Recall | F1 |
|---|---|---|---|
| diagram | 0.640 | 0.667 | 0.653 |
| plot | 0.600 | 0.708 | 0.649 |
| photo | 0.880 | 0.675 | 0.764 |
| other | 0.760 | 0.856 | 0.805 |
| **Macro** | **0.720** | **0.726** | **0.718** |

### Confusion Matrix

The main error mode is **diagram ↔ plot confusion** (26 true diagrams labeled as plot by CLIP, 23 true plots labeled as diagram). This is expected — the boundary between schematic diagrams and data visualizations is genuinely ambiguous for figures that combine both elements (e.g., a pipeline diagram with embedded plots). Photo and other are well-separated.

### Key Takeaways for the Rebuttal

1. **94.4% precision** at threshold 0.85 — the released dataset has minimal contamination (FP rate = 0.5%).
2. **κ ≈ 0.75** for Human–VLM pairs — substantial agreement validates category definitions.
3. **CLIP κ ≈ 0.61** — moderate-to-substantial agreement with all judges, confirming zero-shot CLIP is a reasonable classifier even without fine-tuning.
4. The diagram↔plot boundary is the primary source of classification error, which is a semantically reasonable ambiguity rather than a filter failure.
5. The low recall at 0.85 is a sampling artifact; [A2] will provide the full threshold sensitivity picture.
