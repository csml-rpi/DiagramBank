# A5: Retrieval Ablation — Hierarchical vs Caption-Only

Quantitative comparison of the 3-stage hierarchical retrieval pipeline (title→abstract→caption, 100→30→3) against a flat caption-only baseline on 20 held-out queries.

## Design

- **20 queries**: 4 external papers (from `demo/queries.jsonl`) + 16 leave-one-out papers sampled from the FAISS index with venue stratification (6 ICLR + 6 NeurIPS + 4 ICML)
- **5 conditions**: caption_only, hier_narrow (50/15/3), hier_default (100/30/3), hier_wide (200/50/3), hier_vwide (500/100/3)
- **Primary comparison**: caption_only vs hier_default → 120 results (108 unique after de-dup)
- **Judges**: one author (human, primary 120 only) + GPT 5.4 (all conditions)
- **Metric**: Precision@3 = fraction of top-3 retrieved diagrams judged as relevant

## Relevance Definition

A retrieved diagram is **relevant** if it depicts a sufficiently related architecture, pipeline, or concept that its visual structure could serve as a useful reference or template for designing the query paper's diagram.

## Run

```bash
# 1. Select queries (seed 42)
python select_queries.py

# 2. Run retrieval (needs OPENAI_API_KEY for query embeddings + FIG_RAG_DIR)
python run_retrieval.py

# 3. Prepare the blinded annotation sheet for the human annotator
python prepare_annotation.py

# 4. GPT 5.4 judge (needs Codex OAuth or OPENAI_API_KEY + FIG_RAG_DIR)
python judge_gpt.py [--resume] [--workers 4]

# 5. The annotator fills in annotations_tingwen.jsonl

# 6. Evaluate
python evaluate.py
```

## Files

| File | Description |
|------|-------------|
| `select_queries.py` | Select 20 queries (4 external + 16 leave-one-out) |
| `queries_a5.jsonl` | The 20 query papers |
| `run_retrieval.py` | Run all 5 retrieval conditions |
| `results.jsonl` | 300 retrieval results (20 × 5 × 3) |
| `prepare_annotation.py` | De-dup + randomise → blinded annotation sheet |
| `annotation_sheet.jsonl` | Randomised for human annotation |
| `judge_gpt.py` | GPT 5.4 automated relevance judge |
| `annotations_gpt.jsonl` | GPT judgments |
| `annotations_tingwen.jsonl` | Human judgments |
| `evaluate.py` | Precision@3, Cohen's κ, McNemar's test |
| `results.json` | Final metrics |
| `report.txt` | Human-readable summary |
