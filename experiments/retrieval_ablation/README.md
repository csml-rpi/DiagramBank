# Retrieval field ablation

Exact-target retrieval over the 52,052 released records whose title, abstract, caption, and figure context are all non-empty (25,616 papers).

## Protocol

1. `sample_targets.py` freezes 2,000 targets (seed 20260712), at most one per paper, stratified by venue, year, and caption-length bin → `data/target_manifest.jsonl`, `data/sampling_summary.json`.
2. `make_shards.py` splits the targets into 20 shards of 100. An isolated `gpt-5.6-sol` worker (medium reasoning effort) writes one 35–90-word request per target with `paraphrase_prompt.md`.
3. `validate_paraphrases.py` rejects requests that omit or duplicate an identifier, fall outside the length range, name a figure number, contain the exact title, copy a six-token source span, or keep a coined source name or caption acronym. `prepare_rewrites.py` and `merge_paraphrases.py` handle rewrites: the first pass accepted 1,892 requests, 108 were rewritten, and 7 were rewritten again (`validation_round1/`, `validation_final/`, `validation_round3/`). `query_generation_manifest.json` records the model, rounds, and file hashes.
4. The frozen query set is `validation_round3/accepted_queries.jsonl` (2,000 queries).
5. `prepare_embeddings.py` extracts the released title, abstract, and caption vectors and embeds contexts and queries with `text-embedding-3-large` (3,072 dimensions).
6. `evaluate_retrieval.py` scores each candidate by the mean cosine similarity over its active fields and reports R@1/5/10, MRR@10, paper-level recall, leave-one-field-out and matched wrong-field controls, and 10,000-sample paired bootstrap intervals.

## Results

`results/metrics.json` (all conditions and intervals), `results/per_query_metrics.jsonl`, and the retrieval log `results/top10_rankings.jsonl.gz`.

| Candidate fields | R@1 | R@10 | MRR@10 |
|---|---:|---:|---:|
| Caption only | 72.90 | 86.60 | 77.69 |
| Title, abstract, caption | 64.95 | 85.75 | 72.00 |
| Full (with context) | 68.60 | 88.95 | 75.79 |
| Full with matched wrong context | 56.60 | 80.80 | 64.56 |
| Full without title | 71.75 | 90.75 | 78.48 |

## Re-running

The context and query embeddings and the candidate manifest are hosted with the dataset under `supplement/retrieval_ablation/` (hashes in `embeddings/embedding_manifest.json`). Place them in `embeddings/`, rebuild the title, abstract, and caption arrays from the released FAISS indices (no API calls), and evaluate:

```bash
python prepare_embeddings.py --db $FIG_RAG_DIR/faiss/research.db --faiss-dir $FIG_RAG_DIR/faiss \
  --output-dir embeddings --stage extract-released
python evaluate_retrieval.py --embedding-dir embeddings --targets data/target_manifest.jsonl \
  --queries validation_round3/accepted_queries.jsonl --output-dir results-rerun
```
