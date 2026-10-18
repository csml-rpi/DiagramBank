# Same-paper context swap

Swaps figure context between different figures of the same paper while titles, abstracts, captions, and all 52,052 candidates stay fixed, using the frozen vectors and queries of `../retrieval_ablation`.

A seeded procedure (seed 20260929) forms disjoint same-paper pairs with different context hashes, choosing an unused donor with the nearest log word count; unpaired records keep their contexts. The assignment swaps 35,026 candidate contexts and makes 913 of the 2,000 frozen targets eligible. All conditions use those 913 queries, and intervals use 10,000 paired bootstrap resamples. No model API calls are made.

| Candidate fields (913 queries) | R@1 | R@10 | MRR@10 |
|---|---:|---:|---:|
| Caption only | 72.51 | 85.98 | 77.27 |
| Full without context | 61.99 | 84.67 | 69.84 |
| Full (with context) | 66.48 | 88.61 | 73.98 |
| Full with same-paper context swap | 49.73 | 82.15 | 61.33 |

Full exceeds the swap by 6.46 R@10 points (95% CI 4.60–8.32).

`results/` holds `protocol.json` (assignment rule, counts, input hashes), `within_paper_pairs.jsonl` (the swaps), `per_query_metrics.jsonl`, `within_paper_top10.jsonl`, and `results.json`.

```bash
export FIG_RAG_DIR=/path/to/downloaded/release   # provides faiss/research.db
python run_swap.py --output-dir results-rerun
```

`run_swap.py` needs the full embedding cache in `../retrieval_ablation/embeddings` (see that README). The script hash recorded in `protocol.json` refers to the original run; this copy reads the same inputs from repository paths and omits an offline check of an earlier retrieval demo that does not affect the results.
