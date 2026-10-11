# Experiments

Code behind DiagramBank's quality audits and the early retrieval checks. Each folder
has its own README. The label and log files these scripts read (`*.jsonl`) are not
stored in this repository.

| Folder | What it measures | Earlier name |
|---|---|---|
| `clip_router_audit/` | 500 figures sampled across 4 CLIP classes and 5 confidence bins, labeled by one author and two VLM judges: CLIP precision, recall, and agreement (Cohen's kappa) | A1 |
| `clip_threshold_sweep/` | Precision of "CLIP = diagram and confidence >= t" for t from 0.50 to 0.95 on the same 500 figures, and how many CLIP diagrams each threshold keeps | A2 |
| `retrieval_pilot/` | 20-query comparison of hierarchical and caption-only retrieval, judged by one author and GPT-5.4 | A5 retrieval |
| `retrieval_pooled_recall/` | Pooled retrieval evaluation with LLM relevance labels: recall@K, precision@K, nDCG@K | A5 recall |

The GPT judges use a Codex login (`codex auth login`). The Claude judges call Amazon
Bedrock with your AWS credentials; `BEDROCK_HAIKU_MODEL` selects the inference profile.
Figure images are read from `FIG_RAG_DIR` (default `~/fig-rag-dir`).

The paper's 2,000-query retrieval test, generation study, and 50-record context check
are not in this folder yet.
