#!/usr/bin/env python3
"""
Step 4: Compute Recall@K, Precision@K, and nDCG@K from consensus labels.

Loads ranked_lists.jsonl (rank-ordered results per condition) and
consensus.jsonl (ground truth scores), then computes metrics at
both strict (>=2) and lenient (>=1) thresholds.

Outputs:
  metrics.json       — structured results
  metrics_table.tex  — LaTeX table for rebuttal

Usage:
    python compute_recall.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RANKED_PATH = SCRIPT_DIR / "ranked_lists.jsonl"
CONSENSUS_PATH = SCRIPT_DIR / "consensus.jsonl"
METRICS_PATH = SCRIPT_DIR / "metrics.json"
TEX_PATH = SCRIPT_DIR / "metrics_table.tex"

K_VALUES = [3, 5, 10, 20]
CONDITIONS = ["hier_default", "caption_only"]
THRESHOLDS = {"strict": 2, "lenient": 1}


def dcg(scores: list[float], k: int) -> float:
    """Discounted cumulative gain at rank k."""
    val = 0.0
    for i, s in enumerate(scores[:k]):
        val += s / math.log2(i + 2)  # i+2 because log2(1)=0
    return val


def ndcg(scores: list[float], ideal_scores: list[float], k: int) -> float:
    """Normalized DCG at rank k."""
    idcg = dcg(sorted(ideal_scores, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg(scores, k) / idcg


def main():
    # Load consensus ground truth: judgment_key -> consensus_score
    gt = {}
    with open(CONSENSUS_PATH) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                gt[rec["judgment_key"]] = rec["consensus_score"]

    # Load ranked lists, organized by (query_name, condition) -> [(rank, judgment_key)]
    ranked = defaultdict(list)
    with open(RANKED_PATH) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                key = f"{rec['query_name']}|{rec['retrieved_platform_id']}|{rec['retrieved_figure_number']}"
                ranked[(rec["query_name"], rec["condition"])].append(
                    (rec["rank"], key)
                )

    # Sort each list by rank
    for k in ranked:
        ranked[k].sort(key=lambda x: x[0])

    # Get all query names
    query_names = sorted(set(qn for qn, _ in ranked.keys()))
    print(f"Queries: {len(query_names)}")
    print(f"Conditions: {CONDITIONS}")
    print(f"K values: {K_VALUES}")
    print(f"Ground truth items: {len(gt)}")

    # ── Compute metrics ────────────────────────────────────────────────
    metrics = {}

    for threshold_name, threshold_val in THRESHOLDS.items():
        metrics[threshold_name] = {}

        # For each query, define the relevant set (from the full pool)
        query_relevant = defaultdict(set)  # query_name -> set of relevant judgment_keys
        query_scores = defaultdict(dict)   # query_name -> {judgment_key: consensus_score}
        for jk, score in gt.items():
            qn = jk.split("|")[0]
            query_scores[qn][jk] = score
            if score >= threshold_val:
                query_relevant[qn].add(jk)

        for condition in CONDITIONS:
            cond_metrics = {}
            for K in K_VALUES:
                recall_vals = []
                precision_vals = []
                ndcg_vals = []

                for qn in query_names:
                    rel_set = query_relevant[qn]
                    total_relevant = len(rel_set)

                    rl = ranked.get((qn, condition), [])
                    top_k_keys = [jk for _, jk in rl[:K]]

                    # Recall@K
                    if total_relevant > 0:
                        hits = sum(1 for jk in top_k_keys if jk in rel_set)
                        recall_vals.append(hits / total_relevant)
                    # else: skip this query (undefined recall)

                    # Precision@K
                    if top_k_keys:
                        hits = sum(1 for jk in top_k_keys if jk in rel_set)
                        precision_vals.append(hits / len(top_k_keys))

                    # nDCG@K (uses graded scores, not binary)
                    all_scores = query_scores[qn]
                    retrieved_scores = [all_scores.get(jk, 0) for _, jk in rl[:K]]
                    ideal_scores = list(all_scores.values())
                    if sum(ideal_scores) > 0:
                        ndcg_vals.append(ndcg(retrieved_scores, ideal_scores, K))

                n_queries_with_relevant = len(recall_vals)
                cond_metrics[f"recall@{K}"] = {
                    "mean": sum(recall_vals) / len(recall_vals) if recall_vals else 0,
                    "n_queries": n_queries_with_relevant,
                }
                cond_metrics[f"precision@{K}"] = {
                    "mean": sum(precision_vals) / len(precision_vals) if precision_vals else 0,
                }
                cond_metrics[f"ndcg@{K}"] = {
                    "mean": sum(ndcg_vals) / len(ndcg_vals) if ndcg_vals else 0,
                }

            metrics[threshold_name][condition] = cond_metrics

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

    # ── Print summary ──────────────────────────────────────────────────
    for threshold_name in THRESHOLDS:
        print(f"\n{'='*60}")
        print(f"Threshold: {threshold_name} (score >= {THRESHOLDS[threshold_name]})")
        print(f"{'='*60}")

        header = f"{'Metric':<18}"
        for cond in CONDITIONS:
            header += f"  {cond:>14}"
        print(header)
        print("-" * len(header))

        for K in K_VALUES:
            for metric_type in ("recall", "precision", "ndcg"):
                row = f"{metric_type}@{K:<13}"
                for cond in CONDITIONS:
                    val = metrics[threshold_name][cond][f"{metric_type}@{K}"]["mean"]
                    row += f"  {val:>14.3f}"
                print(row)
            print()

    # ── Generate LaTeX table ───────────────────────────────────────────
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Retrieval performance under multi-VLM consensus ground truth. "
        r"Strict: score $\geq 2$; Lenient: score $\geq 1$.}",
        r"\label{tab:recall}",
        r"\small",
        r"\begin{tabular}{ll" + "r" * len(CONDITIONS) + "}",
        r"\toprule",
    ]

    # Header row
    header = r"Threshold & Metric"
    for cond in CONDITIONS:
        nice = "Hierarchical" if "hier" in cond else "Caption-only"
        header += f" & {nice}"
    header += r" \\"
    tex_lines.append(header)
    tex_lines.append(r"\midrule")

    for ti, (threshold_name, threshold_val) in enumerate(THRESHOLDS.items()):
        label = f"Strict ($\\geq {threshold_val}$)" if threshold_name == "strict" else f"Lenient ($\\geq {threshold_val}$)"

        first = True
        for K in K_VALUES:
            row = f"{label if first else ''} & Recall@{K}"
            for cond in CONDITIONS:
                val = metrics[threshold_name][cond][f"recall@{K}"]["mean"]
                row += f" & {val:.3f}"
            row += r" \\"
            tex_lines.append(row)
            first = False

        if ti == 0:
            tex_lines.append(r"\midrule")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(tex_lines)
    with open(TEX_PATH, "w") as f:
        f.write(tex + "\n")
    print(f"\nLaTeX table saved to {TEX_PATH}")

    # ── Per-query breakdown (for debugging) ────────────────────────────
    print(f"\n{'='*60}")
    print("PER-QUERY RECALL@10 (lenient)")
    print(f"{'='*60}")
    threshold_val = THRESHOLDS["lenient"]
    query_relevant = defaultdict(set)
    for jk, score in gt.items():
        qn = jk.split("|")[0]
        if score >= threshold_val:
            query_relevant[qn].add(jk)

    for qn in query_names:
        rel = len(query_relevant[qn])
        if rel == 0:
            print(f"  {qn:<30s}  relevant=0 (skipped)")
            continue
        row = f"  {qn:<30s}  relevant={rel:<3d}"
        for cond in CONDITIONS:
            rl = ranked.get((qn, cond), [])
            top_10 = [jk for _, jk in rl[:10]]
            hits = sum(1 for jk in top_10 if jk in query_relevant[qn])
            row += f"  {cond}={hits}/{rel}"
        print(row)


if __name__ == "__main__":
    main()
