"""Same-paper context swap control using the frozen retrieval-ablation cache.

Requires the embedding arrays of ../retrieval_ablation/embeddings (hosted with the
dataset) and the release database at $FIG_RAG_DIR/faiss/research.db. No model
API calls are made.

Usage: python run_swap.py --output-dir <new directory>
"""
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import gc
import hashlib
import importlib.util
import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

import faiss
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "experiments/retrieval_ablation"
FIG_RAG_DIR = Path(os.environ.get("FIG_RAG_DIR", Path.home() / "fig-rag-dir"))
OUT = None  # set from --output-dir
SEED = 20260929


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def within_paper_swaps(candidates, seed):
    """Make disjoint pairs with different context text and nearby word counts."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, row in enumerate(candidates):
        groups[row["platform_id"]].append(i)
    permutation = np.arange(len(candidates))
    for indices in groups.values():
        available = set(indices)
        for first in rng.permutation(indices):
            first = int(first)
            if first not in available:
                continue
            options = [i for i in sorted(available) if candidates[i]["context_sha256"] != candidates[first]["context_sha256"]]
            if not options:
                continue
            costs = [abs(np.log((candidates[i]["context_words"] + 1) / (candidates[first]["context_words"] + 1))) for i in options]
            minimum = min(costs)
            closest = [i for i, cost in zip(options, costs) if abs(cost - minimum) < 1e-12]
            second = int(rng.choice(closest))
            permutation[first], permutation[second] = second, first
            available.remove(first)
            available.remove(second)
    assert np.array_equal(np.sort(permutation), np.arange(len(candidates)))
    for i, j in enumerate(permutation):
        assert candidates[i]["platform_id"] == candidates[int(j)]["platform_id"]
        if i != j:
            assert candidates[i]["context_sha256"] != candidates[int(j)]["context_sha256"]
    return permutation


def main():
    if OUT.exists():
        raise FileExistsError("Keep prior runs; select a new output directory")
    OUT.mkdir(parents=True)
    faiss.omp_set_num_threads(4)
    evaluation = load_module("frozen_e4_evaluation", BASE / "evaluate_retrieval.py")
    manifest = json.loads((BASE / "embeddings/embedding_manifest.json").read_text())
    for name, details in manifest["files"].items():
        assert evaluation.sha256_file(BASE / "embeddings" / name) == details["sha256"], name
    assert evaluation.sha256_file(FIG_RAG_DIR / "faiss/research.db") == manifest["database_sha256"]
    assert evaluation.sha256_file(BASE / "embeddings/candidate_manifest.jsonl") == manifest["candidate_manifest_sha256"]
    assert evaluation.sha256_file(BASE / "validation_round3/accepted_queries.jsonl") == manifest["query_file_sha256"]
    candidates = rows(BASE / "embeddings/candidate_manifest.jsonl")
    target_map = {row["query_id"]: row for row in rows(BASE / "data/target_manifest.jsonl")}
    order = json.loads((BASE / "embeddings/query_order.json").read_text())
    assert set(order) == set(target_map) and len(order) == len(set(order))
    targets = [target_map[q] for q in order]
    queries = evaluation.normalize_rows(np.asarray(np.load(BASE / "embeddings/queries.f32.npy"), dtype=np.float32))
    fields = evaluation.load_field_embeddings(BASE / "embeddings")
    saved = defaultdict(dict)
    for row in rows(BASE / "results/per_query_metrics.jsonl"):
        saved[row["condition"]][row["query_id"]] = row
    permutation = within_paper_swaps(candidates, SEED)
    key_to_index = {(r["platform_id"], str(r["figure_number"])): i for i, r in enumerate(candidates)}
    eligible = [i for i, t in enumerate(targets) if permutation[key_to_index[(t['platform_id'], str(t['figure_number']))]] != key_to_index[(t['platform_id'], str(t['figure_number']))]]
    mappings = [{"candidate_index": i, "donor_index": int(j), "platform_id": row["platform_id"], "figure_number": row["figure_number"], "donor_figure_number": candidates[int(j)]["figure_number"], "original_words": row["context_words"], "donor_words": candidates[int(j)]["context_words"]} for i, (row, j) in enumerate(zip(candidates, permutation)) if i != j]
    with (OUT / "within_paper_pairs.jsonl").open("w") as handle:
        for row in mappings:
            handle.write(json.dumps(row) + "\n")
    protocol = {"seed": SEED, "candidate_records": len(candidates), "all_frozen_queries": len(targets), "eligible_queries": len(eligible), "swapped_candidates": len(mappings),
                "assignment": "Disjoint same-paper pairs with different context SHA-256; choose nearest log word count among unused donors, seeded tie breaking. Identity assignment for unpaired records. All 52,052 candidates remain in every condition.",
                "eligibility": "Target record receives a different same-paper context. Eligibility and assignment are fixed before scoring.",
                "conditions": ["Full", "Full-X", "C", "Full-within-paper-X"], "bootstrap_replicates": 10000,
                "scope": "Frozen target-derived known-item queries; multi-figure contexts may share passages. This control does not validate source-PDF extraction.",
                "source_sha256": {"script": evaluation.sha256_file(Path(__file__)), "embedding_manifest": evaluation.sha256_file(BASE / "embeddings/embedding_manifest.json"), "targets": evaluation.sha256_file(BASE / "data/target_manifest.jsonl"), "saved_per_query_results": evaluation.sha256_file(BASE / "results/per_query_metrics.jsonl"), "assignments": evaluation.sha256_file(OUT / "within_paper_pairs.jsonl")}}
    save_json("protocol.json", protocol)
    print(f"Frozen within-paper assignments: {len(eligible)} eligible queries, {len(mappings)} swapped candidate records", flush=True)

    # Check the baseline on 20 fixed queries before reusing its archived rows.
    check_ids = np.sort(np.random.default_rng(SEED + 1).choice(len(targets), 20, replace=False))
    identity = np.arange(len(candidates))
    combined = evaluation.combine_fields(fields, ("caption", "title", "abstract", "context"), identity, identity)
    _, retrieved = evaluation.search_condition(combined, queries[check_ids], 10, 32)
    checks = evaluation.per_query_metrics(retrieved, [targets[i] for i in check_ids], candidates)
    assert all(row["exact_rank_at_10"] == saved["Full"][row["query_id"]]["exact_rank_at_10"] for row in checks)
    del combined
    gc.collect()
    print("Baseline replay matches all 20 archived exact ranks", flush=True)
    combined = evaluation.combine_fields(fields, ("caption", "title", "abstract", "context_shuffled"), permutation, identity)
    _, retrieved = evaluation.search_condition(combined, queries[eligible], 10, 32)
    selected_targets = [targets[i] for i in eligible]
    new_rows = evaluation.per_query_metrics(retrieved, selected_targets, candidates)
    conditions = {name: [saved[name][target['query_id']] for target in selected_targets] for name in ("Full", "Full-X", "C")}
    conditions["Full-within-paper-X"] = new_rows
    with (OUT / "per_query_metrics.jsonl").open("w") as handle:
        for name, records in conditions.items():
            for row in records:
                handle.write(json.dumps({**row, "condition": name}) + "\n")
    with (OUT / "within_paper_top10.jsonl").open("w") as handle:
        for target, indices in zip(selected_targets, retrieved):
            handle.write(json.dumps({"query_id": target['query_id'], "ranked_keys": [f"{candidates[i]['platform_id']}|{candidates[i]['figure_number']}" for i in indices]}) + "\n")
    comparisons = []
    rng = np.random.default_rng(SEED + 2)
    for baseline in ("Full-within-paper-X", "Full-X", "C"):
        for metric in ("exact_r1", "exact_r10", "exact_mrr10"):
            delta = evaluation.paired_bootstrap(np.array([r[metric] for r in conditions['Full']]), np.array([r[metric] for r in conditions[baseline]]), rng, 10000)
            comparisons.append({"condition": "Full", "baseline": baseline, "metric": metric, "mean_difference": delta[0], "ci95": list(delta[1:])})
    result = {"eligible_queries": len(eligible), "excluded_queries": len(targets)-len(eligible), "aggregates": {name: evaluation.aggregate(records) for name, records in conditions.items()}, "paired_comparisons": comparisons, "baseline_replay_exact_rank_matches": 20, "model_api_calls": 0}
    save_json("results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Same-paper context swap control")
    parser.add_argument("--output-dir", type=Path, required=True)
    OUT = parser.parse_args().output_dir.resolve()
    main()
