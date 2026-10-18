#!/usr/bin/env python3
"""Run fixed-query field ablations and compute paired known-item metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np


CONDITIONS = {
    "C": ("caption",),
    "C+T": ("caption", "title"),
    "C+T+A": ("caption", "title", "abstract"),
    "Full": ("caption", "title", "abstract", "context"),
    "Full-T": ("caption", "abstract", "context"),
    "Full-A": ("caption", "title", "context"),
    "Full-C": ("title", "abstract", "context"),
    "Full-X": ("caption", "title", "abstract"),
}


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_text_sha256(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = clean(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("zero vector found")
    return values / norms


def load_field_embeddings(directory: Path) -> dict[str, np.ndarray]:
    paths = {
        "title": directory / "candidate_title.f16.npy",
        "abstract": directory / "candidate_abstract.f16.npy",
        "caption": directory / "candidate_caption.f16.npy",
        "context": directory / "candidate_context.f32.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing candidate embeddings: {missing}")
    output = {field: np.load(path, mmap_mode="r") for field, path in paths.items()}
    for field, values in output.items():
        sample = np.asarray(values[:: max(1, len(values) // 256)], dtype=np.float32)
        norms = np.linalg.norm(sample, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-3):
            raise RuntimeError(f"{field} candidate vectors are not L2 normalized")
    return output


def combine_fields(
    fields: dict[str, np.ndarray],
    names: tuple[str, ...],
    context_permutation: np.ndarray,
    abstract_permutation: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    rows, dimension = fields["caption"].shape
    output = np.zeros((rows, dimension), dtype=np.float32)
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        for name in names:
            if name == "context_shuffled":
                source = fields["context"][context_permutation[start:stop]]
            elif name == "title_shuffled":
                source = fields["title"][abstract_permutation[start:stop]]
            elif name == "abstract_shuffled":
                source = fields["abstract"][abstract_permutation[start:stop]]
            else:
                source = fields[name][start:stop]
            output[start:stop] += np.asarray(source, dtype=np.float32)
    output /= float(len(names))
    return output


def length_bin(word_count: int) -> str:
    if word_count <= 100:
        return "short"
    if word_count <= 250:
        return "medium"
    return "long"


def matched_donor_indices(
    candidates: list[dict],
    length_field: str,
    content_hash_field: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    exact_groups: dict[tuple, list[int]] = defaultdict(list)
    venue_year_groups: dict[tuple, list[int]] = defaultdict(list)
    venue_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(candidates):
        exact_groups[
            (row["venue"], int(row["year"]), length_bin(int(row[length_field])))
        ].append(index)
        venue_year_groups[(row["venue"], int(row["year"]))].append(index)
        venue_groups[row["venue"]].append(index)
    rng = np.random.default_rng(seed)
    donors = np.empty(len(candidates), dtype=np.int64)
    fallback_counts: Counter[str] = Counter()
    global_pool = list(range(len(candidates)))

    def valid(target: dict, donor: dict) -> bool:
        return (
            str(target["platform_id"]) != str(donor["platform_id"])
            and target[content_hash_field] != donor[content_hash_field]
        )

    for index, row in enumerate(candidates):
        pools = (
            (
                "same_venue_year_length",
                exact_groups[(row["venue"], int(row["year"]), length_bin(int(row[length_field])))],
            ),
            ("same_venue_year", venue_year_groups[(row["venue"], int(row["year"]))]),
            ("same_venue", venue_groups[row["venue"]]),
            ("global", global_pool),
        )
        selected = None
        selected_level = None
        for level, pool in pools:
            for _ in range(min(128, max(1, len(pool)))):
                candidate_index = int(pool[int(rng.integers(0, len(pool)))])
                if valid(row, candidates[candidate_index]):
                    selected = candidate_index
                    selected_level = level
                    break
            if selected is None:
                selected = next(
                    (candidate_index for candidate_index in pool if valid(row, candidates[candidate_index])),
                    None,
                )
                if selected is not None:
                    selected_level = level
            if selected is not None:
                break
        if selected is None or selected_level is None:
            raise RuntimeError(f"no valid shuffled donor for candidate {index}")
        donors[index] = selected
        fallback_counts[selected_level] += 1

    for index, donor_index in enumerate(donors):
        if not valid(candidates[index], candidates[int(donor_index)]):
            raise RuntimeError("shuffled control retained same-paper or identical content")
    return donors, dict(sorted(fallback_counts.items()))


def search_condition(
    candidate_vectors: np.ndarray,
    query_vectors: np.ndarray,
    top_k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    index = faiss.IndexFlatIP(candidate_vectors.shape[1])
    index.add(np.ascontiguousarray(candidate_vectors, dtype=np.float32))
    score_batches = []
    index_batches = []
    for start in range(0, len(query_vectors), batch_size):
        scores, indices = index.search(
            np.ascontiguousarray(query_vectors[start : start + batch_size], dtype=np.float32),
            top_k,
        )
        score_batches.append(scores)
        index_batches.append(indices)
    return np.concatenate(score_batches), np.concatenate(index_batches)


def per_query_metrics(
    retrieved: np.ndarray,
    targets: list[dict],
    candidates: list[dict],
) -> list[dict]:
    candidate_keys = [
        (str(row["platform_id"]), str(row["figure_number"])) for row in candidates
    ]
    output = []
    for query_index, indices in enumerate(retrieved):
        target = targets[query_index]
        target_key = (str(target["platform_id"]), str(target["figure_number"]))
        target_paper = str(target["platform_id"])
        keys = [candidate_keys[int(index)] for index in indices]
        exact_rank = next((rank for rank, key in enumerate(keys, start=1) if key == target_key), None)
        paper_rank = next(
            (rank for rank, key in enumerate(keys, start=1) if key[0] == target_paper),
            None,
        )
        output.append(
            {
                "query_id": target["query_id"],
                "platform_id": target_paper,
                "exact_rank_at_10": exact_rank,
                "paper_rank_at_10": paper_rank,
                "exact_r1": float(exact_rank == 1),
                "exact_r5": float(exact_rank is not None and exact_rank <= 5),
                "exact_r10": float(exact_rank is not None),
                "exact_mrr10": 0.0 if exact_rank is None else 1.0 / exact_rank,
                "paper_r1": float(paper_rank == 1),
                "paper_r5": float(paper_rank is not None and paper_rank <= 5),
                "paper_r10": float(paper_rank is not None),
            }
        )
    return output


METRIC_NAMES = (
    "exact_r1",
    "exact_r5",
    "exact_r10",
    "exact_mrr10",
    "paper_r1",
    "paper_r5",
    "paper_r10",
)


def aggregate(rows: list[dict]) -> dict[str, float]:
    return {name: float(np.mean([row[name] for row in rows])) for name in METRIC_NAMES}


def paired_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float, float]:
    differences = values_a - values_b
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.integers(0, len(differences), size=len(differences))
        draws[index] = differences[sampled].mean()
    return (
        float(differences.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    if args.top_k != 10:
        raise ValueError("metrics currently expect --top-k 10")

    candidates = load_jsonl(args.embedding_dir / "candidate_manifest.jsonl")
    embedding_manifest = json.loads(
        (args.embedding_dir / "embedding_manifest.json").read_text(encoding="utf-8")
    )
    candidate_manifest_path = args.embedding_dir / "candidate_manifest.jsonl"
    if embedding_manifest.get("candidate_manifest_sha256") != sha256_file(candidate_manifest_path):
        raise RuntimeError("candidate manifest hash does not match embedding manifest")
    required_embedding_files = (
        "candidate_title.f16.npy",
        "candidate_abstract.f16.npy",
        "candidate_caption.f16.npy",
        "candidate_context.f32.npy",
        "queries.f32.npy",
    )
    for filename in required_embedding_files:
        expected = embedding_manifest.get("files", {}).get(filename, {}).get("sha256")
        path = args.embedding_dir / filename
        if expected is None or not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"unverified embedding cache: {path}")
    target_rows = load_jsonl(args.targets)
    targets_by_id = {row["query_id"]: row for row in target_rows}
    if len(targets_by_id) != len(target_rows):
        raise RuntimeError("target manifest contains duplicate query IDs")
    queries = sorted(load_jsonl(args.queries), key=lambda row: row["query_id"])
    query_ids = [row["query_id"] for row in queries]
    if len(query_ids) != len(set(query_ids)):
        raise RuntimeError("accepted query file contains duplicate query IDs")
    if set(query_ids) != set(targets_by_id):
        missing = sorted(set(targets_by_id) - set(query_ids))
        unexpected = sorted(set(query_ids) - set(targets_by_id))
        raise RuntimeError(
            f"query coverage must equal all frozen targets: missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
    if embedding_manifest.get("query_file_sha256") != sha256_file(args.queries):
        raise RuntimeError("query file hash does not match embedding manifest")
    if embedding_manifest.get("ordered_query_text_sha256") != ordered_text_sha256(
        [row["query"] for row in queries]
    ):
        raise RuntimeError("ordered query text hash does not match embedding manifest")
    targets = [targets_by_id[row["query_id"]] for row in queries]
    query_order = json.loads((args.embedding_dir / "query_order.json").read_text(encoding="utf-8"))
    if query_order != [row["query_id"] for row in queries]:
        raise RuntimeError("query embedding order does not match accepted query order")

    query_vectors = normalize_rows(
        np.asarray(np.load(args.embedding_dir / "queries.f32.npy", mmap_mode="r"), dtype=np.float32)
    )
    fields = load_field_embeddings(args.embedding_dir)
    if any(len(values) != len(candidates) for values in fields.values()):
        raise RuntimeError("candidate embedding row counts do not match manifest")

    context_permutation, context_matching = matched_donor_indices(
        candidates,
        "context_words",
        "context_sha256",
        args.seed + 1,
    )
    abstract_permutation, paper_matching = matched_donor_indices(
        candidates,
        "abstract_words",
        "paper_text_sha256",
        args.seed + 2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_condition: dict[str, list[dict]] = {}
    rankings_path = args.output_dir / "top10_rankings.jsonl"
    with rankings_path.open("w", encoding="utf-8") as rankings:
        condition_fields = dict(CONDITIONS)
        condition_fields["Full-shuffled-X"] = ("caption", "title", "abstract", "context_shuffled")
        condition_fields["Full-shuffled-TA"] = (
            "caption",
            "title_shuffled",
            "abstract_shuffled",
            "context",
        )

        for condition, names in condition_fields.items():
            candidate_vectors = combine_fields(
                fields,
                names,
                context_permutation,
                abstract_permutation,
            )
            scores, retrieved = search_condition(
                candidate_vectors,
                query_vectors,
                args.top_k,
                args.batch_size,
            )
            metrics = per_query_metrics(retrieved, targets, candidates)
            per_condition[condition] = metrics
            for query_index, indices in enumerate(retrieved):
                for rank, candidate_index in enumerate(indices, start=1):
                    candidate = candidates[int(candidate_index)]
                    rankings.write(
                        json.dumps(
                            {
                                "query_id": targets[query_index]["query_id"],
                                "condition": condition,
                                "rank": rank,
                                "score": float(scores[query_index, rank - 1]),
                                "retrieved_platform_id": candidate["platform_id"],
                                "retrieved_figure_number": candidate["figure_number"],
                                "retrieved_figure_path": candidate["figure_path"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            print(condition, aggregate(metrics), flush=True)

    aggregates = {condition: aggregate(rows) for condition, rows in per_condition.items()}
    rng = np.random.default_rng(args.seed)
    comparisons = []
    for condition, baseline in [
        ("C+T", "C"),
        ("C+T+A", "C"),
        ("Full", "C"),
        ("Full-T", "Full"),
        ("Full-A", "Full"),
        ("Full-C", "Full"),
        ("Full-X", "Full"),
        ("Full", "Full-shuffled-X"),
        ("Full", "Full-shuffled-TA"),
    ]:
        for metric in METRIC_NAMES:
            first = np.asarray([row[metric] for row in per_condition[condition]])
            second = np.asarray([row[metric] for row in per_condition[baseline]])
            mean, lower, upper = paired_bootstrap(
                first,
                second,
                rng,
                args.bootstrap_samples,
            )
            comparisons.append(
                {
                    "condition": condition,
                    "baseline": baseline,
                    "metric": metric,
                    "paired_difference": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )

    report = {
        "candidate_records": len(candidates),
        "queries": len(queries),
        "top_k": args.top_k,
        "seed": args.seed,
        "primary_endpoint": {
            "comparison": "Full minus C",
            "metric": "exact_r10",
        },
        "shuffled_control_matching": {
            "context": context_matching,
            "title_abstract": paper_matching,
        },
        "metrics": aggregates,
        "paired_comparisons": comparisons,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_query_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for condition, rows in per_condition.items():
            for row in rows:
                handle.write(json.dumps({"condition": condition, **row}) + "\n")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
