#!/usr/bin/env python3
"""Unblind four complete reviews and aggregate win/tie/loss outcomes."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


CATEGORIES = ("full_field_retrieval", "no_retrieval", "tie")


def load_review(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return center - half, center + half


def fleiss_kappa(item_counts: list[Counter[str]], raters: int) -> float:
    if not item_counts:
        raise ValueError("no item counts")
    observed = []
    totals = Counter()
    for counts in item_counts:
        if sum(counts.values()) != raters:
            raise ValueError("each item must have the same number of ratings")
        observed.append(
            (sum(counts[category] ** 2 for category in CATEGORIES) - raters)
            / (raters * (raters - 1))
        )
        totals.update(counts)
    mean_observed = sum(observed) / len(observed)
    total_ratings = len(item_counts) * raters
    expected = sum((totals[category] / total_ratings) ** 2 for category in CATEGORIES)
    if expected == 1:
        return 1.0
    return (mean_observed - expected) / (1 - expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-key", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs=4, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    key = json.loads(args.condition_key.read_text(encoding="utf-8"))
    build_id = str(key.get("build_id", "")).strip()
    if not build_id:
        raise ValueError("condition key is missing build_id")
    key_by_evaluator = {int(row["evaluator"]): row for row in key["evaluators"]}
    if set(key_by_evaluator) != {1, 2, 3, 4}:
        raise ValueError("condition key must contain evaluators 1-4")

    votes = []
    seen_evaluators = set()
    for review_path in args.reviews:
        review = load_review(review_path)
        if str(review.get("build_id", "")) != build_id:
            raise ValueError(f"build ID mismatch for review {review_path}")
        evaluator = int(review["evaluator"])
        if evaluator in seen_evaluators:
            raise ValueError(f"duplicate evaluator review: {evaluator}")
        seen_evaluators.add(evaluator)
        evaluator_key = key_by_evaluator[evaluator]
        if int(review["seed"]) != int(evaluator_key["seed"]):
            raise ValueError(f"seed mismatch for evaluator {evaluator}")
        item_key = {row["review_id"]: row for row in evaluator_key["items"]}
        responses = review.get("responses", [])
        if len(responses) != 20:
            raise ValueError(f"evaluator {evaluator} has {len(responses)} responses, expected 20")
        response_ids = [row["review_id"] for row in responses]
        if len(response_ids) != len(set(response_ids)) or set(response_ids) != set(item_key):
            raise ValueError(f"review ID coverage mismatch for evaluator {evaluator}")

        for response in responses:
            mapping = item_key[response["review_id"]]
            choice = response["choice"]
            if choice == "Tie":
                condition = "tie"
            elif choice == "A wins":
                condition = mapping["A"]
            elif choice == "B wins":
                condition = mapping["B"]
            else:
                raise ValueError(f"invalid choice from evaluator {evaluator}: {choice}")
            votes.append(
                {
                    "evaluator": evaluator,
                    "review_id": response["review_id"],
                    "case_id": mapping["case_id"],
                    "vote": condition,
                    "reason": str(response.get("reason", "")).strip(),
                }
            )

    if seen_evaluators != {1, 2, 3, 4}:
        raise ValueError("reviews must contain exactly evaluators 1-4")

    by_case: dict[str, list[dict]] = defaultdict(list)
    for vote in votes:
        by_case[vote["case_id"]].append(vote)
    if len(by_case) != 20 or any(len(case_votes) != 4 for case_votes in by_case.values()):
        raise ValueError("expected 20 cases with four votes each")

    case_results = []
    item_counts = []
    for case_id in sorted(by_case):
        counts = Counter(vote["vote"] for vote in by_case[case_id])
        item_counts.append(counts)
        if counts["full_field_retrieval"] >= 3:
            result = "retrieval_win"
        elif counts["no_retrieval"] >= 3:
            result = "no_retrieval_win"
        else:
            result = "tie"
        case_results.append(
            {
                "case_id": case_id,
                "retrieval_votes": counts["full_field_retrieval"],
                "no_retrieval_votes": counts["no_retrieval"],
                "tie_votes": counts["tie"],
                "item_result": result,
            }
        )

    result_counts = Counter(row["item_result"] for row in case_results)
    lower, upper = wilson_interval(result_counts["retrieval_win"], len(case_results))
    report = {
        "build_id": build_id,
        "evaluators": 4,
        "cases": 20,
        "aggregation_rule": "condition wins with at least 3 of 4 votes; all other patterns are ties",
        "item_results": {
            "retrieval_win": result_counts["retrieval_win"],
            "no_retrieval_win": result_counts["no_retrieval_win"],
            "tie": result_counts["tie"],
        },
        "retrieval_win_proportion": result_counts["retrieval_win"] / len(case_results),
        "retrieval_win_wilson_ci95": [lower, upper],
        "fleiss_kappa_three_category_votes": fleiss_kappa(item_counts, 4),
        "interpretation_note": "The 20 cases are deliberately balanced; counts and interval are descriptive and are not a release-wide estimate.",
        "case_results": case_results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "human_vote_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "unblinded_votes.jsonl").open("w", encoding="utf-8") as handle:
        for vote in sorted(votes, key=lambda row: (row["case_id"], row["evaluator"])):
            handle.write(json.dumps(vote, ensure_ascii=False) + "\n")
    print(json.dumps(report["item_results"], indent=2))


if __name__ == "__main__":
    main()
