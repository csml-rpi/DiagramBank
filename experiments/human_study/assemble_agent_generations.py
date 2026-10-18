#!/usr/bin/env python3
"""Validate isolated agent outputs and assemble the blind-review manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


CONDITIONS = ("no_retrieval", "full_field_retrieval")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_task(task: dict) -> dict:
    image_path = Path(task["output_path"])
    metadata_path = Path(task["metadata_path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"missing generated image: {image_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing generation metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for field in ("case_id", "condition", "prompt_sha256"):
        if metadata.get(field) != task[field]:
            raise ValueError(
                f"{task['task_id']}: metadata {field}={metadata.get(field)!r}, "
                f"expected {task[field]!r}"
            )
    if metadata.get("interface") != "Codex built-in image_gen":
        raise ValueError(f"{task['task_id']}: unexpected generation interface")

    expected_references = [str(Path(path).resolve()) for path in task["reference_paths"]]
    actual_references = [
        str(Path(path).resolve()) for path in metadata.get("reference_paths", [])
    ]
    if actual_references != expected_references:
        raise ValueError(f"{task['task_id']}: reference-path mismatch")
    expected_count = 3 if task["condition"] == "full_field_retrieval" else 0
    if len(actual_references) != expected_count:
        raise ValueError(f"{task['task_id']}: expected {expected_count} references")

    digest = sha256_file(image_path)
    if metadata.get("output_sha256") != digest:
        raise ValueError(f"{task['task_id']}: output SHA-256 mismatch")
    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        width, height = image.size
    if metadata.get("width") != width or metadata.get("height") != height:
        raise ValueError(f"{task['task_id']}: output dimension mismatch")
    return {
        **task,
        "output_sha256": digest,
        "width": width,
        "height": height,
        "interface": metadata["interface"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_jsonl(args.tasks)
    if len(tasks) != 40 or len({row["task_id"] for row in tasks}) != 40:
        raise ValueError("expected exactly 40 unique frozen tasks")
    validated = [validate_task(task) for task in tasks]

    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in validated:
        by_case[row["case_id"]][row["condition"]] = row
    if len(by_case) != 20:
        raise ValueError(f"expected 20 cases, found {len(by_case)}")

    manifest = []
    dimensions: dict[str, int] = defaultdict(int)
    for case_id in sorted(by_case):
        conditions = by_case[case_id]
        if set(conditions) != set(CONDITIONS):
            raise ValueError(f"{case_id}: incomplete conditions {sorted(conditions)}")
        no_retrieval = conditions["no_retrieval"]
        retrieval = conditions["full_field_retrieval"]
        if no_retrieval["prompt_sha256"] != retrieval["prompt_sha256"]:
            raise ValueError(f"{case_id}: text prompt differs between conditions")
        for row in conditions.values():
            dimensions[f"{row['width']}x{row['height']}"] += 1
        manifest.append(
            {
                "case_id": case_id,
                "interface": "Codex built-in image_gen",
                "isolation": "one fresh Codex agent and one built-in image_gen call per output",
                "prompt_sha256": no_retrieval["prompt_sha256"],
                "no_retrieval_path": no_retrieval["output_path"],
                "no_retrieval_sha256": no_retrieval["output_sha256"],
                "full_field_retrieval_path": retrieval["output_path"],
                "full_field_retrieval_sha256": retrieval["output_sha256"],
                "reference_keys": retrieval["reference_keys"],
                "reference_sha256": retrieval["reference_sha256"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "cases": 20,
        "generated_outputs": 40,
        "conditions": {name: 20 for name in CONDITIONS},
        "generation_interface": "Codex built-in image_gen",
        "isolated_agent_calls": 40,
        "output_dimensions": dict(sorted(dimensions.items())),
        "all_pairs_share_identical_text_prompt": True,
        "no_retrieval_reference_count": 0,
        "retrieval_reference_count_per_case": 3,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
