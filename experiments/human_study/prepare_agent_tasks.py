#!/usr/bin/env python3
"""Freeze the 40 isolated Codex image-generation tasks for the E4 study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_generation import build_prompt, load_jsonl


CONDITIONS = ("no_retrieval", "full_field_retrieval")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument(
        "--condition",
        choices=("all", *CONDITIONS),
        default="all",
    )
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    if len(cases) != 20 or len({row["case_id"] for row in cases}) != 20:
        raise ValueError("expected exactly 20 unique generation cases")

    selected_conditions = CONDITIONS if args.condition == "all" else (args.condition,)
    references_by_case: dict[str, list[dict]] = {}
    if args.references is not None:
        for row in load_jsonl(args.references):
            references_by_case.setdefault(row["case_id"], []).append(row)
    for rows in references_by_case.values():
        rows.sort(key=lambda row: int(row["rank"]))

    if "full_field_retrieval" in selected_conditions and args.references is None:
        raise ValueError("--references is required for full_field_retrieval tasks")

    tasks = []
    for case in cases:
        case_id = case["case_id"]
        references = references_by_case.get(case_id, [])
        if "full_field_retrieval" in selected_conditions and len(references) != 3:
            raise ValueError(f"{case_id}: expected 3 finalized references, found {len(references)}")
        for reference in references:
            path = Path(reference["local_image_path"])
            if not path.is_file():
                raise FileNotFoundError(f"{case_id}: missing finalized reference {path}")

        prompt = build_prompt(case)
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        output_dir = (args.generation_root / case_id).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = output_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        for condition in selected_conditions:
            condition_references = references if condition == "full_field_retrieval" else []
            tasks.append(
                {
                    "task_id": f"{case_id}--{condition}",
                    "case_id": case_id,
                    "condition": condition,
                    "prompt": prompt,
                    "prompt_path": str(prompt_path),
                    "prompt_sha256": prompt_sha256,
                    "reference_paths": [
                        str(Path(row["local_image_path"]).resolve())
                        for row in condition_references
                    ],
                    "reference_sha256": [
                        row["image_sha256"] for row in condition_references
                    ],
                    "reference_keys": [
                        f"{row['retrieved_platform_id']}|{row['retrieved_figure_number']}"
                        for row in condition_references
                    ],
                    "output_path": str(output_dir / f"{condition}.png"),
                    "metadata_path": str(output_dir / f"{condition}.json"),
                    "isolation": "one fresh Codex agent and one built-in image_gen call",
                }
            )

    expected_tasks = 20 * len(selected_conditions)
    if len(tasks) != expected_tasks:
        raise AssertionError(f"expected {expected_tasks} tasks, built {len(tasks)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in tasks:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(tasks)} frozen tasks to {args.output}")


if __name__ == "__main__":
    main()
