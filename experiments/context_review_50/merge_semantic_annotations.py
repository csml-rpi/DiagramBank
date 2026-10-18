#!/usr/bin/env python3
"""Merge E1 semantic-audit agent annotations and summarize counts."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUTS = [
    HERE / "e1_annotations_agent_a.csv",
    HERE / "e1_annotations_agent_b.csv",
    HERE / "e1_annotations_agent_c.csv",
]
OUT_CSV = HERE / "e1_semantic_annotations_merged.csv"
OUT_MD = HERE / "e1_semantic_annotation_summary.md"


def read_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in INPUTS:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sample_id"]
                if sid in seen:
                    raise ValueError(f"duplicate sample_id {sid}")
                seen.add(sid)
                row["source_file"] = path.name
                rows.append(row)
    rows.sort(key=lambda r: int(r["sample_id"].split("-")[1]))
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_counter(rows: list[dict[str, str]], field: str) -> Counter[str]:
    return Counter(row[field] for row in rows)


def counter_table(counter: Counter[str]) -> list[str]:
    total = sum(counter.values())
    lines = ["| Label | Count | Share |", "|---|---:|---:|"]
    for label, count in counter.most_common():
        lines.append(f"| `{label}` | {count} | {count / total:.1%} |")
    return lines


def write_summary(rows: list[dict[str, str]]) -> None:
    context = label_counter(rows, "context_alignment")
    caption = label_counter(rows, "caption_image_match")
    provenance = label_counter(rows, "same_paper_provenance")
    usable = label_counter(rows, "usable_for_rebuttal")

    context_assessable = len(rows) - context.get("NA_missing", 0)
    explains = context.get("2_explains", 0)
    references = context.get("1_references_only", 0)

    lines = [
        "# E1 Semantic Annotation Summary",
        "",
        "## Scope",
        "",
        "This summary merges three agent annotation files over the deterministic",
        "50-record E1 sample. Agents used local metadata fields only; no network",
        "access was used. The current checkout does not contain release figure",
        "image files for the sampled records, so visual image-caption matching is",
        "not assessed here.",
        "",
        "## Counts",
        "",
        f"- Total sampled records: {len(rows)}",
        f"- Records with assessable non-empty context: {context_assessable}",
        f"- Context spans judged explanatory: {explains}/{context_assessable} ({explains / context_assessable:.1%})",
        f"- Context spans judged explanatory or useful reference: {explains + references}/{context_assessable} ({(explains + references) / context_assessable:.1%})",
        f"- Records missing context: {context.get('NA_missing', 0)}/{len(rows)} ({context.get('NA_missing', 0) / len(rows):.1%})",
        "",
        "### Context Alignment",
        "",
        *counter_table(context),
        "",
        "### Caption-Image Match",
        "",
        *counter_table(caption),
        "",
        "### Same-Paper Provenance",
        "",
        *counter_table(provenance),
        "",
        "### Usable For Rebuttal",
        "",
        *counter_table(usable),
        "",
        "## Interpretation",
        "",
        "The sample supports a cautious context-quality spot check: among records",
        "with non-empty `figure_context`, most spans explicitly explain the same",
        "figure rather than merely mentioning it. However, because the local",
        "checkout lacks the corresponding figure image files, this annotation does",
        "not validate image-caption visual alignment. The rebuttal should describe",
        "this as a small metadata/context semantic audit, not as complete visual",
        "or corpus-level validation.",
        "",
        "## Rebuttal-Ready Wording",
        "",
        "> In a deterministic 50-record spot check of the released metadata slice,",
        "> agents inspected title, abstract, caption, figure number, and stored",
        "> in-text context. Forty records contained non-empty `figure_context`;",
        f"> {explains} of these {context_assessable} context spans were judged to",
        "> explain the same figure, and four more were judged to at least reference",
        "> the correct figure. Ten sampled records had no stored in-text context.",
        "> Because the local checkout did not include the corresponding image files,",
        "> this spot check does not measure visual image-caption alignment; it is a",
        "> small audit of context-field semantic consistency.",
        "",
        "## Files",
        "",
        f"- Merged annotations: `{OUT_CSV.name}`",
        "- Agent notes: `e1_annotations_agent_a_notes.md`,",
        "  `e1_annotations_agent_b_notes.md`, `e1_annotations_agent_c_notes.md`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = read_rows()
    if len(rows) != 50:
        raise ValueError(f"expected 50 rows, got {len(rows)}")
    write_csv(rows)
    write_summary(rows)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
