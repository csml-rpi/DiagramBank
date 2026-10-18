#!/usr/bin/env python3
"""Audit E1 field coverage on the 57,100-record final release table.

This script verifies that $FIG_RAG_DIR/faiss/research.db (table Data) matches the
cascade-final keep set reconstructed from voted.jsonl and
labels_gpt_tiebreak.jsonl, then measures title/abstract/caption/URL/context
coverage and figure-reference traceability on that exact 57,100-row table.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb


REPO = Path(__file__).resolve().parents[2]
FIG_RAG_DIR = Path(os.environ.get("FIG_RAG_DIR", Path.home() / "fig-rag-dir"))
DB_PATH = FIG_RAG_DIR / "faiss" / "research.db"
VOTED_PATH = REPO / "cascade" / "voted.jsonl"
TIEBREAK_PATH = REPO / "cascade" / "labels_gpt_tiebreak.jsonl"
OUT_JSON = Path(__file__).with_name("final_release_57100_coverage.json")
OUT_MD = Path(__file__).with_name("final_release_57100_coverage.md")


def record_key(record: dict[str, Any]) -> str:
    return f"{record['platform_id']}|{record['figure_number']}"


def load_tiebreak_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    with TIEBREAK_PATH.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            labels[record_key(record)] = record.get("label_gpt_tiebreak", "")
    return labels


def reconstruct_final_keys() -> tuple[set[str], Counter[str], Counter[str]]:
    tiebreak = load_tiebreak_labels()
    final_keys: set[str] = set()
    keep_reasons: Counter[str] = Counter()
    venue_counts: Counter[str] = Counter()

    with VOTED_PATH.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record_key(record)
            category = record["vote_category"]

            keep = False
            reason = ""
            if record.get("keep_majority"):
                keep = True
                reason = "tier1_majority_vote"
            elif category == "t1_minority_diagram" and tiebreak.get(key) == "diagram":
                keep = True
                reason = "tier1_tiebreak_confirmed"
            elif category == "t2_vlm_consensus" and tiebreak.get(key) == "diagram":
                keep = True
                reason = "tier2_rescue_confirmed"

            if keep:
                final_keys.add(key)
                keep_reasons[reason] += 1
                venue_counts[record["venue"]] += 1

    return final_keys, keep_reasons, venue_counts


def nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def context_matches_figure_number(context: str, figure_number: str) -> bool:
    digits = re.sub(r"\D", "", str(figure_number))
    if not digits:
        return False
    pattern = re.compile(
        rf"\b(?:Figure|Fig)s?\.?.{{0,50}}?\b{re.escape(digits)}(?!\d)",
        re.IGNORECASE,
    )
    return bool(pattern.search(context))


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00%"
    return f"{100 * numerator / denominator:.2f}%"


def main() -> None:
    final_keys, keep_reasons, cascade_venue_counts = reconstruct_final_keys()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        """
        select platform_id, figure_number, figure_path, venue,
               title, abstract, figure_caption, figure_context, url
        from Data
        """
    ).fetchall()
    con.close()

    db_keys = {f"{platform_id}|{figure_number}" for platform_id, figure_number, *_ in rows}
    missing_from_db = sorted(final_keys - db_keys)
    extra_in_db = sorted(db_keys - final_keys)

    field_counts: Counter[str] = Counter()
    url_shape = 0
    context_nonempty = 0
    context_match = 0
    context_mismatch_examples: list[dict[str, str]] = []
    db_venue_counts: Counter[str] = Counter()

    for platform_id, figure_number, figure_path, venue, title, abstract, caption, context, url in rows:
        db_venue_counts[venue] += 1
        values = {
            "title": title,
            "abstract": abstract,
            "figure_caption": caption,
            "url": url,
            "figure_context": context,
        }
        for field, value in values.items():
            if nonempty(value):
                field_counts[field] += 1

        if url == f"https://openreview.net/forum?id={platform_id}":
            url_shape += 1

        if nonempty(context):
            context_nonempty += 1
            if context_matches_figure_number(str(context), str(figure_number)):
                context_match += 1
            elif len(context_mismatch_examples) < 10:
                context_mismatch_examples.append(
                    {
                        "platform_id": str(platform_id),
                        "figure_number": str(figure_number),
                        "figure_path": str(figure_path),
                        "context_prefix": str(context)[:240],
                    }
                )

    total = len(rows)
    result = {
        "inputs": {
            "db_table": "$FIG_RAG_DIR/faiss/research.db::Data",
            "voted_jsonl": str(VOTED_PATH.relative_to(REPO)),
            "tiebreak_jsonl": str(TIEBREAK_PATH.relative_to(REPO)),
        },
        "key_verification": {
            "cascade_final_keys": len(final_keys),
            "db_rows": total,
            "db_unique_keys": len(db_keys),
            "missing_from_db": len(missing_from_db),
            "extra_in_db": len(extra_in_db),
            "keep_reasons": dict(keep_reasons),
            "cascade_venue_counts": dict(cascade_venue_counts),
            "db_venue_counts": dict(db_venue_counts),
            "missing_examples": missing_from_db[:10],
            "extra_examples": extra_in_db[:10],
        },
        "field_coverage": {
            field: {
                "count": field_counts[field],
                "total": total,
                "share": pct(field_counts[field], total),
            }
            for field in ["title", "abstract", "figure_caption", "url", "figure_context"]
        },
        "url_shape": {
            "count": url_shape,
            "total": total,
            "share": pct(url_shape, total),
        },
        "context_reference_check": {
            "nonempty_contexts": context_nonempty,
            "matching_contexts": context_match,
            "share_of_all_records": pct(context_match, total),
            "share_of_nonempty_contexts": pct(context_match, context_nonempty),
            "mismatch_examples": context_mismatch_examples,
        },
    }

    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# E1 Final Release Coverage Audit",
        "",
        "## Inputs",
        "",
        f"- Final metadata table: `{result['inputs']['db_table']}`",
        f"- Cascade vote file: `{result['inputs']['voted_jsonl']}`",
        f"- Cascade tiebreak file: `{result['inputs']['tiebreak_jsonl']}`",
        "",
        "## Final-Release Key Verification",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| cascade final keys reconstructed from logs | {len(final_keys):,} |",
        f"| rows in `Data` table | {total:,} |",
        f"| unique `(platform_id, figure_number)` keys in `Data` | {len(db_keys):,} |",
        f"| cascade-final keys missing from `Data` | {len(missing_from_db):,} |",
        f"| `Data` keys not in cascade-final set | {len(extra_in_db):,} |",
        "",
        "Cascade keep-reason counts:",
        "",
        "| Keep reason | Count |",
        "|---|---:|",
    ]
    for reason in ["tier1_majority_vote", "tier1_tiebreak_confirmed", "tier2_rescue_confirmed"]:
        lines.append(f"| `{reason}` | {keep_reasons[reason]:,} |")

    lines.extend(
        [
            "",
            "Venue counts:",
            "",
            "| Venue | Cascade final keys | `Data` rows |",
            "|---|---:|---:|",
        ]
    )
    for venue in ["ICLR", "ICML", "NeurIPS", "TMLR"]:
        lines.append(f"| {venue} | {cascade_venue_counts[venue]:,} | {db_venue_counts[venue]:,} |")

    lines.extend(
        [
            "",
            "## Field Coverage Over 57,100 Final Records",
            "",
            "| Field | Count | Share |",
            "|---|---:|---:|",
        ]
    )
    for field in ["title", "abstract", "figure_caption", "url", "figure_context"]:
        data = result["field_coverage"][field]
        lines.append(f"| `{field}` | {data['count']:,} / {data['total']:,} | {data['share']} |")
    lines.append(f"| URL matches `platform_id` shape | {url_shape:,} / {total:,} | {pct(url_shape, total)} |")

    lines.extend(
        [
            "",
            "## Figure-Reference Traceability",
            "",
            "| Check | Count | Share |",
            "|---|---:|---:|",
            f"| non-empty `figure_context` | {context_nonempty:,} / {total:,} | {pct(context_nonempty, total)} |",
            f"| matching stored figure number among all records | {context_match:,} / {total:,} | {pct(context_match, total)} |",
            f"| matching stored figure number among non-empty contexts | {context_match:,} / {context_nonempty:,} | {pct(context_match, context_nonempty)} |",
            "",
            "The figure-reference check uses the same conservative regex family as the earlier E1 audit: it strips non-digits from the stored figure number and searches each non-empty context span for a nearby `Figure` or `Fig.` reference to that number.",
        ]
    )

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
