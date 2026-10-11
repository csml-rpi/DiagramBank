#!/usr/bin/env python3
"""Prepare a blinded annotation sheet for Tingwen.

Pools the primary 120 results (caption_only + hier_default), de-duplicates
(query, figure) pairs, randomises within each query, and strips condition
labels so the annotator is blind to which method produced each result.

Output: annotation_sheet.jsonl
"""

import json
import random
from pathlib import Path

SEED = 42
DIR = Path(__file__).resolve().parent
RESULTS = DIR / "results.jsonl"
QUERIES = DIR / "queries_a5.jsonl"
OUT = DIR / "annotation_sheet.jsonl"


def main():
    rng = random.Random(SEED)

    # Load queries for context
    queries = {}
    with open(QUERIES) as f:
        for line in f:
            q = json.loads(line.strip())
            queries[q["name"]] = q

    # Load primary results
    with open(RESULTS) as f:
        rows = [json.loads(l) for l in f]
    primary = [r for r in rows if r["condition"] in ("caption_only", "hier_default")]

    # De-duplicate by (query_name, platform_id, figure_number)
    # Keep track of which conditions produced each figure
    seen = {}  # (query_name, pid, fig_num) -> record + conditions
    for r in primary:
        key = (r["query_name"], r["retrieved_platform_id"],
               str(r["retrieved_figure_number"]))
        if key not in seen:
            seen[key] = {**r, "_conditions": [r["condition"]]}
        else:
            seen[key]["_conditions"].append(r["condition"])

    # Group by query, then shuffle within each query
    by_query = {}
    for key, rec in seen.items():
        qname = key[0]
        by_query.setdefault(qname, []).append(rec)

    sheet = []
    jid = 0
    # Process queries in a fixed order (sorted by name)
    for qname in sorted(by_query.keys()):
        items = by_query[qname]
        rng.shuffle(items)
        q = queries[qname]
        for item in items:
            jid += 1
            sheet.append({
                "judgment_id": f"j{jid:03d}",
                "query_name": qname,
                "query_title": q["title"],
                "query_abstract": q["abstract"][:500],
                "query_caption": q["caption"],
                "retrieved_figure_path": item["retrieved_figure_path"],
                "retrieved_figure_caption": item["retrieved_figure_caption"],
                "retrieved_title": item["retrieved_title"],
                "retrieved_venue": item["retrieved_venue"],
                "retrieved_year": item["retrieved_year"],
                # Hidden metadata (for evaluate.py to unblind):
                "_conditions": item["_conditions"],
                "_retrieved_platform_id": item["retrieved_platform_id"],
                "_retrieved_figure_number": item["retrieved_figure_number"],
                # Tingwen fills this in:
                "label": None,
            })

    with open(OUT, "w") as f:
        for rec in sheet:
            f.write(json.dumps(rec) + "\n")

    print(f"Annotation sheet: {len(sheet)} judgments")
    print(f"  (de-duplicated from 120 primary results)")

    # Count how many are shared between conditions
    shared = sum(1 for rec in sheet if len(rec["_conditions"]) > 1)
    cap_only = sum(1 for rec in sheet if rec["_conditions"] == ["caption_only"])
    hier_only = sum(1 for rec in sheet if rec["_conditions"] == ["hier_default"])
    print(f"  caption_only exclusive: {cap_only}")
    print(f"  hier_default exclusive: {hier_only}")
    print(f"  shared (both conditions): {shared}")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
