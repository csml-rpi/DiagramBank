#!/usr/bin/env python3
"""Run hierarchical and caption-only retrieval for all 20 A5 queries.

Conditions:
  caption_only       — flat caption-index search (baseline)
  hier_narrow        — (t1=50,  t2=15, k=3)
  hier_default       — (t1=100, t2=30, k=3)  [primary]
  hier_wide          — (t1=200, t2=50, k=3)
  hier_vwide         — (t1=500, t2=100, k=3)

Output: results.jsonl
"""

import json
import sys
from pathlib import Path

# Add demo/ to the path so we can import retrieve.py
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "demo"))

from retrieve import hierarchical_retrieval, caption_only_retrieval

QUERIES = Path(__file__).resolve().parent / "queries_a5.jsonl"
OUT = Path(__file__).resolve().parent / "results.jsonl"

CONDITIONS = [
    ("caption_only",  None,  None, 3),
    ("hier_narrow",   50,    15,   3),
    ("hier_default",  100,   30,   3),
    ("hier_wide",     200,   50,   3),
    ("hier_vwide",    500,   100,  3),
]


def doc_to_record(query_name, condition, rank, doc):
    m = doc.metadata
    return {
        "query_name": query_name,
        "condition": condition,
        "rank": rank,
        "retrieved_id": m.get("id"),
        "retrieved_platform_id": m.get("platform_id"),
        "retrieved_title": m.get("title", "")[:200],
        "retrieved_venue": m.get("venue"),
        "retrieved_year": m.get("year"),
        "retrieved_figure_path": m.get("figure_path"),
        "retrieved_figure_caption": m.get("figure_caption", "")[:500],
        "retrieved_figure_number": m.get("figure_number"),
    }


def main():
    queries = []
    with open(QUERIES) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))

    print(f"Loaded {len(queries)} queries")

    results = []
    for qi, q in enumerate(queries):
        name = q["name"]
        title = q["title"]
        abstract = q["abstract"]
        caption = q["caption"]
        exclude_pid = q.get("platform_id")  # None for external queries

        print(f"\n{'='*60}")
        print(f"Query {qi+1}/{len(queries)}: {name}")
        if exclude_pid:
            print(f"  (leave-one-out: excluding {exclude_pid})")

        for cond_name, t1, t2, k in CONDITIONS:
            print(f"\n  --- Condition: {cond_name} ---")
            if cond_name == "caption_only":
                docs = caption_only_retrieval(caption, k=k,
                                              exclude_platform_id=exclude_pid)
            else:
                docs = hierarchical_retrieval(title, abstract, caption,
                                              t1=t1, t2=t2, k=k,
                                              exclude_platform_id=exclude_pid)

            for rank_i, doc in enumerate(docs):
                rec = doc_to_record(name, cond_name, rank_i + 1, doc)
                results.append(rec)
                # Verify no self-match
                if exclude_pid and rec["retrieved_platform_id"] == exclude_pid:
                    print(f"  WARNING: self-match at rank {rank_i+1}!")

            print(f"  Retrieved {len(docs)} figures")
            for d in docs:
                print(f"    [{d.metadata.get('venue','?')}] "
                      f"{d.metadata.get('title','?')[:60]}")

    # Save
    with open(OUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*60}")
    print(f"Total results: {len(results)}")
    print(f"Saved to {OUT}")

    # Summary stats
    from collections import Counter
    cond_counts = Counter(r["condition"] for r in results)
    for cond, cnt in sorted(cond_counts.items()):
        print(f"  {cond}: {cnt} results")


if __name__ == "__main__":
    main()
