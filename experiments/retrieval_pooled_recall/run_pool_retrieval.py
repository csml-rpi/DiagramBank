#!/usr/bin/env python3
"""
Step 1: Retrieve top-50 from two methods, pool & deduplicate.

For each of the 20 A5 queries, retrieves top-50 diagrams via:
  - hier_default (t1=200, t2=100, k=50)
  - caption_only (k=50)

Outputs:
  ranked_lists.jsonl  — all 2,000 results with rank order preserved
  pool.jsonl          — deduplicated (query, diagram) pairs for judging

Requires: OPENAI_API_KEY env var (for text-embedding-3-large).
"""

import json
import sys
from pathlib import Path

# Add demo/ to the path so we can import retrieve.py
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "demo"))

from retrieve import hierarchical_retrieval, caption_only_retrieval

QUERIES_PATH = Path(__file__).resolve().parents[1] / "retrieval_pilot" / "queries_a5.jsonl"
SCRIPT_DIR = Path(__file__).resolve().parent
RANKED_PATH = SCRIPT_DIR / "ranked_lists.jsonl"
POOL_PATH = SCRIPT_DIR / "pool.jsonl"

# Retrieval conditions: (name, t1, t2, k)
# hier_default widened: t1=200, t2=100 so we can get k=50 results
CONDITIONS = [
    ("caption_only", None, None, 50),
    ("hier_default", 200, 100, 50),
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
    # Load queries
    queries = []
    with open(QUERIES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    print(f"Loaded {len(queries)} queries from {QUERIES_PATH}")

    # Phase 1: Retrieve ranked lists
    ranked_results = []
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
            print(f"  --- {cond_name} (k={k}) ---")
            if cond_name == "caption_only":
                docs = caption_only_retrieval(caption, k=k,
                                              exclude_platform_id=exclude_pid)
            else:
                docs = hierarchical_retrieval(title, abstract, caption,
                                              t1=t1, t2=t2, k=k,
                                              exclude_platform_id=exclude_pid)

            for rank_i, doc in enumerate(docs):
                rec = doc_to_record(name, cond_name, rank_i + 1, doc)
                ranked_results.append(rec)
                if exclude_pid and rec["retrieved_platform_id"] == exclude_pid:
                    print(f"  WARNING: self-match at rank {rank_i+1}!")

            print(f"  Retrieved {len(docs)} figures")

    # Save ranked lists
    with open(RANKED_PATH, "w") as f:
        for r in ranked_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(ranked_results)} ranked results to {RANKED_PATH}")

    # Phase 2: Pool & deduplicate for judging
    queries_by_name = {q["name"]: q for q in queries}
    pool = {}  # (query_name, platform_id, figure_number) -> pool record
    for r in ranked_results:
        key = (r["query_name"], r["retrieved_platform_id"],
               str(r["retrieved_figure_number"]))
        if key not in pool:
            q = queries_by_name[r["query_name"]]
            pool[key] = {
                "judgment_key": f"{r['query_name']}|{r['retrieved_platform_id']}|{r['retrieved_figure_number']}",
                "query_name": r["query_name"],
                "query_title": q["title"],
                "query_abstract": q["abstract"],
                "query_caption": q["caption"],
                "retrieved_platform_id": r["retrieved_platform_id"],
                "retrieved_figure_number": r["retrieved_figure_number"],
                "retrieved_figure_path": r["retrieved_figure_path"],
                "retrieved_title": r["retrieved_title"],
                "retrieved_caption": r["retrieved_figure_caption"],
                "conditions": [r["condition"]],
            }
        else:
            if r["condition"] not in pool[key]["conditions"]:
                pool[key]["conditions"].append(r["condition"])

    pool_records = list(pool.values())
    with open(POOL_PATH, "w") as f:
        for rec in pool_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{'='*60}")
    print(f"Pool: {len(pool_records)} unique (query, diagram) pairs")
    print(f"  from {len(ranked_results)} total ranked results")

    # Stats
    from collections import Counter
    both = sum(1 for p in pool_records if len(p["conditions"]) > 1)
    cond_counts = Counter()
    for p in pool_records:
        for c in p["conditions"]:
            cond_counts[c] += 1
    print(f"  Overlap (in both conditions): {both}")
    for cond, cnt in sorted(cond_counts.items()):
        print(f"  {cond}: {cnt} unique figures")

    per_query = Counter(p["query_name"] for p in pool_records)
    avg = sum(per_query.values()) / len(per_query)
    print(f"  Avg per query: {avg:.1f} (min {min(per_query.values())}, max {max(per_query.values())})")


if __name__ == "__main__":
    main()
