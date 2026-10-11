#!/usr/bin/env python3
"""Select 20 queries for the A5 retrieval experiment.

4 external papers (from demo/queries.jsonl) + 16 leave-one-out papers
sampled from faiss/data.jsonl with venue stratification and diversity
constraints.

Output: queries_a5.jsonl
"""

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[3]  # DiagramBank/
DATA_JSONL = ROOT / "faiss" / "data.jsonl"
DEMO_QUERIES = ROOT / "demo" / "queries.jsonl"
OUT = Path(__file__).resolve().parent / "queries_a5.jsonl"

# --- Load external case study queries ----------------------------------------
def load_external_queries():
    queries = []
    with open(DEMO_QUERIES) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            queries.append({
                "name": d["name"],
                "source": "external",
                "platform_id": None,
                "title": d["title"],
                "abstract": d["abstract"],
                "caption": d["caption"],
                "venue": d.get("venue"),
                "year": d.get("year"),
            })
    return queries


# --- Sample leave-one-out queries from the FAISS index -----------------------
def load_papers():
    """Group data.jsonl by platform_id, return dict of paper -> list of figures."""
    papers = defaultdict(list)
    with open(DATA_JSONL) as f:
        for line in f:
            d = json.loads(line)
            papers[d["platform_id"]].append(d)
    return papers


def select_leave_one_out(papers, n=16):
    """Stratified sampling: 6 ICLR + 6 NeurIPS + 4 ICML, diverse years."""
    rng = random.Random(SEED)

    # Target allocation (roughly proportional to corpus: ICLR ~36%, NeurIPS ~34%, ICML ~20%, TMLR ~10%)
    # TMLR excluded: no review scores, smaller corpus
    venue_targets = {"ICLR": 6, "NeurIPS": 6, "ICML": 4}

    # Group papers by venue, filter to papers with >= 2 figures
    venue_papers = defaultdict(list)
    for pid, figs in papers.items():
        venue = figs[0]["venue"]
        if venue not in venue_targets:
            continue
        if len(figs) < 2:
            continue
        venue_papers[venue].append((pid, figs))

    selected = []
    for venue, target in venue_targets.items():
        pool = venue_papers[venue]
        rng.shuffle(pool)

        # Try to pick from different years
        year_buckets = defaultdict(list)
        for pid, figs in pool:
            year_buckets[figs[0]["year"]].append((pid, figs))

        picked = []
        # Round-robin across years
        years_sorted = sorted(year_buckets.keys())
        idx = 0
        while len(picked) < target:
            year = years_sorted[idx % len(years_sorted)]
            if year_buckets[year]:
                picked.append(year_buckets[year].pop(0))
            idx += 1
            if idx > len(years_sorted) * 50:  # safety
                break

        for pid, figs in picked[:target]:
            # Use figure 1 caption as query, or first diagram figure
            fig1 = None
            for fig in sorted(figs, key=lambda f: int(f.get("figure_number", 99))):
                if fig.get("figure_type") == "diagram":
                    fig1 = fig
                    break
            if fig1 is None:
                fig1 = sorted(figs, key=lambda f: int(f.get("figure_number", 99)))[0]

            selected.append({
                "name": f"loo_{venue.lower()}_{pid[:8]}",
                "source": "leave_one_out",
                "platform_id": pid,
                "title": fig1["title"],
                "abstract": fig1["abstract"],
                "caption": fig1["figure_caption"],
                "venue": venue,
                "year": fig1["year"],
            })

    return selected


def main():
    external = load_external_queries()
    print(f"External queries: {len(external)}")
    for q in external:
        print(f"  {q['name']}")

    papers = load_papers()
    print(f"\nTotal papers in data.jsonl: {len(papers)}")

    loo = select_leave_one_out(papers, n=16)
    print(f"\nLeave-one-out queries: {len(loo)}")
    for q in loo:
        print(f"  {q['name']} | {q['venue']} {q['year']} | {q['title'][:60]}...")

    all_queries = external + loo
    print(f"\nTotal queries: {len(all_queries)}")

    with open(OUT, "w") as f:
        for q in all_queries:
            f.write(json.dumps(q) + "\n")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
