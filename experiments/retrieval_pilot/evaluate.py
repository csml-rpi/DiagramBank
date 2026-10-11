#!/usr/bin/env python3
"""Evaluate retrieval experiment: Precision@3, Cohen's kappa, McNemar's test.

Reads:
  - results.jsonl           (retrieval results for all conditions)
  - annotations_tingwen.jsonl (human judgments on primary 120)
  - annotations_gpt.jsonl   (GPT-5.4 judgments on all conditions)

Outputs:
  - results.json
  - report.txt
"""

import json
import math
from collections import defaultdict
from pathlib import Path

DIR = Path(__file__).resolve().parent


def load_annotations(path):
    """Load annotations as dict: judgment_key -> label."""
    annots = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Build key
            key = rec.get("judgment_key")
            if not key:
                # annotation_sheet uses different field names
                pid = rec.get("_retrieved_platform_id", rec.get("retrieved_platform_id"))
                fnum = rec.get("_retrieved_figure_number", rec.get("retrieved_figure_number"))
                key = f"{rec['query_name']}|{pid}|{fnum}"
            label = rec.get("label") or rec.get("label_gpt")
            if label:
                annots[key] = label.strip().lower()
    return annots


def load_results():
    """Load retrieval results, return list of dicts."""
    with open(DIR / "results.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


def result_key(r):
    return f"{r['query_name']}|{r['retrieved_platform_id']}|{r['retrieved_figure_number']}"


def precision_at_3(results, annotations, condition):
    """Compute per-query Precision@3 for a given condition."""
    cond_results = [r for r in results if r["condition"] == condition]

    by_query = defaultdict(list)
    for r in cond_results:
        by_query[r["query_name"]].append(r)

    precisions = []
    for qname, items in sorted(by_query.items()):
        items.sort(key=lambda x: x["rank"])
        relevant = 0
        for r in items[:3]:
            key = result_key(r)
            label = annotations.get(key, "missing")
            if label == "relevant":
                relevant += 1
        precisions.append(relevant / 3.0)

    mean_p = sum(precisions) / len(precisions) if precisions else 0
    se = (sum((p - mean_p) ** 2 for p in precisions) / (len(precisions) * (len(precisions) - 1))) ** 0.5 if len(precisions) > 1 else 0
    return mean_p, se, precisions


def cohens_kappa(annots_a, annots_b):
    """Compute Cohen's kappa between two annotation dicts on shared keys."""
    shared = set(annots_a.keys()) & set(annots_b.keys())
    if not shared:
        return 0.0, 0, 0.0

    agree = 0
    a_pos = b_pos = 0
    n = len(shared)
    for key in shared:
        la = 1 if annots_a[key] == "relevant" else 0
        lb = 1 if annots_b[key] == "relevant" else 0
        if la == lb:
            agree += 1
        a_pos += la
        b_pos += lb

    po = agree / n
    pe = (a_pos / n) * (b_pos / n) + ((n - a_pos) / n) * ((n - b_pos) / n)
    if pe == 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)
    return kappa, n, po


def mcnemars_test(results, annotations, cond_a, cond_b):
    """McNemar's test comparing two conditions on paired (query, rank) slots.

    For each query and each rank (1,2,3), checks if condition A got relevant
    when condition B did not, and vice versa.
    """
    # Build lookup: (query, condition, rank) -> relevant?
    lookup = {}
    for r in results:
        if r["condition"] not in (cond_a, cond_b):
            continue
        key = result_key(r)
        label = annotations.get(key, "missing")
        is_rel = 1 if label == "relevant" else 0
        lookup[(r["query_name"], r["condition"], r["rank"])] = is_rel

    # Count discordant pairs
    b_only = 0  # A=not_relevant, B=relevant
    c_only = 0  # A=relevant, B=not_relevant
    concordant_both = 0
    concordant_neither = 0

    queries = set(r["query_name"] for r in results)
    for qname in queries:
        for rank in [1, 2, 3]:
            a = lookup.get((qname, cond_a, rank), 0)
            b = lookup.get((qname, cond_b, rank), 0)
            if a == 1 and b == 1:
                concordant_both += 1
            elif a == 0 and b == 0:
                concordant_neither += 1
            elif a == 1 and b == 0:
                c_only += 1
            else:
                b_only += 1

    # McNemar's chi-squared (with continuity correction)
    if b_only + c_only == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        chi2 = (abs(b_only - c_only) - 1) ** 2 / (b_only + c_only)
        # Approximate p-value from chi2(1)
        # Using survival function approximation
        from math import exp, sqrt, pi
        # Simple normal approximation: chi2(1) -> z^2
        z = sqrt(chi2)
        # One-sided -> two-sided
        # Use erfc approximation
        p_val = 2 * _normal_sf(z)

    return {
        "concordant_both": concordant_both,
        "concordant_neither": concordant_neither,
        "hier_only": c_only,  # hier=relevant, caption=not
        "caption_only_only": b_only,  # caption=relevant, hier=not
        "chi2": chi2,
        "p_value": p_val,
    }


def _normal_sf(z):
    """Survival function of standard normal (complementary CDF)."""
    # Abramowitz and Stegun approximation 26.2.17
    if z < 0:
        return 1 - _normal_sf(-z)
    b0 = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    t = 1.0 / (1.0 + b0 * z)
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    return phi * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))


def main():
    results = load_results()

    # Load annotations
    gpt_path = DIR / "annotations_gpt.jsonl"
    human_path = DIR / "annotations_tingwen.jsonl"

    has_gpt = gpt_path.exists()
    has_human = human_path.exists()

    if not has_gpt and not has_human:
        print("ERROR: No annotations found. Run judge_gpt.py or provide annotations_tingwen.jsonl.")
        return

    report = []
    out = {}

    if has_gpt:
        gpt_annots = load_annotations(gpt_path)
        report.append(f"GPT-5.4 annotations: {len(gpt_annots)}")
        relevant_count = sum(1 for v in gpt_annots.values() if v == "relevant")
        report.append(f"  relevant: {relevant_count}, not_relevant: {len(gpt_annots) - relevant_count}")

        # Precision@3 for all conditions
        conditions = sorted(set(r["condition"] for r in results))
        out["gpt"] = {}
        report.append("\nGPT-5.4 Precision@3:")
        for cond in conditions:
            mean_p, se, per_q = precision_at_3(results, gpt_annots, cond)
            out["gpt"][cond] = {"precision_at_3": round(mean_p, 4), "se": round(se, 4)}
            report.append(f"  {cond:20s}: {mean_p:.3f} +/- {se:.3f}")

        # McNemar: hier_default vs caption_only
        mcnemar = mcnemars_test(results, gpt_annots, "hier_default", "caption_only")
        out["gpt"]["mcnemar_hier_vs_caption"] = mcnemar
        report.append(f"\nMcNemar (GPT): hier_default vs caption_only")
        report.append(f"  hier_only={mcnemar['hier_only']}, caption_only_only={mcnemar['caption_only_only']}")
        report.append(f"  chi2={mcnemar['chi2']:.3f}, p={mcnemar['p_value']:.4f}")

    if has_human:
        human_annots = load_annotations(human_path)
        report.append(f"\nHuman annotations: {len(human_annots)}")
        relevant_count = sum(1 for v in human_annots.values() if v == "relevant")
        report.append(f"  relevant: {relevant_count}, not_relevant: {len(human_annots) - relevant_count}")

        out["human"] = {}
        report.append("\nHuman Precision@3:")
        for cond in ("caption_only", "hier_default"):
            mean_p, se, per_q = precision_at_3(results, human_annots, cond)
            out["human"][cond] = {"precision_at_3": round(mean_p, 4), "se": round(se, 4)}
            report.append(f"  {cond:20s}: {mean_p:.3f} +/- {se:.3f}")

        mcnemar = mcnemars_test(results, human_annots, "hier_default", "caption_only")
        out["human"]["mcnemar_hier_vs_caption"] = mcnemar
        report.append(f"\nMcNemar (human): hier_default vs caption_only")
        report.append(f"  hier_only={mcnemar['hier_only']}, caption_only_only={mcnemar['caption_only_only']}")
        report.append(f"  chi2={mcnemar['chi2']:.3f}, p={mcnemar['p_value']:.4f}")

    if has_gpt and has_human:
        kappa, n_shared, po = cohens_kappa(gpt_annots, human_annots)
        out["inter_rater"] = {
            "cohens_kappa": round(kappa, 4),
            "n_shared": n_shared,
            "agreement": round(po, 4),
        }
        report.append(f"\nInter-rater agreement (human vs GPT-5.4):")
        report.append(f"  Shared judgments: {n_shared}")
        report.append(f"  Raw agreement: {po:.3f}")
        report.append(f"  Cohen's kappa: {kappa:.3f}")

    # Save
    out_json = DIR / "results.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_json}")

    report_path = DIR / "report.txt"
    report_text = "\n".join(report)
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"Saved {report_path}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
