#!/usr/bin/env python3
"""Stratified precision estimates for the final-release audit.

path_counts.csv holds, for each retained cascade path h, the population size
N_h, the number of audited records n_h, and the number judged schematic (TP).
With W_h = N_h / N and p_h = TP_h / n_h, the estimate and its normal
interval with finite-population correction are

    p = sum_h W_h p_h
    V = sum_h W_h^2 (1 - n_h / N_h) p_h (1 - p_h) / (n_h - 1)
    p +/- 1.96 sqrt(V)

computed for the full release, DiagramBank-Core (Tier 1 unanimous and Tier 2
consensus), and the remaining records. The item-level judgments and the
within-path venue allocation are not available; only these path totals are.
"""

import csv
import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COUNTS_PATH = SCRIPT_DIR / "path_counts.csv"
OUT_JSON = SCRIPT_DIR / "estimates.json"
Z = 1.959963984540054

SUBSETS = {
    "full_release": ["t1_unanimous_diagram", "t1_majority_diagram", "t1_minority_diagram_gpt_confirmed",
                     "t2_vlm_consensus_gpt_confirmed"],
    "core": ["t1_unanimous_diagram", "t2_vlm_consensus_gpt_confirmed"],
    "remaining": ["t1_majority_diagram", "t1_minority_diagram_gpt_confirmed"],
}


def estimate(paths):
    total = sum(p["population"] for p in paths)
    point = variance = 0.0
    for p in paths:
        weight = p["population"] / total
        rate = p["tp"] / p["sample_n"]
        point += weight * rate
        variance += (weight ** 2 * (1 - p["sample_n"] / p["population"])
                     * rate * (1 - rate) / (p["sample_n"] - 1))
    half = Z * math.sqrt(variance)
    return {"records": total, "audited": sum(p["sample_n"] for p in paths),
            "precision": point, "ci95": [point - half, point + half]}


def main():
    with COUNTS_PATH.open(newline="", encoding="utf-8") as handle:
        rows = {r["cascade_path"]: {"population": int(r["population"]),
                                    "sample_n": int(r["sample_n"]), "tp": int(r["tp"])}
                for r in csv.DictReader(handle)}
    result = {name: estimate([rows[p] for p in members]) for name, members in SUBSETS.items()}

    # Values reported in the paper (Table "Blind human audit of released diagrams").
    expected = {"full_release": (93.67, 90.11, 97.22), "core": (95.2, 91.3, 99.1),
                "remaining": (79.3, 73.1, 85.5)}
    for name, (point, lo, hi) in expected.items():
        digits = 2 if name == "full_release" else 1
        got = result[name]
        assert round(100 * got["precision"], digits) == point, (name, got)
        assert round(100 * got["ci95"][0], digits) == lo, (name, got)
        assert round(100 * got["ci95"][1], digits) == hi, (name, got)

    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, got in result.items():
        lo, hi = got["ci95"]
        print(f"{name:13s} {got['records']:6,d} records, {got['audited']:3d} audited: "
              f"{100 * got['precision']:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}]")


if __name__ == "__main__":
    main()
