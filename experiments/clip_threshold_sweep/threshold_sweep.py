"""
[A2] CLIP Threshold Sensitivity Analysis.

Uses the annotated subset from [A1] to sweep the CLIP confidence threshold
and show how it affects precision/recall/F1 for the "diagram" class.

Outputs:
  - threshold_sweep.png   — P/R/F1 vs. threshold plot
  - threshold_table.json  — metrics at key thresholds
  - dataset_retention.json — extrapolated dataset-wide retention per threshold
  - report.txt            — human-readable summary

Usage:
    python threshold_sweep.py
"""

import json
import os
from pathlib import Path

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
A1_DIR = SCRIPT_DIR.parent / "clip_router_audit"
SAMPLE_PATH = A1_DIR / "sample.jsonl"
ANNOTATIONS_PATH = A1_DIR / "annotations_tingwen.jsonl"

THRESHOLDS = np.arange(0.50, 0.96, 0.05).round(2).tolist()
KEY_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

# Population counts per (type x confidence bin) from DuckDB — used to
# extrapolate dataset-wide retention. These come from sample.py output.
POPULATION = {
    "diagram": {
        "[0.50,0.60)": 8428, "[0.60,0.70)": 8056, "[0.70,0.80)": 9332,
        "[0.80,0.90)": 13036, "[0.90,1.00]": 48825,
    },
    "plot": {
        "[0.50,0.60)": 10979, "[0.60,0.70)": 12175, "[0.70,0.80)": 15704,
        "[0.80,0.90)": 26412, "[0.90,1.00]": 238029,
    },
    "photo": {
        "[0.50,0.60)": 4356, "[0.60,0.70)": 4830, "[0.70,0.80)": 6038,
        "[0.80,0.90)": 8932, "[0.90,1.00]": 25175,
    },
    "other": {
        "[0.50,0.60)": 2840, "[0.60,0.70)": 2832, "[0.70,0.80)": 3085,
        "[0.80,0.90)": 4170, "[0.90,1.00]": 6044,
    },
}

BIN_EDGES = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.001)]
BIN_LABELS = ["[0.50,0.60)", "[0.60,0.70)", "[0.70,0.80)", "[0.80,0.90)", "[0.90,1.00]"]


# ── Load data ───────────────────────────────────────────────────────────
def load_data():
    """Load sample metadata and human annotations, return merged list."""
    # Load sample metadata
    meta = {}
    with open(SAMPLE_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = f"{rec['platform_id']}|{rec['figure_number']}"
            meta[key] = rec

    # Load human annotations
    annotations = {}
    with open(ANNOTATIONS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            key = f"{rec['platform_id']}|{rec['figure_number']}"
            label = rec.get("label_tingwen", "").strip().lower()
            if label in ("diagram", "plot", "photo", "other"):
                annotations[key] = label

    # Merge
    data = []
    for key in meta:
        if key in annotations:
            m = meta[key]
            data.append({
                "clip_type": m["figure_type"],
                "clip_conf": m["confidence"],
                "confidence_bin": m["confidence_bin"],
                "human_label": annotations[key],
            })
    return data


# ── Metrics ─────────────────────────────────────────────────────────────
def compute_metrics_at_threshold(data, tau):
    """Compute diagram-filter metrics at a given threshold tau.
    Filter: CLIP says 'diagram' AND confidence >= tau."""
    tp = fp = fn = tn = 0
    for d in data:
        predicted_diagram = (d["clip_type"] == "diagram" and d["clip_conf"] >= tau)
        is_diagram = (d["human_label"] == "diagram")
        if predicted_diagram and is_diagram:
            tp += 1
        elif predicted_diagram and not is_diagram:
            fp += 1
        elif not predicted_diagram and is_diagram:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": tau,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "included": tp + fp,
    }


def estimate_dataset_retention(tau):
    """Estimate how many figures the dataset would retain at threshold tau.
    Only figures with clip_type='diagram' AND confidence >= tau are included."""
    retained = 0
    for bin_label, (lo, hi) in zip(BIN_LABELS, BIN_EDGES):
        if lo >= tau:
            # Entire bin is above threshold
            retained += POPULATION["diagram"][bin_label]
        elif hi > tau > lo:
            # Threshold falls within this bin — estimate proportionally
            frac = (hi - tau) / (hi - lo)
            retained += int(POPULATION["diagram"][bin_label] * frac)
    total_all = sum(v for cat in POPULATION.values() for v in cat.values())
    total_diagram = sum(POPULATION["diagram"].values())
    return {
        "threshold": tau,
        "estimated_diagrams_retained": retained,
        "total_clip_diagrams": total_diagram,
        "retention_pct": round(100 * retained / total_diagram, 1),
        "total_figures": total_all,
    }


# ── Plotting ────────────────────────────────────────────────────────────
def plot_sweep(results, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available, skipping {save_path}")
        return

    taus = [r["threshold"] for r in results]
    prec = [r["precision"] for r in results]
    rec = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(taus, prec, "o-", color="#2196F3", label="Precision", linewidth=2, markersize=5)
    ax.plot(taus, rec, "s-", color="#FF9800", label="Recall", linewidth=2, markersize=5)
    ax.plot(taus, f1s, "^-", color="#4CAF50", label="F1", linewidth=2, markersize=5)

    # Mark the chosen operating point
    ax.axvline(x=0.85, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="τ = 0.85")

    # Annotate P/R/F1 at 0.85
    idx_85 = next(i for i, r in enumerate(results) if r["threshold"] == 0.85)
    r85 = results[idx_85]
    ax.annotate(
        f"P={r85['precision']:.2f}, R={r85['recall']:.2f}, F1={r85['f1']:.2f}",
        xy=(0.85, r85["f1"]), xytext=(0.72, r85["f1"] + 0.08),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9, color="red",
    )

    ax.set_xlabel("Confidence Threshold (τ)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("CLIP Diagram Filter: Precision / Recall / F1 vs. Threshold", fontsize=13)
    ax.set_xlim(0.48, 0.97)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="center left", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    data = load_data()
    print(f"Loaded {len(data)} annotated figures")

    # Sweep thresholds
    results = []
    for tau in THRESHOLDS:
        m = compute_metrics_at_threshold(data, tau)
        results.append(m)

    # Key threshold table
    key_results = [compute_metrics_at_threshold(data, tau) for tau in KEY_THRESHOLDS]

    # Dataset retention estimates
    retention = [estimate_dataset_retention(tau) for tau in KEY_THRESHOLDS]

    # Save threshold table
    table_path = SCRIPT_DIR / "threshold_table.json"
    with open(table_path, "w") as f:
        json.dump(key_results, f, indent=2)
    print(f"Saved {table_path}")

    # Save retention estimates
    retention_path = SCRIPT_DIR / "dataset_retention.json"
    with open(retention_path, "w") as f:
        json.dump(retention, f, indent=2)
    print(f"Saved {retention_path}")

    # Plot
    plot_sweep(results, SCRIPT_DIR / "threshold_sweep.png")

    # Report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("[A2] CLIP Threshold Sensitivity Analysis — Results")
    report_lines.append("=" * 60)
    report_lines.append(f"Based on {len(data)} annotated figures from [A1]")
    report_lines.append("")

    report_lines.append("--- Precision / Recall / F1 at Key Thresholds ---")
    report_lines.append(f"  {'τ':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}  {'Incl':>5s}")
    for r in key_results:
        report_lines.append(
            f"  {r['threshold']:6.2f}  {r['precision']:6.3f}  {r['recall']:6.3f}  {r['f1']:6.3f}"
            f"  {r['tp']:4d}  {r['fp']:4d}  {r['fn']:4d}  {r['included']:5d}"
        )

    report_lines.append("")
    report_lines.append("--- Estimated Dataset-Wide Retention ---")
    report_lines.append(f"  {'τ':>6s}  {'Diagrams':>10s}  {'of Total':>10s}  {'Retention':>10s}")
    for ret in retention:
        report_lines.append(
            f"  {ret['threshold']:6.2f}  {ret['estimated_diagrams_retained']:>10d}"
            f"  {ret['total_clip_diagrams']:>10d}"
            f"  {ret['retention_pct']:>9.1f}%"
        )

    report_lines.append("")
    report_lines.append("--- Summary ---")
    r85 = next(r for r in key_results if r["threshold"] == 0.85)
    ret85 = next(r for r in retention if r["threshold"] == 0.85)
    report_lines.append(f"At τ=0.85: Precision={r85['precision']:.3f}, Recall={r85['recall']:.3f}, F1={r85['f1']:.3f}")
    report_lines.append(f"Estimated diagrams retained: {ret85['estimated_diagrams_retained']:,} / {ret85['total_clip_diagrams']:,} ({ret85['retention_pct']}%)")

    report_path = SCRIPT_DIR / "report.txt"
    report_text = "\n".join(report_lines)
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"Saved {report_path}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
