"""
[A1] Evaluate CLIP filter quality using human + LLM annotations.

Reads the 3 annotation files (1 human, 2 LLM), computes:
  1. Human ground truth (Tingwen's labels)
  2. CLIP precision/recall/F1 for "diagram" class at threshold 0.85
  3. Per-category precision/recall/F1 (macro & per-class)
  4. Filtering rate, false-positive rate, false-negative rate
  5. Confusion matrix (CLIP vs. human consensus)
  6. Inter-annotator agreement (Cohen's kappa) for all 6 judge pairs

Outputs:
  - results.json           — all metrics in machine-readable form
  - confusion_matrix.png   — CLIP vs. human consensus
  - agreement_matrix.png   — pairwise Cohen's kappa heatmap
  - report.txt             — human-readable summary for paper writing

Usage:
    python evaluate.py

Requires: annotations_tingwen.jsonl,
          annotations_openai.jsonl, annotations_anthropic.jsonl
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
CATEGORIES = ["diagram", "plot", "photo", "other"]
THRESHOLD = 0.85

# ── Annotation file paths ─────���─────────────────────────────────────────
ANNOTATION_FILES = {
    "tingwen":   (SCRIPT_DIR / "annotations_tingwen.jsonl",   "label_tingwen"),
    "openai":    (SCRIPT_DIR / "annotations_openai.jsonl",     "label_openai"),
    "anthropic": (SCRIPT_DIR / "annotations_anthropic.jsonl",  "label_anthropic"),
}

HUMAN_JUDGES = ["tingwen"]
LLM_JUDGES = ["openai", "anthropic"]
ALL_JUDGES = HUMAN_JUDGES + LLM_JUDGES

# Display names for plots and reports
DISPLAY_NAMES = {
    "tingwen": "Human",
    "openai": "GPT-5.4-mini",
    "anthropic": "Haiku 4.5",
    "clip": "CLIP",
}


# ── Load annotations ──────────────────────────────────────��─────────────
def load_annotations(name, path, label_field):
    """Load annotations, keyed by platform_id|figure_number."""
    records = {}
    if not path.exists():
        print(f"WARNING: {path} not found, skipping {name}")
        return records
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                key = f"{rec['platform_id']}|{rec['figure_number']}"
                label = rec.get(label_field, "").strip().lower()
                if label in CATEGORIES:
                    records[key] = label
                else:
                    print(f"  WARNING [{name}] unexpected label '{label}' for {key}")
    return records


# ── Metrics ─────────────────────────────────────────────────────────────
def precision_recall_f1(y_true, y_pred, positive_class):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p == positive_class)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p == positive_class)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p != positive_class)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def confusion_matrix(y_true, y_pred, labels):
    """Row = true, Col = predicted."""
    idx = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t], idx[p]] += 1
    return mat


def cohens_kappa(labels_a, labels_b, keys):
    """Compute Cohen's kappa between two sets of labels."""
    a = [labels_a[k] for k in keys if k in labels_a and k in labels_b]
    b = [labels_b[k] for k in keys if k in labels_a and k in labels_b]
    if len(a) == 0:
        return float("nan"), 0

    n = len(a)
    # Observed agreement
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # Expected agreement
    count_a = Counter(a)
    count_b = Counter(b)
    all_labels = set(count_a.keys()) | set(count_b.keys())
    pe = sum((count_a.get(l, 0) / n) * (count_b.get(l, 0) / n) for l in all_labels)

    if pe == 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)
    return kappa, n


# ── Plotting ────────���─────────────────────────────���─────────────────────
def plot_confusion_matrix(mat, labels, title, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available, skipping {save_path}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("CLIP Predicted Label")
    ax.set_ylabel("Human Ground Truth Label")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if mat[i, j] > mat.max() / 2 else "black"
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color=color, fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_agreement_matrix(kappa_dict, judges, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available, skipping {save_path}")
        return

    display = [DISPLAY_NAMES.get(j, j) for j in judges]
    n = len(judges)
    mat = np.ones((n, n))
    for i, ja in enumerate(judges):
        for j, jb in enumerate(judges):
            if i != j:
                key = f"{ja}_vs_{jb}" if f"{ja}_vs_{jb}" in kappa_dict else f"{jb}_vs_{ja}"
                mat[i, j] = kappa_dict.get(key, {}).get("kappa", float("nan"))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, interpolation="nearest", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(display, fontsize=11)
    ax.set_xlabel("Judge", fontsize=12)
    ax.set_ylabel("Judge", fontsize=12)
    ax.set_title("Pairwise Cohen's κ", fontsize=13)

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            text = f"{val:.3f}" if not np.isnan(val) else "—"
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ── Main ──────────────────────────────────────────���─────────────────────
def main():
    # 1. Load all annotations
    all_labels = {}
    for name, (path, field) in ANNOTATION_FILES.items():
        all_labels[name] = load_annotations(name, path, field)
        print(f"Loaded {name}: {len(all_labels[name])} annotations")

    # 2. Load sample metadata (for CLIP labels and confidence)
    sample_meta = {}
    with open(SCRIPT_DIR / "sample.jsonl") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                key = f"{rec['platform_id']}|{rec['figure_number']}"
                sample_meta[key] = rec

    # 2b. Add CLIP as a judge (using its assigned figure_type labels)
    all_labels["clip"] = {}
    for key, meta in sample_meta.items():
        clip_type = meta["figure_type"]
        if clip_type in CATEGORIES:
            all_labels["clip"][key] = clip_type
    print(f"Loaded clip: {len(all_labels['clip'])} labels (from sample metadata)")

    # 3. Determine common keys (figures annotated by all available judges)
    all_judge_names = ALL_JUDGES + ["clip"]
    available_judges = [j for j in all_judge_names if len(all_labels.get(j, {})) > 0]
    if len(available_judges) == 0:
        print("ERROR: No annotations found. Run the judge scripts first.")
        sys.exit(1)

    common_keys = set(sample_meta.keys())
    for j in available_judges:
        common_keys &= set(all_labels[j].keys())
    common_keys = sorted(common_keys)
    print(f"\nCommon keys across {available_judges}: {len(common_keys)}")

    # 4. Human ground truth (single annotator: Tingwen)
    human_available = [j for j in HUMAN_JUDGES if j in available_judges]
    if len(human_available) >= 1:
        ground_truth_judge = human_available[0]
        print(f"Using {ground_truth_judge} as ground truth")
        consensus = {k: all_labels[ground_truth_judge][k] for k in common_keys
                     if k in all_labels[ground_truth_judge]}
        disagreements = []
    else:
        print("No human annotations found. Using all available judges for agreement only.")
        consensus = {}
        disagreements = []

    # 5. CLIP evaluation against human consensus
    results = {}

    if consensus:
        eval_keys = [k for k in common_keys if k in consensus]

        # Build CLIP predictions at threshold
        clip_labels = []     # CLIP-assigned type
        clip_diagram = []    # binary: does CLIP say "diagram" with conf >= threshold?
        human_labels = []
        human_diagram = []   # binary: is ground truth "diagram"?

        for k in eval_keys:
            meta = sample_meta[k]
            clip_type = meta["figure_type"]
            clip_conf = meta["confidence"]
            human_type = consensus[k]

            clip_labels.append(clip_type)
            human_labels.append(human_type)

            # Binary: CLIP says diagram AND confidence >= threshold
            clip_diagram.append(clip_type == "diagram" and clip_conf >= THRESHOLD)
            human_diagram.append(human_type == "diagram")

        # 5a. Per-category metrics (CLIP category vs human, ignoring threshold)
        per_category = {}
        for cat in CATEGORIES:
            per_category[cat] = precision_recall_f1(human_labels, clip_labels, cat)
        results["per_category"] = per_category

        # Macro-averaged
        macro_p = np.mean([per_category[c]["precision"] for c in CATEGORIES])
        macro_r = np.mean([per_category[c]["recall"] for c in CATEGORIES])
        macro_f1 = np.mean([per_category[c]["f1"] for c in CATEGORIES])
        results["macro"] = {"precision": macro_p, "recall": macro_r, "f1": macro_f1}

        # 5b. Binary diagram filter (type=diagram AND conf>=0.85)
        binary_prf = precision_recall_f1(human_diagram, clip_diagram, True)
        results["diagram_filter_0.85"] = binary_prf

        # Filtering rate: fraction excluded by the filter
        total_included = sum(clip_diagram)
        results["filtering_rate"] = 1.0 - total_included / len(clip_diagram)
        results["false_positive_rate"] = binary_prf["fp"] / max(1, sum(not h for h in human_diagram))
        results["false_negative_rate"] = binary_prf["fn"] / max(1, sum(h for h in human_diagram))

        # 5c. Confusion matrix
        cm = confusion_matrix(human_labels, clip_labels, CATEGORIES)
        results["confusion_matrix"] = cm.tolist()

        plot_confusion_matrix(cm, CATEGORIES, "CLIP vs. Human Consensus", SCRIPT_DIR / "confusion_matrix.png")

    # 6. Inter-annotator agreement (all pairs)
    kappa_results = {}
    for i, ja in enumerate(available_judges):
        for jb in available_judges[i + 1:]:
            pair_keys = [k for k in common_keys if k in all_labels[ja] and k in all_labels[jb]]
            kappa, n = cohens_kappa(all_labels[ja], all_labels[jb], pair_keys)
            pair_name = f"{ja}_vs_{jb}"
            kappa_results[pair_name] = {"kappa": round(kappa, 4), "n": n}
    results["agreement"] = kappa_results

    plot_agreement_matrix(kappa_results, available_judges, SCRIPT_DIR / "agreement_matrix.png")

    # 7. Save results
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = SCRIPT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nSaved results to {results_path}")

    # 8. Human-readable report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("[A1] CLIP Filter Validation — Results")
    report_lines.append("=" * 60)
    report_lines.append(f"Evaluated on {len(eval_keys) if consensus else 0} figures")
    report_lines.append(f"Judges: {', '.join(DISPLAY_NAMES.get(j, j) for j in available_judges)}")

    if consensus:
        report_lines.append("")
        report_lines.append("--- Diagram Filter (type=diagram, conf>=0.85) ---")
        d = results["diagram_filter_0.85"]
        report_lines.append(f"  Precision:  {d['precision']:.3f}")
        report_lines.append(f"  Recall:     {d['recall']:.3f}")
        report_lines.append(f"  F1:         {d['f1']:.3f}")
        report_lines.append(f"  TP={d['tp']}, FP={d['fp']}, FN={d['fn']}")
        report_lines.append(f"  Filtering rate:     {results['filtering_rate']:.3f}")
        report_lines.append(f"  False positive rate: {results['false_positive_rate']:.3f}")
        report_lines.append(f"  False negative rate: {results['false_negative_rate']:.3f}")

        report_lines.append("")
        report_lines.append("--- Per-Category (CLIP label vs. human, no threshold) ---")
        report_lines.append(f"  {'Category':>10s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'TP':>4s}  {'FP':>4s}  {'FN':>4s}")
        for cat in CATEGORIES:
            c = per_category[cat]
            report_lines.append(f"  {cat:>10s}  {c['precision']:6.3f}  {c['recall']:6.3f}  {c['f1']:6.3f}  {c['tp']:4d}  {c['fp']:4d}  {c['fn']:4d}")
        report_lines.append(f"  {'MACRO':>10s}  {macro_p:6.3f}  {macro_r:6.3f}  {macro_f1:6.3f}")

        report_lines.append("")
        report_lines.append("--- Confusion Matrix (rows=human, cols=CLIP) ---")
        report_lines.append(f"  {'':>10s}  " + "  ".join(f"{c:>8s}" for c in CATEGORIES))
        for i, cat in enumerate(CATEGORIES):
            row = "  ".join(f"{cm[i, j]:8d}" for j in range(len(CATEGORIES)))
            report_lines.append(f"  {cat:>10s}  {row}")

    report_lines.append("")
    report_lines.append("--- Inter-Annotator Agreement (Cohen's kappa) ---")
    for pair, vals in kappa_results.items():
        # Format with display names
        parts = pair.split("_vs_")
        display_pair = f"{DISPLAY_NAMES.get(parts[0], parts[0])} vs. {DISPLAY_NAMES.get(parts[1], parts[1])}"
        report_lines.append(f"  {display_pair:>30s}: κ = {vals['kappa']:.4f}  (n={vals['n']})")

    report_path = SCRIPT_DIR / "report.txt"
    report_text = "\n".join(report_lines)
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"Saved report to {report_path}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
