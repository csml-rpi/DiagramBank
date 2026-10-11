"""
Query DuckDB databases to understand CLIP classification distribution across all venues.

Key question: If we want to expand the candidate pool to recover missed diagrams
via a cascade (CLIP -> VLM), how many additional figures would we need to process?

The `confidence` column stores the confidence of the TOP-1 prediction only.
The `figure_type` column stores the top-1 class (diagram, plot, photo, other).
We CANNOT get the diagram softmax score for non-diagram figures from the DB.
"""

import duckdb
import os

FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR", os.path.expanduser("~/fig-rag-dir"))

VENUES = {
    "ICLR":    os.path.join(FIG_RAG_DIR, "OpenReview/ICLR/research.db"),
    "ICML":    os.path.join(FIG_RAG_DIR, "OpenReview/ICML/research.db"),
    "NeurIPS": os.path.join(FIG_RAG_DIR, "OpenReview/NeurIPS/research.db"),
    "TMLR":    os.path.join(FIG_RAG_DIR, "OpenReview/TMLR/research.db"),
}

# VLM cost estimates per image
COST_HAIKU = 0.0009       # Claude Haiku ~$0.0009/image
COST_GPT_MINI = 0.00005   # GPT-4o-mini ~$0.00005/image


def query_venue(db_path, venue_name):
    """Query a single venue database and return raw rows."""
    con = duckdb.connect(db_path, read_only=True)

    # Total counts by figure_type
    type_counts = con.execute("""
        SELECT figure_type, COUNT(*) as cnt
        FROM Figures
        GROUP BY figure_type
        ORDER BY cnt DESC
    """).fetchall()

    # Confidence distribution for ALL figure types in bins
    # Bins: [0, 0.50), [0.50, 0.60), [0.60, 0.70), [0.70, 0.80), [0.80, 0.90), [0.90, 1.00]
    conf_dist = con.execute("""
        SELECT
            figure_type,
            CASE
                WHEN confidence IS NULL THEN 'NULL'
                WHEN confidence < 0.50 THEN '<0.50'
                WHEN confidence < 0.60 THEN '0.50-0.60'
                WHEN confidence < 0.70 THEN '0.60-0.70'
                WHEN confidence < 0.80 THEN '0.70-0.80'
                WHEN confidence < 0.90 THEN '0.80-0.90'
                ELSE '0.90-1.00'
            END as conf_bin,
            COUNT(*) as cnt
        FROM Figures
        GROUP BY figure_type, conf_bin
        ORDER BY figure_type, conf_bin
    """).fetchall()

    # Tier 2 candidate pool sizes at different cutoffs
    tier2_queries = {
        "CLIP=plot (any confidence)": """
            SELECT COUNT(*) FROM Figures WHERE figure_type = 'plot'
        """,
        "CLIP=plot, conf < 0.70": """
            SELECT COUNT(*) FROM Figures WHERE figure_type = 'plot' AND confidence < 0.70
        """,
        "CLIP=plot, conf < 0.60": """
            SELECT COUNT(*) FROM Figures WHERE figure_type = 'plot' AND confidence < 0.60
        """,
        "CLIP!=diagram, conf < 0.70": """
            SELECT COUNT(*) FROM Figures WHERE figure_type != 'diagram' AND confidence < 0.70
        """,
        "CLIP!=diagram, conf < 0.60": """
            SELECT COUNT(*) FROM Figures WHERE figure_type != 'diagram' AND confidence < 0.60
        """,
    }
    tier2_counts = {}
    for label, sql in tier2_queries.items():
        tier2_counts[label] = con.execute(sql).fetchone()[0]

    con.close()
    return type_counts, conf_dist, tier2_counts


def main():
    # Aggregate across all venues
    all_type_counts = {}       # figure_type -> {venue: count}
    all_conf_dist = {}         # (figure_type, conf_bin) -> {venue: count}
    all_tier2 = {}             # cutoff_label -> {venue: count}

    for venue, db_path in VENUES.items():
        print(f"Querying {venue} at {db_path}...")
        type_counts, conf_dist, tier2_counts = query_venue(db_path, venue)

        for ft, cnt in type_counts:
            all_type_counts.setdefault(ft, {})[venue] = cnt

        for ft, cb, cnt in conf_dist:
            all_conf_dist.setdefault((ft, cb), {})[venue] = cnt

        for label, cnt in tier2_counts.items():
            all_tier2.setdefault(label, {})[venue] = cnt

    venues = list(VENUES.keys())

    # ==========================================
    # 1. Total figure counts by figure_type
    # ==========================================
    print("\n" + "=" * 80)
    print("1. TOTAL FIGURE COUNTS BY CLIP TYPE")
    print("=" * 80)

    header = f"{'Type':<12}" + "".join(f"{v:>10}" for v in venues) + f"{'TOTAL':>12}"
    print(header)
    print("-" * len(header))

    grand_total = 0
    type_totals = {}
    for ft in ["diagram", "plot", "photo", "other"]:
        venue_counts = all_type_counts.get(ft, {})
        total = sum(venue_counts.get(v, 0) for v in venues)
        type_totals[ft] = total
        grand_total += total
        row = f"{ft:<12}" + "".join(f"{venue_counts.get(v, 0):>10,}" for v in venues) + f"{total:>12,}"
        print(row)

    print("-" * len(header))
    venue_totals = {v: sum(all_type_counts.get(ft, {}).get(v, 0) for ft in ["diagram", "plot", "photo", "other"]) for v in venues}
    row = f"{'TOTAL':<12}" + "".join(f"{venue_totals[v]:>10,}" for v in venues) + f"{grand_total:>12,}"
    print(row)

    # Percentages
    print(f"\nPercentages of total ({grand_total:,} figures):")
    for ft in ["diagram", "plot", "photo", "other"]:
        pct = type_totals[ft] / grand_total * 100 if grand_total > 0 else 0
        print(f"  {ft:<12} {type_totals[ft]:>8,}  ({pct:.1f}%)")

    # ==========================================
    # 2. Confidence distribution by figure_type
    # ==========================================
    print("\n" + "=" * 80)
    print("2. CONFIDENCE DISTRIBUTION BY CLIP TYPE (all venues combined)")
    print("=" * 80)

    conf_bins_order = ["NULL", "<0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]

    for ft in ["diagram", "plot", "photo", "other"]:
        print(f"\n  --- {ft.upper()} (total: {type_totals.get(ft, 0):,}) ---")
        header2 = f"  {'Confidence':<14}" + "".join(f"{v:>10}" for v in venues) + f"{'TOTAL':>10}" + f"{'%':>8}"
        print(header2)

        ft_total = type_totals.get(ft, 0)
        for cb in conf_bins_order:
            venue_counts = all_conf_dist.get((ft, cb), {})
            total = sum(venue_counts.get(v, 0) for v in venues)
            if total == 0:
                continue
            pct = total / ft_total * 100 if ft_total > 0 else 0
            row = f"  {cb:<14}" + "".join(f"{venue_counts.get(v, 0):>10,}" for v in venues) + f"{total:>10,}" + f"{pct:>7.1f}%"
            print(row)

    # ==========================================
    # 3. Non-diagram figures: cumulative counts below confidence thresholds
    # ==========================================
    print("\n" + "=" * 80)
    print("3. NON-DIAGRAM FIGURES: CUMULATIVE COUNTS BELOW CONFIDENCE THRESHOLDS")
    print("=" * 80)
    print("   (These are potential 'missed diagrams' — lower confidence = more likely borderline)")

    non_diag_types = ["plot", "photo", "other"]

    # Build cumulative from the bins
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    bin_to_upper = {"NULL": None, "<0.50": 0.50, "0.50-0.60": 0.60, "0.60-0.70": 0.70,
                    "0.70-0.80": 0.80, "0.80-0.90": 0.90, "0.90-1.00": 1.01}

    for ft in non_diag_types:
        print(f"\n  --- {ft.upper()} ---")
        print(f"  {'Conf <':<14}{'Count':>10}{'% of type':>12}{'Cumul %':>10}")

        # Get bin totals
        bin_totals = {}
        for cb in conf_bins_order:
            venue_counts = all_conf_dist.get((ft, cb), {})
            bin_totals[cb] = sum(venue_counts.get(v, 0) for v in venues)

        ft_total = type_totals.get(ft, 0)
        cumul = 0
        for thresh in [0.50, 0.60, 0.70, 0.80, 0.90, 1.01]:
            # Which bins contribute to "conf < thresh"?
            for cb, upper in bin_to_upper.items():
                if upper is not None and upper <= thresh and cb not in ["NULL"]:
                    # This bin has values < thresh
                    pass
            # Simpler: just sum the bins whose upper bound <= thresh
            count_below = 0
            if thresh >= 0.50:
                count_below += bin_totals.get("<0.50", 0)
            if thresh >= 0.60:
                count_below += bin_totals.get("0.50-0.60", 0)
            if thresh >= 0.70:
                count_below += bin_totals.get("0.60-0.70", 0)
            if thresh >= 0.80:
                count_below += bin_totals.get("0.70-0.80", 0)
            if thresh >= 0.90:
                count_below += bin_totals.get("0.80-0.90", 0)
            if thresh >= 1.01:
                count_below += bin_totals.get("0.90-1.00", 0)

            pct = count_below / ft_total * 100 if ft_total > 0 else 0
            label = "ALL" if thresh > 1.0 else f"{thresh:.2f}"
            print(f"  {label:<14}{count_below:>10,}{pct:>11.1f}%")

    # ==========================================
    # 4. Tier 2 candidate pool sizes
    # ==========================================
    print("\n" + "=" * 80)
    print("4. TIER 2 CANDIDATE POOL SIZES (figures to re-process with VLM)")
    print("=" * 80)

    header3 = f"{'Cutoff':<35}" + "".join(f"{v:>10}" for v in venues) + f"{'TOTAL':>10}" + f"{'Haiku $':>12}" + f"{'GPT-mini $':>12}"
    print(header3)
    print("-" * len(header3))

    for label in [
        "CLIP=plot (any confidence)",
        "CLIP=plot, conf < 0.70",
        "CLIP=plot, conf < 0.60",
        "CLIP!=diagram, conf < 0.70",
        "CLIP!=diagram, conf < 0.60",
    ]:
        venue_counts = all_tier2.get(label, {})
        total = sum(venue_counts.get(v, 0) for v in venues)
        cost_h = total * COST_HAIKU
        cost_g = total * COST_GPT_MINI
        row = (f"{label:<35}"
               + "".join(f"{venue_counts.get(v, 0):>10,}" for v in venues)
               + f"{total:>10,}"
               + f"${cost_h:>10,.2f}"
               + f"${cost_g:>10,.2f}")
        print(row)

    # ==========================================
    # 5. Summary and recommendations
    # ==========================================
    print("\n" + "=" * 80)
    print("5. SUMMARY: COST-EFFECTIVENESS OF CASCADE TIERS")
    print("=" * 80)

    diagram_count = type_totals.get("diagram", 0)
    total_figures = grand_total
    print(f"\nCurrent dataset: {diagram_count:,} diagrams out of {total_figures:,} total figures ({diagram_count/total_figures*100:.1f}%)")

    # From A1 experiment: CLIP has ~97% precision for diagrams but misses some
    # The cascade would recover missed diagrams from non-diagram classified figures
    print(f"\nTier 1 (CLIP only):    {diagram_count:,} figures classified as diagram")
    print(f"                       Cost: $0 (already done)")

    # Best tier 2 options
    for label in [
        "CLIP=plot, conf < 0.60",
        "CLIP=plot, conf < 0.70",
        "CLIP!=diagram, conf < 0.60",
        "CLIP!=diagram, conf < 0.70",
        "CLIP=plot (any confidence)",
    ]:
        total = sum(all_tier2.get(label, {}).get(v, 0) for v in venues)
        cost_h = total * COST_HAIKU
        cost_g = total * COST_GPT_MINI
        pct_of_total = total / total_figures * 100
        print(f"\nTier 2 ({label}):")
        print(f"  Pool size:  {total:>10,} figures ({pct_of_total:.1f}% of total)")
        print(f"  Haiku cost: ${cost_h:>10,.2f}")
        print(f"  GPT-mini:   ${cost_g:>10,.2f}")

    # Additional context: how many non-diagram figures have low confidence?
    print("\n" + "=" * 80)
    print("6. LOW-CONFIDENCE NON-DIAGRAM FIGURES (combined plot+photo+other)")
    print("=" * 80)

    for thresh_label, thresh_val in [("<0.50", 0.50), ("<0.60", 0.60), ("<0.70", 0.70), ("<0.80", 0.80)]:
        total_low = 0
        for ft in non_diag_types:
            bin_totals = {}
            for cb in conf_bins_order:
                venue_counts = all_conf_dist.get((ft, cb), {})
                bin_totals[cb] = sum(venue_counts.get(v, 0) for v in venues)

            count_below = 0
            if thresh_val >= 0.50:
                count_below += bin_totals.get("<0.50", 0)
            if thresh_val >= 0.60:
                count_below += bin_totals.get("0.50-0.60", 0)
            if thresh_val >= 0.70:
                count_below += bin_totals.get("0.60-0.70", 0)
            if thresh_val >= 0.80:
                count_below += bin_totals.get("0.70-0.80", 0)
            total_low += count_below

        non_diag_total = sum(type_totals.get(ft, 0) for ft in non_diag_types)
        pct = total_low / non_diag_total * 100 if non_diag_total > 0 else 0
        cost_h = total_low * COST_HAIKU
        cost_g = total_low * COST_GPT_MINI
        print(f"  Conf {thresh_label:<8}: {total_low:>8,} figures ({pct:>5.1f}% of non-diagram)  Haiku: ${cost_h:>8,.2f}  GPT-mini: ${cost_g:>8,.2f}")


if __name__ == "__main__":
    main()
