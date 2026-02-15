"""Sync on-disk PDF + figure directories to match the current decision in DuckDB.

Optimizations:
- Scan the filesystem once, restricted to */pending_unknown/ where decision changes occur.
- Use Papers_Staging (from latest tmlr-papers.jsonl) as the source of desired decisions.
- Do not move papers across years; keep the existing year folder on disk.

On-disk layouts (relative to FIG_RAG_DIR):
- papers:  OpenReview/TMLR/papers/<year>/<decision>/<platform_id>/main.pdf
- figures: OpenReview/TMLR/figures/<year>/<decision>/<platform_id>/Figure{n}.png
"""

import argparse
import os
import re
import shutil
import duckdb
from tqdm import tqdm

from utils import TMLR_DIR, DB_PATH, FIG_RAG_DIR

PAPERS_BASE = os.path.join(TMLR_DIR, "papers")
FIGURES_BASE = os.path.join(TMLR_DIR, "figures")


def slugify_decision(decision: str) -> str:
    s = decision.strip().lower()
    if s == "pending/unknown":
        return "pending_unknown"
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "pending_unknown"


def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def move_dir(src_dir: str, dst_dir: str, dry_run: bool = False) -> bool:
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    if src_dir == dst_dir:
        return False
    ensure_parent(dst_dir)
    if dry_run:
        print(f"[dry-run] MOVE: {src_dir} -> {dst_dir}")
        return True
    shutil.move(src_dir, dst_dir)
    return True


def build_pending_indices():
    print("⏳ Indexing pending_unknown assets...")

    pdf_index: dict[str, tuple[str, str]] = {}
    if os.path.exists(PAPERS_BASE):
        for year_part in os.listdir(PAPERS_BASE):
            pending_root = os.path.join(PAPERS_BASE, year_part, "pending_unknown")
            if not os.path.isdir(pending_root):
                continue
            for pid in os.listdir(pending_root):
                pdf_dir = os.path.join(pending_root, pid)
                if os.path.isfile(os.path.join(pdf_dir, "main.pdf")):
                    pdf_index.setdefault(pid, (pdf_dir, year_part))

    fig_index: dict[str, tuple[str, str]] = {}
    if os.path.exists(FIGURES_BASE):
        for year_part in os.listdir(FIGURES_BASE):
            pending_root = os.path.join(FIGURES_BASE, year_part, "pending_unknown")
            if not os.path.isdir(pending_root):
                continue
            for pid in os.listdir(pending_root):
                fig_dir = os.path.join(pending_root, pid)
                if os.path.isdir(fig_dir):
                    fig_index.setdefault(pid, (fig_dir, year_part))

    print(f"✅ Indexed {len(pdf_index)} pending PDFs and {len(fig_index)} pending figure dirs.")
    return pdf_index, fig_index


def update_figure_paths(con, pid: str, figure_dir: str) -> int:
    updates = []
    for fname in os.listdir(figure_dir):
        fig_num = fname[len("Figure") : -len(".png")]
        full_path = os.path.join(figure_dir, fname)
        rel_path = os.path.relpath(full_path, start=FIG_RAG_DIR)
        updates.append((rel_path, pid, fig_num))

    if updates:
        con.executemany(
            """
            UPDATE Figures
            SET figure_path = ?
            WHERE platform_id = ? AND figure_number = ?
            """,
            updates,
        )
    return len(updates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    pdf_index, fig_index = build_pending_indices()

    con = duckdb.connect(DB_PATH)
    con.execute(
        "CREATE OR REPLACE TABLE Papers_Staging AS SELECT * FROM read_json_auto('tmlr-papers.jsonl')"
    )
    rows = con.execute("SELECT platform_id, decision FROM Papers_Staging").fetchall()
    if args.limit:
        rows = rows[: args.limit]

    desired_decision = {pid: slugify_decision(dec) for pid, dec in rows}
    candidate_pids = sorted(set(pdf_index.keys()) | set(fig_index.keys()))

    moved_pdfs = 0
    moved_figs = 0
    updated_paths = 0
    eligible = 0
    missing_in_staging = 0

    for pid in tqdm(candidate_pids, desc="Syncing Assets", unit="paper"):
        target_decision = desired_decision.get(pid)
        if target_decision is None:
            missing_in_staging += 1
            continue
        if target_decision == "pending_unknown":
            continue
        eligible += 1

        if pid in pdf_index:
            src_pdf_dir, year = pdf_index[pid]
            dst_pdf_dir = os.path.join(PAPERS_BASE, str(year), target_decision, pid)
            if not os.path.exists(dst_pdf_dir):
                if move_dir(src_pdf_dir, dst_pdf_dir, dry_run=args.dry_run):
                    moved_pdfs += 1

        if pid in fig_index:
            src_fig_dir, year = fig_index[pid]
            dst_fig_dir = os.path.join(FIGURES_BASE, str(year), target_decision, pid)
            if not os.path.exists(dst_fig_dir):
                if move_dir(src_fig_dir, dst_fig_dir, dry_run=args.dry_run):
                    moved_figs += 1
                    if not args.dry_run:
                        updated_paths += update_figure_paths(con, pid, dst_fig_dir)

    con.close()

    print("=" * 30)
    print("--- SYNC SUMMARY ---")
    print(f"Eligible (decision != pending_unknown): {eligible}")
    print(f"Moved PDF Directories: {moved_pdfs}")
    print(f"Moved Figure Directories: {moved_figs}")
    print(f"Updated DB Figure Paths: {updated_paths}")
    print(f"Missing in staging (skip): {missing_in_staging}")
    print("=" * 30)


if __name__ == "__main__":
    main()
