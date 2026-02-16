# DiagramBank/ICLR — Runbook (Manual Pipeline)

Some stages take **hours** (`extract_figures.py`, `extract_context.py`), and the PDF download stage needs to be **re-run until failures stabilize**.

## Prerequisites

- Activate the conda environment:
  ```bash
  conda activate DiagramBank
  ```
- `FIG_RAG_DIR` is set (this is where all on-disk assets + DuckDB live)

## Stage 1 — Scrape OpenReview → JSONL

Creates the latest paper metadata snapshot (`Papers_Staging`).

```bash
python iclr.py
```

### Incremental updates for future runs

If you are only adding **new papers** (e.g., a new conference year), you typically do **not** need to rescrape all historical years.

Recommended workflow:
1) In `iclr.py`, **comment out old years** in `get_invitations_map()` and keep only the year(s) you want to add/update.
2) Run `python iclr.py` to regenerate `iclr-papers.jsonl` (this becomes the new staging snapshot).
3) Run `python build.py` to:
   - recreate `Papers_Staging` from `iclr-papers.jsonl`
   - **insert** any new `platform_id`s from `Papers_Staging` into the long-term `Papers` table
   - update `decision` for existing papers when it changes

After that, the rest of the pipeline (`extract_figures.py`, `extract_context.py`, etc.) reads from **`Papers_Staging`** (the latest snapshot).

Outputs (in this folder):
- `iclr-papers.jsonl`
- `iclr-urls.json`
- `iclr-replies.json`

## Stage 2 — Download PDFs (repeat until stable)

Downloads PDFs into the canonical on-disk layout:
- `.../OpenReview/ICLR/papers/<year>/<decision>/<platform_id>/main.pdf`

```bash
python download.py
```

This stage often needs to be run multiple times due to rate limits and transient network failures.

**Stop condition:** `failed-downloads.json` stops changing between runs.

Suggested checks:
```bash
wc -l failed-downloads.json
md5sum failed-downloads.json
```

## Stage 3 — Sync DuckDB (`Papers`)

Inserts new `platform_id`s and updates `decision` when it changes.

```bash
python build.py
```

## Stage 4 — Extract figures (long-running)

Runs PDFFigures2 in batches and inserts results into DuckDB table `Figures`.

Recommended (run in background; silence output):
```bash
nohup python extract_figures.py >/dev/null 2>&1 &
```

Expected runtime: hours (depends on number of papers + CPU cores).

## Stage 5 — Extract figure contexts (long-running)

Extracts textual context from PDFs for the figures already in DuckDB.

Recommended (run in background; silence output):
```bash
nohup python extract_context.py >/dev/null 2>&1 &
```

Expected runtime: hours.

## Stage 6 — Classify figures

Classifies extracted figures (diagram/plot/photo/other) and writes to DuckDB.

```bash
python classify.py
```

## Occasional maintenance — Sync/move assets after decision updates

Only needed if you updated decisions and want to **move** already-downloaded PDFs and figures out of `pending_unknown/` into their new decision folders (without re-downloading or re-extracting).

Dry run:
```bash
python sync_assets.py --dry_run
```

Apply moves:
```bash
python sync_assets.py
```
