# DiagramBank/NeurIPS — Runbook (Manual Pipeline)

Some stages take **hours** (`extract_figures.py`, `extract_context.py`), and the PDF download stage needs to be **re-run until failures stabilize**.

## Prerequisites

- Activate the conda environment:
  ```bash
  conda activate DiagramBank
  ```
- `FIG_RAG_DIR` is set
- DuckDB database lives at:
  - `${FIG_RAG_DIR}/OpenReview/NeurIPS/research.db`

## Stage 1 — Scrape OpenReview → JSONL

```bash
python neurips.py
```

### Incremental updates for future runs

If you are only adding **new papers** (e.g., a new year), you typically do **not** need to rescrape all historical years.

Recommended workflow:
1) In `neurips.py`, comment out old years and keep only the year(s) you want.
2) Run `python neurips.py` to regenerate `neurips-papers.jsonl` (staging snapshot).
3) Run `python build.py` to recreate `Papers_Staging` and insert new `platform_id`s into `Papers`.

## Stage 2 — Download PDFs (repeat until stable)

```bash
python download.py
```

Stop condition: `failed-downloads.json` stops changing.

## Stage 3 — Sync DuckDB (`Papers`)

```bash
python build.py
```

## Stage 4 — Extract figures (long-running)

```bash
nohup python extract_figures.py >/dev/null 2>&1 &
```

## Stage 5 — Extract figure contexts (long-running)

```bash
nohup python extract_context.py >/dev/null 2>&1 &
```

## Stage 6 — Classify figures

```bash
python classify.py
```

## Occasional maintenance — Sync/move assets after decision updates

```bash
python sync_assets.py --dry_run
python sync_assets.py
```
