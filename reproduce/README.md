# DiagramBank (reproduce)

This folder contains end-to-end pipelines for building the DiagramBank datasets for multiple venues.

## Prerequisites

- Activate the conda environment:
  ```bash
  conda activate DiagramBank
  ```
- Set `FIG_RAG_DIR` to a writable scratch directory (stores PDFs, extracted figures, and DuckDB files):
  ```bash
  export FIG_RAG_DIR=/path/to/scratch
  ```
- Build / install **pdffigures2** and set:
  ```bash
  export PDFFIGURES2_JAR=/path/to/pdffigures2/target/scala-2.12/pdffigures2-assembly-*.jar
  ```
  See `pdffigures2.md` for setup.

## Venue pipelines

Each venue lives in its own folder:
- `ICLR/`
- `ICML/`
- `NeurIPS/`
- `TMLR/`

Each venue has:
- `<venue>.py` — scrape metadata from OpenReview into `<venue>-papers.jsonl`
- `download.py` — download PDFs (rerun until failures stabilize)
- `build.py` — create `Papers_Staging` from the latest JSONL; insert new papers into `Papers`; update decisions
- `extract_figures.py` — run pdffigures2 and insert into `Figures`
- `extract_context.py` — extract figure contexts from PDFs into `Figures.figure_context`
- `classify.py` — classify figures and update `Figures.figure_type`
- `sync_assets.py` — occasional maintenance to move assets out of `pending_unknown/`
- `REPRODUCE.md` — venue-specific runbook

Start with the venue-specific `REPRODUCE.md`.

## Raw metadata snapshots

Large raw OpenReview metadata snapshots are hosted on Hugging Face instead of GitHub.
Download them before running from an existing snapshot:

```bash
python ../huggingface/download_diagrambank.py --metadata-only
```

This extracts files such as `reproduce/ICLR/iclr-papers.jsonl` and
`reproduce/ICML/icml-replies.json` under `$FIG_RAG_DIR`. Copy or symlink the
needed venue files into the corresponding `reproduce/<venue>/` folder if you
want to resume from the hosted snapshot instead of scraping from scratch.
