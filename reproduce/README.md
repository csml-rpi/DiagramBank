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

## CLIP checkpoint selection and run records

All four `classify.py` scripts use `ViT-B-32` and select one checkpoint explicitly.
The default is `datacomp_xl_s13b_b90k`. Loading failure stops the run; it no longer
switches to LAION automatically. Choose LAION deliberately when it is the
intended configuration:

```sh
python classify.py --clip-pretrained laion2b_s34b_b79k
```

Each invocation creates a new `classification-runs/<run-id>/` directory next to
the venue database, or a new directory supplied with `--run-dir`. Existing run
directories cannot be overwritten. Its files are:

- `manifest.json`: requested and loaded checkpoint, SHA-256 of named model
  tensors, preprocessing, full prompt lists, class order, batch size, scoring,
  CPU/FP32 execution, runtime package versions, script hashes, timestamps, status,
  database hashes before/after, and output hashes;
- `candidates.jsonl`: selected figure keys and paths;
- `image_inputs.jsonl`: hashes of processed images and reasons for skipped files;
- `predictions.jsonl`: figure keys, class labels, and confidence values, written
  before database changes.

Only records with an empty or SQL NULL figure type are selected. The classifier
retains the four prompt centroids, batch size 16, and softmax over 100 times
cosine similarity. Database updates commit in one transaction; an update failure
rolls them all back. A manifest is marked `completed` after the database closes
and its output hash is recorded. Interrupted inference leaves an incomplete
status, which must not be interpreted as a successful classification run.

These records apply to **new runs**. The original release lacks a saved record
identifying which checkpoint its fallback-capable loaders actually used.
Existing labels, release membership, and reported evaluations are unchanged.
The local regression tests use synthetic model tensors and temporary images and
databases; they do not download weights or rerun scientific classification.

Checkpoint tags and loader options were checked against the
[OpenCLIP 3.2.0 registry](https://github.com/mlfoundations/open_clip/blob/v3.2.0/src/open_clip/pretrained.py#L160-L185)
and [factory implementation](https://github.com/mlfoundations/open_clip/blob/v3.2.0/src/open_clip/factory.py#L761-L888).
