# DiagramBank

Code and evaluation artifacts for **DiagramBank: Draw from the Great, Draw Your Best** ([arXiv:2604.20857](https://arxiv.org/abs/2604.20857)).

DiagramBank contains 57,100 schematic diagrams from 26,751 accepted papers at ICLR, NeurIPS, ICML, and TMLR. Each record pairs a diagram with its caption, the body paragraphs that cite it (91.16% of records), and the paper's title and abstract, together with provenance and per-record filtering labels.

- **Dataset:** https://huggingface.co/datasets/ghzlmc/DiagramBank
- **Dataset card:** [DATASET_CARD.md](DATASET_CARD.md)
- **Record fields:** [docs/schema.md](docs/schema.md)

## Setup

```bash
conda env create --file environment.yml
conda activate DiagramBank
export FIG_RAG_DIR=/path/to/scratch   # where the release is downloaded and extracted
```

## Download and verification

```bash
python huggingface/download_diagrambank.py                 # accepted-paper images + core files (default)
python huggingface/download_diagrambank.py --subset all    # images of accepted and rejected papers + core files
python huggingface/download_diagrambank.py --no-core       # images only
```

Core files are `data.jsonl`, the DuckDB databases, and the FAISS indices. The downloader computes the SHA-256 of every downloaded payload, compares it with the commit-pinned manifest `huggingface/DiagramBank_remote_manifest.json` (112 payloads with sizes and SHA-256 values; also `huggingface/SHA256SUMS`), and extracts only verified payloads.

## Repository map

| Path | Paper |
|---|---|
| `reproduce/` | Data collection: OpenReview metadata and PDFs, PDFFigures 2.0 figure and caption extraction, PyMuPDF context extraction, and CLIP classification. `figure_context.py` and `clip_provenance.py` are the revised context parser and CLIP run records; the release keeps its original contexts and labels. |
| `cascade/` | Filtering: Tier 1/Tier 2 routing, Claude Haiku 4.5 and GPT-5.4-mini judges, voting, GPT-5.4 confirmation, and the saved votes for all 94,176 candidates. `python cascade/finalize.py --output-dir <dir> --verify-release-db $FIG_RAG_DIR/faiss/research.db` replays them and checks that the 57,100 released keys are reproduced exactly. |
| `release_supplement/` | `cascade_labels.jsonl`, `paper_groups.jsonl`, and the 55 out-of-label Haiku outputs. |
| `tools/` | Release export and supplement builder. |
| `experiments/final_release_audit/` | Final-release audit estimates for the full release, DiagramBank-Core, and the remaining records. |
| `experiments/context_review_50/` | 50-record diagram–text alignment review and release-wide context checks. |
| `experiments/retrieval_ablation/` | Retrieval field ablation: 2,000 frozen targets and queries, metrics, and retrieval logs. |
| `experiments/same_paper_context_swap/` | Same-paper context swap control. |
| `experiments/human_study/` | 20-case blind human study: briefs, references, outputs, frozen manifest, and votes. |
| `experiments/paperbananabench/generation_reference_protocol.json` | Reference-pool settings of the PaperBananaBench study (pool filters, exclusions, retrieval, reference inputs, generation). |
| `huggingface/` | Downloader, manifest, and SHA-256 list. |

## Citation

```bibtex
@article{yue2026diagrambank,
  title={DiagramBank: Draw from the Great, Draw Your Best},
  author={Yue, Ling and Wang, Jiaying and Lu, Zhexi and Zhu, Mingzhi and Zhang, Tingwen and Xu, Zhen and Pan, Shaowu},
  journal={arXiv preprint arXiv:2604.20857},
  year={2026}
}
```

When you use specific diagrams, please also cite their source papers; every record carries a source URL, and all but the ICML 2023 records carry a BibTeX entry.

## License

The code is released under the MIT License. Figure rights remain with the source papers; see the dataset card for attribution and use.
