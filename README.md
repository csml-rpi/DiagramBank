# DiagramBank

**DiagramBank: Draw from the Great, Draw Your Best** ([arXiv:2604.20857](https://arxiv.org/abs/2604.20857))

DiagramBank is a diagram–text reference resource of **57,100 schematic diagrams** from 26,751 accepted papers at ICLR, NeurIPS, ICML, and TMLR. Each diagram is paired with its caption, the paper's title and abstract, and, for 91.16% of records, the body paragraphs that cite it, together with provenance and per-record filtering labels. It is designed as a pool of reference examples for diagram retrieval and automated diagram generation.

---

## Overview

![DiagramBank Overview](DiagramBank.png)

---

## Motivation

Creating a scientific diagram takes substantial effort in content organization, layout design, and visual refinement. Automated diagram generation systems can reduce this work, and reference examples guide them by showing how scientific ideas are organized and presented visually. Existing curated collections are small, however: PaperBanana retrieves references from a pool of 292 curated diagrams, and the FigureBench evaluation set holds 3,300 pairs.

Diagrams from accepted papers are a much larger source of such examples. DiagramBank collects them with text at several granularities (paper title and abstract, figure caption, and the paragraphs that cite the figure), so a system can retrieve references at the level it needs. Under a fixed generation pipeline, replacing PaperBanana's curated reference pool with DiagramBank raises the overall PaperBanana score from 4.62 to 7.88 on PaperBananaBench.

---

## What's in DiagramBank

Each diagram record carries both **figure-level** and **paper-level** information:

- Diagram image and caption
- Figure context: body paragraphs that cite the figure (52,052 records)
- Paper title and abstract
- OpenReview metadata: venue, year, decision, authors, keywords and subject areas, reviewer scores when available, source URL, and BibTeX (all records except ICML 2023)
- Filtering labels: CLIP label and confidence, the labels of the two VLM judges and the GPT-5.4 tiebreak, and the filtering path, so you can select subsets

The Hugging Face release provides the image archives and the core retrieval artifacts (`data.jsonl`, FAISS indices, and DuckDB databases), so large files stay out of this GitHub repository.

### Release snapshot

- Primary release: **57,100** cascade-filtered schematic diagrams from 26,751 accepted papers
- Venue counts: ICLR 20,516 (2017–2026); NeurIPS 19,655 (2021–2022, 2024–2025); ICML 11,267 (2023–2025); TMLR 5,662 (2022–2026)
- Filtering paths (`cascade_path` in `data.jsonl`): `t1_unanimous` 46,524; `t1_majority` 3,645; `t1_minority_gpt_tiebreak` 1,865; `t2_vlm_consensus_gpt_confirmed` 5,066. `release_supplement/cascade_labels.jsonl` names the first three `t1_unanimous_diagram`, `t1_majority_diagram`, and `t1_minority_diagram_gpt_confirmed`.
- Quality estimate: 93.67% precision with 95% CI [90.11%, 97.22%], from a blind audit of 500 records
- **DiagramBank-Core**: the 51,590 records on `t1_unanimous` and `t2_vlm_consensus_gpt_confirmed`, with 95.2% estimated precision [91.3%, 99.1%]

See [docs/schema.md](docs/schema.md) for the full `data.jsonl` schema and [DATASET_CARD.md](DATASET_CARD.md) for construction, scope, and intended use.

---

## Example use cases

### Retrieval-augmented diagram authoring (RAG)
- Retrieve reference diagrams similar to your paper's **title, abstract, caption**, or the paragraph that explains the figure
- Use the retrieved references to guide layout, grouping, iconography, and style in a diagram generation system

### Multimodal retrieval and benchmarking
- Build and evaluate retrieval systems over diagram-centric scientific content
- Explore coarse-to-fine retrieval (paper-level → figure-level) and compare text fields as retrieval keys

### Diagram understanding, classification, and clustering
- Train or evaluate figure-type classifiers with the CLIP, VLM, and filtering labels
- Study diagram conventions across venues and years

### Paper-level analytics
Because records are linked to paper metadata, you can ask questions like:
- How do diagram conventions differ across venues, years, or presentation types (oral, spotlight, poster)?
- How does caption length or diagram density per paper change over time?

### Context-aware tasks
Citing paragraphs enable tasks that need more than captions:
- Context-aware diagram retrieval
- Diagram–text alignment and grounding
- Studying how authors explain diagrams in scientific writing

---

## Prerequisites

### conda
```bash
conda env create --file environment.yml
conda activate DiagramBank
```

---

## Download the DiagramBank dataset

### 1) Set the target folder

The default download is about 53 GB. Make sure you have enough disk space.

```bash
# Set the target folder using the FIG_RAG_DIR environment variable
export FIG_RAG_DIR=<a scratch folder with enough disk space>
```

### 2) Download options

```bash
# 1. Default: Download accepted-paper images + core files (data.jsonl/FAISS/DBs)
python huggingface/download_diagrambank.py

# 2. Download Everything: All papers (Accept + Reject) + Core files
# python huggingface/download_diagrambank.py --subset all

# 3. Download Rejected papers only + Core files
# python huggingface/download_diagrambank.py --subset reject

# 4. Skip Core Files: Download only images (no DBs or FAISS)
# python huggingface/download_diagrambank.py --no-core

# 5. Combine Flags: Download all images but skip core files
# python huggingface/download_diagrambank.py --subset all --no-core

# 6. Download only raw reproduction metadata archives for ICLR/ICML
# python huggingface/download_diagrambank.py --metadata-only

# 7. Include raw reproduction metadata archives with the selected image subset
# python huggingface/download_diagrambank.py --metadata
```

The script downloads `data.jsonl` and extracts the diagram folders, FAISS indices, and DuckDB databases to `$FIG_RAG_DIR`. Before extracting a file, it checks the file's size and SHA-256 against [huggingface/DiagramBank_remote_manifest.json](huggingface/DiagramBank_remote_manifest.json) (112 files; also listed in `huggingface/SHA256SUMS`) and rejects any mismatch.

---

## Check installation

```bash
du -sh $FIG_RAG_DIR
51G 
```

```bash
tree -L 4 $FIG_RAG_DIR

├── data.jsonl
├── faiss
│   ├── abstract_index
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── caption_index
│   │   ├── index.faiss
│   │   └── index.pkl
│   ├── research.db
│   └── title_index
│       ├── index.faiss
│       └── index.pkl
└── OpenReview
    ├── ICLR
    │   ├── figures
    │   │   ├── 2017
    │   │   ├── 2018
    │   │   ├── 2019
    │   │   ├── 2020
    │   │   ├── 2021
    │   │   ├── 2022
    │   │   ├── 2023
    │   │   ├── 2024
    │   │   ├── 2025
    │   │   └── 2026
    │   └── research.db
    ├── ICML
    │   ├── figures
    │   │   ├── 2023
    │   │   ├── 2024
    │   │   └── 2025
    │   └── research.db
    ├── NeurIPS
    │   ├── figures
    │   │   ├── 2021
    │   │   ├── 2022
    │   │   ├── 2024
    │   │   └── 2025
    │   └── research.db
    └── TMLR
        ├── figures
        │   ├── 2022
        │   ├── 2023
        │   ├── 2024
        │   ├── 2025
        │   └── 2026
        └── research.db
```

### What are these indices?

Each index holds one `text-embedding-3-large` vector per diagram (57,100 entries), with the full record as metadata.

* `title_index`: coarse paper-level filtering (topic/domain alignment)
* `abstract_index`: paper-level refinement (problem/method alignment)
* `caption_index`: figure-level matching (diagram content alignment)

`faiss/research.db` is a DuckDB database whose `Data` table holds the 57,100 records.

---

## Usage

### Load the records

```python
import json
import os

FIG_RAG_DIR = os.environ["FIG_RAG_DIR"]
with open(os.path.join(FIG_RAG_DIR, "data.jsonl"), encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

# DiagramBank-Core: the two filtering paths on which both VLM judges label the figure a diagram
core = [r for r in records if r["cascade_path"] in {"t1_unanimous", "t2_vlm_consensus_gpt_confirmed"}]
print(len(records), len(core))  # 57100 51590

record = core[0]
print(record["title"], record["figure_caption"])
print(os.path.join(FIG_RAG_DIR, record["figure_path"]))  # the diagram image
```

### Set your OpenAI API key

```bash
export OPENAI_API_KEY=<your openai api key>
```

The key is only used to embed the *query text* at runtime, so the cost is very low: $0.13 per 1M tokens with Text Embedding 3 Large. A paper title is 5–25 words, an abstract 150–250 words, and a caption 10–100 words. With an upper bound of 500 words and 1.33 tokens per word, a query uses at most about 1,000 tokens, so 1,000 queries cost about $0.13.

> Note: pricing can change; treat the above as a back-of-the-envelope estimate.

### Retrieve similar diagrams for your figure

`hierarchical_retrieval()` narrows the candidates with your title, then your abstract, and returns the top-`k` diagrams by caption similarity.

```python
import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

FIG_RAG_DIR = os.environ["FIG_RAG_DIR"]
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
title_index, abstract_index, caption_index = (
    FAISS.load_local(os.path.join(FIG_RAG_DIR, "faiss", name), embeddings, allow_dangerous_deserialization=True)
    for name in ("title_index", "abstract_index", "caption_index")
)


def hierarchical_retrieval(title, abstract, caption, t1=100, t2=10, k=3):
    title_ids = {doc.metadata["id"] for doc in title_index.similarity_search(title, k=t1)}
    abstract_docs = abstract_index.similarity_search(
        abstract, k=t2, filter=lambda m: m["id"] in title_ids, fetch_k=len(title_ids) * 100)
    abstract_ids = {doc.metadata["id"] for doc in abstract_docs}
    return caption_index.similarity_search(
        caption, k=k, filter=lambda m: m["id"] in abstract_ids, fetch_k=len(abstract_ids) * 1000)
```

- `t1`: number of diagrams with similar title
- `t2`: number of diagrams with similar abstract
- `k`: number of diagrams with similar caption

```python
results = hierarchical_retrieval(
    title="Transformers for image recognition",
    abstract="This paper proposes a new attention mechanism for vision tasks.",
    caption="Figure 1: The architecture of the Vision Transformer.",
    t1=1000, t2=100, k=5,
)
for rank, doc in enumerate(results, 1):
    print(f"Rank {rank}: {doc.metadata['title']}")
    print(f"  {os.path.join(FIG_RAG_DIR, doc.metadata['figure_path'])}")
```

---

## Hugging Face

The dataset and dataset card are hosted at:
[https://huggingface.co/datasets/ghzlmc/DiagramBank](https://huggingface.co/datasets/ghzlmc/DiagramBank)

---

## Reproduce this work

To rebuild the collection from OpenReview, see [reproduce/README.md](reproduce/README.md). It might take a few days up to a week. The scripts under `reproduce/` cover raw OpenReview collection, PDF figure extraction, context extraction, and first-stage CLIP classification.

The filtering cascade and its saved votes are in [cascade/](cascade/). This command replays the saved votes and checks that the 57,100 released records are reproduced exactly:

```bash
python cascade/finalize.py --output-dir <dir> --verify-release-db $FIG_RAG_DIR/faiss/research.db
```

### Paper experiments

| Path | Contents |
|---|---|
| `reproduce/` | Data collection: OpenReview metadata and PDFs, PDFFigures 2.0 figure and caption extraction, PyMuPDF context extraction, and CLIP classification. `figure_context.py` and `clip_provenance.py` are the revised context parser and CLIP run records; the release keeps its original contexts and labels. |
| `cascade/` | Filtering: Tier 1/Tier 2 routing, Claude Haiku 4.5 and GPT-5.4-mini judges, voting, GPT-5.4 confirmation, and the saved votes for all 94,176 candidates. |
| `release_supplement/` | `cascade_labels.jsonl`, `paper_groups.jsonl`, and the 55 out-of-label Haiku outputs. |
| `tools/` | Release export and supplement builder. |
| `experiments/final_release_audit/` | Final-release audit estimates for the full release, DiagramBank-Core, and the remaining records. |
| `experiments/context_review_50/` | 50-record diagram–text alignment review and release-wide context checks. |
| `experiments/retrieval_ablation/` | Retrieval field ablation: 2,000 frozen targets and queries, metrics, and retrieval logs. |
| `experiments/same_paper_context_swap/` | Same-paper context swap control. |
| `experiments/human_study/` | 20-case blind human study: briefs, references, outputs, frozen manifest, and votes. |
| `experiments/paperbananabench/generation_reference_protocol.json` | Reference-pool settings of the PaperBananaBench study (pool filters, exclusions, retrieval, reference inputs, generation). |
| `huggingface/` | Downloader, manifest, and SHA-256 list. |

---

## Notes on responsible use

* DiagramBank is mined from publicly accessible scientific PDFs and includes metadata for attribution and traceability.
* Figures remain subject to the original authors' and publishers' licenses and terms. Please use them responsibly and cite the relevant sources; see [DATASET_CARD.md](DATASET_CARD.md) for attribution and intended use.
* If you use DiagramBank for generative authoring, we recommend provenance tracking and disclosure for AI-generated figures where appropriate.
* The code in this repository is released under the MIT License.

---

## Citation

If you use DiagramBank in your research, please cite our paper (and consider citing the original papers for any retrieved exemplars you use directly).

```bibtex
@article{yue2026diagrambank,
  title={DiagramBank: Draw from the Great, Draw Your Best},
  author={Yue, Ling and Wang, Jiaying and Lu, Zhexi and Zhu, Mingzhi and Zhang, Tingwen and Xu, Zhen and Pan, Shaowu},
  journal={arXiv preprint arXiv:2604.20857},
  year={2026}
}
```
