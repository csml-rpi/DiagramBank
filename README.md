# DiagramBank
DiagramBank: A Dataset of Diagram Design Exemplars with Paper Metadata for Retrieval-Augmented Generation.


## Prerequsites
### conda
```bash
conda env create --file environment.yml
```

### DiagramBank dataset
```bash
# Run it (downloads ~60GB of diagrams from accepted papers)
# Set the target folder using the FIG_RAG_DIR environment variable
export FIG_RAG_DIR=<a scratch folder with at least 60 GB of space>
```

```bash
# 1. Default: Download Accepted papers + Core files (DBs/FAISS)
python faiss/download_diagrambank.py

# 2. Download Everything: All papers (Accept + Reject) + Core files
# python faiss/download_diagrambank.py --subset all

# 3. Download Rejected papers only + Core files
# python faiss/download_diagrambank.py --subset reject

# 4. Skip Core Files: Download only images (no DBs or FAISS)
# python faiss/download_diagrambank.py --no-core

# 5. Combine Flags: Download all images but skip core files
# python faiss/download_diagrambank.py --subset all --no-core
```

The script will automatically download and extract the diagram folder, FAISS index, duckb database to `$FIG_RAG_DIR`. The process can take 15-30 minutes dependending on network speed.

## Check installation
```
du $FIG_RAG_DIR --summarize --human-readable 
52G 
```

```
tree -L 4 $FIG_RAG_DIR

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

## Usage
`export OPENAI_API_KEY=<your openai api key>` This is only used for embedding, so the cost is very low ($0.13/1M tokens with Text Embedding 3 Large) (https://costgoat.com/pricing/openai-embeddings) A paper title is 5-25 words, an abstract is 150-250 words, and a caption is 10-100 words. Take an upper bound of 500 words, and 1.33 tokens per word, yielding an upper bound of 1000 token per query. Then, for 1000 queries, the cost will be $0.13.

To retrieve the similar diagrams for your figures, go to [query-diagram.ipynb](demo/query-diagrams.ipynb). Set `title`, `abstract`, and `caption` for your paper, and then keep running the third cell to get the similar diagrams.

## Huggingface
The dataset and model card is hosted at [https://huggingface.co/datasets/zhangt20/DiagramBank](https://huggingface.co/datasets/zhangt20/DiagramBank).