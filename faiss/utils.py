import json
import os
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR")
DATA_ROOT = Path(FIG_RAG_DIR).expanduser() if FIG_RAG_DIR else REPO_ROOT

# Optional external data root used by the downloader/demo image paths.
OPENREVIEW_DIR = DATA_ROOT / "OpenReview"

ICLR_DIR = OPENREVIEW_DIR / "ICLR"
ICML_DIR = OPENREVIEW_DIR / "ICML"
NeurIPS_DIR = OPENREVIEW_DIR / "NeurIPS"
TMLR_DIR = OPENREVIEW_DIR / "TMLR"

ICLR_DB = ICLR_DIR / "research.db"
ICML_DB = ICML_DIR / "research.db"
NeurIPS_DB = NeurIPS_DIR / "research.db"
TMLR_DB = TMLR_DIR / "research.db"

FAISS_DIR = DATA_ROOT / "faiss"
DATA_DB = FAISS_DIR / "research.db"
DATA_JSONL = DATA_ROOT / "data.jsonl"
PRIMARY_RELEASE_SIZE = 57100
INDEX_NAMES = ("title_index", "abstract_index", "caption_index")


def load_jsonl_data(file_path: Path) -> List[Dict]:
    """Load a JSONL file into memory."""
    print(f"Loading jsonl data from {file_path}")
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
