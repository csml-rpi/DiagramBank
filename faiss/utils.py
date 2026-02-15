import os
import json
from pathlib import Path
from typing import List, Dict

# file directories
FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR")
ICLR_DIR = os.path.join(FIG_RAG_DIR, "OpenReview", "ICLR") # scratch folder for data from ICLR
ICML_DIR = os.path.join(FIG_RAG_DIR, "OpenReview", "ICML") # scratch folder for data from ICML
NeurIPS_DIR = os.path.join(FIG_RAG_DIR, "OpenReview", "NeurIPS") # scratch folder for data from NeurIPS
TMLR_DIR = os.path.join(FIG_RAG_DIR, "OpenReview", "TMLR") # scratch folder for data from TMLR

FAISS_DIR = os.path.join(FIG_RAG_DIR, "faiss")
if not os.path.exists(FAISS_DIR):
    os.makedirs(FAISS_DIR)

# database
ICLR_DB = os.path.join(ICLR_DIR, "research.db")
ICML_DB = os.path.join(ICML_DIR, "research.db")
NeurIPS_DB = os.path.join(NeurIPS_DIR, "research.db")
TMLR_DB = os.path.join(TMLR_DIR, "research.db")
DATA_DB = os.path.join(FAISS_DIR, "research.db")
DATA_JSONL = "data.jsonl"

THRESHOLD = 0.85 # classification confidence threshold

def load_jsonl_data(file_path: Path) -> List[Dict]:
    """Load data from JSONL file"""
    print(f"Loading jsonl data from {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data