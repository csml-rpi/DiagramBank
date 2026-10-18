import openreview
import os
import subprocess
from datetime import datetime
import json
from pathlib import Path
import sys

# Allow direct invocation of each venue script and package imports.
_REPRODUCE_ROOT = str(Path(__file__).resolve().parents[2])
if _REPRODUCE_ROOT not in sys.path:
    sys.path.insert(0, _REPRODUCE_ROOT)
from reproduce.figure_context import (extract_figure_contexts as _extract_figure_contexts,
                                      normalize_key as _normalize_key, is_caption as _is_caption)
from reproduce.clip_provenance import (load_clip_model as _load_clip_model,
                                       get_ensemble_features as _get_ensemble_features)
from typing import List, Dict
import re
import fitz #PyMuPDF

FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR") # scratch folder to store all data for this project
TMLR_DIR = os.path.join(FIG_RAG_DIR, "OpenReview", "TMLR") # scratch folder for data from TMLR
if not os.path.exists(TMLR_DIR):
    os.makedirs(TMLR_DIR)
DB_PATH = os.path.join(TMLR_DIR, "research.db") # database for TMLR (Papers, Figures)

JAR_PATH = os.environ.get("PDFFIGURES2_JAR")
JAVA_THREADS = "8"

# =========================================================
# general functions
# =========================================================
def load_jsonl_data(file_path: Path) -> List[Dict]:
    """Load data from JSONL file"""
    print(f"Loading jsonl data from {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def setup_clients():
    """Initialize both API v1 and v2 clients."""
    client_v1 = openreview.Client(baseurl='https://api.openreview.net')
    client_v2 = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    return client_v1, client_v2
# =========================================================


# =========================================================
# extract paper metadata
# =========================================================
def extract_decision(note):
    """
    Extracts decision from replies.
    """
    
    if 'withdrawal' in note.content:
        return 'withdrawn'
    
    return note.content['venue']['value'] # to be parsed later

def clean_text(text):
    """Helper to clean None or empty strings."""
    return text if text else ""

def process_paper(note):
    """
    Normalizes a single paper note into the target schema.
    """
    # 1. Basic Content Extraction
    content = note.content

    title = content.get('title', {}).get('value')
    abstract = content.get('abstract', {}).get('value')
    tldr =  content.get('TL;DR', {}).get('value') or content.get('TLDR', {}).get('value') or content.get('one-sentence_summary', {}).get('value')
    authors = content.get('authors', {}).get('value', [])
    author_ids = content.get('authorids', {}).get('value', [])
    keywords = content.get('keywords', {}).get('value', [])
    areas = content.get('primary_area', {}).get('value')
    bibtex = content.get('_bibtex', {}).get('value')
    year = int(re.search(r'year=\{?(\d{4})\}?', bibtex).group(1))

    # 2. Derived Fields
    decision = extract_decision(note)
    scores = []
    
    # Date handling (YYYYMMDD)
    date_obj = datetime.fromtimestamp(note.tcdate / 1000)
    cdate = date_obj.strftime('%Y%m%d')

    # 3. Construct Schema
    paper_data = {
        "platform": "OpenReview",
        "venue": "TMLR",
        "year": year,
        "title": clean_text(title),
        "abstract": clean_text(abstract),
        "keywords": ", ".join(keywords) if keywords else "",
        "areas": clean_text(areas), 
        "tldr": clean_text(tldr), 
        "scores": scores,
        "decision": decision,
        "authors": ", ".join(authors) if authors else "",
        "author_ids": ", ".join(author_ids) if author_ids else "",
        "cdate": cdate, 
        "url": f"https://openreview.net/forum?id={note.id}",
        "platform_id": note.id, 
        "bibtex": bibtex
    }
    
    return paper_data
# =========================================================


# =========================================================
# build path for Papers and Figures
# =========================================================
SLUGIFY = """
    SELECT 
        platform_id, 
        year,
        TRIM(
            REGEXP_REPLACE(
                LOWER(decision), 
                '[^a-z0-9]+', 
                '_', 
                'g'
            ), 
            '_'
        ) AS clean_decision
    FROM Papers
"""
# =========================================================


# =========================================================
# extract figures
# =========================================================
def run_pdffigures2(pdf_dir, json_dir, figure_dir):
    """
    Runs PDFFigures2 on the entire directory at once.
    """
    print(f"Starting PDFFigures2 on directory: {pdf_dir}")
    
    # We pass the directory path, not a list of files
    cmd = [
        "java", "-jar", JAR_PATH,
        pdf_dir,                 # Input: The whole directory
        "-d", json_dir + '/',    # Output JSONs here
        "-m", figure_dir + '/',  # Output Images here
        "-i", "150",             # DPI
        "-t", JAVA_THREADS,      # Parallel threads inside Java
        "-e",                    # Ignore errors
        "-q"                     # Quiet mode
    ]
    
    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"Java execution failed: {e}")
# =========================================================


# =========================================================
# extract context
# =========================================================
def get_clean_paper_text(pdf_path):
    """
    Extracts text using Layout Analysis to preserve paragraph structure.
    """
    if not os.path.exists(pdf_path):
        print(f"ERROR: {pdf_path} does not exist")
        return ""
    
    try:
        doc = fitz.open(pdf_path)
        full_text_blocks = []
        
        for page in doc:
            # block_type 0 is text, 1 is image
            blocks = page.get_text("blocks")
            for b in blocks:
                if b[6] == 0:  # Text block
                    raw_text = b[4]
                    
                    # 1. Fix Hyphenation (e.g., "signifi-\ncant" -> "significant")
                    # We look for a hyphen followed immediately by newline/return
                    clean_text = re.sub(r'-\s*[\n\r]+', '', raw_text)
                    
                    # 2. Fix Line Wrapping (replace remaining newlines with spaces)
                    clean_text = clean_text.replace("\n", " ")
                    
                    # 3. Clean extra whitespace
                    clean_text = " ".join(clean_text.split())
                    
                    if clean_text:
                        full_text_blocks.append(clean_text)
        
        # Join blocks with double newlines to denote paragraphs
        full_text = "\n\n".join(full_text_blocks)

        return full_text

    except Exception as e:
        # print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def normalize_key(k):
    return _normalize_key(k)


def is_caption(paragraph):
    return _is_caption(paragraph)


def extract_figure_contexts(paper_text, target_fig_numbers):
    """Use the shared explicit-reference parser, retaining original figure keys."""
    return _extract_figure_contexts(paper_text, target_fig_numbers)
# =========================================================


# =========================================================
# classify figures
# =========================================================
def load_model_and_tokenizer(pretrained="datacomp_xl_s13b_b90k"):
    return _load_clip_model(pretrained=pretrained)


def get_ensemble_features(model, tokenizer, class_prompts):
    return _get_ensemble_features(model, tokenizer, class_prompts)
# =========================================================
