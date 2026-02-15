import duckdb
import openreview
import os
import subprocess
from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict
import re
import fitz #PyMuPDF

FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR") # scratch folder to store all data for this project
NeurIPS_DIR = os.path.join(FIG_RAG_DIR, "OpenReview", "NeurIPS") # scratch folder for data from NeurIPS
DB_PATH = os.path.join(NeurIPS_DIR, "research.db") # database for NeurIPS (Papers, Figures)

JAR_PATH = os.environ.get("PDFFIGURES2_JAR", "/mnt/home/tzhang/pdffigures2/pdffigures2.jar")
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
def extract_decision(note, default_decision, api_version):
    """
    Extracts decision from replies.
    """
    if default_decision:
        return default_decision
    
    if 'withdrawal' in note.content:
        return 'withdrawn'

    replies = note.details.get('directReplies', [])
    
    # Common decision logic
    for reply in replies:     
        content = reply.get('content', {})
        
        # API v1 direct access
        if api_version == 1:
            invitation = reply.get('invitation', '').lower()
            if 'decision' in invitation or 'meta_review' in invitation or 'acceptance' in invitation:
                return content.get('decision', content.get('recommendation', 'pending/unknown'))
        
        # API v2 nested access
        else:
            title = content.get('title', '') # version 2 has a either 'summary', 'title', or 'metareview' in a reply
            if title and title['value'] == 'Paper Decision': # this reply contains the decision
                # v2 decisions are in 'decision' -> 'value'
                if 'decision' in content:
                    return content['decision'].get('value', 'pending/unknown')
                if 'recommendation' in content:
                    return content['recommendation'].get('value', 'pending/unknown')

    return "pending/unknown"

def extract_scores(note, api_version):
    """
    Extracts reviewer scores from replies.
    """
    scores = []
    replies = note.details.get('directReplies', [])
    
    for reply in replies:
        content = reply.get('content', {})
        rating_str = None
        
        # Extract rating string
        if api_version == 1:
            rating_str = content.get('rating', content.get("recommendation")) # e.g. "8: Top 50%"
        else:
            rating_obj = content.get('rating', content.get("recommendation", {}))
            if isinstance(rating_obj, dict):
                rating_str = rating_obj.get('value')
        
        # Parse integer score from string
        if rating_str:
            try:
                # Handles "8: Top 50%" or just "8"
                score = int(str(rating_str).split(':')[0].strip())
                scores.append(score)
            except:
                pass
    return scores

def clean_text(text):
    """Helper to clean None or empty strings."""
    return text if text else ""

def process_paper(note, year, api_version, default_decision):
    """
    Normalizes a single paper note into the target schema.
    """
    # 1. Basic Content Extraction
    content = note.content
    
    # Handle V1 vs V2 content structure
    if api_version == 1:
        title = content.get('title')
        abstract = content.get('abstract')
        tldr = content.get('TL;DR', content.get('TLDR', content.get("one-sentence_summary")))
        authors = content.get('authors', [])
        author_ids = content.get('authorids', [])
        keywords = content.get('keywords', [])
        areas = content.get('primary_area', content.get('Please_choose_the_closest_area_that_your_submission_falls_into'))
        bibtex = content.get('_bibtex')
    else:
        title = content.get('title', {}).get('value')
        abstract = content.get('abstract', {}).get('value')
        tldr =  content.get('TL;DR', {}).get('value') or content.get('TLDR', {}).get('value') or content.get('one-sentence_summary', {}).get('value')
        authors = content.get('authors', {}).get('value', [])
        author_ids = content.get('authorids', {}).get('value', [])
        keywords = content.get('keywords', {}).get('value', [])
        areas = content.get('primary_area', {}).get('value')
        bibtex = content.get('_bibtex', {}).get('value')

    # 2. Derived Fields
    decision = extract_decision(note, default_decision, api_version)
    scores = extract_scores(note, api_version)
    
    # Date handling (YYYYMMDD)
    date_obj = datetime.fromtimestamp(note.tcdate / 1000)
    cdate = date_obj.strftime('%Y%m%d')

    # 3. Construct Schema
    paper_data = {
        "platform": "OpenReview",
        "venue": "NeurIPS",
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
    """
    Normalizes 'Figure 1', 'Fig 1', '1' -> '1' for dictionary matching.
    """
    return re.sub(r'\D', '', str(k))

def is_caption(paragraph):
    """
    Detects if a paragraph is likely a caption.
    
    Rules 
    - Start with 'Figure' or 'Fig'.
    - Followed by a number.
    - CRITICAL: followed immediately by a separator like ':' or '.' 
      (e.g., "Figure 4:" or "Fig. 4.")
      
    This prevents false positives like "Fig. 4 supports this hypothesis..." 
    from being deleted.
    """
    # Regex Breakdown:
    # ^       : Start of line
    # \s* : Optional whitespace
    # (?:...): Non-capturing group for Figure/Fig
    # \.?     : Optional dot (for Fig.)
    # \s* : Optional space
    # \d+     : The number
    # \s* : Optional space before separator
    # [:.]    : Must have a colon or dot separator
    
    pattern = r'^\s*(?:Figure|Fig)s?\.?\s*\d+\s*[:.]'
    
    return bool(re.match(pattern, paragraph, re.IGNORECASE))

def extract_figure_contexts(paper_text, target_fig_numbers):
    """
    Regex-based context extractor.
    Returns a dict { '1': [para1, para2], '2': [para1] ... }
    """
    paragraphs = paper_text.split('\n\n')
    
    # Normalize targets to handle cases where input might be "1" and "1b" -> both become "1"
    unique_targets = set(normalize_key(n) for n in target_fig_numbers)
    results = {k: [] for k in unique_targets}
    
    # Pre-compile regexes for efficiency
    patterns = {}
    for n_key in unique_targets:
        # REGEX EXPLANATION for case "Fig. (5b and c)":
        # 1. (?i)                 -> Case insensitive
        # 2. \b(?:Figure|Fig)s?\.?-> Matches "Figure", "Fig", "Figs."
        # 3. .{0,50}?             -> Lookahead window (0-50 chars, non-greedy). 
        #                            Allows finding "Figure" then skipping " (" to find "5".
        # 4. \b                   -> Word boundary (start of the number)
        # 5. KEY                  -> The target number (e.g., "5")
        # 6. (?!\d)               -> Negative lookahead: Next char must NOT be a digit.
        #                            - Matches "5" in "5b" (b is not digit)
        #                            - Matches "5" in "5)" () is not digit)
        #                            - REJECTS "5" in "50" (0 is digit)
        
        escaped_key = re.escape(n_key)
        patterns[n_key] = re.compile(
            r'(?i)\b(?:Figure|Fig)s?\.?.{0,50}?\b' + escaped_key + r'(?!\d)'
        )

    for para in paragraphs:
        # Skip empty or very short paragraphs
        if len(para) < 20:
            continue
            
        # 1. EXCLUDE CAPTIONS
        # If the paragraph *starts* with a Figure definition, discard it.
        if is_caption(para):
            continue

        # 2. CHECK REFERENCES
        for n_key in unique_targets:
            if patterns[n_key].search(para):
                results[n_key].append(para)

    return results
# =========================================================
