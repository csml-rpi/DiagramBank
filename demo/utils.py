import os
import json
from pathlib import Path
from typing import List, Dict
import base64

import os
import io
from PIL import Image
from google import genai
from google.genai import types

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


def load_jsonl_data(file_path: Path) -> List[Dict]:
    """Load data from JSONL file"""
    print(f"Loading jsonl data from {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# =========================================================
# utilities for generate.py
# =========================================================
def estimate_cost(input_tokens, generated_pixels):
    # Pricing Constants (Adjust based on your specific API tier)
    PRICE_PER_1M_INPUT_TOKENS = 0.30
    PRICE_PER_IMAGE = 0.04
    STANDARD_PIXEL_COUNT = 1024 * 1024

    input_cost = (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
    
    # We estimate image cost based on the ratio of pixels to a standard 1MP image
    pixel_cost = (generated_pixels / STANDARD_PIXEL_COUNT) * PRICE_PER_IMAGE
    
    total_cost = input_cost + pixel_cost
    return total_cost

def save_image_to_file(image_bytes, filename):
    if image_bytes:
        with open(filename, "wb") as f:
            f.write(image_bytes)
        print(f"Saved Image: {filename}")
    else:
        print(f"Failed to save {filename} (No data)")

def save_prompt_to_file(prompt_sequence, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in prompt_sequence:
            if isinstance(item, str):
                f.write(item + "\n")
            elif hasattr(item, 'format'): # PIL Image check
                f.write(f"[IMAGE OBJECT: {item.format} {item.size}]\n")
            else:
                f.write(f"\n[UNKNOWN OBJECT: {type(item)}]\n")
    print(f"Saved Prompt Log: {filename}")

def image_to_base64_data_url(pil_image):
    buffered = io.BytesIO()
    # Convert RGB to ensure compatibility (remove alpha channel if problematic for some encoders)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"