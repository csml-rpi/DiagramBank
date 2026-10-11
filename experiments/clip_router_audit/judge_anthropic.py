"""
[A1] LLM judge: Claude via AWS Bedrock Converse API.

Reads sample.jsonl, sends each figure image to the model, and writes
annotations to annotations_anthropic.jsonl.

Auth: Uses default AWS credentials (boto3).

Usage:
    python judge_anthropic.py [--resume]

Flags:
    --resume    Skip figures already annotated in the output file.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# ── Config ──────────────────────────────────────────────────────────────
# Claude Haiku 4.5 through its cross-region inference profile. Set BEDROCK_HAIKU_MODEL
# to use another profile or a full ARN from your own AWS account.
MODEL = os.environ.get("BEDROCK_HAIKU_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR", os.path.expanduser("~/fig-rag-dir"))
SCRIPT_DIR = Path(__file__).parent
SAMPLE_PATH = SCRIPT_DIR / "sample.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "annotations_anthropic.jsonl"

PROMPT = (
    "Classify this scientific figure into exactly one category: "
    "diagram, plot, photo, or other.\n\n"
    "Definitions:\n"
    "- diagram: schematic, flowchart, system architecture, pipeline, "
    "block diagram, neural network graph, or computational graph\n"
    "- plot: statistical chart, line/scatter/bar plot, histogram, "
    "heatmap, confusion matrix, or any quantitative data visualization\n"
    "- photo: natural photograph, camera/microscopy image, screenshot "
    "of a video game or simulation, pixel art, or comic-style illustration\n"
    "- other: dense text, standalone math equations, tables of numbers "
    "without graphics, blank pages, or anything that does not fit above\n\n"
    "Reply with ONLY the category name (one word)."
)

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


# ── Helpers ─────────────────────────────────────────────────────────────
def read_image_bytes(image_path: str) -> bytes:
    with open(image_path, "rb") as f:
        return f.read()


def classify(client, image_path: str) -> str:
    image_bytes = read_image_bytes(image_path)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.converse(
                modelId=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": image_bytes},
                                }
                            },
                            {
                                "text": PROMPT,
                            },
                        ],
                    }
                ],
                inferenceConfig={
                    "maxTokens": 16,
                    "temperature": 0.0,
                },
            )
            raw = resp["output"]["message"]["content"][0]["text"].strip().lower()
            # Normalise to one of the 4 categories
            for cat in ("diagram", "plot", "photo", "other"):
                if cat in raw:
                    return cat
            return raw  # keep raw if unexpected, for inspection
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ThrottlingException":
                wait = RETRY_DELAY * (attempt + 1) * 2
                print(f"  Throttled, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Attempt {attempt+1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return "ERROR"
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return "ERROR"
    return "ERROR"


# ── Main ────────────────────────────────────────────────────────────────
def main():
    resume = "--resume" in sys.argv
    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    # Load sample
    with open(SAMPLE_PATH) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    # Load existing annotations if resuming
    done = set()
    if resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    key = f"{rec['platform_id']}|{rec['figure_number']}"
                    done.add(key)
        print(f"Resuming: {len(done)} already annotated, {len(samples)-len(done)} remaining")

    out_mode = "a" if resume else "w"
    with open(OUTPUT_PATH, out_mode) as out:
        for i, s in enumerate(samples):
            key = f"{s['platform_id']}|{s['figure_number']}"
            if resume and key in done:
                continue

            image_path = os.path.join(FIG_RAG_DIR, s["figure_path"])
            if not os.path.exists(image_path):
                print(f"[{i+1}/{len(samples)}] MISSING: {image_path}")
                label = "MISSING"
            else:
                label = classify(client, image_path)
                print(f"[{i+1}/{len(samples)}] {s['figure_type']}({s['confidence']}) → {label}  {s['figure_path']}")

            record = {
                **s,
                "label_anthropic": label,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()

    print(f"\nDone. Annotations written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
