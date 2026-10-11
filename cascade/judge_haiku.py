"""
Stage 2a: Classify candidates with Claude Haiku 4.5 via AWS Bedrock.

Reads candidates.jsonl, sends each figure to Haiku 4.5 for classification,
and writes results to labels_haiku.jsonl.

Uses ThreadPoolExecutor for concurrent API calls with configurable workers.
Supports --resume to continue interrupted runs.

Auth: Uses default AWS credentials (boto3).

Usage:
    python judge_haiku.py [--resume] [--workers 8] [--dry-run]

Estimated cost: ~$83 for 91K images (Haiku 4.5 vision pricing)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import boto3
from botocore.exceptions import ClientError

# ── Config ──────────────────────────────────────────────────────────────
# Claude Haiku 4.5 through its cross-region inference profile. Set BEDROCK_HAIKU_MODEL
# to use another profile or a full ARN from your own AWS account.
MODEL = os.environ.get("BEDROCK_HAIKU_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR", os.path.expanduser("~/fig-rag-dir"))
SCRIPT_DIR = Path(__file__).parent
INPUT_PATH = SCRIPT_DIR / "candidates.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "labels_haiku.jsonl"

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

MAX_RETRIES = 5
BASE_DELAY = 1  # seconds

# ── Helpers ─────────────────────────────────────────────────────────────
write_lock = Lock()
progress_lock = Lock()
progress = {"done": 0, "total": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0}


def read_image_bytes(image_path: str) -> bytes:
    with open(image_path, "rb") as f:
        return f.read()


def classify_one(record: dict) -> dict:
    """Classify a single figure. Called from worker thread."""
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    image_path = os.path.join(FIG_RAG_DIR, record["figure_path"])

    if not os.path.exists(image_path):
        label, in_tok, out_tok = "MISSING", 0, 0
    else:
        label, in_tok, out_tok = _call_api(client, image_path)

    result = {
        "venue": record["venue"],
        "platform_id": record["platform_id"],
        "figure_number": record["figure_number"],
        "figure_path": record["figure_path"],
        "clip_type": record["clip_type"],
        "clip_confidence": record["clip_confidence"],
        "label_haiku": label,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }

    with progress_lock:
        progress["done"] += 1
        progress["input_tokens"] += in_tok
        progress["output_tokens"] += out_tok
        if label == "ERROR":
            progress["errors"] += 1
        if progress["done"] % 100 == 0 or progress["done"] == progress["total"]:
            pct = 100 * progress["done"] / progress["total"]
            print(
                f"  [{progress['done']:>7,d}/{progress['total']:>7,d}] "
                f"{pct:5.1f}%  errors={progress['errors']}  "
                f"tokens: {progress['input_tokens']:,}in/{progress['output_tokens']:,}out",
                flush=True,
            )

    return result


def _call_api(client, image_path: str) -> tuple:
    """Returns (label, input_tokens, output_tokens)."""
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
                            {"text": PROMPT},
                        ],
                    }
                ],
                inferenceConfig={
                    "maxTokens": 16,
                    "temperature": 0.0,
                },
            )
            raw = resp["output"]["message"]["content"][0]["text"].strip().lower()
            usage = resp.get("usage", {})
            in_tok = usage.get("inputTokens", 0)
            out_tok = usage.get("outputTokens", 0)
            for cat in ("diagram", "plot", "photo", "other"):
                if cat in raw:
                    return cat, in_tok, out_tok
            return raw, in_tok, out_tok
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ThrottlingException":
                wait = BASE_DELAY * (2 ** attempt) + (hash(image_path) % 3)
                time.sleep(wait)
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BASE_DELAY * (attempt + 1))
                else:
                    return "ERROR", 0, 0
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY * (attempt + 1))
            else:
                return "ERROR", 0, 0
    return "ERROR", 0, 0


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 2a: Haiku 4.5 judge")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-annotated figures")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent API workers (default: 8)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N candidates (0=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats without making API calls")
    args = parser.parse_args()

    # Load candidates
    with open(INPUT_PATH) as f:
        candidates = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(candidates):,} candidates from {INPUT_PATH}")

    # Load already-done if resuming
    done = set()
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    key = f"{rec['platform_id']}|{rec['figure_number']}"
                    done.add(key)
        print(f"Resuming: {len(done):,} already done, {len(candidates)-len(done):,} remaining")

    # Filter to pending
    pending = []
    for c in candidates:
        key = f"{c['platform_id']}|{c['figure_number']}"
        if key not in done:
            pending.append(c)

    if args.limit > 0:
        pending = pending[:args.limit]

    if args.dry_run:
        # Cost estimate: Haiku 4.5 vision
        # Input: ~250 tokens text + ~1600 tokens image (low detail) = ~1850 tokens
        # Output: ~2 tokens
        # Haiku 4.5 pricing: $0.80/M input, $4.00/M output
        est_input_cost = len(pending) * 1850 * 0.80 / 1_000_000
        est_output_cost = len(pending) * 2 * 4.00 / 1_000_000
        est_total = est_input_cost + est_output_cost
        print(f"\n--- Dry Run ---")
        print(f"Figures to process: {len(pending):,}")
        print(f"Estimated input cost:  ${est_input_cost:,.2f}")
        print(f"Estimated output cost: ${est_output_cost:,.2f}")
        print(f"Estimated total cost:  ${est_total:,.2f}")
        print(f"Workers: {args.workers}")
        est_time = len(pending) / (args.workers * 2)  # ~2 req/s per worker
        print(f"Estimated time: ~{est_time/3600:.1f} hours")
        return

    progress["total"] = len(pending)
    print(f"Processing {len(pending):,} figures with {args.workers} workers...")

    out_mode = "a" if args.resume else "w"
    outfile = open(OUTPUT_PATH, out_mode)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(classify_one, c): c for c in pending}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    with write_lock:
                        outfile.write(json.dumps(result) + "\n")
                        outfile.flush()
                except Exception as e:
                    rec = futures[future]
                    print(f"FATAL: {rec['platform_id']}|{rec['figure_number']}: {e}")
    finally:
        outfile.close()

    print(f"\nDone. Labels written to {OUTPUT_PATH}")
    print(f"Errors: {progress['errors']}")
    print(f"Total tokens: {progress['input_tokens']:,} input, {progress['output_tokens']:,} output")
    # Haiku 4.5 pricing: $0.80/M input, $4.00/M output
    cost = progress["input_tokens"] * 0.80 / 1e6 + progress["output_tokens"] * 4.00 / 1e6
    print(f"Estimated cost: ${cost:,.2f}")


if __name__ == "__main__":
    main()
