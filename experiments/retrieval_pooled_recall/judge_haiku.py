"""
Graded relevance judge: Claude Haiku 4.5 via AWS Bedrock.

Reads pool.jsonl, scores each (query, diagram) pair on a 0/1/2 scale,
writes results to labels_haiku.jsonl.

Auth: Default AWS credentials (boto3).

Usage:
    python judge_haiku.py [--resume] [--workers 8] [--dry-run]
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import boto3
from botocore.exceptions import ClientError

from judge_common import (
    INSTRUCTIONS, encode_image, format_prompt, parse_score,
)

# ── Config ──────────────────────────────────────────────────────────────
# Claude Haiku 4.5 through its cross-region inference profile. Set BEDROCK_HAIKU_MODEL
# to use another profile or a full ARN from your own AWS account.
MODEL = os.environ.get("BEDROCK_HAIKU_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR", os.path.expanduser("~/fig-rag-dir"))
SCRIPT_DIR = Path(__file__).parent
INPUT_PATH = SCRIPT_DIR / "pool.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "labels_haiku.jsonl"

# Haiku 4.5 pricing (USD per 1M tokens)
PRICE_IN = 0.80
PRICE_OUT = 4.00
MAX_RETRIES = 5
BASE_DELAY = 1

# ── Helpers ─────────────────────────────────────────────────────────────
write_lock = Lock()
progress_lock = Lock()
progress = {"done": 0, "total": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0}


def judge_one(item: dict) -> dict:
    """Score a single (query, diagram) pair. Called from worker thread."""
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    image_path = os.path.join(FIG_RAG_DIR, item["retrieved_figure_path"])

    if not os.path.exists(image_path):
        score, in_tok, out_tok = -1, 0, 0
    else:
        prompt = format_prompt(item)
        score, in_tok, out_tok = _call_api(client, image_path, prompt)

    result = {
        "judgment_key": item["judgment_key"],
        "query_name": item["query_name"],
        "score_haiku": score,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }

    with progress_lock:
        progress["done"] += 1
        progress["input_tokens"] += in_tok
        progress["output_tokens"] += out_tok
        if score == -1:
            progress["errors"] += 1
        if progress["done"] % 50 == 0 or progress["done"] == progress["total"]:
            pct = 100 * progress["done"] / progress["total"]
            cost = progress["input_tokens"] * PRICE_IN / 1e6 + progress["output_tokens"] * PRICE_OUT / 1e6
            print(
                f"  [{progress['done']:>5d}/{progress['total']:>5d}] "
                f"{pct:5.1f}%  errors={progress['errors']}  cost=${cost:.3f}",
                flush=True,
            )

    return result


def _call_api(client, image_path: str, prompt: str) -> tuple:
    """Returns (score, input_tokens, output_tokens). score is 0/1/2 or -1 on error."""
    image_bytes = open(image_path, "rb").read()

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.converse(
                modelId=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                        {"text": prompt},
                    ],
                }],
                system=[{"text": INSTRUCTIONS}],
                inferenceConfig={"maxTokens": 16, "temperature": 0.0},
            )
            raw = resp["output"]["message"]["content"][0]["text"]
            usage = resp.get("usage", {})
            in_tok = usage.get("inputTokens", 0)
            out_tok = usage.get("outputTokens", 0)

            score = parse_score(raw)
            return score, in_tok, out_tok

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ThrottlingException":
                wait = BASE_DELAY * (2 ** attempt) + (hash(image_path) % 3)
                time.sleep(wait)
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BASE_DELAY * (attempt + 1))
                else:
                    return -1, 0, 0
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY * (attempt + 1))
            else:
                return -1, 0, 0
    return -1, 0, 0


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Graded relevance judge: Haiku 4.5")
    parser.add_argument("--resume", action="store_true", help="Skip already-judged pairs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Process only first N (0=all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(INPUT_PATH) as f:
        pool = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(pool):,} pool items from {INPUT_PATH}")

    done_keys = set()
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_keys.add(rec["judgment_key"])
        print(f"  Already done: {len(done_keys):,}")

    pending = [item for item in pool if item["judgment_key"] not in done_keys]
    if args.limit > 0:
        pending = pending[:args.limit]

    if args.dry_run:
        est_cost = len(pending) * 1850 * PRICE_IN / 1e6 + len(pending) * 10 * PRICE_OUT / 1e6
        print(f"\n--- Dry Run ---")
        print(f"Pairs to judge: {len(pending):,}")
        print(f"Estimated cost: ${est_cost:.2f}")
        return

    if not pending:
        print("Nothing to process.")
        return

    progress["total"] = len(pending)
    print(f"Processing {len(pending):,} pairs with {args.workers} workers...")

    outfile = open(OUTPUT_PATH, "a" if args.resume else "w")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(judge_one, item): item for item in pending}
            for future in as_completed(futures):
                result = future.result()
                with write_lock:
                    outfile.write(json.dumps(result) + "\n")
                    outfile.flush()
    finally:
        outfile.close()

    cost = progress["input_tokens"] * PRICE_IN / 1e6 + progress["output_tokens"] * PRICE_OUT / 1e6
    print(f"\nDone: {progress['done']}/{progress['total']}, errors={progress['errors']}, cost=${cost:.3f}")


if __name__ == "__main__":
    main()
