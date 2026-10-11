"""
Graded relevance judge: GPT-5.4 via Codex Responses API.

Reads pool.jsonl, scores each (query, diagram) pair on a 0/1/2 scale,
writes results to labels_gpt.jsonl.

This is the most expensive judge ($2.50/M in, $15/M out). Run after
the cheaper Haiku and GPT-mini judges to verify the pipeline works.

Auth: Codex OAuth via ~/.clawdbot auth-profiles.json.

Usage:
    python judge_gpt.py [--resume] [--retry-errors] [--workers 4] [--dry-run]
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Event, Lock

from openai import OpenAI

from judge_common import (
    INSTRUCTIONS, encode_image, format_prompt, get_codex_api_key, parse_score,
)

# ── Config ──────────────────────────────────────────────────────────────
MODEL = "gpt-5.4"
BASE_URL = "https://chatgpt.com/backend-api/codex"
FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR", os.path.expanduser("~/fig-rag-dir"))
SCRIPT_DIR = Path(__file__).parent
INPUT_PATH = SCRIPT_DIR / "pool.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "labels_gpt.jsonl"

# GPT-5.4 pricing (USD per 1M tokens)
PRICE_IN = 2.50
PRICE_OUT = 15.00
MAX_RETRIES = 5
BASE_DELAY = 2  # slower base delay for expensive model

# ── Helpers ─────────────────────────────────────────────────────────────
write_lock = Lock()
progress_lock = Lock()
progress = {"done": 0, "total": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0}
rate_limit_hit = Event()


def judge_one(item: dict, client: OpenAI) -> dict | None:
    """Score a single (query, diagram) pair. Called from worker thread."""
    if rate_limit_hit.is_set():
        return None

    image_path = os.path.join(FIG_RAG_DIR, item["retrieved_figure_path"])

    if not os.path.exists(image_path):
        score, in_tok, out_tok = -1, 0, 0
    else:
        prompt = format_prompt(item)
        score, in_tok, out_tok = _call_api(client, image_path, prompt)

    if score == -99:
        return None

    result = {
        "judgment_key": item["judgment_key"],
        "query_name": item["query_name"],
        "score_gpt": score,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }

    with progress_lock:
        progress["done"] += 1
        progress["input_tokens"] += in_tok
        progress["output_tokens"] += out_tok
        if score == -1:
            progress["errors"] += 1
        if progress["done"] % 20 == 0 or progress["done"] == progress["total"]:
            pct = 100 * progress["done"] / progress["total"]
            cost = progress["input_tokens"] * PRICE_IN / 1e6 + progress["output_tokens"] * PRICE_OUT / 1e6
            print(
                f"  [{progress['done']:>5d}/{progress['total']:>5d}] "
                f"{pct:5.1f}%  errors={progress['errors']}  cost=${cost:.3f}",
                flush=True,
            )

    return result


def _call_api(client: OpenAI, image_path: str, prompt: str) -> tuple:
    """Returns (score, input_tokens, output_tokens). score -1=error, -99=rate limited."""
    b64 = encode_image(image_path)

    for attempt in range(MAX_RETRIES):
        if rate_limit_hit.is_set():
            return -99, 0, 0

        try:
            stream = client.responses.create(
                model=MODEL,
                input=[{
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                    ],
                }],
                instructions=INSTRUCTIONS,
                store=False,
                stream=True,
            )
            text = ""
            usage = None
            for event in stream:
                ev_type = event.type if hasattr(event, "type") else event.get("type")
                if ev_type == "response.output_text.delta":
                    delta = event.delta if hasattr(event, "delta") else event.get("delta", "")
                    text += delta or ""
                elif ev_type == "response.completed":
                    resp = event.response if hasattr(event, "response") else event.get("response")
                    if isinstance(resp, dict):
                        usage = resp.get("usage")
                    else:
                        usage = getattr(resp, "usage", None)

            if usage is None:
                in_tok = out_tok = 0
            elif isinstance(usage, dict):
                in_tok = usage.get("input_tokens", 0) or 0
                out_tok = usage.get("output_tokens", 0) or 0
            else:
                in_tok = getattr(usage, "input_tokens", 0) or 0
                out_tok = getattr(usage, "output_tokens", 0) or 0

            score = parse_score(text)
            return score, in_tok, out_tok

        except Exception as e:
            err_str = str(e)
            if "usage_limit_reached" in err_str or "resets_at" in err_str:
                m = re.search(r"resets_in_seconds['\"]:\s*(\d+)", err_str)
                if m:
                    secs = int(m.group(1))
                    reset_time = datetime.fromtimestamp(time.time() + secs)
                    print(
                        f"\n  USAGE LIMIT REACHED. Resets at {reset_time.strftime('%Y-%m-%d %H:%M')} "
                        f"({secs // 3600}h {(secs % 3600) // 60}m). Stopping...",
                        flush=True,
                    )
                else:
                    print(f"\n  USAGE LIMIT REACHED. Stopping...", flush=True)
                rate_limit_hit.set()
                return -99, 0, 0

            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY * (2 ** attempt))
            else:
                if progress["errors"] < 3:
                    print(f"  ERROR ({type(e).__name__}): {e}", flush=True)
                return -1, 0, 0
    return -1, 0, 0


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Graded relevance judge: GPT-5.4")
    parser.add_argument("--resume", action="store_true", help="Skip already-judged pairs")
    parser.add_argument("--retry-errors", action="store_true", help="Re-process pairs with score=-1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Process only first N (0=all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(INPUT_PATH) as f:
        pool = {
            json.loads(line)["judgment_key"]: json.loads(line)
            for line in f if line.strip()
        }
    print(f"Loaded {len(pool):,} pool items from {INPUT_PATH}")

    done = {}
    error_keys = set()
    if (args.resume or args.retry_errors) and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["judgment_key"]] = rec
                    if rec.get("score_gpt", -1) == -1:
                        error_keys.add(rec["judgment_key"])
        print(f"  Previous: {len(done):,} ({len(done) - len(error_keys):,} OK, {len(error_keys):,} errors)")

    if args.retry_errors:
        pending = [pool[k] for k in pool if k not in done or k in error_keys]
    elif args.resume:
        pending = [pool[k] for k in pool if k not in done]
    else:
        pending = list(pool.values())

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

    client = OpenAI(api_key=get_codex_api_key(), base_url=BASE_URL)
    progress["total"] = len(pending)
    print(f"Processing {len(pending):,} pairs with {args.workers} workers...")

    if args.retry_errors and error_keys:
        keep = [rec for k, rec in done.items() if k not in error_keys]
        with open(OUTPUT_PATH, "w") as f:
            for rec in keep:
                f.write(json.dumps(rec) + "\n")

    outfile = open(OUTPUT_PATH, "a" if (args.resume or args.retry_errors) else "w")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(judge_one, item, client): item for item in pending}
            for future in as_completed(futures):
                if rate_limit_hit.is_set():
                    for f in futures:
                        f.cancel()
                    break
                result = future.result()
                if result is not None:
                    with write_lock:
                        outfile.write(json.dumps(result) + "\n")
                        outfile.flush()
    finally:
        outfile.close()

    cost = progress["input_tokens"] * PRICE_IN / 1e6 + progress["output_tokens"] * PRICE_OUT / 1e6
    print(f"\nDone: {progress['done']}/{progress['total']}, errors={progress['errors']}, cost=${cost:.3f}")

    if rate_limit_hit.is_set():
        print(f"\nStopped early due to usage limit. Run again with --resume when limit resets.")


if __name__ == "__main__":
    main()
