"""
Stage 3: GPT-5.4 tiebreaker for disagreement cases.

Reads disagreements.jsonl (from vote.py), sends each figure to GPT-5.4
for classification, and writes labels_gpt_tiebreak.jsonl. GPT-5.4's
vote breaks ties when the CLIP / Haiku / GPT-5.4-mini judges disagree.

Decision logic after tiebreak (applied downstream by finalize.py):
  - t1_majority_diagram  + GPT-5.4=diagram → keep (3/4 say diagram)
  - t1_majority_diagram  + GPT-5.4≠diagram → still keep (2/3 original)
  - t1_minority_diagram  + GPT-5.4=diagram → keep (2/4 say diagram)
  - t1_minority_diagram  + GPT-5.4≠diagram → reject (CLIP only)
  - t2_vlm_consensus     + GPT-5.4=diagram → keep (3/3 VLMs agree)
  - t2_vlm_consensus     + GPT-5.4≠diagram → reject (CLIP=no wins)

Auth: Codex OAuth via ~/.clawdbot auth-profiles.json.

Usage:
    python judge_gpt_tiebreak.py [--resume] [--retry-errors] [--workers 4] [--dry-run]
"""

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Event, Lock

from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────
MODEL = "gpt-5.4"
BASE_URL = "https://chatgpt.com/backend-api/codex"
INSTRUCTIONS = "You are a scientific figure classifier."
# The ChatGPT Pro Codex backend (chatgpt.com/backend-api/codex) accepts
# a *strict subset* of the public Responses API. Per the ResponsesApiRequest
# struct in openai/codex (codex-rs/codex-api/src/common.rs), the supported
# fields are: model, instructions, input, tools, tool_choice,
# parallel_tool_calls, reasoning, store, stream, include, service_tier,
# prompt_cache_key, text, client_metadata.
#
# Notably UNSUPPORTED (returns 400 "Unsupported parameter"):
#   - temperature         (verified by empirical 400 response)
#   - max_output_tokens   (verified by empirical 400 response)
#   - top_p / top_logprobs
#
# We therefore mirror the minimal known-working payload used by
# judge_gpt_mini.py, which has already completed 94,176 calls against this
# endpoint: {model, input, instructions, store, stream}. The Codex backend
# fills in defaults for everything else (tools=[], tool_choice="auto",
# parallel_tool_calls=False, reasoning={}, include=[]). Reasoning summaries,
# if needed, are requested with `reasoning={"summary": "auto"}`, not
# {"effort": "..."}.
FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR", os.path.expanduser("~/fig-rag-dir"))
SCRIPT_DIR = Path(__file__).parent
INPUT_PATH = SCRIPT_DIR / "disagreements.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "labels_gpt_tiebreak.jsonl"
CODEX_CONFIG_PATHS = [
    Path.home() / ".clawdbot" / "agents" / "main" / "agent" / "auth-profiles.json",
]

# GPT-5.4 pricing (USD per 1M tokens)
PRICE_IN = 2.50
PRICE_OUT = 15.00

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
BASE_DELAY = 2  # seconds (higher for expensive model)

# ── Helpers ─────────────────────────────────────────────────────────────
write_lock = Lock()
progress_lock = Lock()
progress = {"done": 0, "total": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0}
rate_limit_hit = Event()  # Signal all workers to stop


def _get_api_key() -> str:
    """Get OpenAI access token from Codex OAuth auth config."""
    for p in CODEX_CONFIG_PATHS:
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                profiles = data.get("profiles", {})
                last_good = data.get("lastGood", {}).get("openai-codex")
                if last_good and last_good in profiles:
                    profile = profiles[last_good]
                else:
                    for name, prof in profiles.items():
                        if prof.get("provider") == "openai-codex":
                            profile = prof
                            break
                    else:
                        continue
                token = profile.get("access")
                if token:
                    expires = profile.get("expires", 0)
                    if expires and expires < time.time() * 1000:
                        print(f"  WARNING: Codex token expired. Run `codex auth login` to refresh.")
                    print(f"  Using Codex OAuth token from {p}")
                    return token
            except Exception:
                continue
    raise RuntimeError(
        "No Codex auth found. Run `codex auth login`"
    )


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def classify_one(record: dict, client: OpenAI) -> dict:
    """Classify a single figure. Called from worker thread."""
    if rate_limit_hit.is_set():
        return None

    image_path = os.path.join(FIG_RAG_DIR, record["figure_path"])

    if not os.path.exists(image_path):
        label, in_tok, out_tok = "MISSING", 0, 0
    else:
        label, in_tok, out_tok = _call_api(client, image_path)

    if label == "RATE_LIMITED":
        return None

    result = {
        "venue": record["venue"],
        "platform_id": record["platform_id"],
        "figure_number": record["figure_number"],
        "figure_path": record["figure_path"],
        "clip_type": record["clip_type"],
        "clip_confidence": record["clip_confidence"],
        "label_haiku": record.get("label_haiku", ""),
        "label_gpt_mini": record.get("label_gpt_mini", ""),
        "vote_category": record.get("vote_category", ""),
        "label_gpt_tiebreak": label,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }

    with progress_lock:
        progress["done"] += 1
        progress["input_tokens"] += in_tok
        progress["output_tokens"] += out_tok
        if label == "ERROR":
            progress["errors"] += 1
        if progress["done"] % 50 == 0 or progress["done"] == progress["total"]:
            pct = 100 * progress["done"] / progress["total"]
            print(
                f"  [{progress['done']:>7,d}/{progress['total']:>7,d}] "
                f"{pct:5.1f}%  errors={progress['errors']}  "
                f"tokens: {progress['input_tokens']:,}in/{progress['output_tokens']:,}out",
                flush=True,
            )

    return result


def _call_api(client: OpenAI, image_path: str) -> tuple:
    """Returns (label, input_tokens, output_tokens). Uses Codex Responses API."""
    b64 = encode_image(image_path)

    for attempt in range(MAX_RETRIES):
        if rate_limit_hit.is_set():
            return "RATE_LIMITED", 0, 0

        try:
            stream = client.responses.create(
                model=MODEL,
                input=[{
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": PROMPT},
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

            raw = text.strip().lower()
            if usage is None:
                in_tok = out_tok = 0
            elif isinstance(usage, dict):
                in_tok = usage.get("input_tokens", 0) or 0
                out_tok = usage.get("output_tokens", 0) or 0
            else:
                in_tok = getattr(usage, "input_tokens", 0) or 0
                out_tok = getattr(usage, "output_tokens", 0) or 0
            for cat in ("diagram", "plot", "photo", "other"):
                if cat in raw:
                    return cat, in_tok, out_tok
            return raw, in_tok, out_tok
        except Exception as e:
            err_str = str(e)
            # Detect Codex Pro usage limit
            if "usage_limit_reached" in err_str or "resets_at" in err_str:
                try:
                    import re
                    m = re.search(r"resets_in_seconds['\"]:\s*(\d+)", err_str)
                    if m:
                        secs = int(m.group(1))
                        reset_time = datetime.fromtimestamp(time.time() + secs)
                        print(
                            f"\n  ⚠ USAGE LIMIT REACHED. Resets at {reset_time.strftime('%Y-%m-%d %H:%M')} "
                            f"({secs // 3600}h {(secs % 3600) // 60}m). Stopping...",
                            flush=True,
                        )
                    else:
                        print(f"\n  ⚠ USAGE LIMIT REACHED. Stopping...", flush=True)
                except Exception:
                    print(f"\n  ⚠ USAGE LIMIT REACHED. Stopping...", flush=True)
                rate_limit_hit.set()
                return "RATE_LIMITED", 0, 0

            if attempt < MAX_RETRIES - 1:
                wait = BASE_DELAY * (2 ** attempt)
                time.sleep(wait)
            else:
                if progress["errors"] < 3:
                    print(f"  ERROR ({type(e).__name__}): {e}", flush=True)
                return "ERROR", 0, 0
    return "ERROR", 0, 0


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 3: GPT-5.4 tiebreaker")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-annotated figures (including errors)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Re-process figures that previously got ERROR labels")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent API workers (default: 4)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N figures (0=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats without making API calls")
    args = parser.parse_args()

    # Load disagreements
    with open(INPUT_PATH) as f:
        figures = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(figures):,} disagreement cases from {INPUT_PATH}")

    # Load already-done if resuming
    done = {}  # key -> record
    error_keys = set()
    if (args.resume or args.retry_errors) and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    key = f"{rec['platform_id']}|{rec['figure_number']}"
                    done[key] = rec
                    if rec.get("label_gpt_tiebreak") == "ERROR":
                        error_keys.add(key)
        ok_count = len(done) - len(error_keys)
        print(f"  Previous results: {len(done):,} total ({ok_count:,} OK, {len(error_keys):,} errors)")

    # Determine what to process
    if args.retry_errors:
        pending = []
        for fig in figures:
            key = f"{fig['platform_id']}|{fig['figure_number']}"
            if key not in done or key in error_keys:
                pending.append(fig)
        print(f"  Retrying {len(error_keys):,} errors + {len(pending) - len(error_keys):,} new = {len(pending):,} pending")
    elif args.resume:
        pending = []
        for fig in figures:
            key = f"{fig['platform_id']}|{fig['figure_number']}"
            if key not in done:
                pending.append(fig)
        print(f"  Resuming: {len(done):,} already done, {len(pending):,} remaining")
    else:
        pending = list(figures)

    if args.limit > 0:
        pending = pending[:args.limit]

    if args.dry_run:
        # Use GPT-5.4-mini averages as a proxy (same prompt, same images)
        est_in = len(pending) * 500  # ~500 tok/fig observed from mini
        est_out = len(pending) * 5
        est_input_cost = est_in * PRICE_IN / 1_000_000
        est_output_cost = est_out * PRICE_OUT / 1_000_000
        est_total = est_input_cost + est_output_cost
        print(f"\n--- Dry Run ---")
        print(f"Figures to process: {len(pending):,}")
        print(f"Estimated input tokens:  {est_in:,}")
        print(f"Estimated output tokens: {est_out:,}")
        print(f"Estimated input cost:  ${est_input_cost:,.2f}")
        print(f"Estimated output cost: ${est_output_cost:,.2f}")
        print(f"Estimated total cost:  ${est_total:,.2f}")
        print(f"Workers: {args.workers}")
        est_time = len(pending) / (args.workers * 2)
        print(f"Estimated time: ~{est_time/3600:.1f} hours")
        return

    if not pending:
        print("Nothing to process.")
        return

    client = OpenAI(api_key=_get_api_key(), base_url=BASE_URL)
    progress["total"] = len(pending)
    print(f"Processing {len(pending):,} figures with {args.workers} workers...")
    print(f"  model={MODEL}  (Codex backend: no temperature/max_output_tokens)")

    # When retrying errors, rewrite file without error records
    if args.retry_errors and error_keys:
        keep_records = [rec for key, rec in done.items() if key not in error_keys]
        with open(OUTPUT_PATH, "w") as f:
            for rec in keep_records:
                f.write(json.dumps(rec) + "\n")
        print(f"  Rewrote {OUTPUT_PATH.name}: kept {len(keep_records):,} OK records, removed {len(error_keys):,} errors")

    outfile = open(OUTPUT_PATH, "a")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(classify_one, fig, client): fig for fig in pending
            }
            for future in as_completed(futures):
                if rate_limit_hit.is_set():
                    for f in futures:
                        f.cancel()
                    break
                try:
                    result = future.result()
                    if result is not None:
                        with write_lock:
                            outfile.write(json.dumps(result) + "\n")
                            outfile.flush()
                except Exception as e:
                    rec = futures[future]
                    print(f"FATAL: {rec['platform_id']}|{rec['figure_number']}: {e}")
    finally:
        outfile.close()

    # Count errors in output
    error_count = 0
    total_count = 0
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                if line.strip():
                    total_count += 1
                    if '"label_gpt_tiebreak": "ERROR"' in line:
                        error_count += 1

    print(f"\nDone. Tiebreak labels written to {OUTPUT_PATH}")
    print(f"Total records: {total_count:,} ({error_count:,} errors)")
    print(f"Session tokens: {progress['input_tokens']:,} input, {progress['output_tokens']:,} output")
    cost = (
        progress["input_tokens"] * PRICE_IN / 1e6
        + progress["output_tokens"] * PRICE_OUT / 1e6
    )
    print(f"Session cost: ${cost:,.2f}")

    if rate_limit_hit.is_set():
        print(f"\n⚠ Stopped early due to usage limit. Run again with --retry-errors when limit resets.")

    if error_count > 0 and not rate_limit_hit.is_set():
        print(f"\n⚠ {error_count:,} errors remain. Run with --retry-errors to reprocess them.")


if __name__ == "__main__":
    main()
