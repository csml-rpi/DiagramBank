"""
GPT-5.4 relevance judge for the A5 retrieval experiment.

For each (query, retrieved figure) pair across ALL conditions, asks GPT-5.4
whether the retrieved diagram is relevant as a visual reference for the query.

Auth: Codex OAuth via ~/.clawdbot auth-profiles.json or OPENAI_API_KEY.

Usage:
    python judge_gpt.py [--resume] [--workers 4]
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
FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR", os.path.expanduser("~/fig-rag-dir"))

SCRIPT_DIR = Path(__file__).parent
RESULTS_PATH = SCRIPT_DIR / "results.jsonl"
QUERIES_PATH = SCRIPT_DIR / "queries_a5.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "annotations_gpt.jsonl"

CODEX_CONFIG_PATHS = [
    Path.home() / ".clawdbot" / "agents" / "main" / "agent" / "auth-profiles.json",
]

INSTRUCTIONS = "You are a scientific diagram retrieval relevance judge."

PROMPT_TEMPLATE = (
    "You are evaluating whether a retrieved scientific diagram is relevant "
    "as a visual reference for creating a new diagram.\n\n"
    "QUERY PAPER:\n"
    "Title: {query_title}\n"
    "Abstract: {query_abstract}\n"
    "Target caption: {query_caption}\n\n"
    "RETRIEVED DIAGRAM:\n"
    "Source paper: {retrieved_title}\n"
    "Caption: {retrieved_caption}\n\n"
    "A retrieved diagram is RELEVANT if it depicts a sufficiently related "
    "architecture, pipeline, or concept that its visual structure could serve "
    "as a useful reference or template for designing the query paper's diagram. "
    "The diagram does not need to be identical — it should be from a related "
    "enough domain or method that a researcher could learn useful layout "
    "patterns, component conventions, or visual idioms from it.\n\n"
    "Reply with ONLY one word: relevant or not_relevant"
)

# GPT-5.4 pricing (USD per 1M tokens)
PRICE_IN = 2.50
PRICE_OUT = 15.00
MAX_RETRIES = 5
BASE_DELAY = 2

# ── Helpers ─────────────────────────────────────────────────────────────
write_lock = Lock()
progress_lock = Lock()
progress = {"done": 0, "total": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0}
rate_limit_hit = Event()


def _get_api_key() -> str:
    """Get OpenAI access token from Codex OAuth auth config."""
    # Try Codex OAuth first
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
                        print("  WARNING: Codex token expired. Run `codex auth login` to refresh.")
                    print(f"  Using Codex OAuth token from {p}")
                    return token
            except Exception:
                continue
    # Fallback to env var
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        print("  Using OPENAI_API_KEY from environment")
        return key
    raise RuntimeError("No Codex auth or OPENAI_API_KEY found.")


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def judge_one(item: dict, client: OpenAI) -> dict:
    """Judge a single (query, figure) pair. Called from worker thread."""
    if rate_limit_hit.is_set():
        return None

    image_path = os.path.join(FIG_RAG_DIR, item["retrieved_figure_path"])

    if not os.path.exists(image_path):
        label, in_tok, out_tok = "MISSING", 0, 0
    else:
        prompt = PROMPT_TEMPLATE.format(
            query_title=item["query_title"],
            query_abstract=item["query_abstract"][:400],
            query_caption=item["query_caption"],
            retrieved_title=item["retrieved_title"],
            retrieved_caption=item["retrieved_caption"],
        )
        label, in_tok, out_tok = _call_api(client, image_path, prompt)

    if label == "RATE_LIMITED":
        return None

    result = {
        "judgment_key": item["judgment_key"],
        "query_name": item["query_name"],
        "condition": item["condition"],
        "rank": item["rank"],
        "retrieved_platform_id": item["retrieved_platform_id"],
        "retrieved_figure_number": item["retrieved_figure_number"],
        "retrieved_figure_path": item["retrieved_figure_path"],
        "label_gpt": label,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }

    with progress_lock:
        progress["done"] += 1
        progress["input_tokens"] += in_tok
        progress["output_tokens"] += out_tok
        if label == "ERROR":
            progress["errors"] += 1
        if progress["done"] % 20 == 0 or progress["done"] == progress["total"]:
            pct = 100 * progress["done"] / progress["total"]
            cost_in = progress["input_tokens"] * PRICE_IN / 1e6
            cost_out = progress["output_tokens"] * PRICE_OUT / 1e6
            print(
                f"  [{progress['done']:>4d}/{progress['total']:>4d}] "
                f"{pct:5.1f}%  errors={progress['errors']}  "
                f"cost=${cost_in + cost_out:.3f}",
                flush=True,
            )

    return result


def _call_api(client: OpenAI, image_path: str, prompt: str) -> tuple:
    """Returns (label, input_tokens, output_tokens)."""
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

            raw = text.strip().lower()
            if usage is None:
                in_tok = out_tok = 0
            elif isinstance(usage, dict):
                in_tok = usage.get("input_tokens", 0) or 0
                out_tok = usage.get("output_tokens", 0) or 0
            else:
                in_tok = getattr(usage, "input_tokens", 0) or 0
                out_tok = getattr(usage, "output_tokens", 0) or 0

            # Parse label
            if "not_relevant" in raw or "not relevant" in raw:
                return "not_relevant", in_tok, out_tok
            elif "relevant" in raw:
                return "relevant", in_tok, out_tok
            else:
                return raw, in_tok, out_tok

        except Exception as e:
            err_str = str(e)
            if "usage_limit_reached" in err_str or "resets_at" in err_str:
                import re
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
    parser = argparse.ArgumentParser(description="A5: GPT-5.4 relevance judge")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-judged pairs")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent API workers (default: 4)")
    parser.add_argument("--primary-only", action="store_true",
                        help="Judge only caption_only + hier_default (skip HP sweep)")
    args = parser.parse_args()

    # Load queries for context
    queries = {}
    with open(QUERIES_PATH) as f:
        for line in f:
            q = json.loads(line.strip())
            queries[q["name"]] = q

    # Load retrieval results
    with open(RESULTS_PATH) as f:
        results = [json.loads(l) for l in f if l.strip()]

    if args.primary_only:
        results = [r for r in results if r["condition"] in ("caption_only", "hier_default")]

    # Build unique (query, figure) pairs to judge — de-dup across conditions
    # but keep condition info for each
    pairs = {}  # (query_name, pid, fig_num) -> first result row
    items_to_judge = []
    for r in results:
        key = (r["query_name"], r["retrieved_platform_id"],
               str(r["retrieved_figure_number"]))
        if key not in pairs:
            pairs[key] = r
            q = queries[r["query_name"]]
            items_to_judge.append({
                "judgment_key": f"{r['query_name']}|{r['retrieved_platform_id']}|{r['retrieved_figure_number']}",
                "query_name": r["query_name"],
                "query_title": q["title"],
                "query_abstract": q["abstract"],
                "query_caption": q["caption"],
                "condition": r["condition"],
                "rank": r["rank"],
                "retrieved_platform_id": r["retrieved_platform_id"],
                "retrieved_figure_number": r["retrieved_figure_number"],
                "retrieved_figure_path": r["retrieved_figure_path"],
                "retrieved_title": r["retrieved_title"],
                "retrieved_caption": r["retrieved_figure_caption"],
            })

    print(f"Total unique (query, figure) pairs to judge: {len(items_to_judge)}")

    # Resume: skip already-done
    done_keys = set()
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_keys.add(rec["judgment_key"])
        print(f"  Already done: {len(done_keys)}")
        items_to_judge = [it for it in items_to_judge if it["judgment_key"] not in done_keys]

    print(f"  Pending: {len(items_to_judge)}")
    if not items_to_judge:
        print("Nothing to do.")
        return

    # Set up client
    api_key = _get_api_key()
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    progress["total"] = len(items_to_judge)
    progress["done"] = 0
    progress["errors"] = 0
    progress["input_tokens"] = 0
    progress["output_tokens"] = 0

    # Open output file
    mode = "a" if args.resume else "w"
    outf = open(OUTPUT_PATH, mode)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(judge_one, item, client): item
                       for item in items_to_judge}

            for future in as_completed(futures):
                if rate_limit_hit.is_set():
                    break
                result = future.result()
                if result is None:
                    continue
                with write_lock:
                    outf.write(json.dumps(result) + "\n")
                    outf.flush()
    finally:
        outf.close()

    # Summary
    cost_in = progress["input_tokens"] * PRICE_IN / 1e6
    cost_out = progress["output_tokens"] * PRICE_OUT / 1e6
    print(f"\nDone: {progress['done']}/{progress['total']}, "
          f"errors={progress['errors']}, "
          f"cost=${cost_in + cost_out:.3f}")

    # Count labels
    if OUTPUT_PATH.exists():
        labels = {"relevant": 0, "not_relevant": 0, "other": 0}
        with open(OUTPUT_PATH) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    lbl = rec.get("label_gpt", "other")
                    if lbl in labels:
                        labels[lbl] += 1
                    else:
                        labels["other"] += 1
        print(f"Labels: {dict(labels)}")


if __name__ == "__main__":
    main()
