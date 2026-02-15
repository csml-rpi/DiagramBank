import os
import json
import requests
import argparse
import time
import duckdb
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import NeurIPS_DIR, DB_PATH, SLUGIFY

DEFAULT_OUTPUT_DIR = os.path.join(NeurIPS_DIR, "papers")
FAILED_FILE = "failed-downloads.json"
MAX_WORKERS = 4
RETRIES = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def download_pdf(task):
    pid = task["id"]
    year = task["year"]
    decision_slug = task["decision"]
    url = task["url"]
    save_root = task["output_dir"]

    paper_dir = os.path.join(save_root, str(year), decision_slug, pid)
    pdf_path = os.path.join(paper_dir, "main.pdf")

    os.makedirs(paper_dir, exist_ok=True)

    if os.path.exists(pdf_path):
        return "skipped", task

    for attempt in range(RETRIES):
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=30) as r:
                if r.status_code == 200:
                    with open(pdf_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return "downloaded", task
                elif r.status_code == 429:
                    time.sleep((attempt + 1) * 3)
                    continue
                else:
                    return f"error_{r.status_code}", task
        except Exception:
            time.sleep(1)
            continue

    return "failed_max_retries", task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument(
        "--ignore_failed",
        default=False,
        action="store_true",
        help="Ignore failed-downloads.json and scan DB",
    )
    args = parser.parse_args()

    tasks = []
    source_desc = ""

    if os.path.exists(FAILED_FILE) and not args.ignore_failed:
        print(f"Found {FAILED_FILE}. RESUMING from previous failures...")
        source_desc = FAILED_FILE
        with open(FAILED_FILE, "r") as f:
            tasks = json.load(f)
            for t in tasks:
                t["output_dir"] = args.output_dir
    else:
        print(f"Querying DuckDB at {DB_PATH}...")
        source_desc = "DuckDB"

        try:
            con = duckdb.connect(DB_PATH, read_only=True)
            rows = con.execute(SLUGIFY).fetchall()
            con.close()

            for pid, year, clean_decision in rows:
                tasks.append(
                    {
                        "id": pid,
                        "year": year,
                        "decision": clean_decision,
                        "url": f"https://openreview.net/pdf?id={pid}",
                        "output_dir": args.output_dir,
                    }
                )

        except Exception as e:
            print(f"CRITICAL ERROR: Could not query DuckDB. {e}")
            return

    if args.limit:
        print(f"Limiting to first {args.limit} tasks.")
        tasks = tasks[: args.limit]

    failed_tasks = []
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    print(f"Processing {len(tasks)} papers from {source_desc}...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(download_pdf, t): t for t in tasks}

        for future in tqdm(as_completed(future_to_task), total=len(tasks), unit="pdf"):
            status, task = future.result()

            if status == "downloaded":
                stats["downloaded"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
                clean_task = {k: v for k, v in task.items() if k != "output_dir"}
                failed_tasks.append(clean_task)

    if failed_tasks:
        print(f"\nWriting {len(failed_tasks)} failed papers to {FAILED_FILE}...")
        with open(FAILED_FILE, "w") as f:
            json.dump(failed_tasks, f, indent=2)
    else:
        if os.path.exists(FAILED_FILE):
            print(f"\nSuccess! All retry tasks completed. Deleting {FAILED_FILE}...")
            os.remove(FAILED_FILE)

    print("\n--- Summary ---")
    print(f"Downloaded: {stats['downloaded']}")
    print(f"Skipped:    {stats['skipped']}")
    print(f"Failed:     {stats['failed']}")


if __name__ == "__main__":
    main()
