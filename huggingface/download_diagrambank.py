#!/usr/bin/env python3
"""Download, verify, and extract DiagramBank release files.

Every payload is checked against the committed manifest
(DiagramBank_remote_manifest.json): its size and SHA-256 must match before it
is extracted, so a corrupted or altered payload is rejected. By default files
are fetched from the Hugging Face dataset ghzlmc/DiagramBank.

    export FIG_RAG_DIR=/path/to/scratch
    python download_diagrambank.py                 # accepted-paper images + core files
    python download_diagrambank.py --subset all    # accepted and rejected papers + core files
    python download_diagrambank.py --no-core       # images only
"""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BASE_URL = "https://huggingface.co/datasets/ghzlmc/DiagramBank/resolve/main"
MANIFEST_JSON = "DiagramBank_remote_manifest.json"
MANIFEST_SHA256 = "SHA256SUMS"
SCRIPT_DIR = Path(__file__).resolve().parent


def is_accepted(filename):
    """Return True for archive names selected by the accepted-paper mode."""
    name_lower = filename.lower()
    return any(keyword in name_lower for keyword in ("accept", "spotlight", "poster", "oral"))


def is_core_file(filename):
    """Return True for release metadata, FAISS, and database artifacts."""
    return (
        filename == "data.jsonl"
        or filename == "data/faiss_complete.tar.gz"
        or filename.endswith("_research_db.tar.gz")
    )


def is_metadata_archive(filename):
    """Return True for optional raw reproduction metadata archives."""
    return filename.startswith("reproduce/") and filename.endswith("_metadata.tar.gz")


def is_image_archive(filename):
    """Return True for decision-level image archives rather than core files."""
    return (
        filename.startswith("data/")
        and filename.endswith(".tar.gz")
        and not is_core_file(filename)
    )


def _sum_bytes(records):
    return sum(record["size"] for record in records)


def summarize_manifest(records):
    image_records = [r for r in records if is_image_archive(r["path"])]
    accepted_records = [r for r in image_records if is_accepted(r["path"])]
    core_records = [r for r in records if is_core_file(r["path"])]
    metadata_records = [r for r in records if is_metadata_archive(r["path"])]
    return {
        "sha256_payload_files": len(records),
        "sha256_payload_bytes": _sum_bytes(records),
        "image_archives": len(image_records),
        "image_archive_bytes": _sum_bytes(image_records),
        "accepted_image_archives": len(accepted_records),
        "accepted_image_archive_bytes": _sum_bytes(accepted_records),
        "other_image_archives": len(image_records) - len(accepted_records),
        "other_image_archive_bytes": _sum_bytes(image_records) - _sum_bytes(accepted_records),
        "core_payload_files": len(core_records),
        "core_payload_bytes": _sum_bytes(core_records),
        "reproduction_metadata_archives": len(metadata_records),
        "reproduction_metadata_archive_bytes": _sum_bytes(metadata_records),
    }


def build_remote_manifest(repo_id, revision):
    """Maintainer helper: read payload sizes and SHA-256 values from Hub metadata."""
    from huggingface_hub import HfApi

    api = HfApi()
    resolved = api.dataset_info(repo_id=repo_id, revision=revision).sha
    records = []
    for entry in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", revision=resolved, recursive=True):
        lfs = getattr(entry, "lfs", None)
        sha256 = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        if getattr(entry, "blob_id", None) is None or not sha256:
            continue
        records.append({"path": entry.path, "size": int(entry.size), "sha256": sha256})
    records.sort(key=lambda record: record["path"])
    return {
        "schema_version": 2,
        "resolved_commit": resolved,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": "Hugging Face repository-tree metadata; sha256 is the Git LFS object ID of each payload.",
        "summary": summarize_manifest(records),
        "files": records,
    }


def write_manifest(manifest, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / MANIFEST_JSON).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (output_dir / MANIFEST_SHA256).open("w", encoding="utf-8") as handle:
        for record in manifest["files"]:
            handle.write(f"{record['sha256']}  {record['path']}\n")


def fetch(url, destination, attempts=5):
    """Stream a URL to a file, retrying transient failures."""
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "diagrambank-downloader"})
            with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
            return
        except OSError as error:
            if attempt == attempts:
                raise
            print(f"  retry {attempt}/{attempts - 1} after error: {error}")
            time.sleep(min(60, 5 * attempt))


def load_manifest(path, base_url):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fetch(f"{base_url}/{MANIFEST_JSON}", path)
    return json.loads(path.read_text(encoding="utf-8"))


def get_file_list(records, subset, download_core=True, download_metadata=False, metadata_only=False):
    """Return manifest paths selected by the downloader flags."""
    paths = [record["path"] for record in records]
    metadata_archives = [p for p in paths if is_metadata_archive(p)]
    if metadata_only:
        return metadata_archives
    image_tars = [p for p in paths if is_image_archive(p)]
    if subset == "all":
        selected_images = image_tars
    elif subset == "accept":
        selected_images = [p for p in image_tars if is_accepted(p)]
    else:
        selected_images = [p for p in image_tars if not is_accepted(p)]
    selected = [p for p in paths if is_core_file(p)] if download_core else []
    selected += selected_images
    if download_metadata:
        selected += metadata_archives
    return selected


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verify_extract(record, args, dest_dir):
    """Download one payload, verify size and SHA-256, then extract archives."""
    filename = record["path"]
    target = dest_dir / filename
    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} ({record['size']:,} bytes)...")
    if args.repo_id:
        from huggingface_hub import hf_hub_download

        downloaded = Path(hf_hub_download(repo_id=args.repo_id, filename=filename, repo_type="dataset",
                                          revision=args.revision, local_dir=dest_dir))
        downloaded.replace(partial)
    else:
        fetch(f"{args.base_url}/{filename}", partial)

    size = partial.stat().st_size
    digest = sha256_file(partial)
    if size != record["size"] or digest != record["sha256"]:
        partial.unlink()
        raise RuntimeError(f"verification failed for {filename}: expected {record['size']} bytes and "
                           f"SHA-256 {record['sha256']}, got {size} bytes and {digest}")
    print(f"  verified SHA-256 {digest}")
    partial.replace(target)

    if filename.endswith(".tar.gz"):
        print(f"  extracting {filename}...")
        with tarfile.open(target, "r:gz") as archive:
            archive.extractall(path=dest_dir, filter="data")
        target.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description="Download and verify DiagramBank release files.")
    parser.add_argument("--subset", choices=["accept", "reject", "all"], default="accept",
                        help="Which image archives to download (default: accept).")
    parser.add_argument("--no-core", dest="download_core", action="store_false", default=True,
                        help="Skip data.jsonl, FAISS indices, and research databases.")
    parser.add_argument("--metadata", action="store_true", help="Also download raw reproduction metadata archives.")
    parser.add_argument("--metadata-only", action="store_true", help="Download only raw reproduction metadata archives.")
    parser.add_argument("--manifest", type=Path, default=SCRIPT_DIR / MANIFEST_JSON,
                        help="Committed manifest used for verification.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="File server that resolves manifest paths.")
    parser.add_argument("--repo-id", help="Download from this Hugging Face dataset with huggingface_hub instead.")
    parser.add_argument("--revision", default="main", help="Hugging Face revision used with --repo-id.")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Maintainers: rebuild the manifest from Hub metadata of --repo-id and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.base_url = args.base_url.rstrip("/")
    if args.manifest_only:
        if not args.repo_id:
            print("Error: --manifest-only requires --repo-id.")
            return 1
        manifest = build_remote_manifest(args.repo_id, args.revision)
        write_manifest(manifest, args.manifest.parent)
        print(f"Wrote {len(manifest['files'])} payload entries at commit {manifest['resolved_commit']}")
        return 0

    fig_rag_dir = os.environ.get("FIG_RAG_DIR")
    if not fig_rag_dir:
        print("Error: FIG_RAG_DIR is not set. Run: export FIG_RAG_DIR=/path/to/your/data")
        return 1
    dest_dir = Path(fig_rag_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest, args.base_url)
    records = {record["path"]: record for record in manifest["files"]}
    selected = get_file_list(manifest["files"], args.subset, args.download_core,
                             args.metadata or args.metadata_only, args.metadata_only)
    total = sum(records[p]["size"] for p in selected)
    print(f"Manifest: {len(records)} payloads at commit {manifest['resolved_commit']}")
    print(f"Selected {len(selected)} files ({total / 1e9:.1f} GB) for {dest_dir}")

    failures = 0
    for filename in selected:
        try:
            download_verify_extract(records[filename], args, dest_dir)
        except Exception as error:
            failures += 1
            print(f"Failed to process {filename}: {error}")
    if failures:
        print(f"Completed with {failures} failed file(s); rerun to retry them.")
        return 1
    print("Download, verification, and extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
