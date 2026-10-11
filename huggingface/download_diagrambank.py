#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "ghzlmc/DiagramBank"
DEFAULT_REVISION = "main"
MANIFEST_JSON = "DiagramBank_remote_manifest.json"
MANIFEST_SHA256 = "SHA256SUMS"


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


def _metadata_value(metadata, key):
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key, None)


def _sum_bytes(records):
    return sum(record["size"] for record in records)


def summarize_manifest(records):
    sha256_records = [record for record in records if record["sha256"]]
    git_only_records = [record for record in records if not record["sha256"]]
    image_records = [record for record in sha256_records if is_image_archive(record["path"])]
    accepted_records = [record for record in image_records if is_accepted(record["path"])]
    other_image_records = [record for record in image_records if not is_accepted(record["path"])]
    core_records = [
        record for record in sha256_records
        if is_core_file(record["path"])
    ]
    metadata_records = [
        record for record in sha256_records
        if is_metadata_archive(record["path"])
    ]

    return {
        "repository_files": len(records),
        "repository_bytes": _sum_bytes(records),
        "sha256_payload_files": len(sha256_records),
        "sha256_payload_bytes": _sum_bytes(sha256_records),
        "git_blob_only_files": len(git_only_records),
        "git_blob_only_bytes": _sum_bytes(git_only_records),
        "image_archives": len(image_records),
        "image_archive_bytes": _sum_bytes(image_records),
        "accepted_image_archives": len(accepted_records),
        "accepted_image_archive_bytes": _sum_bytes(accepted_records),
        "other_image_archives": len(other_image_records),
        "other_image_archive_bytes": _sum_bytes(other_image_records),
        "core_payload_files": len(core_records),
        "core_payload_bytes": _sum_bytes(core_records),
        "reproduction_metadata_archives": len(metadata_records),
        "reproduction_metadata_archive_bytes": _sum_bytes(metadata_records),
    }


def build_remote_manifest(api, repo_id, revision):
    """Read hashes and sizes from Hub metadata without downloading file payloads."""
    repo_info = api.dataset_info(repo_id=repo_id, revision=revision)
    resolved_revision = repo_info.sha
    if not resolved_revision:
        raise RuntimeError(f"Could not resolve revision {revision!r} for {repo_id}")

    records = []
    for entry in api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        recursive=True,
        expand=False,
    ):
        blob_id = getattr(entry, "blob_id", None)
        if blob_id is None:
            continue

        lfs = getattr(entry, "lfs", None)
        records.append(
            {
                "path": entry.path,
                "size": int(entry.size),
                "sha256": _metadata_value(lfs, "sha256"),
                "xet_hash": getattr(entry, "xet_hash", None),
                "git_blob_oid": blob_id,
            }
        )

    records.sort(key=lambda record: record["path"])
    return {
        "schema_version": 1,
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_commit": resolved_revision,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": (
            "Hugging Face repository-tree metadata only; no repository file payloads "
            "were downloaded. The sha256 field is the Git LFS object ID."
        ),
        "summary": summarize_manifest(records),
        "files": records,
    }


def write_manifest(manifest, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / MANIFEST_JSON
    sha256_path = output_dir / MANIFEST_SHA256

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False)
        handle.write("\n")

    with sha256_path.open("w", encoding="utf-8") as handle:
        for record in manifest["files"]:
            if record["sha256"]:
                handle.write(f"{record['sha256']}  {record['path']}\n")

    return json_path, sha256_path


def print_manifest_summary(manifest, json_path, sha256_path):
    summary = manifest["summary"]
    print(f"Repository: {manifest['repo_id']}@{manifest['resolved_commit']}")
    print(
        "Remote tree: "
        f"{summary['repository_files']} files, "
        f"{summary['repository_bytes']:,} bytes"
    )
    print(
        "SHA-256 coverage: "
        f"{summary['sha256_payload_files']} payload files, "
        f"{summary['sha256_payload_bytes']:,} bytes"
    )
    print(
        "Git blob OID only: "
        f"{summary['git_blob_only_files']} small repository files, "
        f"{summary['git_blob_only_bytes']:,} bytes"
    )
    print(f"Wrote JSON manifest: {json_path}")
    print(f"Wrote SHA-256 list: {sha256_path}")


def get_file_list(records, subset, download_core=True, download_metadata=False, metadata_only=False):
    """Return repository paths selected by the downloader flags."""
    all_files = [record["path"] for record in records]
    core_files = [filename for filename in all_files if is_core_file(filename)]
    metadata_archives = [filename for filename in all_files if is_metadata_archive(filename)]

    if metadata_only:
        print(f"Found {len(metadata_archives)} metadata archives.")
        return metadata_archives

    image_tars = [filename for filename in all_files if is_image_archive(filename)]
    if subset == "all":
        selected_images = image_tars
    elif subset == "accept":
        selected_images = [filename for filename in image_tars if is_accepted(filename)]
    else:
        selected_images = [filename for filename in image_tars if not is_accepted(filename)]

    print(f"Found {len(core_files)} core files.")
    print(f"Found {len(selected_images)} image archives for subset {subset!r}.")
    print(f"Found {len(metadata_archives)} optional metadata archives.")

    selected = []
    if download_core:
        selected.extend(core_files)
    selected.extend(selected_images)
    if download_metadata:
        selected.extend(metadata_archives)
    return selected


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_extract(filename, expected_sha256, repo_id, revision, dest_dir):
    """Download one file, verify its SHA-256 when available, and extract tarballs."""
    print(f"Downloading {filename}...")
    try:
        file_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                local_dir=dest_dir,
            )
        )

        if expected_sha256:
            actual_sha256 = sha256_file(file_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"SHA-256 mismatch for {filename}: expected {expected_sha256}, "
                    f"got {actual_sha256}"
                )
            print(f"Verified SHA-256: {actual_sha256}")
        else:
            print("No raw-file SHA-256 was exposed by repository metadata; skipping local hash check.")

        if filename.endswith(".tar.gz"):
            print(f"Extracting {filename}...")
            with tarfile.open(file_path, "r:gz") as archive:
                archive.extractall(path=dest_dir)
            file_path.unlink()
        else:
            print(f"Saved: {file_path}")
        return True
    except Exception as error:
        print(f"Failed to process {filename}: {error}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate DiagramBank hash manifests without downloading data, or download "
            "and verify selected release files."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write remote hash manifests and exit without downloading repository files.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help=(
            "Directory for generated manifests. Defaults to the current directory in "
            "manifest-only mode and FIG_RAG_DIR during downloads."
        ),
    )
    parser.add_argument(
        "--subset",
        choices=["accept", "reject", "all"],
        default="accept",
        help="Which image archives to download (default: accept).",
    )
    parser.add_argument(
        "--no-core",
        dest="download_core",
        action="store_false",
        default=True,
        help="Skip data.jsonl, FAISS indices, and research databases.",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Also download raw reproduction metadata archives.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only raw reproduction metadata archives.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    api = HfApi()

    print(f"Reading remote metadata from {args.repo_id}@{args.revision}...")
    manifest = build_remote_manifest(api, args.repo_id, args.revision)

    if args.manifest_only:
        manifest_dir = args.manifest_dir or Path.cwd()
        paths = write_manifest(manifest, manifest_dir)
        print_manifest_summary(manifest, *paths)
        return 0

    fig_rag_dir = os.environ.get("FIG_RAG_DIR")
    if not fig_rag_dir:
        print("Error: FIG_RAG_DIR is not set.")
        print("Run: export FIG_RAG_DIR=/path/to/your/data")
        return 1

    dest_dir = Path(fig_rag_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.manifest_dir or dest_dir
    paths = write_manifest(manifest, manifest_dir)
    print_manifest_summary(manifest, *paths)

    records = manifest["files"]
    records_by_path = {record["path"]: record for record in records}
    files_to_download = get_file_list(
        records,
        args.subset,
        args.download_core,
        args.metadata or args.metadata_only,
        args.metadata_only,
    )
    if not files_to_download:
        print("No files found matching the selected options.")
        return 1

    resolved_commit = manifest["resolved_commit"]
    print(f"Starting download of {len(files_to_download)} files to {dest_dir}...")
    failures = 0
    for filename in files_to_download:
        if not download_and_extract(
            filename,
            records_by_path[filename]["sha256"],
            args.repo_id,
            resolved_commit,
            dest_dir,
        ):
            failures += 1

    if failures:
        print(f"Completed with {failures} failed file(s).")
        return 1

    print("Download, hash verification, and extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
