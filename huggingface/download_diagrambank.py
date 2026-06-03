import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import tarfile

# --- CONFIGURATION ---
REPO_ID = "ghzlmc/DiagramBank"

# 1. Setup Destination
FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR")
if not FIG_RAG_DIR:
    print("❌ Error: FIG_RAG_DIR environment variable is not set.")
    print("   Please run: export FIG_RAG_DIR=/path/to/your/data")
    exit(1)

DEST_DIR = Path(FIG_RAG_DIR)
DEST_DIR.mkdir(parents=True, exist_ok=True)


def is_accepted(filename):
    """
    Applies the specific logic to determine if a file represents an accepted paper.
    Logic: (decision like '%accept%' OR '%spotlight%' OR '%poster%' OR '%oral%')
    """
    name_lower = filename.lower()
    keywords = ["accept", "spotlight", "poster", "oral"]
    return any(k in name_lower for k in keywords)


def is_core_file(filename):
    """Return True for release metadata, FAISS, and database artifacts."""
    return (
        filename == "data.jsonl"
        or filename == "SHA256SUMS"
        or filename == "data/faiss_complete.tar.gz"
        or filename.endswith("_research_db.tar.gz")
    )


def is_metadata_archive(filename):
    """Return True for optional raw reproduction metadata archives."""
    return filename.startswith("reproduce/") and filename.endswith("_metadata.tar.gz")


def get_file_list(api, subset, download_core=True, download_metadata=False, metadata_only=False):
    """Returns a list of filenames to download based on the subset and flags."""
    print(f"🔍 Fetching file list from {REPO_ID}...")
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

    # 1. Infrastructure files (dataset JSONL, DBs, FAISS)
    core_files = [f for f in all_files if is_core_file(f)]
    metadata_archives = [f for f in all_files if is_metadata_archive(f)]

    if metadata_only:
        print(f"   Found {len(metadata_archives)} metadata archives.")
        print("   ✅ Downloading only raw reproduction metadata archives.")
        return metadata_archives

    # 2. Decision Tarballs (the images)
    image_tars = [
        f for f in all_files
        if f.startswith("data/") and f.endswith(".tar.gz") and not is_core_file(f)
    ]

    if subset == "all":
        selected_images = image_tars
    elif subset == "accept":
        selected_images = [f for f in image_tars if is_accepted(f)]
    elif subset == "reject":
        selected_images = [f for f in image_tars if not is_accepted(f)]
    else:
        selected_images = []

    print(f"   Found {len(core_files)} core files (data.jsonl/DBs/FAISS).")
    print(f"   Found {len(selected_images)} image archives for subset '{subset}'.")
    print(f"   Found {len(metadata_archives)} optional metadata archives.")

    files = []
    if download_core:
        print("   ✅ Including core files (data.jsonl/FAISS/DBs) in download list.")
        files.extend(core_files)
    else:
        print("   Example: Skipping core files.")

    files.extend(selected_images)

    if download_metadata:
        print("   ✅ Including raw reproduction metadata archives in download list.")
        files.extend(metadata_archives)

    return files


def download_and_extract(filename):
    """Downloads a single file and extracts tarballs to FIG_RAG_DIR."""
    print(f"⬇️  Downloading {filename}...")
    try:
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=DEST_DIR,
            local_dir_use_symlinks=False,
        )

        if filename.endswith(".tar.gz"):
            print(f"   📦 Extracting {filename}...")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=DEST_DIR)

            os.remove(file_path)
            print(f"   ✅ Done: {filename}")
        else:
            print(f"   ✅ Saved: {file_path}")

    except Exception as e:
        print(f"   ❌ Failed to process {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DiagramBank Data.")

    parser.add_argument(
        "--subset",
        choices=["accept", "reject", "all"],
        default="accept",
        help="Which set of papers to download? (Default: accept)",
    )

    # Flag to control core files (Default: True)
    # Using 'store_false' means the variable 'download_core' will be True
    # unless the user explicitly provides --no-core
    parser.add_argument(
        "--no-core",
        dest="download_core",
        action="store_false",
        default=True,
        help="Skip downloading data.jsonl, FAISS indices, and research.db files.",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Also download raw reproduction metadata archives for ICLR/ICML.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only raw reproduction metadata archives for ICLR/ICML.",
    )

    args = parser.parse_args()

    api = HfApi()

    # Get the list of files to process
    files_to_download = get_file_list(
        api,
        args.subset,
        args.download_core,
        args.metadata or args.metadata_only,
        args.metadata_only,
    )

    if not files_to_download:
        print("No files found matching criteria.")
        exit()

    print(f"🚀 Starting download of {len(files_to_download)} files to {DEST_DIR}...")

    for filename in files_to_download:
        download_and_extract(filename)

    print("\n🎉 Download and extraction complete!")
