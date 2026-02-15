import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import tarfile

# --- CONFIGURATION ---
REPO_ID = "zhangt20/DiagramBank"

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
    # Filename format from upload: {Conf}_{Year}_{Decision}.tar.gz
    # We convert to lowercase to match the SQL 'lower()' logic
    name_lower = filename.lower()
    
    # We check if ANY of the keywords exist in the filename
    keywords = ["accept", "spotlight", "poster", "oral"]
    return any(k in name_lower for k in keywords)

def get_file_list(api, subset):
    """Returns a list of filenames to download based on the subset."""
    print(f"🔍 Fetching file list from {REPO_ID}...")
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    
    # Identify the different types of files
    # 1. Infrastructure files (Always download these: DBs, FAISS)
    core_files = [f for f in all_files if "research_db" in f or "faiss" in f]
    
    # 2. Decision Tarballs (The images)
    # Filter for .tar.gz but exclude the core files we already found
    image_tars = [f for f in all_files if f.endswith(".tar.gz") and f not in core_files]
    
    selected_images = []
    
    if subset == "all":
        selected_images = image_tars
    elif subset == "accept":
        selected_images = [f for f in image_tars if is_accepted(f)]
    elif subset == "reject":
        selected_images = [f for f in image_tars if not is_accepted(f)]
        
    print(f"   Found {len(core_files)} core files (DBs/Indices).")
    print(f"   Found {len(selected_images)} image archives for subset '{subset}'.")
    
    # return core_files + 
    return selected_images

def download_and_extract(filename):
    """Downloads a single file and extracts it to FIG_RAG_DIR."""
    print(f"⬇️  Downloading {filename}...")
    try:
        # Download to local cache
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=DEST_DIR, # Downloads directly to target
            local_dir_use_symlinks=False
        )
        
        # Extract
        print(f"   📦 Extracting {filename}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=DEST_DIR)
        
        # Cleanup: Remove the tar file to save space
        os.remove(file_path)
        print(f"   ✅ Done: {filename}")
        
    except Exception as e:
        print(f"   ❌ Failed to process {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DiagramBank Data.")
    parser.add_argument(
        "--subset", 
        choices=["accept", "reject", "all"], 
        default="accept", 
        help="Which set of papers to download (accept, reject, all)? (Default: accept)"
    )
    args = parser.parse_args()

    api = HfApi()
    
    # Get the list of files to process
    files_to_download = get_file_list(api, args.subset)
    
    if not files_to_download:
        print("No files found matching criteria.")
        exit()

    print(f"🚀 Starting download of {len(files_to_download)} files to {DEST_DIR}...")
    
    # Process them one by one
    for filename in files_to_download:
        # The API returns paths like 'data/ICLR_...tar.gz', we need to handle that
        download_and_extract(filename)

    print("\n🎉 Download and extraction complete!")