"""
Validate the downloaded DiagramBank FAISS release.

Run `huggingface/download_diagrambank.py` first and set `FIG_RAG_DIR` to the
same data root. The script expects:

  - $FIG_RAG_DIR/data.jsonl
  - $FIG_RAG_DIR/faiss/research.db
  - $FIG_RAG_DIR/faiss/title_index/
  - $FIG_RAG_DIR/faiss/abstract_index/
  - $FIG_RAG_DIR/faiss/caption_index/

It verifies that the metadata and FAISS index files match the 57,100-record
cascade-filtered primary release.
"""

import json
import struct
from collections import Counter
from pathlib import Path

from utils import DATA_JSONL, FAISS_DIR, INDEX_NAMES, PRIMARY_RELEASE_SIZE


def summarize_jsonl(path: Path):
    venues = Counter()
    cascade_paths = Counter()
    rows = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            venues[record.get("venue", "")] += 1
            cascade_paths[record.get("cascade_path", "")] += 1

    return rows, venues, cascade_paths


def read_faiss_header(index_path: Path):
    with index_path.open("rb") as f:
        header = f.read(16)
    if len(header) < 16:
        raise ValueError(f"Invalid FAISS index header: {index_path}")

    magic = header[:4]
    dimension = struct.unpack("<I", header[4:8])[0]
    total_vectors = struct.unpack("<Q", header[8:16])[0]
    return magic, dimension, total_vectors


def validate_indices():
    for index_name in INDEX_NAMES:
        index_path = FAISS_DIR / index_name / "index.faiss"
        pkl_path = FAISS_DIR / index_name / "index.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing FAISS index: {index_path}. "
                "Run huggingface/download_diagrambank.py and set FIG_RAG_DIR."
            )
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Missing FAISS docstore: {pkl_path}. "
                "Run huggingface/download_diagrambank.py and set FIG_RAG_DIR."
            )

        magic, dimension, total_vectors = read_faiss_header(index_path)
        if total_vectors != PRIMARY_RELEASE_SIZE:
            raise ValueError(
                f"{index_name} has {total_vectors:,} vectors; "
                f"expected {PRIMARY_RELEASE_SIZE:,}"
            )

        print(
            f"  {index_name:<14} vectors={total_vectors:,} "
            f"dimension={dimension} magic={magic.decode(errors='replace')}"
        )


def main():
    if not DATA_JSONL.exists():
        raise FileNotFoundError(
            f"Missing release metadata: {DATA_JSONL}. "
            "Run huggingface/download_diagrambank.py and set FIG_RAG_DIR."
        )

    rows, venues, cascade_paths = summarize_jsonl(DATA_JSONL)
    if rows != PRIMARY_RELEASE_SIZE:
        raise ValueError(
            f"Expected {PRIMARY_RELEASE_SIZE:,} release records, found {rows:,}"
        )

    print(f"Validated release metadata: {DATA_JSONL}")
    print(f"Records: {rows:,}")

    print("\nVenue counts:")
    for venue, count in sorted(venues.items()):
        print(f"  {venue:<8} {count:>6,}")

    print("\nCascade path counts:")
    for path, count in sorted(cascade_paths.items()):
        print(f"  {path:<34} {count:>6,}")

    print("\nFAISS indices:")
    validate_indices()

    db_path = FAISS_DIR / "research.db"
    if db_path.exists():
        print(f"\nDuckDB metadata database: {db_path}")
    else:
        raise FileNotFoundError(
            f"Missing DuckDB metadata database: {db_path}. "
            "Run huggingface/download_diagrambank.py and set FIG_RAG_DIR."
        )


if __name__ == "__main__":
    main()
