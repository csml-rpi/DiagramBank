#!/usr/bin/env python3
"""Extract released FAISS vectors and embed context/query text with the same model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import duckdb
import faiss
import numpy as np
from openai import OpenAI


MODEL = "text-embedding-3-large"
DIMENSION = 3072
FIELD_INDEX_NAMES = {
    "title": "title_index",
    "abstract": "abstract_index",
    "caption": "caption_index",
}


def clean(value: object) -> str:
    return " ".join(str(value or "").split())


def load_complete_records(db_path: Path) -> list[dict]:
    connection = duckdb.connect(str(db_path), read_only=True)
    rows = connection.execute(
        """
        SELECT
            platform_id, figure_number, figure_path, venue, year,
            title, abstract, figure_caption, figure_context
        FROM Data
        WHERE trim(coalesce(title, '')) <> ''
          AND trim(coalesce(abstract, '')) <> ''
          AND trim(coalesce(figure_caption, '')) <> ''
          AND trim(coalesce(figure_context, '')) <> ''
        ORDER BY platform_id, figure_number, figure_path
        """
    ).fetchall()
    columns = [description[0] for description in connection.description]
    connection.close()
    return [dict(zip(columns, row)) for row in rows]


def record_key(record: dict) -> tuple[str, str]:
    return str(record["platform_id"]), str(record["figure_number"])


def write_candidate_manifest(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(
                json.dumps(
                    {
                        "candidate_index": index,
                        "platform_id": str(record["platform_id"]),
                        "figure_number": str(record["figure_number"]),
                        "figure_path": str(record["figure_path"]),
                        "venue": str(record["venue"]),
                        "year": int(record["year"]),
                        "title_words": len(clean(record["title"]).split()),
                        "abstract_words": len(clean(record["abstract"]).split()),
                        "caption_words": len(clean(record["figure_caption"]).split()),
                        "context_words": len(clean(record["figure_context"]).split()),
                        "paper_text_sha256": hashlib.sha256(
                            f"{clean(record['title'])}\n{clean(record['abstract'])}".encode("utf-8")
                        ).hexdigest(),
                        "context_sha256": hashlib.sha256(
                            clean(record["figure_context"]).encode("utf-8")
                        ).hexdigest(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("embedding matrix contains a zero vector")
    return values / norms


def extract_index_vectors(
    index_dir: Path,
    candidate_records: list[dict],
    expected_content_field: str,
) -> np.ndarray:
    index = faiss.read_index(str(index_dir / "index.faiss"))
    if index.d != DIMENSION:
        raise RuntimeError(f"{index_dir} dimension {index.d} != {DIMENSION}")
    with (index_dir / "index.pkl").open("rb") as handle:
        docstore, index_to_docstore_id = pickle.load(handle)
    if index.ntotal != len(index_to_docstore_id):
        raise RuntimeError(f"{index_dir} FAISS/docstore row mismatch")

    source_by_key: dict[tuple[str, str], int] = {}
    for source_index in range(index.ntotal):
        document = docstore.search(index_to_docstore_id[source_index])
        key = (
            str(document.metadata["platform_id"]),
            str(document.metadata["figure_number"]),
        )
        if key in source_by_key:
            raise RuntimeError(f"duplicate key in {index_dir}: {key}")
        source_by_key[key] = source_index

    missing = [record_key(record) for record in candidate_records if record_key(record) not in source_by_key]
    if missing:
        raise RuntimeError(f"{index_dir} is missing {len(missing)} complete-record keys")

    output = np.empty((len(candidate_records), DIMENSION), dtype=np.float32)
    for output_index, record in enumerate(candidate_records):
        source_index = source_by_key[record_key(record)]
        output[output_index] = index.reconstruct(source_index)
        document = docstore.search(index_to_docstore_id[source_index])
        expected = clean(record[expected_content_field])
        if clean(document.page_content) != expected:
            raise RuntimeError(
                f"{index_dir} content mismatch for {record_key(record)}"
            )
    return normalize_rows(output)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_text_sha256(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = clean(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def embed_texts(
    texts: list[str],
    output_path: Path,
    progress_path: Path,
    batch_size: int,
) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to embed new context/query text")

    input_sha256 = ordered_text_sha256(texts)
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    expected_metadata = {
        "model": MODEL,
        "dimension": DIMENSION,
        "records": len(texts),
        "input_texts_sha256": input_sha256,
    }
    if output_path.exists():
        values = np.load(output_path, mmap_mode="r")
        if values.shape != (len(texts), DIMENSION):
            raise RuntimeError(f"unexpected existing cache shape: {values.shape}")
        if not metadata_path.exists():
            raise RuntimeError(f"completed cache lacks input-binding metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError(f"completed cache metadata does not match current inputs: {output_path}")
        if metadata.get("embedding_sha256") != sha256_file(output_path):
            raise RuntimeError(f"completed cache hash mismatch: {output_path}")
        print(f"Using completed embedding cache: {output_path}")
        return

    partial_path = output_path.with_suffix(".partial.npy")
    if partial_path.exists() and progress_path.exists():
        matrix = np.lib.format.open_memmap(partial_path, mode="r+")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress["completed"])
        if matrix.shape != (len(texts), DIMENSION):
            raise RuntimeError("partial embedding cache shape does not match input")
        if any(progress.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError("partial embedding cache metadata does not match input")
    else:
        matrix = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(texts), DIMENSION),
        )
        completed = 0

    client = OpenAI()
    for start in range(completed, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        attempts = 0
        while True:
            try:
                response = client.embeddings.create(model=MODEL, input=texts[start:stop])
                break
            except Exception:
                attempts += 1
                if attempts >= 6:
                    raise
                time.sleep(min(60, 2**attempts))
        batch = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        if batch.shape != (stop - start, DIMENSION):
            raise RuntimeError(f"unexpected embedding response shape: {batch.shape}")
        matrix[start:stop] = normalize_rows(batch)
        matrix.flush()
        progress_path.write_text(
            json.dumps(
                {
                    **expected_metadata,
                    "completed": stop,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Embedded {stop}/{len(texts)}", flush=True)

    del matrix
    partial_path.rename(output_path)
    metadata_path.write_text(
        json.dumps(
            {
                **expected_metadata,
                "embedding_sha256": sha256_file(output_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_queries(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(rows, key=lambda row: row["query_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--faiss-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--stage",
        choices=("extract-released", "embed-context", "embed-queries", "all"),
        default="all",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_complete_records(args.db)
    manifest_path = args.output_dir / "candidate_manifest.jsonl"
    write_candidate_manifest(manifest_path, records)

    if args.stage in ("extract-released", "all"):
        source_fields = {
            "title": "title",
            "abstract": "abstract",
            "caption": "figure_caption",
        }
        for field, index_name in FIELD_INDEX_NAMES.items():
            output_path = args.output_dir / f"candidate_{field}.f16.npy"
            vectors = extract_index_vectors(
                args.faiss_dir / index_name,
                records,
                source_fields[field],
            )
            if output_path.exists():
                existing = np.load(output_path, mmap_mode="r")
                if existing.shape != vectors.shape:
                    raise RuntimeError(f"extracted cache shape mismatch: {output_path}")
                for start in range(0, len(vectors), 512):
                    stop = min(start + 512, len(vectors))
                    if not np.allclose(
                        np.asarray(existing[start:stop], dtype=np.float32),
                        vectors[start:stop],
                        atol=7e-4,
                    ):
                        raise RuntimeError(f"extracted cache content mismatch: {output_path}")
                print(f"Validated extracted cache: {output_path}")
                continue
            np.save(output_path, vectors.astype(np.float16))
            print(f"Saved {output_path} {vectors.shape}")

    if args.stage in ("embed-context", "all"):
        embed_texts(
            [clean(record["figure_context"]) for record in records],
            args.output_dir / "candidate_context.f32.npy",
            args.output_dir / "candidate_context.progress.json",
            args.batch_size,
        )

    if args.stage in ("embed-queries", "all"):
        if args.queries is None:
            raise ValueError("--queries is required for query embedding")
        queries = load_queries(args.queries)
        embed_texts(
            [clean(row["query"]) for row in queries],
            args.output_dir / "queries.f32.npy",
            args.output_dir / "queries.progress.json",
            args.batch_size,
        )
        (args.output_dir / "query_order.json").write_text(
            json.dumps([row["query_id"] for row in queries], indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "model": MODEL,
        "dimension": DIMENSION,
        "database_sha256": sha256_file(args.db),
        "candidate_records": len(records),
        "candidate_manifest": str(manifest_path),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "ordered_source_text_sha256": {
            "title": ordered_text_sha256([record["title"] for record in records]),
            "abstract": ordered_text_sha256([record["abstract"] for record in records]),
            "caption": ordered_text_sha256([record["figure_caption"] for record in records]),
            "context": ordered_text_sha256([record["figure_context"] for record in records]),
        },
        "files": {},
    }
    if args.queries is not None:
        query_rows = load_queries(args.queries)
        manifest["query_file"] = str(args.queries)
        manifest["query_file_sha256"] = sha256_file(args.queries)
        manifest["ordered_query_text_sha256"] = ordered_text_sha256(
            [row["query"] for row in query_rows]
        )
    for path in sorted(args.output_dir.glob("*.npy")):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    (args.output_dir / "embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
