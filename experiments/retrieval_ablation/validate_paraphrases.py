#!/usr/bin/env python3
"""Validate query coverage, format, length, and obvious source-text leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")
RAW_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\b")
FIGURE_RE = re.compile(r"\b(?:figure|fig\.?)[\s#:]*\d+\b", re.IGNORECASE)
COMMON_TECHNICAL_TERMS = {
    "and",
    "ai",
    "bert",
    "cnn",
    "dnn",
    "gan",
    "gnn",
    "gpt",
    "gpu",
    "llm",
    "lstm",
    "mcmc",
    "ml",
    "mlp",
    "mri",
    "nlp",
    "ode",
    "or",
    "pde",
    "rag",
    "rgb",
    "rl",
    "rnn",
    "sde",
    "ssl",
    "svm",
    "transformer",
    "transformers",
    "vae",
    "vit",
}


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def normalized(text: str) -> str:
    return " ".join(tokens(text))


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    values = tokens(text)
    return {tuple(values[index : index + n]) for index in range(len(values) - n + 1)}


def distinctive_source_terms(source: dict) -> set[str]:
    output = set()
    title = str(source["title"])
    caption = str(source["figure_caption"])
    title_prefix = title.split(":", 1)[0].strip()
    prefix_tokens = RAW_TOKEN_RE.findall(title_prefix)
    if len(prefix_tokens) == 1:
        prefix = prefix_tokens[0]
        if prefix.isupper() and 3 <= len(prefix) <= 12:
            output.add(prefix.casefold())

    for raw in RAW_TOKEN_RE.findall(f"{title}\n{caption}"):
        term = raw.casefold()
        if term in COMMON_TECHNICAL_TERMS:
            continue
        is_mixed_case_name = (
            "-" not in raw
            and "_" not in raw
            and not raw.isupper()
            and any(character.isupper() for character in raw[1:])
        )
        is_coined = any(character.isdigit() for character in raw) or is_mixed_case_name
        if is_coined:
            output.add(term)

    for raw in RAW_TOKEN_RE.findall(caption):
        term = raw.casefold()
        if term not in COMMON_TECHNICAL_TERMS and raw.isupper() and 3 <= len(raw) <= 12:
            output.add(term)
    return output


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paraphrase-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sources = {row["query_id"]: row for row in load_jsonl(args.manifest)}
    outputs: dict[str, dict] = {}
    malformed = []
    duplicate_ids = []
    for path in sorted(args.paraphrase_dir.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed.append({"file": str(path), "line": line_number, "error": str(exc)})
                continue
            query_id = str(row.get("query_id", ""))
            if query_id in outputs:
                duplicate_ids.append(query_id)
            outputs[query_id] = {"query_id": query_id, "query": str(row.get("query", "")).strip()}

    accepted = []
    rejected = []
    for query_id in sorted(sources):
        source = sources[query_id]
        output = outputs.get(query_id)
        reasons = []
        if output is None:
            rejected.append({"query_id": query_id, "reasons": ["missing_output"]})
            continue

        query = output["query"]
        word_count = len(query.split())
        if word_count < 35 or word_count > 90:
            reasons.append(f"word_count_{word_count}")
        if FIGURE_RE.search(query):
            reasons.append("explicit_figure_identifier")
        title_norm = normalized(source["title"])
        query_norm = normalized(query)
        if title_norm and title_norm in query_norm:
            reasons.append("exact_title")

        source_text = "\n".join(
            str(source[field])
            for field in ("title", "abstract", "figure_caption", "figure_context")
        )
        overlap = sorted(ngrams(source_text, 6) & ngrams(query, 6))
        if overlap:
            reasons.append("source_six_token_span")

        query_terms = {token.casefold() for token in RAW_TOKEN_RE.findall(query)}
        distinctive_overlap = sorted(
            distinctive_source_terms(source) & query_terms
        )
        if distinctive_overlap:
            reasons.append("distinctive_source_term")

        if reasons:
            rejected.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "reasons": reasons,
                    "overlap_examples": [" ".join(value) for value in overlap[:3]],
                    "distinctive_term_examples": distinctive_overlap[:5],
                }
            )
        else:
            accepted.append({"query_id": query_id, "query": query})

    unexpected = sorted(set(outputs) - set(sources))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.output_dir / "accepted_queries.jsonl"
    rejected_path = args.output_dir / "rejected_queries.jsonl"
    with accepted_path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with rejected_path.open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source_records": len(sources),
        "output_records": len(outputs),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "malformed_lines": malformed,
        "duplicate_query_ids": sorted(set(duplicate_ids)),
        "unexpected_query_ids": unexpected,
        "accepted_queries_sha256": sha256_file(accepted_path),
        "rejected_queries_sha256": sha256_file(rejected_path),
    }
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
