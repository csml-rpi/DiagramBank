"""Export a frozen release and a separately documented citation-enriched copy.

The input database is read-only. Existing citations and all non-citation fields
are preserved. Generated citations use only stored title/author/year/URL fields.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil

import bibtexparser
import duckdb


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with duckdb.connect(str(path), read_only=True) as con:
        fields = con.execute("DESCRIBE Data").fetchall()
        rows = con.execute("SELECT * FROM Data ORDER BY platform_id, figure_number").fetchall()
    names = [field[0] for field in fields]
    return fields, [dict(zip(names, row)) for row in rows]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


def key(row):
    return (str(row["platform_id"]), str(row["figure_number"]))


def tex_text(text):
    escapes = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
               "&": r"\&", "%": r"\%", "_": r"\_", "$": r"\$",
               "#": r"\#", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(char, char) for char in str(text))


def basic_bibtex(row):
    required = ("title", "authors", "author_ids", "year", "url")
    if any(row.get(field) is None or not str(row[field]).strip() for field in required):
        raise ValueError("incomplete citation fields")
    authors = row["authors"].split(", ")
    if len(authors) != len(row["author_ids"].split(", ")):
        raise ValueError("ambiguous author-list serialization")
    pid = row["platform_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", pid):
        raise ValueError("unsupported paper identifier")
    entry = (f"@misc{{diagrambank_{pid},\n"
             f"  title = {{{{{tex_text(row['title'])}}}}},\n"
             f"  author = {{{' and '.join(tex_text(a) for a in authors)}}},\n"
             f"  year = {{{int(row['year'])}}},\n"
             f"  url = {{{row['url']}}},\n"
             "  note = {Entry generated from stored DiagramBank metadata}\n}\n")
    parsed = bibtexparser.loads(entry).entries
    if len(parsed) != 1 or parsed[0]["ID"] != "diagrambank_" + pid:
        raise ValueError("generated BibTeX did not parse")
    if parsed[0]["title"] != "{" + tex_text(row["title"]) + "}":
        raise ValueError("title changed during BibTeX serialization")
    if parsed[0]["url"] != row["url"] or parsed[0]["year"] != str(int(row["year"])):
        raise ValueError("citation identifier fields changed")
    return entry


def export_release(source, output, comparisons=()):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    original_hash = sha256_file(source)
    fields, original = read_rows(source)
    original_by_key = {key(row): row for row in original}
    if len(original_by_key) != len(original):
        raise ValueError("duplicate release keys")
    output.mkdir(parents=True)
    write_jsonl(output / "baseline.jsonl", original)

    # Group first: conflicting paper metadata must never silently determine a citation.
    paper_variants = defaultdict(set)
    missing = {}
    for row in original:
        if not row.get("bibtex") or not row["bibtex"].strip():
            pid = row["platform_id"]
            paper_variants[pid].add(tuple(row.get(k) for k in ("title", "authors", "author_ids", "year", "url")))
            missing[pid] = row
    additions, unresolved = {}, []
    for pid, row in sorted(missing.items()):
        try:
            if len(paper_variants[pid]) != 1:
                raise ValueError("conflicting paper metadata")
            additions[pid] = basic_bibtex(row)
        except ValueError as exc:
            unresolved.append({"platform_id": pid, "reason": str(exc)})

    derived_db = output / "research.db"
    shutil.copy2(source, derived_db)
    with duckdb.connect(str(derived_db)) as con:
        con.execute("BEGIN TRANSACTION")
        for pid, entry in additions.items():
            con.execute("UPDATE Data SET bibtex = ? WHERE platform_id = ? AND (bibtex IS NULL OR trim(bibtex) = '')", [entry, pid])
        con.execute("COMMIT")
    derived_fields, derived = read_rows(derived_db)
    assert fields == derived_fields
    changed_rows = 0
    for before, after in zip(original, derived, strict=True):
        assert key(before) == key(after)
        differences = {name for name in before if before[name] != after[name]}
        if differences:
            assert differences == {"bibtex"} and not (before["bibtex"] or "").strip()
            assert after["bibtex"] == additions[before["platform_id"]]
            changed_rows += 1
    write_jsonl(output / "data.jsonl", derived)
    for filename, expected in (("baseline.jsonl", original), ("data.jsonl", derived)):
        with (output / filename).open() as handle:
            for row, line in zip(expected, handle, strict=True):
                assert json.loads(line) == row

    provenance = [{"platform_id": pid, "citation_key": "diagrambank_" + pid,
                   "method": "generated_from_stored_title_authors_year_url", "publisher_metadata_retrieved": False,
                   "bibtex": entry} for pid, entry in sorted(additions.items())]
    write_jsonl(output / "citation_additions.jsonl", provenance)
    (output / "generated_citations.bib").write_text("\n".join(additions.values()), encoding="utf-8")
    assert len(bibtexparser.loads((output / "generated_citations.bib").read_text()).entries) == len(additions)

    json_types = {"VARCHAR": "string", "BIGINT": "integer", "BIGINT[]": "array of integers or null elements", "FLOAT": "number"}
    schema = {"schema_version": "1.0-local", "table": "Data", "record_key": ["platform_id", "figure_number"],
              "null_policy": "SQL NULL is JSON null. Empty strings stay empty strings. No missing field is imputed except explicitly recorded generated citations.",
              "string_fields": "authors, author_ids, keywords, and areas retain their source strings; consumers must not assume JSON arrays or split compound subject labels blindly.",
              "quality_fields": "figure_type and confidence are original CLIP outputs, including records recovered by the cascade. They are not human labels or calibrated correctness probabilities.",
              "fields": [{"name": f[0], "duckdb_type": f[1], "json_type": json_types.get(f[1], f[1]),
                          "sql_nullable": f[2] == "YES", "observed_nulls": sum(r[f[0]] is None for r in derived),
                          "observed_empty_strings": sum(r[f[0]] == "" for r in derived)} for f in fields]}
    (output / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")

    comparison_reports = []
    for path in comparisons:
        with Path(path).open() as handle:
            old_rows = [json.loads(line) for line in handle if line.strip()]
        old_keys = {key(r) for r in old_rows}
        new_keys = set(original_by_key)
        comparison_reports.append({"path": str(path), "sha256": sha256_file(path), "rows": len(old_rows),
                                   "unique_keys": len(old_keys), "shared_with_baseline": len(old_keys & new_keys),
                                   "only_in_comparison": len(old_keys - new_keys), "only_in_baseline": len(new_keys - old_keys),
                                   "interpretation": "Different local snapshots; no equivalence or collection chronology is inferred."})
    title_groups = defaultdict(set)
    text_groups = defaultdict(set)
    for row in original:
        normalize = lambda value: " ".join(str(value or "").casefold().split())
        title_groups[normalize(row["title"])].add(row["platform_id"])
        text_groups[(normalize(row["title"]), normalize(row["abstract"]))].add(row["platform_id"])
    duplicate_candidates = {"scope": "Metadata matches across distinct paper identifiers; not image deduplication or confirmed duplicate publications.",
                            "same_normalized_title": [sorted(ids) for title, ids in title_groups.items() if title and len(ids) > 1],
                            "same_normalized_title_and_abstract": [sorted(ids) for (title, abstract), ids in text_groups.items() if title and abstract and len(ids) > 1]}
    (output / "metadata_duplicate_candidates.json").write_text(json.dumps(duplicate_candidates, indent=2) + "\n")
    assert sha256_file(source) == original_hash, "Input database changed"
    manifest = {"release_name": output.name, "source_database": str(source), "source_database_sha256": original_hash,
                "records": len(original), "papers": len({r['platform_id'] for r in original}),
                "baseline_matches_source_field_for_field": True, "derived_unchanged_fields_match": True,
                "generated_citation_papers": len(additions), "generated_citation_records": changed_rows,
                "source_missing_citation_records": sum(not (r["bibtex"] or "").strip() for r in original),
                "remaining_missing_citation_records": sum(not (r["bibtex"] or "").strip() for r in derived),
                "unresolved_citations": unresolved, "comparison_snapshots": comparison_reports,
                "metadata_duplicate_title_groups": len(duplicate_candidates["same_normalized_title"]),
                "metadata_duplicate_title_abstract_groups": len(duplicate_candidates["same_normalized_title_and_abstract"]),
                "index_policy": "No historical FAISS index was changed or paired with this new export. Existing evaluation vectors remain tied to their recorded source hashes and 52,052-record manifest.",
                "publication_status": "Local derivative; hosted release and historical experiments are unchanged.",
                "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--compare-jsonl", action="append", default=[], type=Path)
    args = parser.parse_args()
    report = export_release(args.source_db, args.output_dir, args.compare_jsonl)
    print(json.dumps({k: v for k, v in report.items() if k not in {"outputs", "comparison_snapshots"}}, indent=2))
