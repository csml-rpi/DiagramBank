"""Build joinable cascade labels and metadata-match groups from saved local files.

Never modifies the source database, historical labels, or an existing output.
The saved labels are normalized model outputs, not raw responses or human labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import duckdb

FINALIZER = Path(__file__).resolve().parents[1] / "cascade/finalize.py"
spec = importlib.util.spec_from_file_location("cascade_finalize", FINALIZER)
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)
LABELS = {"diagram", "plot", "photo", "other"}
QUERIED_CATEGORIES = {"t1_majority_diagram", "t1_minority_diagram", "t2_vlm_consensus"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row):
    return str(row["platform_id"]), str(row["figure_number"])


def load_unique(path):
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                identity = key(row)
                if identity in rows:
                    raise ValueError(f"Duplicate input key in {Path(path).name}: {identity}")
                rows[identity] = row
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n")


def normalized(value):
    return " ".join(str(value or "").casefold().split())


def expected_category(row):
    if row.get("clip_type") not in LABELS:
        raise ValueError(f"Invalid CLIP label: {key(row)}")
    for field in ("label_haiku", "label_gpt_mini"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip() or value in {"MISSING", "PENDING", "ERROR", "RATE_LIMITED"}:
            raise ValueError(f"Missing or failed model label: {key(row)}, {field}")
    confidence = row.get("clip_confidence")
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError(f"Invalid CLIP confidence: {key(row)}")
    positives = sum(row[field] == "diagram" for field in ("label_haiku", "label_gpt_mini"))
    if row["tier"] == "tier1" and row["clip_type"] == "diagram":
        return ("t1_minority_diagram", "t1_majority_diagram", "t1_unanimous_diagram")[positives]
    # Historical logs round confidence to four places; a score below .70 can
    # serialize as .7000. Preserve its recorded tier instead of moving it.
    if row["tier"] == "tier2" and row["clip_type"] != "diagram" and confidence <= 0.70:
        return ("t2_no_vlm_diagram", "t2_one_vlm_diagram", "t2_vlm_consensus")[positives]
    raise ValueError(f"Inconsistent candidate tier: {key(row)}")


def metadata_matches(rows):
    papers, counts = {}, Counter()
    for row in rows:
        pid = row["platform_id"]
        metadata = {f: row[f] for f in ("title", "abstract", "venue", "year", "url")}
        if pid in papers and papers[pid] != metadata:
            raise ValueError(f"Conflicting metadata within paper: {pid}")
        papers[pid] = metadata
        counts[pid] += 1
    matches = defaultdict(list)
    for pid, row in sorted(papers.items()):
        text = normalized(row["title"]), normalized(row["abstract"])
        if all(text):
            matches[text].append(pid)
    groups, membership = [], {}
    for text, pids in sorted(matches.items()):
        if len(pids) < 2:
            continue
        # Full SHA-256 of the normalized strings gives order-independent IDs.
        group_id = "title-abstract-" + hashlib.sha256(json.dumps(text, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
        for pid in pids:
            membership[pid] = group_id
        groups.append({"metadata_match_group_id": group_id,
                       "title": papers[pids[0]]["title"],
                       "normalized_abstract_sha256": hashlib.sha256(text[1].encode()).hexdigest(),
                       "review_status": "unreviewed_metadata_match",
                       "papers": [{"platform_id": pid, "venue": papers[pid]["venue"],
                                   "year": papers[pid]["year"], "url": papers[pid]["url"],
                                   "released_figures": counts[pid]} for pid in pids]})
    if len({g["metadata_match_group_id"] for g in groups}) != len(groups):
        raise ValueError("Metadata group ID collision")
    paper_map = [{"platform_id": pid, "metadata_match_group_id": membership.get(pid)} for pid in sorted(papers)]
    summary = {"groups": len(groups), "papers_in_groups": len(membership),
               "figures_in_groups": sum(counts[pid] for pid in membership),
               "group_size_counts": dict(sorted(Counter(len(g["papers"]) for g in groups).items())),
               "groups_crossing_venues": sum(len({p["venue"] for p in g["papers"]}) > 1 for g in groups),
               "groups_crossing_years": sum(len({p["year"] for p in g["papers"]}) > 1 for g in groups),
               "groups_by_venue_year": dict(sorted(Counter(" / ".join(sorted({f'{p["venue"]} {p["year"]}' for p in g["papers"]})) for g in groups).items()))}
    return groups, paper_map, summary


def build_supplement(source_db, voted_path, tiebreak_path, output):
    source_db, voted_path, tiebreak_path, output = map(Path, (source_db, voted_path, tiebreak_path, output))
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    inputs = {"release_database": source_db, "voted_labels": voted_path, "gpt_labels": tiebreak_path}
    input_info = {role: {"filename": p.name, "sha256": sha256_file(p)} for role, p in inputs.items()}
    voted, tiebreak = load_unique(voted_path), load_unique(tiebreak_path)
    if set(tiebreak) != {k for k, r in voted.items() if r["vote_category"] in QUERIED_CATEGORIES}:
        raise ValueError("GPT label keys do not match the recorded query branches")
    kept, unexpected_labels = {}, []
    for identity, row in voted.items():
        if row["vote_category"] != expected_category(row):
            raise ValueError(f"Vote category disagrees with saved labels: {identity}")
        gpt_label = None
        if identity in tiebreak:
            response = tiebreak[identity]
            for field in ("venue", "figure_path", "clip_type", "clip_confidence", "label_haiku", "label_gpt_mini", "vote_category"):
                if response[field] != row[field]:
                    raise ValueError(f"GPT input disagrees with voted record: {identity}, {field}")
            gpt_label = response["label_gpt_tiebreak"]
            if gpt_label not in LABELS:
                raise ValueError(f"Invalid normalized GPT label: {identity}")
        path = finalizer.final_keep_path(row, gpt_label)
        for field in ("label_haiku", "label_gpt_mini"):
            if row[field] not in LABELS:
                unexpected_labels.append({"platform_id": identity[0], "figure_number": identity[1],
                                          "field": field, "saved_label": row[field], "vote_category": row["vote_category"],
                                          "cascade_path": path, "retained": path is not None})
        if path:
            kept[identity] = {"platform_id": identity[0], "figure_number": identity[1],
                              "clip_type": row["clip_type"], "clip_confidence": row["clip_confidence"],
                              "label_haiku": row["label_haiku"], "label_gpt_mini": row["label_gpt_mini"],
                              "label_gpt_tiebreak": gpt_label, "cascade_path": path, "label_cascade": "keep"}

    fields = ["platform_id", "figure_number", "venue", "figure_path", "figure_type", "confidence", "title", "abstract", "year", "url"]
    with duckdb.connect(str(source_db), read_only=True) as con:
        records = [dict(zip(fields, row)) for row in con.execute("SELECT " + ",".join(fields) + " FROM Data ORDER BY platform_id, figure_number").fetchall()]
    release_keys = {key(r) for r in records}
    if len(release_keys) != len(records) or release_keys != set(kept):
        raise ValueError("Cascade keep keys do not equal the unique database keys")
    for record in records:
        identity = key(record)
        vote = voted[identity]
        for db_field, vote_field in (("venue", "venue"), ("figure_path", "figure_path"), ("figure_type", "clip_type")):
            if record[db_field] != vote[vote_field]:
                raise ValueError(f"Release metadata mismatch: {identity}, {db_field}")
        if round(record["confidence"], 4) != round(vote["clip_confidence"], 4):
            raise ValueError(f"Release confidence mismatch at saved four-decimal precision: {identity}")
    groups, paper_map, match_summary = metadata_matches(records)
    if any(sha256_file(p) != input_info[role]["sha256"] for role, p in inputs.items()):
        raise ValueError("Input changed during supplement construction")

    output.mkdir(parents=True)
    write_jsonl(output / "cascade_labels.jsonl", (kept[k] for k in sorted(kept)))
    write_jsonl(output / "paper_groups.jsonl", paper_map)
    write_json(output / "metadata_groups.json", groups)
    write_json(output / "unexpected_labels.json", sorted(unexpected_labels, key=lambda r: (*key(r), r["field"])))
    schema = {
        "schema_version": "1.0-local-supplement",
        "cascade_labels.jsonl": {
            "record_key": ["platform_id", "figure_number"],
            "coverage": "One row for every primary-release record; label_cascade is always keep.",
            "fields": {"platform_id": "string", "figure_number": "string", "clip_type": "string: diagram, plot, photo, or other",
                       "clip_confidence": "number: saved CLIP confidence, rounded to four decimal places in the logs",
                       "label_haiku": "string: normalized saved model label; out-of-vocabulary strings preserved and listed in unexpected_labels.json", "label_gpt_mini": "string: normalized saved model label",
                       "label_gpt_tiebreak": "string or null: normalized saved GPT label; null means no recorded query on the unanimous keep path",
                       "cascade_path": "string: retained branch identifier", "label_cascade": "string: keep"},
            "database_mapping": {"figure_type": "clip_type", "confidence": "clip_confidence (compared at four decimal places)"},
            "gpt_role_by_path": {"t1_unanimous_diagram": "not queried", "t1_majority_diagram": "informational; inclusion independent of GPT label",
                                 "t1_minority_diagram_gpt_confirmed": "diagram confirmation required", "t2_vlm_consensus_gpt_confirmed": "diagram confirmation required"},
            "label_provenance": "Normalized historical model labels only; raw response strings and actual model checkpoint provenance are not reconstructed.",
            "unexpected_output_policy": "Historical voting counts only the saved exact string diagram as positive. Other nonempty outputs are preserved as non-diagram votes, without mapping to one of the other rubric classes. ERROR, MISSING, PENDING, and RATE_LIMITED stop this exporter.",
            "tier_boundary_policy": "Tier-2 scores can round to 0.7000; retain the recorded candidate tier at that boundary because the threshold precedes log rounding."},
        "paper_groups.jsonl": {
            "record_key": ["platform_id"], "coverage": "Every paper identifier in the source release, exactly once.",
            "fields": {"platform_id": "string", "metadata_match_group_id": "string or null"},
            "null_policy": "null means no other paper identifier shares both nonempty normalized title and abstract; it is not evidence of image uniqueness.",
            "normalization": "Unicode casefold, split on whitespace and rejoin with one space; punctuation preserved.",
            "group_id": "title-abstract- followed by full SHA-256 of ASCII JSON [normalized_title,normalized_abstract], with compact separators.",
            "split_guidance": "When defining paper splits, use the group ID for matching papers and a namespaced paper identifier for other papers; keep all figures of that unit together. Image similarity and publication-version aliases need separate checks."},
        "metadata_groups.json": {"coverage": "Groups with at least two distinct paper identifiers; review candidates, not confirmed duplicate publications or images."}}
    write_json(output / "schema.json", schema)
    report = {"publication_status": "Local supplement; hosted files and original database unchanged.",
              "inputs": input_info, "generator_sha256": sha256_file(__file__), "finalizer_sha256": sha256_file(FINALIZER),
              "candidate_records": len(voted), "recorded_gpt_queries": len(tiebreak),
              "released_records": len(records), "released_papers": len(paper_map),
              "exact_release_key_match": True, "release_venue_path_clip_match": True,
              "confidence_comparison_decimals": 4,
              "cascade_path_counts": dict(sorted(Counter(r["cascade_path"] for r in kept.values()).items())),
              "null_gpt_labels": sum(r["label_gpt_tiebreak"] is None for r in kept.values()),
              "unexpected_label_records": len({key(r) for r in unexpected_labels}),
              "retained_unexpected_label_records": len({key(r) for r in unexpected_labels if r["retained"]}),
              "unexpected_label_values": dict(sorted(Counter(r["saved_label"] for r in unexpected_labels).items())),
              "tier2_scores_rounded_to_threshold": sum(r["tier"] == "tier2" and r["clip_confidence"] == 0.70 for r in voted.values()),
              "metadata_matches": match_summary, "inputs_unchanged": True,
              "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(output.iterdir())}}
    write_json(output / "manifest.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--voted", type=Path, required=True)
    parser.add_argument("--tiebreak", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_supplement(args.source_db, args.voted, args.tiebreak, args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k not in {"inputs", "outputs"}}, indent=2))
