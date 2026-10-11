"""Integration checks for source-preserving release sidecars and group identities."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import duckdb

SOURCE = Path(__file__).resolve().parents[1] / "tools/build_release_supplement.py"
spec = importlib.util.spec_from_file_location("release_supplement", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseSupplementTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory)
        db, voted, gpt, output = (base / n for n in ("source.db", "voted.jsonl", "gpt.jsonl", "supplement"))
        rows, responses = [], []
        cases = [("a", "diagram", "diagram", "diagram", None),
                 ("b", "diagram", "diagram", "plot", "plot"),
                 ("c", "diagram", "plot", "other", "diagram"),
                 ("d", "plot", "diagram", "diagram", "diagram"),
                 ("e", "plot", "diagram", "diagram", "photo")]
        with duckdb.connect(str(db)) as con:
            con.execute("CREATE TABLE Data(platform_id VARCHAR, figure_number VARCHAR, venue VARCHAR, figure_path VARCHAR, figure_type VARCHAR, confidence FLOAT, title VARCHAR, abstract VARCHAR, year BIGINT, url VARCHAR)")
            for pid, clip, haiku, mini, final in cases:
                row = dict(platform_id=pid, figure_number="1", venue="ICLR", figure_path=f"{pid}/1.png", clip_type=clip,
                           clip_confidence=0.6, tier="tier1" if clip == "diagram" else "tier2", label_haiku=haiku, label_gpt_mini=mini)
                row["vote_category"] = module.expected_category(row)
                rows.append(row)
                if final is not None:
                    responses.append(dict(row, label_gpt_tiebreak=final))
                if pid != "e":
                    title = " shared  TITLE " if pid == "b" else "Shared title" if pid == "a" else pid
                    con.execute("INSERT INTO Data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                [pid, "1", "ICLR", row["figure_path"], clip, 0.6, title, "Same abstract", 2025, f"https://example.test/{pid}"])
        module.write_jsonl(voted, rows)
        module.write_jsonl(gpt, responses)
        return db, voted, gpt, output

    def test_keep_paths_nullable_gpt_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            before = [p.read_bytes() for p in args[:3]]
            report = module.build_supplement(*args)
            self.assertEqual(before, [p.read_bytes() for p in args[:3]])
            labels = module.load_unique(args[3] / "cascade_labels.jsonl")
            self.assertEqual(set(labels), {(p, "1") for p in "abcd"})
            self.assertIsNone(labels[("a", "1")]["label_gpt_tiebreak"])
            self.assertEqual(labels[("b", "1")]["label_gpt_tiebreak"], "plot")
            self.assertEqual(len(report["cascade_path_counts"]), 4)
            self.assertEqual(report["metadata_matches"]["papers_in_groups"], 2)
            with self.assertRaises(FileExistsError):
                module.build_supplement(*args)

    def test_mismatched_database_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            with duckdb.connect(str(args[0])) as con:
                con.execute("UPDATE Data SET figure_type='photo' WHERE platform_id='a'")
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                module.build_supplement(*args)
            self.assertFalse(args[3].exists())

    def test_missing_or_duplicate_gpt_input_is_rejected(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(directory)
                lines = args[2].read_text().splitlines(True)
                args[2].write_text("".join(lines + [lines[0]] if duplicate else lines[:-1]))
                with self.assertRaisesRegex(ValueError, "Duplicate input key|GPT label keys"):
                    module.build_supplement(*args)
                self.assertFalse(args[3].exists())

    def test_invalid_label_and_category_are_rejected(self):
        for field, value in (("label_haiku", "ERROR"), ("label_haiku", "RATE_LIMITED"), ("vote_category", "t1_majority_diagram")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(directory)
                rows = list(module.load_unique(args[1]).values())
                rows[0][field] = value
                module.write_jsonl(args[1], rows)
                with self.assertRaises(ValueError):
                    module.build_supplement(*args)
                self.assertFalse(args[3].exists())

    def test_unexpected_saved_string_is_preserved_and_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            for file in args[1:3]:
                rows = list(module.load_unique(file).values())
                for row in rows:
                    if row["platform_id"] == "c":
                        row["label_haiku"] = "table"
                module.write_jsonl(file, rows)
            report = module.build_supplement(*args)
            self.assertEqual(report["unexpected_label_records"], 1)
            self.assertEqual(report["retained_unexpected_label_records"], 1)
            labels = module.load_unique(args[3] / "cascade_labels.jsonl")
            self.assertEqual(labels[("c", "1")]["label_haiku"], "table")
            self.assertTrue(json.loads((args[3] / "unexpected_labels.json").read_text())[0]["retained"])

    def test_saved_tier_is_retained_at_rounded_threshold(self):
        row = dict(platform_id="x", figure_number="1", tier="tier2", clip_type="plot",
                   clip_confidence=0.7, label_haiku="diagram", label_gpt_mini="diagram")
        self.assertEqual(module.expected_category(row), "t2_vlm_consensus")
        with self.assertRaises(ValueError):
            module.expected_category(dict(row, clip_confidence=0.7001))

    def test_group_ids_are_stable_and_empty_text_does_not_group(self):
        rows = [dict(platform_id=p, title=" A title ", abstract="Some Text", venue="ICLR", year=2025, url=p) for p in "ab"]
        rows += [dict(platform_id=p, title="", abstract="", venue="ICLR", year=2025, url=p) for p in "cd"]
        first = module.metadata_matches(rows)
        self.assertEqual(first, module.metadata_matches(list(reversed(rows))))
        self.assertEqual(first[2]["groups"], 1)
        self.assertEqual(first[2]["figures_in_groups"], 2)
        self.assertEqual([r["metadata_match_group_id"] for r in first[1][2:]], [None, None])


if __name__ == "__main__":
    unittest.main()
