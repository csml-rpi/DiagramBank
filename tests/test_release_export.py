import importlib.util
from pathlib import Path
import tempfile
import unittest

import duckdb

SOURCE = Path(__file__).resolve().parents[1] / "tools/export_release.py"
spec = importlib.util.spec_from_file_location("release_export", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseExportTests(unittest.TestCase):
    def test_source_preserved_and_only_missing_citations_filled(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.db"
            output = Path(directory) / "new-release"
            with duckdb.connect(str(source)) as con:
                con.execute("CREATE TABLE Data(platform_id VARCHAR, figure_number VARCHAR, title VARCHAR, abstract VARCHAR, authors VARCHAR, author_ids VARCHAR, year BIGINT, url VARCHAR, bibtex VARCHAR, scores BIGINT[], confidence FLOAT)")
                for number, citation in (("1", None), ("2", "@misc{existing, title={Original}}")):
                    con.execute("INSERT INTO Data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                ["paper-1", number, r"A {method} with $x^2$ & 50%", "text", "First Author, Second Author", "~First1, ~Second1", 2023, "https://example.test/paper-1", citation, [1, None, 3], 0.75])
            before = source.read_bytes()
            report = module.export_release(source, output)
            self.assertEqual(before, source.read_bytes())
            self.assertEqual(report["generated_citation_records"], 1)
            _, rows = module.read_rows(output / "research.db")
            self.assertEqual(rows[1]["bibtex"], "@misc{existing, title={Original}}")
            self.assertEqual(rows[0]["scores"], [1, None, 3])
            with self.assertRaises(FileExistsError):
                module.export_release(source, output)

    def test_ambiguous_author_lists_are_left_for_review(self):
        row = dict(platform_id="x", title="Title", authors="One, Two", author_ids="~One1", year=2023, url="https://example.test/x")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            module.basic_bibtex(row)


if __name__ == "__main__":
    unittest.main()
