"""Regression checks for release membership, including both confirmation paths."""
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "cascade/finalize.py"
spec = importlib.util.spec_from_file_location("cascade_finalize", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CascadeFinalizationTests(unittest.TestCase):
    def decide(self, category, label=None):
        return module.final_keep_path({"vote_category": category}, label)

    def test_both_confirmation_paths_accept_only_diagram(self):
        for category in ("t1_minority_diagram", "t2_vlm_consensus"):
            with self.subTest(category=category):
                self.assertIsNotNone(self.decide(category, "diagram"))
                for label in ("plot", "photo", "other"):
                    self.assertIsNone(self.decide(category, label))

    def test_original_majority_is_not_overruled_by_informational_tiebreak(self):
        self.assertEqual(self.decide("t1_majority_diagram", "plot"), "t1_majority_diagram")
        self.assertEqual(self.decide("t1_unanimous_diagram"), "t1_unanimous_diagram")

    def test_one_tier_two_vote_cannot_be_rescued(self):
        self.assertIsNone(self.decide("t2_one_vlm_diagram", "diagram"))
        self.assertIsNone(self.decide("t2_no_vlm_diagram", "diagram"))

    def test_incomplete_labels_do_not_silently_produce_a_smaller_release(self):
        for category in ("t1_minority_diagram", "t2_vlm_consensus", "pending", "minority_diagram"):
            with self.subTest(category=category), self.assertRaises(ValueError):
                self.decide(category)


if __name__ == "__main__":
    unittest.main()
