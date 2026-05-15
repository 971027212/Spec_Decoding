import tempfile
import unittest
from pathlib import Path

from profiling import write_csv
from quality_sanity_check import OutputSelector, compare_outputs, first_diff_index, read_summary


class QualitySanityCheckTests(unittest.TestCase):
    def test_compare_outputs_reports_exact_and_similar_matches(self):
        rows = [
            {
                "placement": "cloud",
                "network_profile": "wan",
                "concurrency_level": "1",
                "prompt_id": "0",
                "run_index": "0",
                "concurrent_worker": "0",
                "warmup": "False",
                "output_text": "Hello world",
            },
            {
                "placement": "edge",
                "network_profile": "lan",
                "concurrency_level": "1",
                "prompt_id": "0",
                "run_index": "0",
                "concurrent_worker": "0",
                "warmup": "False",
                "output_text": "hello   world ",
            },
        ]

        detail_rows, aggregate_rows = compare_outputs(
            rows,
            reference=OutputSelector("cloud", "wan"),
            candidates=[OutputSelector("edge", "lan")],
        )

        self.assertFalse(detail_rows[0]["exact_match"])
        self.assertTrue(detail_rows[0]["normalized_exact_match"])
        self.assertEqual(aggregate_rows[0]["normalized_exact_match_rate"], 1.0)

    def test_compare_outputs_marks_missing_candidate(self):
        rows = [
            {
                "placement": "cloud",
                "network_profile": "wan",
                "concurrency_level": "1",
                "prompt_id": "0",
                "run_index": "0",
                "warmup": "False",
                "output_text": "reference",
            }
        ]

        detail_rows, aggregate_rows = compare_outputs(
            rows,
            reference=OutputSelector("cloud", "wan"),
            candidates=[OutputSelector("edge", "lan")],
        )

        self.assertTrue(detail_rows[0]["candidate_missing"])
        self.assertEqual(aggregate_rows[0]["missing"], 1)
        self.assertEqual(aggregate_rows[0]["mean_similarity"], 0.0)

    def test_read_summary_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_summary.csv"
            write_csv(path, [{"placement": "cloud", "output_text": "x"}])

            rows = read_summary(path)

            self.assertEqual(rows[0]["placement"], "cloud")
            self.assertEqual(rows[0]["output_text"], "x")

    def test_first_diff_index(self):
        self.assertEqual(first_diff_index("abc", "abc"), -1)
        self.assertEqual(first_diff_index("abc", "axc"), 1)
        self.assertEqual(first_diff_index("abc", "abcd"), 3)


if __name__ == "__main__":
    unittest.main()
