import tempfile
import unittest
from pathlib import Path

from profiling import TimingRecorder, aggregate_summaries, write_csv, write_jsonl


class ProfilingTests(unittest.TestCase):
    def test_summary_contains_phase_totals_and_metrics(self):
        recorder = TimingRecorder(mode="speculative", prompt_id=0, run_index=0)
        recorder.record("target_upload", 1_000_000)
        recorder.record("target_upload", 2_000_000)
        recorder.set_metric("generated_tokens", 4)

        summary = recorder.summary()

        self.assertEqual(summary["generated_tokens"], 4)
        self.assertEqual(summary["target_upload_ns"], 3_000_000)
        self.assertAlmostEqual(summary["target_upload_ms"], 3.0)

    def test_aggregate_summaries_skips_warmups(self):
        measured = TimingRecorder(mode="target_ar", run_index=0)
        measured.record("generation_total", 10_000_000)
        warmup = TimingRecorder(mode="target_ar", run_index=0, warmup=True)
        warmup.record("generation_total", 99_000_000)

        rows = aggregate_summaries([measured.summary(), warmup.summary()])

        self.assertEqual(rows[0]["samples"], 1)
        self.assertAlmostEqual(rows[0]["generation_total_ms_mean"], 10.0)

    def test_writers_create_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "events.jsonl", [{"phase": "x"}])
            write_csv(root / "summary.csv", [{"phase": "x", "duration_ns": 1}])

            self.assertTrue((root / "events.jsonl").exists())
            self.assertTrue((root / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
