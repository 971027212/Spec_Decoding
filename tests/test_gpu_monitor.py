import tempfile
import unittest
from pathlib import Path

from gpu_monitor import parse_query_csv_line, summarize_gpu_metrics
from profiling import write_csv


class GpuMonitorTests(unittest.TestCase):
    def test_parse_query_csv_line(self):
        row = parse_query_csv_line("2026/05/15 10:00:00.000, 0, NVIDIA GeForce RTX 3090, 87, 42, 12000, 24576, 310.5, 69")

        self.assertEqual(row["gpu_index"], 0)
        self.assertEqual(row["gpu_name"], "NVIDIA GeForce RTX 3090")
        self.assertEqual(row["gpu_util_percent"], 87.0)
        self.assertEqual(row["memory_used_mib"], 12000.0)
        self.assertEqual(row["power_draw_w"], 310.5)

    def test_parse_query_csv_line_handles_na(self):
        row = parse_query_csv_line("2026/05/15 10:00:00.000, 1, GPU, N/A, 0, 1, 2, [N/A], 40")

        self.assertIsNone(row["gpu_util_percent"])
        self.assertIsNone(row["power_draw_w"])
        self.assertEqual(row["temperature_c"], 40.0)

    def test_summarize_gpu_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "gpu_metrics.csv"
            summary = root / "gpu_metrics_summary.csv"
            write_csv(
                metrics,
                [
                    {
                        "sample_index": 0,
                        "elapsed_ms": 0,
                        "gpu_index": 0,
                        "gpu_name": "GPU0",
                        "gpu_util_percent": 10,
                        "memory_util_percent": 20,
                        "memory_used_mib": 100,
                        "power_draw_w": 200,
                        "temperature_c": 50,
                    },
                    {
                        "sample_index": 1,
                        "elapsed_ms": 1000,
                        "gpu_index": 0,
                        "gpu_name": "GPU0",
                        "gpu_util_percent": 30,
                        "memory_util_percent": 40,
                        "memory_used_mib": 300,
                        "power_draw_w": 240,
                        "temperature_c": 60,
                    },
                ],
            )

            rows = summarize_gpu_metrics(metrics, summary)

            self.assertTrue(summary.exists())
            self.assertEqual(rows[0]["samples"], 2)
            self.assertEqual(rows[0]["gpu_util_percent_mean"], 20.0)
            self.assertEqual(rows[0]["memory_used_mib_max"], 300.0)


if __name__ == "__main__":
    unittest.main()
