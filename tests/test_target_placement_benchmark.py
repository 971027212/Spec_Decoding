import csv
import tempfile
import unittest
from pathlib import Path

from target_placement_benchmark import build_decision_rows, benchmark_plan, selected_runs


def minimal_plan():
    return {
        "name": "test_target_placement",
        "max_tokens": 4,
        "warmup_runs": 0,
        "runs": 1,
        "network_profiles": [
            {"name": "cloud_wan", "simulate": True, "rtt_ms": 80, "uplink_mbps": 100, "downlink_mbps": 200},
            {"name": "edge_lan", "simulate": True, "rtt_ms": 2, "uplink_mbps": 1000, "downlink_mbps": 1000},
        ],
        "placements": [
            {
                "name": "cloud_a100_vllm",
                "target_location": "cloud",
                "base_url": "http://cloud.example:8000/v1",
                "model": "qwen14b",
                "network_profiles": ["cloud_wan"],
            },
            {
                "name": "edge_3090_vllm",
                "target_location": "edge",
                "base_url": "http://edge.example:8000/v1",
                "model": "qwen14b",
                "network_profiles": ["edge_lan"],
            },
        ],
        "comparisons": [
            {
                "name": "edge_lan_vs_cloud_wan",
                "mode": "target_generate",
                "cloud": {"placement": "cloud_a100_vllm", "network": "cloud_wan"},
                "edge": {"placement": "edge_3090_vllm", "network": "edge_lan"},
            }
        ],
        "prompts": ["hello"],
    }


class TargetPlacementBenchmarkTests(unittest.TestCase):
    def test_selected_runs_expands_placement_network_pairs(self):
        runs = selected_runs(minimal_plan())

        self.assertEqual([(placement.name, profile.name) for placement, profile in runs], [
            ("cloud_a100_vllm", "cloud_wan"),
            ("edge_3090_vllm", "edge_lan"),
        ])

    def test_build_decision_rows_reports_observed_edge_advantage(self):
        rows = build_decision_rows(
            minimal_plan(),
            [
                {
                    "mode": "target_generate",
                    "placement": "cloud_a100_vllm",
                    "network_profile": "cloud_wan",
                    "generation_total_ms_mean": 100.0,
                    "ttft_ms_mean": 30.0,
                    "target_upload_ms_mean": 10.0,
                    "target_downlink_ms_mean": 10.0,
                },
                {
                    "mode": "target_generate",
                    "placement": "edge_3090_vllm",
                    "network_profile": "edge_lan",
                    "generation_total_ms_mean": 80.0,
                    "ttft_ms_mean": 20.0,
                    "target_upload_ms_mean": 1.0,
                    "target_downlink_ms_mean": 1.0,
                },
            ],
        )

        self.assertEqual(rows[0]["status"], "ok")
        self.assertTrue(rows[0]["edge_worth_it_observed"])
        self.assertAlmostEqual(rows[0]["edge_latency_advantage_ms"], 20.0)
        self.assertAlmostEqual(rows[0]["network_path_advantage_ms"], 18.0)

    def test_fake_benchmark_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = benchmark_plan(minimal_plan(), tmp, fake=True)

            for path in paths.values():
                self.assertTrue(Path(path).exists())

            with Path(paths["placement_decisions"]).open(newline="", encoding="utf-8") as handle:
                decisions = list(csv.DictReader(handle))

            self.assertEqual(decisions[0]["comparison"], "edge_lan_vs_cloud_wan")
            self.assertEqual(decisions[0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
