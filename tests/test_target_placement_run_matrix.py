import tempfile
import unittest
from pathlib import Path

from target_placement_run_matrix import (
    benchmark_command,
    command_block,
    posix_path,
    representative_network,
    write_run_matrix,
)


class TargetPlacementRunMatrixTests(unittest.TestCase):
    def test_representative_network_prefers_cloud_wan_for_cloud(self):
        placement = {"target_location": "cloud", "network_profiles": ["cloud_congested", "cloud_wan"]}

        self.assertEqual(representative_network(placement), "cloud_wan")

    def test_representative_network_prefers_edge_lan_for_edge(self):
        placement = {"target_location": "edge", "network_profiles": ["metro_edge", "edge_lan"]}

        self.assertEqual(representative_network(placement), "edge_lan")

    def test_benchmark_command_can_target_single_concurrency(self):
        command = benchmark_command(
            plan_path="plan.json",
            output_dir="out",
            placement="edge",
            network="edge_lan",
            concurrency_level=1,
        )

        self.assertIn("--concurrency-level", command)
        self.assertIn("1", command)
        self.assertIn("--save-text", command)

    def test_command_block_uses_line_continuations(self):
        block = command_block(["python", "x.py", "--arg", "value"])

        self.assertIn("\\\n", block)

    def test_posix_path_normalizes_windows_separators(self):
        self.assertEqual(posix_path("experiments\\target_placement\\x"), "experiments/target_placement/x")

    def test_write_run_matrix_creates_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "plan.json"
            plan.write_text(
                """
{
  "concurrency_levels": [1, 2],
  "placements": [
    {
      "name": "cloud_a100_vllm_bf16",
      "target_location": "cloud",
      "deployment_method": "vllm_single_gpu",
      "precision": "bf16",
      "model": "/models/qwen",
      "network_profiles": ["cloud_wan"],
      "tensor_parallel_size": 1,
      "pipeline_parallel_size": 1
    },
    {
      "name": "edge_3090x8_vllm_tp8_bf16",
      "target_location": "edge",
      "deployment_method": "vllm_tp8",
      "precision": "bf16",
      "model": "/models/qwen",
      "network_profiles": ["edge_lan"],
      "tensor_parallel_size": 8,
      "pipeline_parallel_size": 1
    },
    {
      "name": "edge_3090x8_vllm_tp4pp2_bf16",
      "target_location": "edge",
      "deployment_method": "vllm_tp4_pp2",
      "precision": "bf16",
      "model": "/models/qwen",
      "network_profiles": ["edge_lan"],
      "tensor_parallel_size": 4,
      "pipeline_parallel_size": 2
    },
    {
      "name": "edge_3090x8_sglang_tp8_bf16",
      "target_location": "edge",
      "deployment_method": "sglang_tp8",
      "precision": "bf16",
      "model": "/models/qwen",
      "network_profiles": ["edge_lan"],
      "tensor_parallel_size": 8,
      "pipeline_parallel_size": 1
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )

            paths = write_run_matrix(plan, root / "out", root / "runbook")

            for path in paths.values():
                self.assertTrue(path.exists())

            nsight_server = paths["nsight_server_script"].read_text(encoding="utf-8")
            self.assertIn("--tensor-parallel-size", nsight_server)
            self.assertIn("--pipeline-parallel-size", nsight_server)


if __name__ == "__main__":
    unittest.main()
