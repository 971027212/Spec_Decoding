from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from typing import Any, Iterable

from profiling import write_csv


DEFAULT_GPU_SAMPLE_PLACEMENTS = [
    "cloud_a100_vllm_bf16",
    "edge_3090x8_vllm_tp8_bf16",
    "edge_3090x8_vllm_tp4pp2_bf16",
    "edge_3090x8_sglang_tp8_bf16",
]
DEFAULT_NSIGHT_PLACEMENTS = [
    "edge_3090x8_vllm_tp8_bf16",
    "edge_3090x8_vllm_tp4pp2_bf16",
]


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def posix_path(value: str | Path) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value).replace("\\", "/")


def load_plan(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def placements_by_name(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(placement["name"]): placement for placement in plan.get("placements", [])}


def representative_network(placement: dict[str, Any]) -> str:
    networks = [str(network) for network in placement.get("network_profiles", [])]
    if not networks:
        raise ValueError(f"Placement {placement.get('name')} does not list any network profiles.")
    preferred = "cloud_wan" if placement.get("target_location") == "cloud" else "edge_lan"
    return preferred if preferred in networks else networks[0]


def command_block(command: list[str]) -> str:
    if len(command) <= 3:
        return " ".join(q(item) for item in command)
    lines = [q(command[0])]
    for item in command[1:]:
        lines[-1] += " \\"
        lines.append(f"  {q(item)}")
    return "\n".join(lines)


def benchmark_command(
    plan_path: str | Path,
    output_dir: str | Path,
    placement: str | None = None,
    network: str | None = None,
    concurrency_level: int | None = None,
    save_text: bool = True,
) -> list[str]:
    command = [
        "python",
        "target_placement_benchmark.py",
        "--plan",
        str(plan_path),
        "--output-dir",
        posix_path(output_dir),
    ]
    if placement:
        command.extend(["--placement", placement])
    if network:
        command.extend(["--network", network])
    if concurrency_level is not None:
        command.extend(["--concurrency-level", str(concurrency_level)])
    if save_text:
        command.append("--save-text")
    return command


def gpu_monitor_command(
    plan_path: str | Path,
    output_root: str | Path,
    placement: dict[str, Any],
    sample_interval_ms: int = 1000,
    dmon_interval_s: int = 1,
) -> list[str]:
    placement_name = str(placement["name"])
    network = representative_network(placement)
    return [
        "python",
        "gpu_monitor.py",
        "run",
        "--output-dir",
        posix_path(Path(output_root) / "gpu" / placement_name),
        "--sample-interval-ms",
        str(sample_interval_ms),
        "--dmon-interval-s",
        str(dmon_interval_s),
        "--",
        *benchmark_command(
            plan_path=plan_path,
            output_dir=Path(output_root) / "client_with_gpu" / placement_name,
            placement=placement_name,
            network=network,
            save_text=True,
        ),
    ]


def nsight_server_command(output_root: str | Path, placement: dict[str, Any], port: int = 8000) -> list[str]:
    deployment_method = str(placement.get("deployment_method", ""))
    if not deployment_method.startswith("vllm"):
        raise ValueError(f"Nsight server command generation currently supports vLLM placements only: {placement['name']}")

    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt,cublas,nccl",
        "--force-overwrite=true",
        "-o",
        posix_path(Path(output_root) / "nsight" / str(placement["name"])),
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(placement["model"]),
        "--served-model-name",
        str(placement["model"]),
        "--dtype",
        "bfloat16" if placement.get("precision") == "bf16" else str(placement.get("precision", "auto")),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    tensor_parallel_size = int(placement.get("tensor_parallel_size") or 1)
    pipeline_parallel_size = int(placement.get("pipeline_parallel_size") or 1)
    if tensor_parallel_size > 1:
        command.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if pipeline_parallel_size > 1:
        command.extend(["--pipeline-parallel-size", str(pipeline_parallel_size)])
    return command


def nsight_client_command(
    plan_path: str | Path,
    output_root: str | Path,
    placement: dict[str, Any],
    concurrency_level: int = 1,
) -> list[str]:
    placement_name = str(placement["name"])
    return benchmark_command(
        plan_path=plan_path,
        output_dir=Path(output_root) / "nsight_client" / placement_name,
        placement=placement_name,
        network=representative_network(placement),
        concurrency_level=concurrency_level,
        save_text=True,
    )


def write_script(path: str | Path, commands: list[list[str]], header: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"# {header}", ""]
    for index, command in enumerate(commands, start=1):
        body.append(f"# {index}")
        body.append(command_block(command))
        body.append("")
    path.write_text("\n".join(body), encoding="utf-8")


def build_matrix_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for placement in plan.get("placements", []):
        for network in placement.get("network_profiles", []):
            for concurrency in plan.get("concurrency_levels", [1]):
                rows.append(
                    {
                        "coverage": "client_visible",
                        "placement": placement["name"],
                        "target_location": placement.get("target_location"),
                        "deployment_method": placement.get("deployment_method"),
                        "network_profile": network,
                        "concurrency_level": concurrency,
                    }
                )
    return rows


def write_run_matrix(
    plan_path: str | Path,
    output_root: str | Path,
    runbook_dir: str | Path,
    gpu_sample_placements: list[str] | None = None,
    nsight_placements: list[str] | None = None,
) -> dict[str, Path]:
    plan = load_plan(plan_path)
    placements = placements_by_name(plan)
    gpu_sample_placements = gpu_sample_placements or DEFAULT_GPU_SAMPLE_PLACEMENTS
    nsight_placements = nsight_placements or DEFAULT_NSIGHT_PLACEMENTS

    missing = [name for name in [*gpu_sample_placements, *nsight_placements] if name not in placements]
    if missing:
        raise ValueError(f"Unknown placements in run matrix: {', '.join(sorted(set(missing)))}")

    runbook_root = Path(runbook_dir)
    output_root = Path(output_root)
    paths = {
        "client_script": runbook_root / "01_client_visible_all.sh",
        "gpu_script": runbook_root / "02_gpu_sampling_representative.sh",
        "nsight_server_script": runbook_root / "03_nsight_server_key_methods.sh",
        "nsight_client_script": runbook_root / "04_nsight_client_key_methods.sh",
        "matrix_csv": runbook_root / "first_round_matrix.csv",
    }

    write_script(
        paths["client_script"],
        [
            benchmark_command(
                plan_path=plan_path,
                output_dir=output_root / "client_visible_all",
                save_text=True,
            )
        ],
        "Full client-visible matrix: all placements, networks, and concurrency levels.",
    )
    write_script(
        paths["gpu_script"],
        [
            gpu_monitor_command(
                plan_path=plan_path,
                output_root=output_root,
                placement=placements[name],
            )
            for name in gpu_sample_placements
        ],
        "Representative GPU sampling runs.",
    )
    write_script(
        paths["nsight_server_script"],
        [nsight_server_command(output_root=output_root, placement=placements[name]) for name in nsight_placements],
        "Run one server command at a time, then run the matching client command in another terminal.",
    )
    write_script(
        paths["nsight_client_script"],
        [
            nsight_client_command(
                plan_path=plan_path,
                output_root=output_root,
                placement=placements[name],
                concurrency_level=1,
            )
            for name in nsight_placements
        ],
        "Client commands for Nsight server runs.",
    )
    write_csv(paths["matrix_csv"], build_matrix_rows(plan))
    return paths


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate first-round target placement runbook scripts.")
    parser.add_argument("--plan", default="configs/target_placement_qwen32b_bf16.example.json")
    parser.add_argument("--output-root", default="experiments/target_placement/qwen32b_bf16")
    parser.add_argument("--runbook-dir", default="experiments/target_placement/qwen32b_bf16_runbook")
    parser.add_argument("--gpu-placement", action="append", default=None)
    parser.add_argument("--nsight-placement", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    paths = write_run_matrix(
        plan_path=args.plan,
        output_root=args.output_root,
        runbook_dir=args.runbook_dir,
        gpu_sample_placements=args.gpu_placement,
        nsight_placements=args.nsight_placement,
    )
    print("Wrote first-round run matrix:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
