from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Iterable

from profiling import write_csv


QUERY_FIELDS = [
    "timestamp",
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
]
QUERY_COLUMNS = [
    "nvidia_timestamp",
    "gpu_index",
    "gpu_name",
    "gpu_util_percent",
    "memory_util_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "temperature_c",
]


def _float_or_none(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_query_csv_line(line: str) -> dict[str, Any]:
    parts = next(csv.reader([line]))
    if len(parts) != len(QUERY_COLUMNS):
        raise ValueError(f"Expected {len(QUERY_COLUMNS)} columns from nvidia-smi, got {len(parts)}: {line!r}")
    row = {column: value.strip() for column, value in zip(QUERY_COLUMNS, parts)}
    row["gpu_index"] = int(row["gpu_index"])
    for key in (
        "gpu_util_percent",
        "memory_util_percent",
        "memory_used_mib",
        "memory_total_mib",
        "power_draw_w",
        "temperature_c",
    ):
        row[key] = _float_or_none(str(row[key]))
    return row


def query_gpu_once(nvidia_smi: str = "nvidia-smi") -> list[dict[str, Any]]:
    command = [
        nvidia_smi,
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        if line.strip():
            rows.append(parse_query_csv_line(line))
    return rows


class CsvAppender:
    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
        self.writer.writeheader()
        self.lock = threading.Lock()

    def write_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        with self.lock:
            for row in rows:
                self.writer.writerow(row)
            self.handle.flush()

    def close(self) -> None:
        with self.lock:
            self.handle.close()


def sample_gpu_metrics(
    output_csv: str | Path,
    stop_event: threading.Event,
    interval_ms: int = 1000,
    nvidia_smi: str = "nvidia-smi",
) -> None:
    started = time.perf_counter()
    fieldnames = ["sample_index", "elapsed_ms", *QUERY_COLUMNS, "error"]
    writer = CsvAppender(output_csv, fieldnames)
    sample_index = 0
    try:
        while not stop_event.is_set():
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            try:
                rows = [
                    {
                        "sample_index": sample_index,
                        "elapsed_ms": elapsed_ms,
                        **row,
                    }
                    for row in query_gpu_once(nvidia_smi=nvidia_smi)
                ]
                writer.write_rows(rows)
            except Exception as exc:
                writer.write_rows(
                    [
                        {
                            "sample_index": sample_index,
                            "elapsed_ms": elapsed_ms,
                            "nvidia_timestamp": "",
                            "gpu_index": -1,
                            "gpu_name": "ERROR",
                            "gpu_util_percent": "",
                            "memory_util_percent": "",
                            "memory_used_mib": "",
                            "memory_total_mib": "",
                            "power_draw_w": "",
                            "temperature_c": "",
                            "error": str(exc),
                        }
                    ]
                )
            sample_index += 1
            stop_event.wait(interval_ms / 1000.0)
    finally:
        writer.close()


def start_dmon_capture(
    output_log: str | Path,
    interval_s: int = 1,
    nvidia_smi: str = "nvidia-smi",
) -> subprocess.Popen | None:
    path = Path(output_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    command = [nvidia_smi, "dmon", "-s", "pucm", "-d", str(interval_s), "-o", "DT"]
    try:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        handle.write(f"nvidia-smi not found: {nvidia_smi}\n")
        handle.close()
        return None
    process._gpu_monitor_log_handle = handle  # type: ignore[attr-defined]
    return process


def stop_process(process: subprocess.Popen | None, timeout_s: float = 5.0) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout_s)
    handle = getattr(process, "_gpu_monitor_log_handle", None)
    if handle is not None:
        handle.close()


def run_with_monitor(args: argparse.Namespace) -> int:
    if not args.command:
        raise ValueError("Pass the benchmark command after --.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "gpu_metrics.csv"
    dmon_log = output_dir / "gpu_dmon_raw.log"
    metadata_path = output_dir / "gpu_monitor_meta.json"

    metadata = {
        "command": args.command,
        "sample_interval_ms": args.sample_interval_ms,
        "dmon_interval_s": args.dmon_interval_s,
        "nvidia_smi": args.nvidia_smi,
        "metrics_csv": str(metrics_csv),
        "dmon_log": str(dmon_log),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stop_event = threading.Event()
    sampler = threading.Thread(
        target=sample_gpu_metrics,
        kwargs={
            "output_csv": metrics_csv,
            "stop_event": stop_event,
            "interval_ms": args.sample_interval_ms,
            "nvidia_smi": args.nvidia_smi,
        },
        daemon=True,
    )
    dmon_process = start_dmon_capture(dmon_log, interval_s=args.dmon_interval_s, nvidia_smi=args.nvidia_smi)
    sampler.start()
    try:
        process = subprocess.Popen(args.command)
        return_code = process.wait()
    finally:
        stop_event.set()
        sampler.join(timeout=max(args.sample_interval_ms / 1000.0 + 1.0, 2.0))
        stop_process(dmon_process)

    summarize_gpu_metrics(metrics_csv, output_dir / "gpu_metrics_summary.csv")
    return return_code


def read_metric_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numeric(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_gpu_metrics(input_csv: str | Path, output_csv: str | Path) -> list[dict[str, Any]]:
    rows = [row for row in read_metric_rows(input_csv) if row.get("gpu_index") not in {None, "", "-1"}]
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["gpu_index"], []).append(row)

    summaries: list[dict[str, Any]] = []
    for gpu_index, gpu_rows in sorted(groups.items(), key=lambda item: int(item[0])):
        summary: dict[str, Any] = {
            "gpu_index": int(gpu_index),
            "gpu_name": gpu_rows[0].get("gpu_name", ""),
            "samples": len(gpu_rows),
        }
        for key in (
            "gpu_util_percent",
            "memory_util_percent",
            "memory_used_mib",
            "power_draw_w",
            "temperature_c",
        ):
            values = [_numeric(row, key) for row in gpu_rows]
            numeric_values = [value for value in values if value is not None]
            summary[f"{key}_mean"] = _mean(numeric_values)
            summary[f"{key}_max"] = max(numeric_values) if numeric_values else 0.0
        summaries.append(summary)

    write_csv(output_csv, summaries)
    return summaries


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command while collecting lightweight NVIDIA GPU metrics.")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark while collecting GPU metrics.")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--sample-interval-ms", type=int, default=1000)
    run_parser.add_argument("--dmon-interval-s", type=int, default=1)
    run_parser.add_argument("--nvidia-smi", default="nvidia-smi")
    run_parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --.")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize an existing gpu_metrics.csv.")
    summarize_parser.add_argument("--metrics-csv", required=True)
    summarize_parser.add_argument("--output-csv", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command_name == "run":
        command = args.command
        if command and command[0] == "--":
            command = command[1:]
        args.command = command
        raise SystemExit(run_with_monitor(args))
    if args.command_name == "summarize":
        summarize_gpu_metrics(args.metrics_csv, args.output_csv)
        print(f"Wrote GPU metric summary: {args.output_csv}")
        return
    raise ValueError(f"Unsupported command: {args.command_name}")


if __name__ == "__main__":
    main()
