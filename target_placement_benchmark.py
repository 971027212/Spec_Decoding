from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
import http.client

from profiling import TimingRecorder, aggregate_summaries, now_ns, write_csv, write_jsonl
from remote_target import NetworkSimulation


DEFAULT_PROMPTS = [
    "Explain why target-model placement matters for low-latency LLM serving.",
    "Write a concise checklist for deploying a quantized model on an edge GPU.",
    "Summarize the trade-off between cloud queueing delay and slower edge compute.",
]
MODE = "target_generate"


@dataclass(frozen=True)
class NetworkProfile:
    name: str
    simulate: bool = False
    rtt_ms: float = 0.0
    uplink_mbps: float = 0.0
    downlink_mbps: float = 0.0

    def simulation(self) -> NetworkSimulation:
        return NetworkSimulation(
            enabled=self.simulate,
            rtt_ms=self.rtt_ms,
            uplink_mbps=self.uplink_mbps,
            downlink_mbps=self.downlink_mbps,
        )


@dataclass(frozen=True)
class Placement:
    name: str
    target_location: str
    base_url: str
    model: str
    protocol: str = "openai_completions"
    deployment_method: str = "unknown"
    hardware: str = "unknown"
    precision: str = "unknown"
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    api_key_env: str | None = None
    network_profiles: tuple[str, ...] = ()
    extra_headers: dict[str, str] | None = None


def _sleep_ns(duration_ns: int) -> None:
    if duration_ns > 0:
        time.sleep(duration_ns / 1_000_000_000)


def _bandwidth_delay_ns(num_bytes: int, mbps: float) -> int:
    if mbps <= 0:
        return 0
    return int((num_bytes * 8 * 1_000_000_000) / (mbps * 1_000_000))


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def load_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not plan.get("placements"):
        raise ValueError(f"{plan_path} must contain at least one placement.")
    if not plan.get("network_profiles"):
        raise ValueError(f"{plan_path} must contain at least one network profile.")
    return plan


def write_json_array(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_network_profiles(plan: dict[str, Any]) -> dict[str, NetworkProfile]:
    profiles: dict[str, NetworkProfile] = {}
    for raw in plan["network_profiles"]:
        profile = NetworkProfile(
            name=str(raw["name"]),
            simulate=bool(raw.get("simulate", False)),
            rtt_ms=float(raw.get("rtt_ms", 0.0)),
            uplink_mbps=float(raw.get("uplink_mbps", 0.0)),
            downlink_mbps=float(raw.get("downlink_mbps", 0.0)),
        )
        profiles[profile.name] = profile
    return profiles


def parse_placements(plan: dict[str, Any]) -> list[Placement]:
    placements = []
    for raw in plan["placements"]:
        protocol = str(raw.get("protocol", "openai_completions"))
        if protocol != "openai_completions":
            raise ValueError(f"Unsupported placement protocol: {protocol}")
        placements.append(
            Placement(
                name=str(raw["name"]),
                target_location=str(raw.get("target_location", "unknown")),
                base_url=str(raw["base_url"]),
                model=str(raw["model"]),
                protocol=protocol,
                deployment_method=str(raw.get("deployment_method", "unknown")),
                hardware=str(raw.get("hardware", "unknown")),
                precision=str(raw.get("precision", "unknown")),
                tensor_parallel_size=(
                    int(raw["tensor_parallel_size"])
                    if raw.get("tensor_parallel_size") is not None
                    else None
                ),
                pipeline_parallel_size=(
                    int(raw["pipeline_parallel_size"])
                    if raw.get("pipeline_parallel_size") is not None
                    else None
                ),
                api_key_env=raw.get("api_key_env"),
                network_profiles=_as_tuple(raw.get("network_profiles")),
                extra_headers={str(key): str(value) for key, value in raw.get("extra_headers", {}).items()},
            )
        )
    return placements


def load_prompts(plan: dict[str, Any]) -> list[str]:
    if plan.get("prompts_file"):
        path = Path(plan["prompts_file"])
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        prompts = [str(item) for item in plan.get("prompts", DEFAULT_PROMPTS)]
    if not prompts:
        raise ValueError("The benchmark needs at least one prompt.")
    return prompts


def selected_runs(
    plan: dict[str, Any],
    placement_filter: str | None = None,
    network_filter: str | None = None,
) -> list[tuple[Placement, NetworkProfile]]:
    profiles = parse_network_profiles(plan)
    placements = parse_placements(plan)
    runs: list[tuple[Placement, NetworkProfile]] = []
    for placement in placements:
        if placement_filter and placement.name != placement_filter:
            continue
        profile_names = placement.network_profiles or tuple(profiles)
        for profile_name in profile_names:
            if network_filter and profile_name != network_filter:
                continue
            if profile_name not in profiles:
                raise ValueError(f"Placement {placement.name} references unknown network profile {profile_name}.")
            runs.append((placement, profiles[profile_name]))
    return runs


def concurrency_levels(plan: dict[str, Any]) -> list[int]:
    values = plan.get("concurrency_levels", [1])
    if isinstance(values, int):
        values = [values]
    levels = [int(value) for value in values]
    if not levels or any(level < 1 for level in levels):
        raise ValueError("concurrency_levels must contain positive integers.")
    return levels


def _connection(base_url: str) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"base_url must start with http:// or https://: {base_url}")
    if not parsed.netloc:
        raise ValueError(f"base_url must include a host: {base_url}")
    connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/completions"):
        path = base_path
    elif base_path.endswith("/v1"):
        path = f"{base_path}/completions"
    else:
        path = f"{base_path}/v1/completions" if base_path else "/v1/completions"
    return connection_cls(parsed.netloc), path


def _headers(placement: Placement) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if placement.api_key_env:
        token = os.environ.get(placement.api_key_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    if placement.extra_headers:
        headers.update(placement.extra_headers)
    return headers


def _completion_piece(event: dict[str, Any]) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return str(choice.get("text") or "")
    delta = choice.get("delta") or {}
    return str(delta.get("content") or "")


def run_openai_completion(
    placement: Placement,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    network: NetworkSimulation,
    save_text: bool,
) -> dict[str, Any]:
    payload = {
        "model": placement.model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    encode_start = now_ns()
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encode_ns = now_ns() - encode_start

    connection, path = _connection(placement.base_url)
    connection.timeout = timeout

    request_headers = _headers(placement)
    raw_bytes = 0
    response_decode_ns = 0
    generated_chunks = 0
    completion_tokens: int | None = None
    output_parts: list[str] = []
    first_content_ns: int | None = None
    last_content_ns: int | None = None
    simulated_upload_ns = network.uplink_delay_ns(len(body))
    simulated_downlink_ns = 0

    total_start = now_ns()
    try:
        upload_start = now_ns()
        _sleep_ns(simulated_upload_ns)
        connection.putrequest("POST", path)
        for key, value in request_headers.items():
            connection.putheader(key, value)
        connection.putheader("Content-Length", str(len(body)))
        connection.endheaders(body)
        upload_ns = now_ns() - upload_start

        wait_start = now_ns()
        response = connection.getresponse()
        response_wait_ns = now_ns() - wait_start
        status = response.status

        if status >= 400:
            error_body = response.read()
            raise RuntimeError(error_body.decode("utf-8", errors="replace"))

        downlink_start = now_ns()
        applied_downlink_latency = False
        while True:
            line = response.readline()
            if not line:
                break
            raw_bytes += len(line)
            if network.enabled and not applied_downlink_latency:
                _sleep_ns(network.one_way_latency_ns)
                simulated_downlink_ns += network.one_way_latency_ns
                applied_downlink_latency = True
            bandwidth_delay_ns = _bandwidth_delay_ns(len(line), network.downlink_mbps) if network.enabled else 0
            _sleep_ns(bandwidth_delay_ns)
            simulated_downlink_ns += bandwidth_delay_ns

            stripped = line.strip()
            if not stripped.startswith(b"data:"):
                continue
            data = stripped[5:].strip()
            if data == b"[DONE]":
                break

            decode_start = now_ns()
            event = json.loads(data.decode("utf-8"))
            response_decode_ns += now_ns() - decode_start

            usage = event.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])

            piece = _completion_piece(event)
            if piece:
                now = now_ns()
                if first_content_ns is None:
                    first_content_ns = now
                last_content_ns = now
                generated_chunks += 1
                if save_text:
                    output_parts.append(piece)
        downlink_ns = now_ns() - downlink_start
    finally:
        connection.close()

    generation_total_ns = now_ns() - total_start
    ttft_ns = first_content_ns - total_start if first_content_ns is not None else None
    if first_content_ns is not None and last_content_ns is not None and generated_chunks > 1:
        itl_ns = (last_content_ns - first_content_ns) / (generated_chunks - 1)
    else:
        itl_ns = None
    generated_units = completion_tokens if completion_tokens is not None else generated_chunks

    result = {
        "target_request_encode_ns": encode_ns,
        "target_upload_ns": upload_ns,
        "target_response_wait_ns": response_wait_ns,
        "target_downlink_ns": downlink_ns,
        "target_response_decode_ns": response_decode_ns,
        "generation_total_ns": generation_total_ns,
        "status": status,
        "request_bytes": len(body),
        "response_bytes": raw_bytes,
        "simulated_upload_ns": simulated_upload_ns,
        "simulated_downlink_ns": simulated_downlink_ns,
        "generated_chunks": generated_chunks,
        "generated_tokens": generated_units,
        "completion_tokens": completion_tokens,
        "characters_generated": sum(len(part) for part in output_parts) if save_text else None,
        "ttft_ms": ttft_ns / 1_000_000 if ttft_ns is not None else None,
        "itl_ms": itl_ns / 1_000_000 if itl_ns is not None else None,
        "throughput_tokens_s": generated_units / (generation_total_ns / 1_000_000_000) if generation_total_ns else None,
    }
    if save_text:
        result["output_text"] = "".join(output_parts)
    return result


def run_fake_completion(
    placement: Placement,
    profile: NetworkProfile,
    prompt: str,
    max_tokens: int,
    run_index: int,
    save_text: bool,
) -> dict[str, Any]:
    prompt_bytes = len(prompt.encode("utf-8"))
    response_bytes = max_tokens * 6
    network = profile.simulation()
    upload_ns = network.uplink_delay_ns(prompt_bytes)
    downlink_ns = network.downlink_delay_ns(response_bytes)
    cloud_like = placement.target_location == "cloud"
    per_token_ms = 45.0 if cloud_like else 80.0
    queue_ms = 70.0 if cloud_like and "congested" in profile.name else (12.0 if cloud_like else 4.0)
    compute_ns = int((per_token_ms * max_tokens + queue_ms + run_index) * 1_000_000)
    total_ns = upload_ns + downlink_ns + compute_ns
    ttft_ms = (upload_ns + downlink_ns / max(max_tokens, 1) + int((per_token_ms + queue_ms) * 1_000_000)) / 1_000_000
    result = {
        "target_request_encode_ns": 50_000,
        "target_upload_ns": upload_ns,
        "target_response_wait_ns": compute_ns,
        "target_downlink_ns": downlink_ns,
        "target_response_decode_ns": 50_000,
        "generation_total_ns": total_ns,
        "status": 200,
        "request_bytes": prompt_bytes,
        "response_bytes": response_bytes,
        "simulated_upload_ns": upload_ns,
        "simulated_downlink_ns": downlink_ns,
        "generated_chunks": max_tokens,
        "generated_tokens": max_tokens,
        "completion_tokens": max_tokens,
        "characters_generated": response_bytes,
        "ttft_ms": ttft_ms,
        "itl_ms": per_token_ms,
        "throughput_tokens_s": max_tokens / (total_ns / 1_000_000_000) if total_ns else None,
    }
    if save_text:
        result["output_text"] = "x" * response_bytes
    return result


def record_completion_result(recorder: TimingRecorder, result: dict[str, Any], metadata: dict[str, Any]) -> None:
    phase_keys = {
        "target_request_encode": "target_request_encode_ns",
        "target_upload": "target_upload_ns",
        "target_response_wait": "target_response_wait_ns",
        "target_downlink": "target_downlink_ns",
        "target_response_decode": "target_response_decode_ns",
        "generation_total": "generation_total_ns",
    }
    for phase, key in phase_keys.items():
        recorder.record(phase, int(result.get(key, 0) or 0), **metadata)
    for key, value in result.items():
        if key.endswith("_ns"):
            continue
        if value is not None:
            recorder.set_metric(key, value)


def run_request_batch(
    placement: Placement,
    profile: NetworkProfile,
    prompt: str,
    prompt_id: int,
    iteration: int,
    run_index: int,
    warmup: bool,
    concurrency_level: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    plan_name: str,
    fake: bool,
    save_text: bool,
) -> list[TimingRecorder]:
    network = profile.simulation()
    batch_start = now_ns()

    def run_worker(worker_index: int) -> tuple[int, dict[str, Any]]:
        if fake:
            return worker_index, run_fake_completion(
                placement,
                profile,
                prompt,
                max_tokens,
                iteration + worker_index,
                save_text,
            )
        return worker_index, run_openai_completion(
            placement=placement,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            network=network,
            save_text=save_text,
        )

    results: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(run_worker, worker_index) for worker_index in range(concurrency_level)]
        for future in as_completed(futures):
            results.append(future.result())
    measured_batch_total_ns = now_ns() - batch_start
    batch_total_ns = (
        max(int(result.get("generation_total_ns") or 0) for _, result in results)
        if fake and results
        else measured_batch_total_ns
    )

    total_generated_tokens = sum(int(result.get("generated_tokens") or 0) for _, result in results)
    batch_throughput_tokens_s = (
        total_generated_tokens / (batch_total_ns / 1_000_000_000)
        if batch_total_ns > 0
        else None
    )

    recorders: list[TimingRecorder] = []
    for worker_index, result in sorted(results, key=lambda item: item[0]):
        extra = {
            "plan": plan_name,
            "mode": MODE,
            "placement": placement.name,
            "target_location": placement.target_location,
            "network_profile": profile.name,
            "base_url": placement.base_url,
            "model": placement.model,
            "protocol": placement.protocol,
            "deployment_method": placement.deployment_method,
            "hardware": placement.hardware,
            "precision": placement.precision,
            "tensor_parallel_size": placement.tensor_parallel_size,
            "pipeline_parallel_size": placement.pipeline_parallel_size,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "concurrency_level": concurrency_level,
            "concurrent_worker": worker_index,
            "batch_run_index": run_index,
            "batch_requests": concurrency_level,
            "batch_total_tokens": total_generated_tokens,
            "batch_total_ms": batch_total_ns / 1_000_000,
            "batch_throughput_tokens_s": batch_throughput_tokens_s,
            **network.metadata(),
        }
        recorder = TimingRecorder(
            mode=MODE,
            prompt_id=prompt_id,
            run_index=run_index,
            warmup=warmup,
            extra=extra,
        )
        record_completion_result(recorder, result, metadata=extra)
        recorder.set_metric("batch_total_tokens", total_generated_tokens)
        recorder.set_metric("batch_total_ms", batch_total_ns / 1_000_000)
        if batch_throughput_tokens_s is not None:
            recorder.set_metric("batch_throughput_tokens_s", batch_throughput_tokens_s)
        recorders.append(recorder)
    return recorders


def benchmark_plan(
    plan: dict[str, Any],
    output_dir: str | Path,
    placement_filter: str | None = None,
    network_filter: str | None = None,
    fake: bool = False,
    save_text: bool = False,
) -> dict[str, Path]:
    prompts = load_prompts(plan)
    runs = selected_runs(plan, placement_filter=placement_filter, network_filter=network_filter)
    levels = concurrency_levels(plan)
    if not runs:
        raise ValueError("No placement/network runs selected.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    max_tokens = int(plan.get("max_tokens", 64))
    warmup_runs = int(plan.get("warmup_runs", 1))
    measured_runs = int(plan.get("runs", 3))
    temperature = float(plan.get("temperature", 0.0))
    timeout = float(plan.get("timeout", 300.0))
    total_runs = warmup_runs + measured_runs
    plan_name = str(plan.get("name", "target_placement"))

    recorders: list[TimingRecorder] = []
    for placement, profile in runs:
        for concurrency_level in levels:
            for prompt_id, prompt in enumerate(prompts):
                for iteration in range(total_runs):
                    warmup = iteration < warmup_runs
                    run_index = iteration - warmup_runs if not warmup else iteration
                    batch_recorders = run_request_batch(
                        placement=placement,
                        profile=profile,
                        prompt=prompt,
                        prompt_id=prompt_id,
                        iteration=iteration,
                        run_index=run_index,
                        warmup=warmup,
                        concurrency_level=concurrency_level,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout,
                        plan_name=plan_name,
                        fake=fake,
                        save_text=save_text,
                    )
                    recorders.extend(batch_recorders)
                    label = "warmup" if warmup else "run"
                    batch_ms = batch_recorders[0].metrics.get("batch_total_ms") if batch_recorders else None
                    batch_tps = batch_recorders[0].metrics.get("batch_throughput_tokens_s") if batch_recorders else None
                    print(
                        f"{placement.name}/{profile.name} c={concurrency_level} prompt={prompt_id} "
                        f"{label}={run_index} batch_ms={float(batch_ms):.3f} "
                        f"batch_tps={float(batch_tps):.3f}"
                    )

    events = [event for recorder in recorders for event in recorder.events]
    summaries = [recorder.summary() for recorder in recorders]
    aggregates = aggregate_summaries(
        summaries,
        group_keys=("mode", "placement", "target_location", "network_profile", "concurrency_level"),
    )
    decisions = build_decision_rows(plan, aggregates)

    paths = {
        "raw_events": output_root / "raw_events.jsonl",
        "run_summary": output_root / "run_summary.csv",
        "aggregate_summary": output_root / "aggregate_summary.csv",
        "placement_decisions": output_root / "placement_decisions.csv",
        "planned_runs": output_root / "planned_runs.json",
    }
    write_jsonl(paths["raw_events"], events)
    write_csv(paths["run_summary"], summaries)
    write_csv(paths["aggregate_summary"], aggregates)
    write_csv(paths["placement_decisions"], decisions)
    write_json_array(
        paths["planned_runs"],
        [
            {
                "placement": placement.name,
                "target_location": placement.target_location,
                "network_profile": profile.name,
                "base_url": placement.base_url,
                "model": placement.model,
                "deployment_method": placement.deployment_method,
                "hardware": placement.hardware,
                "precision": placement.precision,
                "tensor_parallel_size": placement.tensor_parallel_size,
                "pipeline_parallel_size": placement.pipeline_parallel_size,
                "concurrency_levels": levels,
                "simulate_network": profile.simulate,
                "rtt_ms": profile.rtt_ms,
                "uplink_mbps": profile.uplink_mbps,
                "downlink_mbps": profile.downlink_mbps,
            }
            for placement, profile in runs
        ],
    )
    return paths


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def build_decision_rows(plan: dict[str, Any], aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_key = {
        (row.get("placement"), row.get("network_profile"), row.get("mode"), int(row.get("concurrency_level", 1))): row
        for row in aggregates
    }
    decisions: list[dict[str, Any]] = []
    for comparison in plan.get("comparisons", []):
        mode = str(comparison.get("mode", MODE))
        concurrency = int(comparison.get("concurrency_level", comparison.get("concurrency", 1)))
        cloud_ref = comparison["cloud"]
        edge_ref = comparison["edge"]
        cloud_key = (cloud_ref["placement"], cloud_ref["network"], mode, concurrency)
        edge_key = (edge_ref["placement"], edge_ref["network"], mode, concurrency)
        cloud = rows_by_key.get(cloud_key)
        edge = rows_by_key.get(edge_key)
        if cloud is None or edge is None:
            decisions.append(
                {
                    "comparison": comparison["name"],
                    "mode": mode,
                    "concurrency_level": concurrency,
                    "status": "missing_rows",
                    "cloud_key": "/".join(str(item) for item in cloud_key),
                    "edge_key": "/".join(str(item) for item in edge_key),
                }
            )
            continue

        cloud_total = _float(cloud, "generation_total_ms_mean")
        edge_total = _float(edge, "generation_total_ms_mean")
        cloud_network = _float(cloud, "target_upload_ms_mean") + _float(cloud, "target_downlink_ms_mean")
        edge_network = _float(edge, "target_upload_ms_mean") + _float(edge, "target_downlink_ms_mean")
        cloud_ttft = _float(cloud, "ttft_ms_mean")
        edge_ttft = _float(edge, "ttft_ms_mean")
        decisions.append(
            {
                "comparison": comparison["name"],
                "mode": mode,
                "concurrency_level": concurrency,
                "status": "ok",
                "cloud_placement": cloud_ref["placement"],
                "cloud_network": cloud_ref["network"],
                "edge_placement": edge_ref["placement"],
                "edge_network": edge_ref["network"],
                "cloud_generation_total_ms": cloud_total,
                "edge_generation_total_ms": edge_total,
                "edge_latency_advantage_ms": cloud_total - edge_total,
                "edge_worth_it_observed": edge_total < cloud_total,
                "cloud_ttft_ms": cloud_ttft,
                "edge_ttft_ms": edge_ttft,
                "edge_ttft_advantage_ms": cloud_ttft - edge_ttft,
                "cloud_upload_downlink_ms": cloud_network,
                "edge_upload_downlink_ms": edge_network,
                "network_path_advantage_ms": cloud_network - edge_network,
            }
        )
    return decisions


def write_dry_run(
    plan: dict[str, Any],
    output_dir: str | Path,
    placement_filter: str | None = None,
    network_filter: str | None = None,
) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = selected_runs(plan, placement_filter=placement_filter, network_filter=network_filter)
    levels = concurrency_levels(plan)
    path = output_root / "planned_runs.json"
    write_json_array(
        path,
        [
            {
                "placement": placement.name,
                "target_location": placement.target_location,
                "network_profile": profile.name,
                "base_url": placement.base_url,
                "model": placement.model,
                "protocol": placement.protocol,
                "deployment_method": placement.deployment_method,
                "hardware": placement.hardware,
                "precision": placement.precision,
                "tensor_parallel_size": placement.tensor_parallel_size,
                "pipeline_parallel_size": placement.pipeline_parallel_size,
                "concurrency_levels": levels,
                "simulate_network": profile.simulate,
                "rtt_ms": profile.rtt_ms,
                "uplink_mbps": profile.uplink_mbps,
                "downlink_mbps": profile.downlink_mbps,
            }
            for placement, profile in runs
        ],
    )
    return path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cloud-vs-edge target placement with vLLM/OpenAI-compatible endpoints.")
    parser.add_argument("--plan", default="configs/target_placement_qwen32b_bf16.example.json")
    parser.add_argument("--output-dir", default="experiments/target_placement/qwen32b_bf16")
    parser.add_argument("--placement", default=None, help="Run only one placement name from the plan.")
    parser.add_argument("--network", default=None, help="Run only one network profile name from the plan.")
    parser.add_argument("--dry-run", action="store_true", help="Write planned_runs.json without sending requests.")
    parser.add_argument("--fake", action="store_true", help="Generate synthetic timing artifacts without a serving endpoint.")
    parser.add_argument("--save-text", action="store_true", help="Store generated text in run_summary.csv.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    plan = load_plan(args.plan)
    if args.dry_run:
        path = write_dry_run(plan, args.output_dir, placement_filter=args.placement, network_filter=args.network)
        print(f"Wrote planned runs: {path}")
        return
    paths = benchmark_plan(
        plan,
        args.output_dir,
        placement_filter=args.placement,
        network_filter=args.network,
        fake=args.fake,
        save_text=args.save_text,
    )
    print("Wrote target-placement benchmark artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
