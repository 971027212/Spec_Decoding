from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable

from profiling import TimingRecorder, aggregate_summaries, now_ns, write_csv, write_jsonl


DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DRAFTER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPTS = [
    "Explain speculative decoding in two concise paragraphs.",
    "Write a short Python function that computes Fibonacci numbers.",
    "Summarize why latency profiling matters for distributed inference.",
]
PHASES_FOR_PLOTS = [
    "target_request_encode_ms",
    "drafter_generate_ms",
    "target_forward_ms",
    "target_cloud_generate_ms",
    "target_upload_ms",
    "target_cloud_verify_ms",
    "target_server_encode_ms",
    "target_downlink_ms",
    "target_response_decode_ms",
    "target_tensor_materialize_ms",
    "acceptance_sampling_ms",
]


def parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"speculative", "cloud_target_generate", "target_ar", "local_target_ar"}
    unknown = [mode for mode in modes if mode not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unsupported modes: {', '.join(unknown)}")
    return modes


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompt_path = Path(path)
    prompts = [
        line.strip()
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


def write_outputs(recorders: list[TimingRecorder], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = [event for recorder in recorders for event in recorder.events]
    summaries = [recorder.summary() for recorder in recorders]
    aggregates = aggregate_summaries(summaries)

    paths = {
        "raw_events": output_dir / "raw_events.jsonl",
        "run_summary": output_dir / "run_summary.csv",
        "aggregate_summary": output_dir / "aggregate_summary.csv",
    }
    write_jsonl(paths["raw_events"], events)
    write_csv(paths["run_summary"], summaries)
    write_csv(paths["aggregate_summary"], aggregates)
    plot_paths = write_plots(summaries, output_dir)
    paths.update(plot_paths)
    return paths


def write_plots(summaries: list[dict], output_dir: Path) -> dict[str, Path]:
    measured = [row for row in summaries if not row.get("warmup", False)]
    if not measured:
        return {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; CSV/JSONL were written but PNG charts were skipped.")
        return {}

    output_paths: dict[str, Path] = {}
    modes = sorted({row.get("mode") for row in measured})

    stacked_path = output_dir / "phase_stacked.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = [0.0 for _ in modes]
    for phase in PHASES_FOR_PLOTS:
        values = []
        for mode in modes:
            mode_values = [float(row.get(phase, 0.0)) for row in measured if row.get("mode") == mode]
            values.append(sum(mode_values) / len(mode_values) if mode_values else 0.0)
        ax.bar(modes, values, bottom=bottoms, label=phase.replace("_ms", ""))
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_ylabel("mean duration (ms)")
    ax.set_title("Mean phase distribution by mode")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(stacked_path, dpi=160)
    plt.close(fig)
    output_paths["phase_stacked"] = stacked_path

    boxplot_path = output_dir / "phase_boxplot.png"
    labels = []
    values = []
    for phase in PHASES_FOR_PLOTS:
        phase_values = [float(row.get(phase, 0.0)) for row in measured if float(row.get(phase, 0.0)) > 0.0]
        if phase_values:
            labels.append(phase.replace("_ms", ""))
            values.append(phase_values)
    if values:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.boxplot(values, labels=labels, showfliers=False)
        ax.set_ylabel("duration (ms)")
        ax.set_title("Per-phase latency distribution")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(boxplot_path, dpi=160)
        plt.close(fig)
        output_paths["phase_boxplot"] = boxplot_path
    return output_paths


def run_fake(args: argparse.Namespace) -> dict[str, Path]:
    recorders: list[TimingRecorder] = []
    modes = parse_modes(args.modes)
    for mode in modes:
        for run_index in range(args.runs):
            recorder = TimingRecorder(
                mode=mode,
                prompt_id=0,
                run_index=run_index,
                warmup=False,
                extra={"gamma": args.gamma, "max_gen_len": args.max_tokens, "fake": True},
            )
            scale = run_index + 1
            if mode == "local_target_ar":
                recorder.record("target_forward", 4_500_000 + 100_000 * scale)
            elif mode == "cloud_target_generate":
                recorder.record("target_request_encode", 120_000 + 5_000 * scale)
                recorder.record("target_upload", 900_000 + 25_000 * scale)
                recorder.record("target_cloud_generate", 5_000_000 + 100_000 * scale)
                recorder.record("target_server_encode", 80_000 + 5_000 * scale)
                recorder.record("target_downlink", 700_000 + 20_000 * scale)
                recorder.record("target_response_decode", 100_000 + 5_000 * scale)
            else:
                recorder.record("target_request_encode", 120_000 + 5_000 * scale)
                recorder.record("target_upload", 900_000 + 25_000 * scale)
                recorder.record("target_cloud_verify", 4_000_000 + 100_000 * scale)
                recorder.record("target_server_encode", 700_000 + 15_000 * scale)
                recorder.record("target_downlink", 1_100_000 + 20_000 * scale)
                recorder.record("target_response_decode", 500_000 + 10_000 * scale)
            if mode == "speculative":
                recorder.record("drafter_generate", 2_000_000 + 80_000 * scale)
                recorder.record("acceptance_sampling", 200_000 + 5_000 * scale)
                recorder.set_metric("drafts_accepted", 3 + run_index)
                recorder.set_metric("drafts_speculated", 5 + run_index)
                recorder.set_metric("acceptance_rate", (3 + run_index) / (5 + run_index))
            generated_tokens = args.max_tokens
            generation_ns = 10_000_000 + 150_000 * scale
            recorder.record("generation_total", generation_ns)
            recorder.set_metric("generated_tokens", generated_tokens)
            recorder.set_metric("throughput_tokens_s", generated_tokens / (generation_ns / 1_000_000_000))
            recorders.append(recorder)
    return write_outputs(recorders, args.output_dir)


def _resolve_torch_dtype(torch, dtype: str):
    if dtype == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def _resolve_device(torch, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _end_tokens(tokenizer) -> int | list[int]:
    token_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    for token in ("<|eot_id|>", "<|im_end|>"):
        converted = tokenizer.convert_tokens_to_ids(token)
        if isinstance(converted, int) and converted >= 0 and converted not in token_ids:
            token_ids.append(converted)
    return token_ids if token_ids else 1


def _tokenize_prompt(tokenizer, prompt: str, use_chat_template: bool) -> list[int]:
    text = prompt
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return tokenizer(text, return_tensors="pt").input_ids[0].tolist()


def run_real(args: argparse.Namespace) -> dict[str, Path]:
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from remote_target import NetworkSimulation, RemoteTargetModel
    from sampling import autoregressive_generate, speculative_generate
    from utils.logits_processor import GreedyProcessor

    modes = parse_modes(args.modes)
    prompts = load_prompts(args.prompts_file)
    device = _resolve_device(torch, args.device)
    dtype = _resolve_torch_dtype(torch, args.dtype)
    remote_modes = {"speculative", "cloud_target_generate", "target_ar"}
    uses_remote_target = any(mode in remote_modes for mode in modes)
    network_simulation = NetworkSimulation(
        enabled=args.simulate_network,
        rtt_ms=args.sim_rtt_ms,
        uplink_mbps=args.sim_uplink_mbps,
        downlink_mbps=args.sim_downlink_mbps,
    )
    target = None
    if uses_remote_target:
        target = RemoteTargetModel(
            args.target_url,
            output_device=args.target_output_device,
            timeout=args.timeout,
            network_simulation=network_simulation,
            response_format=args.response_format,
            response_dtype=args.response_dtype,
        )
    tokenizer_model = args.tokenizer or (target.metadata.get("model") if target is not None else args.local_target_model)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    drafter = None
    if "speculative" in modes:
        print(f"Loading drafter model {args.drafter_model} on {device}...")
        drafter = AutoModelForCausalLM.from_pretrained(
            args.drafter_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
        )
        drafter.to(device)
        drafter.eval()

    local_target = None
    if "local_target_ar" in modes:
        print(f"Loading local target model {args.local_target_model} on {device}...")
        local_target = AutoModelForCausalLM.from_pretrained(
            args.local_target_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
        )
        local_target.to(device)
        local_target.eval()

    recorders: list[TimingRecorder] = []
    processor = GreedyProcessor()
    end_tokens = _end_tokens(tokenizer)
    total_runs = args.warmup_runs + args.runs

    for prompt_id, prompt in enumerate(prompts):
        input_ids = _tokenize_prompt(tokenizer, prompt, args.chat_template)
        for mode in modes:
            for iteration in range(total_runs):
                warmup = iteration < args.warmup_runs
                run_index = iteration - args.warmup_runs if not warmup else iteration
                _set_seed(torch, args.seed + prompt_id * 1000 + iteration)
                recorder = TimingRecorder(
                    mode=mode,
                    prompt_id=prompt_id,
                    run_index=run_index,
                    warmup=warmup,
                    extra={
                        "gamma": args.gamma,
                        "max_gen_len": args.max_tokens,
                        "prompt_tokens": len(input_ids),
                        "target_url": args.target_url,
                        "local_target_model": args.local_target_model,
                        "drafter_model": args.drafter_model,
                        "response_format": args.response_format,
                        "response_dtype": args.response_dtype,
                        "cloud_generate_use_cache": args.cloud_use_cache,
                        **network_simulation.metadata(),
                    },
                )

                start = now_ns()
                if mode == "speculative":
                    if drafter is None:
                        raise RuntimeError("Speculative mode requires a drafter model.")
                    if target is None:
                        raise RuntimeError("Speculative mode requires a remote target service.")
                    output_ids, accept_rate = speculative_generate(
                        input_ids,
                        drafter,
                        target,
                        tokenizer=tokenizer,
                        gamma=args.gamma,
                        logits_processor=processor,
                        max_gen_len=args.max_tokens,
                        eos_tokens_id=end_tokens,
                        pad_token_id=pad_token_id,
                        use_cache=False,
                        debug=False,
                        profiler=recorder,
                    )
                    recorder.set_metric("acceptance_rate", accept_rate)
                elif mode == "target_ar":
                    if target is None:
                        raise RuntimeError("target_ar mode requires a remote target service.")
                    output_ids = autoregressive_generate(
                        input_ids,
                        target,
                        max_gen_len=args.max_tokens,
                        logits_processor=processor,
                        eos_tokens_id=end_tokens,
                        pad_token_id=pad_token_id,
                        use_cache=False,
                        debug=False,
                        profiler=recorder,
                    )
                elif mode == "cloud_target_generate":
                    if target is None:
                        raise RuntimeError("cloud_target_generate mode requires a remote target service.")
                    output_ids = target.generate(
                        input_ids,
                        max_gen_len=args.max_tokens,
                        eos_tokens_id=end_tokens,
                        pad_token_id=pad_token_id,
                        use_cache=args.cloud_use_cache,
                        profiler=recorder,
                        profile_metadata={"call_kind": "cloud_generate"},
                    )
                elif mode == "local_target_ar":
                    if local_target is None:
                        raise RuntimeError("local_target_ar mode requires a local target model.")
                    output_ids = autoregressive_generate(
                        input_ids,
                        local_target,
                        max_gen_len=args.max_tokens,
                        logits_processor=processor,
                        eos_tokens_id=end_tokens,
                        pad_token_id=pad_token_id,
                        use_cache=False,
                        debug=False,
                        profiler=recorder,
                    )
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                generation_ns = now_ns() - start
                recorder.record("generation_total", generation_ns)
                recorder.set_metric("generated_tokens", len(output_ids))
                recorder.set_metric("throughput_tokens_s", len(output_ids) / (generation_ns / 1_000_000_000))
                if args.save_text:
                    recorder.set_metric("output_text", tokenizer.decode(output_ids, skip_special_tokens=True))
                recorders.append(recorder)
                label = "warmup" if warmup else "run"
                print(f"{mode} prompt={prompt_id} {label}={run_index} tokens={len(output_ids)}")

    return write_outputs(recorders, args.output_dir)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding timing distribution.")
    parser.add_argument("--target-url", default="http://127.0.0.1:8000")
    parser.add_argument("--local-target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--drafter-model", default=DEFAULT_DRAFTER_MODEL)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument(
        "--modes",
        default="speculative,cloud_target_generate",
        help=(
            "Comma-separated modes. Main modes: speculative, cloud_target_generate, "
            "local_target_ar. Diagnostic legacy mode: target_ar (one HTTP request per token)."
        ),
    )
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=35)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-output-device", default="cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--response-format", default="json", choices=["json", "binary"])
    parser.add_argument("--response-dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-text", action="store_true")
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cloud-use-cache", action="store_true", help="Use KV-cache inside the /generate cloud target baseline.")
    parser.add_argument("--local-files-only", action="store_true", help="Load tokenizer/drafter files locally only.")
    parser.add_argument("--simulate-network", action="store_true", help="Add code-level remote network delay simulation.")
    parser.add_argument("--sim-rtt-ms", type=float, default=0.0, help="Simulated round-trip propagation latency.")
    parser.add_argument("--sim-uplink-mbps", type=float, default=0.0, help="Simulated client-to-cloud bandwidth.")
    parser.add_argument("--sim-downlink-mbps", type=float, default=0.0, help="Simulated cloud-to-client bandwidth.")
    parser.add_argument("--fake", action="store_true", help="Generate synthetic timing files without loading models.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    paths = run_fake(args) if args.fake else run_real(args)
    print("Wrote benchmark artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
