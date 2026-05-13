from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from profiling import now_ns


DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    if headers:
        for key, value in headers.items():
            handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    _json_response(handler, status, {"error": message})


def _config_metadata(config) -> dict[str, Any]:
    keys = (
        "vocab_size",
        "max_position_embeddings",
        "max_context_length",
        "model_type",
        "hidden_size",
        "num_hidden_layers",
    )
    return {key: getattr(config, key) for key in keys if hasattr(config, key)}


def _parse_device_map(value: str | None) -> str | dict[str, int | str] | None:
    if value is None or value.strip() == "" or value.lower() in {"none", "single"}:
        return None
    normalized = value.strip()
    if normalized in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return normalized
    if normalized.startswith("{"):
        return json.loads(normalized)
    return normalized


def _parse_max_memory(value: str | None) -> dict[int | str, str] | None:
    if value is None or value.strip() == "":
        return None
    parsed: dict[int | str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, memory = item.split("=", 1)
        elif ":" in item:
            key, memory = item.split(":", 1)
        else:
            raise ValueError(f"Invalid max-memory item: {item!r}. Use entries like 0=22GiB,cpu=64GiB.")
        key = key.strip()
        memory = memory.strip()
        parsed[int(key) if key.isdigit() else key] = memory
    return parsed


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _json_safe_device_map(model) -> dict[str, str] | None:
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return None
    return {str(key): str(value) for key, value in device_map.items()}


def _normalize_cuda_device(value: Any):
    import torch

    if isinstance(value, int):
        return torch.device("cuda", value)
    text = str(value)
    if text.isdigit():
        return torch.device("cuda", int(text))
    try:
        device = torch.device(text)
    except (RuntimeError, TypeError):
        return None
    return device if device.type == "cuda" else None


def _infer_input_device(model, fallback):
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            for parameter in embeddings.parameters(recurse=True):
                if parameter.device.type != "meta":
                    return parameter.device
    except Exception:
        pass
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return fallback


def _cuda_sync_devices(model, input_device) -> list[Any]:
    import torch

    devices = []
    seen: set[str] = set()
    device_map = getattr(model, "hf_device_map", None) or {}
    for value in device_map.values():
        device = _normalize_cuda_device(value)
        if device is not None and str(device) not in seen:
            devices.append(device)
            seen.add(str(device))
    if not devices and getattr(input_device, "type", str(input_device)) == "cuda":
        devices.append(input_device if isinstance(input_device, torch.device) else torch.device(input_device))
    return devices


def make_handler(model, model_name: str, input_device, service_metadata: dict[str, Any] | None = None):
    import torch

    response_dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
    }
    sync_devices = _cuda_sync_devices(model, input_device)

    def synchronize() -> None:
        for sync_device in sync_devices:
            torch.cuda.synchronize(sync_device)

    def normalize_stop_tokens(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(item) for item in value]

    class TargetHandler(BaseHTTPRequestHandler):
        server_version = "SpecDTarget/0.1"

        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                _json_response(self, 200, {"ok": True, "model": model_name, "device": str(input_device), **(service_metadata or {})})
                return
            if self.path == "/metadata":
                _json_response(
                    self,
                    200,
                    {
                        "model": model_name,
                        "device": str(input_device),
                        "config": _config_metadata(model.config),
                        **(service_metadata or {}),
                    },
                )
                return
            _error(self, 404, f"Unknown path: {self.path}")

        def do_POST(self) -> None:
            if self.path == "/forward":
                self._handle_forward()
                return
            if self.path == "/verify_greedy":
                self._handle_verify_greedy()
                return
            if self.path == "/generate":
                self._handle_generate()
                return
            _error(self, 404, f"Unknown path: {self.path}")

        def _read_payload(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            return json.loads(raw_body.decode("utf-8"))

        def _handle_forward(self) -> None:
            try:
                payload = self._read_payload()
                if payload.get("use_cache"):
                    _error(self, 400, "Remote target service does not support use_cache=true for /forward.")
                    return

                input_ids = payload["input_ids"]
                tensor = torch.tensor(input_ids, dtype=torch.long, device=input_device)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)

                logits_start = payload.get("logits_start")
                logits_end = payload.get("logits_end")
                response_format = payload.get("response_format", "json")
                response_dtype_name = payload.get("response_dtype", "float32")
                if response_format not in {"json", "binary"}:
                    _error(self, 400, f"Unsupported response_format: {response_format}")
                    return
                if response_dtype_name not in response_dtypes:
                    _error(self, 400, f"Unsupported response_dtype: {response_dtype_name}")
                    return

                synchronize()
                forward_start = now_ns()
                with torch.no_grad():
                    outputs = model(input_ids=tensor, use_cache=False)
                synchronize()
                model_forward_ns = now_ns() - forward_start

                prepare_start = now_ns()
                logits = outputs.logits
                if logits_start is not None or logits_end is not None:
                    logits = logits[..., logits_start:logits_end, :]
                response_dtype = response_dtypes[response_dtype_name] if response_format == "binary" else torch.float32
                logits = logits.detach().to(device="cpu", dtype=response_dtype).contiguous()
                cloud_verify_ns = model_forward_ns + (now_ns() - prepare_start)

                encode_start = now_ns()
                if response_format == "binary":
                    response_body = logits.numpy().tobytes(order="C")
                    content_type = "application/octet-stream"
                else:
                    response_payload = {"logits": logits.tolist(), "shape": list(logits.shape)}
                    response_body = json.dumps(response_payload, separators=(",", ":")).encode("utf-8")
                    content_type = "application/json"
                response_encode_ns = now_ns() - encode_start

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(response_body)))
                self.send_header("X-Target-Cloud-Verify-Ns", str(cloud_verify_ns))
                self.send_header("X-Target-Model-Forward-Ns", str(model_forward_ns))
                self.send_header("X-Target-Response-Encode-Ns", str(response_encode_ns))
                self.send_header("X-Target-Response-Format", response_format)
                self.send_header("X-Target-Logits-Dtype", response_dtype_name if response_format == "binary" else "float32")
                self.send_header("X-Target-Logits-Shape", json.dumps(list(logits.shape), separators=(",", ":")))
                self.end_headers()
                self.wfile.write(response_body)
            except Exception as exc:
                _error(self, 500, str(exc))

        def _handle_generate(self) -> None:
            try:
                payload = self._read_payload()
                input_ids = payload["input_ids"]
                max_gen_len = int(payload.get("max_gen_len", 35))
                use_cache = bool(payload.get("use_cache", False))
                eos_tokens = normalize_stop_tokens(payload.get("eos_tokens_id", payload.get("eos_token_ids")))
                pad_token_id = int(payload.get("pad_token_id", 0))

                prompt = torch.tensor(input_ids, dtype=torch.long, device=input_device)
                if prompt.ndim == 2:
                    prompt = prompt[0]
                prompt_len = int(prompt.numel())
                max_seq_length = (
                    model.config.max_position_embeddings
                    if hasattr(model.config, "max_position_embeddings")
                    else (model.config.max_context_length if hasattr(model.config, "max_context_length") else 1024)
                )
                total_len = min(max_seq_length, prompt_len + max_gen_len)
                sequence = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=input_device)
                sequence[0, :prompt_len] = prompt
                stop_tokens = torch.tensor(eos_tokens, dtype=torch.long, device=input_device) if eos_tokens else None

                generated: list[int] = []
                model_forward_ns = 0
                cache = None
                stop_reason = "max_tokens"

                synchronize()
                generate_start = now_ns()
                with torch.no_grad():
                    for curr in range(prompt_len, total_len):
                        if use_cache and cache is not None:
                            model_input = sequence[..., curr - 1 : curr]
                        else:
                            model_input = sequence[..., :curr]

                        synchronize()
                        forward_start = now_ns()
                        outputs = model(input_ids=model_input, past_key_values=cache, use_cache=use_cache)
                        synchronize()
                        model_forward_ns += now_ns() - forward_start

                        logits = outputs.logits[..., -1, :]
                        next_token = torch.argmax(logits, dim=-1)
                        sequence[0, curr] = next_token
                        token_id = int(next_token.item())
                        generated.append(token_id)
                        cache = outputs.past_key_values if use_cache else None

                        if stop_tokens is not None and torch.isin(next_token, stop_tokens).item():
                            stop_reason = "eos"
                            break
                synchronize()
                cloud_generate_ns = now_ns() - generate_start

                encode_start = now_ns()
                response_payload = {
                    "output_ids": generated,
                    "generated_tokens": len(generated),
                    "stop_reason": stop_reason,
                    "use_cache": use_cache,
                }
                response_body = json.dumps(response_payload, separators=(",", ":")).encode("utf-8")
                response_encode_ns = now_ns() - encode_start

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.send_header("X-Target-Cloud-Generate-Ns", str(cloud_generate_ns))
                self.send_header("X-Target-Model-Forward-Ns", str(model_forward_ns))
                self.send_header("X-Target-Response-Encode-Ns", str(response_encode_ns))
                self.send_header("X-Target-Generate-Steps", str(len(generated)))
                self.end_headers()
                self.wfile.write(response_body)
            except Exception as exc:
                _error(self, 500, str(exc))

        def _handle_verify_greedy(self) -> None:
            try:
                payload = self._read_payload()
                input_ids = payload["input_ids"]
                current_position = int(payload["current_position"])
                corrected_gamma = int(payload.get("corrected_gamma", 0))

                tensor = torch.tensor(input_ids, dtype=torch.long, device=input_device)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                expected_length = current_position + corrected_gamma
                tensor = tensor[..., :expected_length]

                synchronize()
                forward_start = now_ns()
                with torch.no_grad():
                    outputs = model(input_ids=tensor, use_cache=False)
                synchronize()
                model_forward_ns = now_ns() - forward_start

                verify_start = now_ns()
                logits = outputs.logits[..., current_position - 1 : current_position + corrected_gamma, :]
                greedy_tokens = torch.argmax(logits, dim=-1)[0]
                accepted_count = 0
                for offset in range(corrected_gamma):
                    draft_token = int(tensor[0, current_position + offset].item())
                    target_token = int(greedy_tokens[offset].item())
                    if draft_token != target_token:
                        break
                    accepted_count += 1
                next_token = int(greedy_tokens[accepted_count].item())
                cloud_verify_ns = model_forward_ns + (now_ns() - verify_start)

                encode_start = now_ns()
                response_payload = {
                    "accepted_count": accepted_count,
                    "next_token_id": next_token,
                    "verified_tokens": corrected_gamma,
                    "all_accepted": accepted_count == corrected_gamma,
                }
                response_body = json.dumps(response_payload, separators=(",", ":")).encode("utf-8")
                response_encode_ns = now_ns() - encode_start

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.send_header("X-Target-Cloud-Verify-Ns", str(cloud_verify_ns))
                self.send_header("X-Target-Model-Forward-Ns", str(model_forward_ns))
                self.send_header("X-Target-Response-Encode-Ns", str(response_encode_ns))
                self.send_header("X-Target-Accepted-Count", str(accepted_count))
                self.end_headers()
                self.wfile.write(response_body)
            except Exception as exc:
                _error(self, 500, str(exc))

    return TargetHandler


def _resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _resolve_dtype(dtype_arg: str):
    import torch

    if dtype_arg == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_arg not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_arg}")
    return mapping[dtype_arg]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local HTTP target-model service for timing experiments.")
    parser.add_argument("--model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", help="Single-device fallback when --device-map is not set.")
    parser.add_argument("--device-map", default=None, help="Transformers device_map, e.g. auto, balanced, balanced_low_0, sequential, or a JSON map.")
    parser.add_argument("--max-memory", default=None, help="Comma-separated memory caps, e.g. 0=22GiB,1=22GiB,cpu=64GiB.")
    parser.add_argument("--offload-folder", default=None, help="Folder for Accelerate CPU/disk offload when using --device-map.")
    parser.add_argument("--no-split-module-classes", default=None, help="Comma-separated module classes that Accelerate should not split.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--local-files-only", action="store_true", help="Load model files from local cache/path only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    device_map = _parse_device_map(args.device_map)
    max_memory = _parse_max_memory(args.max_memory)
    no_split_module_classes = _parse_csv(args.no_split_module_classes)
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    if args.offload_folder:
        load_kwargs["offload_folder"] = args.offload_folder
    if no_split_module_classes:
        load_kwargs["no_split_module_classes"] = no_split_module_classes

    placement = f"device_map={device_map}" if device_map is not None else f"device={device}"
    print(f"Loading target model {args.model} with {placement}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if device_map is None:
        model.to(device)
        input_device = device
    else:
        input_device = _infer_input_device(model, device)
    model.eval()

    service_metadata = {
        "input_device": str(input_device),
        "device_map": _json_safe_device_map(model),
        "requested_device_map": str(device_map) if device_map is not None else None,
        "max_memory": {str(key): value for key, value in max_memory.items()} if max_memory else None,
    }
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, args.model, input_device, service_metadata))
    print(f"Target service ready at http://{args.host}:{args.port}")
    print(f"Target input device: {input_device}")
    if service_metadata["device_map"]:
        print(f"Target device map: {service_metadata['device_map']}")
    print("Endpoints: /health, /metadata, /forward, /verify_greedy, /generate")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down target service...")
    finally:
        server.server_close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
