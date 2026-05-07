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


def make_handler(model, model_name: str, device):
    import torch

    def synchronize() -> None:
        if getattr(device, "type", str(device)) == "cuda":
            torch.cuda.synchronize(device)

    class TargetHandler(BaseHTTPRequestHandler):
        server_version = "SpecDTarget/0.1"

        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                _json_response(self, 200, {"ok": True, "model": model_name, "device": str(device)})
                return
            if self.path == "/metadata":
                _json_response(
                    self,
                    200,
                    {
                        "model": model_name,
                        "device": str(device),
                        "config": _config_metadata(model.config),
                    },
                )
                return
            _error(self, 404, f"Unknown path: {self.path}")

        def do_POST(self) -> None:
            if self.path != "/forward":
                _error(self, 404, f"Unknown path: {self.path}")
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                payload = json.loads(raw_body.decode("utf-8"))
                if payload.get("use_cache"):
                    _error(self, 400, "Remote target service does not support use_cache=true.")
                    return

                input_ids = payload["input_ids"]
                tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)

                logits_start = payload.get("logits_start")
                logits_end = payload.get("logits_end")

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
                logits = logits.detach().to(device="cpu", dtype=torch.float32)
                logits_list = logits.tolist()
                cloud_verify_ns = model_forward_ns + (now_ns() - prepare_start)

                response_payload = {"logits": logits_list, "shape": list(logits.shape)}
                encode_start = now_ns()
                response_body = json.dumps(response_payload, separators=(",", ":")).encode("utf-8")
                response_encode_ns = now_ns() - encode_start

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.send_header("X-Target-Cloud-Verify-Ns", str(cloud_verify_ns))
                self.send_header("X-Target-Model-Forward-Ns", str(model_forward_ns))
                self.send_header("X-Target-Response-Encode-Ns", str(response_encode_ns))
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
    parser.add_argument("--device", default="auto")
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
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    print(f"Loading target model {args.model} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.to(device)
    model.eval()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, args.model, device))
    print(f"Target service ready at http://{args.host}:{args.port}")
    print("Endpoints: /health, /metadata, /forward")
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
