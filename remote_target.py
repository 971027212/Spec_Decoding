from __future__ import annotations

import http.client
import json
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from profiling import now_ns


class RemoteTargetModel:
    """A minimal model-like wrapper around the local target HTTP service."""

    supports_logits_window = True
    is_remote_target = True

    def __init__(
        self,
        base_url: str,
        output_device: str = "cpu",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.output_device = output_device
        self.device = output_device
        self.timeout = timeout
        self.metadata = self._get_json("/metadata")
        self.config = SimpleNamespace(**self.metadata.get("config", {}))

    def _connection(self) -> tuple[http.client.HTTPConnection, str]:
        parsed = urlparse(self.base_url)
        scheme = parsed.scheme or "http"
        host = parsed.netloc or parsed.path
        base_path = "" if parsed.netloc else ""
        if parsed.netloc and parsed.path:
            base_path = parsed.path.rstrip("/")
        connection_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        return connection_cls(host, timeout=self.timeout), base_path

    def _get_json(self, path: str) -> dict[str, Any]:
        connection, base_path = self._connection()
        try:
            connection.request("GET", f"{base_path}{path}")
            response = connection.getresponse()
            body = response.read()
            if response.status >= 400:
                raise RuntimeError(body.decode("utf-8", errors="replace"))
            return json.loads(body.decode("utf-8"))
        finally:
            connection.close()

    def __call__(
        self,
        input_ids,
        past_key_values=None,
        use_cache: bool = False,
        logits_start: int | None = None,
        logits_end: int | None = None,
        profiler=None,
        profile_metadata: dict[str, Any] | None = None,
    ):
        if use_cache or past_key_values is not None:
            raise ValueError("RemoteTargetModel intentionally does not support KV-cache transfer.")

        import torch

        ids = input_ids.detach().to("cpu").tolist() if hasattr(input_ids, "detach") else input_ids
        payload = {
            "input_ids": ids,
            "use_cache": False,
            "logits_start": logits_start,
            "logits_end": logits_end,
        }

        encode_start = now_ns()
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        encode_ns = now_ns() - encode_start

        connection, base_path = self._connection()
        headers: dict[str, str] = {}
        status = 0
        raw_body = b""
        total_start = now_ns()
        try:
            upload_start = now_ns()
            connection.putrequest("POST", f"{base_path}/forward")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Accept", "application/json")
            connection.putheader("Content-Length", str(len(body)))
            connection.endheaders(body)
            upload_ns = now_ns() - upload_start

            wait_start = now_ns()
            response = connection.getresponse()
            response_wait_ns = now_ns() - wait_start
            status = response.status
            headers = {key.lower(): value for key, value in response.getheaders()}

            downlink_start = now_ns()
            raw_body = response.read()
            downlink_ns = now_ns() - downlink_start
        finally:
            connection.close()
        total_http_ns = now_ns() - total_start

        decode_start = now_ns()
        decoded = json.loads(raw_body.decode("utf-8"))
        decode_ns = now_ns() - decode_start

        if status >= 400:
            raise RuntimeError(decoded.get("error", raw_body.decode("utf-8", errors="replace")))

        logits = torch.tensor(decoded["logits"], dtype=torch.float32, device=self.output_device)

        if profiler is not None:
            common = {
                "status": status,
                "request_bytes": len(body),
                "response_bytes": len(raw_body),
                "logits_start": logits_start,
                "logits_end": logits_end,
                "logits_shape": decoded.get("shape"),
            }
            if profile_metadata:
                common.update(profile_metadata)
            profiler.record("target_request_encode", encode_ns, **common)
            profiler.record("target_upload", upload_ns, **common)
            profiler.record("target_response_wait", response_wait_ns, **common)
            profiler.record("target_downlink", downlink_ns, **common)
            profiler.record("target_response_decode", decode_ns, **common)
            profiler.record("target_http_total", total_http_ns, **common)
            _record_header_ns(profiler, headers, "x-target-cloud-verify-ns", "target_cloud_verify", common)
            _record_header_ns(profiler, headers, "x-target-model-forward-ns", "target_model_forward", common)
            _record_header_ns(profiler, headers, "x-target-response-encode-ns", "target_server_encode", common)

        return SimpleNamespace(logits=logits, past_key_values=None)


def _record_header_ns(profiler, headers: dict[str, str], header: str, phase: str, metadata: dict[str, Any]) -> None:
    value = headers.get(header)
    if value is None:
        return
    try:
        profiler.record(phase, int(value), **metadata)
    except ValueError:
        return
