from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import time
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from profiling import now_ns


@dataclass(frozen=True)
class NetworkSimulation:
    enabled: bool = False
    rtt_ms: float = 0.0
    uplink_mbps: float = 0.0
    downlink_mbps: float = 0.0

    @property
    def one_way_latency_ns(self) -> int:
        if not self.enabled or self.rtt_ms <= 0:
            return 0
        return int((self.rtt_ms / 2.0) * 1_000_000)

    def uplink_delay_ns(self, request_bytes: int) -> int:
        return self._transfer_delay_ns(request_bytes, self.uplink_mbps)

    def downlink_delay_ns(self, response_bytes: int) -> int:
        return self._transfer_delay_ns(response_bytes, self.downlink_mbps)

    def _transfer_delay_ns(self, num_bytes: int, mbps: float) -> int:
        if not self.enabled:
            return 0
        latency_ns = self.one_way_latency_ns
        if mbps <= 0:
            return latency_ns
        bandwidth_ns = int((num_bytes * 8 * 1_000_000_000) / (mbps * 1_000_000))
        return latency_ns + bandwidth_ns

    def metadata(self) -> dict[str, Any]:
        return {
            "network_simulated": self.enabled,
            "sim_rtt_ms": self.rtt_ms,
            "sim_uplink_mbps": self.uplink_mbps,
            "sim_downlink_mbps": self.downlink_mbps,
        }


def _sleep_ns(duration_ns: int) -> None:
    if duration_ns > 0:
        time.sleep(duration_ns / 1_000_000_000)


class RemoteTargetModel:
    """A minimal model-like wrapper around the local target HTTP service."""

    supports_logits_window = True
    is_remote_target = True

    def __init__(
        self,
        base_url: str,
        output_device: str = "cpu",
        timeout: float = 120.0,
        network_simulation: NetworkSimulation | None = None,
        response_format: str = "json",
        response_dtype: str = "float32",
    ) -> None:
        if response_format not in {"json", "binary"}:
            raise ValueError(f"Unsupported response_format: {response_format}")
        if response_dtype not in {"float32", "float16"}:
            raise ValueError(f"Unsupported response_dtype: {response_dtype}")
        self.base_url = base_url.rstrip("/")
        self.output_device = output_device
        self.device = output_device
        self.timeout = timeout
        self.network_simulation = network_simulation or NetworkSimulation()
        self.response_format = response_format
        self.response_dtype = response_dtype
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
            "response_format": self.response_format,
            "response_dtype": self.response_dtype,
        }

        encode_start = now_ns()
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        encode_ns = now_ns() - encode_start

        connection, base_path = self._connection()
        headers: dict[str, str] = {}
        status = 0
        raw_body = b""
        simulated_upload_ns = self.network_simulation.uplink_delay_ns(len(body))
        simulated_downlink_ns = 0
        total_start = now_ns()
        try:
            upload_start = now_ns()
            _sleep_ns(simulated_upload_ns)
            connection.putrequest("POST", f"{base_path}/forward")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Accept", "application/octet-stream" if self.response_format == "binary" else "application/json")
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
            simulated_downlink_ns = self.network_simulation.downlink_delay_ns(len(raw_body))
            _sleep_ns(simulated_downlink_ns)
            downlink_ns = now_ns() - downlink_start
        finally:
            connection.close()
        total_http_ns = now_ns() - total_start

        decode_start = now_ns()
        decoded = None
        response_format = headers.get("x-target-response-format", "json")
        logits_shape = None
        logits_dtype_name = headers.get("x-target-logits-dtype", self.response_dtype)
        if response_format == "binary":
            logits_shape = json.loads(headers["x-target-logits-shape"])
        else:
            decoded = json.loads(raw_body.decode("utf-8"))
            logits_shape = decoded.get("shape")
        decode_ns = now_ns() - decode_start

        if status >= 400:
            if decoded is None:
                decoded = json.loads(raw_body.decode("utf-8"))
            raise RuntimeError(decoded.get("error", raw_body.decode("utf-8", errors="replace")))

        materialize_start = now_ns()
        if response_format == "binary":
            dtype = torch.float16 if logits_dtype_name == "float16" else torch.float32
            logits = torch.frombuffer(bytearray(raw_body), dtype=dtype).reshape(logits_shape)
            logits = logits.to(device=self.output_device, dtype=torch.float32)
        else:
            logits = torch.tensor(decoded["logits"], dtype=torch.float32, device=self.output_device)
        tensor_materialize_ns = now_ns() - materialize_start

        if profiler is not None:
            common = {
                "status": status,
                "request_bytes": len(body),
                "response_bytes": len(raw_body),
                "response_format": response_format,
                "response_dtype": logits_dtype_name,
                "logits_start": logits_start,
                "logits_end": logits_end,
                "logits_shape": logits_shape,
                "simulated_upload_ns": simulated_upload_ns,
                "simulated_upload_ms": simulated_upload_ns / 1_000_000,
                "simulated_downlink_ns": simulated_downlink_ns,
                "simulated_downlink_ms": simulated_downlink_ns / 1_000_000,
            }
            common.update(self.network_simulation.metadata())
            if profile_metadata:
                common.update(profile_metadata)
            profiler.record("target_request_encode", encode_ns, **common)
            profiler.record("target_upload", upload_ns, **common)
            profiler.record("target_response_wait", response_wait_ns, **common)
            profiler.record("target_downlink", downlink_ns, **common)
            profiler.record("target_response_decode", decode_ns, **common)
            profiler.record("target_tensor_materialize", tensor_materialize_ns, **common)
            profiler.record("target_http_total", total_http_ns, **common)
            _record_header_ns(profiler, headers, "x-target-cloud-verify-ns", "target_cloud_verify", common)
            _record_header_ns(profiler, headers, "x-target-model-forward-ns", "target_model_forward", common)
            _record_header_ns(profiler, headers, "x-target-response-encode-ns", "target_server_encode", common)

        return SimpleNamespace(logits=logits, past_key_values=None)

    def generate(
        self,
        input_ids,
        max_gen_len: int,
        eos_tokens_id: int | list[int],
        pad_token_id: int = 0,
        use_cache: bool = False,
        profiler=None,
        profile_metadata: dict[str, Any] | None = None,
    ) -> list[int]:
        ids = input_ids.detach().to("cpu").tolist() if hasattr(input_ids, "detach") else input_ids
        payload = {
            "input_ids": ids,
            "max_gen_len": max_gen_len,
            "eos_tokens_id": eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id],
            "pad_token_id": pad_token_id,
            "use_cache": use_cache,
        }

        encode_start = now_ns()
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        encode_ns = now_ns() - encode_start

        connection, base_path = self._connection()
        headers: dict[str, str] = {}
        status = 0
        raw_body = b""
        simulated_upload_ns = self.network_simulation.uplink_delay_ns(len(body))
        simulated_downlink_ns = 0
        total_start = now_ns()
        try:
            upload_start = now_ns()
            _sleep_ns(simulated_upload_ns)
            connection.putrequest("POST", f"{base_path}/generate")
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
            simulated_downlink_ns = self.network_simulation.downlink_delay_ns(len(raw_body))
            _sleep_ns(simulated_downlink_ns)
            downlink_ns = now_ns() - downlink_start
        finally:
            connection.close()
        total_http_ns = now_ns() - total_start

        decode_start = now_ns()
        decoded = json.loads(raw_body.decode("utf-8"))
        decode_ns = now_ns() - decode_start

        if status >= 400:
            raise RuntimeError(decoded.get("error", raw_body.decode("utf-8", errors="replace")))

        output_ids = [int(token_id) for token_id in decoded["output_ids"]]
        if profiler is not None:
            common = {
                "status": status,
                "request_bytes": len(body),
                "response_bytes": len(raw_body),
                "response_format": "json",
                "generated_tokens": len(output_ids),
                "stop_reason": decoded.get("stop_reason"),
                "remote_generate_use_cache": use_cache,
                "simulated_upload_ns": simulated_upload_ns,
                "simulated_upload_ms": simulated_upload_ns / 1_000_000,
                "simulated_downlink_ns": simulated_downlink_ns,
                "simulated_downlink_ms": simulated_downlink_ns / 1_000_000,
            }
            common.update(self.network_simulation.metadata())
            if profile_metadata:
                common.update(profile_metadata)
            profiler.record("target_request_encode", encode_ns, **common)
            profiler.record("target_upload", upload_ns, **common)
            profiler.record("target_response_wait", response_wait_ns, **common)
            profiler.record("target_downlink", downlink_ns, **common)
            profiler.record("target_response_decode", decode_ns, **common)
            profiler.record("target_http_total", total_http_ns, **common)
            _record_header_ns(profiler, headers, "x-target-cloud-generate-ns", "target_cloud_generate", common)
            _record_header_ns(profiler, headers, "x-target-model-forward-ns", "target_model_forward", common)
            _record_header_ns(profiler, headers, "x-target-response-encode-ns", "target_server_encode", common)
            steps = headers.get("x-target-generate-steps")
            if steps is not None:
                try:
                    profiler.set_metric("target_generate_steps", int(steps))
                except ValueError:
                    pass
        return output_ids


def _record_header_ns(profiler, headers: dict[str, str], header: str, phase: str, metadata: dict[str, Any]) -> None:
    value = headers.get(header)
    if value is None:
        return
    try:
        profiler.record(phase, int(value), **metadata)
    except ValueError:
        return
