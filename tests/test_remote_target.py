import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from profiling import TimingRecorder
from remote_target import RemoteTargetModel


try:
    import torch
except ModuleNotFoundError:
    torch = None


class FakeTargetHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path != "/metadata":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({"model": "fake", "config": {"vocab_size": 4, "max_position_embeddings": 16}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode())
        if self.path == "/generate":
            body = json.dumps({"output_ids": [2, 3], "generated_tokens": 2, "stop_reason": "max_tokens"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Target-Cloud-Generate-Ns", "456")
            self.send_header("X-Target-Model-Forward-Ns", "400")
            self.send_header("X-Target-Response-Encode-Ns", "56")
            self.send_header("X-Target-Generate-Steps", "2")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/verify_greedy":
            body = json.dumps({"accepted_count": 1, "next_token_id": 3, "verified_tokens": 2, "all_accepted": False}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Target-Cloud-Verify-Ns", "456")
            self.send_header("X-Target-Model-Forward-Ns", "400")
            self.send_header("X-Target-Response-Encode-Ns", "56")
            self.send_header("X-Target-Accepted-Count", "1")
            self.end_headers()
            self.wfile.write(body)
            return

        if payload.get("response_format") == "binary":
            tensor = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float32)
            body = tensor.numpy().tobytes(order="C")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Target-Response-Format", "binary")
            self.send_header("X-Target-Logits-Dtype", "float32")
            self.send_header("X-Target-Logits-Shape", "[1,1,4]")
            self.send_header("X-Target-Cloud-Verify-Ns", "123")
            self.send_header("X-Target-Model-Forward-Ns", "100")
            self.send_header("X-Target-Response-Encode-Ns", "23")
            self.end_headers()
            self.wfile.write(body)
            return

        body = json.dumps({"logits": [[[0.1, 0.2, 0.3, 0.4]]], "shape": [1, 1, 4]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Target-Cloud-Verify-Ns", "123")
        self.send_header("X-Target-Model-Forward-Ns", "100")
        self.send_header("X-Target-Response-Encode-Ns", "23")
        self.end_headers()
        self.wfile.write(body)


@unittest.skipIf(torch is None, "torch is not installed")
class RemoteTargetTests(unittest.TestCase):
    def test_remote_target_returns_logits_and_records_timings(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTargetHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            target = RemoteTargetModel(url)
            recorder = TimingRecorder(mode="speculative")

            output = target(
                torch.tensor([[1, 2, 3]]),
                logits_start=2,
                logits_end=3,
                profiler=recorder,
            )

            self.assertEqual(tuple(output.logits.shape), (1, 1, 4))
            phases = {event["phase"] for event in recorder.events}
            self.assertIn("target_upload", phases)
            self.assertIn("target_cloud_verify", phases)
        finally:
            server.shutdown()
            server.server_close()

    def test_remote_target_decodes_binary_logits(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTargetHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            target = RemoteTargetModel(url, response_format="binary", response_dtype="float32")
            recorder = TimingRecorder(mode="speculative")

            output = target(
                torch.tensor([[1, 2, 3]]),
                logits_start=2,
                logits_end=3,
                profiler=recorder,
            )

            self.assertEqual(tuple(output.logits.shape), (1, 1, 4))
            self.assertAlmostEqual(float(output.logits[0, 0, 3]), 0.4, places=5)
            phases = {event["phase"] for event in recorder.events}
            self.assertIn("target_tensor_materialize", phases)
        finally:
            server.shutdown()
            server.server_close()

    def test_remote_target_verify_greedy_records_small_response(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTargetHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            target = RemoteTargetModel(url)
            recorder = TimingRecorder(mode="speculative_server_accept")

            result = target.verify_greedy(
                [1, 2, 3, 4],
                current_position=2,
                corrected_gamma=2,
                profiler=recorder,
            )

            self.assertEqual(result["accepted_count"], 1)
            self.assertEqual(result["next_token_id"], 3)
            phases = {event["phase"] for event in recorder.events}
            self.assertIn("target_upload", phases)
            self.assertIn("target_cloud_verify", phases)
            self.assertIn("target_downlink", phases)
        finally:
            server.shutdown()
            server.server_close()

    def test_remote_target_generate_records_cloud_generation(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTargetHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            target = RemoteTargetModel(url)
            recorder = TimingRecorder(mode="cloud_target_generate")

            output_ids = target.generate(
                [1, 2],
                max_gen_len=2,
                eos_tokens_id=[0],
                profiler=recorder,
            )

            self.assertEqual(output_ids, [2, 3])
            phases = {event["phase"] for event in recorder.events}
            self.assertIn("target_upload", phases)
            self.assertIn("target_cloud_generate", phases)
            self.assertEqual(recorder.metrics["target_generate_steps"], 2)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
