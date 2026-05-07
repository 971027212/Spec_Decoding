# Edge-cloud Speculative Decoding Experiment Plan

## Goal

Measure the timing distribution of a realistic edge-cloud inference flow:

- The edge/client sends work to a cloud target service.
- The cloud target service performs target-model computation.
- The cloud returns either final generated tokens or verification results.
- The benchmark reports end-to-end latency and phase shares, especially upload and downlink.

## Main Methods

| Method | Benchmark mode | Target location | Communication pattern | Purpose |
|---|---|---|---|---|
| Pure local Target-only | `local_target_ar` | Benchmark process | No HTTP | Local no-network baseline. |
| Cloud Target-only | `cloud_target_generate` | Target HTTP service | Upload prompt once, cloud generates full answer, download answer once | Normal cloud inference baseline. |
| Edge-cloud Speculative | `speculative` | Drafter on client, target HTTP service | Upload draft blocks, cloud verifies, download logits/verification blocks | Main speculative edge-cloud method. |
| Cloud-sim Target-only | `cloud_target_generate --simulate-network` | Target HTTP service | Same as Cloud Target-only plus simulated RTT/bandwidth | Cloud baseline under remote-network assumptions. |
| Cloud-sim Speculative | `speculative --simulate-network` | Drafter on client, target HTTP service | Same as Edge-cloud Speculative plus simulated RTT/bandwidth | Speculative method under remote-network assumptions. |

## Deprecated Diagnostic Mode

`target_ar` sends one HTTP request per generated token. This is not representative of normal cloud inference, where the cloud normally receives a prompt and generates the answer server-side. Keep `target_ar` only for RTT-sensitivity diagnosis; do not use it as the main baseline.

## Timing Phases

| Phase | Meaning |
|---|---|
| `drafter_generate` | Client-side drafter generation. |
| `target_request_encode` | Client request serialization. |
| `target_upload` | Client-to-cloud upload, including simulated uplink when enabled. |
| `target_cloud_generate` | Cloud-side full-answer generation for `cloud_target_generate`. |
| `target_cloud_verify` | Cloud-side target verification for speculative draft blocks. |
| `target_server_encode` | Cloud response serialization. |
| `target_downlink` | Cloud-to-client response transfer, including simulated downlink when enabled. |
| `target_response_decode` | Client response parsing. |
| `target_tensor_materialize` | Client-side binary logits tensor reconstruction for speculative verification. |
| `acceptance_sampling` | Client-side speculative acceptance/rejection sampling. |
| `generation_total` | End-to-end generation latency. |

## Server Commands

Start the target service:

```bash
cd /home/chajiahao/data/Spec_Decoding
conda activate specd
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python serve_target.py \
  --model /home/chajiahao/data/hf_models/Qwen2.5-1.5B \
  --device cuda \
  --local-files-only
```

Run the local no-network baseline:

```bash
python benchmark.py \
  --local-target-model /home/chajiahao/data/hf_models/Qwen2.5-1.5B \
  --tokenizer /home/chajiahao/data/hf_models/Qwen2.5-1.5B \
  --modes local_target_ar \
  --local-files-only \
  --device cuda \
  --output-dir experiments/edge_cloud/local_target
```

Run local-service cloud target and speculative methods:

```bash
python benchmark.py \
  --target-url http://127.0.0.1:8000 \
  --drafter-model /home/chajiahao/data/hf_models/Qwen2.5-0.5B \
  --tokenizer /home/chajiahao/data/hf_models/Qwen2.5-1.5B \
  --modes cloud_target_generate,speculative \
  --local-files-only \
  --target-output-device cuda \
  --response-format binary \
  --response-dtype float32 \
  --device cuda \
  --output-dir experiments/edge_cloud/local_service
```

Run simulated-cloud methods:

```bash
python benchmark.py \
  --target-url http://127.0.0.1:8000 \
  --drafter-model /home/chajiahao/data/hf_models/Qwen2.5-0.5B \
  --tokenizer /home/chajiahao/data/hf_models/Qwen2.5-1.5B \
  --modes cloud_target_generate,speculative \
  --local-files-only \
  --target-output-device cuda \
  --response-format binary \
  --response-dtype float32 \
  --device cuda \
  --simulate-network \
  --sim-rtt-ms 40 \
  --sim-uplink-mbps 100 \
  --sim-downlink-mbps 200 \
  --output-dir experiments/edge_cloud/cloud_sim
```

