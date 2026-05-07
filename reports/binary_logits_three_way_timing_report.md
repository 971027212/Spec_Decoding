# Speculative Decoding 推理阶段耗时与占比组会汇报

## 1. 汇报目标

本次实验的目标是测量 speculative decoding 推理过程中各阶段的耗时和占比，尤其关注端云通信带来的 upload、cloud verify、downlink 开销。为了避免口径混淆，本报告把三类实验明确区分：

| 口径 | 目录 | 含义 |
|---|---|---|
| 纯本地 target-only | `experiments/binary_logits/local_target_ar/` | target 模型直接在 benchmark 进程中推理，无 HTTP、无 upload/downlink。 |
| localhost 端云架构 | `experiments/binary_logits/localhost/` | 客户端和 target 服务在同一台服务器，通过 localhost HTTP 通信，返回 binary logits。 |
| 模拟远端端云架构 | `experiments/binary_logits/cloud_sim/` | 仍在 localhost 上运行，但客户端注入 40 ms RTT、100 Mbps 上行、200 Mbps 下行延迟。 |

需要特别说明：之前结果中的 `target_ar` 不是“纯本地 target-only”，而是“通过 HTTP 调用 target 服务的 target-only baseline”。因此它有 upload/downlink 是合理的；真正没有网络传输的是本次新增的 `local_target_ar`。

## 2. 方法实现

### 2.1 系统结构

本项目在原始 speculative decoding 代码上增加了端云拆分和 profiling：

| 文件 | 作用 |
|---|---|
| `serve_target.py` | target 模型服务端，提供 `/health`、`/metadata`、`/forward`。 |
| `remote_target.py` | 客户端 HTTP wrapper，负责请求编码、上传、等待服务端、下载、响应解析和网络模拟。 |
| `sampling/speculative_decoding.py` | speculative decoding 主流程，记录 drafter、target verify、acceptance sampling 等阶段。 |
| `sampling/base_decoding.py` | autoregressive baseline，支持 HTTP target 和纯本地 target 两种调用方式。 |
| `benchmark.py` | 非交互式实验入口，生成 JSONL、CSV 和 PNG 图表。 |
| `profiling.py` | 记录事件、汇总每次 run、生成 aggregate summary。 |

### 2.2 代码流程

```mermaid
flowchart LR
    P["Prompt tokens"] --> C["Benchmark client"]
    C --> D["Drafter model<br/>Qwen2.5-0.5B"]
    D --> DT["Draft tokens"]
    DT --> U["Upload request"]
    U --> T["Target service<br/>Qwen2.5-1.5B"]
    T --> V["Cloud verify<br/>target forward + logits slice"]
    V --> B["Binary logits encode"]
    B --> R["Downlink response"]
    R --> A["Client decode + tensor materialize"]
    A --> S["Acceptance sampling"]
    S --> O["Generated tokens"]
```

### 2.3 Binary logits 协议

早期 JSON logits 会把完整 vocab logits 转成 JSON list，再由客户端 `json.loads` 解析，序列化和解析开销很高。当前实验使用 binary logits：

1. 客户端请求中设置 `response_format=binary`、`response_dtype=float32`。
2. 服务端 target forward 后只截取本轮需要的 logits window。
3. 服务端将 logits 转成 CPU contiguous tensor，再用 `numpy().tobytes()` 返回。
4. 响应 header 携带 logits shape、dtype 和服务端计时。
5. 客户端用 `torch.frombuffer(...).reshape(shape)` 恢复 tensor。

这个改动不改变 speculative decoding 算法语义，只减少协议层序列化和解析成本。

### 2.4 远端网络模拟

模拟远端实验没有使用真实公网或 `tc/netem`，而是在客户端代码中注入延迟：

```text
target_upload   = RTT / 2 + request_bytes / uplink_bandwidth
target_downlink = RTT / 2 + response_bytes / downlink_bandwidth
```

本次参数：

| 参数 | 数值 |
|---|---:|
| RTT | 40 ms |
| 上行带宽 | 100 Mbps |
| 下行带宽 | 200 Mbps |

因此 upload/downlink 是“按远端网络参数注入的代码级近似”，比 localhost 更接近云端服务场景，但仍不是公网实测。

## 3. 时间拆分口径

为避免重复计算，本报告使用非重叠阶段汇总：

| 阶段 | 含义 |
|---|---|
| `drafter_generate` | 客户端小模型生成 draft tokens。 |
| `target_request_encode` | 客户端构造 HTTP JSON 请求体。 |
| `target_upload` | 请求上传时间；远端模拟时包含半 RTT 和上行带宽延迟。 |
| `target_cloud_verify` | 服务端 target forward、logits 截取和 CPU 转移。 |
| `target_server_encode` | 服务端将 logits 编码为 binary response body。 |
| `server_other_wait` | `target_response_wait - target_cloud_verify - target_server_encode`，主要是服务端框架和排队等待等剩余时间。 |
| `target_downlink` | 客户端读取响应 body；远端模拟时包含半 RTT 和下行带宽延迟。 |
| `target_response_decode` | 客户端解析响应 header 或元数据。 |
| `target_tensor_materialize` | 客户端把 binary logits 恢复成 tensor。 |
| `acceptance_sampling` | speculative decoding 接受/拒绝采样。 |
| `residual_other` | 总耗时减去上述可观测阶段后的剩余本地开销。 |

`target_model_forward` 是 `target_cloud_verify` 的内部子项，不单独加到总占比中，否则会重复计算。

## 4. 实验设计

### 4.1 模型和环境

| 项目 | 设置 |
|---|---|
| Target model | `/home/chajiahao/data/hf_models/Qwen2.5-1.5B` |
| Drafter model | `/home/chajiahao/data/hf_models/Qwen2.5-0.5B` |
| 运行环境 | Conda 环境 `specd` |
| 设备 | CUDA GPU server |
| 解码策略 | GreedyProcessor |
| `gamma` | 4 |
| `max_tokens` | 35 |
| prompts | 默认 3 条 prompt |
| warmup | 每个 prompt/mode 1 次 |
| measured runs | 每个 prompt/mode 3 次，共 9 个样本 |
| HTTP logits 格式 | binary float32 |

### 4.2 实验命令口径

纯本地 target-only：

```bash
python benchmark.py \
  --local-target-model /home/chajiahao/data/hf_models/Qwen2.5-1.5B \
  --tokenizer /home/chajiahao/data/hf_models/Qwen2.5-1.5B \
  --modes local_target_ar \
  --local-files-only \
  --device cuda \
  --output-dir experiments/binary_logits/local_target_ar
```

localhost 端云架构：

```bash
python benchmark.py \
  --target-url http://127.0.0.1:8000 \
  --drafter-model /home/chajiahao/data/hf_models/Qwen2.5-0.5B \
  --tokenizer /home/chajiahao/data/hf_models/Qwen2.5-1.5B \
  --modes speculative,target_ar \
  --response-format binary \
  --response-dtype float32 \
  --local-files-only \
  --device cuda \
  --output-dir experiments/binary_logits/localhost
```

模拟远端端云架构：

```bash
python benchmark.py \
  --target-url http://127.0.0.1:8000 \
  --drafter-model /home/chajiahao/data/hf_models/Qwen2.5-0.5B \
  --tokenizer /home/chajiahao/data/hf_models/Qwen2.5-1.5B \
  --modes speculative,target_ar \
  --response-format binary \
  --response-dtype float32 \
  --simulate-network \
  --sim-rtt-ms 40 \
  --sim-uplink-mbps 100 \
  --sim-downlink-mbps 200 \
  --local-files-only \
  --device cuda \
  --output-dir experiments/binary_logits/cloud_sim
```

## 5. 总体实验结果

| 场景 | mode | 平均总耗时 ms | P50 ms | P95 ms | 吞吐 tok/s | 平均生成 tokens | target 调用次数/run | 通信耗时 ms | 通信占比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 纯本地 | `local_target_ar` | 840.87 | 836.82 | 866.75 | 41.64 | 35.00 | 35 local forwards | 0.00 | 0.00% |
| localhost | `speculative` | 1252.06 | 1274.17 | 1556.77 | 26.34 | 32.78 | 10.33 HTTP calls | 23.82 | 1.90% |
| localhost | `target_ar` | 1550.38 | 1531.48 | 1681.73 | 22.69 | 35.00 | 35 HTTP calls | 43.73 | 2.82% |
| 模拟远端 | `speculative` | 3095.50 | 3132.09 | 3742.04 | 10.59 | 32.78 | 10.33 HTTP calls | 1526.61 | 49.32% |
| 模拟远端 | `target_ar` | 4024.85 | 4023.74 | 4053.23 | 8.70 | 35.00 | 35 HTTP calls | 2306.39 | 57.30% |

主要结论：

1. 纯本地 target-only 最快，平均 840.87 ms，因为没有 HTTP、没有 logits 传输、没有 drafter。
2. 在 localhost 端云架构下，speculative 比 HTTP target-only 快：1550.38 / 1252.06 = 1.24x。
3. 在模拟远端端云架构下，speculative 比 HTTP target-only 快：4024.85 / 3095.50 = 1.30x。
4. 远端模拟下通信成为主导开销：speculative 的 upload+downlink 占 49.32%，HTTP target-only 占 57.30%。

## 6. 阶段耗时与占比

### 6.1 纯本地 target-only

| 阶段 | 平均耗时 ms | 占比 |
|---|---:|---:|
| `target_forward` | 816.15 | 97.06% |
| `residual_other` | 24.72 | 2.94% |

纯本地 baseline 几乎全部时间都在 target forward，没有端云通信阶段。这也是回答“为什么 target-only 会有网络传输”的关键：只有 `local_target_ar` 没有网络；HTTP 版 `target_ar` 为了模拟云端 target 服务，必须经过上传和下行。

### 6.2 localhost 端云架构

| 阶段 | `speculative` ms | `speculative` 占比 | `target_ar` ms | `target_ar` 占比 |
|---|---:|---:|---:|---:|
| `drafter_generate` | 699.99 | 55.91% | 0.00 | 0.00% |
| `target_request_encode` | 0.41 | 0.03% | 3.31 | 0.21% |
| `target_upload` | 2.53 | 0.20% | 17.60 | 1.13% |
| `target_cloud_verify` | 467.79 | 37.36% | 1371.68 | 88.47% |
| `target_server_encode` | 5.82 | 0.46% | 3.54 | 0.23% |
| `server_other_wait` | 20.05 | 1.60% | 59.47 | 3.84% |
| `target_downlink` | 21.29 | 1.70% | 26.13 | 1.69% |
| `target_response_decode` | 0.27 | 0.02% | 1.18 | 0.08% |
| `target_tensor_materialize` | 14.89 | 1.19% | 18.69 | 1.21% |
| `acceptance_sampling` | 3.56 | 0.28% | 0.00 | 0.00% |
| `residual_other` | 15.46 | 1.23% | 48.79 | 3.15% |

localhost 下通信本身不是瓶颈，upload+downlink 只有 1.90% 到 2.82%。speculative 更快的主要原因是 target 服务调用次数从 35 次下降到平均 10.33 次，服务端 target verify 总时间从 1371.68 ms 降到 467.79 ms。

### 6.3 模拟远端端云架构

| 阶段 | `speculative` ms | `speculative` 占比 | `target_ar` ms | `target_ar` 占比 |
|---|---:|---:|---:|---:|
| `drafter_generate` | 1025.43 | 33.13% | 0.00 | 0.00% |
| `target_request_encode` | 0.53 | 0.02% | 3.45 | 0.09% |
| `target_upload` | 213.03 | 6.88% | 725.46 | 18.02% |
| `target_cloud_verify` | 464.03 | 14.99% | 1564.90 | 38.88% |
| `target_server_encode` | 6.36 | 0.21% | 4.58 | 0.11% |
| `server_other_wait` | 21.84 | 0.71% | 72.07 | 1.79% |
| `target_downlink` | 1313.58 | 42.43% | 1580.93 | 39.28% |
| `target_response_decode` | 0.52 | 0.02% | 1.58 | 0.04% |
| `target_tensor_materialize` | 20.46 | 0.66% | 21.49 | 0.53% |
| `acceptance_sampling` | 6.11 | 0.20% | 0.00 | 0.00% |
| `residual_other` | 23.62 | 0.76% | 50.39 | 1.25% |

模拟远端下，通信开销明显变成主导因素：

| 场景 | upload ms | downlink ms | upload+downlink ms | 通信占比 |
|---|---:|---:|---:|---:|
| `speculative` | 213.03 | 1313.58 | 1526.61 | 49.32% |
| `target_ar` | 725.46 | 1580.93 | 2306.39 | 57.30% |

downlink 最大的原因是服务端返回完整 vocab logits。Qwen2.5 vocab 为 151936，float32 下单个 token logits 约 607744 bytes。speculative 的 verify 请求一次会返回多个 draft token 的 logits block，因此单次响应更大；但它的 HTTP 调用次数少很多，平均每 run 10.33 次，而 HTTP target-only 是固定 35 次。

## 7. Speculative 为什么曾经不如 target-only，以及现在为什么改善

早期 JSON logits 结果中，speculative 不如 target-only 的主要原因不是算法本身，而是协议开销：

| 阶段 | JSON localhost speculative | Binary localhost speculative |
|---|---:|---:|
| 服务端 response encode | 约 2970.89 ms | 5.82 ms |
| 客户端 response decode | 约 1182.28 ms | 0.27 ms |
| 总耗时 | 约 6763.84 ms | 1252.06 ms |

换成 binary logits 后，序列化和解析开销基本消失，localhost 端云架构下 speculative 已经快于 HTTP target-only。

但是，speculative 仍然慢于纯本地 target-only：

| 对比 | 倍率 |
|---|---:|
| localhost speculative / 纯本地 target-only | 1.49x slower |
| HTTP target-only localhost / 纯本地 target-only | 1.84x slower |
| HTTP target-only 模拟远端 / 纯本地 target-only | 4.79x slower |

原因是纯本地 target-only 没有端云协议成本，也不需要运行 drafter。当前 target 只有 1.5B，单卡本地 forward 已经很快；在这种场景下，speculative 的 drafter 开销和通信协议开销不一定能被 target 调用减少完全抵消。speculative 更适合“客户端小模型 + 云端大模型 + 网络往返昂贵”的场景。

## 8. 实验结果分析

### 8.1 localhost 结论

localhost 下 upload/downlink 很小，主要时间分布是：

1. `speculative`：drafter 占 55.91%，cloud verify 占 37.36%。
2. HTTP `target_ar`：cloud verify 占 88.47%。
3. speculative 通过减少 target 调用次数，把服务端 target verify 从 1371.68 ms 降到 467.79 ms。
4. binary logits 后协议层开销已经可控，server encode、client decode 都不是主要瓶颈。

### 8.2 模拟远端结论

远端模拟下，通信成为主要瓶颈：

1. `speculative` 通信占 49.32%，其中 downlink 占 42.43%。
2. HTTP `target_ar` 通信占 57.30%，其中 upload 占 18.02%，downlink 占 39.28%。
3. speculative 的优势更明显，因为 HTTP 调用次数从 35 次降到 10.33 次。
4. 即使 speculative 单次 verify 返回更大的 logits block，减少 RTT 次数仍然带来净收益。

### 8.3 纯本地 baseline 结论

纯本地 target-only 是“无网络理想基线”，平均 840.87 ms。它说明：

1. 如果目标是测纯模型推理速度，应该看 `local_target_ar`。
2. 如果目标是测端云协同推理，应该看 `speculative` 和 HTTP `target_ar`。
3. 不能把 HTTP `target_ar` 当作“无网络 target-only”，否则会错误理解 upload/downlink 的来源。

## 9. 局限性

1. 网络模拟是代码级 sleep，不是真实公网链路，不能覆盖 TCP 拥塞、丢包、运营商路径、无线网络抖动等因素。
2. 服务端和客户端仍在同一台物理服务器，GPU 资源、CPU 调度和进程间竞争可能影响绝对耗时。
3. 样本量较小，共 3 个 prompt、9 个 measured runs，适合阶段拆分和趋势分析，但还不足以做严格统计结论。
4. 当前协议返回完整 vocab logits，便于保持算法语义和做精细测量，但不是最省带宽的生产协议。
5. 当前远程 target 服务默认不维护跨请求 KV-cache，因此 target-only 和 speculative 都会重复计算部分上下文。

## 10. 后续优化方向

1. **进一步减少下行数据量**：尝试 `response_dtype=float16`，理论上可把 logits body 从 float32 减半。
2. **服务端执行 verify/acceptance**：让云端只返回 accepted token、必要概率和诊断信息，而不是完整 logits，可显著降低 downlink。
3. **启用 target KV-cache**：服务端保存 session 级 cache，避免每次 `/forward` 重算完整上下文。
4. **调优 gamma**：当前 `gamma=4`，可扫描 gamma=2/4/6/8，寻找“减少 RTT”和“增大 logits block”的平衡点。
5. **真实双机实验**：用一台 client 和一台 GPU server，或在服务器上使用 `tc/netem` 注入真实网络队列延迟。
6. **连接复用和异步流水线**：复用 HTTP keep-alive，减少连接开销；同时尝试让 drafter 生成和网络传输部分重叠。

## 11. 汇报结论

本次三组实验把 target-only 的两种口径拆清楚了：

1. **纯本地 target-only** 没有网络传输，平均 840.87 ms，是单机理想推理基线。
2. **HTTP target-only** 是端云架构 baseline，因此有 upload/downlink；localhost 下平均 1550.38 ms，模拟远端下平均 4024.85 ms。
3. **Speculative decoding** 在 binary logits 后已经快于 HTTP target-only：localhost 下 1.24x，模拟远端下 1.30x。
4. **端云通信是远端场景的核心瓶颈**：模拟远端下 speculative 的 upload+downlink 占 49.32%，HTTP target-only 占 57.30%。
5. 下一步最值得做的是减少 downlink logits 体积、引入 KV-cache、扫描 gamma，并在真实双机或 `tc/netem` 环境下复现实验。

