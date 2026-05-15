# Target 下沉到端边的可行性实验脚手架

这组代码回答一个更窄的问题：把同一个 target model 用成熟 serving 框架部署在云端和边缘端，端到端延迟到底差多少。它不先证明边缘一定更快，而是测出 target 下沉的收益边界。

## 核心判断

边缘 target 值得做，当：

```text
云端网络延迟 + 云端排队/服务边界延迟 - 边缘网络延迟
>
边缘 target 计算变慢的额外成本
```

第一阶段先用 vLLM/SGLang 这类 OpenAI-compatible endpoint 做 target-only 黑盒服务比较。这样可以避开自写 serving stack 的干扰，先把“成熟部署方法下 target 下沉是否可行”测清楚。

## 推荐变量

| 变量 | 第一阶段固定/变化 | 原因 |
|---|---|---|
| 模型 | 固定 Qwen3-32B | 保证云端和边缘比较的是同一个大 target，也符合单张 3090 放不下的研究设定 |
| 精度 | 固定 BF16 | 第一轮先分离 placement 影响，第二轮再研究 INT4/INT8 的质量-延迟权衡 |
| serving | vLLM single GPU、vLLM TP8、vLLM TP4PP2、SGLang TP8 | 比较多种成熟端边分布式部署方法 |
| 网络 | edge_lan、metro_edge、cloud_wan、cloud_congested | 覆盖边缘近端、城域边缘、远端云和拥塞云 |
| 并发 | 1、2、4、8 | 先测单请求基础延迟，再测小并发下 queueing/batching 分叉 |
| 指标 | TTFT、E2E、近似 ITL、throughput、p50/p95 | 不只看平均 ITL |

## 启动服务

云端 A100 示例：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/chajiahao/data/hf_models/Qwen3-32B \
  --served-model-name /home/chajiahao/data/hf_models/Qwen3-32B \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
```

边缘 3090/4090 示例：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/chajiahao/data/hf_models/Qwen3-32B \
  --served-model-name /home/chajiahao/data/hf_models/Qwen3-32B \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
```

边缘 vLLM TP4PP2 示例：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/chajiahao/data/hf_models/Qwen3-32B \
  --served-model-name /home/chajiahao/data/hf_models/Qwen3-32B \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
```

边缘 SGLang TP8 示例：

```bash
python -m sglang.launch_server \
  --model-path /home/chajiahao/data/hf_models/Qwen3-32B \
  --tp 8 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000
```

注意：第一轮不要用 INT4/INT8 或更小模型。一旦边缘用了量化或小模型，这组实验就从“同 target placement”变成“质量-延迟 trade-off”，报告里要单独标注。

## 运行 benchmark

先复制示例配置并替换 `A100_SERVER`、`EDGE_SERVER_VLLM_TP8`、`EDGE_SERVER_VLLM_TP4PP2`、`EDGE_SERVER_SGLANG_TP8`：

```bash
cp configs/target_placement_qwen32b_bf16.example.json configs/target_placement_qwen32b_bf16.local.json
```

查看计划，不发送请求：

```bash
python target_placement_benchmark.py \
  --plan configs/target_placement_qwen32b_bf16.local.json \
  --output-dir experiments/target_placement/qwen32b_bf16 \
  --dry-run
```

做不依赖服务端的 smoke test：

```bash
python target_placement_benchmark.py \
  --plan configs/target_placement_qwen32b_bf16.local.json \
  --output-dir experiments/target_placement/qwen32b_bf16_fake \
  --fake
```

正式运行：

```bash
python target_placement_benchmark.py \
  --plan configs/target_placement_qwen32b_bf16.local.json \
  --output-dir experiments/target_placement/qwen32b_bf16
```

只跑某个 placement 或网络：

```bash
python target_placement_benchmark.py \
  --plan configs/target_placement_qwen32b_bf16.local.json \
  --placement edge_3090x8_vllm_tp8_bf16 \
  --network metro_edge \
  --output-dir experiments/target_placement/qwen32b_bf16_edge_metro
```

## 输出文件

| 文件 | 作用 |
|---|---|
| `raw_events.jsonl` | 每次请求的 phase-level timing |
| `run_summary.csv` | 每个 prompt/run 的 TTFT、E2E、近似 ITL、throughput |
| `aggregate_summary.csv` | 按 placement/network/concurrency 聚合的 mean、p50、p95、std |
| `placement_decisions.csv` | 根据配置里的 comparisons 输出云端 vs 边缘对照 |
| `planned_runs.json` | 实际执行的 placement/network 矩阵 |

当前 benchmark 的 TTFT 和 ITL 来自流式响应的 client-visible timing。prefill、decode、NCCL/通信、GPU utilization 和显存需要从 serving 框架自身 metrics、`nvidia-smi`、Nsight Systems 或框架 profiler 补充采集；这部分不要和 client-visible E2E 混为同一类指标。

## Grill-me 问题

**问题 1：第一轮要不要让边缘用量化模型？**

推荐答案：第一轮不要。先用同一个模型和同一种精度比较云端 A100 与边缘 GPU 的 placement 差异。量化模型放在第二轮，因为它会把问题变成“边缘算得更快但质量可能下降”，容易混淆 target 下沉本身的可行性结论。

**问题 2：如果边缘 3090 跑 14B 明显慢，还值不值得继续？**

推荐答案：值得，但研究问题要转为“边缘下沉的临界条件”。如果 `cloud_congested` 下边缘赢，而 `cloud_wan` 下边缘输，就说明云端排队是关键变量；如果所有云端网络条件下边缘都输，下一步应改研究 quantization、tensor parallel、hybrid fallback，而不是硬证明边缘下沉一定有效。

**问题 3：为什么第一阶段先 target-only，不直接上 speculative decoding？**

推荐答案：因为 speculative decoding 的收益依赖 draft 成本和 acceptance rate。target 下沉是否可行应先测 target serving path 本身：云端强算力+长网络+排队，和边缘弱算力+短网络之间是否存在交叉点。这个交叉点找到后，再把 speculative path 接进去才有解释力。
