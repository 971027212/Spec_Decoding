# Binary Logits Experiments

This folder groups the benchmark artifacts used for the binary logits timing comparison.

| Folder | Meaning |
|---|---|
| `localhost/` | Binary logits with client and target service on localhost, no simulated network delay. |
| `cloud_sim/` | Binary logits with code-level remote network simulation: 40 ms RTT, 100 Mbps uplink, 200 Mbps downlink. |
| `local_target_ar/` | Pure local target-only baseline with no HTTP service and no upload/downlink. |

Each result folder contains:

- `raw_events.jsonl`
- `run_summary.csv`
- `aggregate_summary.csv`
- `phase_stacked.png`
- `phase_boxplot.png`
