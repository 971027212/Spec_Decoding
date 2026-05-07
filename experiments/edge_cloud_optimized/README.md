# Optimized Edge-cloud Benchmark Results

This directory is reserved for the next experiment round after the server-side greedy acceptance optimization.

Keep the previous result set in `experiments/edge_cloud` for comparison. Write new optimized runs here:

- `local_target`: pure local target-only baseline, no HTTP.
- `local_service`: localhost HTTP service with `cloud_target_generate` and `speculative_server_accept`.
- `cloud_sim`: same HTTP service plus simulated RTT and bandwidth.

The optimized speculative mode is `speculative_server_accept`. It sends draft blocks to `/verify_greedy`; the target service verifies accepted greedy drafts server-side and returns only compact acceptance metadata, so downlink no longer contains full-vocabulary logits.
