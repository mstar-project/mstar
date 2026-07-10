# Ming-flash-omni-2.0 T2T scaling sweep — NATIVE M* (mstar), 4×H100 80GB

Native Walk-graph path (`mstar/model/ming_omni_flash/`), thinker-only TP=4,
benchmarked with `--inference-system ours` (native `/generate`). Same grid +
prompts (`benchmark/assets/simple_text_queries.txt`) as the vllm-omni baseline
in [`../ming_t2t_sweep/SUMMARY.md`](../ming_t2t_sweep/SUMMARY.md) for a 1:1
comparison. Dated 2026-06-16.

Stack under test: decode CUDA graphs (`thinker_decode` capture) + request
batching (`forward_batched` / `can_batch`) + the worker↔conductor non-blocking
comm fix. tok/s computed with Ming's own tokenizer (matches the baseline's
basis). All 470 requests succeeded; 0 failures.

| mode | concurrency | reqs | wall (s) | E2E p50 (ms) | E2E p95 (ms) | req/s | tok/s |
|------|-------------|------|----------|--------------|--------------|-------|-------|
| OFFLINE     |           1 |   50 |   47.7   |         574  |        2287  |  1.05 |  104.5 |
| CLOSED_LOOP |           2 |   80 |   45.1   |         760  |        2919  |  1.77 |  164.2 |
| CLOSED_LOOP |           4 |   80 |   32.7   |        1117  |        3908  |  2.44 |  257.3 |
| CLOSED_LOOP |           8 |   80 |   25.7   |        1415  |        5705  |  3.12 |  350.2 |
| CLOSED_LOOP |          16 |   80 |   22.5   |        2782  |        8944  |  3.55 |  410.4 |
| CLOSED_LOOP |          32 |   80 |   16.5   |        4461  |       14242  |  4.85 |  499.3 |

## vs vLLM-omni (same grid)

| concurrency | native req/s | vllm-omni req/s | winner |
|-------------|--------------|-----------------|--------|
|           1 |     **1.05** |            0.72 | native |
|           2 |     **1.77** |            1.30 | native |
|           4 |     **2.44** |            2.36 | native |
|           8 |         3.12 |        **3.71** | vllm   |
|          16 |         3.55 |        **5.81** | vllm   |
|          32 |         4.85 |        **6.96** | vllm   |

## Observations

- **Native beats vLLM-omni through c=4** and is competitive at c=8. At low
  concurrency the decode CUDA graphs give native the edge (offline p50 574 ms
  vs vLLM's 1444 ms; 1.05 vs 0.72 req/s).
- **Single-stream ~104 tok/s** — on par with vLLM's ~110, thanks to the decode
  graph capture (was ~16 tok/s eager before graphs).
- **Throughput scales 1.05 → 4.85 req/s (4.6×)** c=1→32 via request batching
  (the decode batch fills to bs=32 under load). Before batching it was flat at
  ~0.15 req/s.
- **High-concurrency gap (c≥8) remains** vs vLLM. Profiling attributed this to a
  balanced decode step (MoE GEMM ~56%, TP all-reduce ~16%, rest) with no single
  crushable bottleneck on 4-GPU. An expert-parallel MoE was prototyped to close
  it but did not pay off at TP=4 single-node (all-to-all overhead exceeded the
  all_reduce it replaced); not merged. Closing the gap further would need a
  different attack (more ranks, FP8/quantized MoE).
- **Tail latency** grows with batch (p95 2.3 → 14.2 s, c=1→32) — request-mix
  variance under batching, expected.

## Reproduce

Server: `MING …` via `scripts/ming_native/serve_ming_tp4.sh` (TP=4 thinker-only,
`configs/ming_flash_omni_thinker_only_tp4.yaml`). Sweep:
`scripts/ming_native/run_sweep.sh` (offline B=1 + closed-loop c=2/4/8/16/32),
then `scripts/ming_native/summarize.py` for the table. Per-run `results.json` +
`req_*.txt` live alongside this README.
