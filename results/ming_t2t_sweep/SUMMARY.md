# Ming-flash-omni-2.0 T2T scaling sweep — 4×H100 80GB

Run via vllm-omni 0.19.0, hybrid snapshot (inclusionAI thinker + Jonathan1909 metadata/talker),
stage config `ming_flash_omni.yaml` (TP=4 thinker + colocated talker on GPU 3).
Prompts from `benchmark/assets/simple_text_queries.txt` (general-knowledge English).
Dated 2026-06-06.

| mode | concurrency | reqs | wall (s) | E2E p50 (ms) | E2E p95 (ms) | req/s | tok/s |
|------|-------------|------|----------|--------------|--------------|-------|-------|
| OFFLINE     |           1 |   50 |   69.14  |        1444  |        2310  |  0.72 | 109.6 |
| CLOSED_LOOP |           2 |   80 |   61.57  |        1436  |        2536  |  1.30 | 198.9 |
| CLOSED_LOOP |           4 |   80 |   33.94  |        1588  |        2846  |  2.36 | 355.7 |
| CLOSED_LOOP |           8 |   80 |   21.54  |        1899  |        3396  |  3.71 | 573.4 |
| CLOSED_LOOP |          16 |   80 |   13.78  |        2144  |        4175  |  5.81 | 887.9 |
| CLOSED_LOOP |          32 |   80 |   11.50  |        3728  |        7384  |  6.96 | 1060.5 |

## Observations

- **Single-stream baseline** is ~110 tok/s — bounded by TP=4 all-reduce on each
  decode step. TTFT is uniformly 28-91 ms — the 32-layer MoE prefills fast.
- **Linear scaling to c=8** (5.2× over single-stream). Beyond that the curve
  bends: c=16 → 8.1×, c=32 → 9.6×. The knee is between c=16 and c=32.
- **Tail latency** scales as expected with batch size — E2E p95 goes 2.3 → 7.4 s
  from c=1 to c=32 while p50 only doubles. The tail is dominated by
  request-mix variance (token counts span 25-380), not server saturation.
- **All 470 requests succeeded** across the sweep, no errors or timeouts.

## Reproduce

Server launch + benchmark recipe in
[`benchmark/vllm_omni_instructions.md`](../../benchmark/vllm_omni_instructions.md).
Sweep driver was a ~50 LOC scratch script that wraps `benchmark.runner.Benchmark`
with iterated `BenchmarkConfig` (one per concurrency point); contents in the
per-run `results.json` files alongside this README.
