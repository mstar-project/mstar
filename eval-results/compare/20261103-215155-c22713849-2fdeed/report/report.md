## Base vs head matrix: c22713849 against 2c8a8ecbf

No regressions: 2 rows across 1 models, every one starts, answers its requests, keeps its outputs, and stays within 5% (or within noise).
No A/A noise baseline on NVIDIA-B300-SXM6-AC for qwen3_5_2b: those rows are judged on the passes' spread alone (`python -m benchmark.eval calibrate` on a main-vs-main run of them).

Served through `mstar serve` on one NVIDIA-B300-SXM6-AC per model, base and head back to back on the same GPUs (order alternated per model), 3 measured passes per point, 1 server session(s) per arm. Deltas are signed so that positive means head is better (lower latency, higher throughput).

| Model | Modality | Starts | Correct | Errors, base / head | c=1 latency, base to head | c=max throughput, base to head | Verdict |
|---|---|---|---|---|---|---|---|
| qwen3_5_2b | text | yes | yes | 0 / 0 | 181 ms to 154 ms (+14.7%) [matched length 11/16] | 1762 tok/s to 2113 tok/s (+19.9%) [tokens per second, lengths differ] | same (within noise) |
| qwen3_5_2b | image in | yes | yes | 0 / 0 | 107 ms to 109 ms (-1.9%) [matched length 11/16] | 1537 tok/s to 1611 tok/s (+4.9%) [tokens per second, lengths differ] | same |

<details>
<summary>Per-row numbers (mean +- std over passes), correctness detail, extra metrics</summary>

| Model / case | Correctness | c=1 latency | c=max throughput | Extra (c=1) |
|---|---|---|---|---|
| qwen3_5_2b / text | 6/16 identical, similarity 0.86, length ratio 1.00 | 181 ms +- 10.7% to 154 ms +- 13.4% (+14.7%) [matched length 11/16] | 1762 tok/s +- 12.8% to 2113 tok/s +- 4.3% (+19.9%) [tokens per second, lengths differ] | ttft_p50_s 19 ms to 16 ms (+14.5%) |
| qwen3_5_2b / image_in | 9/16 identical, similarity 1.00, length ratio 1.00 | 107 ms +- 17.6% to 109 ms +- 18.3% (-1.9%) [matched length 11/16] | 1537 tok/s +- 3.3% to 1611 tok/s +- 4.6% (+4.9%) [tokens per second, lengths differ] | ttft_p50_s 24 ms to 22 ms (+6.3%) |

</details>

<details>
<summary>Startup, peak VRAM, errors</summary>

| Model | Startup s, base / head | Peak VRAM GB, base / head | Request errors, base / head |
|---|---|---|---|
| qwen3_5_2b | 455 / 156 | 21 / 22 | 0 / 0 |

Startup shares warm compile caches between the arms, so the first arm can pay compiles the second reuses; it is reported, not judged.

</details>

<details>
<summary>Setup</summary>

- base: 2c8a8ecbf (`2c8a8ecbf`), head: c22713849 (`c22713849`)
- GPU: NVIDIA-B300-SXM6-AC; one model per GPU at a time, 1 GPU on 1 host; each arm imports its own checkout (and its own `rust/` build) in one Python environment
- Command: `python -m benchmark.eval compare --base 2c8a8ecbf --head c22713849 --models qwen3_5_2b --no-rerun`
- Checks: greedy text identity (or close similarity with no truncation) at c=1, Whisper-large-v3 WER for speech, PSNR against base for seeded images and video, max action difference for VLA, duplex take-turn and yield rates, status codes for the request path
- Regression: worse than 5%, more than 2 combined standard deviations of the passes, and more than 1.5 times the largest drift the same row showed between two processes of identical code in its last 8 A/A runs, a lone outlier run left out (A/A calibration); better by the same margin is reported as faster. Where base and head returned different output lengths, c=1 latency uses only the requests whose lengths matched (else latency per token or audio second) and throughput counts output tokens or audio seconds per second
- Correctness limits: the manifest's, widened to 1.5 times the largest swing the row showed between two processes of identical code; never tighter than the manifest
- GPU time: 0.2 GPU-hours

</details>
