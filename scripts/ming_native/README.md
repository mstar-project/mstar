# Native M* Ming-flash-omni-2.0 T2T benchmark

Fills in the **native M\* (mstar) Walk-graph** column for Ming-flash-omni-2.0,
to sit alongside the existing vllm-omni 4×H100 sweep in
[`../../results/ming_t2t_sweep/SUMMARY.md`](../../results/ming_t2t_sweep/SUMMARY.md).

This is the *native* path — `MingFlashOmniModel` from
`mstar/model/ming_omni_flash/`, served by `mstar.api_server` and benchmarked
with `--inference-system ours` (native multipart `/generate`). It does **not**
proxy through vllm-omni.

## Scope

Text-only **thinker** path (T2T). Steps 1–5 of the port (config, tokenizer,
Ling-2.0 MoE thinker, encoders, thinker graph walks) are DONE and verified via
TP=8 smoke. The talker (step 6e) and image-gen (step 9) are not needed for T2T.

## Layout: TP=4, true 4 GPUs (GPUs 4–7)

Chosen to be apples-to-apples with the vllm-omni 4-GPU column.

> **Heads-up:** `configs/ming_flash_omni_thinker_only_tp4.yaml` and PORTING_NOTES
> document TP=4 OOMing at ~78.5/80 GB per rank (re-verified 2026-06-08). If load
> OOMs, *that is the finding* for the native 4-GPU column. For a guaranteed
> throughput number, re-run at TP=8 (see fallback below).

## Run

```bash
# 1. Launch the native server (background). Pinned to GPUs 4-7.
./scripts/ming_native/serve_ming_tp4.sh &
#    Weights load takes a few minutes (42 shards). Watch:
#    tail -f scripts/ming_native/server_tp4.log

# 2. Run the sweep (polls /health, then runs all 6 points + summarizes).
./scripts/ming_native/run_sweep.sh
```

Results land in `results/ming_t2t_sweep_native/` — one dir per point with
`results.json` + `req_*.txt`, plus a rolled-up `SUMMARY.md`.

## TP=8 fallback (known-good, but 8-GPU not 4-GPU)

```bash
CONFIG=configs/ming_flash_omni_thinker_only.yaml GPUS=0,1,2,3,4,5,6,7 \
  ./scripts/ming_native/serve_ming_tp4.sh &
OUT=results/ming_t2t_sweep_native_tp8 ./scripts/ming_native/run_sweep.sh
```

## Grid (matches the vllm-omni sweep 1:1)

| point | profiling | reqs | concurrency |
|-------|-----------|------|-------------|
| offline_b1 | offline | 50 | 1 |
| closed_loop_c2 | closed_loop | 80 | 2 |
| closed_loop_c4 | closed_loop | 80 | 4 |
| closed_loop_c8 | closed_loop | 80 | 8 |
| closed_loop_c16 | closed_loop | 80 | 16 |
| closed_loop_c32 | closed_loop | 80 | 32 |

Prompts: `benchmark/assets/simple_text_queries.txt` (same as the vllm-omni run).
tok/s is recomputed by tokenizing the saved outputs with Ming's tokenizer, the
same basis as the vllm-omni SUMMARY.

## Prereqs (already satisfied in this container)

- 8× H100 80GB; GPUs 4–7 free.
- inclusionAI/Ming-flash-omni-2.0 weights cached (42 shards, `~/.cache/huggingface`).
- Ming source repo at `/tmp/ming_repo` (auto-discovered; `MING_CODE_DIR` set by the launch script).
- `mminf` venv at `/root/venvs/mminf`.
