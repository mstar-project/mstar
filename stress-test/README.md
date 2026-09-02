# stress-test

Reproduction package for the four performance findings reported against
`resource_pools_2`. This branch is cut from `resource_pools_2` at `485beeb7`.
The merge base is `6666c623`. The original numbers were taken at `5eeb434f`.

The findings, measured at `5eeb434f` against the merge base:

1. Chat at c=32 stalls. 185 of 200 requests complete, throughput drops from
   13.49 req/s to 0.62 req/s, and the GPU reads 0% SM for about 280 s.
   `485beeb7` fixes this (200/200 at 13.48 req/s).
2. Chat at c=1 is about 30% slower per token. Median ITL goes from 6.6 ms to
   8.6 ms, with TTFT unchanged. The added time is GPU kernel time: the fused
   gated MLP kernel (`triton_red_fused_add_mm_mul_silu_split`) is 49.5% slower
   per launch on an identical launch count, with a different grid and block
   shape at each commit. With a fresh autotune for each commit the gap
   measures about +14%; the rest follows the tile shape a persistent compile
   cache froze into the reference runs (`expected/results.md`).
3. Chat at c=1 delivers tokens in pairs. About 22% of inter-chunk gaps are
   under 1 ms, each followed by a gap of 12 to 14 ms. The merge base has 3
   sub-millisecond gaps in 7620. `engine.commit` runs ahead of
   `_collect_outputs` and blocks in `cudaLaunchKernel` on a full launch queue.
4. `bagel_cfg_parallel` images are about 2x slower at c=6. Latency p50 goes
   from 12.0 s to 24.7 s and per-GPU SM from 81% to 43%, with identical total
   GPU kernel time. The cause is host-side launch starvation in a graph walk
   that captures no CUDA graphs.

## Setup

Run everything from the repository root. The only environment variable the
scripts need is `HF_HOME`, the model cache. Everything the scripts write goes
under `stress-run/` in the repository root.

```bash
export HF_HOME=/path/to/model/cache

git worktree add stress-run/co-base 6666c623 --detach
git worktree add stress-run/co-head 485beeb7 --detach

bash stress-test/setup_venvs.sh           # build both venvs
bash stress-test/setup_venvs.sh --check   # verify SHAs and torch, installs nothing
```

Check out `5eeb434f` as `co-head` instead to reproduce the original numbers.
`setup_venvs.sh` puts the venvs in `stress-run/venvs/`, installs
torch 2.12.1+cu129, then the checkout as `-e .[all]`. flash-attn is optional:
set `FLASH_ATTN_WHEEL` to a wheel path and it goes in with `--no-deps`,
because a resolving install drags torch to 2.14 and invalidates every number.

## Run

One cell is one fresh server boot, the lines of one spec file run in order
under a single `dmon` trace, then teardown. Run cells one at a time; the
pre-boot gate refuses to start while any compute process is on the box.

```bash
# finding 1, the c=32 stall (one GPU)
bash stress-test/perfcell.sh base bagel_single_gpu 0 8501 stress-run/cells/c32_base cold stress-test/specs/stall_c32.txt
bash stress-test/perfcell.sh head bagel_single_gpu 0 8501 stress-run/cells/c32_head cold stress-test/specs/stall_c32.txt

# findings 2 and 3, chat at c=1 and c=16 (one GPU)
bash stress-test/perfcell.sh base bagel_single_gpu 0 8501 stress-run/cells/chat_base cold stress-test/specs/chat.txt
bash stress-test/perfcell.sh head bagel_single_gpu 0 8501 stress-run/cells/chat_head cold stress-test/specs/chat.txt

# finding 4, cfg images (three GPUs)
bash stress-test/perfcell.sh base bagel_cfg_parallel 0,1,2 8501 stress-run/cells/cfg_base cold stress-test/specs/cfg_images.txt
bash stress-test/perfcell.sh head bagel_cfg_parallel 0,1,2 8501 stress-run/cells/cfg_head cold stress-test/specs/cfg_images.txt

# the analysis tables
python3 stress-test/itl_ratio.py
```

Each spec line writes `<celldir>/<name>.json` with per-request
`chunk_arrivals`, `itl_gaps`, `ttft` and `e2e_latency`; the cell also writes
`boot.txt`, `pinning.txt`, `dmon.txt`, `spans.txt` and `server.excerpt.txt`.
`itl_ratio.py` pairs base and head cells, ignores `warmup_*` lines, and prints
the ITL table, the gap distribution and a stall report. `specs/stall_c64.txt`
is the stall cell at c=64, `specs/cfg_images_full.txt` adds a c=16 row to the
cfg cell, and `specs/boot_only.txt` boots and tears down with no load.

## Expected results

`expected/results.md` holds the full reference tables, including chat c=16,
cfg c=16, utilisation and warm boot times. The runner outputs the tables were
computed from are `expected/base_6666c623_c1_rep{1,2}.json`,
`expected/pr_5eeb434f_c1_rep{1,2}.json`, `expected/pr_5eeb434f_c32.json` and
`expected/head_485beeb7_c32.json`; diff a fresh run against them with
`itl_ratio.py` or on the JSON fields directly.

| finding | metric | `5eeb434f` vs base | `485beeb7` vs base |
|---|---|---|---|
| 1: c=32 stall | completed, req/s | 185/200, 0.62 vs 13.49 | 200/200, 13.48 vs 13.49 |
| 2: c=1 ITL | cell median | 6.6 ms to 8.6 ms | +31% cached, +14% fresh tune |
| 3: pairing | gap share under 1 ms | 0% to ~22% | 0% to 17-19% |
| 4: cfg c=6 | L p50 | 12.0 s to 24.7 s | +39% |

A reproduced stall reads: `n_success` 185 of 200 in the c32 JSON, every
failure at `e2e` near 300 s, and a continuous 0% SM stretch of about 280 s in
`dmon.txt`. For findings 2 to 4, expect direction and rough magnitude, not
exact digits. Client NUMA placement shifts absolute latencies between
allocations; `pinning.txt` records it, and A/B pairs in one allocation hold.

## Notes

- Reference allocation: `salloc --nodes=1 --gpus-per-node=4 --mem=600G` on
  H100 80GB nodes; findings 1 to 3 use one H100, finding 4 uses three, and the
  server is pinned to its NUMA node.
- The scripts default `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to a
  per-commit directory under `stress-run/`. If you override them, use a fresh
  directory per commit, or the comparison is between two frozen autotune
  results.
- Do not measure under `MSTAR_LOG_LEVEL=DEBUG`. Logging makes both commits
  CPU-bound: the c=6 cfg gap reads +9.4% under DEBUG and +105.9% clean.
- The first boot in a fresh venv pays a flashinfer JIT compile of several
  minutes; discard it. `server.excerpt.txt` lists the CUDA graphs captured at
  boot: decode buckets `[1,2,4,8,16]` at `5eeb434f`, `[1,2,4,8,16,32,64]` at
  base and `485beeb7`.
- `chaos.py` drives cancellation load; run it with a venv python. Only its
  `--cancel-at-seconds` cancel fires mid-generation on the image endpoint.
