# CTA-pipelining in M*: handover (state as of 2026-09-30 01:00 UTC)

Read this first, then `README.md` in this directory. This file says where things are, what the
numbers are, what to do next and how to run everything. Details, history and per-fix evidence are
in the README and in the per-batch reports listed at the end.

## 1. What this is

An exploration of CTA-pipelining (Liu et al., arXiv 2607.07862: producer CTAs stream GEMM output
tiles straight into a peer GPU's memory over NVLink, consumer CTAs start as soon as their row block
is signalled) for two H100s, applied to the Wan2.2-TI2V-5B DiT FFN (MLP 3072 -> 14336 -> 3072,
gelu_tanh, bias, bf16). Kernels are CUTLASS CuTe DSL 4.7.1 Hopper warp-specialised persistent GEMMs
(128x256x64 tiles, 4 stages, TMA + WGMMA). Everything lives in three untracked directories of the
`cta-pipelining` git worktree (branch `cta-pipelining`, base 65faaa18), **nothing is committed**:

- `mstar/utils/cta_pipelining/` kernels, wrappers, README, this file
- `benchmark/cta_pipelining/` 2-GPU benchmark, IKET driver and analyser
- `test/cta_pipelining/` 41 tests (30 standalone + 11 TP2-extension), all passing on 2026-09-30

Two designs exist:

1. **Standalone** (`cute_mlp.py`, `CuteCTAPipelinedMLP`): GPU 0 computes fc1 (producer), GPU 1
   computes all of fc2 (consumer) plus a slice of fc1. Closed: it cannot beat TP2 on H100 for this
   shape, see section 3.
2. **TP2 extension** (`cute_tp2.py`, `CuteTP2OverlapMLP`, `ROLE_REDUCE` in `cute_gemm.py`): keep
   TP2 sharding; each GPU's fc2 shard computes the peer-owned row blocks first and TMA-stores the bf16
   partial tiles straight into the peer's output with a per-tile signal, then its own row blocks,
   whose epilogue waits for the peer's partial and TMA reduce-adds its own result on top. The NCCL
   all-reduce disappears. Output is row-sharded by default or replicated (full Y on both GPUs, like
   TP2). This is the live design.

## 2. Where things are

| what | path |
|---|---|
| live worktree | `/shared/home/garv901-55613a/mstar-worktrees/cta-pipelining` |
| final tree snapshot (identical to worktree) | `/data/garv901-55613a/cta-logs/b5_snap_final/` |
| batch-4 tree (standalone only, the "before" of batches 5 and 6) | `/data/garv901-55613a/cta-logs/b5_snap_b4_final/` |
| drain worktree (batch 6, reverted to batch-4 code, keep or delete) | `/shared/home/garv901-55613a/mstar-worktrees/cta-pipelining-drain` |
| split-K tail patch from batch 6 (not applied) | `/data/garv901-55613a/cta-logs/b6_A_best.diff`, tree `b6_snap_A_best/` |
| plans (one per batch, read as specs) | `/data/garv901-55613a/cta-logs/plan_*.md` |
| subagent final reports (verbatim) | `/data/garv901-55613a/cta-logs/report_*.md` |
| logs, snapshots, IKET summaries | `/data/garv901-55613a/cta-logs/` (95 % full disk, delete freely what you regenerate) |
| python | `/shared/home/garv901-55613a/mstar-worktrees/graphapi-testing/.venv/bin/python` (torch 2.12.1+cu130, cutlass-dsl 4.7.1, run-iket) |

## 3. Headline numbers (2x H100 SXM, NVLink, Wan2.2 shape, bf16, ms)

**TP2 extension, back-to-back (production condition), medians of 3 rounds in one job,
`b5_fin.log` / `b5_fin32.log`, percent vs cuBLAS+NCCL TP2:**

| M | tp2 (cuBLAS+NCCL) | tp2_cute (same kernels, memcpy+add, sharded) | overlap sharded | overlap replicated (like-for-like) |
|---|---|---|---|---|
| 4k | 0.607 | 0.557 (-8 %) | 0.538 (-12 %) | 0.562 (-7 %) |
| 8k | 1.223 | 1.142 (-6 %) | 1.159 (-5 %) | 1.200 (-2 %) |
| 12k | 1.880 | 1.796 (-5 %) | 1.775 (-6 %) | 1.826 (-3 %) |
| 16k | 2.538 | 2.412 (-6 %) | 2.417 (-5 %) | 2.547 (0 %) |
| 32k | 4.96 | 4.82 (-3 %) | 4.63 (-7 %) | 4.85 (-2 %) |

Paper shape (K=N=8192, no bias/activation) at 8k: overlap -10 % sharded, -8.5 % replicated vs TP2.

**Per-step timing (host sync before every call) tells the opposite story** (overlap +3..+8 % vs
tp2 at 8k-16k) because the forward is four serial DSL launches of ~0.1 ms host time each. Decide on
back-to-back; ship only with CUDA-graph capture (section 5, item 1).

**Standalone, X replicated, batch 4 final:** eager 0.92 / 1.48 / 2.60 / 5.0 at 4k/8k/16k/32k,
graph 0.80 / 1.42 at 4k/8k; TP2 0.65 / 1.23 / 2.45 / 4.76. 15-20 % behind TP2 and balance-bound
(GPU 1 has more work than GPU 0, no CTA ever waits on rows). Paper shape at 32k is the only cell under
TP2 (7.35 vs 7.47). Closed.

**Reduce kernel anatomy at 8k (IKET, `b5_iket_ovR8k.txt`):** CTA life 508 us (GPU 0) / 495 us
(GPU 1) vs 436 us for a plain fc2 half. Own-row epilogue (TMA reduce-add) 3 us per tile. Peer-row
epilogue: remote TMA store 9.3 us + completion wait + fence + signal 10.7 us = 20 us per tile, 12 % of
the kernel, NVLink burst-bound (128 CTAs finish in lockstep, 8 MB per wave). Mainloop tiles 70-75 us in
both kernels.

## 4. What it means for M* (2026-09-30 reading of `mstar/model/wan22`)

- The Wan2.2 block runs its FFN as a plain single-GPU `MLP`. There is no TP or sequence-parallel path
  for Wan today. The framework has row/column-parallel linears with NCCL all-reduce and Ulysses
  sequence parallel (used by cosmos3).
- Token counts: default request 704x1280x81 frames = 18,480 tokens per sample (latent 44x80x21, patch
  2x2), about 37k with CFG cond+uncond batched; 33-frame 720p or 81-frame 480p clips about 8k; one
  720p frame 880.
- FFN share of block work: about 45 % at 8k, 35 % at 18k, 27 % at 37k (attention is quadratic).
- End-to-end estimates: TP2 with replicated output 0 to -1 % at the default request, about -3 % for
  short clips; TP2 with sequence-parallel residuals (sharded output; plumbing does not exist for Wan)
  -2 % default, up to -6 % short clips; under Ulysses SP the overlap kernel has no role (each GPU runs
  the full-width FFN on half the tokens with no reduction) and only the plain CuTe kernels apply,
  0 to -7 % of FFN time.

## 5. What to do next, in order

1. **CUDA-graph capture of `CuteTP2OverlapMLP.forward`** (precondition for shipping). Removes the
   four serial launches (0.13 ms per step at 8k) and the residual GPU-0 skew (13 us per CTA). Copy the
   capture path from `cute_mlp.py` (`use_graph`, counters zeroed by a memset node after the consumer,
   epoch handling). Half a day. Measure per-step and back-to-back; the per-step column should then match
   back-to-back.
2. **Peer-row store bursts** (20 us per tile, 12 % of the reduce kernel). Options: issue the remote
   TMA store per 64-row sub-tile as the next tile's mainloop starts and wait for completion later (the
   whole-tile deferral "S" measured no gain and +10 % at 4k, so granularity matters), or stagger half the
   CTAs / interleave own-row and peer-row tiles so the link sees a steady stream instead of 8 MB waves.
   Expected 3-5 % of the kernel, at most 10 %. One to two days.
3. **16k replicated per-step anomaly**: +0.2 ms per step that vanishes back-to-back, reproducible
   across jobs. Suspect per-call output allocation (2 x 100 MB) with cross-device `record_stream`.
   Persistent output buffers and cached DLPack conversions fix the host side anyway. Half a day.
4. **Integration path decision** (not kernel work): TP2 with sequence-parallel residuals gives the
   sharded numbers; TP2 replicated gives the like-for-like numbers; Ulysses SP makes the overlap moot.
   Stage 2 in the README describes the two-process integration through torch symmetric memory.
5. **Standalone only if ever needed**: producer raster order G>1 in graph mode (in graph mode the
   producer is the critical path; batch 3 found G=8 neutral only because the consumer bound then;
   plain fc1 gained 10-13 % from raster G). Split-K tail and dynamic claiming are measured dead ends
   (batch 6).

Ideas measured and rejected, do not repeat without a new reason: deferred/lagged signal (fix 2 and
"S"), signal warp, bias hoist, 128x128 / 64x256 tiles, fused local fc1 in the consumer, dynamic tile
claiming, split-K tail, L2 prefetch of the partial, batched partial loads, reduce cluster (1,1),
8 epilogue stages, producer raster G=8 in eager mode. Numbers in the README fix list and the reports.

## 6. Rules learnt the hard way (all cost hours)

- **Never touch the signal counters with torch ops** (`add_`, `zero_`, `fill_`) while a peer kernel may
  still be signalling them: the host read-modify-write loses the peer's atomics and the next forward
  hangs. Bump only the slices no peer signals, and synchronise after any zero-fill.
- **Share one compiled `CuteGemmOp` per GEMM across both devices** (or warm both before launching
  mutually dependent kernels): the DSL loads its module into every device on first use, and that load
  blocks behind a spinning peer kernel, which deadlocks.
- **Launch the waiting kernel after the signalling kernel** in eager mode, and never under run-iket
  the other way round.
- **Peer-written outputs need `record_stream` on the writer's stream** too.
- **NVLink low-power trap**: links drop to low power after ~50 ms idle and the next remote
  store/atomic waits 200-250 us. Never time or trace the first launch after a host pause.
- **32k measurements**: the 700 W board cap (hardware maximum, not raisable) engages in loops longer
  than ~100 iterations in every mode; use 20-50 iterations and read `power.draw.instant`.
- **Compare only within one job**: the shared team1 node moves baselines by 3-6 % between jobs.
- **Measure first, verify winners only** (user rule): no tests/ruff/README per variant, one allclose
  smoke while timing, full verification once for a configuration that clears the bar (3 % at 8k).

IKET-specific traps (kernel-name collision per class, untraced second kernel, per-warp range
records, unaligned cross-GPU timers) are in the README section "IKET in-kernel profile".

## 7. How to run

GPUs only through SLURM on the single 8-GPU node (15-minute allocations; other users' jobs queue):

```
cd /shared/home/garv901-55613a/mstar-worktrees/cta-pipelining
PY=/shared/home/garv901-55613a/mstar-worktrees/graphapi-testing/.venv/bin/python
srun -p team1 --gres=gpu:2 --cpus-per-task=8 --time=00:15:00 bash -c "
  cd $PWD && PYTHONPATH=. $PY -m pytest test/cta_pipelining -q"
srun -p team1 --gres=gpu:2 --cpus-per-task=8 --time=00:15:00 bash -c "
  cd $PWD && PYTHONPATH=. $PY -m benchmark.cta_pipelining.bench_mlp_2gpu \
    --modes tp2,tp2_cute,tp2_overlap --tp2-output sharded,replicated --b2b \
    --x-replicated --iters 50 --tokens 4096,8192,12288,16384"
# standalone: --modes single,single_cutlass,tp2,ctapipe_cutlass [--graph] [--paper]
# in-kernel profile: benchmark/cta_pipelining/iket_ctapipe.py, then iket_analyze --by-tile
```

Interleaved A/B scripts (`b4_ab.sh`, `b5_ab.sh`, `b6_ab.sh`), per-kernel timeline tools
(`b4_timeline.py`, `b5_timeline.py`), nvidia-smi samplers and snapshot scripts are in
`/data/garv901-55613a/cta-logs/`; each batch's plan lists which ones it used.

## 8. Log index (the ones that carry deciding numbers)

| batch | logs |
|---|---|
| fix 1 (first-store stall = NVLink low power) | `report_fix1_first_store_stall.md`, `iket_wan8192*` summaries in the README |
| batch 2 (signal, epoch counters, consumer grid) | `b2_final_194417.log`, `b2_paperab_194824.log`, `report_batch2_producer_fixes.md` |
| batch 3 (raster G, column split, conversion cache) | `b3_job10..15_*.log`, `b3_snap_final*`, `report_batch3_rebalance_tiles.md` |
| batch 4 (cluster multicast, CUDA graph, fused fc1 reverted) | `b4_final.log`, `b4_confirm.log`, `b4_sweep_g.log`, `b4_bytile_after.txt`, `report_batch4_gpu1_side.md` |
| gate for batch 5 | `b5_gate_tp2_ar.py`, `b5_gate_tp2_ar.log` |
| batch 5 (TP2 extension) | `b5_fin.log`, `b5_fin32.log`, `b5_ctx.log` (paper), `b5_ctx2.log` (Wan, all modes), `b5_tune1..3.log`, `b5_tl3.log`, `b5_iket_{ovR8k,ovPS8k,cute8k}.txt`, `b5_smokeR.log`, `b5_hang2.log`, `b5_pytest_final.log`, `b5_pytest_tp2.log`, `report_batch5_tp2_overlap.md` |
| batch 6 (drain, closed) | `b6_ab_Afinal.log`, `b6_ab_AM.log`, `b6_share1.log`, `b6_tl_A.log`, `b6_bytile_*.txt`, `report_batch6_drain.md` |

## 9. Ownership and hygiene

- No commits were made in this exploration; nothing carries AI attribution and nothing should.
- Both worktrees show only the three untracked directories in `git status`; no shared M* code was
  touched anywhere.
- `/data` is at 95 %: `b3_snap_final*`, `b3_orig_tree`, `iket_b3_after`, `iket_b4_after` and the
  `batch*_orig` backups are safe to delete once the tree is committed.
