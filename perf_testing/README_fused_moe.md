# Tuning the fused-MoE grouped GEMM

## What changed

`mstar/utils/fused_moe/kernels.py` used to pick tile sizes from a two-branch
heuristic (`M <= num_experts` or not), never set `num_warps` / `num_stages`,
and `runner.py` computed that config **once** from the gate+up GEMM's shape
and reused it for the down GEMM — a different problem. Three fixes:

1. **Per-GEMM configs.** `get_moe_configs()` returns `(gemm1, gemm2)`. They
   share `BLOCK_SIZE_M` because that is the alignment granularity of
   `moe_align_block_size`, which runs once per dispatch; everything else is
   tuned independently.
2. **An offline config table.** `get_moe_configs()` loads
   `mstar/utils/fused_moe/configs/E=<E>,hidden=<H>,inter=<I>,dtype=<dt>,device=<gpu>.json`
   and falls back to the old heuristic when no table matches.
3. **A tighter launch grid.** The grid used to be sized from
   `len(sorted_token_ids)`, the worst case of one partial block per expert.
   It is now bounded by `cdiv(num_valid, BLOCK_M) + min(E, num_valid)`, which
   depends only on host-known shapes and so is CUDA-graph safe.

## Why the table is offline and not `@triton.autotune`

The fused MoE is called from inside CUDA-graph capture
(`mstar/engine/cuda_graph_runner.py`). Triton's autotuner benchmarks on first
call — it launches un-captured work and allocates — which corrupts the graph.
So tuning happens ahead of time and inference only does a dict lookup, keyed
on the captured batch size.

## Re-tuning for a new GPU or a new model shape

```bash
# Add the shape to SHAPES in perf_testing/moe_bench_common.py if it is new.
python perf_testing/tune_fused_moe.py --shape thinker --quick --save
python perf_testing/tune_fused_moe.py --shape talker  --quick --save
```

`--quick` searches a reduced grid and takes ~20 min per shape on an H100; drop
it for the full space. `--save` **merges** into any existing table, so you can
re-tune a few token counts without losing the rest. Every winner is verified
against a per-expert fp32 reference before it is written.

## Checking the result

```bash
python perf_testing/bench_fused_moe.py --shape thinker
```

A/Bs `legacy` (old grid + old heuristic) against `grid` and `tuned`, and prints
each against the bandwidth roofline.

## Timing methodology

`moe_bench_common.bench()` captures the launches into a CUDA graph and times
replays. This is not a nicety: Triton's Python launch path costs tens of
microseconds of CPU time, which at decode exceeds the kernel itself, so an
eager loop measures the launcher and reports the *same* time for every tile
config. The first attempt at this sweep did exactly that and produced a flat
37 µs for all 288 candidates. Clocks are settled with a wall-clock warmup
before each measurement, and the SM clock and throttle mask are recorded
alongside every number.

## The thing to know before optimising further

At decode this kernel is **HBM-bandwidth bound, not FLOP bound**. Each
dispatch must stream `w1` and `w2` for every touched expert exactly once; at
`M >= 16` that is all 128 experts, or 1.2 GB, which is 361 µs at H100 peak.
Measured time is 81–90 % of that bound across `M = 1..128`, so tile tuning has
almost nothing left to give there — the measured gain is 1.02–1.19×. The
tuning wins are all at prefill (1.23× at M=256 rising to 1.72× at M=4096),
where the kernel finally becomes compute bound.

Further decode gains have to come from moving less weight: fp8/int4 expert
weights, keeping hot experts resident, or batching more tokens per dispatch.
Not from tiles.

## The align op

`moe_align_block_size` JIT-compiles a vendored CUDA op and silently falls back
to a torch implementation if the build fails. Check which one you are on:

```bash
python -c "from mstar.utils.fused_moe.align import _cuda_op_available as a; print(a())"
python perf_testing/bench_moe_align.py     # CUDA op vs fallback, per token count
```

On H100 the fallback costs ~520 µs per MoE layer against 16 µs for the CUDA op,
and it cannot be CUDA-graph captured (it syncs to size a `repeat_interleave`),
so it also takes the fused MoE out of the graph. See
`docs/installation.rst` → "Checking the MoE align kernel actually built" for the
build failures and how `align.py` works around them.

## CuTe DSL comparison

`perf_testing/compare_moe_cute.py` benchmarks CUTLASS's Hopper CuTe DSL
grouped GEMM (a warp-specialised TMA + WGMMA persistent kernel) on the same
per-expert problem sizes. It needs the CUTLASS Python example:

```bash
# from https://github.com/NVIDIA/cutlass, examples/python/CuTeDSL/cute/hopper/kernel/grouped_gemm/
export CUTE_GROUPED_GEMM=/path/to/grouped_gemm.py
export PYTHONPATH=/path/to/cutlass_pkgs:/path/to/cutlass_pkgs/nvidia_cutlass_dsl/dsl_packages
python perf_testing/compare_moe_cute.py --shape thinker --m 1 8 64 512 4096
```

Both sides are tuned. The comparison charges CuTe for the `index_select` that
builds the permuted A, because a contiguous grouped GEMM cannot consume the
unpermuted rows the way mstar's gather kernel does.
