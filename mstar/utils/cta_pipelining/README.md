# CTA-pipelining for a two-layer MLP on 2 GPUs

Implementation notes for *CTA-Pipelining: A Latency-Oriented Spatial Scaling Method for
Multi-GPU Systems* (Liu, Andoorveedu, Das, Patel, Kindratenko; arXiv:2607.07862) in M*.

## The idea in one paragraph

Tensor parallelism (TP) shards every GEMM of an MLP across GPUs and pays an all-reduce per
layer. CTA-pipelining instead gives each GPU one *whole* GEMM and runs the dependent kernels
**at the same time**: GPU 0 runs `H = act(X W1ᵀ)`, GPU 1 runs `Y = H W2ᵀ`. Each producer CTA
writes its finished `H` tile directly into GPU 1's memory over NVLink, issues a system-scope
fence, and decrements/increments a per-consumer-CTA scoreboard; consumer CTAs poll a local
queue/counter and start as soon as the row of `H` tiles they need has landed. No all-reduce,
no static micro-batch chunking, kernels stay unmodified except for a prologue/epilogue. On 2
B200s and 16384×8192×8192 the paper reports 31.8% lower latency than the best static
micro-batch schedule and 29% lower than TP2 (Fig. 5, Table I). The gain vanishes when a GEMM
is a single CTA wave (≈1024 rows), so it targets long prefills and diffusion transformers, not
decode.

## Which model / structure in M*

**Recommendation: the Wan2.2-TI2V-5B DiT FFN** (`mstar/model/wan22/components/dit.py`,
`Wan22DiTBlock.ffn`, a shared `components.mlp.MLP(3072 → 14336 → 3072, gelu_tanh, bias)`).

Why it is the easiest and most faithful target:

| criterion | Wan2.2 DiT | Cosmos3 DiT | Orpheus / BAGEL LLM | Qwen3-Omni |
|---|---|---|---|---|
| MLP shape | plain 2-GEMM `fc1 → gelu → fc2` = exactly the paper's `XAB` | 3-GEMM SwiGLU, MoT (two MLP sets, und/gen paths) | SwiGLU 3-GEMM | MoE, not applicable |
| rows per forward | 18,480 tokens/sample at 704×1280×81; ×2 with CFG = 36,960 | ≤ 8,192 | prefill: hundreds (TTS) / ~5k (BAGEL image) | — |
| FFN share of block FLOPs | 45% at 832×480×81 (8,190 tok), 34% at 704×1280×81 | ~55% at 8k tok | high, but rows too few or model too complex | — |
| existing parallel plumbing to disturb | none (plain `nn.Linear`) | TP + Ulysses SP, 1,278-line transformer | TP | TP/MoE |
| state | stateless (no KV cache, no loop) | stateless | KV cache + decode loop | KV cache |
| baseline configs | single GPU only | tp2 / sp2 / tp2+sp2 | tp2 | tp2 |

Wan2.2 is a 50-step, single-request, latency-bound workload with tens of thousands of rows
per GEMM; its FFN is the paper's exact "2-layer GEMM with an elementwise op in between".
Cosmos3 is the runner-up (bigger FFN share, TP2/SP2 baselines already exist) but its MoT
transformer, three-GEMM gated MLP and packed und/gen attention make the integration
several times larger for the same experiment. LLM prefill/decode paths (Orpheus, BAGEL,
Qwen3-Omni) either have too few rows per step or are MoE.

Honest caveat for the end-to-end number: with only 2 GPUs, CTA-pipelining as implemented
here parallelizes the FFN only, while Ulysses SP2 / TP2 also halve attention. At Wan2.2's
default resolution attention is ~45% of the block, so a 2-GPU FFN-only split has an ideal
end-to-end ceiling of roughly 17–22% and will likely trail SP2 end to end. The paper itself
positions CTA-pipelining as orthogonal to TP (their 4- and 8-GPU "TP × CTA" configs). The
2-GPU claim that *is* testable and is what `benchmark/cta_pipelining` measures is the FFN
itself: CTA-pipe vs best micro-batching vs TP2 on Wan2.2's fc1/fc2 shapes.

## What is implemented (stage 1, standalone)

```
mstar/utils/cta_pipelining/kernels.py   Triton producer + consumer persistent GEMMs (protocol in prologue/epilogue)
mstar/utils/cta_pipelining/mlp.py       CTAPipelinedMLP: single-process two-device driver, event-ordered, no host sync
mstar/utils/cta_pipelining/cute_gemm.py CUTLASS CuTe DSL sm_90a warp-specialised persistent WGMMA GEMM with the
                                        producer / consumer hooks (the paper's kernel class), fused bias + activation
mstar/utils/cta_pipelining/cute_mlp.py  CuteCTAPipelinedMLP (same driver on the CuTe kernels; GPU 1 also computes a
                                        column share of fc1 itself; opt-in CUDA-graph forward, use_graph=True)
                                        + CutePlainMLP (1-GPU baseline)
benchmark/cta_pipelining/bench_mlp_2gpu.py  Fig. 5 replication: single / single_cutlass / micro-batch sweep / TP2 / ctapipe / ctapipe_cutlass
benchmark/cta_pipelining/iket_ctapipe.py    run-iket driver (kernels compiled with IKET ranges via CTA_PIPE_IKET=1)
benchmark/cta_pipelining/iket_analyze.py    per-range / per-CTA summary of an IKET trace.json (--by-tile: per tile index)
test/cta_pipelining/test_cta_pipelined_mlp.py       Triton: CPU interpreter protocol test + two-GPU vs cuBLAS tests
test/cta_pipelining/test_cute_cta_pipelined_mlp.py  CuTe: plain GEMM vs cuBLAS (bias/act/M residue) + two-GPU tests
```

### CUTLASS (CuTe DSL) kernels

`cute_gemm.py` is a port of CUTLASS 4.7's
`examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py` (TMA loads, 1 DMA
warp group + 2 MMA warp groups, 128×256×64 tiles, 4-stage smem pipeline, TMA store epilogue) with
the CTA-pipelining protocol inserted where the paper puts it (Sec. III-B.2):

| paper | `cute_gemm.py` |
|---|---|
| tile order that finishes rows early | linear tile id → `(m, n)`, static per CTA (`t = bid + i·grid`): consumer row-block-major (`divmod(t, num_n_tiles)`); plain / local fc1 grouped by `raster_group` G = 32 row blocks, n-major inside a group, so a wave re-reads W1 from L2 (fix 6a/9); the producer stays row-block-major (G = 1), since grouping makes its row blocks complete a group at a time. Producer grid `min(tiles, SMs)`; consumer grid = fewest CTAs with the same number of waves (768 tiles → 128 CTAs × 6 instead of 132 × 5–6, fix 5) |
| producer epilogue: store tile into consumer memory, system fence, signal | TMA store to the peer-mapped `H`; store warp: `cp.async.bulk.wait_group 0` → `fence.proxy.async` → `bar.warp.sync` → `fence.acq_rel.sys` → one `red.relaxed.sys.global.add.s32` on the consumer-side row counter (fence + relaxed `red` is the PTX release pattern; `red.release` would add a second system-scope fence, fix 2) |
| consumer fc2: 2-CTA cluster (batch 4, item 1) | `cluster_shape_mn=(1, 2)` for the consumer when `N2/256` is even (else (1, 1)): the two CTAs of a cluster take n-tiles `2p, 2p+1` of one row block and TMA-multicast the H tile (A); each loads half of it into both CTAs' smem, and each MMA warp releases a stage to both CTAs (`consumer_arrive_cnt` × 2). The static schedule walks n-tile pairs, the grid is whole clusters capped by `get_max_active_clusters(2)`, and the output is bit-identical to (1, 1) (tested). Producer and plain / local fc1 stay at (1, 1) |
| consumer prologue: poll queue before the tile's loads | DMA warp: one lane polls `ld.acquire.sys.global.s32` (256 ns `nanosleep` between polls) until `counter[m] >= target` (counters are never zeroed; `target` = sum of `N1p/256` over the forwards since the buffer was allocated, fix 10), `bar.warp.sync`, `fence.proxy.async`, then the tile's TMA loads; MMA warps need no change (they block on the smem mbarrier) |
| unmodified mainloop | unchanged WGMMA mainloop; bias added in accumulator order, gelu-tanh/silu on the fp32 fragment before the bf16 convert |

**Column split of GEMM 1 (fix 3).** The producer computes only columns `[0, N1p)` of `H` (a column
slice of the `[M, N1]` buffer, row stride `N1`) and GPU 1 computes `[N1p, N1)` itself with the same
kernel in plain role (bias + gelu, no counters) on `c_stream`, just before the consumer. The
consumer then waits for `N1p/256` producer tiles per row block. `N1p = round(f·N1/256)·256` is fixed
per MLP. `producer_share=f` sets it; the default is `DEFAULT_PRODUCER_SHARE`, keyed by shape
`(K, N1, N2)`: 48 of 56 n-tiles for the Wan2.2 FFN, unsplit (f = 1) for the paper's 8192² shape and
any unlisted shape. W1 rows are placed once like TP2 shards: `W1[:N1p]` on GPU 0 and `W1[N1p:]` on
GPU 1. **X must be on both GPUs:** pass
`forward(x, x_consumer)` when the caller already holds X on GPU 1 (the Stage-2 case: both workers
hold the block input, as TP2 assumes); otherwise the forward copies X to GPU 1 over NVLink
(`2·M·K` bytes, ≈ 0.6 ms at 32k rows) and the local share waits for that copy. Host order per
forward: X copy → producer launch → local fc1 launch → consumer launch. Batch 4 also built the
local fc1 *fused into the consumer kernel* (local tiles first in its schedule, each bumping the
row counter). It measured neutral in eager and graph mode and was reverted (fix list, batch 4
item 2).

**CUDA-graph forward (batch 4, item 3; opt-in, `use_graph=True`, default off).** The DSL launch
is capturable: its host function only encodes the TMA descriptors on the host and calls
`cudaLaunchKernelEx`, with no synchronization. The graph contract:
* One graph per `(M, dtype, x_consumer is None)`, captured on first use after 3 eager warm-up
  forwards (compile, module load; they may grow `_h`, which drops all graphs). `forward` copies
  `x` (and `x_consumer`) into the graph's static input buffers, replays on the caller's stream
  and returns the graph's **static output tensor, which the next call with the same `M`
  overwrites** (torch's CUDA-graph contract; clone it to keep it).
* The capture spans both GPUs: the producer stream is the capture stream, GPU 1's stream joins
  through event waits, and the X copy (when `x_consumer` is not given) runs on its own
  producer-side stream so the producer does not wait for it.
* Counters are zeroed inside the graph, after the consumer (one memset node), instead of epoch
  targets: every replay starts from zero and waits for `N1p/256` tiles per row block. A replay
  sets the eager epoch to the int32-guard value, so the next eager forward re-zeroes (sync, zero,
  epoch 0). Zeroing before the producer instead measured 15–18 µs slower.
* Replays are ordered after the previous replay and the previous eager forward (they share `_h`
  and the static buffers).
* Default share in graph mode: `DEFAULT_PRODUCER_SHARE_GRAPH`, picked once by `max_tokens`
  (40/56 up to 4096, 44/56 up to 8192, 48/56 above for the Wan shape): GPU 1 starts ≈ 0.2 ms
  earlier without the serial host launches and can take a larger share at small M. The W1 split
  is fixed at construction, so a later, larger `M` keeps that share. An explicit
  `producer_share=` overrides it, and without `max_tokens` the eager default applies.

The DSL was chosen over a C++ CUTLASS extension because the box only has nvcc 12.9 against a
cu130 torch (the extension build fails the version check) and because IKET only instruments DSL
kernels. `CTA_PIPE_IKET=1` compiles `wait_row`, `tma_tile`, `mma_tile`, `epi_tile`, `signal`
ranges into the kernels for `run-iket`.

### Measured (2× H100 80GB NVLink, driver 595.71, torch 2.12.1+cu130, CUTLASS DSL 4.7.1, 2026-09-29)

Median of 10 after 3 warm-ups, ms. `single` = cuBLAS on one GPU, `1GPU-CuTe` = the same CuTe kernel
in plain role on one GPU, `MB` = best static micro-batch over 2 GPUs (chunk in brackets), `TP2` =
NCCL tensor parallel over 2 processes. `CTA-pipe CuTe` is `forward(x)`: X is copied to GPU 1 inside
the timed forward for GPU 1's share of fc1. `CuTe X on both` is `forward(x, x_consumer)` with X
already on GPU 1, as TP2 assumes (its X is replicated on both ranks). All variants match cuBLAS to
1 bf16 ulp. Every timed call starts after a device sync, so the CuTe columns include the host cost
of the DSL launches (≈ 60–70 µs for the first one, fix 11) while a cuBLAS launch costs a few µs.

Wan2.2 FFN shapes (K=3072, N=14336, gelu-tanh, bias), **after batch 4** (item 1: fc2 cluster
(1, 2); item 3: opt-in CUDA graph). Medians of 3 interleaved rounds against the batch-3 code on one
allocation, `--modes single,single_cutlass,tp2,ctapipe_cutlass --iters 10 --x-replicated`, plus
`--graph` on the after tree. Batch-3 values of the same job are in brackets. All CTA-pipe columns
here use the 48/56 share, the graph columns included (the graph-mode share table came after this
run; next table).

| M | single | 1GPU-CuTe | TP2 | CTA-pipe CuTe, X copied | X on both | graph, X copied | graph, X on both |
|---|---|---|---|---|---|---|---|
| 4096 | 0.992 (0.994) | 1.049 (1.047) | 0.648 (0.657) | 0.972 (0.953, +2.0 %) | 0.923 (0.924, −0.1 %) | 0.853 | 0.841 (−9.0 %) |
| 8192 | 1.993 (1.997) | 2.018 (2.025) | 1.232 (1.241) | 1.502 (1.530, −1.8 %) | 1.485 (1.489, −0.3 %) | 1.464 | 1.475 (−0.9 %) |
| 16384 | 3.975 (4.030) | 3.986 (4.146) | 2.445 (2.451) | 2.687 (2.770, −3.0 %) | 2.601 (2.648, −1.8 %) | 2.759 | 2.584 (−2.4 %) |
| 32768 | 8.597 (8.703) | 8.428 (8.150) | 4.762 (4.766) | 5.381 (5.595, −3.8 %) | 4.992 (5.488, −9.0 %) | 5.640 | 5.317 (−3.1 %) |

The graph percentages compare against batch-3 eager with X on both. The unchanged 1GPU-CuTe kernel
moved by −3.9 % / +3.4 % at 16k / 32k between the two trees, so ≈ 4 % is the noise floor there.
The 32k X-on-both cells spread over 5.13–5.50 ms (before) and 4.92–5.30 ms (after). They are
power-capped. In a 200-forward 32k run with `nvidia-smi` sampling (`b4_job2`), both GPUs reached
the 700 W board limit (peaks 699 / 706 W, which is also the maximum settable limit) and reported
the SW power-cap throttle reason (0x4). Capped samples on GPU 1 ran at 1005–1980 MHz, median
1425 MHz; GPU 0's minimum was 1590 MHz. The per-forward total drifts from 5.1–5.3 ms (first
forwards) to 5.5–5.6 ms, with bursts of 6–9 ms. That is the "unresolved
32k cell" of batch 3: power, not code.

Graph mode with the shipped graph-mode share (`DEFAULT_PRODUCER_SHARE_GRAPH`, picked by
`max_tokens`) against the same graph at an explicit 48/56. Medians of 3 interleaved rounds,
`--modes ctapipe_cutlass --iters 20 --x-replicated --graph`:

| M | graph default share | X copied | X on both | explicit 48/56, X copied | X on both |
|---|---|---|---|---|---|
| 4096 | 40/56 | 0.796 (−6.5 %) | 0.801 (−4.8 %) | 0.851 | 0.841 |
| 8192 | 44/56 | 1.482 (+2.8 %) | 1.424 (−2.7 %) | 1.441 | 1.463 |

At 8k the smaller share helps only with X on both: −7.6 % in the share sweep (1.371 vs 1.483 ms
medians) and 1.375 vs 1.480 ms in the confirmation timeline. With the copy, GPU 1's larger share waits for X. The share cannot
follow `x_consumer`, because the W1 split is fixed at construction. At ≥ 16k graph mode keeps 48/56,
where it is within noise of eager with X on both and 2–5 % slower with the copy. The static-input
copy before each replay costs ≈ 35 µs at 8k and ≈ 140 µs at 32k, and with the copy the NVLink X
copy runs inside the graph alongside the producer. In eager mode every share below 48/56 was
slower at ≥ 8k (batch-4 sweeps), so the eager default stays 48/56.

Wan2.2 FFN shapes (K=3072, N=14336, gelu-tanh, bias), after batch 3 (fixes 3, 6a/9 and 11). The
numbers are medians of 3 interleaved before/after rounds on one allocation, `--modes
single,single_cutlass,tp2,ctapipe_cutlass --iters 10 --x-replicated`. The CTA-pipe CuTe columns in
this A/B are the interim defaults: producer G = 8, share 44/56 at M ≤ 12288 and 48/56 above,
chosen per M. The shipped defaults (producer G = 1, share 48/56 at every M) were measured
against that interim config afterwards (next table). The other columns do not depend on it. The columns marked * (MB
best, CTA-pipe Triton) are from an earlier run on the same day and were not re-run, since the
Triton path did not change. R-MB uses that MB value, and R-X = (X − CTA) / X, so positive means
CTA-pipe is faster. The R columns give `forward(x)` / X on both.

| M | single | 1GPU-CuTe | MB best | TP2 | CTA-pipe Triton | CTA-pipe CuTe | CuTe X on both | CuTe R-MB | CuTe R-TP |
|---|---|---|---|---|---|---|---|---|---|
| 4096 | 0.980 | 1.033 | 1.152 (1024)* | 0.666 | 1.513* | 0.992 | 0.945 | +13.9 / +18.0% | −48.9 / −41.9% |
| 8192 | 1.973 | 2.055 | 2.105 (2048)* | 1.238 | 2.677* | 1.517 | 1.466 | +27.9 / +30.4% | −22.5 / −18.4% |
| 16384 | 3.975 | 3.973 | 3.878 (2048)* | 2.433 | 4.850* | 2.764 | 2.654 | +28.7 / +31.6% | −13.6 / −9.1% |
| 32768 | 8.770 | 8.262 | 7.392 (2048)* | 4.705 | 9.162* | 5.357 | 4.999 | +27.5 / +32.4% | −13.9 / −6.2% |

The same rounds on the batch-2 code gave:

| | M = 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|
| 1GPU-CuTe | 1.155 | 2.182 | 4.306 | 8.855 |
| after batch 3 | 1.033 (−10.6 %) | 2.055 (−5.8 %) | 3.973 (−7.7 %) | 8.262 (−6.7 %) |
| CTA-pipe CuTe | 1.020 | 1.717 | 2.969 | 6.020 |
| after, X copied | 0.992 (−2.7 %) | 1.517 (−11.6 %) | 2.764 (−6.9 %) | 5.357 (−11.0 %) |
| after, X on both | 0.945 (−7.4 %) | 1.466 (−14.6 %) | 2.654 (−10.6 %) | 4.999 (−17.0 %) |

Shipped defaults vs the interim config, CTA-pipe CuTe only, medians of 4 interleaved rounds in a
later job (`--iters 20`; this job ran 1–8 % slower than the A/B above for the same interim config):

| M | interim, X copied | shipped, X copied | interim, X on both | shipped, X on both |
|---|---|---|---|---|
| 4096 | 1.014 (44/56) | 0.946 (−6.7 %) | 0.962 | 0.909 (−5.5 %) |
| 8192 | 1.530 (44/56) | 1.494 (−2.4 %) | 1.507 | 1.458 (−3.3 %) |
| 12288 | 2.178 (44/56) | 2.131 (−2.2 %) | 2.213 | 2.093 (−5.4 %) |
| 16384 | 2.845 (48/56) | 2.869 (+0.8 %) | 2.750 | 2.756 (+0.2 %) |
| 32768 | 5.671 (48/56) | 5.516 (−2.7 %) | 5.421 | 6.280 (+15.8 %) |

The 32k X-on-both cell is unresolved. The four shipped rounds gave 5.91–6.77 ms and the interim
ones 5.38–5.91 ms. A CUDA-event timeline of the same two configs (median of 10 forwards) gave
5.004 vs 4.978 ms, with the producer at 4.19 vs 4.34 ms and the consumer at 4.04 vs 3.99 ms.

cuBLAS `single` (0.980 / 1.974 / 3.948 / 8.510) and TP2 (0.661 / 1.240 / 2.429 / 4.705) did not
change. GPU-only (a `_sleep` ahead of the start event hides the host launch), the plain fc1 with
G = 32 beats cuBLAS: 0.510 vs 0.536 ms at 4k and 1.019 vs 1.081 ms at 8k. So the remaining
1GPU-CuTe gap at 4k is the launch cost. Single runs are only good to a few percent. In batch 2
the unchanged 1GPU-CuTe kernel moved by −6 % to +2.5 % between two jobs, and single 32k runs vary
the most (5.30–5.37 ms with X copied, 4.86–5.15 ms with X on both in these rounds). Before batch 2
the fix-1 one-lane spin measured within noise at 4k/8k and ≈ 2.5 % faster at 16k/32k than the old
32-lane spin. The benchmark never contained the NVLink wake-up described under IKET, item 1.

Paper shapes (K=N=8192, no activation, no bias). The single, 1GPU-CuTe, MB, TP2 and Triton columns
are one `--paper --iters 10 --x-replicated` run after batch 3. The CTA-pipe CuTe column uses the
shipped default for this shape (unsplit, producer G = 1, so X on both makes no difference). Its
values are medians of 4 samples from a separate job of 2 interleaved rounds against the batch-2
code: 1.173 / 2.064 / 3.792 / 8.421 ms before. The R columns mix those two jobs.

| M | single | 1GPU-CuTe | MB best | TP2 | CTA-pipe Triton | CTA-pipe CuTe | CuTe R-MB | CuTe R-TP |
|---|---|---|---|---|---|---|---|---|
| 4096 | 1.357 | 1.492 | 1.216 (1024) | 1.002 | 1.623 | 1.111 | +8.6% | −10.9% |
| 8192 | 2.760 | 2.935 | 2.288 (1024) | 1.923 | 3.005 | 1.989 | +13.1% | −3.4% |
| 16384 | 6.112 | 5.896 | 4.086 (1024) | 3.750 | 5.774 | 3.779 | +7.5% | −0.8% |
| 32768 | 11.712 | 13.290 | 7.839 (2048) | 7.330 | 11.342 | 8.916 | −13.7% | −21.6% |

After batch 4, from one `--paper --iters 10 --x-replicated --graph` run (the fc2 cluster applies:
N2 = 32 n-tiles; unsplit, so X copied and X on both are the same configuration):

| M | single | 1GPU-CuTe | MB best | TP2 | CTA-pipe Triton | CTA-pipe CuTe | X on both | graph | graph, X on both |
|---|---|---|---|---|---|---|---|---|---|
| 4096 | 1.377 | 1.519 | 1.246 (1024) | 0.993 | 1.638 | 1.118 | 1.100 | 1.060 | 1.057 |
| 8192 | 2.799 | 2.982 | 2.288 (1024) | 1.937 | 3.058 | 1.973 | 1.984 | 1.983 | 2.054 |
| 16384 | 5.660 | 6.213 | 4.136 (1024) | 3.773 | 5.880 | 3.991 | 3.861 | 3.825 | 4.164 |
| 32768 | 12.684 | 13.206 | 7.964 (2048) | 7.471 | 11.613 | 7.997 | 7.645 | 8.256 | 9.118 |

No regression against batch 3. In a 2-round interleaved guard in the same job (`--paper --modes
single_cutlass,ctapipe_cutlass --iters 10`), CTA-pipe CuTe went from 1.099–1.116 / 1.977–2.012 /
3.787–4.537 / 7.947–8.245 to 1.098–1.112 / 1.961–1.988 / 3.879–4.233 / 7.338–7.368 ms. That is
32k −9 % with the cluster, the rest within noise. Graph mode helps at 4k (−5 %). At 16k–32k this
single run has it slower, and it was not re-measured.

Here the CTA-pipe change against batch 2 is the host-launch cost only (fix 11): −5.3 / −3.6 / −0.3
/ +5.9 %. The 32k samples spread over 8.0–10.1 ms. 1GPU-CuTe (plain fc1 at G = 32) went 1.551 /
3.016 / 6.236 / 13.476 → 1.492 / 2.935 / 5.896 / 13.290 ms (−3.8 / −2.7 / −5.5 / −1.4 %, one run
against two). The interim Wan defaults applied here (producer G = 8, 25 or 27 of 32 n-tiles) were
a clear regression. X copied: 1.236 / 2.371 / 4.998 / 11.433 ms. X on both: 1.187 / 2.202 / 3.967
/ 10.527 ms. Both causes were measured. Producer G = 8 alone, at f = 1, gave 1.225–1.249 / 2.263–2.327
/ 4.93–5.09 / 10.35–10.74 ms against 1.094–1.130 / 1.963–2.031 / 3.73–4.07 / 8.03–10.13 ms at
G = 1. The split alone at G = 1 (27/32) was slower than f = 1 at every M. This shape's two GEMMs
have equal output sizes, so the producer carries little extra epilogue work and there is nothing
to rebalance. The X copy is also larger here (2·M·8192 bytes). Hence the defaults are keyed by
shape.

Reading: on Wan2.2 shapes the CuTe CTA-pipe cuts 26–35 % (X copied) or 29–40 % (X on both) off
the same kernel run alone on one GPU at ≥ 8k rows. It beats the best micro-batch by 28–32 %. It
still trails TP2: by 6–9 % at 16k–32k with X on both, 14 % with the copy, and 18–42 % at 4k–8k.
The paper's ≈ 30 % over TP2 on B200 is not reproduced on H100. Two reasons are visible without a
profiler:
1. The perfectly pipelined bound is `max(T_fc1, T_fc2) ≈ 0.5 × 1GPU-CuTe`, and the pipeline
   reaches 0.61–0.65 × at 32k rows (0.67 × after batch 2).
2. Each forward still pays ~0.3 ms of fixed cost: serial host launches, the first producer wave,
   and one K=14336 consumer tile of tail. At small M that is large relative to TP2's cuBLAS GEMMs.

TP2 runs cuBLAS on half-size GEMMs, and its all-reduce over NVSwitch is cheap relative to the
compute here. `H` alone is `2·M·14336` bytes over NVLink
(≈ 0.94 GB at 32k rows, ≥ 2 ms at H100's ~450 GB/s per direction), which micro-batching also pays,
but TP2 does not. See the IKET section below for the in-kernel breakdown.

Protocol mapping to the paper (Sec. III-A):

| paper | here |
|---|---|
| producer writes output tile into consumer memory (1) | `tl.store` to `h` allocated on the consumer GPU (unified addressing after `cudaDeviceEnablePeerAccess`) |
| system-wide memory fence | `tl.debug_barrier()` + inline `fence.acq_rel.sys` |
| dependency array + scoreboard on producer (2, 3) | collapsed: dependency is "consumer row block *m* needs all `N1/BLOCK_N` producer tiles of row block *m*"; one int32 counter per row block, on the consumer |
| push ready ID to inter-device work queue (4) | `tl.atomic_add(cnt[m], 1, sem="release", scope="sys")` (remote, fire-and-forget) |
| consumer prologue polls local queue (5) | single-thread `tl.atomic_add(cnt[m], 0, sem="acquire", scope="sys")` spin until `== n_ready`, then the unmodified GEMM tile |
| host: one stream per kernel, launched simultaneously | one `torch.cuda.Stream` per device; counter reset → producer → consumer ordered with CUDA events only (the CuTe driver uses epoch targets instead and never resets, fix 10) |

Verified on 2× H100: `pytest test/cta_pipelining` (30 tests: Triton interpreter protocol test,
Triton and CuTe two-GPU runs vs cuBLAS for M multiple / not multiple of the tile, output on
either GPU, buffer growth, second forward reusing counters; CuTe epoch counters over five forwards
with growing and shrinking M, including a forced int32-guard reset, unsplit and column-split; CuTe
plain GEMM with/without bias and each activation; raster G = 3 / 32 bit-identical to G = 1; column
split at 11 / 8 / 6 / 7 of 11 producer tiles, with X copied or passed on GPU 1; cluster (1, 2)
bit-identical to (1, 1) in plain and consumer role; CUDA-graph mode over three M switches with the
static output checked before each overwrite, X copied or on GPU 1, output on either GPU, unsplit
and split, then eager again). Run everything with

```bash
pytest test/cta_pipelining -q
python -m benchmark.cta_pipelining.bench_mlp_2gpu                 # Wan2.2 FFN shapes, all 6 modes
python -m benchmark.cta_pipelining.bench_mlp_2gpu --paper         # paper's 8192² GEMMs
CTA_PIPE_IKET=1 CUTE_DSL_COMPILER_OPT=iket run-iket -o OUT --clobber profile --postprocess json -- \
    python -m benchmark.cta_pipelining.iket_ctapipe --tokens 8192   # then iket_analyze OUT/*.trace.json --by-tile
```

### IKET in-kernel profile (Wan2.2 shape, M=8192, 2× H100, CUTLASS DSL 4.7.1)

`CTA_PIPE_IKET=1 CUTE_DSL_COMPILER_OPT=iket` compiles per-warp ranges into the kernels and
`run-iket ... profile --postprocess json` records them. Read proportions, not absolute latency:
the instrumented forward took 2.32 ms after batch 3 and 2.09 ms before it, against 1.52 / 1.72 ms
plain. IKET also slows the host launches, which matters more now that a forward launches three
kernels (see the local fc1 below). The after trace is of the interim defaults. At M=8192 the
producer computes 44 of the 56 n-tiles: 2816 tiles (64 row blocks × 44) over 132 persistent CTAs =
21–22 tiles per CTA, with the G = 8 raster. GPU 1 computes the other 12 n-tiles itself in a third
kernel before the consumer. The shipped default (48/56, producer G = 1) was chosen after this trace
and was not traced. The DSL's
per-warp range records are divided by warp, so the shares below are per warp, not summed over the
8 MMA warps. The tables are the third (back-to-back) forward of `iket_ctapipe.py`;
`iket_analyze --by-tile` prints the per-tile-index breakdown, CTA start/end skew and CTA end times
grouped by tile count. They show the state after batch 3 (fixes 3, 6a/9, 11). Values in brackets
are the same-session trace of the batch-2 code (56 of 56 n-tiles, 27–28 tiles per CTA). One trace
per state, so differences of a few percent are noise. Traces: `iket_b3_after` (the batch-3
before-trace was deleted for storage) and `iket_b4_after` (next paragraph).

**After batch 4** (`iket_b4_after`, `b4_bytile_after.txt`; shipped eager defaults: 48/56,
producer G = 1, consumer fc2 as a (1, 2) cluster). The instrumented forward took 2.05–2.22 ms.
* Producer: 3072 tiles on 132 CTAs (96 with 23 tiles, 36 with 24). Span 1131 µs, `mma_tile`
  36.0 µs mean, `signal` 4.0 µs p50, end skew 165 µs. The 24-tile CTAs end 38 µs later on average.
* Local fc1: 512 tiles, span 157 µs (223 µs at 44/56 in batch 3).
* Consumer: same 768 tiles on 128 CTAs × 6. Span 1000 µs (1010). `mma_tile` mean 157.4
  (157.1), p90 164 (168), max 168 (178). `tma_tile` p90 170 (180). End skew 58.6 µs (81.5).
  `wait_row` 0.2 % (first ready 0.9 µs after kernel start).

So the cluster does not speed up the mean fc2 tile. What it removes is the slow tail: the p90 /
max tiles and a third of the end skew. That matches the GPU-only gain (3.96 → 3.79 ms at 32k)
and the 16k / 32k end-to-end gain. The consumer is still fed (no waits). Its last tiles now form
the drain described under PERF ASSUMPTIONS (the "Drain" entry).

Tooling caveats found on the way: (1) the DSL mangles the kernel name from the owning *class*
name, so producer and consumer instances of one class collide and `run-iket` fails with "Kernel
... not found in IKET instrumentation info"; the `CTAPipeProducerGemm` / `CTAPipeConsumerGemm`
subclasses exist only for that. (2) When cuBLAS/torch kernels are loaded *before* the two DSL
kernels, IKET 4.7.1 mis-files the second DSL kernel's GPU-1 module instance under an older module
entry and silently skips its launches (`LaunchShouldInstrument Returns false ...
ModuleHasInstrumentedDeviceFunction`); the driver therefore runs the pipelined forward once before
computing the cuBLAS reference. Single kernels on either GPU never showed the problem. (3) The
traced forward must follow another forward with no idle gap. The cuBLAS reference's first-call
init leaves NVLink idle for well over 50 ms, and the next producer launch then pays a ~200 µs
NVLink wake-up on its first peer stores (item 1 below). An earlier version of this section traced
that forward and reported the wake-up as a per-launch "first-store stall" of 180 µs. The driver
now runs 3 forwards and the last one is read. (4) Launching the consumer before the producer
deadlocks under `run-iket`: the consumer spins and the producer never runs.

Producer (GPU 0, fc1 with gelu+bias, writes `H[:, :N1p]` to GPU 1 over NVLink), per-warp share of
the 0.96 ms mean CTA lifetime (1.21 ms; span 1.05 vs 1.30 ms):

| range | where | per tile (µs) mean / p50 / max | share of CTA lifetime |
|---|---|---|---|
| `mma_tile` (mainloop incl. stalls) | 8 MMA warps | 36.2 / 35.7 / 52 (36.4 / 35.7 / 55) | 80.8 % (81.6 %) |
| `epi_tile` (bias, gelu, TMA store to peer) | 8 MMA warps | 7.1 / 6.4 / 20 (6.9 / 6.3 / 19) | 15.7 % (15.4 %) |
| `signal` (wait_group 0 + `fence.acq_rel.sys` + remote `red`) | store warp | 4.9 / 3.7 / 17 (4.7 / 3.8 / 19) | 11.0 % (10.6 %) of the store warp |
| `tma_tile` (DMA warp issuing loads) | DMA warp | 44 / 43 / 65 (44 / 42 / 69) | 98.0 % (98.4 %, always busy or blocked) |

By tile index within a CTA (last tile = #21 after, #27 before):

| tile # | epi mean / p50 (µs) | mma mean / p50 | signal mean / p50 |
|---|---|---|---|
| 0 | 12.2 / 13.0 (12.3 / 14.0) | 36 / 36 (38 / 39) | 11.9 / 12.1 (11.8 / 11.5) |
| 1 | 10.2 / 11.6 (9.9 / 10.3) | 43 / 44 (45 / 46) | 9.8 / 11.1 (8.4 / 9.1) |
| 2–3 | 7.6–9.3 / 6.9–8.7 (7.9–8.5 / 6.9–7.1) | 37–41 / 39–44 (41 / 42) | 7.0–7.7 / 7.3–8.9 (6.8–9.1 / 6.7–9.7) |
| 5 | 6.5 / 6.2 | 36 / 35 | 4.4 / 3.9 |
| last 3 | 5.3–6.3 / 5.2–6.3 (5.2–6.3 / 5.2–6.2) | 32–35 / 32–35 (32–35 / 32–35) | 2.8–3.9 / 2.8–3.9 (2.9–4.3 / 2.7–4.3) |

The per-tile costs did not change. The G = 8 raster does not show in the producer's mainloop
(36.2 vs 36.4 µs mean), and the producer is shorter because it has 21–22 tiles per CTA instead
of 27–28. End skew is 175 µs (159 µs before). The 44 CTAs with 22 tiles end 40 µs after the 88
with 21, on average.

Local fc1 (GPU 1, plain role, `H[:, 44·256:]`: 768 tiles = 64 row blocks × 12 n-tiles, G = 32,
new in batch 3): mean CTA lifetime 214 µs, span 223 µs, end skew 38 µs (108 CTAs × 6 tiles, 24
CTAs × 5). `mma_tile` is 31.3 / 31.1 / 33 µs, 15 % faster per tile than the producer's (no peer
store traffic, better L2 reuse), and `epi_tile` is 5.0 / 4.9 / 6.5 µs. On GPU 1's clock (both
kernels run there) the consumer's first CTA starts 101 µs after the local fc1's last CTA ends.
The IKET build runs the host launches with extra cost. Without IKET, the CUDA-event timeline of
the same forward (X copied, M=8192) shows the consumer starting 3 µs after the local fc1 ends
(0.504 → 0.507 ms). The consumer's launch returns at 0.385 ms, before the local fc1 ends, so
the host no longer gates GPU 1.

What this says:

1. **There is no per-launch first-store stall. The 180 µs seen earlier is NVLink waking up after
   an idle period (fix 1, resolved).** `nvidia-smi nvlink -gLowPwrInfo` shows all 18 links of GPU 0
   in "Low Power State" mode with a 50 000 µs threshold. After 50 ms with no GPU 0 → GPU 1
   traffic, the first peer stores of the next launch wait ~200–250 µs on every CTA. All 132
   tile-0 epilogues then finish within about 20 µs of each other, which points to one wake-up
   event rather than a bandwidth limit. Measured on the producer alone at M=1024, without IKET:
   * Run back-to-back it takes 352–390 µs.
   * After a host sleep of 3 / 5 / 10 / 20 / 50 / 100 / 300 ms it is slower by
     +0 / +1 / +11 / +30 / +256 / +242 / +304 µs.
   * In a separate run, the same 100 ms sleep cost +129 µs with `H` and the counters local (GPU
     0's own ramp-up) and +374 µs with them on GPU 1 (309 → 438 µs vs 352 → 726 µs).
   * Running cuBLAS GEMMs on GPU 0, GPU 1 or both just before the producer does not remove the
     penalty (636–646 µs), so GPU clocks are not the cause. A GPU 1 → GPU 0 write does not
     remove it either (637 µs), and a 14-store SM poke from GPU 0 → GPU 1 only helps a little
     (594 µs). A 1 MB copy-engine copy from GPU 0 → GPU 1 does remove it (481 µs).

   The benchmark loop never idles for 50 ms, so the table above never contained the stall. In
   the same `run-iket` process, forward 1 (right after the cuBLAS init) still shows it
   (tile-0 epilogues end at +238 µs, p50). Forward 2, run back-to-back, gives tile-0 `epi_tile`
   11.8 / 13.3 µs (mean / p50), against 183 / 186 µs in the traced idle forward. The remaining
   ~2× over steady state is consistent with all 132 CTAs storing 64 KB at once (an 8.4 MB burst).
   What each experiment showed (tile-0 `epi_tile` p50):
   * Consumer spin down to one lane with a 256 ns `nanosleep`: 182 µs vs 186 µs, so no effect.
     It was kept as fix 12; see PERF ASSUMPTIONS.
   * Producer alone: stalls only on the first launch after idle, and the back-to-back second
     launch shows 13.1 µs.
   * A per-launch warm write just before the producer: the stall moves into the warm kernel
     (248–284 µs after 100 ms idle), and back-to-back it adds 68–92 µs of serialized time. Not
     adopted.
   * Local `H`: no epilogue stall (5.4 µs), but the first remote counter `red` after idle stalls
     instead (about 110 µs).
2. **Steady-state per-tile cost ≈ 34–36 µs MMA + 6 µs epilogue + 3–4 µs signal (was 5–7 µs).**
   The signal (`cp.async.bulk.wait_group 0`, `fence.acq_rel.sys`, remote `red`) runs in one warp
   of MMA warp group 1. `wgmma` is collective over the warp group, so the whole group waits for
   the signal, and warp group 0 is throttled through the shared smem pipeline within about 4
   k-tiles. Sub-ranges measured in fix 2 show where the signal's time goes. The `wait_group 0` is
   ≈ 0 by then, because the tile's 8 bulk stores have already completed during the epilogue. The
   cost is the system-scope fences: ≈ 3 µs for `fence.acq_rel.sys` and ≈ 2.5 µs for the second
   fence implied by `red.release.sys`. That second fence was redundant, and the `red` is now relaxed.
   The remaining fence's latency depends on the SM's other outstanding memory traffic. Moved into
   a separate signalling warp, it took ≈ 40 µs, which means it completes about once per tile.
   Net: the epilogue takes ≈ 15 % of each MMA warp's time and the signal ≈ 10 % of the store
   warp's time (was 16 %), on top of the mainloop.
3. **The mainloop itself runs at cuBLAS-like speed once warm.** A 128×256×3072 tile takes 33–35 µs
   (mean) at the end of the kernel, about 5.7–6.1 TFLOP/s per SM or 750–810 TFLOP/s for the chip,
   IKET overhead included. The smem pipeline is pre-filled during the previous epilogue, so the
   range flatters the rate a little. Tiles 1–4 are 10–35 % slower (47 → 38 µs) while the
   synchronized tile-0 store burst and the cold L2 drain.
4. **Tail: producer CTAs finish over a 90–230 µs window** across this batch's traces (180 µs
   after, 188 µs before; start skew 0.1 µs). 3584 tiles over 132 CTAs gives 20 CTAs 28 tiles
   and 112 CTAs 27. The 28-tile CTAs end 26–46 µs later on average (about one tile), and the
   27-tile CTAs alone spread over 127–153 µs, so most of the skew is per-CTA variance
   accumulated over 27 tiles. The last row block the consumer can start is gated by the slowest
   CTA, so this tail sits directly on the critical path. Evening the producer out (128 CTAs × 28)
   measured 117 µs in its trace. The unchanged 132-CTA producer also showed 119 µs in another
   trace, and the benchmark lost 2–3 % at 4k/8k, so that change was reverted (fix 5).

Consumer (GPU 1, fc2 with bias, K=14336: 768 tiles = 64 row blocks × 12 n-tiles, 128 CTAs × 6
tiles (fix 5), 224 k-tiles per tile), per-warp share of the 0.97 ms mean CTA lifetime (1.02 ms):

| range | where | per tile (µs) mean / p50 / max | share of CTA lifetime |
|---|---|---|---|
| `wait_row` (spin on the row counter) | DMA warp | 0.5 / 0.4 / 1.5 (4.0 / 0.4 / 71) | **0.3 %** (2.3 %) |
| `tma_tile` (issuing one tile's 224 loads) | DMA warp | 161 / 160 / 190 (165 / 167 / 191) | 99.0 % (97.0 %) |
| `mma_tile` (mainloop incl. data stalls) | 8 MMA warps | 157 / 157 / 178 (166 / 163 / 242) | 96.9 % (97.2 %) |
| `epi_tile` (bias, TMA store to GPU 0) | 8 MMA warps | 4.0 / 2.6 / 18 (3.8 / 2.7 / 17) | 2.4 % (2.3 %) |

The `wait_row` share is the noisiest number in these traces. It ranged from 5 % to 23 % across
the four traces taken after fix 2. The batch-2 code gave 2.3 % in this session, because IKET's
slower host launches put the consumer launch further behind the producer. After batch 3 no
consumer tile waits: the first `wait_row` ends 0.8 µs after the kernel starts, and waits #1–#5
average 0.3–0.7 µs (before: #3–#5 averaged 4–13 µs, max 71 µs). The consumer starts after the
local fc1 (about 0.3 ms into GPU 1's forward), and by then the producer is ahead. Its CTA
lifetime is now pure fc2: 157 µs per tile, at the ≈ 165 µs fed rate quoted below.
The plan's fallback, the local fc1 on a second GPU-1 stream, is only warranted if `wait_row`
stays above 10 %, so it was not measured. The rest of this section is the batch-2 analysis.

The wait is not a start-up effect. Wait #0 is ≈ 0 (p50 0.3 µs), most likely because the consumer
is launched about 150 µs after the producer (the host-side DSL launch costs 143–148 µs per call)
and by then row blocks 0–1 are done. Waits #1–#4 average 45–65 µs each (58–95 µs before batch 2).
Mechanism: consumer CTA `c` processes tiles `c, c+128, …`, so it jumps 10.7 row blocks per tile.
The producer completes 10.7 row blocks every ≈ 195 µs (2.36 row blocks per ≈ 44 µs wave). A
consumer tile needs only ≈ 165 µs of MMA when fed (128×256×14336 at the cuBLAS rate). `mma_tile` is
193 µs because ≈ 28 µs of every tile is spent blocked on the smem pipeline while the DMA warp
spins. So **the producer is still the bottleneck**. Both sides do equal FLOPs, but the producer's
remote-store epilogue and signalling make it ≈ 1.2× slower than the consumer's compute (1.21 ms
CTA lifetime vs ≈ 0.99 ms of fed consumer MMA for 6 tiles), and the consumer just tracks it. The
consumer's own remote epilogue (writing `Y` back to GPU 0) is cheap (2.5 µs per tile, 1.6 %).

Consumer CTAs end within a 180 µs window in the after trace (141 µs in the fix-5 A/B trace, 206 µs
before batch 2); the producer's window is 180 µs. The last consumer tile can only start after the
producer's slowest CTA has finished row block 63, so end-to-end ≈ producer time + one consumer
tile (≈ 170 µs) + tails. That matches the 1.73 ms uninstrumented forward (1.80 ms before batch 2)
against ≈ 1.15 ms for either GEMM alone.

What to change first, in order of measured payoff (all Stage-1 kernel work, no M* changes):

1. ~~Kill the first-remote-store stall.~~ **Done: not a kernel cost** (item 1 above). It only
   appears after ≥ 50 ms without GPU 0 → GPU 1 traffic, which means once per request or after a
   pause, not per FFN call. Fixed in the measurement harness (`iket_ctapipe.py` traces a
   back-to-back forward): tile-0 `epi_tile` p50 13.3 µs vs 186 µs; producer CTA lifetime 1.29 ms
   vs 1.45–1.47 ms; consumer `wait_row` share 31.8 % vs 41–43 %. The benchmark is unchanged (it
   never idled). The consumer spin throttle landed with it and is neutral to about 2.5 % faster
   (PERF ASSUMPTIONS).
2. ~~Defer the signal by one tile.~~ **Fix 2: done as a relaxed `red`. The planned deferral was
   neutral or worse.** The `wait_group` costs ≈ 0 and the system fences are the real cost (item 2
   above), so moving the wait did not help. Benchmark numbers below are medians of interleaved
   rounds at M = 4k / 8k / 16k / 32k.
   * Deferral as planned: `wait_group 8` after tile *i*'s epilogue, then signal tile *i−1*.
     `signal` p50 went 7.0 → 6.4 µs, since the fences moved but did not shrink. Benchmark vs no
     deferral: +1.0 / +1.8 / −2.9 / +3.7 % (2 rounds). Reverted.
   * Deferral placed before tile *i*'s epilogue: 6.42 vs 6.19 ms at 32k (one run). Reverted.
   * A dedicated signalling warp: its `fence.acq_rel.sys` took ≈ 40 µs, and it gave
     −3.7 / −3.4 / 0 / +7.1 % (5 rounds). Reverted.
   * Kept: `fence.acq_rel.sys` followed by `red.relaxed.sys`, the PTX release pattern. This
     replaced `red.release.sys`, whose implied fence was a second one. `signal` p50 went 7.0 →
     3.8 µs, `mma_tile` p50 39.1 → 35.5 µs, and producer CTA lifetime mean / max 1307 / 1400 →
     1203 / 1282 µs. The consumer `wait_row` share went 33.7 → 16.3 %. Benchmark: 1.087 / 1.789
     / 3.209 / 6.175 → 1.059 / 1.688 / 3.144 / 5.983 ms (−2.6 / −5.6 / −2.0 / −3.1 %, 3 rounds).
3. **Fix 3: rebalance (batch 3). Done as a column split of GEMM 1.** GPU 1 computes the last
   `N1 − N1p` columns of `H` itself (plain-role kernel on `c_stream`), and the producer computes
   only `[0, N1p)` (see "What is implemented"). All numbers are CTA-pipe medians in ms at M = 4k /
   8k / 12k / 16k / 32k.
   * The optimum is erratic in f. At producer G = 8 in one sweep, the best tile counts were
     56/56 (unsplit), 44/56, 44/56, 48/56 and 48/56. At G = 1 in another, 48/56 won everywhere
     except 4k, where 44/56 was 1 % faster.
   * Shipped: 48/56 for the Wan shape, keyed by shape (unsplit elsewhere). Against f = 1 in the
     same rounds (G = 1): X copied 0.946 / 1.494 / 2.131 / 2.869 / 5.516 vs 0.984 / 1.667 / 2.317
     / 2.963 / 5.648 (−3.9 / −10.4 / −8.0 / −3.2 / −2.3 %); X on both 0.909 / 1.458 / 2.093 /
     2.756 / 6.280 vs 0.980 / 1.683 / 2.321 / 2.962 / 5.806 (−7.2 / −13.4 / −9.8 / −7.0 / +8.2 %,
     the 32k cell as discussed under Measured).
   * The plan's model predicted ≈ 1.25–1.3 ms at 8k. It left out about 0.3 ms of fixed offsets
     that CUDA-event timelines show:
     * the local fc1 cannot start before its host launch (≈ 0.2–0.3 ms into the forward, after
       the producer's);
     * with the copy, it also waits for X (50 MB at 8k, 201 MB ≈ 0.6 ms at 32k);
     * one consumer tile of tail (≈ 0.17 ms) remains after the producer ends.
     The IKET trace confirms the consumer no longer waits on rows (`wait_row` 0.3 %).
   * The plan's other options (last n-tiles of each row block, fused or split-grid local fc1)
     are in PERF ASSUMPTIONS.
   * Launch order, measured with the conversion cache in place:
     * **Local-first** (X copy + local fc1 launched before the producer, as first proposed): worse
       than producer-first. X on both, best share per M, 4k / 8k / 12k / 16k / 32k: 0.980 /
       1.542 / 2.144 / 2.831 / 5.290 vs 0.907 / 1.479 / 2.146 / 2.700 / 5.374.
     * **X copy issued before the producer launch, local fc1 after it**: shipped. It helps the
       copy variant, e.g. 48/56 at 16k / 32k 2.938 / 5.914 → 2.863 / 5.667; X on both is
       unaffected.
     * Order now: X copy → producer → local fc1 → consumer.
4. ~~Hoist the bias load.~~ **Fix 4: measured neutral and reverted.** Each thread loaded its 64
   bias values once per tile as 32 `bf16x2` loads, issued while the tile's last `wgmma` is in
   flight, instead of 128 scalar loads. The consumer `epi_tile` fell 2.9 → 2.0 µs in IKET. The producer's did not improve
   (7.0 → 7.9 µs, within trace noise). 1GPU-CuTe at M=4096 measured 1.152 / 1.173 ms vs 1.160 /
   1.154 ms (two rounds each). The 1GPU-CuTe gap to cuBLAS does not come from the bias load. A
   cheaper gelu was not tried.
5. **Fix 5: tail, consumer grid evened out, producer left alone.** End skew (max end − min end)
   after fixes 2 + 10 was 154 µs on the producer, over its one-tile threshold of ≈ 40 µs, and 152
   µs on the consumer, under its one-tile threshold of ≈ 170 µs. Before batch 2 they were 188 /
   206 µs. Option (a) was tried: the fewest CTAs that keep the same number of waves.
   * Both kernels (producer 128 × 28 tiles, consumer 128 × 6): +2 / +3 / −2 / −6 % (6 rounds).
   * Consumer only (768 tiles → 128 × 6, producer stays at 132 CTAs): 1.089 / 1.749 / 3.097 /
     6.240 → 1.087 / 1.760 / 3.026 / 5.830 ms (0 / +0.6 / −2.3 / −6.6 %, 6 rounds). In IKET the
     consumer's last CTA end moved 1337 → 1202 µs. **Kept.**

   Options (b), reversing the last wave, and (c), dynamic claiming, were not needed.
6. **Fix 10: epoch counters instead of a per-forward zero (`cute_mlp.py`, host side only).** The
   per-forward `zero_()` kernel is gone, along with the cross-device event that made the producer
   wait for it. Forward *e* passes `n_ready = e · N1/256`, and the kernel's existing `>=` spin
   waits for `counter[m] >= e · 56`. Row blocks past `M` in a larger reserved buffer get
   `+N1/256` from one small `add_` on the consumer stream, which keeps every counter at
   `e · n_ready`. The epoch restarts when `_reserve` reallocates. A host-synchronizing zero runs
   before `(e+1) · n_ready` would reach 2^30, which is ≈ 19 M forwards at 56. The producer now
   waits on the previous forward's consumer-done event before overwriting `H` (WAR guard). The
   zero kernel's event chain used to provide that ordering implicitly. Measured alone: 1.087 /
   1.789 / 3.209 / 6.175 → 1.054 / 1.747 / 3.164 / 6.148 ms (−3.0 / −2.3 / −1.4 / −0.4 %).
   Together with fix 2: 1.017 / 1.662 / 3.023 / 5.896 ms (−6.4 / −7.1 / −5.8 / −4.5 %), 3 rounds.
   The Triton driver (`mlp.py`) still zeroes; see PERF ASSUMPTIONS.
7. **Fix 6a / 9: L2 raster for fc1 (batch 3).** `raster_group` G (compile-time, one `@cute.jit`
   helper shared by the DMA and MMA loops): G row blocks per group, n-major inside it. G = 1
   reproduces the old order bit for bit (tested).
   * Plain fc1 GPU-only (`_sleep` hides the launch), ms at M = 4k / 8k / 32k:
     * G = 1: 0.611 / 1.181 / 4.765
     * G = 8: 0.582 / 1.164 / 4.62
     * G = 16: 0.526 / 1.069 / 4.35
     * G = 32: 0.508 / 1.017 / 4.147
     * cuBLAS: 0.535 / 1.080 / 4.38–4.71
   * Estimated W1 traffic from HBM at 8k: ≈ 2.3 GB at G = 1, 0.68 GB at G = 16, 0.23 GB at
     G = 32. The kernel gains 14 %, not the ≈ 0.6 ms the traffic estimate suggests, so it was not
     purely HBM-bound.
   * On the paper shape, G = 32 is within noise of G = 16 / 24, and G = 8 was +9 %.
   * Shipped: G = 32 for plain / local fc1. 1GPU-CuTe −5.8 to −10.6 % on Wan shapes (Measured).
   * The producer stays at G = 1.
     * At f = 1 before the split, G = 8 helped (1.049 / 1.719 / 2.996 / 5.725 → 1.050 / 1.656 /
       2.883 / 5.590 at 4k / 8k / 16k / 32k) and G = 16 hurt (+5–14 %): row blocks complete 16 at
       a time and the consumer waits.
     * With the split, G = 1 was equal or better at 4k–16k (e.g. 48/56 at 8k, X on both: 1.458
       vs 1.622).
     * On the paper shape G = 8 cost 5–30 % (Measured).
     * The consumer stays at G = 1 (its 12 n-tiles per row block already share W2 within a wave).
8. **Fix 6b: tile shape by M (batch 3). Measured, not adopted.** Plain GEMM GPU-only, ms at M =
   4096 / 8192, fc1 / fc2:
   * 128×256: 0.527 / 1.068 and 0.444 / 0.891
   * 128×128: 0.698 / 1.389 and 0.675 / 1.360 (+26–52 %)
   * 64×256: 0.830 / 1.685 and 0.805 / 1.681 (+38–91 %)

   Both smaller tiles run a single MMA warp group. 128×256 wins at every M, so the single config
   stays and no table was added.
9. **Fix 11: host launch cost (batch 3).** Each DSL launch cost 103–119 µs on the host: ≈ 10 µs
   per torch → `cute.Tensor` (DLPack) conversion × 5, plus `get_device_properties` and ≈ 27–32
   µs in the compiled call itself. That is 0.5 ms per split forward.
   * `CuteGemmOp.launch(..., static=...)` caches the conversions of buffers the caller owns and
     that never move, keyed by `(data_ptr, shape, stride, dtype)`: weights, biases, `_h` and its
     slices, counters, dummies. `x` and `y` are still converted per call. `_reserve` clears the
     cache on reallocation.
   * The SM count is cached per device.
   * Per launch (local / producer / consumer): 119 / 112 / 103 → 70 / 59 / 58 µs. A split
     forward's host time went 497 → 342 µs (247 µs unsplit).
   * The remainder is the DSL's argument packing plus the launch (`generate_execution_args` ≈
     16–18 µs + `run_compiled_program` ≈ 9–11 µs) and ≈ 16 µs each for `x` / `y`.
   * No documented fast path in the installed DSL was small enough to use, and CUDA graphs were
     out of scope. The < 60 µs target is met for the producer and consumer, not for the local
     launch (70 µs, it converts `x`).
   * Why it matters: the benchmark's timed region starts before the first launch, and the local
     fc1 and the consumer cannot start before their launches return.
10. **Fix 6c / batch 4 item 1: consumer fc2 as a 2-CTA cluster (1, 2) with TMA multicast of A
    (H). Kept.**
    * `CTAPipeGemm(cluster_shape_mn=(1, 2))`: the two CTAs of a cluster share the row block and
      take adjacent n-tiles (`n = 2·n_pair + rank`). Each CTA loads its own W2 tile and half of
      the H tile, multicast to both (`make_layout_image_mask(..., mode=1)`), and the consumer
      barrier arrive count doubles.
    * Launch `cluster=(2, 1, 1)`. The grid is capped by the DSL's `get_max_active_clusters(2)`
      and rounded to whole clusters. Only the consumer uses it, and it falls back to (1, 1) when
      N2 is not a multiple of 2 × 256.
    * Tile order and results are bit-identical to (1, 1) (tested, plain and consumer roles).
    * GPU-only consumer fc2 at 4k / 8k / 32k: 0.472 / 0.920 / 3.96 → 0.452 / 0.911 / 3.79 ms.
    * End to end: 16k copy / X on both 2.774 → 2.704 and 2.742 → 2.673, 32k copy 5.585 → 5.390,
      4k–8k neutral (item-1 A/B). The batch-4 Measured table has the final numbers.
    * Not done:
      * (2, 1) multicasting W. For the producer and the plain fc1 it would need even G and an even
        row-block count with a padded tile.
      * 5 stages. Infeasible: 5 × 48 KB = 240 KB > 227 KB of smem.
      * Consumer G = 2 / 4 did not help.
11. **Batch 4 item 3: CUDA-graph forward (opt-in, `use_graph=True`). Kept, default off.** The DSL
    launch is capturable: host-side TMA descriptor encode, then `cudaLaunchKernelEx`, with no
    implicit sync. Details under "CUDA-graph forward" above. At 48/56, X on both, graph vs eager
    at 4k / 8k / 12k / 16k / 32k: 0.836 / 1.465 / 2.006 / 2.683 / 5.451 vs 0.916 / 1.471 /
    2.073 / 2.777 / 5.427 ms. With the copy, the graph loses at ≥ 12k (2.194 vs 2.110 at 12k)
    because of the static-input copy. The graph-mode share table (40/56 up to 4k, 44/56 up to 8k)
    came from a share sweep of the graph (Measured).
12. **Batch 4 item 2: local fc1 fused into the consumer kernel. Built, measured neutral,
    reverted.**
    * One kernel on GPU 1 did the local fc1 tiles first, statically assigned. Each tile signalled
      its row counter with a gpu-scope fence plus `red.relaxed.sys`, and fc2 then waited for
      `N1 / 256` tiles per row block. Correct (allclose to the reference).
    * Best-share medians, X on both, item 1 vs fused, at 4k / 8k / 16k / 32k:
      * eager: 0.921 vs 0.927, 1.500 vs 1.526, 2.735 vs 2.721, 5.387 vs 5.221;
      * graph: 0.815 vs 0.818, 1.371 vs 1.379, 2.576 vs 2.593, 5.117 vs 5.087.
      All within the ≈ 4 % noise floor.
    * Why it does not pay:
      * The fused kernel's eager launch costs ≈ 145 µs of host time, the same as the two launches
        it replaces, so GPU 1 still starts at ≈ 0.34 ms.
      * fc2 rows wait for H rows anyway: 14.7 µs per row block to consume against 16.7 µs to
        produce.
      * The end-of-forward drain dominates.

    The end-skew fix folded into this item was not done: skew is not the binding term (item-1 IKET
    above).

## TP2 extension (experimental, batch 5)

`cute_tp2.py` (`CuteTP2OverlapMLP`) runs CTA-pipelining on top of Megatron TP2 and fuses the fc2
reduce-scatter (`output="sharded"`) or all-reduce (`"replicated"`) into the fc2 epilogue
(`ROLE_REDUCE` in `cute_gemm.py`).
* **Range A.** Each GPU's fc2 first computes the row blocks the peer owns. It TMA-stores the bf16
  partials straight into the peer's output and bumps the peer's row counter.
* **Range B.** Then it computes its own row blocks. The store warp acquires the counter and TMA
  reduce-adds `acc + b2` onto the peer's partial (`cp.reduce.async.bulk.tensor .add`, in L2).
  Replicated mode also writes both outputs.
* **Numerics** are the same as NCCL.
* **Bug rules.** `_bump` never read-modify-writes counter slices a peer may be signalling. A
  whole-array `add_` lost `red.add` updates and hung. One compiled op per GEMM is shared by both
  devices, because a per-device fc2 op deadlocked on the DSL's module load.

The baseline `tp2_cute` in the bench is the same kernels without overlap: `cudaMemcpyAsync` of the
partials plus `torch.add`. Final A/B: 3 interleaved rounds in one job, eager, X replicated, medians in
ms. `b2b` means back-to-back; per step means synchronized before each call. 32k ran in a separate
job and is power-capped in every mode.

| M | tp2 | tp2_cute | overlap | cute_rep | overlap_rep | per step: tp2 / cute / overlap / cute_rep / overlap_rep |
|---|---|---|---|---|---|---|
| 4096 | 0.607 | 0.557 | 0.538 | 0.611 | 0.562 | 0.627 / 0.765 / 0.767 / 0.824 / 0.800 |
| 8192 | 1.223 | 1.142 | 1.159 | 1.233 | 1.200 | 1.212 / 1.289 / 1.287 / 1.443 / 1.362 |
| 12288 | 1.880 | 1.796 | 1.775 | 1.887 | 1.826 | 1.807 / 1.841 / 1.865 / 2.054 / 1.933 |
| 16384 | 2.538 | 2.412 | 2.417 | 2.555 | 2.547 | 2.444 / 2.433 / 2.433 / 2.649 / 2.851 |
| 32768 | 4.962 | 4.820 | 4.630 | 5.388 | 4.848 | 4.831 / 4.686 / 4.553 / 5.117 / 4.761 |

**At 8k back-to-back** the overlap is +1.5 % (sharded) and -2.7 % (replicated) against `tp2_cute`.
That misses the 3 % bar for productizing, so the mode has no tests.

**IKET at 8k.** The reduce fc2 lives 495-508 us per CTA, against 436 us for `tp2_cute`'s plain fc2.
* A range-B epilogue takes about 3 us per tile.
* Range A costs about 20 us per tile (remote store, completion wait and signal). This is NVLink
  burst-bound.
* GPU 0's remaining wait comes from host launch skew.

## Stage 2: inside M* (two worker processes)

M* runs one process per GPU, so the two kernels are launched from different processes. The
kernels do not change; only buffer allocation does:

1. **Placement.** New node-group option for `dit` on ranks `[0, 1]`, e.g. `cta_pipeline: true`
   (`mstar/distributed/base.py` / `communication.py`). Rank 0 owns every DiT parameter except
   each block's `ffn.linear_out`; rank 1 owns the 30 `linear_out` weights/biases. Memory per
   rank is the same as TP2 for the FFN.
2. **Shared buffers** via `torch.distributed._symmetric_memory`: `symm_mem.empty(...)` +
   `rendezvous(t, group)` for `h [max_rows, 14336]` and the counters (owned by rank 1) and for
   `y [max_rows, 3072]` (owned by rank 0). `handle.get_buffer(peer_rank, shape, dtype)` gives
   each process a tensor over the *other* process's allocation to pass to its kernel.
3. **No counter reset.** Use a monotonically increasing epoch: consumer for (step, block) waits
   for `epoch * n_ready`; rank 1 never zeroes, rank 0 never waits for a reset (int32 wraps after
   ~19M FFN calls at 112 tiles/row; reset at a step boundary if ever needed). The single-process
   `cute_mlp.py` already works this way (fix 10), including the guard against int32 overflow.
4. **Completion back to rank 0.** Add a per-row-block "done" counter on rank 0 that the
   consumer bumps in its epilogue; rank 0 gates the residual add with a tiny spin kernel or
   `cuStreamWaitValue32` (`handle.wait_signal`). Rank 1's loop is fully deterministic (30
   consumer launches per step), so it can enqueue a whole step ahead.
5. **Block forward** (`Wan22DiTBlock.forward`): `norm3 → producer(norm_hidden)` on rank 0,
   `residual + y * c_gate` after the done-wait. Everything else in the block is unchanged.

## SHARED CODE TOUCHED

None. All files are new; nothing under `mstar/model`, `mstar/engine`, `mstar/distributed` or
existing configs was modified.

## PERF ASSUMPTIONS

CuTe DSL path (`cute_gemm.py`, the one the numbers above are quoted for):

* **Producer signals every tile in line, with one system-scope fence per tile.** The store warp
  runs `cp.async.bulk.wait_group 0` → `fence.acq_rel.sys` → relaxed `red` right after the tile's
  epilogue (fix 2). It belongs to MMA warp group 1, whose next `wgmma` waits for it. By then the
  wait itself is ≈ 0; the fence costs the rest, 3.8 µs p50 per tile, or 10.5 % of the store
  warp's time. Faster alternatives not taken:
  * deferring the signal by one tile, which moved the fence cost without shrinking it (+4 % at
    32k);
  * a dedicated signalling warp, whose fence took ≈ 40 µs under the MMA warps' memory traffic
    (+7 % at 32k);
  * one fence per k tiles, which means fewer fences but coarser and later readiness;
  * the paper's fence-free sentinel scheme (Sec. V-A).

  Why not now: the first two measured worse, and the last two change the protocol and need
  their own study. Cost: removing ≈ 3 µs of the old ≈ 7 µs signal gave 2–6 % end to end, so the
  remaining ≈ 4 µs is plausibly worth a few percent more.
* **Bias added with one scalar `ld.global` per accumulator element** (128 per thread per tile,
  bf16, from L1/L2) using `partition_C` of an identity tensor for the column index. Faster:
  32 `bf16x2` loads per thread per tile, issued while the last `wgmma` is in flight (fix 4). It
  measured neutral: 1GPU-CuTe 1.152 / 1.173 vs 1.160 / 1.154 ms at M=4096, consumer `epi_tile`
  −30 %, producer unchanged. Folding the bias into a K+1 GEMM was not tried. Why not now: no
  measurable gain for more code. Cost: ≈ 1 µs per consumer tile epilogue, nothing measurable end
  to end. The 1GPU-CuTe vs cuBLAS gap (≈ 19 % at M=4096 on Wan shapes) is not the bias, and the
  paper's shapes have no bias.
* **Single tile config 128×256×64, 4 stages, one CTA per SM, no autotuning (fix 6b measured).**
  128×128 and 64×256 were 26–91 % slower at M = 4k / 8k (fix list, item 8), so there is no tile
  table. cuBLAS picks per-shape kernels.
* **2-CTA cluster only on the consumer's fc2, (1, 2), multicasting A (H); falls back to (1, 1)
  when N2 is not a multiple of 512** (batch 4 item 1). Pairs share the row block and take
  adjacent n-tiles, so each CTA fetches half of the shared H tile and its own W2 tile.
  * Faster alternatives not taken:
    * (2, 1) multicasting B (W). It halves W's L2 → smem traffic, as cuBLAS / the CUTLASS example
      do.
    * Clusters on the producer and the local / plain fc1.
    * Larger clusters, or a cluster-aware raster.
  * Why not now:
    * A (1, 2) pair shares H, the operand fc2 re-reads 12 times per row block. It also keeps the
      row-block-major order and the one-counter-per-row readiness unchanged, with no pad tiles.
    * (2, 1) needs even G and an even row-block count with a padded, never-signalled OOB tile.
      On the producer that interacts with the per-tile signal.
    * W1 traffic was already cut by G = 32 on the plain fc1.
    * The fallback rule keeps unlisted shapes correct. Wan (N2 = 3072, 12 n-tiles) and paper
      (N2 = 8192) shapes both qualify.
  * Cost: unknown for the untried variants. The consumer gained 4 % GPU-only at 32k and ≈ 0 at
    8k. With the fallback, an odd N2 / 256 loses that.
  * The grid is capped at 2 × `get_max_active_clusters(2)` CTAs and rounded up to whole clusters,
    after the consumer "fewest CTAs with the same waves" rule (fix 5). An odd count gets one more
    CTA; the Wan 768-tile consumer is 128 CTAs either way. Cluster-level load balance at odd
    counts was not measured.
* **Static tile schedule with constant raster groups** (`t = bid + i·grid`). Plain / local fc1 are
  grouped by G = 32 row blocks (n-major inside a group, fix 6a/9). The producer and consumer are
  row-block-major (G = 1). There is no dynamic work claiming, so a consumer CTA can spin on row *m*
  while row *m+1* is ready. Faster: G tuned per shape and M (the producer's best G flipped between
  8 and 1 depending on the share), an atomic tile counter on the consumer, cluster-level swizzle.
  Why not now: one constant per role was within noise or better on both measured shapes, and the
  interplay with the share is erratic. Cost: a few percent per shape at most, by the sweeps.
* **Static producer share, keyed by shape: `DEFAULT_PRODUCER_SHARE` = 48/56 for the Wan2.2 FFN,
  unsplit for 8192² and for every unlisted shape.**
  * Faster alternatives:
    * a per-M share table (a two-entry version was built, measured and dropped);
    * runtime balancing (measure both GPUs' kernel times, adjust f);
    * a cost model that predicts f from the shapes.
  * Why not now:
    * The optimum is erratic in f and moves with the producer raster. With G = 1, 48/56 was
      within 1 % of the best per M.
    * A per-M share needs both W1 slices resident and a cumulative counter target instead of
      `epoch · n_ready`.
    * The paper shape shows a wrong share is a regression, not just a missed gain.
  * Cost:
    * Unlisted FFN shapes run unsplit and miss the split's 2–13 % until someone sweeps them
      (`bench_mlp_2gpu --producer-share ... --x-replicated`).
    * On the Wan shape, ≤ 1 % against the per-M best at G = 1.
* **X copied to GPU 1 inside the forward unless the caller passes `x_consumer`.** Faster: X
  already on both GPUs (Stage 2: both workers hold the block input). Why not now: it is an API
  choice for the caller. The standalone forward must stay self-contained. Cost: `2·M·K` bytes over
  NVLink ahead of the local fc1, 0.13 ms at 8k and ≈ 0.6 ms at 32k on GPU 1's critical path.
  Measured: X on both is 2–7 % faster at 4k–16k (Measured).
* **GPU 1's share of fc1 runs as a separate plain-role kernel, serially on `c_stream` before the
  consumer. The fused variant was built and reverted (batch 4 item 2).**
  * The fused consumer used a static local-first schedule: each CTA did its local fc1 tiles,
    signalled them, then ran its fc2 tiles.
  * Faster alternatives not taken:
    * Dynamic filler tiles: a consumer CTA whose next row block is not ready takes a local fc1
      tile from an atomic queue instead of spinning.
    * Local fc1 concurrently with fc2 on split grids.
    * Computing the local n-tiles of a row block right before its fc2 tiles.
  * Why not now: the static fused version measured within noise of the two-kernel path, eager and
    graph (fix list item 12). fc2 already never waits after the local fc1 (`wait_row` 0.2 %).
    Consumption runs at 14.7 µs per row block against 16.7 µs production, so GPU 1 would still
    be gated by the producer and then the drain. A dynamic queue adds an atomic per tile and a
    second readiness source for no visible slack.
  * Cost: the local fc1 kernel is 157 µs of GPU 1 time at 8k (IKET, 48/56), fully serial before
    the consumer. From item 2's numbers, recovering it is worth ≤ 1–3 % end to end until the
    producer and the drain shrink.
* **Eager by default: serial host launches (X copy → producer → local fc1 → consumer, one thread).
  The CUDA graph is opt-in (`use_graph=True`, batch 4 item 3).** The conversion cache (fix 11)
  brought launches to 58–70 µs each, and eager GPU 1 starts ≈ 0.2–0.3 ms into the forward.
  * Faster: the graph. At 48/56 with X on both it is −9 % at 4k against batch-3 eager and
    neutral at 8k–32k. The graph share table takes another 4.8–6.5 % off at 4k.
  * Why not default:
    * The graph needs static buffers and one graph per M.
    * With the copy it is 2–5 % slower at ≥ 12k.
    * Its best share differs from eager's (below). The caller picks it per deployment.
  * Cost (eager): ≈ 0.34 ms of host time per split forward, ≈ 0.1–0.2 ms of it visible.
* **Graph: one graph per `(M, dtype, x_consumer is None)` with static `x` / `x_consumer` / `y`
  buffers; `forward` returns the static `y` itself.** Faster alternatives:
  * M buckets with padding (fewer graphs, padded FLOPs);
  * writing into caller-owned output buffers.
  Why not now: the standalone op owns its buffers, and returning the static tensor avoids a copy.
  Cost:
  * Each new key runs 3 eager warm-up forwards plus a capture (not timed).
  * Each graph holds its own `M×K` input(s) and `M×N2` output.
  * The returned tensor is overwritten by the next replay with the same M, so the caller must
    consume or clone it (documented).
* **Graph: row counters zeroed by one memset node after the consumer in every replay, not with
  epochs.** Faster: epoch targets. The target is a constant of the captured graph, so bumping it
  per replay would need a device-side epoch read by the consumer. Why not now: the memset node is
  small and off GPU 0's critical path. Zeroing before the producer instead measured 15–18 µs
  slower. Cost: not measured separately. After a replay the op sets its eager epoch to `1 << 30`,
  so the next eager forward takes the int32-guard path (sync, zero, epoch 0) instead of trusting
  stale counters.
* **Graph: `x` (and `x_consumer`) copied into the static inputs before every replay; without
  `x_consumer`, the NVLink X copy for the local share runs inside the graph** on a producer-side
  side stream. Faster: the caller writes the inputs directly into the graph's static buffers.
  Why not now: API change. Cost: ≈ 35 µs at 8k and ≈ 140 µs at 32k for the local copy. The copy
  graph trails eager by 2–5 % at ≥ 12k. That side-stream copy running next to the producer was
  not profiled.
* **Graph-mode producer share picked by `max_tokens` at construction
  (`DEFAULT_PRODUCER_SHARE_GRAPH`: 40/56 for ≤ 4096, 44/56 for ≤ 8192, else 48/56).** The W1
  split is fixed when the op is built, so the share follows the deployment's maximum M, not the
  per-call M. Faster: per-call shares, which would need both W1 slices resident and cumulative
  counter targets. Why not now: the per-M share table was dropped once already for that reason.
  Cost:
  * An op built for 8k but called at 4k runs 44/56 instead of 40/56.
  * The 8k choice favours X on both: −2.7 % there, **+2.8 % with the X copy** (1.482 vs
    1.441 ms). The 44/56 local slice needs X, and the copy gates it.
  * Eager keeps 48/56 at every M: smaller shares were slower in eager at ≥ 8k.
* **Drain at the end of the forward (unaddressed; the main next path).** The producer's last wave
  completes ≈ 3 row blocks at once (132 CTAs, 48 tiles per row block). That leaves 36 fc2 tiles
  of 157 µs (K = 14336) on 128 consumer CTAs after the producer is done. GPU 1 then runs about
  one full fc2 tile alone: 0.16–0.23 ms, or 11–15 % of the 8k forward (8k timelines: producer
  end → consumer end 1.139 → 1.359 at 44/56, 1.250 → 1.479 at 48/56).
  * Faster, not built:
    * split-K (or stream-K) for the last row blocks' fc2 tiles, so 36 tiles × 157 µs spread over
      all 128 CTAs (≈ 45 µs each);
    * smaller fc2 tiles (128×128) for the last row blocks only;
    * a reversed last producer wave, so row blocks finish in a staggered way.
  * Why not now: batch 4 was scoped to items 1–3. Split-K needs a reduction (atomics or a second
    pass) into `Y` on GPU 0. Cost: ≈ 0.1–0.15 ms per forward at 8k (7–10 %), less in relative
    terms at 32k.
* **Producer grid stays `min(tiles, SMs)`; only the consumer grid is evened out** (fix 5). The
  consumer runs on the fewest CTAs with the same number of waves: 768 tiles → 128 × 6. The
  producer at M=8192 keeps 132 CTAs, 20 of which get a 28th tile. Evening out the producer too
  (128 × 28) measured +2 / +3 % at 4k / 8k. Its −2 / −6 % at 16k / 32k was matched by the
  consumer-only change, so it was reverted. A reversed last wave was not tried. Unmeasured:
  the consumer rule at ≤ 2 waves (Wan M ≤ 2816). For example, M=1536 (144 tiles) runs as
  72 CTAs × 2 instead of 132 CTAs with 12 doubled. The ideal makespan is the same, but it was not
  measured.
  Cost of the producer tail: 26–46 µs mean extra for the 28-tile CTAs, on the critical path.
* **Consumer spin: one lane of the DMA warp polls `ld.acquire.sys` with a fixed 256 ns
  `nanosleep` between polls**; the other 31 lanes wait at `bar.warp.sync` (fix 12, landed with
  fix 1). Faster alternatives: a tuned or adaptive backoff, or have the producer push ready row
  IDs into a queue so nobody polls. Why not now: this is already neutral to ≈ 2.5 % faster than the
  old 32-lane back-to-back spin (Measured, re-run note), and the consumer is gated by the producer
  anyway. Cost: up to ~0.5 µs of extra wake-up latency per row-block wait (`nanosleep` can
  oversleep up to 2×), against a 46 µs mean wait.
* **No NVLink wake-up before the producer.** NVLink links drop to their low-power state after
  50 ms idle (`nvidia-smi nvlink -gLowPwrInfo`), and the next producer launch then waits
  ~200–250 µs on its first peer stores. Faster alternatives:
  * issue a ~1 MB copy-engine copy GPU 0 → GPU 1 on a side stream well before the FFN (in Stage 2,
    at the start of the block's attention), which measured as removing the penalty;
  * raise the link low-power threshold with `nvidia-smi nvlink -sLowPwrThres` (needs root).

  Why not now: back-to-back forwards and a Wan2.2 step (an FFN every few ms) never idle for
  50 ms, and a warm write placed in-line before the producer only moved the stall and cost
  68–92 µs when warm. Cost: ~0.2–0.3 ms once per request, or after any pause of 50 ms or more.
* **Producer output goes through TMA store to peer memory**, one 128×32 bf16 TMA op per epilogue
  subtile (8 per tile) over NVLink. Untested alternative: larger epilogue subtiles or `st.global`
  vectors; TMA to peer memory works (verified) but NVLink efficiency of many 8 KB bulk ops vs
  fewer larger ones was not measured. Cost: unknown, possibly the biggest lever after (1).
* **`H` is a whole `[M, N1]` bf16 buffer on the consumer, written once over NVLink**
  (2·M·N1 bytes, ≈ 0.94 GB at 32k rows ≈ 2 ms of NVLink time on H100). Inherent to the paper's
  scheme (TP2 moves only 2·M·K bytes in its all-reduce, 4.7× less here). Not a fixable cost
  inside this kernel; explains part of the TP2 gap on H100 (NVLink 4) vs the paper's B200
  (NVLink 5, 2× the bandwidth).
* **fp32 `tanh.approx`-based gelu-tanh and `exp`-based silu** on the accumulator fragment.
  Faster: `ex2.approx`-based polynomial forms. Cost: the epilogue is a small fraction of a
  K=3072 tile; measured max error stays at 1 bf16 ulp.
* **IKET ranges (`CTA_PIPE_IKET=1`) add ~10–15% to kernel time** (1.88–1.99 ms vs 1.73 ms at
  M=8192 after batch 2, 2.01–2.04 vs 1.78–1.80 ms before; back-to-back forward; the earlier
  "~30%" compared a forward that included the NVLink wake-up), so the in-kernel breakdown is about
  proportions, not absolute latency. With three launches per forward (batch 3), IKET's slower host
  launches add more: 2.32 ms instrumented vs 1.52 ms plain, and the consumer started 101 µs after
  the local fc1 ended (3 µs without IKET).

Triton path (`kernels.py`), kept as the portable reference:

* **Plain Triton `tl.load`/`tl.dot` GEMM, not a warp-specialised kernel.** This is what the
  CuTe DSL path replaces; measured 235 TFLOP/s per GPU vs cuBLAS ~735, so the Triton CTA-pipe is
  slower than one GPU on every shape.
* **Fixed tiles 128×128×64, 8 warps, 3 stages, no autotune.**
* **One remote system-scope atomic per producer tile** onto a consumer-side row counter
  instead of the paper's producer-local scoreboard plus one remote queue push per ready
  consumer CTA (112 NVLink atomics per row block instead of one). Same in the CuTe path.
* **Full `fence.acq_rel.sys` per tile** (paper's default) rather than the Lamport sentinel
  scheme in their Sec. V-A. Same in the CuTe path, where it is now most of the per-tile signal
  cost (see above). The Triton atomic is still `sem="release"`, so it implies a second fence.
* **`.cg` (L1-bypass) loads for `H` in the consumer.**
* **K-dimension bounds masks in the GEMM loop** for arbitrary shapes.

Both paths / driver:

* **Counters: epoch targets in the CuTe driver, per-forward zeroing in the Triton driver.**
  `cute_mlp.py` never zeroes (fix 10, −0.4 to −3 % alone). It keeps three small costs:
  * One `add_` kernel on the consumer stream advances row blocks past `M` whenever the reserved
    buffer is larger than this forward. The producer does not wait for it. Faster: compare
    against a per-row target, which needs a kernel change for a sub-µs kernel.
  * A host `synchronize` + zero runs before `(e+1) · n_ready` would reach 2^30, which is every
    ≈ 19 M forwards at `n_ready` = 56 (22 M at the Wan default of 48). Faster: a wrap-safe compare (`(int)(cnt − target) >= 0`)
    or int64 counters would remove it, but that touches the spin in the kernel for a cost that
    comes once per 19 M calls.
  * A WAR guard: the producer of forward *e+1* waits for consumer *e* before overwriting `H`.
    The old zero kernel imposed the same ordering. Faster: double-buffered `H`, which would let
    back-to-back forwards overlap. It costs 2× the ≈ 1 GB buffer, and in a DiT step other work
    already sits between FFN calls. Cost: nothing in the model; in back-to-back benchmarks each
    forward starts after the previous consumer's tail.

  The Triton driver (`mlp.py`) still runs a `zero_()` kernel plus a cross-device event at the
  head of the critical path. Not mirrored: its `n_ready` is a plain int that Triton specializes
  on (`== 1`, divisible by 16), so a growing target can compile a second variant mid-run for some
  `N1`, and it would need the same fix-up and WAR logic on an unmeasured reference path. Cost:
  one tiny kernel + one event hop per FFN call.
* **`Y` is written remotely by the consumer into producer memory** so the residual add needs
  no copy; that is an extra `2·M·3072` bytes over NVLink per FFN and puts the peer-write
  latency of the last wave on the tail. Alternative: keep `Y` on GPU 1 and move the next
  block's attention there (alternating placement).
* **Whole `[M, 14336]` bf16 activation buffer resident on the consumer** (≈1 GB at 36,960
  rows) instead of a ring of row blocks. Memory, not time.
* **Micro-batch baseline is not CUDA-graph captured** (the paper's was). At small chunks the
  baseline eats launch bubbles, so the reported R-MB is slightly optimistic.
* **TP2 baseline is plain cuBLAS + NCCL all-reduce in two processes**, no fused/overlapped
  reduce-scatter; it is the paper's baseline, not the fastest possible TP2.
