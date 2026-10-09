# Command A+ on M*: GB200 TP4 bring-up

Bring-up notes for serving Command A+ (Cohere2 MoE) on 4x GB200 under M*, with
CUDA-graph decode and fused decode kernels. Covers what was broken, what was
changed and why, how it was validated, how to run it, the numbers, and what is left.

## Model and hardware

| | |
|---|---|
| Checkpoint | `CohereLabs/command-a-plus-05-2026-bf16` @ `5fb6fde5`, `Cohere2VisionForConditionalGeneration`; only the text backbone (`Cohere2MoeForCausalLM`) is loaded |
| Shape | 32 layers, hidden 4096, 128 query heads x 128 head_dim, 8 KV heads, vocab 262144 |
| MoE | 128 experts, top-8, expert intermediate 4096, plus an always-on shared SwiGLU MLP (intermediate 16384), averaged with the routed output |
| Size | ~218B parameters total, ~25B active per token |
| Attention | 3 sliding (window 4096) + 1 full, repeating; checkpoint supports 200k positions |
| Hardware | 4x GB200, 189 GB HBM each, aarch64, NVLink all-to-all |
| Parallelism | TP4, ~110 GB of weights per GPU |

Per decoded token each GPU reads about 12.5 GB of weights (8 of 128 experts per
layer, the dense projections and the LM head). At a practical 6.5-8 TB/s that is a
floor of about 1.6-1.9 ms/token, so roughly 500-550 tok/s for a single BF16 stream.
At concurrency 64 nearly every expert is hit each step, so a step reads about
109 GB per GPU and the floor is about 14-17 ms/step.

## Results

All measured with `test/command_a_plus/bench_http.py` against the live server:
128 tokens per request, `ignore_eos` so every concurrency level does identical
work, greedy (temperature 0). Single-request numbers in tok/s:

| Build | What it adds | conc 1 | TTFT |
|---|---|---|---|
| Eager (no graphs) | baseline | 22 | 59 ms |
| Overnight (changes 1-5) | decode CUDA graphs, sampler fix, async scheduling, one all-reduce per layer | 132 | 57 ms |
| Fused | change 6: fused kernels and concatenated weights | 191 | — |
| v2 | changes 7-8: tuned MoE tiles, flashinfer one-shot all-reduce | 229.5 | 42 ms |
| v3 | changes 9-10: replicated embedding, one-kernel KV write | 241 | 37 ms |
| v4 | changes 11-12: split-K output projection, split-vocab sampler prep | 266 | 42 ms |
| **v5 (current)** | changes 13-15: tight MoE grid, no attention-output copy, fused routing | **277** | **42 ms** |

v5 under load (aggregate and per-request tok/s):

| concurrency | 1 | 8 | 32 | 64 |
|---|---|---|---|---|
| aggregate | 277 | 1517 | 2086 | 2380 |
| per request | 277 | 191 | 66 | 38 |

So **12.6x single-stream over eager, and 2.1x over the overnight build**. A
32,768-token generation (change 16) runs at 283 tok/s on average including prefill,
so long contexts don't slow single-stream decode noticeably.

**Concurrency 32-64 is CPU-bound, not GPU-bound.** At concurrency 64 the GPUs are
about 52% busy while the TP rank-0 worker sits at 100% CPU, spending about 0.35 ms
of Python per request per step in the engine. That is ~22 ms of the ~27 ms step,
against a GPU floor of ~15 ms. None of the kernel work below moves this; it needs
engine work (see "What is left").

## Validation

Every build from the fused one onward was checked three ways before it shipped.

1. **Against Hugging Face.** `validate_checkpoint.py` compares mstar logits at
   every step against an HF transformers reference (3 prompts x 32 tokens):

   | Build | RMSE mean / max | min cosine | argmax == HF | free-running greedy == HF |
   |---|---|---|---|---|
   | Unfused (`COMMAND_A_PLUS_UNFUSED=1`) | 0.0541 / 0.1934 | 0.99742 | 96/96 | 3/3 |
   | v2, v3 | 0.0510 / 0.1745 | 0.99807 | 96/96 | 3/3 |
   | v4, v5 | 0.0517 / 0.1451 | 0.99807 | 96/96 | 3/3 |

   The fused builds are slightly closer to HF than the unfused path, most likely
   because the routed sum and the all-reduce now accumulate in fp32 and round to
   bf16 once. v3 is numerically identical to v2
   and v5 to v4, as expected: the changes between them are exact.

2. **Server determinism.** `parity_http.py --compare` against the previous build:
   6/6 greedy outputs identical, both sequential and concurrent, for every build
   from v2 through v5.

3. **Tests.** See "Test status".

## Changes

Changes 1-5 are from the overnight bring-up; 6-16 are later.

### 1. Every decode step returned HTTP 500 (`mstar/worker/worker.py`)

`Worker._send_outputs` sent new-token counts from every TP rank. The conductor
only absorbs them from rank 0, and ranks 1-3 had already released the tensors, so
the lookup raised `KeyError`. Gated the send on `outputs.is_first_tp_rank`.

### 2. Decode ran eager (`mstar/model/command_a_plus/submodules.py`)

`CommandAPlusLLMSubmodule.get_cuda_graph_configs` returned nothing, so every
decode step paid full launch overhead — ~43 ms/step against ~7.4 ms of actual GPU
work. It now returns a `BatchedCudaGraphConfig` for the decode walk with
`compile=False`: graphs replay the same eager kernels, and `torch.compile` stays off.
Capture covers batch sizes 1/2/4/8/16/32/64 x 2 slots on all 4 ranks.

### 3. Greedy decode was nondeterministic under CUDA graphs (`mstar/engine/resources/sampler/utils.py`)

With graphs on, the same greedy prompt produced about 6 distinct completions in 10
runs. Eager encoded `temperature == 0` as a one-hot at the argmax; the graph path
rewrote greedy rows to `(temperature=1, top_k=1)` and sampled through a Philox
stream. `top_k=1` only returns the argmax when it is unique, and bf16 logits tie
often (spacing 0.0625 at magnitude ~10), so the draw picked between tied tokens at
a per-run RNG offset. The graph path now keeps `temperature = 0` and always passes
`include_greedy=True`, decided per row on device, so ties resolve to the lowest
vocab index in both paths. This affected every model using the graph sampler.

### 4. One all-reduce per layer instead of three (`language_model.py` and the TP components)

Command A+'s decoder layer is a parallel block, so attention, routed-expert and
shared-MLP partials are summed on-rank and reduced once:
`x + AR(attn + (routed + shared) / 2)`. Done with a `reduce_results` flag (default
`True`) on `ParallelAttention`, `ParallelGatedMLP` and `ParallelSparseMoeBlock`.
Cut all-reduces from 97 to 33 per step.

### 5. CUDA toolchain (pod-only, not a repo change)

The fused MoE alignment op JIT-builds against the local toolkit. The system `nvcc`
(13.3) fails against cu128 torch and silently falls back to a torch version that
adds ~134 device syncs per step. Use a CUDA 12.8 redist toolkit, with the pip
`nvidia-*` headers (cuSPARSE etc.) symlinked into its `include/`, and **always
export `CUDA_HOME` pointing at it.**

### 6. Fused decoder layer (`mstar/model/command_a_plus/kernels.py`, `language_model.py`)

Eager decode ran ~1,350 kernels per step. `CommandAPlusLanguageModel.fuse_for_inference()`
(called after weight load unless `COMMAND_A_PLUS_UNFUSED` is set) restructures each layer:

- **Concatenated weights.** `[qkv | shared gate_up | router]` becomes one
  `input_weight` and `[o_proj | shared down]` becomes one `output_weight`, so each
  layer runs two dense GEMMs instead of five. The original parameters are rebound
  as views into the concatenated storage, so nothing is duplicated.
- **Triton kernels**, each rounding to bf16 at the same points as the eager chain
  it replaces:
  - `add_layernorm`: residual add plus bias-free LayerNorm with fp32 statistics,
    replacing ~8 kernels per norm. The next layer's norm is fused with this layer's
    residual add.
  - `sigmoid_topk`: router top-8, sigmoid and renormalisation in one kernel,
    lowest index winning ties, replacing `torch.topk`'s bitonic sort.
  - `silu_mul_into`: the shared MLP's SwiGLU, writing the 1/2 averaging factor
    in-kernel (a power of two, so exact).
  - `moe_combine`: `base + scale * sum_k(routed)` in fp32, rounded once.

Result at batch 1: 1,347 kernels / 6.9 ms GPU time per step down to 568 / 4.0 ms.

### 7. Tuned MoE tiles (`mstar/utils/fused_moe/{kernels,runner,tune}.py`, `configs/`)

`get_config` now reads per-device tuned JSON configs (with separate up and down
GEMM settings sharing one `BLOCK_SIZE_M`) and falls back to `get_default_config`.
`python -m mstar.utils.fused_moe.tune --experts 128 --inter 1024 --hidden 4096 --top-k 8`
sweeps tiles per batch size, timing inside CUDA graphs with rotated routings so
weights come from HBM rather than L2. The GB200 result is
`configs/E=128,N=2048,K=4096,device_name=NVIDIA_GB200.json`.
**That file matches the repo's `*.json` gitignore rule; commit it with `git add -f`.**
Shared with Qwen3-Omni, whose fused-MoE tests pass.

### 8. One-shot all-reduce (`language_model.py`)

The per-layer all-reduce uses flashinfer's trtllm one-shot Lamport kernel
(`allreduce_fusion`, `kAllReduce`, fp32 accumulation) for up to 2048 tokens, and
NCCL above that or when `COMMAND_A_PLUS_NCCL_ALLREDUCE=1`. The workspace is created
collectively in `fuse_for_inference`.

### 9. Replicated embedding (`language_model.py`, `submodules.py`)

The vocab-parallel embedding needed an NCCL all-reduce per step (~45 us). Each rank
now holds the full table (gathered once with `all_gather_into_tensor`, ~2 GB per GPU)
and looks tokens up locally through `model.embed()`.

### 10. One-kernel KV write (`mstar/engine/resources/kv/cache.py`)

`_kv_scatter_nhd_eager` used two `index_put` kernels per layer (K and V). On CUDA
with the NHD layout it now runs one Triton kernel writing both, accepting strided
K/V views; other devices keep the old path. Shared engine code.

### 11. Split-K output projection (`kernels.py`)

cuBLAS picks a non-split kernel for the skinny `[T, 8192] x [8192, 4096]` decode
GEMM and reaches only ~3 TB/s. `splitk_linear` splits K eight ways and writes fp32
partials, which `moe_combine` sums along with the routed experts, so no extra kernel
is added. 23.6 us down to 12.4 us at batch 1. Used for up to 16 tokens; cuBLAS above.
The input GEMM and the LM head were benchmarked too, and cuBLAS is already at
5.7-7 TB/s there.

### 12. Split-vocab sampler prep (`mstar/engine/resources/sampler/utils.py`)

The temperature-softmax prep kernel ran one program per row over the 262k vocab:
at batch 1, one SM doing three passes. It is now two kernels over 2048-token chunks
(per-chunk max, lowest-index argmax and sum of exponentials, then a combine-and-write
pass). 57 us down to 4.4 us at batch 1, 66 us to 35 us at batch 64. This also removes
the `@triton.autotune` and the synchronisation that autotuning needed on first launch.
Shared engine code; all sampler tests pass, plus new ones in `test_fused_prep.py`.

### 13. Tight MoE grid (`mstar/utils/fused_moe/kernels.py`)

The alignment buffer is sized for padding on all 128 experts, and the GEMM grid was
sized from it: 3,872-7,744 thread blocks for one token, of which ~8 do work.
`invoke_fused_moe_kernel` now bounds the grid by `slots + min(E, slots) * (BLOCK_M - 1)`,
the most padding the active experts can produce. Large batches are unchanged.

### 14. No attention-output copy (`kernels.py`, `language_model.py`)

The row GEMM's input was built by copying the attention output into a buffer next
to the shared-MLP activation (~2 us per layer). `splitk_linear` now takes the two
tensors separately and each split reads from whichever one its K range falls in.

### 15. Fused routing (`kernels.py`, `mstar/utils/fused_moe/runner.py`)

For up to 128 routed slots (16 tokens at top-8), `route_align` does `sigmoid_topk`
and `moe_align_block_size` in one single-program kernel, replacing three kernels
(~5.7 us per layer). `fused_experts` gained an optional `alignment=` argument for
precomputed alignment, and `moe_block_m()` reports the block size to align to.
Tested against the two-step path, including ties.

### 16. Serving features (`command_a_plus_model.py`)

- **Multi-turn chat.** `model_kwargs={"messages": [{"role", "content"}, ...]}` with
  `system` / `user` / `assistant` turns is passed through the chat template. Plain
  `text` requests are unchanged.
- **Output length.** The decode loop's hard bound was `get_max_output_tokens()`
  with no arguments, i.e. the global 2048 default, so every request stopped at 2048
  tokens whatever it asked for. It is now `max_seq_len`; the request's
  `max_output_tokens` (default 2048) decides when to stop. A request whose prompt
  plus explicit `max_output_tokens` exceeds `max_seq_len` is rejected with a 400
  naming both numbers, because the CUDA-graph KV buffers are sized to `max_seq_len`.

### Smaller changes

- `compute_logits` skips the multiply when `logit_scale == 1.0`.
- `test/command_a_plus/graph_parity.py` was deleted: it deadlocked capturing NCCL
  collectives, and `sampler_determinism.py` covers what it was written for.

## Running it

Serving config (`command_a_plus_tp4.yaml`):

```
model: command_a_plus
max_seq_len: 65536
model_kwargs:
  max_seq_len: 65536
  checkpoint_dir: /path/to/command-a-plus-05-2026-bf16

resources:
  kv_cache:
    max_num_pages: 8192    # 1M tokens shared across requests, ~32 GB per GPU

node_groups:
  - node_names: [LLM]
    ranks: [0, 1, 2, 3]
    tp_size: 4
    sp_size: 1
    graph_walks: [prefill, decode]
```

Serve:

```
mkdir -p /tmp/cmda-sock
CUDA_HOME=/path/to/cuda-12.8 MSTAR_TP_ASYNC_SCHED=1 setsid nohup mstar-serve \
  --config command_a_plus_tp4.yaml --host 127.0.0.1 --port 8000 \
  --tensor-comm-protocol SHM --socket-path-prefix /tmp/cmda-sock/s \
  > serve.log 2>&1 &
```

**Never start a second server.** Two servers share the socket prefix; the second
OOMs and leaves the first one's result channel broken (requests finish, outputs
never arrive). Startup takes 1-8 minutes depending on page cache.

To stop it, kill the worker processes (`pkill -9 -f <venv>/bin/python`, run as a
standalone command so the pattern can't match your own shell) and wait about 10 s
for the GPUs to free.

Request API: `POST /generate` (form fields `text`, `streaming`, `output_modalities=text`,
`model_kwargs` as JSON). Command A+ has no OpenAI adapter, so `/v1/chat/completions`
returns 404. The model defaults to temperature 0.9, top-p 0.95 and repetition
penalty 1.04, and it reasons before answering
(`<|START_THINKING|>...<|END_THINKING|><|START_TEXT|>...<|END_TEXT|>`), so long
answers need a generous `max_output_tokens`.

## Env knobs

| Variable | Effect |
|---|---|
| `CUDA_HOME` | Required; must point at a CUDA 12.8 toolkit (change 5). Builds the fused MoE align op |
| `MSTAR_TP_ASYNC_SCHED=1` | TP leader speculates step N+1 during step N. +18% at conc 1, +65% at conc 32. No effect on determinism |
| `COMMAND_A_PLUS_DISABLE_CUDA_GRAPH=1` | Decode runs eager |
| `COMMAND_A_PLUS_UNFUSED=1` | Skip `fuse_for_inference` (change 6 onward); the unfused reference path |
| `COMMAND_A_PLUS_NCCL_ALLREDUCE=1` | Use NCCL instead of the flashinfer one-shot all-reduce (change 8) |

## Scripts

All under `test/command_a_plus/`, run from the repo root.

| Script | What it does |
|---|---|
| `validate_checkpoint.py` | `reference` mode writes HF logits and greedy outputs (run alone, needs `accelerate`); `mstar` mode (under `torchrun --nproc-per-node 4`) compares against them. The validation runs above used `--tokens 32 --max-rmse 1 --max-abs 10 --min-cosine 0.9` and were compared by RMSE, cosine and argmax |
| `parity_http.py` | Greedy outputs and speed for 6 prompts from a live server; `--compare` diffs against an earlier run |
| `bench_http.py` | The benchmark above. `--concurrency 1 8 32 64 --max-tokens 128` |
| `../sampling_test/sampler_determinism.py` | Regression test for change 3; no model load, ~8 s |
| `profile_decode.py` | Per-step latency by batch size plus a `torch.profiler` breakdown |
| `test_fused_kernels.py` | Unit tests for every kernel in `kernels.py` and the KV scatter, plus fused-vs-unfused backbone checks |

The HF reference files used above are not checked in; regenerate them with
`reference` mode before validating further changes.

## Where the remaining time goes

v4 at batch 1, from nsys with `--cuda-graph-trace=node`: 3.34 ms of kernel time per
step and only 137 us idle, so the GPU is almost never waiting on the CPU here.

| | us/step | note |
|---|---|---|
| Routed-expert GEMMs (`fused_moe_kernel`, 64 calls) | 1255 | 5.1 TB/s |
| Input GEMM (cuBLAS, 32 calls) | 561 | 6.0 TB/s |
| Output projection (`_splitk_linear_kernel`) | 384 | 5.6 TB/s |
| Attention (prefill-style kernel + split-KV merge) | 327 | ~10 us/layer even at short context |
| All-reduce (one-shot) | 271 | 4-12 us each, mostly waiting on rank skew |
| LM head | 80 | 6.8 TB/s |
| ~20 small kernels per layer | ~460 | 1.3-3 us each, mostly launch latency |

The weight-streaming GEMMs are within ~1.3x of a practical bandwidth ceiling. v5
since removed the attention-output copy and two of the small routing kernels per layer.

## What is left

Roughly in value order:

1. **Engine CPU overhead at high concurrency.** The only thing that moves
   throughput past ~2.4k tok/s: about 0.35 ms of per-request Python per step on TP
   rank 0. Batching the per-request bookkeeping and plan updates is engine work,
   not model work.
2. **FP8 weights.** Halves bytes per token: ~1.6-1.8x single-stream and more room
   for KV cache. flashinfer 0.6.18 ships trtllm FP8 block-scale MoE kernels; about
   1-2 days. NVFP4 is several days.
3. **More kernel fusion** (~5-10% at batch 1): SiLU into the down GEMM's A-load
   (~1.8 us/layer), RoPE into the KV write (~1.4 us/layer), and avoiding the
   split-KV merge for short contexts (~2.9 us/layer).
4. **Seen-token mask copies.** The repetition penalty (default 1.04) copies each
   request's 262k-entry mask into and out of the graph buffers every step, outside
   the graph (~45 us/step).
5. **Prefill CUDA graphs** (`PackedCudaGraphConfig`): about half a day plus a parity check.
6. **Vision encoder.** Not implemented; the SigLIP tower (27 layers) and SwiGLU
   projector weights are skipped at load. M* has SigLIP in `pi05` and `bagel` to build from.

## Running on a preemptible devbox

A Kueue devbox can be preempted, and the pod comes back on another node with an
empty local disk. Keep the checkpoint on shared storage, and script the rebuild
(venv, CUDA 12.8 toolkit with the pip `nvidia-*` headers linked in, serving
config) so a new pod is serving again in about 10 minutes.

## Test status

| Suite | Result |
|---|---|
| `test/command_a_plus/`, `test/sampling_test/`, `test/modular/test_qwen3_omni_fused_moe.py`, `test/modular/test_worker_thread_device.py` | 84 passed, 5 skipped, 134 subtests |
| `test_ragged_attention.py` | 4 failed — **pre-existing**, reproduced on a pristine worktree of HEAD; unrelated to these changes |

## Files touched

```
mstar/worker/worker.py                                   change 1
mstar/model/command_a_plus/submodules.py                 changes 2, 9
mstar/engine/resources/sampler/utils.py                  changes 3, 12
mstar/model/command_a_plus/components/language_model.py  changes 4, 6, 8, 9, 11, 14, 15
mstar/model/components/distributed/attention.py          change 4 (reduce_results flag)
mstar/model/components/distributed/mlp.py                change 4 (reduce_results flag)
mstar/model/components/moe.py                            change 4 (reduce_results flag)
mstar/model/command_a_plus/kernels.py                    new: changes 6, 11, 14, 15
mstar/model/command_a_plus/command_a_plus_model.py       changes 6, 16
mstar/engine/resources/kv/cache.py                       change 10
mstar/utils/fused_moe/kernels.py                         changes 7, 13
mstar/utils/fused_moe/runner.py                          changes 7, 15
mstar/utils/fused_moe/__init__.py                        change 15 (exports moe_block_m)
mstar/utils/fused_moe/tune.py                            new: change 7
mstar/utils/fused_moe/configs/E=128,N=2048,K=4096,device_name=NVIDIA_GB200.json   new: change 7 (gitignored; add -f)
test/command_a_plus/test_fused_kernels.py                new
test/command_a_plus/test_integration.py                  changes 9, 16
test/command_a_plus/validate_checkpoint.py               change 9 (embed())
test/sampling_test/sampler_determinism.py                new
test/command_a_plus/bench_http.py                        new
test/command_a_plus/parity_http.py                       new
test/command_a_plus/profile_decode.py                    new
test/sampling_test/test_fused_prep.py                    new: change 12
```
