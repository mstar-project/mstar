---
name: engine-resources
description: Pitfalls when writing or changing an engine Resource under mstar/engine/resources/ — double-buffering against the async worker, pre-plan, padding rows inside CUDA-graph capture buckets, and attention-backend (FlashInfer, FlashAttention) quirks. Use before adding a new resource kind, changing how a resource stages buffers in plan(), or swapping an attention kernel.
---

# Writing an engine resource

A resource's `plan` writes device buffers that a captured graph later reads, while the async worker is already planning the next step. Mistakes here rarely raise; they make a replay read the wrong layout, or read out of bounds several frames away from the cause. Read `Resource` in [mstar/engine/resources/base.py](../../../mstar/engine/resources/base.py) and an existing manager (`kv/manager.py`, `attn/flashinfer.py`) before starting, and the [async-worker skill](../async-worker/SKILL.md) for what is in flight when `plan` runs.

## Buffers and the async worker

**`force_double_buffer` guards a race between consecutive steps; pre-plan is not what causes it.** If `plan` stages a step's layout into a reused buffer that a replay reads (index tensors, block tables, FlashInfer's per-wrapper plan buffers), `plan(N+1)` can overwrite it before step N's kernels have consumed it, and N replays with N+1's layout. The main runner already double-buffers any resource that `supports_preplan`; the flag extends that to runners with no pre-plan path, notably the piecewise runner. If your resource has the hazard, set it. The docstring on `Resource.force_double_buffer` is authoritative.

**`supports_preplan` pays off even with no kernel-side plan to hoist.** Host work counts. On Whisper, building a page table inline cost 0.052 ms on the critical path, while promoting a pre-planned one cost 0.009 ms.

## Capture buckets

- Batch buckets are geometric (`DEFAULT_CAPTURE_BATCH_SIZES` in `cuda_graph_runner.py`: 1, 2, 4, …, 64), so a replay usually runs wider than the live batch.
- **A padded row must be well-formed, not zeroed.** The replay always runs the bucket's full width. Point padding at `SINK_PAGE` (see `KVPlanState.copy_`) and give it a length of at least 1 unless the target kernel has handling for zero-length rows.
- **Take the step's width from the bucket when captured, and from the live batch otherwise.** Eager buffers are reused and sized to the widest batch seen. Padding an eager step out to the buffer width makes the index tensors describe query rows that `q` does not have, and the kernel reads off the end. That surfaces as an asynchronous illegal memory access far from the cause, so a cheap `q.shape[0] != expected` assertion at the call site pays for itself. A warmup at higher concurrency followed by a timed batch of 1 is what exposes this.

## Attention backends

- **Verify a kernel swap numerically, not just that it runs.** FlashInfer at head_dim 72 runs and returns wrong values (see `check_flashinfer_head_dim` in `attn/wrappers.py`).
- **FlashInfer accepts `(k_cache, v_cache)` as a tuple of 4-D tensors**, not only the paired 5-D layout. With the paired layout it calls `.unbind(dim=1)` on every attention call (`flashinfer/utils.py::_unpack_paged_kv_cache`) and hands the kernel strided views. A split K/V layout avoids that.
- **FlashAttention's causal mask is bottom-right aligned** (the new queries are the last positions); `F.scaled_dot_product_attention(is_causal=True)` is top-left aligned. They disagree whenever `q_len != kv_len`, so build the mask explicitly in a reference or you will "find" an FA bug that isn't there.
- **Check that an FA build has paged KV at all.** It is a compile-time flag (`FLASH_ATTENTION_DISABLE_PAGEDKV`): a build without it imports fine and raises on the first `page_table=` call. Upstream FA2 also requires `page_block_size % 256 == 0`; a paged FA3 build accepts 16–512.

## Scope

KV layout is engine-side Python only; a cache-layout change needs no `rust/` change (invariant 5 covers the graph runtime, wire format and edges). New resource kinds should still land as their own PR under the model that needs them; see "Splitting work across PRs" in [AGENTS.md](../../../AGENTS.md). Document the resource in `docs/adding_models.rst` (invariant 9).
