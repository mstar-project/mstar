---
name: engine-resources
description: Pitfalls when writing or changing an engine Resource under mstar/engine/resources/ — double-buffering against the async worker, pre-plan, padding rows inside accelerator graph capture buckets, and attention-backend (FlashInfer, FlashAttention) quirks. Use before adding a new resource kind, changing how a resource stages buffers in plan(), or swapping an attention kernel.
---

# Writing an engine resource

A resource's `plan` writes device buffers that a captured graph later reads, while the async worker is already planning the next step. Mistakes here rarely raise; they make a replay read the wrong layout, or read out of bounds several frames away from the cause. Read `Resource` in [mstar/engine/resources/base.py](../../../mstar/engine/resources/base.py) and an existing manager (`kv/manager.py`, `attn/flashinfer.py`) before starting, and the [async-worker skill](../async-worker/SKILL.md) for what is in flight when `plan` runs.

## Which resource for which state

Entries marked *open PR* are not on `main` yet. Check the checkout; if your port needs one, stack on that PR or say in the description which resource you used instead.

| state | resource | expresses |
|---|---|---|
| a cache that grows with the request | paged KV (`KVSpec` + `PagedKVConfig`), optionally a `RetentionPolicy` | any length; retention keeps a protected prefix plus the newest pages |
| attention within one forward, no cache | `RaggedAttentionSpec` | block-diagonal per span, optionally causal |
| | `RaggedCrossAttentionSpec` | one contiguous key range per query segment |
| | `RaggedBlockCausalAttentionSpec` (#381, *open PR*) | each block attends its span's prefix of blocks |
| a block-causal stream, running only new tokens | paged KV + `BlockCausalAttentionSpec` | a row's new tokens attend its cached blocks and their own |
| a fixed-size cache per request: a sink plus a sliding window, optionally after a shared read-only prefix | bounded KV (`BoundedKVSpec`, #383, *open PR*) | rows at any position in one batch; per-row source; a per-distance score term |
| fixed-size opaque state per request (recurrent, conv tails) | `RecurrentStatePool` | slots read and written through the step's addressing (`gather` / `scatter_`) |
| per-request adapter weights (LoRA) | `LoraSpec` / `LoraAdapterPool` | a slot per resident adapter, held for a request's lifetime; the step's tokens grouped by slot for the kernels |
| frame rings (Waypoint) | `RingKVManager` | to be merged into bounded KV |

A resource a deployment turns on (LoRA) sets `NodeResourceSpec.opt_in` and is declared in `get_node_resources` like any other; it is built only when the serving YAML's `resources:` names it. A resource whose nodes share no state, and may sit in different lockstep instances (a vision tower beside a TP LLM), sets `NodeResourceSpec.per_node`: `serving_resources` splits it into an instance per node (`for_node`), keyed `<key>@<node>` in every process, so the conductor opens a request with a config under each instance's key; the YAML block names the declared key. A resource whose step on a node is just the node's segments sets `Resource.steps_from_segments`, and the node declares nothing for it: the engine adds the step from the node's segments (`with_segment_steps`), and a piecewise region's runner does the same from the region's own segments.

A resource that holds per-request state gives models operations driven by the step (`gather` / `scatter_`, `attend` over the current plan). It never asks a model to index its storage or to hold and select rows of its plans.

Prototype a new kernel standalone against a dense reference before writing the resource, and for retention, check against a brute-force simulation of the reference's cache on token ids.

## Buffers and the async worker

**`force_double_buffer` guards a race between consecutive steps; pre-plan is not what causes it.** If `plan` stages a step's layout into a reused buffer that a replay reads (index tensors, block tables, FlashInfer's per-wrapper plan buffers), `plan(N+1)` can overwrite it before step N's kernels have consumed it, and N replays with N+1's layout. The main runner already double-buffers any resource that `supports_preplan`; the flag extends that to runners with no pre-plan path, notably the piecewise runner. If your resource has the hazard, set it. The docstring on `Resource.force_double_buffer` is authoritative.

**`supports_preplan` pays off even with no kernel-side plan to hoist.** Host work counts. On Whisper, building a page table inline cost 0.052 ms on the critical path, while promoting a pre-planned one cost 0.009 ms.

## Resident planning

A walk a submodule names in `resident_walks` runs its captured steps off per-request tables on the device (`resources/resident.py`): `plan` stages nothing per row, and kernels recorded around the forward fill the buffers the forward reads. A resource joins by overriding `resident_refusal` and keeping a table; the base refuses, so a new resource is host-planned on its node until then. Per-step state is keyed by resource key, never one per planner: each owner reads its own stream's label (`ResidentWrites.streams[key]`), each KV pool its own block table and gathered lengths, and a pool the walk declares at span 0 is read only (no write addressing, no advance). An attention reports whether it read its pages off the tables (`reads_tables`) or through a host plan (`host_planned`); a bucket skips the host plan only if none needed one. Between the two is `lengths_only`: the graph builds a pool's paged arrays off the tables and the host plans only FlashInfer's schedule from the lengths (`supports_gather_resident`), so that pool's KV plan hands back host lengths and no page lists. Anything that changes a row other than a resident commit has to reach `ResidentPlanner.mark_stale`, or the next step reads the old row. In `KVManager` that is what `_touch_plan_cache(rid)` does. Writes to a table stay on the node's stream, in step order: the plan stream only writes the step's own staging. A slot is a cache entry, not a finite resource: an offload gives it up and a step short of slots takes the least recently run row's, so a table can only hold what a whole-row rewrite can rebuild from the host. The sampler shows the split: its per-request scalars are cells (`sampler/resident.py`), but the RNG offset, last token, seen-token mask and history only the device knows, so they stay by the sampler's own rows, which a request keeps until it is removed, and a resident step reads them through an index row of those rows that its stage uploads when the batch changes.

## Stateful resources move together

KV pages and recurrent (GDN/Mamba) state slots are two forms of the same per-request state. A lifecycle feature added to one usually needs its analogue in the other. Forking for a request that branches, offload/reload, what a partially-run (chunked) step leaves behind: when you change one, check the others. Recurrent forking was initially missed when KV forking learned chunked-prefill semantics.

## Capture buckets

- Batch buckets are geometric (`DEFAULT_CAPTURE_BATCH_SIZES` in `accelerator_graph_config.py`: 1, 2, 4, …, 64), so a replay usually runs wider than the live batch.
- **A padded row must be well-formed, not zeroed.** The replay always runs the bucket's full width. Point padding at `SINK_PAGE` (see `KVPlanState.copy_`) and give it a length of at least 1 unless the target kernel has handling for zero-length rows.
- **Take the step's width from the bucket when captured, and from the live batch otherwise.** Eager buffers are reused and sized to the widest batch seen. Padding an eager step out to the buffer width makes the index tensors describe query rows that `q` does not have, and the kernel reads off the end. That surfaces as an asynchronous illegal memory access far from the cause, so a cheap `q.shape[0] != expected` assertion at the call site pays for itself. A warmup at higher concurrency followed by a timed batch of 1 is what exposes this.

- **A resource whose kernels do nothing for most rows can sit a step out.** Set `lean_capture`, answer `Resource.lean_for(rids)` from state every rank sets alike, say which capture kinds take its twin (`lean_captures`), and stage nothing in `plan` when `StepContext.lean` names the resource: the engine captures each bucket whose step drives the resource again without its kernels and replays that twin for a step whose rows all let it run lean (the LoRA pool, the sampler's argmax). Each rank chooses from its own state, so its lean and plain kernels must issue the same collectives. Decide from `ctx.lean` alone, not from the lease's key or the rows: under the plain bucket the kernels were recorded and need their addressing. If its callers read its plan inside the forward, also override `outside_step`, which the engine enters around compile warm-ups outside any step (the LoRA pool's runs them lean).
- **A kernel that allocates its output with `empty` leaves a bucket's padded tail holding the previous replay's values.** Not zeros, and not random: a captured graph keeps its intermediates in a private pool, so the tail is deterministically whatever the last replay wrote at that address. Varlen kernels that fill only the `cu_seqlens` range do this (the vendored `causal_conv1d_fn` is one), and in a residual stack those values compound from layer to layer until something downstream sees `inf`.
- **Mask such a tail with a select, not a multiply.** `out * mask` is `inf * 0` on exactly the elements being discarded, which is `NaN` — arithmetic masking manufactures what it was added to remove, and the result looks like the kernel's fault. Use `torch.where(keep, out, zeros)`.
- **A kernel's launch geometry can be non-capturable even when the kernel is fine.** `causal_conv1d_fn` derived its Triton grid inside the grid lambda from `query_start_loc.diff().to("cpu")` — a device sync — and then copied the result H2D, which CUDA forbids on a capturing stream. Precompute the geometry in `plan`, where the spans are already Python ints, into buffers with stable addresses; size the grid to the bucket's worst case and point the surplus programs at the kernel's skip sentinel so it stays constant across replays.

## Attention backends

- **Attention over several short key ranges tiles them as one sequence** (key to range by index math) and selects the pointer before loading: per-range tile loops and a masked load per source each cost about a quarter of the kernel on the token2wav DiT.
- **A wrapper over FlashInfer plans through its `Workspace`** (`attn/workspace.py`), not on the FlashInfer wrapper directly. A plan's float workspace need scales with the bucket's rows and heads and the GPU's SM count, and it can only grow before capture: a graph records the address. A `Workspace` grows at the first plan that overflows it, up to `Resource.seal_capture_buffers`, which the engine calls once every bucket is sized.
- **Verify a kernel swap numerically, not just that it runs.** FlashInfer at head_dim 72 runs and returns wrong values (see `check_flashinfer_head_dim` in `attn/wrappers.py`).
- **FlashInfer accepts `(k_cache, v_cache)` as a tuple of 4-D tensors**, not only the paired 5-D layout. With the paired layout it calls `.unbind(dim=1)` on every attention call (`flashinfer/utils.py::_unpack_paged_kv_cache`) and hands the kernel strided views. A split K/V layout avoids that.
- **FlashAttention's causal mask is bottom-right aligned** (the new queries are the last positions); `F.scaled_dot_product_attention(is_causal=True)` is top-left aligned. They disagree whenever `q_len != kv_len`, so build the mask explicitly in a reference or you will "find" an FA bug that isn't there.
- **Check that an FA build has paged KV at all.** It is a compile-time flag (`FLASH_ATTENTION_DISABLE_PAGEDKV`): a build without it imports fine and raises on the first `page_table=` call. Upstream FA2 also requires `page_block_size % 256 == 0`; a paged FA3 build accepts 16–512.

## Ragged (cacheless) attention

- **A ragged resource plans one layout per label.** A step declares, per request, the spans its layers attend over as `(label, length)` segments; the manager plans each label across the batch and a layer asks for its label's callable. Spans with different head geometries need separate resources, since a resource has one head layout.
- **A wrapper skips a plan it already holds.** `_RaggedPrefillBase._plan` compares the layout with what that (label, slot)'s schedule describes, since `run` only reads it. A new wrapper on that base gets this for free; anything that invalidates a schedule without a plan has to clear it (`forget_plan`, as a dropped pre-plan does: its H2Ds ran on the plan stream and nothing orders the replacing step against them).
- **Cross-attention between spans of a request is its own kind** (`RaggedCrossAttentionSpec`, #372): its step names `(q_label, kv_label)` pairs and plans each pair under `cross_label(q, kv)`. It keeps the self-attention resource's API unchanged; extending a resource that almost fits this way beats a model calling SDPA (see AGENTS.md invariant 1).

## Scope

KV layout is engine-side Python only; a cache-layout change needs no `rust/` change (invariant 5 covers the graph runtime, wire format and edges). New resource kinds should still land as their own PR under the model that needs them; see "Splitting work across PRs" in [AGENTS.md](../../../AGENTS.md). Document the resource in `docs/adding_models.rst` (invariant 9).
