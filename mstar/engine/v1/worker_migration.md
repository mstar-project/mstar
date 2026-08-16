# Worker migration to the v1 engine

Plan of record for taking `mstar/worker/` off the old engine and onto
`mstar/engine/v1/engine.py`, then restructuring speculative execution around
it. Phase 0 is a prerequisite for Phase 1: the speculation rework is much
smaller once the worker knows exactly one engine.

Everything the worker does for the OLD engine is being removed, not ported,
unless named below.

**Status**: Phase 0 and Phase 1 applied in one pass over the worker. Three
things landed differently from the plan, all noted inline below:
`prepare_inputs` stays on the main thread (see Phase 1), `check_ready` drives
reload itself, and `prepare_inputs` for a non-speculated batch runs on the
main thread before submission.

---

## Phase 0 — one engine

### 0.1 Engine surface

| worker calls today | v1 |
| --- | --- |
| `execute_with_max_batch_size` | **gone** — see 0.2 |
| `execute_batch` / `prepare_batch` / `plan_batch` / `execute_forward` | `prepare_inputs` → `exec` → `postprocess_batch` (`exec_and_postprocess` pairs the last two) |
| `finalize_batch` | `finalize_batch` (was `publish`; returns rid -> {resource label -> PublishedInfo}) |
| `check_stop_for_batch` | same name; returns `dict[rid, set[loop]]`, failures land on `batch.failed_requests` |
| `reserve_replay_slot` / `pre_plan_for_batch` / `reset_pre_plan_for_batch` | same names, take `ExecutingBatch` |
| `capabilities.requires_kv_cache` | **deleted** — only consumer was `_compute_store_write_policy`, which is dead |
| `node_resources(node)` | **deleted** — see 0.3 |
| `lru_tracked_nodes` | **deleted** — see 0.3 |
| `set_alloc_write_policies` | **deleted**, dead code |
| `pause_request` / `resume_request` | **deleted**, dead code (holdover from a vLLM-style engine; the real mechanism is offload/reload) |

`NodeBatch` / `NodeOutput` → `ExecutingBatch` + a plain `dict[rid,
NameToTensorList]`. Per-batch flags move onto the batch:
`output.allocation_failed` → `batch.allocation_failed`,
`output.failed_requests` → `batch.failed_requests`.

One `Engine` instance serves every node (it already holds many submodules and
shares resources across them), so `EngineManager`'s engine-type and stateless-
flavor grouping collapses to a single construction. `get_engine(node)` returns
the same object for all nodes.

`load_model` differs from the old signature:

- `kv_cache_config` is gone; KV shape comes from the `KVSpec` in
  `model.get_node_resources()`, which is passed as `specs`.
- `default_sampling_config` is gone; v1 takes per-request `SamplingReqConfig`
  through `add_request(overrides=...)`, so `EngineManager.add_request` grows
  that argument and the model's resolved sampling config moves there.

### 0.2 Max batch size → microscheduler

The old engine split an oversized batch into minibatches internally
(`execute_with_max_batch_size`). That is the wrong home: the minibatches run
back to back with no pipelining or async between them.

Instead the microscheduler keeps a **backlog** of batches it should schedule in
succession. A non-empty backlog supersedes the usual scheduling path, so the
pieces flow through the same overlapped pipeline as any other batch.

Landed as: `MicroScheduler.backlog`, filled by `_assemble_batches` (which pops
the whole ready set off the queues and cuts it into steps) and drained by
`_take_backlogged` at the top of `get_next_batch`. The cap comes from
`Engine.get_max_batch_size(node, walk)` — the smaller of what the submodule
says it can batch and the largest captured graph for the walk — so callers no
longer pass one. A backlogged step is skipped when the caller targets a
specific (node, walk) that doesn't match; `exclude_target` is only a fairness
hint, so it doesn't hold back a split set. `fail_rids` / `clear_rid` excise a
request from the backlog, which is the only place holding its popped node.

### 0.3 Eviction and offload

Replaces `node_resources(node)["kv"]` + `lru_tracked_nodes` + `supports_offload`
+ `offload_candidates`.

On `Resource`:

- `supports_eviction: bool` property, default `False`; `True` on `KVManager`.
- `offload(request_id) -> int` — move the request's state off-device, return
  what was freed.
- `reload(request_id) -> bool` — bring it back; `False` when it doesn't fit yet.
- `get_offload_priority(request_id) -> ...` — generalizes `offload_candidates`.

The worker/scheduler still tracks LRU. Two victim policies:

- `LRU` — implemented now.
- `PRIORITY` — chosen by a specific resource's `get_offload_priority`, which
  resource being named in new **eviction policy metadata**. TODO, with a note
  in the code; the abstraction above is what makes it a small change later.

`mstar/engine/cpu_page_pool.py` gets **ported** to v1 (it currently imports the
old `kv_store`), and `KVManager` grows the offload/reload path that uses it —
today it has only `# TODO: CPU page pool`.

Reload is not the worker's to drive: `Engine.check_ready` brings an offloaded
request back and reports not-ready until it fits, which is the scheduler's cue
to run something else. Only victim selection stays on the worker.

### 0.4 Worker code to delete

`_compute_store_write_policy` and its `StoreWritePolicy` plumbing, the
`EngineType` import and dispatch, `_build_node_batch`'s `NodeBatch`
construction (becomes `_build_executing_batch`, also building the
`StepContext`), and the `pause`/`resume` paths.

`_try_offload_cold_request` / `_select_eviction_victim` / `_try_reload_request`
stay, retargeted onto the `Resource` eviction API.

---

## Phase 1 — speculative execution

Target shape, with batch N in flight:

1. Speculatively schedule N+1 and build its `ExecutingBatch`.
2. Wait for the event that says it's safe to prepare inputs.
3. Thread N's outputs over; call `prepare_inputs(N+1)`.
4. Wait for the event that says it's safe to pre-plan.
5. `reserve_replay_slot` + `pre_plan_for_batch(N+1)`.
6. Execute N+1.

`check_stop` and `publish`/`finalize_batch` keep their current placement.

### Two events, both set inside `exec`

They live on `ExecutingBatch` rather than `node_batch.metadata`:

- **`outputs_ready`** — set after `_collect_outputs`, before commit. Gates
  step 2. What step 3 needs is N's output *tensors*, which exist as soon as the
  forward is submitted.
- **`commit_done`** — set after `self._runner.commit(step)`. Gates step 4;
  this is today's `advance_event`, moved into the engine.

Both need the safety-net `finally` the worker has today for `advance_event`, or
a step that raises before commit hangs the plan thread forever.

### Why the split is sound

The two waits are genuinely different dependencies, conflated today into "no
inputs yet, so use dummies and restrict pre-plan to BASIC_BATCHED":

- `prepare_inputs(N+1)` depends on N's output tokens — but only as
  GPU-resident tensors, so it can run before N's kernels finish.
- `admit`/`plan(N+1)` depends on N's commit (stored_len, position counters).

`declare_step` sits with the first group: spans come from `input_seq_len`, it
reads no stream state.

### What landed: pre-plan only, on the plan thread

Steps 2–3 did **not** move off the main thread. Speculative `prepare_inputs`
would require every submodule's `prepare_inputs` to be async-safe, and some
read a token value (`.item()`) — qwen3_tts, for one. Making them all safe is
more churn than this migration should carry, so:

```
# plan thread
_preplan_spec(batch_N, spec_batch):
    batch_N.commit_done.wait(timeout)       # 4
    engine.reserve_replay_slot(spec_batch)  # 5
    engine.pre_plan_for_batch(spec_batch)

# main thread, after await_gpu(N)
thread_outputs(spec_batch, outputs_N)       # 3
engine.prepare_inputs(spec_batch)
```

The batch therefore has no token count when it is pre-planned, so the bucket
comes from `CudaGraphRunner.select_batched_bucket`: a batched capture's token
count is a property of the config (`bs` rows of a fixed length), so the batch
size alone determines it. A packed capture can't be pre-planned — its token
count belongs to the requests. That is the same BASIC_BATCHED-only restriction
the old worker had, kept deliberately.

`pre_plan_for_batch` declares its step over `config.get_node_inputs(...)` for
the leased bucket, and does **not** cache it: `exec` re-declares over the real
inputs and the resources promote what was staged.

**To lift this** (needed for chunked prefill): make `prepare_inputs`
async-safe across submodules, then move steps 2–3 onto the plan thread ahead of
`commit_done`. `ExecutingBatch.outputs_ready` already exists as the hook —
`exec` sets it once N's output tensors are published, before commit. Then
`reserve_replay_slot` sees a real `num_tokens` and `select_bucket` serves
packed captures too.

Because `prepare_inputs` now runs *after* the plan is staged, it can drop rids
the plan covered. `Engine.preplan_is_stale` compares the staged rid tuple
against the batch's current one; the worker drops the plan and lets the GPU
thread plan inline when they diverge.

### Known hazards

**Host syncs in `prepare_inputs`.** The reason it stayed on the main thread.
If it moves, any `.item()` / `.cpu()` / data-dependent Python branch serializes
the pipeline — correct, but the overlap silently disappears. Worth
`torch.cuda.set_sync_debug_mode("warn")` behind an env var when that work
starts.

TODO: let a node opt out of speculative `prepare_inputs` + pre-plan, in which
case its `prepare_inputs` is un-overlapped by design.

**Work done before the stop check.** `prepare_inputs(N+1)` runs before
`check_stop(N)`, so a request whose loop ends at N has already had inputs
prepared and `request_state` mutated. This is not new — today's pipeline
already submits GPU(N+1) before `_postprocess_batch(N)` — the work just moves
earlier in wall-clock. The existing speculation-drop path is what unwinds it.

**Threading before values land.** `_thread_outputs_to_speculative` moves tensor
lists and tests key presence; it never reads a value. That's what makes it safe
to run pre-completion, and it already relies on this today. Keep it that way: a
`if token == EOS` in there would deadlock the plan thread against the GPU
thread.

---

## Files

- `mstar/worker/engine_manager.py` — rewrite `build`; `add_request` takes
  sampling overrides.
- `mstar/worker/worker.py` — `_build_node_batch`, `_execute_on_gpu_thread`,
  `_pre_plan_for_speculative_batch` → `_prepare_and_preplan_spec`,
  `_reset_skip_plan_flags`, the `_thread_outputs_to_speculative` call site,
  `_postprocess_batch`, `_handle_allocation_failure`, the offload/eviction
  block, and the main-loop dispatch (~:2327–2500).
- `mstar/engine/v1/engine.py` — the two events, set in `exec`.
- `mstar/engine/resources/base.py` — eviction API on `Resource`.
- `mstar/engine/v1/kv_manager.py` + a ported CPU page pool — offload/reload.
- `mstar/scheduler/` — batch backlog for max-batch-size splitting.
