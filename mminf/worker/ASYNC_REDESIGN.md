# Worker async-first redesign — sketch

Status: **Phase 1 + speculative scheduling (Option A') implemented.** AR
decode loops now overlap full CPU-side post-processing with GPU work.
Notes for follow-on phases (b)/(c)/sampler-staleness/sampling-sync/
postproc-thread/stream-isolation) at the bottom.

## Phase 1' — speculative scheduling (Option A', SHIPPED)

The previous Phase 1 only overlapped the message/RDMA-polling trio
(~360 µs) with GPU work. Most CPU overhead — `route_outputs`,
`store_outputs`, `send_outputs`, plus `schedule + build` — still sat on
the critical path between GPU steps. With ~6 ms GPU step and ~1 ms
serial CPU per step, ceiling was ~15% speedup.

Option A' moves to vLLM-style 1-deep speculative scheduling, restricted
to AR engine + intra-worker, with a uniform 1-iter deferral of EOS
detection.

Two follow-on fixes shipped alongside the initial Option A' (without
which it regressed both single-request latency and concurrent
throughput vs the synchronous baseline):

1. **New-rid merge into the speculative batch** (`_try_speculate_next`).
   The first cut only carried forward `batch_N`'s rids — new requests
   couldn't join the spec batch until the entire current chain
   finished, regressing concurrent throughput ~40%. Now, after building
   the continuing-rid placeholders, we run `MicroScheduler.get_next_batch`
   and merge in any fresh rids whose ready node matches the spec
   batch's `(node_name, graph_walk)`. Mismatched fresh batches get
   pushed back onto their queues for the non-speculative path.

2. **Stream-isolated D→H + event-gated default-stream sync.** With
   GPU(N+1) queued on default stream behind GPU(N), every
   `default_stream().synchronize()` on the main thread (in
   `_store_outputs_and_finish_loops` and `.cpu()` in `_slow_postprocess`)
   stalled waiting for GPU(N+1) to drain — undoing the overlap. Now
   `_execute_on_gpu_thread` records a `torch.cuda.Event` on the default
   stream right after `execute_with_max_batch_size` returns and stashes
   it on the `NodeOutput`. Downstream, the main thread waits on the
   event (returns as soon as GPU(N) is done) for the
   `register_for_send` sync, and queues a batched D→H copy of the new
   tokens on a worker-owned side stream gated by `wait_event`. The
   side-stream copy assumes the new-token tensors are fresh sampler
   allocations (FlashInfer `top_p_sampling_from_probs`), not views
   into CUDA-graph static buffers — verify this every time the sampler
   path changes.

```
                Main thread                                  GPU thread

iter K start:  pending = (batch_N, future_N)
               _pending_stops, _pending_removes from prior iters

  CPU preamble                                  ◄── overlap GPU(N)
    process_messages
    check_ready_tensors
    poll_stream_buffers

  speculate + build N+1                         ◄── overlap GPU(N)
    continuing_rids = batch_N.rids
                       - _pending_removes
                       - _pending_stops
    spec_batch  = clone(batch_N) with continuing rids only
                  (GraphNode.clone_for_next_iter; fresh edges)
    spec_node_batch = NodeBatch with placeholder loop-back inputs

  await GPU(N) Python return                    ◄── only sync point

  (if alloc_failed: drop speculation, retry batch_N, drain pipeline)

  thread N's outputs → N+1's loop-back inputs   ~ tens of µs
    spec_node_batch.input["text_inputs"] =
        output_N["text_inputs"]                 (Python pointer assign;
                                                 CUDA stream order
                                                 guarantees N's writes
                                                 visible to N+1's reads)

  submit GPU(N+1) ─────────────────────────────►  execute_batch(N+1)

  fast_postprocess(N)                           ◄── overlap GPU(N+1)
    apply _pending_stops_to_batch (1-step-late stops from N-1)
    cleanup_consumed_inputs
    store_outputs_and_finish_loops (incl loop-back so complete_loops
                                    sees the loop-continue signal)
    process_node_outputs (skip loop-back edges that speculation
                          already consumed; manually deref their
                          UUIDs to keep tensor_store refcounts clean)
    register_for_send (intra-worker)

  slow_postprocess(N)                           ◄── overlap GPU(N+1)
    prematerialize new tokens via .cpu()
      ⚠ syncs on default stream → blocks until GPU(N+1) drains
        (streaming-token-latency cost; see "Open follow-ons" below)
    _send_outputs (ZMQ to conductor / api_server)
    check_stop_for_batch(N) → _pending_stops    (will fire next iter)
```

### Submodule API split

Models that previously did `.item()` + `register_loop_stop` *inside*
`submodule.postprocess` (Orpheus, BAGEL, Qwen3-Omni Thinker/Talker)
have been refactored to split that logic:

- `postprocess(rid, info, outputs)` — metadata-only (rebind output
  names, drop optional outputs). MUST NOT read tensor values.
  Runs on the GPU thread inside `execute_batch`.
- `check_stop(rid, info, outputs) → set[str]` — value-reading EOS /
  max-tokens check. Returns loop names to stop. Runs on the worker's
  slow-postprocess path.

vjepa2 retains `register_loop_stop` inside `forward` because its check
is CPU-only (`iter_idx + 1 >= rollout_horizon`) — no GPU sync to
remove. The 1-iter deferral is uniform regardless of where the signal
originates (vjepa2 still pays 1 wasted GPU step per stop).

### Wasted-step semantics

| Source of stop                                 | Wasted steps |
|------------------------------------------------|--------------|
| Conductor `REMOVE_REQUEST` (e.g. external EOS) | 1 (rid was in spec batch when message arrived) |
| Submodule `check_stop` (Orpheus/BAGEL/Qwen3)   | 1 (spec already submitted before stop known) |
| Submodule `register_loop_stop` in `forward` (vjepa2) | 1 (deferred uniformly to next iter's fast_postprocess) |
| `output.allocation_failed` from engine         | up to 1 spec step also discarded; pipeline drained, batch_N retried |

### Speculation eligibility

`_can_speculate(batch_N)` returns True iff:
- `engine.engine_type() == EngineType.AR`, AND
- every `GraphNode` in the batch has `enable_async_scheduling=True`, AND
- `_try_speculate_next` finds at least one continuing rid whose node's
  required inputs are all loop-back (i.e. `next_node == node.name` for
  some output edge with the same name).

Models can opt a node out with `GraphNode(..., enable_async_scheduling=False)`.
For opted-out nodes, non-AR engines (Flow, EncDec, AudioCodec), and AR
steps that aren't pure loop-back (e.g. prefill → decode transitions), the
worker falls through to the non-speculative path — drains the in-flight
step, runs MicroScheduler, submits the next step. Same as Phase 1.

### Cross-iter state on `Worker`

- `_in_flight_rids: set[str]` — rids in the currently-pending GPU step.
  Read by `_remove_request` to defer destructive teardown.
- `_pending_removes: set[str]` — `REMOVE_REQUEST` for rids that were
  in flight at message time. Applied in next iter once no in-flight
  step references them.
- `_pending_stops: dict[rid, set[str]]` — loop names to stop. Filled
  by `check_stop_for_batch` in slow_postprocess; consumed in next
  iter's fast_postprocess (this is the 1-iter deferral).

### Thread safety

The GPU thread only mutates engine-internal state plus
`batch.per_request_info[rid].per_label_seq_info` in `execute_batch`'s
finally block. The main thread, while GPU(N) is in flight, only:
- Reads `worker_graphs_manager` queue state (untouched by GPU thread).
- Reads `batch_N.node_objects[rid]` (`name`, `outputs`, `input_ids`)
  for cloning into spec_batch — no writes.
- Reads `worker_graphs_manager.get_fwd_info(rid)` — returns the live
  fwd_info ref, which `execute_batch(N+1)` reads later on the GPU
  thread *after* `execute_batch(N)` finished its finally write. Strict
  per-step serialization on the GPU thread keeps these races out.

Mutations of `tensor_manager` happen only in fast/slow postprocess on
the main thread (after `await GPU(N)`). The GPU thread doesn't touch
`tensor_manager`.

## What Phase 1' actually delivers (measured)

`MMINF_PHASE_TIMING=<n>` instruments the speculative loop. Steady-state
Orpheus single-request decode (after autotune):

| phase | p50 | mean |
|---|---|---|
| `await_gpu` | 3.90 ms | 3.90 ms |
| `fast_post` | 0.49 ms | 0.49 ms |
| `slow_post` | 0.19 ms | 0.19 ms |
| `speculate` | 0.04 ms | 0.04 ms |
| `submit_spec` | 0.01 ms | 0.01 ms |
| `iter_total` | **4.79 ms** | 4.80 ms |

`iter_total ≈ await_gpu + fast_post + slow_post + preamble`. Concurrent
×4 shows the same pattern: `iter_total = 6.87 ≈ await_gpu (4.64) +
fast_post (1.21) + slow_post (0.59) + preamble`. **The CPU and GPU
work are running serially — there is essentially no overlap.**

Why the design doesn't pay off as drawn:

- The "GPU thread" is a `ThreadPoolExecutor(max_workers=1)`. After
  `submit_spec`, the future is queued, but the GPU thread can only
  execute `engine.execute_with_max_batch_size` once it acquires the
  GIL. The main thread holds the GIL throughout `fast_post` /
  `slow_post` (mostly Python: dict mutations, ZMQ pickling, no CUDA
  calls that release the GIL).
- By the time the main thread reaches `await_gpu` and releases the
  GIL inside `future.result()`, the GPU thread has done **zero**
  kernel launching for N+1 yet. So `await_gpu` measures the full
  3.9 ms of (GPU thread's Python kernel-launch + GPU kernel
  execution), not the residual after CPU work overlaps.
- The `current_stream().synchronize()` in `sampling.py` doesn't help
  either: it sits *inside* the GPU thread's work, so it just makes
  `future.result()` block longer. Removing it (gated to skip after
  the first ~64 calls) was tried and reverted: it caused a different
  concurrent regression on Orpheus (likely a different GIL-yield
  path) and is load-bearing for Qwen.

This is a structural Python-GIL limitation of the executor-thread
pattern, not an mminf-specific bug. The infrastructure (event-gated
default-stream sync, side-stream D→H of new tokens, fresh-rid spec
merge with mismatch-skip) is correct and necessary for any future
approach — it just doesn't translate to wall-clock gains on its own.

### Same-thread-async attempt (Orpheus, measured)

The same-thread-async path was implemented and measured:
- The executor-thread path is now unconditional; the measured inline
  submission fallback has been removed.
- Sample sync (`sampling.py`) gated to a 64-call autotune-warmup budget,
  then permanently skipped.
- New helper `_prematerialize_for_check_stop` does a side-stream D→H of
  every CUDA tensor in the engine output before `check_stop`, so the
  per-rid `.item()` reads don't trigger a default-stream sync (which
  would block on the queued GPU(N+1)).

Result on Orpheus single-request decode (steady state, after autotune):

| phase | inline + budgeted sync (this) | executor (prior) |
|---|---|---|
| `submit_spec` (engine call) | **4.15 ms** | 0.01 ms (returns instantly) |
| `await_gpu` | 0.01 ms | 3.90 ms |
| `fast_post` | 0.30 ms | 0.49 ms |
| `slow_post` | 0.29 ms | 0.19 ms |
| `iter_total` | 4.94 ms | 4.79 ms |

`submit_spec` should have collapsed to ~1 ms once the sample sync was
gone (kernel launch only, no waiting for completion). It didn't. The
4 ms of blocking is **`wrapper.plan()`** in `cache_manager.py` (see the
TODO at line 238): plan does two internal D→H reads of CUB-scan
results that force a synchronous wait on the previous step's KV-state
updates. As long as plan() is on the critical path, the engine is
effectively synchronous regardless of whether the sample sync is in
place — the structural overlap window the same-thread path was meant
to open just isn't there.

Net: same-thread + sample-sync removal is *correct* and the
infrastructure (`completion_event`, side-stream D→H,
`_prematerialize_for_check_stop`) all does what it should — but TPOT
on Orpheus stays within noise of baseline. To actually unlock the
~1 ms / iter win, plan() has to come off the critical path, which
needs the double-buffered FlashInfer wrappers + plan-worker thread
(Phase 3 below). The same-thread path is the right *foundation* for
that work; it just doesn't deliver wins on its own.

### Plausible paths that would actually deliver overlap

1. **Same-thread async with CUDA events.** Drop the executor thread.
   Main thread directly launches kernels via `engine.execute_batch`
   and records a completion event. Post-processing waits the event
   (releases GIL) instead of a Python future. Removes GIL contention
   entirely because only one Python thread is running. Requires the
   engine to be free of internal `current_stream().synchronize()`
   calls — would need the sampling.py:376 sync removed (with a
   model-aware gate so Qwen still gets it).
2. **Postproc thread (original Phase 2).** Move `slow_postprocess`
   onto a separate thread. Main thread keeps doing GPU submissions
   and `fast_postprocess`. Slow path runs in parallel and only blocks
   on its own ZMQ sends. GIL contention shifts but doesn't disappear;
   measured wins likely small (~0.5 ms / iter from `slow_post`).
3. **C++/Cython wrapper that releases the GIL during kernel launch.**
   Wrap the engine's hot Python launch path in an extension that
   uses `Py_BEGIN_ALLOW_THREADS` around the (mostly C++) CUDA API
   calls. Highest invasiveness, biggest potential win.

The phase-timing harness is left in place (env-var-gated) so future
attempts can be measured directly rather than guessed at.

## Open follow-ons (still on the original plan)

The doc below this line is the original staged plan. Phases 0/1 now
correspond to the shipped state (Phase 1 = GPU thread offload, Phase 1'
= speculative scheduling above). The remaining items below are
unchanged in spirit but should be re-read in light of Phase 1':

1. ~~**Stream-isolated D→H of new tokens.**~~ Shipped — see Phase 1'
   above. Side-stream D→H of new tokens + event-gated
   `register_for_send` sync. Sequential single-request latency
   regression dropped from +24% to +7%; what remains is sampler
   staleness (item 4 below) and main-thread CPU residue.
2. **`sampling.py:376` `current_stream().synchronize()`.** Sits inside
   the engine's hot path between `fused_temperature_softmax` and
   FlashInfer's `top_p_sampling_from_probs`. Blocks the GPU thread for
   the full softmax/replay duration. Investigate whether FlashInfer
   actually requires this sync or if it's a vestige; if removable,
   GPU thread returns truly-async and Phase 1' overlap window grows
   from "schedule + build + thread-through" to "the whole iter".
3. **Postproc thread (Phase 2 from original plan).** Move
   `_slow_postprocess` to a separate thread so the main thread can
   start the next iter's CPU preamble without waiting on the .cpu()
   sync above. Less impactful once stream isolation lands; mostly
   useful if a future profile shows the main thread becoming the
   bottleneck again.
4. **Sampler `_seen_token_mask` rep-penalty staleness.** With Option
   A', step N+1 is submitted before N's `check_stop` (slow path) has
   updated the seen-mask with N's token. So N+1's rep-penalty sees a
   one-step-stale mask. Accepted for now — Orpheus + most Qwen
   configs are the affected paths. Fix would need an event-gated mask
   update on the GPU thread between steps.
5. **Speculation extension (b): same-engine, any walk.** Today
   speculation only fires for AR steps whose required inputs are all
   loop-back. Generalize to e.g. flow loop bodies (which iterate the
   same flow node K times); needs the speculation to know the next
   walk's required inputs, not just loop-back.
6. **Speculation extension (c): cross-engine / cross-worker.** AR →
   flow transitions, or LLM-on-worker-A → flow-on-worker-B. Needs
   event-gated RDMA send (sender records event after GPU writes,
   receiver waits on it before reading) instead of the current
   `default_stream().synchronize()` in `register_for_send`.

---

## Motivation (original notes)

Today `Worker._run_loop` is a single serial sequence per decode step:

```
process_messages → poll_stream_buffers → schedule → build_node_batch
  → engine.execute_batch          (≈ 8.5 ms — GPU replay + sample)
  → update_request_info → cleanup_inputs
  → route_outputs → store_outputs → send_outputs
```

The engine block is GPU-bound; everything around it is CPU-bound (Python
dispatch, ZMQ pickling, dict construction, small D→H transfers). Today they
run back-to-back on the same thread, so CPU work never overlaps with GPU work.

Per-step measurements at bs=8 on H200 (after the current round of
micro-optimizations) roughly:

| region | median | nature |
|---|---|---|
| `worker.process_messages` | 329 µs | ZMQ `recv_pyobj` + unpickle (CPU) |
| `worker.poll_stream_buffers` | 29 µs | Python dict scan (CPU) |
| `worker.schedule` | 76 µs | MicroScheduler (CPU) |
| `worker.build_node_batch` | 36 µs | tensor lookup + slicing (CPU) |
| `ar.cuda_graph_path` | 6.1 ms | CUDA graph replay + sample (GPU-bound) |
| `worker.update_request_info` | 26 µs | CPU |
| `worker.cleanup_inputs` | 32 µs | CPU |
| `worker.route_outputs` | 228 µs | CPU (process_node_outputs) |
| `worker.store_outputs` | 158 µs | mostly tensor_manager.register_for_send C++ syncs |
| `worker.send_outputs` | 469 µs | `.cpu()` for new tokens + ZMQ pickle/send |

Total serial per step: **≈ 7.5 ms** — of which the engine block dominates.
But that still leaves ≈ 1.4 ms of CPU-side work per step that runs *between*
replays instead of *during* them. Pipelining that away has a real ceiling of
~15% decode-step speedup, plus better GPU utilization under contention.

## Goal

Redesign the worker so CPU-bound orchestration for step N overlaps with the
GPU work of step N ± k. Specifically, at steady state:

- Step N's `send_outputs` / ZMQ pickling runs while step N+1's CUDA graph replays.
- Step N+1's `flashinfer.plan` runs while step N's sample is finishing and
  while step N's post-processing is handing off.
- Inbound ZMQ (`TENSOR_RECEIVED` ACKs, streaming input signals) never blocks
  the GPU submission loop.

## Architecture

```
          ┌────────────────────┐   ┌────────────────────┐
ZMQ in ───│ io_thread (async)  │──▶│ scheduler_inbox    │ (Python Queue)
          │  recv + unpickle   │   └────────────────────┘
          │  dispatch          │            │
          └────────────────────┘            ▼
                                   ┌────────────────────┐
                                   │ main thread        │
                                   │  (GPU submitter)   │
                                   │                    │
                                   │  for each step N:  │
                                   │   1. drain inbox   │
                                   │   2. pick batch    │
                                   │   3. build inputs  │
                                   │   4. await plan(N) │◀──┐
                                   │   5. replay(N)     │   │
                                   │   6. sample(N)     │   │ CUDA events
                                   │   7. publish N     │   │
                                   │   8. submit        │   │
                                   │      plan(N+1) ────┼───┘
                                   └──────┬─────────────┘
                                          │ postproc_queue
                                          ▼
                                   ┌────────────────────┐
                                   │ postproc_thread    │
                                   │  await sample_done │
                                   │  route_outputs     │
                                   │  store_outputs     │
                                   │  send_outputs (ZMQ)│
                                   └────────────────────┘
```

Thread count: **3** — `io_thread`, `main` (GPU submitter), `postproc_thread`.
A fourth `plan_worker` thread owns the double-buffered FlashInfer wrappers.
All share the same `Worker` object; coordination is queue-based.

### Why threads and not asyncio

PyTorch CUDA calls release the GIL; ZMQ `recv_pyobj` releases the GIL during
the C-level recv. That makes OS threads effective. asyncio would require
wrapping every blocking call in `run_in_executor`, which defeats the purpose.

## Data flow and queues

### `scheduler_inbox` (io_thread → main)

- `io_thread` runs `self.communicator.get_all_new_messages()` in a tight
  `while True` loop; each message is pushed onto the inbox.
- The main thread drains at the top of every step (`process_messages`).
- Eliminates the 330-µs median `worker.process_messages` spike from the
  critical path.

### `postproc_queue` (main → postproc_thread)

- Each entry is a `PostProcessJob(batch, sample_cuda_event, output_handles,
  routing_plan)`.
- Main thread enqueues immediately after sampling kicks off; it does not
  block on sample output.
- `postproc_thread` awaits the CUDA event (`event.synchronize()`), reads
  `.cpu()` tokens, runs routing, fires ZMQ sends.
- Removes ~630 µs of median wall-clock per step from the main thread.

### `plan_future_ring` (main ↔ plan_worker, double-buffered)

- Two `FlashInferDecodeWrapper`s, each with its own persistent plan buffers.
  Main thread alternates: step `N` uses `wrapper[N % 2]`.
- Right after `advance_seq_lens(N)`, main thread posts
  `PlanRequest(seq_lens, page_state_snapshot)` for step N+1 to `plan_worker`.
- `plan_worker` runs `wrapper[(N+1) % 2].plan(...)` and records a CUDA event.
- Main thread awaits that event just before `replay(N+1)`.
- Hides the ~750 µs `flashinfer.plan` currently on the critical path.
  (See `cache_manager.py` TODO.)

## Synchronization points

1. **`advance_seq_lens(N)` → `plan(N+1)` start.** Page indices for N+1 depend
   on N+1's alloc, which needs end-of-step-N state. Main thread snapshots
   `(seq_len, page_indices)` per rid just after `advance_seq_lens` and hands
   the snapshot (plain Python ints/lists) to `plan_worker`. Avoids reading
   shared mutable `KVRequestState` from a second thread.

2. **`plan_worker` done → main thread `replay(N+1)` start.** CUDA event
   recorded at the end of `wrapper.plan()`; main thread calls
   `event.wait(stream=replay_stream)` so the replay stream waits without a
   CPU sync.

3. **`sample(N)` done → `postproc_thread` starts post-processing.** CUDA
   event recorded after sampling kernel is submitted; post-processing thread
   calls `event.synchronize()` (CPU wait) before the batched `.cpu()`.

4. **`post_process(N)` done → engine free to reuse KV-cache static buffers
   for step N+k.** Backpressure: if `postproc_queue` depth exceeds
   `MAX_IN_FLIGHT`, main thread blocks on `postproc_queue.join()` before
   submitting step N+1. Protects against unbounded VRAM pinning.

5. **Seen-token scatter (rep-penalty).** See `sampling.py:_seen_token_mask`
   TODO — runs on `postproc_thread` on the default-stream chain. Does not
   block main thread because mask is read by step N+1's `plan`/`sample` only
   through its stacked copy (stack happens on `postproc_thread` too, before
   the mask is read again).

## Thread safety of existing state

| object | current thread model | post-redesign |
|---|---|---|
| `PagedAllocationManager` (per-worker) | single thread | still single-writer (main). Reads by `plan_worker` use **CPU snapshots** passed in at hand-off time, not direct access. |
| `WorkerGraphsManager.per_request_info` | single thread | single-writer (main). Postproc thread reads via an immutable `NodeOutputRouting` object constructed on main, then handed over. |
| `TensorManager`/`TensorStore` | single thread | needs a lock, or split into read-only vs mutating surfaces. Mutations happen on postproc (register_for_send, increment_ref); main only *reads* for routing decisions. Can probably get away with a coarse `threading.Lock` around the dict mutations. |
| `Sampler._seen_token_mask` | single thread | moves to postproc thread — it's only used in the rep-penalty path on that thread. |
| `Communicator` push sockets | single thread | guard with a `Lock`; io_thread uses `pull_socket` only, postproc_thread uses `push_sockets`. |
| ZMQ context | thread-safe | n/a |

## Phased rollout (low → high risk)

**Phase 0 — instrument.** Add NVTX markers to the existing serial loop at
sub-step granularity (already done for `send_outputs` sub-parts during the
latest round). Capture baseline for each phase.

**Phase 1 (revised) — push the engine call onto a single GPU thread.**

Implemented. Instead of the originally-proposed io_thread split, the actual
landed Phase 1 keeps a single main thread but offloads
`engine.execute_with_max_batch_size` to a 1-worker `ThreadPoolExecutor` and
runs the iteration body in two halves:

```
top of iter:
  process_messages          ┐  run while pending GPU step (from previous
  check_ready_tensors        │  iter) is still in flight on the GPU thread
  poll_stream_buffers       ┘
  ─────────────── await pending future ───────────────
  postprocess(prev step)    (depends on output tensors + updates queues
                             that the next schedule() will read)
  schedule
  build_node_batch
  submit GPU(next step) → pending = future
```

So the per-step CPU work that overlaps with GPU is the message/RDMA polling
trio at the top of the loop (~360 µs median). Schedule + build + post-
processing remain serialized with respect to the GPU step they belong to,
because each has a real read-after-write dependency on the previous step's
post-processing (queues, fwd_info, tensor_manager refcounts).

NVTX caveat: range_push/range_pop with `synchronize=True` calls
`torch.cuda.synchronize()`, which on the main thread would block on the GPU
thread's submitted work and undo the overlap. CPU-side NVTX ranges on the
main thread therefore use `synchronize=False`. The single `worker[…].node[…]`
range that brackets the engine call stays `synchronize=True` and is now
pushed/popped from inside the GPU thread itself, so it gives accurate GPU
timing without polluting the main thread.

Thread safety as actually implemented:
- `engine.execute_with_max_batch_size` is the only thing on the GPU thread.
  It does not touch `tensor_manager` or `worker_graphs_manager`. It does
  mutate `batch.per_request_info[rid].per_label_seq_info` (AR engine,
  `kv_sync_retrieve` finally block).
- During the overlap window the main thread runs only the message-/RDMA-
  polling trio, which mutates queues, tensor_manager state, and stream
  buffers — disjoint from what the GPU thread writes.
- A conductor `INPUT_SIGNALS` arriving in the overlap window can call
  `update_request_info(current_fwd_info=…)` and replace
  `per_partition_info[partition].current_fwd_info`. The GPU thread keeps
  writing to the *old* fwd_info object (held via `node_batch.per_request_info`),
  so the GPU's writes are simply lost relative to the new fwd_info — but
  that is also what the previous serial code did at the iteration boundary,
  because `_build_node_batch` of the next iteration would have re-read the
  fresh fwd_info from `worker_graphs_manager`. No new race.

Subsequent phases (2-4) below remain as planned. The original "io_thread
pulls ZMQ off the critical path" idea is *not* what shipped in Phase 1; it
is now folded into a future Phase that splits inbound ZMQ off if profiling
shows process_messages still on the critical path after the GPU offload.

- **Risk (as shipped):** low-medium. Engine + tensor_manager / queue
  isolation verified by inspection; the only mutation the GPU thread does
  to shared state is `per_label_seq_info` on stale fwd_info, which is
  benign (see above).
- **Expected win:** the message/RDMA-polling trio (~360 µs median at bs=8)
  hides behind a ~6.1 ms GPU step, so per-step wall-clock should drop by
  roughly that amount. Larger wins (route_outputs, send_outputs,
  flashinfer.plan) need the later phases.

**Phase 2 — move `send_outputs` to `postproc_thread`.**
- Queue `PostProcessJob` with (batch, routing, token-cpu-materialized-dict,
  ack-event).
- Main thread kicks off batched `.cpu()` on side stream + event; `postproc`
  awaits the event before sending.
- **Risk:** medium. Tensor lifetime: `sampled` tensor from FlashInfer must
  not be freed before postproc's cpu read completes. Guarantee with an
  explicit reference held in the queue item.
- **Expected win:** ≈ 600 µs median off the main thread loop.

**Phase 3 — double-buffered FlashInfer plan.**
- Instantiate two `FlashInferDecodeWrapper`s per decode config + label.
- Keep existing static_cache_manager structure; swap which wrapper's static
  buffers are in use each step.
- `plan_worker` thread submits `plan(N+1)` after main's `advance_seq_lens(N)`.
- Main thread sees an `await plan_future` before `copy_inputs(N+1)`.
- **Risk:** higher. Mooncake `register_memory` and FlashInfer's internal
  stream syncs need validation under concurrent access.
- **Expected win:** ≈ 750 µs median off the critical path.

**Phase 4 — post-sample rep-penalty scatter off-thread.**
- `seen_token_mask` updates (currently inside `Sampler.sample`) move to
  `postproc_thread`. The stacked mask needed by `plan(N+1)` is materialized
  on `postproc_thread` just after its scatter, handed back to main via a
  future. (Many setups don't use rep-penalty at all; in that case this
  phase is a no-op.)
- **Risk:** low (isolated per-rid writes, only race is with next step's
  stack read — gated by event).
- **Expected win:** ≈ 150 µs when rep-penalty is active (most Orpheus
  configs); 0 otherwise.

## Risks and open questions

- **CUDA graph static buffer aliasing.** CUDA graphs capture static
  input/output tensor addresses. If main thread queues replay(N+1) while
  postproc is still reading sample(N) outputs, we need to be sure postproc
  has cloned or otherwise decoupled those reads. The existing
  `batched_logits[:len(request_ids)]` slice is a view into a static buffer
  — currently safe because post-sample FlashInfer output is a fresh alloc,
  but worth re-checking every time we touch this code.
- **Backpressure tuning.** Too small `MAX_IN_FLIGHT` kills the overlap
  gain; too large grows VRAM pressure (pinned staging tensors, un-ACKed
  RDMA buffers). Start with 2; tune with profiles.
- **Error propagation.** An exception on `postproc_thread` must tear down
  the main loop cleanly. Use a shared `threading.Event` for `stopping` and
  surface exceptions via a result queue.
- **CUDA graph capture reentrance.** New requests with new graph shapes
  trigger `capture_graph` today — that path is synchronous and creates new
  persistent wrappers. Must not race with `plan_worker`. Simplest answer:
  pause `plan_worker` around graph capture.
- **Debugging.** A 4-thread pipeline is ~4× harder to debug. Each thread
  should log with a distinct prefix; every queue hand-off should carry a
  monotonic `step_id` for correlation across threads in nsys.

## Cross-reference

- `mminf/utils/sampling.py` — TODO at the rep-penalty scatter (Phase 4).
- `mminf/engine/cache_manager.py` — TODO at `wrapper.plan(...)` (Phase 3).
- This file pulls those into a single plan and adds Phases 1 & 2 (I/O and
  post-processing offload), which are the lowest-risk starting points.
