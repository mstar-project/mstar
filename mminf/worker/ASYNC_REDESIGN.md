# Worker async-first redesign — sketch

Status: **design note, not yet implemented**. Consolidates the TODOs scattered
across the codebase (`sampling.py` seen-mask scatter, `cache_manager.py`
double-buffered FlashInfer wrappers) into a single staging plan.

## Motivation

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

**Phase 1 — pull ZMQ inbound off the critical path.**
- Single `io_thread`, single `scheduler_inbox` queue.
- `_process_messages` becomes `_drain_inbox` (reads queue, no ZMQ).
- **Risk:** low. ZMQ thread-safety is well-understood.
- **Expected win:** ≈ 300 µs median off `process_messages`.

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
