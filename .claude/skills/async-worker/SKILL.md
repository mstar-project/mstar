---
name: async-worker
description: The worker's asynchronous step loop — what runs on which thread and stream, what each stage waits on, how speculation and pre-plan are rolled back on failure, and what breaks the overlap. Load this before changing anything in mstar/worker/worker.py's main loop, the GPU thread, the plan thread, speculation, or postprocess; and before concluding that a worker-level change has a bug.
---

# The worker's async step loop

The worker does not run one batch at a time. While batch N's kernels are on the GPU it is already scheduling, planning and launching batch N+1, and post-processing batch N-1. Almost every subtlety in `mstar/worker/worker.py` follows from that.

**Do not pattern-match this code.** There is no mechanical rule like "every mutation on the normal path must also appear on the speculative path" — the paths deliberately differ. If you are changing the loop, work out what your change does in terms of the stage order below, which thread it runs on, and what is still in flight at that moment. If you are *reviewing* a change here, either explain the problem in those terms or say nothing; a guess that looks right is usually wrong, and the person who wrote this gets mixed up too.

## Threads and streams

| Runs on | What |
| --- | --- |
| Main thread | The loop below: messages, tensor readiness, scheduling, speculation, postprocess |
| `gpu_executor` (1 worker) | `_execute_on_gpu_thread`: declare → admit → plan → forward launch → commit |
| Plan thread (1 worker) | Pre-plan for N+1, on its own CUDA stream, gated on N's commit event |
| Side stream | D2H copies for `check_stop` pre-materialization |

Knobs, with the non-obvious default called out: **`MSTAR_TP_ASYNC_SCHED` is `0`, so async scheduling is OFF for lockstep-parallel (TP/SP) nodes.** It is expected to be flipped on and simply has not been yet, which has two consequences worth holding onto: the TP speculation path is not what a default `tp2` run exercises, so a change there needs the flag set to be tested at all; and a change that only works with it off will break when it flips. Set to `1` for every parallel node, or to a comma-separated node list. The worker `all_gather`s the value and raises if it disagrees across the ranks of an instance — an instance of invariant 11 made into a startup check.

The others: `MSTAR_PRE_PLAN_SPEC`, `MSTAR_MAX_CONSECUTIVE_SPEC_STEPS`, `MSTAR_SPEC_PEEK_FOR_FAIRNESS`, `MSTAR_LAUNCH_WAIT_MS`, `MSTAR_ENGINE_STEP_SYNC`, `MSTAR_FORWARD_SIDE_STREAM`. Defaults for all of these are in [docs/environment_variables.rst](../../../docs/environment_variables.rst); check it rather than assuming. Per-node opt-out from async scheduling entirely: `enable_async_scheduling=False` on the `GraphNode`.

## One iteration

The loop is circular, so this starts at an arbitrary point: **batch N is running on the GPU and postprocess for N-1 has just finished.** The code carries its own stage comments 1-3; the substages below are what they expand to.

**0. Deferred teardown.** `_apply_pending_removes_safe_to_drop`, then `_fail_requests` on `scheduler.take_admit_errors()`, then `_apply_pending_drains`. The admit-error step matters: those are requests a resource declared unservable during *last* pass's readiness scans. They sit in no batch, so nothing else would ever fail them.

**1. CPU preamble — overlaps GPU(N).** `_process_messages` (ZMQ), `_check_ready_tensors` (read tensors in from other ranks), `_poll_stream_buffers`, request cleanup. Every NVTX range here uses `synchronize=False`, because a `torch.cuda.synchronize()` would drain the in-flight GPU work and destroy the overlap.

**2. Speculate the composition of N+1 — overlaps GPU(N).** Two shapes:

- **Speculative step** (`_try_speculate_next`): assume every output batch N declared in the M* graph will be produced, work out which node(s) become ready next, choose one, then pull in any out-of-batch requests that can run in the same N+1. The TP leader broadcasts the head immediately, during forward N, so followers build theirs concurrently.
- **Yield-away step**: still overlapped, but takes a fresh batch from the micro-scheduler (typically a different node). Chosen when `consecutive_spec_steps` hits `MSTAR_MAX_CONSECUTIVE_SPEC_STEPS`, or when the fairness peek (`scheduler.has_ready_excluding`) shows another `(node, walk)` actually ready on this worker. On single-walk workers the peek returns False and the worker always speculates.

**3. Arm pre-plan.** `_arm_speculation` hands N+1 to the plan thread *now*. It waits on N's **commit event and nothing else**, then reserves a slot and pre-plans — while the main thread sits in `await_gpu` with the GIL released and N's kernels are still running.

**4. Wait for GPU(N)'s CPU side.** `await_gpu` returns when the GPU thread has finished its CPU work for N: all kernels launched, output shapes known. **The GPU is still running.** For TP followers, `follow_await` runs first, watching for the leader's head during N so the leader's decision is settled before N is post-processed.

**5. Reconcile, then roll back on failure.** Thread N's outputs into the speculative batch (`_thread_outputs_to_speculative`), and handle N's failures. Three branches, and they clear speculation for different reasons:

- `admit_error` — admit refused N, so no forward ran. `_handle_admit_failure` pushes the GraphNodes back to the scheduler queue and, on KV OOM, offloads or holds the failed rids.
- `failed_requests` — a per-rid stage (`prepare_inputs` / `postprocess`) blamed specific requests. The speculation was built from N's rids and may thread outputs the failed rids never produced, so it is dropped; the rest of the batch still post-processes and routes.
- Clearing itself splits by kind. A **non-yield-away** speculation depended on N's outputs (the plan thread already threaded them in), so if N's output is invalid the spec batch cannot run. A **yield-away** speculation is independent of N — but `_handle_allocation_failure` may have shifted the engine's KV state (paused or offloaded rids), so its pre-plan is reset anyway.

**6. Launch N+1.** Submit to the GPU thread, which starts queuing kernels. The main thread then blocks on `spec_launch_started` (timeout `MSTAR_LAUNCH_WAIT_MS`) so it holds off the GIL until the GPU thread reaches the forward launch. Staleness is checked on the GPU thread, after the plan future resolves and after prepare. From submission on, the GPU thread owns the pre-plan.

**7. Postprocess N.** Now wait for the actual GPU work (`event_sync`), D2H the outputs, pre-materialize on the side stream, run `check_stop` to look for EOS, extend prefix hash chains from the host copy of sampled tokens, then route and send outputs. The side stream exists because the default stream has GPU(N+1) queued behind GPU(N)'s outputs by this point.

## What breaks it

- **A host sync anywhere in the step path outside `check_stop`.** `check_stop` is fed CPU tensors after a D2H that is already paid for. Any other `.item()`, `.cpu()`, `.tolist()`, or print of a device tensor collapses the pipeline to serial. So does `.numpy()`, or `bool()`/`if` on a device tensor. The cost is invisible in the diff, and in the profile it is charged to the enclosing phase rather than to `event_sync`. Whisper's decoder once pushed its prompt ids to the device and read them straight back with `.tolist()` to recover settings `process_prompt` already knew. That cost 8.4 ms of `prepare_inputs` against a 1.9 ms step, because the read blocked on the encoder's queued work. Host-known values go in `ForwardPassArgs.step_metadata` (precedent in bagel, wan22, omnivoice and pi05).
- **Moving GIL-holding work to another thread.** The comment above `worker.gpu_thread.await_plan` in `worker.py` records that running `prepare_inputs` while the plan thread worked only put the two threads in contention for the GIL, and measured worse. Only work that releases the GIL overlaps. For a host-bound step, delete work rather than relocating it. The inverse is worth knowing too: *adding* a wait can be close to free, because a thread parked on an event releases the GIL its own GPU and plan threads were competing for. A per-iteration wait measured at 1.55 ms on a TP follower left `iter_total` lower, not higher. Measure the net, not the wait.
- **A new resource doing H2D into fixed buffers without `force_double_buffer`.** Step N+1 overwrites buffers step N is still reading. See the [engine-resources skill](../engine-resources/SKILL.md).
- **Per-rank divergence.** Speculation, the fairness peek, admission and pre-plan all run per rank. Anything that makes one rank of a lockstep instance decide differently from another deadlocks the instance on the next collective — see invariant 11 in [AGENTS.md](../../../AGENTS.md). Readiness scans with per-rank side effects are the recurring case.
- **A node that can neither absorb nor predict the wasted last step of a dynamic loop** should set `enable_async_scheduling=False` rather than have the loop work around it. AR decode absorbs it (it cannot know its length). Flow and diffusion nodes know their iteration count at ingestion, so they exit early in `prepare_inputs` for the extra step and pay only overlapped CPU.

## Verifying a change here

`test/modular/test_micro_scheduler.py`, `test_admit_failure_handling.py`, `test_worker_drain.py` and `test_graph_speculation.py` cover the failure and backpressure paths. Several fakes stub worker privates **by name**, so a rename can break them with no grep hit at the definition — run the suite.

For timing effects, `benchmark/worker_phases/` reports these stages by name (`speculate`, `follow_await`, `await_gpu`, `submit_spec`, `gpu_submit_queued`, `iter_total`); see the [benchmarking skill](../benchmarking/SKILL.md). If `await_gpu` collapses toward zero and `iter_total` does not improve, the overlap is gone.
