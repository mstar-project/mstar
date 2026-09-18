# tier 0 — state-machine fuzzing

Tier 0 has six machines. They drive the pure-Python internals of M\*, using no
GPU, no weights and no kernels, at roughly 1000 cases per second. That is cheap
enough for every pull request, but no job runs it yet:
`.github/workflows/ci.yml` does not call `pytest fuzzer/`.

Every module lists the invariants it checks and what it does not cover. Read
that list before treating a pass as evidence. `step_runner` in particular
drives stubs, so it covers the order and scope of the calls into a resource
and no resource behavior.

| machine | system under test | main oracle |
|---|---|---|
| `page_allocator` | `engine/resources/kv/cache.py:11` | page conservation: the free pages and the held pages together are always exactly `0..max-1` |
| `alloc_concurrent` | `engine/resources/kv/cache.py:11`, under threads | the same properties hold at every point of an interleaving; no thread blocks forever on a size check that another thread invalidated |
| `tensor_store` | `communication/tensors.py:50` | the refcounts agree with a model that is balanced by construction; a refcount is never negative; no request entry is empty |
| `graph_io` | `graph/graph_io.py`, `graph/base.py` | a node is ready only when it holds all of its inputs; no edge is lost; the graph always drains |
| `micro_scheduler` | `worker/micro_scheduler.py:102` | conservation: ready work is always on a queue or in the backlog, unless something dropped it on purpose |
| `step_runner` | `engine/resources/runner.py:58` | the sweep follows the dependency order; a failed admit stops the sweep; a sweep for one node stays inside that node; `remove_request` leaves no state |

## What the generators change

`graph_io` gives tier 0 most of its value. It makes graphs that are random but
correct:

* The graph is a chain of stages.
* Each stage holds one node, or a `Parallel` of several nodes.
* Each node consumes one edge from each node of the previous stage. Each of
  these edges has a different name.
* The generator can put one continuous group of stages into a `Loop`. The last
  stage of the loop sends loop-carried edges back to the first stage.

This generalizes the orpheus/wan22 idiom, which is how it reaches shapes no
released model produces.

`micro_scheduler` changes the requests, the nodes, the walks, the worker graphs
and the limit for each (node, walk) pair. It then drives these operations
against a replaced clock:

* ready sets
* `get_next_batch` calls, with a limit and with a target
* holds and failures
* deferred removes
* unservable requests
* tensor-parallel follow batches

`alloc_concurrent` drives more than one thread. The schedule is part of the
case. Each op names the worker to run next, and one worker runs at a time.
The case therefore decides the interleaving, not the clock, and a failure
replays and shrinks in the usual way. The generator varies the page count,
the thread count, the program of each thread and the schedule.

`step_runner` makes a random resource graph. Each resource has a flag for
pre-planning, for publishing and for retrieving. The machine then drives the
full lifecycle against stub resources that record their calls.

## The failures that tier 0 found

Tier 0 found five failures. The shrinker made each one small: 1 to 5 ops. All
five are in `corpus/` with the status `open`.

**`graph_io`**

1. `loop-back-requeues-completed-node` —
   `WorkerGraphStateRegistry.register_ingested_input` must decide if it puts a
   node into the queue again. To decide, it reads `ready_signals.is_ready`.
   That is the slot of the current iteration. A loop-back edge can arrive for a
   node whose current slot is still full. The code puts that edge into
   `ready_next_iter`, which is correct. But the code that adds the node to the
   queue reads the old slot. Thus it queues the node again for an iteration
   that the node already ran. One `drain()` op repeats this failure.
2. `duplicate-completion-inflates-registry-count` —
   `GraphStateRegistry.mark_entity_complete` counts the completions. It does
   not record which entities are complete. If one entity completes two times,
   the count increases two times. The registry then declares that the iteration
   is complete, but one entity never ran. This failure changes failure 1 into a
   node that never runs, a loop that stops early, and a request that hangs.

   No released model reaches either failure today. The repository has two loop
   bodies with more than one entity: bagel `image_gen_cfg` and qwen3-omni
   `talker_decode`. Both put the producer of the loop-back edge last. Thus the
   iteration resets before the code routes the outputs of that producer. A
   `Parallel` tail goes directly into this failure if one branch carries the
   loop-back edge and a different branch is still not complete.

**`micro_scheduler`**

3. `backlog-drain-ignores-pending-removes` — the new scan of the ready sets
   omits the requests in `pending_removes`. The backlog path does not.
   `_schedule_from_backlogged` calls `_filter_cap_and_schedule`, which filters
   `failed_rids` but not `pending_removes`. Thus the scheduler gives new work
   to a request that has a deferred remove.
4. `emptied-backlog-batch-returned-as-work` and
5. `emptied-backlog-batch-reparked` — these are two results of one defect. The
   backlog drain can drop every request ID of a chunk because those requests
   failed. The code does not examine the empty batch again. If there is
   capacity, the scheduler returns the empty batch to the worker as work with
   zero rows. If there is no capacity, the scheduler puts the empty batch back
   into the backlog. This leaves a permanent empty entry under that (node,
   walk) key.

## A limit of the search

A case stops at its first failed op. Thus a frequent failure hides the
invariants that come later in the same case. `graph_io` shows this clearly.
Almost every generated case that has a loop fails
`graph.ready_implies_all_inputs` at the first `drain()`. A search therefore
does not find `graph.drain_terminates` again, although that bug is still open.
Use the corpus to replay a failure that the search hides.

## Gaps to close

* `graph_io`: streaming edges and `consumes_stream`; speculative ingest
  (`ingest_for_speculation`); loops inside loops; destinations in a different
  worker graph.
* `micro_scheduler`: an oracle for fairness and starvation. It must not report
  a false failure when the backlog drains correctly.
* `step_runner`: this tier always generates a step that holds every
  dependency of every resource it names. The runner accepts a step that does
  not. The `plan` method of such a resource then reads a key that nothing
  produced. The runner should reject that step. That is validation work, not
  an assertion for this tier.
* `alloc_concurrent`: the driver follows the schedule that the case carries.
  It does not search the interleavings exhaustively, so a pass means that the
  sampled schedules found nothing. An exhaustive search over the interleavings
  of a small case would be stronger. `test/modular/tp_async_sim.py` is the
  pattern to copy.
* `alloc_concurrent`: the preemption points are the accesses to the free list
  and the lock operations. A fault on a line between two of those points is
  out of reach.
