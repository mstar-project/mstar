# tier 1 — the synthetic model

Tier 0 fuzzes the data structures. Tier 1 fuzzes the engine. To do this, it
makes the model a generated input.

Tier 1 has two lanes. The lane decides what the tier can find.

| lane | hardware | where it runs | what it can find |
|---|---|---|---|
| **1a** | CPU | CI, and each night | graph topology, scheduling, the lifecycle of a request, leaks, deadlocks |
| **1b** | GPU | each night, on `team1` | everything in 1a, plus the behavior of a resource: memory, cache contents, pre-plan, CUDA graphs, eviction |

Lane 1b covers what tier 0 cannot reach. Tier 0 checks the order and the scope
of the calls into a resource. It does not check the work that a resource does,
and it models no thread. A tier 1 on the CPU alone keeps both of those limits
for every part that needs a device.

## What exists now

Three machines. Each one generates a model, runs it, and compares every value
that reaches the client against an interpreter of the same model. They differ
in the code that runs under them.

| machine | what it drives | cost of one case |
|---|---|---|
| `model_run` | the graph layer alone (`mstar/graph/`) | approximately 2 ms |
| `kv_run` | the graph layer and the real resources (`mstar/engine/resources/`) | approximately 25 ms |
| `kv_race` | the real resources under several threads | approximately 2 ms |

`model_run` costs the least, so it runs the most cases. `kv_run` is the only
test in this repository that drives a real resource with a generated workload.
Tier 0 drives the same lifecycle against stubs, which covers the order and the
scope of the calls and no behavior.

## The `model_run` machine

`model_run` is the first machine of the tier, and the first step of lane 1a.
It generates a model. It runs that model over the real graph layer of mstar
(`mstar/graph/`). It then compares every value that reaches the client against
an interpreter of the same model. It uses no GPU, no weights, no kernel and no
process. One case costs a few milliseconds, so it fits the `ubuntu-latest`
runner that `ci.yml` uses.

```bash
pytest fuzzer/tier1                                   # the corpus and a short search
python -m fuzzer.tier1 run --seeds 20000 --all
python -m fuzzer.tier1 replay fuzzer/tier1/corpus/model_run/<case>.json
```

| module | what it holds |
|---|---|
| `spec.py` | the shape of a model, and the wiring that the shape makes |
| `values.py` | the checksum of a step, and the way an edge carries it |
| `reference.py` | the interpreter: the oracle of one forward pass |
| `executor.py` | the driver: the steps of `Worker._postprocess_batch` |
| `resources.py` | the real resources, and the step that drives them |
| `model_run.py` | the machine over the graph layer |
| `kv_run.py` | the machine over the graph layer and the resources |
| `kv_race.py` | the machine that drives the resources on several threads |

### The parts of a case

The **config** of a case is the spec of the model. It is a tree of
dictionaries. Thus it is JSON-able, and it replays:

```python
{"stages": [{"kind": "loop", "max_iters": 3, "ext": 1, "num_outputs": 1,
             "num_accum": 1,
             "body": [{"kind": "nodes", "width": 2, "fanin": 1},
                      {"kind": "nodes", "width": 1, "fanin": 2}]}],
 "num_requests": 2,
 "stops": [{"node": "s0L_s0n0", "run": 2, "loop": "s0L"}]}
```

A stage holds several nodes, or one loop with a chain of its own. A stage of
several nodes becomes a `Parallel`. A loop becomes a `Loop`, with declared
outputs, accumulated outputs and external inputs. The generator also puts a
loop inside a loop, which `tier0/graph_io` does not do.

The **ops** are the schedule: `start`, `step`, `drain` and `abort`. At each
step the driver runs any node that the graph reports ready, and `step(rid, k)`
chooses which one. A stop rule ends a loop at a logical point, which is the
k-th run of one node. It never ends a loop at a time from the clock. Thus a
case replays.

### The oracles

The interpreter in `reference.py` holds no queue, no ready set and no
registry. It runs the stages one after the other, and it runs each loop as a
Python loop. It therefore takes no schedule. Every schedule of the engine must
give the same values. One comparison thus holds the metamorphic oracle of the
tier, and the oracle for the order of a batch.

| invariant | what it holds |
|---|---|
| `emit.runs_match_reference` | each node ran the number of times the interpreter gives |
| `emit.matches_reference` | the values that reach the client are the values the interpreter gives |
| `route.every_edge_lands` | every edge that a node sends reaches a node of this graph or the client |
| `graph.pass_terminates` | a pass that started always ends, under every schedule |
| `state.clear_empties_the_graph` | after a pass, every slot, every loop and the registry are empty again |

`test_tier1.py` breaks the run on purpose one time for each invariant. An
invariant that cannot fail looks like coverage, and it gives none.

### What `model_run` found

`loop-tail-runs-before-a-sibling` (status `open`, 3 ops) is the failure that
`tier0/graph_io` already holds as two separate cases:
`loop-back-requeues-completed-node` and
`duplicate-completion-inflates-registry-count`. Tier 1 shows what those two
cost a client. In the saved case a loop of one iteration runs one node two
times, never runs a second node at all, and sends the client a value that no
correct execution produces.

The mechanism: the tail of the loop completes while a node of the same
iteration is still pending. The loop-back edge of the tail then arrives at a
node whose current slot is still full, so
`WorkerGraphStateRegistry.register_ingested_input` reads `ready_signals` of
the iteration that already ran, and puts that node into the ready queue again.
A schedule that takes the node there runs it a second time inside one
iteration. `GraphStateRegistry.mark_entity_complete` counts completions
without recording which entity completed, so the iteration then reports itself
complete although one entity never ran.

### A limit of the generator

The generator follows two rules:

* Every node sends a minimum of one edge. Two stages beside each other are
  fully connected: each node of a stage sends one edge to each node of the
  stage after it.
* The tail of a loop body holds one node.

Together the two rules make the tail the last entity of each iteration to
complete. That shape avoids the failure above.

A model that breaks either rule meets that failure almost every time. One
frequent failure hides every other failure of the same oracle. The search
keeps one case for each signature, and a case stops at its first failed op.
The corpus holds the shape instead. Take the rules out of `spec.py` when
somebody corrects the underlying bug.

### What `model_run` does not cover

* the resources. There is no batch, no cache and no slot. `kv_run` covers
  them.
* several worker graphs. One worker graph runs the whole model, so a lost
  cross-worker edge is out of reach.
* streaming edges, speculative execution, and tensor parallelism.
* the refcounts of the transport layer. `tier0/tensor_store` holds those.

## The `kv_run` machine — the real resources

`kv_run` generates the same models, drives them with the same ops, and checks
them with the same interpreter. What changes is where the value of a step
comes from. Each node of the model now declares a step against real resources,
and the machine drives the real cycle:

| what | which code |
|---|---|
| the cache | `KVManager`, built through `KVSpec.resource_class.build` |
| the positions | `PositionManager`, when the model declares one |
| the cycle | `StepRunner`: `ingest_request`, `admit`, `plan`, `commit`, `publish`, `remove_request`, and `pre_admit` / `pre_plan` / `clear_preplan` |
| the pages | a real `KVCache`, on the CPU, in `int32` |
| the transport | `LocalTransferEngine`, which a single-node deployment uses |

This machine uses no stub. The position resource names the cache in
`depends_on`. A case therefore also drives `resolve_spec_dependencies` and the
topological order of the runner.

### The checksum comes from the cache

A step reads the tokens that its own stream holds before it writes. That
number is part of every value the step gives.

Without that read, a step covers its identity alone and reads no page. A
resource can then give one page to two requests, address the wrong page, or
keep a page across a free. Every oracle passes. With the read, each of those
faults changes a number.

The value that a step writes into a token depends on the layer, and on the
position of the token inside the step. The V half of a slot is the K half plus
one. A write into the wrong layer, at the wrong offset, or into the wrong half
of a slot, therefore also changes a number.

`test_tier1.py` holds a test named `test_a_step_really_reads_the_cache`. It
changes one token of a stream, and requires the value of the next step to
change. Every cache oracle needs that property.

### One step runs a batch

One step of a node runs every request that is ready on that node. It puts one
`Segment` for each request into one `SubmoduleStep`, as the micro scheduler
does. The requests then share a plan, a packed token order and a cache. An
oracle therefore sees data that crosses between two requests. It cannot see
that in `model_run`, where each request holds its own copy of the graph and
nothing more.

### Pre-planning, capture and forks

Tier 0 reaches none of these three. It drives stubs, so it has no cache to
fork, no buffer to capture into, and no staged state to give back.

**Capture.** A generated model may declare a bucket. Every step then runs under
a `SlotLease`. The rows of the slot pad the batch, and the plan goes through
the static addressing of that slot. The rows that carry no token must address
`SINK_PAGE`. The machine writes those rows the way a replay does, directly
into the cache. A captured graph does not call `write_kv` either. The machine
reports a row that reaches a page of a live stream.

The padding rows hold real state. The engine opens one row for each slot, runs
the bucket full at capture, and leaves those rows with their labels.
`DummyRequestPool.release_all` then gives their pages back. The machine does
the same three things, so a replay finds the state that capture left.

**Pre-planning.** The `preplan` op stages a step through `pre_admit` and
`pre_plan`, one step ahead of its own run. The engine stages a step under a
lease only, and the machine does the same. A staged step must write the plan
buffers of a slot that the replay in flight does not read. A `step` op for the
same node and the same batch then promotes the staged step. An `abandon` op
drops it, and a `step` op for anything else also drops it. That second case is
what `preplan_is_stale` means.

No part of this reads the clock. The ops name the logical step and the action,
as the design notes below require. Thus a case replays, and a failure shrinks.

**Forks.** A step may carry a `pre_fork` or a `post_fork`. Each one copies the
stream of the step onto another label. A pre-fork copies in `plan`, before the
spans of the step land. A post-fork copies in `commit`, after them. Batched
CFG uses them.

The generator picks a target label that only a later stage owns. A label of
the same stage makes the result depend on the order inside that stage, and the
interpreter takes no order.

The interpreter holds a fork as the tokens of one stream copied onto another
label. It names no page and no reservation.

Over 200 generated cases, 114 declare a bucket, 70 pre-plan and 29 fork. Those
cases stage 102 steps, promote 73 of them and drop 29. They write padding rows
on 2941 leased steps, and they carry a fork on 198 steps.

### The oracles of the resources

| invariant | what it holds |
|---|---|
| `kv.admit_succeeds` | the cache is sized for the whole case, so no step is refused |
| `kv.admit_gates_the_plan` | a step that admit accepted plans and commits without raising |
| `kv.readback_matches_the_write` | the slots a step wrote read back as written |
| `kv.publish_matches_the_stream` | what a step publishes is the length and the pages the stream holds |
| `kv.pages_are_not_shared` | no page is held by two streams, and no held page is also free |
| `kv.pages_are_conserved` | the held pages and the free pages together are every page |
| `kv.remove_frees_the_pages` | after the last request goes, only the sink page is held |
| `kv.stream_matches_reference` | after a pass, each stream holds the number of tokens the interpreter gives |
| `pos.ids_match_the_stream` | the position ids of a step are the token indices its stream implies |
| `capture.padding_stays_off_live_pages` | the rows of a bucket that hold no token address the sink page |
| `preplan.staging_does_not_raise` | a step staged a step ahead is admitted and planned without raising |
| `preplan.abandon_restores_state` | a dropped stage gives back the length, the mark and the counter it moved, and keeps no label it invented |

Each invariant has a test that breaks the resource on purpose. The machine
must then report the damage. The tests break these nine things:

* a page that holds what another step wrote
* a page that reaches two streams
* a page that never goes back to the arena
* a write that lands at another address
* a position counter that moves alone
* a fork that never copies
* a padding row that addresses a page of a live stream
* a staged step that raises
* a rollback that does not happen

Two properties need no invariant of their own. A promoted step must give the
values of the same step without a stage. A fork must leave the target with the
tokens of the source. The interpreter holds both, and they appear as
`emit.matches_reference`. The test
`test_a_staged_step_gives_the_same_values_as_an_unstaged_one` also holds the
first one alone.

### What `kv_run` found

**`abandoned-preplan-keeps-its-new-label`** (status `open`, 1 op). A step that
is staged one step ahead, and that then never runs, keeps a label that it
made. It also keeps the pages that the label reserved.

`KVManager.admit` records a new label in `_preplan_new_labels` when a fork
made that label. It does not record a label that the segments of the step
made through `_ensure_label`. `clear_preplan` therefore does not remove that
second label. The comment on `clear_preplan` says that such a stream must go,
"because a stream at 0 that nothing asked for is still a stream, and it holds
the pages the reservation took". A speculative schedule that misses thus
leaves a stream that nothing asked for, holding pages, until the request ends.

**`zero-span-segment-has-no-stream`** (status `open`, 1 op). A `Segment` may
carry a span of zero. Its own docstring says so, and calls it a step that
reads its stream and does not extend it.

`KVManager.admit` skips such a segment on purpose, so it reserves no stream.
Its own comment says the same. The next call of the same cycle,
`KVManager.plan`, then reads `self._streams[rid][label]` with a plain
subscript, and raises `KeyError` (`manager.py:442`, in `_sequence_views`).

So `admit` accepts a step, and that step cannot run. `admit` is the gate, and
a step that it accepted must plan.

The capture path reaches this failure. `SubmoduleStep.declare_step` says that
a padding row declares its segment like any other row.
`CudaGraphRunner.pad_inputs` gives a padding row the length zero. A replay
that is not full therefore declares a zero-span segment for every padding row.

Bagel and cosmos3 both declare their segments under labels of their own, which
are the CFG branches, and not under `main`. `ingest_request` opens `main`
alone. Capture opens the other labels, because it runs the bucket full before
it records the graph. A padding row therefore holds the labels that capture
saw. It does not hold a label that capture never saw, and bagel chooses its
labels for each request from `requires_cfg`. A bucket that capture recorded
without CFG, and that a replay then uses with CFG, is the shape to check.

The machine names the contract `kv.admit_gates_the_plan`. It does not let the
`KeyError` be the signature. A signature that holds a line number moves when
somebody edits the file above that line. The corpus case would then report a
failure that it did not find.

### What `kv_run` does not cover

* **Eviction and offload under real pressure.** The cache takes a size that
  lets every admit succeed. An oracle for a refused admit needs a model of
  eviction, which this tier does not hold. `kv.admit_succeeds` holds the size,
  so this is a gap and not a silent pass. It is the next part to build.
* **The attention resources over this cache, the sampler, and `rms_norm`.**
  Those need a FlashInfer or a Triton kernel, which the CPU does not have.
  They belong to a lane on a device.
* **The two threads of the pre-plan path.** This machine stages a step and
  then promotes or drops it, on one thread. It therefore covers the state
  machine and not the race. `kv_race` drives several threads.
* **Retention, combined labels, and more than one cache in one model.**
  `KVStep` carries `combined_labels`, and `KVConfig` carries a retention
  policy. The generator declares neither one.
* **Retrieve across devices.** There is one walk and one rank. Every step
  drives `publish`, and `admit_retrieve` has nothing to bring in.
* Several worker graphs, streaming edges, speculation and tensor parallelism,
  as in `model_run`.

## The `kv_race` machine — several threads

`kv_run` drives the resource on one thread. `kv_race` drives it on several,
because the worker does. The GPU thread runs the step cycle. The main thread
scans for ready work, and it ends requests.

One thread runs at a time, and the case names which one. The driver comes from
`tier0/alloc_concurrent`. It parks each worker at a preemption point, and it
lets one worker go. Thus an interleaving replays, a race shrinks, and a race
becomes a corpus case. The preemption points are the lock of the manager, and
the reads that `plan` makes outside that lock.

The oracles are the ones that a race needs:

* the pages are conserved and never shared, at every point of the interleaving
* no thread raises
* no thread blocks forever
* every page goes back at quiesce
* a stream holds the tokens that the steps which committed to it wrote

### Where the threads meet

**The pre-plan path is not a race on the CPU.** `_preplan_spec` waits for the
`commit_done` of batch N before it stages batch N+1 (`worker.py:1269`). The
GPU thread then waits for `plan_future.result()` before the forward that reads
the staged plan (`worker.py:1361`). Both ends hold a fence. `kv_run` covers
the state machine between the two fences, on one thread.

The fences leave one window open. `engine.py:617` releases `commit_done`
before `_collect_outputs`, so `pre_plan(N+1)` overlaps the per-request tail of
step N. Two attributes in that window carry no lock: `_default_label` and
`_default_layer_idx` (`resources/base.py:246`). `KVManager.plan` resets them
at its first line, and `read_kv` reads them when its caller names no layer.

No submodule reads the cache in that window today. `_merge_per_rid` copies
tensors alone, and neither implementation of `unpack_packed_outputs` calls the
resource. The window is therefore a hazard with no reader. A lockset pass
reports such a hazard. A search over schedules does not, because no thread
takes the other side.

`kv_race` drives the pair that `publish` names: "`remove_request` can pop the
streams from another thread between the forward and finalize". It also drives
the arena, which a step that allocates and a teardown that frees both use.

### A measured limit

Take the mutual exclusion out of the manager lock, and keep every preemption
point. 400 cases then report nothing. This is not a gap in the oracles. The
threads of this machine touch separate streams, and the page arena holds a
lock of its own that tier 0 covers. The pairs that would contend for one
stream are out of reach here:

* `admit_retrieve` against `commit`. The comment on `commit` names this pair.
  `admit_retrieve` does nothing without a published info from another worker.
  With one, `LocalOnlyKVTransferEngine` refuses: "Cross-worker KV migration is
  unavailable for this accelerator".
* `offload` and `reload` against the step cycle. The comment on `offload`
  calls this pair "the real one". It needs pinned host memory, thus a device.

Both pairs need a lane with a GPU, or a transport that moves pages between
workers. This measurement says where the GPU time must go.

A sweep of injected faults says what the machine catches, and how often:

| the fault | schedules that report it |
|---|---|
| a `publish` without its guard | 19 of 300 |
| an arena that gives one page two times | 73 of 300 |
| a span counted two times | 174 of 300 |
| pages that never come back | 300 of 300 |

The first one needs the one interleaving in which a teardown runs between two
lines of `publish`. Each fault above is a test.

### What is still not tested

* **The search samples the interleavings. It does not cover them.** The driver
  makes every choice, so the next step is a search with a bound on the number
  of context switches. Two switches are enough for most races.
  `tier0/README.md` asks for the same thing.
* **A lockset pass, or a happens-before pass.** A search over schedules finds
  only a race that it reaches. Such a pass records, for each access to the
  shared state, which thread made it and which locks that thread held. It
  reports the hazard above without a reader on the other side. The comments of
  the manager already give the lock discipline that it must check.
* **Everything that CUDA orders.** The double-buffered slots, `preplan_event`,
  and the copies inside `offload`. Lane 1b covers them.

## The rest of lane 1a — the real conductor and the real worker

The step after `model_run` is to run the same generated model over the real
conductor and the real worker. That lane runs on the CPU, over the real ZMQ
and shared memory transports. The spec, the interpreter and the checksum do
not change. One part changes: a `FuzzModel(Model)` class builds its graph from
the spec.

`FuzzModel` implements the abstract base class at `mstar/model/base.py:246`.
It lives in `fuzzer/tier1/`, and it enters `MODEL_REGISTRY` when the fuzzer
imports it, so the shipped tree holds no test-only model. Its
`get_node_resources` gives the specs that `resources.py` already builds by
hand. The generator then samples more of its shape than `spec.py` does today:

* **topology** — edges to a different worker graph, streaming edges, and
  `enable_async_scheduling` for each node.
* **partitions and walks** — several partitions with streaming links between a
  producer and a consumer; generated tables of walk transitions; the split
  between prefill and decode.
* **resources** — `kv_run` generates the page size, the layer count, the head
  counts and the spans today. Add a retention policy, forks, combined labels,
  a sampler, and more than one cache for one model. Also add resources that
  cause trouble on purpose: one that pre-plans, one that only sets
  `force_double_buffer`, and one that refuses `admit` on a schedule.
* **sharding** — the tensor-parallel and sequence-parallel degrees; colocated
  placement and disaggregated placement. The files in `configs/` are the
  hand-written points in this space. The generator covers the space between
  them.

### The checksum must come from the cache

The submodules do no matrix multiplication. Each submodule returns a small
tensor. The contents of that tensor are a checksum.

The choice of what the checksum covers decides what tier 1 finds. Take the
checksum over two groups of inputs together:

1. the identity of the step: the request ID, the node, the walk and the
   iteration
2. the KV pages that this step reads, and the checksums of the inputs

`kv_run` does this. `values.checksum` takes the pages through its `extra`
argument, and `resources.ResourceRig.run_step` reads them out of the plan of
the step. `model_run` passes no `extra`, because it holds no cache.

Group 2 is not optional. A submodule that only hashes its identity never reads
the cache. The pages could hold anything, and every oracle would still pass. A
resource that corrupts a page is then invisible, which is the exact fault class
that tier 0 already cannot see. With group 2, a corrupted page changes a number
that you can compare.

This makes the oracles exact. Each of these faults changes a number that you
can compare:

* a lost edge
* an old buffer
* data that crosses between two requests
* a cache that a resource changed at the wrong time
* loop iterations in the wrong order

You need no tolerance and no reference weights.

### The oracles that the real worker adds

1. **A reference executor.** `reference.py` is that executor. It does not
   change for this lane.
2. **Metamorphic invariance.** Replay the same workload with different engine
   settings. The output for each request must be identical, byte for byte. The
   settings for this lane are `max_batch_size`, `MSTAR_NUM_SLOTS`,
   `MSTAR_MAX_CONSECUTIVE_SPEC_STEPS` and the eviction policy. A client must
   not be able to observe the concurrency or the batching.
3. **Quiesce and leaks.** Each request terminates or aborts. After that, the
   whole system must return to its initial state. Each KV manager must report
   `num_free == max_num_pages`. The `TensorStore` must be empty. The arena must
   reclaim its segments. No dictionary in the conductor, the worker or a
   resource may still hold a key for a dead request ID. Tier 0 already holds
   this property for the individual data structures, and
   `state.clear_empties_the_graph` holds it for the graph layer.
4. **Liveness.** Requests are outstanding and no counter advances for N
   iterations. That is a deadlock.

That lane runs the real resources, so it improves on tier 0. It still cannot
cover a resource on its target hardware. The CPU has no CUDA stream, no graph
capture and no page copy on a device.

What the device adds:

* **The pre-plan path, with its threads.** `pre_plan` runs on `plan_executor`,
  one step ahead of the step that `gpu_executor` still runs
  (`mstar/worker/worker.py:2682,2710,2891`). A resource that pre-plans can
  therefore change its own state from one thread while the step in flight uses
  that state on another thread. Tier 0 models neither thread. Add
  `MSTAR_PRE_PLAN_SPEC` and `MSTAR_TP_ASYNC_SCHED` to the metamorphic settings.
  The output with the pre-plan on must equal the output with the pre-plan off.
* **Rollback of an abandoned pre-plan.** A staged step does not always run.
  `Resource.clear_preplan` reverses it. Drive an abandoned pre-plan, then
  compare the state of each resource against its state before the pre-plan.
* **CUDA graphs.** Capture and replay, `build_cuda_graph_buffers`, and the
  interaction between the double buffer and the pre-plan slot. Compare eager
  mode against CUDA-graph mode against piecewise mode.
* **The attention backends.** FlashInfer and the paged kernels. These need
  `CUDA_HOME` from the shared toolkit; the node has a driver but no system
  `nvcc`.
* **Eviction under real pressure.** Size the cache so that offload and reload
  must run, then hold the quiesce oracle across them.
* **Tensor parallelism with real collectives.** TP1 against TP2, over NCCL.

## What tier 1 cannot do on its own

A race does not shrink. The shrinker keeps a candidate only when it fails with
the same signature. A fault that appears in one run of fifty has no stable
signature, so lane 1b reports it and then loses it.

Two ways to make a race reproducible, in the order to try them:

1. Record each decision of the schedule as a pair of the logical step and the
   action. Do not record the time from the clock. `model_run` already follows
   this rule: the ops name the node to run and the stop rules name the run of a
   node, not a moment. Tier 2 needs the same rule.
2. For a protocol, prefer an exhaustive search over the interleavings.
   `test/modular/tp_async_sim.py` is an explicit-state model checker that
   already does this for the tensor-parallel handoff. The handoff between the
   plan thread and the GPU thread has the same shape.

Also note that a GPU makes numbers less stable. The checksum submodules must
avoid a reduction whose result depends on the order of the additions.
Otherwise the metamorphic oracle reports a false failure.

## What tier 1 reuses

`fuzzer/common/` does not change. A case is still a config and a list of ops.
Here the config is the shape of the model plus the engine settings. Here the
ops are the arrivals of requests, the aborts and the changes of a setting. To
shrink a model is to delta-debug its list of nodes. `spec.shrink_spec` does
this: it drops a stage, it turns a loop into a node, it makes a loop shorter,
and it makes a stage narrower. Every candidate keeps the rules of a chain, so
a shrink step never makes a graph that the generator would refuse.
