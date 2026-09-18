# tier 1 — the synthetic model (not implemented yet)

Tier 0 fuzzes the data structures. Tier 1 fuzzes **the engine**. To do this, it
makes the model a generated input.

Tier 1 has two lanes. The lane decides what the tier can find.

| lane | hardware | where it runs | what it can find |
|---|---|---|---|
| **1a** | CPU | CI, and each night | graph topology, scheduling, the lifecycle of a request, leaks, deadlocks |
| **1b** | GPU | each night, on `team1` | everything in 1a, plus the behavior of a resource: memory, cache contents, pre-plan, CUDA graphs, eviction |

Lane 1b covers what tier 0 cannot reach. Tier 0 checks the order and scope of
the calls into a resource, not the work the resource performs, and it models
no threads. A CPU-only tier 1 would inherit both limits for anything that
needs a device.

## The synthetic model

Write a `FuzzModel(Model)` class. Register it in `mstar/model/registry.py`. It
implements the abstract base class at `mstar/model/base.py:246`. The generator
samples its shape for each case:

* **topology** — `Sequential`, `Parallel` and `Loop` sections inside each
  other. Also the fan-out, static loops, dynamic loops that an EOS token
  stops, edges to a different worker graph, streaming edges, and
  `enable_async_scheduling` for each node.
* **partitions and walks** — several partitions with streaming links between a
  producer and a consumer; generated tables of walk transitions; the split
  between prefill and decode
* **resources** — a KV resource with a generated page size, layer count and
  retention policy; a sampler; a position resource. Also add resources that
  cause trouble on purpose: one that pre-plans, one that only sets
  `force_double_buffer`, and one that refuses `admit` on a schedule.
* **sharding** — the tensor-parallel and sequence-parallel degrees; colocated
  placement and disaggregated placement. The files in `configs/` are the
  hand-written points in this space. The generator covers the space between
  them.

## The checksum must come from the cache

The submodules do no matrix multiplication. Each submodule returns a small
tensor. The contents of that tensor are a checksum.

The choice of **what the checksum covers** decides what tier 1 can find. Take
the checksum over two groups of inputs together:

1. the identity of the step: the request ID, the node, the walk and the
   iteration
2. **the KV pages that this step reads**, and the checksums of the inputs

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

## Lane 1a — CPU

Run the real conductor and the real worker over the synthetic model, on the
CPU, over the real ZMQ and shared memory transports. Use no GPU, no weights and
no HuggingFace. This lane fits the `ubuntu-latest` runner that `ci.yml` uses.

Oracles:

1. **A reference executor.** `run_graph` in `test/modular/test_graph.py:10` is
   already a small sequential interpreter for a graph. Make it the
   specification: no batching, no speculation, no tensor parallelism and no
   CUDA graphs. The output of the engine must be equal to the output of the
   reference.
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
   this property for the individual data structures.
4. **Liveness.** Requests are outstanding and no counter advances for N
   iterations. That is a deadlock.

Lane 1a runs the real resources, so it improves on tier 0. It still cannot
cover a resource on its target hardware. The CPU has no CUDA stream, no graph
capture and no page copy on a device.

## Lane 1b — GPU

Lane 1b is lane 1a on a real device. It adds the settings and the oracles that
need one. Use the owner partition, as the project CLAUDE.md says:

```bash
srun --partition=team1 --gres=gpu:1 --cpus-per-task=24 --mem=200G --time=04:00:00 --pty bash
# tensor parallel needs two: --gres=gpu:2
```

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
   action. Do not record the time from the clock. Tier 2 needs the same rule.
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
shrink a model is to delta-debug its list of nodes. The present shrinker does
this as soon as `shrink_config` can drop a stage.
