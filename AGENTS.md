# Agent instructions for M*

These apply to AI-assisted contributions to `mstar-project/mstar` and to the automated PR reviewer in [`.github/review/`](.github/review/). Humans: look at [CONTRIBUTING.md](CONTRIBUTING.md) for the short version; this file states the key invariants, and the reviewer cites it by number.

A human submitter must understand and defend every line of an AI-assisted PR, and must say in the PR description which tests were run, and what the results were.

When explaining a correctness or performance result (in a PR description, a review, or to the person you're working with), point to the evidence for the mechanism: the code path, a profile, `TORCH_LOGS` output, a test that fails without the fix. A mechanism that sounds right but hasn't been checked should be labeled as a hypothesis. Plausible, unverified explanations have been wrong often enough in this codebase that they cost more than they save.

If a hang, crash, or other bug has several plausible causes, the tempting fix is to add a mechanism per cause. Every mechanism must satisfy the invariants if possible, and must be logically defensible. If a mechanism doesn't work, cannot be reasoned to be independently correct, or breaks an invariant, then it should be removed. Especially in tricky concurrency-sensitive places like TP lockstep, high-level reasoning can go a longer way than trying out several plausible fixes. Don't add a second mechanism while the first is unproven.

## Layer map

| Layer | Path | Owns |
| --- | --- | --- |
| API server | `mstar/api_server/` | HTTP, tokenization, media loading, streaming results back |
| Conductor | `mstar/conductor/` | request lifecycle, graph-walk transitions, worker selection |
| Worker | `mstar/worker/` | one process per GPU: micro-scheduler, async execution, tensor routing |
| Engine | `mstar/engine/engine.py` | compiles forwards, captures CUDA graphs, batches, runs admit/plan/forward/commit |
| Resources | `mstar/engine/resources/` | paged KV, attention plans, positions, samplers, etc. |
| Models | `mstar/model/` | computation graph, tokenization, resource *declarations*, submodules |
| Graph | `mstar/graph/` | M* graph primitives and logic |
| Communication | `mstar/communication/` | ZMQ control mesh, tensor transport (RDMA/TCP/SHM) |
| Rust | `rust/` | vendored transport, graph runtime logic, SHM arena, and API server, selected at runtime |

Background: [docs/architecture.rst](docs/architecture.rst). Model authoring, end to end: [docs/adding_models.rst](docs/adding_models.rst).

## Invariants

### 1. Models declare, the engine executes
In M*, much of the optimizations (e.g., cuda graphs, attention wrappers, sampling, batching) are provided on the engine level, and the model declares specifications for optimizations / resources it needs.

Some heuristics for when something should be implemented on the engine level instead of as model-specific code:

1. **Would any other model want this?** Engine code is available to every model, e.g., `Resource`s (e.g., attention managers, KV cache, samplers), CUDA-graph capture and replay, pre-planning. If your model needs recurrent state, a sampling variant, or a new kind of attention, so will future models.
2. **If the underlying implementation changed, would the fix be a config edit or a sweep through model packages?** Engine-level abstractions allow optimizations to be plug-and-play and easily replaceable. E.g., when a better attention kernel library appears, a model that declared an attention resource moves to it in a few lines of config once the engine work is done. A model that called the kernel directly has to be rewritten, and so does every model that copied it.
3. **Is there ordering-sensitive bookkeeping around capture?** The engine handles a lot of tricky bookkeeping: e.g., preplanning, buffer management and double-buffering, graph capture and replay, allocating resources for requests (along with eviction/reload), incrementing sequence-length counters, etc. These parts are particularly error-prone, and mistakes typically corrupt request data instead of surfacing as an exception.
4. **Does it touch the broader system beyond model execution?** Anything that could touch, e.g., request lifecycle, transfer of data between ranks, or M* graph node-level schedule and readiness should be the concern of the engine or worker (or other parts of the broader system). For instance, if a finite resource like a paged KV cache is over-subscribed, this becomes engine-level *backpressure the scheduler can resolve by eviction*, not an exception inside a forward pass. Code that allocates its own state turns a schedulable condition into an OOM.

Model code that reimplements an engine mechanism is a defect even when it works, because it silently opts out of batching, capture, eviction, offload and backpressure.

For example, nothing under `mstar/model/` should:

- **Capture CUDA graphs.** No `torch.cuda.graph`, `torch.cuda.CUDAGraph`, `graph_pool_handle`, or `make_graphed_callables`. Capture belongs to [mstar/engine/cuda_graph_runner.py](mstar/engine/cuda_graph_runner.py). A model declares a `CudaGraphConfig`: `BatchedCudaGraphConfig`, `PackedCudaGraphConfig`, or for an inner loop `PiecewiseBatchedConfig` / `PiecewisePackedConfig` from [mstar/engine/cuda_graph_config.py](mstar/engine/cuda_graph_config.py), and the engine captures it. As of this writing there are **zero** raw capture calls under `mstar/model/`; keep it that way.
- **Allocate, free, or evict resource state.** No calls into `PageArena`, `PageAllocator`, `CacheStream`, `CPUPagePool`, or `WorkspacePool`, and no direct `admit`/`commit`. Declare resources in `Model.get_node_resources()` and declare each step's effect on them in `NodeSubmodule.declare_step()`. The engine runs the lifecycle.
- **Choose which requests form a batch.** Batch *selection* is the micro-scheduler's job: implement `can_batch` and `forward_batched`, set `input_seq_len` in `prepare_inputs`, and let it decide. (Note: given a batch, mechanically stacking or concatenating tensors in `preprocess` is normal and expected, and some audio submodules also pad to a maximum sequence length themselves for capture compatibility).
- **Manage streams or synchronize.** No `torch.cuda.Stream`, `torch.cuda.synchronize`, or `torch.cuda.Event` under `mstar/model/`.

Things that look model-specific but belong to the system:

- **Anything that should be a new resource.** Especially where preplanning is needed for CUDA-graph compatibility — recurrent/mamba state wrappers, sampling logic. Added as a `Resource` under `mstar/engine/resources/` it is available to every model and lands inside the admit/plan/commit lifecycle. Reimplemented inside one model, it is neither. Before writing one, read [.claude/skills/engine-resources/SKILL.md](.claude/skills/engine-resources/SKILL.md).
- **Fundamental abstractions that cut across layers**, such as session state or bidirectional streaming. These touch the conductor, the worker and the engine, so a per-model version becomes dead weight the moment the system-level feature lands.

Some caveats:
- Sometimes, building a cross-layer feature will be a long-running effort that should not block the implementation of a model. In that case, the right outcome is to note it, land the per-model version, and assign the system-level work.
- There are cases where model-level state is lightweight enough that it can be held in the model itself; e.g., Qwen3-Omni allocates a fixed-size dense KV cache for the code predictor module.

The converse matters just as much: **model-level concerns shouldn't leak into engine code.** If a behavior only makes sense for some models, the engine should provide it as a helper that the model opts into, rather than applying it to every model. For example, with chunked prefill, only the last chunk of a prompt samples a real token. Rather than the engine filtering sampler rows for every model, `keep_final_chunk_samples` ([#365](https://github.com/mstar-project/mstar/pull/365)) wraps a submodule's `SamplerStep`, and each submodule whose sampled token is meaningful only on the final chunk calls it in `declare_step`.

### 2. Resources are declared per graph node, and a node gets only what it declares

`get_node_resources()` returns a `NodeResourceSpec` list; the engine builds one object per declaration and binds it into the submodule and its layers. Layers name their resources (`attn_key` / `kv_key` / `pos_key`). A node that declares nothing receives nothing (e.g., encoders and decoders with no persistent state or attention wrappers).

Per-deployment sizing goes in the config YAML under `resources:` overrides, not into Python constants. See "Tuning resources per deployment" in [docs/adding_models.rst](docs/adding_models.rst).

### 3. Submodule forwards are mostly tensor-in, tensor-out

Aim for `forward` and `forward_batched` to be compilable and CUDA-graph capturable, which in practice means a pure tensor → tensor function: no host syncs, no data-dependent Python control flow, no shapes that vary outside a declared capture bucket, no I/O. Per-request bookkeeping goes in `prepare_inputs` / `preprocess`, and stopping decisions go in `check_stop`. The existing models are the best reference for how far this can be pushed.

### 4. A change for one model must not degrade the others

Changing global state to suit one model, e.g., the `torch.compile` configuration, a default that disables the async worker, a process-wide env var, an allocator setting, could cause a regression for every other model in the stack. If possible, make changes configurable (adding a per-submodule `torch.compile` configuration would be a reasonable move, e.g.) so that other models are not affected.

If working on engine-level changes, don't assume all models have the same shape. Something that works for a simple autoregressive LLM should not break or regress anything for diffusion/flow-matching models, for instance.

If a change is globally beneficial, it should be extensively tested on other models in the system and independently defensible, ideally as another PR that your model work is stacked on top of. Otherwise, scope it to your node or your resource declaration instead.

### 5. For worker-level changes, Python and Rust implementations must not drift

`MSTAR_RUST_GRAPH` and `MSTAR_RUST_ZMQ` select between the Python runtime/transport and the vendored Rust ones under `rust/`; `MSTAR_WIRE_CODEC` selects the encoding. `AUTO` means production may run either. A behavior change to the graph runtime, the wire format, or edge semantics must land on both sides or be explicitly gated, and the CI `rust-transport` job must cover it.

### 6. Be careful about CPU overhead

When making worker- or engine-level changes to a generalizable system like M*, it is easy to introduce heavy per-request CPU overhead. For many models, GPU work per step may be in the 2ms range, so CPU overhead matters for scaling w.r.t. batch size.

Things to be wary of that have caused issues in the past, by no means a comprehensive list:
- Building a dataclass or other Python object, per request, per step
- Scheduling changes that involve checking more requests per-step, or increasing the amount of engine-level checks on the scheduling path
- Per-tensor computation / data transfer that could naturally be batched
- Passing many small objects through the Python <> Rust boundary. For instance, a `list[tuple]` or `list[dataclass]` is going to have a lot more overhead on the boundary than a parallel/columnar format. Specifically: `requests: list[int], values: list[int]` is preferable to `request_vals: list[tuple[int, int]]`.
- Sending large objects over ZMQ
- A field on a per-step message whose declared type the wire codec has no encoder for. It falls back to pickle silently, once per peer per step; `wire._encoder(T).__name__` says which path a type takes. A `@dataclass` of lists encodes typed where a `NamedTuple` of `deque`s does not.
- Work repeated every pass for requests that are waiting. When the ready set exceeds a batch cap, some rows wait in the backlog across many steps, so anything done per pass for them (re-measuring sequence lengths, rebuilding input tensors, readiness checks) is paid every step at high concurrency. Do the cheap checks (e.g., "is the caller's batch already full?") before touching queues or the backlog.

Removing redundant computation and memoizing where possible is good low-hanging fruit. E.g., a backlogged row's inputs don't change while it waits, so its measured length can be computed once and carried on the batch, and a plain readiness check followed by a length-aware one is the same engine call twice. Prefer storing derived per-row facts on the object that already carries the row over a side dict that needs its own invalidation.

If making worker-level changes, measure the change in CPU overhead rather than assuming, especially if trying to optimize. Throughput at high concurrency (32, 64) for a small model like Qwen 3.5 0.8B is a good litmus test (though it doesn't measure things like streaming and inter-worker communication because it's a simple model).

See https://github.com/mstar-project/mstar/pull/368 for some sources of overhead that have bitten.

### 7. Keep the async worker in mind

By default the worker speculates step N+1 through the graph while step N is still running (`enable_async_scheduling`, [mstar/graph/base.py](mstar/graph/base.py#L240), default `True`). It pre-plans it on a separate thread and stream. The full stage order, thread by thread, is in [.claude/skills/async-worker/SKILL.md](.claude/skills/async-worker/SKILL.md) — **read it before changing the worker's main loop, the GPU thread, the plan thread, speculation or postprocess.**

(Note: for lockstep-parallel (TP/SP) nodes this is gated off by `MSTAR_TP_ASYNC_SCHED`, which defaults to `0` and is expected to flip on later)

There's no mechanical rule to follow here (e.g., no "every mutation on the normal follow path must also happen on the speculative one" -- the two paths have different logic on purpose), but it's worthwhile to understand the flow of the speculation path and keep it in mind. The invariant is that a change to this code is reasoned about in terms of the documented stage order, which thread it runs on, and what is still in flight at that point. Unsubstantiated guesses are likely to produce plausible, wrong conclusions.

Some additional information:
- **Speculation "wastes" the last step of a dynamic loop.** AR decode accepts that cost, because it cannot know its length in advance. Flow and diffusion nodes know their iteration count at request ingestion, so they exit early in `prepare_inputs` for the extra step and pay no extra compute (only some CPU work). A node that can do neither sets `enable_async_scheduling=False`.
- **A host sync outside `check_stop` forces synchronous execution.** `check_stop` is fed CPU tensors after a D2H that the worker prematerializes on a side stream, so it is free. Any other `.item()`, `.cpu()`, `.tolist()`, `.numpy()`, `bool()`/`if` on a device tensor, or print of one in the step path collapses the pipeline back to serial and usually costs more than it appears. Values that are already known on the host (e.g., request settings decided in `process_prompt`) should travel in `step_metadata`, not round-trip through the device.
- **A new resource doing H2D into fixed buffers must set `force_double_buffer`** (see `Resource.force_double_buffer` in [mstar/engine/resources/base.py](mstar/engine/resources/base.py)). Without it, step N+1 overwrites buffers step N is still reading.
- **Moving GIL-holding Python to another thread does not parallelize it**; it only adds contention (overlapping `prepare_inputs` with the plan thread measured worse). For a host-bound step, the fix is deleting work, not moving it.

### 8. Configs map nodes to hardware; code stays hardware-agnostic

`configs/*.yaml` `node_groups` map graph node names to GPU ranks. The same model code must run on one GPU or many with only the config changing. Don't branch on rank counts or device indices in model code; any branching should be contained to engine-level code (or worker, conductor, etc.), and should ideally be done in a modular manner (e.g., the `XPUAttentionManager` is a good example -- a new resource that encapsulates the accelerator-specific logic). 

### 9. Update the documentation

**Models and resources**: should be documented in docs/models.rst and docs/adding_models.rst, respectively.

**Installation**: If the PR adds extra installation procedures (e.g., a package that has to be installed separately from the `pyproject.toml`), that should be documented in docs/installation.rst.

**Environment variables**: `MSTAR_*` knobs are read all over the codebase and are the main deployment interface. A new one must land with a row in [docs/environment_variables.rst](docs/environment_variables.rst) giving its default and its meaning, and should name the function that reads it. Environment variables that don't make sense to include in production (e.g., for A/B testing of a decision where there is a clear winner, or for debugging) should be removed. A tuning value a deployment may reasonably want to change (e.g., a per-step token budget) belongs in the serving config, not in an env var added for testing.

The same goes for code paths added for performance: A/B them, and delete the ones that don't measurably help rather than leaving them in behind a switch. On the Qwen3.5 PR, a stream manager and a fused-norm flag both measured neutral and were removed, and segment-count graph buckets were built, measured as no gain, and reverted.

### 10. Performance claims come from the repo's own harnesses

Don't assert a speedup without a number. Which tool depends on *where* you think the time goes:

- **Worker-level timing — start here.** `benchmark/worker_phases/` (see its [README](benchmark/worker_phases/README.md)) gives a per-phase wall-clock breakdown of the worker loop. Prefer it over `mstar/profile/`, whose output is per-request and much harder to read. It is also how you answer the first question of any tuning work, **CPU-bound or GPU-bound**:
  - `event_sync` is the only real GPU wait. Near zero means CPU-bound; a sizeable value means the CPU did wait on the GPU, so there is a GPU-bound component worth profiling at the kernel level. Note that `await_gpu` is *not* that signal — it is waiting on the CPU part of the GPU thread
  - `check_stop` is a side-stream D2H rather than pure CPU work.
  - A host sync is charged to whichever phase encloses it, not to `event_sync`, so a stray `.tolist()` shows up as an expensive `prepare_inputs` and makes `event_sync` look *smaller*.
  - The README lists which phases are waits rather than work. Never add a wait to a work total.
- **End-to-end throughput and latency.** `python -m benchmark.runner`, `benchmark/run_benchmark.sh`, `perf_testing/offline_homogenous.sh`. These may not work for your model; feel free to create more scripts in `benchmark/`, as many models already have.

Getting a trustworthy number out of these is its own skill: concurrency sweeps, warmup, which metrics survive run-to-run variance, and why a cross-process A/B can move without any code change. See [.claude/skills/benchmarking/SKILL.md](.claude/skills/benchmarking/SKILL.md) before running one.

A PR that changes a hot path, e.g., scheduling, attention planning, capture buckets, transport, should say which harness was run and on what hardware, or say why measurement doesn't apply. Benchmarking should be done on many representative models instead of optimizing for a single model only. Micro-optimizations without a measurement are not improvements.

Two things about what the number is for. **Don't optimize against one benchmark.** Being better than or on par with the competition across a comprehensive set is the goal; a win on a single configuration usually means it was tuned for. And **check the workloads you didn't measure**: an optimization that helps `image_to_text` at the expense of `text_to_text` or a mixed workload is a regression, and it is the cross-workload analogue of invariant 5. Say which workloads you checked, not just which one improved.

**Situation-dependent.** Invariants 11-14 will not apply to every task and PR, but should be kept in mind if relevant.

### 11. Ranks in a lockstep instance must stay in lockstep

A node's TP×SP block is its **lockstep instance** ([mstar/distributed/communication.py](mstar/distributed/communication.py)). Every rank in it must execute the same sequence of collectives, in the same order, with the same shapes — including on failure paths.

The failure mode that actually happens: a readiness scan, admission check, capture decision or early `return` that consults **per-rank** state, so one rank takes a branch the others don't and the instance deadlocks on the next collective. Anything that decides control flow for the instance must be derived from state all ranks agree on, or reduced across the group first — `cuda_graph_runner.py` ANDs capture flags across the joint group for exactly this reason, and barriers once per spec whether it passed or failed.

Review any new early return, `continue`, or exception path in worker and engine loops against this. A hang is much more expensive to debug than an error.

Lockstep is also about **when** something is run, not only whether. The main thread runs ahead of the GPU thread within a single step (invariant 7), so "one step in flight" does not mean "that step has admitted": admit is on the GPU thread while the main thread is already applying teardowns and scanning readiness, both of which move the state admit reads. A rank can then admit a step in a window that exists only between two of its own state changes, and a rank replaying those changes in order admits the same step from a different state. Every mutation of replicated state has to land before the step is broadcast or after that step has admitted, never between.

**Replicating a decision the other ranks can't derive.** For instance, with KV cache eviction, LRU orders on wall clock, so the ranks could pick different victims if left to their own devices. One rank decides and the others replay its decisions, which puts four requirements on the journal.

Some lessons learned from debugging KV cache symmetry across TP ranks:
- **Ordered, not summarized.** A set of offloads plus a set of reloads cannot say whether a request ended up resident; the order of actions must be sent, and replayed verbatim.
- **Many parts of the code affect resource state.** A teardown releases pages without offloading anything, so no resident-set delta accounts for it: it has to carry the step it followed, and the rank that stamped it has to honour its own stamp.

Process-group creation is a collective too. `dist.new_group(ranks, backend="gloo")` while the default group is a device-bound NCCL group sends ranks *outside* `ranks` into an `ncclCommSplit` they never return from, which looks like a startup protocol hang. Create subgroups with `use_local_synchronization=True` so only members participate.

### 12. Async partitions coordinate through streaming edges

Distinct from the async worker in invariant 7. An **async partition** keeps its own set of graph walks with independent state transitions; data crosses between partitions on **streaming edges**, buffered in `StreamBuffer`s on the consumer worker and handed to graph nodes according to a chunk policy. Two hazards we have seen so far:
- **Cross-partition races.** Partitions that must advance together need an explicit mechanism. Qwen3-Omni's thinker and talker have to stay in lockstep, and specifically the talker must transition to its decode walk at the right moment: the conductor sends a dummy edge for every new graph walk to mark that walk's start, after which decode runs as a dynamic loop with no further conductor intervention. Making the transition thinker-initiated was attempted and is unfinished. Don't introduce a second coordination mechanism without saying why this one doesn't fit.
- **Deciding a streaming partition is finished.** This is easy to get subtly wrong. Qwen3-Omni and Orpheus are the references worth reading before writing a new one.

### 13. Cross-request prefix reuse

Prefix caching is also its own skill: [.claude/skills/prefix-caching/SKILL.md](.claude/skills/prefix-caching/SKILL.md). Prefix caching retains a finished request's KV pages so the next request with the same prefix skips recomputing them, and has a contract that a contributor needs to understand before touching the KV cache.

### 14. Worker-local request handles don't leave the process

A worker keys its own state by a small integer handle ([mstar/worker/rid_table.py](mstar/worker/rid_table.py)); messages carry the request id string, and the two are translated at the process boundary. Handles are minted per worker and **recycled**, which has three consequences.

- **A handle on the wire is a bug.** Rank 0's handle names a different request on rank 1, so translate at the send site. The same goes for a log line meant to be matched across ranks: with per-rank handles, the two ranks' lines can't be lined up, which is usually the only reason the line exists.
- **New handle-keyed state has to be purged when the handle is released**, or it silently attaches to whichever request is given that handle next. `rid_table.py` lists the state known to be handle-keyed; add to it. Clearing at the top of a per-step routine is not a substitute, because early returns skip it.
- **An unknown rid off the wire is expected; an unknown rid from local state is a bug.** A message can name a request this rank has already removed, so the lookup returns `None` rather than raising — handle that case explicitly at each call site, and say which of the two it is.

## Testing

```bash
ruff check .                          # CI enforces this
python -m pytest @test/cpu-core.txt   # the CPU Core CI job, no GPU or weights
pytest test/modular/                  # broader CPU graph/worker tests
pytest test/integration/              # needs GPU + weights
cd rust && cargo test --release --no-default-features && pytest ../test/rust/
```

Adding a CPU-only module to CI means adding its path to [test/cpu-core.txt](test/cpu-core.txt) — see [CONTRIBUTING.md](CONTRIBUTING.md).

The modular tests exist because these are the parts that are easy to get wrong: `test_resource_runner.py` (the admit/plan/commit lifecycle from `declare_step`), `test_cuda_graph_capture.py` (bucket keys, `cg_key_info`, `additional_key_info`), `test_micro_scheduler.py` (admission, failure, backpressure), `test_admit_failure_handling.py`, `test_kv_offload.py`.

Note that modular tests run models in **dummy mode**, where `get_submodule` returns `None`, and that several fakes stub worker internals **by name** — renaming a private on `Worker` can break them without any grep hit at the definition. Run the suite, don't just grep.

If a bug only reproduces through a running server, write a standalone harness before the third attempt. A server restart can be minutes of compilation, while a harness that builds the resources directly, drives `admit`/`plan`/`commit` by hand, and captures and replays a graph runs in seconds — [test/modular/test_gdn_cuda_graph.py](test/modular/test_gdn_cuda_graph.py) is the shape to copy. Where the bug is inside a captured graph, intermediates are invisible after a replay; returning `torch.isfinite(x).all()` per layer as an extra graph output gives a flag tensor readable afterwards, which names the first bad kernel instead of the first place the damage shows.

Before attributing a test failure to your change, check whether it also fails on `main`; some modular tests have pre-existing failures. Compare the set of failing tests against main, not just the count.

When rebasing onto a `main` that moved, check each resolution for hunks that kept **both** sides. Two got through one rebase here: one reinstated a behaviour the commit existed to remove, and the other defeated a guard's fall-through so a step admitted with nothing reserved. Both read as plausible merges, and neither conflicted again afterwards — the modular tests caught them, which is the argument for running them per step rather than at the end.

The Rust extension under `rust/` is a build artifact, so it goes stale after merging or rebasing onto a `main` that changed `rust/`. A stale build imports fine and then fails at request time (often as an unexpected-keyword `TypeError` from a runtime call), and tests that need it (e.g., `test_graph_runtime_factory.py`) fail in a checkout where it isn't built. Rebuild before believing either.

Unix-socket paths are limited to about 107 characters, so a long `--basetemp` or `TMPDIR` makes the ZMQ-based Rust tests fail with `File name too long`, which reads like a real failure. Use a short `--basetemp`.

## Splitting work across PRs

Recommended practice: the reviewer raises these as notes and never blocks on them.

**Land system changes separately.** A new resource kind, or anything else under `mstar/engine/`, `mstar/worker/` or `mstar/conductor/` that a model change happens to need, reviews better on its own and can usually merge earlier than the model that motivated it. Split it out, or stack it under the model PR.

**Consider stacking capture and optimization on an eager MVP.** Correct eager execution is a coherent thing to merge, and capture plus batching plus tuning stack cleanly on top of it — which also keeps parity debugging out of the same diff as capture debugging. The counterweight is that some models, particularly large ones, are painfully slow end to end without CUDA graphs, so an eager-only merge is not always useful on its own. Judgement call per model, not a requirement.

## Style and structure

Readability rules the reviewer may cite as **S1**, **S2**, **S3**. These are the ones that come up frequently in AI-assisted PRs.

**S1. Name your aggregates.** A `NamedTuple` or `@dataclass` instead of a long tuple. Prefer a `LoraAdapter` with fields `a`, `b` and `sigma` over `tuple[torch.Tensor, torch.Tensor, float]`, which is opaque at every call site and gets worse as it grows. This applies to return types and to anything that crosses a function boundary more than once. Pick the type to signal intent: a `NamedTuple` (or frozen dataclass) for a value that shouldn't change, and a mutable `@dataclass(slots=True)` for state that is updated in place, e.g., a row's chunked-prefill progress.

**S2. Write comments for an outside human reader.** Concise, aimed at someone who has not seen the conversation, plan or debugging session that produced the change.

- No references to a plan's numbered or lettered steps.
- Don't narrate a specific failure in depth when the general statement is the actual justification. "This is required to maintain symmetric resource state between TP ranks and avoid deadlock" is better than several sentences tracing one rank admitting a batch, the other refusing, and a thread spinning on a collective until the NCCL timeout.
- Explain why, not what. Match the comment density of the file you're editing.
- Check comments for staleness when the code under them changes, especially comments that state lifecycle or ordering semantics. A comment that was true of an earlier draft (e.g., "a chunked prompt forks on its first chunk", when pre-fork happens on the first chunk and post-fork on the last) is worse than none.

**S3. Reach for a class when the shape calls for one.** Several related functions with differing implementations — attention backends being the obvious case — want a base class with subclasses, particularly where the alternative is module-level global state. Likewise, a cluster of related fields and methods accreting on `Worker` or `Engine` usually wants to be its own object; the graph runtime on the worker and submodule management / the CUDA graph runner on the engine are the examples worth imitating. And when a new behaviour differs enough from an existing class's that supporting it means an optional argument that switches what the class does, consider a sibling class over a shared base instead. This is a judgement call rather than a rule: the ragged attention resource gained cross-attention as `RaggedCrossAttentionSpec` / `RaggedCrossPrefillWrapper` beside the self-attention classes, rather than an optional `kv_cu_seqlens` that changed what a plan meant, and the self-attention API stayed untouched.

**Also**:

- Keep PRs focused. Mechanical cleanups can ride along with substantive work rather than landing as their own PR, but be careful about scope creep.
- Don't add a dependency without saying why in the PR description.

## Design patterns that have worked

Not rules, and the reviewer shouldn't cite them; these are cleanups that came out of review and made the code easier to work with.

**One hook that returns a policy object, rather than several parallel hooks.** Chunked prefill first added four methods to the submodule base class (whether a walk is chunkable, its token budget, how its outputs accumulate, whether it reuses a cached prefix). They became one `get_chunking_policy(graph_walk) -> ChunkingPolicy` ([#365](https://github.com/mstar-project/mstar/pull/365)), a frozen dataclass with a callable field for the one dynamic case, which the engine caches per (node, walk). A submodule author implements one thing per concern and sees every option in one place, and the engine doesn't rebuild the answer every step.

**If a structural restriction can be enforced at startup, enforce it at startup.** Speculating into a chunkable (node, walk) isn't safe, since chunks are cut at schedule time from lengths a speculative batch doesn't know yet. Rather than the worker filtering speculation targets at runtime and hoping the runtime picks a safe one, the chunkable pairs are computed once when submodules load and passed to the graph runtime's constructor as `disable_spec_node_walks` ([#365](https://github.com/mstar-project/mstar/pull/365)), and the runtime never returns them from `speculate_node`. A component that can't produce the forbidden thing is safer than call sites that each have to remember to check.

## Performance ideas from past work

Not rules, and the reviewer shouldn't cite them. These paid off for Qwen3.5 against vLLM and SGLang, each measured with the harnesses in invariant 10; they are worth checking for a new model, but whether they help depends on where its time goes.

- **Fuse small projections at decode width.** Decode GEMMs are latency-bound, so one wide GEMM beats several narrow ones. Fusing attention's q/k/v through `QKVParallelLinear` and stacked loader rules gave +1.9% at 9B TP2, and one fused GDN input projection matched SGLang running two of them overlapped on a second stream.
- **Remove the eager kernels around a captured graph.** Per-step `.contiguous()` copies and zero-fills outside the graph cost +3.5% at 9B TP2 c=16; passing preallocated output buffers removed them.
- **Check what Inductor generates for small reductions at decode batch sizes.** Its fused residual-add + RMSNorm was about 3x slower than FlashInfer's `fused_add_rmsnorm` at batch 4, run 65 times a step; switching gave +6% at 4B c=4.
- **Under TP, fuse all-reduce, residual add and norm** (FlashInfer's one-shot kernel) rather than NCCL followed by a separate norm: +10-15% at 9B TP2.
- **Cap torch's intra-op threads for CPU work in the API server** such as image preprocessing. The default pool stalled about one preprocess in eight by 35-130 ms on a shared host.
- **Capture launch-bound encoders.** An eager ViT tower was ~4 ms of GPU work in a ~20 ms forward; a piecewise capture of its block loop (`PiecewisePackedConfig`) brought `image_to_text` level with vLLM.
