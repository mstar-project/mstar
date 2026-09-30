# Agent instructions for M*

These apply to AI-assisted contributions to `mstar-project/mstar` and to the automated PR reviewer in [`.github/review/`](.github/review/). Humans: look at [CONTRIBUTING.md](CONTRIBUTING.md) for the short version; this file states the key invariants, and the reviewer cites it by number.

A human submitter must understand and defend every line of an AI-assisted PR, and must say in the PR description that AI assistance was used, which tests were run, and what the results were.

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
A model says *what* it needs; the engine decides *how* and *when*. The slogan is short, so here is the test that actually decides where something belongs.

**Ask these three questions. Any yes means it is the engine's concern, not your model's.**

1. **Would any other model want this?** Engine code runs for every model, and a `Resource` arrives with preplanning, CUDA-graph capture and FlashInfer attention planning already wired up for everyone. If your model needs recurrent state, a sampling variant, or a new kind of attention, so will the next one — there is no reason to roll it by hand inside one package.
2. **If the thing underneath it changed, would the fix be a config edit or a sweep through model packages?** This is the whole point of the abstraction. When a better attention kernel library appears, a model that declared an attention resource moves to it in a few lines of config once the engine work is done. A model that called the kernel directly has to be rewritten, and so does every model that copied it.
3. **Is it ordering-sensitive bookkeeping around capture?** Preplanning, buffer management and double-buffering, graph replay, incrementing sequence-length counters, allocating KV pages outside the captured region — this is the category where hand-written code is not just duplicated but *wrong*, in ways that surface as a corrupted cache or a silent garbage output rather than an exception. One instance worth naming: a step that cannot be admitted has to become *backpressure the scheduler can resolve by eviction*, not an exception inside a forward pass. Code that allocates its own state turns a schedulable condition into an OOM.

Model code that reimplements an engine mechanism is a defect even when it works, because it silently opts out of batching, capture, eviction, offload and backpressure.

Concretely, nothing under `mstar/model/` may:

- **Capture CUDA graphs.** No `torch.cuda.graph`, `torch.cuda.CUDAGraph`, `graph_pool_handle`, or `make_graphed_callables`. Capture belongs to [mstar/engine/cuda_graph_runner.py](mstar/engine/cuda_graph_runner.py). A model declares a `CudaGraphConfig` — `BatchedCudaGraphConfig`, `PackedCudaGraphConfig`, or for an inner loop `PiecewiseBatchedConfig` / `PiecewisePackedConfig` — from [mstar/engine/cuda_graph_config.py](mstar/engine/cuda_graph_config.py), and the engine captures it. As of this writing there are **zero** raw capture calls under `mstar/model/`; keep it that way.
- **Allocate, free, or evict resource state.** No calls into `PageArena`, `PageAllocator`, `CacheStream`, `CPUPagePool`, or `WorkspacePool`, and no direct `admit`/`commit`. Declare resources in `Model.get_node_resources()` and declare each step's effect on them in `NodeSubmodule.declare_step()`. The engine runs the lifecycle.
- **Choose which requests form a batch.** Batch *selection* is the micro-scheduler's job: implement `can_batch` and `forward_batched`, set `input_seq_len` in `prepare_inputs`, and let it decide. To be clear about what this does *not* rule out — mechanically stacking or concatenating tensors in `preprocess` is normal and expected, and padding up to the *captured batch size* is the CUDA graph manager's job rather than something you do yourself. Some audio decoder submodules today pad to a maximum sequence length themselves for capture compatibility; that is accepted current practice and arguably something the engine should provide, so treat it as a precedent to leave alone rather than one to extend.
- **Keep its own device buffers where a pool already owns them.** Don't allocate a private scratch tensor per call.
- **Manage streams or synchronize.** No `torch.cuda.Stream`, `torch.cuda.synchronize`, or `torch.cuda.Event` under `mstar/model/`.

Things that look model-specific but belong to the system:

- **Anything that should be a new resource.** Especially where preplanning is needed for CUDA-graph compatibility — recurrent/mamba state wrappers, sampling logic. Added as a `Resource` under `mstar/engine/resources/` it is available to every model and lands inside the admit/plan/commit lifecycle. Reimplemented inside one model, it is neither.
- **Fundamental abstractions that cut across layers**, such as session state or bidirectional streaming. These touch the conductor, the worker and the engine, so a per-model version becomes dead weight the moment the system-level feature lands.

That second category is a judgement call about *effort*, not correctness. Building a cross-layer feature properly can be weeks of work, which is not a fair thing to demand of the PR that first needs it. The right outcome is to note it, land the per-model version, and assign the system-level work — which is what happened with session state in the cosmos3-edge PR. Flag these; do not block on them.

The one documented escape hatch for capture is a submodule that needs a dense, capture-safe KV path instead of paged FlashInfer — see [talker.py](mstar/model/qwen3_omni/components/talker.py#L445) for the shape of that argument. Note that even there the *engine's* runner still drives capture. A new escape hatch needs the same kind of comment: why the declarative path cannot express it.

### 2. Resources are declared per node, and a node gets only what it declares

`get_node_resources()` returns a `NodeResourceSpec` list; the engine builds one object per declaration and binds it into the submodule and its layers. Layers name their resource keys (`attn_key` / `kv_key` / `pos_key`). A node that declares nothing receives nothing — that is correct for VAE encoders, codec decoders, and projection/combine stages. Reaching around a missing declaration to find a resource off another node is a bug.

Per-deployment sizing goes in the config YAML under `resources:` overrides, not into Python constants. See "Tuning resources per deployment" in [docs/adding_models.rst](docs/adding_models.rst).

### 3. Submodule forwards are tensor-in, tensor-out

Aim for `forward` and `forward_batched` to be compilable and CUDA-graph capturable, which in practice means a pure tensor → tensor function: no host syncs, no data-dependent Python control flow, no shapes that vary outside a declared capture bucket, no I/O. Per-request bookkeeping goes in `prepare_inputs` / `preprocess`, and stopping decisions go in `check_stop`. The existing models are the best reference for how far this can be pushed.

### 4. Don't break the async worker

By default the worker speculates step N+1 through the graph while step N is still running (`enable_async_scheduling`, [mstar/graph/base.py](mstar/graph/base.py#L240), default `True`) — though for lockstep-parallel (TP/SP) nodes this is gated off by `MSTAR_TP_ASYNC_SCHED`, which defaults to `0` and is expected to flip on later, so a default `tp2` run is not exercising the TP speculation path at all. It pre-plans it on a separate thread and stream. The full stage order, thread by thread, is in [.claude/skills/async-worker/SKILL.md](.claude/skills/async-worker/SKILL.md) — **read it before changing the worker's main loop, the GPU thread, the plan thread, speculation or postprocess.**

There is deliberately *no* mechanical rule here, such as "every mutation on the normal follow path must also happen on the speculative one" — the two paths differ on purpose, and their rollback logic differs by speculation kind. The invariant is that a change to this code is reasoned about in terms of the documented stage order, which thread it runs on, and what is still in flight at that point. Guessing produces plausible, wrong conclusions.

Three specific things break the overlap, and all three are easy to do by accident:

- **It "wastes" the last step of a dynamic loop.** AR decode accepts that cost, because it cannot know its length in advance. Flow and diffusion nodes know their iteration count at request ingestion, so they exit early in `prepare_inputs` for the extra step and pay no extra compute — only some CPU work, which is overlapped anyway. A node that can do neither sets `enable_async_scheduling=False`.
- **A host sync outside `check_stop` forces synchronous execution.** `check_stop` is fed CPU tensors after a D2H that the worker prematerializes on a side stream, so it is free. Any other `.item()`, `.cpu()`, `.tolist()`, or print of a device tensor in the step path collapses the pipeline back to serial and costs far more than the code looks like it does.
- **A new resource doing H2D into fixed buffers must set `force_double_buffer`** (see `Resource.force_double_buffer`, [mstar/engine/resources/base.py](mstar/engine/resources/base.py#L191)). Without it, step N+1 overwrites buffers step N is still reading.

### 5. A change for one model must not degrade the others

`mstar/model/` is shared ground. Changing global state to suit one model — the `torch.compile` configuration, a default that disables the async worker, a process-wide env var, an allocator setting — is a regression for every other model in the stack, and one that no test for your model will catch. Scope it to your node or your resource declaration instead.

### 6. Ranks in a lockstep instance must stay in lockstep

A node's TP×SP block is its **lockstep instance** ([mstar/distributed/communication.py](mstar/distributed/communication.py)). Every rank in it must execute the same sequence of collectives, in the same order, with the same shapes — including on failure paths.

The failure mode that actually happens: a readiness scan, admission check, capture decision or early `return` that consults **per-rank** state, so one rank takes a branch the others don't and the instance deadlocks on the next collective. Anything that decides control flow for the instance must be derived from state all ranks agree on, or reduced across the group first — `cuda_graph_runner.py` ANDs capture flags across the joint group for exactly this reason, and barriers once per spec whether it passed or failed.

Review any new early return, `continue`, or exception path in worker and engine loops against this. A hang is much more expensive to debug than an error.

### 7. Python and Rust implementations must not drift

`MSTAR_RUST_GRAPH` and `MSTAR_RUST_ZMQ` select between the Python runtime/transport and the vendored Rust ones under `rust/`; `MSTAR_WIRE_CODEC` selects the encoding. `AUTO` means production may run either. A behavior change to the graph runtime, the wire format, or edge semantics must land on both sides or be explicitly gated, and the CI `rust-transport` job must cover it. A change to one side alone is a finding.

### 8. Every new environment variable is documented

`MSTAR_*` knobs are read all over the codebase and are the main deployment interface. A new one must land with a row in [docs/environment_variables.rst](docs/environment_variables.rst) giving its default and its meaning, and should name the function that reads it.

### 9. Performance claims come from the repo's own harnesses

Don't assert a speedup without a number. Which tool depends on *where* you think the time goes:

- **Worker-level timing — start here.** `benchmark/worker_phases/` (see its [README](benchmark/worker_phases/README.md)) gives a per-phase wall-clock breakdown of the worker loop. Prefer it over `mstar/profile/`, whose output is per-request and much harder to read. It is also how you answer the first question of any tuning work, **CPU-bound or GPU-bound**: `event_sync` is the real GPU wait, so `event_sync` near zero means CPU-bound, and anything else means GPU-bound and you go profile kernels. Note that `await_gpu` is *not* that signal — it is waiting on the CPU part of the GPU thread — and `check_stop` is a side-stream D2H rather than pure CPU work.
- **API-server-level timing.** If the suspicion is above the worker, `--log-stats` (the `mstar/profile/` timings) is the only thing that sees it.
- **End-to-end throughput and latency.** `python -m benchmark.runner`, `benchmark/run_benchmark.sh`, `perf_testing/offline_homogenous.sh`.

Getting a trustworthy number out of these is its own skill — concurrency sweeps, warmup, which metrics survive run-to-run variance, and why a cross-process A/B can move without any code change. See [.claude/skills/benchmarking/SKILL.md](.claude/skills/benchmarking/SKILL.md) before running one.

A PR that changes a hot path — scheduling, attention planning, capture buckets, transport — should say which harness was run and on what hardware, or say why measurement doesn't apply. Micro-optimizations without a measurement are not improvements.

Two things about what the number is for. **Don't optimize against one benchmark.** Being better than or on par with the competition across a comprehensive set is the goal; a win on a single configuration usually means it was tuned for. And **check the workloads you didn't measure**: an optimization that helps `image_to_text` at the expense of `text_to_text` or a mixed workload is a regression, and it is the cross-workload analogue of invariant 5. Say which workloads you checked, not just which one improved.

### 10. Configs map nodes to hardware; code stays hardware-agnostic

`configs/*.yaml` `node_groups` map graph node names to GPU ranks. The same model code must run on one GPU or many with only the config changing. Don't branch on rank counts or device indices in model code.

### 11. Cross-request prefix reuse has a narrow contract

Prefix caching retains a finished request's KV pages so the next request with the same prefix skips recomputing them. Design and rationale: [RFC #210](https://github.com/mstar-project/mstar/issues/210). The rules that make it safe:

- **Sharing is by owner count, and only sealed pages may be shared.** `PageArena` tracks `num_owners` per page and is the only path to the allocator; a page is freed when the count hits zero, not when a request ends. Appends land in the last page until it fills, so a partially filled page under two owners would have one owner mutating bytes another reads — the tail page therefore stays exclusive, and there is an assertion for it. Sealed pages are never written again; a stream that rewinds across one drops its pages and reacquires private ones.
- **The key must name everything the stored bytes depend on.** SHA-256, as a chain (`resources/kv/keys.py`: `fingerprint`, `page_key`, `chain`) rooted in a process fingerprint covering weights, TP head slice, dtype, KV layout, page size, attention backend and position config. Change any of those and every old key becomes unreachable, so the cache empties itself. Media folds in a digest of raw bytes plus preprocessor params plus encoder fingerprint — placeholder token ids alone are identical across images, which is a collision both vLLM and SGLang have shipped. **Adding anything that changes hidden state without adding it to the chain is a correctness bug, not a cache-miss bug.**
- **The matched length is the minimum across the node's resources** (`StepRunner.resolve_cached_prefix`; a resource with no opinion is not an answer of zero), resolved on the CPU before `prepare_inputs`. Bucket selection, attention planning, positions and commit all see a shorter request and never learn a hit happened. Keep it that way.
- **Eligibility is narrow deliberately:** a fresh stream, a keyed walk, a linear position scheme with no custom position ids (asserted), and no extra tensor inputs, kwargs or resource step info. Widening it means proving the stored bytes are valid at the positions they will be read at.
- **A live request must never fail to allocate because of cached data.** Release runs where the shortfall surfaces, in `_alloc` under the arena lock, and removes **leaves only**, so parent pointers never go stale.
- **TP > 1 is not correct yet** — each rank probes its own index and nothing reconciles the matched length across a lockstep instance. Tracked in [#308](https://github.com/mstar-project/mstar/issues/308); don't file it again, and don't assume rank agreement in new code.

### 12. Async partitions coordinate through streaming edges

Distinct from the async worker in invariant 4. An **async partition** keeps its own set of graph walks with independent state transitions; data crosses between partitions on **streaming edges**, buffered in `StreamBuffer`s on the consumer worker and handed to graph nodes according to a chunk policy. Two hazards, both of which have bitten:

- **Cross-partition races.** Partitions that must advance together need an explicit mechanism. qwen3-omni's thinker and talker have to stay in lockstep, and specifically the talker must transition to its decode walk at the right moment: the conductor sends a dummy edge for every new graph walk to mark that walk's start, after which decode runs as a dynamic loop with no further conductor intervention. Making the transition thinker-initiated was attempted and is unfinished. Don't introduce a second coordination mechanism without saying why this one doesn't fit.
- **Deciding a streaming partition is finished.** This is easy to get subtly wrong. qwen3-omni and Orpheus are the references worth reading before writing a new one.

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

## Splitting work across PRs

Recommended practice rather than a rule — the reviewer raises these as notes and never blocks on them.

**Land system changes separately.** A new resource kind, or anything else under `mstar/engine/`, `mstar/worker/` or `mstar/conductor/` that a model change happens to need, reviews better on its own and can usually merge earlier than the model that motivated it. Split it out, or stack it under the model PR.

**Consider stacking capture and optimization on an eager MVP.** Correct eager execution is a coherent thing to merge, and capture plus batching plus tuning stack cleanly on top of it — which also keeps parity debugging out of the same diff as capture debugging. The counterweight is that some models, particularly large ones, are painfully slow end to end without CUDA graphs, so an eager-only merge is not always useful on its own. Judgement call per model, not a requirement.

## Style and structure

Readability rules the reviewer may cite as **S1**, **S2**, **S3**. These are the ones that come up again and again in AI-assisted PRs.

**S1. Name your aggregates.** A `NamedTuple` or `@dataclass` instead of a long tuple. Prefer a `LoraAdapter` with fields `a`, `b` and `sigma` over `tuple[torch.Tensor, torch.Tensor, float]`, which is opaque at every call site and gets worse as it grows. This applies to return types and to anything that crosses a function boundary more than once.

**S2. Write comments for a reader who wasn't there.** Concise, aimed at someone who has not seen the conversation, plan or debugging session that produced the change.

- No references to a plan's numbered or lettered steps.
- Don't narrate a specific failure in depth when the general statement is the actual justification. "This is required to maintain symmetric resource state between TP ranks and avoid deadlock" is better than several sentences tracing one rank admitting a batch, the other refusing, and a thread spinning on a collective until the NCCL timeout.
- Explain why, not what. Match the comment density of the file you're editing.

**S3. Reach for a class when the shape calls for one.** Several related functions with differing implementations — attention backends being the obvious case — want a base class with subclasses, particularly where the alternative is module-level global state. Likewise, a cluster of related fields and methods accreting on `Worker` or `Engine` usually wants to be its own object; the graph runtime on the worker and submodule management / the CUDA graph runner on the engine are the examples worth imitating.

Also:

- Keep PRs focused. Mechanical cleanups ride along with substantive work rather than landing as their own PR.
- Don't add a dependency without saying why in the PR description.
