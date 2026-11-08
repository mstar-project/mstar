# The mstar way: tables and primitives

The derivation the mstar-way pass follows (SKILL.md §3 Phase A and §4.5 Phase B). mstar is a
dataflow runtime: a request is a walk through a graph of nodes, and the runtime already schedules,
batches, speculates, pre-plans, captures, streams, evicts, parallelizes and tears down anything
expressed as graph structure, edges and declared resources. A feature written that way inherits
all of that. A feature built beside the graph has to reimplement or patch each piece: speculation
that can't see it, a scheduler special case, a cursor dict, outputs held by hand, a lifecycle state.

**Precedent isn't proof of fit.** A recent mechanism, an open PR, or an example in AGENTS.md or a
doc may itself have been built beside a primitive. Judge every existing mechanism by the disguise
check before copying its shape. Only the primitives in the catalogue below are given; check each
one you rely on in the code.

## 1. The timeline (5–10 sentences, no class names)

What happens to one request over time: which computations run and in what order; how many times
each runs and whether that count is known on arrival; what each run consumes and when that data
exists (at the start, or arriving over time, and from whom); what is remembered between runs, how
large it grows, whether other requests could share it; what already exists when the request arrives
(cached prefixes, adapters, tables built at load); what decides a phase or the request is over; what
goes out and when; what really differs per model. Also say what the feature is not about: budgets,
fairness and priorities are tuning on top of the shape, not the shape.

## 1b. One thing, or a family?

Many features name a capability (context parallelism, quantization, speculative decoding, caching,
sparse attention, offload) behind which the field has a family of strategies, each winning in a
different regime. Building the first member hard-codes one regime, and every other member then
arrives as a second mechanism beside it.

- List the family from prior art (vLLM, SGLang, Megatron/TransformerEngine, the papers): what each
  member divides, approximates or caches, what it costs per step, when it wins, what it can't handle.
- Find the few shared axes every member is a combination of. Give each axis its owner and timescale
  (per step, per request, fixed at load). An axis lives with the primitive that owns that state, not
  in a central mode switch.
- Make each member a cell, not a code path: a resource declares how it responds to the axes, the
  engine resolves the declaration at load, defaults follow from the other axes, impossible
  combinations are refused at load with the reason, and thresholds are serving config.
- Place every member in the matrix ("later", or "refused, because …"); v1 builds the cells someone
  needs now. A member that fits no cell means the axes are wrong.

**Who decides whether it applies.** List what must be true for a step, request or node to use the
capability, and give each condition to the part that owns the fact (a resource for its state and
kernels, the graph for its shape, the deployment for placement). Each owner answers for itself with
a safe default; the engine composes the answers, not a flag it branches on per model when it could
work the answer out itself, and not a list of model or walk names. A helper the engine provides and a
model opts into is fine (AGENTS.md invariant 1). Facts fixed at load are resolved once at load. When the capability can't apply, one
startup line names the part and the reason. A fact that changes what a step means travels in the
step's own data, not as a side field every consumer must remember to apply.

## 2. Shape table

| If the timeline says… | …the primitive is |
| --- | --- |
| A computation runs more than once for one request, each run continuing from the last | a `Loop` over that node or section. Per-run state in a declared resource, the stop from `check_stop` (or `max_iters` for a fixed count), per-run outputs on edges (`EMIT_TO_CLIENT` each iteration, or `accumulated_outputs` at the end). A node that runs N times for a request *is* a loop, whatever triggers each run. |
| The number of runs is known on arrival | still a `Loop`; exit early in `prepare_inputs` on the overshoot step so the async worker's speculation costs no compute. |
| Data arrives over time, from any producer, at a rate the consumer doesn't control | a streaming edge into the consuming node: a `Connection` with a chunk policy, buffered in a `StreamBuffer`. A node whose stream hasn't delivered is not ready yet: readiness, not a new state. |
| Two parts of a request advance at their own rates | async partitions joined by streaming edges. |
| Different phases or inputs need different computations | separate graph walks, the conductor moving the request between them; walks that should share a batch are combined walks. |
| Engine-owned state at any scope: across one request's runs (KV, recurrent state), across requests (cached prefixes, an adapter pool), or the whole deployment's | a declared resource (`get_node_resources`, each step's effect in `declare_step`). The engine admits, plans, commits, evicts, offloads, prefix-caches and applies backpressure; `fingerprint()` keys every cache derived from it. Never a tensor, dict or cache owned by the model, worker or engine, and never a new state machine beside it. |
| A resource needs a different access pattern (a window, a sparse selection, a layout, another tier) | a new resource kind or a variant of one in `mstar/engine/resources/`. Variants of one role are one resource with the difference as config or a backend, not sibling copies of a lifecycle. |
| Which requests run this step, and how big each step is | the micro-scheduler (`can_batch`, `input_seq_len`, budgets). It never decides that a node runs again for a request; that is the graph's job. |
| Something is done | `check_stop` returning loop names, or the walk ending. Aborts, failures and disconnects use the existing teardown paths. |
| Where something runs (GPU, rank, TP width) | config `node_groups`, never code. |
| Behaviour that differs per model inside a shared mechanism | a declaration or policy object with a default the engine reads, never a method every model must implement when the engine could derive the answer. |
| A fixed-shape step repeated many times | an accelerator-graph config; the engine captures it. |
| The feature changes how the graph runs, not what it computes | not a new shape. State it for every shape it could apply to (a loop, A → B on one GPU, A → B across GPUs); the model declares nothing new. |

A sentence that fits no row doesn't get a new concept yet: find the nearest primitive and write the
extension that makes it fit. A new concept needs a stated reason why extending the nearest primitive
fails. Judge a primitive by its cheapest use, not one expensive variant.

## 3–4. Declaration and primitive extensions

Write what a model would declare (walks with nodes, `Loop`s and edges; partitions and `Connection`s;
`get_node_resources` and `declare_step`; `check_stop`) and the per-model adoption cost as a list. The
target is declarations only; if the feature fits in declarations, the shared-layer diff should be
close to empty.

List exactly what the primitives can't do yet. Each item is a generic extension of an existing
primitive that every model gets. A short list of extensions beats zero extensions plus a mechanism
beside the graph, which grows a patch per runtime feature.

- **Generic over resource kinds and computations.** A shared-layer mechanism never names or assumes
  one resource kind or one computation. State the need as a hook or declaration on the base class
  (`Resource`, `NodeSubmodule`, the scheduler or runner) in terms of what it means ("this step has
  finished on the device", "room needed", "pages to move between tiers"), with the motivating
  resource as one implementation.
- **Don't stop at the first shape.** A mechanism found on a `Loop` usually isn't about loops. Ask
  what it is on a plain A → B edge, across workers and across GPUs; if it holds there, that is the
  unit and the `Loop` is one cell.
- **Keep worker components apart** (scheduling, speculation, admission, lockstep, lifecycle, step
  execution). A feature crossing them gets one read-only table derived at load (as `ChunkLoops` is)
  and at most a field on the batch each component reads; no sideways calls, and the engine and
  resources never learn what the worker decided. Its code goes in its own small module, with call
  sites only in `worker.py`.

## 5. Disguise check

For every mechanism a design or PR puts in the worker, engine, scheduler, conductor or API server:

| Looks like | Is really |
| --- | --- |
| A node pushed back onto the ready queue for the same request, a per-request cursor or progress counter, outputs held until a final pass | a `Loop` built outside the graph runtime |
| Rebuilding readiness by hand, or a check at each step to keep the graph runtime away from something | structure the runtime can't see: put it in the graph, or hand the runtime the fact once at startup (telling it not to do something is fine) |
| Input buffered in the API server or conductor, windows cut by the server, persistent "sessions", polling for more input | a streaming edge |
| A per-request dict, tensor or cache in model, worker or engine code | a declared resource |
| A new lifecycle state ("waiting", "paused", "abandoned", "partial") | readiness on an edge, or the existing abort and teardown paths |
| A deployment-wide version, epoch or mode with its own state machine, for something the engine owns | a resource: its version as `fingerprint()`, requests holding references, admission backpressure for the switch. Only agreement across workers stays in the conductor. |
| A data-dependent loop inside `forward`, with a host sync | a graph `Loop` with `check_stop` and a resource |
| A second copy of an existing lifecycle (a second cache or manager beside a near-identical one) | one primitive generic over the variants |
| A hook every model must implement for a cross-cutting feature | the engine deriving the answer, or a declaration |
| A flag the engine branches on per model or per walk for something it could work out itself, or a central check that names resource kinds | each owning part answering for itself, composed by the engine (§1b); a helper a model opts into is fine (AGENTS.md invariant 1) |
| A side field on the step that each consumer must remember to apply | the step's own data carrying the change, or one shared accessor |
| A feature that needs special cases in several layers with no graph declaration | a feature at the wrong layer |
| New calls between scheduler, speculation, admission, lockstep and step execution, or a method on `Worker` several of them reach into | a read-only table in the worker's layout plus a field on the batch |
| Engine, worker, runner or scheduler code naming one resource kind or computation (`isinstance(KVManager)`, a KV-specific branch in a generic path) | a generic hook or declaration on the base class, with that resource as one implementation |
| A lifecycle, class or decision every resource with per-request state would need (admission class, residency, preemption, replication across ranks), built inside one resource | a contract on the base class or a fact on the request, with that resource as the first implementation |
| One strategy built as "the feature", or a second near-identical feature under another name | a family (§1b): shared axes, a per-resource declaration, this strategy as one cell |
| Followers reporting a per-rank fact to rank 0, and rank 0 waiting for the reports before it schedules | rank 0 deciding and sending the decision as a typed field on the step's `ScheduleTPNode`, as `prefix_hits` and `page_trails` go |
| A hand-kept list or count per node or model (the inputs that set a shape, the sampling keys, a page count) that a new model can get wrong | a value derived from the declarations, the checkpoint config, engine state or what rank 0 already decides |
| A gate every request waits on to cover a failure that almost never happens | handling on the failure path, paid only by the request that fails |

**Altitude check.** For each new concept (a state, class or flag, lifecycle phase, decision and who
makes it, journal or message), ask whether it is a fact about the request, the node or the
deployment, or genuinely about this one resource or computation, and compare where it lives with
where it belongs. A concept that applies to any resource with per-request state, or to any node,
belongs at that altitude even when v1 implements it for one resource. It is a hit when it lives only
inside one concrete resource, or when the feature works only because one resource happens to
implement it. In a stack, each PR that extends a low-altitude concept is a hit again, with the
earlier PR named as the root.

**Seam check.** For each class boundary the feature adds or reshapes, write what crosses it: what it
is constructed with (especially callables), class attributes and hooks subclasses set, flags its
methods and callers branch on, the shape of arguments and returns.

| Looks like | Is really |
| --- | --- |
| A flag or kind that several methods or callers branch on, with methods valid for one value only | one type per value; shared fields on a small base |
| A callable or object passed in whose required methods or signature are written only in a docstring | a `Protocol` or abstract base at the seam |
| What one part supplies arriving in pieces: class attributes only one subclass sets, a closure, a callback threaded through layers that don't use it | one object the supplier provides, with defaults, taken whole |
| Structured data flattened at a call and re-split by a count or position | pass the structure |
| A class generic except for a few lines tied to one representation | the generic part once, those lines behind a small adapter whose interface doesn't assume the representation |

Count concepts, not classes: a flag whose values have different methods is already two concepts, so
splitting it into two types adds none. "Fewest new concepts" never justifies keeping a mode flag or
an unwritten contract. A new class attribute, hook or callback every subclass inherits but one uses
is a new concept.

## 6. Free or patched

For the derived design and the locally easiest one (or the PR's), mark each row **free** (the
runtime already handles it) or **patched** (needs its own code):

- the async worker speculating and pre-planning step N+1;
- batching with other requests' steps, decode and prefill mixed;
- accelerator-graph capture;
- TP lockstep (every rank takes the same branch);
- the Rust graph runtime with no special case for the feature;
- streaming outputs to the client;
- prefix caching, eviction, offload and backpressure;
- abort, failure, disconnect and teardown, including handle-keyed state;
- placement across GPUs (`node_groups`) and disaggregation;
- a second model adopting it (count what it writes).

The design with more free rows and fewer new concepts wins. A design whose patched rows are mostly
"tell X about the new mechanism" is built beside a primitive.

## Primitives catalogue

Paths are on `main`; grep the symbol if a file has moved.

**Graph structure** (`mstar/graph/base.py`)
- `GraphNode`: one computation implemented by a node submodule; ready when its inputs have arrived.
  `enable_async_scheduling` (default on) lets the worker speculate its next step.
- `Sequential`, `Parallel`: composition.
- `Loop`: runs a section repeatedly for one request. `max_iters` is a safety bound; the real stop is
  a submodule's `check_stop` returning loop names, registered through `register_loop_finish_signal`
  (`mstar/graph/graph_io.py`), applied one step late (the async worker's one wasted step). External
  inputs are re-injected each iteration; a streamed input yields the next chunk instead
  (`Loop.ingest_external_input`). Loop-back edges carry per-iteration values; `outputs` are emitted
  at the end, `accumulated_outputs` appended and emitted at the end. A step can advance its loop by
  more than one (`loop_strides`, built for chunk loops; `mstar/worker/chunk_loops.py`). The runtime
  knows the loop, so speculation and readiness handle it without special cases.
- Graph walks (`Model.get_graph_walk_graphs`, `mstar/model/base.py`): named graphs the conductor
  moves a request between. `get_combined_graph_walks` declares walks that may share a batch;
  `get_loop_caps` sets per-request loop caps.
- `EMIT_TO_CLIENT` (`mstar/graph/special_destinations.py`): an edge destination that streams to the
  client; on a loop's node it emits every iteration.

**Data arriving over time**
- Async partitions (`Model.get_partitions` → `PartitionDefinition`, `mstar/conductor/request_info.py`):
  sets of walks advancing independently.
- Streaming edges (`Model.get_partition_topology` → `PartitionTopology`/`Connection`,
  `mstar/streaming/topology.py`), buffered on the consumer in a `StreamBuffer`
  (`mstar/streaming/stream_buffer.py`) and handed over by a chunk policy
  (`mstar/streaming/chunk_policy.py`). A `Connection`'s `consumer_walk` hook assigns streamed items
  their consumer walk (AGENTS.md invariant 12).
- Streaming readiness (`is_ready_for_streaming`, `mstar/graph/base.py`): a node missing only
  streaming inputs is ready when a chunk arrives. That is how "waiting for more input" is expressed.
- Initial inputs: `process_prompt` and `get_initial_forward_pass_args` (`mstar/model/base.py`) turn
  the API request into the first edges, once.
- Examples: Qwen3-Omni (thinker → talker), `nemotron_duplex`, Orpheus, fish_s2_pro.

**State** (`mstar/engine/resources/`)
- `Resource` (`base.py`): engine-owned state a node declares in `Model.get_node_resources`
  (`NodeResourceSpec`, `spec.py`); each step's effect in `NodeSubmodule.declare_step`. The engine
  runs admit → plan → forward → commit (`step.py`); admission failure is backpressure the scheduler
  resolves by eviction or offload. Cross-request hooks: `fingerprint()`, the prefix-cache contract,
  `offload`/`reload`. Resources pre-plan step N+1 during step N (`supports_preplan`); one that copies
  H2D into fixed buffers sets `force_double_buffer`. Sizing goes in the config's `resources:`
  overrides.
- Kinds: paged KV (`kv/`) with prefix caching (`kv/prefix_index.py`), CPU offload
  (`kv/cpu_page_pool.py`), bounded and ring KV (`kv/bounded`, `kv/ring`); attention managers
  (`attn/`); linear-attention state (`linear_attn/`); recurrent state (`recurrent/pool.py`);
  positions; samplers; norm workspaces; resident tables (`resident.py`: per-request device rows a
  captured step reads and advances itself).
- Skills: `.claude/skills/engine-resources/`, `.claude/skills/prefix-caching/`.

**Per-step execution**
- Submodule hooks (`mstar/model/submodule_base.py`): `prepare_inputs` (sets `input_seq_len`,
  bookkeeping, early exit on a known-count overshoot), `preprocess` (stacking a batch), `forward` /
  `forward_batched` (pure tensor → tensor, capturable), `can_batch`, `postprocess`, `check_stop`
  (the only place a host read of outputs is free).
- Declarative stops (`StopPolicy`, `TokenStop`), output spans (`spanned_outputs`), device loop-back
  (`DeviceLoopback`), resident walks (`resident_walks`).
- Micro-scheduler (`mstar/worker/micro_scheduler.py`): picks batch membership and sizes; repetition
  is the graph's job.
- Async worker (`.claude/skills/async-worker/`): speculates step N+1 while step N runs and pre-plans
  it on another thread; it can only speculate what the graph expresses.
- Accelerator graphs (`mstar/engine/accelerator_graph_config.py`, captured by
  `accelerator_graph_runner.py`): batched, packed and piecewise configs the model declares.

**Placement and lifecycle**
- `node_groups` in `configs/*.yaml`: GPUs or ranks per node, TP/SP widths.
- Conductor (`mstar/conductor/`): lifecycle, walk transitions, worker selection.
- Teardown: completed, failed, aborted and disconnected requests share the removal paths;
  handle-keyed worker state is purged there (`mstar/worker/rid_table.py`, invariant 14).

## Worked examples

- **Autoregressive decode.** The LLM runs once per output token, an unknown number of times; each
  run consumes the previous token; every token's keys and values are remembered; EOS or a cap ends
  it; tokens stream out. Shape: a `Loop` over the LLM node with a loop-back edge, a KV resource,
  `check_stop` for EOS, `EMIT_TO_CLIENT` per iteration (`mstar/model/qwen3_5/`). Every §6 row free.
- **A talker that speaks while the thinker writes** (Qwen3-Omni, `nemotron_duplex`): two async
  partitions and a streaming edge with a chunk policy. No coordinator object.
- **A flow or diffusion denoiser:** a fixed step count known at ingestion, so a `Loop` with the
  overshoot step exited early in `prepare_inputs`.
- **Chunked prefill (#365).** A node re-queued for the same request, a cursor in `ChunkProgress`,
  outputs held until the last chunk, plus flags to stop speculation and a special row type to patch
  what the runtime couldn't see: a graph loop built outside the graph runtime, first reviewed as a
  cleanup of scattered state.
- **Multi-LoRA.** First instinct: hook methods in every model. The timeline says every linear layer
  adds a per-request low-rank delta and the adapters are per-request state. Shape: the engine wraps
  linear layers by name, the adapter pool is a resource, a model declares its target-module names.
- **Context parallelism.** First instinct: one scheme (split stored KV across ranks), with SP as a
  separate feature. The family (DCP, PCP variants, Ulysses, recurrent-state schemes) has two axes:
  whether a step's tokens are split or copied (per step, decided by a declared resource) and how
  stored state is laid out (by head or position, fixed at load, owned by the pool). Each layer kind
  declares how it responds; the engine resolves the combination; per model, a declaration and a
  scatter/gather around its layers. Its seams (#1125) also missed: TP and SP placement behind one
  `over_sp` flag with mode-only methods (two types); a kernel contract only in a docstring (a
  protocol); queries and keys flattened into one tuple and re-split by count (pass two tuples); a
  backend's pieces supplied as separate base-class attributes and a closure (one object).
- **Preemption at one resource's altitude.** Request class, victim choice, the drop/rebuild
  lifecycle and its TP replication all lived inside one resource and the worker's calls into it.
  Each concept was about the request or every resource with per-request state, and each later PR
  extended the one-resource version. The altitude check would have named a request class, a
  residency lifecycle on `Resource` and a generic admission journal.
- **A streaming encoder's attention cache.** First instinct: the model keeps a tensor of the last W
  frames' keys and values. "Remembered across runs, bounded window" is a declared KV resource with a
  window, which gets capture, eviction and batching like any other KV.
- **A second copy of a family (#389).** A new linear-attention manager compared only with its nearest
  sibling, missing that another member of the same package repeated the same plan, pre-plan,
  capture-buffer and conv scaffolding: variants of one role belong in one resource, with the
  difference as config and the kernels as a backend.
