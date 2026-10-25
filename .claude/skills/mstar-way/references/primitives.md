# mstar primitives

The building blocks that §2 of the skill maps a feature onto. Line numbers are from `main` as of
2026-10-24; grep the name if they've moved. Read the code for any primitive you rely on.

## Graph structure (what runs, how often, in what order)

- **`GraphNode`** (`mstar/graph/base.py:230`): one computation, implemented by a node submodule.
  It has named `input_names` and output `GraphEdge`s (`:64`). It's ready when its inputs have
  arrived; streaming inputs relax that (below). `enable_async_scheduling` (`:240`, default on)
  lets the worker speculate the node's next step while the current one runs.
- **`Sequential` / `Parallel`** (`:385`, `:439`): composition.
- **`Loop`** (`:480`): runs a section repeatedly for one request.
  - `max_iters` is a safety bound. The real stop is a finish signal: a submodule's `check_stop`
    returns loop names (`mstar/model/submodule_base.py:942`), and the worker calls
    `register_loop_finish_signal` (`mstar/graph/graph_io.py:175`). Stops apply one step late, so
    the async worker's speculated step is the standard one wasted step.
  - External inputs are re-injected each iteration. A streamed input isn't: each iteration takes
    the next chunk (`Loop.ingest_external_input`).
  - Loop-back edges (a node's output edge to itself) carry per-iteration values.
  - `outputs` are emitted when the loop ends. `accumulated_outputs` are appended across
    iterations and emitted all at once at the end.
  - The graph runtime knows the loop. Speculation, readiness and the Python and Rust runtimes
    all handle it without special cases.
- **Graph walks** (`Model.get_graph_walk_graphs`, `mstar/model/base.py:817`): named graphs (for
  example `prefill_text`, `prefill_vision`, `decode`). The conductor moves a request from walk to
  walk. `get_combined_graph_walks` (`:747`) declares walks whose steps may share one batch on a
  node. `get_loop_caps` (`:970`) sets per-request loop caps.
- **`EMIT_TO_CLIENT`** (`mstar/graph/special_destinations.py`): an edge destination that streams
  to the client. Put on a loop's node, it emits every iteration.

## Data arriving over time

- **Async partitions** (`Model.get_partitions` → `PartitionDefinition`,
  `mstar/conductor/request_info.py:89`): sets of walks with their own state transitions,
  advancing independently. `producer_partitions` names who feeds whom.
- **Streaming edges** (`Model.get_partition_topology` → `PartitionTopology` / `Connection`,
  `mstar/streaming/topology.py:24`): a named edge from one partition to another, buffered on the
  consumer in a `StreamBuffer` (`mstar/streaming/stream_buffer.py:47`) and handed over according
  to a `ChunkPolicy` (`mstar/streaming/chunk_policy.py`: fixed, ramp, sliding window, left
  context, scheduled). `continue_after_done` says whether the consumer outlives the producer.
- **Streaming readiness** (`ReadySignals.is_ready_for_streaming`, `mstar/graph/base.py:146`): a
  node whose only missing inputs are streaming ones is ready when a chunk arrives. Until one
  arrives, the node just isn't ready. That's how "waiting for more input" is already expressed.
- **Initial inputs**: `process_prompt` (`mstar/model/base.py:832`) and
  `get_initial_forward_pass_args` (`:821`) turn the API request into the first edges, once,
  through the API server's preprocess worker.
- Examples: Qwen3-Omni (thinker → talker), `nemotron_duplex` (Encoder → LLM → Talker → Codec,
  `mstar/model/nemotron_duplex/nemotron_duplex_model.py`, `get_partition_topology`), Orpheus,
  fish_s2_pro.

## State (anything remembered across steps)

- **Resources** (`Resource`, `mstar/engine/resources/base.py:103`): engine-owned state that a
  node declares in `Model.get_node_resources` (`NodeResourceSpec`,
  `mstar/engine/resources/spec.py:41`). Each step's effect on them is declared in
  `NodeSubmodule.declare_step` (`mstar/model/submodule_base.py:640`). The engine runs admit →
  plan → forward → commit (`mstar/engine/resources/step.py`), with admission failure as
  backpressure the scheduler resolves by eviction or offload.
  - Kinds on `main`: paged KV (`kv/`), with prefix caching (`kv/prefix_index.py`, keyed walks),
    CPU offload (`kv/cpu_page_pool.py`), bounded and ring KV (`kv/bounded`, `kv/ring`);
    attention managers (`attn/`: FlashInfer, MLA, ragged, cross, flex, XPU); linear-attention
    state (`linear_attn/`: GDN, Mamba-2); recurrent state (`recurrent/pool.py`); positions;
    samplers; RMS-norm workspaces.
  - Resources pre-plan step N+1 during step N (`supports_preplan`), and a resource that copies
    host to device into fixed buffers sets `force_double_buffer`.
  - Per-deployment sizing goes in the config's `resources:` overrides.
  - The repo's skills: `.claude/skills/engine-resources/SKILL.md`, and
    `.claude/skills/prefix-caching/SKILL.md`.

## Per-step execution (inside one node step)

- **Node submodule hooks** (`mstar/model/submodule_base.py:495`): `prepare_inputs` (sets
  `input_seq_len`, per-request bookkeeping, early exit on a known-count overshoot), `preprocess`
  (stacking a batch), `forward` / `forward_batched` (pure tensor → tensor, capturable),
  `can_batch`, `postprocess`, `check_stop` / `check_stop_batched` (the only place a host read of
  outputs is free).
- **Micro-scheduler** (`mstar/worker/micro_scheduler.py`): picks which ready node steps form a
  batch, and how big each one is. It decides sizes and membership. Repetition is the graph's job.
- **Async worker** (`.claude/skills/async-worker/SKILL.md`): speculates step N+1 through the
  graph while step N runs, and pre-plans it on another thread. It can only speculate what the
  graph expresses.
- **Accelerator graphs** (`mstar/engine/accelerator_graph_config.py`: batched, packed,
  piecewise): the model declares a config, and the engine captures and replays it.

## Placement and lifecycle

- **`node_groups`** in `configs/*.yaml`: which GPUs or ranks run each node, and TP/SP widths. The
  same model code runs on one GPU or many.
- **Conductor** (`mstar/conductor/`): request lifecycle, walk transitions, worker selection.
- **Teardown**: completed, failed, aborted and disconnected requests all go through the existing
  removal paths. Handle-keyed worker state must be purged on removal
  (`mstar/worker/rid_table.py`).
