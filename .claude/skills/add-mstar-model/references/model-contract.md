# MStar model contract

Use this as a lookup only after reading `docs/adding_models.rst` and the
checked-out sources. The branch is authoritative.

## Model surface

Current required hooks in `mstar/model/base.py`:

| method | purpose |
|---|---|
| `get_node_resources()` | Declare `NodeResourceSpec` objects and the graph nodes sharing each resource. Return `[]` when no engine resource is needed. |
| `get_graph_walk_graphs()` | Build each named Walk from graph sections and exact tensor-edge names. |
| `process_prompt(...)` | Turn request text and loaded media into initial named tensors. |
| `get_initial_forward_pass_args(...)` | Seed each partition's first Walk and inputs. |
| `get_partition_forward_pass_args(...)` | Choose the next Walk, persisted inputs, and completion state. |
| `postprocess(...)` | Encode an emitted tensor as client bytes. |
| `get_submodule(...)` | Construct and load one node, or return `None` in dummy mode. |

`get_request_resource_configs()` optionally returns per-request
`ResourceReqConfig` values keyed by the same resource keys used in
`get_node_resources()`. This is where request sampling settings and other
resource-specific request parameters enter the engine.

Optional partition hooks, media loaders, output metadata, sharding, and token
limits are documented on `Model`. Implement only what the port requires.

## Resource declaration

Import declaration types from `mstar.engine.resources`. Existing kinds include:

| declaration | engine-owned behavior |
|---|---|
| `KVSpec(KVConfig(...))` | Paged persistent KV storage and request admission. |
| `AttentionSpec(AttentionConfig(...))` | Self-attention planned against a named KV resource. |
| `CrossAttentionSpec(CrossAttentionConfig(...))` | Read-once encoder context and decoder cross-attention. |
| `RaggedAttentionSpec(RaggedAttentionConfig(...))` | Cacheless variable-length attention for packed encoder inputs. |
| `PositionSpec(PositionConfig(...))` | Position counters and optional RoPE against a named KV resource. |
| `SamplerSpec(...)` | Sampling buffers and per-request sampling state. |

Each spec has a unique `resource_key`, a set of owning `nodes`, and optional
`depends_on()` keys. YAML deployment overrides live under
`resources.<resource_key>` and may tune only fields accepted by that spec.

## Graph surface

- `GraphNode(name, input_names, outputs)` defines one compute node. Every input
  name must arrive before the node becomes ready.
- `GraphEdge(next_node, name, persist=..., output_modality=...)` routes the
  tensor returned under exactly `name`.
- `Sequential` and `Parallel` compose graph sections.
- `Loop(name, section, max_iters, outputs)` repeats a section; `check_stop()`
  returns loop names that should stop.
- `StreamingGraphEdge` crosses async partitions. Use the partition topology and
  chunk-policy APIs already present on the branch.

Use `EMIT_TO_CLIENT` and `EMPTY_DESTINATION` from
`mstar.graph.special_destinations` for terminal edges.

## NodeSubmodule surface

`NodeSubmodule` and `ARNodeSubmodule` are driven by the engine:

```text
prepare_inputs -> declare_step -> admit -> plan -> preprocess -> forward -> commit
```

- `prepare_inputs(...) -> NodeInputs` performs cheap host-side shaping. Set
  `input_seq_len`; use `resource_step_info` for declaration-only metadata.
- `declare_step(...) -> SubmoduleStep | None` describes work for each resource
  key. A declaring submodule does not independently plan or commit the same
  state.
- `preprocess(...) -> dict` collates prepared inputs. The AR subclass requires
  an implementation.
- `forward(...) -> NameToTensorList` reads resources from
  `engine_inputs.resources[key]` and returns exact graph-edge keys.
- `check_stop(...)` may read tensor values off the GPU thread. `postprocess(...)`
  should normally remain metadata-only.

The engine binds each node's resources once with `bind_node_resources()`, which
also calls `bind_resources()` on child layers that expose it. Layer constructors
should retain resource keys, not build or own managers.

`engine_inputs.per_request_states` is valid for request-local tensors or
metadata whose storage does not need pooling, admission, capacity accounting,
stable shared addresses, dependency ordering, or an independent cleanup policy.
Use an engine resource when any of those properties appears.

## Step declarations

A `SubmoduleStep` maps resource keys to `ResourceStep` subclasses and may share
a default list of `Segment(request_id, label, span)` values. Existing step kinds
include `KVStep`, `AttentionStep`, `PositionStep`, and `SamplerStep`.

Unit-test declarations on CPU. Confirm that:

- every declared key belongs to that node;
- every dependency is declared and acyclic;
- request spans and labels match the forward layout;
- capacity failure is explicit and releases cleanly; and
- resident capacity is not confused with `max_batch_size`.

## Weight and deployment wiring

Build modules on `meta`, cast before `to_empty(device=...)`, then use
`mstar.model.loader.load_weights`. Component `load_weights()` implementations
should delegate to `load_hf_weights` with explicit stacked-parameter rules and
name remapping, and must reject incomplete matches.

Register the model in `mstar/model/registry.py`, add its `configs/<name>.yaml`,
declare only the required pip extra, and add an API adapter only when the target
endpoint needs one. Start with all nodes on one rank unless correctness requires
another placement.

Validate graph and resource structure without weights, then compare component,
node, and live outputs against the independent eager oracle.
