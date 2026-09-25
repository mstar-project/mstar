# Multi-stage models, streaming, and serving

Read this for async partitions, multi-stage streaming, live serving, or a
request that hangs. Confirm every interface against the checkout first.

The closest live patterns are `mstar/model/qwen3_omni/` and
`mstar/model/orpheus/`. Read their Walks, partition definitions, partition
topology, resource declarations, node step declarations, and conductor state
machines before implementing a similar topology.

## How a node runs

The conductor owns request scheduling and the Walk state machine. Workers own
the node submodules and the resources built for those nodes. One node step runs:

```text
prepare_inputs -> declare_step -> admit -> plan -> preprocess -> forward -> commit
```

The resource runner owns admission, dependency-ordered planning, commit, and
request removal. A declaring submodule must not repeat that lifecycle privately.

`graph_walk` is the model's branch key; the engine does not infer prefill or
decode behavior from the name. In a forward, retrieve built resources from
`engine_inputs.resources[resource_key]`. Layers may receive the same resources
through `bind_resources()` during node binding.

For sampling, declare a `SamplerSpec`, return its `SamplingReqConfig` from
`get_request_resource_configs()`, include `SamplerStep` in the node's
`SubmoduleStep`, and call the built sampler resource. Separate token channels
use separate, statically declared sampler resource keys.

Engine-provided per-request state is suitable for request-local metadata and
tensors without their own admission or capacity contract. Any pool, allocator,
stable shared buffer, resident limit, dependency ordering, or cleanup policy is
an engine-owned resource instead.

Start with `can_batch()` false unless the node already has a verified batching
implementation. Resident resource capacity is independent of the number of
requests processed in one step.

## Async partitions

Use partitions when a downstream stage must consume upstream output before the
producer finishes:

- `get_partitions()` returns a `PartitionDefinition` for each independently
  scheduled stage, including its Walks, initial Walk, and producers.
- `StreamingGraphEdge` identifies the destination node, tensor name, and target
  partition.
- `get_partition_topology()` returns the matching `Connection` objects and
  chunk-policy factories.
- `get_initial_forward_pass_args()` seeds non-streamed inputs for each
  partition.
- `get_partition_forward_pass_args()` advances that partition after a Walk and
  marks it done when appropriate.

The streamed tensor arrives from the stream buffer, not from the
forward-pass-argument input list. Every node input name must still be satisfied.
Seed self-fed or loop-back inputs for the first consumer iteration, or that node
never becomes ready.

Use `FixedChunkPolicy` for one item per consumer step and
`LeftContextChunkPolicy` when a causal decoder needs overlap. Copy a matching
policy from a checked-out reference instead of inventing new scheduling rules.

Loop termination remains worker-side: `Loop.max_iters` is the bound and the
node's `check_stop()` returns named loops to stop. A consumer partition normally
sets `request_done=True` after its loop returns control and the upstream stream
has completed according to the reference state machine.

For multiple output modalities, tag each client edge with its
`output_modality`, and branch on that value in `Model.postprocess()`.

## CPU structural validation

Before weights or a GPU:

1. Construct every Walk, partition definition, topology connection, resource
   spec, request config, and node step declaration.
2. Call `get_worker_graphs(config_path)` for every Walk and confirm every graph
   node is mapped by the YAML.
3. Confirm every resource key is unique, each dependency exists, and each node
   step uses only resources owned by that node.
4. Build a dummy-mode `Conductor` to resolve worker and partition routing.
5. Add focused modular tests for first-iteration seeds, exact edge names,
   completion transitions, capacity refusal, and request cleanup.

## Live serving

Launch from the checkout so the new registry entry wins over an installed copy:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
  mstar-serve --config configs/<name>.yaml --host 127.0.0.1 --port 8000 \
  --tensor-comm-protocol SHM --socket-path-prefix /tmp/mstar_<name>/ \
  --upload-dir /tmp/mstar_up_<name>/
```

Use a deterministic, realistic request already verified against the standalone
eager oracle. The first run must confirm exact output routing, termination,
resource release, and a second request reusing released capacity.

Common startup faults:

- `Unknown model`: `PYTHONPATH` points at another checkout or the registry is
  incomplete.
- Transport initialization fails without InfiniBand: use SHM for a single-node
  run.
- Media never reaches the first node: map the data worker's `image_inputs`,
  `audio_inputs`, or `video_inputs` key to the exact graph input in
  `process_prompt()`.
- Media decoding fails before scheduling: use a model-appropriate loader with
  declared runtime dependencies.
- A node is absent from a worker: align YAML `node_names` with graph node names.

## Diagnose a hung request

A hang means a node never became ready or a loop never terminated. Add temporary
logging at each node's `prepare_inputs()` and `forward()`, plus both
forward-pass-argument state-machine hooks. Send one short request and identify
the first missing event.

| last observed event | likely cause | fix |
|---|---|---|
| No first-node preparation | Initial input name or partition seed is missing. | Compare `process_prompt()` output with the first graph node's inputs. |
| Producer runs; consumer never prepares | Stream edge, connection, chunk policy, or first loop-back seed is missing. | Match the producer output key, stream connection, and all consumer inputs. |
| Consumer runs repeatedly | Stop signal or partition completion transition is wrong. | Test `check_stop()` and the next forward-pass arguments directly. |
| Capacity fails after completed requests | A resource claim is not released. | Test ingest, admit, failure, cancellation, and remove lifecycle paths. |

Remove temporary trace logging after the structural test captures the failure.
