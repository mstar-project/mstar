---
name: add-mstar-model
description: >-
  Port a reference or Hugging Face model into MStar as native Walk Graph nodes
  with engine-owned resources and eager-first correctness evidence. Use when
  adding a model package, registry entry, serving config, or model-specific API
  adapter. Do not use this workflow for benchmark-driven optimization of an
  existing port.
---

# Add an MStar model

Map the reference implementation onto the checked-out branch's graph, model,
and resource contracts. Correct eager execution is the MVP. Record benchmark
results after correctness, but do not optimize against them.

## Verify the local contract first

Before choosing a reference model or writing code:

1. Read `docs/adding_models.rst`.
2. Inspect `Model` in `mstar/model/base.py` and list its current abstract hooks.
3. Inspect `NodeSubmodule` and `ARNodeSubmodule` in
   `mstar/model/submodule_base.py`.
4. Inspect the declarations and implementations under
   `mstar/engine/resources/`, including their focused tests.

Treat the checkout as authoritative. If a requested direction needs an API that
is absent, record the missing contract and evidence; do not revive an older API.
Use [the contract lookup](references/model-contract.md) only after this check.

## Write the design before implementation

Create or update `progress-artifacts/evidence/model-port-report.md`. Add one row
per node before editing the implementation:

```text
node | walks | inputs/outputs | persistent state | resource keys | dependencies | resident capacity | eager validation
```

Maintain `progress-artifacts/evidence/model-port-decisions.jsonl` as an
append-only decision log. Before implementation, record whether the target is
`matched`, `composed`, or `unmatched-derived`, then record reference assessments
and persistent-state mappings as those choices are made. Read
[decision logging](references/decision-log.md) for the event schema and developer
feedback signals. Record concise conclusions and evidence, not private
chain-of-thought or command-by-command activity.

After deriving the target design, read
[the model shape map](references/model-shape-map.md) for examples with matching
graph topology and state behavior. The map is non-exhaustive. If no example
fits, implement directly against the checked-out graph and resource contracts;
do not relax engine ownership to make the model resemble a reference.

For every persistent value, decide whether it is:

- a graph-routed tensor;
- engine-provided per-request state without its own allocation lifecycle; or
- an engine-owned resource.

Read [resource pools](references/resource-pools.md) whenever state is pooled,
allocated, capacity-limited, held at a stable address, dependency-ordered, or
cleaned up. Those properties require an engine-owned resource. Do not implement
a parallel lifecycle in a model or submodule.

Read [engine and serving](references/engine-and-serving.md) for async partitions,
multi-stage streaming, live serving, or request hangs.

## Implementation boundary

Reuse KV, attention, cross-attention, ragged-attention, position, and sampler
resources where their contracts fit. A genuinely new resource may add a package
under `mstar/engine/resources/<kind>/`, declaration types, a manager, exports,
and focused tests using the existing spec, request-config, step, and `Resource`
interfaces.

Do not modify `mstar/engine/engine.py`, `mstar/engine/resources/base.py`,
`mstar/engine/resources/runner.py`, `mstar/engine/resources/step.py`, or engine
lifecycle and scheduling semantics. Fit required state into an existing
resource or add a resource kind through the extension interfaces above. Never
fall back to model-owned allocation, pooling, capacity, stable storage,
dependency ordering, or cleanup. Mark a direction `evidence-blocked` only after
documenting the attempted engine mapping and the exact missing extension
interface. A missing reference model, implementation effort, or loss of a
preferred optimization is not a blocker.

Keep resident capacity separate from one-step batching. A pool may retain many
sessions while `max_batch_size` remains one.

Do not wrap the upstream pipeline as the served implementation. Use it only as
an independent oracle. Port the compute, load weights through MStar's loader,
declare Walks and resources, register the model, add its deployment YAML and pip
extra, and wire only the API adapters its client contract needs.

## Eager-first validation

Validate in this order:

1. Compare component forwards and weight mapping with the reference.
2. Build a standalone eager pipeline as the end-to-end oracle.
3. Compare direct node forwards with that oracle.
4. Validate Walks, resource declarations, request configs, and step declarations
   on CPU before loading weights.
5. Compare live serving with the oracle on deterministic, realistic input.

Do not add whole-forward or piecewise CUDA-graph configs during MVP porting.
Do not load [the CUDA graph reference](references/cuda-graphs.md) unless a
separate optimization task explicitly requests capture work.

After correctness, run the supplied benchmark and record latency plus resource
observations in the report. Preserve its `headline_metric`, `result_arg`, and
result JSON format. The measurements do not select or reject MVP implementation
choices.

## Completion report

Mark every requested direction as exactly one of:

- `implemented`, with code and validation evidence;
- `evidence-blocked`, with the missing contract or unavailable evidence; or
- `deferred`, with the separate future task named.

A complete MVP has native eager serving, no model-owned resource pool, no new
CUDA-graph configuration, no engine lifecycle change, compatible benchmark
artifacts, a current model port report, and a valid decision log.
