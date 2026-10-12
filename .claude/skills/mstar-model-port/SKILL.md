---
name: mstar-model-port
description: "Design guide for fitting a model into mstar's machinery (walks, nodes, resources vs submodules, modalities, CUDA graph buckets and gates, request lifecycle, batching, verification). Consumes the user's answered design worksheet from /model-recon and produces a port plan. Use when: the user says /mstar-model-port, has finished reconnaissance on a model, or asks where a model's pieces belong in mstar."
disable-model-invocation: true
---

# mstar-model-port

Input: the user's answered design worksheet and the state/shape ledger from `/model-recon`. If
either is missing, ask for it or run `/model-recon` first; do not substitute an intake form.
Output: a port plan (template at the end). The user's worksheet answers drive every mapping; where
an answer is missing, the plan carries an `open:` line, not a default.

Distilled from the Waypoint port (PR #251). Each item is a decision, its usual resolution, and the
failure it prevents. Exit gate for the resulting PR is `/mstar-pr-review`.

## 1. Map the model onto the machinery

1. Decompose by when things run. Once-per-request work is one walk; repeated work is another. Nodes
   fall out of what each walk emits to the client.
2. Resource vs submodule: a resource is state the engine must admit, plan, commit, and clean up per
   request and that must survive graph capture (persistent cross-step state, and anything planned over
   it). Everything else is a submodule.
3. Modality: a finished artifact and an ordered stream of partial results are different modalities
   even when the bytes look alike. Add one only when the lifecycle differs.
4. Config: split dataclasses when fields do not apply to the model's storage policy. Explicit
   sub-configs that fail on misuse pass review; silently ignored fields do not.
5. Every request-time operation in the recon trace (resize, RNG, cast, sync) is either kept as-is on
   device, moved to a one-time step, or justified in the plan.

## 2. Fit the engine's contracts

1. Grep callers before overriding any base-model or submodule hook; the hook surface contains
   refactored-out methods that are never called.
2. Per-request facts the engine needs for graph bucketing travel via step metadata into the key-info
   hook. Model-side branching on request shape does not reach the graph selector.
3. Declare varying dimensions (batch, sequence) in the graph config. Engine code that infers them
   from tensor shapes is model-specific and gets rejected.
4. Behaviour changes on shared classes (client, base config, engine paths) are opt-in for the new
   model. Existing models keep their defaults.
5. Validation lives in the base-model hook and works from key lists, not per-model dict loops.

## 3. CUDA graphs and static shapes

1. Every varying dim in the ledger gets exactly one treatment: its own bucket, normalized before the
   graph, or eager behind an explicit gate.
2. The static-buffer copy fails as a server error, not a fallback. An end-to-end off-shape request is
   part of the plan's test list.
3. Normalizing inputs before a graph changes numerics; a parity test against the reference precedes
   shipping it.
4. Each bucket costs startup through a Dynamo re-trace. The plan states the bucket list and the
   measured (or to-be-measured) per-bucket cost; batch buckets are geometric.
5. Once-per-request paths that are launch-bound (hundreds of small kernels) are capture candidates.

## 4. Request lifecycle and multi-request execution

1. Per-request state is one struct with one cleanup, called on completed, failed, and aborted paths.
2. For interactive models abort is the normal end. The plan names the test: cancel mid-stream, zero
   output after cancel, slot freed for the next request.
3. A failing request still records its sequence in the reorder buffer.
4. Cross-request batching needs a batched forward, per-request info threading, a row-independence
   test, and a dedicated padding slot; the knee is found by a sweep, not assumed.
5. Speculative execution: loop-external inputs need their own readiness bucket; fan-out after a node
   is unsupported, so fuse or restructure and say so in the plan.

## 5. Verification and what ships

1. Record an oracle from the reference once; keep a parity suite; rerun it after every perf commit and
   add one parity test per perf commit.
2. Millisecond effects are measured server-side (NVTX + nsys). Startup is never compared across
   days or nodes. No expected speedups in the plan without a measurement line next to them.
3. Realtime is a per-stream verdict (TTFF, gap p50/p95, on-time fraction, stalls); budget-edge rows
   are labelled borderline.
4. One config per variant, capacity fields at the measured recommendation, paths resolved from the
   Hub cache.
5. Two benchmark entry points: server-launching and requests-only, both able to save output.

## Port plan template

```
# <model> port plan
Inputs: recon.md @ <path>, worksheet answered <date>
1. Walks and nodes            (walk -> nodes -> what each emits)
2. Resources vs submodules    (item -> class -> why)
3. Modality and client path
4. Config layout
5. Shape ledger -> treatment  (dim -> bucket | normalize+parity | eager gate)
6. Bucket list and startup budget
7. Lifecycle: cleanup, abort, failure ordering
8. Batching / speculation notes
9. Tests to write             (parity, off-shape e2e, cancel, row independence)
10. Benchmarks and configs to ship
open: <every question the worksheet left unanswered>
```
