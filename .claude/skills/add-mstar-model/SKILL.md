---
name: add-mstar-model
description: End-to-end workflow for adding a model to M* — reconnaissance on the reference implementation, decomposing it into walks and nodes, deciding resources vs submodules, registration, eager parity against an oracle, batching and CUDA graphs, human verification of media output, competitor benchmarking and performance tuning. Use when porting a reference or Hugging Face model, or adding a model package, registry entry, serving config or model-specific API adapter.
---

# Adding a model to M*

A model is not done when it produces correct output once. It is done when it is decomposed sensibly, registered everywhere, batched, capturable, benchmarked against the competition, and its media output has been looked at by a human.

Supersedes the `add-mstar-model` skill drafted in PRs [#214](https://github.com/mstar-project/mstar/pull/214) and [#253](https://github.com/mstar-project/mstar/pull/253). Much of the design guidance below is distilled from the `mstar-model-port` and `model-recon` skills on [`garv/new-model-support-agent-2`](https://github.com/mstar-project/mstar/tree/garv/new-model-support-agent-2/.claude/skills), themselves distilled from the Waypoint port — those remain worth reading as standalone deep treatments of reconnaissance and port planning.

Read [AGENTS.md](../../../AGENTS.md) alongside this. Invariants 1, 2, 3, 4 and 8 are the ones model ports trip over, and the PR reviewer cites them by number.

## Two drift patterns seen in real agent-written ports

Check your own work against both before anything else, because both produce code that runs.

**Collapsing the graph onto one node.** The tell is a single node holding the LLM *and* the decoder (or the VAE, or the codec), often named after the model itself. It works, and it throws away disaggregation: nothing in that model can be placed on a separate GPU or scheduled independently again. Fusing stages is a legitimate optimization for capture size — qwen3-omni fuses talker and code predictor deliberately — but it is a choice made after the split, with a reason, not the starting point.

**Drifting to model-owned resources.** State ends up in the submodule because that is the path of least resistance. If you are writing allocation, capacity limits, cleanup or a stable buffer address under `mstar/model/`, that is invariant 1 and the answer is a `Resource`.

## Phase 0 — Verify the contract in the checkout, not in the docs

Read `Model` in `mstar/model/base.py` and list its current abstract hooks; read `NodeSubmodule` and `ARNodeSubmodule` in `mstar/model/submodule_base.py`; read the declarations and implementations under `mstar/engine/resources/` with their focused tests. `docs/adding_models.rst` is the narrative guide and worth reading, but the checkout is authoritative — the docs were wrong about the registry's own type signature for a long time. If a direction you need requires an API that does not exist, record that and stop rather than reviving an older one.

## Phase 1 — Reconnaissance on the reference

Before designing anything, understand what the reference actually does, and write it down with `path:line` citations rather than from memory. Mark anything you inferred as an inference and anything you could not determine as an explicit unknown with the file to look in — a guess recorded as a fact is worse than a gap.

What to extract:

- **File map**: model class, config class, weight loader and key mapping, preprocessing, inference entry point, streaming/output path, any existing vLLM or SGLang port.
- **Runtime trace** of one request through the entry point: for each operation, whether it runs once per request or per iteration; what is carried between iterations; how termination is decided; how output leaves. Flag every request-time operation touching shapes or the host — resizes, RNG draws, `.item()`/`.cpu()` syncs, dtype casts, padding.
- **State and shape ledger**: one row per persistent tensor or buffer — lifetime (global / per request / per iteration), shape, which dims vary at request time and over what range, who allocates, who frees. This table is what Phase 5 consumes; it is the single most useful artifact of this phase.

If a vLLM or SGLang port exists, record what it did — registration, cache config, attention backend, multimodal processor, batching hooks, special-cased code — and what it explicitly did not support.

## Phase 2 — Decompose into a graph

**Split by when things run.** Once-per-request work is one walk; repeated work is another; nodes fall out of what each walk emits to the client. Then:

- Anything you want independently schedulable or independently placeable is its own node — that is what buys disaggregation via `node_groups`.
- A finished artifact and an ordered stream of partial results are **different modalities** even when the bytes look alike. Add a modality only when the lifecycle differs.
- Streaming partitions are for parts of the graph that should run asynchronously, passing data as it is produced rather than at walk boundaries. Read invariant 12 first; the coordination and finish-condition hazards are real.

[model-shape-map.md](references/model-shape-map.md) catalogues how the existing models split — the fastest way to see a reasonable decomposition for a topology like yours. Derive your own first and use the map for precedent, not to pick the least-wrong bucket.

## Phase 3 — Resources vs submodules

A **resource** is state the engine must admit, plan, commit and clean up per request, and that must survive graph capture: persistent cross-step state, and anything planned over it. Everything else is a submodule.

Reuse KV, attention, cross-attention, ragged (cacheless) attention, position and sampler resources wherever their contracts fit. A genuinely new kind is a normal outcome and gets a package under `mstar/engine/resources/<kind>/` with declaration types, a manager, exports and focused tests, built on the existing spec, request-config, step and `Resource` interfaces. Several current model PRs add recurrent-state and linear-attention resources this way, modelled on the KV resource. Read the [engine-resources skill](../engine-resources/SKILL.md) before writing one.

Do not modify `engine.py`, `resources/base.py`, `resources/runner.py` or `resources/step.py` to make your model fit, and do not fall back to model-owned pooling, allocation, capacity, dependency scheduling or cleanup. If the engine genuinely cannot express something, document the missing extension interface — effort is not a blocker, and neither is losing a preferred optimization.

Keep resident capacity separate from one-step batching: a pool may retain many sessions while `max_batch_size` stays at one. Split config dataclasses when fields don't apply to the model's storage policy; explicit sub-configs that fail on misuse pass review, silently ignored fields don't.

## Phase 4 — Write it, register it, prove eager parity

`mstar/model/<name>/<name>_model.py` with the graph, resource and partition declarations, prompt/input processing and output processing. Per submodule: `prepare_inputs` (set `input_seq_len`), `preprocess`, `forward`, `postprocess`, `check_stop`, `declare_step`. Write forwards tensor-in/tensor-out from the start (invariant 3) — retrofitting that is far more work than doing it once.

**Register it: [registration.md](references/registration.md) is the checklist.** `test/modular/test_model_registration.py` enforces the registry, the CLI default config and the docs row, so CI catches those three. Forgetting the CLI entry means `mstar serve <model>` exits `unknown model` no matter how correct the code is.

Grep for callers before overriding any base-model or submodule hook — the hook surface contains refactored-out methods that nothing calls, and overriding one is a silent no-op.

Reuse from `mstar/model/components/`, and put anything reusable across models there rather than in your package. For MoE models, CUDA-graph compatibility comes from the vLLM kernels shipped in `mstar/model/components/moe.py`.

Validate in this order:

1. Component forwards and weight mapping against the reference.
2. A standalone eager pipeline from the reference as an independent oracle. Do not wrap the upstream pipeline as the served implementation — oracle only.
3. Your node forwards against that oracle.
4. Walks, resource declarations, request configs and step declarations on CPU, before weights (`pytest test/modular/`, dummy mode, `get_submodule` returns `None`).
5. Live serving against the oracle on deterministic input.

Record the oracle once and keep the parity suite; rerun it after every performance commit, and add a parity test per performance commit. Do not add CUDA-graph configs while chasing parity — capture makes every discrepancy harder to localize and the two kinds of debugging do not mix.

## Phase 5 — Batching and CUDA graphs

Add `can_batch` / `forward_batched` and the graph declarations. If only part of the forward is capturable, use piecewise graphs (`PiecewiseBatchedConfig` / `PiecewisePackedConfig`) rather than abandoning capture for the node. `unpack_packed_output` and `postprocess` are the hooks for work on the tail end of a graphed forward.

Take the shape ledger from Phase 1 and give **every varying dimension exactly one treatment**: its own bucket, normalized before the graph, or eager behind an explicit gate. Declare varying dims in the graph config — engine code that infers them from tensor shapes is model-specific and gets rejected. Normalizing an input before a graph changes numerics, so the parity test comes before shipping it. Buckets are not free: each costs startup through a Dynamo re-trace, and batch buckets are geometric. Per-request facts the engine needs for bucketing travel via step metadata into the key-info hook; model-side branching on request shape does not reach the graph selector.

Per-request state is one struct with one cleanup, called on the completed, failed **and** aborted paths. For interactive models abort is the normal ending, so test it: cancel mid-stream, zero output after cancel, slot freed for the next request. A failing request still records its sequence in the reorder buffer.

Cross-request batching needs a batched forward, per-request info threading, a row-independence test and a dedicated padding slot; find the knee with a sweep rather than assuming it.

Consider making capture a separate PR stacked on the eager MVP, and any **system** change your model needed a separate PR underneath — see "Splitting work across PRs" in AGENTS.md. Recommended, not required; large models are painfully slow without capture, so an eager-only merge is not always useful alone.

## Phase 6 — Look at the output yourself

**Media output must be verified by a human, here and again before the PR goes up.** An agent is fine for sanity checks and for reading text, but audio, images and video need a person to listen to or look at them. Numerical parity passing is not evidence that generated audio sounds right.

## Phase 7 — Benchmark against the competition

Against vLLM, vllm-omni, TensorRT-LLM, SGLang, sglang-omni — whichever serve this model. The goal is genuinely to be better than or on par with them; this is a target, not an observation.

Across a **comprehensive** set, though, not one configuration. A single winning benchmark usually means it got tuned for, and the workloads you didn't measure are where the regression hides: don't buy `image_to_text` throughput with `text_to_text` or mixed-workload throughput. Report which workloads you checked, including the ones that didn't improve.

Use the [benchmarking skill](../benchmarking/SKILL.md) for methodology — warmup, concurrency sweeps, and the traps that make a cross-process A/B move on their own. Millisecond-scale effects are measured server-side (NVTX, nsys), and startup is never compared across days or nodes. For streaming models, realtime is a per-stream verdict (TTFF, gap p50/p95, on-time fraction, stalls), with budget-edge rows labelled borderline. Ship one config per variant with capacity fields at the measured recommendation.

## Phase 8 — Performance tuning

Establish CPU-bound or GPU-bound first; the benchmarking skill covers how (`event_sync` near zero means CPU-bound). That much is settled. Beyond it **there is no established best practice in this repo yet** — current tuning work is iterative and exploratory, so treat any confident-sounding recipe with suspicion, including one you generate yourself. What holds regardless: same harness, same box, two sides adjacent in time, and re-check the workloads you are not targeting before calling a change a win.

Two things that are settled: a host sync in your `prepare_inputs` hides inside that phase rather than in `event_sync`, and a host-bound step gets faster by deleting work, not by moving it to another thread. Values already known on the host travel in `step_metadata`.

If tuning pushes you into the worker or engine rather than your model, read the [async-worker skill](../async-worker/SKILL.md) first.

## Done means

Decomposed by when things run, not collapsed onto one node; registered everywhere [registration.md](references/registration.md) lists; eager parity against an oracle, with the suite kept; every varying dim with exactly one treatment; one cleanup covering completed, failed and aborted; batching and capture declared or a comment saying why a node can't have them; modular tests passing; a human has checked the media; a benchmark against at least one competitor across more than one workload; and no model-owned resource lifecycle anywhere.
