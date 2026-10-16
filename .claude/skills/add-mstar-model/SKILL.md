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

Then check you can actually run the reference, before any design work:

- **Access.** Gated repos need a token and an accepted licence; ask for it now rather than mid-recon.
- **Every format the weights ship in.** A model card may point at a single-file (e.g. ComfyUI) checkpoint while the diffusers- or transformers-format weights the oracle loads live in a separate repo under the same author.
- **The reference's generation defaults.** Pin the oracle to greedy single-beam; a `chat()` helper may default to beam search even with sampling off.
- **The oracle's library versions against the venv.** A newer model often needs a newer `transformers` or `diffusers` than the serving venv pins. Every other model imports those too, so a bump is an invariant-4 change and its own PR; for the oracle, use a separate venv with the same torch build.

## Phase 1 — Reconnaissance on the reference

Before designing anything, understand what the reference actually does, and write it down with `path:line` citations rather than from memory. Mark anything you inferred as an inference and anything you could not determine as an explicit unknown with the file to look in — a guess recorded as a fact is worse than a gap.

What to extract:

- **File map**: model class, config class, weight loader and key mapping, preprocessing, inference entry point, streaming/output path, any existing vLLM or SGLang port.
- **Runtime trace** of one request through the entry point: for each operation, whether it runs once per request or per iteration; what is carried between iterations; how termination is decided; how output leaves. Flag every request-time operation touching shapes or the host — resizes, RNG draws, `.item()`/`.cpu()` syncs, dtype casts, padding.
- **State and shape ledger**: one row per persistent tensor or buffer — lifetime (global / per request / per iteration), shape, which dims vary at request time and over what range, who allocates, who frees. This table is what Phase 5 consumes; it is the single most useful artifact of this phase.
- **Size per component**: weight bytes per component (from the safetensors index) and a rough peak-activation estimate. It answers "is this hardware enough?" and is the starting point for placement in Phase 2.

If a vLLM or SGLang port exists (check local checkouts and installed packages before upstream), record what it did — registration, cache config, attention backend, multimodal processor, batching hooks, special-cased code — and what it explicitly did not support.

Diff your `state_dict()` against the checkpoint's safetensors index by **name and shape** before running anything — on `meta` device, so it costs nothing. A full match proves loading needs no reshaping or splitting, and it catches a misread projection layout before it becomes a parity hunt.

## Phase 2 — Decompose into a graph

**Split by when things run.** Once-per-request work is one walk; repeated work is another; nodes fall out of what each walk emits to the client. Then:

- Anything you want independently schedulable or independently placeable is its own node — that is what buys disaggregation via `node_groups`.
- Where to draw the line between two adjacent stages is a judgement call. Start from precedent in the repo, then reason about performance: what crosses the edge and how large it is, whether either side is worth placing, scaling, or scheduling independently, and how it affects capture. LTX-2.5 keeps its text encoder and text connectors in one node because Gemma's stacked hidden states are ~30× the connectors' output; splitting there would ship the large tensor between nodes to gain nothing.
- Decide now which nodes need tensor or sequence parallelism. TP/SP has to be designed into the attention and linear layers from the start (`tp_enabled_nodes` / `sp_enabled_nodes` in the sharding config; cosmos3 and its `*_tp2_sp2.yaml` are the precedent), and retrofitting it is a rewrite of the layers.
- A finished artifact and an ordered stream of partial results are **different modalities** even when the bytes look alike. Add a modality only when the lifecycle differs.
- Streaming partitions are for parts of the graph that should run asynchronously, passing data as it is produced rather than at walk boundaries. Read invariant 12 first; the coordination and finish-condition hazards are real.

[model-shape-map.md](references/model-shape-map.md) catalogues how the existing models split — the fastest way to see a reasonable decomposition for a topology like yours. Derive your own first and use the map for precedent, not to pick the least-wrong bucket.

## Phase 3 — Resources vs submodules

A **resource** is state the engine must admit, plan, commit and clean up per request, and that must survive graph capture: persistent cross-step state, and anything planned over it. Everything else is a submodule.

**Do an explicit resource pass before writing submodules**, one row per piece of per-request or per-step state and per attention, sampling or position operation:

| state or op | lifetime and size | existing resource, as is | existing resource, changed (how; own PR?) | new resource (why nothing fits) | allocated in which walk; what happens when full | how the model reaches it |
|---|---|---|---|---|---|---|

Each row must hold up: the resource's layout expresses the mask or retention, checked against the reference (read the planner, not the kind's name; when retention isn't obvious, simulate the reference's cache on token ids); capacity becomes engine backpressure, and the request that waits is the one that needs it; eager and captured paths address it the same way; and the model reaches it only through the resource's API (invariant 1). The [engine-resources skill](../engine-resources/SKILL.md) lists what each existing resource expresses. MiniCPM-o (#381-#383) is a worked example: paged KV and ragged attention as is, a new block-causal ragged kind, an extended sampler, `RecurrentStatePool` for the vocoder's carry state, and a new bounded KV once paged KV with retention and `RingKVManager` were ruled out.

Reuse KV, attention, cross-attention, ragged (cacheless) attention, position and sampler resources wherever their contracts fit. A genuinely new kind is a normal outcome and gets a package under `mstar/engine/resources/<kind>/` with declaration types, a manager, exports and focused tests, built on the existing spec, request-config, step and `Resource` interfaces. Several current model PRs add recurrent-state and linear-attention resources this way, modelled on the KV resource. Read the [engine-resources skill](../engine-resources/SKILL.md) before writing one.

Don't drop to calling a kernel from the model because no resource quite fits. When the closest resource almost expresses what you need, extend it (in its own package, alongside the existing kind); when nothing is close but a new resource is a reasonable amount of work, build one. Qwen3.5's recurrent state and GDN resources resembled nothing that existed and were still the right call. Attention in particular is typically (though not always) resource-driven, cross-attention and attention with fixed shapes inside a capture bucket included: "capture still works with SDPA" is true and is not by itself a reason, since the model then misses the engine's kernels and a backend change becomes a sweep through model code. Exceptions need a stated reason (performance, or implementation complexity out of proportion to the gain), e.g. a text encoder whose head dim no FlashInfer kernel supports and whose attention is a negligible share of its node's time.

Do not modify `engine.py`, `resources/base.py`, `resources/runner.py` or `resources/step.py` to make your model fit, and do not fall back to model-owned pooling, allocation, capacity, dependency scheduling or cleanup. If the engine genuinely cannot express something, document the missing extension interface — effort is not a blocker, and neither is losing a preferred optimization.

Keep resident capacity separate from one-step batching: a pool may retain many sessions while `max_batch_size` stays at one. Split config dataclasses when fields don't apply to the model's storage policy; explicit sub-configs that fail on misuse pass review, silently ignored fields don't.

## Phase 4 — Write it, register it, prove eager parity

`mstar/model/<name>/<name>_model.py` with the graph, resource and partition declarations, prompt/input processing and output processing. Per submodule: `prepare_inputs` (set `input_seq_len`), `preprocess`, `forward`, `postprocess`, `check_stop`, `declare_step`. Write forwards tensor-in/tensor-out from the start (invariant 3) — retrofitting that is far more work than doing it once.

**Register it: [registration.md](references/registration.md) is the checklist.** `test/modular/test_model_registration.py` enforces the registry, the CLI default config and the docs row, so CI catches those three. Forgetting the CLI entry means `mstar serve <model>` exits `unknown model` no matter how correct the code is.

Grep for callers before overriding any base-model or submodule hook — the hook surface contains refactored-out methods that nothing calls, and overriding one is a silent no-op.

The engine calls `submodule.to(device, dtype)` *after* `get_submodule` (`mstar/worker/engine_manager.py:82`), so per-parameter dtype fixes applied at construction do not survive. If a module has dtype-sensitive parameters — FlashInfer's GDN kernels assert fp32 on the decay terms — enforce it in the module's own `_apply`, so it holds against any caller rather than one loader.

Reuse from `mstar/model/components/`, and put anything reusable across models there rather than in your package. For MoE models, CUDA-graph compatibility comes from the vLLM kernels shipped in `mstar/model/components/moe.py`. For image and video flow models, build the `dit` node on the DiT scaffold in `mstar/model/components/diffusion/`; read the [diffusion-scaffold skill](../diffusion-scaffold/SKILL.md) first.

**Port natively by default; reusing an upstream leaf module is accepted but second best.** A native port can be optimized (TP, capture, fused kernels) and doesn't break when `transformers` or `diffusers` moves. The hot path — the backbone or DiT — should always be native. Importing a cold leaf module such as a VAE or vocoder from the upstream library, lazily as wan22 and cosmos3 do, is accepted precedent when porting it would be high effort for little gain; say which parts are reused and why in the PR.

A self-contained native port delegates well: record the oracle tensors first, make a parity harness the deliverable, and state which files the agent owns.

Validate in this order:

1. Component forwards and weight mapping against the reference.
2. A standalone eager pipeline from the reference as an independent oracle. Do not wrap the upstream pipeline as the served implementation — oracle only.
3. Your node forwards against that oracle.
4. Walks, resource declarations, request configs and step declarations on CPU, before weights (`pytest test/modular/`, dummy mode, `get_submodule` returns `None`).
5. Live serving against the oracle on deterministic input.

When a parity check fails, diff per-layer hidden states against the oracle, then split the first bad layer into its parts — each norm, the mixer, the MLP. A mixer compared in isolation can pass while its enclosing layer is wrong, because the isolated comparison feeds both sides the same input; the norms and the residual are where convention mismatches live.

**When parity is close but not exact, find the configuration where it is exact before accepting a tolerance.** Served paths legitimately differ from the reference in shapes and kernel routes (running only the real prompt tokens instead of a padded batch, a different SDPA backend), and those differences alone produce ~0.3–1% per layer. Reproduce the reference's shapes and kernel route in a test-only mode; bit-exact output there proves the port and attributes the residue to the shape and kernel choices, and that mode becomes a regression test. Test each attribution by making it vanish rather than accepting it because it sounds right; on LTX-2.5's Gemma, one of two plausible causes was false.

The reference is not ground truth either: measure both it and the port against a more accurate third configuration (fp32). A kernel toggle on an already-instantiated remote-code model may not take effect, so check that a reference-vs-reference spread is nonzero, and reset `torch.set_float32_matmul_precision("highest")` in a harness comparing against fp32 (`apply_torch_config` sets TF32 on import).

**For a sampler that amplifies small differences** (a few-step distilled diffusion sampler, any sampled AR stage), the oracle's bytes are not the bar. Measure the reference against itself under a different, equally valid kernel (flash vs memory-efficient SDPA) and hold the served output to that spread, for per-step tensors and for the final media (PSNR for video, a log-spectrum distance for audio).

Make the weight loader raise if any parameter goes unfilled. A silently half-loaded model produces plausible output instead of an error.

Record the oracle once and keep the parity suite; rerun it after every performance commit, and add a parity test per performance commit. Do not add CUDA-graph configs while chasing parity — capture makes every discrepancy harder to localize and the two kinds of debugging do not mix.

The oracle and the server are two sides of an A/B, so they must never share a GPU at the same time: run them on different GPUs when nothing is being timed, and back to back when something is. Record every oracle input you expect to need in one session up front, so you don't stop the server to re-record later.

## Small details to double-check

Each of these is a one-line convention that a reference implementation states somewhere unobvious, runs fine when wrong, and produces output that reads as plausible. They are cheap to verify up front and expensive to find by bisection.

- **Which RoPE variant.** Llama-style scaling applied where it doesn't belong (or omitted where it does) has manifested as a TTS model skipping words and phrases — fluent output, missing content.
- **mRoPE position ids for multimodal spans.** Whether a component should be constant across a media span or advance per token is a per-model decision; the wrong choice has produced plausible but wrong audio transcription.
- **Whether a norm is Gemma-style.** `(1 + weight)` versus `weight`, and a model can use both — Qwen3.5's plain `RMSNorm` scales by `(1 + weight)` while its gated norm does not. The tell is in the checkpoint: a weight tensor whose mean is ~0 rather than ~1 is storing `weight - 1`. Reading `mean()` off one norm tensor is faster than any amount of parity debugging.
- **Stop tokens: trust the tokenizer over `config.json`.** Qwen3.5's `config.json` names `<|endoftext|>` while `tokenizer.eos_token` is `<|im_end|>`, which is what a chat turn actually ends on. Stopping on the config's alone means every reply runs to `max_tokens`, which reads as a sampler or scheduler bug. Union the two.
- **Base versus instruct.** Variants may differ only by a suffix — Qwen tags base checkpoints `-Base`, so the unsuffixed repo is the instruct one and needs the chat template. Applying no template to an instruct model gives fluent, wrong continuations.
- **Key per-request submodule state by `fwd_info.rid_handle` in `prepare_inputs`**, not `request_id`: the forward and the engine's cleanup use the handle.
- **The conductor merges each pass's `step_metadata` into the request's `metadata.kwargs`.** A model whose walks send *different* step metadata (multi-stage, multi-resolution) and also keeps request-level facts in kwargs under the same names has those facts overwritten by the previous walk's values: a stage-1 half-resolution `height` becomes the request's height for stage 2. Keep request-level facts under keys step metadata never uses (LTX-2.5 nests them under `kwargs["request"]`).

## Phase 5 — Batching and CUDA graphs

Add `can_batch` / `forward_batched` and the graph declarations. If only part of the forward is capturable, use piecewise graphs (`PiecewiseBatchedConfig` / `PiecewisePackedConfig`) rather than abandoning capture for the node. `unpack_packed_output` and `postprocess` are the hooks for work on the tail end of a graphed forward.

Take the shape ledger from Phase 1 and give **every varying dimension exactly one treatment**: its own bucket, normalized before the graph, or eager behind an explicit gate. Declare varying dims in the graph config — engine code that infers them from tensor shapes is model-specific and gets rejected. Normalizing an input before a graph changes numerics, so the parity test comes before shipping it. Buckets are not free: each costs startup through a Dynamo re-trace, and batch buckets are geometric. Per-request facts the engine needs for bucketing travel via step metadata into the key-info hook; model-side branching on request shape does not reach the graph selector.

Per-request state is one struct with one cleanup, called on the completed, failed **and** aborted paths. For interactive models abort is the normal ending, so test it: cancel mid-stream, zero output after cancel, slot freed for the next request. A failing request still records its sequence in the reorder buffer.

Cross-request batching needs a batched forward, per-request info threading, a row-independence test and a dedicated padding slot; find the knee with a sweep rather than assuming it. Then check the batch sizes the forward actually runs under concurrency: a per-request counter or position in a grouping key silently batches nothing.

- A model with several KV-backed nodes sizes every KV in its yaml; the defaults assume one, and the last resource to build OOMs.
- A node whose work splits into several piecewise regions runs every call through the runners with `eager_fallback=True` (#383), with eager-only regions for shapes it never captures, rather than keeping a hand-written eager path: each region then plans exactly the rows it runs.
- Stream positions derive from what was written, never from a step count.

Consider making capture a separate PR stacked on the eager MVP, and any **system** change your model needed a separate PR underneath — see "Splitting work across PRs" in AGENTS.md. Recommended, not required; large models are painfully slow without capture, so an eager-only merge is not always useful alone.

## Phase 6 — Look at the output yourself

**Media output must be verified by a human, here and again before the PR goes up.** An agent is fine for sanity checks and for reading text, but audio, images and video need a person to listen to or look at them. Numerical parity passing is not evidence that generated audio sounds right.

Before a person listens, transcribe generated speech (the served model's own audio input works) and compare it with the intended text; it tells wrong tokens from a broken vocoder.

Hand the person files they can play: audio as WAV alongside any mp4, since AAC-in-MP4 does not decode in every player. Put the reference's output next to yours, and the reference-vs-reference numbers from Phase 4 next to the served-vs-reference ones.

## Phase 7 — Benchmark against the competition

Against vLLM, vllm-omni, TensorRT-LLM, SGLang, sglang-omni — whichever serve this model. The goal is genuinely to be better than or on par with them; this is a target, not an observation. Benchmark the competitor's best working config, not its default deploy (which may run eager at batch 1), and say which knobs you tried.

Across a **comprehensive** set, though, not one configuration. A single winning benchmark usually means it got tuned for, and the workloads you didn't measure are where the regression hides: don't buy `image_to_text` throughput with `text_to_text` or mixed-workload throughput. Report which workloads you checked, including the ones that didn't improve.

**The deployment is an axis too.** For a multi-node model, especially a large DiT, the first-order lever is often the config: which nodes are colocated, which GPU each sits on, and TP vs SP vs both on the heavy node. A full sweep is combinatorial and mostly wasted, so reason first: which configs fit in memory, what the per-node time split from the profile says is worth parallelizing, which edges are large enough that colocating their ends matters. Then benchmark the handful of candidates that survive, and ship the winner as the default config.

Use the [benchmarking skill](../benchmarking/SKILL.md) for methodology — warmup, concurrency sweeps, and the traps that make a cross-process A/B move on their own. Millisecond-scale effects are measured server-side (NVTX, nsys), and startup is never compared across days or nodes. For streaming models, realtime is a per-stream verdict (TTFF, gap p50/p95, on-time fraction, stalls), with budget-edge rows labelled borderline. Ship one config per variant with capacity fields at the measured recommendation.

## Phase 8 — Performance tuning

Establish CPU-bound or GPU-bound first; the benchmarking skill covers how (`event_sync` near zero means CPU-bound). That much is settled. Beyond it **there is no established best practice in this repo yet** — current tuning work is iterative and exploratory, so treat any confident-sounding recipe with suspicion, including one you generate yourself. What holds regardless: same harness, same box, two sides adjacent in time, and re-check the workloads you are not targeting before calling a change a win.

Two things that are settled: a host sync in your `prepare_inputs` hides inside that phase rather than in `event_sync`, and a host-bound step gets faster by deleting work, not by moving it to another thread. Values already known on the host travel in `step_metadata`.

If tuning pushes you into the worker or engine rather than your model, read the [async-worker skill](../async-worker/SKILL.md) first.

## Done means

Decomposed by when things run, not collapsed onto one node; registered everywhere [registration.md](references/registration.md) lists; eager parity against an oracle, with the suite kept; every varying dim with exactly one treatment; one cleanup covering completed, failed and aborted; batching and capture declared or a comment saying why a node can't have them; modular tests passing; a human has checked the media; a benchmark against at least one competitor across more than one workload; and no model-owned resource lifecycle anywhere.
