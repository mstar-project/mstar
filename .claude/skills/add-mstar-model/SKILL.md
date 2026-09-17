---
name: add-mstar-model
description: >-
  Port a new model into the M* serving system, typically from a reference
  implementation such as a HuggingFace `transformers`/`diffusers` model or a
  research codebase (NeMo, etc.). Use when asked to "add a model", "support
  <model> in M*", "port <HF model>", implement a new `Model` subclass under
  `mstar/model/`, or wire a checkpoint into a `configs/*.yaml`. Covers the
  reference-to-M* translation workflow, which reference model to copy, the
  required contract, how to verify numerics, and how to bring the full engine +
  serving stack up (including multi-stage / streaming / batched models).
---

# Adding / porting a model into M*

M* serves any-to-any multimodal models as a **dataflow graph of components**, where
every request is a **Walk** over that graph. Adding a model means expressing the model
in that abstraction — NOT re-implementing a `.generate()` loop.

**Always read `docs/adding_models.rst` first** — it is the canonical, maintained contract
reference. This skill is the *porting workflow* layered on top of it. `references/model-contract.md`
is a fast contract lookup; `references/engine-and-serving.md` is the deep-dive for multi-stage
async models, the live serving stack, and debugging a hung request — read it when your model has
more than one autoregressive stage, streams output, or you're bringing up `mstar-serve`.

## The key reframe (read this before touching code)

A reference model is **monolithic**: one `from_pretrained` + `.generate()` that hides
tokenization, the KV cache, sampling, the decode loop, and stopping. M* **owns** the KV
cache, sampling, batching, CUDA graphs, and the decode loop. So porting is mostly
**subtraction and re-mapping**, not rewriting the math.

| In the reference | Becomes in M* |
|---|---|
| `config.json` / `PretrainedConfig` | `config.py` dataclass, read from the checkpoint |
| `AutoTokenizer` / `AutoProcessor` | reused as-is inside `process_prompt` |
| `nn.Module` math (attention, MLP, decoder layers) | `components/` — keep the math, swap attention + linears |
| the model's own KV cache / `past_key_values` | **deleted** — the M* engine owns it |
| `.generate()` loop, sampling, stopping criteria | **deleted** — a graph `Loop` + engine sampler + `check_stop` |
| `from_pretrained` weight names | `load_weights` remap (`stacked_params` + `name_remapper`) |
| the single `forward()` | split into `prepare_inputs → preprocess → forward` |

## The three layers you produce

| Layer | Where | GPU compute? | Your job |
|---|---|---|---|
| `Model` subclass | `mstar/model/<name>/<name>_model.py` | No — pure contract | tokenize, declare graph, map nodes→engines, build fwd-pass args, postprocess |
| `NodeSubmodule`s | `mstar/model/<name>/submodules.py` | Yes — the `nn.Module` wrappers | `prepare_inputs → preprocess → forward`, weight loading, batching/CUDA-graph opt-in |
| Config YAML | `configs/<name>.yaml` | — | map graph nodes → GPU ranks (single-GPU vs disaggregated, **no code change**) |

## Verification strategy: build a verified oracle, then check every layer against it

This is the single most important habit for a non-trivial port. The engine has many moving
parts (weight remap, attention/cache port, walk wiring, streaming, batching, serving). When
the final output is wrong you need to know *which layer* broke — and a live end-to-end run
is the worst place to first discover a bug. So build verified ground truth from the bottom up
and never let an unverified layer sit under the one you're debugging.

1. **Component forwards vs the reference, in isolation.** Feed each `components/` module
   random inputs and diff its output against the reference module's forward (bf16/fp32
   tolerance). This localizes numeric bugs (weight remap, attention math, grouped/tiled
   reshapes, norm epsilons) to the smallest unit. These checks are independent — parallelize
   them across subagents.
2. **A standalone pipeline that bypasses the engine.** For anything past a single LLM, write a
   plain offline path that calls the raw component modules directly (no conductor/workers) and
   reproduces the reference end-to-end. Verify it matches the reference on realistic input.
   **This standalone path is now your oracle.** It decouples "is the math right" from "is the
   engine wiring right" — you'll reuse it to check the engine port, per-request, without a
   reference. (For a plain LLM this collapses to "one M* forward vs HF logits".)
3. **Engine node forwards vs the oracle.** Each `submodule.forward`, driven directly, must
   reproduce the standalone path's output for that stage.
4. **Walks derive vs the engine, without a GPU.** `get_worker_graphs(config)` and constructing
   the `Conductor` resolve every edge/partition/stream — a dangling edge name or unresolved
   partition fails here. Cover this with a `test/modular/` structural test.
5. **Live serving vs the oracle.** Only now run `mstar-serve` + a request, and compare the
   per-request output to the oracle (`offline path`) on the same input.

**Two traps that make verification lie to you:**
- **The reference's own example may not exercise the code.** A toy/near-empty input (a silent
  audio clip, a one-token prompt, a request that stays on the happy path) can pass while real
  inputs hit unported branches. Verify with *realistic* inputs, and probe internal activity
  (e.g. output variety), not just "it ran".
- **A stochastic reference won't match bit-for-bit.** Seed the RNG (or disable sampling/guidance)
  to get a deterministic diff for correctness; test the stochastic path separately for quality.

## Workflow

### Step 0 — Analyze the reference model's shape
Answer three questions — they determine the whole structure:
1. **How many distinct compute stages** that could live on separate GPUs? Each is a **graph node**. (LLM + codec → 2 nodes; encoder + decoder → 2 nodes; plain LLM → 1 node; encoder + LLM + talker + codec → 4 nodes.)
2. **Which stages are autoregressive / carry cross-step state?** Those are `EngineType.KV_CACHE` nodes; encoders, decoders, VAEs, codecs, and projections are `EngineType.STATELESS`. Note any **non-attention** recurrent state (Mamba/SSM conv+ssm state) — the engine's KV pool does *not* cover it; see below.
3. **What are the graph walks, and is the model turn-based or streaming?** text→text ⇒ `prefill` + `decode`; encoder-decoder ⇒ `prefill_<enc>` + `decode`; multi-stage streaming (stage A streams to stage B as it generates) ⇒ **async partitions** — read `references/engine-and-serving.md`.

### Step 1 — Pick the closest reference and copy it
Do NOT start from scratch. Copy the nearest existing model package and rename:

| Target shape | Copy | Path |
|---|---|---|
| Plain or streaming autoregressive LLM (± codec) | **Orpheus** | `mstar/model/orpheus/` |
| Encoder-decoder (audio/vision → text, cross-attention) | **Whisper** | `mstar/model/whisper/` |
| ASR / audio-tower + LLM | **Higgs-Audio** | `mstar/model/higgs_audio/` |
| Two AR stages streaming (LLM → talker → vocoder), aux sampling, async partitions | **Qwen3-Omni** / **Qwen3-TTS** | `mstar/model/qwen3_omni/`, `mstar/model/qwen3_tts/` |
| Unified understanding + generation, CFG-parallel | **BAGEL** | `mstar/model/bagel/` |
| Diffusion / LLM-as-denoiser, video/world model, TP+SP | **Cosmos3** / **Wan2.2** | `mstar/model/cosmos3/`, `mstar/model/wan22/` |
| Vision-language-action policy | **Pi0.5** | `mstar/model/pi05/` |

Scaffold: `mstar/model/<name>/{__init__.py, config.py, <name>_model.py, submodules.py, components/}`.

### Step 2 — `config.py` (config-driven, no hardcoded hyperparameters)
Translate the checkpoint's `config.json` into `@dataclass`es, and **read every dimension and
hyperparameter from that file** at load time — sizes, head counts, rope theta, norm eps, vocab,
special-token ids, *and* generation/sampling defaults (temperature, top-p, guidance/noise scales,
seeds). Hardcoded constants are the most common source of silent drift when a checkpoint variant
differs from the one you first read. Provide a `from_pretrained(ckpt_dir)` that populates the
dataclasses from the JSON.

### Step 3 — Port the `nn.Module`s into `components/` (the real work)
Keep the reference layer math; swap exactly two things:
- **Attention** → M*'s attention that reads/writes the paged KV cache via `cache_handle`
  (`mstar/model/components/attention.py`; TP variant in `components/distributed/attention.py`).
  Remove all `past_key_values` plumbing — the engine plans attention and owns the cache. Keep a
  separate eager full-sequence forward for the standalone oracle.
- **Linears / embeddings** → for tensor parallelism, build from `mstar/model/components/distributed/`
  with the `comm_group`. If you don't need TP, plain `nn.Linear` works and the node is replicated.

Give each module **`load_weights(self, weights)`** delegating to `load_hf_weights(...)` with
`stacked_params` + a `name_remapper`. **Assert nothing is left unmatched or missing** — a silent
partial load produces plausible-but-wrong output that wastes hours downstream.

**Now do Verification steps 1–2** (component diffs + standalone oracle) before wiring the engine.

### Step 4 — `submodules.py`: wrap each node
One `NodeSubmodule` per node (AR nodes subclass `ARNodeSubmodule`). Split the reference forward:
- **`prepare_inputs`** — cheap host-side only; runs off the GPU thread.
- **`preprocess`** — collate a batch → `forward` kwargs; plan attention into the cache. **Abstract for AR nodes.**
- **`forward`** — the forward body minus cache/sampling; pure tensor → `NameToTensorList`. Auto-`torch.compile`d — keep it compile-friendly, or set `disable_torch_compile = True` if it holds capture-unsafe Python state (custom per-request state, internal sampling).
- **`check_stop`** — where stopping / EOS go; reads token values off the GPU thread and returns which `Loop`s to stop. **This is how decode terminates.**
- The forward output must be keyed by the **graph edge name** the downstream node consumes (see pitfalls). Opt into perf later: `can_batch`/`forward_batched`, CUDA-graph configs.

Sampling is NOT in your forward — the engine samples from `get_sampling_config` (and per-channel `get_aux_sampling_configs` for a second token stream). Verify each `forward` against the oracle (step 3).

### Step 5 — `<name>_model.py`: the abstract methods (pure glue, no math)
Crib from the nearest reference. `get_kv_cache_config`, `get_node_engine_types`,
`get_graph_walk_graphs`, `process_prompt` (reuse the tokenizer/processor; derive multimodal
tensors), `get_initial_forward_pass_args` / `get_partition_forward_pass_args` (the walk state
machine), `postprocess`, `get_submodule`. For multi-stage streaming also implement
`get_partitions` / `get_partition_topology` and the per-partition state machine —
**`references/engine-and-serving.md`**.

### Step 6 — Register + config
- `mstar/model/registry.py`: add to `MODEL_REGISTRY` (name→class) and `HF_MODELS` (name→checkpoint).
- `configs/<name>.yaml`: `node_groups` → ranks. Start single-GPU with one group holding all nodes.

### Step 7 — Validate the plumbing WITHOUT weights or a GPU
```bash
ruff check .            # CI enforces this
pytest test/modular/    # CPU graph/scheduling tests; submodules in dummy mode (get_submodule → None)
```
Add a `test/modular/test_<name>_model.py` (mirror `test_qwen3_tts_model.py`) asserting the walks,
engine types, partitions/topology, `get_worker_graphs` derivation, and forward-pass-args routing —
this locks in Verification step 4 as CI-checked.

### Step 8 — Bring up real weights and verify the engine vs the oracle
`load_weights`, then per Verification steps 3 & 5. Compare M* forwards / per-request serving output
to the **standalone oracle** on realistic input, before trusting anything.

### Step 9 — Live serving
`mstar-serve --config configs/<name>.yaml` then `POST /generate`. Multi-stage streaming and the
first-run serving gotchas (they cost real time) are in **`references/engine-and-serving.md`** —
read it before this step.

### Step 10 — Performance (optional, after correctness)
Continuous batching (`can_batch`/`forward_batched`), CUDA graphs, then tensor parallelism via YAML
`tp_size` + `get_default_sharding_config`.

## Models with non-attention recurrent state (Mamba / SSM / conv)
The engine's paged-KV pool covers **attention layers only**. Any other per-step state (Mamba conv
+ ssm state, RNN state) is yours to manage: keep it in the submodule's `PerRequestState` and thread
it into the mixer forward via a small accessor. For **true batched inference**, the mixer must
stack each request's state into a batch tensor, run one fused step, and write each slice back —
attention batches for free via the paged pool, but the recurrent state does not. Do this eager
first (state in a per-request dict, `disable_torch_compile = True`); a CUDA-graph-capturable
fixed-buffer state pool is a later perf layer. Verify the batched step equals the per-request path.

## Common pitfalls

**Porting the math**
- **Leftover `past_key_values` logic** in the ported forward — remove it; the engine owns the cache.
- **Sampling / stopping inside `forward`** — sampling belongs to the engine; stopping to `check_stop`.
- **Silent partial weight load** — assert no missing/unmatched params; wrong-but-plausible output otherwise.
- **Reshape/grouping quirks** (grouped-query, group→head for SSM, interleave-vs-tile) — the reference's
  *naive* path may differ from its optimized kernel; diff against the deployed kernel, not the fallback.
- **Norms in the wrong dtype** — a fused bf16 norm vs an fp32 reference norm shifts a bit-exact diff.

**Engine wiring / hangs (a hang = a node never became ready, or a loop never terminated)**
- **Output key ≠ edge name.** A `submodule.forward` must return tensors under the *exact* name of the
  graph edge the next node consumes. A mismatch means the consumer waits forever (hang, no error).
- **Decode-loop inputs not seeded on the first iteration.** If a loop node's `input_names` include
  fed-back tensors (previous token, previous embeds), the first iteration has no prior value to feed
  them → the node never becomes ready. Seed them (mirror how the reference partition seeds its first
  decode input) or the request hangs right after submission.
- **Consumer partition scheduling.** A stage that consumes another's stream must gate on that stream;
  copy the producer→consumer partition + `StreamingGraphEdge` + chunk-policy + forward-pass-args
  state machine from `qwen3_omni` rather than inventing one.
- **Host syncs (`.item()`/`.cpu()`) in `forward`/`postprocess`** — they stall the GPU thread; value-dependent decisions go in `check_stop`.
- **Sampling params as Python scalars into a captured forward** — baked into the CUDA graph; use sampler buffers / aux configs.
- **`node_names` in YAML not matching graph node names**, or a `tp_size>1` group naming a non-TP-enabled node — rejected at load.

**Live serving (first-run gotchas — see the reference for the fixes)**
- **The installed `mstar-serve` imports the main checkout, not your worktree** — set `PYTHONPATH` to your
  checkout so `import mstar` resolves your code, or your new model isn't in the registry.
- **`mstar serve <name>`** (quickstart) has a hardcoded model allow-list; use **`mstar-serve --config <yaml>`** for a new model.
- **Default tensor transport is RDMA (Mooncake)** and fails without InfiniBand — use `--tensor-comm-protocol SHM` single-node.
- **Media ingestion goes through `model.load_audio`/`load_image`**, whose base impl may use libraries with fragile native deps — override it (e.g. decode via `soundfile`).
- **The data worker keys uploaded media as `f"{modality}_inputs"`** (e.g. `audio_inputs`) — map that to your node's input name inside `process_prompt`.
- **Debug a hang with trace logging** at each node's forward + the forward-pass-args state machine; a hang localizes to the first node that never logs.

## Definition of done (checklist)
```
[ ] config.py from the checkpoint json (no hardcoded hyperparameters)
[ ] components/ nn.Modules + load_weights  — component forwards diffed vs reference
[ ] standalone offline path                — verified vs reference (the oracle)
[ ] submodules.py NodeSubmodule per node   — each forward diffed vs the oracle
[ ] <name>_model.py abstract methods       — walks + (partitions/topology if streaming)
[ ] registry.py + configs/<name>.yaml
[ ] ruff check .  &&  pytest test/modular/  — green (incl. a structural walk test)
[ ] engine forwards / serving verified vs the oracle on realistic input
[ ] mstar-serve --config … + POST /generate — end-to-end stream confirmed
[ ] (if recurrent state) batched step verified vs per-request
[ ] (optional) batching / CUDA graphs / TP
```
