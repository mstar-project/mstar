---
name: add-mstar-model
description: >-
  Port a new model into the M* serving system, typically from a reference
  implementation such as a HuggingFace `transformers`/`diffusers` model. Use
  when asked to "add a model", "support <model> in M*", "port <HF model>",
  implement a new `Model` subclass under `mstar/model/`, or wire a checkpoint
  into a `configs/*.yaml`. Covers the HF-to-M* translation workflow, which
  reference model to copy, the required contract, and how to validate.
---

# Adding / porting a model into M*

M* serves any-to-any multimodal models as a **dataflow graph of components**, where
every request is a **Walk** over that graph. Adding a model means expressing the model
in that abstraction — NOT re-implementing a `.generate()` loop.

**Always read `docs/adding_models.rst` first** — it is the canonical, maintained contract
reference (abstract methods, submodule lifecycle, CUDA graphs, tensor parallelism, worked
examples). This skill is the *porting workflow* layered on top of it; `references/model-contract.md`
in this skill folder is a fast lookup of the contract when you don't want to reread the full doc.

## The key reframe (read this before touching code)

An HF model is **monolithic**: one `from_pretrained` + `.generate()` that hides
tokenization, the KV cache, sampling, the decode loop, and stopping. M* **owns** the KV
cache, sampling, batching, CUDA graphs, and the decode loop. So porting is mostly
**subtraction and re-mapping**, not rewriting the math.

| In the HF reference | Becomes in M* |
|---|---|
| `config.json` / `PretrainedConfig` | `config.py` dataclass |
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

## Workflow

### Step 0 — Analyze the reference model's shape
From the HF impl, answer three questions — they determine the whole structure:
1. **How many distinct compute stages** that could live on separate GPUs? Each is a **graph node**. (Orpheus = LLM + SNAC → 2 nodes; Whisper = encoder + decoder → 2 nodes; plain LLM → 1 node; BAGEL = ViT + VAE-enc + LLM + VAE-dec.)
2. **Which stages are autoregressive** (need a persistent KV cache across steps)? Those become `EngineType.KV_CACHE` nodes; everything else (encoders, decoders, VAE, codecs, projections) is `EngineType.STATELESS`.
3. **What are the graph walks?** text→text ⇒ `prefill` + `decode`; encoder-decoder ⇒ `prefill_<enc>` + `decode`; diffusion / LLM-as-denoiser ⇒ an `image_gen`/denoise `Loop`; unified understanding+generation ⇒ several walks (see BAGEL).

### Step 1 — Pick the closest reference and copy it
Do NOT start from scratch. Copy the nearest existing model package and rename:

| Target shape | Copy | Path |
|---|---|---|
| Plain or streaming autoregressive LLM (± codec) | **Orpheus** | `mstar/model/orpheus/` |
| Encoder-decoder (audio/vision → text, cross-attention) | **Whisper** | `mstar/model/whisper/` |
| ASR / audio-tower + LLM | **Higgs-Audio** | `mstar/model/higgs_audio/` |
| Unified understanding + generation, CFG-parallel | **BAGEL** | `mstar/model/bagel/` |
| Full omni (text/audio/vision in and out) | **Qwen3-Omni** | `mstar/model/qwen3_omni/` |
| Diffusion / LLM-as-denoiser, video/world model, TP+SP | **Cosmos3** / **Wan2.2** | `mstar/model/cosmos3/`, `mstar/model/wan22/` |
| Vision-language-action policy | **Pi0.5** / **Cosmos3-DROID** | `mstar/model/pi05/`, `mstar/model/cosmos3/` |

Scaffold: `mstar/model/<name>/{__init__.py, config.py, <name>_model.py, submodules.py, components/}`.

### Step 2 — `config.py`
Translate the HF `config.json` fields you need into a `@dataclass` (hidden size, layers,
attention & kv heads, head_dim, vocab, `max_position_embeddings`, rope theta) plus
generation defaults (temperature, top_p, etc.). If you support multiple sizes, read dims
from the checkpoint's `config.json` at load time (see Cosmos3).

### Step 3 — Port the `nn.Module`s into `components/` (the real work)
Keep the HF layer math; swap exactly two things:
- **Attention** → M*'s attention that reads/writes the paged KV cache via `cache_handle`
  (`mstar/model/components/attention.py`; TP variant in `components/distributed/attention.py`).
  Remove all `past_key_values` plumbing — the engine plans attention and owns the cache.
- **Linears / embeddings** → for tensor parallelism, build from `mstar/model/components/distributed/`
  (`ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`, `ParallelGatedMLP`,
  `ParallelAttention`) constructed with the `comm_group`. If you don't need TP, plain
  `nn.Linear` works and the node is simply replicated on each rank.

Then give the module a **`load_weights(self, weights)`** delegating to
`load_hf_weights(self, weights, stacked_params=..., name_remapper=...)` from `mstar.model.loader`:
- Use `LLAMA_STACKED_PARAMS` for Llama-family models (fuses HF `q/k/v_proj` → `qkv_proj`).
- Write a `name_remapper` for keys that don't line up (prefixes, dropped buffers).
- Sharding happens automatically inside each parameter's `weight_loader` when built with a `comm_group`.

### Step 4 — `submodules.py`: wrap each node
One `NodeSubmodule` per node (AR nodes subclass `ARNodeSubmodule`). Split HF's `forward`:
- **`prepare_inputs`** — cheap host-side only (slice token ids, build position info). Runs off the GPU thread; no heavy compute.
- **`preprocess`** — collate a batch into `forward` kwargs; plan attention into the cache. **Abstract for AR nodes.**
- **`forward`** — the HF forward body minus cache/sampling; pure tensor → `NameToTensorList`. **Auto-`torch.compile`d** — keep it compile-friendly.
- **`check_stop`** — where HF stopping criteria / EOS go; reads token values off the GPU thread and returns which `Loop`s to stop. **This is how decode terminates — not an edge flag.**
- Opt into perf later: `can_batch`/`forward_batched`, `get_cuda_graph_configs`, `get_piecewise_cuda_graph_configs`.

Sampling is NOT in your forward — the engine samples from `get_sampling_config`.

### Step 5 — `<name>_model.py`: the 8 abstract methods (pure glue, no math)
Crib directly from `mstar/model/orpheus/orpheus_model.py`:
- `get_kv_cache_config` — fill from the config dataclass (encoder-decoder: add `cross_attn`).
- `get_node_engine_types` — node → `KV_CACHE`/`STATELESS` dict.
- `get_graph_walk_graphs` — declare the `prefill` node + `decode` `Loop` (and any others).
- `process_prompt` — **reuse the HF tokenizer/processor here**; compute derived tensors (`pixel_values`, `audio_features`) for multimodal.
- `get_initial_forward_pass_args` / `get_partition_forward_pass_args` — the walk state machine (prefill→decode until EOS).
- `postprocess` — encode output bytes (utf-8 / PNG / PCM).
- `get_submodule` — build on `meta`, cast to `autocast_dtype`, `to_empty(device)`, `load_weights(...)`, wrap, cache. Reuse HF `from_pretrained` for stages you don't shard (Orpheus does this for SNAC).

### Step 6 — Register + config
- `mstar/model/registry.py`: add to `MODEL_REGISTRY` (name→class) and `HF_MODELS` (name→checkpoint).
- `configs/<name>.yaml`: `node_groups` → ranks. Start single-GPU with one group.
- Optional: add a `DEFAULT_CONFIGS` entry in `mstar/cli/main.py` so `mstar serve <name>` resolves it.

### Step 7 — Validate the plumbing WITHOUT weights
Modular tests run submodules in **dummy mode** (`get_submodule` returns `None`) — catch
graph/wiring bugs before touching checkpoints:
```bash
ruff check .            # CI enforces this
pytest test/modular/    # CPU graph/scheduling/adapter tests
```

### Step 8 — Bring up real weights and verify numerics
The two bug-prone areas are the **weight remap** (Step 3) and the **attention/cache port**. Verify against HF:
1. After `load_weights`, assert no params were left unconsumed or missing (the loader reports loaded names).
2. Run one M* forward and compare logits/hidden states to the HF reference on the same input — should match within bf16 tolerance. Do this **before** trusting generated output.
3. `mstar-serve --config configs/<name>.yaml --port 8000`, then `POST /generate` and confirm the stream.

### Step 9 — Opt into performance (optional, after correctness)
Continuous batching (`can_batch`/`forward_batched`), CUDA graphs (`get_cuda_graph_configs`),
then tensor parallelism via YAML `tp_size` + `get_default_sharding_config`. See
`docs/adding_models.rst` Steps 5 and 7.

## Where the time goes
Steps 3 and 8 — the attention/KV-cache surgery and the weight-name remap — are the real
work. Steps 0, 2, 5, 6, 7 are mechanical translation you can crib almost line-for-line
from the nearest reference.

## Common pitfalls
- **Leftover `past_key_values` logic** in the ported forward — remove it; the engine owns the cache.
- **Sampling / stopping inside `forward`** — sampling belongs to the engine (`get_sampling_config`); stopping belongs in `check_stop`.
- **Host syncs (`.item()`/`.cpu()`) in `forward`/`postprocess`** — they stall the GPU thread. Value-dependent decisions go in `check_stop`.
- **Passing sampling params as Python scalars into a captured forward** — they get baked into the CUDA graph; use sampler buffers / aux configs.
- **Building weights on the real device** — build on `meta`, cast to `autocast_dtype`, then `to_empty(device)` to avoid a fp32 VRAM peak.
- **`node_names` in YAML not matching graph node names** — they must match `get_node_engine_types` keys exactly.
- **Forgetting `tp_enabled_nodes`** — a `tp_size>1` group naming a node not declared TP-enabled is rejected at load.

## Definition of done (checklist)
```
[ ] mstar/model/<name>/config.py           — config dataclass
[ ] mstar/model/<name>/components/          — nn.Modules + load_weights
[ ] mstar/model/<name>/submodules.py        — NodeSubmodule per node
[ ] mstar/model/<name>/<name>_model.py      — 8 abstract methods
[ ] mstar/model/registry.py                 — MODEL_REGISTRY (+ HF_MODELS)
[ ] configs/<name>.yaml                     — node_groups → ranks
[ ] ruff check .  &&  pytest test/modular/  — green in dummy mode
[ ] numerics verified vs HF reference
[ ] mstar-serve + POST /generate            — end-to-end stream confirmed
[ ] (optional) batching / CUDA graphs / TP
```
