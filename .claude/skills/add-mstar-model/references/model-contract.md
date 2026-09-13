# M* model contract — quick reference

Condensed lookup for the interfaces a new model implements. The authoritative source is
`docs/adding_models.rst` plus `mstar/model/base.py` and `mstar/model/submodule_base.py`.
Read those when this summary is not enough.

## `Model` abstract methods (`mstar/model/base.py`)

Must implement (all `@abstractmethod`):

| Method | Returns | Purpose |
|---|---|---|
| `get_kv_cache_config()` | `list[KVCacheConfig]` | per-node paged-KV configs (layers, kv_heads, head_dim, max_seq_len, num_qo_heads). Empty if no AR node. Encoder-decoder: declare `cross_attn={source: CrossAttnKVConfig}`. |
| `get_node_engine_types()` | `dict[str, EngineType]` | each node → `KV_CACHE` or `STATELESS`. |
| `get_graph_walk_graphs()` | `dict[str, GraphSection]` | `{walk_name: graph}` built from the four primitives. |
| `process_prompt(prompt, in_mods, out_mods, tensors=None, **kw)` | `NameToTensorList` | tokenize + initial tensors; reuse the HF tokenizer/processor here; may read raw media tensors to derive `pixel_values` etc. |
| `get_initial_forward_pass_args(partition, in_mods, out_mods, input_signals, model_kwargs=None)` | `ForwardPassArgs` | seed the first walk. |
| `get_partition_forward_pass_args(partition, metadata, persist_signals, incoming_connections=None)` | `ForwardPassArgs` | state machine — next walk / inputs / `request_done`. |
| `postprocess(output, modality, request_kwargs=None)` | `bytes` | encode final tensor (utf-8 / PNG / PCM). |
| `get_submodule(node_name, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None)` | `NodeSubmodule \| None` | lazily build+load one node; `None` = dummy mode. |

Overridable (not abstract): `get_sampling_config`, `get_aux_sampling_configs`,
`resolve_sampling_configs`, `get_max_output_tokens`, `get_autocast_dtype` (default bf16),
`get_default_sharding_config` (declare `tp_enabled_nodes`/`sp_enabled_nodes`),
`load_image`/`load_audio`/`load_video`, `get_output_sample_rate`/`get_output_audio_channels`,
and the partition API (`get_partition_topology`, `get_partitions`) for async streaming.

## Engine types (`mstar/engine/base.py:EngineType`)

- **`KV_CACHE`** — persistent paged KV cache across forwards: autoregressive LLMs and
  LLM-as-denoiser flow loops. Pairs with `ARNodeSubmodule` + an entry in `get_kv_cache_config`.
- **`STATELESS`** — no cross-step KV state: ViT/VAE/audio encoders & decoders, embedding &
  projection stages, flow-matching combine, codec (waveform) decoders.

## Graph primitives (`mstar/graph/base.py`)

- `GraphNode(name, input_names, outputs)` — one compute unit; `name` matches a
  `get_node_engine_types` key; `outputs` is a list of `GraphEdge`.
- `GraphEdge(next_node, name, persist=?, output_modality=?)` — routes an output tensor.
  `persist=True` carries a tensor across steps/walks (e.g. prefill's token into the decode
  loop). `output_modality` + `next_node=EMIT_TO_CLIENT` streams to the client.
- `Sequential([...])` / `Parallel([...])` — compose subgraphs in order / concurrently.
- `Loop(name, section, max_iters, outputs)` — iterating subgraph; body feeds its outputs back.
  Give it a `name` so a submodule's `check_stop` can stop it early (EOS). This is the decode loop.
- Special destinations: `EMIT_TO_CLIENT`, `EMPTY_DESTINATION` (`mstar/graph/special_destinations.py`).
- `StreamingGraphEdge(next_node, name, target_partition=...)` — cross-partition async edge.

## `NodeSubmodule` contract (`mstar/model/submodule_base.py`)

- `prepare_inputs(graph_walk, fwd_info, inputs, **kw) -> NodeInputs` — cheap host-side only.
- `preprocess(graph_walk, engine_inputs, inputs) -> dict` — collate batch → forward kwargs
  (base handles bs=1; **abstract for `ARNodeSubmodule`**).
- `forward(graph_walk, engine_inputs, **kw) -> NameToTensorList` — pure compute; auto-compiled.
- `postprocess(...)` — metadata-only, **no tensor value reads** (runs on GPU thread).
- `check_stop(...) -> set[str]` — may read tensor values (off GPU thread); returns loop names to stop.
- `cleanup_request(request_id)` — free per-request state; call `super()` if overriding.
- Batching: `can_batch` (default `False`), `forward_batched`, `max_batch_size`.
- CUDA graphs: `get_cuda_graph_configs` (whole forward), `get_piecewise_cuda_graph_configs`
  (inner region, e.g. a block loop), `can_use_cuda_graphs`.
- Stateless flavor: `get_stateless_flavor()` → `"enc_dec"` (default) or `"audio_codec"`.
- `ARNodeInputs` fields: `input_seq_len` (required), `input_ids` or `input_embeds`, `custom_pos_ids`.
- Per-request state: `self.request_state(rid)` (a `PerRequestState`); engine injects the
  batch's states via `ModelInputsFromEngine.per_request_states` and drops them on removal.

## Weight loading (`mstar/model/loader/`)

Three layers:
1. In `get_submodule`: build on `meta`, cast to `autocast_dtype`, `to_empty(device)`, then
   `load_weights(module, source, device=...)` (top-level driver picks single-file vs sharded HF dir).
2. Module implements `load_weights(self, weights)` → delegates to
   `load_hf_weights(self, weights, stacked_params=..., name_remapper=...)`.
3. `stacked_params` (list of `StackedParamRule`) fuse several checkpoint keys into one param
   (e.g. `LLAMA_STACKED_PARAMS` for `q/k/v_proj` → `qkv_proj`). `name_remapper` rewrites/drops keys.

Each param's `weight_loader` also shards along its shard dim when the module was built with a
`comm_group` — one load path serves single-GPU and tensor-parallel.

## Tensor parallelism / sequence parallelism

- Declare shardable nodes: override `get_default_sharding_config()` →
  `ShardingConfig(groups=[], tp_enabled_nodes={...}, sp_enabled_nodes={...}, shard_dim={})`.
- Build components from `mstar/model/components/distributed/` with the `comm_group`; otherwise
  a node is replicated on every rank.
- Config: add `tp_size` (and `sp_size`) to a `node_groups` entry with that many ranks. A
  `tp_size>1` group naming a non-TP-enabled node is rejected at load.
- Activation sharding across a node boundary: `shard_dim` map (edge/signal name → split dim);
  absent ⇒ replicated. Overridable per-run under a `sharding_config:` YAML block.

## Config YAML (`configs/<name>.yaml`)

```yaml
model: "<registry_key>"
max_seq_len: 2048
node_groups:
  - node_names: ["LLM"]
    ranks: [0]
    # optional: tp_size, sp_size, graph_walks: [prefill, decode]
```
Naming convention encodes layout: `_tp2`, `_sp2`, `_pd_disaggregated`, `_colocated`, etc.
Disaggregation = pin the same node to different ranks per `graph_walks`.

## Validate

```bash
ruff check .          # CI enforces
pytest test/modular/  # CPU, dummy-mode submodules (get_submodule → None)
mstar-serve --config configs/<name>.yaml --port 8000
```
