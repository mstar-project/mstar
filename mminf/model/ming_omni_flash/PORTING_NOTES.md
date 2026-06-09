# Ming-flash-omni-2.0 — porting notes

Native mminf port of `inclusionAI/Ming-flash-omni-2.0`. This directory is a
scaffold today; everything below is the punch list to make it real.

## Status

- `benchmark/base.py` has `MingFlashOmni` + `ModelType.MING_FLASH_OMNI`.
  Benchmarking against a vllm-omni server **works today** with
  `--inference-system vllm_omni` (see `benchmark/vllm_omni_instructions.md`).
- Step 1 (config port) — DONE. `mminf/model/ming_omni_flash/config.py`
  loads the released ckpt; 10 tests in `test/modular/test_ming_flash_omni_config.py`.
- Step 2 (tokenizer + processor wiring) — DONE.
  `MingFlashOmniModel.__init__` resolves the snapshot, stages Ming source
  files (see "Ming source dependency" below), and loads
  `BailingTokenizer` + `BailingMM2Processor` with graceful fallback;
  11 tests in `test/modular/test_ming_flash_omni_tokenizer.py`.
- Everything else in `MingFlashOmniModel` still raises `NotImplementedError`
  — `mminf-serve --config configs/ming_flash_omni.yaml` will fail at
  startup until step 3+ lands.

## Ming source dependency (loading the tokenizer/processor)

The released HF checkpoint `inclusionAI/Ming-flash-omni-2.0` ships
**only weights and sub-dir configs**. The tokenizer/processor Python
modules (`configuration_bailingmm2.py`, `tokenization_bailing.py`,
`processing_bailingmm2.py`, etc.) live in the source repo at
https://github.com/inclusionAI/Ming . To load the tokenizer/processor:

```bash
# 1. Clone the source repo
git clone https://github.com/inclusionAI/Ming.git /path/to/Ming

# 2. Install extra Python deps Ming's modules depend on
pip install opencv-python-headless openai-whisper

# 3. Tell mminf where to find the source repo
export MING_CODE_DIR=/path/to/Ming
# (or pass ming_code_dir="/path/to/Ming" to MingFlashOmniModel)
```

`MingFlashOmniModel.__init__` (via `_prepare_tokenizer_dir`) symlinks
the required .py and .json files from `$MING_CODE_DIR` alongside the
snapshot's `config.json` so transformers' `trust_remote_code` machinery
can resolve them. The snapshot dir is also pushed onto `sys.path` so
the dynamic-module loader's sibling imports resolve.

## Role-handling nuance (chat templates)

Ming-flash-omni-2.0 ships **two** chat-template implementations with
**different role conventions**:

- `tokenizer.apply_chat_template(messages)` — uses the **jinja template
  in `tokenizer_config.json`**. Accepts standard OpenAI roles
  (`user` / `assistant` / `system`) and remaps them to Ming's uppercase
  `HUMAN` / `ASSISTANT` / `SYSTEM` inside the template. This is the path
  vllm-omni's serving layer uses → the benchmark side works unchanged.

- `processor.apply_chat_template(messages, sys_prompt_exp=..., use_cot_system_prompt=...)`
  — uses the **Python implementation in `BailingMM2Processor`** (Ming
  source repo). **Strict**: asserts `role in [HUMAN, ASSISTANT]` and
  raises `AssertionError` on lowercase OpenAI roles. The native mminf
  `process_prompt` (step 7) will need this path for the multimodal
  preprocessing (vision feature extraction, audio padding, etc.) and
  must explicitly remap roles before calling.

## Upstream reference

Treat the vllm-omni port as the source of truth for architecture. Files to
read (totals ~6.5 KLOC):

| Concern | vllm-omni file |
|---|---|
| Pipeline glue | `vllm_omni/model_executor/models/ming_flash_omni/pipeline.py` (141 LOC) |
| Top-level model | `ming_flash_omni.py` (255 LOC) |
| Thinker (Ling-2.0 MoE + multimodal) | `ming_flash_omni_thinker.py` (1,164 LOC) |
| Talker (CFM + LLM) | `ming_flash_omni_talker.py` (586) + `talker_module.py` (1,145) |
| Audio VAE | `audio_vae.py` (392) |
| Audio encoder | `audio_encoder.py` (246) |
| Vision encoder | `vision_encoder.py` (125) + `projectors.py` (184) |
| Ling MoE backbone | `modeling_bailing_moe_v2.py` (892) |
| Prompt utils | `prompt_utils.py` (134) — `IMAGE_PATCH_TOKEN`, `DEFAULT_NUM_QUERY_TOKENS=256`, TTS caption template |
| Text processing | `text_processing.py` (535) |
| Speaker presets | `spk_embedding.py` (44) + `voice_presets.py` (289) |
| Config | `vllm_omni/transformers_utils/configs/ming_flash_omni.py` (420) |
| Stage input processor | `vllm_omni/model_executor/stage_input_processors/ming_flash_omni.py` |
| ImageGen pipeline | `vllm_omni/diffusion/models/ming_flash_omni/` |
| Deploy yamls | `vllm_omni/deploy/ming_flash_omni{,_image,_thinker_only,_tts}.yaml` |

## mminf parallels

Mirror the structure of `mminf/model/qwen3_omni/` end-to-end. That model is
the closest analog (multimodal thinker + speech talker + vocoder), and the
graph-walk / partition / streaming patterns transfer 1:1.

| mminf surface | Qwen3-Omni reference | Ming-flash-omni equivalent |
|---|---|---|
| Model class | `qwen3_omni_model.py` (1,529) | `ming_omni_flash_model.py` |
| Submodules | `submodules.py` (2,016) | `submodules.py` (TODO) |
| Config | `config.py` (544) | `config.py` |
| Talker | `components/talker.py` (549) + `code2wav.py` (534) | `components/talker.py` + `audio_vae.py` (TODO) |
| Thinker | `components/thinker.py` (259) | `components/thinker.py` (TODO) |
| Attention / RoPE | `components/attention.py` + `rope.py` | likely shareable; check Ling-2.0 attention shape |

## Punch list (in order)

1. **Config port — DONE.** `mminf/model/ming_omni_flash/config.py`
   loads `config.json` + sibling subdir configs (talker / image-gen) into
   a dataclass tree. Verified via 10 tests in
   `test/modular/test_ming_flash_omni_config.py`.

2. **Tokenizer + processor — DONE.** `MingFlashOmniModel.__init__`
   resolves the snapshot, stages Ming source files alongside it (see
   "Ming source dependency" above), and loads `BailingTokenizer` +
   `BailingMM2Processor` with graceful fallback. The chat-template role
   handling has two paths (see "Role-handling nuance" above); the native
   `process_prompt` (step 7) will use the strict processor path and must
   remap roles. Verified via 11 tests in
   `test/modular/test_ming_flash_omni_tokenizer.py`.

3. **Ling-2.0 thinker LLM port — IN PROGRESS.**
   - **3a — DONE** (`components/router.py`, `rope.py`, `attention.py`):
     architecture-novel pieces (MultiRouter group-limited top-k, partial
     3D `video_rope`, QK-norm attention). 12 tests in
     `test/modular/test_ming_flash_omni_components.py`.
   - **3b — DONE** (`components/moe.py`, `decoder_layer.py`, `model.py`):
     `LingMoeBlock` (3-router text/image/audio with `torch.where`
     per-token swap), `LingDecoderLayer` (hybrid dense/MoE per
     `first_k_dense_replace`), full `LingMoeModel` (embed + N layers +
     RMSNorm + lm_head). 9 tests in `test_ming_flash_omni_model.py`.
   - **3c — DONE** (`loader.py`): weight loader that maps the released
     ckpt's `model.model.*` namespace to `LingMoeModel`'s state_dict,
     with per-expert gate/up/down fusion into the packed
     `experts.gate_up_proj` tensor via mminf's existing
     `WeightConverter` machinery. Real-ckpt smoke test loads embed +
     dense layer 0 + lm_head from the released shards and runs a
     forward — output is finite bf16 logits at the expected
     `(T, vocab_size)` shape. 6 tests in
     `test_ming_flash_omni_loader.py` (4 pure-Python + 2 CUDA+snapshot).
   - **3e — DONE** (TP-aware variants): `LingAttention` uses
     `QKVParallelLinear` + `RowParallelLinear` (per-rank heads + dense
     row-parallel); `LingMoeBlock` shards fused experts by
     `shard_inter = moe_intermediate_size / tp_size` and uses mminf's
     existing `_gate_up_weight_loader` / `_down_proj_weight_loader`
     for per-rank weight slicing; dense layer-0 MLP uses
     `ParallelGatedMLP`; `LingMoeModel` threads `comm_group` through
     every decoder layer. Weight loader refactored onto mminf's
     `load_hf_weights` + 770 `StackedParamRule`s (3 per expert ×
     num_experts + dense MLP + synthetic QKV). The packed
     `attention.query_key_value.weight` from the checkpoint is split
     into synthetic `q_proj` / `k_proj` / `v_proj` keys by
     `_split_packed_qkv` so `QKVParallelLinear`'s standard weight
     loader handles per-rank head slicing.

     **Verified via TP=8 mminf-serve smoke** (8 H100s): server starts,
     all 8 workers load 507 thinker params each (one per packed
     parameter; per-rank ~40 GB), KVCacheEngine warmup_and_capture
     completes, torch.compile applies, dedicated GPU threads spin up,
     port 8092 listens. Per-rank model + KV cache is well under 80 GB.
     TP=4 was tried first and OOMed at 78.58 GB / 80 GB; TP=8 has
     plenty of headroom.

     **Known gap (resolved in 3f)**: see step 3f.

   - **3d — DONE** (cache wiring + submodule + engine integration):
     `LingAttention` now uses `cache_handle.run_attention` for paged
     KV-cache attention (keeps the custom partial-3D rope inline);
     `BailingMoeV2ThinkerSubmodule` in `submodules.py` implements
     `prepare_inputs` / `preprocess` / `forward` / `check_stop` for
     the prefill + decode walks; `MingFlashOmniModel.__init__` no
     longer raises NotImplementedError and all Model ABC methods
     (`get_kv_cache_config`, `get_graph_walk_graphs`, `get_partitions`,
     `process_prompt`, `postprocess`, `get_submodule`, etc.) are
     implemented for the text-only path. 12 tests in
     `test_ming_flash_omni_model.py` + the existing 30+ Ming tests
     still pass.

     **Verified via `mminf-serve` smoke**: the engine instantiates the
     model class, calls `get_submodule("Thinker")`, and reaches
     `load_thinker_weights` — failing with OOM on a single GPU
     (loaded ~75 GB before exhausting the 80 GB H100). The engine
     plumbing itself works; **single-GPU OOM is the expected blocker
     until step 3e brings TP-aware variants**. To actually serve the
     full 100B model we need TP=4 distributing the experts + attention
     across 4 H100s.

   - **3f — DONE** (graph wiring for the text-only generate loop):
     two model-side bugs blocked the first end-to-end `/generate`
     response on top of step 3e.

     (a) `BailingMoeV2ThinkerSubmodule` had no `postprocess` hook.
     The decode loop's output edge is named `text_inputs` so the
     loop feeds the previous sampled token back into the next
     iteration. `submodule.forward` returns `{"logits": [...]}`;
     the KV-cache engine samples into `{"new_token": [...]}`; but
     the graph router needs a `text_inputs` key under that name.
     Added `postprocess` that rebinds `new_token → text_inputs`,
     mirroring :meth:`OrpheusLLMSubmodule.postprocess`. Without
     this, every decode iteration hit `IndexError` at
     `prepare_inputs` (`text_inputs` list arrived empty), which
     is the same symptom the 3e notes called out.

     (b) The prefill / decode output edges used `EMPTY_DESTINATION`
     + `conductor_new_token=True` rather than
     `EMIT_TO_CLIENT` + `output_modality="text"`. With (a) fixed
     the loop produced tokens, but the API server received
     `{"outputs": {}}` because no edge routed `new_token` to the
     client. Switched to Qwen3-Omni's pattern: prefill emits its
     first token to the client and the decode-loop section emits
     each subsequent sampled token via a parallel
     `EMIT_TO_CLIENT, name="new_token", output_modality="text"`
     edge alongside the `text_inputs` loopback.

     **Environment / dependency patches collected along the way**
     (not Ming code, but required on this box to reach a working
     forward):

     * `BailingTokenizer` doesn't load under transformers >= 5.0:
       (i) accessor properties reference `self.verbose`, removed
       in 5.x — set a class-level `verbose = False`; (ii)
       `__init__` sets `self.add_bos_token` before
       `super().__init__()` and the 5.x setter calls
       `update_post_processor()` which dereferences the not-yet-
       built `self._tokenizer`. Both patches live in
       `_patch_bailing_tokenizer_for_transformers5` in
       `ming_omni_flash_model.py`, applied once after the first
       `AutoTokenizer.from_pretrained` raises an `AttributeError`
       matching either signature.

     * `LingMoeBlock._dispatch_tp` always called
       `mminf.utils.fused_moe.fused_experts`, which hard-requires
       `sgl_kernel`. On boxes where the installed `sgl_kernel.so`
       has an ABI mismatch against the running torch (the
       importlib-level error doesn't propagate as a normal
       `ImportError` until you actually call into the .so), this
       crashes mid-forward. Added a naive fallback that calls
       `dispatch_experts_fused` on each rank's expert shard then
       all-reduces; math is equivalent because sum-over-TP and
       sum-over-top-k commute.

     * `flashinfer-python` 0.6.6 ships a Python wrapper that
       passes 10 args to the bundled `top_p_sampling_from_probs`
       op while `flashinfer-jit-cache` 0.6.2 expects 8. Pin
       `flashinfer-python==0.6.2` (via `pip install --no-deps`)
       to match the jit-cache; the alternative would be rebuilding
       the cache against 0.6.6.

     **Verified via `mminf-serve` smoke (TP=8 on 8 H100s)**:
     /generate returns real model text. <details to be filled in
     by the verification curl in step 3g (benchmark wiring).>

   Note: expert layout doesn't share with Qwen3-Omni's MoE block —
   `MultiRouter` (3 gates + modality masks) is Ling-specific, and
   the per-expert fused weight tensor has its own shape constraints.

4. **Vision + audio encoders.** Stateless graph nodes. Port
   `vision_encoder.py` + `projectors.py` and `audio_encoder.py`. Wire into
   the prefill graph walks.

   - **4a — DONE** (`components/projectors.py`,
     `components/vision_encoder.py`, `components/audio_encoder.py`):
     pure-port encoder + projector modules with weight-key parity
     against the released ckpt's top-level prefixes
     (`vision.*`, `audio.*`, `linear_proj.*`, `linear_proj_audio.*`).

     * `MingVisionProjector` / `MingAudioProjector` mirror the
       `nn.Sequential` chains built inline in
       `modeling_bailingmm2.py` (Linear→GELU→Linear for vision,
       Conv1d→Transpose→GELU→Linear→Transpose for audio). Layer
       indices match the on-disk keys (`linear_proj.{0,2}` vision,
       `linear_proj_audio.{0,3}` audio).

     * `build_vision_encoder` constructs Ming's
       `Qwen3MoeVisionTransformer` via dynamic import from the staged
       Ming source dir (same path used by the tokenizer + processor).
       Reused as-is rather than forked — no vLLM dep, ~1 GB at bf16,
       runs on a single GPU.

     * `MingAudioEncoder` is a self-contained port of vllm-omni's
       packed-sequence Whisper encoder (~250 LOC) — no
       `openai-whisper` runtime dep, optional flash-attn varlen fast
       path with a manual fallback. Param names match upstream
       Whisper (`query` / `key` / `value` / `out`,
       `mlp.{0,2}.{weight,bias}`) so the released ckpt's
       `audio.blocks.N.*` keys load by state-dict equality.

     * 17 tests in `test/modular/test_ming_flash_omni_encoders.py`:
       12 pure-Python (projector shape / layer indices / forward /
       audio encoder weight-key parity / packed-attention fallback
       shape) + 1 snapshot-gated (vision encoder builds from the
       real `VisionEncoderConfig`) + 1 CUDA-gated (forward smoke
       under eager attention — currently skipped on this box for
       missing libnvrtc-builtins, not a code bug).

   - **4b — DONE** (encoder weight loading): `loader.py` now exposes
     `load_vision_encoder_weights`, `load_audio_encoder_weights`,
     `load_vision_projector_weights`, `load_audio_projector_weights`
     on top of a shared `_load_prefixed_state_dict` helper. None of
     these are TP-aware — vision + audio encoders colocate on rank 0
     in the typical topology (see `configs/ming_flash_omni.yaml`) so
     a plain prefix-strip + `load_state_dict` path suffices. The
     projector loaders also prepend `proj.` to the stripped key so
     the on-disk `linear_proj.{0,2}.*` / `linear_proj_audio.{0,3}.*`
     keys hit the `nn.Sequential` slot by integer index.

     Verified by 4 snapshot-gated tests in
     `test_ming_flash_omni_encoders.py` against the real
     `/dev/shm/ming-hybrid` ckpt — all four prefixes load strictly
     (no missing / unexpected). The audio encoder's
     `positional_embedding` is loaded as a buffer (overrides the
     sinusoidal init); the vision encoder loads all 27 blocks +
     merger + deepstack_merger_list cleanly.

5. **Thinker graph walks.** `prefill_text`, `prefill_audio`, `prefill_vision`,
   `prefill_video`, `thinker_decode`. Follow Qwen3-Omni's pattern for
   conditional walks based on `input_modalities`.

   - **5a — DONE** (`submodules.py`, `ming_omni_flash_model.py`): the two
     encoder NodeSubmodules and their construction paths.

     * `VisionEncoderSubmodule` wraps Ming's `Qwen3MoeVisionTransformer`
       + `MingVisionProjector`, mirrors
       `modeling_bailingmm2.extract_image_feature` (encoder → projector
       → L2 norm). `prepare_inputs` raises clearly on missing
       `pixel_values` / `image_grid_thw` and promotes 1-D
       `[T, H, W]` grid_thw to `(1, 3)`.

     * `AudioEncoderSubmodule` wraps `MingAudioEncoder` +
       `MingAudioProjector`. Accepts either a single `(n_mels, T)` clip
       or a `(B, n_mels, T)` batched tensor and optionally trims the
       padded tail using `audio_seqlens`. Per-clip embeddings are
       concatenated along time; L2-norm is applied when
       `audio_config.norm_query_embeds` is set (true on the released
       ckpt — matches `modeling_bailingmm2.extract_audio_feature`).

     * `get_node_engine_types` now registers
       `vision_encoder` / `audio_encoder` as `EngineType.STATELESS`
       alongside the KV-cache Thinker. Construction routes through
       new `_create_vision_encoder_submodule` /
       `_create_audio_encoder_submodule` helpers that build, dtype-cast,
       and weight-load via the loaders from step 4b.

     * 12 tests in `test/modular/test_ming_flash_omni_submodules.py`:
       10 pure-Python (input-validation, output shape, L2 norm,
       audio batched-vs-single equivalence, audio_seqlens trim,
       grid_thw promotion, node-type registration, friendly error on
       unknown node) + 2 snapshot-gated (full
       `_create_audio_encoder_submodule` on the real ckpt — verifies
       Conv1 + projector params are non-zero post-load).

   - **5b — DONE** (Thinker prefill dispatch + position helpers):
     `BailingMoeV2ThinkerSubmodule.prepare_inputs` now dispatches on
     `graph_walk` and emits either `input_ids` (text-only walks) or
     `input_embeds` + `custom_pos_ids` (multimodal walks). `preprocess`
     and `forward` route both shapes through to `LingMoeModel`'s
     existing dual input_ids/input_embeds + 1D/3D position_ids
     handling — no new model.py path needed.

     Three new position-id helpers live in `components/positions.py`,
     each producing `(3, T)` long tensors compatible with
     `LingPartialMRotaryEmbedding`'s `video_rope` branch:

     * `get_rope_index_text(seq_len, start_pos)` — three identical
       sequential rows. Matches `modeling_bailing_moe_v2.get_rope_index`'s
       pure-text branch (`:658-675`).
     * `get_rope_index_audio` — alias to the text helper (Ming
       does not special-case audio in `get_rope_index`).
     * `get_rope_index_vision(grid_thw, start_pos, spatial_merge_size,
       second_per_grid_t=None, tokens_per_second=2)` — per-image
       3D grid math from `:625-647`. Optional video timestamp
       scaling via `second_per_grid_t * tokens_per_second`.

     The Thinker dispatch:

     * `prefill` / `prefill_text` — backward-compat text path
       (unchanged from step 3f).
     * `prefill_audio` — wraps `audio_embeds` with `audio_start`
       / `audio_end` sentinel embeddings, builds text-like positions
       for the span.
     * `prefill_vision` / `prefill_video` — wraps `vision_embeds`
       with `image_start`/`image_end` (or `video_start`/`video_end`),
       builds grid-aware 3D positions; `eos` sentinel sits at
       `global_max(vision_pos) + 1` so the next walk's text positions
       can resume without collision (matches Ming source's
       `llm_pos_ids_list[-1].max() + 1` accounting).
     * `decode` / `thinker_decode` — single-token AR step (unchanged).

     Sentinel embeds are lazily computed per device on first use.
     The model.py construction now passes `config=self.config` to the
     submodule so it can read `vision.spatial_merge_size`,
     `thinker_llm.tokens_per_second`, and the `*_start_token` /
     `*_end_token` ids.

     Step 5b restricts to single-image / single-clip requests
     (multi-image splice via `Sequential` graph wiring lands in 5c).

     21 new tests across `test_ming_flash_omni_positions.py` (11) and
     `test_ming_flash_omni_submodules.py` (10): position-id shape /
     offset / abs-time math, missing-input error paths,
     multi-image rejection, sentinel embed correctness for audio /
     image / video walks, start_pos advancement, legacy `prefill`
     walk name compat. All green.

   - **5c — DONE** (graph wiring + multimodal scheduling):
     `get_graph_walk_graphs` now returns five walks instead of the
     step 3f text-only `prefill` / `decode` pair:

     * `prefill_text` — bare `Thinker` node.
     * `prefill_audio` — `Sequential([audio_encoder, Thinker])`
       where the encoder emits `audio_embeds` into the Thinker.
     * `prefill_vision` — `Sequential([vision_encoder, Thinker])`;
       `image_grid_thw` routes to BOTH the encoder (for spatial
       positions on the patches) AND the Thinker (for 3D MRoPE math
       around the vision span).
     * `prefill_video` — same shape as `prefill_vision` plus
       `video_second_per_grid` routed into the Thinker.
     * `thinker_decode` — AR loop, renamed from step 3f's `decode`.

     `get_partitions` lists all five walks under the single `Thinker`
     partition with `initial_walk="prefill_text"`. Two new helpers
     drive the scheduling:

     * `_build_thinker_prefill_schedule(input_modalities, input_signals)`
       — one schedule step per modality, in `input_modalities` order;
       each step is `(walk_name, {input_name: TensorPointerInfo})`.
       Modalities listed without matching tensors in `input_signals`
       are silently skipped (parity with qwen3_omni).
     * `_get_thinker_prefill_inputs(metadata, input_signals)` — emits
       one `GraphEdge` per input for the current step, routing each
       to the right node (encoder vs Thinker), including the dual
       `image_grid_thw` edge for vision walks.

     `get_initial_forward_pass_args` builds the schedule, picks the
     first walk, and stashes the schedule + step counter on the
     metadata. `get_partition_forward_pass_args` is the Thinker state
     machine: advance schedule → transition to `thinker_decode` →
     return `request_done=True` after the decode loop unwinds. Mirrors
     `mminf/model/qwen3_omni/qwen3_omni_model.py:765+` minus the
     Talker / Code2Wav partitions (which land in step 6+).

     Empty-schedule edge case (no usable modalities) short-circuits
     to `request_done=True` so the conductor doesn't hang.

     21 tests in `test/modular/test_ming_flash_omni_graph.py`:
     graph-walk structure (5 walks, encoder→Thinker chaining, dual
     grid_thw edge, loop feedback edge), partition listing, prefill
     schedule construction for text-only / text+audio+image / video /
     unknown-modality / no-inputs cases, edge routing for each walk
     type, full state-machine drive across a text+audio request
     (init → audio prefill → decode → done).

6. **Talker + Audio VAE.** Port `ming_flash_omni_talker.py` + `talker_module.py`
   + `audio_vae.py`. The talker is CFM-based (continuous flow matching) rather
   than discrete-codec-AR like Qwen3-Omni's — the streaming topology will
   differ. Re-read `mminf/streaming/topology.py` before wiring connections.

   Broken out into sub-steps because the upstream code is ~2,100 LOC
   across three files (`ming_flash_omni_talker.py` 586 LOC +
   `talker_module.py` 1,145 LOC + `audio_vae.py` 392 LOC):

   - **6a — DONE** (config port): replaced the step-1 raw-dict
     skeleton `TalkerConfig` with typed sub-config dataclasses so the
     modeling code (CFM head + DiT blocks + Aggregator + AudioVAE)
     can read dims off `config.talker.*` directly.

     New dataclasses in `components/config.py` (under `TalkerConfig`):
     * `TalkerLLMConfig` — Qwen2 backbone (896-dim, 24L, 14H/2KV,
       sliding-window=False, RoPE θ=1e6). Distinct from
       `ThinkerLLMConfig` (different vocab, no MoE, smaller dims).
       `head_dim` property computes 896/14=64.
     * `DiTBlockConfig` — shared shape for `flowmodel` and
       `aggregator` (depth=8, hidden_size=1024, num_heads=16,
       mlp_ratio=4, in_channels=64); only `dropout` differs (0 vs
       0.1 on the released ckpt). `head_dim` / `intermediate_size`
       properties for convenience.
     * `AudioVAEConfig` — encoder + decoder dims (latent_dim=64,
       input_dim=80, hop_size=320, output_dim=882),
       `sample_rate=44100`, `patch_size=4`. Encoder/decoder Qwen2
       backbones kept as raw dicts (`enc_backbone` /
       `dec_backbone`) for the eventual block-builder to lift.
       Discriminator + loss-weight fields retained for round-trip
       fidelity but not consumed at inference.

     `TalkerConfig.from_subdir` now constructs the typed sub-configs
     directly (was raw-dict assignment); `vae_sample_rate` /
     `vae_patch_size` retained as `@property` accessors for backward
     compat with `Model.get_output_sample_rate`.

     8 new tests in `test_ming_flash_omni_config.py` (7 freshly
     authored + 1 updated to assert the new typed shape):
     - `TalkerLLMConfig` defaults / head_dim / unknown-key filter
     - `DiTBlockConfig` intermediate_size / head_dim derivations
     - `AudioVAEConfig` enc/dec kwarg lifting + fallback when
       enc_kwargs missing latent_dim
     - `TalkerConfig.from_subdir` end-to-end with synthetic tmp dirs
       (round-trips all three sub-configs)
     - Default-factory check that `TalkerConfig()` with no args yields
       typed sub-configs

     Verified by re-running the existing snapshot-gated
     `test_subdir_configs_load_when_present` against the real
     `/dev/shm/ming-hybrid/talker/` tree — typed fields read
     correctly (LLM hidden_size=896, VAE sample_rate=44100,
     flowmodel depth=8, aggregator dropout=0.1).

   - **6b — DONE** (CFM + DiT building blocks): new
     `components/talker_dit.py` ports the modeling primitives from
     upstream `talker_module.py:1-402`. Module names mirror upstream
     so the released ckpt's `talker/model.safetensors` keys
     (`flowmodel.blocks.N.attn.to_q.weight`,
     `flowmodel.blocks.N.mlp.ff.0.0.weight` etc.) will load by
     state-dict equality once the loader path lands.

     Two external deps replaced with minimal in-tree ports:
     * `DiTTimestepEmbedding` — sinusoidal pos-emb + Linear+SiLU+Linear
       MLP, matching vllm-omni's `timestep_embedding.DiTTimestepEmbedding`.
     * `RotaryEmbedding` — non-xpos 1-D RoPE matching
       `x_transformers.RotaryEmbedding.forward_from_seq_len` exactly,
       including the INTERLEAVED-pair `rotate_half(x1, x2) = (-x2, x1)`
       layout. This is DIFFERENT from Ling-2.0 thinker's neox-cat
       layout — adjacent freq pairs share the same value here, while
       Ling's halves repeat across the split. Required so the released
       weights line up with the same RoPE shape they were trained
       against.

     The CFM module wraps the DiT and integrates an ODE/SDE step grid
     from `get_epss_timesteps` with classifier-free guidance.
     Sway-sampling-coef remap is honored (`-1.0` default packs more
     steps near `t=0`). The released ckpt's `steps=10` schedule is
     the predefined `[0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32] / 32`.

     Skipped from `talker_module.py`: `CFMGraphExecutor`,
     `CFMGraphExecutorPool` (vllm-specific batching), `Aggregator`
     (lands in 6c), the resampling / silence-trim / `build_tts_input`
     / `MingAudioGenerator` orchestration utilities (lands in 6e
     where the Talker submodule wires the streaming graph).

     New factory `build_talker_cfm(talker_config, llm_cond_dim=None,
     dtype=..., device=...)` constructs DiT + CFM directly from a
     `TalkerConfig` so 6e's `_create_talker_submodule` will be a
     one-liner. `llm_cond_dim` defaults to `talker.llm.hidden_size`
     (896 on the released ckpt).

     28 tests in `test_ming_flash_omni_talker_dit.py`:
     - RotaryEmbedding layout: rotate-half pair negation,
       freqs.shape `(1, T, dim)`, adjacent-pair-shared-frequency
       invariant, partial-rotary apply preserves passed-through tail.
     - DiTTimestepEmbedding: shape, dtype-stability, even-dim guard.
     - RMSNorm normalises to unit-rms per row.
     - FeedForward layer indices align with the ckpt's
       `ff.0.0` / `ff.0.1` / `ff.2` keys.
     - Attention: `to_q/to_k/to_v/to_out.0` param names, qk_norm
       branches, rope on/off shape preservation, unknown-qk_norm
       rejection.
     - DiTBlock + FinalLayer + CondEmbedder round-trip.
     - DiT.forward output `(B, 1 + his + patch, out_channels)` for
       no-spk and `(B, 2 + his + patch, ...)` with spk; CFG forward
       returns trailing `x.shape[1]` rows.
     - CFM.sample shape preservation + length / sde_rnd validation,
       sway=None branch.
     - `build_talker_cfm` from real `TalkerConfig` defaults yields
       the expected DiT dims (1024 hidden, 8 layers, 16 heads,
       cond_embedder input = 896) + `llm_cond_dim` override.

   - **6c — DONE** (Aggregator + Qwen2 backbone + heads):

     `_Attention` / `_DiTBlock` grew a `mask` parameter to match
     upstream API exactly. For the CFM path the caller passes
     `mask=None`, so behaviour is unchanged; the Aggregator's mask
     branch is now exercised. Mask semantics mirror upstream's
     `talker_module.Attention.forward`:
     * `attn_mask_enabled=True` builds an SDPA `attn_mask` from the
       (B, T) key-padding mask so padded keys are excluded from
       softmax.
     * Regardless of `attn_mask_enabled`, the masked-out output rows
       are zeroed via `masked_fill(~mask, 0)` — matches upstream's
       unconditional zeroing branch.

     `Aggregator` (port of `talker_module.Aggregator:702-744`): same
     DiTBlock stack as the CFM head, but the input embedder is
     `nn.Linear` (audio-latent → hidden) plus a learnable [CLS]-style
     `word_embedder` (`nn.Embedding(1, hidden_size)`) prepended to the
     sequence. The output is the `[CLS]` row only, projected through
     `final_layer` to `llm_input_dim` so the condition feedback loops
     back into the talker LLM's embedding space.

     `build_aggregator(talker_config, llm_input_dim=None, ...)` and
     `build_talker_cfm(...)` both honor `attn_mask_enabled` from the
     respective DiTBlockConfig (False on the released ckpt).

     **Talker LLM backbone** — `build_talker_llm(talker_llm_config,
     attn_implementation="sdpa", ...)` constructs a stock
     `transformers.Qwen2Model` from `TalkerLLMConfig`. No custom modeling
     path: the talker LLM colocates on a single rank in the typical
     topology and the ckpt's `talker/model.safetensors` keys are
     plain `model.*` Qwen2 keys, so reusing HF keeps the surface small
     and inherits HF's KV-cache + attention impl. Matches what the
     upstream `MingFlashOmniTalkerForConditionalGeneration.__init__`
     does (line 116: `self.model = Qwen2Model(llm_config)`).

     **Talker heads** — `build_talker_heads(talker_config,
     spk_embed_dim=192, ...)` returns a dict of two `nn.Linear` heads:
     * `stop_head` — `Linear(hidden_size, 2, bias=True)`: binary
       end-of-audio classifier consumed during the generation loop.
     * `spk_head` — `Linear(192, hidden_size, bias=True)`: projects
       a CAMPPlus speaker embedding into the LLM hidden space; the
       projected embedding is prepended to the prompt as a voice-
       condition token.

     13 new tests appended to `test_ming_flash_omni_talker_dit.py`:
     - Attention mask output-zeroing (unconditional), SDPA attn_mask
       branch (attn_mask_enabled=True), no-mask no-zeroing regression
       guard.
     - Aggregator: `[CLS]` row only output `(B, 1, llm_input_dim)`,
       single-row `word_embedder`, mask propagation through DiT
       blocks, shape stability across varying T, `build_aggregator`
       from real TalkerConfig + `llm_input_dim` override.
     - `build_talker_llm`: returns `transformers.Qwen2Model` with
       correct dims; tiny-input forward returns hidden states.
     - `build_talker_heads`: stop_head (h→2) + spk_head (192→h) with
       biases; `spk_embed_dim` override.

     Total talker_dit tests: 41 (28 from 6b + 13 from 6c). Full
     Ming step-1..7 + 6a/6b/6c suite: **162 pass / 9 skipped / 0 fail
     / 1 deselected** (deselected is pre-existing cuDNN-broken
     attention forward, unrelated).

   - **6d — DONE** (AudioVAE): new `components/audio_vae.py` ports
     `vllm_omni/.../audio_vae.py` (~392 LOC). Module tree mirrors
     upstream so the released ckpt's `talker/vae/model.safetensors`
     keys load 1:1 by state-dict equality once the loader path
     lands (6f).

     Building blocks:
     * `_ISTFT` — sliding-window OLA inverse-STFT. Two padding
       modes: `"center"` wraps `torch.istft` directly; `"same"` is
       the hand-rolled `F.fold` reconstruction with optional
       streaming buffers (carries the trailing `win_length - hop`
       samples + window envelope across chunks).
     * `_ISTFTHead` — Linear → STFT mag (exp+clip) / phase → `_ISTFT`.
     * `_StreamingLinearUpsample` — chunked linear upsampler with
       1-step lookahead so chunked output matches single-shot output
       at chunk boundaries.
     * `_Encoder` — waveform → latent params. `get_frames` windows
       the waveform with stride `hop_size`, `fc1` projects to hidden,
       Qwen2 backbone runs, then optional `aggregator` (4-layer
       Qwen2 + `cls_embed`) summarises each patch.
     * `_Decoder` — latent → waveform. `fc1` to hidden, optional
       `_StreamingLinearUpsample`, Qwen2 backbone with sliding-window
       bridge for streaming KV cache, `_ISTFTHead` to audio.
     * `AudioVAE` — wraps encoder+decoder, exposes `encode_latent`
       (with an inline `_oobleck_sample()` so we don't depend on the
       broken-on-this-box `diffusers` package) and `decode`.

     **Defaults fixed**: `AudioVAEConfig.encoder_input_dim` /
     `encoder_hop_size` were previously 80 / 320 (placeholder from
     step 6a); updated to 882 / 882 to match the released ckpt
     (`enc_kwargs: {hop_size: 882, input_dim: 882, latent_dim: 64}`).
     The existing 6a tests still pass since they explicitly pass
     overrides through `from_dict`.

     `build_audio_vae(audio_vae_config, dtype, device, attn_implementation=None)`:
     auto-picks `"sdpa"` on CPU and FA2 when available on CUDA;
     caller can pin explicitly. Mirrors vllm-omni's runtime choice
     for the talker LLM (`llm_config._attn_implementation = "sdpa"`).

     18 tests in `test_ming_flash_omni_audio_vae.py` covering:
     - Oobleck sampler shape + mean-collapse-on-small-scale.
     - ISTFT padding-mode validation + center / same forward paths.
     - StreamingLinearUpsample: single-shot path, deferred-first-chunk
       path, **chunked-vs-single-shot equivalence** (the key
       correctness property — proves boundary lookahead is wired
       correctly so chunked streaming doesn't introduce artefacts).
     - ISTFTHead output shape (audio + x_pred).
     - Encoder: `get_frames` padding arithmetic, forward without
       patching, forward with patching (aggregator path collapses
       to per-patch latents).
     - Decoder: non-streaming reconstruct shape, patching path
       routes through the upsampler.
     - AudioVAE: construction + encode_latent shape (incl. per-clip
       frame counts) + decode end-to-end.
     - **Snapshot-gated parity**: built `AudioVAE.state_dict()` keys
       contain all representative entries present in
       `talker/vae/model.safetensors` (fc1/fc2/fc3/norm/cls_embed,
       encoder.encoder, encoder.aggregator, decoder.fc1,
       decoder.head.out, decoder.head.istft.window, decoder.decoder)
       and vice versa — proves the eventual loader will be a clean
       prefix-strip + load_state_dict.

   - **6e — IN PROGRESS** (Talker submodule + graph walks): split into
     6e-1 (orchestration helper) + 6e-2 (mminf graph wiring).

     - **6e-1 — DONE** (`components/talker_generator.py`): port of
       upstream `MingAudioGenerator` (talker_module.py:854-1146) plus
       the streaming-decode utilities `silence_holder` /
       `trim_trailing_silence`. Stateless-per-request `TalkerGenerator`
       binds Qwen2 LLM + CFM + Aggregator + stop_head + AudioVAE and
       exposes:
       * `generate_latents(inputs_embeds, ...)` — the AR loop:
         repeated (`llm_step` → `cfm_sample_step` → stop check). Each
         step emits one `(B, patch_size, latent_dim)` latent; the
         Aggregator output becomes the next step's `inputs_embeds`;
         the stop_head softmax gates early termination after
         `min_new_token` steps.
       * `cfm_sample_step` — one CFM substep-integration + Aggregator
         + stop classification.
       * `llm_step` — single Qwen2 forward with `StaticCache`
         `cache_position` bookkeeping on step > 0.
       * `decode_to_waveform(latents, stream_decode=True)` — one-shot
         or chunked AudioVAE decode; the streaming path threads
         `silence_holder` + a sliding `decode_pad` window across chunks.
       * `duration_capped_steps` — the text-length → max-steps prosody
         heuristic.
       * `_init_his_lat` / `_update_his_lat` — history-latent sliding
         window (right-aligns a voice-prompt latent when supplied).

       Skipped from upstream: `CFMGraphExecutorPool` / `CFMGraphExecutor`
       (vllm CUDA-graph batching — mminf's engine handles capture);
       `build_tts_input` / `_looks_like_music_prompt` (→ step 8).

       24 tests in `test_ming_flash_omni_talker_generator.py`:
       trim_trailing_silence (empty / short-clip / silent-tail trim /
       weird-shape passthrough), silence_holder (cache init, sub-frame
       buffering until last_chunk), generator construction (with /
       without VAE), his-lat zeros + right-align + window update +
       unsupported-shape guard, cfm_sample_step output shapes +
       stop-softmax-sums-to-1, llm_step step-0 path, generate_latents
       per-step collection + max_steps cap, duration_capped_steps
       heuristic, decode_to_waveform one-shot / streaming / empty /
       no-VAE-raises, instance trim_trailing_silence.

     - **6e-2 — DONE** (TalkerSubmodule + construction + node
       registration): the talker is a STATELESS node, not an AR /
       streaming-codec node. Ming's thinker→talker bridge passes
       DETOKENIZED TEXT (the talker re-encodes with its own
       `talker/llm` tokenizer — see vllm-omni `pipeline.py`'s
       `thinker2talker`), and the CFM step count is stop_head-
       determined rather than a conductor decode loop. So the whole
       per-request generation (LLM prefill + CFM AR decode + AudioVAE
       decode) runs inside one `TalkerSubmodule.forward` call.

       * `TalkerSubmodule` (`submodules.py`): `prepare_inputs` embeds
         `talker_text_inputs` token ids via the talker LLM's
         `embed_tokens`; `forward` runs `generate_latents` →
         `decode_to_waveform` → `trim_trailing_silence` and returns
         `{"audio_chunk": [waveform]}` (`(1, 1, num_samples)` at the
         VAE sample rate). `get_stateless_flavor` returns
         `"audio_codec"` (no autocast / no torch.compile — the CFM
         ODE loop + ISTFT are numerically sensitive).

       * `get_node_engine_types` registers `Talker` as
         `EngineType.STATELESS` when the snapshot ships a `talker/`
         subdir; thinker-only configs omit it.

       * `_create_talker_submodule` builds the full stack
         (`build_talker_llm` + `build_talker_cfm` + `build_aggregator`
         + `build_talker_heads` + `build_audio_vae`), loads every
         subtree via the step-6f loaders, wraps in a
         `TalkerGenerator` → `TalkerSubmodule`.

       12 tests across `test_ming_flash_omni_talker_submodule.py` (9)
       + an updated `test_get_submodule_rejects_unknown_node`:
       stateless flavor, prepare_inputs embed (1-D + 2-D ids) +
       missing-input guard, forward returns finite audio_chunk,
       node-type registration (with / without talker config),
       `_create_talker_submodule` no-talker guard, plus a
       snapshot-gated end-to-end that builds the full talker from
       real weights and generates a finite waveform.

     - **6e-3 — DONE** (graph walks + Thinker→Talker bridge): the
       talker is now a second partition wired off the Thinker, gated
       entirely on `config.talker is not None` (thinker-only configs
       are byte-for-byte unchanged from step 5c).

       Graph + partition additions (all in `ming_omni_flash_model.py`):
       * `get_graph_walk_graphs` adds a `talker` walk — a single
         `Talker` node consuming `thinker_tokens`, emitting one
         `audio_chunk` `EMIT_TO_CLIENT` edge. The `thinker_decode`
         loop gains a `StreamingGraphEdge(name="thinker_tokens",
         target_partition="Talker")` so each decoded token streams to
         the talker.
       * `get_partition_topology` declares the Thinker→Talker
         `Connection` with a `FixedChunkPolicy(chunk_size=1,
         continue_after_done=True)` — the talker needs the FULL text
         before it generates, so the policy keeps the consumer alive
         past the Thinker's text EOS.
       * `get_partitions` adds the `Talker` partition
         (`producer_partitions=["Thinker"]`, `initial_walk=None`).
       * `get_output_sample_rate("audio")` returns the talker VAE
         sample rate (44.1 kHz).
       * `get_initial_forward_pass_args` / `get_partition_forward_pass_args`
         dispatch a Talker branch: `_get_talker_forward` waits for
         `producer_done`, then fires the single `talker` walk once and
         reports `request_done` on the next invocation.

       Thinker→Talker text bridge: Ming passes DETOKENIZED TEXT, not
       hidden states. `thinker_text_to_talker_inputs` decodes the
       thinker output ids with the thinker tokenizer and re-encodes
       with the talker's own `talker/llm` tokenizer (loaded lazily +
       cached via `_get_talker_tokenizer`). `_create_talker_submodule`
       injects this as the `TalkerSubmodule.text_bridge`, and
       `TalkerSubmodule.prepare_inputs` accepts either pre-bridged
       `talker_text_inputs` or raw `thinker_tokens` (running the
       bridge in the latter case).

       18 tests in `test_ming_flash_omni_talker_graph.py`: thinker-only
       path unchanged (no talker walk / partition / streaming edge),
       talker-enabled graph structure (walk, audio edge, streaming
       edge to Talker), partition + topology + chunk-policy
       continue-after-done, node-type registration, audio sample rate,
       Talker state machine (waits for producer_done, fires once,
       then done; audio-output gating), and the text bridge
       (decode→re-encode round-trip + missing-tokenizer guard).
       Updated two pre-existing tests that asserted Talker was an
       unknown node/partition.

     **Step 6 complete** — audio-out `/generate` is now wireable
     end-to-end at the model layer (live bring-up still blocked by the
     TP=4 OOM on the 4-GPU dev box; needs TP=8 thinker + talker on a
     spare rank).

   - **6f — DONE** (weight loaders): `loader.py` exposes five new
     entry points on top of the step-4b `_load_prefixed_state_dict`
     helper. The helper grew two args: `subdir` (relative to
     `local_dir` — lets us point at `talker/` or `talker/vae/` instead
     of the snapshot root) and `allow_unexpected` (set of post-rename
     keys allowed to appear in the ckpt without a target module slot).

     Five loaders:
     * `load_talker_llm_weights` — strips `model.` from
       `talker/model.safetensors` for a `transformers.Qwen2Model`.
     * `load_talker_cfm_weights` — strips `cfm.` for a `CFM(DiT)`.
       Allows the ckpt's `model.rotary_embed.inv_freq` (we register
       it as `persistent=False` and recompute locally — deterministic
       from head_dim + rope_theta).
     * `load_talker_aggregator_weights` — strips `aggregator.` for
       an `Aggregator`. Same `rotary_embed.inv_freq` allow.
     * `load_talker_heads_weights` — loads `stop_head.*` +
       `spk_head.*` into the dict produced by `build_talker_heads`.
     * `load_talker_audio_vae_weights` — empty-prefix load from
       `talker/vae/model.safetensors` (the ckpt's `encoder.*` /
       `decoder.*` are top-level siblings with no shared prefix —
       no strip needed).

     7 snapshot-gated tests in `test_ming_flash_omni_talker_loader.py`
     verify strict load against `/dev/shm/ming-hybrid/talker/`:
     - Talker LLM: representative key parity + non-zero embed table
       after load.
     - CFM: `model.x_embedder.weight` / `model.blocks.0.attn.to_q.weight`
       / `model.blocks.0.mlp.ff.0.0.weight` / `model.final_layer.linear.weight`.
     - Aggregator: `x_embedder` / `word_embedder` / `blocks.0.attn.to_q`
       / `final_layer.linear`.
     - Heads: `stop_head` + `spk_head` weights both load; non-zero
       post-load; missing-key guard fires before disk I/O.
     - AudioVAE: full encoder + decoder + aggregator + ISTFT window
       keys loaded; CPU end-to-end decode on a real-weights latent
       produces a finite waveform (catches catastrophic
       dtype/layout misloads that key-name parity alone wouldn't
       surface).

     Full Ming step-1..7 + 6a/6b/6c/6d/6f suite: 187 pass / 9 skipped
     / 0 fail / 1 deselected.

7. **Process_prompt — DONE.** `MingFlashOmniModel.process_prompt` now
   produces the full `NameToTensorList` consumed by step 5c's prefill
   scheduler. Strategy mirrors `qwen3_omni`'s `process_prompt`: apply
   the chat template to TEXT-ONLY messages (so the tokenizer doesn't
   insert placeholder tokens we'd later have to strip), then run the
   image / video / audio sub-processors separately for each modality.
   The Ming chat template path uses `tokenizer.apply_chat_template`
   (jinja, accepts OpenAI roles `user`/`assistant`/`system`) rather
   than `processor.apply_chat_template` (Python implementation in
   `BailingMM2Processor`, asserts on lowercase OpenAI roles — see
   "Role-handling nuance" above).

   Input convention (`tensors: NameToTensorList`):
     * `image_inputs` — list of CHW float [0,1] tensors per image.
       Internal `_image_to_processor_input` converts to HWC uint8 to
       avoid the upstream's double-rescale near-zero bug
       (`qwen3_omni_model.py:1033-1038` documents the same gotcha).
       Single-channel inputs auto-broadcast to 3 channels.
     * `audio_inputs` — list of either raw 1-D float tensors (sample
       rate inferred from processor default 16 kHz) or
       `(waveform, sample_rate)` tuples.
     * `video_inputs` — list of (T, C, H, W) float tensors. Per-frame
       `second_per_grid` defaults to 1.0; override via
       `kwargs["input_metadata"]["video"][i]["second_per_grid"]`.

   Output keys consumed by `_build_thinker_prefill_schedule`:
     * `text_inputs` — list of 1-D long tensors (one per text turn).
     * `pixel_values`, `image_grid_thw` — one entry per image.
     * `pixel_values_videos`, `video_grid_thw`,
       `video_second_per_grid` — one entry per video clip.
     * `audio_features` (n_mels, T) + `audio_seqlens` (length-1 long)
       — one entry per audio clip. Note: upstream returns audio_feats
       as (B, T, n_mels); we transpose to (n_mels, T) per clip so
       `AudioEncoderSubmodule.prepare_inputs` can splice without a
       reshape.

   17 tests in `test/modular/test_ming_flash_omni_process_prompt.py`:
   text-only happy path, no-prompt audio-only path, image conversion
   correctness (CHW float [0,1] → HWC uint8, grayscale broadcast,
   uint8 pass-through), per-modality dispatch, missing-processor
   error paths, multi-image / mixed-modality combinations, video
   metadata override, snapshot-gated text+image E2E with the real
   `BailingMM2Processor`. 16 green + 1 env-skip on this box.

   Image-gen-specific `<image><imagePatch>*256</image>` block (the
   query-token expansion for the imagegen DiT path) is deferred to
   step 9 (ImageGen partition), since today's prefill schedule only
   covers text-out generation.

8. **TTS caption template (optional, talker-only deploy).** Port
   `prompt_utils.BASE_CAPTION_TEMPLATE` + `create_instruction` so the
   `ming_flash_omni_tts` deploy variant accepts the same JSON caption shape
   that vllm-omni speaks.

9. **ImageGen partition (deferred).** Separate from the omni pipeline; lives
   under vllm-omni's diffusion tree. Wire as a fourth partition with its own
   graph walk once #1–8 are landed. Needs `FlowEngine`-style integration.

10. **Configs.** Update `configs/ming_flash_omni*.yaml` to match the final
    node names emerging from #5 and #6. Add an image-gen variant when #9
    lands.

11. **Benchmark `OursOpenAI` parity.** Once `mminf-serve` boots the model,
    extend `benchmark/request.py:OursOpenAI` to route Ming TTS through the
    correct endpoint (likely `/v1/chat/completions` with `modalities=["audio"]`,
    matching the Qwen3-Omni path — `MingFlashOmni` declares no Orpheus-style
    speech-only fallback).

12. **Tests.** Add `test/modular/test_ming_flash_omni_*.py` covering config
    load, submodule weight load on a tiny shard, and a smoke graph walk on
    a single GPU. Mirror `test/modular/test_qwen3_omni_*.py` if present.

## Things to verify against the released checkpoint (not in vllm-omni)

- Exact `max_position_embeddings` and `rope_theta` for thinker vs talker
  (read from `config.json`, not the deploy yaml).
- Whether `default_sampling_params.repetition_penalty=1.05` from the deploy
  yaml is a serving default or a hard requirement — affects
  `benchmark/base.py:MingFlashOmni.get_model_kwargs`.
- The output sample rate for the talker (Qwen3-Omni is 24 kHz; check
  `audio_vae.py` for Ming's). Override
  `Model.get_output_sample_rate` if it differs.
