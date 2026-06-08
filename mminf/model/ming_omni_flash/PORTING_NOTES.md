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

     **Known gap (not blocking 3e commit)**: first text request to
     `/generate` hits `IndexError` in
     `BailingMoeV2ThinkerSubmodule.prepare_inputs` — the per-request
     `text_inputs` list arrives empty. This is an integration bug
     between `get_initial_forward_pass_args` / graph-walk wiring /
     the conductor's prompt-to-input-signals routing (NOT a model
     code bug — all the heavy machinery loaded and warmed up cleanly).
     Likely fix: either change the graph node's `input_names` /
     ckpt edge-naming or add a fallback in `prepare_inputs` that
     pulls the prompt tokens from `fwd_info` when the input list is
     empty. Standalone follow-up.

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

   - **3e — TODO**: TP-aware variants (`ParallelAttention` replacement
     of `nn.Linear` QKV, `ParallelMoeBlock` for routed experts,
     TP-rank-aware weight loader slicing per-expert tensors per rank).
     Then `mminf-serve --config configs/ming_flash_omni_thinker_only.yaml`
     with TP=4 should actually answer a text request.

   Note: expert layout doesn't share with Qwen3-Omni's MoE block —
   `MultiRouter` (3 gates + modality masks) is Ling-specific, and
   the per-expert fused weight tensor has its own shape constraints.

4. **Vision + audio encoders.** Stateless graph nodes. Port
   `vision_encoder.py` + `projectors.py` and `audio_encoder.py`. Wire into
   the prefill graph walks.

5. **Thinker graph walks.** `prefill_text`, `prefill_audio`, `prefill_vision`,
   `prefill_video`, `thinker_decode`. Follow Qwen3-Omni's pattern for
   conditional walks based on `input_modalities`.

6. **Talker + Audio VAE.** Port `ming_flash_omni_talker.py` + `talker_module.py`
   + `audio_vae.py`. The talker is CFM-based (continuous flow matching) rather
   than discrete-codec-AR like Qwen3-Omni's — the streaming topology will
   differ. Re-read `mminf/streaming/topology.py` before wiring connections.

7. **Process_prompt.** Build the ChatML-ish prompt via the processor's
   `apply_chat_template(messages, sys_prompt_exp=None, use_cot_system_prompt=False)`.
   For image-gen requests append the `<image><imagePatch>*256</image>`
   query-token block (see `prompt_utils.maybe_expand_image_gen_prompt`).

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
