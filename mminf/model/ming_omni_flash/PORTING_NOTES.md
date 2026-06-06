# Ming-flash-omni-2.0 — porting notes

Native mminf port of `inclusionAI/Ming-flash-omni-2.0`. This directory is a
scaffold today; everything below is the punch list to make it real.

## Status

- `benchmark/base.py` has `MingFlashOmni` + `ModelType.MING_FLASH_OMNI`.
  Benchmarking against a vllm-omni server **works today** with
  `--inference-system vllm_omni` (see `benchmark/vllm_omni_instructions.md`).
- `mminf/model/ming_omni_flash/` ships only the file/class shape.
  `MingFlashOmniModel.__init__` and every abstractmethod raise
  `NotImplementedError`. `mminf-serve --config configs/ming_flash_omni.yaml`
  will fail at startup until the work below is done.

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

1. **Config port.** Fill `config.py` by mirroring vllm-omni's
   `MingFlashOmniConfig` field tree. Add `from_pretrained` that reads
   `config.json` from the HF snapshot. Verify by loading the released
   checkpoint and printing key dims.

2. **Tokenizer + processor.** In `MingFlashOmniModel.__init__`, load the
   HF `AutoTokenizer` + `AutoProcessor` from the snapshot with
   `trust_remote_code=True`. Chat-template role map is `user→HUMAN`,
   `assistant→ASSISTANT`, `system→SYSTEM` (uppercase internally); the HF
   processor handles this — the wire-level OpenAI shape is unchanged.

3. **Submodules (one per node) — start with the Thinker.** Define
   `submodules.py` registering each `NodeSubmodule` and a weight loader.
   Port the Ling-2.0 MoE backbone (`modeling_bailing_moe_v2.py`) first;
   it's the largest single chunk and unblocks everything else. Don't try to
   share with Qwen3-Omni's MoE block — expert layout differs.

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
