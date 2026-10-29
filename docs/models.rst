Supported Models
================

``mstar`` ships the following model families. The table below summarizes the registered
families, their registry key (the value of ``model:`` in a config YAML), and a
representative Hugging Face identifier.

Registry keys live in ``mstar/model/registry.py`` (``MODEL_REGISTRY`` / ``HF_MODELS``).

.. list-table:: Registered model families
   :header-rows: 1
   :widths: 14 34 30

   * - Registry key
     - Example Hugging Face model ID
     - Description
   * - ``bagel``
     - ``ByteDance-Seed/BAGEL-7B-MoT``
     - Unified multimodal model (text + image understanding and generation).
   * - ``chatterbox`` / ``chatterbox_multilingual`` / ``chatterbox_turbo``
     - ``ResembleAI/chatterbox``, ``ResembleAI/chatterbox-turbo``
     - Zero-shot voice-cloning TTS: T3 speech-token LM (Llama-520M, or GPT-2-medium
       for Turbo) with CFG and exaggeration control, S3Gen flow-matching decoder,
       HiFT vocoder, PerTh watermark. 24 kHz. Multilingual covers 23 languages.
   * - ``command_a_plus`` *(Experimental)*
     - ``CohereLabs/command-a-plus-05-2026-bf16``
     - Command A+ text and image chat: a 218B-parameter MoE (25B active) with
       sliding-window attention. Tested on 4x GB200 at TP4; see the notes below.
   * - ``cosmos3``
     - ``nvidia/Cosmos3-Nano``
     - Cosmos3 world model: t2i/t2v/i2v/v2v diffusion, robot-action modes, opt-in sound.
   * - ``cosmos3_droid``
     - ``nvidia/Cosmos3-Nano-Policy-DROID``
     - Cosmos3 action-policy fine-tune for the DROID platform (``domain_name``
       ``droid_lerobot``, 10-dim raw actions); no sound pathway. The config
       serves the released policy sampling defaults (4 steps, guidance 3.0).
   * - ``cosmos3_edge``
     - ``nvidia/Cosmos3-Edge``
     - Cosmos3-Edge (4B): dense Nemotron backbone, 480p-native t2i/t2v/i2v and
       robot-action modes, plus the reasoner (image/video chat through
       ``/v1/chat/completions``) on the shared understanding tower.
   * - ``cosmos3_edge_droid``
     - ``nvidia/Cosmos3-Edge-Policy-DROID``
     - Edge action-policy fine-tune for DROID (``domain_name``
       ``droid_lerobot``); serves the released 4-step, guidance-3.0 policy
       defaults.
   * - ``cosmos3_super``
     - ``nvidia/Cosmos3-Super``
     - Cosmos3-Super (64B) variant of the above; TP/SP for multi-GPU serving.
   * - ``cosmos3_super_t2i_4step`` / ``cosmos3_super_i2v_4step``
     - ``nvidia/Cosmos3-Super-Text2Image-4Step`` / ``…-Image2Video-4Step``
     - 4-step distilled Super task checkpoints (guidance baked in, fixed-sigma
       stochastic sampler); TP=2 deployments.
   * - ``gr00t_n17`` *(Beta)*
     - ``nvidia/GR00T-N1.7-3B``
     - Vision-language-action robotics model: a Qwen3-VL backbone cut to 16 layers
       with DeepStack, and a 32-block flow-matching action head that denoises a
       40-step action chunk in four steps. One request is a language instruction
       plus the robot's camera views and its proprioceptive state; the reply is
       msgpack, one entry per joint group, each with its ``dtype``, ``shape`` and
       raw ``data``. Request knobs: ``embodiment_tag`` (the checkpoint's own
       spelling, matched case-insensitively — eight are shipped, and the error
       lists them), ``state`` as ``{joint_group: float32 (1, 1, D)}`` (one
       request, one timestep: the checkpoint's ``state_history_length``) with the
       embodiment's own joint-group names, and ``walk``, which selects one of the
       two reuse walks (``backbone`` or ``state``) instead of running the full
       path. A reuse walk answers ``output_modalities=("tensor",)`` with its
       features (``backbone_features``, ``state_features``) as one msgpack entry
       laid out like an action group. **Camera frames are attached as images,
       not passed in ``model_kwargs``**: one lossless image per camera view per
       timestep, in time-major order over the embodiment's view order — ``(t0,v0)``,
       ``(t0,v1)``, ``(t1,v0)``, … — so a two-camera robot sends four. The format
       must be **lossless** (PNG or lossless WebP): the pixels are compared against
       the reference's bit for bit, so JPEG moves them 167x further than PNG while
       passing every other check. JPEG is refused; a lossy WebP is not detected
       (Pillow reports one format for both WebP modes), so send WebP lossless. The
       backbone's config **and its tokenizer and image
       processor** come from the **gated** ``nvidia/Cosmos-Reason2-2B``; accept its
       terms on the Hub first. Needs the ``gr00t_n17`` extra: OpenCV for the
       ``INTER_AREA`` resize the reference's image transform uses, ``diffusers``
       for the action head's feed-forward, and the Hugging Face loaders.
   * - ``kokoro``
     - ``hexgrad/Kokoro-82M``
     - TTS (82M, not autoregressive): misaki G2P + PL-BERT prosody + iSTFTNet
       decoder, 54 bundled voices and voice blends, sentence-chunked streaming,
       batched across requests.
   * - ``nemotron_duplex`` *(Beta)*
     - ``nvidia/NVIDIA-NemotronLabs-VoiceChat-11B``
     - Full-duplex speech-to-speech (NemotronLabs VoiceChat-11B): Fast-Conformer
       encoder + Nemotron-H hybrid Mamba-2/attention backbone + EarTTS talker +
       RVQ codec; user speech in, agent text and 22.05 kHz speech out.
   * - ``glm52``
     - ``zai-org/GLM-5.2-FP8``
     - GLM-5.2 (753B/40B): MLA + DSA MoE causal LM, fp8 block-scale weights, TP8.
   * - ``orpheus``
     - ``canopylabs/orpheus-3b-0.1-ft``
     - TTS: Llama 3.2 3B LLM emitting audio tokens + SNAC 24 kHz decoder.
   * - ``pi05``
     - ``lerobot/pi05_base``
     - Pi0.5 vision-language-action robotics model (ViT encoder + LLM + flow action expert).
   * - ``qwen3_5_{0.8,2,4,9,27}b``
     - ``Qwen/Qwen3.5-4B``
     - Hybrid-attention VLM (text + image in, text out): gated DeltaNet linear
       attention interleaved with full attention, plus a ViT tower.
   * - ``qwen3_8_27b``
     - ``Qwen/Qwen3.8-27B``
     - Qwen3.8-27B on the Qwen3.5 code above (same architecture); ships a TP4 config.
   * - ``omnivoice``
     - ``k2-fsa/OmniVoice``
     - Massively multilingual zero-shot TTS: masked-diffusion canvas over a Qwen3-0.6B
       backbone + audio codec. Clones a voice from a reference clip.
   * - ``qwen3_omni``
     - ``Qwen/Qwen3-Omni-30B-A3B-Instruct``
     - Omni-modal (text/image/audio/video in, text/audio out): Thinker + Talker + codec.
   * - ``qwen3_tts``
     - ``Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice``
     - Streaming text-to-speech with built-in speakers: Talker + 12 Hz speech codec.
   * - ``qwen3_tts_1p7b``
     - ``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice``
     - 1.7B CustomVoice: built-in speakers plus style/emotion ``instruct`` control.
   * - ``qwen3_tts_voicedesign``
     - ``Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign``
     - 1.7B VoiceDesign: the voice is described by the request's ``instruct`` text.
   * - ``qwen3_tts_base``
     - ``Qwen/Qwen3-TTS-12Hz-1.7B-Base``
     - 1.7B Base: zero-shot voice clone from one reference clip (x-vector, optional in-context transcript).
   * - ``fish_s2_pro``
     - ``fishaudio/s2-pro``
     - Fish Audio S2-Pro: dual-AR TTS (Qwen3-4B Slow AR + 4-layer Fast AR over 10 codebooks),
       44.1 kHz causal DAC codec, voice cloning from one reference clip, inline ``[tag]``
       prosody control. Weights under the Fish Audio Research License (non-commercial use
       only); check it before deploying.
   * - ``vjepa2``
     - ``facebook/vjepa2-vitl-fpc64-256``
     - V-JEPA 2 video encoder + masked predictor.
   * - ``vjepa2_ac``
     - ``vjepa2-ac-vitg``
     - V-JEPA 2-AC encoder + action-conditioned predictor. Video frames take the official
       repo's central square crop, not the HF processor's resize and crop.
   * - ``whisper_large_v3_turbo``
     - ``openai/whisper-large-v3-turbo``
     - Whisper ASR: batched, CUDA-graph-captured audio encoder + paged-KV
       decoder with write-once cross-attention; language detection, forced
       language/task, ``initial_prompt`` context, ``/v1/audio/transcriptions``.
       The 4-decoder-layer distillation of large-v3, same class as below.
   * - ``whisper_large``
     - ``openai/whisper-large-v3``
     - Whisper large-v3 (32 decoder layers) on the same scaffold.
   * - ``qwen3_asr``
     - ``Qwen/Qwen3-ASR-1.7B``
     - Qwen3-ASR: AuT audio encoder (8 s attention windows, ragged attention)
       + dense Qwen3 decoder; up to 20 min of audio per request, 30 languages
       (detected or forced), ``/v1/audio/transcriptions``. A repetition loop
       in the transcript is collapsed as the reference SDK does.
   * - ``qwen3_asr_realtime``
     - ``Qwen/Qwen3-ASR-0.6B``
     - The 0.6B Qwen3-ASR checkpoint on the same class, sized in its config for the
       short steps of ``/v1/realtime`` sessions.
   * - ``higgs_audio`` *(Beta)*
     - ``bosonai/higgs-audio-v3-stt``
     - Audio-tower + Qwen3 LLM speech-to-text. Beta / un-optimized.
   * - ``omnilingual_asr_ctc_{300m,1b,3b,7b}``
     - ``facebook/omniASR-CTC-300M`` (v1); v2 from Meta's CDN (default)
     - Meta Omnilingual ASR, CTC variants: wav2vec 2.0 encoder + CTC head, speech-to-text
       in 1,600+ languages with no language code. Not autoregressive; batched across
       requests. Apache-2.0.
   * - ``canary``
     - ``nvidia/canary-1b-v2``
     - FastConformer encoder + Transformer decoder: speech recognition in 25 European
       languages and translation to or from English (``language``, English if unset, and
       ``target_language`` or ``task="translate"``). Clips up to 40 s; longer uploads are
       windowed. No timestamps.
   * - ``canary_qwen``
     - ``nvidia/canary-qwen-2.5b``
     - English speech recognition by an LLM: Canary's FastConformer encoder projected into
       Qwen3-1.7B (LoRA merged at load). Clips up to 40 s; longer uploads are windowed. The
       OpenAI ``prompt`` is ignored. No timestamps.
   * - ``parakeet``
     - ``nvidia/parakeet-tdt-0.6b-v3``
     - FastConformer + token-and-duration transducer, 25 European languages, language detected
       from the speech. Clips up to 10 min; longer uploads are windowed. No timestamps.
   * - ``nemotron_asr``
     - ``nvidia/nemotron-3.5-asr-streaming-0.6b``
     - Cache-aware streaming FastConformer + RNN-T with a language prompt (``language``: a locale
       such as ``en-US``, a bare code, or ``auto``); served over whole clips, with the streaming
       model's chunked attention at its widest lookahead (``lookahead`` in the YAML's
       ``model_kwargs``). Clips up to 10 min; longer uploads are windowed. No timestamps.
   * - ``wan22``
     - ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``
     - Wan2.2-TI2V-5B video diffusion: text-to-video and image-to-video, 5B dense DiT.
   * - ``waypoint``
     - ``Overworld/Waypoint-1.5-1B``
     - Waypoint-1.5 interactive video world model: 1.28B DiT generating one latent
       frame per step, TAEHV encode/decode. Live RGB frame output; no OpenAI surface.
   * - ``lingbot_world_1_3b``
     - ``robbyant/lingbot-world-v2-1.3b-causal-fast``
     - LingBot-World-V2 1.3B causal-fast: camera-controlled image-to-world video, streamed
       chunk by chunk (Wan2.1-layout causal DiT, rolling KV cache, 4 distilled steps).
       A request carries the whole camera trajectory as JSON in ``model_kwargs``, which the API
       server caps at 1 MiB: about 3,900 frames of ``[4, 4]`` float32 poses; send ``[F, 3, 4]``
       rows for up to the 4,077-frame maximum. One rollout per GPU. The DiT's attention runs
       through the engine's ragged attention resource, its blocks are compiled (Inductor's
       deterministic mode), the condition encode runs channels-last, and the VAE decode is
       channels-last, compiled and bf16 where cuDNN >= 9.16 (frames 51 dB from the fp32 decode).
       ``model_kwargs: {reference_numerics: true}`` runs the reference's numerics instead
       (FlashAttention-2, needs ``flash-attn``; an eager DiT; the fp32 contiguous encode and
       decode), with which a rollout is bit-identical to the reference's.
       ``lingbot_world_1_3b_fast.yaml`` compiles the DiT with autotuned kernels
       (``dit_compile_autotune``): ~5% faster, but the rollout for a seed then differs from one
       server process to the next.
       The weights are CC-BY-NC-SA-4.0 (non-commercial, share-alike), not Apache-2.0; check it
       before deploying. The code is Apache-2.0, adapted from LingBot-World v1 (``Robbyant/lingbot-world``).
   * - ``flux2_klein``
     - ``black-forest-labs/FLUX.2-klein-4B``
     - FLUX.2 [klein] 4B: step-distilled (4 steps, no CFG) text-to-image and multi-reference
       image editing; Qwen3-4B hidden-state text encoder + FLUX.2 VAE. Apache-2.0.
   * - ``flux2_klein_9b``
     - ``black-forest-labs/FLUX.2-klein-9B``
     - FLUX.2 [klein] 9B (Qwen3-8B encoder, 4096-wide DiT), same class. Released under
       the FLUX Non-Commercial License; check it before deploying.
   * - ``ltx2_5`` *(Beta)*
     - ``Lightricks/LTX-2.5-Diffusers``
     - LTX-2.5 (22B) text-to-audio+video: the distilled 8-step recipe, no guidance; native
       joint audio-video DiT and Gemma-4 text encoder, emitting an H.264 video and 48 kHz
       stereo audio. Gated; LTX-2.x Community License.
   * - ``z_image_turbo``
     - ``Tongyi-MAI/Z-Image-Turbo``
     - Z-Image-Turbo: 8-step distilled single-stream flow DiT (6B) with a Qwen3-4B caption
       encoder and the FLUX.1 VAE; text-to-image, no CFG. Apache-2.0. Same DiT scaffold as klein.

Notes
-----

- Models marked *(Beta)* are functionally supported but not yet
  performance-optimized; treat their throughput/latency as provisional.
- The IDs above are representative. You may use local paths or compatible variants.
- Some families accept multimodal input (image/audio/video); see the model's
  ``process_prompt`` for the inputs it expects.
- Models on the fused MoE kernel (Qwen3-Omni, Command A+) use a tuned tile table
  from ``mstar/utils/fused_moe/configs/`` when one matches the expert shape and
  GPU, and default tiles otherwise. ``python -m mstar.utils.fused_moe.tune``
  writes a table for the local GPU.
- To add a new family, see :doc:`adding_models`.

OmniVoice notes
~~~~~~~~~~~~~~~

- Zero-shot only: there are no built-in speakers. Pass ``reference_audio`` with its
  transcript in ``ref_text`` to clone a voice, or describe one in ``voice``.
  ``ref_text`` is required alongside ``reference_audio``.
- ``language`` takes either the name (``Vietnamese``) or the id (``vi``): a
  name is resolved to the id the model was trained on before the prompt is
  built, and an unrecognised value warns and falls back to language-agnostic
  mode.
- The ``omnivoice`` package is installed separately from git rather than by an
  extra — see :doc:`installation`.
- The backbone is not autoregressive: it fills a fixed canvas of eight codebook
  rows over a few unmasking steps, so there is no KV cache and no per-token
  sampling loop. Serve it with ``mstar serve omnivoice --gpus 0``.

Omnilingual ASR notes
~~~~~~~~~~~~~~~~~~~~~

- Serve with ``mstar serve omnilingual_asr_ctc_300m`` (or ``_1b``, ``_3b``, ``_7b``) and
  transcribe through ``POST /v1/audio/transcriptions``. No extra is needed;
  ``pip install -e '.[audio]'`` adds non-WAV containers (mp3, flac, ...).
- The model is a port of Meta's `Omnilingual ASR <https://github.com/facebookresearch/omnilingual-asr>`_
  CTC checkpoints (code and weights Apache-2.0) that loads Meta's own ``.pt`` files without
  fairseq2. ``weights_version`` in the YAML ``model_kwargs`` picks the release: ``v2``
  (default; lower CER per Meta; its vocabulary adds capitals and punctuation, though on our
  FLEURS clips it still wrote lowercase, unpunctuated text) is only on Meta's
  CDN (``dl.fbaipublicfiles.com/mms/``), which publishes no checksums, so the files are
  checked against sha256 values pinned in ``mstar/model/omnilingual_asr/config.py``
  (computed from our downloads); ``v1`` comes from the ``facebook/omniASR-CTC-*`` Hub repos
  at pinned revisions. ``checkpoint_path`` / ``tokenizer_path`` take local files instead.
- Preprocessing follows the reference exactly: decode at the file's rate, resample each
  channel to 16 kHz (torchaudio), average the channels, normalize the utterance to zero
  mean and unit variance. ``/v1/audio/transcriptions`` transcribes a clip of up to 20 s
  whole and cuts a longer upload into the fewest windows of at most 16 s, of about equal
  length, each ending at the quietest 0.1 s within 5 s of its boundary, transcribed in
  parallel. The model hears up to 40 s (the reference's limit, and the cap for
  ``window_seconds`` in ``extra_body``, which sets both lengths for one request), but on
  FLEURS concatenations 40 s windows give more than twice the character error rate of
  this policy. ``/generate`` refuses a clip over 40 s with a 400 rather than truncating it.
- The CTC head covers every language at once: ``language``, ``prompt`` and ``temperature``
  have no effect (the reference ignores a language code for CTC models too), and
  decoding is a per-frame argmax. There are no timestamps: ``srt`` / ``vtt`` get one cue
  per window and ``verbose_json`` one segment per window, timed to the windows. Nothing detects the
  language either: ``verbose_json``'s ``language`` is the request's ``language`` as given,
  or null. ``/v1/realtime`` is not supported.
- ``dtype`` defaults to ``bfloat16``, the reference's inference dtype; ``float32`` also
  turns off TF32 for the node's matmuls. ``max_batch_size`` (16) caps the requests in one
  step; the step sorts them by length and splits them so that each forward's padded audio
  (rows times the longest clip) stays within ``max_batch_seconds`` (320). In bf16 a
  request batched with others can differ from the same request run alone where two
  characters nearly tie, as the reference's own batched ``transcribe`` does;
  ``max_batch_size: 1`` or ``dtype: float32`` give batch-invariant transcripts.
- Parity scripts against Meta's pipeline live in ``test/omnilingual_asr/``
  (``record_oracle.py`` runs the reference in its own venv, as it pins torch 2.8 and
  fairseq2).

FastConformer ASR (``canary``, ``canary_qwen``, ``parakeet``, ``nemotron_asr``) notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Install with ``.[asr]``. Uploads that libsndfile can't read (M4A, WebM) are decoded by
  torchcodec, which needs FFmpeg.
- One GPU per deployment: both nodes share a rank, and there is no tensor parallelism.
- Request fields, the same on ``/v1/audio/transcriptions`` and in ``/generate``'s
  ``model_kwargs``:

  - ``parakeet``: detects the language from the speech. ``language`` may name one of the 25
    it knows (bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro,
    sk, sl, es, sv, ru, uk) and is then ignored; any other is refused.
  - ``nemotron_asr``: ``language`` is its prompt: a locale from the checkpoint's
    ``processor_config.json`` (such as ``en-US``), a bare code, or ``auto`` (the default).
  - ``canary``: ``language`` (the speech's, default ``en``; one of Parakeet's 25),
    ``target_language`` (a translation to or from English; the OpenAI route also takes
    ``task="translate"``) and ``punctuation`` (default on).
  - ``canary_qwen``: English only (checked on the OpenAI route). ``instruction``
    replaces the trained one, "Transcribe the following:"; ``/generate``'s text is unused.

- A request holds up to 610 s on Parakeet and Nemotron, and 40.96 s on Canary and
  Canary-Qwen. ``/v1/audio/transcriptions`` cuts longer uploads (up to 30 min) into windows
  at quiet moments and joins their text. None of the four emits timestamps.
- Parakeet and Nemotron hold no cache. Each clip's decoder state (the LSTM's, and the
  cursor) sits in a recurrent state slot from its first decode iteration until the request
  is torn down; the YAMLs size ``resources.transducer_state.max_slots`` (129) for
  ``max_concurrent_requests`` (128) and the sink slot, and startup refuses a cap the pool
  can't hold. A decode iteration is 64 greedy steps, and its tokens stream out as it ends. ``compile_step: false`` in
  their ``model_kwargs`` decodes with eager kernels instead of the compiled step.
- Canary's and Canary-Qwen's YAMLs size the KV caches for ``max_concurrent_requests`` (64),
  and startup refuses a cap the caches can't hold.
- Canary-Qwen's default KV cache takes about 15 GB (64 requests at 2048 tokens); on a
  smaller GPU, lower ``max_concurrent_requests`` and ``resources.kv_cache.max_num_pages``
  together.

Qwen3.5 / Qwen3.8 (``qwen3_5_*``, ``qwen3_8_27b``)
--------------------------------------------------

Text, image and video chat on the Qwen3.5 dense family (five sizes; MoE
variants are not supported yet). Served on both ``POST /generate`` and
``/v1/chat/completions`` (``image_url`` and ``video_url`` parts). Images and
videos may be interleaved with text anywhere in the prompt, and prefill follows
the order they were written::

    mstar serve qwen3_5_4b --gpus 0

Video follows transformers' ``Qwen3VLProcessor`` exactly: frames are decoded
with torchcodec on the host (FFmpeg's shared libraries must be installed;
without them a video request gets a 400, as there is no fallback decoder),
sampled at the checkpoint's ``video_preprocessor_config.json`` rate (2 fps,
4 to 768 frames for the released checkpoints), resized within its pixel
budget, and paired into temporal groups of two frames. Each group is preceded
by a ``<t seconds>`` timestamp in the prompt and placed in MRoPE like a
one-frame image. The prompt layout is the one transformers uses from 5.10
(and vLLM and SGLang): the template's ``<|vision_start|>``/``<|vision_end|>``
around a video stay, with each group's own pair inside. Earlier transformers
replaced the outer pair too; mstar keeps the 5.10 layout whatever version is
installed. Resizing follows the installed transformers version:
transformers 5.22 turns on the per-frame pixel cap of qwen-vl-utils by
default, which lowers the token count of short clips. Notes:

- The vision encoder attends within each image and each video frame group
  separately, and its captured CUDA graphs hold up to 16384 patches and 64 such
  segments per prompt. A prompt over either limit runs the encoder without the
  graph: a long video at ordinary resolution exceeds the patches, and one over
  ~64 s with frames resized below ~256x256 pixels exceeds the segments.
- A long video is one wide prefill: one 1-3 minute video is ~11-12k tokens,
  beyond the largest captured ``prefill_vision`` bucket (4096), so it runs
  eagerly.
- Size the KV pool for video. ``configs/qwen3_5_*.yaml`` keep the default
  2048 pages of 128 tokens (262k tokens per deployment); 32 concurrent long
  videos need about twice that. Measured on Qwen3.8-27B at TP4: 32 concurrent
  long videos on the default pool gave 67 tok/s with most requests timing out,
  and 1,585 tok/s with none failing on 8192 pages. Raise
  ``resources: kv_cache: max_num_pages`` to fit the concurrency you serve.
- Decoding happens in the API server process, before the request is queued,
  and counts against ``MSTAR_MAX_VIDEO_PIXELS``: at the default, a 1080p
  video over about 259 s gets a 400.

Most layers are gated DeltaNet, so a request holds a recurrent-state slot as
well as a KV allocation. Set ``gdn_state.max_slots`` in
``configs/qwen3_5_*.yaml`` to the concurrency you want plus one for the sink;
CUDA-graph capture and padded replays use the sink, not slots. A slot is
~20 MiB for the 0.8B and ~50 MiB for the 27B at TP4 (half that in bf16), hence
not the 256-slot default. The state defaults to the checkpoint's
``mamba_ssm_dtype`` (fp32 for the released checkpoints);
``gdn_state.state_dtype: bfloat16`` halves state traffic at some precision and
needs FlashInfer's fused bf16 decode kernel (K = V = 128).

``temperature``, ``top_p``, ``max_tokens`` and ``seed`` are the standard fields.
``repetition_penalty`` and ``enable_thinking`` (default true; the template opens
a ``<think>`` block) are read by the model but are not OpenAI fields — pass them
via ``extra_body``.

Qwen3.8-27B (``qwen3_8_27b``) is the Qwen3.5 architecture and runs on the same
code, served from ``configs/qwen3_8_27b_tp4.yaml`` (TP4)::

    mstar serve qwen3_8_27b --gpus 0,1,2,3

Its ``output_gate_type: swish`` names the gated DeltaNet norm's activation,
which is the SiLU every Qwen3.5 checkpoint already uses. transformers ignores
the key and hard-codes SiLU. Video is still
refused and the MTP head is skipped. The chat template's new
``reasoning_effort`` and ``preserve_thinking`` are not forwarded: every
request gets the template's defaults (``xhigh`` and preserved), whatever the
client sends.

Command A+ (``command_a_plus``)
-------------------------------

Text and image chat on Command A+: 32 MoE layers (128 routed experts, top-8,
plus an always-on shared expert), with three sliding-window (4096) layers for
every global one. Served on ``POST /generate`` (there is no OpenAI adapter
yet)::

    mstar serve command_a_plus --gpus 0,1,2,3
    curl -s http://localhost:8000/generate -F 'text=What is in this image?' -F 'files=@cat.jpg'

Images are tiled like transformers' ``Cohere2VisionProcessor`` (up to 12
512x512 tiles plus a thumbnail, 256 tokens each) and encoded on the first LLM
rank. In ``messages``, an image is a ``{"type": "image"}`` item; the uploaded
files fill them in order.

Install the ``command_a_plus`` extra, and point ``CUDA_HOME`` at a toolkit
matching PyTorch's CUDA version so the fused-MoE alignment op can JIT-build.
The ~437 GB of BF16 weights are never downloaded automatically: download the
checkpoint and set ``model_kwargs.checkpoint_dir`` in
``configs/command_a_plus_tp4.yaml``, the default (TP4, 65536-token
``max_seq_len``). Its 8192-page KV pool fits on GB200; on 141 GB GPUs such as
H200, set ``resources.kv_cache.max_num_pages`` to 2048. Tokenizer and config
metadata are pinned to one checkpoint revision.
``configs/command_a_plus_tp8.yaml`` targets 8x 80 GB GPUs and keeps NCCL's
all-reduce until FlashInfer's is measured there.

``model_kwargs={"messages": [{"role": ..., "content": ...}, ...]}`` with
``system`` / ``user`` / ``assistant`` turns goes through the chat template; a
plain ``text`` prompt is one user turn. Sampling defaults to temperature 0.9,
top-p 0.95 and repetition penalty 1.04. The model reasons before answering, so
give it a generous ``max_output_tokens``; a prompt plus ``max_output_tokens``
(2048 when unset) longer than ``max_seq_len`` is rejected.

On CUDA, decode runs in CUDA graphs over a fused layer: two concatenated GEMMs,
Triton kernels, and one all-reduce per layer (``tp_allreduce: flashinfer`` at TP4).
On 4x GB200 it decodes about 290 tok/s for one request and 3700 tok/s across
64 (``benchmark/command_a_plus/bench_http.py``); ``validate_checkpoint.py``
there checks parity against Hugging Face, with ``--images`` for image prompts.
Not supported yet: FP8 weights, prefill CUDA graphs, and reclaiming KV pages
outside the sliding window.

Kokoro notes
------------

- Install the G2P dependencies with ``pip install -e '.[kokoro]'`` and fetch the
  spaCy tagger once with ``python -m spacy download en_core_web_sm`` (misaki does
  this itself on first use when it has network access). ``.[all]`` includes the G2P
  dependencies, and the Docker images also install the tagger. ``pip install 'misaki[en]'``
  additionally bundles espeak-ng (GPL), which Kokoro uses only as the fallback for
  out-of-dictionary English words and as the G2P for Spanish, French, Hindi,
  Italian and Portuguese voices; without it those words are skipped and those
  languages are unavailable. Japanese and Mandarin voices need ``misaki[ja]`` /
  ``misaki[zh]``.
- Serve with ``mstar serve kokoro``. Request knobs: ``voice`` (a bundled voice such
  as ``af_heart``, or a blend ``af_bella+af_sky`` / ``af_bella(2)+af_sky(1)``),
  ``speed`` (0.25-4.0), ``lang_code`` (defaults to the voice prefix: ``a`` American
  English, ``b`` British, ``e`` Spanish, ``f`` French, ``h`` Hindi, ``i`` Italian,
  ``p`` Portuguese, ``j`` Japanese, ``z`` Mandarin) and ``phonemes`` (skip G2P and
  synthesize a phoneme string directly).
- Deployment-wide options go in the YAML's ``model_kwargs`` (see ``configs/kokoro.yaml``):
  ``lang_code`` fixes the G2P language, ``espeak_fallback: false`` disables the espeak-ng
  fallback even when it is installed, ``chunk_target_phonemes`` and
  ``first_chunk_target_phonemes`` set the sentence packing, ``text_buckets``,
  ``frame_buckets``, ``capture_batch_sizes`` and ``max_batch_frames`` shape the CUDA
  graphs captured at start-up (fewer buckets on a smaller GPU), ``compile_decoder: false``
  skips the ``torch.compile`` of the vocoder (start-up in seconds instead of minutes, about
  half the throughput on an H100) and ``decoder_dtype: bfloat16`` runs the vocoder trunk
  in bf16 (about 13% more throughput at concurrency 32 in our runs, with the harmonic
  source and the iSTFT kept in fp32; the parity test covers fp32 only).
- Text is cut at sentence boundaries into chunks of at most 510 phonemes (the
  PL-BERT window); each chunk is emitted to the client as soon as it is
  synthesized, so ``stream=True`` on ``/v1/audio/speech`` returns audio sentence by
  sentence. The first chunk is kept short so time to first audio is one short
  synthesis.
- Output is 24 kHz mono PCM16. The model runs in fp32 by default: its vocoder is
  phase-sensitive, so reduced precision is opt-in. On CUDA the text half and the
  frame half of the forward are captured as CUDA graphs per length bucket and the
  frame half is compiled with dynamic shapes, so the first start-up on a GPU takes
  about two minutes; rows of one step are grouped by frame bucket
  (``frame_grouping: single`` pads them into one group instead, kept for comparison).
- ``examples/livekit_kokoro.py`` and ``examples/pipecat_kokoro.py`` plug the server
  into LiveKit Agents and Pipecat through their OpenAI TTS plugins (``base_url``
  pointed at M*, ``response_format="pcm"``); ``GET /v1/audio/voices`` lists the
  voices for a picker: the bundled voices whose G2P extra is installed on the server
  (a request for one of the others gets a 400 with the ``pip install`` hint).

Qwen3-Omni notes
----------------

- Streamed audio starts after 12 codec frames (0.96 s of audio), then 13, then
  25-frame chunks behind 25 frames of left context. Playback stays gapless
  while the Talker produces frames at about 1.5x real time over the first 50
  frames (about 19 frames/s per request); the Talker batches at most 32
  requests, so requests past that queue and can fall below it and pause. Set
  ``codec_chunk_schedule: []`` under the deployment YAML's ``model_kwargs`` to
  start with a 25-frame chunk instead: first audio comes later, and the
  Talker needs only about 1.0 to 1.3x real time.

Qwen3-TTS notes
---------------

- Install the model-specific dependencies with ``pip install -e '.[qwen3_tts]'``
  plus ``pip install --no-deps 'qwen-tts==0.1.1'`` (see :doc:`installation`), and launch a single-GPU deployment with ``mstar serve <key> --gpus 0`` where
  ``<key>`` is one of ``qwen3_tts`` (0.6B CustomVoice), ``qwen3_tts_1p7b``,
  ``qwen3_tts_voicedesign`` or ``qwen3_tts_base``. One model class serves every
  12 Hz checkpoint; the variant (speakers, instruction support, reference
  audio) is read from the checkpoint's ``config.json``.
- Requests: ``voice`` selects a built-in speaker (CustomVoice), ``language``
  defaults to automatic detection, ``instruct`` (OpenAI: ``instructions``)
  carries a style instruction (1.7B CustomVoice) or the voice description
  (VoiceDesign, required). Base clones a voice from one reference clip: on
  ``/v1/audio/speech`` pass ``ref_audio`` (data URL, base64, or an http(s) URL
  when the server sets ``MSTAR_ALLOW_REMOTE=1``; paths on the server are refused) plus
  ``ref_text`` (its transcript) or ``x_vector_only_mode: true``; with the SDK,
  ``client.tts(text, reference_audio="ref.wav", ref_text="...")``. The clip's
  conditioning (x-vector and codec frames) is memoised by content, so a voice
  reused across requests, or by the sentence chunks of one long request, is
  encoded once; named, persisted voices arrive with the shared voice registry.
- Text layout follows the reference defaults: CustomVoice and VoiceDesign put
  the whole text in the prefill; Base feeds it one token per frame. Override
  per request with ``non_streaming_mode``.
- ``/v1/audio/speech`` inputs of 600 or more characters are synthesized as
  ordered sentence chunks of about 400 characters (one Talker request each,
  two kept in flight while the current one streams); set
  ``sentence_chunking: false`` (or ``true`` for shorter texts) per request.
- Audio streams in a ramp of codec chunks: the first frame is decoded on its
  own (first audio one Talker step after prefill), the next windows add 3, 8
  and 16 frames, then 25 new frames behind up to 72 frames of already decoded
  left context, the decoder transformer's sliding window. The
  transformer stacks 8 such layers, so a window still differs slightly from a
  whole-utterance decode once an utterance outgrows the context; 25 frames of
  context differed far more. Each window size is a CUDA-graph bucket captured
  for batch sizes 1 to 32; the stream buffer reports how many leading frames of
  a window are repeated context, and the codec trims their audio. A voice
  clone's reference tail (its last 72 frames) is the first window's context.
- The codec runs its transformer over the whole window but its conv stack (all
  but a few percent of the codec's time) only over the new frames plus the
  stack's causal receptive field, derived from the decoder's modules at load
  (10 frames for the 12 Hz decoder; the new frames' audio equals the
  whole-window decode). It also fuses the SnakeBeta activations into one kernel.
- Talker prefill replays a packed CUDA Graph for the smallest token bucket
  (32 to 1024 tokens) that holds the batch; only the clone prefill, which also
  pushes the reference clip's frames into the codec stream, runs eager. Decode
  always uses the whole-walk CUDA Graph, with the 15-step
  CodePredictor loop captured inside it; request-local EOS suppression is
  carried as a graph tensor input so replay does not consult capture-slot dummy
  request state. Residual ``subtalker_*`` sampling is per-request through the
  ``code_predictor`` aux sampler, so custom values neither block batching nor
  fall off the graph. On the 1.7B checkpoints the CodePredictor projects the
  Talker-width inputs through ``small_to_mtp_projection`` before its depth loop.
- ``configs/qwen3tts_1p7b_split.yaml`` runs the Codec on a second worker that
  shares GPU 0 (``rank_devices``) and caps the Talker KV pool so both fit.
  On its own it lowers first-audio latency at high concurrency; with
  ``mps: true`` in the YAML (see :doc:`serving`) the two workers' kernels also
  overlap, which raised c=32 throughput by about a fifth on an H100.
- The codec decoder runs in float32 by default, reproducing the reference
  decoder bit for bit. ``model_kwargs: {codec_dtype: bfloat16}`` in the
  deployment YAML selects a bf16 codec (the Talker is bf16 either way). On a
  B300 that is slower, not faster: cuDNN's heuristics run several of its
  convs on a kernel without tensor cores, so measure before switching.
- Opt-in AudioSeal watermark on the codec's output (off by default; install
  ``pip install -e '.[audioseal]'``). Turn it on in the deployment YAML::

     model_kwargs:
       audio_watermark:
         enabled: true      # default false
         message: 0xBEEF    # 16-bit payload: int 0..65535 or a 16-character 0/1 string (default 0)
         alpha: 1.0         # strength, in (0, 4] (default 1.0)
         generator: audioseal_wm_streaming   # AudioSeal card name or checkpoint path (default)

  Unknown keys or out-of-range values fail startup, and so does ``enabled: true``
  without the package. Each request's PCM is marked chunk by chunk with its own
  streaming state, so a streamed utterance carries the same mark as a whole one.
  At startup the engine captures the generator per chunk length of the codec
  schedule (by default 1, 3, 8, 16 and 25 frames) and batch bucket, so marking a step's
  chunks costs one graph replay per distinct length. The captures hold GPU memory: about
  2.5 GiB for the default 30 graphs (5 lengths, 6 batch buckets) on an H200, 5.7 GiB with
  25 lengths. A chunk outside them (another length, more rows than the largest bucket,
  a partial frame) runs eagerly, and the first one of each kind is logged.
  Check files with ``python -m mstar.model.audio_watermark detect --message 0xBEEF out.wav``
  (``--message``, the deployment's payload: without it the verdict is the
  score alone, and an unmarked file re-encoded as MP3 can score above 0.5;
  ``--threshold``, default 0.5; ``--device``; needs ``.[audio]`` for
  ``soundfile``); a file at another rate than 24 kHz is resampled to 24 kHz
  first, since the mark is found only at the rate it was added at. Detect on
  whole utterances: a clip under a few seconds can miss the mark. The helper lives in ``mstar/model/audio_watermark.py``.
- Weight loading checks coverage in both directions: a parameter the checkpoint
  does not fill, or a checkpoint tensor the port does not load, fails startup.
- The 12 Hz codec does not require the system SoX executable. M* imports only
  the exact upstream speech-tokenizer modules, avoiding qwen-tts's unrelated
  25 Hz SoX probe during worker startup.

For throughput/latency validation, run the native serving benchmark with the
Qwen3-TTS model metadata rather than the Orpheus compatibility entry::

   python -m benchmark.runner \
       --url localhost:8000 \
       --model qwen3_tts \
       --profiling-type closed_loop \
       --request-type text_to_speech \
       --num-requests 20 \
       --inference-system ours \
       --num-warmup 2 \
       --max-concurrency 4 \
       --dataset seed_tts \
       --output-dir .bench_outs

The benchmark stops on the model's natural codec EOS by default. Use
``--ignore-eos --output-len-min N --output-len-max N`` only when measuring
fixed-length decode throughput rather than end-user latency.
The first process-local request can include eager FlashInfer kernel JIT, so
keep the warmup requests enabled when reporting steady-state latency.

For cross-engine comparisons (M*, vLLM-Omni, SGLang-Omni) use the shared
streaming ``/v1/audio/speech`` client, which measures time-to-first-audio, RTF
and audio-seconds per second with the same request for every engine, and score
the saved WAVs with ``benchmark/tts_speech_wer.py``::

   python -m benchmark.tts_speech_bench --engine mstar --url http://127.0.0.1:8000 \
       --model qwen3_tts_1p7b --sentences sentences_200.txt --voice vivian \
       --language English --concurrency 8 --repeats 3 --out results/mstar_c8.json

Fish Audio S2-Pro notes
-----------------------

- The ``fishaudio/s2-pro`` weights are under the Fish Audio Research License:
  research and non-commercial use only. M* downloads them at runtime and does
  not redistribute them; read the license before deploying.
- ``pip install -e '.[fish_s2_pro]'`` then ``mstar serve fish_s2_pro --gpus 0``.
- The voice-clone warmup at startup uses ``samples/af_heart_0.wav`` from
  ``hexgrad/Kokoro-82M``. An offline deployment that pre-fills its HF cache
  needs that file as well; without it the warmup is skipped with a warning and
  the first clone request pays the one-time cost.
- Requests: ``/v1/audio/speech`` with ``input``; inline ``[tag]`` prosody
  control is part of the text. Clone a voice with ``ref_audio`` (data URL,
  base64, or an http(s) URL when the server allows remote fetch) and its transcript in
  ``ref_text``; with the SDK, ``client.tts(text, reference_audio="ref.wav",
  ref_text="...")``. ``voice`` and ``speed`` are ignored.
- Sampling defaults follow the reference (temperature 0.8, top-p 0.8, top-k 30);
  ``do_sample: false`` is greedy. The order differs: M\*'s sampler applies the
  temperature first and takes top-k and top-p over the tempered probabilities,
  while the reference takes top-p over the untempered ones, so at the same
  values M\* keeps a slightly different set of tokens per step. The top-p
  boundary differs as well: M\*'s sampler (FlashInfer) keeps the token whose
  probability carries the running total past top-p, while the reference drops
  it unless it is the most likely token. ``max_new_tokens`` (default 2048, at most
  4096 frames of 2048 samples at 44.1 kHz) and ``ignore_eos`` pass through.
- Streamed audio matches the whole-utterance decode to within one int16 step:
  each codec window carries the decoder's whole receptive field as context, and
  the codec runs in fp32 with TF32 off.

Nemotron VoiceChat (``nemotron_duplex``) notes
----------------------------------------------

- Three nodes in three async partitions, joined by streaming edges, with the
  user's audio a client stream (``audio_in``) into the first:
  ``conformer_encoder`` (16 kHz audio → one LLM-space embedding per 80 ms frame,
  streamed: each chunk runs the layers over its new frames only, against the
  call's cached last 70 frames, so a step costs the same however long the call
  is; it emits the frames no later audio can change, the last ones at the end
  of input)
  → ``nano_llm`` (frame-synchronous decode loop: each step fuses one audio frame
  with the previous agent-text and tool-call tokens, emits one agent-text token)
  → ``eartts_talker`` (one text token → 31 RVQ codes, decoded by the RVQ codec
  in the same step to that frame's 80 ms of PCM, which goes straight to the
  client: no codec step of its own, so a frame's audio leaves with the step
  that made its codes. The decoder is causal, so each call keeps its stream
  state in a recurrent-state pool, ``codec_state``: each ConvNeXt block's last
  6 inputs and the inverse STFT's 12-sample overlap, ~95 KiB a slot in bf16. A
  step decodes only the call's new frame against that state, and a fresh,
  zeroed slot is the decoder's causal padding, so a call's PCM equals the full
  decode of its frames from its first frame; at 32 sessions the decode is
  ~100 GMAC per 80 ms tick, inside the talker's CUDA graph).
  The decode loops are stream-terminated: they end when the upstream stream
  closes, not on EOS, which is an ordinary per-frame token in duplex speech.
- ``nano_llm`` declares a paged KV cache over its four attention layers with
  the attention plan on it (``resources: nano_kv`` / ``nano_attn``; no
  positional encoding, so no position resource), a recurrent-state pool holding
  the 27 Mamba-2 layers' conv window and SSM state, one slot per session
  (``mamba_state``: 32 sessions by default, ~137 MB per slot in fp32, tunable
  with ``max_slots``), the Mamba-2 resource planned on it
  (``mamba``) and the agent-text sampler (``nano_sampler``). Its decode step is
  captured as a CUDA graph for batch sizes 1 to 32.
- Every per-session resource is sized for 32 sessions: a slot in each of
  ``mamba_state``, ``encoder_state``, ``encoder_front`` and ``codec_state`` (``max_slots``, one
  more than the sessions, for the sink) and in ``encoder_kv`` (``max_slots``,
  no sink), and its pages in ``nano_kv`` and
  ``talker_kv`` (``max_num_pages``; both sized from ``max_session_s``, and the
  nano's from ``max_prompt_tokens`` too). The shipped configs set a
  conservative ``max_concurrent_requests: 16`` (below); raising it to 32 needs
  no other change, and to serve more, raise the cap and all six with it. The server
  refuses to start with a cap one of them cannot hold, since a session past
  any of them would wait on admit, and its frames with it, until another ends.
  The nano, the talker and the encoder each run at most 32 sessions
  a step, their widest CUDA graph; more are split over several steps rather
  than run eagerly.
- ``eartts_talker`` advances every live session in one backbone pass per frame
  through the engine's paged KV (``talker_kv`` / ``talker_attn`` /
  ``talker_pos``): each session owns two streams, the conditional and the
  unconditional CFG context, planned together under one combined label, so a
  step is one packed batch of two tokens per session (a session's first step
  prefills its 38-token speaker/text warm-up instead). The pool's head dim is
  128: FlashInfer computes 64/128/256 exactly and returns wrong values for the
  talker's 72, so q/k/v are zero-padded (exact) and RoPE is applied in torch
  from the positions the position resource planned. Sampling noise is drawn
  per row from the request's seeded generator before the step, so a session's
  speech does not depend on which other sessions share it. The per-frame step
  is captured as a CUDA graph for 1 to 32 sessions; a session's first step
  (its 38-token speaker warm-up) runs eager. The talker KV pool is 1025 pages
  of 128 positions (28 GiB): 32 sessions x 2 streams x 16 pages plus a sink
  page, enough for ``max_session_s`` (160 s) per session, with no
  sliding-window eviction yet; longer audio is refused (below).
- Measured on one H100 80GB (``benchmark/nemotron_duplex/sessions.py`` on the
  106-frame demo clip, three repeats): 24-25 ms per 80 ms tick with one
  session, 61-64 ms with 32 concurrent sessions, 76-85 ms with 40, 87-92 ms
  with 48 and 95-97 ms with 64 (2026-10-08). 32 is the most that keep up with
  real time, hence the pools' size; every session received all its frames.
  Those are one-shot replays (the whole clip uploaded, so the server runs as
  fast as it can). Live calls paced in real time over ``/generate/ws``
  (``test/nemotron_duplex/live_ws_request.py``, 80 ms appends, 2026-10-17, one
  H100): input-to-token lag p50 48 ms with one call and about 100-120 ms with
  8, which keep up; from 24 calls up the server fell behind real time, so lag
  grew over the call. Capturing the encoder's live step (below) took its host
  time at 32 calls from 23.6 to 8.4 ms and cut live lag 2-6x at 16-32 calls
  (paced runs interleaved against the uncaptured encoder, ``MSTAR_SHM_ARENA=1``);
  32 calls staggered over 2 s all finish, heard latency p50 2.1-2.3 s (from the
  user stopping to agent audio playable without gaps). With the codec in the
  talker's step, on one H200: 16 live calls keep real time whether they start
  together or 2 s apart; 32 all finish but sit at the edge of real time (one run
  kept up with 32 started together, another did not; later runs kept up at 32),
  so the shipped cap is a conservative 16. Served audio is checked for intelligibility by
  transcribing it (``test/nemotron_duplex/asr_check.py``, Whisper
  large-v3-turbo): MaskGIT sampling is knife-edge, so attention-backend
  numerics change the waveform but not the words.
- Against vLLM-Omni 0.28's native duplex serving of the same checkpoint (three
  eager fp32 engines, one session per GPU by its deploy config, its realtime
  client): on the demo clip both systems' response latency is the model's
  turn-taking decision (first agent text about 1.8 s after the user stops;
  first voiced audio 1.6 s here vs 2.0 s in stream time there, 2.3-2.6 s wall
  there), while M* serves 32 concurrent sessions with margin on one H100.
- The nano text tokenizer is read from the ``nano/`` folder of
  ``pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark`` (the base checkpoint
  ships only the RNN-T tokenizer); prefetch both repositories on machines
  without egress.
- Inputs are user-only 16 kHz mono audio plus an optional system prompt (the
  request's text, primed before the first frame exactly as the reference
  does): one clip uploaded with the request, or a live call over
  ``/generate/ws``, ``append`` messages until ``end_input`` (see
  :doc:`clients`; a one-shot request over the socket carries ``end_input:
  true`` on its opening message, or it waits for appends). Output is clocked by
  the input, one agent frame per 80 ms of user audio, so a live client keeps
  streaming (silence included) for as long as it wants the agent to talk. The
  model answers in the frames *after* the user stops, so append a few seconds
  of trailing silence to a clip. The checkpoint's demo recordings are
  two-channel conversations (user left, agent right); feed the left channel
  only.
- Limits: a call is capped at ``max_session_s`` (160 s) of audio. An upload, or
  any one file, past it gets a 400, as does one of more than two channels, and so
  does a live call whose appends pass it: ``audio_in`` declares the session's
  samples as its ``max_items``, which the client stream counts before the encoder
  sees them (raise ``max_session_s``, not the stream's limit, for longer calls). A
  call that ends with no more than 16 ms of audio, which the encoder cannot
  featurize, fails with a 500 (the worker reports it without a client status).
  Appends must be 16 kHz audio (they are neither
  resampled nor rescaled one by one, which would put a seam at every append).
  Send appends of 80 ms or more: the encoder takes one append per step, so
  shorter appends need several steps per frame and fall behind once a step
  takes longer than an append.
- Encoder numerics, measured on CPU at the checkpoint's dimensions with random
  weights: an upload is one encoder pass, bit for bit the offline encoder's in
  fp32 and under bf16 autocast, save the last frame when the end of input
  reaches the encoder after the audio (about 1e-6 in fp32). 80 ms appends put
  the frames within about 2e-6 of offline in fp32 and about 0.7% under bf16
  autocast. So live and one-shot text should agree token for token in fp32
  (inferred from those frames, not measured on served text); under the served
  bf16 a different chunking can flip a knife-edge frame such as the agent's
  turn start.
- The encoder keeps each call's state in a slot per session of three
  resources: its attention keys and values in a bounded KV (``encoder_kv``),
  which keeps a call's last 70 frames per layer and runs the attention, each new
  frame over the 70 before it, with the relative-position term as a band of
  distances; its conv inputs (``encoder_state``); and its newest audio and mel
  frames (``encoder_front``), the last two recurrent-state pools. A
  call that appends one frame of audio (1280 samples) on the frame grid, every
  live step of an 80 ms client, runs its whole step as a CUDA graph (the grid is
  the call's start: an opening clip or append that is not a multiple of 1280
  samples keeps the call off it for good); other calls take the front end
  eagerly, then the layer stack as a graph when each got at most one frame (a
  live step, the call's end), or eagerly (a one-shot upload, a large append:
  a chunk over a second is not batched with other calls, so the engine runs
  each call of that step on its own). Appends of other sizes are served the same,
  without the whole-step graph.
- Checks: ``test/nemotron_duplex/duplex_request.py`` prepares a clip from the
  bundled ``turn_taking.wav`` and sends it; ``parity_compare.py`` runs the
  standalone reference path on the same clip and checks the served text
  against it token for token (bf16, the serving precision, must match save the
  last frame; the fp32 oracle can differ where the agent's turn-taking decision
  is a knife-edge, which its printed logit margins show).
  ``live_ws_request.py`` streams a clip in 80 ms appends and reports per-frame
  lag; with ``--compare-one-shot`` it checks that a live call sending the whole
  clip as one append matches the upload save the last frame, and reports where
  the paced call's text first differs from it.
- Default deployment: ``configs/nemotron_duplex.yaml`` (all three nodes on one
  GPU); ``configs/nemotron_duplex_disagg.yaml`` puts the encoder+LLM and the
  talker (with its codec) on two ranks. Launch with ``--tensor-comm-protocol SHM`` on a
  single node. Text samples at ``temperature: 0.7, top_p: 0.9`` by default;
  send ``temperature: 0`` for greedy. The talker's mixture-of-Gaussians sampling
  is stochastic by design and seeded per request.
- Known limitation of the checkpoint: it often answers speech that is not
  addressed to it, such as a side conversation or background speech, where it
  should keep its turn (about half of such pairs on Full-Duplex-Bench v1.5).
  NVIDIA's NeMo reference inference (``examples/speechlm2/offline_voicechat_infer.py``
  on the ``nemotron-labs-voicechat`` branch of NVIDIA-NeMo/Speech) answers the
  same side speech on the same inputs, and the model card's known limitations
  say the model is not suitable for noisy environments, especially where
  background speech may occur. Nothing in the serving path tells speakers
  apart. Real interruptions and backchannels are handled well.

Chatterbox notes
----------------

- ``pip install -e '.[chatterbox]'`` then ``mstar serve chatterbox --gpus 0``
  (``chatterbox_turbo`` for the distilled Turbo checkpoint,
  ``chatterbox_multilingual`` for the 23-language one). All variants are one
  model class; the variant follows the registry key or ``model_kwargs:
  variant``.
- Multilingual: the same graph, S3Gen and voice encoder as the English model
  with the ``t3_mtl23ls_v2`` T3 weights (a 2454-token grapheme vocabulary)
  and a language token in front of the text. Requests pass ``language_id``
  in ``extra_body`` (``ar da de el en es fi fr he hi it ja ko ms nl no pl pt
  ru sv sw tr zh``; the deployment's ``model_kwargs: default_language``, ``en``
  by default, applies when a request gives none). The text is lower-cased and
  NFKD-normalised, Korean is decomposed into jamo and Chinese spelled as
  Cangjie codes from the checkpoint's table; Japanese kana reading, Hebrew
  diacritics and Russian stress marks use the same optional packages as the
  reference (``pip install -e '.[chatterbox_multilingual]'`` brings
  ``pykakasi`` and the Chinese word segmenter; ``dicta_onnx`` and
  ``russian_text_stresser`` are installed separately) and are skipped with a
  warning when a package is missing, as the reference does.
- Requests: ``/v1/audio/speech`` with ``input``, ``voice`` (``default`` = the
  voice shipped in the checkpoint, or a preset name resolved under the
  deployment's ``model_kwargs: voices_dir``), and in ``extra_body``
  ``ref_audio`` (data URL or bare base64 of a reference clip, 5-30 s,
  cloning; an http(s) URL only when the server sets ``MSTAR_ALLOW_REMOTE=1``;
  paths on the server, clips over 60 s and clips below 8 kHz are refused),
  ``exaggeration`` (0-2, emotion intensity, default 0.5),
  ``cfg_weight`` (0-1, default 0.5; 0 disables guidance and halves the T3 work),
  ``temperature`` (up to 5; under 1e-5 is greedy)/``top_p``/``top_k`` (the
  whole vocab or more is no filter)/``min_p``/``repetition_penalty`` (up to 2;
  the upper bounds are the reference demo's), ``seed``, ``n_cfm_timesteps``
  (S3Gen Euler steps, 10; Turbo 2), ``max_new_tokens`` (up to the deployment's
  ``model_kwargs: max_new_tokens_limit``, 1000 in the shipped configs; with
  ``max_concurrent_requests`` set, the server refuses to start unless that
  many requests of that length fit the KV cache, so raise it together with
  ``max_num_pages`` or a lower cap),
  ``ignore_eos`` (T3 decodes all ``max_new_tokens``
  past the stop token, for fixed-length benchmarks; the speech tokens among
  them are vocoded, but T3 keeps emitting stop and control tokens after the
  end of speech and those are dropped, so the audio is shorter than
  ``max_new_tokens`` / 25 s: 4096 tokens gave 117 s on base and 141 s on
  Turbo, not 164 s) and ``watermark`` (default on). Turbo ignores
  ``cfg_weight``, ``exaggeration`` and ``min_p`` like the reference package.
  The native ``/generate`` route and ``client.tts(...)`` take the same knobs;
  a clip uploaded as ``audio`` input is the reference voice.
- Graph: ``voice_encoder`` (speaker LSTM + S3 tokenizer over the reference,
  cached per clip hash) -> ``T3`` (paged KV, continuous batching; guidance runs
  the conditional and unconditional streams through one attention plan and one
  captured decode graph per batch size) -> ``s3gen`` (own streaming partition).
- Outputs are watermarked with Resemble's PerTh network when ``resemble-perth``
  is installed; ``watermark: false`` per request or in ``model_kwargs`` turns it
  off, and a deployment without the package logs that outputs are unmarked.
  The mark is embedded on the worker's device end to end (the package's own
  routine resamples on the CPU); it decodes with the package's detector like
  the package's output, from which it differs by the resampler only.
- Streaming (``stream: true``) emits WAV chunks as the speech tokens arrive:
  the first after 35 tokens on Chatterbox (1.12 s of audio, enough to
  cover the wait for the next chunk at 4 to 8 streams) and after 15 on
  Multilingual and Turbo, then 50, 100 and 200 tokens (``model_kwargs: stream_first_chunk_tokens`` / ``stream_chunk_tokens``
  / ``stream_chunk_growth`` / ``stream_max_chunk_tokens``): each chunk buys
  the playback time to produce a bigger one, so a stream costs three or four
  flow solves instead of one per 25 tokens. Each chunk re-runs
  the flow decoder over all tokens so far with a fixed noise field, holds back
  the three look-ahead tokens and crossfades the vocoder tail, so the stream
  is continuous but not sample-identical to the whole-utterance decode;
  ``stream_chunk_tokens: 0`` synthesises whole utterances (the reference
  path, bit-exact with the package at a fixed seed). ``stream_context_tokens``
  (default 25; 0 = whole history) bounds how many settled tokens a chunk's
  flow solve keeps as left context, making the per-chunk cost constant; the
  window stays as close to the whole-utterance decode as the full history
  does (log-mel correlation 0.988 vs 0.985 on CPU). Requests whose chunks
  are ready together share one padded flow solve (up to 8 per step).
- S3Gen runs from CUDA graphs by default (``s3gen_graphs``): the flow solve
  (one graph per rows x frames x steps; the estimator's hundreds of tiny
  kernels per Euler step make the eager solve launch-bound, 170 ms vs 45 ms
  for one row on an H100), the token encoder (per rows x token bucket) and
  the HiFT vocoder up to its output spectrum (per exact chunk length, its
  excitation noise drawn outside the graph in the reference's order; the
  inverse STFT runs eagerly since ``torch.istft`` synchronises). Rows are
  padded to the powers
  of two up to ``s3gen_max_batch_size`` (8) and frames to
  ``s3gen_frame_bucket`` (64); the built-in voice's chunk shapes are captured
  at startup, other shapes on first use; ``s3gen_graph_stages`` (default
  ``solve,encoder,vocoder``) picks the stages. The flow estimator runs in
  ``s3gen_estimator_dtype`` float16 by default with the Euler state in
  float32 (within 0.02-0.15 of the float32 log-mel; ``bfloat16`` is as fast
  and further off; ``float32`` is the reference path, bit-exact with the
  package at a fixed seed). ``t3_prefill_graphs`` (default on) captures T3
  prefill as packed CUDA graphs by token bucket for batches of up to four
  requests; decode graphs are always captured. ``s3gen_graphs: false`` gives
  the eager path; ``s3gen_compile: true`` is the older alternative and turns
  the graphs off.
- Sampling follows the reference order inside the sampler resource:
  repetition penalty -> temperature -> ``min_p`` -> ``top_p``; the T3 node
  declares ``enable_min_p`` on its ``SamplerSpec`` (see
  :doc:`adding_models`). T3 runs in bf16 (``model_kwargs: t3_dtype:
  float16`` is the alternative; float32 is refused because the paged
  attention has no float32 kernels, and fp32 token parity is checked by the
  CPU tests in ``test/chatterbox``).
- Reference clips are decoded with ``soundfile`` (WAV/FLAC/OGG/MP3 through
  the bundled libsndfile); other codecs fall back to ``torchcodec``, which
  needs FFmpeg's shared libraries on the node.
- Keep each request short, a sentence or two. Text longer than 512 tokens is
  rejected, but quality drops well before that, as it does in the reference
  package: on three-sentence inputs (600-720 characters) Turbo often drops or
  babbles the last sentence (median WER 0.10, reference 0.12; base 0.02), and
  1,500-2,000 characters come back from Turbo as a second or so of unrelated
  speech, with a 200. For long text send ``sentence_chunking: true`` in
  ``extra_body``: the server splits it into sentence groups of up to 400
  characters (at most 32) and synthesises them in order, or split it
  client-side.
- Benchmarks and parity scripts live in ``benchmark/chatterbox/``
  (``bench_all.sh`` drives M*, Chatterbox-TTS-Server and chatterbox-vllm on
  one GPU; ``reference_greedy.py`` + ``serve_parity.py`` compare a served
  greedy synthesis with the reference package; ``wer_eval.py`` is the Whisper
  intelligibility guard; ``node_breakdown.py`` turns a served run's
  ``--log-stats`` profiles into a per-node time table; ``profile_s3gen.py``
  micro-benchmarks the flow estimator, solve, encoder, vocoder and watermark
  by precision, eager vs. CUDA graph; ``gpu_util.py`` summarises
  ``nvidia-smi dmon`` samples taken during runs).

Cosmos3 environment requirements
--------------------------------

- ``flashinfer`` is required: it is the paged KV/attention backend used by the
  prefill, the captured CUDA graphs, and multi-request batches.
- The default denoise attention backend is ``dense_gen``
  (``Cosmos3Config.attention_backend``), which runs bs=1 eager generation
  attention as one FlashAttention-3 varlen kernel from the ``fa3-fwd`` wheel
  (AITER's varlen kernel on ROCm).
  That wheel is ABI-tied to the installed torch/CUDA build (Hopper builds
  exist for at least torch 2.9 + cu12.8 and torch 2.11 + cu13.0); install the
  one matching your environment. It ships kernels for sm_80 and sm_90a only.
  Where it is not importable or has no kernel for the GPU (Blackwell), the
  same dense pass runs on FlashInfer's ragged prefill, whose ``auto`` kernel on
  a B300 is CUTLASS's FMHA: 1.5 ms per layer for the 832x480x121 guidance pair
  against 6.3 ms for the paged prefill. ``resources: attn_gen:
  flashinfer_backend:`` picks that kernel (``cudnn``, ``cutlass``, ``fa2``); the
  server checks it at startup with a tiny plan and run, and a backend this GPU or
  FlashInfer build cannot run (a typo, ``fa3`` on Blackwell, ``cutlass`` on
  Hopper) stops it with a ``ValueError`` naming the backend.
  Without a usable FlashInfer either, the engine logs a warning and switches to
  the paged backend, but that is FlashInfer too (the default backend on every
  platform for now), so it cannot run there either: the dense pass needs
  fa3-fwd (AITER on ROCm) or FlashInfer.
  ``model_kwargs.attention_backend: flashinfer`` in the config YAML selects the
  paged backend explicitly.
- Video-input requests (video-to-video, action inverse-dynamics) decode the
  conditioning clip with ``torchcodec``; environments without it reject those
  requests at preprocessing (other modes are unaffected).
- Generated video containers are written with ``torchcodec``'s ``VideoEncoder``
  when available (torchcodec >= 0.9), otherwise with ``torchvision``'s
  ``write_video``, which needs the PyAV (``av``) package.
- Sound-enabled video responses mux the AAC track with the ``ffmpeg`` and
  ``ffprobe`` binaries, which must be on ``PATH`` (system packages, not
  pip-installable).
- The Wan-VAE decode dtype is gated on the cuDNN build: bf16 needs cuDNN >=
  9.16 (fast Hopper bf16 conv3d); older cuDNN serves the decode in fp32/TF32
  automatically.

Cosmos3 image-to-video conditioning
-----------------------------------

An image-to-video request's conditioning image is scaled to cover the output
size with its aspect ratio kept, center-cropped and rounded to 8 bits, as in
NVIDIA's cosmos-framework and diffusers 0.40 and later. Setting
``conditioning_resize: stretch`` under the config YAML's ``model_kwargs``
resizes it to the output size instead (antialiased bilinear, aspect ratio not
kept), as diffusers 0.39 did. Action requests ignore the setting.

Cosmos3-Edge reasoner and action loop
-------------------------------------

``cosmos3_edge`` serves the understanding tower as a vision-language model on the
same transformer instance and KV pool as the generator. ``/v1/chat/completions``
takes image and video content parts (URLs or data URIs) and streams tokens; the
chat template opens a ``<think>`` block by default. ``extra_body`` knobs:
``enable_thinking`` (or ``chat_template_kwargs.enable_thinking``), ``top_k``,
``repetition_penalty``, and for video attachments ``video_fps`` / ``video_num_frames``
(frames are sampled at 2 fps by default, each frame a timestamped span).

.. code-block:: bash

   curl -sN http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "cosmos3_edge", "stream": true, "max_tokens": 256,
     "messages": [{"role": "user", "content": [
       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
       {"type": "text", "text": "The task is to put the flower into the red bottle. Plan the next steps."}]}],
     "enable_thinking": false}'

The decode step is captured into CUDA graphs per batch bucket and, by default,
compiled first (``compile_reasoner_decode: true``; ``COSMOS3_REASONER_COMPILE=0``
turns it off): the eager step is over a thousand tiny kernels, and the fused
step runs at the weight-streaming floor (about twice the uncompiled rate at
batch size 1 on an H100). Concurrent chat requests share decode steps
(continuous batching over the captured decode graphs, padded to the next batch
bucket). Requests never see each other's data, but the batch bucket changes the
bf16 arithmetic of the step, so a long greedy answer can part from its solo run
where two candidate tokens tie: in every measured divergence the two tokens'
logits were equal or one bf16 ulp apart, the first differing token came after
tens to hundreds of identical ones, and identical prompts in one batch agreed
with each other. Short answers (128 tokens) came out identical in 8 of 8 runs;
256-token answers with thinking on in 2 of 8. Generation requests batch
into one denoise pass too, which is the same maths but not the same bf16
arithmetic — under classifier-free guidance the branch rounding is amplified,
so an image or clip produced alongside other requests differs from its solo
result at the kernel-drift level (~30 dB PSNR at guidance 6). Serve with one
request at a time when outputs must be bitwise repeatable.

The denoise step is compiled (``compile_denoise``) in Inductor's deterministic
mode by default (``denoise_compile_mode: deterministic``, the engine's compile
mode of that name): kernel configs are picked by rule, so a seed gives the same
video from every server process. ``default`` is torch.compile's own defaults,
and ``level`` takes the ``mstar serve -O`` level's options (``-O2``:
max-autotune).

Video clips decode through the shared Wan VAE decoder (``vae_decode_fast``):
one decoder call per latent, channels-last, in the decode dtype (bf16 with
cuDNN >= 9.16, otherwise fp32; see above), on the decoder-only copy of the VAE
the channels-last decoders on a GPU share. The first two calls (the clip-start latent and the one after it)
run eager; every later call is compiled in Inductor's deterministic mode. An
image (one latent) keeps the compiled stock ``vae.decode`` on that copy, which
is faster for a single frame. False decodes everything with the compiled stock
``vae.decode`` on the shared instance.

Two or more requests at the same denoise step run one cross-request forward
(``max_gen_batch_size``, default 8) for video, image and sound; the action step
runs eager. With ``compile_denoise_batched`` (default on) its layer stack is
compiled with the same mode, once, while the server starts, with the packed
length dynamic: no batch size or mix of resolutions compiles mid-traffic. The
per-request patchify, rotary tables and unpatchify around it run eager. At
832x480x121 one request's guidance pair is 24k tokens, which already fills the
GPU's GEMM and attention throughput, so a batch of N costs about N single
steps: batching keeps requests in step rather than adding throughput.

The action policy (``cosmos3_edge_droid``, or ``cosmos3_edge`` with an action
``domain_name``) predicts a chunk of robot actions from the current observation:
``output_modalities=action`` with ``model_kwargs``
``{"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 10,
"action_chunk_size": 32}``; the reply's ``action`` payload is float32
``[chunk, action_dim_padded]`` and the first ``raw_action_dim`` columns are the
embodiment's actions. For a control loop use ``/generate/ws`` (one connection,
pipelined observations): ``examples/cosmos3_action_ws_client.py`` runs it and
reports chunks/s, actions/s and latency percentiles.

Cosmos3 streaming rollout (windowed video)
------------------------------------------

Long clips can be generated window by window instead of in one denoise loop,
with each finished window streamed to the client while the next one is being
denoised. The deployment opts in with ``enable_windowed_video: true`` in the
config YAML (``configs/cosmos3_edge.yaml`` and ``configs/cosmos3_nano_ar.yaml``
do), which adds the ``video_gen_ar`` walk and a ``vae_decoder_ar`` node in its
own ``window_decoder`` partition; a request opts in per call:

.. list-table:: Windowed request knobs (``model_kwargs`` or the video request body)
   :header-rows: 1
   :widths: 22 14 64

   * - Knob
     - Default
     - Meaning
   * - ``window_mode``
     - — (``chained`` with ``stream_video``)
     - ``chained``: every window is a full bidirectional denoise conditioned on
       the previous window's tail (``overlap_frames`` pinned clean: the last
       frames are decoded and encoded again as a clip of their own, the
       video-to-video conditioning layout, so the window's first latent frame
       is a single-frame one as in training). ``kv``
       (experimental, see below): the finished window's clean K/V is committed
       to the cache and later windows attend to it block-causally; no overlap,
       and frames older than ``context_frames`` behind the frontier are
       released from the cache (the persistent world state of a long rollout
       stays bounded).
   * - ``window_frames``
     - 61 on Edge, 29 elsewhere; ``kv``: 29
     - Pixel frames per window (quantized to latent frames).
   * - ``overlap_frames``
     - 32 on Edge, 8 elsewhere
     - ``chained`` only: frames re-pinned from the previous window, as k = 1 +
       (overlap_frames - 1) // 4 latent frames (at least two), which pin
       1 + 4(k - 1) frames: 5 and 29 at the defaults.
   * - ``context_frames``
     - 61
     - ``kv`` only: committed context kept behind the frontier; ``0`` keeps all.
   * - ``stream_video``
     - ``false``
     - Emit each window as its own video chunk as it is decoded instead of one
       assembled clip. ``/generate`` streams the chunks as NDJSON lines,
       ``/generate/ws`` as frames, ``/v1/videos/generations`` switches to an
       NDJSON body (``video`` lines with a running ``index``, closed by ``done``).
   * - ``session_id``
     - —
     - Names a world-state session: the DiT node keeps the rollout's last window
       and the decoder its context. The most recent ``session_store_size`` idle
       sessions are kept; a session with a request in flight is never evicted, and
       a second request on it is refused until the first finishes.
   * - ``resume_session``
     - ``false``
     - Continue the named session: window 0 is conditioned on the stored last
       frames (pinned clean, like a chained overlap) and only the ``num_frames``
       new frames are delivered — a new prompt steers the same world.
   * - ``session_timeout_s``
     - 600
     - Seconds an idle session is kept after its request finishes, capped at the
       deployment's ``session_timeout_max_s`` (3600). An expired session resumes
       like an unknown one: the request is rejected.
   * - ``end_session``
     - ``false``
     - Drop the named session once this request is done (with or without
       ``resume_session``) instead of keeping its state.

.. code-block:: bash

   curl -sN http://localhost:8000/generate \
     -F 'text=a drone flies over a coastal town at dawn' \
     -F 'output_modalities=video' \
     -F 'model_kwargs={"num_frames":241,"window_mode":"chained","stream_video":true}'

.. warning::

   ``window_mode: kv`` is experimental, and the server logs a warning at
   startup when windowed video is enabled. The checkpoints were not trained
   block-causally: committed K/V conditions the next window far more weakly
   than frames pinned in the sequence, so an image-conditioned kv rollout
   jumps away from the scene at the first window boundary and, unless the
   prompt describes the image, can turn to texture noise within a few
   windows. ``chained`` with Edge's 61/32 defaults holds the scene over 241
   frames when the prompt describes the image; its first chunk arrives later,
   since window 0 is 61 frames. A prompt that
   contradicts the image still takes over within a few windows in every
   windowed mode; non-windowed clips are not affected.

The schedule is padded up to whole windows and the video trimmed back to
``num_frames``; a seeded request is deterministic end to end (later windows draw
their noise from the same generator). Windowed requests batch with each other
and with plain requests at the same walk. ``gen_capture_video`` lists (height,
width, frames) tiers whose denoise steps replay a per-step CUDA graph (one graph per
latent shape, the clean/noisy frame layout carried as a mask input; plain t2v/i2v and
``chained`` windows, never ``kv`` windows). It is empty by default: at 832x480 the
graph, which captures the paged attention, measured 3-6% slower than the eager dense
FA3 step for both the 121-frame clip and the 29-frame window, so it only pays for
small, launch-bound tiers.

Wan2.2 (``wan22``)
------------------

Text-to-video and image-to-video on **Wan2.2-TI2V-5B** — the dense 5B variant
(``Wan-AI/Wan2.2-TI2V-5B-Diffusers``): a native video DiT, a UMT5-XXL prompt
encoder and the Wan2.2-VAE, all four nodes stateless. The A14B (MoE dual-DiT)
variants are **not** supported; ``wan22`` rejects any other variant explicitly.

Install and serve on one GPU:

.. code-block:: bash

   pip install -e ".[wan22]"
   mstar serve wan22                              # configs/wan22.yaml
   # or: mstar-serve --config configs/wan22.yaml --port 8000

Two routes are served. ``POST /generate`` is the native one (multipart form, like
every other model); the mp4 comes back base64-encoded in ``outputs.video[0].data``:

.. code-block:: bash

   curl -s http://localhost:8000/generate \
     -F 'text=a fluffy cat walking across a sunlit floor' \
     -F 'output_modalities=video' -F 'streaming=false' \
     -F 'model_kwargs={"height":480,"width":832,"num_frames":33,"num_inference_steps":50,"guidance_scale":5.0}'

``POST /v1/videos/generations`` is the OpenAI-shaped surface (JSON body, mp4 in
``data[0].b64_json``). Here the size is a single ``WxH`` string — **width first**,
the opposite order to the ``height``/``width`` kwargs above — and supplying an
``image`` (URL or data URI) turns the request into image-to-video:

.. code-block:: bash

   curl -sS -X POST http://localhost:8000/v1/videos/generations \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "a fluffy cat walking across a sunlit floor",
          "size": "832x480", "num_frames": 33, "seed": 42,
          "num_inference_steps": 50, "guidance_scale": 5.0}' \
     | python -c "import sys,json,base64; d=json.load(sys.stdin); \
                  open('out.mp4','wb').write(base64.b64decode(d['data'][0]['b64_json']))"

``test/wan22/t2v_request.sh`` and ``i2v_request.sh`` wrap these two calls.

Generation knobs (per request, via ``model_kwargs`` or the request body):

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Knob
     - Default
     - Notes
   * - ``height`` / ``width``
     - 704 / 1280
     - The checkpoint's native 720P tier. **Both must be multiples of 32** — see
       below. Rejected with a 400 otherwise.
   * - ``num_frames``
     - 81
     - **Must be 4k+1** — see below. Rejected with a 400 otherwise. Latent
       frames = ``(num_frames - 1) // 4 + 1``.
   * - ``num_inference_steps``
     - 50
     - Clamped to ``max_denoise_steps`` (100), the denoise loop's ceiling.
   * - ``guidance_scale``
     - 5.0
     - Classifier-free guidance; run as a single batched forward.
   * - ``negative_prompt``
     - ``""``
     - Empty by default, matching the reference pipeline.
   * - ``fps``
     - 24
     - **Playback rate only** — it is the mp4 container rate, not a generation
       knob. Wan2.2 always generates a fixed ``num_frames`` clip at an implied
       24 fps, so another value just rescales the clip's duration.

**The ÷32 rule.** Height and width must each be an exact multiple of **32**:
a pixel dimension is downsampled 16x by the VAE and then patchified 2x by the
DiT, and only exact multiples survive both. So ``720x1280`` is **not** a valid
size for this model (720/32 = 22.5) — the 720p-class tier is **704**x1280. An
unaligned size is rejected at the request seam with a 400 naming the rule and
the nearest valid sizes, because it has no clean failure deeper in: the two
paths round the latent extent differently and the DiT dies mid-forward.

**The 4k+1 frame rule.** ``num_frames`` must be one more than a multiple of 4
(33, 81, 121 …): the VAE compresses time by 4 around an anchor frame, so only
``4k+1`` survives the round trip. Anything else is *silently floored* — ask for
32 frames and you would get 29 — so it too is rejected with a 400 naming the
nearest valid counts.

**Size ceiling.** ``height x width x num_frames`` may not exceed
``max_video_pixels`` (default 268,435,456, about 3.6x the default clip; raise
it under the yaml's ``model_kwargs``), each axis must fit the DiT's 1,024
rotary positions (``num_frames`` up to 4,093), and height and width are at most
16,384, the largest side the mp4 encoder (libx264) opens. Each is a 400 naming
the limit.

**Compiled DiT shapes.** With ``compile_dit`` the DiT compiles one graph per
input shape, so only the ``"WxHxF"`` shapes in ``compile_dit_sizes`` (default
``1280x704x81``) use the compiled transformer; each compiles on first use, once
with CFG and once without, so at most two compiles per listed shape. Any other
size or frame count runs the same transformer eager, with no compile, which is
slower per request once warm: on an H200, about 49% more at 640x352x17 and 55%
more at 832x480x33 than the same shape compiled. List the shapes a deployment
serves under the yaml's ``model_kwargs``; each one listed costs its compiles on
first use. The first DiT call after startup is traced twice (a guard on the
rotary table), so a server compiles at most two graphs per listed shape plus
one.

**UniPC runs inline, inside the DiT node.** Unlike cosmos3, the scheduler is not
a separate stage: the solver state (the order-2 history buffer and the
corrector's ``last_sample``) is carried on the denoise loop's own edges, so it
travels with the request rather than living in a scheduler object on one rank.
Requests are therefore independent and the loop is resumable across ranks.

**What is accelerated by default.** The DiT runs under ``torch.compile``
(``compile_dit``) and the untiled VAE decode runs the shared Wan VAE decoder,
channels-last and compiled (``vae_decode_fast``; set it to ``false`` under the
yaml's ``model_kwargs`` for stock ``vae.decode``). Both compile on first use:
the DiT only for the ``compile_dit_sizes`` shapes (see Compiled DiT shapes), so
the first request at a listed shape is slower. The VAE decode compiles once per resolution, and only
for the sizes in ``vae_decode_fast_sizes`` (default ``1280x704``); any other
size decodes with stock ``vae.decode``, frames identical to
``vae_decode_fast: false``. The compile uses Inductor's deterministic mode, so
every server process decodes the same latents to the same frames; torch 2.9.1
has no such mode, and there two processes can differ by a few uint8 levels.
The VAE decode
tiles only when free VRAM at decode time is short of the untiled estimate
(the ``auto`` policy; ``WAN22_VAE_DECODE_TILING`` forces a path), which keeps the untiled conv3d from OOMing a
32 GiB card. No CUDA-graph capture, no continuous batching (one request per
step) and no component offload.

LTX-2.5 (``ltx2_5``)
--------------------

Text-to-audio+video on **Lightricks LTX-2.5** (``Lightricks/LTX-2.5-Diffusers``, gated):
a 22B joint audio-video DiT, a Gemma-4 12B text encoder, and video/audio VAEs with a
48 kHz vocoder. The distilled checkpoint runs the model card's recipes without
guidance (single-stage: 8 sigmas); the two-stage recipe, the full (SFT) transformer,
image-to-video and the duration head are not ported yet.

Four nodes: ``text_encoder`` (native Gemma-4 + the text connectors), ``dit`` (the
DiT scaffold's denoise loop; video and audio latents carried in fp32; every attention
through two ragged attention resources), ``vae_decoder`` and ``audio_decoder``. The walk follows the requested output
modalities, so ``["video"]`` skips the audio decode (the model still denoises both).

.. code-block:: bash

   mstar serve ltx2_5            # configs/ltx2_5.yaml: DiT on GPU 0, the other nodes on GPU 1

Request knobs (``model_kwargs``): ``height`` / ``width`` (multiples of 32; default
544x960), ``num_frames`` (8k+1; default 121), ``fps``, ``seed``. Frames above 720p
decode in VAE tiles (``vae_tile_min_pixels``) to bound the decode's memory.

Outputs are an H.264 ``video`` and a 48 kHz stereo ``audio`` chunk (16-bit PCM).
``POST /v1/videos/generations`` returns one mp4 with the audio muxed in as AAC
(``"audio": false`` in the body for video alone); the mux needs the ``ffmpeg`` binary
on the server's ``PATH`` and falls back to video only without it.

Performance (2x H100, warm, against diffusers 0.41 with the same recipe and seeds):

======================================  ==============  ==============
                                        diffusers       ``ltx2_5``
======================================  ==============  ==============
544x960x121, one request                7.87 s          6.23 s
544x960x121, 2 concurrent               7.6 videos/min  12.7 videos/min
544x960x121, 4 concurrent               7.6 videos/min  14.0 videos/min
======================================  ==============  ==============

Concurrency pays because the text encoder and decoders run on the second GPU while the
DiT denoises another request; the DiT itself is GPU-bound (batching two requests costs
two steps' time).

The DiT is compiled and its denoise step captured per shape (``capture_shapes``);
other shapes run the compiled eager path, and the first request of a new shape pays
its compile.

FLUX.2 [klein] (``flux2_klein`` / ``flux2_klein_9b``)
-----------------------------------------------------

Text-to-image and reference-image editing on the step-distilled **FLUX.2 [klein]**
checkpoints. Four stateless nodes: a native Qwen3 encoder that runs only the 27 layers
whose hidden states the DiT consumes (taps 9/18/27 concatenated), the rectified-flow
transformer as the body of a ``denoise_loop`` (one Euler step per iteration, 4 by
default, no classifier-free guidance), the FLUX.2 VAE encode of reference images and
the VAE decode. All of them are exact ports; the CPU suite pins them bit-for-bit against
the diffusers modules on tiny random configs and the GPU suite compares real-weight
trajectories and PSNR against a recorded pipeline run
(``test/flux2_klein/record_oracle.py``).

Serve on one GPU and generate::

   mstar serve flux2_klein --gpus 0
   python - <<'PY'
   from mstar import MStarClient
   client = MStarClient("http://localhost:8000")
   png = client.generate_image("a cat holding a sign that says hello world", width=1024, height=1024, seed=0)
   open("cat.png", "wb").write(png)
   open("edit.png", "wb").write(client.edit_image("make it a watercolor painting", "cat.png", seed=1))
   PY

The OpenAI routes are ``POST /v1/images/generations`` (``size`` as ``WxH``, ``seed``,
``n``; ``num_inference_steps`` through ``extra_body``) and ``POST /v1/images/edits``
(multipart ``image`` + ``prompt``; up to four reference images are concatenated as
conditioning tokens, in order).

Deployment knobs live under ``model_kwargs`` in ``configs/flux2_klein.yaml``:

``attention_backend``
   ``sdpa`` (default) is the reference kernel, cuDNN on an H100 and measured as fast as
   FlashInfer in the served path. ``flashinfer`` runs the DiT's joint attention on the
   engine's ragged FlashInfer resource; both are CUDA-graph replayable.

``compile``
   ``torch.compile`` of the transformer, one trace per shape.

``compile_eager_rounding``
   Inductor rounds intermediates where eager PyTorch does, and fuses no FMAs.

``compile_exact_ops``
   ``true``, or a list of op classes among ``norms`` and ``activations``: those modules
   stay on the eager kernels inside the compiled forward, so inductor only fuses the
   chains around the GEMMs and attention. Each excluded module is a graph break, so this
   is a latency-for-exactness trade:

   * The norms alone make the transformer bit-exact with eager (measured on klein-4B and
     9B). The shipped ``[norms]`` costs klein-4B 4% of its B=1 latency (0.371 vs
     0.356 s); excluding the activations too adds cost and nothing else.
   * Z-Image's 180 norms per step cost 45% (1.25 vs 0.86 s at B=1), so it ships ``false``
     and lands at a median 34 dB. ``[norms]`` turns it reference-faithful (at least
     55.9 dB on every prompt).
   * Without it, the plain compile — even with eager rounding — lands at a median 35 to
     38 dB PSNR from eager over the 100 protocol prompts, because a 4- or 8-step
     distilled sampler amplifies the last bit of inductor's own reductions and activation
     decompositions.

``cuda_graph``, ``capture_sizes``, ``capture_batch_sizes``
   The denoise step, Euler update included, is captured per listed ``[height, width]``
   and batch size. Other shapes run the eager batched path.

``capture_edit_sizes``
   klein only: edit buckets with one reference image of the output size, the shape of an
   edit that keeps its reference's size. Defaults to ``capture_sizes``; ``[]`` captures
   none. The eight edit buckets of the 1024² default cost about 8 GiB of peak VRAM on
   klein-4B and 6 GiB on 9B, and every extra capture size adds roughly 7 minutes of
   startup.

``async_scheduling``
   The engine's speculative scheduling of the image nodes: the worker assembles a batch's
   next denoise step while the current one runs and merges requests that became ready
   meanwhile, so requests at different steps share a forward. On by default; a step a
   request speculates past its schedule is vetoed before any forward, and ``false``
   restores lockstep. Measured on one H100 back to back against lockstep: klein-4B 0.365
   to 0.372 s vs 0.377 to 0.390 s at B=1 and about 3% more images per second at 4 and 16
   concurrent requests, klein-9B 2%, Z-Image neutral, served images unchanged — the
   denoise phase is compute-bound either way.

``max_batch_size``
   Rows batched per denoise step.

``vae_compile``
   ``torch.compile`` of the VAE decode with inductor autotuning: 89 to 29 ms at 1024² on
   an H100. Its fused reductions move the image by about 55 dB PSNR from the eager decode
   on every prompt, and the autotuner may pick different conv kernels in another server
   process, so set it to ``false`` for bit-exact, repeatable output — which is how the
   parity suite runs. The compiled decode is warmed at load for the ``capture_sizes``
   grids and the decode batch sizes; any other latent shape decodes eagerly rather than
   paying a 40 to 100 s autotune inside a request. Memory cuts the other way: the eager
   decode of an 8-image batch at 1024² allocates about 40 GiB of activations, for served
   peaks of 75 GiB on klein-4B and 77 GiB on Z-Image at 16 concurrent requests against
   36 to 44 GiB compiled.

``max_image_area``
   Pixels, default 2048². Bounds the output size a request may ask for; larger requests
   are rejected with a 400 before scheduling instead of occupying the worker for minutes
   (an 8192² request means 262k tokens of quadratic attention).

With the klein defaults every served image is within 53 dB of the eager path on all 100
protocol prompts, and with ``vae_compile: false`` it is bit-exact. For scale, the served
images of the other engines are 10 to 16 dB from the diffusers reference for the same
seeds.
Requests at the same output size batch across users in every node, including the
text encoder, whose input is always 512 tokens. ``lora`` lists adapters to fold into the
transformer weights at load time (``[{path: ..., scale: ...}]``; diffusers/PEFT-format or
BFL-layout safetensors), so a styled deployment runs at the base model's speed. The adapter's
own scaling follows PEFT: an ``alpha`` key in the weights, else the ``lora_alpha`` (and
``use_rslora``) of an ``adapter_config.json`` / ``config.json`` next to them, else ``alpha = r``;
``scale`` multiplies that (a public rank-64 rsLoRA adapter lands within 53 dB of diffusers'
fused result — M* merges in fp32 with one rounding, PEFT fuses in bf16).
