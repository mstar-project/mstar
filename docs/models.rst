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
       ``droid_lerobot``, 8-dim raw actions); no sound pathway. The config
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
   * - ``cosyvoice3``
     - ``FunAudioLLM/Fun-CosyVoice3-0.5B-2512``
     - Zero-shot voice-cloning TTS: Qwen2-0.5B speech-token LM with RAS sampling,
       DiT flow-matching decoder and causal HiFT vocoder, streamed in chunks.
       Clones from a preset or an uploaded clip. 24 kHz.
   * - ``glm5_next``
     - ``zai-org/GLM-5.3-Flash``
     - GLM-5.3-Flash (320B/18B): hybrid KDA linear-attention + MLA MoE causal
       LM, text in / text out, fp8 checkpoint, TP8. ``configs/glm5_next_tp8_mtp.yaml``
       drafts with MTP k=3 (greedy only).
   * - ``kokoro``
     - ``hexgrad/Kokoro-82M``
     - TTS (82M, not autoregressive): misaki G2P + PL-BERT prosody + iSTFTNet
       decoder, 54 bundled voices and voice blends, sentence-chunked streaming,
       batched across requests.
   * - ``voxcpm2``
     - ``openbmb/VoxCPM2``
     - Tokenizer-free TTS: MiniCPM-4 base and residual LMs emit continuous latent
       patches through a local flow-matching DiT, decoded by AudioVAE V2 in
       streamed chunks. Zero-shot, voice design, and cloning from a clip. 48 kHz.
   * - ``muse_glimmer`` *(Beta)*
     - ``meta-models/Muse-Glimmer-30B``
     - Meta's 30B dense agentic VLM (text + image in, text out): sliding-window and
       NoPE global attention layers 3:1, plus a windowed ViT perception encoder.
   * - ``ripple_65b`` *(Experimental)*
     - ``ricursiveAI/Ripple-65B-Preview``
     - Ricursive's 65B dense text LLM: sliding-window and NoPE global attention layers
       4:1, with a value residual from the token embeddings. BF16, TP2 or TP4.
   * - ``nemotron_duplex`` *(Beta)*
     - ``nvidia/NVIDIA-NemotronLabs-VoiceChat-11B``
     - Full-duplex speech-to-speech (NemotronLabs VoiceChat-11B): Fast-Conformer
       encoder + Nemotron-H hybrid Mamba-2/attention backbone + EarTTS talker +
       RVQ codec; user speech in, agent text and 22.05 kHz speech out.
   * - ``glm52``
     - ``zai-org/GLM-5.2-FP8``
     - GLM-5.2 (753B/40B): MLA + DSA MoE causal LM, fp8 block-scale weights, TP8.
       ``model_kwargs.kernels`` is ``auto`` (the tuned kernels) or ``reference``, the numerics
       anchor, served eager. ``configs/glm52_tp8_mtp.yaml`` drafts with MTP k=3 (greedy only).
   * - ``orpheus``
     - ``canopylabs/orpheus-3b-0.1-ft``
     - TTS: Llama 3.2 3B LLM emitting audio tokens + SNAC 24 kHz decoder.
   * - ``pi05``
     - ``lerobot/pi05_base``
     - Pi0.5 vision-language-action robotics model (ViT encoder + LLM + flow action expert).
       A request sends one image per camera, up to the config's ``num_cameras`` (3 in
       ``configs/pi05.yaml``). Cameras it leaves out are masked, as openpi does, and more
       images than that are a 400.
       ``configs/pi05_droid.yaml`` serves the DROID fine-tune (``lerobot/pi05_droid``) with
       openpi's DROID normalization: raw joint state in, joint velocities out. It also
       serves openpi's policy WebSocket, so openpi's DROID client connects unchanged
       (see :doc:`clients`).
       ``LLM`` and ``action_expert`` share a KV cache, so ``node_groups`` must list them in
       one entry; the server refuses to start otherwise.
   * - ``qwen3_5_{0.8,2,4,9,27}b``
     - ``Qwen/Qwen3.5-4B``
     - Hybrid-attention VLM (text + image in, text out): gated DeltaNet linear
       attention interleaved with full attention, plus a ViT tower.
   * - ``qwen3_8_27b``
     - ``Qwen/Qwen3.8-27B``
     - Qwen3.8-27B on the Qwen3.5 code above (same architecture); ships a TP4 config.
   * - ``minicpm_o`` *(Beta)*
     - ``openbmb/MiniCPM-o-4_5``
     - Omni-modal chat (text/image/audio in, text and speech out): Qwen3-8B over a navit
       SigLIP + resampler and a Whisper-medium encoder; a Llama TTS and a streaming
       flow-matching vocoder for speech. Half-duplex only.
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
   * - ``phi4mm`` *(Beta)*
     - ``microsoft/Phi-4-multimodal-instruct``
     - Phi-4-mini with a SigLIP vision tower and a conformer audio tower: text, images
       and audio in, text out. Each request runs under the checkpoint's vision or speech
       LoRA as its media call for, from the engine's LoRA pool.
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
       A request carries a seed image and ``model_kwargs.actions`` (1 to ``max_frames``,
       512 latent frames, controller dicts); it generates one frame per action. A live
       ``/generate/ws`` call (opened with ``end_input: false``) appends more as
       ``{"append": {"model_kwargs": {"actions": [...]}}}`` and plays until it ends its
       input, or for at most ``max_frames`` actions in all.
   * - ``lingbot_world_1_3b``
     - ``robbyant/lingbot-world-v2-1.3b-causal-fast``
     - LingBot-World-V2 1.3B causal-fast: camera-controlled image-to-world video, streamed
       chunk by chunk (Wan2.1-layout causal DiT, rolling KV cache, 4 distilled steps).
       A request carries the whole camera trajectory as JSON in ``model_kwargs``, which the API
       server caps at 1 MiB: about 3,900 frames of ``[4, 4]`` float32 poses; send ``[F, 3, 4]``
       rows for up to the 4,077-frame maximum. One rollout at a time: the model refuses any
       other ``max_concurrent_requests`` until each rollout's text K/V and VAE caches, which
       are outside the rolling K/V pool (``resources: dit_kv: pool_tokens``), are engine
       resources. The DiT's attention runs
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
   * - ``minimax_h3`` *(Beta)*
     - ``MiniMaxAI/MiniMax-H3``
     - MiniMax-H3 (FL2VA) text- and first/last-frame-to-audio+video: a 33B packed-sequence
       audio-video DiT, CFG-distilled (one forward per step, 50 sigma points by default), a
       Qwen3-VL-32B conditioner (keyframes through its vision tower) and a ViT video VAE,
       emitting an H.264 video at 24 fps with 32 kHz stereo audio (one mp4 with AAC on
       ``/v1/videos/generations``). MiniMax H3 Community License Agreement.
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
- Models on the fused MoE kernel (Qwen3-Omni, Command A+) take their tiles from the
  ``moe_gemm`` table for the GPU (``mstar/kernels/tables/moe_gemm/``) when it has
  the expert shape, and default tiles otherwise. The server logs the sweep command
  for each shape it serves untuned; see "Tuned kernel configs" in :doc:`adding_models`.
- To add a new family, see :doc:`adding_models`.
- Unset sampling knobs take the checkpoint's ``generation_config.json``, then the
  model's config. BAGEL and Orpheus are the exception: their files are the
  backbone's chat presets (Qwen2.5-7B-Instruct, Llama-3.2-3B), so only
  ``do_sample: false`` is read and the recipes stay (BAGEL temperature 0.6,
  top_k 0, top_p 1.0; Orpheus temperature 0.6, top_p 0.8). MiniCPM-o keeps
  upstream ``chat``'s recipe over its file (temperature 0.7, top_p 0.8, top_k
  100, repetition_penalty 1.02).
- Qwen3-Omni's Talker cap comes from ``talker_max_new_tokens``, 4096 on the
  released checkpoint (it was 2048). Plain sampling keys target the Thinker, except
  on ``/v1/audio/speech``, where ``temperature`` and ``top_p`` set the Talker;
  ``talker_*`` and ``code_predictor_*`` set those stages on every route.
- Qwen3-TTS: ``talker_<knob>`` wins over a plain ``<knob>`` when a request
  sends both.

Per-request LoRA adapters
~~~~~~~~~~~~~~~~~~~~~~~~~

A ``resources: lora:`` block in a serving config serves PEFT adapters per request
(sized and loaded as described under Qwen3.5 below). A request picks one with the
``model`` field on any OpenAI route, or with ``lora_adapter: <name>`` in ``extra_body``
or ``/generate``'s ``model_kwargs``; an adapter the deployment does not serve is a 400.
Without the block a model runs exactly as before. The table lists the models that take
one and their targets, by the names a ``targets:`` list in the block uses. Every row is
validated on CPU (each model's own stack against merged weights at simulated tensor
parallelism); the GPU checks are in ``benchmark/lora/README.md``. Qwen3.5 passes them,
Orpheus agrees with transformers + PEFT on every row and misses the merged check on a
few, and Higgs Audio passes them, though its transformers check has not been rerun since
that reference moved to fp32 RoPE. The rows marked *not validated* do not yet, and serve
adapters at your own risk. None of their misses in the eager checks so far is an adapter
error: the checks are tighter than bf16 noise on these models, and when each layer is fed
an fp32 reference's input, an adapter's median error is within about 10% of a bf16
merge's.

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Model (node)
     - Targets
     - Notes
   * - Qwen3.5 / Qwen3.8 (``LLM``, ``vision_encoder``)
     - every linear of the text stack, ``lm_head``, ``embed_tokens``; the vision tower's
       blocks, merger and patch embed
     - See the Qwen3.5 notes.
   * - Orpheus (``LLM``)
     - ``qkv_proj``, ``o_proj``, ``gate_up_proj``, ``down_proj``, ``lm_head``,
       ``embed_tokens``
     - Hub Orpheus adapters (PEFT on ``q/k/v/o/gate/up/down``, ranks 8 to 512) are
       served; set ``max_rank`` to the largest.
   * - Higgs Audio (``LLM``)
     - ``qkv_proj``, ``o_proj``, ``gate_up_proj``, ``down_proj``, ``lm_head`` (the
       checkpoint's ``audio_decoder_proj.text_lm_head``)
     - No embedding target.
   * - Qwen3-Omni (``Thinker``), *not validated*
     - ``qkv_proj``, ``o_proj``, dense-layer ``gate_up_proj`` / ``down_proj``, ``lm_head``;
       the routed experts' ``experts_gate_up_proj`` / ``experts_down_proj`` when the YAML's
       ``targets:`` names them
     - Expert targets are opt-in: each slot holds every expert's A and B (about 1.7 GB per
       slot at rank 16, and as much pinned on the host per adapter held there), and with
       128 experts a pool serves at most 7 slots. Under TP only: an ``ep_config`` with
       ``parallel: ep`` over several ranks refuses them at startup. Per-expert adapters
       only (``experts.<N>.<proj>``), not PEFT's fused 3-D form. The router takes none.
       Talker, code predictor and encoders take none. Not validated: this MoE model's
       bf16 noise is about 10x a dense model's (its router flips experts on near-ties),
       so even base rows, run in a larger batch than the reference's, miss the check
       against merged weights.
   * - Qwen3-TTS (``Talker``), *not validated*
     - ``qkv_proj``, ``o_proj``, ``gate_up_proj``, ``down_proj``, ``codec_head``
     - Not the text projection (runs before the step) or the CodePredictor. Not
       validated: both checks fail on some rows, but against an fp32 reference LoRA
       gets every row's top-1 right while a bf16 merge of the adapter (the merged
       check's yardstick) flips one.
   * - Chatterbox (``T3``), *not validated*
     - ``qkv_proj``, ``o_proj``, ``gate_up_proj``, ``down_proj``, ``speech_head``
     - English and multilingual; a request's guidance (uncond) rows run its adapter too.
       Not Turbo (GPT-2's fused ``c_attn``), nor the conditioning encoder. Not
       validated: the check against merged weights fails on some rows, and an early
       layer's massive activation (its largest value grows from 8 to about 1200)
       amplifies bf16 noise.

Not yet: Whisper (its encoder node plans no step rows, and the decoder's cross-attention
K/V run over encoder rows, so the usual ``q_proj`` / ``v_proj`` adapters cannot be served
whole), BAGEL and Cosmos3 (mixture-of-transformers weights chosen per row inside one
forward, and guidance branches run as separate forwards), π0.5 (openpi's adapters are
not PEFT), FLUX.2 [klein] (merges a LoRA into its weights at load, not per request), and
every other family, which declares no LoRA specs yet: Canary, Canary-Qwen, Command A+,
CosyVoice3, Fish Audio S2-Pro, GLM-5.2, Kokoro, LingBot-World, LTX-2.5, Nemotron
VoiceChat, Omnilingual ASR, OmniVoice, Parakeet and Nemotron ASR, Qwen3-ASR, V-JEPA 2,
VoxCPM2, Wan2.2, Waypoint and Z-Image.

OmniVoice notes
~~~~~~~~~~~~~~~

- Zero-shot only: there are no built-in speakers. Pass ``ref_audio`` (an uploaded
  ``audio`` input on ``/generate``) with its transcript in ``ref_text`` to clone a voice,
  or describe one in ``instruct`` (``voice`` on ``/v1/audio/speech``).
  ``ref_text`` is required alongside the reference clip.
- ``language`` takes either the name (``Vietnamese``) or the id (``vi``): a
  name is resolved to the id the model was trained on before the prompt is
  built, and an unrecognised value warns and falls back to language-agnostic
  mode.
- The ``omnivoice`` package is installed separately from git rather than by an
  extra — see :doc:`installation`.
- The backbone is not autoregressive: it fills a fixed canvas of eight codebook
  rows over a few unmasking steps, so there is no KV cache and no per-token
  sampling loop. Serve it with ``mstar serve omnivoice --gpus 0``.
- The backbone's layer stack replays CUDA graphs per (requests, packed-token
  bucket up to 8192 tokens) with attention through a ragged attention resource;
  larger steps run the same layer loop eagerly. The layer loop is mstar's own over
  the reference's fused weights: q/k norms read the fused QKV output in place
  and each residual add folds into the next RMSNorm (flashinfer
  ``fused_add_rmsnorm``), so logits match the reference to half-precision
  rounding rather than bit for bit. ``model_kwargs: {backbone_graphs: false}``
  runs the same layer loop eagerly, and the serving config's
  ``prefill_token_buckets: {backbone: {backbone_layers: [...]}}`` sets the
  packed-token buckets (see :doc:`serving`).
- ``configs/omnivoice_split.yaml`` runs code2wav (the codec decode) on a
  second worker sharing GPU 0, so the unmask loop no longer stops for each
  finished request's decode; the silence trim and the rest of the output
  shaping run on the API side in either config. It sets ``mps: true`` (see
  :doc:`serving`) so the two workers' kernels overlap. On an H200 it raised
  concurrency-32 throughput from ~153 to 176-184 audio-s/s and cut TTFA p95
  from ~9 s to ~3 s, measured while code2wav still did the shaping on its
  worker; concurrency 1 was ~10% slower.

Higgs Audio notes
~~~~~~~~~~~~~~~~~

- The audio tower runs outside ``torch.compile`` (a clip's chunk count changes
  its shape, and every new shape recompiled on the serving path). Requests
  whose 4 s chunks share a mel width are encoded together, and full-width
  chunks replay a CUDA graph per chunk count; a clip under one chunk is padded
  to the full width and masked (``HiggsAudioTower``'s ``valid_frames``) so it
  replays a graph too. ``model_kwargs: {encoder_graphs: false}`` runs the
  tower eagerly.
- The model drops or repeats text on long audio, so ``/v1/audio/transcriptions``
  cuts an upload longer than 30 s into windows of at most 30 s run in parallel.
- One prefill step per request: the ChatML text around the audio and the audio
  embeddings go into the KV cache as three spans.
- Decode keeps each request's last token in the sampler's slot master and
  sends the client's copy inline (``MSTAR_DEVICE_LOOPBACK``, on by default),
  as Qwen3.5 does: a decode step stores and routes no tensors.

Orpheus notes
~~~~~~~~~~~~~

- Orpheus only ever emits ``<custom_token_*>`` ids (SNAC codes and the
  speech/turn markers), so its LM head computes the logits of those rows alone
  (``ColumnParallelLinear(output_start=...)``, about 18% of the head) and reads
  ``-inf`` for the Llama text rows, at every TP size. Sampling therefore
  renormalizes over the custom tokens: the reference samples the full
  vocabulary, where a text token's probability during speech is negligible but
  not zero, so sampled output is not identical to the reference's.
- ``/v1/audio/speech`` input of 250+ characters is synthesized as sentence chunks of
  about 200: the default 2,048 output tokens are about 25 s of speech, and a longer
  input would be cut off there (``sentence_chunking: false`` opts out). An input of
  more than 32 chunks (some 6,400 characters) is a 400.

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
  (rows times the longest clip) stays within ``max_batch_seconds`` (320). In bf16 the
  encoder packs a step's clips into one unpadded forward (a captured graph up to 16384
  frames) when their total audio fits ``max_batch_seconds``; a step over it encodes padded,
  per split. In bf16 a
  request batched with others can differ from the same request run alone where two
  characters nearly tie, as the reference's own batched ``transcribe`` does;
  ``max_batch_size: 1`` or ``dtype: float32`` give batch-invariant transcripts.
- Parity scripts against Meta's pipeline live in ``test/omnilingual_asr/``
  (``record_oracle.py`` runs the reference in its own venv, as it pins torch 2.8 and
  fairseq2).

Ripple-65B (``ripple_65b``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    mstar serve ripple_65b --gpus 0,1,2,3
    mstar serve ripple_65b --config configs/ripple_65b_tp2.yaml --gpus 0,1

- ``configs/ripple_65b_tp4.yaml``, the default, needs four 80 GB GPUs and
  ``configs/ripple_65b_tp2.yaml`` two 141 GB ones. TP can't pass the model's 8 KV heads. Both
  serve 131072 tokens a request.
  For more, raise ``resources.kv_cache.max_seq_len`` and ``max_num_pages`` together: a prompt
  holds all 80 layers' K/V until its prefill commits, so the shipped pool runs out near 260K
  tokens and two 128K prompts can't prefill at once.
- A chat with no ``temperature`` samples at 1.0, as Ricursive's server does, and one with no
  ``max_tokens`` stops at 2048 tokens, which a long reasoning reply can reach inside
  ``<think>``. Set ``max_tokens`` for hard problems.
- Text in, text out, on ``/generate`` and ``/v1/chat/completions``. A reply's reasoning stays
  inline as ``<think>...</think>``, and a chat that asks for tool calls is refused.
- Not supported yet: the NVFP4 checkpoint, prefix caching (a pool that keeps a sliding window
  doesn't prefix cache) and DP replicas (a window pool can't span instances).
- The GPU parity scripts live in ``test/ripple/``: ``run_parity.py`` drives a server and gates
  the result, and ``record_oracle.py`` teacher-forces its replies through the checkpoint's own
  ``modeling_ricursive.py``, in its own venv, with the attention gate and the value residual
  each turned off as controls.

FastConformer ASR (``canary``, ``canary_qwen``, ``parakeet``, ``nemotron_asr``) notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Install with ``.[asr]``. Uploads that libsndfile can't read (M4A, WebM) are decoded by
  torchcodec, which needs FFmpeg.
- One GPU per deployment: both nodes share a rank, and there is no tensor parallelism.
- All four run the FastConformer encoder ``nemotron_duplex`` streams
  (``mstar/model/components/fastconformer.py``), over the whole clip. Canary, Parakeet and
  Nemotron ASR read its dimensions from the HF ``encoder_config``; Canary-Qwen from NeMo's
  ``perception.encoder`` (``FastConformerConfig.from_nemo``), which refuses an encoder
  whose parts NeMo configures as partly causal.
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
  - ``canary_qwen``: English only. ``instruction``
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
- Request knobs must have their documented types (a string ``language``, a boolean
  ``punctuation``, a string ``instruction``), or the request is a 400. Canary-Qwen refuses a
  ``language`` other than English on every route. Canary and Canary-Qwen decode greedily
  unless the request sets ``temperature`` above 0 (``top_p`` and ``top_k`` then apply);
  neither checkpoint declares sampling defaults, so a ``min_p`` or ``repetition_penalty``
  is refused.
  ``max_output_tokens`` may be at most 1014 on Canary (the position table less its 10-token
  prompt) and 1024 on Canary-Qwen; a larger value is refused rather than cut. Parakeet and
  Nemotron decode greedily and read only ``language``: a sent ``temperature`` is logged
  as ignored.

Qwen3-ASR (``qwen3_asr``, ``qwen3_asr_realtime``) notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Greedy, as the checkpoint's ``do_sample: false`` declares. A request reads ``language``
  (an ISO code or name; it forces the language and drops the model's ``language X`` line),
  ``initial_prompt`` (or the request text) as system-turn context, and ``assistant_prefix``,
  a string the model continues from (the realtime route's running hypothesis).
- ``max_output_tokens`` defaults to 4096; a larger value is served up to the text model's
  ``max_position_embeddings`` while the KV pool has room, and past it is a 400. The model
  writes no timestamps: ``/v1/audio/transcriptions`` gives a timed format's segments one
  per window. ``min_p`` and ``repetition_penalty`` are refused.

BAGEL notes
~~~~~~~~~~~

- Text-to-image runs only the CFG branches a denoising step reads: two with
  image guidance off, and the main branch alone outside ``cfg_interval``, about
  -37% latency at 1024² on an H200. A step then computes fewer rows and returns
  the main branch's velocity outside the interval where it used to recombine all
  three, which is equal only up to rounding, so images move. Against the
  previous output on one GPU, seeded text-to-image at the default guidance is
  34 dB PSNR at worst and 45 dB at the median (58–60 dB between two boots of
  one version), text-to-image without guidance 42 / 53 dB, and edits 24 / 40 dB,
  about what two runs of one version's edit differ by (23 / 39 dB).
- A guided request's steps inside ``cfg_interval`` run as a walk of their own,
  ``image_gen_guided2`` (``image_gen_guided3`` with image guidance), between
  ``image_gen_head`` and ``image_gen``. A config that places ``LLM`` per walk
  (``configs/bagel_*_disaggregated.yaml``) lists all four on the rank that
  generates images.
- On a CFG-parallel deployment (``configs/bagel_cfg_parallel*.yaml``) those
  steps run on the guidance GPUs instead, as the ``image_gen_cfg2`` walk
  (``image_gen_cfg3`` with image guidance); the other steps run on the main
  GPU. A config that places ``LLM_cfg_text`` or ``LLM_cfg_img`` with
  ``graph_walks`` names these walks: ``LLM_cfg_text`` and ``combine_cfg`` both,
  ``LLM_cfg_img`` only ``image_gen_cfg3``.

GLM-5.2 / GLM-5.3-Flash (``glm52``, ``glm5_next``)
--------------------------------------------------

- Sampling defaults come from the checkpoint's ``generation_config.json`` (temperature
  1.0 and top_p 0.95 for both), read from a local checkpoint or the cached Hub snapshot;
  the conductor never fetches it, and without it the same values apply. A request may set
  ``temperature``, ``top_p``, ``top_k``, ``min_p``, ``repetition_penalty``,
  ``penalize_prompt`` and ``ignore_eos``. ``min_p`` above 0 is refused, since neither
  checkpoint defaults one. With MTP drafting, only greedy, unpenalized sampling is served.
- GLM-5.2's ``max_tokens`` may run to its serving window (``index_topk``, or
  ``max_seq_len`` with ``dsa_long_context``). GLM-5.3-Flash holds contexts to
  ``index_topk`` (2048), where dense MLA computes exactly what its sparse attention does,
  so ``max_tokens`` above 2048 is a 400 rather than being shortened; under MTP, a
  ``max_tokens`` past the room the prompt leaves is refused too.

Muse Glimmer (``muse_glimmer``)
-------------------------------

Meta's Muse Glimmer 30B: text-and-image chat (video is refused), on ``POST
/generate`` and ``/v1/chat/completions``. bf16 needs one 141 GB GPU, or two
80 GB ones::

    mstar serve muse_glimmer --gpus 0
    mstar serve muse_glimmer --config configs/muse_glimmer_tp2.yaml --gpus 0,1

Three in four layers attend a 2048-token sliding window; every layer writes the
one paged KV cache, and the sliding layers attend it through a windowed
attention resource. The cache keeps only the window of the sliding layers'
K/V (``AttentionConfig.layers``) and returns the pages behind it to the
pool, so a 32k-token request holds about 3.4x less KV than with every layer's
history. A page holds 192 tokens of the full layers, or 64 of the sliding
ones. Such a cache doesn't prefix cache yet, so the configs set
``prefix_cache: false``. Sampling defaults to the checkpoint's (temperature 1,
top_p 0.95, top_k 64); a request's ``min_p`` above 0 is refused (a 400)
unless the checkpoint's generation_config sets a ``min_p``, which turns the
sampler's min-p filter on. ``reasoning_effort`` (``low``, ``medium``, ``high``,
``xhigh``; default ``high``) sets the template's reasoning strength, and
``enable_thinking: false`` means ``low``, as the template has no off: the model
still reasons, at the template's lowest strength. Either, or the template's own
``reasoning_strength`` and ``current_date``, may also come in
``chat_template_kwargs``; a flat key wins. The reply
is the model's channel stream as generated: reasoning addressed ``to=self``,
then the answer ``to=user``. Tool calls are refused for now. Images resize with the
reference's antialiased Lanczos, which torchvision has from 0.27 (torch 2.12); an older
torchvision falls back to PIL's Lanczos with a warning, and its pixels differ slightly.

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
  graph (still compiled, so only the launches are extra): a long video at
  ordinary resolution exceeds the patches, and one over ~64 s with frames
  resized below ~256x256 pixels exceeds the segments.
- An encoder step packs the images and videos of up to 8 requests into one
  replay. Requests that fit alone but would overflow it together run one
  after another in that step. While the block loop is captured, a request too
  large for a replay alone (a long video) gets a step of its own. The packed
  GEMMs' shapes follow the step's total patches, so an image's
  embeddings can differ at the last bit with what it is batched beside.
- On two GPUs, ``configs/qwen3_5_9b_dp2.yaml`` (a whole replica per GPU)
  serves video traffic faster than ``configs/qwen3_5_9b_tp2.yaml``, whose
  vision encoder runs on one rank while the other waits for it in the lockstep
  LLM: 9B on 2xH200 with TP-async on, DP2 gave 12.5-44% more output tok/s on
  30 s and 60 s videos at c=16 and c=32.
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

Text prefill and mixed steps replay captured graphs by total tokens, on the
ladder ``32 ... 8192`` (powers of two, plus 2176 as a mixed step's slack: a
step passes its budget only when its decode rows leave its next prompt less
than ``min_chunk_tokens`` of it, and then by at most its decode rows); a step
pads to its bucket and a longer one runs eagerly. A text-prefill step, one with
no decode rows beside it, also has 1152, 1280, 1536 and 1792, so a prompt a
little past 1024 tokens doesn't pad to 2048; a mixed step captures only the
ladder's powers of two and 2176. 4096 and 8192 stay in the default ladder for a
deployment that raises ``max_batch_tokens`` or turns chunking off, which fills
them exactly. The ladder leaves out slack rungs above them (4352, 8704): such a
deployment adds them to its own ladder, and without them a step just past a
power of two pads to the next one and a step past 8192 runs eagerly. The
serving config's ``prefill_token_buckets`` sets a deployment's own ladder per
walk (see :doc:`serving`): ``configs/qwen3_5_9b*.yaml`` take the powers of two
to 2048 for ``prefill_text`` and ``mixed``, and 1152-1792 for ``prefill_text``,
since the LLM's 2048-token step budget (``MAX_BATCH_TOKENS``) caps both and at
the default ``min_chunk_tokens`` no step passes it. Each bucket adds captures
and startup time.

On ``sp_size`` above 1 the LLM declares a sequence shard (``llm_shard``, its
prefill and mixed walks; decode never splits, and its compiled frame keeps the
bracket out). With its KV pool stored by position (``kv_cache:
{context_parallel: true}``) and the shard's default ``exchange`` (``none``), every step
runs copied, each rank attending its share of the stored context (more KV
capacity, no faster prefill). Over KV stored by head the LLM on SP ranks is
refused unless the deployment sets ``llm_shard: {exchange: ulysses}``, which
has run on CPU only. With ``llm_shard: {exchange: all_gather}`` a
step of at least ``llm_shard.min_tokens`` tokens splits over
the SP ranks: each runs the layers on its part of the prompt, attention gathers
the step's queries, keys and values, and the gated DeltaNet layers trade tokens
for heads (``ulysses``, or ``state_passing`` where
their heads per TP rank don't divide by ``sp_size``). The DeltaNet split runs
eagerly only, so the LLM never splits a captured step: the shard sets its own floor for
each of its text prefill, vision prefill and mixed walks, one past that walk's largest
captured bucket (:doc:`serving`). A prompt that fits a bucket runs copied in its graph, a longer
one runs eagerly and splits, and decode keeps its graphs; ``llm_shard.min_tokens``
only raises the floor. Chunked prefill is off
on SP ranks, so a long prompt splits whole.
The recurrent state is held whole on every SP rank, so decode is unchanged. No
shipped config splits yet: a TP2 x SP2 9B config lands with its GPU parity run.
Prefix caching stays on at one rank as before; above one rank it is off.

A captured decode step builds its rotary tables inside the graph from the
planned positions, rather than staging tables built eagerly before the replay
(that build sat on the per-step critical path: +13% output tok/s at c=32 on
0.8B, H200). Greedy output can differ from earlier releases at near-tie
tokens; prefill and eager steps are unchanged (``knobs: {ingraph_decode_rope: true}``
does the same for captured text prefill and mixed steps).

The prefill, mixed and vision captures compile in Inductor's deterministic mode
whatever ``-O`` says: autotuning them bought no GPU time and took about a fifth of
the cold start. The decode capture keeps the ``-O`` level's tuned kernels. ``knobs:
{compile_deterministic: true}`` compiles it deterministically too: greedy output then
reproduces across server processes and the cold start more than halves. One-row
decode pays for that at the larger sizes (c=1 at 9B on B300: 270 against 286 tok/s;
at 27B TP4 on H200: 151 against 169) and gains at 4B on B300 (372 against 353); from
c=4 up it makes no difference.

``temperature``, ``top_p``, ``max_tokens`` and ``seed`` are the standard fields.
``top_k``, ``min_p``, ``repetition_penalty``, ``penalize_prompt`` and
``enable_thinking`` (default true; the template opens a ``<think>`` block;
``chat_template_kwargs.enable_thinking`` also works) are read by the model but
are not OpenAI fields — pass them via ``extra_body``.
Sampling defaults to the checkpoint's generation_config, under whatever the
request sets. Qwen3.5-27B's sets temperature 0.6, top_k 20 and top_p 0.95,
and Qwen3.8-27B's temperature 1, top_k 20 and top_p 0.95; the 0.8B, 2B, 4B
and 9B ship none, so the engine sampler's defaults apply. A request's
``min_p`` above 0 is refused (a 400) unless the checkpoint sets a ``min_p``,
which turns the sampler's min-p filter on; none of these does.
``max_output_tokens`` may not exceed ``max_position_embeddings``.

With ``resources: {sampler: {enable_logprobs: true}}`` in the deployment's
YAML, ``"return_logprobs": true`` in ``model_kwargs`` on ``/generate`` puts each
token's raw log-prob in its chunk's ``metadata`` (``{"logprobs": [...]}``, one per
token, as ``token_ids``). Qwen3.5 reports them; other models refuse the request
with a 400. Add ``enable_processed_logprobs: true`` and ``"logprobs_mode":
"processed"`` for the log-prob under the distribution it was actually drawn from
(penalties and top-k/top-p applied); greedy requests want ``"raw"``, as a greedy
token scores 0 there. Top alternatives and OpenAI's ``logprobs`` are not
supported yet. On any deployment,
``"return_token_ids": true`` adds each text chunk's sampled ids
(``{"token_ids": [...]}``), stop tokens included, so a trainer need not
re-tokenize the text.

**LoRA adapters.** A ``resources: lora:`` block in the serving config serves
per-request LoRA adapters (PEFT ``adapter_model.safetensors``; the scaling comes
from the ``adapter_config.json`` beside it). The OpenAI ``model`` field picks one,
as in vLLM: the base model's name (or its HF repo id, or no ``model`` at all) runs
the base model, an adapter's name runs that adapter, and any other name is a 400.
``GET /v1/models`` lists the adapters beside the base model. On ``POST /generate``,
pass ``model_kwargs: {lora_adapter: <name>}``, checked the same way. Rows on different adapters, and on
none, share a batch::

    resources:
      lora:
        max_slots: 4        # adapters resident on the GPU at once
        max_rank: 16        # the largest rank served; slots are sized to it
        cpu_capacity: 16    # adapters kept in pinned host memory
        adapters:
          sql: /adapters/qwen3_5-4b-sql/adapter_model.safetensors
          chat: /adapters/qwen3_5-4b-chat   # a PEFT directory names its adapter_model.safetensors

Every adapter the config names is checked at startup, before any worker starts (LoRA
layers present, rank within ``max_rank``, only the targets below, shapes that fit them),
and one that fails stops the server with an error naming it.

Adapters may touch every module PEFT adapters for Qwen3.5 on the Hub touch:

* every linear of the text stack, as ``target_modules="all-linear"`` picks them: the
  full-attention layers' ``q_proj`` (its output gate included), ``k_proj``, ``v_proj``
  and ``o_proj``, the gated DeltaNet layers' ``linear_attn.in_proj_qkv`` (or
  ``in_proj_q`` / ``_k`` / ``_v`` split, as Tinker trains them), ``in_proj_z``,
  ``in_proj_a``, ``in_proj_b`` and ``out_proj``, and every layer's ``gate_proj``,
  ``up_proj`` and ``down_proj``;
* the LM head (``lm_head``, or Tinker's ``unembed_tokens``) and the token embedding
  (PEFT's ``lora_embedding_A`` / ``_B`` on ``embed_tokens``). With tied weights (0.8B,
  2B, 4B) each delta stays on its own side, as PEFT applies them;
* the vision tower, as unsloth's vision fine-tunes adapt it: each block's
  ``attn.qkv``, ``attn.proj``, ``mlp.linear_fc1`` and ``mlp.linear_fc2``, the merger's
  ``linear_fc1`` / ``linear_fc2``, and the conv ``patch_embed.proj``. One spec covers
  both nodes, sized by the one ``lora:`` block; the vision encoder node has a pool of
  its own (``lora@vision_encoder``, beside the LLM's ``lora@LLM``) and runs the
  request's adapter too.

Module names may be in any layout PEFT writes for the model: the released checkpoint's
(``model.language_model.layers.*``, ``model.visual.*``), the text-only causal LM's
(``model.layers.*``), the inner model's (``language_model.layers.*``) or the bare text
model's (``layers.*``). An adapter on the MTP head is accepted and that part ignored
(the head never runs). The DeltaNet in-projections are one fused GEMM, so their deltas
are one shrink and one expand per layer, added before the conv reads the output.

Refused with a 400 that says why, when the adapter is named: DoRA (``use_dora``),
``modules_to_save`` (full copies of a module, e.g. a classifier ``score`` head or a
retrained ``lm_head``), trainable tokens, bias LoRA, other PEFT methods (OFT), a module
outside the list above, a module whose shapes do not fit (an adapter for another size),
and a linear-shaped LoRA on ``embed_tokens`` (``lora_A.weight`` of ``[r, hidden]``, as
some Tinker exports name a tied head's LoRA; rename its keys to ``lm_head`` to serve it
on the head).

Every target is sized into each slot and runs its kernels on every step that has an
adapter row, whether that row's adapter uses it or not. The LM head and embedding are
vocabulary-sized: at ``max_rank: 16`` and 4 slots, about 40 MB each per GPU (the head's
split over TP ranks, the embedding's whole on each). So with ``runtime_updates`` off
(no adapter can arrive after startup), the pools keep only the targets the configured
adapters touch, and a vision encoder whose adapters are all text-only builds no pool.
With it on, every target is kept unless the YAML lists the ones to keep::

    resources:
      lora:
        targets: [qkv_proj, o_proj, in_proj_fused, out_proj, gate_up_proj, down_proj,
                  qkv, attn_proj, mlp_linear_fc1, mlp_linear_fc2, merger_linear_fc1,
                  merger_linear_fc2, patch_embed_proj]

An adapter deployment captures and compiles the LLM node as one without adapters does,
each bucket twice: with the LoRA kernels, and base-only for steps none of whose rows names
an adapter. The vision encoder's block loop is captured base-only; a step with an adapter
row runs it uncaptured. The LoRA pool keeps no resident table yet, so a ``lora:`` block
turns resident decode planning off on the LLM node (the server logs it): every decode
step, base-only ones included, is planned on the host as before resident planning.

Every size takes the same ``resources: lora:`` block; ``max_rank`` must cover the
largest adapter served. Example blocks (Hub adapters, ranks as published; no shipped
config has one, so add the line to that size's serving config)::

    # e.g. in configs/qwen3_5_0.8b.yaml: a vision fine-tune (blocks and merger, rank 16)
    lora: {max_slots: 4, max_rank: 16, adapters: {agnews: <hammamwahab/jev-agnews-lora>/adapter_model.safetensors}}
    # e.g. in configs/qwen3_5_2b.yaml: vision with the conv patch embed (rank 32)
    lora: {max_slots: 4, max_rank: 32, adapters: {comma: <wjbmattingly/comma-qwen-3.5-2b>/adapter_model.safetensors}}
    # e.g. in configs/qwen3_5_4b.yaml: LM head and embedding (rank 8), Tinker (rank 32)
    lora: {max_slots: 4, max_rank: 32, adapters: {pirate: <William-Gao1/qwen3.5-4b-pirate-lora>/...,
                                                  oneinbox: <dKhRr/oneinbox-qwen3.5-4b-lora>/...}}
    # e.g. in configs/qwen3_5_9b.yaml or qwen3_5_9b_tp2.yaml: released layout (16), Tinker (32)
    lora: {max_slots: 4, max_rank: 32, adapters: {nimble: <bespokelabs/Bespoke-Nimble-9B>/...,
                                                  rlvr: <lucabaroni/qwen3.5-9b-rlvr-reward-hacking>/...}}
    # e.g. in configs/qwen3_5_27b_tp4.yaml: text and vision (rank 16)
    lora: {max_slots: 4, max_rank: 16, adapters: {tcg: <heftyTuna/qwen35-27b-tcg-lora>/...}}

Under TP the LLM's targets split as its linears do (the 27B's single K/V head per rank
at TP4, and the DeltaNet ``a`` gate padded per rank, included); the vision encoder runs
on one GPU in every shipped config.

Both frontends serve LoRA: a request's ``model`` names the adapter it runs under
(``ServedModels``), and ``/v1/models`` lists the adapters beside the base model.
Both serve runtime updates (below) too: ``mstar-server`` selects the adapter itself,
and the API server pushes it the served set after every change, so a request pays no
extra round trip for it.

Adapters can also be loaded and unloaded while the server runs, with vLLM's
endpoints and bodies. They make the server read paths a client names, so they are off
unless the ``lora`` block opts in, and answer 404 otherwise (as vLLM's do without
``VLLM_ALLOW_RUNTIME_LORA_UPDATING``). The block may name no adapters at all
(``resources: {lora: {max_slots: 4, runtime_updates: true}}``)::

    curl -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
        -d '{"lora_name": "sql", "lora_path": "/adapters/qwen3_5-4b-sql"}'
    curl -X POST localhost:8000/v1/unload_lora_adapter -H 'Content-Type: application/json' \
        -d '{"lora_name": "sql"}'

``lora_path`` is a PEFT directory (its ``adapter_model.safetensors`` is read) or a
safetensors file, readable at the same path by the API server and every worker. A load is checked
first (LoRA layers present, rank within ``max_rank``, only the targets above, shapes
that fit them), then read by every worker that hosts a node with a LoRA pool (the LLM's
or the vision encoder's); the name is selectable, and listed by ``/v1/models``, once all
of them have it. A load that fails on any worker is undone on all of them and answered
with a 400, as is a name that is already loaded. A refusal says why the adapter cannot be
served (its rank, its modules, its shapes); a file that cannot be read at all gets the
same answer whether it is missing, unreadable or not an adapter, and the details go to
the server log. A worker that does not answer within ``runtime_update_timeout_s``
(default 120) fails the load with a 504, and the load is undone everywhere, so the name
is free to try again.

To change an adapter's weights, unload it and load it again: each adapter is pinned to
the digest of the weights checked when it was configured or loaded, and a worker that
reads other weights from its path (the file overwritten in place) refuses them, on every
rank, rather than serve them under the old name and prefix-cache entries. An unload stops
new requests on the name at once (they get the 400 an unknown adapter gets; an unknown
name in the unload is a 404); if the conductor refuses it, the name is selectable again.
Requests already running on it finish on its weights, after which its slot and host copy
are freed; loading the name again before then is a 400 that says so.

Qwen3.8-27B (``qwen3_8_27b``) is the Qwen3.5 architecture and runs on the same
code, served from ``configs/qwen3_8_27b_tp4.yaml`` (TP4)::

    mstar serve qwen3_8_27b --gpus 0,1,2,3

Its ``output_gate_type: swish`` names the gated DeltaNet norm's activation,
which is the SiLU every Qwen3.5 checkpoint already uses. transformers ignores
the key and hard-codes SiLU. Video is still
refused and the MTP head is skipped. The chat template's new
``reasoning_effort`` and ``preserve_thinking`` are not forwarded: every
request gets the template's defaults (``xhigh`` and preserved), whatever the
client sends, and a request that sends one is logged as ignoring it.

MiniCPM-o 4.5 (``minicpm_o``)
------------------------------

Text, image and audio in; text, or text and 24 kHz speech, out, as upstream's half-duplex
``chat()``: the LLM (Qwen3-8B) writes the reply, a TTS reads it into speech codes, and a
streaming flow-matching vocoder (Step-Audio2's token2wav) turns them into audio. Ask for
speech with ``output_modalities`` including ``"audio"``; the bundled ``default`` voice is the
only one. A spoken reply's system message is upstream's voice-clip prompt, so a spoken
request with a system prompt of its own is refused (400) unless it sets ``voice_prompt:
false``, which uses the request's ``system_prompt`` instead, as vllm-omni does.
``streaming_generate``, video input and full duplex are not supported.

``configs/minicpm_o.yaml`` runs three workers on one GPU (LLM and encoders, TTS, vocoder);
it sets ``mps: true``, so their kernels overlap (see *Several workers on one GPU* in :doc:`serving`). Each spoken
reply holds a slot of each ``t2w_*`` resource (~0.4 GiB in all) while it speaks, so their
``max_slots`` cap the replies being voiced at once. ``t2w_state`` has one more than each of the
three ``t2w_*_kv`` (its sink); startup refuses counts that disagree. The slots don't hold
back the TTS, which keeps a spoken reply's ``tts_kv`` pages (up to 32) until its last code,
so ``max_concurrent_requests`` caps the requests in flight and startup refuses a ``tts_kv``
that can't hold that many full replies: overfilled, the TTS would stall. A reply's LLM
pages have no such bound, so ``llm_kv`` has host pages (``cpu_offload_pages``): a full
pool moves a request to the host until those fill. At the cap of 32, with 2048 tokens
out on an H200 (one run per size), a burst of prompts around 3.7k tokens or longer still
waits, 2.5k does not (follow-up: reserve a request's whole layout).

The vocoder's DiT runs its GEMMs, convolutions and attention, and keeps its caches, in
float16; its residual stream, its norms' arithmetic, the Euler solver, the token encoder
and HiFT stay float32. Its mel is within about 0.1% of a float32 DiT's, and the served
speech's Whisper WER is unchanged. HiFT up to its spectrum is compiled (``compile_regions``),
and the TTS's decode capture runs cuBLAS GEMMs (``compile_mode="default"``).

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
H200, set ``resources.kv_cache.max_num_pages`` to 2048. It keeps only the
sliding layers' window per request, and no prefix cache yet. Tokenizer and config
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
64 (``benchmark/command_a_plus/bench_http.py``, measured before the sliding layers
kept only their window); ``validate_checkpoint.py``
there checks parity against Hugging Face, with ``--images`` for image prompts.
Not supported yet: FP8 weights and prefill CUDA graphs.

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
  synthesize a phoneme string directly). An empty ``voice`` / ``lang_code`` or a
  ``speed`` outside 0.25-4.0 is a 400. ``seed`` fixes the vocoder's noise (absent, the
  server derives one from the request id), so a seeded request run at the same batch
  layout repeats exactly; batched differently, its noise is the same but padding rounding
  moves the waveform's phase, as it does in the reference.
- Deployment-wide options go in the YAML's ``model_kwargs`` (see ``configs/kokoro.yaml``):
  ``lang_code`` fixes the G2P language, ``espeak_fallback: false`` disables the espeak-ng
  fallback even when it is installed, ``chunk_target_phonemes`` and
  ``first_chunk_target_phonemes`` set the sentence packing, ``text_buckets``,
  ``frame_buckets``, ``capture_batch_sizes`` and ``max_batch_frames`` shape the CUDA
  graphs captured at start-up (fewer buckets on a smaller GPU), ``compile_decoder: false``
  skips the ``torch.compile`` of the vocoder (start-up in seconds instead of minutes, about
  half the throughput on an H100) and ``decoder_dtype: bfloat16`` runs the vocoder trunk
  in bf16 (about 13% more throughput at concurrency 32 on an H100 and 20-25% at
  concurrency 32-64 on an H200, Whisper WER unchanged, with the harmonic source and the
  iSTFT kept in fp32; the parity test covers fp32 only; ``configs/kokoro_bf16.yaml``).
  Each BiLSTM's two directions run as one block-diagonal recurrence, and cuDNN times
  the conv algorithms of both captured halves once per bucket, at capture
  (``NodeSubmodule.capture_cudnn_benchmark``).
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

- The default ``configs/qwen3omni_2gpu.yaml`` holds 1,024 GPU pages (128k tokens)
  of Thinker KV plus 1,024 host pages it offloads to. On H200-class GPUs, when
  concurrent requests hold more than 128k tokens of Thinker KV, serve
  ``configs/qwen3omni_2gpu_large_kv.yaml``, which keeps 2,048 pages (256k tokens)
  on the GPU and no host pages; it should fit beside its captures on an H200.
  On 80 GB GPUs such as the H100, serve ``configs/qwen3omni_2gpu_h100.yaml``.
  Its 512-page Thinker pool (64k tokens) was sized on an H100 when the encoders shared the Thinker's GPU and its
  text ladder ran to 8192: beside 1,024 pages 27 Thinker captures, all among those
  of 4096 tokens and more (text and vision), ran eagerly, and 640 pages still left
  two vision buckets out. Chunked prefill now caps a Thinker step at 1,024
  tokens, so those large buckets no longer replay, but they are still captured
  and still take that memory; whether 1,024 pages fit with today's ladders and
  encoder placement is unmeasured. The config runs the audio and vision encoders on the Talker's GPU,
  so an encoder step doesn't wait behind a Thinker step.
- None of the three configs sets ``max_concurrent_requests``. With the Thinker's
  prefix cache on, a decode step that finds the GPU pool full offloads a running
  request to the host pages. Only when that frees nothing does it hold requests
  until they time out: with no host pages (``large_kv``), with the host pages full,
  or when the running requests' pages are mostly a prefix they share (see the
  prefix cache under "Config files" in :doc:`serving`). Where one of those can
  happen and concurrent requests can together hold more Thinker KV than the GPU
  pool, set the cap to the GPU pages divided by the pages of the longest request
  served, prompt and output together, at 128 tokens a page: at the full 32k
  context, 8 for ``large_kv``, 4 for the default and 2 for the H100 config.
- The audio and vision encoders attend through ragged attention resources
  (``audio_attn`` within each 8 s window, ``vision_attn`` within each image or
  video frame group), FlashInfer by default, so no ``flash-attn`` install is
  needed. A deployment can pick the kernel under ``resources:`` in its YAML,
  e.g. ``vision_attn: {flashinfer_backend: cutlass}``. Each encoder's block stack
  is captured as a CUDA graph per token-count bucket; a request past the
  largest bucket (8192 patches, about a 1920x1080 image; 4096 audio tokens)
  runs it eagerly.
- A request with no audio output decodes on its own walk, ``thinker_text_decode``:
  its token stays on the device between steps and reaches the client inline, as
  Qwen3.5's does, and nothing goes to the Talker. A request with audio output
  keeps ``thinker_decode``, which streams every step's states to the Talker. A
  captured step of either walk takes token ids and builds its embeds and rotary
  tables in the graph.
- The audio encoder is the AuT node Qwen3-ASR runs, fed the API server's
  log-mel features: a step's clips (up to 8 requests) go through one packed
  forward, up to 4096 tokens. A step over 4096 tokens runs its clips one by
  one, each replayed if it fits a bucket. A clip's last 1 s chunk is
  zero-padded to a full chunk before the convolutions, as HF's encoder pads
  it whenever its batch holds a full chunk, so a clip under 1 s encoded alone
  differs from HF's: in its last conv row and, through the windowed
  attention, in every embedding of the clip. A clip's embeddings can differ at
  the last bit with what it is batched beside, since the step's total tokens
  pick the captured bucket (or the eager GEMMs' shapes), and a prefix-cached
  audio prefix keeps the embeddings it was first computed with, so a later
  request reusing it inherits that first batch's rounding.
- A prompt runs one Thinker prefill per attachment: each attachment's row also
  holds the text before it, and the last one the text after it, so text, a clip
  and text is one step. A prompt with no attachment is one text prefill.
- Streamed audio starts after 12 codec frames (0.96 s of audio), then 13, then
  25-frame chunks behind 25 frames of left context. Playback stays gapless
  while the Talker produces frames at about 1.5x real time over the first 50
  frames (about 19 frames/s per request); the Talker batches at most 32
  requests, so requests past that queue and can fall below it and pause. Set
  ``codec_chunk_schedule: []`` under the deployment YAML's ``model_kwargs`` to
  start with a 25-frame chunk instead: first audio comes later, and the
  Talker needs only about 1.0 to 1.3x real time.
- ``configs/qwen3omni_thinker_tp4_sp2.yaml`` (GPU-tested before its last merge of main; a rerun is pending) runs the
  Thinker on 8 GPUs as 4 TP ranks by 2 SP ranks, for long prompts. Its KV pool
  is stored by position over each SP pair (``context_parallel``). A prefill of
  at least ``min_tokens`` tokens (on the ``thinker_shard`` resource) splits over
  the pair: each rank runs the layers on its half of the tokens, and attention
  gathers the step's queries, keys and values (``exchange: all_gather`` on
  ``thinker_shard``; with the Thinker's default, ``none``, it runs every step copied, and over KV
  stored by head a Thinker on SP ranks is refused unless the deployment sets ``exchange: ulysses``,
  which has run on CPU only). Decode and
  shorter prefills run copied on both ranks, each attending its half of the
  stored context, and the two partials are merged. A captured prefill decides on its bucket's size.
  ``min_tokens`` is 16384: on 8 H100s all-gather prefilled faster than
  the copied path from 16K-token prompts and slower at 4K to 8K. The config serves prompts up to the
  model's own 65536 positions (greedy outputs were checked to 60K; the 64K and 128K rows measured time and
  memory only). Chunked prefill is off on SP ranks, so a prompt prefills in one step.
- Numerics: the Thinker's q/k RMSNorm and multimodal RoPE run in one kernel
  with the KV write, in fp32 throughout and rounded once, where the unfused ops
  (and Hugging Face) round to bf16 after the norm and after each step of the
  rotation: no less accurate, but not bit-exact. (A Thinker step that splits
  over SP ranks runs unfused, through the exchange, and so does a copied prefill; a decode step keeps the fused kernel.) The code predictor (shared
  with Qwen3-TTS, see its notes) runs each projection with its RMSNorm folded
  in and its residual add in the same Triton kernel (``cp_linear``), and its
  attention as another (``cp_attend``), neither bit-exact with the unfused
  ops. Its 15 depth steps feed each sampled code into the next, so small
  differences can change later codes at near-ties.
- The Thinker's text and audio prefill replay captured graphs on the ladder
  ``128 ... 2048`` tokens, and a longer step runs eagerly. The serving
  config's ``prefill_token_buckets: {Thinker: {prefill_text: [...]}}`` sets a
  deployment's own ladder (see :doc:`serving`). Each bucket adds captures,
  graph memory and startup time.
- The MoE layers route through the fused softmax-top-k kernel
  (``TopKRouter(fused_softmax_topk=True)``), which gives equal probabilities to
  the lower expert id where ``torch.topk`` may pick another. On one GPU a
  decode batch of up to 16 tokens also gets its block alignment from that
  kernel, unless it is small enough to run its experts as per-pair GEMVs,
  which need none: up to the ``moe_decode`` kernel table's crossover for the
  GPU and expert shape (``mstar/kernels/tables/moe_decode``), else the
  built-in 7, swept on an H100 (aligned from 8). On an H200 the TP1 Thinker
  runs no per-pair GEMVs: it takes the grouped GEMMs, aligned, from 1 token.
  At 1-2 tokens per-pair decode was 5-11% slower than grouped on one H200 box
  (torch 2.10, Triton 3.6) and 4-6% faster on another (torch 2.12.1, Triton
  3.7.1); at 4 tokens grouped beat the built-in 7 on both. A crossover keyed
  by GPU name can't suit both boxes, so it takes grouped, which trails
  per-pair by at most about 6% at 1-2 tokens, whereas per-pair would trail
  grouped by up to about 11%.
  The Thinker at TP2 (expert width 384) has no H200 entry yet, so it keeps 7
  there, unmeasured.
- ``configs/qwen3omni_pd_2gpu.yaml`` runs ``qwen3omni_pd_disaggregated.yaml``'s
  prefill/decode split on two GPUs: the prefill Thinker shares rank 0 with the
  encoders, the Talker and Code2Wav, and rank 1 runs the Thinker's decode. At
  concurrency 1 its text matches ``configs/qwen3omni_2gpu.yaml``'s.

CosyVoice3 notes
----------------

- Install with ``pip install -e '.[cosyvoice3]'`` and serve with
  ``mstar serve cosyvoice3 --gpus 0``. Every request clones a voice: a
  ``ref_audio`` clip (with its transcript in ``ref_text`` for zero-shot
  cloning) or a preset from ``voices_dir`` (see ``configs/cosyvoice3.yaml``).
  ``instructions`` carries a style instruction. Output is 24 kHz.
- A streamed row solves only each hop's new frames, reading the K/V of the
  frames before them from ``resources.flow_kv`` (about 1.8 MB a frame). Past
  ``flow_cache_frames`` (768 by default, 15 s of prompt and speech) a row goes
  back to re-solving every frame. The default pool holds that for 32 requests,
  about 41 GiB. ``flow_cache_frames: 0`` turns it off, and so does a float32
  estimator, which FlashInfer's kernels can't run.
- ``configs/cosyvoice3_split.yaml`` runs the flow and vocoder on their own
  worker on the same GPU, with MPS and batches of 16, for throughput.
- Keep the reference clip short, a few seconds like CosyVoice's own 3.5 s
  prompt clip: with a 23 s clip as the preset, round-trip word error rate rose
  from 1.4% to 13.4%, with words dropped mid-sentence.
- Request knobs are strictly typed: ``ras_window`` and ``n_cfm_timesteps`` take a whole
  float but not a string or a boolean; ``top_k`` (1-128) and ``max_new_tokens`` follow
  the server's integer rule (:ref:`generation-parameters`); an out-of-range value is a
  400. ``voice``, ``instruct`` and ``ref_text``
  must be strings. ``max_new_tokens`` above the deployment's speech-token limit is
  refused, not cut. ``seed`` fixes the LM's sampling; the flow and vocoder noise is
  fixed, as in the reference. ``speed`` has no effect and is logged as ignored.
- On ``/v1/audio/speech``, text longer than one piece is synthesized as ordered
  sentence pieces, as the reference frontend does: about 80 tokens (320
  characters) of English, or 80 characters of text with Chinese, each piece
  cloning the same voice. In one request the LM ends long text early or
  garbles its end. A Chinese sentence over 300 characters, which the
  reference keeps whole, is also cut at its commas, so that each piece stays
  under the LM's 512 tokens of transcript and text.
  ``sentence_chunking: false`` sends the text whole.
- Japanese written in kana alone, with spaces between words, is read poorly,
  as it is by the official model: in tests both garbled common words such as
  ゆっくり and ありがとうございました on every seed.
- The flow's DiT estimator runs in bfloat16 by default; the Euler state, the
  rest of the flow and the HiFT vocoder stay float32 (HiFT's F0 predictor
  float64).
  ``model_kwargs: {flow_estimator_dtype: float32}`` in the deployment YAML runs
  the estimator at the reference's precision instead, at lower throughput.
- The flow's Euler solve is captured at startup as CUDA graphs, one per frame
  bucket and batch size (``flow_capture_frame_buckets``,
  ``flow_capture_batch_sizes``, ``flow_capture_batch_frames``), which takes GPU
  memory and startup time. ``model_kwargs: {flow_capture_frame_buckets: []}``
  captures none, and every solve runs eagerly.
- The streaming vocoder is captured the same way, one graph per window bucket
  and batch size (``hift_capture_frame_buckets``, ``hift_capture_batch_sizes``,
  about 3.3 GB at the defaults); each stream's mel tail and phase carry live in
  the ``hift_state`` pool, a slot per request (``resources: hift_state:
  {max_slots: N}``, one more than the requests). A chunk past the largest bucket
  runs as several steps. ``hift_capture_frame_buckets: []`` runs the vocoder
  eagerly.
- The LM's prefill is captured with its decode, per prompt token bucket and
  batch size (the serving config's ``prefill_token_buckets: {llm: {prefill_voice:
  [...]}}``, see :doc:`serving`, and ``lm_prefill_batch_sizes``; the LM's graphs
  take about 130 MB together). A prompt past the largest bucket runs eagerly, and
  ``lm_prefill_batch_sizes: []`` captures no prefill. The flow's new-frame region
  keeps ``model_kwargs.flow_new_frame_buckets``, which the key can't set.

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
  not encoded again once its conditioning is memoised; named, persisted voices
  arrive with the shared voice registry.
  A new clip of up to 20.5 s replays the engine's capture of its frame bucket
  (``RefEncoderSubmodule.FRAME_BUCKETS``), padded to the bucket and encoded as
  the unpadded clip would be, up to kernel choice; a longer clip runs the same
  encode eagerly.
- Text layout follows the reference defaults: CustomVoice and VoiceDesign put
  the whole text in the prefill; Base feeds it one token per frame. Override
  per request with ``non_streaming_mode``.
- Live text: with ``non_streaming_mode: false`` (Base, or x-vector-only
  cloning), a ``/generate/ws`` call opened with ``end_input: false`` takes the
  rest of its text as ``append`` messages, one token per frame as it arrives,
  through the ``text_in`` client stream (see :doc:`clients`). Each append is
  tokenized alone, which matches tokenizing the whole text when the cut falls
  before a space or at punctuation. In ``non_streaming_mode`` and for an
  in-context clone, the prefill's pairing fixes the text's length and where TTS
  EOS sits, so a live opening (``end_input: false``) gets a 400; send those
  one-shot.
- ``/v1/audio/speech`` inputs of 600 or more characters are synthesized as
  ordered sentence chunks of about 400 characters (one Talker request each,
  two kept in flight while the current one streams); set
  ``sentence_chunking: false`` (or ``true`` for shorter texts) per request.
- Audio streams in a ramp of codec chunks: the first frame is decoded on its
  own (first audio one Talker step after prefill), the next chunks add 3, 8
  and 16 frames, then 25 frames a chunk. The decoder is causal, so the codec
  decodes every frame once: per stream it keeps each causal conv's last input
  samples, the transformer's sliding-window keys and values (71 per layer) and
  the frame count, about 5 MiB in float32, in the ``codec_state`` recurrent
  pool. ``resources: {codec_state: {max_slots: N}}`` sizes it for N - 1
  streams (one slot is the pool's sink); 256 streams by default. Each chunk
  width is a CUDA-graph bucket captured for batch sizes 1 to 32 (densely: the
  codec's cost is per row, so a padding row costs as much as a real one). A
  step takes ready chunks of any width and pads them to its widest, since a
  step of its own for each width costs more than the padding. A voice clone's
  stream leads with the reference's last 72 frames, decoded with the first
  chunk to warm the state on the voice and trimmed from its audio.
  The SnakeBeta activations run as one fused kernel. A deployment sets the
  ramp and the steady chunk with ``model_kwargs: {codec_chunk_schedule: [1, 3,
  8, 16], codec_chunk_frames: 25}``. ``configs/qwen3tts_1p7b.yaml`` opens with
  4 frames ([4, 8, 16]): on an H200 the streaming SLO (first audio within 200
  ms, under 10 ms of pauses) then holds 101 Seed-TTS sentences a second
  against 90 with [1, 3, 8, 16], which halves idle first audio (15 vs 31 ms
  p50 at 1 req/s).
- Talker prefill, the clone prefill included, replays a packed CUDA Graph for
  the smallest token bucket (32 to 1024 tokens) that holds the batch; a clone's
  reference frames join the codec stream in ``postprocess``. Prefill and decode
  are one combined walk (``talker_mixed``): up to four new prompts ride a decode
  step instead of a prefill-only step every decode row waits behind, replaying
  a packed capture sized to the decode batch. A decode-only step uses the
  whole-walk CUDA Graph (up to 128 rows a step), with the 15-step CodePredictor
  loop captured inside it; request-local EOS suppression is
  carried as a graph tensor input so replay does not consult capture-slot dummy
  request state. Residual ``code_predictor_*`` sampling (``subtalker_*`` is a deprecated alias) is per-request through the
  ``code_predictor`` aux sampler, so custom values neither block batching nor
  fall off the graph. On the 1.7B checkpoints the CodePredictor projects the
  Talker-width inputs through ``small_to_mtp_projection`` before its depth loop.
  It is the same component as Qwen3-Omni's code predictor
  (``ResidualCodePredictor`` in ``mstar.model.components.code_predictor_kernels``):
  the depth sequence opens with the Talker hidden state in a one-token pass
  and takes one token per group after it, five launches a layer. Each
  projection is one ``cp_linear`` launch with the RMSNorm folded into its
  weight (a row scale from partial sums of squares its producer recorded) and
  the residual add or the SwiGLU in its epilogue; the q/k norm, RoPE, KV write
  and attention are one ``cp_attend`` launch, and the frame's 16 codec
  embeddings are summed in one launch per group. Where the predictor projects
  its inputs (``small_to_mtp_projection``, the 1.7B), every embedding table a
  later depth step reads is kept projected. None of this is bit-exact with the
  unfused ops: the sampled residual codes can differ at near-ties. Each Talker
  keeps the sum it had before the two shared the component: Qwen3-TTS
  accumulates in the model dtype, rounding after every add (the reference sums
  in fp32 and rounds once), and Qwen3-Omni in fp32, rounded once. With
  ``depth_kernel: true`` in ``model_kwargs`` on a Hopper GPU with 128 or more
  SMs, a step of up to 64 rows whose residual draws the graph sampler can hand
  over runs the whole loop, sampling included, as one persistent kernel
  (``mstar.model.qwen3_tts.components.depth_frame``). The kernel needs Triton
  3.7, which comes with torch 2.12. Under torch 2.11's Triton 3.6, startup logs
  why and runs the loop.
- The Codec does not run beside the Talker on a side stream (``side_stream``,
  see :doc:`serving`): its windows read and write their streams' ``codec_state``
  rows, and the side lane only takes a node that holds no resources, since it
  neither admits nor plans a step ahead. No Qwen3-TTS config sets the field.
- ``configs/qwen3tts_1p7b_split.yaml`` (and ``configs/qwen3tts_split.yaml``
  for the 0.6B) runs the Codec on a second worker that
  shares GPU 0 (``rank_devices``) and caps the Talker KV pool so both fit.
  On its own it lowers first-audio latency at high concurrency; with
  ``mps: true`` in the YAML (see :doc:`serving`; ``qwen3tts_split.yaml`` sets
  it) the two workers' kernels also overlap, which raised c=32 throughput by
  about a fifth on an H100.
- ``configs/qwen3tts_base.yaml`` keeps the Base model's nodes on one worker, on
  any GPU. ``configs/qwen3tts_base_split.yaml`` runs the Talker, the Codec and
  the RefEncoder on workers of their own, all on GPU 0, as clients of one MPS
  daemon (``mps: true``, NVIDIA only), so the Talker's decode steps overlap codec
  windows and reference encodes. On an H200, serving seed-tts voice clones (two
  boots each), it served 83-87 audio seconds a second at c=8 and 189-215 at
  c=32, against 50-81 and 113-165 on one worker and 46-61 and 76-140 for
  sglang-omni; at c=1 it is level with one worker (16.6-17.1 against 15.0-18.7).
  Both start streaming after one frame and ramp to steady 16-frame chunks
  (``codec_chunk_schedule: [1, 2, 4]``, ``codec_chunk_frames: 16``), and size
  the Talker's KV pool at 2304 pages, which fits an 80 GB GPU. Every Qwen3-TTS
  request reserves its whole footprint at admission, its prompt plus its length
  cap: a streamed one can't pause (``unpausable_outputs``), and none can be
  rebuilt after a drop (its prompt is rows, not ids) without a host pool. That is
  a few pages for a clone of a sentence, about 66 for a request that asks for the
  checkpoint's 8192 frames (34 of those fit).
- A clone without ``max_output_tokens`` stops within a budget set by its target
  text: twice the slowest natural stop of the 1.7B checkpoint, 1.38 s + 0.39 s a
  text token (sglang-omni's fit; 2 of 78k natural stops of M*'s seed-tts clones
  ran past it), about 18 s for a 20-token sentence. Without it, a clone that
  runs away would decode to the checkpoint's 8192 frames (11 minutes); about 1
  in 7000 did. A request's own ``max_output_tokens`` replaces the budget, and
  live text (``text_in`` left open) has none, since its length isn't known at
  the prompt.
- The Talker's layers run ``Qwen3Attention``, so each layer's q/k norm, RoPE and
  KV write are one kernel (``qk_norm_rope_kv_write``). It reads the RoPE theta off
  the position resource, which declares the checkpoint's.
- The Talker's decode rows run on FlashInfer's CUDA-core decode kernel. Every
  checkpoint has 16 query heads over 8 KV heads, and the tensor-core kernel pads
  the 2 query heads per KV head out to a tile. On an H200 the 1.7B serves 1.01x
  to 1.03x the audio from c1 to c128 with the same WER. It's unmeasured on the
  0.6B and on B300. ``talker_attn: decode_tensor_cores: true`` in a deployment's
  YAML selects the tensor-core kernel.
- The codec decoder runs in float32 by default: a stream's audio is the
  reference decoder's whole decode within 1 LSB of PCM16. ``model_kwargs:
  {codec_dtype: bfloat16}`` selects a bf16 codec (RMSNorm and RoPE stay
  float32), which ``configs/qwen3tts_1p7b.yaml`` serves. Against the float32
  whole decode of the 1.7B's frames it gives 35 dB SNR median (20 dB worst),
  1.0 dB mel cepstral distortion, 0.9999 speaker cosine and the same WER, and
  the streaming SLO holds 101 Seed-TTS req/s against float32's 90 (H200). Its
  dilated convs run as undilated ones over their interleaved subsequences:
  cuDNN takes some bfloat16 dilated convs (384 and 768 channels) through a
  direct kernel without tensor cores, about 100x slower on an H100. On a B300
  the bf16 codec measured slower while those convs still ran dilated; measure
  there before switching. The decode is compiled (Inductor) twice per chunk
  width, batch 1 apart: its batch dimension is dynamic.
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
       --language English --concurrency 8 --repeats 3 --out results/mstar_c8.json \
       --save-audio-dir results/mstar_c8_wav

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
  ref_text="...")``. ``voice`` and ``speed`` are logged as ignored.
- Sampling defaults follow the reference (temperature 0.8, top-p 0.8, top-k 30);
  ``do_sample: false`` is greedy. The order differs: M\*'s sampler applies the
  temperature first and takes top-k and top-p over the tempered probabilities,
  while the reference takes top-p over the untempered ones, so at the same
  values M\* keeps a slightly different set of tokens per step. The top-p
  boundary differs as well: M\*'s sampler (FlashInfer) keeps the token whose
  probability carries the running total past top-p, while the reference drops
  it unless it is the most likely token. ``max_new_tokens`` (default 2048, at most
  4096 frames of 2048 samples at 44.1 kHz) and ``ignore_eos`` pass through.
- The codec decodes each chunk's new frames from its stream's decoder state
  (the ``codec_state`` recurrent pool, 64 streams by default, as many as the AR
  decodes in a step, about 0.55 GiB; size it with
  ``resources: {codec_state: {max_slots: ...}}``), so every frame is decoded
  once. It decodes with TF32 by default: TF32 roughly halves its conv stack
  and the audio stays within ~60 dB SNR of the exact decode, both measured on
  the earlier windowed codec, not yet on the frame-once codec. With
  ``model_kwargs: {codec_exact_fp32: true}`` it runs in fp32 with TF32 off and
  the streamed audio matches the whole-utterance decode to within one int16 step.

VoxCPM2 notes
-------------

- Install with ``pip install -e '.[voxcpm2]'`` and serve with
  ``mstar serve voxcpm2 --gpus 0``. ``/v1/audio/speech`` takes the text alone
  (zero-shot), a voice description in ``instructions`` (voice design), a
  ``ref_audio`` clip (cloning) or ``ref_audio`` with its transcript in
  ``ref_text`` (continuation). Output is 48 kHz.
- The Talker's decode step (both LMs, the local DiT solve, the stop head) is
  ``torch.compile``'d and captured as CUDA graphs at start-up, so a first
  start-up with a cold Inductor cache takes 10 to 30 minutes (591 s on a
  B200, 1,723 s on an H100), and about two minutes once the cache is warm.
  ``model_kwargs: {compile_talker: false}``
  in the deployment YAML captures the eager step instead: on a B200 the server
  is up in about 15 seconds, with about a third less throughput at
  concurrency 1 and a sixth less at concurrency 8.
- The Talker's prefill (both LMs over the prompt, the first patch's solve and
  the stop head) is captured too, per total prompt tokens (up to 2,048) and
  prompts in a step (up to 32); a step past them runs the LMs eagerly.
- ``configs/voxcpm2.yaml`` runs the VAE decoder's windows on the worker's side
  lane (``side_stream: true``), beside the Talker's steps rather than taking
  turns with them.
- The streaming VAE decoder is ``torch.compile``'d at start-up, about a
  quarter faster per window batch; ``model_kwargs: {compile_vae: false}``
  decodes eagerly.
- A request may hold the whole 8,192-position context, and nothing preempts a
  request once a KV cache is full, so each LM's cache defaults to 2,049 pages
  (32 such requests and the sink page, about 9.7 GB for both) and the YAML sets
  ``max_concurrent_requests: 32``; the conductor queues the rest. Start-up
  refuses a cap the caches cannot hold, so raise ``resources.base_kv`` and
  ``resources.res_kv`` ``max_num_pages`` with it. Text over
  ``max_text_tokens`` (1,365 by default) gets a 400: English speech runs at up
  to about 2.9 patches per token, so longer text may not fit in the
  4,096-patch (655 s) audio budget.
- One request with long text can end early: on some seeds the stop head fires
  mid-text, as it does in the official ``voxcpm`` package (a 983-character
  Russian text stopped about three quarters of the way in on 5 of 10 seeds
  there). Split long text into several requests.
- ``min_len``, ``max_len`` and ``inference_timesteps`` must be integers (a whole float
  such as ``16.0`` is accepted; ``2.7`` and ``"20"`` are 400s), ``max_new_tokens`` /
  ``max_output_tokens`` (160 ms patches) follow the server's integer rule, and
  ``cfg_value`` is a number in [0, 10].
  ``inference_timesteps`` is a deployment setting a request may only repeat. A budget past
  the model's ``max_len`` is refused. ``seed`` fixes the diffusion noise. ``voice`` is
  dropped (VoxCPM2 has no named voices) and a sent ``speed`` is logged as ignored.

Nemotron VoiceChat (``nemotron_duplex``) notes
----------------------------------------------

- Three nodes in three async partitions, joined by streaming edges, with the
  user's audio a client stream (``audio_in``) into the first:
  ``conformer_encoder`` (16 kHz audio → one LLM-space embedding per 80 ms frame,
  streamed: each chunk runs the layers over its new frames only, against the
  call's cached last 70 frames, so a step costs the same however long the call
  is; it emits the frames no later audio can change, the last ones at the end
  of input. It is the shared FastConformer, Canary's encoder, as a stream node
  (``FastConformerStreamSubmodule``), with the featurizer of the VoiceChat
  reference, which predates NeMo 2.5: reflected clip edges and
  ``samples // 160 + 1`` mel frames. The checkpoint's RNN-T transcription head
  is not loaded)
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
  (paced runs interleaved against the uncaptured encoder, on the SHM arena);
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
  does): one clip uploaded with the request, a live call over
  ``/generate/ws`` (an opening with ``end_input: false``, then ``append``
  messages until ``end_input``; see :doc:`clients`), or an OpenAI
  Realtime conversation over ``/v1/realtime``, which takes PCM16 at one of
  the eight rates :doc:`clients` lists (8 to 48 kHz) or G.711, resamples it
  to 16 kHz, appends it as it comes, and sends each agent reply as one
  response (BOS / EOS on the agent text channel mark the reply,
  ``metadata.turn`` on ``/generate``). The voice is ``Aria``. Output is clocked by
  the input, one agent frame per 80 ms of user audio, so a ``/generate/ws``
  client keeps streaming (silence included) for as long as it wants the agent
  to talk; ``/v1/realtime`` pads a pause in the client's audio with silence,
  up to 30 s. The
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
  ``/generate/ws`` appends must be 16 kHz audio (they are neither
  resampled nor rescaled one by one, which would put a seam at every append);
  ``/v1/realtime`` resamples for the client. On both, ``audio_in`` declares the
  encoder's 80 ms frame (``ClientAudio``), and the data worker sends a live
  call's audio in whole frames, a step each, the rest waiting for the next
  append or the call's end.
- Encoder numerics, measured on CPU at the checkpoint's dimensions with random
  weights on the duplex encoder the shared FastConformer replaced, and not
  since: an upload was one encoder pass, bit for bit that encoder's offline
  pass in fp32 and under bf16 autocast, save the last frame when the end of
  input reaches the encoder after the audio (about 1e-6 in fp32). 80 ms appends
  put the frames within about 2e-6 of offline in fp32 and about 0.7% under bf16
  autocast. The shared encoder's stream attention adds the relative-position
  term before scaling and its offline attention after, so an upload is no
  longer expected to match the offline pass bit for bit. From those figures,
  live and one-shot text should agree token for token in fp32 (inferred, not
  measured on served text); under the served bf16 a different chunking can flip
  a knife-edge frame such as the agent's turn start. Under bf16 autocast the
  stream node keeps the blocks' residual in fp32, as the replaced encoder did
  from its second layer on; the offline FastConformer encoders keep theirs in
  bf16. Against an fp32 run on 13 clips (one H200, real checkpoint,
  2026-10-29), live frames have a mean per-frame relative error of 1.35%,
  level with the replaced encoder's 1.39%, where a bf16 residual gave 1.63%;
  uploads and offline passes are as close as the replaced encoder's too.
- The encoder keeps each call's state in a slot per session of three
  resources: its attention keys and values in a bounded KV (``encoder_kv``),
  which keeps a call's last 70 frames per layer and runs the attention, each new
  frame over the 70 before it, with the relative-position term as a band of
  distances; its conv inputs (``encoder_state``); and its newest audio and mel
  frames (``encoder_front``), the last two recurrent-state pools. Every live
  step, one frame (1280 samples) whatever the client's append size, runs whole
  as a CUDA graph; other steps take the front end eagerly, then the layer stack
  as a graph when each got at most one frame (the call's end), or eagerly (a
  one-shot upload, an opening clip, the call's last append: a chunk over a
  second is not batched with other calls, so the engine runs each call of that
  step on its own).
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
  the upper bounds are the reference demo's), ``penalize_prompt`` (whether the
  penalty counts the prompt's tokens, default true), ``seed``, ``n_cfm_timesteps``
  (S3Gen Euler steps, 10; Turbo 2), ``max_output_tokens`` (up to the deployment's
  ``model_kwargs: max_new_tokens_limit``, 1000 in the shipped configs; with
  ``max_concurrent_requests`` set, the server refuses to start unless that
  many requests of that length fit the KV cache, so raise it together with
  ``max_num_pages`` or a lower cap),
  ``ignore_eos`` (T3 decodes all ``max_output_tokens``
  past the stop token, for fixed-length benchmarks; the speech tokens among
  them are vocoded, but T3 keeps emitting stop and control tokens after the
  end of speech and those are dropped, so the audio is shorter than
  ``max_output_tokens`` / 25 s: 4096 tokens gave 117 s on base and 141 s on
  Turbo, not 164 s) and ``watermark`` (default on). Turbo has no
  ``cfg_weight``, ``exaggeration`` or ``min_p``: a non-zero value is a 400.
  ``language_id`` on a non-multilingual checkpoint is reported as ignored.
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
  the three look-ahead tokens and crossfades the vocoder tail (which a first chunk
  of at least 7 tokens fills), so the stream
  is continuous but not sample-identical to the whole-utterance decode;
  ``stream_chunk_tokens: 0`` synthesises whole utterances (the reference
  path, bit-exact with the package at a fixed seed). ``stream_context_tokens``
  (default 25; 0 = whole history) bounds how many settled tokens a chunk's
  flow solve keeps as left context, making the per-chunk cost constant; the
  window stays as close to the whole-utterance decode as the full history
  does (log-mel correlation 0.988 vs 0.985 on CPU). Requests whose chunks
  are ready together share one padded flow solve (up to 8 per step).
- ``configs/chatterbox_split.yaml`` (and ``chatterbox_turbo_split.yaml``,
  ``chatterbox_multilingual_split.yaml``) runs S3Gen on a second worker that
  shares GPU 0 (``rank_devices``), so T3's decode steps stop alternating with
  S3Gen's flow solves on one worker loop. It only pays under CUDA MPS, which
  lets the two workers' kernels overlap, so the configs set ``mps: true`` (see
  :doc:`serving`): on an H200 at concurrency 32 Chatterbox
  went from 36-38 to 58-61 audio-s/s and Turbo from 69 to 94, at the cost of a
  later first chunk (TTFA p50 1.6-1.8 s -> 1.9-2.1 s; Turbo 0.36 s -> 1.0 s);
  concurrency 1 is unchanged, and WER held (1.22% -> 1.08%). Without MPS the
  split measured the same as one worker.
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
  when available (torchcodec >= 0.9), otherwise directly with PyAV (``av``,
  ``video_encode.encode_mp4``), whose wheel bundles FFmpeg with libx264; that also
  covers a torchcodec that cannot load FFmpeg.
- Sound-enabled video responses mux the AAC track with the ``ffmpeg`` and
  ``ffprobe`` binaries, which must be on ``PATH`` (system packages, not
  pip-installable).
- The Wan-VAE decode dtype is gated on the cuDNN build: bf16 needs cuDNN >=
  9.16 (fast Hopper bf16 conv3d); older cuDNN serves the decode in fp32/TF32
  automatically.
- Two GPUs (H200 numbers, measured before the Cosmos3-Edge Blackwell changes,
  so the one-GPU baselines are older): ``configs/cosmos3_nano_sp2.yaml`` halves
  one request's denoise (Nano 832x480x189x35: 1.47 -> 0.78 s per step), while
  ``configs/cosmos3_nano_dp2.yaml`` (a whole replica per GPU) serves more under
  concurrency: 2.13 videos/min at 2 concurrent vs 1.63 for sp2 and 1.07 on one GPU.
  For Edge (832x480x121, 20 steps): ``configs/cosmos3_edge_sp2.yaml`` takes one
  request from 6.3 to 4.4 s (generation only: the reasoner has no SP path), and
  ``configs/cosmos3_edge_dp2.yaml`` serves 15.8 videos/min at 4 concurrent vs 7.9 on
  one GPU.

Cosmos3 image-to-video conditioning
-----------------------------------

An image-to-video request's conditioning image is scaled to cover the output
size with its aspect ratio kept, center-cropped and rounded to 8 bits, as in
NVIDIA's cosmos-framework and diffusers 0.40 and later. Setting
``conditioning_resize: stretch`` under the config YAML's ``model_kwargs``
resizes it to the output size instead (antialiased bilinear, aspect ratio not
kept), as diffusers 0.39 did. Action requests ignore the setting. The image
pins the first latent frame, so the request must ask for a video of more than
one latent frame (``num_frames`` above 4); an image output, or a shorter video,
is refused with a 400.

Cosmos3-Edge reasoner and action loop
-------------------------------------

``cosmos3_edge`` serves the understanding tower as a vision-language model on the
same transformer instance and KV pool as the generator. ``/v1/chat/completions``
takes image and video content parts (URLs or data URIs) and streams tokens; the
chat template opens a ``<think>`` block by default. ``extra_body`` knobs:
``enable_thinking`` (or ``chat_template_kwargs.enable_thinking``), ``top_k``,
``repetition_penalty``, ``penalize_prompt`` (``min_p`` is refused unless the checkpoint's
``generation_config.json`` sets one; ``max_tokens``
defaults to 2048), and for video attachments ``video_fps`` / ``video_num_frames``
(frames are sampled at 2 fps by default, each frame a timestamped span).

.. code-block:: bash

   curl -sN http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "cosmos3_edge", "stream": true, "max_tokens": 256,
     "messages": [{"role": "user", "content": [
       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
       {"type": "text", "text": "The task is to put the flower into the red bottle. Plan the next steps."}]}],
     "enable_thinking": false}'

The reasoner is a ``SampledTokenDecoder`` on the VLM scaffold: its decode steps
are captured into CUDA graphs per batch bucket, each compiled first: the eager
decode step is over a thousand tiny kernels, and the fused step runs at the
weight-streaming floor (about twice the uncompiled rate at batch size 1 on an
H100). Text prefills, mixed prefill + decode steps and image or video prefills
(up to four prompts a step, each prompt one row with its vision tokens spliced
in) are captured too, and a prompt past the largest bucket prefills eagerly.
Concurrent chat requests share decode steps (continuous batching over the
captured decode graphs, padded to the next batch bucket). Requests never see
each other's data, but the batch bucket changes the
bf16 arithmetic of the step, so a long greedy answer can part from its solo run
where two candidate tokens tie: in every measured divergence the two tokens'
logits were equal or one bf16 ulp apart, the first differing token came after
tens to hundreds of identical ones, and identical prompts in one batch agreed
with each other. Short answers (128 tokens) came out identical in 8 of 8 runs;
256-token answers with thinking on in 2 of 8. Generation requests batch
into one denoise pass too, which is the same maths but not the same bf16
arithmetic — under classifier-free guidance the branch rounding is amplified,
so an image or clip produced alongside other requests differs from its solo
result at the kernel-drift level (~30 dB PSNR at guidance 6). A generation
step whose requests total more than ``max_batched_vision_tokens`` (a
``model_kwargs`` key, off by default since it is a per-GPU number; 16384 on an
H100 is, at 832x480, more than 42 latent frames across the step's requests,
e.g. two clips of 85 or more frames each; a windowed request counts its current
window) runs every request in it alone, since batching such steps cost more
than it saved on an H100, so those outputs match their solo results. Serve with one request at a time
when outputs must be bitwise repeatable.

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
(``max_gen_batch_size``, default 8) for video, image, sound and action. With
``compile_denoise_batched`` (default on) the layer stack these steps share is
compiled with the same mode, once, while the server starts, with the packed
length dynamic: no batch size or mix of resolutions compiles mid-traffic. The
per-request patchify, rotary tables and unpatchify around it run eager.
Compiled, a batched action step's outputs differ from the eager stack's at the
kernel-drift level. At 832x480x121 one request's guidance pair is 24k tokens,
which already fills the GPU's GEMM and attention throughput, so a batch of N
costs about N single steps: batching keeps requests in step rather than adding
throughput.

The action policy (``cosmos3_edge_droid``, or ``cosmos3_edge`` with an action
``domain_name``) predicts a chunk of robot actions from the current observation:
``output_modalities=action`` with ``model_kwargs``
``{"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": 8,
"action_chunk_size": 32}``; the reply's ``action`` payload is float32
``[chunk, action_dim_padded]`` and the first ``raw_action_dim`` columns are the
embodiment's actions. A policy request may also send the robot's current state
as ``robot_state``, one row of ``raw_action_dim`` finite numbers (a NaN or inf is
a 400), as pi0.5 takes it. The state is one more action token, kept clean at
video frame 0 in front of the chunk, so ``num_frames`` is ``action_chunk_size + 1``
and the reply carries only the predicted rows. The released DROID policies were
trained this way. A request that sends a ``robot_state`` is a policy request, and
its chunk, fps and domain default to the checkpoint's ``checkpoint.json`` (Edge
DROID: 32 rows, 15 fps, ``droid_lerobot``; ``configs/cosmos3_droid.yaml`` sets the
same for Nano, whose file is empty), so NVIDIA's RoboLab request is the image, the
prompt, the state and ``size="736x544"``: the wrist frame above the two exterior
frames reflection-padded to 736x544, the 7 joint positions and the gripper as
DROID records them, and the task with view, duration and resolution text (a JSON
caption for Edge). The DROID configs map the gripper to the checkpoint's
``1 - gripper`` in every action a request sends and back in every action it
gets (``inverted_action_dims``),
and ``cosmos3_edge_droid`` guides only inside the cookbook's interval
``[960, 1001]`` (``guidance_interval_action``; outside it the conditional branch
runs alone, batched requests too). For a control loop use ``/generate/ws`` (one
connection, pipelined observations): ``examples/cosmos3_action_ws_client.py``
builds that request, runs the loop and reports chunks/s, actions/s and latency
percentiles.

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
     - Seconds an idle session is kept after its request finishes, at most the
       deployment's ``session_timeout_max_s`` (3600); more is a 400. An expired session resumes
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
and with plain requests at the same walk, under the same 16384-vision-token
cap on the step (each windowed request counts its current window).
``gen_capture_video`` lists (height,
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
     - An integer in 1-100; the upper bound is the denoise loop's ``max_iters``
       (``max_denoise_steps``). A 400 outside it.
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

**UniPC runs inline, inside the DiT node.** The scheduler is the DiT scaffold's
(``components/diffusion/unipc.py``, which cosmos3 runs too), not a separate
stage: the solver state (the order-2 history buffer and the corrector's
``last_sample``) is carried on the denoise loop's own edges, so it travels with
the request rather than living in a scheduler object on one rank. Requests are
therefore independent and the loop is resumable across ranks. ``dit_max_batch_size``
(default 1) lets requests at one latent grid and guidance scale share a forward,
each row count compiling its own graph.

**What is accelerated by default.** The DiT runs under ``torch.compile``
(``compile_dit``); its self-attention runs on the engine's ragged attention
resource (FlashInfer, FA3 on Hopper; on a B300 the ``ragged_attention`` kernel table
picks torch's cuDNN SDPA, the kernel ``backend: sdpa`` runs, and the two
configs give the same videos at the same step time) as an opaque op inside the compiled graph,
6% off a 1280x704x81 denoise step on an H200 against cuDNN SDPA, measured on
an earlier version of the DiT block, before the adaLN was computed in one pass
and the q/k norm and RoPE were fused
(``backend: sdpa`` on the ``dit_self_attn`` resource restores the reference kernel; the
cross-attention over the 512 text tokens stays on SDPA, which is faster at that
shape). The untiled VAE decode runs the shared Wan VAE decoder,
channels-last and compiled (``vae_decode_fast``; set it to ``false`` under the
yaml's ``model_kwargs`` for stock ``vae.decode``). Both compile on first use:
the DiT only for the ``compile_dit_sizes`` shapes (see Compiled DiT shapes), so
the first request at a listed shape is slower. The VAE decode compiles once per resolution, and only
for the sizes in ``vae_decode_fast_sizes`` (model default ``1280x704``; the
shipped configs add ``832x480``, where it decodes 1.59x faster); any other
size decodes with stock ``vae.decode``, frames identical to
``vae_decode_fast: false``. The compile uses Inductor's deterministic mode, so
every server process decodes the same latents to the same frames; torch 2.9.1
has no such mode, and there two processes can differ by a few uint8 levels.
The VAE decode
tiles only when free VRAM at decode time is short of the untiled estimate
(the ``auto`` policy; ``WAN22_VAE_DECODE_TILING`` forces a path), which keeps the untiled conv3d from OOMing a
32 GiB card. No CUDA-graph capture, no continuous batching (one request per
step) and no component offload.

**Two GPUs.** ``configs/wan22_cfg_parallel_2gpu.yaml`` is CFG-parallel: it runs
the dit over two ranks (``cfg_size: 2``) that split the classifier-free guidance
batch: each rank runs one branch over the batch's rows and the two noise
predictions are all-gathered
every step, so a guided step costs one branch's forward (502 vs 1000 ms at
1280x704x81 on H200s; one request 13.4 s vs 23.3 s end to end at 20 steps;
measured on that earlier version). The
text encoder and VAE run on rank 0. Without guidance both ranks compute the same
step, correct but no faster. For throughput under concurrency,
``configs/wan22_dp2.yaml`` serves a whole replica per GPU instead: 5.2 videos/min
at 4 concurrent vs 4.7 CFG-parallel (whose rank 1 idles through rank 0's VAE
decode; measured on that earlier version), while one request takes the 1-GPU time.

**Four GPUs.** ``configs/wan22_cfg2_sp2_4gpu.yaml`` also splits each branch's
tokens over a pair of ranks (``cfg_size: 2, sp_size: 2``, Ulysses on the dit's
self-attention) to cut one request's latency. Split the guidance branches
first (``cfg_size: 2``), then add ``sp_size`` on top: on two GPUs,
``sp_size: 2`` alone is slower than ``cfg_size: 2``.
``sp_size`` must divide the dit's 24 attention heads, and SP needs the
ragged resource (``backend: sdpa`` on ``dit_self_attn`` runs outside the engine's resources, so
it is refused on SP ranks). Under concurrency, data
parallelism stays the throughput layout: at 1280x704x81 and 4 concurrent, four
replicas (``wan22_dp2.yaml``'s layout over four ranks) serve 8.6 videos/min
against 7.2 for this layout on H200s, and 11.0 against 7.4 in a separate run on
H100s.

LTX-2.5 (``ltx2_5``)
--------------------

Text-to-audio+video on **Lightricks LTX-2.5** (``Lightricks/LTX-2.5-Diffusers``, gated):
a 22B joint audio-video DiT, a Gemma-4 12B text encoder, and video/audio VAEs with a
48 kHz vocoder. The distilled checkpoint runs the model card's recipes without
guidance; the full (SFT) transformer, image-to-video and the duration head are not
ported yet.

Five nodes: ``text_encoder`` (native Gemma-4 + the text connectors), ``dit`` (the
DiT scaffold's denoise loop; video and audio latents carried in fp32; every attention
through two ragged attention resources), ``latent_upsampler`` (two-stage recipe only),
``vae_decoder`` and ``audio_decoder``. The walk follows the requested output
modalities, so ``["video"]`` skips the audio decode (the model still denoises both).

.. code-block:: bash

   mstar serve ltx2_5                             # configs/ltx2_5.yaml: DiT on GPU 0, the rest on GPU 1
   mstar serve ltx2_5 --config configs/ltx2_5_tp2.yaml   # DiT tensor-parallel over 2 GPUs
   mstar serve ltx2_5 --config configs/ltx2_5_sp2.yaml   # DiT sequence-parallel over 2 GPUs (fastest)

Request knobs (``model_kwargs``): ``height`` / ``width`` (multiples of 32; 64 for the
two-stage recipe), ``num_frames`` (8k+1), ``fps``, ``seed``, and ``recipe``:

- ``single`` (default): 8 distilled sigmas at the requested size (default 544x960x121).
- ``two_stage``: the card's better-quality path — 8 sigmas at half size, the x2 latent
  upsampler, then 3 sigmas at full size (default 1088x1920). Frames above 720p decode
  in VAE tiles, as the card does for this decode.

Outputs are an H.264 ``video`` and a 48 kHz stereo ``audio`` chunk (16-bit PCM).
``POST /v1/videos/generations`` returns one mp4 with the audio muxed in as AAC
(``"audio": false`` in the body for video alone); the mux needs the ``ffmpeg`` binary
on the server's ``PATH`` and falls back to video only without it.

``ltx2_5_sp2`` splits each step's video tokens over the two ranks (the DiT's sequence
shard): each rank holds the whole DiT and half the tokens, and only the video
self-attention exchanges them (Ulysses, one all-to-all each way); the cross-attentions
run on each rank's own tokens, and the audio queries over the video merge the ranks'
partial attentions. The attention resources do the exchanges, so it needs
the ragged resources (``backend: sdpa`` on one is refused on SP ranks where it reads split
keys). ``ltx2_5_tp2`` shards
the weights instead and all-reduces after every block's attentions and feed-forwards.

Performance (2x H100, warm; diffusers with the same recipe and seeds; the ``ltx2_5_sp2`` column
was measured before the exchanges moved into the attention resources):

======================================  ================  ================  ================
                                        diffusers         ``ltx2_5_tp2``    ``ltx2_5_sp2``
======================================  ================  ================  ================
544x960x121, one request                7.87 s            3.70 s            3.15 s
544x960x121, 4 concurrent               7.6 videos/min    19.6 videos/min   22.3 videos/min
two-stage 1088x1920x121, one request    21.6 s            11.0 s            11.3 s
two-stage, 2 concurrent                 2.8 videos/min    5.9 videos/min    6.1 videos/min
======================================  ================  ================  ================

The DiT is compiled and its denoise step captured per shape (``capture_shapes``, which
the two-stage recipe's stages replay too); other shapes run the compiled eager path, and the first
request of a new shape pays its compile. ``vae_compile`` (on in these configs) compiles the
video VAE's decoder at load for ``vae_compile_shapes``, by default the same shapes (~55 dB
from the eager decode; other shapes decode eager). The SP2 config doesn't capture 1088x1920
but lists it there, so its two-stage decode runs compiled too.

MiniMax-H3 (``minimax_h3``)
---------------------------

Text- and first/last-frame-to-audio+video on **MiniMax-H3** (``MiniMaxAI/MiniMax-H3``,
its ``FL2VA`` checkpoint; MiniMax H3 Community License Agreement): a 33B packed-sequence
audio-video DiT, guidance-distilled (one forward per step), a Qwen3-VL-32B conditioner
and a ViT video VAE, at 24 fps with 32 kHz stereo audio. Not ported: ``Ref2VA``, the
H3-Context-IR prompt rewriter and the 2K regenerator.

Nodes: ``text_encoder`` (Qwen3-VL to layer 50, keyframes through its vision tower, then
the DiT's ``condition_proj`` and token refiner; once per request), ``vae_encoder``
(keyframes), ``dit`` (the DiT scaffold's denoise loop, Ulysses SP), ``vae_decoder`` and
``audio_decoder``. In the SP configs ``vae_decoder`` sits in the DiT's SP group: each rank
decodes a share of the clips, between denoise steps.

.. code-block:: bash

   mstar serve minimax_h3                                        # 4 GPUs: DiT SP4
   mstar serve minimax_h3 --config configs/minimax_h3_sp8.yaml   # 8 GPUs: DiT SP8
   mstar serve minimax_h3 --config configs/minimax_h3_sp2.yaml   # 2 GPUs: DiT SP2
   mstar serve minimax_h3 --config configs/minimax_h3_2gpu.yaml  # 2 GPUs, no SP

Request knobs (``model_kwargs``): ``height`` / ``width`` (multiples of 32, aspect 1:4 to
4:1; default 768x1344, or the first keyframe's aspect), ``num_frames`` (snapped up to
17n+5, 5 to 15 s; default 124), ``num_inference_steps`` (sigma points, default 50: 49
forwards), ``flow_shift`` / ``audio_flow_shift`` (12 / 3), ``seed``, and with one or two
images ``keyframe_anchors`` (``first`` / ``last`` per image). These are a 400: a prompt
over the text encoder's 262,144 positions (keyframe tokens included; ``max_prompt_tokens``
lowers it), a packed sequence over ``max_packed_rows`` (default 131072: the longest 768p
clip, 14.4 s, with a ~25k-token prompt), Qwen's vision tokens in the prompt, and a last
keyframe outside 1:4 to 4:1 next to a first one (it is cropped to the canvas). One request
runs per denoise step. ``POST /v1/videos/generations``
returns one mp4 with the audio as AAC (``image`` is the first frame,
``extra_body.last_image`` the last; the mux needs ``ffmpeg`` on ``PATH``).

Deployment knobs (``model_kwargs``; the SP configs set ``compile``, ``vae_tile_batch: 28`` and
``vae_compile``):

- ``compile``: the DiT's blocks, static per row bucket: each new bucket compiles (5-8 s)
  mid-request. At ``MSTAR_RECOMPILE_LIMIT``'s default of 84 (measured at SP2 on H200), a
  bucket past the 42nd runs partly eager and one past the 84th fully eager; buckets compiled
  before stay compiled.
- ``row_bucket``: the DiT's packed rows pad up to a multiple of this and of the SP size, as
  a tail segment that attends only to itself and reaches no output. Default 256 with
  ``compile`` (one graph per 256 prompt tokens per canvas, at most 0.7% more rows at
  768p), else 1.
- ``vae_tile_batch``: 256-px tiles per decoder call (28 is one 768p clip).
- ``vae_compile``: the decoder's blocks, static per tile batch; a canvas or clip share with
  another tile count compiles again. A 768p clip's share decodes in 2.1 s instead of 3.2 s
  on one H200, within 62.9 dB of eager.

The attention kernel is a resource's setting: ``backend: sdpa`` on the ``dit_attn`` and
``refiner_attn`` resources runs the reference's kernel, and FlashInfer runs without it. SDPA on
``dit_attn`` is refused on SP ranks, which every shipped config but ``minimax_h3_2gpu.yaml`` is.

Async scheduling is on for the DiT and the decoder, the default for parallel nodes
(``tp_async_sched: false`` turns it off); on and off measured within noise. Runtime needs:
PyAV, transformers' image processor (a server that can't import it, e.g. next to a
torchaudio built for another torch, refuses to start), FlashInfer, and ``ffmpeg``. Each DiT
rank holds the whole DiT, and the SP configs put the conditioner beside it on rank 0 (117 GiB
in use once loaded), so they need 141 GB GPUs (H200); no config has been run on 80 GB cards.

On SDPA the nodes' modules, driven by ``test/minimax_h3/parity.py`` (fp32 matmuls at
"highest"), were bit-identical to diffusers' MiniMax-H3 pipeline end to end (latents, frames,
waveform), measured at ``f6260f2677``; with ``--exact`` the script exits nonzero when one isn't.
FA3 and the compiled DiT are not bit-exact. Served at 768p (50 steps, SP2 on H200, three
seeds), measured on main's earlier SP path, before the DiT and the decoder moved onto the sequence
shard, clips are 23.7-32.8 dB (video) and 17.8-22.3 dB (audio) from diffusers', where the
reference on another SDPA kernel is 23.5-33.1 and 21.6-22.3 dB from itself; per seed, the
served clip is below that spread on one seed's video and another's audio.

Seconds per clip, one request at a time, each engine on the same H200s, M* measured before the
DiT and the decoder moved onto the sequence shard (M* on torch
2.12.1+cu132 and FlashInfer 0.7.2; text only, 1344x768, 124 frames, 49 DiT forwards, seed 0,
median of 3 after a warmup; ``benchmark/minimax_h3/bench_video.py --seed 0`` sends these
requests):

.. list-table::
   :header-rows: 1

   * - s/clip
     - M*
     - SGLang 0.5.23
     - vllm-omni 0.33.0rc1
   * - SP4, 4x H200 (``configs/minimax_h3.yaml``)
     - 76.8
     - 71.8
     - 78.2
   * - SP8, 8x H200 (``configs/minimax_h3_sp8.yaml``)
     - 39.4
     - 38.1
     - 41.8

SGLang ran ``--performance-mode speed`` at Ulysses 4 / 8, without compile or step caching;
vllm-omni USP 4 / 8 with the text encoder and VAE split as wide. The DiT's 49 steps are
~74 s of the SP4 clip. Not done: CUDA-graph capture, more than one request per step, tensor
parallelism, ``Ref2VA``.

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

``backend: sdpa`` on the ``dit_attn`` resource, as the shipped configs set it, runs the
reference kernel: torch's SDPA called by the model, cuDNN on an H100 and a B300. Without it the
DiT's joint attention runs on the engine's ragged attention resource, whose kernel the GPU's
``ragged_attention`` table picks: torch's cuDNN SDPA on ``sm_103`` (B300), FlashInfer's ragged
prefill (FA3 on Hopper) where no table covers the GPU. ``backend: flashinfer`` keeps
FlashInfer. Both are CUDA-graph replayable. Z-Image takes the same block.

Deployment knobs live under ``model_kwargs`` in ``configs/flux2_klein.yaml``:

``compile``
   ``torch.compile`` of the transformer, one trace per shape.

``compile_eager_rounding``
   Inductor rounds intermediates where eager PyTorch does, and fuses no FMAs.

``compile_exact_ops``
   ``true``, or a list of op classes among ``norms`` and ``activations``: those modules
   keep the eager result inside the compiled forward, so inductor only fuses the chains
   around the GEMMs and attention. Each kept module other than a 1-D ``torch.nn.RMSNorm``
   is a graph break, so this is a latency-for-exactness trade:

   * The norms alone make the transformer bit-exact with eager (measured on klein-4B and
     9B while the q/k norms were graph breaks), so klein-4B and 9B ship ``[norms]``.
     Klein's per-head q/k norms (1-D ``RMSNorm``) run as an op the compiled graph keeps
     whole. On NVIDIA GPUs that op is a Triton replica of torch's kernel with the same
     arithmetic order: bit-identical and faster than torch's one-block-per-row launch.
     Kept norms of one class share a class-level forward, so the blocks share one
     compiled frame: with a forward per module, dynamo hit its recompile limit and the
     small batches ran the blocks eagerly (klein-4B 2.86 / 3.10 / 3.69 against 3.33 /
     3.75 / 3.80 img/s at concurrency 1 / 4 / 16, H200). Keeping the activations too
     adds cost and nothing else.
   * Z-Image's 180 norms per step cost 45% of its B=1 latency, so it ships ``false``
     and lands at a median 34 dB (the ``false`` figures were measured before its compiled
     forward fused the q/k norms and rotary into one kernel). ``[norms]``, which keeps
     that fusion off, turns it reference-faithful (at least 55.9 dB on every prompt).
   * Without ``[norms]``, the plain compile — even with eager rounding — lands at a
     median 37.6 dB (klein-4B) and 35.6 dB (9B) PSNR from eager over the 100 protocol
     prompts (measured with the earlier compiled VAE decode), because a 4- or 8-step
     distilled sampler amplifies the last bit of inductor's own reductions and
     activation decompositions.

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
   meanwhile, so requests at different steps share a forward. On by default; the loop's cap
   (the request's step count) keeps speculation from running past its schedule, and ``false``
   restores lockstep. Measured on one H100 back to back against lockstep: klein-4B 0.365
   to 0.372 s vs 0.377 to 0.390 s at B=1 and about 3% more images per second at 4 and 16
   concurrent requests, klein-9B 2%, Z-Image neutral, served images unchanged — the
   denoise phase is compute-bound either way.

``max_batch_size``
   Rows batched per denoise step.

``vae_compile``
   ``torch.compile`` of the VAE decode with inductor autotuning, each nearest-x2 upsample
   folded into its conv and the GroupNorms (with their SiLU) on a split-reduction kernel.
   Before those two it measured 89 to 29 ms at 1024² on an H100 and about 55 dB PSNR from
   the eager decode on every prompt; neither figure is re-measured since. The compiled
   decode is not bit-exact, and the autotuner may pick different conv kernels in another
   server process, so set it to ``false`` for bit-exact, repeatable output — which is how the
   parity suite runs. The compiled decode is warmed at load for the ``capture_sizes``
   grids and the decode batch sizes; any other latent shape decodes eagerly rather than
   paying a 40 to 100 s autotune inside a request. Memory cuts the other way: the eager
   decode of an 8-image batch at 1024² allocates about 40 GiB of activations, for served
   peaks of 75 GiB on klein-4B and 77 GiB on Z-Image at 16 concurrent requests against
   36 to 44 GiB compiled.

``linear_dtype``
   ``bf16`` (default), ``mxfp8`` or ``nvfp4``; klein 4B, 9B and Z-Image. Runs the block linears
   the model declares (``QUANTIZABLE_LINEARS`` beside each transformer) in that format: a
   ``QuantLinear`` holding the weight quantized once at load, after any ``lora``, whose forward
   is one op (``mstar::mxfp8_linear`` / ``mstar::nvfp4_linear``) that quantizes the activation
   with a Triton kernel and runs cuBLASLt's block-scaled GEMM. Modulation, timestep, patch and
   caption embedders and the final projection stay bf16. A GPU below compute capability 10.0
   stays on bf16 with a warning, and a linear whose K isn't a whole number of scale tiles (MXFP8
   128, NVFP4 64), whose N isn't a multiple of 16, or whose kind can't take a format (a
   tensor-parallel linear) is refused at load.

   NVFP4 quantizes an activation's 16-value blocks with e4m3 scales in units of a per-linear
   tensor scale ``s``, so a block clips above ``2688 * s``. Over the 100 protocol prompts,
   klein's text feed-forward outputs reach 4080 (4B) and 5056 (9B) and Z-Image's feed-forward
   outputs up to 340k; every other input stays under 1800. Those linears take the power of two
   that holds their measured maximum (``input_amax`` in the declaration), the rest ``s = 1``.

   It stays off by default because the images move. Against the bf16 image of the same prompt
   and seed, over the 100 protocol prompts with the shipped config on one B300 (compute capability
   10.3): klein 4B MXFP8 lands at a median 22.3 dB, lowest 13.6 dB, and NVFP4 at 15.6 dB,
   lowest 9.3 dB. Two bf16 images of a prompt at different seeds are a median 10.4 dB apart.
   MXFP8 keeps the scene with its details shifted; NVFP4 sometimes changes the composition. On
   the same GPU the 4B's denoise step goes from 31.4 ms to 24.4 ms (MXFP8) or 22.9 ms (NVFP4),
   and images per second at 8 concurrent requests rise 25% or 39%. Device memory after 8
   concurrent requests falls by 2.3 GiB on klein 4B with MXFP8 (5.1 GiB with NVFP4), 7.6 GiB on
   9B and 6.9 GiB on Z-Image (MXFP8, each model's shipped config with ``backend: sdpa``).

``linear_keep_bf16``
   A list of module-name patterns over the transformer's linears to keep in bf16 for one
   deployment, on top of the declaration. A pattern matches name segment by segment, ``*``
   within one segment, and a segment may be an index range: ``single_blocks.0-3.*``. A pattern
   that names none of the candidate linears, or a bare string, is refused at load; the log
   names the linears each pattern kept.

``linear_dtype_steps``
   Some denoise steps in bf16 under a low-precision ``linear_dtype``: a count of leading steps
   (``1``: the first step in bf16), or a dtype per step, each ``bf16`` or the ``linear_dtype``,
   past the list its last entry. Those steps run a bf16 copy of the transformer taken before
   quantizing, so the deployment holds both sets of weights. Rows batch, and each capture bucket
   is captured, per dtype, and the scheduler keeps a step of one dtype apart from the other.
   One leading bf16 step takes klein 4B MXFP8 from a median 22.3 dB to 27.0 dB, lowest 16.0 dB,
   at a 26.0 ms step (24.4 ms MXFP8, 31.4 ms bf16); at 8 concurrent requests it serves 17% more
   images per second than bf16, against MXFP8's 25%. Ignored when nothing is quantized (``bf16``,
   or no block-scaled GEMM).

``max_image_area``
   Pixels, default 2048². Bounds the output size a request may ask for; larger requests
   are rejected with a 400 before scheduling instead of occupying the worker for minutes
   (an 8192² request means 262k tokens of quadratic attention). Z-Image also refuses a side
   over 8192 px, the 512 tokens its rotary table covers.

With the klein defaults every served image was within 53 dB of the eager path on all 100
protocol prompts, and with ``vae_compile: false`` it was bit-exact; both were measured while
the q/k norms were graph breaks, and the 53 dB before the folded, split-GroupNorm VAE decode.
For scale, SGLang 0.5.22's images sit at a median 23.8 dB (minimum 13.1) on klein-4B and
24.0 dB (14.5) on 9B from klein's output at the same seeds.
Requests at the same output size batch across users in every node, including the
text encoder, whose input is always 512 tokens. ``lora`` lists adapters to fold into the
transformer weights at load time (``[{path: ..., scale: ...}]``; diffusers/PEFT-format or
BFL-layout safetensors), so a styled deployment runs at the base model's speed. The adapter's
own scaling follows PEFT: an ``alpha`` key in the weights, else the ``lora_alpha`` (and
``use_rslora``) of an ``adapter_config.json`` / ``config.json`` next to them, else ``alpha = r``;
``scale`` multiplies that (a public rank-64 rsLoRA adapter lands within 53 dB of diffusers'
fused result — M* merges in fp32 with one rounding, PEFT fuses in bf16).
A DoRA adapter (PEFT's ``lora_magnitude_vector`` or Kohya's ``dora_scale``) is refused,
since its magnitudes would be dropped from the merge.
