Supported models
================

Choose a model by the task you need to serve, then open its guide for checkpoint access,
dependencies, configuration, request options and limits. A registry key is the value of
``model:`` in a serving YAML file; the CLI maps model keys to deployment configs in
``mstar/cli/main.py``. Checkpoints and classes are registered in ``mstar/model/registry.py``.

Start with :doc:`quickstart` for a first request. :doc:`installation` covers dependencies,
and :doc:`serving` explains GPU placement. The config a model ships is a starting
layout, not a hardware minimum: use the model's measured setup and capacity notes
before changing GPU count, precision or concurrency.

Text and multimodal chat
------------------------

.. list-table::
   :header-rows: 1

   * - Registry keys
     - Model guide
     - Use it for
   * - ``qwen3_5_{0.8,2,4,9,27}b``, ``qwen3_8_27b``
     - :doc:`Qwen3.5 and Qwen3.8 </models/qwen>`
     - Text, image and video → text
   * - ``muse_glimmer``
     - :doc:`Muse Glimmer </models/muse-glimmer>`
     - Text and image → text; beta
   * - ``ripple_65b``
     - :doc:`Ripple-65B </models/ripple>`
     - Text → text; experimental
   * - ``command_a_plus``
     - :doc:`Command A+ </models/command-a-plus>`
     - Text and image → text; experimental
   * - ``glm52``, ``glm5_next``
     - :doc:`GLM-5.2 and GLM-5.3-Flash </models/glm>`
     - Text → text; large MoE checkpoints
   * - ``qwen3_omni``
     - :doc:`Qwen3-Omni </models/qwen3-omni>`
     - Text, image, audio and video → text and speech
   * - ``bagel``
     - :doc:`BAGEL </models/bagel>`
     - Text and image understanding, generation and editing

Speech generation and conversation
----------------------------------

.. list-table::
   :header-rows: 1

   * - Registry keys
     - Model guide
     - Use it for
   * - ``kokoro``
     - :doc:`Kokoro </models/kokoro>`
     - Bundled voices and voice blends
   * - ``orpheus``
     - :doc:`Orpheus </models/orpheus>`
     - Speech from text
   * - ``chatterbox``, ``chatterbox_multilingual``, ``chatterbox_turbo``
     - :doc:`Chatterbox </models/chatterbox>`
     - Voice cloning; English, multilingual and Turbo
   * - ``cosyvoice3``
     - :doc:`CosyVoice3 </models/cosyvoice3>`
     - Voice cloning from a clip or preset
   * - ``qwen3_tts``, ``qwen3_tts_1p7b``, ``qwen3_tts_voicedesign``, ``qwen3_tts_base``
     - :doc:`Qwen3-TTS </models/qwen3-tts>`
     - Built-in speakers, voice design or cloning
   * - ``fish_s2_pro``
     - :doc:`Fish Audio S2-Pro </models/fish-s2-pro>`
     - Voice cloning and inline prosody tags; non-commercial weights
   * - ``voxcpm2``
     - :doc:`VoxCPM2 </models/voxcpm2>`
     - Zero-shot speech, voice design, cloning or continuation
   * - ``omnivoice``
     - :doc:`OmniVoice </models/omnivoice>`
     - Multilingual voice cloning or voice description
   * - ``nemotron_duplex``
     - :doc:`Nemotron VoiceChat </models/nemotron-voicechat>`
     - Live speech → agent text and speech; beta

Speech recognition
------------------

.. list-table::
   :header-rows: 1

   * - Registry keys
     - Model guide
     - Use it for
   * - ``whisper_large``, ``whisper_large_v3_turbo``, ``qwen3_asr``, ``qwen3_asr_realtime``
     - :doc:`Whisper and Qwen3-ASR </models/whisper-qwen-asr>`
     - File transcription; Whisper timestamps; Qwen realtime input
   * - ``higgs_audio``
     - :doc:`Higgs Audio </models/higgs-audio>`
     - Speech → text; beta
   * - ``omnilingual_asr_ctc_{300m,1b,3b,7b}``
     - :doc:`Omnilingual ASR </models/omnilingual-asr>`
     - CTC transcription in 1,600+ languages
   * - ``canary``, ``canary_qwen``, ``parakeet``, ``nemotron_asr``
     - :doc:`Canary, Parakeet and Nemotron ASR </models/fastconformer-asr>`
     - Recognition and selected translation tasks

Images, video and robotics
--------------------------

.. list-table::
   :header-rows: 1

   * - Registry keys
     - Model guide
     - Use it for
   * - ``flux2_klein``, ``flux2_klein_9b``, ``z_image_turbo``
     - :doc:`FLUX.2 klein and Z-Image </models/flux-z-image>`
     - Image generation; FLUX reference-image editing
   * - ``cosmos3``, ``cosmos3_droid``, ``cosmos3_edge``, ``cosmos3_edge_droid``, ``cosmos3_super``, ``cosmos3_super_i2v_4step``, ``cosmos3_super_t2i_4step``
     - :doc:`Cosmos3 </models/cosmos3>`
     - Images, videos and robot actions; Edge reasoning
   * - ``wan22``
     - :doc:`Wan2.2 </models/wan22>`
     - Text or image → video; TI2V-5B only
   * - ``ltx2_5``
     - :doc:`LTX-2.5 </models/ltx2-5>`
     - Text → audio and video; beta, gated weights
   * - ``minimax_h3``
     - :doc:`MiniMax-H3 </models/minimax-h3>`
     - Text and keyframes → audio and video; beta
   * - ``pi05``, ``vjepa2``, ``vjepa2_ac``, ``waypoint``, ``lingbot_world_1_3b``
     - :doc:`Robotics and interactive world models </models/world-models>`
     - Robot actions, video prediction and interactive rollouts

.. _notes:

Compatibility and defaults
--------------------------

- Models marked *(Beta)* are functionally supported but not yet
  performance-optimized; treat their throughput/latency as provisional.

- The checkpoint IDs in the model guides are representative. You may use local paths or compatible variants.

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
  top_k 0, top_p 1.0; Orpheus temperature 0.6, top_p 0.8).


Per-request adapters
--------------------

See :doc:`/models/lora` for supported targets, validation status and limitations.

Model guide links
-----------------

The former sections on this page now have dedicated guides.

.. _qwen3-5-qwen3-8-qwen3-5-qwen3-8-27b:

.. rubric:: Qwen3.5 / Qwen3.8 (``qwen3_5_*``, ``qwen3_8_27b``)

See :doc:`/models/qwen`.

.. _muse-glimmer-muse-glimmer:

.. rubric:: Muse Glimmer (muse_glimmer)

See :doc:`/models/muse-glimmer`.

.. _ripple-65b-ripple-65b:

.. rubric:: Ripple-65B (ripple_65b)

See :doc:`/models/ripple`.

.. _command-a-command-a-plus:

.. rubric:: Command A+ (command_a_plus)

See :doc:`/models/command-a-plus`.

.. _glm-5-2-glm-5-3-flash-glm52-glm5-next:

.. rubric:: GLM-5.2 / GLM-5.3-Flash (glm52, glm5_next)

See :doc:`/models/glm`.

.. _qwen3-omni-notes:

.. rubric:: Qwen3-Omni

See :doc:`/models/qwen3-omni`.

.. _kokoro-notes:

.. rubric:: Kokoro

See :doc:`/models/kokoro`.

.. _orpheus-notes:

.. rubric:: Orpheus

See :doc:`/models/orpheus`.

.. _chatterbox-notes:

.. rubric:: Chatterbox

See :doc:`/models/chatterbox`.

.. _cosyvoice3-notes:

.. rubric:: CosyVoice3

See :doc:`/models/cosyvoice3`.

.. _qwen3-tts-notes:

.. rubric:: Qwen3-TTS

See :doc:`/models/qwen3-tts`.

.. _fish-audio-s2-pro-notes:

.. rubric:: Fish Audio S2-Pro

See :doc:`/models/fish-s2-pro`.

.. _voxcpm2-notes:

.. rubric:: VoxCPM2

See :doc:`/models/voxcpm2`.

.. _omnivoice-notes:

.. rubric:: OmniVoice

See :doc:`/models/omnivoice`.

.. _nemotron-voicechat-nemotron-duplex-notes:

.. rubric:: Nemotron VoiceChat (nemotron_duplex)

See :doc:`/models/nemotron-voicechat`.

.. _higgs-audio-notes:

.. rubric:: Higgs Audio

See :doc:`/models/higgs-audio`.

.. _omnilingual-asr-notes:

.. rubric:: Omnilingual ASR

See :doc:`/models/omnilingual-asr`.

.. _fastconformer-asr-canary-canary-qwen-parakeet-nemotron-asr-notes:

.. rubric:: FastConformer ASR (canary, canary_qwen, parakeet, nemotron_asr)

See :doc:`/models/fastconformer-asr`.

.. _qwen3-asr-qwen3-asr-qwen3-asr-realtime-notes:

.. rubric:: Qwen3-ASR (qwen3_asr, qwen3_asr_realtime)

See :doc:`/models/whisper-qwen-asr`.

.. _bagel-notes:

.. rubric:: BAGEL

See :doc:`/models/bagel`.

.. _cosmos3-environment-requirements:

.. rubric:: Cosmos3 environment requirements

See :doc:`/models/cosmos3`.

.. _cosmos3-image-to-video-conditioning:

.. rubric:: Cosmos3 image-to-video conditioning

See :doc:`/models/cosmos3`.

.. _cosmos3-edge-reasoner-and-action-loop:

.. rubric:: Cosmos3-Edge reasoner and action loop

See :doc:`/models/cosmos3`.

.. _cosmos3-streaming-rollout-windowed-video:

.. rubric:: Cosmos3 streaming rollout (windowed video)

See :doc:`/models/cosmos3`.

.. _wan2-2-wan22:

.. rubric:: Wan2.2 (wan22)

See :doc:`/models/wan22`.

.. _ltx-2-5-ltx2-5:

.. rubric:: LTX-2.5 (ltx2_5)

See :doc:`/models/ltx2-5`.

.. _minimax-h3-minimax-h3:

.. rubric:: MiniMax-H3 (minimax_h3)

See :doc:`/models/minimax-h3`.

.. _flux-2-klein-flux2-klein-flux2-klein-9b:

.. rubric:: FLUX.2 [klein] (flux2_klein / flux2_klein_9b)

See :doc:`/models/flux-z-image`.

.. _per-request-lora-adapters:

.. rubric:: Per-request LoRA adapters

See :doc:`/models/lora`.

.. toctree::
   :hidden:
   :maxdepth: 1

   models/qwen
   models/muse-glimmer
   models/ripple
   models/command-a-plus
   models/glm
   models/qwen3-omni
   models/bagel
   models/kokoro
   models/orpheus
   models/chatterbox
   models/cosyvoice3
   models/qwen3-tts
   models/fish-s2-pro
   models/voxcpm2
   models/omnivoice
   models/nemotron-voicechat
   models/whisper-qwen-asr
   models/higgs-audio
   models/omnilingual-asr
   models/fastconformer-asr
   models/flux-z-image
   models/cosmos3
   models/wan22
   models/ltx2-5
   models/minimax-h3
   models/world-models
   models/lora
