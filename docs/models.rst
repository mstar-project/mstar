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
   * - ``cosmos3``
     - ``nvidia/Cosmos3-Nano``
     - Cosmos3 world model: t2i/t2v/i2v/v2v diffusion, robot-action modes, opt-in sound.
   * - ``cosmos3_super``
     - ``nvidia/Cosmos3-Super``
     - Cosmos3-Super (64B) variant of the above; TP/SP for multi-GPU serving.
   * - ``orpheus``
     - ``canopylabs/orpheus-3b-0.1-ft``
     - TTS: Llama 3.2 3B LLM emitting audio tokens + SNAC 24 kHz decoder.
   * - ``pi05``
     - ``lerobot/pi05_base``
     - Pi0.5 vision-language-action robotics model (ViT encoder + LLM + flow action expert).
   * - ``qwen3_omni``
     - ``Qwen/Qwen3-Omni-30B-A3B-Instruct``
     - Omni-modal (text/image/audio/video in, text/audio out): Thinker + Talker + codec.
   * - ``vjepa2``
     - ``facebook/vjepa2-vitl-fpc64-256``
     - V-JEPA 2 video encoder + masked predictor.
   * - ``vjepa2_ac``
     - ``vjepa2-ac-vitg``
     - V-JEPA 2-AC encoder + action-conditioned predictor.
   * - ``wan22``
     - ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``
     - Wan2.2-TI2V-5B video diffusion: text-to-video and image-to-video, 5B dense DiT.

Notes
-----

- The IDs above are representative. You may use local paths or compatible variants.
- Some families accept multimodal input (image/audio/video); see the model's
  ``process_prompt`` for the inputs it expects.
- To add a new family, see :doc:`adding_models`.

Cosmos3 environment requirements
--------------------------------

- ``flashinfer`` is required: it is the paged KV/attention backend used by the
  prefill, the captured CUDA graphs, and multi-request batches.
- The default denoise attention backend is ``dense_gen``
  (``Cosmos3Config.attention_backend``), which runs bs=1 eager generation
  attention as one FlashAttention-3 varlen kernel from the ``fa3-fwd`` wheel.
  That wheel is ABI-tied to the installed torch/CUDA build (Hopper builds
  exist for at least torch 2.9 + cu12.8 and torch 2.11 + cu13.0); install the
  one matching your environment. When it is not importable, the engine logs a
  warning at startup and automatically falls back to the paged ``flashinfer``
  backend — serving still works, only the bs=1 dense fast path is lost.
  ``model_kwargs.attention_backend: flashinfer`` in the config YAML selects
  the paged backend explicitly.
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

**UniPC runs inline, inside the DiT node.** Unlike cosmos3, the scheduler is not
a separate stage: the solver state (the order-2 history buffer and the
corrector's ``last_sample``) is carried on the denoise loop's own edges, so it
travels with the request rather than living in a scheduler object on one rank.
Requests are therefore independent and the loop is resumable across ranks.

**Nothing is accelerated by default.** wan22 serves the DiT eager: no
``torch.compile``, no CUDA-graph capture, no continuous batching, no component
offload, and the VAE decode is always tiled (which bounds its workspace so the
untiled conv3d cannot OOM a 32 GiB card).
