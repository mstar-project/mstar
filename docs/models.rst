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
   * - ``orpheus``
     - ``canopylabs/orpheus-3b-0.1-ft``
     - TTS: Llama 3.2 3B LLM emitting audio tokens + SNAC 24 kHz decoder.
   * - ``pi05``
     - ``lerobot/pi05_base``
     - Pi0.5 vision-language-action robotics model (ViT encoder + LLM + flow action expert).
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
   * - ``vjepa2``
     - ``facebook/vjepa2-vitl-fpc64-256``
     - V-JEPA 2 video encoder + masked predictor.
   * - ``vjepa2_ac``
     - ``vjepa2-ac-vitg``
     - V-JEPA 2-AC encoder + action-conditioned predictor.
   * - ``whisper_large`` *(Beta)*
     - ``openai/whisper-large-v3``
     - Encoder-decoder ASR (audio in, transcript out). Beta / un-optimized.
   * - ``higgs_audio`` *(Beta)*
     - ``bosonai/higgs-audio-v3-stt``
     - Audio-tower + Qwen3 LLM speech-to-text. Beta / un-optimized.
   * - ``wan22``
     - ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``
     - Wan2.2-TI2V-5B video diffusion: text-to-video and image-to-video, 5B dense DiT.

Notes
-----

- Models marked *(Beta)* are functionally supported but not yet
  performance-optimized; treat their throughput/latency as provisional.
- The IDs above are representative. You may use local paths or compatible variants.
- Some families accept multimodal input (image/audio/video); see the model's
  ``process_prompt`` for the inputs it expects.
- To add a new family, see :doc:`adding_models`.

OmniVoice notes
~~~~~~~~~~~~~~~

- Zero-shot only: there are no built-in speakers. Pass ``ref_audio`` with its
  transcript in ``ref_text`` to clone a voice, or describe one in ``voice``.
  ``ref_text`` is required alongside ``ref_audio``.
- ``language`` takes either the name (``Vietnamese``) or the id (``vi``): a
  name is resolved to the id the model was trained on before the prompt is
  built, and an unrecognised value warns and falls back to language-agnostic
  mode.
- The ``omnivoice`` package is installed separately from git rather than by an
  extra — see :doc:`installation`.
- The backbone is not autoregressive: it fills a fixed canvas of eight codebook
  rows over a few unmasking steps, so there is no KV cache and no per-token
  sampling loop. Serve it with ``mstar serve omnivoice --gpus 0``.

Qwen3-TTS notes
---------------

- Install the model-specific dependencies with ``pip install -e '.[qwen3_tts]'``
  and launch the default single-GPU deployment with
  ``mstar serve qwen3_tts --gpus 0``.
- The first integration supports the CustomVoice checkpoint and text-to-audio
  requests. ``voice`` selects one of the checkpoint's built-in speakers and
  ``language`` defaults to automatic detection.
- Codec CUDA graphs are captured through batch size 8. The upstream decoder's
  batch-16 capture can exhaust an H100 after Talker weights and CodePredictor
  graphs are resident; larger Codec batches therefore use the scheduler's safe
  ceiling.
- Talker prefill remains eager because it runs once with variable sequence
  lengths. Decode always uses the whole-walk CUDA Graph, with the 15-step
  CodePredictor loop captured inside it; request-local EOS suppression is
  carried as a graph tensor input so replay does not consult capture-slot dummy
  request state. Residual ``subtalker_*`` sampling is per-request through the
  ``code_predictor`` aux sampler, so custom values neither block batching nor
  fall off the graph.
- The 12 Hz decoder does not require the system SoX executable. M* imports only
  the exact upstream decoder modules, avoiding qwen-tts's unrelated 25 Hz SoX
  probe during worker startup.

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
bucket); a request's tokens are the same whether it runs alone or in a batch. Generation requests batch
into one denoise pass too, which is the same maths but not the same bf16
arithmetic — under classifier-free guidance the branch rounding is amplified,
so an image or clip produced alongside other requests differs from its solo
result at the kernel-drift level (~30 dB PSNR at guidance 6). Serve with one
request at a time when outputs must be bitwise repeatable.

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
     - —
     - ``chained``: every window is a full bidirectional denoise conditioned on
       the previous window's tail (``overlap_frames`` pinned clean). ``kv``: the
       finished window's clean K/V is committed to the cache and later windows
       attend to it block-causally; no overlap, and frames older than
       ``context_frames`` behind the frontier are released from the cache
       (the persistent world state of a long rollout stays bounded).
   * - ``window_frames``
     - 29
     - Pixel frames per window (quantized to latent frames).
   * - ``overlap_frames``
     - 8
     - ``chained`` only: frames re-pinned from the previous window (at least two
       latent frames).
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
       and the decoder its context (the most recent ``session_store_size`` sessions).
   * - ``resume_session``
     - ``false``
     - Continue the named session: window 0 is conditioned on the stored last
       frames (pinned clean, like a chained overlap) and only the ``num_frames``
       new frames are delivered — a new prompt steers the same world.

.. code-block:: bash

   curl -sN http://localhost:8000/generate \
     -F 'text=a drone flies over a coastal town at dawn' \
     -F 'output_modalities=video' \
     -F 'model_kwargs={"num_frames":241,"window_mode":"kv","window_frames":29,"context_frames":61,"stream_video":true}'

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

**UniPC runs inline, inside the DiT node.** Unlike cosmos3, the scheduler is not
a separate stage: the solver state (the order-2 history buffer and the
corrector's ``last_sample``) is carried on the denoise loop's own edges, so it
travels with the request rather than living in a scheduler object on one rank.
Requests are therefore independent and the loop is resumable across ranks.

**Nothing is accelerated by default.** wan22 serves the DiT eager: no
``torch.compile``, no CUDA-graph capture, no continuous batching, no component
offload, and the VAE decode is always tiled (which bounds its workspace so the
untiled conv3d cannot OOM a 32 GiB card).
