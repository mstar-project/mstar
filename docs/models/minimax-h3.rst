MiniMax-H3
==========

MiniMax-H3 (FL2VA) text- and first/last-frame-to-audio+video: a 33B packed-sequence
audio-video DiT, CFG-distilled (one forward per step, 50 sigma points by default), a
Qwen3-VL-32B conditioner (keyframes through its vision tower) and a ViT video VAE,
emitting an H.264 video at 24 fps with 32 kHz stereo audio (one mp4 with AAC on
``/v1/videos/generations``). MiniMax H3 Community License Agreement.

Registry key: ``minimax_h3`` *(Beta)*. Checkpoint: ``MiniMaxAI/MiniMax-H3``.

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``minimax_h3``
     - ``configs/minimax_h3.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve minimax_h3

For client requests and response formats, see :doc:`/clients`.

Text- and first/last-frame-to-audio+video on **MiniMax-H3** (``MiniMaxAI/MiniMax-H3``,
its ``FL2VA`` checkpoint; MiniMax H3 Community License Agreement): a 33B packed-sequence
audio-video DiT, guidance-distilled (one forward per step), a Qwen3-VL-32B conditioner
and a ViT video VAE, at 24 fps with 32 kHz stereo audio. Not ported: ``Ref2VA``, the
H3-Context-IR prompt rewriter and the 2K regenerator.

Execution graph
---------------

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

Request fields and limits
-------------------------

Request knobs (``model_kwargs``): ``height`` / ``width`` (multiples of 32, aspect 1:4 to 4:1; default 768x1344, or the first keyframe's aspect), ``num_frames`` (snapped up to 17n+5, 5 to 15 s; default 124), ``num_inference_steps`` (sigma points, default 50: 49 forwards), ``flow_shift`` / ``audio_flow_shift`` (12 / 3), ``seed``, and with one or two images ``keyframe_anchors`` (``first`` / ``last`` per image). These are a 400: a prompt over the text encoder's 262,144 positions (keyframe tokens included; ``max_prompt_tokens`` lowers it), a packed sequence over ``max_packed_rows`` (default 131072: the longest 768p clip, 14.4 s, with a ~25k-token prompt), Qwen's vision tokens in the prompt, and a last keyframe outside 1:4 to 4:1 next to a first one (it is cropped to the canvas).

One request runs per denoise step. ``POST /v1/videos/generations`` returns one mp4 with the audio as AAC (``image`` is the first frame, ``extra_body.last_image`` the last; the mux needs ``ffmpeg`` on ``PATH``).

Deployment settings
-------------------

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
- ``attention_backend``: ``flashinfer``, or ``sdpa`` (the reference's kernel).

Runtime dependencies and GPU memory
-----------------------------------

Async scheduling is on for the DiT and the decoder, the default for parallel nodes
(``tp_async_sched: false`` turns it off); on and off measured within noise. Runtime needs:
PyAV, transformers' image processor (a server that can't import it, e.g. next to a
torchaudio built for another torch, refuses to start), FlashInfer, and ``ffmpeg``. Each DiT
rank holds the whole DiT, and the SP configs put the conditioner beside it on rank 0 (117 GiB
in use once loaded), so they need 141 GB GPUs (H200); no config has been run on 80 GB cards.

Numerical parity
----------------

At ``attention_backend: sdpa`` the nodes' modules, driven by ``test/minimax_h3/parity.py``
(fp32 matmuls at "highest"), are bit-identical to diffusers' MiniMax-H3 pipeline end to end
(latents, frames, waveform); with ``--exact`` the script exits nonzero when one isn't.
FA3 and the compiled DiT are not bit-exact. Served at 768p (50 steps, SP2 on H200, three
seeds), clips are 23.7-32.8 dB (video) and 17.8-22.3 dB (audio) from diffusers', where the
reference on another SDPA kernel is 23.5-33.1 and 21.6-22.3 dB from itself; per seed, the
served clip is below that spread on one seed's video and another's audio.

Measured latency
----------------

Seconds per clip, one request at a time, each engine on the same H200s (M* on torch
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
