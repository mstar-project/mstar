Qwen3-Omni
==========

Omni-modal (text/image/audio/video in, text/audio out): Thinker + Talker + codec.

Registry key: ``qwen3_omni``. Checkpoint: ``Qwen/Qwen3-Omni-30B-A3B-Instruct``.

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``qwen3_omni``
     - ``configs/qwen3omni_2gpu.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve qwen3_omni

For client requests and response formats, see :doc:`/clients`.

Choose a deployment
-------------------

The default ``configs/qwen3omni_2gpu.yaml`` holds 1,024 GPU pages (128k tokens) of Thinker KV plus 1,024 host pages it offloads to. On H200-class GPUs, when concurrent requests hold more than 128k tokens of Thinker KV, serve ``configs/qwen3omni_2gpu_large_kv.yaml``, which keeps 2,048 pages (256k tokens) on the GPU and no host pages; it should fit beside its captures on an H200. On 80 GB GPUs such as the H100, serve ``configs/qwen3omni_2gpu_h100.yaml``. Its 512-page Thinker pool (64k tokens) was sized on an H100 when the encoders shared the Thinker's GPU and its text ladder ran to 8192: beside 1,024 pages 27 Thinker captures, all among those of 4096 tokens and more (text and vision), ran eagerly, and 640 pages still left two vision buckets out.

Chunked prefill now caps a Thinker step at 1,024 tokens, so those large buckets no longer replay, but they are still captured and still take that memory; whether 1,024 pages fit with today's ladders and encoder placement is unmeasured. The config runs the audio and vision encoders on the Talker's GPU, so an encoder step doesn't wait behind a Thinker step.

Bound concurrent context
------------------------

None of the three configs sets ``max_concurrent_requests``. With the Thinker's
prefix cache on, a decode step that finds the GPU pool full offloads a running
request to the host pages. Only when that frees nothing does it hold requests
until they time out: with no host pages (``large_kv``), with the host pages full,
or when the running requests' pages are mostly a prefix they share (see the
prefix cache under "Config files" in :doc:`/serving`). Where one of those can
happen and concurrent requests can together hold more Thinker KV than the GPU
pool, set the cap to the GPU pages divided by the pages of the longest request
served, prompt and output together, at 128 tokens a page: at the full 32k
context, 8 for ``large_kv``, 4 for the default and 2 for the H100 config.

Attachment order
----------------

A prompt runs one Thinker prefill per attachment: each attachment's row also
holds the text before it, and the last one the text after it, so text, a clip
and text is one step. A prompt with no attachment is one text prefill.

Audio chunks and playback
-------------------------

Streamed audio starts after 12 codec frames (0.96 s of audio), then 13, then
25-frame chunks behind 25 frames of left context. Playback stays gapless
while the Talker produces frames at about 1.5x real time over the first 50
frames (about 19 frames/s per request); the Talker batches at most 32
requests, so requests past that queue and can fall below it and pause. Set
``codec_chunk_schedule: []`` under the deployment YAML's ``model_kwargs`` to
start with a 25-frame chunk instead: first audio comes later, and the
Talker needs only about 1.0 to 1.3x real time.

Sampling precedence
-------------------

Qwen3-Omni's Talker cap comes from ``talker_max_new_tokens``, 4096 on the
released checkpoint (it was 2048). Plain sampling keys target the Thinker, except
on ``/v1/audio/speech``, where ``temperature`` and ``top_p`` set the Talker;
``talker_*`` and ``code_predictor_*`` set those stages on every route.


Disaggregated prefill and decode
--------------------------------

``configs/qwen3omni_pd_2gpu.yaml`` runs ``qwen3omni_pd_disaggregated.yaml``'s
prefill/decode split on two GPUs: the prefill Thinker shares rank 0 with the
encoders, the Talker and Code2Wav, and rank 1 runs the Thinker's decode. At
concurrency 1 its text matches ``configs/qwen3omni_2gpu.yaml``'s.



Audio and vision capture
------------------------

The audio and vision encoders attend through ragged attention resources
(``audio_attn`` within each 8 s window, ``vision_attn`` within each image or
video frame group), FlashInfer by default, so no ``flash-attn`` install is
needed. A deployment can pick the kernel under ``resources:`` in its YAML,
e.g. ``vision_attn: {flashinfer_backend: cutlass}``. Each encoder's block stack
is captured as a CUDA graph per token-count bucket; a request past the
largest bucket (8192 patches, about a 1920x1080 image; 4096 audio tokens)
runs it eagerly.

Text-only and audio output walks
--------------------------------

A request with no audio output decodes on its own walk, ``thinker_text_decode``:
its token stays on the device between steps and reaches the client inline, as
Qwen3.5's does, and nothing goes to the Talker. A request with audio output
keeps ``thinker_decode``, which streams every step's states to the Talker. A
captured step of either walk takes token ids and builds its embeds and rotary
tables in the graph.

Audio encoding and numerical differences
----------------------------------------

The audio encoder is the AuT node Qwen3-ASR runs, fed the API server's log-mel features: a step's clips (up to 8 requests) go through one packed forward, up to 4096 tokens. A step over 4096 tokens runs its clips one by one, each replayed if it fits a bucket. A clip's last 1 s chunk is zero-padded to a full chunk before the convolutions, as HF's encoder pads it whenever its batch holds a full chunk, so a clip under 1 s encoded alone differs from HF's: in its last conv row and, through the windowed attention, in every embedding of the clip.

A clip's embeddings can differ at the last bit with what it is batched beside, since the step's total tokens pick the captured bucket (or the eager GEMMs' shapes), and a prefix-cached audio prefix keeps the embeddings it was first computed with, so a later request reusing it inherits that first batch's rounding.

Decoder numerical differences
-----------------------------

Numerics: the Thinker's q/k RMSNorm and multimodal RoPE run in one kernel
with the KV write, in fp32 throughout and rounded once, where the unfused ops
(and Hugging Face) round to bf16 after the norm and after each step of the
rotation: no less accurate, but not bit-exact. The code predictor (shared
with Qwen3-TTS, see :doc:`/models/qwen3-tts`) runs each projection with its RMSNorm folded
in and its residual add in the same Triton kernel (``cp_linear``), and its
attention as another (``cp_attend``), neither bit-exact with the unfused
ops. Its 15 depth steps feed each sampled code into the next, so small
differences can change later codes at near-ties.

Thinker prefill buckets
-----------------------

The Thinker's text and audio prefill replay captured graphs on the ladder
``128 ... 2048`` tokens, and a longer step runs eagerly. The serving
config's ``prefill_token_buckets: {Thinker: {prefill_text: [...]}}`` sets a
deployment's own ladder (see :doc:`/serving`). Each bucket adds captures,
graph memory and startup time.

MoE routing and GPU kernels
---------------------------

The MoE layers route through the fused softmax-top-k kernel (``TopKRouter(fused_softmax_topk=True)``), which gives equal probabilities to the lower expert id where ``torch.topk`` may pick another. On one GPU a decode batch of up to 16 tokens also gets its block alignment from that kernel, unless it is small enough to run its experts as per-pair GEMVs, which need none: up to the ``moe_decode`` kernel table's crossover for the GPU and expert shape (``mstar/kernels/tables/moe_decode``), else the built-in 7, swept on an H100 (aligned from 8). On an H200 the TP1 Thinker runs no per-pair GEMVs: it takes the grouped GEMMs, aligned, from 1 token.

At 1-2 tokens per-pair decode was 5-11% slower than grouped on one H200 box (torch 2.10, Triton 3.6) and 4-6% faster on another (torch 2.12.1, Triton 3.7.1); at 4 tokens grouped beat the built-in 7 on both. A crossover keyed by GPU name can't suit both boxes, so it takes grouped, which trails per-pair by at most about 6% at 1-2 tokens, whereas per-pair would trail grouped by up to about 11%. The Thinker at TP2 (expert width 384) has no H200 entry yet, so it keeps 7 there, unmeasured.
