Fish Audio S2-Pro
=================

Fish Audio S2-Pro: dual-AR TTS (Qwen3-4B Slow AR + 4-layer Fast AR over 10 codebooks),
44.1 kHz causal DAC codec, voice cloning from one reference clip, inline ``[tag]``
prosody control. Weights under the Fish Audio Research License (non-commercial use
only); check it before deploying.

Registry key: ``fish_s2_pro``. Checkpoint: ``fishaudio/s2-pro``.

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``fish_s2_pro``
     - ``configs/fish_s2_pro.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve fish_s2_pro

For client requests and response formats, see :doc:`/clients`.

Weights license
---------------

The ``fishaudio/s2-pro`` weights are under the Fish Audio Research License:
research and non-commercial use only. M* downloads them at runtime and does
not redistribute them; read the license before deploying.

Install and serve
-----------------

``pip install -e '.[fish_s2_pro]'`` then ``mstar serve fish_s2_pro --gpus 0``.

Offline voice-clone warmup
--------------------------

The voice-clone warmup at startup uses ``samples/af_heart_0.wav`` from
``hexgrad/Kokoro-82M``. An offline deployment that pre-fills its HF cache
needs that file as well; without it the warmup is skipped with a warning and
the first clone request pays the one-time cost.

Voice cloning and prosody
-------------------------

Requests: ``/v1/audio/speech`` with ``input``; inline ``[tag]`` prosody
control is part of the text. Clone a voice with ``ref_audio`` (data URL,
base64, or an http(s) URL when the server allows remote fetch) and its transcript in
``ref_text``; with the SDK, ``client.tts(text, reference_audio="ref.wav",
ref_text="...")``. ``voice`` and ``speed`` are logged as ignored.

Sampling and output limits
--------------------------

Sampling defaults follow the reference (temperature 0.8, top-p 0.8, top-k 30);
``do_sample: false`` is greedy. The order differs: M\*'s sampler applies the
temperature first and takes top-k and top-p over the tempered probabilities,
while the reference takes top-p over the untempered ones, so at the same
values M\* keeps a slightly different set of tokens per step. The top-p
boundary differs as well: M\*'s sampler (FlashInfer) keeps the token whose
probability carries the running total past top-p, while the reference drops
it unless it is the most likely token. ``max_new_tokens`` (default 2048, at most
4096 frames of 2048 samples at 44.1 kHz) and ``ignore_eos`` pass through.

Streaming codec state and precision
-----------------------------------

The codec decodes each chunk's new frames from its stream's decoder state
(the ``codec_state`` recurrent pool, 64 streams by default, as many as the AR
decodes in a step, about 0.55 GiB; size it with
``resources: {codec_state: {max_slots: ...}}``), so every frame is decoded
once. It decodes with TF32 by default: TF32 roughly halves its conv stack
and the audio stays within ~60 dB SNR of the exact decode, both measured on
the earlier windowed codec, not yet on the frame-once codec. With
``model_kwargs: {codec_exact_fp32: true}`` it runs in fp32 with TF32 off and
the streamed audio matches the whole-utterance decode to within one int16 step.
