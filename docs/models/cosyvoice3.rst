CosyVoice3
==========

Zero-shot voice-cloning TTS: Qwen2-0.5B speech-token LM with RAS sampling,
DiT flow-matching decoder and causal HiFT vocoder, streamed in chunks.
Clones from a preset or an uploaded clip. 24 kHz.

Registry key: ``cosyvoice3``. Checkpoint: ``FunAudioLLM/Fun-CosyVoice3-0.5B-2512``.

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``cosyvoice3``
     - ``configs/cosyvoice3.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve cosyvoice3

For client requests and response formats, see :doc:`/clients`.

Install and supply a voice
--------------------------

Install with ``pip install -e '.[cosyvoice3]'`` and serve with
``mstar serve cosyvoice3 --gpus 0``. Every request clones a voice: a
``ref_audio`` clip (with its transcript in ``ref_text`` for zero-shot
cloning) or a preset from ``voices_dir`` (see ``configs/cosyvoice3.yaml``).
``instructions`` carries a style instruction. Output is 24 kHz.

Reference clip length
---------------------

Keep the reference clip short, a few seconds like CosyVoice's own 3.5 s
prompt clip: with a 23 s clip as the preset, round-trip word error rate rose
from 1.4% to 13.4%, with words dropped mid-sentence.

Request validation and ignored fields
-------------------------------------

Request knobs are strictly typed: ``ras_window`` and ``n_cfm_timesteps`` take a whole
float but not a string or a boolean; ``top_k`` (1-128) and ``max_new_tokens`` follow
the server's integer rule (:ref:`generation-parameters`); an out-of-range value is a
400. ``voice``, ``instruct`` and ``ref_text``
must be strings. ``max_new_tokens`` above the deployment's speech-token limit is
refused, not cut. ``seed`` fixes the LM's sampling; the flow and vocoder noise is
fixed, as in the reference. ``speed`` has no effect and is logged as ignored.

Long text
---------

On ``/v1/audio/speech``, text longer than one piece is synthesized as ordered
sentence pieces, as the reference frontend does: about 80 tokens (320
characters) of English, or 80 characters of text with Chinese, each piece
cloning the same voice. In one request the LM ends long text early or
garbles its end. A Chinese sentence over 300 characters, which the
reference keeps whole, is also cut at its commas, so that each piece stays
under the LM's 512 tokens of transcript and text.
``sentence_chunking: false`` sends the text whole.

Japanese text limitations
-------------------------

Japanese written in kana alone, with spaces between words, is read poorly,
as it is by the official model: in tests both garbled common words such as
ゆっくり and ありがとうございました on every seed.

Split flow and vocoder deployment
---------------------------------

``configs/cosyvoice3_split.yaml`` runs the flow and vocoder on their own
worker on the same GPU, with MPS and batches of 16, for throughput.

Flow cache memory
-----------------

A streamed row solves only each hop's new frames, reading the K/V of the
frames before them from ``resources.flow_kv`` (about 1.8 MB a frame). Past
``flow_cache_frames`` (768 by default, 15 s of prompt and speech) a row goes
back to re-solving every frame. The default pool holds that for 32 requests,
about 41 GiB. ``flow_cache_frames: 0`` turns it off, and so does a float32
estimator, which FlashInfer's kernels can't run.

Flow precision
--------------

The flow's DiT estimator runs in bfloat16 by default; the Euler state, the
rest of the flow and the HiFT vocoder stay float32 (HiFT's F0 predictor
float64).
``model_kwargs: {flow_estimator_dtype: float32}`` in the deployment YAML runs
the estimator at the reference's precision instead, at lower throughput.

Flow capture
------------

The flow's Euler solve is captured at startup as CUDA graphs, one per frame
bucket and batch size (``flow_capture_frame_buckets``,
``flow_capture_batch_sizes``, ``flow_capture_batch_frames``), which takes GPU
memory and startup time. ``model_kwargs: {flow_capture_frame_buckets: []}``
captures none, and every solve runs eagerly.

Vocoder capture and state
-------------------------

The streaming vocoder is captured the same way, one graph per window bucket
and batch size (``hift_capture_frame_buckets``, ``hift_capture_batch_sizes``,
about 3.3 GB at the defaults); each stream's mel tail and phase carry live in
the ``hift_state`` pool, a slot per request (``resources: hift_state:
{max_slots: N}``, one more than the requests). A chunk past the largest bucket
runs as several steps. ``hift_capture_frame_buckets: []`` runs the vocoder
eagerly.

Language-model prefill capture
------------------------------

The LM's prefill is captured with its decode, per prompt token bucket and
batch size (the serving config's ``prefill_token_buckets: {llm: {prefill_voice:
[...]}}``, see :doc:`/serving`, and ``lm_prefill_batch_sizes``; the LM's graphs
take about 130 MB together). A prompt past the largest bucket runs eagerly, and
``lm_prefill_batch_sizes: []`` captures no prefill. The flow's new-frame region
keeps ``model_kwargs.flow_new_frame_buckets``, which the key can't set.
