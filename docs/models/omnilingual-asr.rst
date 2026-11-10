Omnilingual ASR
===============

Meta Omnilingual ASR, CTC variants: wav2vec 2.0 encoder + CTC head, speech-to-text
in 1,600+ languages with no language code. Not autoregressive; batched across
requests. Apache-2.0.

Registry key: ``omnilingual_asr_ctc_{300m,1b,3b,7b}``. Checkpoint: ``facebook/omniASR-CTC-300M`` (v1); v2 from Meta's CDN (default).

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``omnilingual_asr_ctc_300m``
     - ``configs/omnilingual_asr_ctc_300m.yaml``
   * - ``omnilingual_asr_ctc_1b``
     - ``configs/omnilingual_asr_ctc_1b.yaml``
   * - ``omnilingual_asr_ctc_3b``
     - ``configs/omnilingual_asr_ctc_3b.yaml``
   * - ``omnilingual_asr_ctc_7b``
     - ``configs/omnilingual_asr_ctc_7b.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve omnilingual_asr_ctc_300m

For client requests and response formats, see :doc:`/clients`.

Install and serve
-----------------

Serve with ``mstar serve omnilingual_asr_ctc_300m`` (or ``_1b``, ``_3b``, ``_7b``) and
transcribe through ``POST /v1/audio/transcriptions``. No extra is needed;
``pip install -e '.[audio]'`` adds non-WAV containers (mp3, flac, ...).

Checkpoint versions and offline files
-------------------------------------

The model is a port of Meta's `Omnilingual ASR <https://github.com/facebookresearch/omnilingual-asr>`_
CTC checkpoints (code and weights Apache-2.0) that loads Meta's own ``.pt`` files without
fairseq2. ``weights_version`` in the YAML ``model_kwargs`` picks the release: ``v2``
(default; lower CER per Meta; its vocabulary adds capitals and punctuation, though on our
FLEURS clips it still wrote lowercase, unpunctuated text) is only on Meta's
CDN (``dl.fbaipublicfiles.com/mms/``), which publishes no checksums, so the files are
checked against sha256 values pinned in ``mstar/model/omnilingual_asr/config.py``
(computed from our downloads); ``v1`` comes from the ``facebook/omniASR-CTC-*`` Hub repos
at pinned revisions. ``checkpoint_path`` / ``tokenizer_path`` take local files instead.

Audio preprocessing and window limits
-------------------------------------

Preprocessing follows the reference exactly: decode at the file's rate, resample each
channel to 16 kHz (torchaudio), average the channels, normalize the utterance to zero
mean and unit variance. ``/v1/audio/transcriptions`` transcribes a clip of up to 20 s
whole and cuts a longer upload into the fewest windows of at most 16 s, of about equal
length, each ending at the quietest 0.1 s within 5 s of its boundary, transcribed in
parallel. The model hears up to 40 s (the reference's limit, and the cap for
``window_seconds`` in ``extra_body``, which sets both lengths for one request), but on
FLEURS concatenations 40 s windows give more than twice the character error rate of
this policy. ``/generate`` refuses a clip over 40 s with a 400 rather than truncating it.

Ignored fields and timestamps
-----------------------------

The CTC head covers every language at once: ``language``, ``prompt`` and ``temperature``
have no effect (the reference ignores a language code for CTC models too), and
decoding is a per-frame argmax. There are no timestamps: ``srt`` / ``vtt`` get one cue
per window and ``verbose_json`` one segment per window, timed to the windows. Nothing detects the
language either: ``verbose_json``'s ``language`` is the request's ``language`` as given,
or null. ``/v1/realtime`` is not supported.

Batching and numerical precision
--------------------------------

``dtype`` defaults to ``bfloat16``, the reference's inference dtype; ``float32`` also
turns off TF32 for the node's matmuls. ``max_batch_size`` (16) caps the requests in one
step; the step sorts them by length and splits them so that each forward's padded audio
(rows times the longest clip) stays within ``max_batch_seconds`` (320). In bf16 the
encoder packs a step's clips into one unpadded forward (a captured graph up to 16384
frames) when their total audio fits ``max_batch_seconds``; a step over it encodes padded,
per split. In bf16 a
request batched with others can differ from the same request run alone where two
characters nearly tie, as the reference's own batched ``transcribe`` does;
``max_batch_size: 1`` or ``dtype: float32`` give batch-invariant transcripts.

Reference checks
----------------

Parity scripts against Meta's pipeline live in ``test/omnilingual_asr/``
(``record_oracle.py`` runs the reference in its own venv, as it pins torch 2.8 and
fairseq2).
