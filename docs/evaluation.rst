Evaluation and Benchmarks
=========================

``python -m benchmark.eval`` measures a served model end to end: speed (latency,
throughput, streaming stalls) and quality (WER, speaker similarity, MOS, duplex turn
timing) from the same requests, repeated, with every number written to one results
file that also records the commit, GPU, versions and settings that produced it. The
same client can drive M\*, vLLM-Omni and SGLang-Omni, so comparisons use one metric
definition.

It lives in ``benchmark/eval/`` (not shipped in the wheel). Install the optional
dependencies with the ``eval`` extra, ideally in a separate client environment::

    pip install -e ".[eval]"

Quick start
-----------

Build a dataset once, then run a suite against a server the runner launches itself::

    python -m benchmark.eval data librispeech --out ~/eval-data/ls_clean --n 200

    python -m benchmark.eval run tts --serve kokoro --voice af_heart \
        --data benchmark/assets/t2s.txt --concurrency 1,4,16 --repeats 3 --out results/kokoro

    python -m benchmark.eval report results/kokoro --md kokoro.md \
        --chart kokoro.png --chart-metrics ttfa_ms_p50,audio_s_per_s,wer

``--serve MODEL`` runs ``mstar serve MODEL`` with its own port, ZMQ socket prefix and
upload directory, waits for ``GET /health``, logs to ``<out>/server-<label>.log`` and
stops the whole process group afterwards. ``--url`` measures a server that is already
running, and ``--serve-cmd "<command with {port}>"`` launches anything else (for
example another engine for a side-by-side run).

How a run is measured
---------------------

A run is one suite at one or more load points. For each point the runner sends
``--warmup`` untimed requests at the measured concurrency, then ``--repeats`` passes
over the items, scoring each pass on its own. A results file holds one point.

* **Closed loop** (``--mode closed``, default): ``--concurrency`` requests in flight at
  all times. ``--concurrency 1,4,16`` sweeps; each level writes its own file.
* **Open loop** (``--mode open --rate 1,2,4``): Poisson arrivals at each rate, seeded
  (``--arrival-seed``) so every system sees the same schedule. ``--concurrency N`` caps
  in-flight requests; latency is measured from the scheduled arrival, so time spent
  waiting for a slot counts (``slot_wait_s``). Each request also records how late the
  client was against the schedule (``arrival_lateness_s``); a large
  ``arrival_lateness_s_p99`` means the client, not the server, set the pace, and the
  point should not be used.
* **Repeats**: every metric is one number per repeat (a p50 over that repeat's
  requests, a corpus WER, ...); the summary is mean +- sample std over repeats.
  Three repeats is the default and what a PR table should show.

Suites
------

``tts``
  Streaming ``/v1/audio/speech``. Per request: time to first audio (first PCM byte
  after the WAV header), end-to-end latency, audio seconds, RTF, and the playback
  stalls a real-time player fed by the stream would hit (``--stall-prebuffer-ms``,
  ``--stall-min-ms``). Per repeat, optionally: round-trip WER (``--asr-judge``, Whisper
  large-v3 by default; CER for zh/ja/ko/th), speaker similarity to each item's
  reference clip (``--clone`` sends it as ``ref_audio``; ``--sim-judge``) and UTMOS
  (``--mos``). ``--engine vllm-omni|sglang-omni`` reads their raw PCM streams.
``asr``
  ``/v1/audio/transcriptions``. Corpus WER (CER for character languages) against the
  manifest, latency percentiles, RTFx (audio seconds per wall second), and with
  ``--stream`` the time to the first transcript delta. A failed request scores as an
  empty transcript, so errors never lower the WER.
``duplex``
  Streamed ``/generate`` for frame-synchronous speech-to-speech models
  (``nemotron_duplex``). The output is time-aligned with the input, so latencies are
  measured in interaction time from a replay: response latency after the user stops,
  whether the agent took the turn, and on barge-in clips the stop latency and whether
  it yielded; plus wall time per frame and whether each session kept up with real
  time. ``data duplex-synth`` builds turn-taking and barge-in clips from any 16 kHz
  ASR manifest.
``fdb`` (opt-in, non-commercial data)
  `Full-Duplex-Bench <https://github.com/DanielLin94144/Full-Duplex-Bench>`_ v1.5 on the
  same streamed ``/generate`` client: stop and response latency, LLM-judged behavior
  labels, semantic quality, and the agreement between judges. See
  :ref:`full-duplex-bench` below.

Datasets
--------

A dataset is a JSONL manifest, one item per line, with paths relative to the
manifest (a ``.txt`` file of sentences also works for ``tts``)::

    {"id": "1089-134686-0000", "text": "...", "audio_path": "audio/1089-134686-0000.wav", "lang": "en"}

``ref_audio`` and ``ref_text`` add a voice-clone prompt; ``extra`` carries suite
fields (duplex ``category``, ``speech_end_s``, ``interrupt_s``). An ``id`` names the
item's saved audio, so it must be a plain name (letters, digits, ``.``, ``_``, ``-``).
Builders:

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - ``data`` kind
     - Source
     - Terms
   * - ``librispeech``
     - ``openslr/librispeech_asr`` (``--split test.clean``)
     - CC BY 4.0
   * - ``fleurs``
     - ``google/fleurs`` (``--lang en_us``, ``cmn_hans_cn``, ...)
     - CC BY 4.0
   * - ``seedtts``
     - Seed-TTS-eval test set (target text plus prompt clip), Google Drive via gdown
     - research use; EN prompts come from Common Voice, whose terms forbid trying to
       identify speakers
   * - ``duplex-synth``
     - composed from an ASR manifest
     - inherits the source's terms
   * - ``fdb-v15``
     - Full-Duplex-Bench v1.5 (498 pairs), the official Google Drive via gdown
       (``--per-category N`` takes the first N of each category)
     - CC BY-NC 4.0, non-commercial use only; never vendored

Downloads go to the Hugging Face cache, so set ``HF_HOME`` to a large disk on GPU
hosts and reuse it across runs.

Quality judges
--------------

Model-based scores depend on the judge, so every results file names it, and tables
should only compare runs scored by the same judge and normalizer.

* WER normalizers (``--normalizer``): ``whisper_en`` (Whisper's English normalizer, the
  ASR convention, default), ``seedtts`` (the Seed-TTS-eval recipe: punctuation removed
  except the apostrophe), ``whisper_basic``, ``basic``. Both the corpus rate (``wer``)
  and the per-utterance mean (``wer_utt_mean``, what Seed-TTS-eval publishes) are kept.
* Speaker similarity uses ``microsoft/wavlm-base-plus-sv`` x-vectors; Seed-TTS-eval's
  published SIM uses a WavLM-large ECAPA checkpoint, so absolute values differ.
* UTMOS is UTMOS22 strong from ``tarepan/SpeechMOS`` on torch.hub.

Judges run between repeats, after the load has finished, so they never compete with
measured requests; ``--judge-device`` moves them to another GPU or the CPU.

.. _full-duplex-bench:

Full-Duplex-Bench v1.5
----------------------

.. note::

   Full-Duplex-Bench's code and data are licensed
   `CC BY-NC 4.0 <https://github.com/DanielLin94144/Full-Duplex-Bench/blob/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/LICENSE>`_:
   non-commercial use only. This repository contains none of it. ``data fdb-v15``
   downloads the v1.5 subsets from the benchmark's official Google Drive, and the
   suite downloads the official scoring code (``get_timing.py``, ``eval_behavior.py``,
   ``instruction/behavior.txt``, ``get_transcript/asr.py``) at the pinned revision
   ``3e799c45`` into the Hugging Face cache, checking each file's SHA-256. Using the
   suite means accepting those terms.

The ``fdb`` suite follows sglang-omni's runbook (PR #2668, ``benchmarks/duplex/fdb_v15``)
metric for metric, so its numbers can be compared with that runbook's. v1.5 has 498
pairs in four categories: ``user_interruption`` (200), ``user_backchannel`` (98),
``talking_to_other`` (100) and ``background_speech`` (100). A pair is the user's query
with a later overlapping event (``input.wav``) and the query alone
(``clean_input.wav``); each is its own session::

    python -m benchmark.eval data fdb-v15 --out ~/eval-data/fdb --per-category 12   # or all

    # a local judge on the same GPU, started first so it gets a fixed share of memory
    vllm serve Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
        --served-model-name qwen3.8-27b --port 30000 --reasoning-parser qwen3 \
        --gpu-memory-utilization 0.42 --max-model-len 65536

    python -m benchmark.eval run fdb --serve nemotron_duplex \
        --data ~/eval-data/fdb/manifest-12.jsonl --concurrency 16 --repeats 3 --out results/fdb \
        --judge qwen=http://127.0.0.1:30000/v1 --judge-revision qwen=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
        --judge gpt4o=openai

    python -m benchmark.eval report results/fdb --md fdb.md --chart fdb.png
    python -m benchmark.eval agreement results/fdb --md agreement.md --chart confusion.png

What is measured (each number is per category and for ``all``, mean +- std over repeats):

* **Observation window.** The model's output over ``[0, T]``, T the input length, as
  16 kHz mono with exactly the input's sample count. A frame-synchronous model emits
  one output frame per input frame, so this is the audio a client would hear with no
  transport delay; sglang-omni streams in real time and plays what arrives, so its
  window also contains the server's processing and transport delay.
  ``wall_ms_per_frame`` and ``on_time_rate`` show whether the server kept up live.
* **Stop and response latency.** The official ``get_timing.py`` on Silero VAD speech
  spans (``silero-vad`` package, user spans merged across 0.6 s gaps, model spans
  across 0.5 s). Stop intervals are overlaps of user and model speech; response
  intervals run from the end of a user span to the next later model start. These are
  whole-file intervals of the overlap session, pooled: the mean over every interval of
  every timed session, null when there is none. Lower stop latency is better on
  ``user_interruption``; on the other three categories the model should keep its turn,
  so the overlap mostly tracks how long the other speech lasts.
* **Behavior labels.** The four files of a pair are transcribed with word timestamps by
  the official ASR script on ``nvidia/parakeet-tdt-0.6b-v2`` (pinned revision and
  SHA-256; needs ``nemo_toolkit[asr]``), then each judge labels the pair with the exact
  official prompt: ``C_RESPOND`` (addressed the overlap), ``C_RESUME`` (ignored it and
  continued), ``C_UNCERTAIN_HANDLING`` (asked for a repeat) or ``C_UNKNOWN`` (off-target
  or silent). The share that should be high is ``C_RESPOND`` on ``user_interruption``
  and ``C_RESUME`` on the other three. Shares are over valid labels; ``judged_pairs``
  says how many there were.
* **Semantic quality** (non-official, one local judge, as in sglang-omni). Each pair is
  graded ``accept``, ``fail`` or ``unresolved`` on ``interaction_handling``,
  ``relevance`` and ``grounding`` with sglang-omni's frozen prompt and schema (fetched
  at a pinned commit); ``joint`` fails if any axis fails and accepts only if all three
  accept. A verdict without an exact quote of its source becomes ``unresolved``.
  ``quality = A/(A+F)`` and ``coverage = (A+F)/N``, N every selected pair: read them
  together. Before grading, six frozen control cases must reproduce all 18 expected
  statuses, or the run is not graded (a rubric check, not judge accuracy).

Failed sessions stay in every denominator and are never timed or judged.

Judges and agreement
~~~~~~~~~~~~~~~~~~~~

A judge is any OpenAI-compatible chat endpoint, named with ``--judge NAME=...``:

* **local**: ``NAME=http://host:port/v1`` (vLLM, SGLang, or M\* if it serves the model).
  The model sent is the endpoint's ``/v1/models`` entry (``URL#name`` picks one) and
  every reply must report it. Behavior labels decode as sglang-omni's custom judge:
  non-thinking, temperature 0, top_p 1, 512 tokens, retry seeds 1 to 3. The semantic
  judge runs thinking mode, temperature 0, seed 1, in batches of 8 pairs; serve with a
  reasoning parser (``--reasoning-parser qwen3``) so the JSON answer arrives in
  ``content``. ``Qwen/Qwen3.8-27B`` is sglang-omni's choice.
* **API**: ``NAME=openai`` is ``gpt-4o-2024-08-06``, the paper's judge, with the
  official decoding. It reads ``OPENAI_API_KEY`` (and ``OPENAI_BASE_URL``) from the
  environment and is skipped, with the reason in the results file and the report,
  when the key is not set. Keys are never written anywhere.

Labels from different judges are kept apart (``<category>.<judge>.<label>``); compare
runs only under the same judge. With two or more judges, every repeat also records
``agreement.<a>.<b>.agreement`` and ``.kappa`` (Cohen's) over the pairs both labeled
validly, and ``agreement`` pools them over repeats into a confusion matrix with
per-category agreement and each judge's label shares. sglang-omni measured Qwen
against gpt-4o on Nemotron VoiceChat (the model behind ``nemotron_duplex``): 84.9%
agreement, kappa 0.72, with Qwen giving far fewer ``C_UNKNOWN`` labels, so a local
judge's numbers are for comparing like with like, and the API judge's for comparing
with published results.

``rejudge`` adds a judge to finished results from the transcripts they hold, without a
server or GPU (for example once an API key is available), and recomputes the label
and agreement metrics::

    python -m benchmark.eval rejudge results/fdb --judge gpt4o=openai --out results/fdb-gpt4o

``--semantic-only --semantic-judge NAME`` regrades only the semantic A/F/U axes and keeps
every judge's labels; ``--retry-failed`` relabels only the pairs whose label from an
unchanged judge failed. Rate limits, server errors and dropped connections are retried
with the same request and seed, with exponential backoff (``Retry-After`` honored), so they
never move the official loop to its next seed; a pair whose retries run out is ``failed``.

Results and reports
-------------------

Each results file (``schema: "mstar-eval/1"``) has ``config`` (everything that decides
the numbers: data slice, load, repeats, request fields, judges), ``meta`` (git commit
and dirty flag, GPUs, driver, Python/torch/CUDA versions, ``MSTAR_*`` environment, the
server's ``/v1/models`` identity and launch settings, the CLI command and flag names),
``repeats`` (per-repeat metrics and per-request rows) and ``summary`` (mean, std, min,
max, n per metric).

Results files are meant to be shared, so they never carry addresses, secrets or home
paths: values under secret-looking keys are dropped, host-like keys keep only loopback
values, a ``--serve-cmd`` is recorded as a fingerprint and ``--serve-arg`` values as flag
names plus a fingerprint, and raw command lines are not recorded. A failed request keeps only an error category (``HTTP 503``, an
exception type); the server's message and the exception text go to
``<out>/errors-<suite>-<label>-<load>.log`` next to the server log.

``report`` turns any set of files into one markdown table per suite, headed by a
provenance line, plus optional PNG charts of metrics against concurrency or rate.
``--metrics`` picks the columns. For a PR, paste the table and link the chart.

``rescore`` recomputes WER/CER from the transcripts a ``tts`` or ``asr`` results file
already holds, with another normalizer, and records that it did; transcribing is the
expensive part, so runs scored differently can be compared without a GPU. It refuses a
file whose transcripts were redacted when written (they held an address or a home path),
since it would score the redaction::

    python -m benchmark.eval rescore results/tts --data eval-data/fleurs_en/manifest.jsonl \
        --normalizer seedtts --out results/tts_seedtts

Adding a suite
--------------

Subclass ``benchmark.eval.suites.base.Suite``: declare its CLI flags in ``add_args``,
send one request in ``request`` (return a row dict; keys starting with ``_`` stay in
memory, for example decoded audio for scoring), compute one number per metric in
``score``, list it in ``benchmark/eval/suites/__init__.py``, and add its default table
columns to ``benchmark/eval/report.py``. ``test/modular/test_eval_framework.py`` runs
every suite against a stdlib fake server on CPU; add a case there.
