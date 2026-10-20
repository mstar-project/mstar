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
``e2e``
  The request kinds ``compare`` needs beyond speech (``--kind``): greedy chat with text, image, audio or video in
  and optionally speech out, seeded image and video generation, voice-clone uploads, VLA actions and streamed
  world-model frames. Latency percentiles, time to first output and requests/s; the kept repeat's outputs are
  saved for ``compare`` to diff. It builds its own prompts, so ``--data`` is optional (``gen_clone`` takes a
  voice-clone manifest) and ``--media`` points at ``data e2e``'s ``media/``.

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

Downloads go to the Hugging Face cache, so set ``HF_HOME`` to a large disk on GPU
hosts and reuse it across runs.

Quality judges
--------------

Model-based scores depend on the judge, so every results file names it, and tables
should only compare runs scored by the same judge and normalizer.

* WER normalizers (``--normalizer``): ``whisper`` (default; Whisper's convention, per
  item: its English normalizer for English items and items with no language, its basic
  normalizer for every other language), ``whisper_en`` (English only; a run whose items
  are in another language is refused before it starts), ``seedtts`` (the Seed-TTS-eval
  recipe: punctuation removed except the apostrophe), ``whisper_basic``, ``basic``. Both
  the corpus rate (``wer``) and the per-utterance mean (``wer_utt_mean``, what
  Seed-TTS-eval publishes) are kept.
* Speaker similarity uses ``microsoft/wavlm-base-plus-sv`` x-vectors; Seed-TTS-eval's
  published SIM uses a WavLM-large ECAPA checkpoint, so absolute values differ.
* UTMOS is UTMOS22 strong from ``tarepan/SpeechMOS`` on torch.hub.

Judges run between repeats, after the load has finished, so they never compete with
measured requests; ``--judge-device`` moves them to another GPU or the CPU.

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

Base vs head: ``compare``
-------------------------

``compare`` serves two trees (usually ``main`` and a PR) on the same GPUs and checks, per model and case, that
the PR keeps its outputs and does not get slower::

    python -m benchmark.eval compare --base main --head 548                    # every model, local GPUs
    python -m benchmark.eval compare --base main --head 548 --models tag:tts,qwen3_omni --rp
    python -m benchmark.eval compare --base main --head 548 --hosts bench@gpu1,bench@gpu2:1
    python -m benchmark.eval compare --list                                    # models, tags, cases

``--head`` and ``--base`` take a branch (fetched fresh from ``mstar-project/mstar``, then ``origin``), a PR
number, a commit, or another local ref such as ``HEAD`` or a tag (this checkout's commit). Both trees are exported from the checkout
running ``compare`` (each arm imports its own tree and builds its own ``rust/`` extension, in one Python
environment); the client, the checks and the report always come from that checkout, so ``main`` can be measured
before it has this framework. Each model runs on one GPU: base and head back to back, arm order alternating from
model to model, every case at each concurrency in its ``conc`` with ``repeats`` passes after warmup waves.
``--sessions 2`` runs two server sessions per arm (ABBA). Each arm's first server start gets twice the model's
``startup_timeout_s``, since it may compile from cold caches whichever arm goes first; later starts get it once.
The ``rust/`` extension is built with the environment's ``pip``, or ``uv pip`` when it has none (a uv venv); if a
tree both arms share cannot be built, both run the environment's ``mstar_rust`` and the report says so.
Exported trees, ``rust/`` builds, local jobs' outputs and the inputs live in ``--work-dir`` (default
``$XDG_CACHE_HOME/mstar-eval``, else ``~/.cache/mstar-eval``; ``--data-dir`` moves the inputs alone).
With ``--hosts``, idle hosts take whatever models are
still queued; a host that stops answering is dropped and its model requeued. Hosts need key-based ``ssh``,
``rsync`` at both ends, an M\* environment (``--host-python``) and the Hugging Face Hub (with a login for gated repos such
as Orpheus) or a warm cache; their work directory is ``--host-dir``. Machines that must
be started first are a launcher's job: ``--host-provider module:factory`` supplies the hosts and is told when the
run ends (``benchmark/eval/e2e/hosts.py``).

The models and cases are ``benchmark/eval/e2e/cases.yaml``: TTS, voice clone, ASR, duplex, text, image, audio
and video in, speech out, image and video generation, streamed frames, VLA actions, and with ``--rp`` request-path
cases and served invariants. Speech cases use the ``tts``, ``asr`` and ``duplex`` suites; the rest use the ``e2e``
suite (``run e2e --kind chat_image|image|video|pi05|frames|...``). Inputs are built once into ``--data-dir``
(``python -m benchmark.eval data e2e --out DIR``: Seed-TTS and LibriSpeech sets plus generated media).

The run directory holds every results file and ``report/report.md`` (a table for a PR body, details in toggles),
``report/chart.png`` and ``report/summary.json``. ``compare`` exits with 1 when any row is flagged, else with 2
when any row has no verdict (its job failed or was interrupted, the model was skipped or never got a GPU, base
is broken, or base answered no request at some server session and concurrency, whatever head did), else 0; on Ctrl-C it stops every job, prints one line and exits with 130. A row is **flagged** when:

* **correctness changed**: greedy text is neither identical (share at least ``min_match``) nor close (median
  similarity 0.6 or more, length ratio 0.8 to 1.25), or any head reply is base's cut below 80% of its length
  (empty replies included); speech WER (Whisper
  large-v3 round trip for TTS) rose more than ``max_increase`` points; a seeded image, video or frame stream fell
  below ``min_db`` PSNR against base; VLA actions moved more than ``tol``; duplex take-turn or yield rates dropped
  more than ``tol``; a request-path status code changed; head had more request errors than base; or head answered
  no request at a server session and concurrency where base answered some (request errors per arm are a column of
  the main table);
* **a regression**: c=1 latency or c=max throughput is worse by more than 5%, by more than two combined standard
  deviations of the passes, and by more than 1.5 times the largest drift that row showed between two server
  processes of identical code in its last 8 A/A runs (its A/A noise floor; a lone drift over 10% and over three
  times the median of the others is left out until a second run agrees). When base and head return different output lengths for the
  same requests, latency is compared over the requests whose lengths matched, else per token or audio second;
* **head does not build** (its ``rust/`` extension or Rust frontend) while base does, or **head does not start**
  while base does.

A perf row flagged with one session per arm is rerun with two (ABBA) before it is reported, since a server process
can settle at one of two speeds; ``--no-rerun`` turns that off.

**Calibrating.** Noise floors and correctness swings come from A/A runs (base and head the same commit), per GPU
type, in ``benchmark/eval/e2e/calibration.yaml``. After a large change on ``main``, a new model, or on a new GPU
type, run ``compare --base main --head main`` for the models concerned, then
``python -m benchmark.eval calibrate <run dir>``; rows without a floor on the run's GPU are named in the report
and judged on the passes' spread alone. A calibrated swing only widens a ``cases.yaml`` limit, never tightens it.
Rows of the A/A run that failed or had request errors stay out of the floors; ``calibrate`` prints how many per model.

Adding a suite
--------------

Subclass ``benchmark.eval.suites.base.Suite``: declare its CLI flags in ``add_args``,
send one request in ``request`` (return a row dict; keys starting with ``_`` stay in
memory, for example decoded audio for scoring), compute one number per metric in
``score``, list it in ``benchmark/eval/suites/__init__.py``, and add its default table
columns to ``benchmark/eval/report.py``. ``test/modular/test_eval_framework.py`` runs
every suite against a stdlib fake server on CPU; add a case there.
