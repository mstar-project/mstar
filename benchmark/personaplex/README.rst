PersonaPlex offline validation
=============================

This tooling validates prerecorded-input, batch-size-1 replay. Concurrency is
independent requests, not model batching. It does not implement live duplex or
streamed model audio. NVIDIA recordings come from independent reference-model
execution, not NVIDIA's web server.

Repository benchmark entry point
-------------------------------

Run both the server and benchmark client with ``MSTAR_SHM_ARENA=1``. The server
must already have cached weights and NATF2 configured. No asset downloads or
credentials are handled by this package. A local manifest contains::

  [{"id": "english", "path": "/private/path/question.wav", "sha256": "<file SHA256>"}]

Paths may be absolute or relative to the manifest. Optional ``persona`` defaults
to the pinned NVIDIA teacher prompt. The model supplies NATF2 and greedy mode;
the dataset never substitutes a transcription instruction.

For each clip, use a separate manifest and output directory::

  MSTAR_SHM_ARENA=1 python -m benchmark.runner \
    --url http://127.0.0.1:8000 --model personaplex \
    --inference-system ours --dataset local_audio --audio-manifest inputs.json \
    --request-type audio_to_speech --profiling-type closed_loop \
    --max-concurrency 1 --num-warmup 3 --num-requests 10 --verbose \
    --output-dir results/clip-c1

Repeat at concurrency2 and4 after lower-concurrency correctness checks pass.
The existing runner warms at the measurement concurrency. Headline results are
its ``results.json``: request latency, RTF and audio-seconds throughput. File
saving happens after timing. Buffered time-to-audio is not live speech latency;
an80ms model frame is not measured processing performance.

Numerical and lifecycle orchestration
------------------------------------

``python -m benchmark.personaplex.suite`` accepts ``preflight``, ``smoke``,
``full``, ``resume``, ``status`` and ``report``. Supply ``--output`` outside the
checkout and ``--settings`` for initial runs. Settings are a private JSON object
with these keys:

* ``expected_head``: exact committed model branch SHA; tracked state must be clean.
* ``source``: clean NVIDIA checkout pinned to3428dfd95309a7f3c84fd93259ded0f810d1ff91.
* ``assets``: local snapshot containing LM, Mimi and SentencePiece files.
* ``voices``: extracted voice directory containing NATF2.pt.
* ``cache`` and ``server_config``: existing serving cache and YAML configuration.
* ``extension``: private directory holding the tested Rust extension.
* ``existing_clip``: original222-frame mono24kHz PCM16 recording.
* ``english_clip``: optional user-supplied English WAV. Missing means BLOCKED.
* ``reference_site``: optional dependency-only NVIDIA environment site-packages;
  appended after importing production Torch, never substituted for that runtime.
* ``budget_file``: shared private reservation ledger across smoke/full allocations.
* ``port``: optional localhost port, default8000. Occupied ports are left untouched.

Run ``preflight`` before allocating a GPU. It verifies identity and input digests;
it does not establish GPU correctness. For GPU execution, the allocation wrapper
must set ``PERSONAPLEX_ALLOCATION_START`` to its start epoch,
``PERSONAPLEX_ALLOCATION_SECONDS`` and ``SLURM_JOB_ID``. Reserve at most1200seconds
for smoke and9600seconds for full, total10800seconds across the shared ledger.
The conservative ledger counts reservations, including an early-finished job;
it does not silently reclaim budget for additional allocations. Full runs stop
launching work15minutes before expiry; smoke reserves2minutes for cleanup inside
its20minute cap. Queue time does not count. Setup/loading does count.

The suite launches independent subprocesses in priority order: greedy parity;
Python/Rust lifecycle;20minute Rust soak; uninstrumented Rust warm benchmarks.
It preserves raw commands, timeouts, logs, inputs, hashes, code SHA, numerical
arrays/events, WAVs and incremental reports. Missing English cases do not get
replaced. A16-frame prefix of the existing clip is a labelled lifecycle control.
Numeric failures do not suppress independent serving checks. Failed Rust lifecycle
checks prevent soak/performance escalation. Both lifecycle paths use SHM arena.

Resume uses the same settings, source, input, environment and extension identity.
Changed identity requires a fresh evidence directory. Completed PASS/FAIL cases
are reused; ``--retry-failed`` reruns failed cases and preserves previous output
directories. Resume does not create an additional GPU allocation or extend its
budget. All performance cells still execute through ``benchmark.runner``.

Results and limits
------------------

``REPORT.rst``, ``results.json`` and embedded ``listen.html`` are regenerated
incrementally. PASS, FAIL, SKIP, BLOCKED and NOT RUN are distinct. Required
failures/incomplete cases give a nonzero exit. Exact-token divergence is FAIL
even when execution completes. Float-waveform MAE/MSE/RMSE/max error and first
decision margins are measurements, not an automatic numerical/quality approval.
No broad audio-quality score or supported-reference-runtime claim is made.

The historical76frame count was a reporting error: original event indices and
finalized arrays both identify73frames,216agent-code differences, firstframe39.
``compare`` recomputes counts directly and requires them to agree with events::

  python -m benchmark.personaplex.compare /path/to/parity-output

It exits1 for token divergence. The regression suite explicitly checks that a
deliberately failed comparison produces FAIL and a nonzero process exit.

The soak observes decoder tensor ownership before hard cleanup, request entries
after teardown, periodic total GPU usage, and drained allocated/reserved memory.
Any final idle allocation growth over the first five-minute checkpoint fails for
investigation; reservation growth alone is reported separately. This is bounded
coverage, not proof against every possible leak. Listening remains a human task.

Warm compilation detection uses server log markers around the runner's warmup
boundary, not an exhaustive kernel tracer. A detected build during measurement
invalidates the cell; one complete rewarm/retry is allowed within the deadline.
Small-sample percentiles are descriptive. No optimization is performed here.

Deferred: second voice, injected worker error, supported NVIDIA GPU runtime,
accepted near-limit request, larger corpus/longer soak, optimization, batching,
streamed output and live input. Keep shared ownership and model commits separate
for review. Private launch scripts, reports, recordings and worklogs stay local.
