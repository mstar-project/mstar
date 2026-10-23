Environment variables
=====================

Runtime knobs M* reads from the environment. New variables should be
documented here as they are introduced.

Communication
-------------

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Meaning
   * - ``MSTAR_RUST_ZMQ``
     - ``AUTO``
     - Transport selection for the ZeroMQ control mesh (see
       :func:`mstar.communication.communicator.make_communicator`).
       ``AUTO``: the Rust-backed ``RustZMQCommunicator`` when the vendored
       ``rust/`` extension imports successfully, pyzmq otherwise.
       ``1``: the Rust communicator, raising if the extension is missing.
       ``0``: always pyzmq. The two transports are wire-compatible, so
       this can be set per-process while the rest of the mesh stays on
       pyzmq. Under ``MSTAR_RUST_GRAPH=AUTO``, ``0`` also keeps that
       worker's graph runtime on Python -- the Rust one cannot share a
       pyzmq transport.
   * - ``MSTAR_RUST_GRAPH``
     - ``AUTO``
     - Which graph runtime a worker builds (see
       :func:`mstar.graph.runtime.utils.resolve_graph_runtime_type`).
       ``AUTO``: Rust where the vendored ``rust/`` extension imports *and*
       the Rust transport is in use, Python otherwise. ``1``: the Rust
       runtime, failing at startup if the transport or the tensor bookkeeper
       is a Python one -- it holds a share of both, so there is no degraded
       mode to fall back to. ``0``: always the Python runtime. Resolved once
       per process, because it also selects the tensor bookkeeper the
       runtime and the tensor store must share.
   * - ``MSTAR_WIRE_CODEC``
     - ``msgpack``
     - How messages are encoded on every communicator edge (see
       :func:`mstar.communication.codec.default_codec`). ``msgpack``: the
       typed encoding in :mod:`mstar.communication.wire`, language-neutral
       so an edge can terminate in a Rust process. ``pickle`` is no longer
       supported and fails at startup. Loosely typed metadata uses data-only
       extensions instead of pickle. Upgrade all processes in a deployment
       together: older processes cannot decode the new extensions, and
       updated processes reject legacy pickle messages and metadata.
   * - ``MSTAR_ZMQ_TRANSPORT``
     - constructor's protocol
     - Overrides the communicator protocol (``IPC`` or ``TCP``) for a
       process, e.g. to run entities on separate hosts.
   * - ``MSTAR_ZMQ_TCP_HOST``
     - ``127.0.0.1``
     - Host used to build peer endpoints when the protocol is ``TCP``.
   * - ``MSTAR_ZMQ_TCP_BASE_PORT``
     - ``19000``
     - Base of the deterministic entity-id → TCP port map (``api_server``
       = base, ``conductor`` = base+1, ``worker_<rank>`` = base+100+rank).
   * - ``MSTAR_REQUIRE_ACCELERATOR_GRAPHS``
     - ``0``
     - ``1`` makes a worker fail at startup when any accelerator graph bucket
       could not be captured, instead of serving that bucket eagerly at
       10-20x the latency. Off by default: a failed capture is logged at
       ERROR with a per-runner summary and the rest keeps running.
   * - ``MSTAR_SHM_ARENA``
     - ``0``
     - SHM tensor-transport implementation. ``0``: per-uuid files.
       ``1``: the Rust shared-memory arena (requires the ``rust/``
       extension; raises if missing). ``AUTO``: the arena when the
       extension imports, files otherwise. Must match across the
       deployment — arena locations ride in the tensor descriptors.
   * - ``MSTAR_KV_SHM_DIR``
     - ``/dev/shm`` (or ``/tmp`` if unavailable)
     - Base directory for host-staged KV snapshots when remote KV transfer
       uses SHM. Each deployment creates a private child directory here;
       request cleanup removes snapshots, and startup removes directories
       left by dead deployments. All workers sharing KV snapshots must see
       the same filesystem.
   * - ``MSTAR_SHM_ARENA_SEGMENT_MB``
     - ``256``
     - Size of each arena segment. The arena grows segment by segment;
       existing segments never move (registrations stay valid). A
       segment's /dev/shm pages are committed when it is created (pinned
       segments always were, by ``cudaHostRegister``), so the whole
       segment counts against /dev/shm, as do spill files, which land in
       /dev/shm by default. A segment /dev/shm cannot hold counts as arena
       full (sends spill, see ``MSTAR_SHM_ARENA_SPILL``).
   * - ``MSTAR_SHM_ARENA_MAX_SEGMENTS``
     - ``32``
     - Growth cap PER ENTITY. Every entity (workers + the api-server data
       worker) creates its own arena, so node-wide /dev/shm demand can
       reach ``MAX_SEGMENTS x SEGMENT_MB x num_entities`` — size against
       ``df -h /dev/shm`` (tmpfs defaults to ~50% of RAM). Construction
       fails fast if one entity's ceiling exceeds /dev/shm. At the cap,
       sends spill (see ``MSTAR_SHM_ARENA_SPILL``).
   * - ``MSTAR_SHM_ARENA_FULL_TIMEOUT_S``
     - ``30``
     - Strict mode only (``MSTAR_SHM_ARENA_SPILL=0``): how long a send
       backpressures on a full arena before failing.
   * - ``MSTAR_SHM_ARENA_SPILL``
     - ``1``
     - Degrade gracefully at the segment cap: stage the tensor through the
       per-uuid file protocol instead — slower, never fails, matching the
       file transport's saturation behavior. ``0`` restores strict
       backpressure + timeout — only meaningful where ANOTHER thread
       drains consumer ACKs (the threaded api-server); on a worker the
       ACKs arrive on the very thread that would be waiting.
   * - ``MSTAR_SHM_ARENA_SPILL_AFTER_S``
     - ``0``
     - Optional grace before spilling, for deployments where another
       thread frees slots concurrently. Default 0: spill immediately
       (a worker cannot receive ACKs while it waits).
   * - ``MSTAR_SHM_ARENA_PIN``
     - ``1``
     - ``cudaHostRegister`` each mapped segment (both sides) so D2H/H2D
       copies through the side streams run at page-locked bandwidth and
       stay asynchronous. ``0`` disables (pageable copies).
   * - ``MSTAR_SHM_ARENA_PIN_MAX_MB``
     - ``4096``
     - Budget for TOTAL pinned host memory PER PROCESS, distinct from the
       segment cap (pinned pages come out of the OS's pageable pool
       system-wide). Node-wide pinned demand is approx
       ``PIN_MAX_MB x num_entities`` — a consumer pins peer segments too,
       so one process can pin more than its own arena holds. Segments
       past the budget stay unpinned: copies work, without async overlap.
   * - ``MSTAR_SHM_ARENA_SLOT_TTL_S``
     - ``0``
     - TTL backstop for abort-orphaned slots (a request aborted after
       staging but before all consumer ACKs defers reclaim forever).
       A slot older than the request timeout cannot have a legitimate
       reader, so a bound safely above it (recommend >= 2x the request
       timeout) cannot race a real consumer. ``0`` disables (default,
       pending review discussion); reclaims run under capacity pressure
       and with the periodic stats sweep, logging loudly.
   * - ``MSTAR_SHM_ARENA_STATS_INTERVAL_S``
     - ``60``
     - Under ``--log-stats``: how often the arena logs its occupancy /
       fragmentation snapshot (segments, free bytes, largest contiguous
       free block, pinned bytes).
   * - ``NCCL_SOCKET_IFNAME``
     - ``lo``
     - NCCL's own variable. Every worker runs on the conductor's host, so
       the conductor sets ``lo`` for its workers and NCCL's bootstrap
       sockets listen on loopback only. A value you set, or a set
       ``NCCL_SOCKET_FAMILY``, is kept as is.
   * - ``GLOO_SOCKET_IFNAME``
     - ``lo``
     - PyTorch's variable for the Gloo groups next to each TP/SP group.
       Same rule: the conductor sets ``lo`` for its workers unless you set
       it, so Gloo does not listen on the address the hostname resolves to.

Models
------

Per-model knobs, read when the submodule is built. They are scoped to one
model, so they are named for it.

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Meaning
   * - ``MSTAR_VIT_BATCHING``
     - ``0``
     - BAGEL: batch several requests through the ViT encoder in one forward.
       Off by default because flash-attn's varlen reductions across packed
       images produce small bf16 drift, which at greedy ``temperature=0`` can
       flip a downstream LLM argmax. When off, ``prefill_vit`` runs one
       request at a time.
   * - ``MSTAR_VIT_ACCELERATOR_GRAPH``
     - ``0``
     - BAGEL: accelerator graph capture of the ViT block loop, over the node's
       ragged attention resource (see :doc:`adding_models`). Requires the
       deployment option ``model_kwargs.accelerator_graph`` to be true;
       BAGEL defaults this option to true (see :doc:`models`).
       Off by default: capture costs one graph per (batch size, token bucket)
       and the eager flash-attn path is already fast. The win is removing
       per-layer launch overhead on small images.
   * - ``MSTAR_VIT_ACCELERATOR_GRAPH_TOKEN_BUCKETS``
     - ``512,1024,2048,4096,4900``
     - Comma-separated token-count buckets to capture, when
       ``MSTAR_VIT_ACCELERATOR_GRAPH=1``. A batch longer than the largest bucket runs
       eagerly. ``4900`` is ``70*70``, the exact length the ``vllm``
       preprocess option emits.
   * - ``MSTAR_VIT_ACCELERATOR_GRAPH_BATCH_SIZES``
     - ``1,2,4``
     - Comma-separated batch sizes to capture. Read only when
       ``MSTAR_VIT_BATCHING=1``; otherwise only batch size 1 is captured.
   * - ``COSMOS3_DISABLE_ACCELERATOR_GRAPH``
     - unset
     - Cosmos3: any nonempty value disables denoise, prefill and reasoner
       decode capture. The deployment option
       ``model_kwargs.accelerator_graph: false`` also disables all these
       captures and cannot be overridden by this variable.
   * - ``COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH``
     - unset
     - Cosmos3: any nonempty value disables only the understanding-tower
       prefill capture. Denoise and reasoner decode captures remain eligible
       when ``model_kwargs.accelerator_graph`` is true.
   * - ``MSTAR_OMNIVOICE_MODEL_PATH``
     - unset
     - OmniVoice: load weights from this local directory instead of the
       checkpoint the registry names. For serving a fine-tune of the same
       architecture off a mounted volume, where the registry entry would
       otherwise pull the public weights from the Hub.

Serving (Python frontend)
-------------------------

Read by ``mstar-serve`` / ``mstar.api_server.entrypoint``.

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Meaning
   * - ``MSTAR_MAX_BODY_MB``
     - ``128``
     - Request body limit on every route, as in the Rust frontend: a larger
       body gets a 413, refused on its ``Content-Length`` or, for a chunked
       upload, while it is read. ``/v1/audio/transcriptions`` keeps its own
       512 MiB file cap when this is lower. A body refused on its
       ``Content-Length`` is answered before it is read, so a client that
       writes the whole body before reading the reply can see a connection
       reset instead of the 413; one that sends ``Expect: 100-continue``
       gets the 413 before it sends the body.
   * - ``MSTAR_ROOT_PATH``
     - ``""``
     - Mount the app under a URL sub-path. Set it when an ingress routes the
       deployment at a prefix and does not strip that prefix before the
       request reaches the pod (Run:AI serves a workload at
       ``/<project>/<job-name>/``, for instance); without it FastAPI 404s
       every route. Empty by default, so a direct deployment is unaffected.
   * - ``MSTAR_MAX_IMAGES_PER_REQUEST``
     - ``10``
     - Largest ``n`` one ``/v1/images/generations`` request may ask for (see
       :func:`mstar.api_server.openai.serving_images.max_images_per_request`);
       ``n`` outside ``1`` to this value is a 400 before anything is
       submitted. The Rust frontend and the Dynamo bridge apply the same
       variable and default.
   * - ``MSTAR_API_KEY``
     - unset
     - API key (read by :func:`mstar.api_server.auth.resolve_api_key`;
       ``--api-key`` overrides it). When set, every route except
       ``/health`` and ``/metrics`` needs ``Authorization: Bearer <key>``, WebSocket
       handshakes included; unset or empty turns the check off. The
       launcher passes it on to the Rust frontend only (the conductor and
       workers do not inherit it), and
       :class:`mstar.client.MStarClient` sends it when not given
       ``api_key``. See :ref:`api-key`.
   * - ``MSTAR_SILERO_VAD_PATH``
     - unset
     - Silero VAD ONNX file (``silero_vad.onnx``) for ``/v1/realtime``
       ``server_vad`` turn detection, read by
       ``realtime_vad.silero_weights``. Unset, the copy bundled with the
       ``silero-vad`` package (``mstar[vad]``) is used; with neither, or
       with a file whose inputs or outputs differ (the package's other
       exports), ``server_vad`` sessions are refused. Never downloaded.

Serving (Rust frontend)
-----------------------

Read by the ``mstar-server`` binary and its bridge
(``mstar-serve --rust-frontend``; see :doc:`installation`).

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Meaning
   * - ``MSTAR_SERVER_BIN``
     - unset
     - Path to the ``mstar-server`` binary. Fallback order:
       ``--rust-frontend-bin``, this variable, ``$PATH``, then the in-repo
       ``rust/server/target/release`` build.
   * - ``MSTAR_API_KEY``
     - unset
     - Same rule as the Python server (``rust/server/src/auth.rs``, read
       once at startup): every route except ``/health`` and ``/metrics`` needs
       ``Authorization: Bearer <key>``. ``mstar-serve --rust-frontend``
       sets it from ``--api-key`` / its own environment.
   * - ``MSTAR_REQUEST_TIMEOUT_S``
     - ``600``
     - Per-request budget in the Rust frontend; on expiry the client gets
       an error and the request is aborted in the backend.
   * - ``MSTAR_SAMPLE_RATE``
     - ``24000``
     - Sample rate stamped on ``/v1/audio/speech`` WAV output.
   * - ``MSTAR_ALLOW_REMOTE``
     - ``0``
     - Allow ``http(s)`` media URLs in requests (fetched server-side,
       30 s timeout). Off by default.
   * - ``MSTAR_MAX_CONCURRENT_REQUESTS``
     - ``256``
     - Admission cap on in-flight generation requests; beyond it clients
       get an immediate 503 instead of queueing into the request timeout.
       ``/health``, ``/metrics`` and ``/v1/models`` bypass the cap.
   * - ``MSTAR_MAX_BODY_MB``
     - ``128``
     - Request body limit (multipart uploads included).
   * - ``MSTAR_TOKENIZE_PARALLEL``
     - ``1``
     - When a long prompt is encoded as parallel chunks, set
       ``TOKENIZERS_PARALLELISM=true`` for the process first: the Rust
       tokenizer only uses its thread pool while that variable reads true,
       and serving environments often export it false. ``0`` leaves the
       variable as it is (the chunked encode then runs serially).
   * - ``MSTAR_OPENAI_ADAPTER``
     - set by the launcher
     - The class name of the model's adapter in ``ADAPTER_REGISTRY``
       (``openai_adapter_class`` in ``mstar/api_server/rust_frontend.py``), which
       picks the OpenAI translation ``mstar-server`` uses
       (``Adapter::from_python``). ``mstar-serve --rust-frontend`` always sets it;
       a ``mstar-server`` started by hand without it answers 404 on ``/v1/*``.
   * - ``MSTAR_TOKENIZER``
     - unset
     - Path to a HuggingFace ``tokenizer.json`` enabling frontend
       tokenization. Leave unset with the Python backend — its preprocess
       worker owns tokenization, and the bridge rejects pre-tokenized
       ingest.

Rust frontend limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Rust frontend carries no audio encoder or media muxer (the Python frontend
uses ``soundfile`` / ``ffmpeg``), so two surfaces degrade:

* ``/v1/audio/speech`` produces only ``wav`` and ``pcm``. Any other
  ``response_format`` (``mp3``, ``opus``, ``flac``, …) is rejected with a 400
  rather than silently returned as WAV. Run the Python frontend for compressed
  containers.
* ``/v1/videos/generations`` (Cosmos3) returns the **video-only** mp4. When a
  request sets ``generate_sound``, Cosmos3 generates a separate audio track that
  mstar muxes into the mp4 as an AAC track; the Rust frontend cannot mux it, so
  the audio is dropped and a warning is logged. Do not set ``generate_sound`` on
  the Rust frontend (it spends compute on a track the client won't receive), or
  run the Python frontend for sound video.

Request media
-------------

Read by the model's media loaders in the data worker's preprocess thread,
behind either frontend. All four are also checked once when the data worker
starts (:func:`mstar.model.media_limits.check_media_limits`), so an invalid
value stops the server instead of failing every media request. The pixel and
audio caps hold for a request's files together (:class:`mstar.model.media_limits.RequestMediaBudget`),
so attaching more files does not multiply them.

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Meaning
   * - ``MSTAR_MAX_IMAGE_PIXELS``
     - ``89478485``
     - Pixels a request's images may decode to, summed over them (see
       :func:`mstar.model.media_limits.max_image_pixels`), checked from each
       header before decoding; over it, the request gets a 400. PNG, JPEG,
       WebP and GIF are read (FLUX.2 klein, which decodes with Pillow, also
       reads BMP and TIFF); other formats get a 400. A GIF or an animated
       WebP is read as its first frame.
   * - ``MSTAR_MAX_VIDEO_PIXELS``
     - ``1073741824``
     - Pixels a request's videos may decode to (see
       :func:`mstar.model.media_limits.max_video_pixels`), summed over the
       frames the model's loader decodes, each frame counting as at least
       64x64; over it, the request gets a 400. At 1080p30 the default refuses
       clips over about 17 s on Cosmos3, which decodes every frame, and over
       about 259 s on Qwen3-Omni and Qwen3.5, which decode 2 frames a second
       (Qwen3-Omni decodes every frame, as Cosmos3, when the clip's header
       gives no frame count).
   * - ``MSTAR_MAX_MEDIA_FILES``
     - ``16``
     - Images, audio clips and videos one request may attach, together (see
       :func:`mstar.model.media_limits.max_media_files`); more get a 400
       before any is decoded. A model can take fewer of one kind
       (:meth:`mstar.model.base.Model.max_input_files`; Qwen3-TTS,
       Chatterbox, Whisper and Higgs-Audio take one audio clip).
   * - ``MSTAR_MAX_AUDIO_SECONDS``
     - ``3600``
     - Seconds a request's audio clips may decode to, summed over them (see
       :func:`mstar.model.media_limits.max_audio_seconds`), checked from each
       header before the model decodes it and again from the decoded clip (a
       file whose header gives no length, such as a FLAC with no sample count,
       is counted as it decodes, or from its size for a WAV with no data
       length); over it, the request gets a 400.

Worker scheduling
-----------------

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Meaning
   * - ``MSTAR_TP_ASYNC_SCHED``
     - config's ``tp_async_sched``, else ``0``
     - Async scheduling for lockstep-parallel (TP / SP) nodes; overrides the
       deployment config's ``tp_async_sched``. ``1`` (or ``true``): the
       instance leader speculates step N+1 of the parallel node during
       forward N (the existing single-worker speculation machinery, gate
       opened) and broadcasts it at once as a speculative
       ``ScheduleTPNode``; followers rebuild the identical batch during
       their own forward N and every rank submits N+1 the moment N
       completes. Removes the per-step serial build/plan from the group's
       critical path. Voids (allocation failure, per-rid failure, a
       continuing request without loop-back output) are derived by each
       rank from replicated state, never signalled. A comma-separated
       list of node names (``thinker,talker``) enables it for those
       parallel nodes only. ``0`` (or ``false``): the serial path — leader schedules
       after N, followers rebuild after the broadcast. Set it identically
       on every rank of an instance: the workers compare it at startup and
       refuse to start on a mismatch. Leave ``MSTAR_ENGINE_STEP_SYNC`` at
       ``0`` with it: that throttle holds the GPU thread until step N drains,
       which serialises the very overlap this flag buys.
   * - ``MSTAR_PRE_PLAN_SPEC``
     - ``1``
     - Pre-plan the speculative batch's attention on a dedicated thread
       while the previous replay runs. ``0`` plans inline on the GPU
       thread. With ``MSTAR_TP_ASYNC_SCHED`` on, set it identically on every
       rank of a lockstep instance: a follower builds a voided speculative step
       as rank 0 pre-planned it, so the workers compare it at startup and
       refuse to start on a mismatch.
   * - ``MSTAR_DECODE_BUCKETS``
     - ``1,2,4,8,16,32,48,64,96,128``
     - The decode batch sizes a model captures as CUDA graphs (Qwen3.5 reads
       it; a model with its own ladder reads it the same way). A decode step
       runs at the smallest bucket that holds its rows, so the ladder bounds
       both the rows per step and the graph memory; narrow it (for example
       ``1,2,4,8,16,32``) on a memory-tight GPU or for an A/B against the
       former ladder. ``MSTAR_QWEN35_DECODE_BUCKETS`` overrides it for
       Qwen3.5 alone.
   * - ``MSTAR_LAZY_PUBLISH``
     - ``1``
     - Publish resource state (KV pages, positions) only for the requests
       whose frame or completion carries it: ``finalize_batch`` keeps a
       cheap per-request snapshot and the worker finishes it on demand.
       ``0`` builds the full publication for every request on every step,
       the old behaviour; it was a quarter of the GPU thread's host time at
       batch 32 in a decode loop that never reads it.
   * - ``MSTAR_MAX_CONSECUTIVE_SPEC_STEPS``
     - ``1024``
     - Cap on back-to-back speculative steps before the leader yields to
       other ready work. ``0`` turns speculation off. A value that is not an
       integer is ignored with a warning when the worker starts.
   * - ``MSTAR_LAUNCH_WAIT_MS``
     - ``5``
     - Milliseconds the speculative submitter holds off the GIL waiting for
       the GPU thread to reach the launch before it carries on. A value that
       is not a number, is ``nan`` or ``inf``, or is past
       ``threading.TIMEOUT_MAX`` seconds is ignored with a warning when the
       worker starts and ``5`` is used; a negative value is raised to ``0``
       with a warning.
   * - ``MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS``
     - ``32,64,128,256,512,1024,2048,4096,8192``
     - Total tokens of a packed Qwen3.5 text-prefill step that get a captured
       graph (one per bucket and batch size). A prompt past the top bucket
       prefills eagerly, which is launch-bound: an eager 8k prefill on 0.8B
       spends 78 ms of wall time on 36 ms of kernels. The default now reaches
       8192; narrow it to save graph memory and capture time, or extend it.
   * - ``MSTAR_KV_CHAIN_LAZY_STEPS``
     - ``1``
     - How many decode steps of sampled ids the KV manager holds back before
       it extends the requests' prefix chains (the per-page keys the prefix
       cache is indexed by), one extend per request per batch of steps instead
       of one per step. The per-request Python of that extension was 0.3 to
       0.4 ms a step at 128 rows. A page key then appears up to that many
       steps late and a later commit indexes it, as the single-request path
       already allows; a request that finishes in between may leave its last
       page unindexed. The default ``1`` extends on every step: ``16``
       measured as noise at c64/c128 on 0.8B (the extension is a small part
       of the stop check's tail).
   * - ``MSTAR_KV_PINNED_INDPTRS``
     - ``0``
     - The KV plan writes FlashInfer's index arrays (qo_indptr, kv_indptr,
       kv_indices, last_page_len) into a ring of page-locked host buffers and
       the decode wrapper copies the indices straight into its captured
       device buffer from there, so the copies FlashInfer's ``plan`` issues
       are really asynchronous. From pageable memory the driver stages each
       one and FlashInfer makes the indices copy blocking. No measurable gain
       at c32-c128 on 0.8B, so it is off by default. The cost is one CUDA
       event per plan, which a buffer set waits on before it is rewritten.
   * - ``MSTAR_KV_PLAN_CACHE``
     - ``0``
     - The KV manager plans a captured decode step off the previous step's
       plan when the batch is the same rows in the same order and nothing
       but that step's commit touched the streams: every length is one
       more, pages change only where a row crossed a page boundary, and the
       five FlashInfer index arrays come from numpy instead of a per-row
       Python pass. The first version measured a loss at c64/c128 on 0.8B
       (batches change every step or two there, and a hit still rebuilt the
       views); the reworked path keeps the views lazy and is off by default
       until measured. ``0`` plans every step from the streams. ``2`` runs
       both and logs a difference, for checking a run.
   * - ``MSTAR_DEVICE_LOOPBACK``
     - ``1``
     - Whether a decode node may keep its loop-back token on the device:
       the sampler writes each request's last token to a slot-addressed
       master and the next step reads its input ids from it, so the worker
       routes the signal with no tensor (no per-request store, hold or
       cleanup per step). Only nodes that declare the read
       (``reads_device_loopback``: Qwen3.5 decode) and only when the sampler
       has graph buffers; the sampler keeps the masters for those nodes
       alone, so a node that takes its tokens on the host (Whisper's
       decoder) no longer pays the eager-step master write, a small
       synchronous host-to-device copy per step. ``0`` keeps every node on
       the per-request tensor path.
   * - ``MSTAR_INGRAPH_DECODE_ROPE``
     - ``0``
     - A captured decode step builds its rotary tables inside the graph from
       the position resource's planned positions instead of staging them.
       Experiment knob, measured and gated on its own.
   * - ``MSTAR_SAMPLER_INGRAPH_SCATTER``
     - ``0``
     - The captured sampler gathers its RNG offsets from the slot masters
       and scatters the advanced offsets and the sampled tokens back inside
       the graph, so a replayed decode step issues none of those launches on
       the gpu thread afterwards (each one is a GIL release the main thread's
       postprocess wins). Padding rows then address a trash master row and
       the masters are allocated once for ``MSTAR_SAMPLER_SLOTS`` concurrent
       requests (default 4096, a hard cap with this on). Only with the device
       loop-back on. An experiment knob; measure before turning it on.
   * - ``MSTAR_RUST_SEND_TIMING``
     - unset
     - ``1`` makes the Rust runtime print, every 500 calls, how long the
       worker's per-step send spends in its own work against the whole call
       (which includes taking the GIL back when the send had to release it).
       A diagnostic for the host-overhead work; off by default.
   * - ``MSTAR_SPEC_PEEK_FOR_FAIRNESS``
     - ``1``
     - Yield the speculation chain only when another (node, walk) is
       actually ready right now; ``0`` uses the consecutive-step cap alone.
   * - ``MSTAR_PHASE_TIMING``
     - ``0``
     - ``N > 0``: every N iterations log per-phase p50/p95/mean of the
       worker main loop (speculate, await_gpu, submit_spec, ...).
   * - ``MSTAR_KV_DEBUG_ASSERTS``
     - ``0``
     - ``1``: after every ``admit``, ``commit``, ``reset_request``, ``remove_request``
       and change to an admitted request's pages, check the KV page bookkeeping against
       the streams holding the pages, the reservations against the free and evictable
       pages, and each commit against its request's layout. Logs a label left under its
       footprint at removal. Tests and debugging only.

Compilation
-----------

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Meaning
   * - ``MSTAR_RECOMPILE_LIMIT``
     - ``84``
     - Dynamo's recompile limit, clamped to [8, 256]. Raise it if the logs
       show ``hit config.recompile_limit``.
