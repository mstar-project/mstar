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
       pyzmq.
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
   * - ``MSTAR_SHM_ARENA``
     - ``0``
     - SHM tensor-transport implementation. ``0``: per-uuid files.
       ``1``: the Rust shared-memory arena (requires the ``rust/``
       extension; raises if missing). ``AUTO``: the arena when the
       extension imports, files otherwise. Must match across the
       deployment — arena locations ride in the tensor descriptors.
   * - ``MSTAR_SHM_ARENA_SEGMENT_MB``
     - ``256``
     - Size of each arena segment. The arena grows segment by segment;
       existing segments never move (registrations stay valid).
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

Graph / scheduler core
----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Meaning
   * - ``MSTAR_RUST_WALK``
     - ``0``
     - ``shadow``: run the Rust walk core in lockstep with every
       per-request ``WorkerGraphIO`` on real traffic — Python stays
       authoritative; ready-set / doneness / loop-counter divergence is
       logged as an error (events the core does not model yet suspend
       comparison for that request with a logged reason). ``0``: off.
   * - ``MSTAR_RUST_WALK_STRICT``
     - ``0``
     - With shadow mode, raise on divergence instead of logging (CI /
       debugging).

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
       ``/health`` and ``/v1/models`` bypass the cap.
   * - ``MSTAR_MAX_BODY_MB``
     - ``128``
     - Request body limit (multipart uploads included).
   * - ``MSTAR_TOKENIZER``
     - unset
     - Path to a HuggingFace ``tokenizer.json`` enabling frontend
       tokenization. Leave unset with the Python backend — its preprocess
       worker owns tokenization, and the bridge rejects pre-tokenized
       ingest.
