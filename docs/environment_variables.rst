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
   * - ``MSTAR_VIT_CUDA_GRAPH``
     - ``0``
     - BAGEL: CUDA-graph capture of the ViT block loop, over the node's ragged
       attention resource (see :doc:`adding_models`). Off by default: capture
       costs one graph per (batch size, token bucket) and the eager
       flash-attn path is already fast. The win is removing per-layer launch
       overhead on small images.
   * - ``MSTAR_VIT_CG_TOKEN_BUCKETS``
     - ``512,1024,2048,4096,4900``
     - Comma-separated token-count buckets to capture, when
       ``MSTAR_VIT_CUDA_GRAPH=1``. A batch longer than the largest bucket runs
       eagerly. ``4900`` is ``70*70``, the exact length the ``vllm``
       preprocess option emits.
   * - ``MSTAR_VIT_CG_BATCH_SIZES``
     - ``1,2,4``
     - Comma-separated batch sizes to capture. Read only when
       ``MSTAR_VIT_BATCHING=1``; otherwise only batch size 1 is captured.

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

Worker scheduling
-----------------

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Meaning
   * - ``MSTAR_TP_ASYNC_SCHED``
     - ``0``
     - Async scheduling for lockstep-parallel (TP / SP) nodes. ``1``: the
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
       parallel nodes only. ``0``: the serial path — leader schedules
       after N, followers rebuild after the broadcast. Set it identically
       on every rank of an instance: the workers compare it at startup and
       refuse to start on a mismatch. Leave ``MSTAR_ENGINE_STEP_SYNC`` at
       ``0`` with it: that throttle holds the GPU thread until step N drains,
       which serialises the very overlap this flag buys.
   * - ``MSTAR_PRE_PLAN_SPEC``
     - ``1``
     - Pre-plan the speculative batch's attention on a dedicated thread
       while the previous replay runs. ``0`` plans inline on the GPU
       thread.
   * - ``MSTAR_MAX_CONSECUTIVE_SPEC_STEPS``
     - ``1024``
     - Cap on back-to-back speculative steps before the leader yields to
       other ready work.
   * - ``MSTAR_SPEC_PEEK_FOR_FAIRNESS``
     - ``1``
     - Yield the speculation chain only when another (node, walk) is
       actually ready right now; ``0`` uses the consecutive-step cap alone.
   * - ``MSTAR_PHASE_TIMING``
     - ``0``
     - ``N > 0``: every N iterations log per-phase p50/p95/mean of the
       worker main loop (speculate, await_gpu, submit_spec, ...).

Tensor-parallel collectives
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Variable
     - Default
     - Meaning
   * - ``MSTAR_FAST_ALLREDUCE``
     - ``1``
     - Route small TP all-reduces through a symmetric-memory kernel on the
       compute stream instead of NCCL. A decode step issues two all-reduces
       per layer plus one for the vocab-parallel embedding, all small (8 KiB
       at bs=1, hidden 4096), and they serialise — so the per-call fork/join
       between the compute stream and ``ProcessGroupNCCL``'s internal stream
       is unhideable. Prefers NVLink SHARP (``multimem``) where the fabric
       exposes multicast, falling back to ``one_shot``, then to NCCL.

       Measured on 2xH100 (NV18), Qwen3.5-9B TP2, collectives per decode step
       in situ: NCCL 1.811 ms, one_shot 0.597 ms, multimem 0.379 ms.

       End-to-end at 2304-token outputs, ``0`` vs ``1`` in one job, 3 trials
       each: conc=1 222.6 -> 231.5 tok/s (+4.0%), conc=4 775.7 -> 800.4
       (+3.2%). Trials repeat to within 0.2% at those concurrencies. conc=16
       is not quoted: its run-to-run spread is ~8%, wider than the effect.
       Greedy output is bitwise identical to the NCCL path.

       ``0`` forces NCCL. Falls back automatically wherever the fast path
       does not apply, so it is safe to leave on.
   * - ``MSTAR_SP_CAPTURE_ALL_GATHER``
     - ``1``
     - Which Ulysses exchange the *captured* Cosmos3 denoise step uses.
       ``1`` (default) all-gather; ``0`` all-to-all. Both compute the same
       thing — identical output pixels — so this is purely a perf knob, and
       the default is the fast one.

       The all-gather was chosen because grouped point-to-point send/recv
       would not replay from a CUDA graph. That constraint is gone (torch
       2.12 / current NCCL captures and replays it correctly, verified at
       SP2 and SP4 over 20 replays), but lifting it does not make the
       all-to-all preferable here. The two trade off by sequence length:
       the all-to-all moves ``P`` times fewer bytes and wins when
       bandwidth-bound, the all-gather is one tuned collective instead of
       ``P-1`` send/recv pairs and wins when latency-bound. The captured
       path only sees short sequences (~240 tokens at 256p, ~1560 at 480p),
       so it is firmly in the second regime: on cosmos3-nano SP4,
       setting ``0`` made t2i **slower** — 256p 0.733 -> 0.866 s (+18%),
       480p 0.990 -> 1.033 s (+4%).

       Keep it at ``1``. It is worth revisiting only if a captured path
       ever gets long sequences; at seq 8192 one exchange is 491 us
       all-gather vs 190 us all-to-all at SP4. Video is unaffected either
       way — it runs eager and already uses the all-to-all (t2v control:
       3.764 -> 3.700 s, i.e. noise).
   * - ``MSTAR_FAST_ALLREDUCE_MAX_KIB``
     - ``2048``
     - Size ceiling for the fast path; above it NCCL wins and mstar hands
       off. Both kernels are small-message wins only — on 2xH100, us per
       all-reduce (chained, graph-replayed, result landed back in place):
       8 KiB NCCL 18.48 / one_shot 8.69 / multimem 7.76; 128 KiB 20.71 /
       9.87 / 8.33; 8 MiB 58.72 / 64.53 / 74.88. Re-measure with
       ``test/scratch/collective_shootout.py`` on new topology — multicast
       availability and the crossover are fabric-dependent.
