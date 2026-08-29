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
   * - ``MSTAR_DIST_TIMEOUT_S``
     - config's ``dist_timeout_s``
     - Timeout in seconds for the NCCL world group and its parallel
       subgroups (:func:`mstar.distributed.communication.resolve_dist_timeout`).
       Overrides the deployment config's ``dist_timeout_s``; with neither set,
       PyTorch's default applies. Raise it only where weight load, JIT or
       CUDA-graph capture can exceed that default (a 1T MoE at TP8 does) —
       a hung collective takes correspondingly longer to abort. Must be set
       before the conductor spawns workers, which inherit it.
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
   * - ``MSTAR_TP_ALLREDUCE``
     - ``nccl``
     - All-reduce backend for small TP messages
       (:mod:`mstar.distributed.communication`). ``nccl``: torch.distributed's
       ring/tree. ``symm_oneshot`` / ``symm_multimem``: route messages up to
       ``MSTAR_TP_SYMM_AR_MAX_KB`` through torch symmetric memory (one-shot:
       each rank reads all peers over NVLink and reduces locally; multimem:
       NVLS multicast, needs NVSwitch + driver support). A B=1 TP8 decode step
       is ~160–230 all-reduces of ~10 KB, where NCCL's per-call latency
       dominates (measured in-graph: NCCL 26.4 µs vs multimem 7.4 µs per
       call). Larger messages stay on NCCL. CUDA-graph capturable. Reduction
       order differs from NCCL's, so bf16 results can differ in the last bit
       — measure before flipping. Falls back to ``nccl`` with a warning if
       symmetric memory is unavailable.
   * - ``MSTAR_TP_SYMM_AR_MAX_KB``
     - ``512``
     - Message-size cap (KB) for the symmetric-memory all-reduce path above;
       larger tensors use NCCL.
   * - ``MSTAR_TP_STEP_BARRIER``
     - ``1``
     - Per-step ``dist.barrier()`` at the engine's execute entry
       (``KVCacheEngine.execute_forward``, ``StatelessEngine.execute_batch``).
       On NCCL a barrier is a dummy all-reduce plus a current-stream
       synchronize, so the host blocks until the previous step has drained
       — it forbids enqueueing step N+1 behind step N. The forward's own
       collectives keep ranks in lockstep without it; capture-time barriers
       are separate and unaffected. ``0`` removes it. Turning it off widens
       the host run-ahead window, which is why pinned staging buffers are
       event-fenced (``mstar/utils/sampling.py``).

GLM-5.2 (MTP, capture, collectives)
-----------------------------------

Read by :mod:`mstar.model.glm52`. Every knob below is a measured
default with an escape hatch; the ones that change numerics (reduction
order or norm convention) are called out. "Bit-exact" refers to the
3264-token greedy stream of the TP8 bench against plain (MTP-off,
captured) decode.

.. list-table::
   :header-rows: 1
   :widths: 34 8 58

   * - Variable
     - Default
     - Meaning
   * - ``MSTAR_GRAPH_COMPILE_MODE``
     - ``max-autotune-no-cudagraphs``
     - ``torch.compile`` mode for the forward captured by the piecewise /
       full-graph runners (``mstar/engine/cuda_graph_runner.py``). ``default``
       keeps Inductor's fusion but takes cuBLAS for every GEMM and skips
       autotuning: at decode shapes (M=4) max-autotune's warm-L2 benchmark
       picks Triton GEMM templates that run 2× slower than cuBLAS's split-K
       kernels once the weights stream from HBM — measured −8 % on the GLM-5.2
       trunk at per-rank dims, capture 130 s → 27 s. Any mode string
       ``torch.compile`` accepts.
   * - ``MSTAR_INDUCTOR_GEMM_BACKENDS``
     - unset
     - When set (e.g. ``ATEN`` or ``ATEN,TRITON``), restricts Inductor's
       ``max_autotune_gemm_backends`` for the captured forward; only matters
       under an autotuning compile mode.
   * - ``MSTAR_GLM52_GRAPH_COMPILE``
     - ``1``
     - Capture the ``torch.compile``'d forward into the CUDA graphs. ``0``
       captures the eager forward instead — escape hatch for an
       Inductor-subprocess Triton crash (``per_token_group_quant_fp8_kernel``
       failing in ``make_llir`` under the compile pool) that once failed
       every capture and silently degraded the fast config to eager. Stream
       capture alone still removes launch overhead; Inductor fusion is what
       ``0`` gives up.
   * - ``MSTAR_GLM52_MTP_PAIR_POSTNORM``
     - ``1``
     - MTP draft input pairing uses the trunk's post-norm hidden state (the
       reference convention). Recovers per-position acceptance p1/p2 to
       0.89/0.74 from pre-norm's 0.77/0.33. ``0`` restores pre-norm pairing.
       Changes which tokens are drafted, never which are emitted (greedy
       verify emits the trunk's own argmax).
   * - ``MSTAR_GLM52_MTP_CAPTURE_SYNC``
     - ``1``
     - Capture the decode sync pass as a padded ``(bs, k+1)`` piecewise
       graph. Bit-identical to the eager pass (pinned by
       ``test_sync_capture_matches_eager_bit_identically``). ``0`` runs it
       eager; with post-norm pairing also on, the eager path has a known FP
       near-tie fork of ~0.25 % of the stream, so strict bit-identity to
       plain decode then also needs ``MSTAR_GLM52_MTP_PAIR_POSTNORM=0``.
   * - ``MSTAR_GLM52_MTP_CAPTURE_PREFILL``
     - ``1``
     - Capture the MTP *prefill* trunk (embed + all layers over the packed
       prompt) as a piecewise graph over the same token buckets the k=0
       config captures. Without it the whole prefill runs eager under MTP:
       TTFT 305 ms vs 57 ms at k=0. Sample, plane sync and draft chain stay
       outside the graph. ``0`` runs prefill eager; per-bucket capture
       failures fall back to eager on their own.
   * - ``MSTAR_GLM52_MTP_PREFILL_DRAFTS``
     - ``1``
     - Bundle the first drafts with the prefill output so the first decode
       step is a speculated ``(k+1)``-row step like every other. ``0`` drops
       the bundle — one wasted unspeculated step per request. Read through a
       single helper so the edge that writes the bundle and the transition
       that reads it cannot disagree.
   * - ``MSTAR_GLM52_MTP_DRAFT_PHASE_GRAPH``
     - ``1``
     - The whole decode draft phase as ONE captured graph: padded sync pass,
       draft-1 head and the ``k-1`` chain iterations, with ``k`` FlashInfer
       plan slots planned before a single replay. The three-replay version
       was host-bound at ~5 ms/step (1.2–2 ms of host per piecewise
       ``run()``); one replay is ~1.2 ms. Requires sync capture. ``0``
       restores the three-graph path (still the fallback for missing
       buckets).
   * - ``MSTAR_GLM52_MTP_PHASE_PREPARE``
     - ``0``
     - Hoist the accepted-count-independent half of the draft phase (sync
       inputs, contiguous positions, slot-0 FlashInfer plan) above the
       verify readback, so the host does ~0.6–0.9 ms of it while blocked in
       the verify ``.tolist()`` instead of stalling the GPU afterwards.
       Bit-exact by construction; default off until a TP8 arm measures it.
   * - ``MSTAR_GLM52_PLAN_ROPE``
     - ``1``
     - Keep the per-plan ``plan_rope`` staging. GLM-5.2 applies RoPE
       explicitly in its attention and never calls ``cache_handle.apply_rope``,
       so every ``plan_rope`` is dead work: a pinned host build plus an
       async H2D per plan, 3–4 per MTP step, two of them on the post-verify
       critical path. ``0`` skips them; default on until a TP8 arm confirms
       bit-identity (the reduced-dims GPU test already does).
   * - ``MSTAR_GLM52_MOE_FUSED_ALLREDUCE``
     - ``0``
     - Add the shared-expert output to the routed partial *before* the TP
       all-reduce and reduce once (the ``DeepseekV2MoE`` layout) instead of
       reducing each separately: one collective fewer per MoE layer (76 at
       TP8; the step went 233 → 158 all-reduces measured). ``1`` enables. Off by default because bf16 rounding
       order moves (sum-then-reduce vs reduce-then-sum); the 08-19 TP8 arm
       that enabled it together with ``MSTAR_TP_ALLREDUCE=symm_multimem``
       stayed bit-exact on the bench, but that is a measurement, not a
       guarantee.
   * - ``MSTAR_GLM52_MTP_STEP_TIMING``
     - ``0``
     - ``N`` > 0 logs the per-phase GPU|host split (trunk, verify readback,
       draft phase, tail) of every ``N``-th decode step from CUDA events —
       an nsys-lite for the MTP step. The GPU column is stream-elapsed time
       between events and so *includes* idle the GPU spends waiting on the
       host to enqueue; read it next to the host column, not alone.

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
