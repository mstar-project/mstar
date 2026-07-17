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
     - Growth cap per producer entity. At the cap, sends backpressure
       until consumers ACK and space is reclaimed.
   * - ``MSTAR_SHM_ARENA_FULL_TIMEOUT_S``
     - ``30``
     - Strict mode only (``MSTAR_SHM_ARENA_SPILL=0``): how long a send
       backpressures on a full arena before failing.
   * - ``MSTAR_SHM_ARENA_SPILL``
     - ``1``
     - Degrade gracefully at the segment cap: after a short backpressure
       grace (``MSTAR_SHM_ARENA_SPILL_AFTER_S``, default ``0.05``), stage
       the tensor through the per-uuid file protocol instead — slower,
       never fails, matching the file transport's saturation behavior.
       ``0`` restores strict backpressure + timeout.
   * - ``MSTAR_SHM_ARENA_SPILL_AFTER_S``
     - ``0.05``
     - How long a send waits for consumer ACKs before spilling.
   * - ``MSTAR_SHM_ARENA_PIN``
     - ``1``
     - ``cudaHostRegister`` each mapped segment (both sides) so D2H/H2D
       copies through the side streams run at page-locked bandwidth and
       stay asynchronous. ``0`` disables (pageable copies).
   * - ``MSTAR_SHM_ARENA_PIN_MAX_MB``
     - ``4096``
     - Budget for TOTAL pinned host memory, distinct from the segment cap
       (pinned pages come out of the OS's pageable pool system-wide).
       Segments past the budget — and oversized dedicated segments, whose
       one-shot transfer doesn't amortize the registration — stay
       unpinned: copies work, without async overlap.
