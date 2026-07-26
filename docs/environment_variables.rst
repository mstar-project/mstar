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

Engine
------

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Variable
     - Default
     - Meaning
   * - ``MSTAR_DISABLE_CUDA_GRAPH``
     - ``0``
     - ``1`` skips CUDA-graph capture for **every** node in the worker
       process — the paged-KV engine's decode / prefill graphs, the
       stateless engine's codec graphs, and the piecewise runners — so
       forwards are dispatched kernel by kernel. ``torch.compile`` is
       unaffected. Useful for debugging shape or memory issues that
       capture hides, and for profilers that can't see inside a replayed
       graph; expect a large decode-latency regression.
   * - ``MSTAR_DISABLE_TORCH_COMPILE``
     - ``0``
     - ``1`` skips every ``torch.compile`` the **engines** apply, for all
       nodes: the post-warmup compile of ``forward`` / ``forward_batched``
       and the ``max-autotune-no-cudagraphs`` compile the CUDA-graph
       runners do before capture. Graph capture still happens, around
       eager kernels. Use it to cut Inductor warmup time or to rule out a
       compile-induced numerical difference. It does **not** affect
       ``torch.compile`` calls inside model code — a submodule opts out of
       those via its own ``disable_torch_compile`` attribute.
   * - ``MSTAR_DISABLE_BATCHING``
     - ``0``
     - ``1`` forces sequential execution: every engine reports a max batch
       size of ``1``, so a scheduled batch of *n* requests is split into
       *n* single-request forwards
       (``BaseEngine.execute_with_max_batch_size``) instead of one batched
       one. CUDA graphs stay enabled, but each runner captures only its
       ``bs=1`` bucket — the larger ones could never be replayed, and
       skipping them cuts warmup time and capture memory. A capture config
       that doesn't declare ``bs=1`` at all is skipped entirely (logged as
       a warning) and that walk runs eager. Use it to isolate a
       batching-dependent correctness bug, or to measure single-request
       latency without queueing effects. Throughput drops accordingly.
   * - ``MSTAR_AUTOCAST_DTYPE``
     - unset
     - Overrides the engines' autocast dtype for every node
       (``float32``, ``bfloat16`` or ``float16``); wins over both the
       model's ``get_autocast_dtype`` and the config's
       ``autocast_dtype``. ``float32`` disables autocast and keeps
       weights and activations in full precision. Exception: FlashInfer
       has no float32 attention kernels, so paged-KV pools and the
       attention-kernel boundary stay bf16 (q/k/v are cast on entry, the
       output is cast back). Per-submodule ``get_autocast_dtype``
       overrides still win for their node.

``MSTAR_DISABLE_CUDA_GRAPH`` and ``MSTAR_DISABLE_TORCH_COMPILE`` are the
process-wide form of the per-node ``disable_cuda_graph_nodes`` /
``disable_torch_compile_nodes`` config keys (see
:ref:`Running nodes eagerly <eager-node-overrides>`); a node is skipped if
either its config key or the env var says so. ``MSTAR_DISABLE_BATCHING`` has
no per-node counterpart — batch composition is a whole-worker property.
