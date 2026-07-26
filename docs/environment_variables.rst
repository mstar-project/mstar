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
     - unset
     - ``1``/``true``: skip all CUDA graph capture (AR, stateless and
       piecewise runners) so every submodule runs its eager path.
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
