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
     - ``0``
     - Transport selection for the ZeroMQ control mesh (see
       :func:`mstar.communication.communicator.make_communicator`).
       ``0``: the pyzmq ``ZMQCommunicator``. ``1``: the Rust-backed
       ``RustZMQCommunicator`` (requires the vendored ``rust/`` extension;
       raises if it is not installed). ``AUTO``: Rust when the extension
       imports successfully, pyzmq otherwise. The two transports are
       wire-compatible, so this can be set per-process while the rest of
       the mesh stays on pyzmq.
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
