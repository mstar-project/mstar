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
