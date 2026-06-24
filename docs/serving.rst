Serving
=======

``mstar`` has two ways to start a server:

- ``mstar serve <model>`` — a one-command wrapper with sensible per-model defaults (most
  single-GPU; a few multi-GPU). Best for getting started and single-node runs.
- ``mstar-serve --config <yaml>`` — the low-level entry point that takes an explicit
  config. Use it for custom layouts, disaggregation, and tensor parallelism.

``mstar serve`` resolves a default config for the model, fills in the plumbing that
``mstar-serve`` needs (socket/upload dirs, a single-node-safe tensor protocol, the HF
cache), and then delegates to it.

mstar serve
-----------

.. code-block:: bash

   mstar serve <model> [options]

``<model>`` is one of ``bagel``, ``bagel_cfg_parallel``, ``qwen3_omni``, ``orpheus``,
``pi05``, ``vjepa2``, ``vjepa2_ac`` (or pass ``--config`` for any other deployment).

.. list-table::
   :header-rows: 1
   :widths: 28 24 48

   * - Option
     - Default
     - Description
   * - ``--host``
     - ``0.0.0.0``
     - Bind address.
   * - ``--port``
     - ``8000``
     - HTTP port.
   * - ``--gpus``
     - all visible
     - Sets ``CUDA_VISIBLE_DEVICES`` (e.g. ``0`` or ``0,1,2``).
   * - ``--config``
     - model default
     - Override the resolved default config with a path to your own YAML.
   * - ``--cache-dir``
     - HF default
     - HuggingFace weight cache directory.
   * - ``--tensor-comm-protocol``
     - ``SHM``
     - Tensor transport: ``SHM`` (safe single-node default), ``TCP``, or ``RDMA``.
   * - ``--socket-path-prefix``
     - ``/tmp/mstar_<user>/``
     - ZMQ IPC socket prefix.
   * - ``--upload-dir``
     - ``/tmp/mstar_uploads_<user>/``
     - Temp directory for uploaded media.
   * - ``--log-level``
     - ``INFO``
     - ``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``.
   * - ``--log-stats``
     - off
     - Print a per-request profile when each request finishes (see
       :ref:`request-profiling`).
   * - ``--log-stats-file``
     - stdout
     - Append the per-request profiles to this file instead of stdout
       (implies ``--log-stats``).

Each model maps to a default config (override with ``--config``):

.. list-table::
   :header-rows: 1
   :widths: 22 44 12

   * - Model
     - Default config
     - GPUs
   * - ``bagel``
     - ``configs/bagel_single_gpu.yaml``
     - 1
   * - ``bagel_cfg_parallel``
     - ``configs/bagel_cfg_parallel.yaml``
     - 3
   * - ``orpheus``
     - ``configs/orpheus_colocated.yaml``
     - 1
   * - ``qwen3_omni``
     - ``configs/qwen3omni_2gpu.yaml``
     - 2
   * - ``pi05``
     - ``configs/pi05.yaml``
     - 1
   * - ``vjepa2``
     - ``configs/vjepa2.yaml``
     - 1
   * - ``vjepa2_ac``
     - ``configs/vjepa2_ac.yaml``
     - 1

mstar-serve
-----------

.. code-block:: bash

   mstar-serve --config configs/<model>.yaml [options]

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Option
     - Default
     - Description
   * - ``--config`` *(required)*
     - —
     - Path to the YAML config.
   * - ``--host`` / ``--port``
     - ``0.0.0.0`` / ``8000``
     - Bind address / HTTP port.
   * - ``--tensor-comm-protocol``
     - ``RDMA``
     - ``RDMA``, ``TCP``, or ``SHM``.
   * - ``--cache-dir``
     - HF default
     - HuggingFace weight cache directory.
   * - ``--socket-path-prefix``
     - ``/tmp/mstar``
     - ZMQ IPC socket prefix (shared with conductor/workers).
   * - ``--upload-dir``
     - ``/tmp/mstar_uploads``
     - Temp directory for uploaded media.
   * - ``--timeout``
     - ``600``
     - Per-request timeout (seconds).
   * - ``--mooncake-port``
     - ``8080``
     - Port for the Mooncake RDMA transfer engine.
   * - ``--tcp-transfer-device``
     - (auto)
     - Network device for TCP tensor transport.
   * - ``--enable-nvtx``
     - off
     - Emit NVTX markers for profiling.
   * - ``--log-stats``
     - off
     - Print a per-request profile when each request finishes (see
       :ref:`request-profiling`).
   * - ``--log-stats-file``
     - stdout
     - Append the per-request profiles to this file instead of stdout
       (implies ``--log-stats``).
   * - ``--log-level``
     - ``INFO``
     - ``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``.

.. note::

   ``mstar serve`` defaults the tensor protocol to ``SHM`` (safe on a single node),
   whereas ``mstar-serve`` defaults to ``RDMA``. On a single node without RDMA, pass
   ``--tensor-comm-protocol SHM`` to ``mstar-serve``.

Config files
------------

A config maps the model's computation-graph nodes to physical GPU ranks. The keys:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Meaning
   * - ``model``
     - Registry key of the model (see :doc:`models`).
   * - ``max_seq_len``
     - Maximum sequence length (sizes the KV cache).
   * - ``node_groups``
     - List of placements. Each entry assigns ``node_names`` to ``ranks``, optionally
       scoped to specific ``graph_walks`` and/or sharded with ``tp_size``.
   * - ``model_kwargs``
     - *(optional)* Server-init model parameters (see below).

Node names are model-specific — they are the keys of the model's
``get_node_engine_types`` (e.g. BAGEL's ``vit_encoder`` / ``vae_encoder`` / ``LLM``,
Orpheus's ``LLM`` / ``snac_decoder``).

**Single GPU.** Everything on rank 0:

.. code-block:: yaml

   model: "bagel"
   max_seq_len: 32768
   node_groups:
     - {node_names: [vit_encoder], ranks: [0]}
     - {node_names: [vae_encoder, vae_decoder], ranks: [0]}
     - {node_names: [LLM], ranks: [0]}

**Disaggregation.** The same node can live on different GPUs *per graph walk* — e.g.
prefill, decode, and image generation on three GPUs:

.. code-block:: yaml

   node_groups:
     - {node_names: [LLM], ranks: [0], graph_walks: [prefill_text, prefill_vit, prefill_vae]}
     - {node_names: [LLM], ranks: [1], graph_walks: [decode]}
     - {node_names: [LLM], ranks: [2], graph_walks: [image_gen]}

Because placement is config-only, the *same* model code runs single-GPU or fully
disaggregated. ``configs/`` ships several layouts per model (``*_single_gpu``,
``*_colocated``, ``*_pd_disaggregated``, ``*_cfg_parallel``, …).

**Tensor parallelism.** Shard a node across GPUs with ``tp_size`` and that many ``ranks``:

.. code-block:: yaml

   model: "orpheus"
   max_seq_len: 131072
   node_groups:
     - {node_names: [LLM], ranks: [0, 1], tp_size: 2, graph_walks: [prefill, decode]}
     - {node_names: [snac_decoder], ranks: [0], graph_walks: [snac_chunk]}

A node must be declared TP-enabled by the model to be eligible for ``tp_size > 1``; the
weight loaders then shard parameters automatically, with no model-code changes. See
:ref:`Tensor parallelism <tensor-parallelism>` in the model guide for the model-side
details.

model_kwargs
------------

``model_kwargs`` are model parameters fixed at server start (not per request) — they are
baked into the model's config dataclass and into CUDA-graph captures. For example, the
Pi0.5 DROID variant fixes the action horizon:

.. code-block:: yaml

   model: "pi05"
   max_seq_len: 2048
   model_kwargs:
     action_horizon: 15
   node_groups:
     - {node_names: [vit_encoder], ranks: [0]}
     - {node_names: [LLM], ranks: [0]}

Per-request knobs (``temperature``, ``voice``, ``max_output_tokens``, …) are sent by the
client instead — see :doc:`clients`.

.. _tensor-transport:

Tensor transport
----------------

Workers route tensors directly to one another using one of three transports, selected with
``--tensor-comm-protocol``:

- ``SHM`` — shared memory; the safe default for single-node deployments.
- ``TCP`` — works anywhere; used for some multi-node setups.
- ``RDMA`` — lowest latency for multi-GPU / multi-node, via the Mooncake transfer engine
  (requires RDMA-capable networking).

.. _request-profiling:

Request profiling
-----------------

Pass ``--log-stats`` (to either ``mstar serve`` or ``mstar-serve``) to print a per-request
profile when each request finishes. Add ``--log-stats-file <path>`` to append the profiles
to a file instead of stdout (it implies ``--log-stats``):

.. code-block:: bash

   mstar serve bagel --log-stats
   mstar-serve --config configs/orpheus_colocated.yaml --log-stats-file stats.txt

The report has four sections (any that has no data is omitted):

.. code-block:: text

   ============================================================
    Request profile: 3f9c1e2a-…
   ============================================================
    Inputs:
      text         x1           72 B
      image        x1        1.1 MiB
    Outputs:
      image        x1        1.1 MiB
   ------------------------------------------------------------
    Timeline:
      recv → preprocess done                       62.6 ms
      preprocess done → conductor ingest            2.3 ms
      conductor ingest → first chunk              133.7 ms
      first chunk → last chunk                    683.3 ms
      last chunk → conductor done                   1.5 ms
      conductor done → finish                       3.1 ms
      total                                       886.5 ms
   ------------------------------------------------------------
    Graph timings (CPU, ms) — total over request (avg per exec):
                                    all                 fwd                 pre               post*
      LLM
        decode     n=588      3277.9 (   5.57)    2858.0 (   4.86)     243.0 (   0.41)     176.9 (   0.30)
        prefill    n=1          8.78 (   8.78)      4.24 (   4.24)      3.47 (   3.47)      1.06 (   1.06)
      * post overlaps the next step / another in-flight batch under
        speculative scheduling, so it is not necessarily additive.
   ------------------------------------------------------------
    Tensor transfer   size / transport-time / count:
      rx (received over the wire), by source → dest:
        worker_0 → worker_1
          hidden                  106.7 MiB      78.7 ms  (x49)
      tx (registered/written for send), by source:
        worker_0
          hidden                  106.7 MiB       0.90 ms  (x49)
   ============================================================

**Inputs / Outputs** — per-modality count and byte size. Inputs are the *raw*
client-supplied bytes (uploaded file sizes on disk, UTF-8 length of the prompt), not the
much larger decoded tensors, so they line up with what was sent over the wire.

**Timeline** — wall-clock stage boundaries for the request as it flows
``api server → conductor → workers → api server``. Stages are ordered by their actual
timestamp, so a stage is never negative even when two checkpoints race (e.g. the API
server's ``last chunk`` and the conductor's ``done`` can arrive in either order). All
timestamps are ``time.perf_counter()``; they are comparable across the api-server,
conductor, and worker processes because those run on the **same host** (``perf_counter`` is
``CLOCK_MONOTONIC``, which is boot-relative and shared).
This implementation will be updated when multi-node support is added.

**Graph timings** — CPU time per ``(node, graph_walk)``, accumulated over the request.
Each cell shows the **total over the request** and, in parentheses, the **average per
execution** (``n`` is the execution count, so e.g. one entry covers all ``588`` decode
steps). The columns:

- ``all`` — the full node execution (= ``pre`` + ``fwd`` + ``post``).
- ``fwd`` — the GPU forward region. Because execution is asynchronous and *not* synced
  (a sync would serialize the pipeline), this measures the CPU launch/enqueue span of the
  whole batch, not per-request GPU time.
- ``pre`` — input preparation / preprocessing before the launch.
- ``post`` — CPU postprocess after the forward. Marked ``post*`` because under
  speculative scheduling it can overlap the next step's forward or another in-flight
  batch, so it is **not necessarily additive**.

**Tensor transfer** — bytes moved between entities via the tensor transport
(:ref:`above <tensor-transport>`). ``rx`` (received) is grouped by ``source → dest``; ``tx``
(sent) is grouped by source only — the sender registers/writes data without knowing a
priori which worker will read it (and may register data that is never sent), so there is no
per-destination breakdown. ``size`` is total bytes, the time is transport cost (the
RDMA-read or SHM file-read time on ``rx``; the registration / serialize-and-write time on
``tx``), and ``count`` is the number of tensors.

.. note::

   Profiling is gated end-to-end on ``--log-stats``: when it is off, the workers,
   conductor, and data worker skip the timing/transfer bookkeeping entirely.
