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
   * - ``--api-key``
     - ``MSTAR_API_KEY``, else off
     - Require ``Authorization: Bearer <key>`` on every route except ``/health``
       and ``/metrics`` (see :ref:`api-key`).
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
     - ``/tmp/mstar_<user>_<port>/``
     - ZMQ IPC socket prefix; one per server, so servers on other ports do
       not share sockets.
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
   * - ``--enable-metrics``
     - off
     - Serve Prometheus metrics at ``/metrics`` (see :ref:`serving-metrics`).
       Needs ``.[metrics]``.

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
   * - ``--api-key``
     - ``MSTAR_API_KEY``, else off
     - Require ``Authorization: Bearer <key>`` on every route except ``/health``
       and ``/metrics`` (see :ref:`api-key`).
   * - ``--tensor-comm-protocol``
     - ``RDMA``
     - ``RDMA``, ``TCP``, or ``SHM``.
   * - ``--cache-dir``
     - HF default
     - HuggingFace weight cache directory.
   * - ``--socket-path-prefix``
     - ``/tmp/mstar_<user>_<port>/``
     - ZMQ IPC socket prefix (shared with conductor/workers). The Dynamo
       worker (``python -m mstar.integrations.dynamo``) has no HTTP port, so
       it defaults to a new directory per launch and removes it at exit.
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
     - Emit NVTX markers for CUDA profiling. Disabled on XPU and CPU.
   * - ``--log-stats``
     - off
     - Print a per-request profile when each request finishes (see
       :ref:`request-profiling`).
   * - ``--log-stats-file``
     - stdout
     - Append the per-request profiles to this file instead of stdout
       (implies ``--log-stats``).
   * - ``--enable-metrics``
     - off
     - Serve Prometheus metrics at ``/metrics`` (see :ref:`serving-metrics`).
       Needs ``.[metrics]``.
   * - ``--log-level``
     - ``INFO``
     - ``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``.

.. note::

   ``mstar serve`` defaults the tensor protocol to ``SHM`` (safe on a single node),
   whereas ``mstar-serve`` defaults to ``RDMA``. On a single node without RDMA, pass
   ``--tensor-comm-protocol SHM`` to ``mstar-serve``.

.. _api-key:

API key
-------

Both commands listen on all interfaces by default and accept requests from anyone who
can reach the port. If no gateway in front of the server authenticates clients, set a
key or bind ``--host 127.0.0.1``; the server logs a warning at startup while no key is
set.

.. code-block:: bash

   export MSTAR_API_KEY="$(openssl rand -hex 32)"
   mstar serve qwen3_omni            # or: mstar-serve --config ... [--rust-frontend]

With a key set, every HTTP route except ``/health`` and ``/metrics`` (see
:ref:`serving-metrics`) answers ``401`` unless the request carries
``Authorization: Bearer <key>``, and WebSocket routes (``/v1/realtime``,
``/generate/ws``) refuse the handshake without it. The check is one middleware over the
whole app (FastAPI and the Rust frontend alike), and the comparison takes the same time
however much of the key matches. CORS preflights still get their answer, since browsers
send them without credentials. ``--api-key`` overrides the environment variable, but
prefer the variable: a command-line flag is visible to every local user in the process
list. An empty value turns the check off, so ``--api-key ""`` disables a key set in the
environment, and so does a variable that expands to nothing (``--api-key "$UNSET_VAR"``).

Clients send the key as their API key: ``MStarClient(base_url, api_key=...)`` (it reads
``MSTAR_API_KEY`` when not given one) or ``OpenAI(base_url=..., api_key=...)``; see
:doc:`clients`. The standalone Nemotron realtime server
(``python -m mstar.model.nemotron_duplex.realtime_api``) takes the same flag and
variable.

Under NVIDIA Dynamo, Dynamo's own frontend serves HTTP and does not forward
``Authorization`` to backends, so the M\* worker cannot check a key there: authenticate at
Dynamo's frontend or a gateway in front of it.

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
       scoped to specific ``graph_walks`` and/or sharded with ``tp_size``. Each node of a
       walk must be placed by exactly one entry for that walk.
   * - ``resources``
     - *(optional)* Per-resource overrides, keyed by the model's resource names (see
       below).
   * - ``max_concurrent_requests``
     - *(optional)* How many requests run at once. Requests past the cap wait in the
       conductor's queue.
   * - ``gpu_memory_fraction``
     - *(optional)* The share of its GPU each worker may hold once CUDA-graph capture is
       done, for workers that share a GPU (see below). Unset, capture plans against
       whatever is free.
   * - ``model_kwargs``
     - *(optional)* Server-init model parameters (see below).
   * - ``speculation_mixes_walks``
     - *(optional)* Per node, ``true``/``false`` for every walk of the node's combined
       walks, or a mapping of walk -> ``true``/``false``. On for a walk, a speculation
       chain on it merges ready rows of the other walks of its combined walk, so
       decode steps take in new prompts as they arrive. Off (the default), a chain
       keeps to its own walk and yields to the others, and a step mixes walks only
       when the scheduler builds one fresh. Naming a node with no combined walk, or a
       walk outside its combined walks, fails loading.
   * - ``warmup_requests``
     - *(optional)* Requests the server runs after the workers are ready and before it
       binds, so no client pays a shape's first-request cost (torch.compile of a step,
       a cold cache). Each entry is a ``/generate`` request in yaml form: ``text``,
       ``output_modalities``, ``model_kwargs``, and ``files``, a list of media
       inputs in prompt order (text follows them). A file is a path or
       ``hf://<owner>/<repo>/<path>`` (a file in a model's HF repo); its modality
       comes from the extension, or give ``{path: ..., modality: ...}``.
       ``repeat: N`` sends ``text`` N times over, a long prompt for a path only
       long prompts take (a long-context prefill). A malformed or failing warmup
       is logged and skipped.
   * - ``dist_timeout_s``
     - *(optional)* Timeout in seconds for the NCCL world group and its parallel
       subgroups during weight load and warmup; serving keeps PyTorch's default.
       Unset keeps PyTorch's default throughout. Raise it only where weight load,
       JIT or CUDA-graph capture can exceed that default.
   * - ``tp_allreduce``
     - *(optional)* Small-message all-reduce path of every TP and SP group:
       ``nccl`` (default), ``symm_oneshot`` or ``symm_multimem``; any other value
       fails at startup. The last two send
       contiguous bf16 and fp32 CUDA messages up to ``tp_allreduce_max_kb`` through
       torch symmetric memory: one-shot reads every peer's buffer over NVLink and
       reduces locally, multimem uses NVLS multicast (NVSwitch and driver support).
       Larger messages stay on NCCL, and so does one-shot inside a
       ``torch.compile``'d forward. Both are CUDA-graph capturable. Reduction order
       differs from NCCL's, so bf16 results can differ at the last bit. Falls back
       to NCCL, with a warning, where symmetric memory cannot be set up on every
       rank of the group.
   * - ``tp_allreduce_max_kb``
     - *(optional)* Largest message, in KiB, the symmetric-memory path takes;
       larger ones go to NCCL. It is also the size of the buffer each group
       rendezvouses. Default 512.
   * - ``tp_async_sched``
     - *(optional)* Async scheduling for the parallel nodes: ``true``, ``false``
       (default) or a list of node names (see ``MSTAR_TP_ASYNC_SCHED``, which
       overrides it).

Node names are model-specific — they are the node names appearing in the model's graph
walks (e.g. BAGEL's ``vit_encoder`` / ``vae_encoder`` / ``LLM``, Orpheus's ``LLM`` /
``snac_decoder``).

**Resource overrides.** A model declares the KV caches, attention backends, positions and
samplers that its nodes need. Each declaration has a ``resource_key``. A deployment tunes
them by that name:

.. code-block:: yaml

   resources:
     kv_cache:
       max_num_pages: 1024      # also: page_size, max_seq_len, cpu_offload_pages,
                                # prefix_cache, prefix_cache_salt
     talker_attn:
       flashinfer_backend: fa2  # also: backend (flashinfer / dense)
     vit_attn:                  # a cacheless (ragged) attention resource
       max_segments_per_request: 2  # also: max_tokens_per_request, flashinfer_backend

Use one block per resource. A model with two caches, such as Whisper's decoder cache and
its encoder context, can therefore size each one separately. If a block names a resource
that the model does not declare, or sets a key that the resource does not accept, loading
fails. A misspelled setting is never silently ignored. A top-level ``kv_cache:`` block is
no longer read. It raises an error with a message describing the migration.

``prefix_cache: false`` turns cross-request prefix reuse off for that cache, and
``prefix_cache_salt`` keeps two otherwise identical deployments off each other's
cached pages. A single request opts out by sending ``prefix_cache=False``, which
travels with its other ``model_kwargs``.

On a single rank, a request waits at the head of its pool's queue until its prompt fits
beside the prompts already admitted, and no request behind it starts first. A cache hit on
a page only the prefix index holds counts as a page the request takes. A request that
can't fit in its pool at its full output length gets a 400 if it asked for that length,
and otherwise has the model's default length cut until it fits. The pool
reserves only prompts, so a decode step can find no page outside the admitted prompts. The
worker then moves the least recently active request to the host if ``cpu_offload_pages`` has
room for it. Otherwise it drops the newest requests it can rebuild from their token ids. Either
way, the request that ran short gets the pages freed, up to its shortage, unless it was the request
moved to the host or dropped. A dropped request waits at its place in the queue until its history and
one more page fit, then recomputes its KV in one prefill row.

Above one rank, on a worker that runs only some of a node's walks, on a node with a second
paged KV cache, such as Whisper's decoder, and while no request on a node keeps its token ids,
as on Qwen3-TTS's Talker, a decode step that finds no page and nothing to offload holds its
requests until they time out. Elsewhere at one rank, a request that runs short and can't be
dropped waits for room when nothing can be offloaded and dropping newer requests wouldn't
cover its shortage. That is the case for a request with an
image or audio in its KV, and for every request under ``MSTAR_SAMPLER_INGRAPH_SCATTER=1``,
since the sampler then refuses every drop. Several such requests can wait on each other.
``max_concurrent_requests`` prevents both. It caps how many requests run at once, and the
pool's pages divided by the pages one request needs, prompt and output together, is a
safe value. ``cpu_offload_pages`` lets a full pool move a request to the host, and an offload
comes before any drop. An offload frees only the device pages no other running request reads,
though, so a request that shares most of its prompt frees little.
Offload helps a pool that runs short now and then. It doesn't replace the cap.

**Single GPU.** Everything on rank 0:

.. code-block:: yaml

   model: "bagel"
   max_seq_len: 32768
   node_groups:
     - {node_names: [vit_encoder], ranks: [0]}
     - {node_names: [vae_encoder, vae_decoder], ranks: [0]}
     - {node_names: [LLM], ranks: [0]}

**Several workers on one GPU.** A worker is one process per rank, and by default rank
``n`` runs on device ``n``. ``rank_devices`` maps ranks onto devices explicitly, so two
node groups can run as separate workers on the same GPU: a heavy autoregressive node then
never waits for a light node's steps on its worker loop (a TTS talker and its codec):

.. code-block:: yaml

   node_groups:
     - {node_names: [Talker], ranks: [0]}
     - {node_names: [Codec], ranks: [1]}
   rank_devices: {1: 0}

At start-up a worker sizes every CUDA-graph bucket of its nodes with a capture it throws
away, then captures the cheapest buckets across all of them while its largest eager step
still fits beside them; a bucket left out runs eagerly. Workers sharing a GPU see each
other's memory only as used, so the first to capture can take the other's room. Set
``gpu_memory_fraction`` (``0.45`` for two workers, say) to bound what each plans for. It
counts PyTorch allocations, not the CUDA context or NCCL buffers, so leave room for those.

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

``TCP`` and ``RDMA`` run Mooncake's transfer engine, which does not authenticate its peers.
Each worker's engine picks its handshake and data ports at startup, so restrict by host: only
the deployment's own hosts should reach the workers. The model extras require
``mooncake-transfer-engine`` 0.3.13.post1
or newer: earlier than 0.3.12.post1, any host that reaches the TCP data port can read or
write the worker's memory (CVE-2026-103764), and a worker warns at startup if it finds such a
version. Two denial-of-service issues on the handshake port (CVE-2026-103760 and
CVE-2026-103761) are still open in 0.3.13.post1.

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

.. _serving-metrics:

Metrics
-------

Pass ``--enable-metrics`` (to either ``mstar serve`` or ``mstar-serve``) to serve
Prometheus metrics at ``GET /metrics``. It needs the ``metrics`` extra
(``pip install 'mstar[metrics]'``); without it the server stops at startup with that
hint. Off by default, and ``/metrics`` answers 404 then. Like ``/health``, it needs no
API key even when one is set, the usual convention for scrape endpoints. It exposes
request counts, latencies and the model name (never prompts or outputs); if that
matters for a deployment, keep the port on an internal network. With
``--rust-frontend``, ``mstar-server`` serves the same text: it asks the API server for
it over the bridge, so both frontends show the same names, and it answers ``503`` if the
bridge does not reply within two seconds, as ``/health`` does.

:doc:`metrics` documents every family in one place, with a Prometheus and Grafana setup.

Requests are counted when they leave the API server, from the first request after the
warmup requests. Every family has a ``model_name`` label (the config's ``model``);
``output_modalities`` is the request's output modalities, sorted and comma-joined
(e.g. ``audio,text``).

- ``mstar_requests_total{model_name, output_modalities, outcome}``: counter.
  ``outcome`` is ``success`` (the result went to the HTTP layer), ``error`` (the
  request failed in the engine, preprocessing or result delivery), ``abort`` (the client
  went away, or its finished result was dropped unread) or ``timeout`` (``--timeout``
  expired). Requests refused before they were submitted (a 400 for a bad form, a 409
  for a ``request_id`` in flight) are not counted here but in
  ``mstar_http_refusals_total``.
- ``mstar_requests_in_flight{model_name}``: gauge, requests the API server holds
  (submitted and not yet returned, failed or aborted).
- ``mstar_e2e_request_latency_seconds{model_name, output_modalities}``: histogram over
  successful requests, from submission (after the request body was read) to the result
  leaving the API server.
- ``mstar_time_to_first_output_seconds{model_name, output_modalities}``: histogram over
  successful requests, from submission to the first result chunk of any modality
  reaching the API server. It is measured on the server, so it does not include the
  network to the client, and a non-streaming client sees nothing until the end.
- ``mstar_http_refusals_total{model_name, route, status}``: counter, HTTP requests the
  frontend answered with a 4xx status without submitting them to the API server: ``401``
  (missing or wrong API key), ``413`` (body over ``MSTAR_MAX_BODY_MB``), ``422`` (body
  that fails validation), ``400`` (a bad form field or request), ``409`` (a
  ``request_id`` in flight), ``404`` (unknown route, or a model without the endpoint),
  ``405`` (wrong method), plus ``503`` when the Rust frontend's
  ``MSTAR_MAX_CONCURRENT_REQUESTS`` cap turns a request away. ``route`` is the route's
  path as declared (``/v1/chat/completions``), ``unmatched`` when no route matches.
  A request that was submitted and then failed counts once, as an ``error`` in
  ``mstar_requests_total``, never here. WebSocket handshakes are not counted. With
  ``--rust-frontend``, ``mstar-server`` keeps these counts and sends them with each
  ``/metrics`` request, so the API server exports them under the same name.
- ``mstar_request_queue_seconds{model_name, output_modalities}``: histogram over
  successful requests, from the conductor taking the request in to admitting it for
  dispatch. Zero unless the request queued at ``max_concurrent_requests``.

While metrics are on, the conductor sends the API server a status message every second
(nothing when they are off). These gauges show its latest one, so they can be up to a
second old:

- ``mstar_conductor_requests{model_name, state}``: requests the conductor holds.
  ``state`` is ``waiting`` (queued at ``max_concurrent_requests``, not dispatched yet),
  ``running`` (dispatched to workers, not finished) or ``draining`` (completed, aborted
  or failed, and waiting for every reader to release its outputs before the workers free
  its state; it still counts toward ``max_concurrent_requests``).
- ``mstar_component_in_flight{model_name, component}``: per graph node (the
  ``node_names`` of the config's ``node_groups``), running requests with a pass on that
  node that the conductor dispatched and has not heard back from. The conductor sees
  whole passes, so a node running a request's decode loop counts it once for the loop.
- ``mstar_conductor_status_age_seconds{model_name}``: seconds since the API server
  received the last status message, or since metrics were enabled before the first one.
  More than a few seconds means the conductor stopped reporting.

The engine workers also send the conductor their own metrics every second, which it
adds to its status message, so these can be up to two seconds old. ``component`` is the
graph node; a node with replicas on several workers shows their sum, and the ranks of a
tensor or sequence parallel instance count once (the instance's first rank reports).
The histograms and counters count from the first request after the warmup requests, as
the API server's do (the gauges show the current state), so read them with ``rate()``
or ``increase()``.

- ``mstar_kv_pages_used{model_name, component, resource}``,
  ``mstar_kv_pages_total{model_name, component, resource}``: pages of each paged KV
  cache (``resource`` is its key in the model, since a node can have several, e.g.
  Whisper's decoder self and cross attention caches). Used counts every page off the
  free list, including the pages the prefix cache keeps; a cache two nodes of one
  worker share shows under each node.
- ``mstar_kv_pages_cached{model_name, component, resource}``: pages the prefix cache
  names. Some may also be in use by running requests; the cache frees the others when a
  request needs the room.
- ``mstar_step_seconds{model_name, component, graph_walk}``: histogram over engine
  steps that ran a forward. A step counts from when the worker hands it to its GPU
  thread, or from the end of the previous step if that was later, to the end of its
  postprocess (outputs on the host, stop check, routing). Overlapping steps are not
  counted twice: in steady state this is the interval between step completions, and
  ``rate(..._sum)`` is the share of time the worker had a step running.
- ``mstar_batch_size{model_name, component, graph_walk}``: histogram, rows the step's
  forward ran.
- ``mstar_prefix_cache_query_tokens_total{model_name, component}``,
  ``mstar_prefix_cache_hit_tokens_total{model_name, component}``: counters, prompt
  tokens looked up in the prefix cache (once per request, its whole keyed prompt) and
  tokens found there. Only models that key their prompts for prefix reuse (BAGEL and
  Orpheus) count; the cache is off under tensor parallelism.
- ``mstar_component_pending_steps{model_name, component}``: gauge, requests whose next
  step on the node is ready (its inputs arrived) and not running: waiting for room in a
  step or for the worker to pick the node.

Tensor transfers between graph nodes (:ref:`tensor transport <tensor-transport>`) are
counted by the reader, once per edge read: the tensors of one graph edge that one worker
read from one producer. The workers send them with the engine metrics above, and the
API server adds the results its data worker read:

- ``mstar_transfer_bytes_total{model_name, src, dst, transport}``: counter, tensor bytes
  read over the edge.
- ``mstar_transfers_total{model_name, src, dst, transport}``: counter, tensors read over
  the edge.
- ``mstar_transfer_seconds{model_name, src, dst, transport}``: histogram, one observation
  per edge read: the reader's time to read that edge's tensors, which is the SHM file
  read and copy, the copy out of the SHM arena, or the RDMA or TCP read. A GPU reader of
  the arena only issues its host-to-device copies in that time; they finish
  asynchronously. The producer's own work (serializing, staging into the arena,
  registering memory) is not included; it is the ``tx`` time of
  :ref:`request profiling <request-profiling>`.

``src`` is the graph node that produced the tensors and ``dst`` the node that reads them.
An edge the conductor joined from several producers (one input name produced in several
walks or nodes) is one read per producing node, with its read time split between them by
bytes.
``api_server`` stands for the request's inputs (as ``src``) and its results (as ``dst``),
which the API server's data worker sends and reads. ``transport`` is the protocol:
``SHM`` (per-tensor files or the arena, see ``MSTAR_SHM_ARENA``), ``TCP`` or ``RDMA``. The
Python and Rust runtimes and communicators (``MSTAR_RUST_GRAPH``, ``MSTAR_RUST_ZMQ``)
report the same series. A tensor already on the reader (one it made, or one it read once
for two of its nodes) moves nothing and is not counted. With tensor parallelism every
rank counts its own read, of its slice or of the whole tensor when the edge is
replicated, so an edge two ranks read is two reads. Like the data worker, the workers
count from the first request after the warmup requests.

Where a successful request's time went, from the timestamps every request already
carries (same ``model_name`` and ``output_modalities`` labels; the conductor's
timestamps are compared with the API server's, which assumes both run on one host, as
they do today):

- ``mstar_request_preprocess_seconds``: histogram, from submission to the data worker
  handing the preprocessed request to the conductor, including its wait in the data
  worker's queue.
- ``mstar_request_engine_seconds``: histogram, from the conductor dispatching the
  request to the workers (after any admission wait) to the conductor finishing it.
- ``mstar_request_delivery_seconds``: histogram, from the conductor finishing the
  request to the result leaving the API server (the last chunks' postprocessing and
  transfer, and the client handler picking it up).

Output, per output modality (``modality`` is the chunk's modality, e.g. ``audio``):

- ``mstar_output_chunks_total{model_name, modality}`` and
  ``mstar_output_bytes_total{model_name, modality}``: counters of the result chunks (and
  their bytes) that reached the API server, for every request whatever its outcome;
  error chunks are not counted.
- ``mstar_output_audio_seconds_total{model_name}``: counter, seconds of audio in those
  chunks (16-bit PCM at the chunk's sample rate and channel count).
- ``mstar_time_to_first_chunk_seconds{model_name, modality}``: histogram over
  successful requests, from submission to the first chunk of each output modality
  reaching the API server (time to first audio, first image, first video frame).

The data worker (a thread in the API server that loads each request's media and runs
the model's prompt processing, preprocessing one request at a time) reports its own
work.
``input_modalities`` is the request's input modalities, sorted and comma-joined like
``output_modalities``:

- ``mstar_data_worker_preprocess_seconds{model_name, input_modalities}``: histogram over
  requests whose preprocessing finished, from the data worker taking the request off its
  queue to handing it to the conductor. Unlike ``mstar_request_preprocess_seconds`` it
  leaves out the wait in the queue, and it counts requests whatever their outcome later.
- ``mstar_data_worker_preprocess_failures_total{model_name, input_modalities}``:
  counter, requests whose preprocessing failed in the data worker, each once, whatever
  raised: a file count or media length over its limit, a file that cannot be decoded,
  the model's prompt processing, or handing the inputs on. The client gets the error
  (a 400 for a bad input), so the request also ends as ``outcome="error"`` in
  ``mstar_requests_total`` unless its client had already gone. That counter counts
  failures from every stage by output modalities; this one only the data worker's
  preprocessing, by input modalities (failures turning results into output chunks are
  not counted here).
- ``mstar_upload_size_bytes{model_name, modality}``: histogram, the size on disk of
  each image, audio or video file a request carried to the data worker (``modality`` is
  ``image``, ``audio`` or ``video``), whether the request succeeded or failed. Every
  route writes its media to a file first: an upload as received, inline media (a
  ``data:`` URL, base64 files on ``/generate/ws``) after base64 decoding. The realtime
  route and long transcriptions send WAV files they write from the decoded audio, so
  those count the WAV sizes. Requests refused before they were submitted (an oversized
  body, a bad form) are not counted.
- ``mstar_data_worker_queue_depth{model_name}``: gauge, read at scrape time, requests
  submitted and not yet taken up by the data worker (the one it is working on is not
  counted). A request whose client went away still counts until the data worker takes
  it up.

.. code-block:: bash

   mstar serve kokoro --enable-metrics
   curl -s localhost:8000/metrics | grep '^mstar_'
