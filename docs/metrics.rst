.. _metrics-reference:

Metrics reference
=================

Every Prometheus metric family M\* serves at ``GET /metrics``, what moves it, and which
process measures it. How to turn metrics on is in :ref:`serving-metrics`; a ready-to-run
Prometheus and Grafana setup is in :ref:`metrics-setup`.

.. code-block:: bash

   mstar serve kokoro --enable-metrics
   curl -s localhost:8000/metrics | grep '^mstar_'

Metrics are off by default. With ``--enable-metrics`` off, ``/metrics`` answers ``404``,
the conductor sends no status message and the workers record nothing. Like ``/health``,
``/metrics`` needs no API key; it exposes counts, latencies and the names of the model,
its graph nodes and routes, never prompts or outputs.

Where the numbers come from
---------------------------

Only the API server process renders ``/metrics``, from one registry. The other
processes send it plain numbers on messages they already exchange, so the time a value
reflects depends on where it was measured:

- **API server.** Request counts, request latencies, output volume and HTTP refusals
  are recorded as requests pass through it. The data worker is a thread of the API
  server, so its families are recorded in the same process. These are current at every
  scrape.
- **Conductor.** While metrics are on, the conductor sends the API server a status
  message every second. Its gauges show the latest message, so they can be up to a
  second old; ``mstar_conductor_status_age_seconds`` says how old.
- **Engine workers.** Each worker sends the conductor its cumulative numbers every
  second, and the conductor adds them to its next status message, so worker families
  can be up to two seconds old. A lost or reordered message cannot corrupt a counter:
  every message carries totals, not increments, and the API server keeps the latest.
- **Rust frontend.** With ``--rust-frontend``, ``mstar-server`` serves ``/metrics`` by
  asking the API server for the rendered text over the bridge, so both frontends show
  the same families. It answers ``503`` if the bridge does not reply within two
  seconds. ``mstar-server`` counts its own HTTP refusals and sends the totals with each
  ``/metrics`` request, and the API server exports them in
  ``mstar_http_refusals_total``.

The registry holds only ``mstar_`` families: no ``process_`` or ``python_`` metrics.
Every value starts from zero when the server starts.

Conventions
-----------

- Names use Prometheus base units (``_seconds``, ``_bytes``), and counters end in
  ``_total``. Read counters and histograms with ``rate()`` or ``increase()``.
- Every family has a ``model_name`` label: the config's ``model`` key (the registry
  name, e.g. ``kokoro`` or ``qwen3_omni``).
- Label values come from fixed sets or from the model and its config, never from
  requests, so the number of series is bounded:

  - ``outcome``: ``success``, ``error``, ``abort``, ``timeout``.
  - ``output_modalities`` and ``input_modalities``: the request's modalities, sorted
    and comma-joined (``audio,text``), from ``text``, ``image``, ``audio``, ``video``,
    ``video_frame``, ``action``, ``scalar``, ``tensor``. In practice a model has a few
    combinations.
  - ``modality``: one of those modality names.
  - ``state``: ``waiting``, ``running``, ``draining``.
  - ``component``: a graph node name from the config's ``node_groups``.
  - ``graph_walk``: a graph walk name of the model.
  - ``resource``: the key of a paged KV cache in the model.
  - ``src`` and ``dst``: a graph node name, or ``api_server``.
  - ``transport``: ``SHM``, ``TCP``, ``RDMA``.
  - ``route``: a route path as declared (``/v1/chat/completions``), or ``unmatched``.
  - ``status``: an HTTP status code the frontend sent (a ``4xx``, or ``503``).

- A labeled series appears the first time it is used. For a new set of output
  modalities, ``mstar_requests_total`` exports all four outcomes at zero from that
  request on, and the data worker's two families start together, so a later first
  error shows in ``rate()``. ``rate()`` and ``increase()`` miss what was counted before
  the first scrape that saw a series: if a set's first request fails, its error series
  starts at 1.

What is counted from when
-------------------------

- **API server families count from the first request after warmup.** The server
  enables metrics after its warmup requests, so first-shape compile times stay out of
  the request latency histograms.
- **Engine worker histograms and counters, and the transfers workers read, count
  from the same point.** When the API server enables its metrics it has the conductor
  tell every worker, so a worker counts the steps whose time starts after that, and
  the prefix-cache lookups and tensor reads after it. The gauges (KV pages, pending
  steps) show the current state from worker start.
- **Requests refused before submission** (a bad form, a missing API key, a body over
  the limit, a ``request_id`` already in flight) are counted only in
  ``mstar_http_refusals_total``. A request that was submitted and then failed is
  counted once, as ``outcome="error"`` in ``mstar_requests_total``.

Requests
--------

Exported by the API server, recorded where a request leaves it, the same way for
``/generate``, ``/generate/ws``, the OpenAI routes and both frontends.

``mstar_requests_total{model_name, output_modalities, outcome}``
   Counter. Requests that left the API server. ``success``: the result went to the
   HTTP layer. ``error``: the request failed in the engine, preprocessing or result
   delivery, or the conductor went away. ``abort``: the client went away, or its
   finished result was dropped unread. ``timeout``: ``--timeout`` expired.

``mstar_requests_in_flight{model_name}``
   Gauge, read at scrape time. Requests the API server holds: submitted and not yet
   returned, failed or aborted.

``mstar_e2e_request_latency_seconds{model_name, output_modalities}``
   Histogram, seconds, successful requests. From submission (after the request body was
   read) to the result leaving the API server. Buckets: :ref:`E2E <metrics-buckets>`.

``mstar_time_to_first_output_seconds{model_name, output_modalities}``
   Histogram, seconds, successful requests. From submission to the first result chunk
   of any modality reaching the API server. Measured on the server: it leaves out the
   network to the client, and a non-streaming client sees nothing until the end.
   Buckets: :ref:`first output <metrics-buckets>`.

Request stages
--------------

Exported by the API server, for successful requests, from the timestamps every request
carries. The conductor's timestamps are compared with the API server's, which assumes
both run on one host.

``mstar_request_preprocess_seconds{model_name, output_modalities}``
   Histogram, seconds. From submission to the data worker handing the preprocessed
   request to the conductor, including the wait in the data worker's queue. Buckets:
   :ref:`stage <metrics-buckets>`.

``mstar_request_queue_seconds{model_name, output_modalities}``
   Histogram, seconds. From the conductor taking the request in to admitting it for
   dispatch: zero unless the request queued at ``max_concurrent_requests``. The
   conductor sends it with the request's completion. Buckets:
   :ref:`queue <metrics-buckets>`.

``mstar_request_engine_seconds{model_name, output_modalities}``
   Histogram, seconds. From the conductor dispatching the request to the workers (after
   any wait in ``mstar_request_queue_seconds``) to the conductor finishing it. Buckets:
   :ref:`E2E <metrics-buckets>`.

``mstar_request_delivery_seconds{model_name, output_modalities}``
   Histogram, seconds. From the conductor finishing the request to the result leaving
   the API server: the last chunks' postprocessing and transfer, and the client handler
   picking the result up. Buckets: :ref:`stage <metrics-buckets>`.

Output
------

Exported by the API server, counted per result chunk as it reaches the API server.

``mstar_output_chunks_total{model_name, modality}``
   Counter. Result chunks, for every request whatever its outcome. Error chunks are not
   counted.

``mstar_output_bytes_total{model_name, modality}``
   Counter. Bytes of those chunks.

``mstar_output_audio_seconds_total{model_name}``
   Counter, seconds. Audio in those chunks, from their byte length as 16-bit PCM at the
   chunk's sample rate and channel count. Audio chunks that carry no sample rate are
   not counted.

``mstar_time_to_first_chunk_seconds{model_name, modality}``
   Histogram, seconds, successful requests. From submission to the first chunk of each
   output modality reaching the API server: time to first audio, first image, first
   video frame. Buckets: :ref:`first output <metrics-buckets>`.

Data worker
-----------

Exported by the API server. The data worker is the API server thread that loads each
request's media and runs the model's prompt processing, one request at a time.

``mstar_data_worker_preprocess_seconds{model_name, input_modalities}``
   Histogram, seconds. From the data worker taking a request off its queue to handing
   it to the conductor, for every request whose preprocessing finished, whatever its
   outcome later. Unlike ``mstar_request_preprocess_seconds``, it leaves out the wait in
   the queue. Buckets: :ref:`stage <metrics-buckets>`.

``mstar_data_worker_preprocess_failures_total{model_name, input_modalities}``
   Counter. Requests whose preprocessing failed in the data worker, each once: a file
   count or media length over its limit, a file that cannot be decoded, the model's
   prompt processing, or handing the inputs on. Such a request also ends as
   ``outcome="error"`` in ``mstar_requests_total`` unless its client had already gone.

``mstar_upload_size_bytes{model_name, modality}``
   Histogram, bytes. Size on disk of each ``image``, ``audio`` or ``video`` file a
   request carried to the data worker, whether the request succeeded or failed: an
   upload as received, inline media (a ``data:`` URL, base64 files on
   ``/generate/ws``) after decoding. The realtime route and long transcriptions count
   the WAV files they write from the decoded audio. Buckets:
   :ref:`upload size <metrics-buckets>`.

``mstar_data_worker_queue_depth{model_name}``
   Gauge, read at scrape time. Inputs (requests and client-stream appends) waiting in
   the data worker: submitted, and neither being prepared nor handed on to the
   conductor yet. One whose client went away counts until its turn comes and it is
   dropped.

HTTP refusals
-------------

Exported by the API server, counted by whichever frontend serves the deployment.

``mstar_http_refusals_total{model_name, route, status}``
   Counter. HTTP requests the frontend answered with an error without submitting them
   to the API server: ``401`` (missing or wrong API key), ``413`` (body over
   ``MSTAR_MAX_BODY_MB``), ``422`` (body that fails validation), ``400`` (a bad form
   field or request), ``409`` (a ``request_id`` in flight), ``404`` (unknown route, or
   a model without the endpoint), ``405`` (wrong method), and ``503`` when the Rust
   frontend's ``MSTAR_MAX_CONCURRENT_REQUESTS`` cap turns a request away. WebSocket
   handshakes are not counted. With ``--rust-frontend``, the values are the totals
   ``mstar-server`` sent with the latest ``/metrics`` request.

Conductor
---------

From the conductor's status message, sent every second while metrics are on.

``mstar_conductor_requests{model_name, state}``
   Gauge, up to a second old. Requests the conductor holds. ``waiting``: queued at
   ``max_concurrent_requests``, not dispatched yet. ``running``: dispatched to workers,
   not finished. ``draining``: completed, aborted or failed, and waiting for every
   reader to release its outputs before the workers free its state; it still counts
   toward ``max_concurrent_requests``.

``mstar_component_in_flight{model_name, component}``
   Gauge, up to a second old. Running requests with a pass on the graph node that the
   conductor dispatched and has not heard back from. The conductor sees whole passes,
   so a node running a request's decode loop counts it once for the loop.

``mstar_conductor_status_age_seconds{model_name}``
   Gauge, seconds, computed at scrape time. Time since the API server received the
   conductor's last status message, or since metrics were enabled if none arrived yet.
   More than a few seconds means the conductor stopped reporting, and the conductor,
   worker and transport values its message carries are stale.

Engine workers
--------------

From the workers' numbers, forwarded in the conductor's status message, so up to two
seconds old. A node with replicas on several workers shows their sum. The ranks of a
tensor or sequence parallel instance count once: the instance's first rank reports.

``mstar_kv_pages_used{model_name, component, resource}``
   Gauge. Pages of a paged KV cache not on its free list: held by requests, or kept by
   the prefix cache. A node can have several caches (e.g. a decoder's self and cross
   attention), told apart by ``resource``. A cache two nodes of one worker share shows
   under each node, so summing over ``component`` counts it twice.

``mstar_kv_pages_total{model_name, component, resource}``
   Gauge. Pages of the cache that requests can hold.

``mstar_kv_pages_cached{model_name, component, resource}``
   Gauge. Pages the prefix cache names. Running requests may also hold some of them;
   the cache frees the others when a request needs the room.

``mstar_step_seconds{model_name, component, graph_walk}``
   Histogram, seconds, engine steps that ran a forward. From when the worker hands the
   step to its GPU thread, or from the end of the previous step if that was later, to
   the end of the step's postprocess (outputs on the host, stop check, routing).
   Overlapping steps are not counted twice: in steady state this is the interval
   between step completions, and ``rate(mstar_step_seconds_sum[1m])`` is the share of
   time the worker had a step running. Buckets: :ref:`step <metrics-buckets>`.

``mstar_batch_size{model_name, component, graph_walk}``
   Histogram, rows the step's forward ran. Buckets: :ref:`batch <metrics-buckets>`.

``mstar_prefix_cache_query_tokens_total{model_name, component}``
   Counter. Prompt tokens looked up in the prefix cache, once per request (its whole
   keyed prompt). Only models that key their prompts for prefix reuse count (BAGEL,
   Orpheus and Qwen3.5). Under tensor parallelism only rank 0 counts a lookup.

``mstar_prefix_cache_hit_tokens_total{model_name, component}``
   Counter. Tokens the prefix cache already held at that lookup. The hit rate is
   ``rate(mstar_prefix_cache_hit_tokens_total[5m])`` divided by
   ``rate(mstar_prefix_cache_query_tokens_total[5m])``. For Qwen3.5 these are the KV
   cache's tokens before the recurrent pool cuts the hit to a snapshot, so a hit the pool
   cuts to 0 still counts.

``mstar_component_pending_steps{model_name, component}``
   Gauge. Requests whose next step on the node is ready (its inputs arrived) and not
   running: waiting for room in a step, or for the worker to pick the node.

Tensor transport
----------------

Tensor transfers between graph nodes (:ref:`tensor-transport`), counted by the reader,
once per edge read: the tensors of one graph edge that one worker read from one
producer. Workers report through the conductor (up to two seconds old); the API server
adds what its data worker read since metrics were enabled, which the data worker
publishes once a second. ``src`` is the node that produced the tensors, ``dst``
the node that reads them, and ``api_server`` stands for the request's inputs (as
``src``) and its results (as ``dst``). The Python and Rust runtimes report the same
series.

``mstar_transfer_bytes_total{model_name, src, dst, transport}``
   Counter, bytes. Tensor bytes read over the edge.

``mstar_transfers_total{model_name, src, dst, transport}``
   Counter. Tensors read over the edge.

``mstar_transfer_seconds{model_name, src, dst, transport}``
   Histogram, seconds, one observation per edge read. The reader's time to read the
   edge's tensors: the SHM file read and copy, the copy out of the SHM arena, or the
   RDMA or TCP read. A GPU reader of the arena only issues its host-to-device copies in
   that time. The producer's own work (serializing, staging, registering memory) is not
   included; it is the ``tx`` time of :ref:`request-profiling`. Buckets:
   :ref:`transfer <metrics-buckets>`.

A tensor already on the reader moves nothing and is not counted. An edge the conductor
joined from several producers is one read per producing node, with its read time split
between them by bytes. Under tensor parallelism every rank counts its own read, so an
edge two ranks read is two reads.

.. _metrics-buckets:

Histogram buckets
-----------------

Upper bounds, without ``+Inf``:

- **E2E** (``mstar_e2e_request_latency_seconds``, ``mstar_request_engine_seconds``):
  0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 60, 120, 300, 600,
  1200 s.
- **First output** (``mstar_time_to_first_output_seconds``,
  ``mstar_time_to_first_chunk_seconds``): 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2,
  0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 20, 30, 60, 120, 300, 600 s.
- **Stage** (``mstar_request_preprocess_seconds``, ``mstar_request_delivery_seconds``,
  ``mstar_data_worker_preprocess_seconds``): 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025,
  0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600 s.
- **Queue** (``mstar_request_queue_seconds``): 0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
  0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600 s.
- **Upload size** (``mstar_upload_size_bytes``): 1, 4, 16, 32, 64, 128, 256, 512 KiB,
  then 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024 MiB.
- **Step** (``mstar_step_seconds``): 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01,
  0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30 s.
- **Batch** (``mstar_batch_size``): 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 rows.
- **Transfer** (``mstar_transfer_seconds``): 10, 25, 50, 100, 250, 500 µs, then 1, 2.5,
  5, 10, 25, 50, 100, 250, 500 ms, then 1, 2.5, 5, 10 s.

Example queries
---------------

.. code-block:: text

   # Requests per second by outcome
   sum by (outcome) (rate(mstar_requests_total[1m]))

   # Share of requests that did not succeed
   sum(rate(mstar_requests_total{outcome!="success"}[5m]))
     / sum(rate(mstar_requests_total[5m]))

   # p95 end-to-end latency and time to first output
   histogram_quantile(0.95, sum by (le) (rate(mstar_e2e_request_latency_seconds_bucket[5m])))
   histogram_quantile(0.95, sum by (le) (rate(mstar_time_to_first_output_seconds_bucket[5m])))

   # KV cache use per node and cache
   mstar_kv_pages_used / mstar_kv_pages_total

   # The conductor stopped reporting (alert)
   mstar_conductor_status_age_seconds > 10

.. _metrics-setup:

Prometheus and Grafana
----------------------

``examples/observability/`` runs Prometheus and Grafana in Docker Compose against an
M\* server on the host:

.. code-block:: bash

   mstar serve kokoro --enable-metrics          # on the host, port 8000
   docker compose -f examples/observability/docker-compose.yml up -d

Prometheus (``localhost:9090``) scrapes ``/metrics`` every 5 seconds; Grafana
(``localhost:3000``) has Prometheus as its default data source and loads any dashboard
JSON placed in ``examples/observability/grafana/dashboards/``. Its README covers how
the containers reach the host on Linux, a server on another port, and the optional
anonymous viewer.

The ``M* serving`` dashboard there (``mstar.json``) has a panel for every family on
this page, in one row per component: requests, request stages and output, data worker,
conductor, engine workers, tensor transport and HTTP refusals. Its ``model`` variable
picks the ``model_name`` values to show.

To scrape from an existing Prometheus, add a job for the server's address:

.. code-block:: yaml

   scrape_configs:
     - job_name: mstar
       metrics_path: /metrics
       scrape_interval: 5s
       static_configs:
         - targets: ["localhost:8000"]

The conductor and workers report once a second, so a scrape interval under a second
only repeats values. Keep the scrape timeout above two seconds with
``--rust-frontend``, which waits that long for the bridge before answering ``503``.
