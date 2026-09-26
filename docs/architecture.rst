Architecture
============

High-level components
---------------------

``mstar`` is organized as a set of cooperating processes:

- **API server** (``mstar/api_server/``): FastAPI layer that accepts ``POST /generate``,
  tokenizes/loads media, dispatches the request, and streams results back to the client.
  Entry point: ``mstar.api_server.entrypoint:main`` (the ``mstar-serve`` console script).
- **Conductor** (``mstar/conductor/``): central coordinator. It manages the request
  lifecycle, handles graph-walk transitions, selects workers, routes inputs, and
  detects completion.
- **Workers** (``mstar/worker/``): one process per GPU. Each runs an engine manager and a
  micro-scheduler (continuous batching), drives eviction and offload, and routes tensors
  directly to downstream workers.
- **Engine** (``mstar/engine/engine.py``): the single execution backend that runs
  submodules on the GPU. It compiles forwards, captures CUDA graphs, batches requests, and
  runs each step's resource lifecycle: admit, plan, forward, commit.
- **Resources** (``mstar/engine/resources/``): the state that a node's compute uses. This
  includes paged KV caches, the attention planned over them (FlashInfer or dense),
  cross-attention over a fixed context, cacheless (ragged) attention for encoder towers
  that attend within one packed forward, position embeddings, and samplers. A model
  declares which resources each node needs, and the engine builds them. A node that
  declares no resources receives none. VAE encoders, codec decoders, and projection and
  combine stages are examples.
- **Models** (``mstar/model/``): each model declares its computation graph, tokenization,
  node resources, and submodules. Registered via ``mstar/model/registry.py``.
- **Graph** (``mstar/graph/``): computation-graph primitives — ``GraphNode``,
  ``Sequential``, ``Parallel``, ``Loop``, ``GraphEdge``.
- **Communication** (``mstar/communication/``): ZMQ-based IPC/TCP messaging; tensor
  transport over RDMA or TCP.
- **Streaming** (``mstar/streaming/``): streaming output with configurable chunking
  policies and async partition topology.

Process failures
----------------

A worker that dies (an exception during setup, an OOM kill, a segfault) sends nothing, so
the conductor polls its worker process handles during the startup wait and, twice a
second, from its main loop. A dead worker is fatal for the deployment. The conductor logs
the worker, its pid, its exit code or signal and the nodes it hosted, fails every request
that is still waiting with a 503 that names it, terminates the remaining workers and exits
non-zero. The API server polls the conductor the same way. If it exits before the workers
are ready, the server never binds. If it exits while serving, pending requests get a 503,
new ones are refused, the HTTP server stops and the process exits non-zero. Each worker
also watches its parent and exits if the conductor is gone, so a conductor that is killed
outright cannot leave workers behind holding GPU memory. The conductor watches the API
server the same way, so an API server that is killed outright takes the whole deployment
down with it. SIGINT and SIGTERM shutdowns are unaffected.

Core design principles
----------------------

- **Models define execution plans.** Each model provides its own graph walks (e.g.
  ``prefill``, ``decode``, ``image_gen``) via ``get_graph_walk_graphs()``.
- **Disaggregated.** Logical computation nodes map to physical workers via the YAML
  config's ``node_groups`` (node names → GPU ranks).
- **Models declare, the engine executes.** A model declares its resources in
  ``get_node_resources()``, and declares what each step does to them in
  ``NodeSubmodule.declare_step()``. The engine performs admission, planning, capture and
  commit. A step that cannot be admitted therefore becomes backpressure that the scheduler
  can resolve by eviction, instead of an error inside a forward pass.
- **Graph-driven scheduling.** The conductor schedules graph walks and their transitions
  to coordinate multi-engine pipelines, including async producer/consumer partitions.

Execution flow (simplified)
---------------------------

1. The API server receives a request, loads media, and calls the model's
   ``process_prompt`` to produce the initial tensors.
2. The conductor seeds the initial graph walk (e.g. ``prefill``) and asks the model for
   the next forward-pass arguments after each graph walk completes.
3. Workers execute the ready graph nodes on GPU through the engine. The engine declares
   and admits each step's resource work around the forward pass. Workers then route the
   output tensors to downstream nodes and workers.
4. Outputs marked for the client are post-processed (``postprocess``) and streamed back.
