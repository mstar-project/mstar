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
- **Workers** (``mstar/worker/``): one process per GPU. Each runs an engine manager, a
  micro-scheduler (continuous batching), and a KV cache manager, and routes tensors
  directly to downstream workers.
- **Engines** (``mstar/engine/``): execution backends that actually run submodules on the
  GPU — ``KVCacheEngine`` (nodes with a persistent paged KV cache, e.g. autoregressive
  LLMs and LLM-as-denoiser flow loops) and ``StatelessEngine`` (everything else: ViT/VAE
  encoders and decoders, codec decoders, projection/combine stages).
- **Models** (``mstar/model/``): each model declares its computation graph, tokenization,
  engine types, and submodules. Registered via ``mstar/model/registry.py``.
- **Graph** (``mstar/graph/``): computation-graph primitives — ``GraphNode``,
  ``Sequential``, ``Parallel``, ``Loop``, ``GraphEdge``.
- **Communication** (``mstar/communication/``): ZMQ-based IPC/TCP messaging; tensor
  transport over RDMA or TCP.
- **Streaming** (``mstar/streaming/``): streaming output with configurable chunking
  policies and async partition topology.

Core design principles
----------------------

- **Models define execution plans.** Each model provides its own graph walks (e.g.
  ``prefill``, ``decode``, ``image_gen``) via ``get_graph_walk_graphs()``.
- **Disaggregated.** Logical computation nodes map to physical workers via the YAML
  config's ``node_groups`` (node names → GPU ranks).
- **Graph-driven scheduling.** The conductor schedules graph walks and their transitions
  to coordinate multi-engine pipelines, including async producer/consumer partitions.

Execution flow (simplified)
---------------------------

1. The API server receives a request, loads media, and calls the model's
   ``process_prompt`` to produce the initial tensors.
2. The conductor seeds the initial graph walk (e.g. ``prefill``) and asks the model for
   the next forward-pass arguments after each graph walk completes.
3. Workers execute the ready graph nodes on GPU through the appropriate engine and route
   outputs (tensors) to downstream nodes/workers.
4. Outputs marked for the client are post-processed (``postprocess``) and streamed back.
