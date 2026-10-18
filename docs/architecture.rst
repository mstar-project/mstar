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
  includes paged KV caches, the attention planned over them (FlashInfer, AITER or dense),
  cross-attention over a fixed context, cacheless (ragged) attention for encoder towers
  that attend within one packed forward, recurrent state pools and the linear attention
  planned over them (gated DeltaNet, for hybrid models whose non-KV layers carry a
  fixed-size state instead of a growing cache), position embeddings, and samplers. A model
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

Hardware platforms
------------------

M* runs on NVIDIA GPUs (CUDA), AMD Instinct GPUs (ROCm) and, for some models, Intel GPUs
(XPU). Model code is the same everywhere. Platform differences stay in the engine's
resources, which pick their kernels when they're built.

``mstar.platforms.current_platform`` says which stack a process runs on (``is_cuda()``,
``is_rocm()``, ``is_xpu()``, ``is_cpu()``) and the GPU's architecture (``arch()``, e.g.
``sm_90`` or ``gfx942``). Ask it instead of reading ``torch.version.hip`` or a compute
capability. PyTorch's ROCm build calls AMD GPUs ``cuda`` devices and reports an MI300X as
compute capability 9.4, so neither says who made the GPU. ``device.type == "cuda"`` is
still the right check for "is this a GPU". Each process resolves the platform on first
use and keeps it. Workers are spawned, so each one resolves its own.

Kernel choice per resource
~~~~~~~~~~~~~~~~~~~~~~~~~~

Resource configs default their ``backend`` to ``auto``, which the resource turns into a
kernel library once, in ``build``. On NVIDIA ``auto`` is what M* has always run. A
backend named in the model or in the deployment YAML (``resources:`` overrides, see
:ref:`adding_models:Tuning resources per deployment`) is used as named, and a backend that
can't run on the platform fails at startup. Dense attention is the exception. Without its
kernel it falls back to the paged backend and logs a warning. Ops with no ``backend``
field (norms, RoPE, sampling) pick their kernel by platform on first call.

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Resource
     - NVIDIA
     - AMD (ROCm)
   * - Paged attention (``AttentionSpec``)
     - FlashInfer
     - AITER's Triton unified attention. At a power-of-two page size of 64 or less it
       uses AMD's FlashInfer build (``fa2``) when that's installed, because AITER's kernel
       reads the tail of a row's last page without masking it there. Non-causal steps
       with more than one query per row (DiT denoise, prefix-LM) also go to FlashInfer's
       ``fa2``, since AITER's kernel is causal only.
   * - Cross-attention (``CrossAttentionSpec``)
     - FlashInfer
     - AITER's unified attention, one kernel row per query token, with the same page
       size rule
   * - Dense attention (``backend: dense``)
     - FlashAttention-3 (``fa3-fwd``)
     - AITER's ``flash_attn_varlen_func``
   * - Ragged and ragged-cross attention (``RaggedAttentionSpec``,
       ``RaggedCrossAttentionSpec``)
     - FlashInfer's ragged prefill
     - AITER's ``flash_attn_varlen_func``, or AMD's FlashInfer if AITER is missing
   * - Flex attention (``backend: flex``)
     - ``flash-attn-4`` if it imports, else Triton (``MSTAR_FLEX_BACKEND``)
     - Same rule, so Triton (``flash-attn-4`` is NVIDIA only)
   * - Gated DeltaNet (``LinearAttnSpec``)
     - FlashInfer's GDN kernels
     - AITER. Chunked prefill runs eagerly, captured prefill buckets use a sequential
       kernel, decode uses FlyDSL (``gfx942``/``gfx950``) or the sequential kernel.
   * - Mamba-2 and causal conv1d
     - M*'s Triton kernels
     - The same kernels, without programmatic dependent launch (Hopper and newer only)
   * - ``mstar::rms_norm``
     - FlashInfer
     - AMD's FlashInfer
   * - Fused add + RMSNorm (``RMSNorm.forward_residual``)
     - FlashInfer's ``fused_add_rmsnorm``, called directly
     - AMD's FlashInfer, through the ``mstar::fused_add_rms_norm`` op
   * - RoPE (``mstar::rope_apply_qk_inplace``)
     - FlashInfer, angles computed in the kernel
     - AMD's FlashInfer, reading an fp32 cos/sin cache that ``PositionManager`` builds
       from ``PositionConfig``. A layer that passes other RoPE arguments gets the
       in-kernel version.
   * - Sampler (``SamplerSpec``)
     - FlashInfer's samplers with per-row seed and offset tensors
     - AMD's FlashInfer, same calls
   * - TP all-reduce + add + RMSNorm (``mstar::allreduce_add_rmsnorm``)
     - FlashInfer's ``allreduce_fusion``
     - AITER's custom all-reduce with its fused RMSNorm (one-stage kernel, TP 2/4/8,
       up to 80 tokens and fewer at TP4/TP8, AITER 0.1.19.post1 or newer). Anything
       else gets RCCL plus a separate norm.

When nothing on ROCm can serve a resource (neither AITER nor AMD's FlashInfer imports),
the worker fails at startup and the error says what to install.

Adding a backend
~~~~~~~~~~~~~~~~

There are a few places to plug in a new kernel library or platform. None of them run per
step.

- **The resource's backend enum.** Add a value and the class that runs the new kernels,
  in its own module that imports the library lazily. Then teach the resource's resolver
  (``resolve_attn_backend``, ``resolve_ragged_attn_backend``,
  ``resolve_linear_attn_backend``) when ``auto`` should pick it. The resolver is the only
  place that maps ``auto`` to a kernel.
- **PlatformDispatch** (``mstar.platforms.dispatch``). On ROCm torch runs a custom op's
  ``cuda`` kernel too, so an op whose ``cuda`` kernel has to call a different library
  per vendor wraps the call in ``PlatformDispatch(name, cuda=...,
  rocm=..., xpu=..., cpu=..., default=...)``. Each argument is a zero-argument factory that
  imports the library and returns the function. The first call picks one and later calls
  go straight to it. ``mstar::rms_norm``, the RoPE op, the samplers and the all-reduce
  fusion work this way.
- **Inductor settings.** The engine wraps each ``torch.compile`` it does in
  ``with_inductor_overrides(fn, num_tokens)``, which patches in
  ``current_platform.inductor_overrides(num_tokens)`` (``num_tokens`` is the capture
  bucket's token count). No platform sets any today. An override should come with a
  measured win.
- **Process environment.** ``Platform.process_env_defaults()`` is applied with
  ``os.environ.setdefault`` in the conductor before it spawns workers, so values you
  export win. It's empty on every platform today.
- **Capture refusal.** A resource whose kernel can't run some capture bucket inside a
  graph returns a reason from ``Resource.capture_refusal(spec)``. Both graph runners ask
  before capturing, and a refused bucket runs eagerly. Answer from the spec and the
  resource's config only, never per-rank state, so every rank refuses the same buckets.

Capture and compile on ROCm
~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA graphs on ROCm are hipGraphs, captured and replayed through ``torch.cuda.graph`` with
the same capture configs as on NVIDIA. ``torch.compile`` settings are the same too. The
differences are in what each kernel can do inside a graph.

- AITER's chunked GDN prefill builds its chunk schedule on the host, so it can't be
  captured. Prefill buckets up to ``linear_attn.max_captured_prefill_tokens`` (512 by
  default) are captured with a sequential kernel whose cost grows with the longest row.
  Larger buckets are refused and run eagerly. Decode is always captured.
- AITER's paged attention stages its block table from pinned host memory into static
  buffers kept per bucket, slot and label. It also pins the kernel's ``max_seqlen`` hints
  per bucket, so a replay runs the kernel variant its capture recorded.
- Dense attention never runs under capture, on any platform.
- AMD's FlashInfer samplers raise when traced by dynamo, so M* calls them through
  ``torch.compiler.disable``. Captured graphs still draw fresh numbers on each replay
  from the seed and offset tensors.
- Under TP every capture runs inside the TP group's ``graph_capture()`` scope. With the
  AITER all-reduce fusion it registers the buffers the capture recorded, on every rank.
  Elsewhere it does nothing.
- After warmup on NVIDIA, ``reset_device_scheduling`` creates and destroys a throwaway
  CUDA context to undo late graph-node launches. It goes through the CUDA driver, so it
  does nothing on ROCm.
- Kernels compile during warmup, before capture. AITER ships most of its kernels prebuilt
  in its release wheel, and FlyDSL and FlashInfer compile theirs the first time they
  see a new configuration. Cold starts take minutes, see :doc:`installation`.
