Adding a New Model
==================

This page walks through everything you need to implement to add support for a new model
in ``mstar``. By the end you will have a model that the conductor can schedule, that
workers can execute on GPU, and that you can launch with ``mstar-serve``.

Mental model
------------

A model in ``mstar`` is split into a handful of well-defined responsibilities:

- **The** ``Model`` **class** (``mstar/model/base.py``) is the contract the rest of the
  system talks to. It tokenizes prompts, declares the computation graph, says which
  engine runs each node, builds forward-pass arguments, and post-processes outputs. It
  contains *no* GPU compute.
- **Submodules** (``NodeSubmodule`` in ``mstar/model/submodule_base.py``) are the
  ``torch.nn.Module`` s that *do* the compute. Each graph node maps to one submodule.
- **Engines** (``mstar/engine/``) wrap submodules with execution machinery (KV cache,
  FlashInfer, CUDA graphs, batching). You pick an engine *type* per node; you rarely
  write a new engine.
- **The graph** (``mstar/graph/base.py``) is how a model declares *what runs in what
  order*: nodes, edges between them, and loops. Conceptually all of a model's work is
  one large computation graph, and each named "graph walk" (e.g. ``prefill``,
  ``decode``) is a *path* through it. For implementation purposes, though, you declare
  each walk as its own standalone graph; nodes that appear in several walks (e.g. the
  ``LLM``) are simply referenced by name in each, and the engine/submodule behind that
  name — and its KV cache — is shared across them.
- **The config YAML** (``configs/``) maps graph nodes to physical GPU ranks via
  ``node_groups``. This is where disaggregation happens — the same model code runs
  single-GPU or sharded across many depending only on the config.

A note on vocabulary used throughout this page: a **tensor bundle** routed between nodes
is a ``NameToTensorList`` — i.e. ``dict[str, list[torch.Tensor]]``
(``mstar/communication/tensors.py``), mapping an edge name to a (usually length-1) list
of tensors.

The flow at request time. This is the **conductor/model-level** flow, at the granularity
of a *graph walk*: the conductor is notified when a walk completes and only then asks the
model what to do next, so everything happening *within* a walk (the per-node execution
described in later steps) is abstracted away here::

   process_prompt()              # text/media -> initial tensors
        │
        ▼
   get_initial_forward_pass_args()   # seed the first graph walk (e.g. prefill)
        │
        ▼   (conductor walks the graph, running nodes via engines)
   get_partition_forward_pass_args() # asked after each graph walk completes:
                                     #   what's next? done?
        │
        ▼
   postprocess()                 # model output tensor -> bytes for the client

What you will create
--------------------

A typical model lives in its own package under ``mstar/model/<your_model>/``:

.. code-block:: text

   mstar/model/<your_model>/
   ├── __init__.py
   ├── config.py            # a @dataclass with architecture + generation params
   ├── <your_model>_model.py # the Model subclass (the contract)
   ├── submodules.py        # NodeSubmodule subclasses (the compute wrappers)
   └── components/          # the actual nn.Modules (attention, decoder, etc.)

Plus two things outside that package:

- an entry in ``mstar/model/registry.py`` so the model is discoverable, and
- a config YAML in ``configs/`` mapping nodes to ranks.

Step 1 — Register the model
---------------------------

Open ``mstar/model/registry.py`` and add your class to ``MODEL_REGISTRY`` (and, if it
loads weights from Hugging Face, to ``HF_MODELS``). The dict key is the string you put
under ``model:`` in a config YAML.

.. code-block:: python

   from mstar.model.your_model.your_model_model import YourModel

   MODEL_REGISTRY: dict[str, type[Model]] = {
       # ...
       "your_model": YourModel,
   }

   HF_MODELS: dict[str, dict] = {
       # ...
       "your_model": {"model_path_hf": "org/your-model-id"},
   }

That is the only wiring step — there is no plugin scan; the registry import is the
single source of truth.

Step 2 — Implement the ``Model`` class
--------------------------------------

Subclass :class:`mstar.model.base.Model` and implement its abstract methods. The
constructor receives ``model_path_hf`` (from ``HF_MODELS``) plus any ``**kwargs``; it
typically loads the tokenizer and stores a config dataclass. Defer heavy weight loading
to ``get_submodule`` so the conductor process never allocates GPU memory.

The abstract methods you **must** implement:

``get_kv_cache_config(self) -> list[KVCacheConfig]``
   Per-node KV cache configs for autoregressive nodes (``num_layers``,
   ``num_kv_heads``, ``head_dim``, ``max_seq_len``, ``num_qo_heads``). Return a single
   config if all AR nodes share one. Models with no AR node may return an empty list.

``get_node_engine_types(self) -> dict[str, EngineType]``
   Maps each graph-node name to an :class:`mstar.engine.base.EngineType`
   (``KV_CACHE`` or ``STATELESS``). See `Step 6 — Choose engine types`_ below.

``get_graph_walk_graphs(self) -> dict[str, GraphSection]``
   The heart of the model: returns ``{walk_name: graph}``. See
   `Step 3 — Declare the computation graph`_.

``process_prompt(self, prompt, input_modalities, output_modalities, tensors=None, **kwargs) -> NameToTensorList``
   Tokenize the prompt and produce the initial request tensors (e.g.
   ``{"text_inputs": [token_ids]}``). It runs in the API-server data worker *after*
   raw media tensors have been loaded, so it may read ``tensors`` (e.g.
   ``image_inputs`` / ``audio_inputs`` / ``video_inputs``) to compute derived tensors
   such as ``pixel_values``. The returned dict is merged into the request's tensors.

``get_initial_forward_pass_args(self, partition_name, input_modalities, output_modalities, input_signals, model_kwargs=None) -> ForwardPassArgs``
   Build the first :class:`mstar.model.base.ForwardPassArgs` for a partition — which
   graph walk to start on and which input edges feed it.

``get_partition_forward_pass_args(self, partition_name, partition_metadata, persist_signals, incoming_connections=None) -> ForwardPassArgs``
   Called by the conductor after each graph walk completes to decide the *next* walk, its
   inputs, and whether the request is done (``request_done=True``). For a simple
   prefill→decode model this flips ``is_prefill`` once and then loops decode until EOS.

``postprocess(self, output, modality) -> bytes``
   Encode a finished output tensor to bytes for the client (``utf-8`` for text, PNG for
   images, raw PCM for audio, …).

``get_submodule(self, node_name, device="cpu", tp_group=None) -> NodeSubmodule | None``
   Lazily build and return the ``NodeSubmodule`` for ``node_name`` (load weights here,
   on ``device``; ``tp_group`` is the node's tensor-parallel communicator when sharded —
   forward it to parallel-linear constructors, see :ref:`Step 7 <tensor-parallelism>`).
   Cache the result. Return ``None`` for dummy mode. See
   `Step 4 — Implement the submodules`_.

Useful overridable defaults (not abstract): ``get_sampling_config`` (temperature/top-p
per node), ``get_max_output_tokens``, ``get_autocast_dtype``, ``load_image`` /
``load_audio`` / ``load_video``, and the partition API below.

Step 3 — Declare the computation graph
--------------------------------------

``get_graph_walk_graphs`` returns one graph per *walk*. The primitives
(``mstar/graph/base.py``):

- ``GraphNode(name, input_names, outputs)`` — one unit of compute. ``name`` must match a
  key in ``get_node_engine_types``. ``input_names`` are the tensor names that must be
  present before the node can run; ``outputs`` is a list of ``GraphEdge``.
- ``GraphEdge(next_node, name, ...)`` — routes an output tensor named ``name`` to
  ``next_node``. Flags: ``persist=True`` keeps the tensor available for later steps/walks
  (this is how a generated token is carried from ``prefill`` into the ``decode`` loop),
  and ``output_modality`` — one of ``"text"``, ``"image"``, ``"audio"``, ``"video"``,
  ``"action"`` — with ``next_node=EMIT_TO_CLIENT`` streams the tensor to the client.
  Special destinations live in ``mstar/graph/special_destinations.py``
  (``EMIT_TO_CLIENT``, ``EMPTY_DESTINATION``). A ``decode`` loop stops when a submodule's
  ``check_stop`` registers a stop signal against the ``Loop`` (e.g. on EOS) — see Step 4 —
  not through any edge flag.
- ``Sequential([...])`` / ``Parallel([...])`` — compose subgraphs in order or
  concurrently.
- ``Loop(name, section, max_iters, outputs)`` — an iterating subgraph whose body feeds
  its own outputs back as the next iteration's inputs. It runs up to ``max_iters`` but
  can also stop early: give it a ``name`` so a submodule's ``check_stop`` can register a
  stop signal against that loop (e.g. on EOS). This is the usual ``decode`` loop.

A minimal text generator has two walks — a one-shot ``prefill`` node and a ``decode``
``Loop`` whose body feeds its own output back as the next input:

.. code-block:: python

   def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
       prefill = GraphNode(
           name="LLM",
           input_names=["text_inputs"],
           # the generated token persists so the decode loop can pick it up
           outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name="new_token",
                              persist=True)],
       )
       decode = Loop(
           name="decode_loop",
           section=GraphNode(
               name="LLM",
               input_names=["text_inputs"],
               outputs=[GraphEdge(next_node="LLM", name="text_inputs")],  # loop-back
           ),
           max_iters=self.get_max_output_tokens(),
           outputs=[],
       )
       return dict(prefill=prefill, decode=decode)

Step 4 — Implement the submodules
---------------------------------

Each node name maps to a :class:`mstar.model.submodule_base.NodeSubmodule` (a
``torch.nn.Module``). Autoregressive nodes use the ``ARNodeSubmodule`` subclass. The
contract:

``prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs``
   Convert the routed ``NameToTensorList`` into a typed ``NodeInputs`` (or
   ``ARNodeInputs`` with ``input_ids`` / ``input_embeds`` / ``input_seq_len``). Runs once
   per request and should do only **cheap, host-side ("CPU-ish") work** — shape/length
   bookkeeping, building position metadata, slicing token id lists. Do not launch the
   heavy GPU compute here (that is ``forward``); the engine may call this off the GPU
   thread and well ahead of execution.

``preprocess(self, graph_walk, engine_inputs, inputs) -> dict``
   Collate a list of ``NodeInputs`` into the kwargs your ``forward`` expects. The base
   default handles batch size 1 but can be overridden to support batching.
   ``ARNodeSubmodule`` makes this method abstract, as autoregressive submodules typically
   support continuous batching (though you can choose to disable batching for a node or
   for specific graph walk(s) if needed — see :ref:`Step 5 <step-5>`).

``forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList``
   The pure tensor → tensor computation. Keys in the returned dict are the edge
   ``name`` s the graph routes downstream. ``forward`` **is wrapped with** ``torch.compile``
   **automatically** (and CUDA-graph captured when you declare configs — see
   :ref:`Step 5 <step-5>`); for autoregressive (``KV_CACHE``) nodes ``forward_batched`` is
   compiled too, while the stateless engine leaves ``forward_batched`` eager. Keep the
   compiled paths compile-friendly; any helper that must *not* be traced (data-dependent
   Python control flow, host syncs) has to be excluded explicitly, e.g. with
   ``@torch.compiler.disable``.

``postprocess(...)`` (optional)
   Metadata-only fixups that run on the GPU thread — must **not** read tensor values (no
   ``.item()``/``.cpu()``/``.tolist()``). This is a **performance**, not a correctness,
   constraint: reading a value here forces a host sync that stalls the GPU thread and
   forfeits the worker's async-scheduling overlap. Use it only to rebind output names for
   routing. Value-dependent decisions belong in ``check_stop``.

``check_stop(...) -> set[str]`` (optional)
   Runs off the GPU thread and *may* read tensor values. Return the names of the
   ``Loop`` s to stop (e.g. when you see the EOS token). This is how decode terminates.

``cleanup_request(self, request_id)`` (optional)
   Free any submodule-internal per-request state when a request finishes — buffers,
   per-request caches, counters. See Qwen3-Omni's ``Code2WavSubmodule`` for an example.

The two batching/CUDA-graph knobs — ``can_batch`` / ``forward_batched`` and
``get_cuda_graph_configs`` — are important enough to get their own step below.

Loading weights
~~~~~~~~~~~~~~~

``get_submodule`` is where a node's parameters are actually loaded. Weight loading is
standardized through ``mstar/model/loader/`` — using it (rather than an ad-hoc
``load_state_dict``) is what lets the *same* code load both a single-GPU checkpoint and a
tensor-parallel shard (see :ref:`Step 7 <tensor-parallelism>`). There are three layers:

1. In ``get_submodule`` you build the ``nn.Module`` on the ``meta`` device, materialize it
   with ``to_empty(device=...)``, then call the top-level driver
   ``load_weights(module, source, device=...)`` from ``mstar.model.loader``. The driver
   picks the right safetensors iterator (single file *or* a sharded HF directory) and
   calls ``module.load_weights(weights)``.
2. Your module implements ``load_weights(self, weights)`` and delegates to
   ``load_hf_weights(self, weights, stacked_params=..., name_remapper=...)``, which streams
   the ``(name, tensor)`` pairs and dispatches each to the matching parameter's
   ``weight_loader``.
3. ``stacked_params`` (a list of ``StackedParamRule``) route several checkpoint keys into
   one fused parameter — e.g. HF ``q_proj`` / ``k_proj`` / ``v_proj`` into a single
   ``qkv_proj`` (``LLAMA_STACKED_PARAMS`` is the ready-made Llama set). ``name_remapper``
   rewrites or drops checkpoint keys that don't line up with your parameter paths.

.. code-block:: python

   # in the Model: build on meta, materialize, hand off to the driver
   def _create_llm_submodule(self, device, tp_group=None):
       from mstar.model.loader import load_weights
       with torch.device("meta"):
           language_model = OrpheusForCausalLM(self.config, comm_group=tp_group)
       language_model.to_empty(device=device)
       load_weights(language_model, local_dir, device=device)  # → module.load_weights(...)
       ...

   # in the nn.Module: declare the fused-shard routing and delegate
   def load_weights(self, weights):
       from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights
       return load_hf_weights(self, weights, stacked_params=LLAMA_STACKED_PARAMS)

Each parameter's ``weight_loader`` is also where tensor-parallel sharding happens: when the
module is built with a ``comm_group`` (``tp_world_size > 1``), the loader slices the
incoming tensor along that parameter's shard dim before copying it in. That is why one
``load_weights`` path serves both single-GPU and tensor-parallel runs without change.

.. _step-5:

Step 5 — Continuous batching and CUDA graphs
--------------------------------------------

Continuous batching and CUDA graphs are the two fundamental throughput optimizations a
submodule opts into. They are optional, but for any autoregressive node you almost
certainly want both, and getting their contracts right is the subtlest part of writing a
submodule — hence this dedicated step.

**Continuous batching.** The worker's micro-scheduler groups compatible in-flight
requests into one GPU call. A submodule controls this with three methods:

- ``can_batch(self, batch, model_inputs) -> bool`` — may these requests share a forward
  pass? Returns ``False`` by default (batching off). Override to admit batches (e.g. same
  graph walk, compatible shapes).
- ``forward_batched(self, graph_walk, engine_inputs, **kwargs) -> dict[str, NameToTensorList]``
  — the batched compute, returning per-request (``request_id`` → tensors) outputs. You
  implement this *instead of* relying on the single-request ``forward`` when batched.
- ``max_batch_size(self, graph_walk)`` — optional cap.

``ARNodeSubmodule`` makes ``preprocess`` abstract precisely because AR nodes are expected
to collate a batch; you can still disable batching for a specific node or walk by having
``can_batch`` return ``False`` there.

**CUDA graphs.** A submodule declares its capturable shapes from
``get_cuda_graph_configs(self, device, tp_world_size=1) -> list[CudaGraphConfig]`` (empty
by default → eager). The capture runs ``torch.compile`` first (the per-config ``compile``
flag, default ``True``), then records a CUDA graph it can replay. Two config types live
in ``mstar/engine/cuda_graph_config.py``, and they differ in *which* stage of the
submodule pipeline they freeze:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Config type
     - When to use / what it captures
   * - ``BasicBatchedCudaGraphConfig``
     - **Decode-style** forward passes, where every request in the batch has the *same*
       (typically single-token) length. You pass ``single_request_inputs`` (a one-request
       ``ARNodeInputs``); the runner replicates it to size each captured batch. This config
       fixes the output of ``prepare_inputs``, and ``preprocess`` is invoked on **both**
       capture and replay.
   * - ``FlashInferPackedCudaGraphConfig``
     - **Prefill-style** AR forward passes that operate on **packed / ragged** sequences.
       You pass ``packed_seq_len_to_inputs`` (a bucket → preprocess-output mapping), so this
       config fixes the output of ``preprocess``. The CUDA-graph runner *plans* FlashInfer
       attention at **capture** time; on **replay** the submodule's ``preprocess`` performs
       the per-step attention planning into the captured buffers.

Both share the base ``CudaGraphConfig`` fields: ``capture_graph_walk`` (the walk captured),
``replay_graph_walks`` (walks allowed to replay this capture — lets one capture serve
aliased walks, e.g. ``prefill_audio`` reusing ``prefill_text``), ``capture_batch_sizes``
(which batch sizes to record), ``labels`` (which KV-cache labels are live, e.g.
``["main"]`` or ``["main", "cfg_img"]``), and ``requires_cfg``.

A concrete example — Orpheus's LLM submodule captures a basic-batched ``decode`` graph and
a packed ``prefill`` graph:

.. code-block:: python

   def get_cuda_graph_configs(self, device, tp_world_size=1):
       prefill_packed = {
           n: self._build_prefill_packed(n, device) for n in self.PREFILL_TOKEN_BUCKETS
       }
       return [
           BasicBatchedCudaGraphConfig(
               capture_graph_walk="decode",
               single_request_inputs=ARNodeInputs(
                   input_ids=torch.zeros(1, dtype=torch.long, device=device),
                   input_seq_len=1,
               ),
           ),
           FlashInferPackedCudaGraphConfig(
               capture_graph_walk="prefill",
               replay_graph_walks=["prefill"],
               packed_seq_len_to_inputs=prefill_packed,
               causal_attention=True,
               capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
           ),
       ]

Step 6 — Choose engine types
----------------------------

You almost never write an engine; you assign one of the two
:class:`~mstar.engine.base.EngineType` values per node in ``get_node_engine_types``.
The engine type — not the submodule — decides whether the node gets a managed KV cache:

.. list-table::
   :header-rows: 1
   :widths: 20 52

   * - ``EngineType``
     - Use for
   * - ``KV_CACHE``
     - Any node that needs a persistent, paged KV cache across forward passes —
       autoregressive LLMs (text decode) and LLM-as-denoiser flow loops alike. Runs on
       :class:`~mstar.engine.kv_cache_engine.KVCacheEngine`; pairs with an
       ``ARNodeSubmodule`` and an entry in ``get_kv_cache_config``.
   * - ``STATELESS``
     - Every node *without* cross-step KV state — ViT / VAE / audio encoders and
       decoders, embedding and projection stages, flow-matching combine steps, codec
       (waveform) decoders. Runs on
       :class:`~mstar.engine.stateless_engine.StatelessEngine`.

A model's job is just to label each node; the worker instantiates the right engine and
gives ``KV_CACHE`` nodes their cache from ``get_kv_cache_config``.

Step 7 — Write a config YAML
----------------------------

A config maps nodes to GPU ranks. The key under ``model:`` is your registry key; each
``node_groups`` entry assigns one or more ``node_names`` to ``ranks``, optionally scoped
to specific ``graph_walks`` (this is how prefill/decode disaggregation is expressed).

.. code-block:: yaml

   model: "your_model"
   max_seq_len: 2048
   node_groups:
     - node_names: ["LLM"]
       ranks: [0]

Run it with:

.. code-block:: bash

   mstar-serve --config configs/your_model.yaml --host 0.0.0.0 --port 8000

.. _tensor-parallelism:

Tensor parallelism (sharding)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A node can be sharded across several GPUs by adding ``tp_size`` to its ``node_groups``
entry and listing ``tp_size`` ranks. The runtime splits the group's ranks into
TP groups of that size and builds one ``comm_group`` per shard. For a node to actually
shard (rather than be replicated), its components must be built from the tensor-parallel
modules in ``mstar/model/components/distributed`` — ``ParallelAttention``,
``ParallelGatedMLP``, ``ColumnParallelLinear`` / ``RowParallelLinear``,
``VocabParallelEmbedding``, and friends — whose ``weight_loader`` s (see `Loading
weights`_) slice each sharded parameter automatically. A node whose components do *not*
use those modules is simply **replicated** on every rank (e.g. a tensor-parallel
Qwen3-Omni Talker leaves its code predictor replicated). Once a component is built this
way, going from single-GPU to tensor-parallel needs only the YAML and a small sharding
declaration — no further model-code changes. For example, running the Orpheus LLM tensor-parallel across two
GPUs (``configs/orpheus_tp2.yaml``):

.. code-block:: yaml

   model: "orpheus"
   node_groups:
     - node_names: [LLM]
       ranks: [0, 1]
       tp_size: 2
       graph_walks: [prefill, decode]
     - node_names: [snac_decoder]
       ranks: [0]
       graph_walks: [snac_chunk]

For a node to be eligible for ``tp_size > 1`` it must be declared TP-enabled by the model.
Override ``get_default_sharding_config`` to return a ``ShardingConfig`` naming the
shardable nodes (and any non-default shard dimensions):

.. code-block:: python

   def get_default_sharding_config(self):
       from mstar.distributed.base import ShardingConfig
       return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={})

Two pieces split work across the TP group, and it helps to keep them separate:

- **Weights** are sharded inside the components, by each parameter's ``weight_loader``
  (column/row-parallel linear layers built with the ``comm_group``). This is automatic
  once the module is constructed tensor-parallel — nothing extra in the config.
- **Activations** crossing a node boundary are handled by ``shard_dim`` in the
  ``ShardingConfig`` — a map from an inter-node *edge/signal name* to the dimension along
  which that tensor is split across the group (absent or ``None`` ⇒ the tensor is
  replicated to every rank). You only need entries here for edges whose producer and
  consumer keep the data sharded; the common case (replicated activations) needs nothing.
  ``shard_dim`` can also be supplied per-run under a ``sharding_config`` block in the YAML.

A node group whose ``tp_size > 1`` names a node *not* in ``tp_enabled_nodes`` is rejected
at load time. See ``configs/qwen3omni_thinker_tp2.yaml`` for a multi-node-type example.

Worked example: Orpheus
-----------------------

Orpheus (``mstar/model/orpheus/``) is a compact, complete reference. It is a TTS model:
a Llama 3.2 3B LLM emits audio tokens, and a SNAC decoder turns them into 24 kHz PCM.

Two nodes, two engines — the LLM needs a KV cache, the SNAC decoder doesn't:

.. code-block:: python

   def get_node_engine_types(self) -> dict[str, EngineType]:
       return {
           "LLM": EngineType.KV_CACHE,
           "snac_decoder": EngineType.STATELESS,
       }

Three graph walks — ``prefill`` and a ``decode`` ``Loop`` on the LLM, plus a
``snac_chunk`` node that emits audio to the client:

.. code-block:: python

   snac_chunk = GraphNode(
       name="snac_decoder",
       input_names=["new_token"],
       outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio_chunk",
                          output_modality="audio")],
   )

``process_prompt`` formats ``"{voice}: {text}"``, tokenizes, and wraps the ids in the
model's special start/end tokens, returning ``{"text_inputs": [ids]}``.
``get_submodule`` lazily builds either the Llama LLM submodule (an ``ARNodeSubmodule``)
or the SNAC decoder submodule and caches it. ``postprocess`` returns the audio tensor's
raw bytes for the ``audio`` modality.

Orpheus also demonstrates the **async partition** API (next section): the LLM and SNAC
run as two partitions connected by a streaming edge, so audio is decoded in a sliding
window *while* the LLM is still generating.

Worked example: BAGEL
---------------------

Orpheus is a single pipeline. BAGEL (``mstar/model/bagel/``) is the opposite end of the
spectrum and a better illustration of *why* the graph abstraction exists: it is a
**unified** model that does both image *understanding* (image → text) and image
*generation* (text → image) with the **same** Qwen2 LLM, which also serves as the
denoiser for rectified-flow image generation. Walking through the same steps shows how
the pieces scale up.

**Step 1 — Register.** Already done in ``registry.py``: ``"bagel": BagelModel`` plus an
``HF_MODELS`` entry pointing at ``ByteDance-Seed/BAGEL-7B-MoT``.

**Step 2/6 — Nodes and engine types.** Only the LLM nodes carry a KV cache; everything
else is stateless. The core of the model is four nodes — a ViT encoder, a VAE encoder, the
LLM, and a VAE decoder — plus a few extra nodes for the CFG-parallel image-generation path
described below (``init_latents``, the per-branch ``LLM_cfg_*`` nodes, and ``combine_cfg``).
All are declared up front:

.. code-block:: python

   def get_node_engine_types(self) -> dict[str, EngineType]:
       return {
           "vit_encoder":  EngineType.STATELESS,   # SigLIP2 ViT (understanding)
           "vae_encoder":  EngineType.STATELESS,   # FLUX VAE encode (editing/gen)
           "init_latents": EngineType.STATELESS,   # seed the image-gen latents
           "LLM":          EngineType.KV_CACHE,    # Qwen2: embed + transformer + lm_head + CFG
           "LLM_cfg_text": EngineType.KV_CACHE,    # CFG-parallel branch (text-only cond)
           "LLM_cfg_img":  EngineType.KV_CACHE,    # CFG-parallel branch (image cond)
           "combine_cfg":  EngineType.STATELESS,   # CFG combine + Euler step
           "vae_decoder":  EngineType.STATELESS,   # FLUX VAE decode → pixels
       }

The CFG nodes are always *declared* here, but they are only *used* when the config opts
into CFG-parallel mode (covered under Step 7); a single-GPU config simply never routes to
them.

The ``LLM`` is intentionally a **"fat" node**: it absorbs text embedding, the lm_head,
and the flow projection, because those always live on the same GPU and splitting them
into separate graph nodes would only add IPC overhead. This is a recurring modeling
choice — *make a node as coarse as the colocation boundary allows.*

**Step 3 — Graph walks.** Because understanding and generation are different pipelines,
BAGEL returns *five* walks from ``get_graph_walk_graphs`` instead of two:

.. list-table::
   :header-rows: 1
   :widths: 18 54

   * - Graph walk
     - What it does
   * - ``prefill_text``
     - Embed text tokens and prefill the LLM (causal).
   * - ``prefill_vit``
     - ``vit_encoder`` → LLM: encode an input image for *understanding* (bidirectional).
   * - ``prefill_vae``
     - ``vae_encoder`` → LLM: VAE-encode an image for *editing / generation*.
   * - ``decode``
     - Autoregressive text generation (a ``Loop``, exactly like Orpheus).
   * - ``image_gen``
     - The flow-matching denoising ``Loop`` (LLM does CFG + one Euler step per iter),
       then ``vae_decoder`` turns the final latents into pixels.

The encoder walks are two-node ``Sequential`` chains, and ``image_gen`` is a ``Loop``
followed by the decoder — note how the loop body loops ``latents`` and ``time_index``
back to itself, and the loop's ``outputs`` hand the final latents to ``vae_decoder``:

.. code-block:: python

   prefill_vit = Sequential([
       GraphNode(name="vit_encoder", input_names=["image_inputs"],
                 outputs=[GraphEdge(next_node="LLM", name="img_emb")]),
       GraphNode(name="LLM", input_names=["img_emb"],
                 outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token",
                                    output_modality="text", persist=True)]),
   ])

   image_gen = Sequential([
       Loop(
           section=GraphNode(
               name="LLM",
               input_names=["latents", "time_index"],
               outputs=[GraphEdge(next_node="LLM", name="latents"),
                        GraphEdge(next_node="LLM", name="time_index")],
           ),
           max_iters=self.config.num_timesteps - 1,   # one Euler step per interval
           outputs=[GraphEdge(next_node="vae_decoder", name="latents")],
       ),
       GraphNode(
           name="vae_decoder",
           input_names=["latents"],
           outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="image_output",
                              output_modality="image")],
       ),
   ])

**Declared outputs are conditional.** A node's ``outputs`` list is the set of edges it
*can* emit; what it actually emits on a given step is limited to whatever its submodule produces.
``new_token`` above is the clearest case — the LLM samples a token only when the request
needs text out (the understanding path, and every ``decode`` step). On the
image-generation / editing path the same node still runs and writes the KV cache but
samples no token, so ``new_token`` is not produced. It is in the graph because understanding
requests use it; treat declared edges as the *possible* outputs and let the submodule decide
which actually fire on each step.

**Choosing the walk per request.** Unlike Orpheus, BAGEL's transitions are
*schedule-driven*: the output modality is known up front from the request's
``output_modalities``, so ``get_initial_forward_pass_args`` builds a prefill *schedule*
(walking interleaved text/image inputs) and ``get_partition_forward_pass_args`` steps
through it, then transitions to ``decode`` (text out) or ``image_gen`` (image out). With
``think_mode`` the model decodes a reasoning trace first, *then* the EOS transitions it
into ``image_gen``. This is the same two methods Orpheus implements — they just encode a
richer state machine.

**Step 4 — Submodules.** Each node maps to a ``NodeSubmodule`` in ``bagel/submodules.py``
(``ViTEncoderSubmodule``, ``VAEEncoderSubmodule``, ``LLMSubmodule``,
``VAEDecoderSubmodule``); ``get_submodule`` builds them lazily so a worker that only runs
``vit_encoder`` never allocates the 7B LLM. ``process_prompt`` tokenizes the prompt (and
a system prompt when ``think_mode``); ``postprocess`` branches on modality — ``decode`` →
``utf-8`` text, ``image`` → PNG bytes.

**Step 7 — Config and disaggregation.** This is where BAGEL pays off. The *same* model
code runs on one GPU:

.. code-block:: yaml

   model: "bagel"
   max_seq_len: 32768
   node_groups:
     - {node_names: [vit_encoder], ranks: [0]}
     - {node_names: [vae_encoder, vae_decoder], ranks: [0]}
     - {node_names: [LLM], ranks: [0]}

…or disaggregated across GPUs by pinning the **same** ``LLM`` node to different ranks
*per graph walk* — prefill on GPU 0, decode on GPU 1, image generation on GPU 2:

.. code-block:: yaml

   node_groups:
     - {node_names: [LLM], ranks: [0], graph_walks: [prefill_text, prefill_vit, prefill_vae]}
     - {node_names: [LLM], ranks: [1], graph_walks: [decode]}
     - {node_names: [LLM], ranks: [2], graph_walks: [image_gen]}

BAGEL also supports a **CFG-parallel** mode: when the config names extra
``LLM_cfg_text`` / ``LLM_cfg_img`` nodes (see ``configs/bagel_cfg_parallel.yaml``), the
model swaps in an ``image_gen_cfg`` walk whose loop body is a ``Parallel`` of the three
classifier-free-guidance branches — each on its own GPU — feeding a ``combine_cfg`` node.
The model code detects this purely from the node names present in the config, so the
extra parallelism is opt-in via YAML with no code change. This is the disaggregation
principle taken to its conclusion: **one model, many physical layouts.**

Advanced: async partitions and streaming
----------------------------------------

Single-partition models can ignore this — the defaults in ``Model`` give you one
``"default"`` partition containing all walks. For pipelines where one stage should run
asynchronously while another keeps producing (LLM → vocoder, thinker → talker),
override:

- ``get_partition_topology()`` — declare partitions and the streaming
  ``Connection`` s between them, including a ``chunk_policy_factory`` (e.g.
  ``SlidingWindowChunkPolicy(window=..., stride=...)``).
- ``get_partitions()`` — declare each ``PartitionDefinition`` (its walks, its initial
  walk, and which partitions produce into it).
- Route cross-partition tensors with ``StreamingGraphEdge(next_node=..., name=...,
  target_partition=...)`` instead of a plain ``GraphEdge``.

The consumer partition's ``get_partition_forward_pass_args`` reads
``incoming_connections`` (token counts, ``producer_done``) to decide when to fire.

Checklist
---------

.. code-block:: text

   [ ] mstar/model/<your_model>/config.py        — config dataclass
   [ ] mstar/model/<your_model>/components/       — the nn.Modules + weight loading
   [ ] mstar/model/<your_model>/submodules.py     — NodeSubmodule per node
   [ ] mstar/model/<your_model>/<your_model>_model.py — Model subclass:
         [ ] get_kv_cache_config
         [ ] get_node_engine_types
         [ ] get_graph_walk_graphs
         [ ] process_prompt
         [ ] get_initial_forward_pass_args
         [ ] get_partition_forward_pass_args
         [ ] postprocess
         [ ] get_submodule
   [ ] mstar/model/registry.py                    — add to MODEL_REGISTRY (+ HF_MODELS)
   [ ] configs/<your_model>.yaml                  — node_groups → ranks
   [ ] (optional) async partitions if pipelined

Testing
-------

Validate the graph and worker plumbing before touching real weights — the CPU modular
tests exercise models with submodules in dummy mode (``get_submodule`` returning ``None``):

.. code-block:: bash

   ruff check .
   pytest test/modular/                  # CPU graph/worker tests
   pytest test/integration/              # requires GPU + weights
   mstar-serve --config configs/your_model.yaml --port 8000

Then send a ``POST /generate`` request and confirm the streamed output. Modeling a new
family on the closest existing one (Orpheus for a streaming LLM + codec, BAGEL for
multi-engine understanding + generation, Qwen3-Omni for full omni-modal) is by far the
fastest path.
