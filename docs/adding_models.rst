Adding a New Model
==================

This page describes everything you must implement to add a new model to ``mstar``. When
you finish, the conductor can schedule your model, workers can execute it on GPU, and you
can launch it with ``mstar-serve``.

Overview
--------

A model in ``mstar`` has several separate responsibilities:

- **The** ``Model`` **class** (``mstar/model/base.py``) is the interface that the rest of
  the system calls. It tokenizes prompts, declares the computation graph, declares the
  resources each node needs, builds forward-pass arguments, and post-processes outputs.
  It contains no GPU compute.
- **Submodules** (``NodeSubmodule`` in ``mstar/model/submodule_base.py``) are the
  ``torch.nn.Module`` s that perform the compute. Each graph node maps to one submodule.
- **Resources** (``mstar/engine/resources/``) hold the state that a node's compute uses:
  a paged KV cache, the attention planned over that cache, position embeddings, and a
  sampler. The model declares its resources in ``get_node_resources()``. The engine then
  builds one object per declaration and binds it into the submodule and its layers. A
  node that declares no resources was called a "stateless" node in earlier versions.
- **The engine** (``mstar/engine/engine.py``) is a single class that runs every node. It
  compiles forwards, captures CUDA graphs, batches requests, and runs each step's
  resource lifecycle (admit, plan, forward, commit). You never write an engine. Earlier
  versions required you to choose an engine type per node. That is no longer true. A
  node's capabilities now follow from the resources it declares.
- **The graph** (``mstar/graph/base.py``) declares what runs in what order: nodes, edges
  between them, and loops. All of a model's work forms one large computation graph. Each
  named "graph walk" (for example ``prefill`` or ``decode``) is one path through that
  graph. In code, however, you declare each walk as a separate standalone graph. A node
  that appears in several walks is referenced by name in each walk. The submodule behind
  that name, and its resources, are shared across all of them.
- **The config YAML** (``configs/``) maps graph nodes to physical GPU ranks through
  ``node_groups``. Disaggregation is configured here. The same model code runs on one GPU
  or on many, and only the config changes.

One vocabulary note for this page. A **tensor bundle** routed between nodes is a
``NameToTensorList``, which is ``dict[str, list[torch.Tensor]]``
(``mstar/communication/tensors.py``). It maps an edge name to a list of tensors. The list
usually has length 1.

The diagram below shows the request flow at the conductor and model level. Its
granularity is one graph walk. The conductor is notified when a walk completes, and only
then asks the model what to do next. Everything that happens inside a walk is described
in later steps and is omitted here::

   process_prompt()              # text/media -> initial tensors
        │
        ▼
   get_initial_forward_pass_args()   # seed the first graph walk (e.g. prefill)
        │
        ▼   (conductor walks the graph, the engine runs each node)
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

This is the only wiring step. There is no plugin scan. The registry import is the single
source of truth.

Step 2 — Implement the ``Model`` class
--------------------------------------

Subclass :class:`mstar.model.base.Model` and implement its abstract methods. The
constructor receives ``model_path_hf`` (from ``HF_MODELS``) and any ``**kwargs``. It
normally loads the tokenizer and stores a config dataclass. Do not load weights in the
constructor. Load them in ``get_submodule``, so that the conductor process never
allocates GPU memory.

You must implement these abstract methods:

``get_node_resources(self) -> list[NodeResourceSpec]``
   Declare every resource the engine builds for this model, and which nodes share each
   one. This is the largest single part of a model, so it has its own section. See
   `Step 2a — Declare your resources`_ below.

``get_graph_walk_graphs(self) -> dict[str, GraphSection]``
   Return ``{walk_name: graph}``. See `Step 3 — Declare the computation graph`_.

``process_prompt(self, prompt, input_modalities, output_modalities, tensors=None, **kwargs) -> NameToTensorList``
   Tokenize the prompt and produce the initial request tensors, for example
   ``{"text_inputs": [token_ids]}``. This method runs in the API-server data worker,
   after raw media tensors are loaded. It can therefore read ``tensors`` (for example
   ``image_inputs``, ``audio_inputs`` or ``video_inputs``) to compute derived tensors
   such as ``pixel_values``. The returned dict is merged into the request's tensors.

``get_initial_forward_pass_args(self, partition_name, input_modalities, output_modalities, input_signals, model_kwargs=None) -> ForwardPassArgs``
   Build the first :class:`mstar.model.base.ForwardPassArgs` for a partition. It names
   the graph walk to start on and the input edges that feed it.

``get_partition_forward_pass_args(self, partition_name, partition_metadata, persist_signals, incoming_connections=None) -> ForwardPassArgs``
   The conductor calls this after each graph walk completes. It returns the next walk,
   the inputs for that walk, and whether the request is finished
   (``request_done=True``). A simple prefill-decode model sets ``is_prefill`` to false
   once, then repeats the decode walk until EOS.

``postprocess(self, output, modality) -> bytes``
   Encode a finished output tensor to bytes for the client: ``utf-8`` for text, PNG for
   images, raw PCM for audio, and so on.

``get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None) -> NodeSubmodule | None``
   Build and return the ``NodeSubmodule`` for ``node_name``, and cache the result. Load
   weights here, on ``device``. Return ``None`` for dummy mode.

   ``tp_group`` and ``sp_group`` are the node's tensor-parallel and sequence-parallel
   communicators when the node is sharded. Pass ``tp_group`` to the parallel-linear
   constructors. See :ref:`Step 6 <tensor-parallelism>`.

   ``autocast_dtype`` is the dtype in which the node's parameters must be allocated. If
   you build the module on the ``meta`` device, cast it to this dtype before calling
   ``to_empty(device)``. Casting after ``to_empty`` allocates every parameter in float32
   first, which doubles the peak VRAM during loading.

   See `Step 4 — Implement the submodules`_.

.. note::

   ``model_kwargs`` reaches your model from clients through the OpenAI routes'
   ``extra_body`` passthrough. The Dynamo bridge
   (``mstar/integrations/dynamo/bridges.py``) strips OpenAI-standard fields no
   model consumes (``_STRIP_KEYS``) before that passthrough runs. If your model
   starts reading such a field from ``model_kwargs`` — the way ``ignore_eos`` is
   read — also delete the key from ``_STRIP_KEYS``, or requests arriving through
   the Dynamo frontend will silently lose it.

The following methods have defaults and are optional to override:
``get_request_resource_configs`` (described below), ``get_sampling_config``,
``get_max_output_tokens``, ``get_autocast_dtype``, ``load_image``, ``load_audio``,
``load_video``, and the partition methods described at the end of this page.
``Model.nodes`` is a read-only property. It returns the sorted set of node names that
appear in any graph walk.

Per-request resource parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A resource spec is fixed at load time and is shared by every request. Parameters that
differ per request, such as sampling parameters and the set of cache labels a request
uses, come from a separate method::

   get_request_resource_configs(
       self, partition_fwd_args: dict[str, ForwardPassArgs],
       model_kwargs: dict | None = None,
   ) -> dict[str, ResourceReqConfig]

The returned dict is keyed by ``resource_key``. The conductor calls this method once per
request, and the engine passes each config to its resource when the request is ingested.
There are two ``ResourceReqConfig`` subclasses:

- ``SamplingReqConfig`` holds ``temperature``, ``top_k``, ``top_p``,
  ``repetition_penalty`` and ``ignore_eos``. The conductor fills in the per-request seed.
- ``KVReqConfig`` holds ``needed_labels``, ``needed_labels_per_node`` and
  ``needed_labels_per_node_walk``. These name the cache streams that the request will
  actually read. In a PD-disaggregated deployment, a KV transfer then copies only those
  streams.

Orpheus shows the simple case. It has one sampler, and reads the parameters from
``model_kwargs``:

.. code-block:: python

   def get_request_resource_configs(self, partition_fwd_args, model_kwargs=None):
       model_kwargs = model_kwargs or {}
       keys = ["temperature", "top_p", "repetition_penalty", "ignore_eos"]
       return {
           SAMPLER: SamplingReqConfig(
               **{k: model_kwargs.get(k, getattr(self.config, k)) for k in keys}
           )
       }

BAGEL (``bagel_model.py``) returns both config types. A BAGEL request uses
classifier-free guidance or does not, and that choice determines which cache labels the
request reads.

``get_sampling_config(node_name, model_kwargs)`` still exists as a helper for assembling
sampling parameters. The engine no longer reads it directly. Pass its result into a
``SamplingReqConfig`` here.

.. warning::

   Never pass sampling parameters into a captured forward as Python scalars. Their values
   are recorded into the captured kernel launch, and every replay then reuses the values
   from capture time. No error is raised. Parameters that reach the sampler resource
   through ``SamplingReqConfig`` are stored in buffers whose addresses do not change
   across replays, so they stay per-request and remain safe under CUDA graphs.

.. _Step 2a — Declare your resources:

Step 2a — Declare your resources
--------------------------------

``get_node_resources`` returns a flat list of ``NodeResourceSpec`` objects. Every spec has
three common fields:

- ``resource_key`` is the name of this resource. Three places use this name, and all three
  must agree: the layers bind to it, ``declare_step`` uses it as the key of its
  per-resource steps, and a deployment YAML tunes it under ``resources:`` (see
  :ref:`Step 6 <config-yaml>`). Existing models define these names as module-level
  constants in the model's ``config.py``, for example ``KV_CACHE = "kv_cache"`` and
  ``ATTN = "attn"``, so that all three places read one definition.
- ``nodes`` is the set of graph-node names that share this resource. When two nodes name
  the same resource, they share one object. For example, BAGEL's ``LLM`` node and its CFG
  branch nodes share one KV pool. A node that no spec names receives no resources. Such a
  node was called "stateless" in earlier versions.
- ``depends_on()`` returns the keys of other specs that this spec is built against.
  ``AttentionSpec`` and ``PositionSpec`` depend on the ``kv_cache`` key they name. If a
  spec names a dependency that the model does not declare, loading fails immediately,
  rather than during a forward pass.

The spec types are:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Spec
     - What it builds
   * - ``KVSpec(config=KVConfig(...))``
     - A paged KV cache. ``KVConfig`` holds ``num_layers``, ``num_kv_heads``,
       ``head_dim``, ``max_seq_len`` and ``num_qo_heads``. It also holds three fields that
       a deployment can tune: ``max_num_pages``, ``page_size`` and ``cpu_offload_pages``
       (the number of pinned host pages used for offload; 0 disables offload).
   * - ``AttentionSpec(config=AttentionConfig(kv_cache=...))``
     - Self-attention planned over the named cache. ``backend`` selects
       ``AttnBackend.FLASHINFER`` (the default) or ``AttnBackend.DENSE``.
       ``flashinfer_backend`` selects a kernel generation: ``"auto"``, ``"fa2"`` or
       ``"fa3"``.
   * - ``CrossAttentionSpec(config=CrossAttentionConfig(...))``
     - Attention over a context that is written once and never extended. See
       `Cross-attention (encoder-decoder models)`_.
   * - ``PositionSpec(config=PositionConfig(kv_cache=...))``
     - Position tracking and RoPE. ``scheme`` is ``PosScheme.SEQUENTIAL`` or
       ``PosScheme.BLOCK``. The RoPE parameters are set here: ``rope_theta``,
       ``rope_scale``, and the Llama-3.1 parameters ``low_freq_factor``,
       ``high_freq_factor`` and ``old_context_len``. A model with learned position
       embeddings also declares this spec, because it needs the position counter. Such a
       model never applies RoPE.
   * - ``SamplerSpec(vocab_size=..., enable_repetion_penalty=...)``
     - A sampler with its own per-request parameter buffers, philox stream, and optional
       seen-token mask. ``enable_repetion_penalty`` declares a capability, not an
       intention. It selects which kernel variant is recorded into the captured graph.
       Whether the penalty runs on a given step is decided from the
       ``repetition_penalty`` values of the resident requests.

Orpheus declares four specs for its one autoregressive node. Its ``snac_decoder`` node
appears in no spec, so it receives no resources:

.. code-block:: python

   # mstar/model/orpheus/config.py
   KV_CACHE, ATTN, SAMPLER, ROPE = "kv_cache", "attn", "sampler", "rope"

   # mstar/model/orpheus/orpheus_model.py
   def get_node_resources(self) -> list[NodeResourceSpec]:
       kv_config = KVConfig(
           num_layers=self.config.num_hidden_layers,
           num_kv_heads=self.config.num_key_value_heads,
           head_dim=self.config.head_dim,
           max_seq_len=self.config.max_position_embeddings,
           num_qo_heads=self.config.num_attention_heads,
       )
       return [
           KVSpec(resource_key=KV_CACHE, nodes={"LLM"}, config=kv_config),
           AttentionSpec(
               resource_key=ATTN, nodes={"LLM"},
               config=AttentionConfig(kv_cache=KV_CACHE),
           ),
           SamplerSpec(
               resource_key=SAMPLER, nodes={"LLM"},
               vocab_size=self.config.vocab_size,
               enable_repetion_penalty=True,
           ),
           PositionSpec(
               resource_key=ROPE, nodes={"LLM"},
               config=PositionConfig(
                   kv_cache=KV_CACHE,
                   rope_theta=self.config.rope_theta,
                   rope_scale=self.config.rope_scaling["factor"],
                   low_freq_factor=self.config.rope_scaling["low_freq_factor"],
                   high_freq_factor=self.config.rope_scaling["high_freq_factor"],
                   old_context_len=self.config.rope_scaling["original_max_position_embeddings"],
               ),
           ),
       ]

**Which resources does a node need?**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Node
     - Declare
   * - Self-attending LLM (text decode, or an LLM used as a denoiser in a flow loop)
     - ``KVSpec``, ``AttentionSpec`` and ``PositionSpec``. Add a ``SamplerSpec`` if the
       node samples.
   * - Decoder of an encoder-decoder model
     - The three or four specs above, plus a second ``KVSpec`` for the encoder context
       and a ``CrossAttentionSpec`` over that second cache.
   * - A node that samples with two different sets of parameters
     - Two ``SamplerSpec`` objects that both name the node, under different keys.
   * - ViT, VAE or audio encoder, codec decoder, projection stage, combine stage
     - Nothing. Declare no spec that names the node.

Two samplers on one node
~~~~~~~~~~~~~~~~~~~~~~~~

If a node's forward samples more than once per step with different parameters, declare
one ``SamplerSpec`` per parameter set. Both specs name the same node. Qwen3-Omni's Talker
node does this. The Talker LLM samples codec group 0. The CodePredictor samples groups 1
to N-1, using its own vocabulary size and no repetition penalty:

.. code-block:: python

   SamplerSpec(resource_key=TALKER_SAMPLER, nodes={"Talker"},
               vocab_size=self.config.talker_text.vocab_size,
               enable_repetion_penalty=True),
   SamplerSpec(resource_key=CODE_PRED_SAMPLER, nodes={"Talker"},
               vocab_size=self.config.code_predictor.vocab_size,
               enable_repetion_penalty=False),

The forward looks up each sampler by key and calls ``.sample()`` on it::

   layer0 = engine_inputs.resources[TALKER_SAMPLER].sample(request_ids, logits)
   code   = engine_inputs.resources[CODE_PRED_SAMPLER].sample(request_ids, logits)

Each resource owns its own buffers. The two parameter sets are therefore independent,
per-request, and safe under CUDA graphs.

.. _Cross-attention (encoder-decoder models):

Cross-attention (encoder-decoder models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A decoder that attends to a fixed encoder context declares two KV caches: its own
self-attention cache, and a second cache that holds the context.

Use two separate resources, not two labels on one cache. The self-attention resource
plans one wrapper per label of the cache it names. If the context shared that cache, the
resource would re-plan the context on every decode step. The context is large and never
changes, so this work would be wasted.

``CrossAttentionConfig`` names both sides:

- ``kv_cache`` is the KV resource that holds the context.
- ``query_kv_cache`` is the KV resource that drives the queries. Its plan defines how the
  step packs its queries. Set it to ``None`` if the query side caches nothing across
  steps. The packing is then taken from the cross-attention step's own segments, with one
  entry per segment, in declaration order.
- ``context_label`` is the label under which the context is written. The default is
  ``"context"``.

Whisper (``mstar/model/whisper/``) is the reference implementation:

.. code-block:: python

   KVSpec(resource_key=KV_CACHE, nodes={"decoder"}, config=kv_config),
   AttentionSpec(resource_key=ATTN, nodes={"decoder"},
                 config=AttentionConfig(kv_cache=KV_CACHE)),
   KVSpec(resource_key=CROSS_KV_CACHE, nodes={"decoder"}, config=context_kv_config),
   CrossAttentionSpec(
       resource_key=CROSS_ATTN, nodes={"decoder"},
       config=CrossAttentionConfig(
           kv_cache=CROSS_KV_CACHE,
           query_kv_cache=KV_CACHE,
           context_label=CONTEXT_LABEL,
       ),
   ),

There is no special API for writing the context. You express it in the step declaration.
The prefill step declares a non-zero span on the context cache's ``KVStep``. Every later
step declares a span of 0. A span of 0 reads the stream without extending it.

The encoder K and V tensors are written once per request by the decoder's own code, in
``whisper/components/decoder.py:write_cross_kv``. That method iterates over the layers and
writes into the context label::

   for layer_idx, layer in enumerate(self.layers):
       k, v = layer.encoder_attn.compute_kv(encoder_states)
       layer.encoder_attn.context_kv.set_default_layer_idx(layer_idx)
       layer.encoder_attn.context_kv.write_kv(k, v, label=CONTEXT_LABEL)

A model with several context sources, such as "audio" and "image", declares one
``CrossAttentionSpec`` per source. Each source is a separate resource, and the source name
is also the key that the layer binds to.

Step 3 — Declare the computation graph
--------------------------------------

``get_graph_walk_graphs`` returns one graph per walk. The primitives are defined in
``mstar/graph/base.py``:

- ``GraphNode(name, input_names, outputs)`` is one unit of compute. ``name`` is the node
  name that ``get_submodule`` receives, and the name that resource specs put in their
  ``nodes`` set. ``Model.nodes`` returns all such names across all walks. ``input_names``
  lists the tensor names that must be present before the node can run. ``outputs`` is a
  list of ``GraphEdge``.
- ``GraphEdge(next_node, name, ...)`` routes an output tensor named ``name`` to
  ``next_node``. Two flags are important. ``persist=True`` keeps the tensor available for
  later steps and walks, which is how a generated token is carried from ``prefill`` into
  the ``decode`` loop. ``output_modality``, combined with ``next_node=EMIT_TO_CLIENT``,
  streams the tensor to the client. Its value is one of ``"text"``, ``"image"``,
  ``"audio"``, ``"video"`` or ``"action"``. The special destinations
  ``EMIT_TO_CLIENT`` and ``EMPTY_DESTINATION`` are defined in
  ``mstar/graph/special_destinations.py``. Note that no edge flag stops a ``decode``
  loop. A loop stops when a submodule's ``check_stop`` registers a stop signal against
  that ``Loop``, for example on EOS. See Step 4.
- ``Sequential([...])`` and ``Parallel([...])`` compose subgraphs in order or
  concurrently.
- ``Loop(name, section, max_iters, outputs)`` is a subgraph that iterates. Its body feeds
  its own outputs back as the inputs of the next iteration. It runs at most ``max_iters``
  times, and can stop earlier. Give the loop a ``name`` so that a submodule's
  ``check_stop`` can register a stop signal against it. This is the usual ``decode`` loop.

A minimal text generator has two walks: a ``prefill`` node that runs once, and a
``decode`` ``Loop`` whose body feeds its own output back as the next input.

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

Each node name maps to a :class:`mstar.model.submodule_base.NodeSubmodule`, which is a
``torch.nn.Module``. Autoregressive nodes use the ``ARNodeSubmodule`` subclass. The
methods are:

``prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs``
   Convert the routed ``NameToTensorList`` into a typed ``NodeInputs``, or into an
   ``ARNodeInputs`` with ``input_ids`` or ``input_embeds``. This method runs once per
   request. Do only cheap host-side work here: shape and length bookkeeping, building
   position metadata, slicing token id lists. Do not launch GPU compute here. GPU compute
   belongs in ``forward``. The engine may call ``prepare_inputs`` on a different thread
   from the GPU thread, and much earlier than execution.

   Always set ``input_seq_len``. This field is on the base ``NodeInputs`` class and has
   two important uses. The engine sums it across the batch to select a CUDA-graph capture
   bucket and to compute padding sizes, and ``declare_step`` normally computes its spans
   from it. Leave it at 0 only when the submodule's inputs are not sequence-shaped. If
   ``declare_step`` needs a value that ``forward`` does not use, put that value in
   ``resource_step_info``.

``declare_step(self, graph_walk, request_ids, inputs, slot_lease=None, piecewise_leases=None, **kwargs) -> SubmoduleStep | None``
   Declare what this batch's step does to the node's resources. See
   `Step 4a — Declare the step`_ below.

``preprocess(self, graph_walk, engine_inputs, inputs) -> dict``
   Collate a list of ``NodeInputs`` into the keyword arguments that ``forward`` expects.
   The default implementation handles batch size 1. Override it to support batching.
   ``ARNodeSubmodule`` declares this method abstract, because autoregressive submodules
   normally support continuous batching. You can still disable batching for one node, or
   for specific graph walks. See :ref:`Step 5 <step-5>`.

``forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList``
   The tensor-to-tensor computation. The keys of the returned dict are the edge names
   that the graph routes downstream. Read resources from
   ``engine_inputs.resources[key]``. See `Reaching your resources`_.

   The engine applies ``torch.compile`` to both ``forward`` and ``forward_batched``, for
   every submodule. It also captures CUDA graphs for them when you declare capture
   configs (see :ref:`Step 5 <step-5>`). To disable compilation for a submodule, set the
   class attribute ``disable_torch_compile = True``. Keep the compiled paths
   compile-friendly. If a helper must not be traced, because it uses data-dependent
   Python control flow or forces a host synchronization, exclude it explicitly with
   ``@torch.compiler.disable``.

   Some submodules must run in their own parameter dtype. One example is an fp32 vocoder
   that is numerically sensitive. Such a submodule sets ``disable_autocast = True``. The
   engine then does not cast its parameters to the autocast dtype, does not wrap its
   forward in autocast, and explicitly disables any enclosing autocast. These two class
   attributes replace the removed ``get_stateless_flavor`` method.

``postprocess(...)`` (optional)
   Metadata-only fixups that run on the GPU thread. This method must not read tensor
   values. Do not call ``.item()``, ``.cpu()`` or ``.tolist()`` here. This is a
   performance requirement, not a correctness requirement. Reading a value forces a host
   synchronization, which stalls the GPU thread and loses the worker's asynchronous
   scheduling overlap. Use this method only to rename outputs for routing. Put decisions
   that depend on tensor values in ``check_stop``.

``check_stop(...) -> set[str]`` (optional)
   Runs off the GPU thread, and may read tensor values. Return the names of the ``Loop``
   objects to stop, for example after seeing the EOS token. This is how a decode loop
   terminates.

``cleanup_request(self, request_id)`` (optional)
   Free per-request state held inside the submodule when a request finishes: buffers,
   per-request caches, counters. See Qwen3-Omni's ``Code2WavSubmodule`` for an example.

``filter_batched_output(...)`` and ``unpack_packed_outputs(...)`` (optional)
   Output fixups that run after the forward. A captured forward always emits the same set
   of keys, because the graph shape is fixed. Use ``filter_batched_output`` to drop keys
   that a particular request must not receive. Use ``unpack_packed_outputs`` to slice a
   packed ``(total_tokens, ...)`` tensor into per-request entries, when the slice
   boundaries depend on the real sequence lengths and therefore cannot be computed inside
   the captured region. Both methods are defined on ``NodeSubmodule``, so any node can use
   them.

Two more methods control batching and CUDA graphs: ``can_batch`` with
``forward_batched``, and ``get_cuda_graph_configs``. They are described in Step 5.

.. _Step 4a — Declare the step:

Step 4a — Declare the step
~~~~~~~~~~~~~~~~~~~~~~~~~~

``declare_step`` tells the engine what one batched step does to the node's resources:
which cache streams it touches, how much each stream grows, what the attention plan is
based on, which streams fork, and what is committed after the forward completes.

The runner calls the declaration and the resource lifecycle around your forward::

   declare_step()  →  admit  →  plan  →  preprocess  →  forward  →  commit

The runner owns this lifecycle. A submodule that declares a step must therefore contain
no plan calls and no advance calls of its own.

Declaring the work also lets the worker handle a cache that is full. If a step cannot be
admitted, the engine returns an admit failure. The scheduler can then apply backpressure
and evict other requests. Without a declaration, the same situation raises a
``RuntimeError`` inside the forward.

Returning ``None`` means that the submodule manages its own resources. This is the legacy
path, kept for submodules that have not been migrated.

A ``SubmoduleStep`` contains a list of ``Segment`` objects and one ``ResourceStep`` per
resource key:

- ``Segment(request_id, label, span)`` is one request's contribution to one cache stream
  in this step. ``span`` is the number of tokens by which the stream grows. A ``span`` of
  0 reads the stream without extending it: admission reserves nothing, and commit does
  nothing. A request contributes one segment per label that is active for it.
- The ``segments`` argument of ``SubmoduleStep`` is a default value. Any ``ResourceStep``
  that does not set its own ``segments`` uses this list.
- ``KVStep(commit=, combined_labels=, pre_forks=, post_forks=)``. See below.
- ``AttentionStep(causal=)`` describes the attention plan.
- ``PositionStep(pos_ids=, advance=)``. With ``pos_ids=None``, positions are derived from
  the stream counters. Pass explicit ids, keyed by plan label, when the model computes
  positions itself.
- ``SamplerStep(apply_penalty=, prefill_tracked_tokens=)``. ``prefill_tracked_tokens``
  initializes the repetition-penalty mask with the prompt tokens. Only the prefill step
  passes them, because the sampler tracks every token it samples afterwards.

The Orpheus declaration is the simplest form. It uses one label, computes spans directly
from the prepared inputs, and steps all four resources together:

.. code-block:: python

   def declare_step(self, graph_walk, request_ids, inputs,
                    slot_lease=None, piecewise_leases=None, **kwargs):
       prefill_tokens = {}
       if graph_walk == "prefill":
           prefill_tokens = {
               rid: inp.input_ids
               for rid, inp in zip(request_ids, inputs, strict=True)
           }
       return SubmoduleStep(
           segments=[
               Segment(request_id=rid, label="main", span=inp.input_seq_len)
               for rid, inp in zip(request_ids, inputs, strict=True)
           ],
           steps={
               KV_CACHE: KVStep(),
               ATTN: AttentionStep(causal=True),
               SAMPLER: SamplerStep(apply_penalty=True,
                                    prefill_tracked_tokens=prefill_tokens),
               ROPE: PositionStep(),
           },
       )

**Padding rows.** Under a captured graph, the batch is padded to the shape of the capture
bucket, and ``request_ids`` also contains the ids of the padding rows. Declare segments
for those rows in the same way as for real rows. The ``zip(..., strict=True)`` in the
example above already does this.

**The two leases.** ``slot_lease`` is the CUDA-graph slot on which this step will replay.
It is ``None`` for an eager step. If a submodule's declaration differs between the
captured case and the eager case, it must check the lease, not its own capture key. The
capture key only says that the batch could be captured. The lease says that the batch was
captured. For example, Cosmos3 packs both guidance branches into a single plan for the
captured shape, and uses the dense backend otherwise.

``piecewise_leases`` names the inner regions of this node that hold their own slot for
this step. Such a region declares, plans and commits its own work. Any resource that the
region owns must therefore be excluded from the outer declaration. See
`Piecewise CUDA graphs (capturing an inner loop)`_.

**Advanced ``KVStep`` fields.** These fields cover cases that the engine previously
handled as special cases. BAGEL uses all of them, in
``mstar/model/bagel/submodules.py``:

- ``commit=False``: the step reads and plans, but its writes do not become resident yet.
- ``combined_labels={("main", "cfg_img"): "cfg_batched"}``: pack several labels into a
  single plan. Batched classifier-free guidance uses this to run two branches through one
  attention call. Positions take their packing from the KV plan, so the grouping is
  declared once here, and ``PositionStep`` keys its ``pos_ids`` by the combined label.
- ``pre_forks`` and ``post_forks``: a value such as ``(("main", "cfg_text"),)`` forks one
  cache stream from another. ``pre_forks`` forks before any planning or writing, for a
  branch that must keep the context from before this step. ``post_forks`` forks at commit
  time, for a branch that must include this step's
  writes.

**Testing.** ``declare_step`` performs only host-side bookkeeping, so you can unit-test it
on CPU without model weights. ``test/modular/vjepa2/fake_resources.py`` shows how to stub
the resources, and ``test/modular/test_resource_runner.py`` tests the runner's lifecycle
directly.

.. _Reaching your resources:

Reaching your resources
~~~~~~~~~~~~~~~~~~~~~~~

Two places need access to resources, and each has its own mechanism.

**In the forward**, read ``engine_inputs.resources``, keyed by ``resource_key``:

.. code-block:: python

   def _forward(self, graph_walk, engine_inputs, text_inputs):
       sampler: SamplerResource = engine_inputs.resources[SAMPLER]
       attn: AttentionManager = engine_inputs.resources[ATTN]
       hidden = self.language_model(self.embed_tokens(text_inputs), label="main")
       if graph_walk == "prefill":
           hidden = attn.select_last_hidden(hidden)
       return sampler.sample(engine_inputs.request_ids, logits=self.lm_head(hidden))

``ModelInputsFromEngine`` carries four other useful fields:

- ``step`` is this step's ``SubmoduleStep``. A forward that must match its own declaration
  reads it here instead of computing the same information again.
- ``captured`` is true when this forward runs under a CUDA-graph capture or replay. This
  field replaces the old ``cache_manager.is_captured``. It is useful when ``preprocess``
  packs its inputs differently for the fixed capture shape.
- ``per_request_states`` is a ``Mapping`` that is resolved on first read.
- ``piecewise_runners`` holds the piecewise CUDA-graph runners, keyed by region name.

**In a layer**, resources are resolved once at load time. The engine calls
``submodule.bind_node_resources(resources)``. That method stores the resources, then
iterates over ``self.modules()`` and calls ``bind_resources(resources)`` on every module
that defines it. A layer names the keys it needs in its constructor and resolves them in
``bind_resources``. If a layer names a key that its node does not have, the error occurs
at bind time rather than during a forward pass:

.. code-block:: python

   ParallelAttention(..., attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE)

Inside the layer, ``AttentionCallable`` (``mstar/engine/resources/convenience.py``) wraps
the KV write and the attention call together. The label and the layer index are cursors
stored on the resources. They are not passed as arguments on each call. The code that
drives the layer stack binds the label once, then advances the index for each layer:

.. code-block:: python

   def forward(self, query_sequence: torch.Tensor, *, label: str) -> torch.Tensor:
       self.layers[0].self_attn.attend.bind_step(label)
       for layer_idx, decoder_layer in enumerate(self.layers):
           decoder_layer.self_attn.attend.set_layer_idx(layer_idx)
           query_sequence = decoder_layer(hidden_states=query_sequence)
       return self.norm(query_sequence)   # the runner advances the cache, from the step

Cursors are used instead of arguments because passing the layer index as an argument makes
inductor specialize on that integer. The frame is then retraced once per layer.

.. warning::

   Where the callable is passed into a traced function, create one ``AttentionCallable``
   per transformer, not one per layer. Dynamo specializes ``ulysses_attention`` on the
   identity of its ``run_attention`` argument. A per-layer callable therefore retraces
   that frame once per layer and exceeds the recompile limit. The shared ``Attention``
   layer creates one callable per layer. That is safe only because the callable is never
   passed into a traced function.

Loading weights
~~~~~~~~~~~~~~~

``get_submodule`` loads a node's parameters. Weight loading is standardized through
``mstar/model/loader/``. Use it instead of a custom ``load_state_dict`` call. Only then
can the same code load both a single-GPU checkpoint and a tensor-parallel shard. See
:ref:`Step 6 <tensor-parallelism>`. There are three layers:

1. In ``get_submodule``, build the ``nn.Module`` on the ``meta`` device, materialize it
   with ``to_empty(device=...)``, then call ``load_weights(module, source, device=...)``
   from ``mstar.model.loader``. This function selects the correct safetensors iterator,
   for a single file or for a sharded Hugging Face directory, and then calls
   ``module.load_weights(weights)``.
2. Your module implements ``load_weights(self, weights)`` and delegates to
   ``load_hf_weights(self, weights, stacked_params=..., name_remapper=...)``. That
   function streams the ``(name, tensor)`` pairs and dispatches each pair to the
   ``weight_loader`` of the matching parameter.
3. ``stacked_params`` is a list of ``StackedParamRule``. Each rule routes several
   checkpoint keys into one fused parameter, for example the Hugging Face ``q_proj``,
   ``k_proj`` and ``v_proj`` keys into a single ``qkv_proj`` parameter.
   ``LLAMA_STACKED_PARAMS`` is a predefined rule set for Llama models. ``name_remapper``
   rewrites or drops checkpoint keys that do not match your parameter paths.

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

Each parameter's ``weight_loader`` also performs tensor-parallel sharding. When the module
is built with a ``comm_group`` and ``tp_world_size > 1``, the loader slices the incoming
tensor along that parameter's shard dimension before copying it. For this reason, one
``load_weights`` path serves both single-GPU runs and tensor-parallel runs without change.

.. _step-5:

Step 5 — Continuous batching and CUDA graphs
--------------------------------------------

Continuous batching and CUDA graphs are the two main throughput optimizations. Both are
optional. For an autoregressive node, you normally want both. These two mechanisms have
the most detailed rules in the submodule interface, so they are described here separately.

**Continuous batching.** The worker's micro-scheduler groups compatible in-flight requests
into one GPU call. A submodule controls this behavior with three methods:

- ``can_batch(self, batch: ExecutingBatch, model_inputs) -> bool`` returns whether these
  requests can share one forward pass. The default is ``False``, which disables batching.
  Override it to accept batches, for example when the requests use the same graph walk and
  have compatible shapes.
- ``forward_batched(self, graph_walk, engine_inputs, **kwargs) -> dict[str, NameToTensorList]``
  performs the batched compute. It returns per-request outputs, keyed by ``request_id``.
  When a batch runs, the engine calls this method instead of the single-request
  ``forward``.
- ``max_batch_size(self, graph_walk)`` sets an optional upper limit.

``ARNodeSubmodule`` declares ``preprocess`` abstract because autoregressive nodes are
expected to collate a batch. You can still disable batching for one node, or for specific
walks, by returning ``False`` from ``can_batch`` in those cases.

**CUDA graphs.** A submodule declares the shapes it can capture in
``get_cuda_graph_configs(self, device, tp_world_size=1) -> list[CudaGraphConfig]``. The
default is an empty list, which means eager execution. For each config, the engine first
runs ``torch.compile``, controlled by the config's ``compile`` flag, which defaults to
``True``. It then records a CUDA graph and replays it. Two config types are defined in
``mstar/engine/cuda_graph_config.py``. They differ in which stage of the submodule
pipeline they freeze:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Config type
     - Use and captured stage
   * - ``BatchedCudaGraphConfig``
     - Decode-style forward passes, in which every request in the batch has the same
       length. That length is usually one token. Pass ``single_request_inputs``, which is
       a ``NodeInputs`` for one request. The runner clones it to build each captured
       batch. This config fixes the output of ``prepare_inputs``, and ``preprocess`` runs
       on both capture and replay.
   * - ``PackedCudaGraphConfig``
     - Prefill-style forward passes that operate on packed, variable-length sequences.
       Pass ``capture_token_lengths``, which lists the token-count buckets to record, and
       ``make_node_input(n)``, a factory that builds a ``NodeInputs`` for one request of
       length ``n``. The runner partitions each bucket across the batch. Attention is
       planned at capture time from the step declaration, and planned again into the
       captured buffers on each replay.

Both types share the base ``CudaGraphConfig`` fields:

- ``capture_graph_walk`` is the walk to capture.
- ``replay_graph_walks`` lists the walks that may replay this capture. One capture can
  therefore serve several walks, for example ``prefill_audio`` reusing ``prefill_text``.
- ``capture_batch_sizes`` lists the batch sizes to record.
- ``additional_key_info`` is an extra hashable component of the capture bucket key. Use it
  when one walk must be captured more than once because the batch shape alone does not
  identify the capture. Its value must equal both the value returned by the submodule's
  ``cg_key_info(graph_walk, per_request_info)`` for a batch and the ``cg_key_info`` value
  that the batch's ``declare_step`` sets on its step. A mismatch raises no error. The
  batch simply does not match any capture and runs eagerly. Compute all three values from
  one place. This field replaces the removed ``labels`` and ``requires_cfg`` fields.
- ``capture_forward_method`` names the method to capture. The default is
  ``"forward_batched"``.
- ``caps_eager_batch_size`` controls whether this config's captured sizes also limit the
  engine's eager batch size for the walk. The default is ``True``, so the engine never
  batches beyond a captured size.
- ``compile`` runs ``torch.compile`` before capture. The default is ``True``.

``BatchedCudaGraphConfig`` also accepts ``total_tokens_multiplier``. Use it when one
request's step commits KV across several labels that are combined into a single plan, as
in batched guidance that packs the conditional and unconditional sequences together. The
static buffer must hold all of them, and this field scales the buffer independently of the
per-label span.

For example, the Orpheus LLM submodule captures a batched ``decode`` graph and a packed
``prefill`` graph:

.. code-block:: python

   PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
   PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

   def get_cuda_graph_configs(self, device, tp_world_size=1):
       return [
           BatchedCudaGraphConfig(
               capture_graph_walk="decode",
               single_request_inputs=ARNodeInputs(
                   input_ids=torch.zeros(1, dtype=torch.long, device=device),
                   input_seq_len=1,
               ),
           ),
           PackedCudaGraphConfig(
               capture_graph_walk="prefill",
               capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
               make_node_input=lambda n: ARNodeInputs(
                   input_ids=torch.zeros((n,), dtype=torch.long, device=device),
                   input_seq_len=n,
               ),
               capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
           ),
       ]

BAGEL shows the case of two captures for one walk. It captures ``decode`` twice, once with
``additional_key_info=False`` and once with ``additional_key_info=True``. A decode step
with guidance enabled and a decode step with guidance disabled declare different segments
over the same token count. BAGEL's ``cg_key_info`` reports which of the two a batch is.

Piecewise CUDA graphs (capturing an inner loop)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configs above capture the whole ``forward_batched`` method of a submodule. The engine
then drives the replay, including sampling and output remapping.

Sometimes you want to capture only one inner region of a forward, such as a transformer
block loop, and keep the surrounding code in eager Python. The code before the region
computes embeddings and assembles the sequence. The code after it applies the final norm
and projection. A piecewise CUDA graph supports this.

A submodule enables piecewise capture by returning one or more configs from::

   get_piecewise_cuda_graph_configs(self, device, autocast_dtype, tp_world_size=1)
       -> dict[str, PiecewiseCudaGraphConfig]

The dict key is a region name, which is any string that identifies the captured region.
Several keys capture several independent graphs. At warmup, the engine builds one
``PiecewiseCudaGraphRunner`` per key and puts them in
``engine_inputs.piecewise_runners``. Your forward looks up the runner by key and calls it.
Nothing is stored on the submodule.

Two config types are defined in ``mstar/engine/cuda_graph_config.py``. They correspond to
the two whole-forward types:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Config type
     - Use and captured shape
   * - ``PiecewiseBatchedConfig``
     - Batched inputs of equal length. The captured region sees ``[bs, seq_len, D]``, and
       every request has the same ``seq_len``. Pass ``seq_len``, the number of tokens per
       request. One graph is captured per batch size.
   * - ``PiecewisePackedConfig``
     - Packed, variable-length inputs. The captured region sees ``[total_tokens, D]``,
       with several sequences of different lengths packed together. Pass ``total_tokens``,
       a list of token-count buckets. One graph is captured per pair of batch size and
       bucket.

Both types share the base ``PiecewiseCudaGraphConfig`` fields:

- ``capture_fn`` is the callable to capture. Its interface is described below.
- ``make_static_inputs`` is a factory with signature
  ``(shape: PiecewiseCaptureShape) -> dict[str, Tensor]``. It returns the persistent
  buffers that the captured region reads. The runner owns these buffers. Before each
  replay, it copies your real tensors into them by name. ``shape`` provides ``bs``,
  ``seq_lens`` and ``total_tokens`` for the bucket being built. Allocate the hidden-state
  buffer with dtype ``autocast_dtype``, so that the copy before replay does not convert
  dtypes.
- ``forward_kwargs`` holds static keyword arguments that are passed to ``capture_fn`` on
  every call, for example a layer count or an ``is_causal`` flag.
- ``declare_step`` has signature
  ``(request_ids, seq_lens) -> SubmoduleStep | None``. It declares the region's own
  resource work over the padded batch. It is separate from the submodule's
  ``declare_step`` because the region has its own shape. The runner admits, plans and
  commits this step on each replay. This field replaces the removed ``uses_kv_cache``,
  ``plan_fn``, ``advance_seq_lens`` and ``cache_labels`` fields. A region that reads a KV
  cache declares a ``KVStep`` and an ``AttentionStep`` here.
- ``lease_before_step`` makes the runner take this region's CUDA-graph slot before the
  outer ``declare_step`` runs, and report the slot in that call's ``piecewise_leases``
  argument. The region still declares, plans and commits its own work. The lease only
  tells the outer declaration which resources are already taken, so that it can exclude
  them.
- ``capture_batch_sizes`` lists the batch sizes to record. ``None`` uses the runner
  default.
- ``compile`` runs ``torch.compile`` on ``capture_fn`` before capture. The default is
  ``False``.

**Splitting the declaration.** When a region leases its own slot, exactly one of the two
declarations must own each resource. The common pattern is for the outer ``declare_step``
to return ``None`` when the region holds a lease, and to declare the resources itself
otherwise, which covers the eager fallback path:

.. code-block:: python

   def declare_step(self, graph_walk, request_ids, inputs,
                    slot_lease=None, piecewise_leases=None, **kwargs):
       if (piecewise_leases or {}).get(BLOCK_LOOP_REGION):
           return None                      # the region declares its own KV + attention
       return SubmoduleStep(                # eager path: nobody else reserves these pages
           segments=[
               Segment(request_id=rid, label="main", span=inp.input_seq_len)
               for rid, inp in zip(request_ids, inputs, strict=True)
           ],
           steps={KV_CACHE: KVStep(), ATTN: AttentionStep(causal=False)},
       )

**The capture_fn interface.** The captured callable takes one ``PiecewiseCallInputs``
argument and returns a dict::

   capture_fn(inp: PiecewiseCallInputs) -> dict[str, Tensor]

``PiecewiseCallInputs`` has four fields. ``static_inputs`` holds the buffers owned by the
runner. ``engine_inputs`` is the region's view of the batch: request ids and per-request
info padded to the capture bucket, plus the node's resources. ``kwargs`` is the config's
``forward_kwargs``, unchanged. ``resources`` is a shortcut for
``engine_inputs.resources``. Capture and replay both use this same type, so a region
cannot read a field that exists on only one of the two paths.

.. warning::

   Read tensors from ``inp.static_inputs``, and never assign to its entries. The runner
   passes the same dict object at capture time, and updates those buffers in place before
   each replay. An assignment such as ``static_inputs["x"] = ...`` replaces the buffer,
   and the region then no longer uses the memory address that the graph recorded.

A ``capture_fn`` may also return a single ``Tensor``. The runner wraps it as
``{"x": ...}``.

**Calling the runner.** Look up the runner by region name and pass your real inputs to it.
It returns a ``PiecewiseOutput``, which behaves like a dict. Indexing and ``.get`` return
a clone that you own and can keep. ``.get_view`` returns a view without copying, which is
only valid until the next call to ``run``. See ``mstar/engine/cuda_graph_runner.py``. The
runner handles input padding, admission and planning of the region's declared step,
replay, commit, and output slicing.

The V-JEPA2 AC predictor is the reference implementation. See
``mstar/model/vjepa2/submodules.py``, region ``"block_loop"``. It has an eager section
before the region, a captured block loop that reads the KV cache over a fixed per-step
sequence, and an eager section after the region.

.. code-block:: python

   from mstar.engine.cuda_graph_config import (
       PiecewiseBatchedConfig, PiecewiseCallInputs, PiecewiseCaptureShape,
   )

   # --- the captured inner region ---
   def _block_loop_capture(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
       # READ out of inp.static_inputs; never reassign its entries
       cond_tokens = inp.kwargs.get("cond_tokens")
       fn = self.predictor.make_block_loop_fn("main", inp.static_inputs, cond_tokens)
       return {"x": fn(inp.static_inputs["x"])}

   # --- declare the region ---
   def get_piecewise_cuda_graph_configs(self, device, autocast_dtype, tp_world_size=1, **kwargs):
       def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
           # hidden state in autocast_dtype so the replay copy_ is a same-dtype memcpy;
           # position buffers stay float32 (RoPE frequency precision matters more)
           return {
               "x": torch.zeros(shape.bs, capture_seq_len, embed_dim,
                                dtype=autocast_dtype, device=device),
               "d_pos": torch.zeros(N * N, dtype=torch.float32, device=device),
               ...
           }

       def declare_step(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
           return SubmoduleStep(
               segments=[
                   Segment(request_id=rid, label="main", span=seq_len)
                   for rid, seq_len in zip(request_ids, seq_lens, strict=True)
               ],
               steps={KV_CACHE: KVStep(), ATTN: AttentionStep(causal=False)},
           )

       return {
           BLOCK_LOOP_REGION: PiecewiseBatchedConfig(
               capture_fn=self._block_loop_capture,
               make_static_inputs=make_static_inputs,
               declare_step=declare_step,
               # take the slot before the outer declaration, so it can leave
               # this region's KV to the runner
               lease_before_step=True,
               seq_len=capture_seq_len,
               forward_kwargs={"cond_tokens": cond_tokens},
               capture_batch_sizes=[1, 2, 4, 8],
           )
       }

   # --- invoke it inside the forward ---
   runner = engine_inputs.piecewise_runners.get(BLOCK_LOOP_REGION)
   if runner is not None and runner.can_run(x.size(0)):
       out = runner.run(                       # admits + plans + replays + commits
           static_inputs={"x": x, "d_pos": d_pos, ...},
           request_ids=engine_inputs.request_ids,
       )
       x = out["x"]                            # owned clone

Three points in this example are worth noting. First, positions are computed eagerly and
passed in through ``static_inputs``, so the captured region does not compute them again.
Second, the same block loop is used on both the captured path and the eager path, so the
code exists in one place only. Third, the region's ``declare_step`` covers only the
captured path. The eager path is covered by the submodule's own ``declare_step``, as shown
under **Splitting the declaration** above.

.. _config-yaml:

Step 6 — Write a config YAML
----------------------------

A config maps nodes to GPU ranks. The value under ``model:`` is your registry key. Each
``node_groups`` entry assigns one or more ``node_names`` to ``ranks``. An entry can also
name specific ``graph_walks``, which is how prefill-decode disaggregation is expressed.

.. code-block:: yaml

   model: "your_model"
   max_seq_len: 2048
   node_groups:
     - node_names: ["LLM"]
       ranks: [0]

Run it with:

.. code-block:: bash

   mstar-serve --config configs/your_model.yaml --host 0.0.0.0 --port 8000

Tuning resources per deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model declares the shapes that the model requires. A deployment then tunes the values
that suit the machine it runs on. Use a ``resources:`` block for this, with one sub-block
per ``resource_key``. A model with two caches of the same kind, such as Whisper's decoder
cache and its encoder context, can therefore tune each one separately:

.. code-block:: yaml

   model: "whisper_large"
   max_seq_len: 448
   resources:
     kv_cache:
       cpu_offload_pages: 128
     cross_kv_cache:
       cpu_offload_pages: 128

Each spec declares which keys it accepts. An unknown key raises an error at load time, so
that a misspelled setting is never silently ignored:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Spec
     - Accepts
   * - ``KVSpec``
     - ``max_num_pages``, ``page_size``, ``max_seq_len``, ``cpu_offload_pages``
   * - ``AttentionSpec`` / ``CrossAttentionSpec``
     - ``backend`` (``flashinfer`` / ``dense``), ``flashinfer_backend``
       (``auto`` / ``fa2`` / ``fa3``)

Tune the cache shape on the KV resource, not on the attention resource that reads it. For
example, ``configs/qwen3tts.yaml`` selects FA2 under ``talker_attn``, while
``configs/cosmos3_nano.yaml`` sets the page count under its KV key.

.. note::

   A top-level ``kv_cache:`` block is no longer read. It raises an error at load time,
   with a message describing the migration. Move the block under ``resources:``, keyed by
   the resource name.

.. _tensor-parallelism:

Tensor parallelism (sharding)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To shard a node across several GPUs, add ``tp_size`` to its ``node_groups`` entry and list
``tp_size`` ranks. The runtime splits the group's ranks into TP groups of that size and
builds one ``comm_group`` per shard.

A node is sharded only if its components are built from the tensor-parallel modules in
``mstar/model/components/distributed``: ``ParallelAttention``, ``ParallelGatedMLP``,
``ColumnParallelLinear``, ``RowParallelLinear``, ``VocabParallelEmbedding`` and others.
The ``weight_loader`` of each such parameter slices it automatically. See `Loading
weights`_. A node whose components do not use these modules is replicated on every rank
instead. For example, a tensor-parallel Qwen3-Omni Talker keeps its code predictor
replicated.

Once a component is built from these modules, moving from one GPU to tensor parallelism
requires only the YAML change and a small sharding declaration. No model code changes. The
example below runs the Orpheus LLM with tensor parallelism across two GPUs. See
``configs/orpheus_tp2.yaml``:

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
Override ``get_default_sharding_config`` to return a ``ShardingConfig`` that names the
shardable nodes, and any non-default shard dimensions:

.. code-block:: python

   def get_default_sharding_config(self):
       from mstar.distributed.base import ShardingConfig
       return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={})

Two separate mechanisms split work across the TP group. Keep them distinct:

- **Weights** are sharded inside the components, by each parameter's ``weight_loader``.
  Column-parallel and row-parallel linear layers built with the ``comm_group`` do this
  automatically, once the module is constructed for tensor parallelism. The config needs
  nothing more.
- **Activations** that cross a node boundary are handled by ``shard_dim`` in the
  ``ShardingConfig``. It maps an inter-node edge or signal name to the dimension along
  which that tensor is split across the group. If a name is absent, or maps to ``None``,
  the tensor is replicated to every rank. You need an entry only for edges where the
  producer and the consumer both keep the data sharded. The common case is replicated
  activations, which needs no entry. You can also set ``shard_dim`` per run, under a
  ``sharding_config`` block in the YAML.

If a node group has ``tp_size > 1`` and names a node that is not in ``tp_enabled_nodes``,
loading fails. See ``configs/qwen3omni_thinker_tp2.yaml`` for an example with several node
types.

Worked example: Orpheus
-----------------------

Orpheus (``mstar/model/orpheus/``) is a small and complete reference. It is a TTS model. A
Llama 3.2 3B LLM emits audio tokens, and a SNAC decoder converts them into 24 kHz PCM.

The two nodes have different needs. The LLM declares the four standard resources, shown
under :ref:`Step 2a <Step 2a — Declare your resources>`. No spec names ``snac_decoder``,
so that node receives no resources. The SNAC decoder instead declares how it must be run,
using two class attributes on the submodule:

.. code-block:: python

   class SNACDecoderSubmodule(NodeSubmodule):
       disable_torch_compile = True   # runs in fp32, without compilation
       disable_autocast = True

There are three graph walks: ``prefill`` and a ``decode`` ``Loop`` on the LLM, plus a
``snac_chunk`` node that emits audio to the client:

.. code-block:: python

   snac_chunk = GraphNode(
       name="snac_decoder",
       input_names=["new_token"],
       outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio_chunk",
                          output_modality="audio")],
   )

``process_prompt`` formats the string ``"{voice}: {text}"``, tokenizes it, wraps the ids in
the model's start and end tokens, and returns ``{"text_inputs": [ids]}``.
``get_submodule`` builds either the Llama LLM submodule, which is an ``ARNodeSubmodule``,
or the SNAC decoder submodule, and caches the result. ``postprocess`` returns the raw
bytes of the audio tensor for the ``audio`` modality.

Orpheus also demonstrates the async partition API, described in the next section. The LLM
and the SNAC decoder run as two partitions connected by a streaming edge. Audio is
therefore decoded in a sliding window while the LLM is still generating.

Worked example: BAGEL
---------------------

Orpheus is a single pipeline. BAGEL (``mstar/model/bagel/``) is much more complex, and it
shows why the graph abstraction is useful. BAGEL is a unified model. It performs image
understanding, which maps an image to text, and image generation, which maps text to an
image. Both use the same Qwen2 LLM. That LLM is also the denoiser for rectified-flow image
generation. The steps below follow the same order as the rest of this page.

**Step 1 — Register.** This is already done in ``registry.py``, with the entry
``"bagel": BagelModel`` and an ``HF_MODELS`` entry that points to
``ByteDance-Seed/BAGEL-7B-MoT``.

**Step 2 and 2a — Nodes and resources.** The model has four core nodes: a ViT encoder
(SigLIP2, used for understanding), a VAE encoder (FLUX, used for editing and generation),
the LLM (Qwen2, which contains the embedding, the transformer, the lm_head and the CFG
logic), and a VAE decoder. It has four more nodes for the CFG-parallel image-generation
path described below: ``init_latents``, the two branch nodes ``LLM_cfg_text`` and
``LLM_cfg_img``, and ``combine_cfg``.

Only the three LLM nodes need resources, and all three share one set of them: the same KV
pool, attention, positions and sampler. Each spec names all three nodes in its ``nodes``
field:

.. code-block:: python

   def get_node_resources(self) -> list[NodeResourceSpec]:
       nodes = set(self._LLM_NODES)   # {"LLM", "LLM_cfg_text", "LLM_cfg_img"}
       return [
           KVSpec(resource_key="kv", nodes=nodes, config=self._kv_config()),
           AttentionSpec(resource_key="attn", nodes=nodes,
                         config=AttentionConfig(kv_cache="kv")),
           PositionSpec(resource_key="rope", nodes=nodes,
                        config=PositionConfig(kv_cache="kv",
                                              rope_theta=self.config.rope_theta)),
           SamplerSpec(resource_key="sampler", nodes=nodes,
                       vocab_size=self.config.vocab_size),
       ]

No spec names the encoders, ``init_latents``, ``combine_cfg`` or the VAE decoder, so those
nodes receive no resources. The CFG nodes are always declared, but they are used only when
the config enables CFG-parallel mode, described under Step 6. A single-GPU config never
routes requests to them.

**Per-request resources.** Whether a request uses classifier-free guidance determines
which cache labels it reads. This is a property of the request, not of the deployment.
BAGEL's ``get_request_resource_configs`` therefore returns a ``KVReqConfig`` that names
the active labels per node and walk, together with a ``SamplingReqConfig``. A request with
guidance disabled names only ``"main"``, so a PD-disaggregated transfer copies only that
stream.

The ``LLM`` node is deliberately coarse. It contains the text embedding, the lm_head and
the flow projection. These always run on the same GPU, so splitting them into separate
graph nodes would only add IPC overhead. This is a general modeling rule: make a node as
coarse as the colocation boundary allows.

**Step 3 — Graph walks.** Understanding and generation are different pipelines, so BAGEL
returns five walks from ``get_graph_walk_graphs`` instead of two:

.. list-table::
   :header-rows: 1
   :widths: 18 54

   * - Graph walk
     - What it does
   * - ``prefill_text``
     - Embed text tokens and prefill the LLM. Attention is causal.
   * - ``prefill_vit``
     - ``vit_encoder``, then the LLM. Encodes an input image for understanding. Attention
       is bidirectional.
   * - ``prefill_vae``
     - ``vae_encoder``, then the LLM. Encodes an image for editing or generation.
   * - ``decode``
     - Autoregressive text generation. This is a ``Loop``, the same as in Orpheus.
   * - ``image_gen``
     - The flow-matching denoising ``Loop``. The LLM applies CFG and one Euler step per
       iteration. ``vae_decoder`` then converts the final latents into pixels.

The two encoder walks are ``Sequential`` chains of two nodes. ``image_gen`` is a ``Loop``
followed by the decoder. In the loop body, ``latents`` and ``time_index`` are routed back
to the same node, and the loop's ``outputs`` pass the final latents to ``vae_decoder``:

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

**Declared outputs are conditional.** A node's ``outputs`` list is the set of edges that
the node can emit. What it emits on a given step depends on what its submodule produces.
``new_token`` above is the clearest example. The LLM samples a token only when the request
requires text output, which is the case on the understanding path and on every ``decode``
step. On the image-generation and editing paths, the same node still runs and writes the
KV cache, but samples no token, so it does not produce ``new_token``. The edge is present
in the graph because understanding requests need it. Treat declared edges as the possible
outputs, and let the submodule decide which of them are produced on each step.

**Choosing the walk per request.** BAGEL's transitions are driven by a schedule, unlike
those of Orpheus. The output modality is known in advance from the request's
``output_modalities``. ``get_initial_forward_pass_args`` therefore builds a prefill
schedule, by iterating over the interleaved text and image inputs, and
``get_partition_forward_pass_args`` advances through that schedule. It then transitions to
``decode`` for text output, or to ``image_gen`` for image output. In ``think_mode``, the
model first decodes a reasoning trace, and the EOS token then triggers the transition to
``image_gen``. These are the same two methods that Orpheus implements. BAGEL only encodes
a more complex state machine in them.

**Step 4 — Submodules.** Each node maps to a ``NodeSubmodule`` in ``bagel/submodules.py``:
``ViTEncoderSubmodule``, ``VAEEncoderSubmodule``, ``LLMSubmodule`` and
``VAEDecoderSubmodule``. ``get_submodule`` builds them on demand, so a worker that runs
only ``vit_encoder`` never allocates the 7B LLM. ``process_prompt`` tokenizes the prompt,
and also a system prompt in ``think_mode``. ``postprocess`` selects an encoding by
modality: ``utf-8`` text for ``decode``, and PNG bytes for ``image``.

**Step 4a — Step declarations.** BAGEL's ``LLMSubmodule.declare_step`` is the most complex
declaration in the tree. Read it to see every advanced ``KVStep`` field in context:

- ``pre_forks`` and ``post_forks``: guidance requires a ``cfg_text`` stream forked from
  ``main``, and the fork time differs by walk. In ``prefill_text``, the branch must keep
  the context from before the text, so it forks before any planning or writing. In
  ``prefill_vit`` and ``prefill_vae``, the branch must include the image, so it forks at
  commit time, after this step's writes are applied.
- ``combined_labels``: in the batched-guidance denoise step, the conditional and
  unconditional labels are packed into one plan under a single combined label.
  ``PositionStep.pos_ids`` is keyed by that combined label, with the ids concatenated in
  the same label-major order.
- ``commit=False``: the step plans and reads, but its writes do not become resident yet.
- ``cg_key_info``: the step records whether guidance is enabled. This value matches the
  ``additional_key_info`` of the two ``decode`` capture configs, so each replay uses its
  own bucket. The submodule's ``cg_key_info()`` method reports the same value from the
  batch, because the engine leases a slot before the step is declared.

**Step 6 — Config and disaggregation.** This is the main benefit of the graph
abstraction. The same model code runs on one GPU:

.. code-block:: yaml

   model: "bagel"
   max_seq_len: 32768
   node_groups:
     - {node_names: [vit_encoder], ranks: [0]}
     - {node_names: [vae_encoder, vae_decoder], ranks: [0]}
     - {node_names: [LLM], ranks: [0]}

The same code also runs disaggregated across GPUs. Assign the same ``LLM`` node to
different ranks per graph walk: prefill on GPU 0, decode on GPU 1, and image generation on
GPU 2:

.. code-block:: yaml

   node_groups:
     - {node_names: [LLM], ranks: [0], graph_walks: [prefill_text, prefill_vit, prefill_vae]}
     - {node_names: [LLM], ranks: [1], graph_walks: [decode]}
     - {node_names: [LLM], ranks: [2], graph_walks: [image_gen]}

BAGEL also supports a CFG-parallel mode. When the config names the extra
``LLM_cfg_text`` and ``LLM_cfg_img`` nodes, as in ``configs/bagel_cfg_parallel.yaml``, the
model uses an ``image_gen_cfg`` walk instead. The loop body of that walk is a ``Parallel``
of the three classifier-free-guidance branches, each on its own GPU, feeding a
``combine_cfg`` node. The model code detects this mode only from the node names present in
the config, so the extra parallelism is enabled in YAML and requires no code change. One
model can therefore run in many physical layouts.

Worked example: Whisper
-----------------------

Whisper (``mstar/model/whisper/``) is the reference for encoder-decoder models. It is the
smallest complete example of cross-attention over a fixed context.

It has two nodes. ``audio_encoder`` declares no resources. ``decoder`` declares six: its
own KV cache and attention, a second KV cache for the encoder context, a
``CrossAttentionSpec`` over that second cache, a ``PositionSpec``, and a ``SamplerSpec``.
The ``PositionSpec`` exists only for the position counter. Whisper uses learned position
embeddings, so the planned ids index an ``embed_positions`` table instead of driving RoPE.
The ``SamplerSpec`` sets ``enable_repetion_penalty=False``, because ASR transcription
decodes greedily.

The page counts show how to size a cache for a specific model instead of using the
default. A sequence is at most ``max_target_positions`` (448) tokens, which is 4 pages per
request. 128 pages therefore support about 32 concurrent requests and use about 2.7 GB.
The 2048-page default would use about 43 GB. The fixed 30-second context window is
``max_source_positions`` (1500) tokens, which is 12 pages per request, so 192 pages
support about 16 concurrent requests.

Whisper's ``declare_step`` shows how a write-once context is expressed with spans. The
context segments have a non-zero span in the prefill step that writes them, and a span of
0 in every later step. The ``commit`` in the prefill step converts the reservation into
resident pages that the later steps read.

Advanced: async partitions and streaming
----------------------------------------

Models with a single partition can skip this section. The defaults in ``Model`` provide
one partition, named ``"default"``, that contains all walks.

Use several partitions when one stage must run asynchronously while another continues to
produce data, for example an LLM feeding a vocoder, or a thinker feeding a talker.
Override these methods:

- ``get_partition_topology()`` declares the partitions and the streaming ``Connection``
  objects between them, including a ``chunk_policy_factory`` such as
  ``SlidingWindowChunkPolicy(window=..., stride=...)``.
- ``get_partitions()`` declares each ``PartitionDefinition``: its walks, its initial walk,
  and the partitions that produce data into it.
- Route cross-partition tensors with ``StreamingGraphEdge(next_node=..., name=...,
  target_partition=...)`` instead of a plain ``GraphEdge``.

The consuming partition's ``get_partition_forward_pass_args`` reads
``incoming_connections``, which provides token counts and a ``producer_done`` flag, and
uses them to decide when to run.

Checklist
---------

.. code-block:: text

   [ ] mstar/model/<your_model>/config.py        — config dataclass + resource-key constants
   [ ] mstar/model/<your_model>/components/       — the nn.Modules + weight loading
         [ ] layers name their resource keys (attn_key / kv_key / pos_key)
   [ ] mstar/model/<your_model>/submodules.py     — NodeSubmodule per node
         [ ] prepare_inputs (set input_seq_len)
         [ ] declare_step
         [ ] preprocess / forward (+ forward_batched, can_batch)
   [ ] mstar/model/<your_model>/<your_model>_model.py — Model subclass:
         [ ] get_node_resources
         [ ] get_graph_walk_graphs
         [ ] process_prompt
         [ ] get_initial_forward_pass_args
         [ ] get_partition_forward_pass_args
         [ ] postprocess
         [ ] get_submodule
         [ ] (optional) get_request_resource_configs
   [ ] mstar/model/registry.py                    — add to MODEL_REGISTRY (+ HF_MODELS)
   [ ] configs/<your_model>.yaml                  — node_groups → ranks (+ resources: overrides)
   [ ] (optional) async partitions if pipelined

Testing
-------

Validate the graph and the worker integration before you use real weights. The modular
tests run on CPU and exercise models in dummy mode, where ``get_submodule`` returns
``None``:

.. code-block:: bash

   ruff check .
   pytest test/modular/                  # CPU graph/worker tests
   pytest test/integration/              # requires GPU + weights
   mstar-serve --config configs/your_model.yaml --port 8000

These modular tests are useful while writing a new model. They cover the parts that are
easy to get wrong, and they run on CPU:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Test
     - Covers
   * - ``test/modular/test_resource_runner.py``
     - The admit, plan and commit lifecycle driven by ``declare_step``.
   * - ``test/modular/vjepa2/fake_resources.py``
     - Stub resources for testing a submodule's step declaration without a GPU.
   * - ``test/modular/test_cuda_graph_capture.py``
     - Capture-bucket keys, including ``cg_key_info`` and ``additional_key_info``.
   * - ``test/modular/test_micro_scheduler.py``
     - Batch admission, and the paths for failed and backpressured requests.
   * - ``test/modular/test_admit_failure_handling.py``
     - Behavior when a resource cannot admit a step.
   * - ``test/modular/test_kv_offload.py``
     - Offload and reload of a request's cache state.

Then send a ``POST /generate`` request and check the streamed output.

The fastest way to add a new model is to base it on the existing model that is closest to
it. Use Orpheus for a streaming LLM with a codec, Whisper for encoder-decoder
cross-attention, BAGEL for a unified understanding and generation model, and Qwen3-Omni
for a full omni-modal model.
