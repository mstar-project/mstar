Adding a New Model
==================

This page walks through everything you need to implement to add support for a new model
in ``mminf``. By the end you will have a model that the conductor can schedule, that
workers can execute on GPU, and that you can launch with ``mminf-serve``.

Mental model
------------

A model in ``mminf`` is split into a handful of well-defined responsibilities:

- **The** ``Model`` **class** (``mminf/model/base.py``) is the contract the rest of the
  system talks to. It tokenizes prompts, declares the computation graph, says which
  engine runs each node, builds forward-pass arguments, and post-processes outputs. It
  contains *no* GPU compute.
- **Submodules** (``NodeSubmodule`` in ``mminf/model/submodule_base.py``) are the
  ``torch.nn.Module`` s that *do* the compute. Each graph node maps to one submodule.
- **Engines** (``mminf/engine/``) wrap submodules with execution machinery (KV cache,
  FlashInfer, CUDA graphs, batching). You pick an engine *type* per node; you rarely
  write a new engine.
- **The graph** (``mminf/graph/base.py``) is how a model declares *what runs in what
  order*: nodes, edges between them, and loops. Each named "graph walk" (e.g.
  ``prefill``, ``decode``) is one graph.
- **The config YAML** (``configs/``) maps graph nodes to physical GPU ranks via
  ``node_groups``. This is where disaggregation happens — the same model code runs
  single-GPU or sharded across many depending only on the config.

The flow at request time::

   process_prompt()              # text/media -> initial tensors
        │
        ▼
   get_initial_forward_pass_args()   # seed the first graph walk (e.g. prefill)
        │
        ▼   (conductor walks the graph, running nodes via engines)
   get_partition_forward_pass_args() # asked after each step: what's next? done?
        │
        ▼
   postprocess()                 # model output tensor -> bytes for the client

What you will create
--------------------

A typical model lives in its own package under ``mminf/model/<your_model>/``:

.. code-block:: text

   mminf/model/<your_model>/
   ├── __init__.py
   ├── config.py            # a @dataclass with architecture + generation params
   ├── <your_model>_model.py # the Model subclass (the contract)
   ├── submodules.py        # NodeSubmodule subclasses (the compute wrappers)
   └── components/          # the actual nn.Modules (attention, decoder, etc.)

Plus two things outside that package:

- an entry in ``mminf/model/registry.py`` so the model is discoverable, and
- a config YAML in ``configs/`` mapping nodes to ranks.

Step 1 — Register the model
---------------------------

Open ``mminf/model/registry.py`` and add your class to ``MODEL_REGISTRY`` (and, if it
loads weights from Hugging Face, to ``HF_MODELS``). The dict key is the string you put
under ``model:`` in a config YAML.

.. code-block:: python

   from mminf.model.your_model.your_model_model import YourModel

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

Subclass :class:`mminf.model.base.Model` and implement its abstract methods. The
constructor receives ``model_path_hf`` (from ``HF_MODELS``) plus any ``**kwargs``; it
typically loads the tokenizer and stores a config dataclass. Defer heavy weight loading
to ``get_submodule`` so the conductor process never allocates GPU memory.

The abstract methods you **must** implement:

``get_kv_cache_config(self) -> list[KVCacheConfig]``
   Per-node KV cache configs for autoregressive nodes (``num_layers``,
   ``num_kv_heads``, ``head_dim``, ``max_seq_len``, ``num_qo_heads``). Return a single
   config if all AR nodes share one. Models with no AR node may return an empty list.

``get_node_engine_types(self) -> dict[str, EngineType]``
   Maps each graph-node name to an :class:`mminf.engine.base.EngineType`
   (``AR``, ``FLOW``, ``ENC_DEC``, ``AUDIO_CODEC``, ``CODE_PREDICTOR``). See
   `Step 5 — Choose engine types`_ below.

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
   Build the first :class:`mminf.model.base.ForwardPassArgs` for a partition — which
   graph walk to start on and which input edges feed it.

``get_partition_forward_pass_args(self, partition_name, partition_metadata, persist_signals, new_tokens, incoming_connections=None) -> ForwardPassArgs``
   Called by the conductor after each completed step to decide the *next* walk, its
   inputs, and whether the request is done (``request_done=True``). For a simple
   prefill→decode model this flips ``is_prefill`` once and then loops decode until EOS.

``postprocess(self, output, modality) -> bytes``
   Encode a finished output tensor to bytes for the client (``utf-8`` for text, PNG for
   images, raw PCM for audio, …).

``get_submodule(self, node_name, device="cpu") -> NodeSubmodule | None``
   Lazily build and return the ``NodeSubmodule`` for ``node_name`` (load weights here,
   on ``device``). Cache the result. Return ``None`` for dummy mode. See
   `Step 4 — Implement the submodules`_.

Useful overridable defaults (not abstract): ``get_sampling_config`` (temperature/top-p
per node), ``get_max_output_tokens``, ``get_autocast_dtype``, ``load_image`` /
``load_audio`` / ``load_video``, and the partition API below.

Step 3 — Declare the computation graph
--------------------------------------

``get_graph_walk_graphs`` returns one graph per *walk*. The primitives
(``mminf/graph/base.py``):

- ``GraphNode(name, input_ids, outputs, optional_input_ids=...)`` — one unit of compute.
  ``name`` must match a key in ``get_node_engine_types``. ``input_ids`` are the tensor
  names that must be present before the node can run; ``outputs`` is a list of
  ``GraphEdge``.
- ``GraphEdge(next_node, name, ...)`` — routes an output tensor named ``name`` to
  ``next_node``. Flags: ``persist=True`` keeps the tensor for later steps,
  ``conductor_new_token=True`` reports a generated token to the conductor, and
  ``output_modality="audio"`` (with ``next_node=EMIT_TO_CLIENT``) streams it to the
  client. Special destinations live in ``mminf/graph/special_destinations.py``
  (``EMIT_TO_CLIENT``, ``EMPTY_DESTINATION``).
- ``Sequential([...])`` / ``Parallel([...])`` — compose subgraphs in order or
  concurrently.
- ``Loop(section, max_iters, outputs, ...)`` — fixed-count loop.
- ``DynamicLoop(name, section, max_iters, outputs)`` — loop that stops early when a
  ``check_stop`` condition fires (e.g. EOS). This is the usual ``decode`` loop.

A minimal text generator has two walks — a one-shot ``prefill`` node and a ``decode``
``DynamicLoop`` whose body feeds its own output back as the next input:

.. code-block:: python

   def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
       prefill = GraphNode(
           name="LLM",
           input_ids=["text_inputs"],
           outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name="new_token",
                              conductor_new_token=True, persist=True)],
       )
       decode = DynamicLoop(
           name="decode_loop",
           section=GraphNode(
               name="LLM",
               input_ids=["text_inputs"],
               outputs=[GraphEdge(next_node="LLM", name="text_inputs")],  # loop-back
           ),
           max_iters=self.get_max_output_tokens(),
           outputs=[],
       )
       return dict(prefill=prefill, decode=decode)

Step 4 — Implement the submodules
---------------------------------

Each node name maps to a :class:`mminf.model.submodule_base.NodeSubmodule` (a
``torch.nn.Module``). Autoregressive nodes use the ``ARNodeSubmodule`` subclass. The
contract:

``prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs``
   Convert the routed ``NameToTensorList`` into a typed ``NodeInputs`` (or
   ``ARNodeInputs`` with ``input_ids`` / ``input_embeds`` / ``input_seq_len``). Runs
   per request, on CPU-ish metadata only.

``preprocess(self, graph_walk, engine_inputs, inputs) -> dict``
   Collate a batch of ``NodeInputs`` into the kwargs your ``forward`` expects. The base
   default handles batch size 1; ``ARNodeSubmodule`` makes it abstract (implement
   batching).

``forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList``
   The pure tensor → tensor computation (this is what gets CUDA-graphed/compiled). Keys
   in the returned dict are the edge ``name`` s the graph routes downstream.

``postprocess(...)`` (optional)
   Metadata-only fixups that run on the GPU thread — **must not** read tensor values
   (no ``.item()``/``.cpu()``). Use it to rebind output names for routing.

``check_stop(...) -> set[str]`` (optional)
   Runs off the GPU thread and *may* read tensor values. Return the names of
   ``DynamicLoop`` s to stop (e.g. when you see the EOS token). This is how decode
   terminates.

Optional knobs: ``can_batch`` / ``forward_batched`` (batching is off by default),
``get_cuda_graph_configs`` (declare CUDA-graph captures), and ``cleanup_request``.

Step 5 — Choose engine types
----------------------------

You almost never write an engine; you assign one of the existing
:class:`~mminf.engine.base.EngineType` values per node in
``get_node_engine_types``:

.. list-table::
   :header-rows: 1
   :widths: 22 50

   * - ``EngineType``
     - Use for
   * - ``AR``
     - Autoregressive token generation with a KV cache (LLMs). Pairs with
       ``ARNodeSubmodule`` and ``get_kv_cache_config``.
   * - ``ENC_DEC``
     - Stateless encoders/decoders — ViT / audio encoders, embedding stages.
   * - ``FLOW``
     - Diffusion / ODE flow-matching steps (e.g. Pi0.5's action expert, image gen).
   * - ``AUDIO_CODEC``
     - Audio token → waveform decoders (e.g. SNAC).
   * - ``CODE_PREDICTOR``
     - Depth/code prediction heads used by some speech codecs.

Step 6 — Write a config YAML
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

   mminf-serve --config configs/your_model.yaml --host 0.0.0.0 --port 8000

Worked example: Orpheus
-----------------------

Orpheus (``mminf/model/orpheus/``) is a compact, complete reference. It is a TTS model:
a Llama 3.2 3B LLM emits audio tokens, and a SNAC decoder turns them into 24 kHz PCM.

Two nodes, two engines:

.. code-block:: python

   def get_node_engine_types(self) -> dict[str, EngineType]:
       return {
           "LLM": EngineType.AR,
           "snac_decoder": EngineType.AUDIO_CODEC,
       }

Three graph walks — ``prefill`` and a ``decode`` ``DynamicLoop`` on the LLM, plus a
``snac_chunk`` node that emits audio to the client:

.. code-block:: python

   snac_chunk = GraphNode(
       name="snac_decoder",
       input_ids=["new_token"],
       outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio_chunk",
                          output_modality="audio")],
   )

``process_prompt`` formats ``"{voice}: {text}"``, tokenizes, and wraps the ids in the
model's special start/end tokens, returning ``{"text_inputs": [ids]}``.
``get_submodule`` lazily builds either the Llama ``ARNodeSubmodule`` or the SNAC
``AudioCodecSubmodule`` and caches it. ``postprocess`` returns the audio tensor's raw
bytes for the ``audio`` modality.

Orpheus also demonstrates the **async partition** API (next section): the LLM and SNAC
run as two partitions connected by a streaming edge, so audio is decoded in a sliding
window *while* the LLM is still generating.

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

   [ ] mminf/model/<your_model>/config.py        — config dataclass
   [ ] mminf/model/<your_model>/components/       — the nn.Modules + weight loading
   [ ] mminf/model/<your_model>/submodules.py     — NodeSubmodule per node
   [ ] mminf/model/<your_model>/<your_model>_model.py — Model subclass:
         [ ] get_kv_cache_config
         [ ] get_node_engine_types
         [ ] get_graph_walk_graphs
         [ ] process_prompt
         [ ] get_initial_forward_pass_args
         [ ] get_partition_forward_pass_args
         [ ] postprocess
         [ ] get_submodule
   [ ] mminf/model/registry.py                    — add to MODEL_REGISTRY (+ HF_MODELS)
   [ ] configs/<your_model>.yaml                  — node_groups → ranks
   [ ] (optional) async partitions if pipelined

Testing
-------

Start with the dummy plumbing to validate the graph before touching real weights, then:

.. code-block:: bash

   ruff check .
   pytest test/modular/                  # CPU graph/worker tests
   pytest test/integration/              # requires GPU + weights
   mminf-serve --config configs/your_model.yaml --port 8000

Then send a ``POST /generate`` request and confirm the streamed output. Modeling a new
family on the closest existing one (Orpheus for AR+codec, Pi0.5 for VLA+flow,
Qwen3-Omni for full omni-modal) is by far the fastest path.
