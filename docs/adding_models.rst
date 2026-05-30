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
   (``KV_CACHE`` or ``STATELESS``). See `Step 5 — Choose engine types`_ below.

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

- ``GraphNode(name, input_names, outputs)`` — one unit of compute. ``name`` must match a
  key in ``get_node_engine_types``. ``input_names`` are the tensor names that must be
  present before the node can run; ``outputs`` is a list of ``GraphEdge``.
- ``GraphEdge(next_node, name, ...)`` — routes an output tensor named ``name`` to
  ``next_node``. Flags: ``persist=True`` keeps the tensor for later steps,
  ``conductor_new_token=True`` reports a generated token to the conductor, and
  ``output_modality="audio"`` (with ``next_node=EMIT_TO_CLIENT``) streams it to the
  client. Special destinations live in ``mminf/graph/special_destinations.py``
  (``EMIT_TO_CLIENT``, ``EMPTY_DESTINATION``).
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
           outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name="new_token",
                              conductor_new_token=True, persist=True)],
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
   Runs off the GPU thread and *may* read tensor values. Return the names of the
   ``Loop`` s to stop (e.g. when you see the EOS token). This is how decode terminates.

Optional knobs: ``can_batch`` / ``forward_batched`` (batching is off by default),
``get_cuda_graph_configs`` (declare CUDA-graph captures), and ``cleanup_request``.

Step 5 — Choose engine types
----------------------------

You almost never write an engine; you assign one of the two
:class:`~mminf.engine.base.EngineType` values per node in ``get_node_engine_types``.
The engine type — not the submodule — decides whether the node gets a managed KV cache:

.. list-table::
   :header-rows: 1
   :widths: 20 52

   * - ``EngineType``
     - Use for
   * - ``KV_CACHE``
     - Any node that needs a persistent, paged KV cache across forward passes —
       autoregressive LLMs (text decode) and LLM-as-denoiser flow loops alike. Runs on
       :class:`~mminf.engine.kv_cache_engine.KVCacheEngine`; pairs with an
       ``ARNodeSubmodule`` and an entry in ``get_kv_cache_config``.
   * - ``STATELESS``
     - Every node *without* cross-step KV state — ViT / VAE / audio encoders and
       decoders, embedding and projection stages, flow-matching combine steps, codec
       (waveform) decoders. Runs on
       :class:`~mminf.engine.stateless_engine.StatelessEngine`.

A model's job is just to label each node; the worker instantiates the right engine and
gives ``KV_CACHE`` nodes their cache from ``get_kv_cache_config``.

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

Orpheus is a single pipeline. BAGEL (``mminf/model/bagel/``) is the opposite end of the
spectrum and a better illustration of *why* the graph abstraction exists: it is a
**unified** model that does both image *understanding* (image → text) and image
*generation* (text → image) with the **same** Qwen2 LLM, which also serves as the
denoiser for rectified-flow image generation. Walking through the same steps shows how
the pieces scale up.

**Step 1 — Register.** Already done in ``registry.py``: ``"bagel": BagelModel`` plus an
``HF_MODELS`` entry pointing at ``ByteDance-Seed/BAGEL-7B-MoT``.

**Step 2/5 — Nodes and engine types.** BAGEL declares four logical nodes; only the LLM
carries a KV cache, everything else is stateless:

.. code-block:: python

   def get_node_engine_types(self) -> dict[str, EngineType]:
       return {
           "vit_encoder": EngineType.STATELESS,   # SigLIP2 ViT (understanding)
           "vae_encoder": EngineType.STATELESS,   # FLUX VAE encode (editing/gen)
           "LLM":         EngineType.KV_CACHE,     # Qwen2: embed + transformer + lm_head + CFG
           "vae_decoder": EngineType.STATELESS,   # FLUX VAE decode → pixels
       }

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
       GraphNode(name="LLM", input_names=["img_emb"], outputs=[]),
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
                              output_modality="image", persist=True)],
       ),
   ])

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

**Step 6 — Config and disaggregation.** This is where BAGEL pays off. The *same* model
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
family on the closest existing one (Orpheus for a streaming LLM + codec, BAGEL for
multi-engine understanding + generation, Qwen3-Omni for full omni-modal) is by far the
fastest path.
