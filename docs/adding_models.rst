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

Plus a handful of things outside that package:

- an entry in ``mstar/model/registry.py`` so the model is discoverable,
- a config YAML in ``configs/`` mapping nodes to ranks,
- an entry in ``DEFAULT_CONFIGS`` (``mstar/cli/main.py``) so ``mstar serve`` knows the name,
- a row in ``docs/models.rst``, and
- an OpenAI adapter, only if the model needs OpenAI-compatible routes.

Step 1 covers all of them.

Step 1 — Register the model
---------------------------

Open ``mstar/model/registry.py`` and add your model to ``MODEL_REGISTRY`` (and, if it
loads weights from Hugging Face, to ``HF_MODELS``). The dict key is the string you put
under ``model:`` in a config YAML.

The value is a ``(module_path, class_name)`` tuple, not an imported class: the module is
imported lazily through ``import_module`` when the model is actually used, which keeps
``mstar.model.registry`` cheap to import. A direct class import here would pull every
model's dependencies into every process.

.. code-block:: python

   MODEL_REGISTRY: dict[str, tuple[str, str]] = {
       # ...
       "your_model": ("mstar.model.your_model.your_model_model", "YourModel"),
   }

   HF_MODELS: dict[str, dict] = {
       # ...
       "your_model": {"model_path_hf": "org/your-model-id"},
   }

There is no plugin scan, so the registry is the only place the class is discovered — but it
is **not** the only wiring step. A model also needs:

- ``DEFAULT_CONFIGS`` in ``mstar/cli/main.py``, mapping the name to its default config file.
  Without it, ``mstar serve <your_model>`` exits with ``error: unknown model`` before
  anything loads.
- a row in ``docs/models.rst``, which is the model list users read.
- an ``OpenAIAdapter`` subclass in ``ADAPTER_REGISTRY``
  (``mstar/api_server/openai/adapters.py``), *only* if the model maps onto OpenAI semantics
  such as ``chat.completions``, ``audio.speech`` or ``images.generate``. A model reached
  only through ``POST /generate`` needs none.
- the client-example tuples further down ``mstar/cli/main.py``, so ``mstar serve`` prints a
  usage snippet. Cosmetic, but user-visible.

``test/modular/test_model_registration.py`` enforces the registry, CLI and docs entries, so
CI fails rather than shipping a model nobody can reach.

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

``process_prompt(self, prompt, input_modalities, output_modalities, tensors=None, prompt_parts=None, **kwargs) -> NameToTensorList``
   Tokenize the prompt and produce the initial request tensors, for example
   ``{"text_inputs": [token_ids]}``. This method runs in the API-server data worker,
   after raw media tensors are loaded. It can therefore read ``tensors`` (for example
   ``image_inputs``, ``audio_inputs`` or ``video_inputs``) to compute derived tensors
   such as ``pixel_values``. The returned dict is merged into the request's tensors.

   ``input_modalities`` is the request layout: one entry per prompt element in the
   order written, so ``["image", "text", "image"]`` is a distinct request from
   ``["image", "image", "text"]``. ``prompt_parts`` carries the same sequence with
   the text attached, and is ``None`` for entrypoints that submit files and text
   separately. :mod:`mstar.model.multimodal` turns either into the prefill plan
   that ``process_prompt`` and the schedule builder both work from.

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

``get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None, moe_parallel=None) -> NodeSubmodule | None``
   Build and return the ``NodeSubmodule`` for ``node_name``, and cache the result. Load
   weights here, on ``device``. Return ``None`` for dummy mode.

   ``tp_group`` and ``sp_group`` are the node's tensor-parallel and sequence-parallel
   communicators when the node is sharded. Pass ``tp_group`` to the parallel-linear
   constructors. See :ref:`Step 6 <tensor-parallelism>`.

   ``moe_parallel`` is the node group's ``ep_config``, parsed into a
   ``MoeParallelConfig``. It is passed only to an EP-enabled node whose group sets one,
   so a model without MoE layers never receives it. See `Expert parallelism (MoE)`_.

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

.. note::

   ``model_kwargs`` cross the control plane to the conductor as plain data, so a
   value that ``process_prompt`` moves onto a tensor edge must not travel there
   too. List those kwargs in ``PREPROCESS_ONLY_KWARGS``: the data worker keeps the
   originals for ``postprocess`` and leaves them out of the conductor's copy. Pi0.5
   lists ``robot_state``, and Cosmos3 lists ``action`` (it passes only the shape on as
   ``action_shape`` metadata). A tensor left in ``model_kwargs`` fails at send
   with an error that names the kwarg.

.. note::

   The metadata ``process_prompt`` returns is laid over the request's own
   ``model_kwargs``, so a key the model reads back there (a prompt layout, a
   stream lead) must be set in every mode and listed in ``INTERNAL_KWARGS``: the
   data worker drops a client's value for it before ``process_prompt``. Qwen3-TTS
   lists its prompt layout, speaker, language, clip digest and codec stream lead.

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
  ``repetition_penalty``, ``min_p`` and ``ignore_eos``. The conductor fills in the
  per-request seed. ``min_p`` follows the HF processor order (after the penalty and
  temperature, before top-k/top-p) and needs ``enable_min_p=True`` on the node's
  ``SamplerSpec``, which adds the filter to that node's captured sampler only.
  Further knobs, each needing a ``SamplerSpec`` capability (CUDA only):

  - ``repetition_window`` (``max_repetition_window``): the penalty counts a token's
    occurrences in the last N generated tokens, ``repetition_penalty ** n``.
  - ``min_tokens`` (``min_tokens_stop_ids``): those ids are barred until
    ``min_tokens`` tokens exist; applied after top-k/top-p.
  - ``top_p_first``, ``top_p_min_keep`` (``enable_top_p_first``): HF's order (top-p,
    then top-k) instead of FlashInfer's, and HF's ``min_tokens_to_keep``.

  ``SamplerStep(apply_filters=False)`` samples a step with top-k/top-p off (e.g. a
  TTS's first code); declare it the same way on every step of a walk.
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

Paged KV layouts
~~~~~~~~~~~~~~~~

A model with a paged KV pool states, for every request, what its walks write into each
label of the pool. The KV manager turns that layout into the request's footprint, the most
pages each label can hold, before the request runs. A model declares its labels once and
returns each request's layout from ``process_prompt``.

``kv_labels(self) -> dict[str, dict[str, KVLabel]]``
   Return ``{resource_key: {label: KVLabel}}`` for every label of every paged pool. A
   :class:`mstar.model.base.KVLabel` has two fields:

   - ``growth``: ``(walk, loop)`` pairs, each a walk that keeps writing after admission,
     bounded by the named loop. Only a named loop can bound growth, since an unnamed one
     is renamed in every process.
   - ``keys``: a :class:`mstar.model.base.PrefixKeys`, or ``None``. See **Cross-request
     prefix reuse** below.

   A label written only by spans and scratch, such as a guidance branch, is ``KVLabel()``.

``process_prompt`` returns, under ``kv_layout`` in its metadata, a
:class:`mstar.model.base.LabelLayout` for every label the request opens: the labels its
``KVReqConfig`` can name on any node and walk, ``main`` when it names none. A layout may be
empty where an opened label is never written. It may also name labels the request opens
only on another deployment, as an unguided BAGEL generation does for a CFG-parallel one.
A layout has five fields:

- ``spans``: the committed writes known at intake, in walk order. A
  :class:`mstar.model.base.Span` has a ``kind``: ``"ids"`` for token ids from the tensor
  ``source`` names, ``"digest"`` for KV made from an uploaded file (``source`` is
  ``(modality, index)``), and ``"rows"`` for KV made from anything else, such as another
  node's outputs. ``length`` is the slots the write fills, ``advance`` how far it moves
  the position counter. A span's ``scratch`` is the most one non-committing step writes
  before the span commits into those slots, as a window's denoise steps do.
- ``growth``: ``(loop, slots)`` pairs, ``slots`` per iteration of ``loop``.
- ``scratch``: the most KV slots one non-committing step writes past the committed length.
- ``retention``: the ``RetentionPolicy`` the label's committing steps declare.
- ``keyed_spans``: how many leading spans the label's keys cover, by default every span
  before the first ``rows`` span.

The footprint adds the spans, the growth at each loop's iterations and the scratch, and
rounds up to whole pages per label. A span's own scratch counts only with the spans before
it, so a window's denoise steps and its commit count once. Under a ``RetentionPolicy`` it
takes the window's peak instead. ``KVReqConfig.validate`` refuses a request whose footprint
is bigger than its pool with a 400 at intake, and ``KVReqConfig.loop_room`` cuts the default
length of a request that asks for none until it fits. With every loop at 0 iterations, the
same sum is the prompt a request reserves when its pool admits it.

``prompt_spans(plan, texts, text_walk, item_spans)`` builds the common case: a prompt's
text and attachments in ``prefill_plan`` order. Qwen3.5 lays out its one label with it:

.. code-block:: python

   layout = LabelLayout(
       spans=tuple(prompt_spans(prefill_plan(parts), segments, walk, image)),
       growth=(("decode_loop", 1),),
   )
   return ProcessPromptOutput(result, {"kv_layout": {KV_CACHE: {"main": layout}}})

``get_loop_caps(self, **model_kwargs) -> dict[str, int]``
   The iterations each named loop may run for this request. The conductor refuses a request
   whose cap is past its loop's graph ``max_iters``, or whose layout at its caps holds more
   positions than its pool's ``max_seq_len``, with a 400. A request that sets no length has
   the default ``max_output_tokens`` cut to the context its prompt leaves, and a stop check
   reads that length from ``request_info.max_tokens``, which carries the cut. A model that
   reads a length from another key lists it in ``LENGTH_KWARGS``. Speculation never runs past a
   cap. A model whose stop
   check ends a loop at a count returns that count, so the cap and the stop check end the
   loop on the same iteration; ``test/modular/test_model_kv_layouts.py`` checks this for
   every model. Growth on an uncapped loop is counted to its graph ``max_iters``.

Where ``process_prompt`` computes a length that a walk computes another way, such as a
media item's rows, the walk asserts the two agree: BAGEL's ViT walk against
``vit_patches``, Qwen3-Omni's vision encoder against the processor's grid.

Loading fails in three cases, and a request fails in two:

- A paged pool has no labels. The message is ``<Model> declares no labels for its paged
  pool '<key>'; add them to `kv_labels`, and return each request's layout for them from
  `process_prompt```. A ring pool needs none.
- A label's growth or keys name a walk the model never runs. The message is ``<Model>
  declares '<key>'/'<label>' over ['<walk>'], which it never runs; its walks are [...]``.
- A label grows on a loop its walk doesn't run. The message is ``<Model> declares
  '<key>'/'<label>' growing on '<loop>', which '<walk>' doesn't run; its loops are [...]``.
- A request's layout leaves out a label the request opens, or lays out one the model
  doesn't declare. The request fails with a 500, ``<Model> returned no kv_layout for
  '<key>'/'<label>', which the request opens``, or ``<Model> returned a kv_layout for
  '<key>'/'<label>', which it does not declare``.

A request that takes a page past its footprint logs once per label, with the layout that
under-counted. A request is held only to the labels it opens, and to no pages on a label
it doesn't open. A page there logs the same line, reported apart from a real overrun on
that label so neither hides the other. Under ``MSTAR_KV_DEBUG_ASSERTS=1`` each commit is also
checked against the layout: the walk that wrote each span, growth past its bound, scratch
past its declaration and the retention a step declares. At removal, a label left a page
or more under its footprint logs the most pages it held.

Cross-request prefix reuse
~~~~~~~~~~~~~~~~~~~~~~~~~~

Two requests that begin with the same tokens fill the same KV pages, and the second can
attend the pages the first filled instead of recomputing them. A model opts a label in by
giving it ``keys``, and names its checkpoint. A model with no keyed label is never cached.
The deployment switch and the per-request opt-out are described under **Resource
overrides** in :doc:`serving`.

``PrefixKeys(tensor, walks, growth=False)``
   - ``tensor`` names the request tensor whose entries the layout's ids spans are keyed
     by, as ``process_prompt`` returns it.
   - ``walks``: only a write from one of these walks keeps the keys going. A write from
     another walk ends them, and the rest of the label is computed, never served.
   - ``growth``: the ids the label's growth walk samples extend the keys, so a later
     request that re-sends the reply matches it too.

   The keys cover the label's first ``keyed_spans`` spans. An ids span is keyed by its
   token ids, a digest span by its file.

   BAGEL keys its text and ViT walks and the decode that follows them:

   .. code-block:: python

      return {"kv": {"main": KVLabel(
          growth=((DECODE, "decode_loop"),),
          keys=PrefixKeys("text_inputs", ("prefill_text", "prefill_vit"), growth=True),
      ), ...}}

   Orpheus keys its prompt walk:

   .. code-block:: python

      return {KV_CACHE: {"main": KVLabel(
          growth=(("decode", "decode_loop"),), keys=PrefixKeys("text_inputs", ("prefill",)),
      )}}

``checkpoint_path(self) -> str | None``
   Return the local directory that holds the weights and ``config.json``. The cache
   hashes the checkpoint's safetensors index and config into the root of every key, so
   naming the checkpoint is what keeps two builds off each other's pages. The default
   returns ``None``, which leaves the weights out of the root. A model without a
   checkpoint therefore needs ``prefix_cache_salt`` set to use the cache at all. Without
   one, the engine keeps the cache shut and logs why.

A recurrent layer can't skip tokens: its state after a token is all it keeps of the tokens
before it. On a node with a recurrent state pool, a hit is cut to the longest snapshot of
that state the pool holds under it. Name the KV cache in the pool's config
(``RecurrentStateConfig(kv_cache=...)``), and give the deployment ``prefix_slots`` under
the pool's ``resources:`` key. Each snapshot slot costs as much as a request slot.

- A prefill keeps the state at its last page boundary before its end, since a resend's KV
  hit always leaves the last page to recompute, and at its KV hit's last page
  boundary when that is past what it restored. A shared prefix is then served from the third request
  that shares it. A step longer than the largest captured bucket checkpoints less often, so that
  its checkpoints fit the captured steps' buffer, and keeps its boundaries on that interval.
- A decode step that ends a page keeps the state there, so a later turn that re-sends the
  reply restores it. It replaces the request's previous decode snapshot once the KV chain
  keys its page; a step whose token never reaches the chain, such as the one speculated
  past the reply's end, leaves the previous one in place.
- A snapshot is keyed by the KV chain's key at its page. A full set of slots evicts the
  least recently used snapshot that no request holds.

Snapshots stay off without ``prefix_slots``, and also above one rank, without a named KV
cache, when that cache isn't on every node of the pool, or when its ``prefix_cache`` is
off or can't open at the pool's world size; in those cases the pool allocates no snapshot
slots. Neither does a pool whose cache the engine keeps shut for another reason, such as a
model that names no checkpoint and sets no salt. Every cache on the node of a pool without
snapshots stays shut, and the engine logs which pool kept it shut.
``opening_prefix_caches`` in ``mstar/engine/resources/kv/config.py`` is that rule over the
specs. The engine evaluates it once, before it builds any resource, and opens exactly the
caches it returns; the conductor's routing and the API server's prompt hashing evaluate it
for each rank as that rank's engine does (``prefix_caches_by_rank``). Each spec answers for itself: ``NodeResourceSpec.prefix_cache_opens``
says whether the resource keeps a prefix cache of its own, and ``restores_prefix`` marks
one whose state a hit must restore (the pool, and the resource over it), so another
resource takes part through its spec alone.

A model with a keyed label is checked at load time. Loading fails in two more cases:

- A position resource over a keyed cache uses a scheme other than sequential. A page is
  reusable only where its tokens land at the positions they were written at. The message
  is ``resource '<key>' places positions by <scheme>, but it sits over ['<kv key>'],
  which the model keyed for prefix reuse; either drop the stream or give the resource a
  sequential scheme``.
- A resource on a keyed node does not set ``prefix_skip_safe``, and is neither a
  recurrent state pool nor a resource over pools alone. A hit takes the front of the
  request away from every resource on the node, including resources that never saw those
  tokens. The message is ``resource '<key>' cannot serve a request whose prefix
  came from the cache, but it sits on ['<node>'], which the model keyed for prefix
  reuse; either drop the stream or give the resource the prefix hooks``.

Some requests stay uncached even when the model keys a label. The engine checks each
request's inputs on the keyed walk and cuts a prefix only from inputs that
``split_inputs`` can slice. A request whose inputs carry ``kwargs`` or
``resource_step_info`` is never probed; BAGEL's guided requests carry ``requires_cfg``
there. The default ``split_inputs`` slices only the sequence, so a request whose inputs
carry ``tensor_inputs`` is probed only when the submodule sets ``cuts_tensor_inputs``
and overrides ``split_inputs`` to cut those tensors. For a label keyed over several
walks, where a keyed node streams into a partition other than its own, a request that
runs that partition is not cached either, since a hit would skip the rows the node
streams for the tokens the cache holds. BAGEL
keys an understanding prompt's text and images, and a generation's first span when it
is text: an edit's images are resized before their walks see them, and a file's digest
doesn't cover the resize. A label under a sliding window files pages only up to its
protected prefix, which the window never releases.

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
  ``AttentionSpec``, ``PositionSpec`` and ``DSASpec`` depend on the ``kv_cache`` key they
  name. If a spec names a dependency that the model does not declare, loading fails
  immediately, rather than during a forward pass. ``RaggedAttentionSpec`` is the one
  attention spec that depends on nothing: it names no cache, because it has none.

The spec types are:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Spec
     - What it builds
   * - ``KVSpec(config=PagedKVConfig(...))``
     - A paged KV cache. ``KVConfig`` is the abstract base for common model geometry;
       ``PagedKVConfig`` adds ``max_seq_len`` and five fields that
       a deployment can tune: ``max_num_pages``, ``page_size``, ``cpu_offload_pages``
       (the number of pinned host pages used for offload; 0 disables offload),
       ``prefix_cache`` and ``prefix_cache_salt``. ``layout=KVLayout.MLA`` stores one
       Multi-head Latent Attention latent per token in place of K and V. It needs
       ``kv_lora_rank`` and ``qk_rope_head_dim``, and sets ``num_kv_heads`` to 1 and
       ``head_dim`` to their sum. Every rank holds the whole latent; only
       ``num_qo_heads`` shards, and MLA attention needs it set to the model's query heads.
       A spec whose nodes can run on several workers, as Bagel's LLM nodes do under CFG
       parallel, sets ``leader`` to the node whose pool admits each request. Left unset, each
       worker's pool admits requests on its own.
   * - ``KVSpec(config=RingKVConfig(...))``
     - A fixed-capacity frame ring. It adds ``tokens_per_frame``, one
       ``RingKVLayerConfig`` per layer, and the deployment-tunable ``num_sessions``.
       Ring storage is currently paired with FlexAttention.
   * - ``BoundedKVSpec(config=BoundedKVConfig(...))``
     - Fixed-size per-request KV with sink + window retention, and its attention. A
       request's stream is an optional read-only source (e.g. a voice prompt's keys,
       shared by the requests that use it) followed by what it writes; ``retention`` maps
       the source length to the ``SinkWindow`` kept (the first ``sink`` entries and the
       last ``window``). Each step declares its rows' ``StreamPosition`` and writes
       (``BoundedKVStep``), so rows at different positions batch, and nothing is
       reserved or released per step. ``attend`` takes an optional per-distance score
       term (``rel_bias``) for relative-position attention. Float32 by default (Triton
       kernels). See `Bounded (sink + window) caches`_.
   * - ``AttentionSpec(config=AttentionConfig(kv_cache=...))``
     - Self-attention planned over the named cache. ``backend`` selects
       ``AttnBackend.AUTO`` (the default, which lets the engine pick the platform's
       paged backend, FlashInfer for now), ``AttnBackend.FLASHINFER``,
       ``AttnBackend.DENSE``, ``AttnBackend.FLEX`` or ``AttnBackend.FLASHINFER_MLA``.
       Leave it at ``AUTO`` unless the model needs a specific kernel. FlashInfer, dense
       and MLA attention require a ``PagedKVConfig``; FlexAttention requires a
       ``RingKVConfig``. ``DENSE`` runs one varlen pass over each stream's frozen
       prefix (gathered from the pages once) and the step's fresh K/V, eager only:
       FA3 (AITER on ROCm), or FlashInfer's ragged prefill where FA3 has no kernel for
       the GPU, the same choice the ragged resources make (``dense_or_flashinfer``), with
       ``flashinfer_backend`` picking its kernel. The engine plans and runs a tiny layout on
       that kernel when it builds the resource, so a backend the GPU or FlashInfer build
       cannot run stops the server with a ``ValueError`` naming it. Its wrappers share one
       workspace, taken at build.
       ``flashinfer_backend`` selects a kernel generation: ``"auto"``, ``"fa2"`` or
       ``"fa3"`` when the FlashInfer backend is selected. ``decode_tensor_cores``
       (default ``True``) runs one-token decode rows on FlashInfer's tensor-core
       kernel; ``False`` picks the CUDA-core decode kernel, which can be faster for
       MHA and narrow GQA (few query heads per KV head). Measure before changing it:
       Whisper and Qwen3-ASR declare ``False``. ``FLASHINFER_MLA`` is
       Multi-head Latent Attention over a ``KVLayout.MLA`` cache. It needs ``sm_scale``,
       the softmax scale of the model's original qk head dim, which the other backends
       refuse. Its ``run`` takes the absorbed query (``[tokens, heads, kv_lora_rank]``)
       with the rope part as ``q_pe``, and returns the output in the latent space, so
       the layer applies ``W_UV``. FlashInfer's kernel covers a 512-wide latent with a
       64-wide rope part, or none from FlashInfer 0.6.18; any other shape runs a torch
       fallback, which cannot be captured in a CUDA graph.
       ``sliding_window=N`` lets each query attend to the last ``N`` positions, its
       own included. It needs the FlashInfer backend and causal attention, and the
       pages behind the window stay allocated.
   * - ``DSASpec(config=DSAConfig(kv_cache=..., topk=...))``
     - DeepSeek sparse attention (DSA) over the named ``KVLayout.MLA`` cache. A step
       declares ``DSAStep()``, and the resource takes each request's rows from the KV plan.
       The model's indexer layers store their index keys (``store_keys``) and pick each
       query's top ``topk`` positions by the indexer's score (``select``); a layer that
       reuses an earlier layer's pick reads it with ``shared_selection``. ``attend`` runs
       absorbed MLA over just the picked latents. The keys sit in per-request buffers
       outside the paged pool and the selection runs per request on the host, so the
       resource is eager only, a prompt must fit in ``topk`` and past ``topk`` a request
       decodes one row per step. Once one row of a batch passes ``topk``, every row of it
       attends its pick. GLM-5.2 declares it under ``dsa_long_context``. Planned: the keys
       move onto the MLA cache's pages under its page table, so the allocator, eviction and
       prefix caching cover them, and selection gets fixed-shape buffers so decode captures.
   * - ``CrossAttentionSpec(config=CrossAttentionConfig(...))``
     - Attention over a paged context that is written once and never extended.
       ``backend`` defaults to ``AttnBackend.AUTO``. Only FlashInfer is implemented.
       A step whose rows each query one token runs FlashInfer's decode kernel, and
       ``decode_tensor_cores`` picks its variant as for ``AttentionConfig``.
       See `Cross-attention (encoder-decoder models)`_.
   * - ``RaggedAttentionSpec(config=RaggedAttentionConfig(...))``
     - Cacheless (ragged) varlen self-attention over the segments packed into one
       forward. Nothing is paged, and nothing carries to the next step.
       ``RaggedAttentionConfig`` holds ``num_qo_heads``, ``num_kv_heads`` and
       ``head_dim`` (all pre-sharding, as in ``KVConfig``), an ``sm_scale`` that
       defaults to ``head_dim ** -0.5``, ``backend`` (``RaggedAttnBackend.AUTO``, the
       default, ``RaggedAttnBackend.FLASHINFER``, or ``RaggedAttnBackend.DENSE``),
       ``flashinfer_backend``, the
       two CUDA-graph ceilings ``max_segments_per_request`` and
       ``max_tokens_per_request``, and ``compile_op``, described under `Cacheless
       attention (encoder towers)`_. ``compile_op`` is the model's declaration; the
       serving config's ``resources:`` overrides take only the backends and the two
       ceilings. ``DENSE`` runs FA3 ``flash_attn_varlen`` (AITER on ROCm)
       and falls back to FlashInfer if that kernel is missing. See `Cacheless attention (encoder towers)`_.
   * - ``RaggedCrossAttentionSpec(config=RaggedAttentionConfig(...))``
     - The cross-attention counterpart: one span of each request attends another
       (``RaggedCrossAttentionStep(pairs=...)``). Same config. See `Cross-attention
       between spans`_.
   * - ``RaggedBlockCausalAttentionSpec(config=RaggedAttentionConfig(...), block_size=...)``
     - Block-causal self-attention within each span (``AttentionStep(causal=False)``): causal
       between ``block_size``-token blocks, bidirectional within one. Same config. See
       `Block-causal attention within spans`_.
   * - ``PositionSpec(config=PositionConfig(kv_cache=...))``
     - Position tracking and RoPE. ``scheme`` is ``PosScheme.SEQUENTIAL`` or
       ``PosScheme.BLOCK``. The RoPE parameters are set here: ``rope_theta``,
       ``rope_scale``, and the Llama-3.1 parameters ``low_freq_factor``,
       ``high_freq_factor`` and ``old_context_len``. Layers rotate with
       ``apply_qk(q, k, label)``. ``pos_ids=`` rotates at positions the model passes
       instead of the label's plan, and ``k=None`` rotates one tensor. ``interleave``
       selects GPT-J pairing, ``rotary_dim`` a partial rotation, and ``rope_table_dtype``
       a cos/sin table built once (GLM-5.2 uses float32). A model with learned position
       embeddings also declares this spec, because it needs the position counter. Such a
       model never applies RoPE.
   * - ``SamplerSpec(vocab_size=..., enable_repetion_penalty=...)``
     - A sampler with its own per-request parameter buffers, philox stream, and optional
       seen-token mask. ``enable_repetion_penalty`` declares a capability, not an
       intention. It selects which kernel variant is recorded into the captured graph.
       Whether the penalty runs on a given step is decided from the
       ``repetition_penalty`` values of the resident requests.
       ``enable_min_p``, ``max_repetition_window``, ``min_tokens_stop_ids`` and
       ``enable_top_p_first`` are capabilities of the same kind. See
       ``SamplingReqConfig`` above. ``loopback_state`` declares per-request state a
       decode loop keeps on the device beside its last sampled token. See
       `Device loop-back`_.
   * - ``RecurrentStateSpec(config=RecurrentStateConfig(...))``
     - Fixed-size per-request state that carries between steps (an LSTM's ``h`` and
       ``c``, a cursor, a conv tail, a Mamba or gated-DeltaNet layer's conv window and SSM
       state, a streaming encoder's caches): one slot per request and label, ``labels``
       naming every label a request's steps address or fork into (``("main",)`` by default)
       so that admission reserves a slot for each, ``blocks`` naming a shape and dtype for
       each piece (dtype None: the engine's compute dtype,
       what its KV caches store), ``num_layers`` copies of each, ``max_slots`` slots
       (deployment-tunable) with slot 0 the sink that padding rows address. A recurrent
       kernel reads the step's ``addressing``. ``declare_step`` gives each row a
       ``RecurrentStep`` segment with span >= 1. The forward reads its rows with
       ``pool.gather(name, start=...)`` and writes them back with
       ``pool.scatter_(name, value)``; both follow the step's plan, so there is no
       addressing to fetch, and eager and captured steps reach the same slots. A row with
       no saved state (a request's first step, or a padding row) reads ``start``, zeros
       when none is given. ``gather`` / ``scatter_`` need the sink, so leave
       ``disable_sink_slot`` unset. Parakeet's transducer decode is the example.

Orpheus declares four specs for its one autoregressive node. Its ``snac_decoder`` node
appears in no spec, so it receives no resources:

.. code-block:: python

   # mstar/model/orpheus/config.py
   KV_CACHE, ATTN, SAMPLER, ROPE = "kv_cache", "attn", "sampler", "rope"

   # mstar/model/orpheus/orpheus_model.py
   def get_node_resources(self) -> list[NodeResourceSpec]:
       kv_config = PagedKVConfig(
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
   * - An LLM with DeepSeek sparse attention (GLM-5.2)
     - The specs above over an MLA cache (``KVLayout.MLA``, ``AttnBackend.FLASHINFER_MLA``),
       plus a ``DSASpec`` over the same cache.
   * - A node that carries per-request state between steps of a loop (an RNN-T
       decoder, recurrent attention)
     - A ``RecurrentStateSpec``, read and written with ``gather`` / ``scatter_``.
   * - A node that samples with two different sets of parameters
     - Two ``SamplerSpec`` objects that both name the node, under different keys.
   * - An encoder tower that attends within the segments of one packed forward
     - A ``RaggedAttentionSpec``, and nothing else. No ``KVSpec`` and no
       ``PositionSpec``: there is no cache to page and no counter to advance. Needed
       only if the tower's attention must run inside a CUDA graph; see
       `Cacheless attention (encoder towers)`_.
   * - ViT, VAE or audio encoder, codec decoder, projection stage, combine stage
     - Usually nothing. Declare no spec that names the node, unless the node needs the
       row above.

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

.. _Cacheless attention (encoder towers):

Cacheless attention (encoder towers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ViT or audio encoder attends within its own sequence and keeps nothing afterwards.
Several images are packed into one forward, and each must attend only within its own
span. That is varlen, or "ragged", attention: the whole layout belongs to this step, and
nothing carries to the next.

Such a tower needs no resource at all if it calls a varlen kernel directly, passing
``cu_seqlens`` as an argument. Declare a ``RaggedAttentionSpec`` when the tower must run
inside a CUDA graph. The plan is what makes that possible: the engine plans the layout
outside the graph, into buffers whose addresses do not change, and the captured region
attends through them. A kernel that reads ``cu_seqlens`` as an argument, or builds its
mask from it, cannot be captured this way.

``RaggedAttnBackend.AUTO`` (and an explicit ``FLASHINFER``) plan through FlashInfer's
ragged prefill wrapper. ``RaggedAttnBackend.DENSE`` plans the same layouts into static
``cu_seqlens`` buffers and runs FA3 ``flash_attn_varlen`` (AITER on ROCm) instead, with
no page scatter. If that kernel is not importable, or the spec is a
``RaggedBlockCausalAttentionSpec`` (whose overlapping prefixes varlen cannot express),
the factory logs a warning and builds FlashInfer. The fallback does not cover a GPU or
dtype the kernel lacks: ``fa3-fwd`` 0.0.3 is built for Hopper (``sm_90a``) and runs bf16
but not fp16.

BAGEL's SigLIP2 tower is the reference implementation. The spec stands alone, over the
encoder node only, and names no cache:

.. code-block:: python

   RaggedAttentionSpec(
       resource_key=VIT_ATTN, nodes={"vit_encoder"},
       config=RaggedAttentionConfig(
           num_qo_heads=vit.num_attention_heads,
           num_kv_heads=vit.num_attention_heads,
           head_dim=vit.hidden_size // vit.num_attention_heads,
           # one image, hence one attending segment, per request
           max_segments_per_request=1,
       ),
   )

A layer binds the key as it binds any other resource, then attends through the resource's
``run(q, k, v, label=None)``. There is no KV write to pair it with, so no
``AttentionCallable`` is involved. A compiled caller breaks at ``run``; with
``RaggedAttentionConfig(compile_op=True)``, ``run`` traces as the ``mstar::ragged_attention``
custom op instead, so the region keeps one graph across the attention (worth it for a
compiled, uncaptured region; it costs startup time and capture memory where the region
is also captured). An eager call goes straight to the kernel.

The step is an ordinary ``AttentionStep``, and its ``segments`` are the entire layout.
There is no cache to read the layout off, so a label that carries no segment in the
declaration cannot be attended: the forward raises, rather than silently reusing the
previous step's plan. Because the tower is normally captured as a piecewise region, that
declaration lives in the region's ``declare_step``, and names no ``KVStep``. See
``mstar/model/bagel/submodules.py``, region ``"vit_block_loop"``, and `Piecewise CUDA
graphs (capturing an inner loop)`_.

Cross-attention between spans
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A diffusion transformer whose streams attend each other (video queries over text keys, or
audio over video) has several spans per request and no cache either. Declare each span as
its own label, and the cross-attentions as a ``RaggedCrossAttentionSpec`` beside the
``RaggedAttentionSpec`` for the self-attention: a resource is one kind of attention. Its
step names the ``(q_label, kv_label)`` pairs:

.. code-block:: python

   RaggedCrossAttentionStep(
       segments=segments,              # one Segment per (request, label) the pairs use
       pairs=(("video", "text"), ("audio", "video")),
   )

Each pair is planned as its own layout, pairing each request's ``q_label`` span with that
request's ``kv_label`` span; cross-attention is never causal. A layer attends through
``resource.run(q, k, v, label=cross_label(q_label, kv_label))``, passing the key
span's tokens as ``k`` and ``v``. One resource has one head geometry, so a model whose
attentions differ in head count or head dim declares one spec per (kind, geometry).
LTX-2.5 is the reference (``mstar/model/ltx2_5/submodules.py``).

Block-causal attention within spans
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Whisper-style encoder trained for streaming masks attention by chunk: a frame in chunk
``b`` sees every frame of chunks ``0..b`` and nothing later. Declare it as a
``RaggedBlockCausalAttentionSpec`` with the chunk length as ``block_size``, and step it like
any ragged self-attention, one segment per independently attending span, with
``AttentionStep(causal=False)``. Blocks are counted from each span's start, so a span need
not be a multiple of ``block_size``.

The prefixes overlap, which a ragged layout cannot express, so this kind runs FlashInfer's
paged prefill over the packed keys viewed as one-token pages, each query block listing its
prefix; nothing is copied. Under a captured graph the block count and page list are sized
for the worst layout the bucket's token count allows.

.. note::

   The eager varlen path is sometimes good to have. The BAGEL ViT has a head dimension of
   72, which the FlashInfer kernel — it supports only a few fixed head dimensions — forces
   us to zero-pad to 128. So that tower attends through the resource only when it is
   replaying a captured graph, and runs flash-attn varlen otherwise.

Bounded (sink + window) caches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A streaming decoder that keeps a fixed amount of context per request (a prompt or voice
prefix plus the last N steps, as StreamingLLM-style sinks or a streaming vocoder's chunk
caches do) declares a ``BoundedKVSpec``. A request takes a slot when it first runs and
keeps it; the window is a ring overwritten in place, so a step reserves and releases
nothing. A step's layout is at most five key ranges and three write ranges per row,
computed on the host from the position the step declares, which is also what lets rows at
different positions share a batch and a captured graph.

The model calls ``attend(layer_idx, q, k, v, source, plan)``: each row attends its
retained keys in stream order and the step's keys it writes (all of them for a row that
writes none), unmasked, and its writes land after. Pass
``source`` (``[sources, rows, H, L, 2 * head_dim]``, keys then values) when requests start
from shared prefixes that the slots should not copy; each row reads the one its
``StreamPosition.source`` names, so rows of different prefixes share a batch and a graph. ``rel_bias`` adds a term that depends only on
the query-key distance, for Transformer-XL-style relative positions. A streaming vocoder
like MiniCPM-o's token2wav fits three of them: its two conformer caches and its DiT's.

Two flags cover retention rules that are not quite a token FIFO.
``SinkWindow(window_includes_step=True)`` counts the step's own tokens toward the window.
``BoundedKVConfig(reverse_step_order=True)`` appends each step's tokens last-first, which
only makes sense for attention that ignores key order; it decides which of the oldest
step's tokens leave first.

By default the window is per step: every query of a step sees the same keys, the retained
ones and the step's own. A streaming encoder whose frames each see a fixed left
context (a conformer's ``att_context_size=[left, 0]``) sets
``BoundedKVConfig(retention=SinkWindow(0, left), query_window=left)`` instead: each query
attends the keys at distances ``0..left`` from it, so a step of several frames is causal
within itself and matches the offline mask. ``rel_bias`` is then indexed by that distance,
``[rows, H, T, left + 1]``, and passed unscaled (the kernel scales the sum by
``head_dim ** -0.5``). A per-query window takes no sink, and its right context is 0.
``retention`` may be a single ``SinkWindow`` when it does not depend on the source length.
``BoundedKVConfig(dtype=None)`` stores the cache in the engine's KV dtype, for layers that
compute in it. Nemotron VoiceChat's streaming encoder (``ENCODER_KV``,
``encoder_kv_config`` in ``mstar/model/nemotron_duplex/components/conformer.py``) is the
per-query example.

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
  times, and can stop earlier. A decode loop's ``max_iters`` is the model's context length,
  and each request's own length is its cap from ``get_loop_caps``. Give the loop a ``name`` so that a submodule's
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
           max_iters=self.config.max_position_embeddings,
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
   class attribute ``disable_torch_compile = True``. An encoder whose output feeds a
   prefix cache layout walk compiles with dynamic shapes instead. One that marks every
   dimension that varies with ``torch._dynamo.mark_dynamic`` in ``prepare_inputs`` sets
   ``marks_dynamic_dims = True`` and compiles as any other submodule. Keep the compiled
   paths compile-friendly. If a helper must not be traced, because it uses
   data-dependent Python control flow or forces a host synchronization, exclude it
   explicitly with ``@torch.compiler.disable``.

   Some submodules must run in their own parameter dtype. One example is an fp32 vocoder
   that is numerically sensitive. Such a submodule sets ``disable_autocast = True``. The
   engine then does not cast its parameters to the autocast dtype, does not wrap its
   forward in autocast, and explicitly disables any enclosing autocast. These two class
   attributes replace the removed ``get_stateless_flavor`` method.

   The engine runs float32 matmuls as TF32 in every process
   (``torch.set_float32_matmul_precision("high")``). A node that needs full fp32, for
   example a flow solve that amplifies TF32 rounding, sets
   ``float32_matmul_precision = "highest"``, and ``cudnn_allow_tf32 = False`` for cuDNN
   convolutions and RNNs. The engine applies these around that node's forward, its
   compile and its CUDA-graph capture, and restores the default afterwards, so other
   nodes keep TF32. ``None`` (the default) keeps the process setting. Don't flip
   ``torch.backends`` flags in model code: they are process-wide, and a CUDA graph keeps
   the precision it was captured with. The setting covers ``forward`` and
   ``forward_batched`` only, not ``preprocess`` or ``postprocess``. It can also be set
   per instance in ``__init__``, for example only when the model loads in fp32.

   Under TP or SP the engine sizes every resource it builds for a node (KV, attention,
   recurrent state) for one rank's slice of the heads. A node whose SP ranks split other
   work instead, such as one guidance branch per rank, sets ``splits_heads = False``,
   and its SP ranks keep all the heads (TP ranks still split them). Nodes sharing a
   resource must agree.

``postprocess(...)`` (optional)
   Metadata-only fixups that run on the GPU thread. This method must not read tensor
   values. Do not call ``.item()``, ``.cpu()`` or ``.tolist()`` here. This is a
   performance requirement, not a correctness requirement. Reading a value forces a host
   synchronization, which stalls the GPU thread and loses the worker's asynchronous
   scheduling overlap. Use this method only to rename outputs for routing. Put decisions
   that depend on tensor values in ``check_stop``.

``check_stop(...) -> set[str] | StopCheck`` (optional)
   Runs off the GPU thread, and may read tensor values. Return the names of the ``Loop``
   objects to stop, for example after seeing the EOS token. This is how a decode loop
   terminates. An AR submodule whose loop ends on a stop token sets
   ``self.token_stop = TokenStop(...)`` in ``__init__`` instead of writing one; that also
   gives it ``check_stop_batched``, which checks a whole step off one host copy of its
   tokens. The token budget is the loop's cap from ``get_loop_caps``, not a stop check.
   When a prefill walk outside the loop samples the reply's first token, list it
   in ``prefill_walks``; that token counts toward the budget, so ``get_loop_caps``
   gives the loop ``max_output_tokens - 1``. A stop id sampled
   there ends the loop at its first step, which withholds its ``buffer`` (see
   ``StopCheck`` below), so the loop has to emit the token under that name (GLM-5.2,
   Command A+).
   To also keep some of this step's outputs from being routed, return
   ``StopCheck(stops, withheld=(names...))``, for example a frame streamed alongside the
   EOS token: its consumer then never receives it, instead of finding and dropping it with
   a device read. ``check_stop_batched`` returns a ``BatchedStopCheck`` for the same.
   The Qwen3-TTS and Qwen3-Omni Talkers and Fish S2 Pro's AR keep their EOS frame out of
   the codec stream this way.

``on_request_added(self, request_id, request_info)`` (optional)
   Called on the worker's main thread when the request reaches the worker, before any of
   its steps, with the request's first ``CurrentForwardPassInfo``. For host work worth
   starting early on a thread of the submodule's own; it must not touch the device. The
   DiT scaffold draws a request's seeded noise here (``PREFETCH_SEEDS``).

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

``spanned_outputs(graph_walk) -> frozenset[str]`` (optional)
   Outputs whose valid length per row is known only on the device, such as audio with
   silence to trim. ``forward_batched`` returns a ``BatchedModelOutput`` whose
   ``output_spans`` maps the output's name to a ``[bs, 2]`` int64 tensor on the device:
   row ``i`` holds the first and one-past-last index of row ``i``'s valid samples along
   the output's last dimension. A step that returns spans returns one for every output
   declared here, since per-request forwards are merged by name. The forward keeps a
   fixed width and does no host read.
   The worker cuts each row to its span after the step's completion event, before the stop
   check, transport or a streaming edge reads it. The cut is a view, not a copy. Engine-side
   ``postprocess`` still sees the uncut output. A row whose span is empty is left out of the
   store after the stop check, as a withheld output is (``StopCheck``): a streaming edge
   gets no item for it. A spanned output must not feed a node that the same worker
   speculates into.

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
- ``KVStep(commit=, eager_write=, combined_labels=, pre_forks=, post_forks=)``. See
  below.
- ``AttentionStep(causal=, context_only=)`` describes the attention plan.
  ``context_only=True`` has the queries attend to the stored context alone, not to the
  tokens this step appends, and the step's own ``segments`` then give the query counts.
  Only the MLA backend reads it; the others refuse it. A ``RaggedAttentionSpec``
  steps through this same type. There, the segments are the whole layout rather than an
  extension of a cache. See `Cacheless attention (encoder towers)`_.
- ``PositionStep(pos_ids=, advance=)``. With ``pos_ids=None``, positions are derived from
  the stream counters. Pass explicit ids, keyed by plan label, when the model computes
  positions itself.
- ``SamplerStep(apply_penalty=, prefill_tracked_tokens=, loopback_writes=)``.
  ``prefill_tracked_tokens`` initializes the repetition-penalty mask with the prompt
  tokens. Only the prefill step passes them, because the sampler tracks every token it
  samples afterwards. ``loopback_writes`` names the declared loop-back states this step's
  forward stores (see `Device loop-back`_).

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
- ``eager_write=True``: the step's tokens for this cache are written outside any captured
  graph, for example in ``preprocess`` before the replay. Their addressing stays an
  ordinary plan rather than the slot's static buffers, which are sized for the bucket's
  in-graph tokens. Whisper's prefill uses it to write its 1500-token encoder context
  while replaying a graph sized by its few prompt tokens.

**Writing K/V yourself.** A layer that writes its K/V inside a fused kernel of its own,
rather than through ``write_kv``, asks the KV resource where the step's tokens go:
``kv.write_target(layer_idx, label)`` returns ``(layer cache, page of each token, slot in
page of each token, number of tokens to write)`` for the current plan (NHD layout only).
Tokens past that number are the plan's padding and must not be written. Qwen3-ASR's
fused q/k norm, RoPE and KV-write kernel (``Qwen3Attention``) is the example.

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

``ModelInputsFromEngine`` carries six other useful fields:

- ``step`` is this step's ``SubmoduleStep``. A forward that must match its own declaration
  reads it here instead of computing the same information again.
- ``captured`` is true when this forward runs under a CUDA-graph capture or replay. This
  field replaces the old ``cache_manager.is_captured``. It is useful when ``preprocess``
  packs its inputs differently for the fixed capture shape.
- ``synthetic`` is true when every row is a dummy request, during CUDA-graph capture or
  its warmup. Use it to keep statistics such as routing counts free of dummy traffic.
- ``real_seq_len`` is the token count of the real requests. Their rows come first; any
  rows after them are padding that a replay adds to reach the bucket shape. It is
  ``None`` when nothing was padded, so every row is real. Read it instead of assuming
  anything about the padding rows' own inputs.
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

.. _Device loop-back:

**Device loop-back.** A decode loop normally routes its sampled token back to itself as a
tensor per request. A node that sets ``reads_device_loopback = True`` can instead read
each row's last sampled token off the sampler on the device. It names those signals in
``device_loopback_signals(graph_walk)``, so the worker routes them without a tensor, and
reads them in ``preprocess`` with ``sampler.loopback_tokens(request_ids)``. A row whose
last step did not keep its value on the device (its first decode step, after the prefill)
still brings it routed; put it over the one read from the device with
``sampler.overwrite_rows(rows, (ids, routed_ids))``.

State that a decode loop carries from step to step beside the token, such as Whisper's
timestamp rule state, is declared on the sampler, not kept in the model:

.. code-block:: python

   SamplerSpec(
       resource_key=SAMPLER, nodes={"decoder"}, vocab_size=vocab,
       loopback_state={"ts_rules": LoopbackState(
           width=4, dtype=torch.long, default=tuple(inactive_state()),
       )},
   )

The sampler keeps one row per request by slot, grows it with its other per-request
buffers, and resets a row to ``default`` when its slot is given to a new request. A step
reads the rows with ``sampler.loopback_state(name, request_ids)`` in ``preprocess`` and
stores the next ones from the forward with ``sampler.store_loopback_state(name,
request_ids, values)``. Under a CUDA-graph slot the store is captured into a per-step
buffer and commit persists it, so the step must list the name in
``SamplerStep(loopback_writes=...)``. An eager step writes the rows by request. Both paths
address rows the same way, and the model never sees a slot index.

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

**Combined walks.** By default a step holds one walk. A model lets several walks of a node
share a step by returning ``{node: {combined_walk: {walks}}}`` from
``get_combined_graph_walks``, for example decode rows beside a prompt's prefill. The
combined walk is a name of its own, which ``max_batch_size`` and the CUDA-graph configs
can key on. A step of one walk keeps that walk's cap. A step that mixes walks takes the
combined walk's ``max_batch_size`` when it has one, else the smallest cap of the walks it
holds, so a small prefill cap never limits a decode-only step.
``max_rows_in_combined(self, graph_walk)`` also bounds how many rows of one walk a mixed
step may hold, for kernels sized to a few prompt rows (see ``CombinedSplit``).

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
- ``compile_mode`` is that ``torch.compile``'s mode. The default, ``None``, takes the
  deployment's compile level (``mstar serve -O``, ``MSTAR_COMPILE_LEVEL``); naming a
  mode takes the config out of that level. ``default`` keeps Inductor's fusion but takes cuBLAS for
  every GEMM and skips autotuning. ``deterministic`` is the engine's own: Inductor's
  defaults in its deterministic mode, which picks kernel configs by rule rather than timing
  them on the device, so every process compiles the same kernels (a torch without the mode
  logs a warning and compiles with the defaults). A compile outside the runner takes the same
  options from ``accelerator_graph_runner.compile_kwargs("deterministic")``, or
  ``deterministic_inductor_options()`` where it builds its own; the Wan VAE decode, the DiT
  scaffold's ``compile_blocks`` and Cosmos3's denoise step do. Building the config refuses
  an unknown mode, and one that turns on Inductor's own CUDA graphs (``reduce-overhead``,
  ``max-autotune``), which fail inside the runner's capture.
- ``required`` makes every bucket in the config mandatory. If capture fails locally or
  on another participating rank, warmup raises after rank-wide agreement instead of
  dropping the bucket and falling back to eager execution. The default is ``False``.

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
  cache declares a ``KVStep`` and an ``AttentionStep`` here. A cacheless region that uses
  ragged attention declares only an ``AttentionStep``, against its
  ``RaggedAttentionSpec`` key.
- ``lease_before_step`` makes the runner take this region's CUDA-graph slot before the
  outer ``declare_step`` runs, and report the slot in that call's ``piecewise_leases``
  argument. The region still declares, plans and commits its own work. The lease only
  tells the outer declaration which resources are already taken, so that it can exclude
  them.
- ``can_lease`` has signature ``(inputs: list[NodeInputs]) -> bool``. It applies only with
  ``lease_before_step``, and lets the region refuse a batch for a limit that the
  capture bucket key doesn't express, such as Qwen3.5's per-request segment count. The
  engine then gives the region no lease for that step. Your forward must replay only
  when the region was leased, for example when the outer step is ``None``, and not
  whenever ``runner.can_run`` is true: a refused batch can still fit a bucket, and
  replaying it would declare the region's resources twice.
- ``one_row_per_step`` makes the runner lease, check (``can_run``) and replay the region as
  one row, whatever the step's batch size, under the step's first request. It is for a
  packed region that runs every request's segments as one ragged batch, such as a vision
  tower, and needs ``lease_before_step`` and ``capture_batch_sizes=[1]``. The region's
  ``declare_step`` then receives ``[first_rid]`` with every request's ``seq_lens``, so the
  whole step's segments count against that one request's ``max_segments_per_request``,
  which for this region is a per-step ceiling. Pair it with a ``can_lease`` that refuses a
  step with more.
- ``capture_batch_sizes`` lists the batch sizes to record. ``None`` uses the runner
  default.
- ``compile`` runs ``torch.compile`` on ``capture_fn`` before capture. The default is
  ``False``.
- ``compile_mode`` is that ``torch.compile``'s mode, as on ``AcceleratorGraphConfig``.
- ``required`` makes every declared shape mandatory. If any participating rank cannot
  capture one, warmup raises instead of leaving that shape on the eager path. The default
  is ``False``.

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

BAGEL's ViT tower is the packed, cacheless counterpart. See
``mstar/model/bagel/submodules.py``, region ``"vit_block_loop"``: a
``PiecewisePackedConfig`` whose region declares only ragged attention, with patch
embedding and the RoPE gathers left eager because they are data-dependent indexing.

.. _config-yaml:

Step 6 — Write a config YAML
----------------------------

A config maps nodes to GPU ranks. The value under ``model:`` is your registry key. Each
``node_groups`` entry assigns one or more ``node_names`` to ``ranks``. An entry can also
name specific ``graph_walks``, which is how prefill-decode disaggregation is expressed.
Every node of a walk must be placed by exactly one entry that applies to that walk; the
conductor refuses a config that places it twice or not at all.

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
   * - ``KVSpec`` with ``PagedKVConfig``
     - ``max_num_pages``, ``page_size``, ``max_seq_len``, ``cpu_offload_pages``,
       ``prefix_cache``, ``prefix_cache_salt``
   * - ``KVSpec`` with ``RingKVConfig``
     - ``num_sessions``
   * - ``BoundedKVSpec``
     - ``max_slots``
   * - ``RecurrentStateSpec``
     - ``max_slots``, ``state_dtype``, ``prefix_slots`` (see **Cross-request prefix reuse**)
   * - ``AttentionSpec``
     - ``backend`` (``auto`` / ``flashinfer`` / ``dense`` / ``flex`` / ``xpu_paged``),
       ``flashinfer_backend`` (``auto`` / ``fa2`` / ``fa3``), ``decode_tensor_cores``
       (``true`` / ``false``)
   * - ``CrossAttentionSpec``
     - ``backend`` (``auto`` / ``flashinfer``), ``flashinfer_backend``
       (``auto`` / ``fa2`` / ``fa3``), ``decode_tensor_cores`` (``true`` / ``false``)
   * - ``RaggedAttentionSpec``
     - ``backend`` (``auto`` / ``flashinfer``), ``flashinfer_backend``
       (``auto`` / ``fa2`` / ``fa3``), ``max_segments_per_request``,
       ``max_tokens_per_request``. The two ceilings size CUDA-graph buckets, which is a
       memory-against-coverage trade the deployment makes, not the model.

Tune the cache shape on the KV resource, not on the attention resource that reads it. For
example, ``configs/qwen3tts.yaml`` selects FA2 under ``talker_attn``,
``configs/cosmos3_nano.yaml`` sets the page count under its paged KV key, and
``configs/waypoint.yaml`` sets the resident world count under its ring KV key.

.. note::

   A top-level ``kv_cache:`` block is no longer read. It raises an error at load time,
   with a message describing the migration. Move the block under ``resources:``, keyed by
   the resource name.

Routing across data-parallel replicas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a node group has several replicas, the conductor sends each new request to the
replica where it is cheapest (see "Data-parallel routing" in :doc:`serving`). A resource
whose state makes a request cheaper on one replica than another takes part through its
spec, so models declare nothing for it.

**The hook.** ``NodeResourceSpec.routing_mirror(ctx)`` returns a ``RoutingMirror`` or
``None``, the default. The router asks once per set of node groups with replicas that
holds one of the spec's nodes, and ``ctx`` (a ``RoutingContext``) gives the replica count,
the walks of the worker graphs there that hold the spec's nodes, the resource's keyed
labels with the walks that key each (empty for a resource that is not a KV cache), and
whether every replica's engine opens its prefix cache, which the router derives with
``prefix_caches_by_rank``.
A mirror is conductor-side bookkeeping that follows the resource's own rules as far as
the conductor's inputs allow. The module that defines one must stay free of kernels,
since the conductor builds it.

**The mirror's methods.** Each is given the request's ``ResourceReqConfig`` for the
resource, which the conductor builds and validates before the pick.

- ``facts(config)`` reports the request's ``RoutingFacts`` on each replica: named counts,
  such as ``work`` (prompt positions) and ``kv_hit`` (of those, the ones predicted cached).
  The router adds every mirror's row on a replica field by field, then prices the sum
  once with ``routing_cost`` in ``mstar/conductor/routing.py``, one term per
  ``routing_weights`` key, each 0 when its facts are absent.
- ``add_request(config, replica)`` records the placement once the request is dispatched,
  and returns a handle.
- ``remove_request(handle, ran)`` releases it when the request leaves. ``ran`` says
  whether a pass came back for one of the request's partitions with a walk on the
  resource's node groups.

**Adding a resource.** A new resource adds its fields to ``RoutingFacts``, its mirror,
and its term to ``routing_cost`` and ``RoutingWeights``. ``ReplicaRouter`` changes only
if its fields are not counts that add, or if its mirror reads a model declaration other
than the KV ones, which then becomes a ``RoutingContext`` field that ``ReplicaRouter``
fills. ``KVSpec`` with a ``PagedKVConfig`` is the one implementation. Its mirror runs the
worker's own ``PrefixIndex`` per replica over a count of the pool's pages, takes each
request's lease, pages and index entries at dispatch, and exists only where a walk keying
the cache runs and ``opening_prefix_caches`` says the workers open it.

Preprocessing threads
~~~~~~~~~~~~~~~~~~~~~

The API server runs ``load_image`` / ``process_prompt`` and output encoding with torch's
intra-op pool. Torch sizes it at one thread per host core, which in a container can be
several times the CPUs the cgroup quota lets run, and an op waits on any of its threads
the scheduler has parked. The API server therefore defaults to a quarter of the CPUs it
may use (its affinity, capped by the cgroup ``cpu.max`` quota), at most 4. A model sets
its own size with ``PREPROCESS_TORCH_THREADS`` on the class (Qwen3.5 uses 4). A
deployment overrides either at the top level of its config; ``null`` keeps torch's
default:

.. code-block:: yaml

   model: "qwen3_5_4b"
   preprocess_torch_threads: 4

The setting applies only to the API server process. The conductor and workers keep
torch's default. It does not size video decoding: each torchcodec decode that asks
``media_limits.video_decode_threads`` (``Model.load_video`` does) runs FFmpeg with up to 8
threads, no more than the CPUs the process may use (the same affinity and cgroup quota).

.. _tensor-parallelism:

Tensor parallelism (sharding)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To shard a node across several GPUs, add ``tp_size`` to its ``node_groups`` entry and list
``tp_size`` ranks. The runtime splits the group's ranks into TP groups of that size and
builds one ``comm_group`` per shard.

A node is sharded only if its components are built from the tensor-parallel modules in
``mstar/model/components/distributed``: ``ParallelAttention``, ``ParallelCrossAttention``, ``ParallelGatedMLP``,
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

.. _Expert parallelism (MoE):

Expert parallelism (MoE)
~~~~~~~~~~~~~~~~~~~~~~~~

A sharded MoE node can split its routed experts in two ways. Under TP, the default, every
rank holds a slice of every expert. Under EP, each rank holds whole experts. A deployment
picks one per node group with an ``ep_config`` block; see :doc:`serving` for the keys. The
shared pieces are in ``mstar/model/components/moe_parallel.py``. The Qwen3-Omni Thinker
(``mstar/model/qwen3_omni/components/thinker.py``) is the reference user.

To support it in a model:

1. **Declare the node EP-enabled.** Add it to ``ep_enabled_nodes`` in
   ``get_default_sharding_config``. An ``ep_config`` on any other node fails at start-up.

   .. code-block:: python

      return ShardingConfig(
          groups=[], tp_enabled_nodes={"LLM"}, shard_dim={}, ep_enabled_nodes={"LLM"},
      )

2. **Build the MoE layers with** ``instantiate_moe_block``. Accept ``moe_parallel`` in
   ``get_submodule`` and pass it, with the node's ``tp_group``, down to each MoE layer.
   The helper returns a ``ParallelSparseMoeBlock`` or an ``ExpertParallelSparseMoeBlock``,
   whose weight loaders keep only the rank's slice or its own experts. ``moe_parallel=None``
   gives TP.

   .. code-block:: python

      self.mlp = instantiate_moe_block(
          hidden_size=cfg.hidden_size,
          moe_intermediate_size=cfg.moe_intermediate_size,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          norm_topk_prob=cfg.norm_topk_prob,
          comm_group=tp_group,
          moe_parallel=moe_parallel,
      )

3. **Optionally, mix in** ``MoeParallelMixin`` on the model's top-level module, and call
   ``self.init_moe_parallel(moe_parallel)`` at the end of its ``__init__``. It provides:

   - ``maybe_log_expert_load(synthetic)``, which implements ``log_expert_load_every``.
     Call it from eager code before each forward, usually from ``preprocess``, with
     ``synthetic=engine_inputs.synthetic`` so capture and warmup routing are discarded.
   - ``moe_token_valid(hidden_states, num_valid_tokens)``, which implements
     ``mask_padding``. It returns the ``token_valid`` mask for the MoE blocks, or
     ``None`` when masking is off. Pass the mask only to MoE layers, not dense MLPs.
     ``num_valid_tokens`` must be a ``(1,)`` device tensor that ``preprocess`` returns
     as a static input, so a replay can update it; fill it from
     ``engine_inputs.real_seq_len``. ``ThinkerSubmodule.preprocess`` shows the wiring.

The shared-expert block (``ParallelSparseMoeBlockWithSharedExpert``) has no EP variant, so
a node built from it stays TP-only.

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

The three LLM nodes share one set of resources: the same KV pool, attention, positions and
sampler. Each of those specs names all three nodes in its ``nodes`` field. The ViT encoder
declares one resource of its own, a ``RaggedAttentionSpec`` with no cache behind it, so
that its block loop can be CUDA-graph captured:

.. code-block:: python

   def get_node_resources(self) -> list[NodeResourceSpec]:
       nodes = set(self._LLM_NODES)   # {"LLM", "LLM_cfg_text", "LLM_cfg_img"}
       return [
           RaggedAttentionSpec(resource_key=VIT_ATTN, nodes={"vit_encoder"},
                               config=RaggedAttentionConfig(...)),
           KVSpec(resource_key="kv", nodes=nodes, config=self._kv_config(), leader="LLM"),
           AttentionSpec(resource_key="attn", nodes=nodes,
                         config=AttentionConfig(kv_cache="kv")),
           PositionSpec(resource_key="rope", nodes=nodes,
                        config=PositionConfig(kv_cache="kv",
                                              rope_theta=self.config.rope_theta)),
           SamplerSpec(resource_key="sampler", nodes=nodes,
                       vocab_size=self.config.vocab_size),
       ]

No spec names the VAE encoder, ``init_latents``, ``combine_cfg`` or the VAE decoder, so
those nodes receive no resources. The CFG nodes are always declared, but they are used
only when the config enables CFG-parallel mode, described under Step 6. A single-GPU
config never routes requests to them.

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
default. Both caches use 128-token pages and hold ``WhisperModel.MAX_CONCURRENT_REQUESTS``
(64) requests at full length, plus the sink page that every ``KVManager`` keeps out of
circulation. A sequence is at most ``max_target_positions`` (448) tokens, which is 4 pages
per request, so the self-attention cache has 64 * 4 + 1 = 257 pages, about 5 GiB for
large-v3 in bf16. The 2048-page default would use about 40 GiB. The fixed 30-second context
window is ``max_source_positions`` (1500) tokens, which is 12 pages per request held from
prefill to the last token, so the context cache has 64 * 12 + 1 = 769 pages, about 15 GiB.
Sizing the caches is only half of it: ``configs/whisper_large.yaml`` also sets
``max_concurrent_requests: 64``, so the conductor queues further requests instead of the
worker offloading decoding requests to admit them. ``WhisperModel.validate_config_yaml``
refuses at startup a cap that either cache cannot hold after the YAML's ``resources:``
overrides, through ``mstar.model.utils.check_kv_pools_fit``, which any model with KV caches
can call the same way.

The decoder node's ``declare_step`` shows how a write-once context is expressed with spans.
Canary's comes from ``EncoderDecoderNode`` (``mstar/model/components/encoder_decoder.py``);
Whisper's declares its own, since it writes the context eagerly and captures prefill. The
context segments have a non-zero span in the prefill step that writes them, and a span of
0 in every later step. The ``commit`` in the prefill step converts the reservation into
resident pages that the later steps read.

Worked example: an image DiT on the diffusion scaffold
------------------------------------------------------

Flow-matching image and video transformers (FLUX-family DiTs, Z-Image, Wan-style
video) share one graph shape, so M* ships it as a scaffold in
``mstar/model/components/diffusion/``; a model adds only its exact component
ports, its checkpoint remap and a few hooks.

Graph and Walks::

    text_encoder ──text_embeds──▶ Loop("denoise_loop", dit) ──latents──▶ vae_decoder ──▶ EMIT image
    vae_encoder  ──ref_latents──▶ (image_edit only)

    encode_text  : text_inputs  -> text_embeds  (persist)
    encode_image : image_inputs -> ref_latents  (persist)
    image_gen    : Sequential[Loop(dit) -> vae_decoder]
    image_edit   : image_gen with ref_latents as an extra Loop input

``latents`` is the loop-back edge; every other input is re-injected each
iteration. The step index is the engine's loop counter
(``fwd_info.dynamic_loop_iter_counts``), so there is no host sync per step.

The ``dit`` node subclasses ``DenoiseLoopSubmodule`` (``denoise_loop.py``), which owns
the engine contract — seeded initial noise, per-request schedules, equal-key request
batching (``can_batch``), stacked ``forward_batched``, ``check_stop`` at the request's own
step count, the ragged-attention ``declare_step`` and one CUDA-graph bucket per
``bucket_key`` with the Euler update inside the graph — and asks the model for:

.. code-block:: python

    class MyDenoise(DenoiseLoopSubmodule):
        def bucket_key_for(self, fwd_info): ...       # hashable: latent grid, text length, ...
        def schedule_for(self, fwd_info, key): ...    # FlowMatchSchedule (flow_match.py)
        def seed_latents(self, fwd_info, key, gen): ...  # [L, C] initial noise, seeded
        def request_inputs(self, fwd_info, inputs, key): ...  # per-row conditioning tensors
        def num_tokens(self, key): ...                # image tokens for the attention step
        def capture_request_inputs(self, key, device): ...   # dummy rows for graph capture
        def denoise(self, engine_inputs, key, latents, timestep, sigma, sigma_next, **cond): ...
        # optional: extra attention spans (a refiner over the image tokens alone, ...)
        def attention_segments(self, key): return (("main", self.num_tokens(key)),)

``bucket_key`` is whatever has to match for two rows to share a forward, and it is also
the CUDA-graph bucket key — shape, but equally a knob like whether CFG is on.

A multi-step scheduler that carries solver state between iterations (wan22's UniPC, with
``unipc_model_outputs`` and ``unipc_last_sample``) names it in ``SOLVER_STATE``; those
become loop-back edges alongside ``latents``, seeded by ``seed_loop_back`` and returned
from ``denoise`` as a ``{name: tensor}`` mapping. A scheduler needing another per-step
scalar adds a ``STEP_SCALARS`` entry naming the attribute of its ``FlowMatchSchedule``
to slice it from — the base stages that tensor on the device and slices it per step, so
nothing in the batching path has to change.

A model whose ``seed_loop_back`` depends on nothing but the bucket key and the request's
seed sets ``PREFETCH_SEEDS = True`` (klein and Z-Image do): the base then draws the noise
when the request reaches the worker, on a thread of its own, and iteration 0 only copies
it to the device. Otherwise the CPU draw runs on the GPU thread at iteration 0, once per
joining row.

Every span a layer attends over must be declared with its own label (default: one
``"main"`` segment per request); ``ragged_for(label)`` hands that span's kernel to the
layers, which fall back to SDPA when no ragged resource is bound.

Shared pieces to build the components from: ``flow_match.py`` (shift schedules and the
exact Euler step), ``rope.py`` (multi-axis rotary embeddings), ``attention.py`` (joint
attention over packed tokens: SDPA eagerly, the ragged FlashInfer resource under graphs),
``text_encoder.py`` (native Qwen3 hidden-state taps), ``autoencoder_kl.py`` (the KL VAE
family), ``image_io.py`` (latent packing, PNG) and ``lora.py`` (static LoRA merge at load,
also into fused ``q``/``k``/``v`` shards via ``FusedColumnLinear.shard_slice``).

On the API side ``DiffusionImageAdapter`` (``mstar/api_server/openai/adapters.py``) serves
``/v1/images/generations`` and ``/v1/images/edits`` for any such model (``size`` becomes
``width``/``height``, ``seed`` and other knobs pass through); register it under the
model's registry key. The SDK exposes ``generate_image`` and ``edit_image``.

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
  and the partitions that produce data into it. Every walk belongs to exactly one
  partition.
- Route cross-partition tensors with ``StreamingGraphEdge(next_node=..., name=...,
  target_partition=...)`` instead of a plain ``GraphEdge``.

The consuming partition's ``get_partition_forward_pass_args`` reads
``incoming_connections``, which provides token counts and a ``producer_done`` flag, and
uses them to decide when to run.

Keep a walk that consumes a stream to one node or one ``Loop`` on one node group, and move
later work into a downstream partition fed by its own ``StreamingGraphEdge``. A consumer
whose every input is streamed runs again as soon as the next chunk arrives, and the
runtime and conductor count passes per worker graph, so the conductor refuses at startup
such a consumer sharing its worker graph with another top-level node or ``Loop``, and a
stream-consuming walk split over worker graphs, unless every consumer also waits on a
non-streamed input and all the walk's worker graphs run on the same workers. That input
must arrive once per pass (from the conductor, say); startup cannot check this.

A consumer whose walk should follow the producer's progress, rather than a count fixed at
ingest, is *producer-triggered*: the producer decides the consumer's walk. Set the same
``consumer_walk`` function on every connection into the consumer, all from one producer
partition. The producer worker calls it for each row of a pass that emits on those edges,
with a ``ProducerWalkCtx``: the walk that row ran (its own, even in a step that mixed
walks), the node that just completed, the iteration each enclosing dynamic loop is on,
and the producer's full ``CurrentForwardPassInfo`` (``step_metadata`` and all; the hook
runs on the producer). The row's streamed items run under the walk it returns. The
context is reused across calls, so a hook must not keep it.

- Each streamed item carries its assigned walk, and a chunk never spans two walks, so
  these connections need a ``FixedChunkPolicy``. A client stream cannot set
  ``consumer_walk``.
- The consumer's worker switches walk before popping a chunk, and only once the partition
  has no pass under way. The conductor learns each pass's walk from the worker's
  ``WORKER_GRAPHS_DONE``.
- A walk the consumer enters on its own, with conductor inputs, takes effect when they
  arrive, so send them only once the stream has moved the consumer through the walk
  before it.

Qwen3-Omni's Talker (``_talker_walk``) is the reference: Thinker prefill passes run
``talker_prefill``, the first decode iteration runs ``talker_last_prefill``, and later
ones run ``talker_decode``; the conductor feeds ``talker_decode`` its first input once
``talker_last_prefill`` has run.

Streams from the client
~~~~~~~~~~~~~~~~~~~~~~~

A model that listens while it runs (a full-duplex speech model, say) takes its live input
as a stream the client feeds. Declare it like any other connection, with the reserved
producer ``CLIENT``:

.. code-block:: python

   from mstar.streaming.topology import CLIENT, Connection

   Connection(from_partition=CLIENT, to_partition="Encoder", edge_name="audio_in",
              chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1))

``CLIENT`` is not a partition: it has no walks and no worker, and it must not appear in
``partitions``. The node that lists ``audio_in`` in its ``input_names`` (with
``consumes_stream=True``) consumes it through a ``StreamBuffer`` and the connection's
chunk policy, exactly as it would a stream from another partition. Its partition
self-triggers from the stream, so its forward-pass args send no inputs. That node must be
the only one, in any walk, that takes the edge; the conductor checks this at startup, and
that the model implements ``process_stream_chunk``. A partition may take several client
streams (audio and video, say): each has its own buffer and its own end.

- ``process_stream_chunk(chunk)`` turns each append the client sends into stream items,
  keyed by client edge name. ``chunk`` is a ``ClientChunk`` (``mstar.model.base``):
  ``text``; ``tensors`` and ``input_metadata``, the append's files loaded as for a request
  (``audio_inputs`` and so on); ``streams``, per ``tensors`` list the client stream each
  file names (``None`` where the client named none, so the model decides);
  ``model_kwargs``, the append's own; and ``request_kwargs``, the opening message's.
  Raise ``ValueError`` for an append the model doesn't take (the client gets a 400). An
  append of text alone that yields no item is refused rather than dropped, but the data
  worker can't tell which items text sent with files yielded: refuse such text yourself
  if the model takes none. Keep it stateless; the
  consuming node keeps its own state, in its resources or ``request_state``.
- Tensors ``process_prompt`` returns under a client edge's name are the stream's first
  chunk, so a one-shot request (one uploaded file, then nothing) is a stream of one chunk.
- A client stream may declare limits, ``Connection(..., limits=ClientStreamLimits(...))``
  (``mstar.streaming.topology``), which the client-stream layer enforces before any of
  the model's code runs: ``max_items``, what the stream may carry over a call's life,
  each item counting ``item_size(item)`` (1 by default; a stream of variable-length
  chunks counts their frames or samples, named by ``unit``). A call past
  ``max_items`` ends with a 400 that names the limit, so the consumer need not check a
  session cap of its own. Declare the most the consumer can take: a deployment may
  lower a declared bound under ``client_streams:`` (see :doc:`serving`), but not raise
  it.
- A stream ends when the client ends it: ``end_input: true`` ends all of the call's
  streams, a list of names only those. The consumer gets ``producer_done`` for that edge
  and a final chunk, and a loop fed by the stream ends as any stream-terminated loop does.
  Over ``/generate/ws`` the client appends and ends input explicitly (see
  :doc:`clients`); every other route ends the input as soon as the request is submitted.
  ``/generate/ws`` is on the Python frontend only, not ``--rust-frontend``.

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
         [ ] kv_labels, kv_layout from process_prompt (+ get_loop_caps) — with a paged KV pool
         [ ] (optional) keys on a KVLabel + checkpoint_path — cross-request prefix reuse
   [ ] mstar/model/registry.py                    — (module_path, class_name) in MODEL_REGISTRY
         [ ] HF_MODELS too, if weights come from Hugging Face
   [ ] configs/<your_model>.yaml                  — node_groups → ranks (+ resources: overrides)
   [ ] mstar/cli/main.py DEFAULT_CONFIGS          — else `mstar serve <name>` says "unknown model"
         [ ] the client-example tuples lower down (cosmetic, user-visible)
   [ ] docs/models.rst                            — one row in the registry-key table
   [ ] (optional) mstar/api_server/openai/adapters.py — only for OpenAI-compatible routes
   [ ] (optional) pyproject.toml extra            — if it needs deps the base install lacks
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
