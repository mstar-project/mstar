Model resources
===============

Declare state through ``Model.get_node_resources()`` and per-step work through
``NodeSubmodule.declare_step()``. The engine owns allocation, planning, admission,
capture buffers, eviction, and commit. Start with the resource declarations of your
chosen `scaffold <model_examples.html>`__; use this reference when its resource needs
differ.

.. _Step 2a — Declare your resources:

.. _step-2a-declare-your-resources:

Declare resources per node
--------------------------

``get_node_resources`` returns a flat list of ``NodeResourceSpec`` objects. Every spec
has these common members:

- ``resource_key`` is the name of this resource. Three places use this name, and all
  three must agree: the layers bind to it, ``declare_step`` uses it as the key of its
  per-resource steps, and a deployment YAML tunes it under ``resources:`` (see `Map nodes to ranks <model_deployment.html#config-yaml>`__). Existing models define these names as
  module-level constants in the model’s ``config.py``, for example
  ``KV_CACHE = "kv_cache"`` and ``ATTN = "attn"``, so that all three places read one
  definition.
- ``nodes`` is the set of graph-node names that share this resource. When two nodes name
  the same resource, they share one object. For example, BAGEL’s ``LLM`` node and its
  CFG branch nodes share one KV pool. A node that no spec names receives no resources.
  Such a node was called “stateless” in earlier versions.
- ``depends_on()`` returns the keys of other specs that this spec is built against.
  ``AttentionSpec``, ``PositionSpec`` and ``DSASpec`` depend on the ``kv_cache`` key
  they name. If a spec names a dependency that the model does not declare, loading fails
  immediately, rather than during a forward pass. ``RaggedAttentionSpec`` is the one
  attention spec that depends on nothing: it names no cache, because it has none.
- ``combined_splits()`` returns a ``CombinedSplit`` for each combined walk this spec
  runs on split kernels (see `combined walks <model_capture.html#combined-walks>`__);
  the default is none. The engine reads only this hook to order a split walk’s rows, so
  a spec whose config carries ``splits`` must return them here. ``AttentionSpec`` and
  ``LinearAttnSpec`` return their config’s ``splits``.

Two lockstep-parallel nodes on one worker (TP, SP or CFG nodes placed on the same ranks)
must not share a resource whose state rank 0 decides for the other ranks outside the
step that uses it: a resource implementing ``supports_eviction``,
``lease_cached_prefix`` or ``page_trail``, which the paged KV cache does with or without
host offload. The other ranks replay each decision with the next step of the node it was
recorded under, so two nodes on one such resource could admit out of step. The conductor
refuses that placement (``Resource.journals_resident_moves`` says which resources it
covers); give each node a spec of its own, or place the nodes on different workers.

Choose the state or attention contract your node needs. Each entry below states its
configuration and limitations.

Paged KV cache
~~~~~~~~~~~~~~

``KVSpec(config=PagedKVConfig(...))``

A paged KV cache. ``KVConfig`` is the abstract base for common model geometry;
``PagedKVConfig`` adds ``max_seq_len`` and five fields that a deployment can tune:
``max_num_pages``, ``page_size``, ``cpu_offload_pages`` (the number of pinned host pages
used for offload; 0 disables offload), ``prefix_cache`` and ``prefix_cache_salt``.
``layout=KVLayout.MLA`` stores one Multi-head Latent Attention latent per token in place
of K and V. It needs ``kv_lora_rank`` and ``qk_rope_head_dim``, and sets
``num_kv_heads`` to 1 and ``head_dim`` to their sum. Every rank holds the whole latent;
only ``num_qo_heads`` shards, and MLA attention needs it set to the model’s query heads.
A pool keeps only a window of the layers a windowed attention over it runs
(``AttentionConfig.sliding_window``, below); its ``page_size``, ``max_num_pages`` and
``cpu_offload_pages`` then work as described there. A spec whose nodes can run on
several workers, as Bagel’s LLM nodes do under CFG parallel, sets ``leader`` to the node
whose pool admits each request. Left unset, each worker’s pool admits requests on its
own.

Frame-ring KV cache
~~~~~~~~~~~~~~~~~~~

``KVSpec(config=RingKVConfig(...))``

A fixed-capacity frame ring. It adds ``tokens_per_frame``, one ``RingKVLayerConfig`` per
layer, and the deployment-tunable ``num_sessions``. Ring storage is currently paired
with FlexAttention.

BoundedKVSpec
~~~~~~~~~~~~~

``BoundedKVSpec(config=BoundedKVConfig(...))``

Fixed-size per-request KV with sink + window retention, and its attention. A request’s
stream is an optional read-only source (e.g. a voice prompt’s keys, shared by the
requests that use it) followed by what it writes; ``retention`` maps the source length
to the ``SinkWindow`` kept (the first ``sink`` entries and the last ``window``). Each
step declares its rows’ ``StreamPosition`` and writes (``BoundedKVStep``), so rows at
different positions batch, and nothing is reserved or released per step. ``attend``
takes an optional per-distance score term (``rel_bias``) for relative-position
attention. Float32 by default (Triton kernels). See `Bounded (sink + window)
caches <model_resources.html#bounded-sink-window-caches>`__.

AttentionSpec
~~~~~~~~~~~~~

``AttentionSpec(config=AttentionConfig(kv_cache=...))``

Self-attention planned over the named cache. ``backend`` selects ``AttnBackend.AUTO``
(the default, which lets the engine pick the platform’s paged backend, FlashInfer for
now), ``AttnBackend.FLASHINFER``, ``AttnBackend.DENSE``, ``AttnBackend.FLEX`` or
``AttnBackend.FLASHINFER_MLA``. Leave it at ``AUTO`` unless the model needs a specific
kernel. FlashInfer, dense and MLA attention require a ``PagedKVConfig``; FlexAttention
requires a ``RingKVConfig``.

Dense attention
^^^^^^^^^^^^^^^

``DENSE`` runs one varlen pass over each stream’s frozen prefix (gathered from the pages
once) and the step’s fresh K/V, eager only: FA3 (AITER on ROCm), or FlashInfer’s ragged
prefill where FA3 has no kernel for the GPU, the same choice the ragged resources make
(``dense_or_flashinfer``), with ``flashinfer_backend`` picking its kernel. The engine
plans and runs a tiny layout on that kernel when it builds the resource, so a backend
the GPU or FlashInfer build cannot run stops the server with a ``ValueError`` naming it.
Its wrappers share one workspace, taken at build.

FlashInfer kernel selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``flashinfer_backend`` selects a kernel generation: ``"auto"``, ``"fa2"`` or ``"fa3"``
when the FlashInfer backend is selected. ``decode_tensor_cores`` (default ``True``) runs
one-token decode rows on FlashInfer’s tensor-core kernel; ``False`` picks the CUDA-core
decode kernel, which can be faster for MHA and narrow GQA (few query heads per KV head).
Measure before changing it: Whisper and Qwen3-ASR declare ``False``.

Captured decode with fmha
^^^^^^^^^^^^^^^^^^^^^^^^^

``decode_fmha`` (default ``False``) runs captured one-token decode rows on FlashInfer’s
trtllm-gen fmha kernel on Blackwell, with no FlashInfer plan per step; a geometry
trtllm-gen has no kernel for keeps the paged kernel, with a warning. Eager steps and
prompt rows stay on the paged kernels. Qwen3.5 declares ``True``.

Softmax scale and MLA
^^^^^^^^^^^^^^^^^^^^^

``sm_scale`` (default ``1/sqrt(head_dim)``) sets the softmax scale; FlashInfer and
``FLASHINFER_MLA`` read it, and the other backends refuse it. ``FLASHINFER_MLA`` is
Multi-head Latent Attention over a ``KVLayout.MLA`` cache. It needs ``sm_scale``, the
softmax scale of the model’s original qk head dim. Its ``run`` takes the absorbed query
(``[tokens, heads, kv_lora_rank]``) with the rope part as ``q_pe``, and returns the
output in the latent space, so the layer applies ``W_UV``. FlashInfer’s kernel covers a
512-wide latent with a 64-wide rope part, or none from FlashInfer 0.6.18; any other
shape runs a torch fallback, which cannot be captured in a CUDA graph.

Sliding-window layers
^^^^^^^^^^^^^^^^^^^^^

``sliding_window=N`` lets each query attend to the last ``N`` positions, its own
included, and the cache keeps only about the last ``N`` tokens of this attention’s
layers per request, returning the rest to the pool, so long requests take far fewer
pages. It needs the FlashInfer backend, causal attention, a paged NHD cache and
``layers=(...)``, the KV layer indices this attention runs (the ones ``set_layer_idx``
takes). The cache’s other layers keep every token, and the attentions over it that
declare no window run them (Command A+); each attention refuses a layer it doesn’t run.
The cache works this out from its attentions before anything is built
(``derive_kv_groups``); a window at least the cache’s ``max_seq_len`` keeps every token,
so it needs no ``layers`` and costs nothing.

Unsupported sliding-window configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Startup refuses a window over another kind of cache, a window without ``layers``, a
layer declared twice, two different windows over one cache (one per cache for now), a
window over every layer of a cache, other layers that no attention without a window
reads, cross attention over such a cache, and a backend other than FlashInfer; its
kernel must be ``auto`` or ``fa2``, or ``fa3`` on SM90, and from SM100 ``auto`` runs
``fa2``.

Sliding-window page sizing
^^^^^^^^^^^^^^^^^^^^^^^^^^

The window’s pages hold ``page_size × full layers / sliding layers`` tokens, which must
be whole. Left unset, the page (``tokens_per_page``) is the smallest whole one whose
pages of either group hold at least 64 tokens (192 for 13 full and 39 sliding layers,
184 for 16 and 46); a ``page_size`` in the YAML must be a multiple of
``sliding layers / gcd(full, sliding)``. A window is kept in whole pages of its layers
plus up to one more, so with few sliding layers among many full ones (1984-token sliding
pages at 31 full and 1 sliding) a short window saves little. ``max_num_pages`` and
``cpu_offload_pages`` count pages of every layer at the declared ``page_size``, or at
128 tokens when it is unset, so the pool’s bytes don’t move with the derived page.

Sliding-window integration limits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Such a cache doesn’t prefix cache yet, so its ``prefix_cache`` goes off (logged at
info), and a serving config’s ``prefix_cache: true`` is refused. It doesn’t fork, serve
a node on another instance (replicas included) or hand out ``write_slots`` yet either;
each of these is refused.

DSASpec
~~~~~~~

``DSASpec(config=DSAConfig(kv_cache=..., topk=...))``

DeepSeek sparse attention (DSA) over the named ``KVLayout.MLA`` cache. A step declares
``DSAStep()``, and the resource takes each request’s rows from the KV plan. The model’s
indexer layers store their index keys (``store_keys``) and pick each query’s top
``topk`` positions by the indexer’s score (``select``); a layer that reuses an earlier
layer’s pick reads it with ``shared_selection``. ``attend`` runs absorbed MLA over just
the picked latents. The keys sit in per-request buffers outside the paged pool and the
selection runs per request on the host, so the resource is eager only, a prompt must fit
in ``topk`` and past ``topk`` a request decodes one row per step. Once one row of a
batch passes ``topk``, every row of it attends its pick. GLM-5.2 declares it under
``dsa_long_context``. Planned: the keys move onto the MLA cache’s pages under its page
table, so the allocator, eviction and prefix caching cover them, and selection gets
fixed-shape buffers so decode captures.

CrossAttentionSpec
~~~~~~~~~~~~~~~~~~

``CrossAttentionSpec(config=CrossAttentionConfig(...))``

Attention over a paged context that is written once and never extended. ``backend``
defaults to ``AttnBackend.AUTO``. Only FlashInfer is implemented. A step whose rows each
query one token runs FlashInfer’s decode kernel, and ``decode_tensor_cores`` picks its
variant as for ``AttentionConfig``. See `Cross-attention (encoder-decoder
models) <model_resources.html#cross-attention-encoder-decoder-models>`__.

RaggedAttentionSpec
~~~~~~~~~~~~~~~~~~~

``RaggedAttentionSpec(config=RaggedAttentionConfig(...))``

Cacheless (ragged) varlen self-attention over the segments packed into one forward.
Nothing is paged, and nothing carries to the next step. ``RaggedAttentionConfig`` holds
``num_qo_heads``, ``num_kv_heads`` and ``head_dim`` (all pre-sharding, as in
``KVConfig``), an ``sm_scale`` that defaults to ``head_dim ** -0.5``, ``backend``
(``RaggedAttnBackend.AUTO``, the default, ``RaggedAttnBackend.FLASHINFER``,
``RaggedAttnBackend.DENSE``, or ``RaggedAttnBackend.CUDNN``), ``flashinfer_backend``,
the two CUDA-graph ceilings ``max_segments_per_request`` (``max_segments_per_step`` for
a ``one_row_per_step`` region) and ``max_tokens_per_request``, ``compile_op``,
``shard_over_sp`` and ``workspace_mb``. ``CUDNN`` runs torch’s cuDNN SDPA over
equal-length segments (see `Cacheless
attention <#cacheless-attention-encoder-towers>`__). ``DENSE`` runs FA3
``flash_attn_varlen`` (AITER on ROCm) and falls back to FlashInfer if that kernel is
missing. By default the heads are split across all TP × SP ranks of the node;
``shard_over_sp=False`` splits them across the TP ranks only, for an attention each
sequence-parallel rank runs whole over its own tokens (one over a replicated stream, or
a cross-attention from the rank’s query shard). ``workspace_mb`` is the floor of each
label’s FlashInfer workspace per slot (default ``MSTAR_WORKSPACE_BUFFER_MB``, 512, which
doesn’t apply once it is declared): a plan that asks for more at startup grows it. A
node with many labels can declare less, since a ragged prefill only uses it for split-KV
partials. ``compile_op``, ``shard_over_sp`` and ``workspace_mb`` are the model’s
declaration; the serving config’s ``resources:`` overrides take only the backends and
the ceilings. See `Cacheless attention (encoder
towers) <model_resources.html#cacheless-attention-encoder-towers>`__ for ``compile_op``.

RaggedCrossAttentionSpec
~~~~~~~~~~~~~~~~~~~~~~~~

``RaggedCrossAttentionSpec(config=RaggedAttentionConfig(...))``

The cross-attention counterpart: one span of each request attends another
(``RaggedCrossAttentionStep(pairs=...)``). Same config. See `Cross-attention between
spans <model_resources.html#cross-attention-between-spans>`__.

RaggedBlockCausalAttentionSpec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RaggedBlockCausalAttentionSpec(config=RaggedAttentionConfig(...), block_size=...)``

Block-causal self-attention within each span (``AttentionStep(causal=False)``): causal
between ``block_size``-token blocks, bidirectional within one. Same config. See
`Block-causal attention within
spans <model_resources.html#block-causal-attention-within-spans>`__.

BlockCausalAttentionSpec
~~~~~~~~~~~~~~~~~~~~~~~~

``BlockCausalAttentionSpec(config=AttentionConfig(kv_cache=...), block_size=...)``

The same mask over a paged KV cache: a row’s new tokens attend its cached blocks and
their own, and a step runs only new tokens. FlashInfer only. See `Block-causal attention
over a KV cache <#block-causal-attention-over-a-kv-cache>`__.

PositionSpec
~~~~~~~~~~~~

``PositionSpec(config=PositionConfig(kv_cache=...))``

Position tracking and RoPE. ``scheme`` is ``PosScheme.SEQUENTIAL`` or
``PosScheme.BLOCK``. The RoPE parameters are set here: ``rope_theta``, ``rope_scale``,
and the Llama-3.1 parameters ``low_freq_factor``, ``high_freq_factor`` and
``old_context_len``. Layers rotate with ``apply_qk(q, k, label)``. ``pos_ids=`` rotates
at positions the model passes instead of the label’s plan, and ``k=None`` rotates one
tensor. ``interleave`` selects GPT-J pairing, ``rotary_dim`` a partial rotation, and
``rope_table_dtype`` a cos/sin table built once (GLM-5.2 uses float32). A model with
learned position embeddings also declares this spec, because it needs the position
counter. Such a model never applies RoPE.

SamplerSpec
~~~~~~~~~~~

``SamplerSpec(vocab_size=..., enable_repetition_penalty=...)``

A sampler with its own per-request parameter buffers, philox stream, and optional
seen-token mask. ``enable_repetition_penalty`` declares a capability, not an intention.
It selects which kernel variant is recorded into the captured graph. Whether the penalty
runs on a given step is decided from the ``repetition_penalty`` values of the resident
requests. ``enable_min_p``, ``max_repetition_window``, ``min_tokens_stop_ids`` and
``enable_top_p_first`` are capabilities of the same kind. See `per-request sampling
parameters <#per-request-resource-parameters>`__. ``loopback_state`` declares
per-request state a decode loop keeps on the device beside its last sampled token. See
`Device loop-back <model_submodules.html#device-loop-back>`__.

RecurrentStateSpec
~~~~~~~~~~~~~~~~~~

``RecurrentStateSpec(config=RecurrentStateConfig(...))``

Fixed-size per-request state that carries between steps (an LSTM’s ``h`` and ``c``, a
cursor, a conv tail, a Mamba or gated-DeltaNet layer’s conv window and SSM state, a
streaming encoder’s caches): one slot per request and label, ``labels`` naming every
label a request’s steps address or fork into (``("main",)`` by default) so that
admission reserves a slot for each, ``blocks`` naming a shape and dtype for each piece
(dtype None: the engine’s compute dtype, what its KV caches store), ``num_layers``
copies of each, ``max_slots`` slots (deployment-tunable) with slot 0 the sink that
padding rows address. A recurrent kernel reads the step’s ``addressing``.
``declare_step`` gives each row a ``RecurrentStep`` segment with span >= 1. The forward
reads its rows with ``pool.gather(name, start=...)`` and writes them back with
``pool.scatter_(name, value)``; both follow the step’s plan, so there is no addressing
to fetch, and eager and captured steps reach the same slots. A row with no saved state
(a request’s first step, or a padding row) reads ``start``, zeros when none is given.
``gather`` / ``scatter_`` need the sink, so leave ``disable_sink_slot`` unset.
Parakeet’s transducer decode is the example.

LoraSpec
~~~~~~~~

``LoraSpec`` (built by ``module_lora_spec``)

Per-request LoRA adapters on the node’s LoRA-capable linears: a pool of device slots, a
pinned host tier, and the per-step addressing a LoRA kernel reads. Opt-in
(``NodeResourceSpec.opt_in``): declared in this list, built only when the serving YAML
has a ``resources: lora:`` block. One spec spans every node adapters reach, with a pool
per node. See `Per-request LoRA
adapters <model_resources.html#per-request-lora-adapters>`__.

Lean captures
~~~~~~~~~~~~~

A resource whose kernels can be left out of a step whose rows all allow it sets
``Resource.lean_capture``, answers ``lean_for(request_ids)`` from state every rank of a
lockstep instance sets alike (its ingests), and says which kinds of capture get its twin
(``lean_captures``). Beside each bucket whose step drives such resources, the engine
captures a twin for each set of them, keyed by that set (``BucketKey.lean``). A step
replays the twin of every resource its rows let run lean, else one with fewer of them
(largest set first, ties in declaration order), else the plain bucket; the engine names
the set in ``StepContext.lean``, eager steps included, and each resource checks its rows
again at plan. A resource whose callers read its plan inside the forward also overrides
``Resource.outside_step``, which the engine enters around compile warm-ups outside any
step (the LoRA pool’s runs them lean). Twins are sized, planned and captured after every
plain bucket, and a twin left out (under the capture budget, say) is logged at INFO and
is not a dropped bucket: its steps replay the plain bucket. Lean resources: the LoRA
pool, for steps none of whose rows names an adapter, in every kind of capture; and a
sampler (see ``SamplingReqConfig``), for steps whose rows all sample greedily, in
batched captures. A model declares nothing for these.

Orpheus declares four specs for its one autoregressive node. Its ``snac_decoder`` node
appears in no spec, so it receives no resources:

.. code:: python

   # mstar/model/orpheus/config.py: the keys its SampledTokenDecoder reads, and its attention's
   KV_CACHE, SAMPLER, ROPE = keys.KV_CACHE, keys.SAMPLER, keys.ROPE
   ATTN = "attn"

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
               enable_repetition_penalty=True,
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

+-------------------------------------------+-----------------------------------------------------------------------+
| Node                                      | Declare                                                               |
+===========================================+=======================================================================+
| Self-attending LLM (text decode, or an    | ``KVSpec``, ``AttentionSpec`` and ``PositionSpec``. Add a             |
| LLM used as a denoiser in a flow loop)    | ``SamplerSpec`` if the node samples.                                  |
+-------------------------------------------+-----------------------------------------------------------------------+
| Decoder of an encoder-decoder model       | The three or four specs above, plus a second ``KVSpec`` for the       |
|                                           | encoder context and a ``CrossAttentionSpec`` over that second cache.  |
+-------------------------------------------+-----------------------------------------------------------------------+
| An LLM with DeepSeek sparse attention     | The specs above over an MLA cache (``KVLayout.MLA``,                  |
| (GLM-5.2)                                 | ``AttnBackend.FLASHINFER_MLA``), plus a ``DSASpec`` over the same     |
|                                           | cache.                                                                |
+-------------------------------------------+-----------------------------------------------------------------------+
| A node that carries per-request state     | A ``RecurrentStateSpec``, read and written with ``gather`` /          |
| between steps of a loop (an RNN-T         | ``scatter_``.                                                         |
| decoder, recurrent attention)             |                                                                       |
+-------------------------------------------+-----------------------------------------------------------------------+
| A node that samples with two different    | Two ``SamplerSpec`` objects that both name the node, under different  |
| sets of parameters                        | keys.                                                                 |
+-------------------------------------------+-----------------------------------------------------------------------+
| An encoder tower that attends within the  | A ``RaggedAttentionSpec``, and nothing else. No ``KVSpec`` and no     |
| segments of one packed forward            | ``PositionSpec``: there is no cache to page and no counter to         |
|                                           | advance. Needed only if the tower’s attention must run inside a CUDA  |
|                                           | graph; see `Cacheless attention (encoder                              |
|                                           | towers) <model_resources.html#cacheless-attention-encoder-towers>`__. |
+-------------------------------------------+-----------------------------------------------------------------------+
| ViT, VAE or audio encoder, codec decoder, | Usually nothing. Declare no spec that names the node, unless the node |
| projection stage, combine stage           | needs the row above.                                                  |
+-------------------------------------------+-----------------------------------------------------------------------+

Two samplers on one node
~~~~~~~~~~~~~~~~~~~~~~~~

If a node’s forward samples more than once per step with different parameters, declare
one ``SamplerSpec`` per parameter set. Both specs name the same node. Qwen3-Omni’s
Talker node does this. The Talker LLM samples codec group 0. The CodePredictor samples
groups 1 to N-1, using its own vocabulary size and no repetition penalty:

.. code:: python

   SamplerSpec(resource_key=TALKER_SAMPLER, nodes={"Talker"},
               vocab_size=self.config.talker_text.vocab_size,
               enable_repetition_penalty=True),
   SamplerSpec(resource_key=CODE_PRED_SAMPLER, nodes={"Talker"},
               vocab_size=self.config.code_predictor.vocab_size,
               enable_repetition_penalty=False),

The forward looks up each sampler by key and calls ``.sample()`` on it:

.. code:: text

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
- ``query_kv_cache`` is the KV resource that drives the queries. Its plan defines how
  the step packs its queries. Set it to ``None`` if the query side caches nothing across
  steps. The packing is then taken from the cross-attention step’s own segments, with
  one entry per segment, in declaration order.
- ``context_label`` is the label under which the context is written. The default is
  ``"context"``.

Whisper (``mstar/model/whisper/``) is the reference implementation:

.. code:: python

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
The prefill step declares a non-zero span on the context cache’s ``KVStep``. Every later
step declares a span of 0. A span of 0 reads the stream without extending it.

The encoder K and V tensors are written once per request by the decoder’s own code, in
``whisper/components/decoder.py:write_cross_kv``. That method iterates over the layers
and writes into the context label:

.. code:: text

   for layer_idx, layer in enumerate(self.layers):
       k, v = layer.encoder_attn.compute_kv(encoder_states)
       layer.encoder_attn.context_kv.set_default_layer_idx(layer_idx)
       layer.encoder_attn.context_kv.write_kv(k, v, label=CONTEXT_LABEL)

A model with several context sources, such as “audio” and “image”, declares one
``CrossAttentionSpec`` per source. Each source is a separate resource, and the source
name is also the key that the layer binds to.

.. _Cacheless attention (encoder towers):

Cacheless attention (encoder towers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ViT or audio encoder attends within its own sequence and keeps nothing afterwards.
Several images are packed into one forward, and each must attend only within its own
span. That is varlen, or “ragged”, attention: the whole layout belongs to this step, and
nothing carries to the next.

Such a tower needs no resource at all if it calls a varlen kernel directly, passing
``cu_seqlens`` as an argument. Declare a ``RaggedAttentionSpec`` when the tower must run
inside a CUDA graph. The plan is what makes that possible: the engine plans the layout
outside the graph, into buffers whose addresses do not change, and the captured region
attends through them. A kernel that reads ``cu_seqlens`` as an argument, or builds its
mask from it, cannot be captured this way.

``RaggedAttnBackend.AUTO`` (and an explicit ``FLASHINFER``) plan through FlashInfer’s
ragged prefill wrapper. For a self-attention whose config names neither a backend nor a
``flashinfer_backend`` (``None``, the default; an explicit ``auto`` is FlashInfer’s own
choice), ``AUTO`` first reads the ``ragged_attention`` kernel table
(``mstar/kernels/tables/ragged_attention/``, see `Tuned kernel
configs <#tuned-kernel-configs>`__) for the rank’s head geometry on the GPU. An eager
plan runs the row its layout’s longest query segment falls in, so one slot can run a
different kernel from one layout to the next; a captured bucket runs the row of the
layout that first plans it. On ``sm_103`` the table names torch’s cuDNN SDPA
(``RaggedAttnBackend.CUDNN``) for klein 4B / 9B and Z-Image’s geometries (24, 32 and 30
heads of 128; Wan2.2-TI2V-5B’s 24 heads read klein 4B’s entry, LTX-2.5’s 32-head video
self-attention klein 9B’s): under a CUDA graph FlashInfer’s ``auto`` stays on FA2 there
(its cutlass plan can’t be replanned in graph mode, and its cuDNN path needs cuDNN
9.24), which is 3.5-5x slower from 1k to 64k tokens. An entry whose larger sizes no
served model has run bounds them with a numeric last row and a ``"*"`` row naming
``flashinfer``, the resource’s own default. A GPU of the architecture reads the file
whichever model it is, and the build-time log names the GPU the row was swept on. The
cuDNN wrapper runs a layout of equal-length segments, a DiT batch’s, as one dense
``[segments, S, H, D]`` call. A captured one replays only the layout it was planned
with, so a captured bucket takes it only when the step that plans it declares
``AttentionStep(fixed_layout=True)``, and stays on FlashInfer otherwise. The DiT
scaffold’s denoise step declares it (``DenoiseLoopSubmodule.attention_step``): every row
of a denoise bucket, padding rows included, spans the bucket’s tokens. An eager layout
of unequal segments runs FlashInfer. It is self-attention only. A config that names
``backend: cudnn`` fails at build where cuDNN can’t run it: on a cross-attention or
block-causal resource, under GQA, in a dtype other than bf16 / fp16, or at a head dim
FlashInfer pads. Where no table has the geometry, ``AUTO`` is FlashInfer with its own
``auto`` choice. ``RaggedAttnBackend.DENSE`` plans the same layouts into static
``cu_seqlens`` buffers and runs FA3 ``flash_attn_varlen`` (AITER on ROCm) instead, with
no page scatter. If that kernel is not importable, or the spec is a
``RaggedBlockCausalAttentionSpec`` (whose overlapping prefixes varlen cannot express),
the factory logs a warning and builds FlashInfer. The fallback does not cover a GPU or
dtype the kernel lacks: ``fa3-fwd`` 0.0.3 is built for Hopper (``sm_90a``) and runs bf16
but not fp16.

BAGEL’s SigLIP2 tower is the reference implementation. The spec stands alone, over the
encoder node only, and names no cache:

.. code:: python

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

A layer binds the key as it binds any other resource, then attends through the
resource’s ``run(q, k, v, label=None)``. There is no KV write to pair it with, so no
``AttentionCallable`` is involved. A compiled caller breaks at ``run``; with
``RaggedAttentionConfig(compile_op=True)``, ``run`` traces as the
``mstar::ragged_attention`` custom op instead, so the region keeps one graph across the
attention (worth it for a compiled, uncaptured region; it costs startup time and capture
memory where the region is also captured). An eager call goes straight to the kernel.

The step is an ordinary ``AttentionStep``, and its ``segments`` are the entire layout.
There is no cache to read the layout off, so a label that carries no segment in the
declaration cannot be attended: the forward raises, rather than silently reusing the
previous step’s plan. Because the tower is normally captured as a piecewise region, that
declaration lives in the region’s ``declare_step``, and names no ``KVStep``. See
``mstar/model/bagel/submodules.py``, region ``"vit_block_loop"``, and `Capture an inner
loop <model_capture.html#piecewise-cuda-graphs-capturing-an-inner-loop>`__.

Cross-attention between spans
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A diffusion transformer whose streams attend each other (video queries over text keys,
or audio over video) has several spans per request and no cache either. Declare each
span as its own label, and the cross-attentions as a ``RaggedCrossAttentionSpec`` beside
the ``RaggedAttentionSpec`` for the self-attention: a resource is one kind of attention.
Its step names the ``(q_label, kv_label)`` pairs:

.. code:: python

   RaggedCrossAttentionStep(
       segments=segments,              # one Segment per (request, label) the pairs use
       pairs=(("video", "text"), ("audio", "video")),
   )

Each pair is planned as its own layout, pairing each request’s ``q_label`` span with
that request’s ``kv_label`` span; cross-attention is never causal. A layer attends
through ``resource.run(q, k, v, label=cross_label(q_label, kv_label))``, passing the key
span’s tokens as ``k`` and ``v``. One resource has one head geometry, so a model whose
attentions differ in head count or head dim declares one spec per (kind, geometry).
LTX-2.5 is the reference (``mstar/model/ltx2_5/submodules.py``).

Called with ``return_lse=True``, a ragged or ragged cross-attention resource’s ``run``
on the FlashInfer backend also returns each query row’s log-sum-exp
(``[total_tokens, H]``, fp32, natural log) from the same plan, which is what merging
attentions over disjoint key sets needs; a compiled caller breaks there even with
``compile_op``. Under sequence parallelism LTX-2.5’s audio queries attend each rank’s
share of the video tokens, and ``attention_over_sharded_keys``
(``mstar/model/components/distributed/sequence_parallel.py``) merges them.

Block-causal attention within spans
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Whisper-style encoder trained for streaming masks attention by chunk: a frame in chunk
``b`` sees every frame of chunks ``0..b`` and nothing later. Declare it as a
``RaggedBlockCausalAttentionSpec`` with the chunk length as ``block_size``, and step it
like any ragged self-attention, one segment per independently attending span, with
``AttentionStep(causal=False)``. Blocks are counted from each span’s start, so a span
need not be a multiple of ``block_size``.

The prefixes overlap, which a ragged layout cannot express, so this kind runs
FlashInfer’s paged prefill over the packed keys viewed as one-token pages, each query
block listing its prefix; nothing is copied. Under a captured graph the block count and
page list are sized for the worst layout the bucket’s token count allows.

.. note::

   The eager varlen path is sometimes good to have. The BAGEL ViT has a head dimension of
   72, which the FlashInfer kernel — it supports only a few fixed head dimensions — forces
   us to zero-pad to 128. So that tower attends through the resource only when it is
   replaying a captured graph, and runs flash-attn varlen otherwise.

Block-causal attention over a KV cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under a block-causal mask a frame’s output never depends on a later block, so what a
stream has already computed stays valid and need not be run again. Declare a paged
``KVSpec`` and a ``BlockCausalAttentionSpec`` on it, with the block length as
``block_size``, and step only the new tokens: ``KVStep(commit=True)`` and
``AttentionStep(causal=False)``, the layer writing K/V as usual. A step’s new tokens
must start on a block, and may run several blocks: the KV plan’s rows expand into one
paged request per block, each reading its row’s prefix pages. CosyVoice3’s flow solves
its hops this way, one KV layer per DiT block per Euler step.

Bounded (sink + window) caches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A streaming decoder that keeps a fixed amount of context per request (a prompt or voice
prefix plus the last N steps, as StreamingLLM-style sinks or a streaming vocoder’s chunk
caches do) declares a ``BoundedKVSpec``. A request takes a slot when it first runs and
keeps it; the window is a ring overwritten in place, so a step reserves and releases
nothing. A step’s layout is at most five key ranges and three write ranges per row,
computed on the host from the position the step declares, which is also what lets rows
at different positions share a batch and a captured graph.

The model calls ``attend(layer_idx, q, k, v, source, plan)``: each row attends its
retained keys in stream order and the step’s keys it writes (all of them for a row that
writes none), unmasked, and its writes land after. Pass ``source``
(``[sources, rows, H, L, 2 * head_dim]``, keys then values) when requests start from
shared prefixes that the slots should not copy; each row reads the one its
``StreamPosition.source`` names, so rows of different prefixes share a batch and a
graph. ``rel_bias`` adds a term that depends only on the query-key distance, for
Transformer-XL-style relative positions. A streaming vocoder like MiniCPM-o’s token2wav
fits three of them: its two conformer caches and its DiT’s.

Two flags cover retention rules that are not quite a token FIFO.
``SinkWindow(window_includes_step=True)`` counts the step’s own tokens toward the
window. ``BoundedKVConfig(reverse_step_order=True)`` appends each step’s tokens
last-first, which only makes sense for attention that ignores key order; it decides
which of the oldest step’s tokens leave first.

By default the window is per step: every query of a step sees the same keys, the
retained ones and the step’s own. A streaming encoder whose frames each see a fixed left
context (a conformer’s ``att_context_size=[left, 0]``) sets
``BoundedKVConfig(retention=SinkWindow(0, left), query_window=left)`` instead: each
query attends the keys at distances ``0..left`` from it, so a step of several frames is
causal within itself and matches the offline mask. ``rel_bias`` is then indexed by that
distance, ``[rows, H, T, left + 1]``, and passed unscaled (the kernel scales the sum by
``head_dim ** -0.5``). A per-query window takes no sink, and its right context is 0.
``retention`` may be a single ``SinkWindow`` when it does not depend on the source
length. ``BoundedKVConfig(dtype=None)`` stores the cache in the engine’s KV dtype, for
layers that compute in it. The streaming FastConformer (``encoder_kv_config`` in
``mstar/model/components/fastconformer_stream.py``, Nemotron VoiceChat’s ``ENCODER_KV``)
is the per-query example.

Per-request LoRA adapters
~~~~~~~~~~~~~~~~~~~~~~~~~

A ``LoraSpec`` serves LoRA adapters per request over frozen base weights: each target
linear adds ``scaling * (x @ A^T) @ B^T`` for the adapter of the token’s request, so
rows of one batch can run different adapters, or none. The engine owns the adapter
weights. A model declares which linears adapters may touch; it never indexes the pool
itself.

A model declares LoRA from its own modules: the shared linears add the delta themselves
once bound, and ``module_lora_spec`` builds the spec by walking each node’s weights
root. The spec is opt-in (``NodeResourceSpec.opt_in``), declared in
``get_node_resources`` beside the model’s other resources and built only when the
serving YAML has a ``resources: lora:`` block. Orpheus is the reference
(``OrpheusModel.get_node_resources``):

.. code:: python

   from mstar.model.components.lora.modules import PER_SEGMENT, module_lora_spec

   def get_node_resources(self) -> list[NodeResourceSpec]:
       return [
           ...,  # KV, attention, positions, sampler
           module_lora_spec(
               # builds the module the node's weight loader fills, at one rank, on the
               # meta device: only once the spec is read, which a deployment without
               # a `lora:` block never does
               lambda group: OrpheusForCausalLM(self.config, comm_group=group),
               nodes={"LLM"},
               root_path="model",               # the root's attribute path in the node submodule
               adapter_roots=[("", "model.")],  # adapters saved on the bare LlamaModel
               rows={"lm_head": PER_SEGMENT},    # rows that are not the step's tokens
               embeddings=True,                  # every embedding call is in the forward
           ),
       ]

The weights root states how the checkpoint lands on it once
(``checkpoint_rules() -> CheckpointRules``: a prefix, a remap, the stacked-shard rules);
its weight loader and ``module_lora_spec`` both use them, so adapters are placed exactly
as weights are.

What takes a delta
^^^^^^^^^^^^^^^^^^

``ColumnParallelLinear`` (and ``MergedColumnParallelLinear``, ``QKVParallelLinear``,
``KVColumnParallelLinear``), ``RowParallelLinear``, ``FusedColumnLinear`` and
``VocabParallelEmbedding`` describe their own target from their own shard layout
(slices, heads replicated past their count, packs, per-rank padding:
``LoraCapable.lora_target``). Binding swaps a module to its family’s twin class, which
adds the delta after the base GEMM and before the module’s own gather or all-reduce
(after it, for the embedding) through ``apply_lora(delta, x, y) -> y``, the one call
site. A plain ``nn.Linear``, a ``Conv1d/2d/3d`` whose kernel is its stride, and
subclasses of either that keep their base’s forward are swapped alike, and are whole on
every rank. An unbound module keeps its class, so a deployment without a ``lora:`` block
runs, traces and guards exactly as before (tests hold the aten ops, the dynamo graph and
its guards). Anything else holding weight matrices (MoE experts, a router, a depthwise
conv, ``nn.Embedding``, a forward overridden on an instance or hooked) takes none: the
walk records why, and an adapter touching it is refused with that reason. So does a
child its parent runs through the weight itself rather than its forward (folded at load,
an ``F.linear`` over it), which the parent names in ``lora_unservable`` (a path below it
-> reason; a path that is not one of its submodules is a ``ValueError``): MLA’s absorbed
projections, SigLIP’s patch embedding, Command-A+‘s fused decoder-layer projections, the
Qwen3 talkers’ code-predictor heads. Nothing else can tell such a child from one whose
forward runs: an undeclared one is bound, and its delta silently never added, so code
that reads a linear’s weight directly declares it there.

Targets inferred from modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One per role in a layer stack (``<stack>.N.<role>``, the first ``ModuleList`` or
``Sequential`` in the path, whatever it is named; ``layers`` set where a role is on some
layers only), named by the role’s last component (``names`` overrides), plus one per
other module: one outside the stacks, or at a role whose entries differ in shape (UNet
levels, vision stages), which is named by its path but keeps its role for ``rows``.

What the model states
^^^^^^^^^^^^^^^^^^^^^

Only what modules cannot know. ``rows``: a target that sees one row per segment
(``PER_SEGMENT``: a head after the last-token select) or one per ``k`` folded tokens.
``exclude`` (with a reason): modules called outside the planned step, a text projection
or a conditioning encoder run in ``prepare_inputs``. ``embeddings=True`` only when every
embedding call is in the step. ``adapter_roots`` / ``aliases`` /
``adapter_stacked_params``: the prefixes, names and split projections adapters write,
rewritten to the checkpoint’s before the root’s rules place them. ``ignore``: modules an
adapter may carry that never run.

Several nodes
^^^^^^^^^^^^^

A model whose adapters reach several nodes (an LLM and its vision encoder, which a TP
config may put on different ranks) builds a spec per node with ``module_lora_spec`` and
declares one, ``join_lora_specs(llm, vision)``: its targets name their nodes, a
checkpoint module belongs to the node that places it (the one whose ``adapter_roots``
claim it most specifically, when several do), so one adapter file serves both, and each
node gets a pool of its own (``LoraSpec.per_node``, keyed ``lora@<node>``). The one
``lora:`` block sizes them all.

Binding and rows
^^^^^^^^^^^^^^^^

``NodeSubmodule.bind_node_resources`` binds every module placed on its node under
``root_path``, after checking it is the declared target and that the pool’s shard is its
rank-local shape (a module built whole on a TP node fails here, not in a forward). The
pool’s step is the node’s own segments (``Resource.steps_from_segments``): the engine
(``engine_step``) adds a ``ResourceStep`` of them under the pool’s key, the rows its
linears see, in order (a request may have several: Chatterbox’s cond and uncond copies;
Qwen3.5’s vision encoder: one segment per frame, the merger’s rows folding them). A step
that lays its rows out otherwise declares its own step under the pool’s key, which the
engine keeps. A step that is ``None`` or declares no segments gets none, and an adapted
module it runs refuses, so a node serving adapters returns a step with its segments
wherever one runs.

Only inside the step
^^^^^^^^^^^^^^^^^^^^

The pool holds a step’s addressing from its plan to its commit (or rollback), for the
thread that planned it, so a delta called outside a step (``prepare_inputs``, which the
async worker runs on the main thread while the GPU thread’s step is planned; an output
cached from an earlier step) refuses, eager or compiled, instead of adding another
step’s adapters to these rows. The engine’s own warm-ups outside any step
(``compile_regions``, ``warmup_compiles``) run as a step whose rows name no adapter: a
compile there traces no delta, and the first step with an adapter row compiles its own.
A piecewise region’s step takes its rows as the node’s does, from its own segments (an
``attn_key`` region’s: one per row), planned per replay and handing the enclosing step’s
plan back at its commit. A region replayed as one row per step (``one_row_per_step``)
addresses every row to the first request’s adapter, so the engine captures it with the
pool lean (``BucketKey.lean`` names the pool) and leases it only for a step none of
whose rows runs an adapter; a step with one runs the region eagerly under the node’s
step.

``module_lora_spec`` produces an ordinary ``LoraSpec``, whose parts follow.

Explicit target declarations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``LoraTarget`` gives the unsharded linear’s shape and how the base layer is split under
tensor parallelism. For a column-parallel linear, A is replicated and B is split by
output rows, per slice (a slice with fewer heads than ranks is replicated, as
``QKVParallelLinear`` does with K/V). For a row-parallel linear, A is split along the
input and B is replicated; the delta is a partial sum, added to the partial base output
*before* the all-reduce. LoRA adds no collective to the forward. ``layers`` restricts a
target to some layers of a hybrid stack. A checkpoint tensor that packs several of the
target’s slices, each split across ranks on its own, is named in ``slice_packs``
(Qwen3.5’s delta-net ``in_proj_qkv``: one A for the three, B cut by their widths), and a
base linear that pads a slice per rank for alignment says so in ``slice_align``.
``parallel="none"`` is a target whole on every rank, its delta added to an output that
is already whole.

Target kinds and rows
^^^^^^^^^^^^^^^^^^^^^

``kind="embedding"`` is a token embedding: ``in_features`` is the vocabulary, the layer
passes token ids as ``x``, and the shrink is a gather of A’s column per token
(``mstar::lora_gather_shrink``), PEFT’s ``lora_embedding_A`` / ``_B``; an id outside the
vocabulary (an image position’s ``-1``) adds nothing. ``per_segment`` is one row per
segment and ``token_fold=k`` one row per ``k`` consecutive tokens of a segment. The pool
plans an addressing per grouping its targets use; the call checks the rows it gets.

Checkpoint layouts
^^^^^^^^^^^^^^^^^^

Adapters are validated from the safetensors header: modules map to targets, shapes fit
them (an adapter for another model size is a 400, not an admit failure), and a conv
adapter whose kernel is its stride flattens to the linear over the patch. DoRA (PEFT’s
``lora_magnitude_vector``, Kohya’s ``dora_scale``), ``modules_to_save``, trainable
tokens and bias LoRA are refused by name. A YAML ``targets: [...]`` keeps a subset of
the model’s targets (dropping vocabulary-sized ones saves their slots); an adapter
touching a dropped target is refused, and its modules stay unbound. Without the list, a
deployment with ``runtime_updates`` off keeps the targets its configured adapters touch
(a node left with none gets no pool): the API server reads them once at startup
(``resolve_lora_adapters``) and the conductor, handed that config, spawns every worker
with the list, so every process builds the same pools; an adapter it cannot read fails
the deployment.

Requests
^^^^^^^^

Nothing per model. ``LoraSpec.request_config`` opens a request that names an adapter in
the ``lora_adapter`` kwarg, which the OpenAI ``model`` field maps onto, with a
``LoraReqConfig``, which each node’s pool takes; a base-model request carries none. The
API server refuses, with a 400, an adapter it does not serve (``ServedModels``), and the
conductor, with a 409, one a runtime load or unload is in flight for; an adapter is
checked when it is registered (rank within ``max_rank``, only declared targets, shapes
that fit), not per request. A request on an adapter is prefix-cached in its adapter’s
own namespace (see **Cross-request prefix reuse**).

Capture and compile
^^^^^^^^^^^^^^^^^^^

Nothing per model: a node serving adapters captures the same buckets and compiles the
same as one that does not. A capture plans every row on the zero adapter; a replay’s
plan writes each row’s slot into the buffers that capture recorded, so a captured graph
serves any mix of adapters without recapture. The kernels are the custom ops
``mstar::lora_shrink`` and ``mstar::lora_expand_`` (the latter declares the ``y`` it
writes), so ``torch.compile`` takes them as single nodes without a graph break. Under
Triton the launch grid depends on the token count and the slot count alone. The pool is
a lean resource (``Resource.lean_capture``; see `lean captures <#lean-captures>`__), so
the engine captures every bucket whose step drives it a second time with no LoRA kernel
in it: a step none of whose rows names an adapter replays that twin (or runs its deltas
as ``y`` untouched, eagerly), and the LoRA capture when the twin was not captured. The
choice reads only which adapter each request named, the same on every rank, and the pool
checks it again at plan; ``cg_key_info`` and ``declare_step`` still state only the
node’s own key.

dtype
^^^^^

The slot weights take the nodes’ compute dtype (the serving config’s ``autocast_dtype``,
e.g. ``float16``). A node that runs outside autocast sets ``LoraSpec.dtype`` to its
weights’ dtype.

Slot admission and lifetime
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A request takes its adapter’s slot on its first admit and holds it until it is removed.
Slot 0 is a zero adapter (rank 0) that base-model rows and capture padding rows address,
and the kernels skip it. A miss loads the adapter from the host tier into a free slot,
or into the unheld slot released longest ago.

Waiting for capacity
~~~~~~~~~~~~~~~~~~~~

A request whose adapter cannot get a slot (every slot held by running requests, or
reserved for requests admitted ahead of it) waits at the runner’s admission
(``LoraAdapterPool.ask`` / ``reserve``), in arrival order with every other resource that
asks, so it holds no KV pages meanwhile and the rows already running keep stepping; the
admit’s ``AllocationFailed`` stays only as a backstop. Only the first waiting request is
admitted, so a cold adapter is not starved by steady traffic on resident ones: requests
behind it wait too, and the held slots drain. On one GPU, where KV admits every request
as well, that holds back base-model requests too: a request waiting for a slot, or for
its adapter’s host copy, stops every newer admission on its node until a slot drains or
the read lands. Under tensor parallelism KV does not ask, so base requests go on past
it. The slot wait does not arise while the adapters in rotation fit in ``max_slots``,
and the copy wait is at most one read, which starts at the request’s ingest.

Host reads and reservations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A slot is reserved for a request only once its adapter’s host copy is in (even for an
adapter resident in a slot, which a miss admitted ahead may evict), so its admit does
not wait on a read there; a request whose adapter failed to read, and is in no slot, is
let through at once, and its admit fails it before handing out any slot, so it takes
none that a counted request needs. A reservation counts slots and pins none: which slot
an admit takes follows from the slot table alone.

Adapters that do not target this node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An adapter with nothing for a pool’s targets (a text-only adapter on a vision tower’s
pool) takes no slot there and runs no delta (its steps can replay the lean capture; see
**Lean captures**): the pool knows it from the file’s header at startup, or from a
runtime load’s read, and its rows address the zero adapter. Only the rank that chooses
batches admits: on a lockstep instance the leader, whose choice the followers run as
sent, so its reservations need no agreement across ranks.

Replay buffers and stream ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plan writes each step’s addressing (for each row grouping its targets read, per
token, per segment or per fold, the rows grouped by slot: the only thing the kernels
read) into one fixed buffer per capture slot, sized to its bucket, so a captured replay
reads stable addresses; a replay’s padding rows address the zero adapter and its tail
tokens no slot. The pool admits and plans on the GPU thread only (no pre-plan), in step
order, and every write it makes to the device (an adapter load, the addressing) goes on
the stream the forwards run on, so a load into a slot lands after every step queued
before it.

LoRA kernel backend
~~~~~~~~~~~~~~~~~~~

``backend`` picks the kernels a pool registers behind the custom ops, which every device
runs: ``"triton"`` (slot-grouped shrink and expand, the default on CUDA) or ``"torch"``,
which registers none and leaves the reference (no data-dependent shapes, the default
elsewhere). Registration is process-wide: once anything in the process has registered
the Triton kernels, every pool on CUDA runs them.

LoRA agreement across TP ranks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under tensor parallelism each rank of the node’s lockstep instance keeps its own pool
and makes its own slot decisions, with nothing journalled from rank 0, and they come out
the same on every rank: admits run in the order rank 0 broadcasts, and the worker lands
each teardown between the same two admits on every rank (the ``after_tp_seq`` stamp on a
forwarded remove, honoured on rank 0 by waiting for the admit of the step it is stamped
behind). That is why the LRU clock counts admits rather than commits, which may land on
either side of a teardown. Which adapters are resident, which slot a miss takes and
whether an admit refuses then follow from the same state on every rank.

Disk reads and rank-local failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reading an adapter from disk is the one step that can fail on one rank only, so the
ranks agree where each one’s copy stands (in, still being read, or failed) before any
waits for one, and every rank installs it or refuses the request: one gloo all-reduce
per admit with an adapter to install (all of its misses at once), a second when a rank
was still reading, none in steady state. Files are read on the pool’s own reader thread:
every rank starts reading an adapter into the host tier when it ingests a request that
names it, the leader reserves a slot for the request only once its copy is in, and an
admit never reads a file itself.

Follower read waits
~~~~~~~~~~~~~~~~~~~

A follower whose read is still in flight waits for it, unless another rank’s read has
failed. That wait is short when the follower’s read started at the same ingest as the
leader’s, the usual case, but it can be a whole read: a tier keeps what the ingests and
removes on its rank leave it, a follower takes removes from rank 0 and ingests from the
conductor, and a remove that races an ingest can leave a follower without a copy rank 0
kept. Neither the wait nor the agreement holds the pool’s lock, so the main thread’s
ingests and removes go on meanwhile.

Unsupported LoRA parallel layouts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A node that runs sequence parallelism (SP > 1) or CFG parallelism (cfg_size > 1) cannot
serve LoRA yet: the pool lays rows out per request over the whole batch, while each SP
rank holds a shard of the tokens and each CFG rank one guidance branch’s rows, so
building the pool refuses the deployment.

Runtime adapter changes
~~~~~~~~~~~~~~~~~~~~~~~

Adapters loaded and unloaded at runtime (``/v1/load_lora_adapter``) reach each worker as
a control message from the conductor, which each rank applies whenever it arrives rather
than at a step the instance agrees on. ``LoraAdapterPool.load_adapter_async`` only reads
the adapter into the host tier, on the reader thread (the worker acknowledges the load
from its main loop once every pool has it), and ``unload_adapter`` only stops new
requests from naming it; once no live request names it (immediately, or at the last
one’s teardown) the pool forgets its host copy and its name’s slot entry, and leaves the
slot’s weights and LRU standing as they were. Neither touches what an admit decides
from, and the next request that names the adapter is ingested after a reload, which
reaches every rank after the unload did. Nothing per model is needed for either.

Per-request resource parameters
-------------------------------

A resource spec is fixed at load time and is shared by every request. Parameters that
differ per request, such as sampling parameters and the set of cache labels a request
uses, come from a separate method:

.. code:: text

   get_request_resource_configs(
       self, input_modalities: list[str], output_modalities: list[str], model_kwargs: dict,
   ) -> dict[str, ResourceReqConfig]

The returned dict is keyed by ``resource_key``. The conductor calls this method once per
request, with the request’s modalities and kwargs as ``walk_plan`` gets them, and the
engine passes each config to its resource when the request is ingested. There are two
``ResourceReqConfig`` subclasses:

- ``SamplingReqConfig`` holds ``temperature``, ``top_k``, ``top_p``,
  ``repetition_penalty``, ``min_p``, ``penalize_prompt`` and ``ignore_eos``. The
  conductor fills in the per-request seed, derived per resource key so each sampler of a
  request draws its own stream. ``min_p`` follows the HF processor order (after the
  penalty and temperature, before top-k/top-p) and needs ``enable_min_p=True`` on the
  node’s ``SamplerSpec``, which adds the filter to that node’s captured sampler only.
  Further knobs, each needing a ``SamplerSpec`` capability (CUDA only):

  - ``repetition_window`` (``max_repetition_window``): the penalty counts a token’s
    occurrences in the last N generated tokens, ``repetition_penalty ** n``.
  - ``min_tokens`` (``min_tokens_stop_ids``): those ids are barred until ``min_tokens``
    tokens exist; applied after top-k/top-p.
  - ``top_p_first``, ``top_p_min_keep`` (``enable_top_p_first``): HF’s order (top-p,
    then top-k) instead of FlashInfer’s, and HF’s ``min_tokens_to_keep``.

  ``return_logprobs`` (``enable_logprobs``, any device) reports each sampled token’s
  log-prob under ``softmax(logits / T)``, before the penalty and filters, with T = 1 for
  greedy requests; it reaches the client as ``{"logprobs": [...]}`` in the chunk’s
  metadata. The model passes ``return_logprobs`` into its ``SamplingReqConfig`` (the
  conductor refuses the request otherwise), its forward samples once per step and hands
  over row-addressed stop buffers (``BatchedModelOutput.check_stop_buffers``): the
  log-probs ride the same copy to the host. The captured sampler scores every step once
  it is on, so it is off by default and a deployment turns it on with
  ``resources: {<sampler key>: {enable_logprobs: true}}``.
  ``logprobs_mode: "processed"`` (``enable_processed_logprobs``, CUDA) scores the
  distribution the token was drawn from instead: after the penalty, temperature, min-p,
  top-k/top-p and min-tokens floor. A greedy token scores 0 there, so a trainer sampling
  greedily wants ``"raw"``. It renorms every step’s distribution again, about 0.15 ms at
  batch 1 and 0.5 ms at batch 128 on a GB200 with a 248k vocabulary, so it is a separate
  switch.

  ``SamplerStep(apply_filters=False)`` samples a step with top-k/top-p off (e.g. a TTS’s
  first code); declare it the same way on every step of a walk.

  A step whose every row samples greedily (``samples_greedy``: temperature 0, no
  repetition penalty, no ``min_tokens`` and no log-probs) takes each row’s argmax
  instead of the full sampler’s draw, eagerly or in a twin of each batched capture (see
  `lean captures <#lean-captures>`__); any other step runs the full sampler. By default
  only a sampler whose ``vocab_size`` is past the one-kernel draw (4096) does this,
  since a narrower one draws in one kernel already. ``SamplerSpec.greedy_capture``
  overrides that: ``false`` turns it off, and with it the twins’ capture memory and
  time, for a deployment that never samples greedily, and ``true`` turns it on whatever
  the vocabulary. A deployment sets it in its ``resources:`` block. The model declares
  nothing for it.

- ``KVReqConfig`` holds ``needed_labels``, ``needed_labels_per_node`` and
  ``needed_labels_per_node_walk``. These name the cache streams that the request will
  actually read. In a PD-disaggregated deployment, a KV transfer then copies only those
  streams.

Orpheus shows the simple case. It has one sampler, and reads the parameters from
``model_kwargs``:

.. code:: python

   def get_request_resource_configs(self, input_modalities, output_modalities, model_kwargs):
       keys = ("temperature", "top_p", "top_k", "min_p",
               "repetition_penalty", "penalize_prompt", "ignore_eos")
       return {
           SAMPLER: SamplingReqConfig(
               **{k: model_kwargs.get(k, getattr(self.config, k)) for k in keys}
           )
       }

BAGEL (``bagel_model.py``) returns both config types. A BAGEL request uses
classifier-free guidance or does not, and that choice determines which cache labels the
request reads.

Sampling knobs mean the same thing on every model (`Generation
parameters <clients.html#generation-parameters>`__), and the server range-checks them
before your model sees them. A model’s part of that contract:

- Default each knob the request leaves unset from the checkpoint’s own
  ``generation_config.json`` (``mstar.model.utils.load_generation_defaults``), falling
  back to your config only when the checkpoint ships none.
- Raise ``ValueError`` (a 400) for an input you cannot serve; never clamp, coerce, or
  fall back silently. A ``SamplerSpec`` without ``enable_repetition_penalty`` or
  ``enable_min_p`` refuses those knobs at admission, so do not default a penalty onto
  it.
- Declare every ``model_kwargs`` key you read in ``request_kwargs()``; any other key a
  client sends is logged as ignored by the server. ``test_model_contracts.py`` fails for
  a registered model that does not override it.
- Bound a request’s length, or a denoise loop’s step count, through ``get_loop_caps``:
  the conductor refuses a cap past the loop’s ``max_iters`` instead of cutting the
  request short, so do not clamp it, or check it against a typed-in limit, yourself.
- Draw any model-local randomness from the request’s seed (``random_seed`` on the
  forward-pass info), never the global RNG.

A spec can also build its own request’s config from ``model_kwargs``
(``NodeResourceSpec.request_config``), which the conductor adds beside the model’s. That
is how a resource the deployment declares, such as a LoRA pool, takes per-request
settings without model code. A row with no config under a resource’s key may let that
resource run lean for the step; see `lean captures <#lean-captures>`__.

``get_sampling_config(node_name, model_kwargs)`` still exists as a helper for assembling
sampling parameters. The engine no longer reads it directly. Pass its result into a
``SamplingReqConfig`` here.

.. warning::

   Never pass sampling parameters into a captured forward as Python scalars. Their values
   are recorded into the captured kernel launch, and every replay then reuses the values
   from capture time. No error is raised. Parameters that reach the sampler resource
   through ``SamplingReqConfig`` are stored in buffers whose addresses do not change across
   replays, so they stay per-request and remain safe under CUDA graphs.

Paged KV layouts
----------------

A model with a paged KV pool states, for every request, what its walks write into each
label of the pool. The KV manager turns that layout into the request’s footprint, the
most pages each label can hold, before the request runs. A model declares its labels
once and returns each request’s layout from ``process_prompt``.

``kv_labels(self) -> dict[str, dict[str, KVLabel]]`` Return
``{resource_key: {label: KVLabel}}`` for every label of every paged pool. A
`mstar.model.base.KVLabel <https://github.com/mstar-project/mstar/blob/main/mstar/model/base.py>`__
has two fields:

- ``growth``: ``(walk, loop)`` pairs, each a walk that keeps writing after admission,
  bounded by the named loop. Only a named loop can bound growth, since an unnamed one is
  renamed in every process.
- ``keys``: a
  `mstar.model.base.PrefixKeys <https://github.com/mstar-project/mstar/blob/main/mstar/model/base.py>`__,
  or ``None``. See `cross-request prefix reuse <#cross-request-prefix-reuse>`__.

A label written only by spans and scratch, such as a guidance branch, is ``KVLabel()``.

``process_prompt`` returns, under ``kv_layout`` in its metadata, a
`mstar.model.base.LabelLayout <https://github.com/mstar-project/mstar/blob/main/mstar/model/base.py>`__
for every label the request opens: the labels its ``KVReqConfig`` can name on any node
and walk, ``main`` when it names none. A layout may be empty where an opened label is
never written. It may also name labels the request opens only on another deployment, as
an unguided BAGEL generation does for a CFG-parallel one. A layout has five fields:

- ``spans``: the committed writes known at intake, in walk order. A
  `mstar.model.base.Span <https://github.com/mstar-project/mstar/blob/main/mstar/model/base.py>`__
  has a ``kind``: ``"ids"`` for token ids from the tensor ``source`` names, ``"digest"``
  for KV made from an uploaded file (``source`` is ``(modality, index)``), and
  ``"rows"`` for KV made from anything else, such as another node’s outputs. ``length``
  is the slots the write fills, ``advance`` how far it moves the position counter. A
  span’s ``scratch`` is the most one non-committing step writes before the span commits
  into those slots, as a window’s denoise steps do.
- ``growth``: ``(loop, slots)`` pairs, ``slots`` per iteration of ``loop``.
- ``scratch``: the most KV slots one non-committing step writes past the committed
  length.
- ``retention``: the ``RetentionPolicy`` the label’s committing steps declare.
- ``keyed_spans``: how many leading spans the label’s keys cover, by default every span
  before the first ``rows`` span.

The footprint adds the spans, the growth at each loop’s iterations and the scratch, and
rounds up to whole pages per label. A span’s own scratch counts only with the spans
before it, so a window’s denoise steps and its commit count once. Under a
``RetentionPolicy`` it takes the window’s peak instead. ``KVReqConfig.validate`` refuses
a request whose footprint is bigger than its pool with a 400 at intake, and
``KVReqConfig.loop_room`` cuts the default length of a request that asks for none until
it fits. With every loop at 0 iterations, the same sum is the prompt a request reserves
when its pool admits it.

``prompt_spans(plan, texts, text_walk, item_spans)`` builds the spans of a prompt whose
text and attachments, in ``prefill_plan`` order, are each written by their own walk, as
BAGEL lays out its prompt:

.. code:: python

   layout = LabelLayout(
       spans=tuple(prompt_spans(prefill_plan(parts), segments, walk, image)),
       growth=(("decode_loop", 1),),
   )
   return ProcessPromptOutput(result, {"kv_layout": {KV_CACHE: {"main": layout}}})

``get_loop_caps(self, **model_kwargs) -> dict[str, int]`` The iterations each named loop
may run for this request. The conductor refuses a request whose cap is below 1 or past
its loop’s graph ``max_iters``, or whose layout at its caps holds more positions than
its pool’s ``max_seq_len``, with a 400. A request that sets no length has the default
``max_output_tokens`` cut to the context its prompt leaves, and
``request_info.max_tokens`` carries the cut. A model that reads a length from another
key lists it in ``LENGTH_KWARGS``. Speculation never runs past a cap, so the cap is the
loop’s budget. A model with an AR decode loop names it in the class attribute
``DECODE_LOOP``, and the default ``get_loop_caps`` caps it at ``max_output_tokens - 1``,
the prefill having sampled the first token; its stop check counts nothing. A model with
other loops whose count each request knows up front (denoise steps, rollout frames)
overrides ``get_loop_caps`` to add them, calling ``super()`` for the decode loop, and
its stop check counts nothing either. A model whose own stop check still ends a loop at
a count returns that count, so the two end the loop on the same iteration;
``test/modular/test_model_kv_layouts.py`` checks caps per model. Growth on an uncapped
loop is counted to its graph ``max_iters``.

Where ``process_prompt`` computes a length that a walk computes another way, such as a
media item’s rows, the walk asserts the two agree: BAGEL’s ViT walk against
``vit_patches``, Qwen3-Omni’s vision encoder against the processor’s grid.

Loading fails in three cases, and a request fails in two:

- A paged pool has no labels. The message is
  :literal:`<Model> declares no labels for its paged pool '<key>'; add them to \`kv_labels\`, and return each request's layout for them from \`process_prompt\``.
  A ring pool needs none.
- A label’s growth or keys name a walk the model never runs. The message is
  ``<Model> declares '<key>'/'<label>' over ['<walk>'], which it never runs; its walks are [...]``.
- A label grows on a loop its walk doesn’t run. The message is
  ``<Model> declares '<key>'/'<label>' growing on '<loop>', which '<walk>' doesn't run; its loops are [...]``.
- A request’s layout leaves out a label the request opens, or lays out one the model
  doesn’t declare. The request fails with a 500,
  ``<Model> returned no kv_layout for '<key>'/'<label>', which the request opens``, or
  ``<Model> returned a kv_layout for '<key>'/'<label>', which it does not declare``.

A request that takes a page past its footprint logs once per label, with the layout that
under-counted. A request is held only to the labels it opens, and to no pages on a label
it doesn’t open. A page there logs the same line, reported apart from a real overrun on
that label so neither hides the other. Under ``MSTAR_KV_DEBUG_ASSERTS=1`` each commit is
also checked against the layout: the walk that wrote each span, growth past its bound,
scratch past its declaration and the retention a step declares. At removal, a label left
a page or more under its footprint logs the most pages it held.

Cross-request prefix reuse
--------------------------

Two requests that begin with the same tokens fill the same KV pages, and the second can
attend the pages the first filled instead of recomputing them. A model opts a label in
by giving it ``keys``, and names its checkpoint. A model with no keyed label is never
cached. See `resource capacity and overrides <serving.html#resource-overrides>`__ for the deployment switch and per-request opt-out.

``PrefixKeys(tensor, walks, growth=False)`` - ``tensor`` names the request tensor whose
entries the layout’s ids spans are keyed by, as ``process_prompt`` returns it. -
``walks``: only a write from one of these walks keeps the keys going. A write from
another walk ends them, and the rest of the label is computed, never served. -
``growth``: the ids the label’s growth walk samples extend the keys, so a later request
that re-sends the reply matches it too.

The keys cover the label’s first ``keyed_spans`` spans. An ids span is keyed by its
token ids, a digest span by its file. Each span is its own step unless it sets
``opens_step=False``, which puts it in one row with the span before it, of the same walk
(``LabelLayout.row()`` fails a layout where it isn’t): a match cuts the row, and the row
keeps placing its own positions, so its walk declares the advance of the whole row
whatever the cut left (Qwen3-Omni’s prefill of text around an attachment).

BAGEL keys its text and ViT walks and the decode that follows them:

.. code:: python

   return {"kv": {"main": KVLabel(
       growth=((DECODE, "decode_loop"),),
       keys=PrefixKeys("text_inputs", ("prefill_text", "prefill_vit"), growth=True),
   )}}  # Add other declared labels here.

Orpheus keys its prompt walk:

.. code:: python

   return {KV_CACHE: {"main": KVLabel(
       growth=(("decode", "decode_loop"),), keys=PrefixKeys("text_inputs", ("prefill",)),
   )}}

``checkpoint_path(self) -> str | None`` Return the local directory that holds the
weights and ``config.json``. The cache hashes the checkpoint’s safetensors index and
config into the root of every key, so naming the checkpoint is what keeps two builds off
each other’s pages. The default returns ``None``, which leaves the weights out of the
root. A model without a checkpoint therefore needs ``prefix_cache_salt`` set to use the
cache at all. Without one, the engine keeps the cache shut and logs why.

A recurrent layer can’t skip tokens: its state after a token is all it keeps of the
tokens before it. On a node with a recurrent state pool, a hit is cut to the longest
snapshot of that state the pool holds under it. Name the KV cache in the pool’s config
(``RecurrentStateConfig(kv_cache=...)``), and give the deployment ``prefix_slots`` under
the pool’s ``resources:`` key. Each snapshot slot costs as much as a request slot.

- A prefill keeps the state at its last page boundary before its end, since a resend’s
  KV hit always leaves the last page to recompute, and at its KV hit’s last page
  boundary when that is past what it restored. A shared prefix is then served from the
  third request that shares it. A step longer than the largest captured bucket
  checkpoints less often, so that its checkpoints fit the captured steps’ buffer, and
  keeps its boundaries on that interval.
- A prompt the step budget cuts into chunks keeps each chunk’s last boundary (a chunk
  with more of its prompt to come keeps its own end), so a resend restores past its
  first chunk, and a later chunk’s snapshot makes the earlier one the first to evict, as
  a reply’s is. Checkpoints fall a page apart from where a chunk starts, so the pool
  declares a page as its ``chunk_alignment`` and the scheduler cuts its chunks whole
  pages long.
- A decode step that ends a page keeps the state there, so a later turn that re-sends
  the reply restores it. It replaces the request’s previous decode snapshot once the KV
  chain keys its page; a step whose token never reaches the chain, such as the one
  speculated past the reply’s end, leaves the previous one in place.
- A snapshot is keyed by the KV chain’s key at its page. A full set of slots evicts the
  least recently used snapshot that no request holds, a decode snapshot before a
  prompt’s, so the slots hold about one prompt each. A decode snapshot that a later
  prompt restores is kept like a prompt’s from then on.

Snapshots stay off without ``prefix_slots``, and also above one rank, without a named KV
cache, when that cache isn’t on every node of the pool, or when its ``prefix_cache`` is
off or can’t open at the pool’s world size; in those cases the pool allocates no
snapshot slots. Neither does a pool whose cache the engine keeps shut for another
reason, such as a model that names no checkpoint and sets no salt. Every cache on the
node of a pool without snapshots stays shut, and the engine logs which pool kept it
shut. ``opening_prefix_caches`` in ``mstar/engine/resources/kv/config.py`` is that rule
over the specs. The engine evaluates it once, before it builds any resource, and opens
exactly the caches it returns of those it holds; the conductor’s routing and the API
server’s prompt hashing take, for each rank, the caches the rank holds and its engine
opens (``prefix_caches_by_rank``). Each spec answers for itself:
``NodeResourceSpec.keeps_prefix_cache`` says whether the resource keeps a prefix cache
of its own, and ``restores_prefix`` marks one whose state a hit must restore (the pool,
and the resource over it), so another resource takes part through its spec alone.

A model with a keyed label is checked at load time. Loading fails in two more cases:

- A position resource over a keyed cache uses a scheme other than sequential. A page is
  reusable only where its tokens land at the positions they were written at. The message
  is
  ``resource '<key>' places positions by <scheme>, but it sits over ['<kv key>'], which the model keyed for prefix reuse; either drop the stream or give the resource a sequential scheme``.
- A resource on a keyed node does not set ``prefix_skip_safe``, and is neither a
  recurrent state pool nor a resource over pools alone. A hit takes the front of the
  request away from every resource on the node, including resources that never saw those
  tokens. The message is
  ``resource '<key>' cannot serve a request whose prefix came from the cache, but it sits on ['<node>'], which the model keyed for prefix reuse; either drop the stream or give the resource the prefix hooks``.

Some requests stay uncached even when the model keys a label. The engine checks each
request’s inputs on the keyed walk and cuts a prefix only from inputs that
``split_inputs`` can slice. A request whose inputs carry ``kwargs`` or
``resource_step_info`` is never probed; BAGEL’s guided requests carry ``requires_cfg``
there. The default ``split_inputs`` slices only the sequence, so a request whose inputs
carry ``tensor_inputs`` is probed only when the submodule sets ``cuts_tensor_inputs``
and overrides ``split_inputs`` to cut those tensors.

For a label keyed over several walks, where a keyed node streams into a partition other
than its own, a request that runs that partition is not cached either, since a hit would
skip the rows the node streams for the tokens the cache holds. BAGEL keys an
understanding prompt’s text and images, and a generation’s first span when it is text:
an edit’s images are resized before their walks see them, and a file’s digest doesn’t
cover the resize. A label under a step’s retention (``RetentionPolicy``) files pages
only up to its protected prefix, which the retention never releases.

A request on a LoRA adapter fills different KV for the same tokens, so its keys chain
from a root that names the adapter’s weights: a SHA-256 over every module’s A, B (with
dtype and shape) and alpha, read once when the adapter is registered
(``adapter_prefix_root`` in ``mstar/engine/resources/lora/identity.py``). The API server
takes the root when it selects a request’s adapter (``ServedModels.select``) and the
preprocess worker keys the prompt from it. The root travels with the request as
``KVReqConfig.prefix_root`` so that a page generation fills is keyed under it too.

The same tokens under two adapters, or under an adapter and the base model, never share
pages; two names for identical weights do; and a name registered again with other
weights misses everything the old weights filled; a request selected under the old
weights carries their digest (``LoraReqConfig.digest``), and a pool serving the new ones
refuses it at ingest.

A base-model request chains from the empty root, so its keys are what they were before
LoRA. Only a registered adapter can be selected, so every request on one has its root;
one that arrives without is not cached. Nothing on the model side is needed for this.
