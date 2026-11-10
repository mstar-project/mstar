Batching and graph capture
==========================

First verify the node in eager execution. Then test compatible requests as a batch, and
finally add capture buckets. Each stage must preserve the same outputs, stopping
behavior, and resource effects. These mechanisms are optional; a scaffold may already
implement them for your model.

.. _step-5:

Batch requests and capture forwards
-----------------------------------

Batching changes which requests share a forward. Accelerator graphs replay work recorded
for a declared shape. Test each mechanism independently before using them together; a
capture declaration does not prove a throughput improvement.

Batch compatible rows
~~~~~~~~~~~~~~~~~~~~~

The worker’s micro-scheduler groups compatible in-flight requests into one GPU call. A
submodule controls this behavior with these methods:

- ``can_batch(self, batch: ExecutingBatch, model_inputs) -> bool`` returns whether these
  requests can share one forward pass. The default is ``False``, which disables
  batching. Override it to accept batches, for example when the requests use the same
  graph walk and have compatible shapes.
- ``forward_batched(self, graph_walk, engine_inputs, **kwargs)`` performs the batched
  compute. It returns per-request outputs keyed by request handle, or a
  ``BatchedModelOutput`` carrying batched tensors and optional stop buffers/spans. When
  a batch runs, the engine calls this method instead of the single-request ``forward``.
- ``max_batch_size(self, graph_walk)`` sets an optional upper limit.
- ``step_order(self, graph_walk) -> StepOrder`` says who runs first when more rows are
  ready than ``max_batch_size`` allows. The default, ``StepOrder.ARRIVAL``, is first
  come, first served to completion: the requests holding the step keep it, then the
  request that arrived first. ``StepOrder.TURNS`` is longest-waiting first: a request
  that just ran gives way to one that waited, so a node capped at one row advances its
  requests a step each in rotation. Declare it on a walk whose sessions have to keep
  moving while others run, as LingBot’s rollout and Cosmos3’s chained windows do. A
  combined walk takes turns if any of its walks does. The order applies to the rows a
  speculation chain continues, the backlog and the queued rows alike; when every ready
  row fits, it doesn’t matter. An admit that runs out of KV pages offloads the row last
  in that same line.

Prioritize walks
~~~~~~~~~~~~~~~~

Which (node, walk) a worker’s next step goes to is a ``SchedulingPolicy``
(``mstar/worker/scheduling_policy.py``). By default it is round robin: the least
recently scheduled key first, and a speculation chain keeps its first step before it
gives way to other ready work. A model that returns ``WalkPriority`` entries from
``Model.get_walk_priorities`` gets ``Priority`` instead. An entry names a node (all its
walks), a walk (all its nodes) or one (node, walk), and where entries overlap the most
specific decides: a pair over a walk over a node. Keys with ready rows go in priority
order, round robin among equals, and a chain gives way at once when a key above its own
has a row the engine would take. That trades the other keys’ steps for the listed one’s
latency, so it suits a walk whose steps are short, such as a TTS model’s prefill, whose
first audio waits on it while decode streams have time to spare: list the walk, and
every node on it is covered.

Bound priority starvation
^^^^^^^^^^^^^^^^^^^^^^^^^

A key that stays ready holds lower ones back for as long as it does, unless they have a
``max_wait``: a ready key that has gone that many of the worker’s chain steps (every
step on its main chain, side-lane steps not counted) since it last ran goes first,
longest waiting first, and a chain gives it its next step. A priority on one walk of a
combined walk lifts the whole combined key in that order, its other walks’ rows
included, and a backlogged key, split over several steps, still finishes first whatever
its priority.

Override priorities per deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A deployment replaces the entries, sets a default ``max_wait`` or turns priorities off
with the serving config’s ``scheduling`` (see `Serving <serving.html>`__). An entry
naming nothing the model runs, a negative priority, or two entries for the same target
refuse the worker’s start.

.. _combined-walks:

.. raw:: html

   <span id="combined-walks"></span>

Combine walks in one step
~~~~~~~~~~~~~~~~~~~~~~~~~

By default a step holds one walk. A model lets several walks of a node share a step by
returning ``{node: {combined_walk: {walks}}}`` from ``get_combined_graph_walks``, for
example decode rows beside a prompt’s prefill. The combined walk is a name of its own,
which ``max_batch_size`` and the CUDA-graph configs can key on. A step of one walk keeps
that walk’s cap. A step that mixes walks takes the combined walk’s ``max_batch_size``
when it has one, else the smallest cap of the walks it holds, so a small prefill cap
never limits a decode-only step. ``max_rows_in_combined(self, graph_walk)`` also bounds
how many rows of one walk a mixed step may hold, for kernels sized to a few prompt rows
(see ``CombinedSplit``). To run a combined walk on such split kernels, pass a
``CombinedSplit`` in the ``splits`` of each resource config that supports one
(``AttentionConfig`` on the ``FLASHINFER`` backend, for captured steps only, and
``LinearAttnConfig`` with the GDN variant; other backends ignore ``splits`` and run the
walk unsplit); their specs return it from ``combined_splits()``, which is what the
engine reads, and resources of one node must agree on a walk’s ``prefill_walks``. The
engine then orders that walk’s steps with the ``prefill_walks`` rows last, so the
kernels read the decode rows as the step’s first tokens, in place. Every other walk of
the combined walk must be one token per row.

Run a walk on a side stream
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A worker’s main loop runs one chain of steps, and any step on it costs a pass of the
loop and holds the next one behind its kernels. A node that shares a worker with a
heavier one can set ``side_stream_walks`` on its submodule to the walks that may run
beside the chain instead, and a deployment turns that on with ``side_stream: true`` on
the node’s group (see `Serving <serving.html>`__). Such a step neither takes a turn on
the chain nor makes it yield, runs on a second stream, and the windows that come ready
while it runs batch into the next one (see ``StepPipeline._launch_side`` in
``mstar/worker/step_pipeline.py``). Its graphs capture on a stream of their own, since
graphs captured on one stream must not overlap. It suits a stream-fed walk that nothing
on the chain reads, such as a codec.

Side-stream eligibility
^^^^^^^^^^^^^^^^^^^^^^^

Where the config asks, ``EngineManager.build`` refuses what the lane can’t serve, since
a side step is neither speculated nor planned ahead and lands between two chain steps: a
TP, SP or CFG node, a node alone on its worker, a device without streams, a node with
chain walks or resources, and a side walk that is not the node’s alone, sits in a
combined walk, or has output spans or a stop check. A config without the field keeps the
node on the chain, whatever it declares.

``ARNodeSubmodule`` declares ``preprocess`` abstract because autoregressive nodes are
expected to collate a batch. You can still disable batching for one node, or for
specific walks, by returning ``False`` from ``can_batch`` in those cases.

Declare whole-forward captures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A submodule declares the shapes it can capture in
``get_accelerator_graph_configs(self, device, tp_world_size=1) -> list[AcceleratorGraphConfig]``.
The default is an empty list, which means eager execution. For each config, the engine
first runs ``torch.compile``, controlled by the config’s ``compile`` flag, which
defaults to ``True``. It then records a CUDA graph and replays it. Two config types are
defined in ``mstar/engine/accelerator_graph_config.py``. They differ in which stage of
the submodule pipeline they freeze:

+-------------------------------------------+-------------------------------------------+
| Config type                               | Use and captured stage                    |
+===========================================+===========================================+
| ``BatchedAcceleratorGraphConfig``         | Decode-style forward passes, in which     |
|                                           | every request in the batch has the same   |
|                                           | length. That length is usually one token. |
|                                           | Pass ``single_request_inputs``, which is  |
|                                           | a ``NodeInputs`` for one request. The     |
|                                           | runner clones it to build each captured   |
|                                           | batch. This config fixes the output of    |
|                                           | ``prepare_inputs``, and ``preprocess``    |
|                                           | runs on both capture and replay.          |
+-------------------------------------------+-------------------------------------------+
| ``PackedAcceleratorGraphConfig``          | Prefill-style forward passes that operate |
|                                           | on packed, variable-length sequences.     |
|                                           | Pass ``capture_token_lengths``, which     |
|                                           | lists the token-count buckets to record,  |
|                                           | and ``make_node_input(n)``, a factory     |
|                                           | that builds a ``NodeInputs`` for one      |
|                                           | request of length ``n``. The runner       |
|                                           | partitions each bucket across the batch.  |
|                                           | Attention is planned at capture time from |
|                                           | the step declaration, and planned again   |
|                                           | into the captured buffers on each replay. |
+-------------------------------------------+-------------------------------------------+

Both types share the base ``AcceleratorGraphConfig`` fields:

- ``capture_graph_walk`` is the walk to capture.
- ``replay_graph_walks`` lists the walks that may replay this capture. One capture can
  therefore serve several walks, for example ``prefill_audio`` reusing ``prefill_text``.
- ``capture_batch_sizes`` lists the batch sizes to record.
- ``additional_key_info`` is an extra hashable component of the capture bucket key. Use
  it when one walk must be captured more than once because the batch shape alone does
  not identify the capture. Its value must equal both the value returned by the
  submodule’s ``cg_key_info(graph_walk, per_request_info)`` for a batch and the
  ``cg_key_info`` value that the batch’s ``declare_step`` sets on its step. A mismatch
  raises no error. The batch simply does not match any capture and runs eagerly. Compute
  all three values from one place. This field replaces the removed ``labels`` and
  ``requires_cfg`` fields.
- ``capture_forward_method`` names the method to capture. The default is
  ``"forward_batched"``.
- ``caps_eager_batch_size`` controls whether this config’s captured sizes also limit the
  engine’s eager batch size for the walk. The default is ``True``, so the engine never
  batches beyond a captured size.
- ``compile`` runs ``torch.compile`` before capture. The default is ``True``.
- ``compile_mode`` is that ``torch.compile``\ ’s mode. The default, ``None``, takes the
  deployment’s compile level (``mstar serve -O``, ``MSTAR_COMPILE_LEVEL``); naming a
  mode takes the config out of that level. ``default`` keeps Inductor’s fusion but takes
  cuBLAS for every GEMM and skips autotuning. ``deterministic`` is the engine’s own:
  Inductor’s defaults in its deterministic mode, which picks kernel configs by rule
  rather than timing them on the device, so every process compiles the same kernels (a
  torch without the mode logs a warning and compiles with the defaults). A compile
  outside the runner takes the same options from
  ``accelerator_graph_runner.compile_kwargs("deterministic")``, or
  ``mstar.platforms.compile.deterministic_inductor_options()`` where it builds its own;
  the Wan VAE decode, the DiT scaffold’s ``compile_blocks`` and Cosmos3’s denoise step
  do. Building the config refuses an unknown mode, and one that turns on Inductor’s own
  CUDA graphs (``reduce-overhead``, ``max-autotune``), which fail inside the runner’s
  capture.
- ``inductor_options`` are Inductor config entries the compile runs under, over the
  mode’s or the compile level’s own, e.g. ``{"max_autotune_gemm_backends": "ATEN"}`` to
  keep max-autotune’s fusion choices but cuBLAS for the GEMMs (autotune’s timings of
  decode-width GEMMs can vary between processes). Building the config refuses an entry
  Inductor doesn’t have, and ``triton.cudagraphs``.

``BatchedAcceleratorGraphConfig`` also accepts ``total_tokens_multiplier``. Use it when
one request’s step commits KV across several labels that are combined into a single
plan, as in batched guidance that packs the conditional and unconditional sequences
together. The static buffer must hold all of them, and this field scales the buffer
independently of the per-label span.

For example, a token decoder that captures a batched ``decode`` graph and a packed
``prefill`` graph declares the following (``SampledTokenDecoder`` builds these, and a
mixed one, for its subclasses):

This method excerpt uses module-level bucket constants; place the method on your
submodule.

.. code:: python

   import torch
   from mstar.engine.accelerator_graph_config import (
       BatchedAcceleratorGraphConfig, PackedAcceleratorGraphConfig,
   )
   from mstar.model.submodule_base import ARNodeInputs

   PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024]
   PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

   def get_accelerator_graph_configs(self, device, tp_world_size=1):
       return [
           BatchedAcceleratorGraphConfig(
               capture_graph_walk="decode",
               single_request_inputs=ARNodeInputs(
                   input_ids=torch.zeros(1, dtype=torch.long, device=device),
                   input_seq_len=1,
               ),
           ),
           PackedAcceleratorGraphConfig(
               capture_graph_walk="prefill",
               capture_token_lengths=PREFILL_TOKEN_BUCKETS,
               make_node_input=lambda n: ARNodeInputs(
                   input_ids=torch.zeros((n,), dtype=torch.long, device=device),
                   input_seq_len=n,
               ),
               capture_batch_sizes=PREFILL_CAPTURE_BATCH_SIZES,
           ),
       ]

A deployment replaces a packed capture’s ``capture_token_lengths`` with the serving
config’s ``prefill_token_buckets`` (node -> capture walk -> ladder, see
`Serving <serving.html>`__). The engine applies it to the configs
``get_accelerator_graph_configs`` returns, so a model plumbs nothing. A
``PiecewisePackedConfig`` region’s ``total_tokens`` takes the ladder its label names the
same way.

BAGEL shows the case of two captures for one walk. It captures ``decode`` twice, once
with ``additional_key_info=False`` and once with ``additional_key_info=True``. A decode
step with guidance enabled and a decode step with guidance disabled declare different
segments over the same token count. BAGEL’s ``cg_key_info`` reports which of the two a
batch is.

.. _piecewise-cuda-graphs-capturing-an-inner-loop:

Capture an inner loop
~~~~~~~~~~~~~~~~~~~~~

The configs above capture the whole ``forward_batched`` method of a submodule. The
engine then drives the replay, including sampling and output remapping.

Sometimes you want to capture only one inner region of a forward, such as a transformer
block loop, and keep the surrounding code in eager Python. The code before the region
computes embeddings and assembles the sequence. The code after it applies the final norm
and projection. A piecewise CUDA graph supports this.

A submodule enables piecewise capture by returning one or more configs from:

.. code:: text

   get_piecewise_accelerator_graph_configs(self, device, autocast_dtype, tp_world_size=1)
       -> dict[str, PiecewiseAcceleratorGraphConfig]

The dict key is a region name, which is any string that identifies the captured region.
Several keys capture several independent graphs. At warmup, the engine builds one
``PiecewiseAcceleratorGraphRunner`` per key and puts them in
``engine_inputs.piecewise_runners``. Your forward looks up the runner by key and calls
it. Nothing is stored on the submodule.

Two config types are defined in ``mstar/engine/accelerator_graph_config.py``. They
correspond to the two whole-forward types:

+-------------------------------------------+-------------------------------------------+
| Config type                               | Use and captured shape                    |
+===========================================+===========================================+
| ``PiecewiseBatchedConfig``                | Batched inputs of equal length. The       |
|                                           | captured region sees                      |
|                                           | ``[bs, seq_len, D]``, and every request   |
|                                           | has the same ``seq_len``. Pass            |
|                                           | ``seq_len``, the number of tokens per     |
|                                           | request. One graph is captured per batch  |
|                                           | size.                                     |
+-------------------------------------------+-------------------------------------------+
| ``PiecewisePackedConfig``                 | Packed, variable-length inputs. The       |
|                                           | captured region sees                      |
|                                           | ``[total_tokens, D]``, with several       |
|                                           | sequences of different lengths packed     |
|                                           | together. Pass ``total_tokens``, a list   |
|                                           | of token-count buckets. One graph is      |
|                                           | captured per pair of batch size and       |
|                                           | bucket.                                   |
+-------------------------------------------+-------------------------------------------+

Both types share the base ``PiecewiseAcceleratorGraphConfig`` fields:

- ``capture_fn`` is the callable to capture. Its interface is described below.
- ``make_static_inputs`` is a factory with signature
  ``(shape: PiecewiseCaptureShape) -> dict[str, Tensor]``. It returns the persistent
  buffers that the captured region reads. The runner owns these buffers. Before each
  replay, it copies your real tensors into them by name. ``shape`` provides ``bs``,
  ``seq_lens`` and ``total_tokens`` for the bucket being built. Allocate each buffer in
  the dtype of the tensor you pass for it, which for a hidden state is usually
  ``autocast_dtype``. The copy before replay casts a value into a wider dtype of its
  kind (int32 token ids into an int64 buffer, a bf16 table into an fp32 one). A value
  the buffer may round (a narrower dtype, such as an fp32 hidden state into a bf16
  buffer, or an integer into a floating-point or complex buffer) is refused with a
  ``ValueError`` naming the buffer.
- ``forward_kwargs`` holds static keyword arguments that are passed to ``capture_fn`` on
  every call, for example a layer count or an ``is_causal`` flag.
- ``required`` refuses to start the worker when any of the region’s shapes is left out
  of capture, for memory or by a failure, rather than serving those shapes eagerly. Set
  it on a region with no eager path, such as one whose per-stream state only its
  captures read and write. If any participating rank cannot capture a declared shape,
  warmup raises after rank agreement. The default is ``False``.
- ``declare_step`` has signature ``(request_ids, seq_lens) -> SubmoduleStep | None``. It
  declares the region’s own resource work over the padded batch. It is separate from the
  submodule’s ``declare_step`` because the region has its own shape. The runner admits,
  plans and commits this step on each replay. This field replaces the removed
  ``uses_kv_cache``, ``plan_fn``, ``advance_seq_lens`` and ``cache_labels`` fields. A
  region that reads a KV cache declares a ``KVStep`` and an ``AttentionStep`` here. A
  cacheless region that uses ragged attention declares only an ``AttentionStep``,
  against its ``RaggedAttentionSpec`` key.
- ``lease_before_step`` makes the runner take this region’s CUDA-graph slot before the
  outer ``declare_step`` runs, and report the slot in that call’s ``piecewise_leases``
  argument. The region still declares, plans and commits its own work. The lease only
  tells the outer declaration which resources are already taken, so that it can exclude
  them.
- ``can_lease`` has signature ``(inputs: list[NodeInputs]) -> bool``. It applies only
  with ``lease_before_step``, and lets the region refuse a batch for a limit that the
  capture bucket key doesn’t express, such as Qwen3.5’s segment count per step. The
  engine then gives the region no lease for that step. Your forward must replay a
  ``lease_before_step`` region exactly when its label is in
  ``engine_inputs.piecewise_leases``, and not whenever ``runner.can_run`` is true: a
  refused batch can still fit a bucket, and replaying it would declare the region’s
  resources twice.
- ``one_row_per_step`` makes the runner lease, check (``can_run``) and replay the region
  as one row, whatever the step’s batch size, under the step’s first request. It is for
  a packed region that runs every request’s segments as one ragged batch, such as a
  vision tower, and needs ``lease_before_step`` and ``capture_batch_sizes=[1]``. The
  region’s ``declare_step`` then receives ``[first_rid]`` with every request’s
  ``seq_lens``, so the whole step’s segments count against one replay’s plan: size its
  attention resource with ``max_segments_per_step``, and pair it with a ``can_lease``
  that refuses a step with more. On a node serving LoRA adapters it is leased only for a
  step whose rows run none, since its replay would address every request’s tokens to the
  first request’s adapter.
- ``capture_batch_sizes`` lists the batch sizes to record. ``None`` uses the runner
  default.
- ``compile`` runs ``torch.compile`` on ``capture_fn`` before capture. The default is
  ``False``.
- ``compile_mode`` is that ``torch.compile``\ ’s mode, as on ``AcceleratorGraphConfig``.

Assign each resource one declaration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a region leases its own slot, exactly one of the two declarations must own each
resource. The common pattern is for the outer ``declare_step`` to return ``None`` when
the region holds a lease, and to declare the resources itself otherwise, which covers
the eager fallback path:

.. code:: python

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

Define the captured callable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The captured callable takes one ``PiecewiseCallInputs`` argument and returns a dict:

.. code:: text

   capture_fn(inp: PiecewiseCallInputs) -> dict[str, Tensor]

``PiecewiseCallInputs`` has four fields. ``static_inputs`` holds the buffers owned by
the runner. ``engine_inputs`` is the region’s view of the batch: request ids and
per-request info padded to the capture bucket, plus the node’s resources. ``kwargs`` is
the config’s ``forward_kwargs``, unchanged. ``resources`` is a shortcut for
``engine_inputs.resources``. Capture and replay both use this same type, so a region
cannot read a field that exists on only one of the two paths.

.. warning::

   Read tensors from ``inp.static_inputs``, and never assign to its entries. The runner
   passes the same dict object at capture time, and updates those buffers in place before
   each replay. An assignment such as ``static_inputs["x"] = ...`` replaces the buffer, and
   the region then no longer uses the memory address that the graph recorded.

A ``capture_fn`` may also return a single ``Tensor``. The runner wraps it as
``{"x": ...}``.

Replay and own the output
~~~~~~~~~~~~~~~~~~~~~~~~~

Look up the runner by region name and pass your real inputs to it. It returns a
``PiecewiseOutput``, which behaves like a dict. Indexing and ``.get`` return a clone
that you own and can keep. ``.get_view`` returns a view without copying, which is only
valid until the next call to ``run``. See ``mstar/engine/accelerator_graph_runner.py``.
The runner handles input padding, admission and planning of the region’s declared step,
replay, commit, and output slicing.

The V-JEPA2 AC predictor is the reference implementation. The following is a structural
excerpt: ``capture_seq_len``, ``embed_dim``, ``N``, ``cond_tokens``, resource keys, and
the omitted buffers come from that model. It is not a standalone runnable snippet. See
``mstar/model/vjepa2/submodules.py``, region ``"block_loop"``. It has an eager section
before the region, a captured block loop that reads the KV cache over a fixed per-step
sequence, and an eager section after the region.

.. code:: python

   from mstar.engine.accelerator_graph_config import (
       PiecewiseBatchedConfig, PiecewiseCallInputs, PiecewiseCaptureShape,
   )

   # --- the captured inner region ---
   def _block_loop_capture(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
       # READ out of inp.static_inputs; never reassign its entries
       cond_tokens = inp.kwargs.get("cond_tokens")
       fn = self.predictor.make_block_loop_fn("main", inp.static_inputs, cond_tokens)
       return {"x": fn(inp.static_inputs["x"])}

   # --- declare the region ---
   def get_piecewise_accelerator_graph_configs(self, device, autocast_dtype, tp_world_size=1, **kwargs):
       def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
           # hidden state in autocast_dtype so the replay copy_ is a same-dtype memcpy;
           # position buffers stay float32 (RoPE frequency precision matters more)
           return {
               "x": torch.zeros(shape.bs, capture_seq_len, embed_dim,
                                dtype=autocast_dtype, device=device),
               "d_pos": torch.zeros(N * N, dtype=torch.float32, device=device),
               # Other model-specific buffers go here.
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
   if BLOCK_LOOP_REGION in engine_inputs.piecewise_leases:
       runner = engine_inputs.piecewise_runners[BLOCK_LOOP_REGION]
       out = runner.run(                       # admits + plans + replays + commits
           static_inputs={"x": x, "d_pos": d_pos},  # include every declared buffer
           request_ids=engine_inputs.request_ids,
       )
       x = out["x"]                            # owned clone

Positions are computed eagerly and passed through ``static_inputs``. Both paths use the
same block loop. With ``lease_before_step=True``, replay is conditional on the granted
lease, not only on whether a bucket fits: ``can_lease`` can reject a fitting batch. The
region’s ``declare_step`` covers only the captured path. The eager path is covered by
the submodule’s own ``declare_step``, as shown under **Assign each resource one declaration**
above.

BAGEL’s ViT tower is the packed, cacheless counterpart. See
``mstar/model/bagel/submodules.py``, region ``"vit_block_loop"``: a
``PiecewisePackedConfig`` whose region declares only ragged attention, with patch
embedding and the RoPE gathers left eager because they are data-dependent indexing. A
packed region whose rows each attend within themselves passes ``attn_key`` (the
attention resource) instead of a ``declare_step``: the config then declares one
non-causal segment per row, or under ``one_row_per_step`` one per length of its row.
