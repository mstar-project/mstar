.. raw:: html

   <span id="adding-a-new-model"></span>

Add a model
===========

Add a model by making one request work from prompt processing through client output.
Start from an existing scaffold, keep the first deployment to one supported device
layout, and test graph wiring before loading weights. Add batching, capture, and
streaming after that path works.

You need a `source installation <installation.html>`__, the model’s checkpoint and
reference implementation, and hardware supported by its components. Run commands below
from the repository root. The code fragments explain interfaces; they are not a complete
model that can be launched unchanged.

.. _overview:

Choose a starting point
-----------------------

Use the implementation closest to your model’s computation:

- **Text or vision-language decoder:** ``MultimodalDecoderModel`` with
  ``SampledTokenDecoder``; see the `decoder
  scaffold <model_examples.html#worked-example-a-vision-language-model-on-the-vlm-scaffold>`__.
  It supplies prompt planning, decode, sampling, and common capture declarations.
- **Image diffusion:** ``DenoiseLoopSubmodule``; see the `diffusion
  scaffold <model_examples.html#worked-example-an-image-dit-on-the-diffusion-scaffold>`__.
  It supplies schedules, loop state, batching, and step declarations.
- **Encoder-decoder:** `Whisper <model_examples.html#worked-example-whisper>`__ shows a
  separate encoder and cross-attention decoder.
- **Token decoder followed by an audio codec:**
  `Orpheus <model_examples.html#worked-example-orpheus>`__ shows resource ownership and
  streaming partitions.
- **Interleaved understanding and generation:**
  `BAGEL <model_examples.html#worked-example-bagel>`__ explains multiple cache labels
  and graph walks. Its span-split prompt path serves that graph; use the decoder
  scaffold for a new conventional VLM.

A model declares the work; the engine executes it. ``Model`` handles prompt processing,
graph walks, resource declarations, and output encoding. Each graph node has a
``NodeSubmodule`` for tensor computation. The engine owns resource allocation,
admission, planning, compilation, graph capture, and commit. Deployment YAML maps nodes
to ranks. Do not implement those engine mechanisms inside a model.

A **graph walk** is one named route through the model, such as ``prefill`` or
``decode``. Declare each walk as its own graph; a node named in several walks shares its
submodule and resources across them. A **tensor bundle**, ``NameToTensorList``, maps
edge names to lists of tensors: ``dict[str, list[torch.Tensor]]``, usually with one
tensor per list.

.. code:: text

   process_prompt()  -> initial tensors and request metadata
   walk_plan()       -> ordered walks for each partition
   engine            -> execute each walk's nodes and resource steps
   postprocess()     -> encode output tensors as bytes for the client

.. _what-you-will-create:

Create the package
------------------

Keep the model’s interface separate from its compute components:

.. code:: text

   mstar/model/<your_model>/
   ├── __init__.py
   ├── config.py            # architecture and generation parameters; resource keys
   ├── <your_model>_model.py # Model subclass
   ├── submodules.py        # NodeSubmodule subclasses
   └── components/          # attention, decoder, and other nn.Modules

You will also add registry and CLI entries, a deployment YAML, tests, and a model guide
in both documentation formats. If you expose an OpenAI-compatible API, implement its
adapter in both frontends in the same change.

.. _step-1-register-the-model:

Register the model
------------------

Add a lazy entry to ``MODEL_REGISTRY`` in ``mstar/model/registry.py``. The key must
match ``model:`` in the deployment YAML. Add ``HF_MODELS`` when weights come from
Hugging Face. These are entries to insert into the existing dictionaries:

.. code:: python

   MODEL_REGISTRY: dict[str, tuple[str, str]] = {
       # Existing entries remain here.
       "your_model": ("mstar.model.your_model.your_model_model", "YourModel"),
   }
   HF_MODELS: dict[str, dict] = {
       # Existing entries remain here.
       "your_model": {"model_path_hf": "org/your-model-id"},
   }

The registry imports the named module only when the model is used. Importing the class
directly here would load that model’s dependencies into unrelated processes. There is no
plugin scan.

Also add the model’s config to ``DEFAULT_CONFIGS`` in ``mstar/cli/main.py``; without it,
``mstar serve your_model`` exits with ``error: unknown model``. Update the
client-example tuples in that file so the CLI prints an appropriate request. Add the
model to the registry-key table in ``docs/models.rst`` and its Mintlify counterpart.

For OpenAI routes, register an ``OpenAIAdapter`` in
``mstar/api_server/openai/adapters.py`` and an implementation in
``rust/server/src/adapters/``, including its ``ADAPTERS`` entry. A model served only
through native ``POST /generate`` needs no OpenAI adapter. New API behavior must be
covered by the frontend parity tests in ``test/rust/test_rust_openai_adapters.py``.

**Checkpoint:** once the package, config, CLI, and catalog entries exist, run
``python -m pytest test/modular/test_model_registration.py -q --timeout=120``. It checks
registration and file wiring without loading model weights. Passing it does not validate
the forward pass.

.. raw:: html

   <span id="step-2-implement-the-model-class"></span>

Implement the request interface
-------------------------------

Subclass the selected scaffold, or ``Model`` if none fits. Keep the constructor limited
to configuration, tokenizer setup, and metadata. Load and cache weights in
``get_submodule`` on the supplied device, so the conductor never allocates the model’s
GPU weights.

Implement one supported input/output combination first. Follow its data through these
hooks:

1. ``process_prompt`` converts the prompt and loaded media into named tensors. Return
   ``ProcessPromptOutput`` when request metadata must accompany them. Preserve prompt
   element order; ``prompt_parts`` distinguishes interleaved text and attachments.
2. ``walk_plan`` chooses the ordered ``WalkStep``\ s for each partition. For prefill
   followed by decode, the decode step binds its ``text_inputs`` to the persisted
   ``new_token`` from prefill.
3. ``get_graph_walk_graphs`` names the nodes and edges each walk uses.
4. ``get_submodule`` returns the compute wrapper for each node, or ``None`` in dummy
   mode.
5. ``text_output`` or ``postprocess`` encodes the result for the client. Text
   detokenization must preserve raw token bytes across chunk boundaries; decoding one
   token at a time can corrupt a split UTF-8 character.

These hooks are specified in the `model interface <model_contract.html>`__. A scaffold
may implement several for you. Declare accepted request keys in ``request_kwargs()``,
refuse unsupported inputs with ``ValueError``, and use ``get_loop_caps`` for
request-specific iteration limits. Keep tensor payloads on tensor edges rather than in
control-plane kwargs.

**Checkpoint:** add CPU tests for prompt conversion, accepted and rejected kwargs, walk
selection, and loop caps. Inspect the produced tensor names and shapes and the ordered
walk plan. Test a minimum-length request and a rejected request as well as the usual
path.

.. raw:: html

   <span id="step-2a-declare-your-resources"></span>

Declare state per node
----------------------

Return ``NodeResourceSpec`` declarations from ``get_node_resources()``. Name each
resource with a stable key and the nodes that use it. Layers refer to those keys, such
as ``attn_key``, ``kv_key``, and ``pos_key``; the engine builds and binds the resource.
A node that declares no resources receives none.

A token decoder typically needs KV storage, attention over that storage, positions, and
a sampler. An encoder may need cacheless attention instead. Start with your scaffold’s
declarations, then check the `resource
reference <model_resources.html#step-2a-declare-your-resources>`__ for your attention
and state requirements. Keep deployment capacities in YAML ``resources:`` overrides, not
model constants.

Per-request settings, including sampling parameters, come from
``get_request_resource_configs``. Never pass changing sampling parameters as Python
scalars to a captured forward: capture records their values. Resource request configs
stage them in buffers that can change safely between replays.

**Checkpoint:** every resource key used by a layer or step must have a declaration for
that node. A model with paged KV also needs its request’s labels, layout, and iteration
caps; follow `paged KV layouts <model_resources.html#paged-kv-layouts>`__ before
attempting a real decode.

.. _step-3-declare-the-computation-graph:

Connect the graph
-----------------

``get_graph_walk_graphs()`` returns ``{walk_name: graph}``. A ``GraphNode`` names a
submodule, its required input edges, and its output edges. ``Sequential`` and
``Parallel`` compose nodes; ``Loop`` repeats a section with a named iteration cap and
loop-back edges.

For a prefill/decode model, verify three distinct routes:

1. Prefill persists its first sampled token for the decode step’s ``bind``.
2. Decode feeds each next token back to the same node’s input.
3. Outputs intended for the client have an edge to ``EMIT_TO_CLIENT`` with the
   appropriate ``output_modality``.

A persisted edge alone does not emit to the client, and an output edge does not stop a
loop. Use ``get_loop_caps`` for the request’s token budget and the submodule’s stop
contract for EOS. The `graph fragment and
primitives <model_contract.html#graph-primitives>`__ show the internal wiring; the
`existing model examples <model_examples.html>`__ include complete model-specific
routes.

**Checkpoint:** test the graph in dummy mode before loading weights. Confirm that every
external input exists, each ``bind`` names a walk input, and the expected outputs reach
the client. Include a request that stops during prefill, if that is valid for your
model.

.. raw:: html

   <span id="step-4-implement-the-submodules"></span>

Implement and test the forward
------------------------------

For each node, implement the compute hooks the scaffold leaves to you:

- ``prepare_inputs`` performs cheap host bookkeeping and sets ``input_seq_len``; put
  resource-only metadata in ``resource_step_info``.
- ``declare_step`` describes this step’s resource effects from those inputs. It does not
  allocate or commit state.
- ``preprocess`` assembles forward arguments. Override it when collating more than one
  row.
- ``forward`` performs tensor computation and returns named outputs. Keep it free of
  I/O, host reads of device values, and data-dependent Python branches that prevent
  capture.
- ``check_stop`` makes decisions from results materialized on the host. Submodule
  ``postprocess`` is for metadata-only output changes on the GPU thread; it must not
  synchronize to read device values.

Read the `submodule reference <model_submodules.html>`__, especially `step
declarations <model_submodules.html#step-4a-declare-the-step>`__, `resource
access <model_submodules.html#reaching-your-resources>`__, and `weight
loading <model_submodules.html#loading-weights>`__. Cast a module built on ``meta`` to
its requested parameter dtype before ``to_empty(device)``; casting afterward first
allocates float32 parameters and increases loading memory.

**Checkpoint:** compare the eager forward with the reference implementation on fixed
inputs. Check tensor shapes, numerical agreement at the intended dtype, stopping, and
output encoding. Record the hardware, torch/accelerator build, checkpoint, and
tolerance; a dummy graph test cannot establish numerical correctness.

.. raw:: html

   <span id="step-5-continuous-batching-and-cuda-graphs"></span>

.. raw:: html

   <span id="step-5"></span>

Add batching and capture
------------------------

When eager execution is correct, enable batching for compatible requests through
``can_batch`` and ``forward_batched``. Compare single-request and batched outputs,
including mixed request lengths and stopping at different times.

Declare whole-forward capture buckets with ``get_accelerator_graph_configs``, or inner
regions with ``get_piecewise_accelerator_graph_configs``. The engine owns capture and
replay. Use the `batching and capture guide <model_capture.html>`__ for scheduling,
capture keys, compiler options, and piecewise leases. If your scaffold already supplies
these hooks, verify its limits before overriding it.

**Checkpoint:** compare eager, batched, and captured paths at supported shapes and at an
eager fallback shape. Verify resource effects as well as output tensors. Report
performance only after measuring with the repository’s `benchmarking
workflow <evaluation.html>`__; record the workload and hardware.

.. raw:: html

   <span id="step-6-write-a-config-yaml"></span>

.. raw:: html

   <span id="config-yaml"></span>

Place the nodes and send a request
----------------------------------

Create ``configs/your_model.yaml``. Use the registry key under ``model:`` and the
graph’s exact node names under ``node_groups``. For a model with only an ``LLM`` node,
the starting placement is:

.. code:: yaml

   model: "your_model"
   max_seq_len: 2048
   node_groups:
     - node_names: ["LLM"]
       ranks: [0]

This is a placement example: choose limits supported by your model and declare every
other node it needs. Each node of each walk must be placed exactly once. Start the
registered model with your config:

.. code:: bash

   mstar serve your_model --config configs/your_model.yaml --port 8000

In another terminal, send the native or OpenAI request documented for that model. Check
the returned text, media, or action data, then send concurrent requests and a request
that reaches its stopping limit. Server health alone does not verify model correctness.

For resource sizing, preprocessing threads, replica routing, and TP/SP/CFG/EP layouts,
use `model deployment <model_deployment.html>`__. For an independently running consumer
such as a codec, use `streaming model graphs <model_streaming.html>`__.

.. _checklist:

Before opening the PR
---------------------

- Register the model, its checkpoint defaults when applicable, the CLI config, and
  client example.
- Test prompt handling, graph walks, resource declarations, and invalid requests on CPU;
  guard tests that need optional packages, weights, or a GPU.
- Test numerical output and the supported batched/captured paths on the target hardware.
  State what could not run.
- Add both Python and Rust adapters when exposing OpenAI routes, with parity tests.
- Document model setup, checkpoint access, supported inputs, limitations, and any extra
  dependencies. Update both RST and Mintlify guides.
- Keep node placement and deployment sizing in YAML. Check that request-local state is
  cleaned up when requests finish.
- Report the tests run and their results in the PR. Include measurements for changed hot
  paths; do not infer a speedup from the design alone.

.. _testing:

Run focused checks
------------------

During development, run the modules covering the code you changed. Registration checks
are independent of weights:

.. code:: bash

   python -m pytest test/modular/test_model_registration.py -q --timeout=120
   ruff check .

Use ``test/modular/test_resource_runner.py`` for resource lifecycle patterns,
``test/modular/vjepa2/fake_resources.py`` for stub resources,
``test/modular/test_accelerator_graph_capture.py`` for capture registration, and the
scheduler/admission/offload tests for those integration paths. See `the test
map <model_contract.html#test-map>`__ for the complete list from this guide.

Before pushing, run the CPU suite specified in ``AGENTS.md`` with GPUs hidden and the
Rust extension built. Run relevant ``test/integration/`` cases separately with real
weights and the target accelerator. Tests that skip for missing checkpoints do not
establish model correctness.

.. _reference-sections:

Reference sections from the original guide
------------------------------------------

The detailed contracts now live on the pages below. Existing section links remain
available here.

.. raw:: html

   <span id="per-request-resource-parameters"></span>

.. raw:: html

   <span id="paged-kv-layouts"></span>

.. raw:: html

   <span id="cross-request-prefix-reuse"></span>

.. raw:: html

   <span id="two-samplers-on-one-node"></span>

.. raw:: html

   <span id="cross-attention-encoder-decoder-models"></span>

.. raw:: html

   <span id="cacheless-attention-encoder-towers"></span>

.. raw:: html

   <span id="cross-attention-between-spans"></span>

.. raw:: html

   <span id="block-causal-attention-within-spans"></span>

.. raw:: html

   <span id="block-causal-attention-over-a-kv-cache"></span>

.. raw:: html

   <span id="bounded-sink-window-caches"></span>

.. raw:: html

   <span id="per-request-lora-adapters"></span>

Resource declarations and request state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Per-request resource
  parameters <model_resources.html#per-request-resource-parameters>`__
- `Paged KV layouts <model_resources.html#paged-kv-layouts>`__
- `Cross-request prefix reuse <model_resources.html#cross-request-prefix-reuse>`__
- `Two samplers on one node <model_resources.html#two-samplers-on-one-node>`__
- `Cross-attention (encoder-decoder
  models) <model_resources.html#cross-attention-encoder-decoder-models>`__
- `Cacheless attention (encoder
  towers) <model_resources.html#cacheless-attention-encoder-towers>`__
- `Cross-attention between spans <model_resources.html#cross-attention-between-spans>`__
- `Block-causal attention within
  spans <model_resources.html#block-causal-attention-within-spans>`__
- `Block-causal attention over a KV
  cache <model_resources.html#block-causal-attention-over-a-kv-cache>`__
- `Bounded (sink + window) caches <model_resources.html#bounded-sink-window-caches>`__
- `Per-request LoRA adapters <model_resources.html#per-request-lora-adapters>`__

.. raw:: html

   <span id="step-4a-declare-the-step"></span>

.. raw:: html

   <span id="reaching-your-resources"></span>

.. raw:: html

   <span id="device-loop-back"></span>

.. raw:: html

   <span id="loading-weights"></span>

Submodule hooks and step declarations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Step 4a — Declare the step <model_submodules.html#step-4a-declare-the-step>`__
- `Reaching your resources <model_submodules.html#reaching-your-resources>`__
- `Device loop back <model_submodules.html#device-loop-back>`__
- `Loading weights <model_submodules.html#loading-weights>`__

.. raw:: html

   <span id="combined-walks"></span>

.. raw:: html

   <span id="piecewise-cuda-graphs-capturing-an-inner-loop"></span>

Batching and accelerator graph capture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Combined walks <model_capture.html#combined-walks>`__
- `Piecewise CUDA graphs (capturing an inner
  loop) <model_capture.html#piecewise-cuda-graphs-capturing-an-inner-loop>`__

.. raw:: html

   <span id="tuning-resources-per-deployment"></span>

.. raw:: html

   <span id="routing-across-data-parallel-replicas"></span>

.. raw:: html

   <span id="preprocessing-threads"></span>

.. raw:: html

   <span id="tensor-parallelism-sharding"></span>

.. raw:: html

   <span id="tensor-parallelism"></span>

.. raw:: html

   <span id="guidance-cfg-parallelism"></span>

.. raw:: html

   <span id="expert-parallelism-moe"></span>

.. raw:: html

   <span id="tuned-kernel-configs"></span>

Deployment tuning and parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Tuning resources per
  deployment <model_deployment.html#tuning-resources-per-deployment>`__
- `Routing across data-parallel
  replicas <model_deployment.html#routing-across-data-parallel-replicas>`__
- `Preprocessing threads <model_deployment.html#preprocessing-threads>`__
- `Tensor parallelism (sharding) <model_deployment.html#tensor-parallelism>`__
- `Guidance (CFG) parallelism <model_deployment.html#guidance-cfg-parallelism>`__
- `Expert parallelism (MoE) <model_deployment.html#expert-parallelism-moe>`__
- `Tuned kernel configs <model_deployment.html#tuned-kernel-configs>`__

.. raw:: html

   <span id="worked-example-orpheus"></span>

.. raw:: html

   <span id="worked-example-bagel"></span>

.. raw:: html

   <span id="worked-example-whisper"></span>

.. raw:: html

   <span id="worked-example-an-image-dit-on-the-diffusion-scaffold"></span>

.. raw:: html

   <span id="worked-example-a-vision-language-model-on-the-vlm-scaffold"></span>

Worked models and scaffolds
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Worked example: Orpheus <model_examples.html#worked-example-orpheus>`__
- `Worked example: BAGEL <model_examples.html#worked-example-bagel>`__
- `Worked example: Whisper <model_examples.html#worked-example-whisper>`__
- `Worked example: an image DiT on the diffusion
  scaffold <model_examples.html#worked-example-an-image-dit-on-the-diffusion-scaffold>`__
- `Worked example: a vision-language model on the VLM
  scaffold <model_examples.html#worked-example-a-vision-language-model-on-the-vlm-scaffold>`__

.. raw:: html

   <span id="advanced-async-partitions-and-streaming"></span>

.. raw:: html

   <span id="streams-from-the-client"></span>

Async partitions and client streams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Advanced: async partitions and
  streaming <model_streaming.html#advanced-async-partitions-and-streaming>`__
- `Streams from the client <model_streaming.html#streams-from-the-client>`__
