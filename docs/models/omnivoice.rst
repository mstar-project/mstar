OmniVoice
=========

Massively multilingual zero-shot TTS: masked-diffusion canvas over a Qwen3-0.6B
backbone + audio codec. Clones a voice from a reference clip.

Registry key: ``omnivoice``. Checkpoint: ``k2-fsa/OmniVoice``.

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``omnivoice``
     - ``configs/omnivoice.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve omnivoice

For client requests and response formats, see :doc:`/clients`.

Install dependencies
--------------------

The ``omnivoice`` package is installed separately from git rather than by an
extra — see :doc:`/installation`.

Serve the model
---------------

The backbone is not autoregressive: it fills a fixed canvas of eight codebook
rows over a few unmasking steps, so there is no KV cache and no per-token
sampling loop. Serve it with ``mstar serve omnivoice --gpus 0``.

Voice input and language
------------------------

Zero-shot only: there are no built-in speakers. Pass ``ref_audio`` (an uploaded
``audio`` input on ``/generate``) with its transcript in ``ref_text`` to clone a voice,
or describe one in ``instruct`` (``voice`` on ``/v1/audio/speech``).
``ref_text`` is required alongside the reference clip.

Language values
---------------

``language`` takes either the name (``Vietnamese``) or the id (``vi``): a
name is resolved to the id the model was trained on before the prompt is
built, and an unrecognised value warns and falls back to language-agnostic
mode.

Canvas execution and precision
------------------------------

The backbone's layer stack replays CUDA graphs per (requests, packed-token
bucket up to 8192 tokens) with attention through a ragged attention resource;
larger steps run the same layer loop eagerly. The layer loop is mstar's own over
the reference's fused weights: q/k norms read the fused QKV output in place
and each residual add folds into the next RMSNorm (flashinfer
``fused_add_rmsnorm``), so logits match the reference to half-precision
rounding rather than bit for bit. ``model_kwargs: {backbone_graphs: false}``
runs the same layer loop eagerly, and the serving config's
``prefill_token_buckets: {backbone: {backbone_layers: [...]}}`` sets the
packed-token buckets (see :doc:`/serving`).

Split codec deployment
----------------------

``configs/omnivoice_split.yaml`` runs code2wav (the codec decode) on a
second worker sharing GPU 0, so the unmask loop no longer stops for each
finished request's decode; the silence trim and the rest of the output
shaping run on the API side in either config. It sets ``mps: true`` (see
:doc:`/serving`) so the two workers' kernels overlap. On an H200 it raised
concurrency-32 throughput from ~153 to 176-184 audio-s/s and cut TTFA p95
from ~9 s to ~3 s, measured while code2wav still did the shaping on its
worker; concurrency 1 was ~10% slower.
