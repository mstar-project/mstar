Muse Glimmer
============

Meta's 30B dense agentic VLM (text + image in, text out): sliding-window and
NoPE global attention layers 3:1, plus a windowed ViT perception encoder.

Registry key: ``muse_glimmer`` *(Beta)*. Checkpoint: ``meta-models/Muse-Glimmer-30B``.

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``muse_glimmer``
     - ``configs/muse_glimmer.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve muse_glimmer

For client requests and response formats, see :doc:`/clients`.

Meta's Muse Glimmer 30B: text-and-image chat (video is refused), on ``POST
/generate`` and ``/v1/chat/completions``. bf16 needs one 141 GB GPU, or two
80 GB ones::

    mstar serve muse_glimmer --gpus 0
    mstar serve muse_glimmer --config configs/muse_glimmer_tp2.yaml --gpus 0,1

Attention and request controls
------------------------------

Three in four layers attend a 2048-token sliding window; every layer writes the one paged KV cache, and the sliding layers attend it through a windowed attention resource. The cache keeps only the window of the sliding layers' K/V (``AttentionConfig.layers``) and returns the pages behind it to the pool, so a 32k-token request holds about 3.4x less KV than with every layer's history. A page holds 192 tokens of the full layers, or 64 of the sliding ones. Such a cache doesn't prefix cache yet, so the configs set ``prefix_cache: false``.

Sampling defaults to the checkpoint's (temperature 1, top_p 0.95, top_k 64); a request's ``min_p`` above 0 is refused (a 400) unless the checkpoint's generation_config sets a ``min_p``, which turns the sampler's min-p filter on. ``reasoning_effort`` (``low``, ``medium``, ``high``, ``xhigh``; default ``high``) sets the template's reasoning strength, and ``enable_thinking: false`` means ``low``, as the template has no off: the model still reasons, at the template's lowest strength. Either, or the template's own ``reasoning_strength`` and ``current_date``, may also come in ``chat_template_kwargs``; a flat key wins. The reply is the model's channel stream as generated: reasoning addressed ``to=self``, then the answer ``to=user``.

Tool calls are refused for now. Images resize with the reference's antialiased Lanczos, which torchvision has from 0.27 (torch 2.12); an older torchvision falls back to PIL's Lanczos with a warning, and its pixels differ slightly.
