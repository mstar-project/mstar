GLM-5.2 and GLM-5.3-Flash
=========================

``glm5_next``
-------------

GLM-5.3-Flash (320B/18B): hybrid KDA linear-attention + MLA MoE causal
LM, text in / text out, fp8 checkpoint, TP8. ``configs/glm5_next_tp8_mtp.yaml``
drafts with MTP k=3 (greedy only).

Checkpoint: ``zai-org/GLM-5.3-Flash``.

``glm52``
---------

GLM-5.2 (753B/40B): MLA + DSA MoE causal LM, fp8 block-scale weights, TP8.
``model_kwargs.kernels`` is ``auto`` (the tuned kernels) or ``reference``, the numerics
anchor, served eager. ``configs/glm52_tp8_mtp.yaml`` drafts with MTP k=3 (greedy only).

Checkpoint: ``zai-org/GLM-5.2-FP8``.

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``glm52``
     - ``configs/glm52_tp8.yaml``
   * - ``glm5_next``
     - ``configs/glm5_next_tp8.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve glm52

For client requests and response formats, see :doc:`/clients`.

- Sampling defaults come from the checkpoint's ``generation_config.json`` (temperature
  1.0 and top_p 0.95 for both), read from a local checkpoint or the cached Hub snapshot;
  the conductor never fetches it, and without it the same values apply. A request may set
  ``temperature``, ``top_p``, ``top_k``, ``min_p``, ``repetition_penalty``,
  ``penalize_prompt`` and ``ignore_eos``. ``min_p`` above 0 is refused, since neither
  checkpoint defaults one. With MTP drafting, only greedy, unpenalized sampling is served.
- GLM-5.2's ``max_tokens`` may run to its serving window (``index_topk``, or
  ``max_seq_len`` with ``dsa_long_context``). GLM-5.3-Flash holds contexts to
  ``index_topk`` (2048), where dense MLA computes exactly what its sparse attention does,
  so ``max_tokens`` above 2048 is a 400 rather than being shortened; under MTP, a
  ``max_tokens`` past the room the prompt leaves is refused too.
