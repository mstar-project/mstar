Ripple-65B
==========

Ricursive's 65B dense text LLM: sliding-window and NoPE global attention layers
4:1, with a value residual from the token embeddings. BF16, TP2 or TP4.

Registry key: ``ripple_65b`` *(Experimental)*. Checkpoint: ``ricursiveAI/Ripple-65B-Preview``.

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``ripple_65b``
     - ``configs/ripple_65b_tp4.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve ripple_65b

For client requests and response formats, see :doc:`/clients`.

.. code-block:: bash

    mstar serve ripple_65b --gpus 0,1,2,3
    mstar serve ripple_65b --config configs/ripple_65b_tp2.yaml --gpus 0,1

- ``configs/ripple_65b_tp4.yaml``, the default, needs four 80 GB GPUs and
  ``configs/ripple_65b_tp2.yaml`` two 141 GB ones. TP can't pass the model's 8 KV heads. Both
  serve 131072 tokens a request.
  For more, raise ``resources.kv_cache.max_seq_len`` and ``max_num_pages`` together: a prompt
  holds all 80 layers' K/V until its prefill commits, so the shipped pool runs out near 260K
  tokens and two 128K prompts can't prefill at once.
- A chat with no ``temperature`` samples at 1.0, as Ricursive's server does, and one with no
  ``max_tokens`` stops at 2048 tokens, which a long reasoning reply can reach inside
  ``<think>``. Set ``max_tokens`` for hard problems.
- Text in, text out, on ``/generate`` and ``/v1/chat/completions``. A reply's reasoning stays
  inline as ``<think>...</think>``, and a chat that asks for tool calls is refused.
- Not supported yet: the NVFP4 checkpoint, prefix caching (a pool that keeps a sliding window
  doesn't prefix cache) and DP replicas (a window pool can't span instances).
- The GPU parity scripts live in ``test/ripple/``: ``run_parity.py`` drives a server and gates
  the result, and ``record_oracle.py`` teacher-forces its replies through the checkpoint's own
  ``modeling_ricursive.py``, in its own venv, with the attention gate and the value residual
  each turned off as controls.
