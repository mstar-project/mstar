Orpheus
============

TTS: Llama 3.2 3B LLM emitting audio tokens + SNAC 24 kHz decoder.

Registry key: ``orpheus``. Checkpoint: ``canopylabs/orpheus-3b-0.1-ft``.

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``orpheus``
     - ``configs/orpheus_colocated.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve orpheus

For client requests and response formats, see :doc:`/clients`.

Sampling and numerical differences
----------------------------------

Orpheus only ever emits ``<custom_token_*>`` ids (SNAC codes and the
speech/turn markers), so its LM head computes the logits of those rows alone
(``ColumnParallelLinear(output_start=...)``, about 18% of the head) and reads
``-inf`` for the Llama text rows, at every TP size. Sampling therefore
renormalizes over the custom tokens: the reference samples the full
vocabulary, where a text token's probability during speech is negligible but
not zero, so sampled output is not identical to the reference's.

Long text and sentence chunks
-----------------------------

``/v1/audio/speech`` input of 250+ characters is synthesized as sentence chunks of
about 200: the default 2,048 output tokens are about 25 s of speech, and a longer
input would be cut off there (``sentence_chunking: false`` opts out). An input of
more than 32 chunks (some 6,400 characters) is a 400.
