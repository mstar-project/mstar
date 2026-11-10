Higgs Audio
===========

Audio-tower + Qwen3 LLM speech-to-text. Beta / un-optimized.

Registry key: ``higgs_audio`` *(Beta)*. Checkpoint: ``bosonai/higgs-audio-v3-stt``.

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``higgs_audio``
     - ``configs/higgs_audio.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve higgs_audio

For client requests and response formats, see :doc:`/clients`.

Audio encoding and capture
--------------------------

The audio tower runs outside ``torch.compile`` (a clip's chunk count changes
its shape, and every new shape recompiled on the serving path). Requests
whose 4 s chunks share a mel width are encoded together, and full-width
chunks replay a CUDA graph per chunk count; a clip under one chunk is padded
to the full width and masked (``HiggsAudioTower``'s ``valid_frames``) so it
replays a graph too. ``model_kwargs: {encoder_graphs: false}`` runs the
tower eagerly.

Long audio
----------

The model drops or repeats text on long audio, so ``/v1/audio/transcriptions``
cuts an upload longer than 30 s into windows of at most 30 s run in parallel.

Prompt prefill
--------------

One prefill step per request: the ChatML text around the audio and the audio
embeddings go into the KV cache as one span.

Token delivery
--------------

Decode keeps each request's last token in the sampler's slot master and
sends the client's copy inline (``MSTAR_DEVICE_LOOPBACK``, on by default),
as Qwen3.5 does: a decode step stores and routes no tensors.
