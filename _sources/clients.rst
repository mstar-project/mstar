Using a Server
==============

Once a server is running (see :doc:`serving`), you can reach it three ways: the native
``/generate`` endpoint, the Python SDK, or the OpenAI-compatible API. Every model is
reachable via ``/generate`` and the SDK; the OpenAI routes cover the chat, speech, and
image models.

Native ``/generate``
--------------------

``POST /generate`` takes a multipart form and returns either a single JSON document or an
NDJSON stream.

.. list-table:: Form fields
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Default
     - Meaning
   * - ``text``
     - —
     - Text prompt (optional if media is provided).
   * - ``files``
     - —
     - One or more media uploads; each file's modality is inferred from its extension.
   * - ``input_modalities``
     - auto
     - Comma-separated input modalities; auto-detected from the data when omitted.
   * - ``output_modalities``
     - ``text``
     - Comma-separated desired outputs (e.g. ``text``, ``image``, ``audio``, ``video``,
       ``action``).
   * - ``streaming``
     - ``true``
     - ``true`` → NDJSON stream of chunks; ``false`` → one JSON document.
   * - ``model_kwargs``
     - —
     - JSON object of model-specific parameters (e.g. ``{"voice": "tara"}``).
   * - ``request_id``
     - *(uuid)*
     - Optional client-supplied id; the server generates one when omitted.

A non-streaming response groups outputs by modality, each payload base64-encoded:

.. code-block:: json

   {
     "request_id": "…",
     "outputs": {
       "text":  [{"data": "<base64>",     "metadata": {}}],
       "image": [{"data": "<base64-png>", "metadata": {}}]
     }
   }

A streaming response is ``application/x-ndjson`` — one JSON object per line as chunks
arrive. ``GET /health`` returns ``{"status": "healthy"}``.

.. code-block:: bash

   # text (non-streaming → JSON)
   curl -s http://localhost:8000/generate -F 'text=Hello' -F 'streaming=false'

   # image understanding (image in, text out)
   curl -s http://localhost:8000/generate -F 'text=What is in this image?' -F 'files=@cat.jpg'

   # text-to-speech (base64 PCM in outputs.audio)
   curl -s http://localhost:8000/generate \
     -F 'text=hello there' -F 'output_modalities=audio' \
     -F 'model_kwargs={"voice":"tara"}' -F 'streaming=false'

Python SDK
----------

The SDK (:class:`mstar.client.MStarClient`) is a thin HTTP client over ``/generate``. It
depends only on ``requests`` (plus ``numpy`` for the audio helpers) — no torch — so it can
run anywhere:

.. code-block:: python

   from mstar import MStarClient
   client = MStarClient("http://localhost:8000")   # optional: timeout=600.0

The core method is ``generate``:

``generate(*, text=None, images=None, audio=None, video=None, output_modalities=("text",), input_modalities=None, stream=False, request_id=None, **model_kwargs)``
   Submit a request. ``images`` / ``audio`` / ``video`` accept a path, raw ``bytes``, a
   ``(filename, bytes)`` tuple, or a list of those. Extra keyword args are forwarded as the
   model's ``model_kwargs`` (e.g. ``voice="tara"``, ``temperature=0.7``,
   ``max_output_tokens=256``); ``None`` values are dropped. Returns a ``GenerateResult``
   when ``stream=False``, or an iterator of stream events when ``stream=True``.

Convenience wrappers:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Returns
   * - ``chat(prompt, *, images=None, audio=None, output_modalities=("text",), stream=False, **kw)``
     - Text generation (and, with ``output_modalities=("text", "audio")``, speech).
   * - ``generate_image(prompt, **kw)``
     - PNG ``bytes`` (e.g. BAGEL text-to-image).
   * - ``tts(text, *, voice=None, **kw)``
     - An ``AudioBuffer`` (``.to_wav(path)``, ``.to_numpy()``, ``len(...)`` samples).
   * - ``stream(**kw)``
     - Sugar for ``generate(stream=True, ...)``.
   * - ``health()``
     - ``True`` if the server is healthy.

Result and event types live in ``mstar.client``:

- ``GenerateResult`` — ``.text``, ``.images`` (list of PNG bytes), ``.audio``
  (an ``AudioBuffer`` or ``None``), ``.raw``; plus ``.save_image(path)`` /
  ``.save_audio(path)``.
- ``AudioBuffer`` — decoded PCM with ``.sample_rate``; ``.to_wav(path)``, ``.to_numpy()``,
  ``len(...)``.
- Stream events — ``TextChunk(text)``, ``ImageChunk(data)`` (``.save(path)``),
  ``AudioChunk(pcm, sample_rate)``.

.. code-block:: python

   res = client.chat("Hello!")                       # GenerateResult
   print(res.text)

   open("cat.png", "wb").write(client.generate_image("a cat in a hat"))

   client.tts("Hi there", voice="tara").to_wav("out.wav")

   for event in client.stream(text="Tell me a story"):
       print(getattr(event, "text", ""), end="", flush=True)

OpenAI-compatible API
---------------------

``mstar`` mounts OpenAI-style routes under ``/v1`` for the models with standard OpenAI
semantics. Point any OpenAI client at ``http://<host>:<port>/v1``:

.. code-block:: python

   from openai import OpenAI
   client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

Endpoints and model coverage:

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Endpoint
     - Models
     - Notes
   * - ``GET /v1/models``
     - all
     - Lists the served model.
   * - ``POST /v1/chat/completions``
     - ``bagel``, ``qwen3_omni``
     - Text chat (streaming + non-streaming). Qwen3-Omni can also emit speech.
   * - ``POST /v1/audio/speech``
     - ``orpheus``, ``qwen3_omni``
     - Text-to-speech.
   * - ``POST /v1/images/generations``
     - ``bagel``
     - Text-to-image.
   * - ``POST /v1/images/edits``
     - ``bagel``
     - Image editing (image + prompt → image).

Models without an OpenAI surface (``pi05``, ``vjepa2``, ``vjepa2_ac``) return ``404`` on
``/v1/*``; use ``/generate`` or the SDK for them.

.. code-block:: python

   # chat
   client.chat.completions.create(model="bagel", messages=[{"role": "user", "content": "hi"}])

   # text-to-speech
   client.audio.speech.create(model="orpheus", input="hello there", voice="tara")

   # image generation
   client.images.generate(model="bagel", prompt="a cat in a hat")

Per-model notes:

- **BAGEL** — chat returns text only; use ``/v1/images/generations`` and
  ``/v1/images/edits`` for image output.
- **Qwen3-Omni** — text sampling uses ``thinker_*`` keys and speech uses ``talker_*``; set
  the speaker with ``voice`` (default ``Ethan``) and request audio output by including
  ``"audio"`` in ``modalities``. Non-OpenAI knobs (e.g. ``talker_top_k``) go through
  ``extra_body``.
- **Orpheus** — set the speaker with ``voice`` — one of ``tara`` (default), ``zoe``,
  ``zac``, ``jess``, ``leo``, ``mia``, ``julia``, ``leah`` (the ``available_voices`` list
  in the Orpheus config).
