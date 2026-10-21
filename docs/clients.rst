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
     - Comma-separated input modalities, one entry per prompt element in order.
       Auto-detected from the uploads and text when omitted, which is what keeps
       the ordering and the count of same-modality attachments; an explicit list
       replaces it.
   * - ``output_modalities``
     - ``text``
     - Comma-separated desired outputs (e.g. ``text``, ``image``, ``audio``, ``video``,
       ``video_frame``, ``action``). ``video_frame`` is streaming-only raw RGB24.
   * - ``streaming``
     - ``true``
     - ``true`` → NDJSON stream of chunks; ``false`` → one JSON document.
   * - ``model_kwargs``
     - —
     - JSON object of model-specific parameters (e.g. ``{"voice": "tara"}``).
   * - ``request_id``
     - *(uuid)*
     - Optional client-supplied id; the server generates one when omitted. An id
       that still has a request in flight is refused with ``409``; it is free
       again once that request finished, failed or was aborted (a client that
       drops a stream can resubmit its id at once).

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
arrive. A request that fails after its stream has started ends it with one chunk of
modality ``error`` (the HTTP status in ``metadata.status``); nothing follows it. A client that sends ``Accept: application/vnd.mstar.frames`` receives
length-framed binary chunks instead, which skips the base64 pass on multi-megabyte
``video_frame`` payloads; a server without the framing keeps answering NDJSON, so a
client must parse by the response ``Content-Type``. ``GET /health`` returns
``{"status": "healthy"}``.

.. code-block:: bash

   # text (non-streaming → JSON)
   curl -s http://localhost:8000/generate -F 'text=Hello' -F 'streaming=false'

   # image understanding (image in, text out)
   curl -s http://localhost:8000/generate -F 'text=What is in this image?' -F 'files=@cat.jpg'

   # text-to-speech (base64 PCM in outputs.audio)
   curl -s http://localhost:8000/generate \
     -F 'text=hello there' -F 'output_modalities=audio' \
     -F 'model_kwargs={"voice":"tara"}' -F 'streaming=false'

WebSocket ``/generate/ws``
--------------------------

``/generate/ws`` is the same request over one persistent WebSocket, for control loops
that cannot afford an HTTP round trip per step (robot policies, streaming world models).
Each message is one request with the ``/generate`` fields — ``text``, ``files`` as
``[{"name": ..., "data": ...}]``, ``input_modalities``, ``output_modalities``,
``model_kwargs``, ``request_id`` — sent either as a JSON text frame (``data`` base64) or
as a msgpack binary frame (``data`` raw bytes). Replies use the same encoding as the
message: one frame per result chunk, ``{"request_id", "modality", "data", "metadata"}``,
then ``{"request_id", "finish": true}``. A rejected message answers
``{"request_id", "error": ...}`` and the socket stays open; a request that fails after it
was accepted ends with one such reply. Both carry the HTTP ``status`` when there is one
(``409`` for a ``request_id`` still in flight, as on ``/generate``), and no
``finish`` follows. Messages may be pipelined —
send the next observation before the current action chunk has returned — and the
``request_id`` tells the replies apart. Pipelined messages are submitted in the order
they were sent, so of two under one ``request_id`` it is the later one that gets the
``409`` while the earlier one runs; their replies may interleave. Closing the socket aborts whatever is still in flight.

.. code-block:: python

   import msgpack, numpy as np, websockets.sync.client

   with websockets.sync.client.connect("ws://localhost:8000/generate/ws", max_size=None) as ws:
       ws.send(msgpack.packb({
           "text": "pick up the mug",
           "files": [{"name": "obs.jpg", "data": open("obs.jpg", "rb").read()}],
           "output_modalities": ["action"],
           "model_kwargs": {"action_mode": "policy", "domain_name": "droid_lerobot",
                            "raw_action_dim": 8, "action_chunk_size": 32},
           "request_id": "step-0",
       }, use_bin_type=True))
       while True:
           reply = msgpack.unpackb(ws.recv(), raw=False)
           if reply.get("modality") == "action":
               actions = np.frombuffer(reply["data"], dtype=np.float32).reshape(32, -1)
           if reply.get("finish") or reply.get("error"):
               break

``examples/cosmos3_action_ws_client.py`` is a complete openpi-style client that runs this
loop at a fixed observation rate and reports chunks/s, actions/s and latency percentiles.

Python SDK
----------

The SDK (:class:`mstar.client.MStarClient`) is a thin HTTP client over ``/generate``. It
depends only on ``requests`` (plus ``numpy`` for the audio helpers) — no torch — so it can
run anywhere:

.. code-block:: python

   from mstar import MStarClient
   client = MStarClient("http://localhost:8000")   # optional: timeout=600.0

If the server requires an API key (``--api-key`` / ``MSTAR_API_KEY``, see
:ref:`api-key`), pass ``api_key="..."``; without one the client reads ``MSTAR_API_KEY``
from its own environment, and sends nothing when that is unset too.

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
     - PNG ``bytes`` (e.g. BAGEL, FLUX.2 klein text-to-image).
   * - ``edit_image(prompt, image, **kw)``
     - PNG ``bytes`` for an edit of ``image`` (a path, bytes, or a list of them for
       multi-reference models).
   * - ``tts(text, *, voice=None, **kw)``
     - An ``AudioBuffer`` (``.to_wav(path)``, ``.to_numpy()``, ``len(...)`` samples).
   * - ``voices()``
     - The ``voice`` ids the served speech model accepts (``GET /v1/audio/voices``).
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
  ``AudioChunk(pcm, sample_rate)``, and ``VideoFrameChunk(data, metadata)``. A
  video-frame chunk validates its width, height, fps, pixel format and frame range;
  ``.to_numpy()`` returns a zero-copy ``[frame_count, height, width, 3]`` uint8 view.
  Raw ``video_frame`` requests require ``stream=True``. The SDK asks for the binary
  framing automatically for ``video_frame``; other modalities opt in with
  ``MStarClient(prefer_binary=True)``. The server does not pace generation to the
  consumer: frames are produced at model speed and buffered by the API server until
  read, bounded by the request's frame count, so a consumer slower than realtime
  accumulates that backlog in server memory (about 11 MiB per chunk at 720p).

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

``api_key`` can be anything unless the server sets a key (:ref:`api-key`); then pass
that key, which the client sends as ``Authorization: Bearer <key>``.

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
     - ``kokoro``, ``orpheus``, ``qwen3_omni``
     - Text-to-speech (``kokoro`` streams sentence by sentence with ``stream=True``).
   * - ``GET /v1/audio/voices``
     - speech models that publish a voice list through their speech adapter (``kokoro``, ``orpheus``; 404 otherwise)
     - The ``voice`` ids the served model accepts, plus its default.
   * - ``POST /v1/audio/transcriptions``
     - ``whisper_large``, ``higgs_audio``, ``omnilingual_asr_ctc_*``
     - Speech-to-text (multipart upload; ``json`` / ``text`` / ``verbose_json`` /
       ``srt`` / ``vtt``; streaming via ``transcript.text.delta`` events).
   * - ``WS /v1/realtime?intent=transcription``
     - models whose adapter can continue a hypothesis
     - Streaming speech-to-text: ``input_audio_buffer.append`` PCM16 chunks in,
       ``conversation.item.input_audio_transcription.delta`` (append-only) and
       ``mstar.transcription.partial`` (whole current hypothesis) out; ``commit``
       finishes the utterance. Input is ``pcm16`` at 16 kHz (the default) or
       ``{"type": "audio/pcm", "rate": 24000}``, resampled to 16 kHz. Turn
       detection is null (client commits) or ``server_vad``; see below.
   * - ``POST /v1/images/generations``
     - ``bagel``
     - Text-to-image.
   * - ``POST /v1/images/edits``
     - ``bagel``
     - Image editing (image + prompt → image).

Models without an OpenAI surface (``pi05``, ``vjepa2``, ``vjepa2_ac``, ``waypoint``)
return ``404`` on ``/v1/*``; use ``/generate`` or the SDK for them. In particular,
Waypoint emits live RGB frame chunks and is not routed through the encoded-video
``/v1/videos/generations`` endpoint.

.. code-block:: python

   # chat
   client.chat.completions.create(model="bagel", messages=[{"role": "user", "content": "hi"}])

   # text-to-speech
   client.audio.speech.create(model="orpheus", input="hello there", voice="tara")
   # (mstar extension) the voices the served model accepts
   requests.get("http://localhost:8000/v1/audio/voices").json()["voices"]

   # speech-to-text
   client.audio.transcriptions.create(model="whisper_large", file=open("speech.wav", "rb"),
                                      language="en")

   # image generation
   client.images.generate(model="bagel", prompt="a cat in a hat")

Per-model notes:

- **BAGEL** — chat returns text only; use ``/v1/images/generations`` and
  ``/v1/images/edits`` for image output.
- **FLUX.2 klein / Z-Image** — ``/v1/images/generations`` (and ``/v1/images/edits`` for
  klein) only: ``size`` as ``WxH`` (multiples of 16); ``seed``; ``num_inference_steps``,
  ``output_format`` (``png`` default, ``jpeg``, ``webp``), ``output_compression`` (JPEG /
  WebP quality) and ``png_compress_level`` (0-9, default 1) via ``extra_body``.
- **Qwen3-Omni** — text sampling uses ``thinker_*`` keys, speech uses ``talker_*``, and the
  residual codec groups use ``code_predictor_*``; set the speaker with ``voice`` (default
  ``Ethan``) and request audio output by including ``"audio"`` in ``modalities``.
  Non-OpenAI knobs (e.g. ``talker_top_k``, ``code_predictor_top_p``) go through
  ``extra_body``.
- **Whisper / Higgs-Audio** — ``language`` (ISO-639-1) skips language detection,
  ``prompt`` conditions the decoder on prior text (it reaches the model as
  ``initial_prompt``), and ``response_format``
  ``verbose_json`` / ``srt`` / ``vtt`` (or ``timestamp_granularities[]``) asks the
  model for timestamps. Whisper's language and timestamp tokens travel in the
  text stream and are lifted into ``language`` / ``segments`` by the server; a
  streaming client receives only the spoken words. Uploads longer than the
  model's clip (30 s for Whisper) are served as consecutive windows. By default
  they run in order, openai-whisper style: each window gets the transcript so
  far as ``initial_prompt`` and the first window's detected language, is decoded
  with timestamps so the next window can start where its last closed segment
  ended (no word is split by a boundary), and is decoded again when its text is
  a repetition loop (gzip compression ratio above 2.4): first without the
  conditioning text, then at rising temperatures — after which the transcript
  so far stops conditioning later windows.
  Leave ``temperature`` at 0 to get that fallback; a pinned temperature is used
  as is. ``long_form="parallel"`` in ``extra_body`` submits fixed windows all at
  once, each cut at the quietest moment before its boundary. Segment timestamps
  are offset to the whole file. For Whisper, ``window_seconds`` in ``extra_body``
  (5 to 30) sets the window length for one request, and an upload longer than it
  is windowed. An upload, like any audio a request attaches, is refused once it
  decodes past 30 min, whatever its header says (a 413 here, a 400 from the data
  worker). The decode up to that point still runs, so this bounds memory, not
  time: refusing a 31 min file takes about as long as decoding it.
- **Omnilingual ASR** (``omnilingual_asr_ctc_*``): no language code is needed for any of
  its 1,600+ languages, and ``language`` / ``prompt`` / ``temperature`` have no effect. A
  clip of up to 20 s is one request; a longer upload is cut into windows of about equal
  length, at most 16 s, at quiet points and transcribed in parallel (``window_seconds``
  in ``extra_body``, 5 to 40, sets both lengths). There are no timestamps: ``srt`` / ``vtt``
  get one cue spanning the file. Nothing detects the language: ``verbose_json``'s
  ``language`` is the request's ``language`` as given, or null.
- **Realtime transcription** — every ``chunk_seconds`` (default 2 s) of new audio the
  session re-transcribes everything heard so far as one engine request whose assistant
  turn is prefilled with the previous hypothesis minus its last ``unfixed_tokens``
  tokens (the Qwen3-ASR SDK's streaming algorithm), so only the tail is ever revised.
  Tune ``chunk_seconds`` / ``unfixed_chunks`` / ``unfixed_tokens`` under ``session.mstar``
  in ``transcription_session.update``; a smaller ``chunk_seconds`` is raised to 0.5. An item holds
  at most 300 s of audio (``max_item_seconds`` in the session, lower if the model hears
  less): commit before then, since an append past it is refused with an ``error``.
- **Realtime turn detection** — by default the client ends each utterance with
  ``input_audio_buffer.commit``. With ``"turn_detection": {"type": "server_vad"}`` the
  server does it, as OpenAI's server VAD does: ``threshold`` (default 0.5),
  ``prefix_padding_ms`` (300) and ``silence_duration_ms`` (500). Silero VAD scores each
  32 ms frame; a turn starts on a frame at or above ``threshold`` with up to
  ``prefix_padding_ms`` of the audio before it. Silence starts on a frame below
  ``threshold`` minus 0.15, and a frame at or above ``threshold`` cancels it; the turn
  ends on whichever frame completes ``silence_duration_ms`` of whole frames counted from
  that first silent frame, borderline frames included. The hysteresis is that of
  Silero's own ``VADIterator``, but ``VADIterator`` counts from the frame after the first
  silent one and ends only on a frame below ``threshold`` minus 0.15, so with a nonzero
  ``silence_duration_ms`` it ends a turn one frame or more later. The server then sends
  ``input_audio_buffer.speech_started`` (``audio_start_ms``),
  ``input_audio_buffer.speech_stopped`` (``audio_end_ms``) and ``committed``, and the
  transcription completes. Times are milliseconds of audio since the session started,
  cleared audio included. A commit takes exactly the turn's audio; what follows, even
  in the same append, goes to the next turn. A turn never passes the item cap: it is
  committed before the frame that would pass it, and speech goes on in a new turn that
  starts with that frame. A client ``commit`` ends an open turn; between turns it
  drops what the VAD has not put in a turn yet. ``create_response`` and
  ``interrupt_response`` are accepted and echoed but do nothing in a transcription
  session; ``idle_timeout_ms`` and other types are refused. With 24 kHz input each
  decision waits 3 ms more for the resampler. Install ``mstar[vad]`` (Silero VAD, MIT,
  bundled 2 MB ONNX weights, run by onnxruntime on one CPU thread) or point
  ``MSTAR_SILERO_VAD_PATH`` at the ONNX file (``silero_vad.onnx``; the package's other
  exports take other inputs, and with one of them ``server_vad`` is refused); weights are
  never downloaded. The Dynamo bridge has no turn detection yet and
  refuses it.
- **Kokoro** — ``voice`` is one of the 54 bundled voices (default ``af_heart``) or a
  blend such as ``af_bella+af_sky`` or ``af_bella(2)+af_sky(1)``; ``speed`` scales the
  speaking rate (0.25-4.0). ``lang_code`` and ``phonemes`` go through ``extra_body``.
  ``temperature`` / ``top_p`` are ignored: Kokoro does not sample.
- **Orpheus** — set the speaker with ``voice`` — one of ``tara`` (default), ``zoe``,
  ``zac``, ``jess``, ``leo``, ``mia``, ``julia``, ``leah`` (the ``available_voices`` list
  in the Orpheus config).
