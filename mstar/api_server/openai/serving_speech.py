"""/v1/audio/speech and /v1/audio/voices handlers (text-to-speech).

Non-streaming speech returns the full audio as a container blob (WAV by
default). Streaming returns one open-ended response as the audio is produced:
a WAV header followed by PCM16 frames, or, with ``response_format="pcm"``,
the bare PCM16 frames that the OpenAI TTS clients (LiveKit, Pipecat) expect.
A request streams when it sets ``stream`` or OpenAI's ``stream_format``, which
is what the OpenAI SDK sends: ``"audio"`` streams the bytes as above, ``"sse"``
wraps them in ``speech.audio.delta`` events closed by ``speech.audio.done``.
``/v1/audio/voices`` lists the ``voice`` values the served model accepts.

Long inputs can be synthesized as ordered sentence chunks (one engine request
per chunk, ``speech_chunking.split_sentences``): the adapter's
``speech_chunk_min_chars`` turns this on for long texts, and a client can force
or suppress it per request with ``sentence_chunking: true|false``. The next
chunks are submitted while the current one streams, so the engine batches them
and playback never waits for a prefill. An input needing more than the
adapter's ``speech_chunk_max_pieces`` chunks is rejected.

A streaming request only commits to HTTP 200 once its first result chunk has
arrived and is not an error; an error chunk before that becomes the HTTP error
it carries (the non-streaming path gets the same from ``collect_results``).

The adapter maps the request before anything is submitted, and that step can
decode an uploaded clip or, where the server allows it, fetch one over HTTP.
It runs in a worker thread so a slow reference cannot stall the streams the
event loop is serving.
"""

from __future__ import annotations

from mstar.api_server.openai.protocol import VoiceCard, VoiceList


def list_voices(api) -> VoiceList | None:
    """The served model's voices, or ``None`` when it has no fixed list."""
    model = api.model
    voices = model.get_voices() if model is not None else None
    if voices is None:
        return None
    return VoiceList(
        voices=[VoiceCard(id=v, name=v) for v in voices],
        default_voice=model.get_default_voice(),
    )

