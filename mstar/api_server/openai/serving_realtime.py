"""``/v1/realtime``: OpenAI's Realtime API over a WebSocket, two intents.

The conversation intent (:class:`RealtimeConversation`, GA events) serves a
full-duplex model, one that declares a ``ClientAudio`` client stream and
decides when to speak; see its docstring. The transcription intent
(``?intent=transcription``, :class:`RealtimeTranscription`, the beta
transcription-session events) streams speech-to-text with partial results.
Both take audio in OpenAI's formats, converted on the socket to what the
model hears.

The transcription intent is a subset of OpenAI's Realtime transcription
session, enough for a client that streams microphone audio and wants partial
transcripts:

Client events
    ``transcription_session.update``  ``{"session": {"input_audio_transcription":
    {"model", "language", "prompt"}, "input_audio_format", "turn_detection",
    "mstar": {"chunk_seconds", "unfixed_chunks", "unfixed_tokens"}}}``
    ``input_audio_buffer.append``      ``{"audio": <base64 PCM16 mono>}``
    ``input_audio_buffer.commit``      end of the utterance: flush and finish
    ``input_audio_buffer.clear``       drop buffered audio and start over

``input_audio_format`` is ``"pcm16"`` (PCM16 at 16 kHz, the default; OpenAI
reads it as 24 kHz), ``"g711_ulaw"`` or ``"g711_alaw"`` (8 kHz), or a GA format
object (``{"type": "audio/pcm", "rate": N}`` at a rate in ``REALTIME_RATES``,
``{"type": "audio/pcmu"}``, ``{"type": "audio/pcma"}``).
Audio at another rate is resampled to 16 kHz as it arrives, so partial
``audio_seconds`` stay on the client's timeline. The format changes only with
an empty buffer.

``turn_detection`` null (the default) leaves ending an utterance to the
client's commit. ``{"type": "server_vad", "threshold", "prefix_padding_ms",
"silence_duration_ms"}`` lets the server end it (``realtime_vad``), adding
``input_audio_buffer.speech_started`` / ``.speech_stopped`` events with
``audio_start_ms`` / ``audio_end_ms`` on the session's audio timeline; a
turn that reaches the item cap is committed there instead of refused.
``{"type": "semantic_vad", "eagerness"}`` asks an end-of-turn model at each
pause instead, and the update is refused when that model is unavailable.

``chunk_seconds`` below ``MIN_CHUNK_SECONDS`` is raised to it and must not
pass the item cap, and the unfixed counts are non-negative integers; numeric
strings count as numbers. ``language`` and ``prompt`` are strings (or null),
``language`` at most ``MAX_LANGUAGE_CHARS`` characters. Any other value is an
``error`` and the update changes nothing.

A failed step is an ``error`` too. A request error (a 4xx from the engine,
such as a model refusing the language) quotes its message, with server paths
reduced to their base names and cut to ``MAX_ERROR_CHARS``; any other failure
says only that the step failed, with an error id that the server log carries
the details under.

Server events
    ``transcription_session.created`` / ``.updated``
    ``conversation.item.input_audio_transcription.delta``      append-only text
    ``mstar.transcription.partial``   the whole current hypothesis, unstable tail included
    ``conversation.item.input_audio_transcription.completed``  ``{"transcript"}``
    ``error``

The decoding follows the Qwen3-ASR SDK's streaming algorithm: every
``chunk_seconds`` of new audio, the audio heard so far is transcribed again
as one engine request whose assistant turn is prefilled with the previous
hypothesis minus its last ``unfixed_tokens`` tokens (once ``unfixed_chunks``
chunks are in), so the tail can still be revised while the head is fixed.
The adapter decides how that prefix reaches its model
(``realtime_step_request``). Each step is an ordinary request, so sessions
batch on the engine like any other traffic.

Since every step decodes the whole item again, an item's work grows with the
square of its length. An item is capped at ``MAX_ITEM_SECONDS`` (or the
adapter's ``max_audio_seconds`` if shorter): an append past the cap is an
``error`` and the client commits or clears. An append that completes several
chunks at once is decoded as one step.
"""

from __future__ import annotations

import base64
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

def decode_base64(payload: str) -> bytes:
    """An append's base64 payload, strictly: whitespace is allowed
    (base64.encodebytes wraps lines), anything else outside the alphabet
    raises, as does a payload that is not a string."""
    # translate stays linear in the payload where split() would build a list of its pieces
    return base64.b64decode(payload.encode("ascii").translate(None, b" \t\n\r\v\f"), validate=True)


def server_event(kind: str, **fields) -> dict:
    """A server event: ``kind`` with a fresh ``event_id``."""
    return {"event_id": f"event_{uuid.uuid4().hex[:12]}", "type": kind, **fields}


def error_event(message: str, kind: str = "invalid_request_error", code: str | None = None) -> dict:
    error = {"type": kind, "message": message}
    if code is not None:
        error["code"] = code
    return server_event("error", error=error)


def response_object(response_id: str, status: str, modalities: list[str], output: list | None = None,
                    status_details: dict | None = None) -> dict:
    """A GA ``realtime.response``, as ``response.created`` / ``.done`` carry it."""
    response = {"id": response_id, "object": "realtime.response", "status": status,
                "output_modalities": modalities, "max_output_tokens": "inf", "output": output or []}
    if status_details is not None:
        response["status_details"] = status_details
    return response


def content_event(kind: str, response_id: str, item_id: str, **fields) -> dict:
    """A GA event about a response's one content part (``response.output_audio.delta`` and the like)."""
    return server_event(kind, response_id=response_id, item_id=item_id, output_index=0, content_index=0, **fields)

