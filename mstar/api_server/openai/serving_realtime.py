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

``input_audio_format`` is ``"pcm16"`` (PCM16 at 24 kHz, as OpenAI reads it),
``"g711_ulaw"`` or ``"g711_alaw"`` (8 kHz), or a GA format object
(``{"type": "audio/pcm", "rate": N}`` at a rate in ``REALTIME_RATES``, ``{"type": "audio/pcmu"}``,
``{"type": "audio/pcma"}``); left out, it is PCM16 at the model's 16 kHz.
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

import asyncio
import base64
import contextlib
import itertools
import json
import logging
import math
import re
import reprlib
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from mstar.api_server import media_io
from mstar.api_server.openai import realtime_vad
from mstar.api_server.openai._util import error_type, now, rid
from mstar.api_server.openai.protocol import TranscriptionRequest
from mstar.streaming.topology import Connection
from mstar.utils.utf8_stream import Utf8Stream, decode_joined

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# OpenAI's PCM16 rate, and G.711's
OPENAI_PCM_RATE = 24000
G711_RATE = 8000
# The PCM rates a socket takes. A closed set bounds what resampling one
# append costs (at most 6x its samples, about 100 taps each), which a bound on
# the kernel's size alone does not.
REALTIME_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000)
# Appends up to this many input samples (4 ms at 24 kHz) cost less to resample
# than a thread hop, so they resample on the event loop; longer ones go to a thread.
INLINE_RESAMPLE_SAMPLES = 96
# Longest server_vad prefix padding or silence window.
MAX_VAD_MS = 10_000
DEFAULT_CHUNK_SECONDS = 2.0
DEFAULT_UNFIXED_CHUNKS = 2
DEFAULT_UNFIXED_TOKENS = 5
# Each chunk is one engine request over the whole item, so this floor bounds a
# session at 2 requests per second of audio. A shorter chunk is under a word
# and adds little to the hypothesis.
MIN_CHUNK_SECONDS = 0.5
# Longest uncommitted item: the most audio one step decodes. One append is
# capped at this too, on both intents, as it is resampled before any other cap.
MAX_ITEM_SECONDS = 300.0
# Longest excerpt of a client value quoted in an error.
EXCERPT_CHARS = 100
# Longest language a session accepts: a code ("en", "zh-Hant") or a name.
MAX_LANGUAGE_CHARS = 32
# Longest message a failed step sends.
MAX_ERROR_CHARS = 300
class _ExcerptRepr(reprlib.Repr):
    """``reprlib.Repr`` that reads the first items of a dict or set instead
    of sorting all of them, so the work does not grow with the input."""

    def repr_dict(self, x, level):
        if not x:
            return "{}"
        if level <= 0:
            return "{" + self.fillvalue + "}"
        pieces = [f"{self.repr1(k, level - 1)}: {self.repr1(v, level - 1)}"
                  for k, v in itertools.islice(x.items(), self.maxdict)]
        if len(x) > self.maxdict:
            pieces.append(self.fillvalue)
        return "{" + ", ".join(pieces) + "}"

    def repr_set(self, x, level):
        return self._repr_iterable(x, level, "{", "}", self.maxset) if x else "set()"

    def repr_frozenset(self, x, level):
        return self._repr_iterable(x, level, "frozenset({", "})", self.maxfrozenset) if x else "frozenset()"


# Bounds the work of rendering a nested value; EXCERPT_CHARS bounds the result.
_EXCERPT_REPR = _ExcerptRepr(
    maxlevel=2, maxtuple=4, maxlist=4, maxarray=4, maxdict=4, maxset=4, maxfrozenset=4, maxdeque=4,
    maxstring=EXCERPT_CHARS, maxlong=EXCERPT_CHARS, maxother=EXCERPT_CHARS,
)


@dataclass
class RealtimeSession:
    """One WebSocket's transcription state (the SDK's ``ASRStreamingState``)."""
    session_id: str
    request: TranscriptionRequest
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    unfixed_chunks: int = DEFAULT_UNFIXED_CHUNKS
    unfixed_tokens: int = DEFAULT_UNFIXED_TOKENS
    max_item_seconds: float = MAX_ITEM_SECONDS
    pending: list[np.ndarray] = field(default_factory=list)   # audio not yet in a chunk
    pending_samples: int = 0
    heard: list[np.ndarray] = field(default_factory=list)     # every chunk decoded so far
    chunk_id: int = 0
    raw_hypothesis: str = ""   # the model's raw output for the audio so far
    emitted: str = ""          # clean text already sent as deltas
    item_id: str = field(default_factory=lambda: f"item_{uuid.uuid4().hex[:12]}")
    # error ids of this item's distinct step failures (at most
    # MAX_FAILURES_PER_ITEM), so a failure that repeats is logged once per item
    failures: tuple[str, dict] = ("", {})
    # the first byte of a sample whose second byte is in the next append;
    # kept across commit (the byte stream goes on), dropped by clear
    odd_byte: bytes = b""
    # the client's audio; any rate but SAMPLE_RATE goes through resampler
    input_format: AudioFormat = field(default_factory=lambda: AudioFormat("audio/pcm", SAMPLE_RATE))
    resampler: media_io.StreamResampler | None = None
    # samples the last chunk was short of its boundary (see take_chunk)
    short: int = 0
    # server turn detection (None: the client commits); the segmenter, like
    # the resampler, spans turns, so a commit does not reset either
    turn_detection: realtime_vad.ServerVAD | realtime_vad.SemanticVAD | None = None
    segmenter: realtime_vad.TurnSegmenter | None = None
    stream_samples: int = 0    # SAMPLE_RATE samples written this session: the event timeline
    # the input clock: stream_samples at the last rate change, and input samples since
    clock_base: int = 0
    clock_in: int = 0

    @property
    def chunk_samples(self) -> int:
        return int(round(self.chunk_seconds * SAMPLE_RATE))

    def set_input_format(self, fmt: AudioFormat) -> None:
        self.clock_base, self.clock_in = self.stream_samples, 0
        self.input_format = fmt
        self.resampler = None if fmt.rate == SAMPLE_RATE else media_io.StreamResampler(fmt.rate, SAMPLE_RATE)

    def resample(self, audio: np.ndarray) -> np.ndarray:
        self.clock_in += int(audio.size)
        out = audio if self.resampler is None else self.resampler.push(audio)
        self.stream_samples += int(out.size)
        return out

    def segment(self, audio: np.ndarray) -> list[tuple]:
        """Resample and segment ``audio``: the segmenter's operations."""
        return self.segmenter.feed(self.resample(audio))

    def reset_stream(self) -> None:
        """A clear: drop what the resampler and the segmenter hold, keeping
        the timeline (the dropped audio still counts)."""
        skipped = self.resampler.held() if self.resampler is not None else 0
        self.stream_samples += skipped
        if self.resampler is not None:
            self.resampler.reset()
        if self.segmenter is not None:
            self.segmenter.reset(skipped)
        self.realign()

    def realign(self) -> None:
        """After the resampler starts over (a commit or a clear), put the
        timeline back on the input clock: each start-over rounds its output
        up by less than a sample, which would add up over many of them."""
        exact = self.clock_base + self.clock_in * SAMPLE_RATE // self.input_format.rate
        if self.segmenter is not None:
            self.segmenter.shift(exact - self.stream_samples)
        self.stream_samples = exact

    def drain_resampler(self) -> np.ndarray:
        """What the resampler still holds, as if silence followed; it starts over."""
        out = np.zeros(0, dtype=np.float32) if self.resampler is None else self.resampler.flush()
        self.stream_samples += int(out.size)
        return out

    def add(self, audio: np.ndarray) -> None:
        """Add SAMPLE_RATE audio to the item."""
        if audio.size:
            self.pending.append(audio)
            self.pending_samples += int(audio.size)

    def append(self, audio: np.ndarray) -> None:
        self.add(self.resample(audio))

    def item_samples(self, extra: int = 0) -> int:
        """SAMPLE_RATE samples in the item once ``extra`` more input samples
        arrive, rounded once over the whole item (not per append), so the cap
        does not depend on how the audio was split."""
        held = self.resampler.held(extra) if self.resampler is not None else extra
        return self.pending_samples + held + sum(int(a.size) for a in self.heard)

    def take_chunk(self) -> np.ndarray | None:
        """Every full chunk of pending audio as one array, or None until
        enough arrived. Output the resampler still holds for its lookahead
        counts toward the chunk, so a chunk of 24 kHz input is decoded on the
        append that completes it, not the one after; the chunk is then short
        by those samples (``short``), which go with the next one."""
        held = self.resampler.held() if self.resampler is not None else 0
        k = (self.pending_samples + held - self.short) // self.chunk_samples
        if k <= 0:
            return None
        n = min(self.pending_samples, k * self.chunk_samples + self.short)
        self.short = k * self.chunk_samples + self.short - n
        buf = np.concatenate(self.pending)
        chunk, rest = buf[:n], buf[n:]
        self.pending = [rest] if rest.size else []
        self.pending_samples = int(rest.size)
        return chunk

    def flush(self) -> np.ndarray | None:
        if not self.pending_samples:
            return None
        buf = np.concatenate(self.pending)
        self.pending, self.pending_samples = [], 0
        return buf

    def audio_so_far(self) -> np.ndarray:
        return np.concatenate(self.heard) if self.heard else np.zeros(0, dtype=np.float32)

    def clear(self) -> None:
        """Start a new item; the stream state (resampler, segmenter, odd byte) goes on."""
        self.pending, self.pending_samples, self.heard, self.short = [], 0, [], 0
        self.chunk_id, self.raw_hypothesis, self.emitted = 0, "", ""
        self.item_id = f"item_{uuid.uuid4().hex[:12]}"


class AudioFormat(NamedTuple):
    """A socket's audio on the wire: ``codec`` ``audio/pcm`` (PCM16, little
    endian, mono), ``audio/pcmu`` or ``audio/pcma`` (G.711), at ``rate`` Hz."""
    codec: str
    rate: int

    def view(self) -> dict:
        return {"type": self.codec, "rate": self.rate} if self.codec == "audio/pcm" else {"type": self.codec}


def _audio_format(fmt, name: str) -> AudioFormat:
    """The GA format object ``fmt``: ``{"type": "audio/pcm", "rate": N}``
    (24 kHz when left out) or ``{"type": "audio/pcmu" | "audio/pcma"}``
    (8 kHz)."""
    fmt = _object(fmt, name)
    codec = fmt.get("type")
    if not isinstance(codec, str) or codec not in ("audio/pcm", "audio/pcmu", "audio/pcma"):
        raise ValueError(f'{name}.type must be "audio/pcm", "audio/pcmu" or "audio/pcma"; got {_excerpt(codec)}')
    if extra := sorted(set(fmt) - {"type", "rate"}):
        # channels, encoding and the like would change how the bytes read
        raise ValueError(f"{name} takes only type and rate; got {_excerpt(extra)}")
    rate = fmt.get("rate", OPENAI_PCM_RATE if codec == "audio/pcm" else G711_RATE)
    if codec != "audio/pcm" and rate != G711_RATE:
        raise ValueError(f"{name}: {codec} is {G711_RATE} Hz; got rate {_excerpt(rate)}")
    if isinstance(rate, bool) or not isinstance(rate, int) or rate not in REALTIME_RATES:
        raise ValueError(f"{name}.rate {_excerpt(rate)} Hz is not supported; use one of {list(REALTIME_RATES)}")
    return AudioFormat(codec, rate)


def decode_base64(payload: str) -> bytes:
    """An append's base64 payload, strictly: whitespace is allowed
    (base64.encodebytes wraps lines), anything else outside the alphabet
    raises, as does a payload that is not a string."""
    # translate stays linear in the payload where split() would build a list of its pieces
    return base64.b64decode(payload.encode("ascii").translate(None, b" \t\n\r\v\f"), validate=True)


def decode_append(payload: str, carry: bytes = b"", codec: str = "audio/pcm") -> tuple[np.ndarray, bytes]:
    """A base64 append in ``codec`` -> (float32 in [-1, 1), the odd trailing
    byte). ``carry`` is the previous append's odd byte (PCM16 only; a G.711
    sample is one byte), so a sample split across two appends is joined
    instead of shifting every sample after it. Strict decoding: a malformed
    payload is an error, not silence."""
    raw = carry + decode_base64(payload)
    if codec != "audio/pcm":
        return media_io.g711_decode(raw, codec).astype(np.float32) / 32768.0, b""
    whole = len(raw) - len(raw) % 2
    return np.frombuffer(raw[:whole], dtype="<i2").astype(np.float32) / 32768.0, raw[whole:]


def _append_error(audio: np.ndarray, rate: int) -> dict | None:
    """The ``error`` for an append of more than ``MAX_ITEM_SECONDS`` of
    ``rate`` audio, or None."""
    if audio.size <= MAX_ITEM_SECONDS * rate:
        return None
    return _error(f"an append holds at most {MAX_ITEM_SECONDS:g} s of audio; send it in shorter appends")


def encode_audio(audio: np.ndarray, codec: str) -> str:
    """float32 audio -> base64 in ``codec``."""
    pcm = media_io.to_pcm16(audio)
    return base64.b64encode(pcm.tobytes() if codec == "audio/pcm" else media_io.g711_encode(pcm, codec)).decode("ascii")


# The transcription intent's beta names for formats; "pcm16" is OpenAI's 24 kHz.
_BETA_FORMATS = {
    "pcm16": AudioFormat("audio/pcm", OPENAI_PCM_RATE),
    "g711_ulaw": AudioFormat("audio/pcmu", G711_RATE),
    "g711_alaw": AudioFormat("audio/pcma", G711_RATE),
}
_BETA_NAMES = {fmt: name for name, fmt in _BETA_FORMATS.items()}


def _excerpt(value) -> str:
    """The repr of a client-supplied value, at most ``EXCERPT_CHARS`` long."""
    text = _EXCERPT_REPR.repr(value)
    return text if len(text) <= EXCERPT_CHARS else text[: EXCERPT_CHARS - 3] + "..."


# distinct step failures remembered per item; past this each new one is
# logged, so alternating failures cannot grow the session's memory
MAX_FAILURES_PER_ITEM = 8


def _clip(message: str, limit: int = MAX_ERROR_CHARS) -> str:
    return message if len(message) <= limit else message[: limit - 3] + "..."


# An absolute path with at least one directory, not inside a word; a URL's
# "//" never matches, so a path right after a ":" is still one.
_ABS_PATH = re.compile(r"(?<![\w./<>])/(?:[^\s/'\"]+/)+([^\s/'\"]+)")
# A quoted one, taken whole: its directories may hold spaces.
_QUOTED_PATH = re.compile(r"(['\"])/(?:[^'\"\n/]*/)+([^'\"\n/]+)\1")


def _scrub_paths(message: str, upload_dir) -> str:
    """``message`` with the upload directory as ``<upload>`` and any other
    absolute path cut to its base name, so a kept error does not show the
    server's layout (#592)."""
    message = _QUOTED_PATH.sub(r"\1\2\1", message.replace(str(upload_dir), "<upload>"))
    return _ABS_PATH.sub(r"\1", message)


class StepError(RuntimeError):
    """The engine's error for a step, with the HTTP status it carried: a 4xx
    is the request's fault (a model's ValueError), anything else the server's."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def _object(value, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _input_format(fmt, current: AudioFormat) -> AudioFormat:
    """The transcription intent's ``input_audio_format`` (``current`` when absent)."""
    if fmt is None:
        return current
    if isinstance(fmt, str) and fmt in _BETA_FORMATS:
        return _BETA_FORMATS[fmt]
    if isinstance(fmt, dict):
        return _audio_format(fmt, "input_audio_format")
    raise ValueError(f"input_audio_format must be pcm16, g711_ulaw, g711_alaw or an object; got {_excerpt(fmt)}")


def _number(knobs: dict, name: str, default, lo: float, hi: float = math.inf, integer: bool = False,
            clamp_low: bool = False, scope: str = "mstar"):
    """``knobs[name]`` (or ``default``) checked to be a finite number in
    ``[lo, hi]``, and integral when ``integer``. A numeric string is read as
    its number, and with ``clamp_low`` a value below ``lo`` is raised to it.
    JSON allows NaN, Infinity and integers too big for a float, so each is
    checked before any conversion."""
    value = knobs.get(name, default)
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
    kind = "an integer" if integer else "a number"
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    try:
        ok = ok and math.isfinite(value) and (not integer or value == int(value))
    except OverflowError:
        ok = False
    if ok and clamp_low and value < lo:
        value = lo
    ok = ok and lo <= value <= hi
    if not ok:
        bound = f">= {lo:g}" if hi == math.inf else f"between {lo:g} and {hi:g}"
        raise ValueError(f"{scope}.{name} must be {kind} {bound}; got {_excerpt(value)}")
    return int(value) if integer else float(value)


_TURN_DETECTION_KEYS = {
    "server_vad": frozenset({"type", "threshold", "prefix_padding_ms", "silence_duration_ms",
                             "create_response", "interrupt_response", "idle_timeout_ms"}),
    "semantic_vad": frozenset({"type", "eagerness", "create_response", "interrupt_response"}),
}


def _turn_detection(value) -> realtime_vad.ServerVAD | realtime_vad.SemanticVAD | None:
    """A session's ``turn_detection``: null (the client commits),
    ``server_vad`` or ``semantic_vad``; fields left out take OpenAI's defaults."""
    if value is None:
        return None
    td = _object(value, "turn_detection")
    kind = td.get("type")
    if not isinstance(kind, str) or kind not in _TURN_DETECTION_KEYS:
        what = f"type {_excerpt(kind)}" if isinstance(kind, str) else _excerpt(td)
        raise ValueError(f"turn_detection {what} is not supported; use server_vad, semantic_vad, "
                         "or null to commit with input_audio_buffer.commit")
    unknown = next((k for k in td if k not in _TURN_DETECTION_KEYS[kind]), None)
    if unknown is not None:
        raise ValueError(f"turn_detection.{_excerpt(unknown)} is not supported")
    if td.get("idle_timeout_ms") is not None:
        raise ValueError("turn_detection.idle_timeout_ms is not supported")
    flags = {}
    for name in ("create_response", "interrupt_response"):
        flags[name] = td.get(name, True)
        if not isinstance(flags[name], bool):
            raise ValueError(f"turn_detection.{name} must be a boolean; got {_excerpt(flags[name])}")
    if kind == "semantic_vad":
        eagerness = td.get("eagerness", "auto")
        if not isinstance(eagerness, str) or eagerness not in realtime_vad.EAGERNESS:
            raise ValueError(f"turn_detection.eagerness must be one of low, medium, high, auto; "
                             f"got {_excerpt(eagerness)}")
        return realtime_vad.SemanticVAD(eagerness=eagerness, **flags)
    d = realtime_vad.ServerVAD()
    return realtime_vad.ServerVAD(
        threshold=_number(td, "threshold", d.threshold, 0.0, 1.0, scope="turn_detection"),
        prefix_padding_ms=_number(td, "prefix_padding_ms", d.prefix_padding_ms, 0, MAX_VAD_MS, integer=True,
                                  scope="turn_detection"),
        silence_duration_ms=_number(td, "silence_duration_ms", d.silence_duration_ms, 0, MAX_VAD_MS,
                                    integer=True, scope="turn_detection"),
        **flags,
    )


def stable_prefix(raw_text: str, tokenizer, unfixed_tokens: int) -> str:
    """``raw_text`` minus its last ``unfixed_tokens`` tokens, backed off
    further when the cut lands inside a multi-byte character."""
    if not raw_text or tokenizer is None:
        return ""
    ids = tokenizer.encode(raw_text, add_special_tokens=False)
    k = max(int(unfixed_tokens), 0)
    while True:
        end = max(0, len(ids) - k)
        prefix = tokenizer.decode(ids[:end]) if end > 0 else ""
        if "�" not in prefix or end == 0:
            return prefix
        k += 1


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


async def _run_step(api, adapter, session: RealtimeSession, prefix: str) -> str:
    """Transcribe the audio heard so far, continuing from ``prefix``; return
    the model's raw output (prefix included)."""
    # a fresh id per step: a failed step does not advance chunk_id, and its
    # id is still draining when the next step goes out
    request_id = rid(f"rt-{session.session_id}")
    path = media_io.write_wav(
        session.audio_so_far(), str(Path(api.upload_dir) / f"{request_id}.wav"), SAMPLE_RATE,
    )
    try:
        args = adapter.realtime_step_request(session.request, path, prefix)
        api.submit_request(
            text=args.text,
            file_paths=args.file_paths,
            input_modalities=args.input_modalities,
            output_modalities=args.output_modalities,
            model_kwargs=args.model_kwargs,
            prompt_parts=args.prompt_parts,
            streaming=False,
            request_id=request_id,
        )
        try:
            chunks = await api.collect_results(request_id)
        except asyncio.CancelledError:
            api.release_request(request_id)
            raise
        except HTTPException as exc:
            # a request the engine failed: the status says whose fault it was
            raise StepError(str(exc.detail), exc.status_code) from exc
    finally:
        Path(path).unlink(missing_ok=True)
    text = decode_joined(c.data for c in chunks if c.modality == "text")
    errors = [c for c in chunks if c.modality == "error"]
    if errors:
        raise StepError(
            errors[0].data.decode("utf-8", "replace"), int((errors[0].metadata or {}).get("status") or 500),
        )
    return prefix + text


def server_event(kind: str, **fields) -> dict:
    """A server event: ``kind`` with a fresh ``event_id``."""
    return {"event_id": f"event_{uuid.uuid4().hex[:12]}", "type": kind, **fields}


def _error(message: str, kind: str = "invalid_request_error", code: str | None = None) -> dict:
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


async def _events(ws: WebSocket, send):
    """A socket's client events as dicts, until it disconnects. A frame that
    is not a JSON object is answered with an ``error`` and skipped."""
    while True:
        # receive_text() raises KeyError on a binary frame
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return
        if message.get("text") is None:
            await send(_error("an event must be a JSON text message, not binary"))
            continue
        try:
            event = json.loads(message["text"])
        # ValueError covers JSONDecodeError and integers past Python's
        # int-string digit limit; RecursionError is deep nesting
        except (ValueError, RecursionError):
            await send(_error("not JSON"))
            continue
        if not isinstance(event, dict):
            await send(_error("an event must be a JSON object"))
            continue
        yield event


class RealtimeTranscription:
    """Drives one WebSocket session against the API server."""

    def __init__(self, api, model_name: str, adapter, websocket: WebSocket):
        self.api = api
        self.model_name = model_name
        self.adapter = adapter
        self.ws = websocket
        self.session = RealtimeSession(
            session_id=uuid.uuid4().hex[:16],
            request=TranscriptionRequest(model=model_name),
            max_item_seconds=min(MAX_ITEM_SECONDS, getattr(adapter, "max_audio_seconds", None) or math.inf),
        )

    async def send(self, event: dict) -> None:
        await self.ws.send_text(json.dumps(event))

    def _session_view(self) -> dict:
        s = self.session
        return {
            "id": s.session_id,
            "object": "realtime.transcription_session",
            "input_audio_format": _BETA_NAMES.get(s.input_format) or s.input_format.view(),
            "turn_detection": s.turn_detection.view() if s.turn_detection is not None else None,
            "input_audio_transcription": {
                "model": self.model_name,
                "language": s.request.language,
                "prompt": s.request.prompt or "",
            },
            "mstar": {
                "chunk_seconds": s.chunk_seconds,
                "unfixed_chunks": s.unfixed_chunks,
                "unfixed_tokens": s.unfixed_tokens,
                "max_item_seconds": s.max_item_seconds,
                "sample_rate": s.input_format.rate,
            },
        }

    async def run(self) -> None:
        await self.ws.accept()
        await self.send(server_event("transcription_session.created", session=self._session_view()))
        try:
            async for event in _events(self.ws, self.send):
                await self.handle(event)
        except WebSocketDisconnect:
            return

    async def handle(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "transcription_session.update":
            try:
                self.update(_object(event.get("session"), "session"))
            except (TypeError, ValueError) as exc:
                await self.send(_error(str(exc)))
                return
            await self.send(server_event("transcription_session.updated", session=self._session_view()))
        elif kind == "input_audio_buffer.append":
            s = self.session
            try:
                # not `or ""`: a null is an error, not an empty append, on every realtime path
                audio, odd_byte = decode_append(event.get("audio", ""), s.odd_byte, s.input_format.codec)
            except Exception:  # noqa: BLE001 — bad base64
                await self.send(_error("audio is not base64", code="invalid_audio"))
                return
            if (error := _append_error(audio, s.input_format.rate)) is not None:
                await self.send(error)
                return
            # with server_vad the segmenter ends a turn at the cap instead
            if s.segmenter is None and s.item_samples(audio.size) > s.max_item_seconds * SAMPLE_RATE:
                await self.send(_error(
                    f"the item would pass {s.max_item_seconds:g} s of audio; "
                    "commit or clear the input audio buffer first",
                ))
                return
            s.odd_byte = odd_byte
            if not audio.size:
                return
            if s.segmenter is None:
                # off the event loop: resampling a long append takes long enough
                # to stall every other session on this server
                if s.resampler is None or audio.size <= INLINE_RESAMPLE_SAMPLES:
                    audio = s.resample(audio)
                else:
                    audio = await asyncio.to_thread(s.resample, audio)
                await self.apply([("audio", audio)])
            else:
                # off the event loop: one large append is thousands of VAD frames,
                # and an end-of-turn model call takes about 45 ms
                await self.apply(await asyncio.to_thread(s.segment, audio))
        elif kind == "input_audio_buffer.commit":
            s = self.session
            tail = s.drain_resampler()
            if s.segmenter is None:
                s.realign()
                await self.apply([("audio", tail)])
            else:
                ops = await asyncio.to_thread(s.segmenter.feed, tail) + s.segmenter.end_turn()
                s.realign()
                await self.apply(ops)
                if ops and ops[-1] == ("commit",):
                    # the drained tail ended a turn: that commit was this one
                    return
            await self.commit()
        elif kind == "input_audio_buffer.clear":
            s = self.session
            s.clear()
            s.odd_byte = b""
            s.reset_stream()
            await self.send(server_event("input_audio_buffer.cleared"))
        else:
            await self.send(_error(f"unknown event {_excerpt(kind)}"))

    async def apply(self, ops: list[tuple]) -> None:
        """Carry out segmenter operations in order (see ``TurnSegmenter``).
        Audio steps once, after the last operation, so one long append is one
        step, as without turn detection; a turn's commit decodes the rest."""
        s = self.session
        if any(op[0] == "end_of_turn_failed" for op in ops):
            for op in ops:
                if op[0] == "end_of_turn_failed":
                    await self._end_of_turn_failed(op[1])
            ops = [op for op in ops if op[0] != "end_of_turn_failed"]
        for i, op in enumerate(ops):
            if op[0] == "audio":
                s.add(op[1])
                if i == len(ops) - 1 and (chunk := s.take_chunk()) is not None:
                    await self.step(chunk)
            elif op[0] == "commit":
                await self.commit()
            elif op[0] == "speech_started":
                await self.send(server_event("input_audio_buffer.speech_started", audio_start_ms=op[1],
                                             item_id=s.item_id))
            else:
                await self.send(server_event("input_audio_buffer.speech_stopped", audio_end_ms=op[1],
                                             item_id=s.item_id))

    async def _end_of_turn_failed(self, exc: Exception) -> None:
        """semantic_vad's end-of-turn model raised: the segmenter no longer
        asks it, so tell the client how this session's turns end now. The
        error is logged in full under an id; the event carries only the id,
        as for a failed step."""
        s = self.session
        error_id = uuid.uuid4().hex[:12]
        logger.error("realtime end-of-turn model failed (error id %s, session %s); semantic_vad turns now end "
                     "on silence alone", error_id, s.session_id, exc_info=exc)
        silence_s = realtime_vad.EAGERNESS[s.turn_detection.eagerness][2] / 1000
        await self.send(_error(
            f"semantic_vad's end-of-turn model failed (error id {error_id}); this session's turns now end after "
            f"{silence_s:g} s of silence; send transcription_session.update to switch to server_vad",
            kind="server_error", code="end_of_turn_failed",
        ))

    async def commit(self) -> None:
        """Finish the item: decode what is left, report it, start a new one."""
        s = self.session
        tail = s.flush()
        if tail is not None and tail.size:
            await self.step(tail, final=True)
        else:
            # nothing left to hear: what was held back is final now
            await self.emit(self.current_text(), hold=0)
        await self.send(server_event("input_audio_buffer.committed", item_id=s.item_id))
        await self.send(server_event(
            "conversation.item.input_audio_transcription.completed",
            item_id=s.item_id, transcript=self.current_text(),
        ))
        s.clear()

    def update(self, session: dict) -> None:
        """Apply a session update, all or nothing."""
        s = self.session
        fmt = _input_format(session.get("input_audio_format"), s.input_format)
        if fmt != s.input_format and s.item_samples():
            raise ValueError("commit or clear the input audio buffer before changing the input audio format")
        if fmt != s.input_format and s.odd_byte:
            # a commit keeps the half sample, so only clear helps
            raise ValueError("half a sample is pending; clear the input audio buffer before changing "
                             "the input audio format")
        td = _turn_detection(session["turn_detection"]) if "turn_detection" in session else s.turn_detection
        vad = end_of_turn = None
        if td != s.turn_detection:
            # with server_vad on, the resampler's lookahead is in no item between turns
            in_item = (s.pending_samples or s.heard or s.segmenter.speaking) if s.segmenter is not None \
                else s.item_samples()
            if in_item:
                raise ValueError("commit or clear the input audio buffer before changing turn_detection")
            if td is not None and (vad := realtime_vad.silero_factory()) is None:
                raise ValueError("server_vad is unavailable here (Silero VAD is missing or not the expected "
                                 "model; see the server log); "
                                 "set turn_detection to null and commit with input_audio_buffer.commit")
            if isinstance(td, realtime_vad.SemanticVAD) and (end_of_turn := realtime_vad.smart_turn()) is None:
                raise ValueError(f"semantic_vad is unavailable here ({realtime_vad.smart_turn_problem()}; see the "
                                 "server log); use server_vad, or set turn_detection to null and commit with "
                                 "input_audio_buffer.commit")
        knobs = _object(session.get("mstar"), "mstar")
        chunk_seconds = _number(knobs, "chunk_seconds", s.chunk_seconds, MIN_CHUNK_SECONDS, s.max_item_seconds,
                                clamp_low=True)
        unfixed_chunks = _number(knobs, "unfixed_chunks", s.unfixed_chunks, 0, integer=True)
        unfixed_tokens = _number(knobs, "unfixed_tokens", s.unfixed_tokens, 0, integer=True)
        transcription = _object(session.get("input_audio_transcription"), "input_audio_transcription")
        fields = {k: v for k, v in transcription.items() if k in ("language", "prompt")}
        for name, value in fields.items():
            if value is not None and not isinstance(value, str):
                raise ValueError(f"input_audio_transcription.{name} must be a string; got {_excerpt(value)}")
        language = fields.get("language")
        if language is not None and len(language) > MAX_LANGUAGE_CHARS:
            raise ValueError(
                f"input_audio_transcription.language must be at most {MAX_LANGUAGE_CHARS} characters; "
                f"got {_excerpt(language)}"
            )
        request = TranscriptionRequest.model_validate({**s.request.model_dump(), **fields}) if fields else s.request
        s.request = request
        s.chunk_seconds, s.unfixed_chunks, s.unfixed_tokens = chunk_seconds, unfixed_chunks, unfixed_tokens
        if fmt != s.input_format:
            s.set_input_format(fmt)
        if td != s.turn_detection:
            old, s.turn_detection = s.segmenter, td
            s.segmenter = None if td is None else realtime_vad.TurnSegmenter(
                vad(), td, int(s.max_item_seconds * SAMPLE_RATE), pos=s.stream_samples, end_of_turn=end_of_turn)
            if old is not None:
                # audio the old segmenter had not scored yet is not dropped
                if s.segmenter is not None:
                    s.segmenter.take_over(old)
                else:
                    s.add(old.unscored())

    def current_text(self) -> str:
        return self.adapter.parse_transcript(self.session.raw_hypothesis, self.session.request).text

    def _tokenizer(self):
        return getattr(getattr(self.api, "model", None), "tokenizer", None)

    async def step(self, chunk: np.ndarray, final: bool = False) -> None:
        """Hear ``chunk``, re-decode everything so far from the stable prefix,
        then send the newly firm text and the whole hypothesis."""
        s = self.session
        s.heard.append(chunk)
        tokenizer = self._tokenizer()
        prefix = "" if s.chunk_id < s.unfixed_chunks else stable_prefix(s.raw_hypothesis, tokenizer, s.unfixed_tokens)
        try:
            s.raw_hypothesis = await _run_step(self.api, self.adapter, s, prefix)
        except Exception as exc:  # noqa: BLE001 — report it as an event, keep the session
            await self._step_failed(exc)
            return
        # a coalesced step counts every chunk it heard, so the prefix that keeps
        # sent text fixed starts after the same audio as with one step per chunk
        # (rounded: at 24 kHz a chunk can be short by the resampler's lookahead)
        s.chunk_id += max(1, round(chunk.size / s.chunk_samples))
        text = self.current_text()
        if final or tokenizer is None:
            hold = 0
        else:
            # hold back the unstable tail: what the next step may still rewrite
            stable_text = self.adapter.parse_transcript(
                stable_prefix(s.raw_hypothesis, tokenizer, s.unfixed_tokens), s.request,
            ).text
            hold = max(0, len(text) - len(stable_text))
        await self.emit(text, hold)
        await self.send(server_event(
            "mstar.transcription.partial", item_id=s.item_id, text=text, chunk_id=s.chunk_id,
            audio_seconds=sum(a.size for a in s.heard) / SAMPLE_RATE, created=now(),
        ))

    async def _step_failed(self, exc: Exception) -> None:
        """Report a failed step. A request error (a ValueError or TypeError
        here, a 4xx from the engine) keeps its message, cut to
        ``MAX_ERROR_CHARS`` and without server paths; anything else is
        a generic message with an error id, and the full error is logged under
        that id. Each distinct failure is logged once per item, not once per
        step."""
        s = self.session
        detail = str(exc)
        client_error = isinstance(exc, (ValueError, TypeError)) or (
            isinstance(exc, StepError) and 400 <= exc.status < 500
        )
        if s.failures[0] != s.item_id:
            s.failures = (s.item_id, {})
        seen = s.failures[1]
        failure = (type(exc).__name__, detail)
        error_id = seen.get(failure)
        if error_id is None:
            error_id = uuid.uuid4().hex[:12]
            if len(seen) < MAX_FAILURES_PER_ITEM:
                seen[failure] = error_id
            logger.log(
                logging.INFO if client_error else logging.ERROR,
                "realtime step failed (error id %s, session %s, %s); repeats in this item are not logged",
                error_id, s.session_id, s.item_id, exc_info=exc,
            )
        if client_error:
            message = _clip(_scrub_paths(detail, self.api.upload_dir))
            await self.send(_error(message))
        else:
            await self.send(_error(f"realtime step failed (error id {error_id})", kind="server_error"))

    async def emit(self, text: str, hold: int) -> None:
        """Send the part of ``text`` past what was already sent, minus the
        last ``hold`` characters. A revision that reaches into sent text is
        not resent — deltas are append-only — the partial event carries it."""
        s = self.session
        stable = _common_prefix_len(s.emitted, text)
        if stable < len(s.emitted):
            s.emitted = text[:stable]
        delta = text[len(s.emitted):]
        firm = delta[: max(0, len(delta) - hold)]
        if firm:
            s.emitted += firm
            await self.send(server_event(
                "conversation.item.input_audio_transcription.delta", item_id=s.item_id, delta=firm,
            ))


# Client events a conversation refuses with an ``unsupported`` error, the call
# going on: the model decides when it speaks and has no turn to cancel or edit.
_UNSUPPORTED_EVENTS = frozenset({
    "response.create", "response.cancel", "output_audio_buffer.clear",
    "conversation.item.truncate", "conversation.item.delete", "conversation.item.retrieve",
})


@dataclass(slots=True)
class _AgentTurn:
    """One agent reply: the text steps from its start mark to its end mark,
    and the audio of the same steps."""
    start: int                  # its first step
    end: int | None = None      # its last step, once the end mark arrived
    response_id: str = field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:16]}")
    item_id: str = field(default_factory=lambda: f"item_{uuid.uuid4().hex[:12]}")
    text: Utf8Stream = field(default_factory=Utf8Stream)
    transcript: list[str] = field(default_factory=list)

    def item(self, status: str) -> dict:
        content = [] if status == "in_progress" else [{"type": "output_audio", "transcript": "".join(self.transcript)}]
        return {"id": self.item_id, "object": "realtime.item", "type": "message", "role": "assistant",
                "status": status, "content": content}


class RealtimeConversation:
    """Drives one ``/v1/realtime`` conversation (OpenAI's GA events) with a
    full-duplex model: one whose client stream declares ``ClientAudio`` and
    that decides when to speak (see docs/clients.rst for the protocol).

    The call is one engine request, opened by the first append with the
    session's instructions as its prompt; formats and instructions are fixed
    from then on. Input is decoded, resampled to the stream's rate and
    appended in whole frames, a partial frame waiting for the next append.
    Each reply between the model's ``turn_tokens`` marks is one response:
    text and audio chunks are paired by the loop step the data worker
    stamps on them (``metadata["step"]``), audio that arrives before its
    step's text waits for it, and a response closes once its last step's
    audio is sent. One response is in progress at a time, as OpenAI's
    clients expect: a reply that starts while the one before is still
    speaking opens when that one is done. Audio between replies is silence
    and is not sent.
    """

    def __init__(self, api, model_name: str, websocket: WebSocket, stream: Connection):
        self.api, self.model_name, self.ws = api, model_name, websocket
        # the stream's rate, and its step: the frame each append is cut into
        self.stream, self.rate, self.frame = stream.edge_name, stream.audio.sample_rate, stream.step_rows
        model = api.model
        self.output_rate = model.get_output_sample_rate("audio")
        self.voice = model.get_default_voice()
        self.session_id = f"sess_{uuid.uuid4().hex[:16]}"
        self.instructions = ""
        self.turn_detection = "server_vad"
        self.odd_byte = b""
        self.pending = np.zeros(0, dtype=np.float32)   # model-rate audio short of a frame
        self.request_id: str | None = None
        self.key: str | None = None
        self.relay_task: asyncio.Task | None = None
        self.turns: deque[_AgentTurn] = deque()   # replies whose last audio is not sent yet, oldest first
        self.text_steps = 0                       # steps whose text has arrived
        self.early: deque[tuple[int, bytes]] = deque()   # audio steps waiting for their text
        self.closed = False
        self._send_lock = asyncio.Lock()
        self._set_formats(AudioFormat("audio/pcm", OPENAI_PCM_RATE), AudioFormat("audio/pcm", OPENAI_PCM_RATE))

    async def send(self, event: dict) -> None:
        text = json.dumps(event)
        async with self._send_lock:
            if self.closed:   # the call ended: events still arriving get no answer
                return
            try:
                await self.ws.send_text(text)
            except Exception:  # noqa: BLE001 — the client is gone, and the call ends with it (run() aborts it)
                self.closed = True

    async def close(self, code: int) -> None:
        async with self._send_lock:
            if not self.closed:
                self.closed = True
                with contextlib.suppress(Exception):   # the client may have gone first
                    await self.ws.close(code=code)

    def _session_view(self) -> dict:
        return {
            "type": "realtime", "object": "realtime.session", "id": self.session_id, "model": self.model_name,
            "instructions": self.instructions, "output_modalities": ["audio"],
            "audio": {
                "input": {"format": self.input_format.view(), "turn_detection": {"type": self.turn_detection},
                          "transcription": None},
                "output": {"format": self.output_format.view(), "voice": self.voice},
            },
        }

    async def run(self) -> None:
        await self.ws.accept()
        await self.send(server_event("session.created", session=self._session_view()))
        try:
            async for event in _events(self.ws, self.send):
                await self.handle(event)
        except WebSocketDisconnect:
            pass
        finally:
            if self.relay_task is not None:
                self.relay_task.cancel()
                await asyncio.gather(self.relay_task, return_exceptions=True)
            if self.key is not None:
                self.api.abort_submission(self.key)

    async def handle(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "session.update":
            try:
                self.update(_object(event.get("session"), "session"))
            except (TypeError, ValueError) as exc:
                await self.send(_error(str(exc)))
                return
            await self.send(server_event("session.updated", session=self._session_view()))
        elif kind == "input_audio_buffer.append":
            await self.append(event.get("audio", ""))
        elif kind == "input_audio_buffer.commit":
            # the model hears the audio as it comes: nothing to flush
            await self.send(server_event("input_audio_buffer.committed", previous_item_id=None,
                                   item_id=f"item_{uuid.uuid4().hex[:12]}"))
        elif kind == "input_audio_buffer.clear":
            self.odd_byte, self.pending = b"", np.zeros(0, dtype=np.float32)
            if self.in_resampler is not None:
                self.in_resampler.reset()
            await self.send(server_event("input_audio_buffer.cleared"))
        elif kind == "conversation.item.create":
            await self.create_item(event.get("item"))
        elif isinstance(kind, str) and kind in _UNSUPPORTED_EVENTS:
            await self.send(_error(f"{kind} is not supported: this model decides when it speaks; keep streaming "
                                   "input_audio_buffer.append", code="unsupported"))
        else:
            await self.send(_error(f"unknown event {_excerpt(kind)}"))

    def update(self, session: dict) -> None:
        """Apply a session update, all or nothing."""
        if session.get("type", "realtime") != "realtime":
            raise ValueError(f'session.type must be "realtime"; got {_excerpt(session.get("type"))} '
                             "(transcription sessions are /v1/realtime?intent=transcription)")
        instructions = session.get("instructions", self.instructions)
        if instructions is None:
            instructions = ""
        if not isinstance(instructions, str):
            raise ValueError(f"instructions must be a string; got {_excerpt(instructions)}")
        modalities = session.get("output_modalities")
        if modalities is not None and modalities != ["audio"]:
            raise ValueError(f'output_modalities must be ["audio"] (this model always speaks); '
                             f'got {_excerpt(modalities)}')
        if session.get("tools"):
            raise ValueError("tools are not supported")
        if session.get("max_output_tokens") not in (None, "inf"):
            raise ValueError("max_output_tokens is not supported: the model decides how long it speaks")
        audio = _object(session.get("audio"), "audio")
        audio_in = _object(audio.get("input"), "audio.input")
        audio_out = _object(audio.get("output"), "audio.output")
        in_fmt = (_audio_format(audio_in["format"], "audio.input.format")
                  if audio_in.get("format") is not None else self.input_format)
        out_fmt = (_audio_format(audio_out["format"], "audio.output.format")
                   if audio_out.get("format") is not None else self.output_format)
        for name in ("transcription", "noise_reduction"):
            if audio_in.get(name) is not None:
                raise ValueError(f"audio.input.{name} is not supported")
        turn_detection = self.turn_detection
        if "turn_detection" in audio_in:
            td = audio_in["turn_detection"]
            if td is None:
                raise ValueError("audio.input.turn_detection null is not supported: this model decides when it "
                                 "speaks; use server_vad or semantic_vad")
            td = _object(td, "audio.input.turn_detection")
            if td.get("type") not in ("server_vad", "semantic_vad"):
                raise ValueError(f"audio.input.turn_detection.type must be server_vad or semantic_vad; "
                                 f"got {_excerpt(td.get('type'))}")
            for name in ("create_response", "interrupt_response"):
                if td.get(name, True) is not True:
                    raise ValueError(f"audio.input.turn_detection.{name} must be true: this model always "
                                     "answers and always yields")
            turn_detection = td["type"]
        if audio_out.get("voice", self.voice) != self.voice:
            raise ValueError(f"audio.output.voice must be {self.voice!r}, the voice this server speaks in; "
                             f"got {_excerpt(audio_out['voice'])}")
        if audio_out.get("speed") not in (None, 1):
            raise ValueError("audio.output.speed is not supported")
        if self.key is not None and (instructions, in_fmt, out_fmt) != (
                self.instructions, self.input_format, self.output_format):
            raise ValueError("instructions and audio formats are fixed once audio flows")
        self.instructions, self.turn_detection = instructions, turn_detection
        if (in_fmt, out_fmt) != (self.input_format, self.output_format):
            self._set_formats(in_fmt, out_fmt)

    def _set_formats(self, in_fmt: AudioFormat, out_fmt: AudioFormat) -> None:
        """Before audio flows: the resamplers hold nothing yet."""
        self.input_format, self.output_format = in_fmt, out_fmt
        self.in_resampler = (None if in_fmt.rate == self.rate
                             else media_io.StreamResampler(in_fmt.rate, self.rate))
        self.out_resampler = (None if out_fmt.rate == self.output_rate
                              else media_io.StreamResampler(self.output_rate, out_fmt.rate))

    async def create_item(self, item) -> None:
        """``conversation.item.create`` of a user message of ``input_audio``
        parts: their audio goes on the stream as appends do."""
        item = item if isinstance(item, dict) else {}
        content = item.get("content") if item.get("type") == "message" and item.get("role") == "user" else None
        if not content or not isinstance(content, list) or any(
                not isinstance(part, dict) or part.get("type") != "input_audio" for part in content):
            await self.send(_error("conversation.item.create takes only a user message of input_audio parts",
                                   code="unsupported"))
            return
        if not await self.append(*(part.get("audio", "") for part in content)):
            return   # the error is sent; no part was taken
        # the audio is not echoed back
        item = {**item, "content": [{k: v for k, v in part.items() if k != "audio"} for part in content],
                "id": item.get("id") or f"item_{uuid.uuid4().hex[:12]}", "object": "realtime.item",
                "status": "completed"}
        await self.send(server_event("conversation.item.added", previous_item_id=None, item=item))
        await self.send(server_event("conversation.item.done", previous_item_id=None, item=item))

    async def append(self, *payloads) -> bool:
        """Put the audio of an append, or of an item's parts, on the call,
        all or none; False, with an ``error`` sent (or the call over), if it
        was not taken."""
        odd_byte, pieces = self.odd_byte, []
        for payload in payloads:
            try:
                # not `or ""`: a null is an error, not an empty append, on every realtime path
                audio, odd_byte = decode_append(payload, odd_byte, self.input_format.codec)
            except Exception:  # noqa: BLE001 — bad base64
                await self.send(_error("audio is not base64", code="invalid_audio"))
                return False
            pieces.append(audio)
        audio = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        if (error := _append_error(audio, self.input_format.rate)) is not None:
            await self.send(error)
            return False
        self.odd_byte = odd_byte
        if self.key is None and not await self.open():
            return False
        if not audio.size:
            return True
        # off the event loop: resampling and writing the frames is about a
        # millisecond per frame, which every other socket would wait on
        paths = await asyncio.to_thread(self._frames, audio)
        if not paths:
            return True
        try:
            appended = self.api.stream_input(self.key, {"audio": paths},
                                             file_streams={"audio": [self.stream] * len(paths)})
        except ValueError as exc:  # past the call's input budget: the call ends with it
            appended = False
            self.api.reject_input(self.key, f"append rejected: {exc}", 400)
        if not appended:   # the call has ended; the relay reports why
            for path in paths:
                Path(path).unlink(missing_ok=True)
        return appended

    def _frames(self, audio: np.ndarray) -> list[str]:
        """Resample ``audio`` to the stream's rate and write each whole frame
        as a WAV file; the rest waits for the next append."""
        if self.in_resampler is not None:
            audio = self.in_resampler.push(audio)
        audio = np.concatenate([self.pending, audio]) if self.pending.size else audio
        n = self.frame
        whole = audio.size - audio.size % n
        self.pending = audio[whole:]
        upload = Path(self.api.upload_dir)
        return [media_io.write_wav(audio[i:i + n], str(upload / f"{self.request_id}-{uuid.uuid4().hex[:8]}.wav"),
                                   self.rate) for i in range(0, whole, n)]

    async def open(self) -> bool:
        """Open the call's engine request; False, with an ``error`` sent, if
        the server refused it (at capacity, say)."""
        text = self.instructions or None
        try:
            self.request_id = self.api.submit_request(
                text=text, input_modalities=["text"] if text else [], output_modalities=["text", "audio"],
                streaming=True, request_id=rid(f"rt-{self.session_id}"), end_input=False,
            )
        except HTTPException as exc:
            await self.send(_error(str(exc.detail), kind=error_type(exc.status_code)))
            return False
        except ValueError as exc:
            await self.send(_error(_clip(_scrub_paths(str(exc), self.api.upload_dir))))
            return False
        self.key = self.api.submission_key(self.request_id)
        # bound now, before anything else runs (see APIServer._bind)
        self.relay_task = asyncio.create_task(self.relay(self.api.iter_result_chunks(self.request_id)))
        return True

    async def relay(self, chunks) -> None:
        """Turn the call's output into response events until it ends, then
        finish any reply still open and close the socket: a call that ended
        takes no more audio."""
        status, code = "incomplete", 1000
        try:
            async for chunk in chunks:
                if chunk.modality == "text":
                    await self._on_text(chunk)
                elif chunk.modality == "audio":
                    await self._on_audio(chunk)
                elif chunk.modality == "error":
                    http_status = int((chunk.metadata or {}).get("status") or 500)
                    message = bytes(chunk.data).decode("utf-8", "replace")
                    await self.send(_error(_clip(_scrub_paths(message, self.api.upload_dir)),
                                           kind=error_type(http_status)))
                    status, code = "failed", 1011 if http_status >= 500 else 1000
        except HTTPException as exc:
            await self.send(_error(str(exc.detail), kind=error_type(exc.status_code)))
            status, code = "failed", 1011
        except Exception as exc:  # noqa: BLE001 — the socket must still close
            error_id = uuid.uuid4().hex[:12]
            logger.error("realtime conversation failed (error id %s, session %s)", error_id, self.session_id,
                         exc_info=exc)
            await self.send(_error(f"the call failed (error id {error_id})", kind="server_error"))
            status, code = "failed", 1011
        # no more text will come: held audio goes to its reply, and a reply still open ends here
        while self.early:
            await self._play(*self.early.popleft())
        while self.turns:
            await self._finish_turn(status)
        await self.close(code)

    async def _on_text(self, chunk) -> None:
        meta = chunk.metadata or {}
        step = meta["step"]
        self.text_steps = step + 1
        turn = self.turns[-1] if self.turns and self.turns[-1].end is None else None
        data = bytes(chunk.data)
        for mark in meta.get("turn", ()):
            if mark == "start" and turn is None:
                turn = await self._start_turn(step)
            elif mark == "end" and turn is not None:
                await self._feed(turn, data, final=True)
                data, turn.end, turn = b"", step, None
        if turn is not None:   # between replies the text channel is padding
            await self._feed(turn, data)
        while self.early and self.early[0][0] < self.text_steps:
            await self._play(*self.early.popleft())

    async def _start_turn(self, step: int) -> _AgentTurn:
        turn = _AgentTurn(start=step)
        self.turns.append(turn)
        if len(self.turns) == 1:
            await self._open_turn(turn)
        return turn

    async def _open_turn(self, turn: _AgentTurn) -> None:
        """Send the opening events of the oldest reply, and its transcript so far."""
        await self.send(server_event("response.created", response=response_object(
            turn.response_id, "in_progress", ["audio"])))
        await self.send(server_event("conversation.item.added", previous_item_id=None, item=turn.item("in_progress")))
        await self.send(server_event("response.output_item.added", response_id=turn.response_id, output_index=0,
                               item=turn.item("in_progress")))
        await self.send(content_event("response.content_part.added", turn.response_id, turn.item_id,
                                      part={"type": "audio", "transcript": ""}))
        if turn.transcript:
            await self.send(content_event("response.output_audio_transcript.delta", turn.response_id,
                                          turn.item_id, delta="".join(turn.transcript)))

    async def _feed(self, turn: _AgentTurn, data: bytes, final: bool = False) -> None:
        delta = turn.text.feed(data) + (turn.text.flush() if final else "")
        if delta:
            turn.transcript.append(delta)
            if turn is self.turns[0]:   # a later reply's text goes out when it opens
                await self.send(content_event("response.output_audio_transcript.delta", turn.response_id,
                                              turn.item_id, delta=delta))

    async def _on_audio(self, chunk) -> None:
        step = chunk.metadata["step"]
        if step >= self.text_steps:
            self.early.append((step, bytes(chunk.data)))   # its text decides which reply it is in
        else:
            await self._play(step, bytes(chunk.data))

    async def _play(self, step: int, pcm: bytes) -> None:
        """Send step ``step``'s audio in the reply it belongs to, if any."""
        while self.turns and self.turns[0].end is not None and self.turns[0].end < step:
            await self._finish_turn("completed")   # a reply whose last step had no audio
        turn = self.turns[0] if self.turns and self.turns[0].start <= step else None
        if turn is None:
            return   # silence between replies
        last = turn.end == step
        audio = await asyncio.to_thread(self._encode_out, pcm, last)
        if audio:
            await self.send(content_event("response.output_audio.delta", turn.response_id, turn.item_id, delta=audio))
        if last:
            await self._finish_turn("completed")

    def _encode_out(self, pcm: bytes, last: bool) -> str:
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if self.out_resampler is not None:
            audio = self.out_resampler.push(audio)
            if last:   # the reply ends: what the resampler holds goes out with it
                audio = np.concatenate([audio, self.out_resampler.flush()])
        return encode_audio(audio, self.output_format.codec) if audio.size else ""

    async def _finish_turn(self, status: str) -> None:
        """Close the oldest reply as ``status``: ``completed``, or
        ``incomplete`` / ``failed`` when the call ended inside it; then
        open the next."""
        turn = self.turns[0]
        if turn.end is None:
            await self._feed(turn, b"", final=True)
        self.turns.popleft()
        if self.out_resampler is not None:
            self.out_resampler.reset()   # already flushed, unless the reply ended early
        transcript = "".join(turn.transcript)
        await self.send(content_event("response.output_audio.done", turn.response_id, turn.item_id))
        await self.send(content_event("response.output_audio_transcript.done", turn.response_id, turn.item_id,
                                      transcript=transcript))
        await self.send(content_event("response.content_part.done", turn.response_id, turn.item_id,
                                      part={"type": "audio", "transcript": transcript}))
        item = turn.item("completed" if status == "completed" else "incomplete")
        await self.send(server_event("response.output_item.done", response_id=turn.response_id, output_index=0,
                                     item=item))
        await self.send(server_event("conversation.item.done", previous_item_id=None, item=item))
        await self.send(server_event("response.done", response=response_object(
            turn.response_id, status, ["audio"], output=[item],
            status_details=None if status == "completed" else {"type": status})))
        if self.turns:
            await self._open_turn(self.turns[0])
