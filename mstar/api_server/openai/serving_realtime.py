"""``/v1/realtime`` (transcription intent): streaming speech-to-text over a WebSocket.

A subset of OpenAI's Realtime transcription session, enough for a client
that streams microphone audio and wants partial transcripts:

Client events
    ``transcription_session.update``  ``{"session": {"input_audio_transcription":
    {"model", "language", "prompt"}, "input_audio_format", "turn_detection",
    "mstar": {"chunk_seconds", "unfixed_chunks", "unfixed_tokens"}}}``
    ``input_audio_buffer.append``      ``{"audio": <base64 PCM16 mono>}``
    ``input_audio_buffer.commit``      end of the utterance: flush and finish
    ``input_audio_buffer.clear``       drop buffered audio and start over

``input_audio_format`` is ``"pcm16"`` (16 kHz, the default) or OpenAI's
``{"type": "audio/pcm", "rate": 24000}``; 24 kHz audio is resampled to the
model's 16 kHz as it arrives, so partial ``audio_seconds`` stay on the
client's timeline. The rate changes only with an empty buffer.

``turn_detection`` null (the default) leaves ending an utterance to the
client's commit. ``{"type": "server_vad", "threshold", "prefix_padding_ms",
"silence_duration_ms"}`` lets the server end it (``realtime_vad``), adding
``input_audio_buffer.speech_started`` / ``.speech_stopped`` events with
``audio_start_ms`` / ``audio_end_ms`` on the session's audio timeline; a
turn that reaches the item cap is committed there instead of refused.

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
import itertools
import json
import logging
import math
import re
import reprlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from mstar.api_server import media_io
from mstar.api_server.openai import realtime_vad
from mstar.api_server.openai._util import now, rid
from mstar.api_server.openai.protocol import TranscriptionRequest

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Input rates a session accepts; 24 kHz (OpenAI's pcm rate) is resampled to SAMPLE_RATE.
INPUT_SAMPLE_RATES = (16000, 24000)
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
# Longest uncommitted item: the most audio one step decodes.
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
    # the client's audio rate; anything but SAMPLE_RATE goes through resampler
    input_rate: int = SAMPLE_RATE
    resampler: media_io.StreamResampler | None = None
    # samples the last chunk was short of its boundary (see take_chunk)
    short: int = 0
    # server turn detection (None: the client commits); the segmenter, like
    # the resampler, spans turns, so a commit does not reset either
    turn_detection: realtime_vad.ServerVAD | None = None
    segmenter: realtime_vad.TurnSegmenter | None = None
    stream_samples: int = 0    # SAMPLE_RATE samples written this session: the event timeline
    # the input clock: stream_samples at the last rate change, and input samples since
    clock_base: int = 0
    clock_in: int = 0

    @property
    def chunk_samples(self) -> int:
        return int(round(self.chunk_seconds * SAMPLE_RATE))

    def set_input_rate(self, rate: int) -> None:
        self.clock_base, self.clock_in = self.stream_samples, 0
        self.input_rate = rate
        self.resampler = None if rate == SAMPLE_RATE else media_io.StreamResampler(rate, SAMPLE_RATE)

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
        exact = self.clock_base + self.clock_in * SAMPLE_RATE // self.input_rate
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


def pcm16_to_float(payload: str, carry: bytes = b"") -> tuple[np.ndarray, bytes]:
    """Base64 PCM16 (little-endian, mono) -> (float32 in [-1, 1), the odd
    trailing byte). ``carry`` is the previous append's odd byte, so a sample
    split across two appends is joined instead of shifting every sample after
    it. Strict decoding: a malformed payload is an error, not silence."""
    # whitespace is allowed (base64.encodebytes wraps lines); anything else
    # outside the alphabet is an error. translate stays linear in the payload
    # where split() would build a list of its pieces
    raw = carry + base64.b64decode(payload.encode("ascii").translate(None, b" \t\n\r\v\f"), validate=True)
    whole = len(raw) - len(raw) % 2
    return np.frombuffer(raw[:whole], dtype="<i2").astype(np.float32) / 32768.0, raw[whole:]


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


def _input_rate(fmt, current: int) -> int:
    """The input rate an ``input_audio_format`` declares (``current`` when
    absent): ``"pcm16"`` is 16 kHz, as this endpoint has always read it;
    OpenAI's ``{"type": "audio/pcm", "rate": N}`` names its rate (24 kHz when
    left out, OpenAI's only one)."""
    if fmt is None:
        return current
    if fmt == "pcm16":
        return SAMPLE_RATE
    if isinstance(fmt, dict) and fmt.get("type") == "audio/pcm":
        if extra := sorted(set(fmt) - {"type", "rate"}):
            # channels, encoding and the like would change how the bytes read
            raise ValueError(f"input_audio_format takes only type and rate; got {_excerpt(extra)}")
        rate = fmt.get("rate", 24000)
        if not isinstance(rate, bool) and rate in INPUT_SAMPLE_RATES:
            return int(rate)
        raise ValueError(f"input_audio_format.rate must be 16000 or 24000; got {_excerpt(rate)}")
    raise ValueError(
        f'input_audio_format must be "pcm16" or {{"type": "audio/pcm", "rate": 24000}}; got {_excerpt(fmt)}'
    )


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


_TURN_DETECTION_KEYS = frozenset({"type", "threshold", "prefix_padding_ms", "silence_duration_ms",
                                  "create_response", "interrupt_response", "idle_timeout_ms"})


def _turn_detection(value) -> realtime_vad.ServerVAD | None:
    """A session's ``turn_detection``: null (the client commits) or
    ``server_vad``; fields left out take OpenAI's defaults."""
    if value is None:
        return None
    td = _object(value, "turn_detection")
    kind = td.get("type")
    if kind != "server_vad":
        what = f"type {_excerpt(kind)}" if isinstance(kind, str) else _excerpt(td)
        raise ValueError(f"turn_detection {what} is not supported; use server_vad, "
                         "or null to commit with input_audio_buffer.commit")
    unknown = next((k for k in td if k not in _TURN_DETECTION_KEYS), None)
    if unknown is not None:
        raise ValueError(f"turn_detection.{_excerpt(unknown)} is not supported")
    if td.get("idle_timeout_ms") is not None:
        raise ValueError("turn_detection.idle_timeout_ms is not supported")
    flags = {}
    for name in ("create_response", "interrupt_response"):
        flags[name] = td.get(name, True)
        if not isinstance(flags[name], bool):
            raise ValueError(f"turn_detection.{name} must be a boolean; got {_excerpt(flags[name])}")
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
    text = "".join(c.data.decode("utf-8", "replace") for c in chunks if c.modality == "text")
    errors = [c for c in chunks if c.modality == "error"]
    if errors:
        raise StepError(
            errors[0].data.decode("utf-8", "replace"), int((errors[0].metadata or {}).get("status") or 500),
        )
    return prefix + text


def _event(kind: str, **fields) -> dict:
    return {"event_id": f"event_{uuid.uuid4().hex[:12]}", "type": kind, **fields}


def _error(message: str, kind: str = "invalid_request_error", code: str | None = None) -> dict:
    error = {"type": kind, "message": message}
    if code is not None:
        error["code"] = code
    return _event("error", error=error)


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
            "input_audio_format": (
                "pcm16" if s.input_rate == SAMPLE_RATE else {"type": "audio/pcm", "rate": s.input_rate}
            ),
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
                "sample_rate": s.input_rate,
            },
        }

    async def run(self) -> None:
        await self.ws.accept()
        await self.send(_event("transcription_session.created", session=self._session_view()))
        try:
            while True:
                # receive_text() raises KeyError on a binary frame
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", 1000), message.get("reason"))
                if message.get("text") is None:
                    await self.send(_error("an event must be a JSON text message, not binary"))
                    continue
                try:
                    event = json.loads(message["text"])
                # ValueError covers JSONDecodeError and integers past Python's
                # int-string digit limit; RecursionError is deep nesting
                except (ValueError, RecursionError):
                    await self.send(_error("not JSON"))
                    continue
                if not isinstance(event, dict):
                    await self.send(_error("an event must be a JSON object"))
                    continue
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
            await self.send(_event("transcription_session.updated", session=self._session_view()))
        elif kind == "input_audio_buffer.append":
            s = self.session
            try:
                # not `or ""`: a null is an error, not an empty append, on every realtime path
                audio, odd_byte = pcm16_to_float(event.get("audio", ""), s.odd_byte)
            except Exception:  # noqa: BLE001 — bad base64
                await self.send(_error("audio is not base64", code="invalid_audio"))
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
                # off the event loop: one large append is thousands of VAD frames
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
            await self.send(_event("input_audio_buffer.cleared"))
        else:
            await self.send(_error(f"unknown event {_excerpt(kind)}"))

    async def apply(self, ops: list[tuple]) -> None:
        """Carry out segmenter operations in order (see ``TurnSegmenter``).
        Audio steps once, after the last operation, so one long append is one
        step, as without turn detection; a turn's commit decodes the rest."""
        s = self.session
        for i, op in enumerate(ops):
            if op[0] == "audio":
                s.add(op[1])
                if i == len(ops) - 1 and (chunk := s.take_chunk()) is not None:
                    await self.step(chunk)
            elif op[0] == "commit":
                await self.commit()
            elif op[0] == "speech_started":
                await self.send(_event("input_audio_buffer.speech_started", audio_start_ms=op[1], item_id=s.item_id))
            else:
                await self.send(_event("input_audio_buffer.speech_stopped", audio_end_ms=op[1], item_id=s.item_id))

    async def commit(self) -> None:
        """Finish the item: decode what is left, report it, start a new one."""
        s = self.session
        tail = s.flush()
        if tail is not None and tail.size:
            await self.step(tail, final=True)
        else:
            # nothing left to hear: what was held back is final now
            await self.emit(self.current_text(), hold=0)
        await self.send(_event("input_audio_buffer.committed", item_id=s.item_id))
        await self.send(_event(
            "conversation.item.input_audio_transcription.completed",
            item_id=s.item_id, transcript=self.current_text(),
        ))
        s.clear()

    def update(self, session: dict) -> None:
        """Apply a session update, all or nothing."""
        s = self.session
        rate = _input_rate(session.get("input_audio_format"), s.input_rate)
        if rate != s.input_rate and s.item_samples():
            raise ValueError("commit or clear the input audio buffer before changing the input sample rate")
        if rate != s.input_rate and s.odd_byte:
            # a commit keeps the half sample, so only clear helps
            raise ValueError("half a sample is pending; clear the input audio buffer before changing "
                             "the input sample rate")
        td = _turn_detection(session["turn_detection"]) if "turn_detection" in session else s.turn_detection
        vad = None
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
        if rate != s.input_rate:
            s.set_input_rate(rate)
        if td != s.turn_detection:
            old, s.turn_detection = s.segmenter, td
            s.segmenter = None if td is None else realtime_vad.TurnSegmenter(
                vad(), td, int(s.max_item_seconds * SAMPLE_RATE), pos=s.stream_samples)
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
        await self.send(_event(
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
            await self.send(_event(
                "conversation.item.input_audio_transcription.delta", item_id=s.item_id, delta=firm,
            ))
