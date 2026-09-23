"""``/v1/realtime`` (transcription intent): streaming speech-to-text over a WebSocket.

A subset of OpenAI's Realtime transcription session, enough for a client
that streams microphone audio and wants partial transcripts:

Client events
    ``transcription_session.update``  ``{"session": {"input_audio_transcription":
    {"model", "language", "prompt"}, "input_audio_format": "pcm16", "mstar":
    {"chunk_seconds", "unfixed_chunks", "unfixed_tokens"}}}``
    ``input_audio_buffer.append``      ``{"audio": <base64 PCM16 mono 16 kHz>}``
    ``input_audio_buffer.commit``      end of the utterance: flush and finish
    ``input_audio_buffer.clear``       drop buffered audio and start over

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
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from mstar.api_server import media_io
from mstar.api_server.openai._util import now
from mstar.api_server.openai.protocol import TranscriptionRequest

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DEFAULT_CHUNK_SECONDS = 2.0
DEFAULT_UNFIXED_CHUNKS = 2
DEFAULT_UNFIXED_TOKENS = 5


@dataclass
class RealtimeSession:
    """One WebSocket's transcription state (the SDK's ``ASRStreamingState``)."""
    session_id: str
    request: TranscriptionRequest
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    unfixed_chunks: int = DEFAULT_UNFIXED_CHUNKS
    unfixed_tokens: int = DEFAULT_UNFIXED_TOKENS
    pending: list[np.ndarray] = field(default_factory=list)   # audio not yet in a chunk
    pending_samples: int = 0
    heard: list[np.ndarray] = field(default_factory=list)     # every chunk decoded so far
    chunk_id: int = 0
    raw_hypothesis: str = ""   # the model's raw output for the audio so far
    emitted: str = ""          # clean text already sent as deltas
    item_id: str = field(default_factory=lambda: f"item_{uuid.uuid4().hex[:12]}")

    @property
    def chunk_samples(self) -> int:
        return int(round(self.chunk_seconds * SAMPLE_RATE))

    def append(self, audio: np.ndarray) -> None:
        if audio.size:
            self.pending.append(audio)
            self.pending_samples += int(audio.size)

    def take_chunk(self) -> np.ndarray | None:
        """A full chunk of pending audio, or None until enough arrived."""
        if self.pending_samples < self.chunk_samples:
            return None
        buf = np.concatenate(self.pending)
        chunk, rest = buf[: self.chunk_samples], buf[self.chunk_samples:]
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
        self.pending, self.pending_samples, self.heard = [], 0, []
        self.chunk_id, self.raw_hypothesis, self.emitted = 0, "", ""
        self.item_id = f"item_{uuid.uuid4().hex[:12]}"


def pcm16_to_float(payload: str) -> np.ndarray:
    """Base64 PCM16 (little-endian, mono) -> float32 in [-1, 1). Strict
    decoding: a malformed payload is an error, not silence."""
    raw = base64.b64decode(payload, validate=True)
    return np.frombuffer(raw[: len(raw) - len(raw) % 2], dtype="<i2").astype(np.float32) / 32768.0


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
    path = media_io.write_wav(
        session.audio_so_far(), str(Path(api.upload_dir) / f"rt_{session.session_id}_{session.chunk_id:05d}.wav"),
        SAMPLE_RATE,
    )
    args = adapter.realtime_step_request(session.request, path, prefix)
    request_id = f"rt-{session.session_id}-{session.chunk_id}"
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
    chunks = await api.collect_results(request_id)
    text = "".join(c.data.decode("utf-8", "replace") for c in chunks if c.modality == "text")
    errors = [c for c in chunks if c.modality == "error"]
    if errors:
        raise RuntimeError(errors[0].data.decode("utf-8", "replace"))
    return prefix + text


def _event(kind: str, **fields) -> dict:
    return {"event_id": f"event_{uuid.uuid4().hex[:12]}", "type": kind, **fields}


def _error(message: str, kind: str = "invalid_request_error") -> dict:
    return _event("error", error={"type": kind, "message": message})


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
        )

    async def send(self, event: dict) -> None:
        await self.ws.send_text(json.dumps(event))

    def _session_view(self) -> dict:
        s = self.session
        return {
            "id": s.session_id,
            "object": "realtime.transcription_session",
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": self.model_name,
                "language": s.request.language,
                "prompt": s.request.prompt or "",
            },
            "mstar": {
                "chunk_seconds": s.chunk_seconds,
                "unfixed_chunks": s.unfixed_chunks,
                "unfixed_tokens": s.unfixed_tokens,
                "sample_rate": SAMPLE_RATE,
            },
        }

    async def run(self) -> None:
        await self.ws.accept()
        await self.send(_event("transcription_session.created", session=self._session_view()))
        try:
            while True:
                message = await self.ws.receive_text()
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    await self.send(_error("not JSON"))
                    continue
                await self.handle(event)
        except WebSocketDisconnect:
            return

    async def handle(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "transcription_session.update":
            self.update(event.get("session") or {})
            await self.send(_event("transcription_session.updated", session=self._session_view()))
        elif kind == "input_audio_buffer.append":
            try:
                audio = pcm16_to_float(event.get("audio") or "")
            except Exception:  # noqa: BLE001 — bad base64
                await self.send(_error("audio is not base64"))
                return
            self.session.append(audio)
            while (chunk := self.session.take_chunk()) is not None:
                await self.step(chunk)
        elif kind == "input_audio_buffer.commit":
            tail = self.session.flush()
            if tail is not None and tail.size:
                await self.step(tail, final=True)
            else:
                # nothing left to hear: what was held back is final now
                await self.emit(self.current_text(), hold=0)
            await self.send(_event("input_audio_buffer.committed", item_id=self.session.item_id))
            await self.send(_event(
                "conversation.item.input_audio_transcription.completed",
                item_id=self.session.item_id, transcript=self.current_text(),
            ))
            self.session.clear()
        elif kind == "input_audio_buffer.clear":
            self.session.clear()
            await self.send(_event("input_audio_buffer.cleared"))
        else:
            await self.send(_error(f"unknown event {kind!r}"))

    def update(self, session: dict) -> None:
        s = self.session
        transcription = session.get("input_audio_transcription") or {}
        fields = {k: v for k, v in transcription.items() if k in ("language", "prompt")}
        if fields:
            s.request = s.request.model_copy(update=fields)
        knobs = session.get("mstar") or {}
        if "chunk_seconds" in knobs:
            s.chunk_seconds = max(float(knobs["chunk_seconds"]), 0.1)
        if "unfixed_chunks" in knobs:
            s.unfixed_chunks = int(knobs["unfixed_chunks"])
        if "unfixed_tokens" in knobs:
            s.unfixed_tokens = int(knobs["unfixed_tokens"])
        fmt = session.get("input_audio_format")
        if fmt not in (None, "pcm16"):
            raise ValueError(f"only pcm16 input is supported; got {fmt!r}")

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
        except Exception as exc:  # noqa: BLE001 — surface as an event, keep the session
            logger.exception("realtime step failed")
            await self.send(_error(str(exc), kind="server_error"))
            return
        s.chunk_id += 1
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
