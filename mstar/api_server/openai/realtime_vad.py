"""Server turn detection for ``/v1/realtime`` (OpenAI's ``server_vad``).

``TurnSegmenter`` cuts a 16 kHz stream into turns from a per-frame speech
probability, and returns what to do in stream order: start a turn, add audio
to it, stop it, commit it. A commit covers exactly the frames up to the end
of the turn's silence window; audio after that point, even in the same
append, is segmented for the next turn, so a turn end inside one append never
drops the start of the next utterance, and each sample is committed exactly once.

The probability comes from Silero VAD (MIT), its 2 MB ONNX export run by
onnxruntime on one CPU thread, about 0.15 ms per 32 ms frame. Its weights are
read from disk only: ``MSTAR_SILERO_VAD_PATH``, else the copy bundled with
the ``silero-vad`` package (``pip install 'mstar[vad]'``).
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME = 512              # samples per decision: Silero's window at 16 kHz, 32 ms
# Silence inside a turn starts on a frame below threshold - HYSTERESIS and a
# frame at or above threshold cancels it; a borderline frame in between does
# neither (the same hysteresis as Silero's VADIterator). The turn ends on
# whichever frame completes the silence window, borderline frames included,
# counting the first silent frame; VADIterator ends only on a frame below
# threshold - HYSTERESIS and counts from the frame after the first silent one.
HYSTERESIS = 0.15


def samples_to_ms(n: int) -> int:
    return n * 1000 // SAMPLE_RATE


class SpeechProbability(Protocol):
    """Per-session speech probability of one FRAME of 16 kHz audio; stateful."""

    def __call__(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class ServerVAD:
    """OpenAI's ``server_vad`` settings. ``create_response`` and
    ``interrupt_response`` are accepted and echoed; a transcription session
    has no response to create or interrupt."""

    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    create_response: bool = True
    interrupt_response: bool = True

    def view(self) -> dict:
        return {"type": "server_vad", **asdict(self)}


class SileroVAD:
    """Silero VAD for one stream. The ONNX session is shared and stateless;
    the stream keeps the model's recurrent state and the last CONTEXT
    samples, which the model reads before each frame."""

    CONTEXT = 64

    def __init__(self, session):
        self._session = session
        self.reset()

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, np.asarray(frame, dtype=np.float32)])[None]
        prob, self._state = self._session.run(None, {"input": x, "state": self._state, "sr": _SR})
        self._context = x[0, -self.CONTEXT:]
        return float(prob[0, 0])

    def reset(self) -> None:
        self._state = np.zeros(_STATE, dtype=np.float32)
        self._context = np.zeros(self.CONTEXT, dtype=np.float32)


_SR = np.array(SAMPLE_RATE, dtype=np.int64)
_STATE = (2, 1, 128)
# what SileroVAD feeds the model
_INPUTS = {"input": "tensor(float)", "state": "tensor(float)", "sr": "tensor(int64)"}


def _mismatch(session) -> str | None:
    """Why ``session`` cannot score frames the way SileroVAD runs it, or None.
    The silero-vad package ships other exports next to the right one (half
    precision, OpenVINO, sequence) that take other inputs; without this check
    they load fine and then fail on a session's first append."""
    inputs = {i.name: i.type for i in session.get_inputs()}
    if inputs != _INPUTS:
        return f"its inputs are {inputs}, not {_INPUTS}"
    x = np.zeros((1, SileroVAD.CONTEXT + FRAME), dtype=np.float32)
    try:
        out = session.run(None, {"input": x, "state": np.zeros(_STATE, dtype=np.float32), "sr": _SR})
    except Exception as exc:  # noqa: BLE001 — any failure here would come back on the first append
        return f"it cannot score a silent frame: {exc}"
    shapes = [np.shape(o) for o in out]
    if shapes != [(1, 1), _STATE]:
        return f"its outputs have shapes {shapes}, not [(1, 1), {_STATE}]"
    return None


def silero_weights() -> Path | None:
    """The Silero ONNX file this server uses, or None."""
    path = os.environ.get("MSTAR_SILERO_VAD_PATH")
    if path:
        return Path(path)
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        return None
    bundled = Path(next(iter(spec.submodule_search_locations))) / "data" / "silero_vad.onnx"
    return bundled if bundled.is_file() else None


@functools.lru_cache(maxsize=1)
def silero_factory() -> Callable[[], SpeechProbability] | None:
    """Make one SileroVAD per session over a model loaded once, or return
    None (with one warning) when the weights or onnxruntime are missing."""
    path = silero_weights()
    if path is None:
        logger.warning("server_vad is unavailable: no Silero VAD weights (set MSTAR_SILERO_VAD_PATH "
                       "or install mstar[vad])")
        return None
    try:
        import onnxruntime as ort

        # one thread: a 32 ms frame is too small to split, and a multi-thread
        # pool's workers would spin between every session's frames
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    except Exception:  # noqa: BLE001 — a bad file disables server_vad, not the server
        logger.exception("server_vad is unavailable: cannot load Silero VAD from %s (needs onnxruntime, "
                         "mstar[vad])", path)
        return None
    if (why := _mismatch(session)) is not None:
        logger.warning("server_vad is unavailable: %s is not the Silero VAD export this server runs "
                       "(silero_vad.onnx): %s", path, why)
        return None
    return lambda: SileroVAD(session)


class TurnSegmenter:
    """Splits a 16 kHz stream into turns (OpenAI ``server_vad`` semantics).

    ``feed`` takes audio of any length and returns operations in stream
    order: ``("speech_started", audio_start_ms)``, ``("audio", samples)`` for
    the open turn, ``("speech_stopped", audio_end_ms)``, ``("commit",)``.
    Times are milliseconds from the start of the session's audio. A turn
    starts on a frame at or above ``threshold`` and takes up to
    ``prefix_padding_ms`` of the audio before it; it ends after the frame
    that brings ``silence_duration_ms`` of whole frames, counted from a frame
    below ``threshold - HYSTERESIS`` (included) with none at or above
    ``threshold`` since, whatever that last frame's probability (the frames
    belong to the turn). A turn holds at most ``max_turn_samples``: it is
    committed before the frame that would pass that, and in mid-speech the
    next turn starts with that frame. Audio between turns, beyond the
    prefix, is in no turn. A partial frame waits for the next ``feed``.
    """

    def __init__(self, vad: SpeechProbability, cfg: ServerVAD, max_turn_samples: int, pos: int = 0):
        self.vad, self.cfg = vad, cfg
        self.max_turn = max_turn_samples
        self._prefix_cap = cfg.prefix_padding_ms * SAMPLE_RATE // 1000
        self._silence_cap = cfg.silence_duration_ms * SAMPLE_RATE // 1000
        self._neg = max(cfg.threshold - HYSTERESIS, 0.01)
        self._rest = np.zeros(0, dtype=np.float32)   # the partial frame
        self._pos = pos                              # stream index of _rest[0]
        self._prefix = np.zeros(0, dtype=np.float32)
        self.speaking = False
        self._silence = 0                            # trailing silent samples in the turn
        self._turn = 0                               # samples in the turn

    def feed(self, audio: np.ndarray) -> list[tuple]:
        ops: list[tuple] = []
        buf = np.concatenate([self._rest, np.asarray(audio, dtype=np.float32)]) if self._rest.size else \
            np.asarray(audio, dtype=np.float32)
        whole = buf.size - buf.size % FRAME
        for i in range(0, whole, FRAME):
            self._frame(buf[i:i + FRAME], self._pos + i, ops)
        self._rest = buf[whole:].copy()
        self._pos += whole
        return ops

    def _frame(self, frame: np.ndarray, start: int, ops: list) -> None:
        p = self.vad(frame)
        if self.speaking and self._turn + FRAME > self.max_turn:
            # the item cap: the turn ends before this frame; in mid-speech the
            # next turn starts with it, whatever its probability
            mid_speech = not self._silence
            self._stop(start, ops)
            if mid_speech:
                self._open(start, ops)
        if not self.speaking:
            if p < self.cfg.threshold:
                if self._prefix_cap:
                    self._prefix = np.concatenate([self._prefix, frame])[-self._prefix_cap:]
                return
            self._open(start, ops)
        ops.append(("audio", frame))
        self._turn += FRAME
        # the hysteresis of Silero's VADIterator: a frame below threshold -
        # HYSTERESIS starts the silence, one at or above threshold cancels it,
        # and one in between leaves it running
        if p >= self.cfg.threshold:
            self._silence = 0
        elif p < self._neg or self._silence:
            self._silence += FRAME
        # at least one silent frame, so silence_duration_ms 0 does not end a turn mid-speech
        if self._silence and self._silence >= self._silence_cap:
            self._stop(start + FRAME, ops)

    def _open(self, start: int, ops: list) -> None:
        """Start a turn at stream index ``start``, with the prefix padding
        (cut so the turn's first frame still fits the cap)."""
        keep = min(self._prefix.size, max(self.max_turn - FRAME, 0))
        prefix = self._prefix[self._prefix.size - keep:]
        self.speaking, self._turn, self._silence = True, prefix.size, 0
        self._prefix = np.zeros(0, dtype=np.float32)
        ops.append(("speech_started", samples_to_ms(start - prefix.size)))
        if prefix.size:
            ops.append(("audio", prefix))

    def _stop(self, end: int, ops: list) -> None:
        ops.append(("speech_stopped", samples_to_ms(end)))
        ops.append(("commit",))
        self.speaking, self._silence, self._turn = False, 0, 0

    def end_turn(self) -> list[tuple]:
        """A client commit: an open turn takes the partial frame and stops
        there. Between turns the partial frame and the prefix padding are
        dropped, so audio from before the commit never reaches the next turn."""
        ops: list[tuple] = []
        rest, self._rest = self._rest, np.zeros(0, dtype=np.float32)
        if self.speaking:
            if rest.size and self._turn + rest.size > self.max_turn:
                self._stop(self._pos, ops)
                self._open(self._pos, ops)
            if rest.size:
                ops.append(("audio", rest))
            self._pos += rest.size
            self._stop(self._pos, ops)
            ops.pop()   # the client's commit is the commit
        else:
            self._pos += rest.size
            self._prefix = np.zeros(0, dtype=np.float32)
        return ops

    def reset(self, skipped: int = 0) -> None:
        """A clear: drop buffered audio, keep the timeline (``skipped`` more
        samples were dropped before they reached the segmenter)."""
        self._pos += self._rest.size + skipped
        self._rest = np.zeros(0, dtype=np.float32)
        self._prefix = np.zeros(0, dtype=np.float32)
        self.speaking, self._silence, self._turn = False, 0, 0
        self.vad.reset()

    def shift(self, n: int) -> None:
        """Move the stream position by ``n`` samples (the session's timeline
        correction after a resampler start-over, with no partial frame held)."""
        self._pos += n

    def take_over(self, old: TurnSegmenter) -> None:
        """Go on from ``old`` between turns (a settings change): its stream
        position, its partial frame, not scored yet, and its prefix padding."""
        self._pos, self._rest = old._pos, old._rest
        self._prefix = old._prefix[old._prefix.size - min(old._prefix.size, self._prefix_cap):]

    def unscored(self) -> np.ndarray:
        """The partial frame, not scored yet."""
        return self._rest
