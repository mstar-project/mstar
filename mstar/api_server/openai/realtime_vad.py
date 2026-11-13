"""Server turn detection for ``/v1/realtime`` (OpenAI's ``server_vad`` and
``semantic_vad``).

``TurnSegmenter`` cuts a 16 kHz stream into turns from a per-frame speech
probability, and returns what to do in stream order: start a turn, add audio
to it, stop it, commit it. A commit covers exactly the frames up to the end
of the turn's silence window; audio after that point, even in the same
append, is segmented for the next turn, so a turn end inside one append never
drops the start of the next utterance, and each sample is committed exactly once.

The probability comes from Silero VAD (MIT), its 2 MB ONNX export run by
onnxruntime on one CPU thread, about 0.15 ms per 32 ms frame. Its weights are
read from disk only: ``MSTAR_SILERO_VAD_PATH``, else the copy bundled with
the ``silero-vad`` package (``pip install 'mstar-ai[vad]'``). ``semantic_vad`` adds
Pipecat's Smart Turn v3 (BSD-2), an 8 MB ONNX model that judges from the audio
whether a pause ends the turn; without it (or onnxruntime), or with a file
that is not that model, ``semantic_vad`` is refused with an error event.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME = 512              # samples per decision: Silero's window at 16 kHz, 32 ms
class SpeechProbability(Protocol):
    """Per-session speech probability of one FRAME of 16 kHz audio; stateful."""

    def __call__(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


SMART_TURN_SECONDS = 8       # the model hears the last 8 s of the turn
SMART_TURN_REPO, SMART_TURN_FILE = "pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"


class SmartTurn:
    """Pipecat Smart Turn v3 (BSD-2, 8 MB int8 ONNX on CPU): P(the speaker
    has finished) from the last 8 s of a turn, about 45 ms a call on one
    thread. Stateless, so one instance serves every session."""

    def __init__(self, path: Path):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        so = ort.SessionOptions()
        so.inter_op_num_threads = so.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self._features = WhisperFeatureExtractor(chunk_length=SMART_TURN_SECONDS)
        # another ONNX file loads fine and then fails on a session's first pause
        inputs = {i.name: i.type for i in self._session.get_inputs()}
        if inputs != _SMART_TURN_INPUTS:
            raise _NotSmartTurnError(f"its inputs are {inputs}, not {_SMART_TURN_INPUTS}")
        shapes = [np.shape(o) for o in self._run(np.zeros(0, dtype=np.float32))]
        if shapes[:1] != [(1, 1)]:
            raise _NotSmartTurnError(f"its outputs have shapes {shapes}, not [(1, 1)]")

    def _run(self, audio: np.ndarray) -> list:
        n = SMART_TURN_SECONDS * SAMPLE_RATE
        audio = np.asarray(audio, dtype=np.float32)[-n:]
        audio = np.pad(audio, (n - audio.size, 0))   # the reference pads at the front
        features = self._features(audio, sampling_rate=SAMPLE_RATE, return_tensors="np", padding="max_length",
                                  max_length=n, truncation=True, do_normalize=True).input_features
        return self._session.run(None, {"input_features": features.astype(np.float32)})

    def __call__(self, audio: np.ndarray) -> float:
        return float(self._run(audio)[0][0][0])


_SMART_TURN_INPUTS = {"input_features": "tensor(float)"}


class _NotSmartTurnError(ValueError):
    """The file loads, but its inputs or outputs are not Smart Turn v3's."""


def smart_turn_weights() -> Path | None:
    """``MSTAR_SMART_TURN_PATH``, else the file in the local Hugging Face
    cache (``hf download pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx``);
    never downloaded here."""
    path = os.environ.get("MSTAR_SMART_TURN_PATH")
    if path:
        return Path(path)
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILE, local_files_only=True))
    except Exception:  # noqa: BLE001 — not cached, or no huggingface_hub
        return None


_smart_turn_problem = "no Smart Turn model is loaded"


def smart_turn_problem() -> str:
    """Why ``smart_turn()`` gave None, in words safe to send a client (no paths)."""
    return _smart_turn_problem


@functools.lru_cache(maxsize=1)
def smart_turn() -> Callable[[np.ndarray], float] | None:
    """The shared end-of-turn model, or None (with one warning, and the
    reason in ``smart_turn_problem``): semantic_vad is then refused."""
    global _smart_turn_problem
    _smart_turn_problem = "no Smart Turn model is loaded"
    path = smart_turn_weights()
    if path is None or not path.is_file():
        logger.warning("semantic_vad is unavailable: no Smart Turn weights%s (set MSTAR_SMART_TURN_PATH "
                       "or hf download %s %s)", f" at {path}" if path else "", SMART_TURN_REPO, SMART_TURN_FILE)
        _smart_turn_problem = "no Smart Turn weights on this server"
        return None
    try:
        return SmartTurn(path)
    except Exception as exc:  # noqa: BLE001 — missing onnxruntime/transformers or a bad file
        logger.warning("semantic_vad is unavailable: cannot load Smart Turn from %s (%s: %s); "
                       "it needs mstar-ai[vad] and the %s file", path, type(exc).__name__, exc, SMART_TURN_FILE)
        if isinstance(exc, ImportError):
            _smart_turn_problem = "onnxruntime or transformers is not installed"
        elif isinstance(exc, _NotSmartTurnError):
            _smart_turn_problem = "the Smart Turn file is not the expected model"
        else:
            _smart_turn_problem = "the Smart Turn model did not load"
        return None


def report_availability() -> None:
    """Load the turn detection models once at startup, so a missing one is a
    startup warning rather than a surprise in the first session."""
    if silero_factory() is not None:
        smart_turn()


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
                       "or install mstar-ai[vad])")
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
                         "mstar-ai[vad])", path)
        return None
    if (why := _mismatch(session)) is not None:
        logger.warning("server_vad is unavailable: %s is not the Silero VAD export this server runs "
                       "(silero_vad.onnx): %s", path, why)
        return None
    return lambda: SileroVAD(session)

