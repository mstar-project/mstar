"""Step-Audio2's prompt: the chat rendering, its tokenization, and the audio front end.

Mirrors ``StepAudio2.apply_chat_template`` and ``StepAudio2Base.__call__``
(``stepaudio2.py:14-27``, ``:94-127``): each turn is ``<|BOT|>{role}\\n{content}<|EOT|>``
with role ``user`` written ``human``; a turn's content is text, a list of text / audio /
token items, or None (an open turn, the generation prompt); ``eot=False`` leaves a turn
open (a spoken reply's ``<|BOT|>assistant\\n<tts_start>``). Audio is cut into 25 s windows,
each a log-mel clip whose embedding rows replace the ``<audio_patch>`` run right after its
``<audio_start>``.

The reference tokenizes every rendered piece on its own and concatenates the ids
(``stepaudio2.py:18-26``), so two adjacent text pieces never merge into one BPE token; the
rendering here keeps its pieces for the same reason.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from mstar.model.components.audio_features import librosa_mel_filters
from mstar.model.step_audio2.config import (
    AUDIO_PATCH,
    AUDIO_START,
    HOP_LENGTH,
    MEL_PADDING,
    N_FFT,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    audio_tokens,
)

# A rendered piece: text to tokenize, or ids as they are (a ``token`` item).
Piece = str | list[int]


def log_mel(audio: torch.Tensor, n_mels: int = 128) -> torch.Tensor:
    """``log_mel_spectrogram(audio, n_mels, padding=479)`` (``utils.py:33-54``): float32
    16 kHz samples ``[T]`` -> ``[n_mels, mel_frames(T)]``."""
    audio = F.pad(audio.to(torch.float32), (0, MEL_PADDING))
    window = torch.hann_window(N_FFT, device=audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = torch.from_numpy(librosa_mel_filters(SAMPLE_RATE, N_FFT, n_mels)).to(audio.device)
    log_spec = torch.clamp(filters @ magnitudes, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def audio_clips(audio: torch.Tensor) -> list[torch.Tensor]:
    """A 16 kHz waveform -> its 25 s windows' log-mels (``stepaudio2.py:113-115``)."""
    return [log_mel(audio[i:i + WINDOW_SAMPLES]) for i in range(0, audio.shape[0], WINDOW_SAMPLES)]


def _audio_piece(mel: torch.Tensor) -> str:
    return "<audio_start>" + "<audio_patch>" * audio_tokens(mel.shape[-1]) + "<audio_end>"


def render(
    messages: Sequence[dict[str, Any]], load_audio: Callable[[Any], torch.Tensor],
) -> tuple[list[Piece], list[torch.Tensor]]:
    """The reference's chat rendering: its pieces in order and the audio clips' log-mels.

    ``load_audio`` turns an audio item's ``audio`` value into 16 kHz float32 samples.
    """
    pieces: list[Piece] = []
    mels: list[torch.Tensor] = []
    for msg in messages:
        role = "human" if msg["role"] == "user" else msg["role"]
        content = msg["content"]
        eot = msg.get("eot", True)
        if isinstance(content, str):
            pieces.append(f"<|BOT|>{role}\n{content}" + ("<|EOT|>" if eot else ""))
        elif isinstance(content, list):
            pieces.append(f"<|BOT|>{role}\n")
            for item in content:
                kind = item["type"]
                if kind == "text":
                    pieces.append(item["text"])
                elif kind == "audio":
                    for mel in audio_clips(load_audio(item["audio"])):
                        mels.append(mel)
                        pieces.append(_audio_piece(mel))
                elif kind == "token":
                    pieces.append(list(item["token"]))
                else:
                    raise ValueError(f"unsupported content item type {kind!r}")
            if eot:
                pieces.append("<|EOT|>")
        elif content is None:
            pieces.append(f"<|BOT|>{role}\n")
        else:
            raise ValueError(f"unsupported content type {type(content).__name__}")
    return pieces, mels


def tokenize(pieces: Sequence[Piece], encode: Callable[[str], list[int]]) -> list[int]:
    """Each text piece tokenized on its own, ids pieces as they are, end to end.
    ``encode`` is the tokenizer's ``text -> ids`` with no special tokens added."""
    ids: list[int] = []
    for piece in pieces:
        ids += list(piece) if isinstance(piece, list) else encode(piece)
    return ids


def audio_positions(ids: Sequence[int], tokens_per_clip: Sequence[int]) -> list[int]:
    """The prompt rows each clip's embeddings fill, clips end to end: the run right after
    each ``<audio_start>`` (``modeling_step_audio_2.py:353-357``). Raises if a run is not
    the clip's ``<audio_patch>`` placeholders."""
    starts = [i for i, t in enumerate(ids) if t == AUDIO_START]
    if len(starts) != len(tokens_per_clip):
        raise ValueError(f"{len(starts)} <audio_start> tokens for {len(tokens_per_clip)} audio clips")
    out: list[int] = []
    for start, n in zip(starts, tokens_per_clip, strict=True):
        run = range(start + 1, start + 1 + n)
        if any(ids[i] != AUDIO_PATCH for i in run if i < len(ids)) or run.stop > len(ids):
            raise ValueError(f"the clip at {start} has {n} rows but its placeholder run differs")
        out += list(run)
    return out
