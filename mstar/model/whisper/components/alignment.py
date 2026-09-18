"""Word-level timestamps from the decoder's cross-attention.

Whisper was not trained to emit word boundaries; openai-whisper derives them
from a handful of cross-attention heads (``alignment_heads`` in the
generation config) that track the audio position of the token being
predicted. The recipe, reproduced here on M*'s own components:

1. run the decoder once more over ``<|sot|><|lang|><|task|><|notimestamps|>
   text... <|eot|>`` with teacher forcing and keep those heads' attention
   probabilities over the encoder frames that hold audio;
2. normalize each head over its tokens, smooth along time with a median
   filter, average the heads: one ``(tokens, frames)`` alignment matrix
   (row ``i`` is the step that predicts text token ``i``; the last row
   predicts end-of-text);
3. dynamic time warping through that matrix gives a monotonic path; the
   frame at which the path first enters a row is when that token starts;
4. tokens are grouped into words (a new word at a leading space or a lone
   punctuation mark); a word starts with its first token and ends where the
   next word starts, the last one where end-of-text is predicted.

The functions here are pure tensor/numpy code with no engine dependencies,
so the same math is unit-tested on the CPU and run by the decoder
submodule's ``align`` walk on the GPU.
"""
from __future__ import annotations

import string

import numpy as np
import torch
import torch.nn.functional as F

# encoder positions per second: a 30 s window is 1500 positions
FRAMES_PER_SECOND = 50
MEDIAN_FILTER_WIDTH = 7
PREPENDED_PUNCTUATION = "\"'“¿([{-"
APPENDED_PUNCTUATION = "\"'.。,，!！?？:：”)]}、"


def median_filter(x: torch.Tensor, width: int = MEDIAN_FILTER_WIDTH) -> torch.Tensor:
    """Sliding median over the last dim (odd ``width``), edges reflected."""
    if width <= 1 or x.shape[-1] <= 1:
        return x
    pad = width // 2
    mode = "reflect" if x.shape[-1] > pad else "replicate"
    padded = F.pad(x, (pad, pad), mode=mode)
    return padded.unfold(-1, width, 1).median(dim=-1).values


def _dtw_trace(cost: np.ndarray) -> np.ndarray:
    """Fill the DTW table for ``cost`` (rows x frames) and return, per cell,
    which neighbour it came from: 0 diagonal, 1 the row above, 2 the frame
    before. Row 0 / frame 0 of the padded table are the sentinels."""
    rows, frames = cost.shape
    acc = np.full((rows + 1, frames + 1), np.inf, dtype=np.float64)
    trace = np.zeros((rows + 1, frames + 1), dtype=np.int8)
    acc[0, 0] = 0.0
    for j in range(1, frames + 1):
        for i in range(1, rows + 1):
            diag = acc[i - 1, j - 1]
            up = acc[i - 1, j]
            left = acc[i, j - 1]
            if diag <= up and diag <= left:
                best, move = diag, 0
            elif up <= left:
                best, move = up, 1
            else:
                best, move = left, 2
            acc[i, j] = cost[i - 1, j - 1] + best
            trace[i, j] = move
    return trace


try:  # the double loop is ~60k cells for a full window; numba makes it ~1 ms
    from numba import njit

    _dtw_trace_fast = njit(cache=True, nogil=True)(_dtw_trace)
except Exception:  # noqa: BLE001 — numba missing or unable to compile: plain Python
    _dtw_trace_fast = _dtw_trace


def dtw(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cheapest monotonic path from the top-left to the bottom-right of
    ``cost`` (rows x frames), moving right, down or diagonally. Returns the
    path's row and frame indices, each of length ``len(path)``."""
    cost = np.ascontiguousarray(cost, dtype=np.float64)
    trace = _dtw_trace_fast(cost)
    trace[0, :] = 2
    trace[:, 0] = 1
    i, j = cost.shape
    path = []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        move = trace[i, j]
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    arr = np.array(path, dtype=np.int64)
    return arr[:, 0], arr[:, 1]


def alignment_matrix(weights: torch.Tensor, num_frames: int, medfilt_width: int = MEDIAN_FILTER_WIDTH) -> torch.Tensor:
    """``weights: (heads, rows, frames)`` attention probabilities of the
    alignment heads -> the ``(rows, audio_frames)`` matrix DTW walks.

    Only the frames that hold audio count (``num_frames`` mel frames, two per
    encoder position); each head is standardized over its rows so heads with
    different sharpness weigh alike, smoothed along time, then averaged.
    """
    audio_positions = max(1, min(weights.shape[-1], num_frames // 2))
    w = weights[:, :, :audio_positions].float()
    std, mean = torch.std_mean(w, dim=-2, keepdim=True, unbiased=False)
    w = (w - mean) / std.clamp_min(1e-6)
    w = median_filter(w, medfilt_width)
    # a head that never varies (or a broken pass) must not derail the DTW
    return torch.nan_to_num(w.mean(dim=0), nan=0.0)


def token_start_frames(matrix: torch.Tensor) -> np.ndarray:
    """The frame at which the DTW path first enters each row of ``matrix``
    (a high value is a good alignment, so the path minimizes ``-matrix``)."""
    rows, frames = dtw(-matrix.detach().cpu().numpy())
    first = np.ones(len(rows), dtype=bool)
    first[1:] = np.diff(rows) > 0
    return frames[first]


def split_words(decode, token_ids: list[int]) -> list[tuple[str, list[int]]]:
    """Group text tokens into words: multi-byte characters are kept whole
    (a token whose text decodes to the replacement character continues into
    the next), and a word begins at a leading space or a lone punctuation
    mark. ``decode(ids) -> str`` is the tokenizer's plain decode."""
    pieces: list[tuple[str, list[int]]] = []
    pending: list[int] = []
    for tok in token_ids:
        pending.append(tok)
        text = decode(pending)
        if "�" not in text:
            pieces.append((text, pending))
            pending = []
    if pending:
        pieces.append((decode(pending), pending))
    words: list[tuple[str, list[int]]] = []
    for text, toks in pieces:
        if not words or text.startswith(" ") or text.strip() in string.punctuation:
            words.append((text, list(toks)))
        else:
            prev_text, prev_toks = words[-1]
            words[-1] = (prev_text + text, prev_toks + list(toks))
    return words


def merge_punctuation(words: list[dict]) -> list[dict]:
    """Attach opening marks to the word after them and closing marks to the
    word before them, keeping the neighbour's timing (openai-whisper does the
    same before reporting words)."""
    merged: list[dict] = []
    for w in words:
        text = w["word"]
        if merged and not merged[-1]["word"].endswith(" ") and text.strip() in APPENDED_PUNCTUATION:
            merged[-1]["word"] += text
            merged[-1]["end"] = w["end"]
            continue
        merged.append(dict(w))
    out: list[dict] = []
    for w in merged:
        if out and out[-1]["word"].startswith(" ") and out[-1]["word"].strip() in PREPENDED_PUNCTUATION:
            opener = out.pop()
            w = {"word": opener["word"] + w["word"], "start": opener["start"], "end": w["end"]}
        out.append(w)
    return out


def word_timings(
    weights: torch.Tensor, text_tokens: list[int], decode, num_frames: int,
) -> list[dict]:
    """Words with ``start`` / ``end`` seconds for one window.

    ``weights`` holds the alignment heads' attention for the rows that
    predict ``text_tokens`` plus the final row that predicts end-of-text
    (``len(text_tokens) + 1`` rows); ``num_frames`` is how many mel frames
    of the window carry audio.
    """
    if not text_tokens:
        return []
    matrix = alignment_matrix(weights, num_frames)
    starts = token_start_frames(matrix) / FRAMES_PER_SECOND  # one per row
    words = split_words(decode, text_tokens)
    bounds = np.concatenate([[0], np.cumsum([len(toks) for _, toks in words])])
    timed = [
        {"word": text, "start": float(starts[b0]), "end": float(starts[b1])}
        for (text, _), b0, b1 in zip(words, bounds[:-1], bounds[1:], strict=True)
    ]
    return [{**w, "word": w["word"].strip()} for w in merge_punctuation(timed) if w["word"].strip()]
