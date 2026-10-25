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
   punctuation mark, or at every run of tokens that decodes to whole
   characters in a language written without spaces); a word starts with its first token and ends where the next word
   starts, the last one where end-of-text is predicted.

Steps 1 and 2 are GPU work: the decoder submodule's ``align`` walk emits the
matrix (``alignment_output``). Steps 3 and 4 run on the host in the API
server, when ``WhisperModel.postprocess`` renders that output
(``word_timings``). The functions here are pure tensor/numpy code with no
engine dependencies, so the same math is unit-tested on the CPU.
"""
from __future__ import annotations

import logging
import string

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# encoder positions per second: a 30 s window is 1500 positions
FRAMES_PER_SECOND = 50
MEDIAN_FILTER_WIDTH = 7
# Rows of the teacher-forced sequence before the first that predicts a text
# token: <|sot|><|lang|><|task|> (the <|notimestamps|> row predicts it)
PROMPT_ROWS = 3
PREPENDED_PUNCTUATION = "\"'“¿([{-"
APPENDED_PUNCTUATION = "\"'.。,，!！?？:：”)]}、"
# Written without spaces between words: without morpheme analysis, each run of
# tokens that decodes to whole characters is a word (openai-whisper's
# split_to_word_tokens).
CHARACTER_SPLIT_LANGUAGES = frozenset({"zh", "ja", "th", "lo", "my", "yue"})


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


# The double loop is up to ~670k cells (444 rows by 1500 frames): milliseconds
# compiled by numba (the asr extra), far longer as plain Python. ``warm_up``
# swaps the compiled one in.
_dtw_trace_fast = _dtw_trace


def warm_up() -> None:
    """Compile the DTW now, in the process that runs it (the API server's data
    worker, at model load), rather than inside the first word-timestamp
    request. Without numba, or if it cannot compile, the DTW stays plain
    Python, loudly."""
    global _dtw_trace_fast
    try:
        from numba import njit
    except ImportError:
        logger.warning("numba is not installed: word timestamps run the DTW in plain Python, far slower. "
                       "Install mstar[asr].")
        return
    try:
        fast = njit(cache=True, nogil=True)(_dtw_trace)
        fast(np.zeros((2, 2), dtype=np.float64))
    except Exception as exc:  # noqa: BLE001 — any compile failure: keep serving
        logger.warning("numba could not compile the DTW (%s): word timestamps fall back to plain Python", exc)
        return
    _dtw_trace_fast = fast


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


def alignment_output(weights: torch.Tensor, tokens: torch.Tensor, num_frames: int) -> torch.Tensor:
    """The align walk's one output tensor, from the alignment heads'
    ``weights`` over the whole teacher-forced ``tokens``
    (``<|sot|><|lang|><|task|><|notimestamps|> text <|eot|>``).

    Layout, float32 ``[n_text + 1, frames + 1]``: ``alignment_matrix`` of the
    rows that predict each text token and end-of-text, then one column of
    token ids (exact in float32): row ``i``'s text token, and on the last row
    the language token, which word splitting needs. Device ops only, so the
    walk never waits on the host.
    """
    matrix = alignment_matrix(weights[:, PROMPT_ROWS:-1], num_frames)
    ids = torch.cat([tokens[PROMPT_ROWS + 1:-1], tokens[1:2]]).to(matrix.dtype)
    return torch.cat([matrix, ids[:, None]], dim=1)


def unpack_output(output: torch.Tensor) -> tuple[torch.Tensor, list[int], int]:
    """``alignment_output``'s tensor -> (matrix, text token ids, language token id)."""
    ids = output[:, -1].long().tolist()
    return output[:, :-1], ids[:-1], ids[-1]


def token_start_frames(matrix: torch.Tensor) -> np.ndarray:
    """The frame at which the DTW path first enters each row of ``matrix``
    (a high value is a good alignment, so the path minimizes ``-matrix``)."""
    rows, frames = dtw(-matrix.detach().cpu().numpy())
    first = np.ones(len(rows), dtype=bool)
    first[1:] = np.diff(rows) > 0
    return frames[first]


def split_words(
    decode, token_ids: list[int], language: str | None = None,
) -> list[tuple[str, list[int]]]:
    """Group text tokens into words, as openai-whisper's ``split_to_word_tokens``:
    first into pieces that decode to whole characters (a token ending inside a
    multi-byte character continues into the next), then, unless ``language``
    is written without spaces, into words that begin at a leading space or a
    lone punctuation mark. ``decode(ids) -> str`` is the tokenizer's plain decode,
    without clean-up, so the pieces' texts add up to ``decode(token_ids)``."""
    full = decode(token_ids)
    pieces: list[tuple[str, list[int]]] = []
    pending: list[int] = []
    offset = 0
    for tok in token_ids:
        pending.append(tok)
        text = decode(pending)
        # a replacement character the whole transcript also has there is in the
        # text itself, not an unfinished character, and must not swallow the rest
        at = offset + text.index("�") if "�" in text else None
        if at is None or (at < len(full) and full[at] == "�"):
            pieces.append((text, pending))
            pending = []
            offset += len(text)
    if pending:
        # the transcript ends inside a character (cut at the position table)
        pieces.append((decode(pending), pending))
    if language in CHARACTER_SPLIT_LANGUAGES:
        return pieces
    words: list[tuple[str, list[int]]] = []
    for text, toks in pieces:
        if not words or text.startswith(" ") or text.strip() in string.punctuation:
            words.append((text, list(toks)))
        else:
            prev_text, prev_toks = words[-1]
            words[-1] = (prev_text + text, prev_toks + list(toks))
    return words


def merge_punctuation(words: list[dict]) -> list[dict]:
    """openai-whisper's ``merge_punctuations``: opening marks join the word
    after them (scanning right to left), then closing marks join the word
    before them. A word keeps its own timing; the mark's is dropped."""
    words = [dict(w) for w in words]
    i, j = len(words) - 2, len(words) - 1
    while i >= 0:
        prev, following = words[i], words[j]
        if prev["word"].startswith(" ") and prev["word"].strip() in PREPENDED_PUNCTUATION:
            following["word"] = prev["word"] + following["word"]
            prev["word"] = ""
        else:
            j = i
        i -= 1
    i, j = 0, 1
    while j < len(words):
        prev, following = words[i], words[j]
        if not prev["word"].endswith(" ") and following["word"] in APPENDED_PUNCTUATION:
            prev["word"] += following["word"]
            following["word"] = ""
        else:
            i = j
        j += 1
    return [w for w in words if w["word"]]


def word_timings(
    matrix: torch.Tensor, text_tokens: list[int], decode, language: str | None = None,
) -> list[dict]:
    """Words with ``start`` / ``end`` seconds for one window.

    ``matrix`` is ``alignment_matrix`` over the rows that predict
    ``text_tokens`` plus the final row that predicts end-of-text
    (``len(text_tokens) + 1`` rows); ``language`` is the transcript's
    language code.
    """
    if not text_tokens:
        return []
    starts = token_start_frames(matrix) / FRAMES_PER_SECOND  # one per row
    words = split_words(decode, text_tokens, language)
    bounds = np.concatenate([[0], np.cumsum([len(toks) for _, toks in words])])
    timed = [
        {"word": text, "start": float(starts[b0]), "end": float(starts[b1])}
        for (text, _), b0, b1 in zip(words, bounds[:-1], bounds[1:], strict=True)
    ]
    return [{**w, "word": w["word"].strip()} for w in merge_punctuation(timed) if w["word"].strip()]
