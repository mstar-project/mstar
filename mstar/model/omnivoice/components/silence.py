"""The reference's silence trim (``omnivoice.utils.audio.remove_silence``),
vectorized and sample-exact.

The reference runs pydub: an RMS window every 10 ms, each a fresh
``AudioSegment`` slice and an ``audioop.rms`` call, which costs about 14 ms of
Python per second of audio and ran inside the code2wav forward, stalling the
worker's loop for every finished request. This computes the same windows from
one prefix sum of squared int16 samples and replays pydub's millisecond
slicing (``int(ms * rate / 1000)`` samples, zero-filled past the end), its
range merging and its edge trim, so the output is the reference's to the
sample; ``test_omnivoice_silence`` checks it against pydub.
"""

import copy
import itertools

import numpy as np

# pydub's dBFS: amplitude relative to the int16 full scale
_FULL_SCALE = 32768.0


class _Segment:
    """An int16 mono signal addressed the way ``pydub.AudioSegment`` slices it.

    It reads ``samples[lo:lo + n]``, zeros past the end of ``samples``, and
    backwards when ``flipped``, so a trimmed or reversed segment shares its
    parent's prefix sums instead of building its own.
    """

    def __init__(self, samples: np.ndarray, rate: int):
        self.samples = samples
        self.rate = rate
        self.per_ms = rate / 1000.0
        squares = samples.astype(np.int64) ** 2
        self.prefix = np.concatenate([[0], np.cumsum(squares)])
        self.lo, self.n, self.flipped = 0, len(samples), False

    def __len__(self) -> int:
        # milliseconds, rounded as pydub rounds them
        return round(1000 * (self.n / self.rate))

    def bounds(self, start_ms, end_ms):
        """Sample bounds and zero-filled length of ``segment[start_ms:end_ms]``."""
        n_ms = len(self)
        start = (np.minimum(start_ms, n_ms) * self.per_ms).astype(np.int64)
        end = (np.minimum(end_ms, n_ms) * self.per_ms).astype(np.int64)
        return start, end

    def reversed_tail(self, start_ms: int) -> "_Segment":
        """``segment[start_ms:].reverse()``, over the same prefix sums."""
        if self.flipped:
            raise ValueError("reversed_tail of a reversed segment")
        start, end = (int(x) for x in self.bounds(np.asarray(start_ms), np.asarray(len(self))))
        tail = copy.copy(self)
        tail.lo, tail.n, tail.flipped = self.lo + start, end - start, True
        return tail

    def _span(self, start, end):
        """Where ``[start, end)`` of the segment, clipped to it, lies in ``samples``."""
        start, end = np.minimum(start, self.n), np.minimum(end, self.n)
        if self.flipped:
            start, end = self.n - end, self.n - start
        return self.lo + start, self.lo + end

    def rms(self, start_ms, end_ms) -> np.ndarray:
        """``audioop.rms`` of each ``[start_ms, end_ms)`` slice (an int, floored)."""
        start, end = self.bounds(np.asarray(start_ms), np.asarray(end_ms))
        lo, hi = self._span(start, end)
        n = len(self.samples)
        squares = (self.prefix[np.minimum(hi, n)] - self.prefix[np.minimum(lo, n)]).astype(np.float64)
        width = (end - start).astype(np.float64)  # past the end pydub pads zeros
        with np.errstate(divide="ignore", invalid="ignore"):
            rms = np.floor(np.sqrt(squares / width))
        return np.where(width > 0, rms, 0.0)

    def slice(self, start_ms: int, end_ms: int) -> np.ndarray:
        start, end = (int(x) for x in self.bounds(np.asarray(start_ms), np.asarray(end_ms)))
        lo, hi = (int(x) for x in self._span(start, end))
        data = self.samples[lo:hi]
        out = np.zeros(max(end - start, 0), dtype=self.samples.dtype)
        if self.flipped:
            # the segment's [start, min(end, n)) is `samples[lo:hi]` backwards;
            # what lies past the samples' end are its leading zeros
            last = min(end, self.n) - start
            out[last - len(data):last] = data[::-1]
        else:
            out[:len(data)] = data
        return out


def _silent_ranges(seg: _Segment, min_silence_len: int, threshold: float, seek_step: int) -> list[list[int]]:
    """``pydub.silence.detect_silence``."""
    seg_len = len(seg)
    if seg_len < min_silence_len:
        return []
    last = seg_len - min_silence_len
    starts = list(range(0, last + 1, seek_step))
    if last % seek_step:
        starts.append(last)
    starts = np.asarray(starts, dtype=np.int64)
    silent = starts[seg.rms(starts, starts + min_silence_len) <= threshold].tolist()
    if not silent:
        return []
    ranges = []
    prev = range_start = silent[0]
    for s in silent[1:]:
        if s != prev + seek_step and s > prev + min_silence_len:
            ranges.append([range_start, prev + min_silence_len])
            range_start = s
        prev = s
    ranges.append([range_start, prev + min_silence_len])
    return ranges


def _nonsilent_ranges(seg: _Segment, min_silence_len: int, threshold: float, seek_step: int) -> list[list[int]]:
    """``pydub.silence.detect_nonsilent``."""
    silent = _silent_ranges(seg, min_silence_len, threshold, seek_step)
    seg_len = len(seg)
    if not silent:
        return [[0, seg_len]]
    if silent[0][0] == 0 and silent[0][1] == seg_len:
        return []
    ranges, prev_end = [], 0
    for start, end in silent:
        ranges.append([prev_end, start])
        prev_end = end
    if end != seg_len:
        ranges.append([prev_end, seg_len])
    if ranges[0] == [0, 0]:
        ranges.pop(0)
    return ranges


def _split_on_silence(seg: _Segment, min_silence_len: int, threshold: float, keep: int, seek_step: int) -> np.ndarray:
    """``pydub.silence.split_on_silence``, its pieces joined back together."""
    ranges = [[s - keep, e + keep] for s, e in _nonsilent_ranges(seg, min_silence_len, threshold, seek_step)]
    for a, b in itertools.pairwise(ranges):
        if b[0] < a[1]:
            a[1] = (a[1] + b[0]) // 2
            b[0] = a[1]
    n_ms = len(seg)
    pieces = [seg.slice(max(s, 0), min(e, n_ms)) for s, e in ranges]
    return np.concatenate(pieces) if pieces else seg.samples[:0]


def _leading_silence(seg: _Segment, threshold_db: float, chunk: int = 10) -> int:
    """``pydub.silence.detect_leading_silence``."""
    n_ms = len(seg)
    starts = np.arange(0, n_ms, chunk, dtype=np.int64)
    rms = seg.rms(starts, starts + chunk)
    with np.errstate(divide="ignore"):
        db = np.where(rms > 0, 20 * np.log10(rms / _FULL_SCALE), -np.inf)
    loud = np.flatnonzero(db >= threshold_db)
    trim = int(starts[loud[0]]) if len(loud) else len(starts) * chunk
    return min(trim, n_ms)


def remove_silence(
    audio: np.ndarray, sampling_rate: int, mid_sil: int = 300, lead_sil: int = 100, trail_sil: int = 300,
) -> np.ndarray:
    """``omnivoice.utils.audio.remove_silence`` for mono ``(1, T)`` float audio."""
    if audio.shape[0] != 1:
        raise ValueError(f"expected mono (1, T) audio, got {audio.shape}")
    samples = (audio[0] * 32768.0).clip(-32768, 32767).astype(np.int16)
    threshold_db = -50.0
    threshold = 10 ** (threshold_db / 20) * _FULL_SCALE
    if mid_sil > 0:
        samples = _split_on_silence(_Segment(samples, sampling_rate), mid_sil, threshold, mid_sil, 10)

    seg = _Segment(samples, sampling_rate)
    # the trailing trim is the leading one of the lead-trimmed signal reversed
    tail = seg.reversed_tail(max(0, _leading_silence(seg, threshold_db) - lead_sil))
    start = max(0, _leading_silence(tail, threshold_db) - trail_sil)
    samples = tail.slice(start, len(tail))[::-1]
    return (samples.astype(np.float32) / 32768.0)[np.newaxis, :]
