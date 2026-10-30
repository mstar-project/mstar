"""The vectorized silence trim against the reference's pydub one: sample-exact."""

import numpy as np
import pytest

from mstar.model.omnivoice.components import silence
from mstar.model.omnivoice.components.silence import remove_silence

RATE = 24000


@pytest.fixture
def reference():
    return pytest.importorskip("omnivoice.utils.audio")


def _speech(rng, seconds, rate=RATE):
    """Bursts of tone and noise separated by silences of varied length."""
    out = []
    while sum(len(x) for x in out) < seconds * rate:
        n = int(rng.uniform(0.05, 0.9) * rate)
        if rng.random() < 0.5:
            t = np.arange(n) / rate
            out.append((rng.uniform(0.05, 0.6) * np.sin(2 * np.pi * rng.uniform(80, 400) * t)).astype(np.float32))
        else:
            level = rng.choice([0.0, 1e-5, 2e-3, 3.2e-3, 1e-2])
            out.append((level * rng.standard_normal(n)).astype(np.float32))
    return np.concatenate(out)[None, : int(seconds * rate) + int(rng.integers(0, 24))]


@pytest.mark.parametrize("seed", range(40))
def test_matches_pydub(seed, reference):
    rng = np.random.default_rng(seed)
    audio = _speech(rng, rng.uniform(0.05, 12.0))
    for mid, lead, trail in ((500, 100, 100), (300, 100, 300), (0, 100, 300)):
        want = reference.remove_silence(audio, RATE, mid_sil=mid, lead_sil=lead, trail_sil=trail)
        got = remove_silence(audio, RATE, mid_sil=mid, lead_sil=lead, trail_sil=trail)
        assert got.dtype == want.dtype and np.array_equal(got, want), (seed, mid, got.shape, want.shape)


@pytest.mark.parametrize("audio", [
    np.zeros((1, 0), np.float32),
    np.zeros((1, 7), np.float32),
    np.zeros((1, RATE), np.float32),
    np.full((1, 13), 0.5, np.float32),
    # full scale both ways between silences: the int16 conversion clips
    np.concatenate([np.zeros(RATE // 5), np.ones(RATE // 10), -np.ones(RATE // 10), np.zeros(RATE // 2)])
    .astype(np.float32)[None],
])
def test_edge_cases_match_pydub(audio, reference):
    want = reference.remove_silence(audio, RATE, mid_sil=500, lead_sil=100, trail_sil=100)
    got = remove_silence(audio, RATE, mid_sil=500, lead_sil=100, trail_sil=100)
    assert np.array_equal(got, want)


def _trim_over_a_reversed_copy(audio, rate, mid_sil, lead_sil, trail_sil):
    """The edge trim as pydub runs it: the trailing trim reverses the lead-trimmed
    samples and measures them afresh."""
    samples = (audio[0] * 32768.0).clip(-32768, 32767).astype(np.int16)
    threshold = 10 ** (-50.0 / 20) * 32768.0
    if mid_sil > 0:
        samples = silence._split_on_silence(silence._Segment(samples, rate), mid_sil, threshold, mid_sil, 10)
    seg = silence._Segment(samples, rate)
    samples = seg.slice(max(0, silence._leading_silence(seg, -50.0) - lead_sil), len(seg))
    seg = silence._Segment(samples[::-1].copy(), rate)
    samples = seg.slice(max(0, silence._leading_silence(seg, -50.0) - trail_sil), len(seg))[::-1]
    return (samples.astype(np.float32) / 32768.0)[np.newaxis, :]


@pytest.mark.parametrize("rate", [24000, 22050, 44100, 16000])
@pytest.mark.parametrize("seed", range(25))
def test_trailing_trim_reads_the_shared_prefix_sums(seed, rate):
    """Guards the flipped-bounds path (``_Segment.reversed_tail``) where the pydub
    comparison skips, as on CI: the trailing trim reads the lead-trimmed segment
    backwards over its prefix sums, sample-exact against measuring a reversed
    copy, including where a rate's fractional samples per millisecond pad or
    drop the edge."""
    rng = np.random.default_rng(seed)
    audio = _speech(rng, rng.uniform(0.0, 4.0), rate)
    if seed % 5 == 0:  # loud to both edges
        audio = (0.3 * rng.standard_normal(audio.shape)).astype(np.float32)
    for args in ((500, 100, 100), (0, 100, 300), (0, 0, 0), (0, 500, 0)):
        want = _trim_over_a_reversed_copy(audio, rate, *args)
        got = remove_silence(audio, rate, *args)
        assert got.dtype == want.dtype and np.array_equal(got, want), (seed, rate, args)
