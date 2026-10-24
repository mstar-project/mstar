"""The vectorized silence trim against the reference's pydub one: sample-exact."""

import numpy as np
import pytest

from mstar.model.omnivoice.components.silence import remove_silence

reference = pytest.importorskip("omnivoice.utils.audio")

RATE = 24000


def _speech(rng, seconds):
    """Bursts of tone and noise separated by silences of varied length."""
    out = []
    while sum(len(x) for x in out) < seconds * RATE:
        n = int(rng.uniform(0.05, 0.9) * RATE)
        if rng.random() < 0.5:
            t = np.arange(n) / RATE
            out.append((rng.uniform(0.05, 0.6) * np.sin(2 * np.pi * rng.uniform(80, 400) * t)).astype(np.float32))
        else:
            level = rng.choice([0.0, 1e-5, 2e-3, 3.2e-3, 1e-2])
            out.append((level * rng.standard_normal(n)).astype(np.float32))
    return np.concatenate(out)[None, : int(seconds * RATE) + int(rng.integers(0, 24))]


@pytest.mark.parametrize("seed", range(40))
def test_matches_pydub(seed):
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
])
def test_edge_cases_match_pydub(audio):
    want = reference.remove_silence(audio, RATE, mid_sil=500, lead_sil=100, trail_sil=100)
    got = remove_silence(audio, RATE, mid_sil=500, lead_sil=100, trail_sil=100)
    assert np.array_equal(got, want)
