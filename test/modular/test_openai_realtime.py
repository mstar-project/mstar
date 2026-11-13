"""The ``/v1/realtime`` pieces still on the Python side: the strict base64
reader the Dynamo realtime bridge decodes appends with, and the streaming
resampler. mstar-server's ``/v1/realtime`` is tested in test/rust/."""

import numpy as np
import pytest

from mstar.api_server.media_io import StreamResampler
from mstar.api_server.openai.serving_realtime import decode_base64


def test_whitespace_in_base64_costs_memory_linear_in_the_payload():
    """A frame of short base64 pieces between spaces is decoded without a
    Python object per piece, so a client cannot turn one large frame into
    hundreds of megabytes; non-ASCII is refused like any non-base64."""
    import tracemalloc

    payload = "AAAA " * 200_000  # 1 MB
    tracemalloc.start()
    audio = decode_base64(payload)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert audio == bytes(600_000)
    assert peak < 5_000_000, peak
    with pytest.raises(ValueError):
        decode_base64("AAAA AAAA")
    with pytest.raises(ValueError):
        decode_base64("!!!!")


def test_stream_resampler_is_split_invariant_and_keeps_time():
    rng = np.random.default_rng(0)
    n = 24000 * 2 + 7
    x = (np.sin(2 * np.pi * 1000 * np.arange(n) / 24000) * 0.5).astype(np.float32)
    r = StreamResampler(24000, 16000)
    whole = np.concatenate([r.push(x), r.flush()])
    pieces, i = [], 0
    while i < n:
        step = int(rng.integers(1, 700))
        pieces.append(r.push(x[i:i + step]))
        i += step
    pieces.append(r.flush())
    assert whole.size == -(-n * 2 // 3)
    assert np.array_equal(np.concatenate(pieces), whole)
    exact = np.sin(2 * np.pi * 1000 * np.arange(whole.size) / 16000) * 0.5
    assert np.abs(whole[50:-50] - exact[50:-50]).max() < 1e-4
    # above the 8 kHz Nyquist of the output: filtered, not folded back
    for hz in (8250, 10000):  # just past the Nyquist folds to just under it
        alias = np.sin(2 * np.pi * hz * np.arange(n) / 24000).astype(np.float32)
        out = np.concatenate([r.push(alias), r.flush()])
        assert np.sqrt((out[100:-100] ** 2).mean()) < 1e-3, hz
