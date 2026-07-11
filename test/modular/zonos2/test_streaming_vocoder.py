"""Streaming-decode regression tests for the Zonos2 DAC vocoder.

DAC's decoder is convolutional, so decoding each streamed chunk of frames in
isolation and hard-concatenating the waveforms puts a discontinuity click at
every chunk boundary (an ~86 Hz buzz for 16-frame / 512-hop chunks). The fix
in ``StreamingDacDecoder`` re-decodes ``overlap_frames`` of already-emitted
frames as left context and overlap-add crossfades the seam with the previous
chunk's withheld tail, flushing that tail on the final call.

These tests monkeypatch ``decode_dac`` with a deterministic fake (integer
codes -> float waveform, DAC's role), so they run pure-CPU with no GPU and
no ``descript-audio-codec`` install. They check:
  * frame bookkeeping — each real output frame emitted exactly once, in order;
  * the final-flush emits exactly the withheld overlap tail (no double-count);
  * the crossfade turns a hard boundary jump into a smooth ramp.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import mstar.model.zonos2.vocoder as V
from mstar.model.zonos2.vocoder import StreamingDacDecoder

HOP = 8            # tiny hop for a fast test
C = 9              # n_codebooks
CHUNK = 16         # frames per transport chunk (FixedChunkPolicy delivery size)


def _g(code):
    """Clean per-code audio value in [-1, 1] — the fake DAC's codes->wave map."""
    return 0.9 * torch.sin(code.to(torch.float32) * 0.1)


def _to_i16(v) -> int:
    t = torch.as_tensor(v, dtype=torch.float32)
    return int(t.clamp(-1, 1).mul(32767).to(torch.int16))


def _stream(dec, codes, value_fn, monkeypatch, chunk=CHUNK):
    """Feed integer ``codes`` (one per frame) in ``chunk``-frame batches.

    ``value_fn(col0, call_idx)`` maps a decode's code column to a float
    waveform ``(T*HOP,)``. Returns the concatenated int16 output.
    """
    n = len(codes)
    call = {"i": 0}

    def fake_decode_dac(codes_in, *a, **k):
        aud = value_fn(codes_in[0, :, 0], call["i"])
        call["i"] += 1
        return aud.unsqueeze(0)

    monkeypatch.setattr(V, "decode_dac", fake_decode_dac)
    frames_all = torch.tensor([[c] * C for c in codes], dtype=torch.int64)
    out, i = [], 0
    while i < n:
        j = min(i + chunk, n)
        out.append(dec.add_frames("r", frames_all[i:j], is_final=(j >= n)))
        i = j
    return torch.cat(out) if out else torch.empty(0, dtype=torch.int16)


def test_streaming_reconstructs_frames_in_order(monkeypatch):
    dec = StreamingDacDecoder(n_codebooks=C, overlap_frames=4, hop_length=HOP,
                              min_decode_chunk=1)
    N = 100
    out = _stream(dec, list(range(N)),
                  lambda col0, _: _g(col0).repeat_interleave(HOP), monkeypatch).tolist()

    target = N - (C - 1)  # trailing shear-alignment frames carry no audio
    expected = []
    for i in range(target):
        expected += [_to_i16(_g(torch.tensor(i)))] * HOP
    assert out == expected


def test_final_flush_emits_withheld_tail_once(monkeypatch):
    dec = StreamingDacDecoder(n_codebooks=C, overlap_frames=4, hop_length=HOP,
                              min_decode_chunk=1)
    N = 64
    frames_all = torch.tensor([[c] * C for c in range(N)], dtype=torch.int64)
    monkeypatch.setattr(
        V, "decode_dac",
        lambda ci, *a, **k: _g(ci[0, :, 0]).repeat_interleave(HOP).unsqueeze(0),
    )

    streamed, i = [], 0
    while i < N:
        j = min(i + CHUNK, N)
        streamed.append(dec.add_frames("r", frames_all[i:j], is_final=False))
        i = j
    before = int(sum(t.numel() for t in streamed))
    flush = dec.add_frames("r", torch.empty(0, C, dtype=torch.int64), is_final=True)

    target = N - (C - 1)
    assert flush.numel() == 4 * HOP                 # exactly the withheld overlap tail
    assert before + flush.numel() == target * HOP   # every real frame once, no dupes


def test_crossfade_smooths_boundary_discontinuity(monkeypatch):
    """Per-call DC bias models independent decodes disagreeing at the seam."""
    N = 96
    codes = [0] * N  # flat clean signal -> only the bias creates jumps

    def biased(col0, call_idx):
        bias = 0.3 * (call_idx % 2)   # alternating per decode call -> hard seam
        return (_g(col0) + bias).repeat_interleave(HOP)

    off = _stream(StreamingDacDecoder(n_codebooks=C, overlap_frames=0,
                                      hop_length=HOP, min_decode_chunk=1),
                  codes, biased, monkeypatch).to(torch.float32)
    on = _stream(StreamingDacDecoder(n_codebooks=C, overlap_frames=4,
                                     hop_length=HOP, min_decode_chunk=1),
                 codes, biased, monkeypatch).to(torch.float32)

    jump_off = off.diff().abs().max().item()
    jump_on = on.diff().abs().max().item()
    assert jump_off > 4000                 # ~9830: a hard int16 click at each seam
    assert jump_on < jump_off * 0.25       # crossfade spreads it into a ramp
