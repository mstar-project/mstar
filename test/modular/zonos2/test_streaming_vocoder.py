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
  * the final flush decodes the held-back shear frames and the withheld tail;
  * eoa ends the audio at its aligned frame, unless ``ignore_eos``;
  * the crossfade turns a hard boundary jump into a smooth ramp.
"""
from __future__ import annotations

import pytest

import mstar.model.zonos2.vocoder as V
from mstar.model.zonos2.vocoder import StreamingDacDecoder

torch = pytest.importorskip("torch")

HOP = 8            # tiny hop for a fast test
C = 9              # n_codebooks
CHUNK = 16         # frames per transport chunk (FixedChunkPolicy delivery size)
EOA = 1024


def _g(code):
    """Clean per-code audio value in [-1, 1] — the fake DAC's codes->wave map."""
    return 0.9 * torch.sin(code.to(torch.float32) * 0.1)


def _to_i16(v) -> int:
    t = torch.as_tensor(v, dtype=torch.float32)
    return int(t.clamp(-1, 1).mul(32767).to(torch.int16))


def _stream(dec, codes, value_fn, monkeypatch, chunk=CHUNK, frames_all=None,
            ignore_eos=False):
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
    if frames_all is None:
        frames_all = torch.tensor([[c] * C for c in codes], dtype=torch.int64)
    out, i = [], 0
    while i < n:
        j = min(i + chunk, n)
        out.append(dec.add_frames("r", frames_all[i:j], is_final=(j >= n),
                                  ignore_eos=ignore_eos))
        i = j
    return torch.cat(out) if out else torch.empty(0, dtype=torch.int16)


def test_streaming_reconstructs_frames_in_order(monkeypatch):
    dec = StreamingDacDecoder(n_codebooks=C, overlap_frames=4, hop_length=HOP,
                              min_decode_chunk=1)
    N = 100
    out = _stream(dec, list(range(N)),
                  lambda col0, _: _g(col0).repeat_interleave(HOP), monkeypatch).tolist()

    expected = []
    for i in range(N):  # the final call decodes the shear tail too
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

    # The flush emits the withheld tail plus the C - 1 frames held for shear context.
    assert flush.numel() == (4 + C - 1) * HOP
    assert before + flush.numel() == N * HOP   # every frame once, no dupes


def _eoa_frames(N, step, col):
    """Code ``i`` in frame ``i``; eoa in ``col`` at ``step`` and everywhere after."""
    frames = torch.tensor([[c] * C for c in range(N)], dtype=torch.int64)
    frames[step, col] = EOA
    frames[step + 1:] = EOA
    return frames


@pytest.mark.parametrize("step, col, kept", [
    (40, 0, 40), (40, 8, 32), (40, 3, 37), (17, 8, 9), (5, 8, 0),
])
def test_eoa_ends_audio_at_its_aligned_frame(monkeypatch, step, col, kept):
    N = 60
    dec = StreamingDacDecoder(n_codebooks=C, overlap_frames=4, hop_length=HOP,
                              min_decode_chunk=1, eoa_id=EOA)
    out = _stream(dec, list(range(N)), lambda col0, _: _g(col0).repeat_interleave(HOP),
                  monkeypatch, frames_all=_eoa_frames(N, step, col))
    expected = []
    for i in range(kept):
        expected += [_to_i16(_g(torch.tensor(i)))] * HOP
    assert out.tolist() == expected


def test_ignore_eos_keeps_every_frame(monkeypatch):
    N = 60
    dec = StreamingDacDecoder(n_codebooks=C, overlap_frames=4, hop_length=HOP,
                              min_decode_chunk=1, eoa_id=EOA)
    out = _stream(dec, list(range(N)), lambda col0, _: _g(col0).repeat_interleave(HOP),
                  monkeypatch, frames_all=_eoa_frames(N, 40, 0), ignore_eos=True)
    assert out.numel() == N * HOP


def test_eos_state_is_per_request(monkeypatch):
    monkeypatch.setattr(
        V, "decode_dac",
        lambda ci, *a, **k: _g(ci[0, :, 0]).repeat_interleave(HOP).unsqueeze(0),
    )
    dec = StreamingDacDecoder(n_codebooks=C, overlap_frames=4, hop_length=HOP,
                              min_decode_chunk=1, eoa_id=EOA)
    a = dec.add_frames("a", _eoa_frames(30, 10, 0), is_final=True)
    b = dec.add_frames("b", _eoa_frames(30, 29, 0)[:20], is_final=True)
    assert a.numel() == 10 * HOP
    assert b.numel() == 20 * HOP
    assert not dec._eos_frames


def test_one_frame_budget_still_gives_audio(monkeypatch):
    monkeypatch.setattr(
        V, "decode_dac",
        lambda ci, *a, **k: _g(ci[0, :, 0]).repeat_interleave(HOP).unsqueeze(0),
    )
    dec = StreamingDacDecoder(n_codebooks=C, hop_length=HOP, eoa_id=EOA)
    out = dec.add_frames("r", torch.full((1, C), 3, dtype=torch.int64), is_final=True)
    assert out.numel() == HOP


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
