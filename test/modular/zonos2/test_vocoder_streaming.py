"""Streaming and batched parity for ``StreamingDacDecoder``.

``decode_dac`` is a per-frame stub, so a stream must equal one decode of the
whole sequence. The oracle shears and trims on its own, sharing no helper with
the decoder.
"""
from __future__ import annotations

import pytest

from mstar.model.zonos2 import vocoder as V

torch = pytest.importorskip("torch")

NC = 3            # n_codebooks
HOP = 4           # hop_length
OVERLAP = 2       # overlap_frames
CBSIZE = 8        # codebook_size
EOA = CBSIZE      # eoa_id, one past the codebook as in the real config
PAD = 9           # audio_pad_id


def _stub_decode(codes, *a, **k):
    """``(B, W, C) -> (B, W * HOP)``: each frame becomes a constant block."""
    B, W, _ = codes.shape
    val = codes.clamp(0, CBSIZE - 1).to(torch.float64).mean(dim=-1) / CBSIZE
    val = (val * 2.0 - 1.0).to(torch.float32)
    return val.unsqueeze(-1).expand(B, W, HOP).reshape(B, W * HOP).contiguous()


@pytest.fixture(autouse=True)
def _stub_dac(monkeypatch):
    monkeypatch.setattr(V, "decode_dac", _stub_decode)


def _eos_frame(frames, ignore_eos):
    """The reference rule: the first eoa row, shifted back by its last eoa column."""
    if ignore_eos:
        return None
    for step, row in enumerate(frames.tolist()):
        cols = [j for j, c in enumerate(row) if c == EOA]
        if cols:
            return max(0, step - cols[-1])
    return None


def _oneshot(frames, ignore_eos=False):
    """Undo the shear for the whole stream, trim at EOS, decode once."""
    T = frames.shape[0]
    codes = torch.full_like(frames, PAD)
    for j in range(NC):
        codes[: T - j, j] = frames[j:, j]
    end = T
    eos = _eos_frame(frames, ignore_eos)
    if eos is not None:
        end = min(end, eos)
    audio = _stub_decode(codes[:end].unsqueeze(0))[0]
    return (audio.clamp(-1, 1) * 32767.0).to(torch.int16)


def _assert_pcm_close(got, want):
    # The crossfade of two equal blocks may round one LSB differently.
    assert got.numel() == want.numel(), f"length {got.numel()} != {want.numel()}"
    if got.numel():
        assert (got.to(torch.int32) - want.to(torch.int32)).abs().max() <= 1


def _new_decoder():
    return V.StreamingDacDecoder(
        n_codebooks=NC, audio_pad_id=PAD, codebook_size=CBSIZE,
        overlap_frames=OVERLAP, hop_length=HOP, min_decode_chunk=1, eoa_id=EOA,
    )


def _frames(seed, T=40, eos_at=None):
    """Random codes; ``eos_at=(step, col)`` puts eoa there and in every later frame."""
    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, CBSIZE, (T, NC), generator=g, dtype=torch.int64)
    if eos_at is not None:
        step, col = eos_at
        frames[step, col] = EOA
        frames[step + 1:, :] = EOA
    return frames


def _splits(kind, T, g):
    if kind == "ones":
        return [1] * T
    if kind == "sixteens":
        return [16] * (T // 16) + ([T % 16] if T % 16 else [])
    out, rem = [], T
    while rem > 0:
        n = min(int(torch.randint(1, 7, (1,), generator=g)), rem)
        out.append(n)
        rem -= n
    return out


def _drive(dec, frames, splits, rid="r", trailing_flush=False, ignore_eos=False):
    outs, idx = [], 0
    for i, n in enumerate(splits):
        final = i == len(splits) - 1 and not trailing_flush
        outs.append(dec.add_frames(rid, frames[idx:idx + n], final, ignore_eos))
        idx += n
    if trailing_flush:
        outs.append(dec.add_frames(rid, frames[:0], True, ignore_eos))
    return torch.cat(outs)


EOS_CASES = [None, (20, 0), (20, 2), (1, 2), (35, 1)]


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("chunking", ["ones", "sixteens", "random"])
@pytest.mark.parametrize("trailing_flush", [False, True])
@pytest.mark.parametrize("eos_at", EOS_CASES)
@pytest.mark.parametrize("ignore_eos", [False, True])
def test_stream_matches_oneshot(seed, chunking, trailing_flush, eos_at, ignore_eos):
    frames = _frames(seed, eos_at=eos_at)
    splits = _splits(chunking, frames.shape[0], torch.Generator().manual_seed(seed))
    got = _drive(_new_decoder(), frames, splits, trailing_flush=trailing_flush,
                 ignore_eos=ignore_eos)
    _assert_pcm_close(got, _oneshot(frames, ignore_eos))


def test_oracle_is_not_vacuous():
    assert _oneshot(_frames(0)).numel() == 40 * HOP
    assert _oneshot(_frames(0, eos_at=(20, 2))).numel() == 18 * HOP
    assert _oneshot(_frames(0, eos_at=(1, 2))).numel() == 0


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_batched_matches_oneshot(seed):
    """Ragged schedules, mixed finals and mixed EOS in one shared decoder."""
    streams = {
        "a": (_frames(seed), [16, 16, 8], False),
        "b": (_frames(seed + 1, eos_at=(12, 1)), [8, 16, 16], False),
        "c": (_frames(seed + 2, eos_at=(25, 2)), [16, 8, 16], True),
        "d": (_frames(seed + 3, eos_at=(30, 0)), [20, 20], False),
    }
    dec = _new_decoder()
    out = {r: [] for r in streams}
    cursor = dict.fromkeys(streams, 0)
    for step in range(3):
        rids, frames, finals, ignore = [], [], [], []
        for r, (f, sched, ign) in streams.items():
            if step >= len(sched):
                continue
            n = sched[step]
            rids.append(r)
            frames.append(f[cursor[r]:cursor[r] + n])
            finals.append(step == len(sched) - 1)
            ignore.append(ign)
            cursor[r] += n
        res = dec.add_frames_batched(rids, frames, finals, ignore)
        for r in rids:
            out[r].append(res[r])
    for r, (f, _, ign) in streams.items():
        _assert_pcm_close(torch.cat(out[r]), _oneshot(f, ign))


def test_batched_single_group_homogeneous():
    streams = {r: _frames(i, T=48) for i, r in enumerate("xyz")}
    dec = _new_decoder()
    out = {r: [] for r in streams}
    for step in range(3):
        lo = step * 16
        res = dec.add_frames_batched(
            list(streams), [f[lo:lo + 16] for f in streams.values()], [step == 2] * 3,
        )
        for r in streams:
            out[r].append(res[r])
    for r, f in streams.items():
        _assert_pcm_close(torch.cat(out[r]), _oneshot(f))
