"""Parity test for the GPU-tensor rewrite of ``StreamingDacDecoder``.

The rewrite moved the decoder's per-request state from Python lists to device
tensors and the crossfade from CPU to on-device, but the *algorithm* (framing,
watermark, overlap withhold, raised-cosine crossfade) is unchanged. This test
pins that: the tensor-native decoder must produce **byte-identical** int16 PCM
to a faithful copy of the original list-based algorithm (``_LegacyDecoder``
below), for several chunkings of the same frame stream.

To isolate the streaming orchestration from the DAC net (which is unchanged and
optional), ``decode_dac`` is replaced with a deterministic per-frame stub. Both
decoders run through the same stub on CPU, so any divergence is a bug in the
rewrite's buffering/indexing — not device float noise. (The stub is per-frame
independent, so it exercises windowing/withhold logic; boundary-continuity of
the real conv vocoder is out of scope here.)
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mstar.model.zonos2 import vocoder as V

# Small geometry so windows/overlap are easy to reason about.
NC = 3            # n_codebooks
HOP = 4           # hop_length
OVERLAP = 2       # overlap_frames
CBSIZE = 8        # codebook_size
PAD = 9           # audio_pad_id (> codebook_size, like the real config)


def _stub_decode(codes: torch.Tensor, model_type: str = "44khz",
                 codebook_size: int = CBSIZE) -> torch.Tensor:
    """Deterministic ``(B, W, C) -> (B, W * HOP)`` decode.

    Each frame maps to a constant block of ``HOP`` samples in [-1, 1] from its
    mean code value — device-preserving and reproducible, so two decoders fed
    identical code windows return identical audio.
    """
    B, W, _ = codes.shape
    frame_val = (codes.to(torch.float64).mean(dim=-1) / codebook_size)  # (B, W)
    frame_val = (frame_val * 2.0 - 1.0).to(torch.float32)
    return frame_val.unsqueeze(-1).expand(B, W, HOP).reshape(B, W * HOP).contiguous()


class _LegacyDecoder:
    """Verbatim copy of the original list-based algorithm (pre-rewrite), used
    as the golden reference. Calls the module-level ``decode_dac`` /
    ``shear_up`` / ``to_int16_pcm`` so the stub applies to it too."""

    def __init__(self):
        self.n_codebooks = NC
        self.audio_pad_id = PAD
        self.codebook_size = CBSIZE
        self.model_type = "44khz"
        self.overlap_frames = OVERLAP
        self.hop_length = HOP
        self.min_decode_chunk = 1
        self._buffers: dict[str, list[list[int]]] = {}
        self._decoded: dict[str, int] = {}
        self._overlap_tails: dict[str, torch.Tensor] = {}
        self._window_cache: dict[int, torch.Tensor] = {}

    def reset(self, rid=None):
        if rid is None:
            self._buffers.clear(); self._decoded.clear(); self._overlap_tails.clear()
        else:
            self._buffers.pop(rid, None); self._decoded.pop(rid, None)
            self._overlap_tails.pop(rid, None)

    def _fade_in(self, length):
        win = self._window_cache.get(length)
        if win is None:
            if length <= 1:
                win = torch.ones(length, dtype=torch.float32)
            else:
                t = torch.linspace(0.0, torch.pi, length, dtype=torch.float32)
                win = 0.5 * (1.0 - torch.cos(t))
            self._window_cache[length] = win
        return win

    def add_frames(self, rid, frames, is_final):
        buf = self._buffers.setdefault(rid, [])
        self._decoded.setdefault(rid, 0)
        if frames.numel():
            buf.extend(frames.tolist())
        total = len(buf)
        decoded = self._decoded[rid]
        target = max(total - (self.n_codebooks - 1), 0)
        new_decodable = target - decoded
        if is_final:
            should = new_decodable > 0 or rid in self._overlap_tails
        else:
            should = new_decodable >= self.min_decode_chunk
        if not should:
            if is_final:
                self.reset(rid)
            return torch.empty(0, dtype=torch.int16)
        if new_decodable <= 0:
            tail = self._overlap_tails.pop(rid, None)
            out = V.to_int16_pcm(tail) if tail is not None else torch.empty(0, dtype=torch.int16)
            if is_final:
                self.reset(rid)
            return out
        overlap = min(self.overlap_frames, decoded)
        decode_start = decoded - overlap
        raw_end = min(target + self.n_codebooks - 1, total)
        raw = buf[decode_start:raw_end]
        codes = torch.tensor(raw, dtype=torch.int64)
        codes = V.shear_up(codes, self.audio_pad_id)
        out_count = target - decode_start
        codes = codes[:out_count].unsqueeze(0)
        audio = V.decode_dac(codes, self.model_type, self.codebook_size)
        audio = audio[0].detach().float().cpu()
        prev_tail = self._overlap_tails.get(rid)
        if overlap > 0 and prev_tail is not None:
            k = min(overlap * self.hop_length, prev_tail.numel(), audio.numel())
            if k > 0:
                fade = self._fade_in(k)
                audio[:k] = (1.0 - fade) * prev_tail[-k:] + fade * audio[:k]
        if is_final:
            output = audio
            self._overlap_tails.pop(rid, None)
        else:
            tail_samples = min(self.overlap_frames * self.hop_length, audio.numel())
            if tail_samples > 0:
                self._overlap_tails[rid] = audio[-tail_samples:].clone()
                output = audio[:-tail_samples]
            else:
                self._overlap_tails.pop(rid, None)
                output = audio
        self._decoded[rid] = target
        pcm = V.to_int16_pcm(output)
        if is_final:
            self.reset(rid)
        return pcm


def _new_decoder():
    return V.StreamingDacDecoder(
        n_codebooks=NC, audio_pad_id=PAD, codebook_size=CBSIZE,
        overlap_frames=OVERLAP, hop_length=HOP, min_decode_chunk=1,
    )


def _drive(dec, frames, splits, trailing_final_flush=False):
    """Feed ``frames`` to ``dec`` in ``splits`` chunks; return concatenated PCM.

    The last chunk carries ``is_final`` unless ``trailing_final_flush`` is set,
    in which case an extra empty ``is_final`` call flushes the withheld tail.
    """
    outs, idx = [], 0
    for i, n in enumerate(splits):
        chunk = frames[idx:idx + n]; idx += n
        final = (i == len(splits) - 1) and not trailing_final_flush
        outs.append(dec.add_frames("r", chunk, is_final=final).cpu())
    if trailing_final_flush:
        empty = frames[:0]
        outs.append(dec.add_frames("r", empty, is_final=True).cpu())
    return torch.cat(outs) if outs else torch.empty(0, dtype=torch.int16)


@pytest.fixture(autouse=True)
def _stub_dac(monkeypatch):
    monkeypatch.setattr(V, "decode_dac", _stub_decode)


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("chunking", ["ones", "sixteens", "random", "flush"])
def test_rewrite_matches_legacy_bytes(seed, chunking):
    g = torch.Generator().manual_seed(seed)
    T = 40
    frames = torch.randint(0, CBSIZE, (T, NC), generator=g, dtype=torch.int64)

    trailing = False
    if chunking == "ones":
        splits = [1] * T
    elif chunking == "sixteens":
        splits = [16, 16, 8]
    elif chunking == "flush":
        splits = [16, 16, 8]
        trailing = True
    else:  # random chunk sizes summing to T
        splits, rem = [], T
        while rem > 0:
            n = int(torch.randint(1, 7, (1,), generator=g).item())
            n = min(n, rem); splits.append(n); rem -= n

    legacy = _drive(_LegacyDecoder(), frames, splits, trailing_final_flush=trailing)
    new = _drive(_new_decoder(), frames, splits, trailing_final_flush=trailing)

    assert new.dtype == torch.int16
    assert new.numel() == legacy.numel(), (
        f"length mismatch: new={new.numel()} legacy={legacy.numel()}"
    )
    assert torch.equal(new, legacy), "PCM bytes diverge from the legacy algorithm"


def test_produces_nonempty_audio():
    # Guard against a degenerate stub/chunking that would make the parity check
    # vacuous: the stream must actually emit samples.
    frames = torch.randint(0, CBSIZE, (40, NC), dtype=torch.int64)
    out = _drive(_new_decoder(), frames, [16, 16, 8])
    assert out.numel() > 0


# -- batched decode parity --------------------------------------------------
@pytest.mark.parametrize("seed", [0, 3, 11])
def test_batched_matches_per_request(seed):
    """``add_frames_batched`` over N requests must equal N independent
    ``add_frames`` streams, byte-for-byte — including when requests sit at
    different stream positions (ragged windows -> multiple decode groups) and
    finish on different steps (mixed finals)."""
    g = torch.Generator().manual_seed(seed)
    rids = ["a", "b", "c", "d"]
    T = 40
    # Independent frame streams and independent chunkings per request, so they
    # advance out of lock-step (exercises the ragged/multi-group path).
    streams = {r: torch.randint(0, CBSIZE, (T, NC), generator=g, dtype=torch.int64) for r in rids}
    # Per-request chunk schedule; unequal so windows differ across requests.
    schedules = {
        "a": [16, 16, 8],
        "b": [8, 16, 16],
        "c": [16, 8, 16],
        "d": [20, 20],
    }

    # Reference: one decoder per request, driven independently.
    ref = {r: _new_decoder() for r in rids}
    ref_out = {r: [] for r in rids}
    for r in rids:
        idx = 0
        for i, n in enumerate(schedules[r]):
            chunk = streams[r][idx:idx + n]; idx += n
            final = i == len(schedules[r]) - 1
            ref_out[r].append(ref[r].add_frames(r, chunk, is_final=final).cpu())

    # Batched: a single shared decoder, stepped over all requests together.
    dec = _new_decoder()
    bat_out = {r: [] for r in rids}
    n_steps = max(len(s) for s in schedules.values())
    cursors = {r: 0 for r in rids}
    for step in range(n_steps):
        active, frames_list, finals = [], [], []
        for r in rids:
            if step >= len(schedules[r]):
                continue
            n = schedules[r][step]
            chunk = streams[r][cursors[r]:cursors[r] + n]; cursors[r] += n
            active.append(r); frames_list.append(chunk)
            finals.append(step == len(schedules[r]) - 1)
        res = dec.add_frames_batched(active, frames_list, finals)
        for r in active:
            bat_out[r].append(res[r].cpu())

    for r in rids:
        ref_pcm = torch.cat(ref_out[r]) if ref_out[r] else torch.empty(0, dtype=torch.int16)
        bat_pcm = torch.cat(bat_out[r]) if bat_out[r] else torch.empty(0, dtype=torch.int16)
        assert bat_pcm.numel() == ref_pcm.numel(), f"{r}: length differs"
        assert torch.equal(bat_pcm, ref_pcm), f"{r}: batched decode diverges from per-request"


def test_batched_single_group_homogeneous():
    """When all requests advance identically, they collapse into one decode
    group (the common steady state) and still match per-request output."""
    g = torch.Generator().manual_seed(5)
    rids = ["x", "y", "z"]
    streams = {r: torch.randint(0, CBSIZE, (48, NC), generator=g, dtype=torch.int64) for r in rids}
    splits = [16, 16, 16]  # identical schedule -> homogeneous windows each step

    ref = {r: _new_decoder() for r in rids}
    dec = _new_decoder()
    for step, n in enumerate(splits):
        lo, hi = step * 16, step * 16 + n
        final = step == len(splits) - 1
        batched = dec.add_frames_batched(
            rids, [streams[r][lo:hi] for r in rids], [final] * len(rids)
        )
        for r in rids:
            single = ref[r].add_frames(r, streams[r][lo:hi], is_final=final)
            assert torch.equal(batched[r].cpu(), single.cpu()), f"{r} step {step}"
