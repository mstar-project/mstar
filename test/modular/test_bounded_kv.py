"""Bounded (sink + window) KV: the host layout against a brute-force stream,
token2wav's three cache rules expressed as sink + window, and the kernels
against torch."""

import random

import pytest
import torch

from mstar.engine.resources.kv.bounded.config import SinkWindow
from mstar.engine.resources.kv.bounded.layout import (
    READ_FROM_SOURCE,
    flatten_row,
    row_layout,
)


class SlotSim:
    """Applies layouts to a list-valued slot and reads back token ids."""

    def __init__(self, source: list, policy: SinkWindow, sink_capacity: int, reverse: bool = False):
        self.reverse = reverse
        self.source = source
        self.policy = policy
        self.sink_capacity = sink_capacity
        self.slot = [None] * (sink_capacity + policy.window)
        self.written = 0

    def step(self, fresh: list, write: bool = True) -> list:
        """The ids a step attends to (retained, then its own), then its writes."""
        span = len(fresh) if write else 0
        layout = row_layout(len(self.source), self.written, span, self.policy, self.sink_capacity,
                            self.reverse)
        keys = []
        for from_source, (start, count) in zip(READ_FROM_SOURCE, layout.reads, strict=True):
            buf = self.source if from_source else self.slot
            keys += buf[start:start + count]
        assert None not in keys, "read an unwritten slot entry"
        for offset, start, count in layout.writes:
            if self.reverse:
                self.slot[start:start + count] = fresh[offset - count + 1:offset + 1][::-1]
            else:
                self.slot[start:start + count] = fresh[offset:offset + count]
        self.written += span
        return keys + fresh


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_layout_matches_stream(seed, reverse):
    rng = random.Random(seed)
    source_len = rng.randrange(0, 40)
    policy = SinkWindow(rng.randrange(0, 50), rng.randrange(0, 30))
    sink_capacity = max(0, policy.sink - source_len)
    source = [("s", i) for i in range(source_len)]
    sim = SlotSim(source, policy, sink_capacity, reverse)
    stream = list(source)
    for step in range(12):
        fresh = [("g", len(stream) - source_len + j) for j in range(rng.randrange(1, 20))]
        n = len(stream)
        retained = stream[:min(policy.sink, n)] + stream[max(policy.sink, n - policy.window):n]
        assert sim.step(fresh) == retained + fresh, step
        stream += fresh[::-1] if reverse else fresh


# token2wav: a voice of P tokens (2P frames at 50 Hz); 28-code windows advance by 25
P, KEEP = 60, 100


def _upstream(windows: int):
    """Upstream's caches as id lists, window by window (``Token2WavState.truncate``):
    returns what each window attends to, per cache."""
    p2, limit = 2 * P, 2 * P + KEEP
    voice1 = [("v1", i) for i in range(P)]
    voice2 = [("v2", i) for i in range(p2)]
    voiced = [("vd", i) for i in range(p2)]
    enc1, enc2, dit = list(voice1), list(voice2), list(voiced)
    seen = []
    for w in range(windows):
        new1 = [("g1", 25 * w + j) for j in range(25)]
        new2 = [("g2", 50 * w + j) for j in range(50)]
        newd = [("gd", 50 * w + j) for j in range(50)]
        seen.append({"enc1": enc1 + new1, "enc2": enc2 + new2, "dit": newd + dit})
        enc1, enc2, dit = enc1 + new1, enc2 + new2, newd + dit
        if len(dit) > limit:
            dit = dit[:p2] + dit[len(dit) - KEEP:]
        if len(enc2) > limit:
            n2, n1 = len(enc2), len(enc1)
            kept = list(range(p2)) + list(range(n2 - KEEP, n2))
            enc1 = [enc1[i % n1] for i in kept[:limit // 2]]
            enc2 = enc2[:p2] + enc2[n2 - KEEP:]
    return seen


def test_token2wav_caches_are_sink_window():
    p2 = 2 * P
    windows = 9
    upstream = _upstream(windows)
    voice1 = [("v1", i) for i in range(P)]
    voice2 = [("v2", i) for i in range(p2)]
    voiced = [("vd", i) for i in range(p2)]
    # The DiT's attention is order-free, and upstream keeps the first 2P of [newest
    # window, ..., oldest, voice] (each window in its own order): sink + window
    # over a stream of the voice's tail, then the rest of it and each window
    # last-first.
    dit_source = voiced[p2 - KEEP:] + voiced[:p2 - KEEP][::-1]
    caches = {
        "enc1": SlotSim(voice1, SinkWindow(P + KEEP // 2, 0), KEEP // 2),
        "enc2": SlotSim(voice2, SinkWindow(p2, KEEP), 0),
        "dit": SlotSim(dit_source, SinkWindow(KEEP, p2), 0, reverse=True),
    }
    for w in range(windows):
        fresh = {
            "enc1": [("g1", 25 * w + j) for j in range(25)],
            "enc2": [("g2", 50 * w + j) for j in range(50)],
            "dit": [("gd", 50 * w + j) for j in range(50)],
        }
        for name, sim in caches.items():
            got = sim.step(fresh[name])
            want = upstream[w][name]
            if name == "dit":
                assert sorted(got) == sorted(want), (w, name)
            else:
                # the conformer's attention is positional: same order
                assert got == want, (w, name)


def _reference(q, k, v, cache, source, rows_layout, slots):
    """Torch: attention over each row's ranges and its own keys, then the writes."""
    n, h, t, d = q.shape
    rows = cache.shape[1]
    out = torch.empty(n, t, h, d, dtype=q.dtype, device=q.device)
    new_cache = cache.clone()
    for i in range(n):
        b, c = divmod(i, rows)
        layout, slot = rows_layout[b], slots[b]
        parts = []
        for from_source, (start, count) in zip(READ_FROM_SOURCE, layout.reads, strict=True):
            buf = source[c] if from_source else cache[slot, c]
            parts.append(buf[:, start:start + count])
        kv = torch.cat(parts, dim=1)
        keys = torch.cat([k[i], kv[..., :d]], dim=1)
        values = torch.cat([v[i], kv[..., d:]], dim=1)
        att = torch.softmax(q[i] @ keys.transpose(-1, -2) * d ** -0.5, dim=-1)
        out[i] = (att @ values).transpose(0, 1)
        for offset, start, count in layout.writes:
            new_cache[slot, c, :, start:start + count, :d] = k[i][:, offset:offset + count]
            new_cache[slot, c, :, start:start + count, d:] = v[i][:, offset:offset + count]
    return out, new_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")
@pytest.mark.parametrize("with_source", [True, False])
def test_kernels_match_torch(with_source):
    from mstar.engine.resources.kv.bounded.kernels import bounded_attention, bounded_store

    torch.manual_seed(0)
    rng = random.Random(0)
    dev = "cuda"
    rows, h, d, t = 2, 4, 64, 50
    source_len = 120 if with_source else 0
    policy = SinkWindow(30 if with_source else 0, 90)
    sink_capacity = max(0, policy.sink - source_len)
    slots_n = 6
    cache = torch.randn(slots_n, rows, h, sink_capacity + policy.window, 2 * d, device=dev)
    source = torch.randn(rows, h, max(source_len, 1), 2 * d, device=dev) if with_source else None
    # rows at different positions, including a read-only one
    written = [0, 50, 100, 170, 260]
    spans = [t, t, t, t, 0]
    slots = [3, 0, 5, 1, 2]
    layouts = [row_layout(source_len, w, s, policy, sink_capacity) for w, s in zip(written, spans, strict=True)]
    table = torch.tensor([flatten_row(s, lay) for s, lay in zip(slots, layouts, strict=True)],
                         dtype=torch.int32, device=dev)
    b = len(slots)
    q, k, v = (torch.randn(b * rows, h, t, d, device=dev) for _ in range(3))
    src = source if with_source else torch.zeros(rows, h, 1, 2 * d, device=dev)
    # float64, so the reference is not itself TF32
    want, want_cache = _reference(q.double(), k.double(), v.double(), cache.double(), src.double(),
                                  layouts, slots)
    want, want_cache = want.float(), want_cache.float()
    got = bounded_attention(q, k, v, cache, source, table, ieee=True)
    bounded_store(k, v, cache, table)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cache, want_cache, rtol=0, atol=0)
