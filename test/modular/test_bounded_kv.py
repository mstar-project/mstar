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


@pytest.mark.parametrize("source_len", [0, 5, 29, 30, 60])
def test_slots_hold_the_sink_of_the_shortest_source(source_len):
    """A fixed sink longer than a short source spills into the slot; the slots are
    sized for the shortest source allowed, so a request with a longer one (whose
    retention alone needs no sink region) shares the layout."""
    from mstar.engine.resources.kv.bounded.config import BoundedKVConfig

    config = BoundedKVConfig(
        num_layers=1, num_heads=1, head_dim=1, rows_per_request=1,
        min_source_len=0, max_source_len=60, retention=lambda n: SinkWindow(30, 20),
    )
    assert config.sink_capacity == 30 and config.max_retained == 50
    source = [("s", i) for i in range(source_len)]
    sim = SlotSim(source, config.retention_for(source_len), config.sink_capacity)
    stream = list(source)
    for step in range(8):
        fresh = [("g", len(stream) - source_len + j) for j in range(7)]
        n = len(stream)
        assert sim.step(fresh) == stream[:min(30, n)] + stream[max(30, n - 20):n] + fresh, step
        stream += fresh


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


def _reference(q, k, v, cache, source, rows_layout, slots, sources, rel_bias=None, row_source=None):
    """Torch: attention over each row's ranges (in stream order) and its own keys,
    plus each pair's position term; then the writes."""
    n, h, t, d = q.shape
    rows = cache.shape[1]
    out = torch.empty(n, t, h, d, dtype=q.dtype, device=q.device)
    new_cache = cache.clone()
    for i in range(n):
        b, c = divmod(i, rows)
        layout, slot = rows_layout[b], slots[b]
        parts = []
        for from_source, (start, count) in zip(READ_FROM_SOURCE, layout.reads, strict=True):
            if not from_source:
                buf = cache[slot, c]
            elif sources[b] < 0:
                buf = row_source[0][row_source[1][b], c]
            else:
                buf = source[sources[b], c]
            parts.append(buf[:, start:start + count])
        kv = torch.cat(parts, dim=1)
        total = kv.shape[1]
        keys = torch.cat([kv[..., :d], k[i]], dim=1)
        values = torch.cat([kv[..., d:], v[i]], dim=1)
        scores = q[i] @ keys.transpose(-1, -2)
        if rel_bias is not None:
            for qi in range(t):
                for j in range(total + t):
                    scores[:, qi, j] += rel_bias[i, :, qi, total + qi - j + t - 1]
        att = torch.softmax(scores * d ** -0.5, dim=-1)
        out[i] = (att @ values).transpose(0, 1)
        for offset, start, count in layout.writes:
            new_cache[slot, c, :, start:start + count, :d] = k[i][:, offset:offset + count]
            new_cache[slot, c, :, start:start + count, d:] = v[i][:, offset:offset + count]
    return out, new_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")
@pytest.mark.parametrize("with_row_source", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("with_source", [True, False])
def test_kernels_match_torch(with_source, with_bias, with_row_source):
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
    # three sources (voices, say), each row starting from its own
    source = torch.randn(3, rows, h, max(source_len, 1), 2 * d, device=dev) if with_source else None
    # rows at different positions, including a read-only one
    written = [0, 50, 100, 170, 260]
    spans = [t, t, t, t, 0]
    slots = [3, 0, 5, 1, 2]
    sources = [2, 0, 1, 2, 0] if with_source else [0] * 5
    row_source = None
    if with_row_source and with_source:
        # rows 1 and 3 read their own entry of a second, per-row source
        sources = [2, -1, 1, -1, 0]
        row_source = (torch.randn(4, rows, h, source_len + 7, 2 * d, device=dev),
                      torch.tensor([0, 3, 0, 1, 0], dtype=torch.int32, device=dev))
    layouts = [row_layout(source_len, w, s, policy, sink_capacity) for w, s in zip(written, spans, strict=True)]
    table = torch.tensor([flatten_row(s, lay, src) for s, lay, src in zip(slots, layouts, sources, strict=True)],
                         dtype=torch.int32, device=dev)
    b = len(slots)
    q, k, v = (torch.randn(b * rows, h, t, d, device=dev) for _ in range(3))
    src = source if with_source else torch.zeros(1, rows, h, 1, 2 * d, device=dev)
    longest = max(sum(c for _, c in lay.reads) for lay in layouts)
    bias = torch.randn(b * rows, h, t, longest + 2 * t - 1, device=dev) * 4 if with_bias else None
    # float64, so the reference is not itself TF32
    ref_rows = None if row_source is None else (row_source[0].double(), row_source[1].tolist())
    want, want_cache = _reference(q.double(), k.double(), v.double(), cache.double(), src.double(),
                                  layouts, slots, sources, None if bias is None else bias.double(), ref_rows)
    want, want_cache = want.float(), want_cache.float()
    got = bounded_attention(q, k, v, cache, source, table, bias, ieee=True, row_source=row_source)
    bounded_store(k, v, cache, table)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cache, want_cache, rtol=0, atol=0)


@pytest.mark.parametrize("ring_buckets,dilation", [(4, 1), (6, 1), (3, 4), (5, 8)])
def test_waypoint_ring_is_sink_window(ring_buckets, dilation):
    """Waypoint's ring (``RingKVManager`` + flex visibility) as a sink + window: no
    sink, a window of ``ring_buckets`` frames that counts the current one, over a
    stream that only every ``dilation``-th frame writes to (the others attend
    without writing). Compared slot set by slot set against its block tables."""
    from types import SimpleNamespace

    from mstar.engine.resources.attn.flex import FlexAttentionManager

    tokens = 128
    manager = SimpleNamespace(_kv_config=SimpleNamespace(tokens_per_frame=tokens))
    geometry = (ring_buckets, ring_buckets, dilation)
    policy = SinkWindow(0, ring_buckets * tokens, window_includes_step=True)
    written = 0
    for frame in range(4 * ring_buckets * dilation):
        visible = FlexAttentionManager._visible_blocks_for(manager, geometry, session_idx=0, frame_pos=frame)
        # their ring slots, without the scratch frame the current one sits in
        want = sorted({b // (tokens // 128) for b in visible} - {ring_buckets})
        span = tokens if frame % dilation == 0 else 0
        layout = row_layout(0, written, span, policy, 0)
        got = sorted({pos // tokens for start, count in layout.reads[3:] for pos in range(start, start + count)})
        assert got == want, (frame, got, want)
        assert sum(c for _, c in layout.reads[:3]) == 0
        written += span


# The resource end to end (layout, attention, writes) over a stream of steps,
# against brute force over each row's whole stream.

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")
WINDOW = 70


def _manager(window: int, per_query: bool, dtype, device, heads: int = 2, head_dim: int = 128, slots: int = 4):
    from mstar.engine.resources.kv.bounded.config import BoundedKVConfig
    from mstar.engine.resources.kv.bounded.manager import BoundedKVManager

    config = BoundedKVConfig(num_layers=1, num_heads=heads, head_dim=head_dim, rows_per_request=1,
                             max_source_len=0, retention=SinkWindow(0, window), max_slots=slots, dtype=dtype,
                             query_window=window if per_query else None)
    return BoundedKVManager(config, torch.device(device))


def _step(rids, written, spans):
    from mstar.engine.resources.kv.bounded.config import BoundedKVStep, StreamPosition
    from mstar.engine.resources.step import Segment, StepContext

    step = BoundedKVStep(
        segments=tuple(Segment(r, "main", s) for r, s in zip(rids, spans, strict=True)),
        positions={r: StreamPosition(0, w) for r, w in zip(rids, written, strict=True)},
    )
    return step, StepContext(request_ids=tuple(rids), graph_walk="w", slot=0, capture=False)


def _stream_reference(streams, q, k, v, written, spans, window, per_query, bias):
    """float64: each row's query ``i`` (stream position ``w + i``) over its stream so
    far ``streams[b] = (keys, values) [H, w, D]`` and the step's own tokens; per
    query, the keys at distances [0, window], with the band indexed by distance;
    per step, the last ``window`` stream keys and the step's first ``span`` (all T
    for a row that writes none)."""
    n, h, t, d = q.shape
    out = torch.empty(n, t, h, d, dtype=torch.float64)
    for b in range(n):
        w = written[b]
        keys = torch.cat([streams[b][0], k[b]], 1)
        values = torch.cat([streams[b][1], v[b]], 1)
        lo = max(0, w - window)
        own = spans[b] or t
        for i in range(t):
            if per_query:
                seen = torch.arange(max(0, w + i - window), w + i + 1)
                rel = w + i - seen
            else:
                seen = torch.arange(lo, w + own)
                rel = (w - lo) + i - (seen - lo) + t - 1
            scores = (q[b, :, i:i + 1] @ keys[:, seen].transpose(-1, -2))[:, 0] + bias[b, :, i, rel]
            att = torch.softmax(scores * d ** -0.5, dim=-1)
            out[b, i] = (att.unsqueeze(1) @ values[:, seen])[:, 0]
    return out


def _check_stream(mgr, schedule, ieee, tol):
    """Runs ``schedule`` [(T, each row's span)] through ``mgr`` against brute force."""
    c = mgr.config
    h, d, window, per_query = c.num_heads, c.head_dim, c.window_capacity, c.query_window is not None
    device, dtype = mgr.layer(0).device, mgr.layer(0).dtype
    rows = len(schedule[0][1])
    rids = list(range(100, 100 + rows))
    streams = [(torch.zeros(h, 0, d, dtype=torch.float64),) * 2 for _ in rids]
    written = [0] * rows
    for t, spans in schedule:
        q, k, v = (torch.randn(rows, h, t, d).to(device=device, dtype=dtype) for _ in range(3))
        width = window + 1 if per_query else window + 2 * t - 1
        bias = (4 * torch.randn(rows, h, t, width)).to(device=device, dtype=dtype)
        want = _stream_reference(streams, q.double().cpu(), k.double().cpu(), v.double().cpu(),
                                 written, spans, window, per_query, bias.double().cpu())
        step, ctx = _step(rids, written, spans)
        assert mgr.admit(step, ctx).ok
        mgr.plan(step, ctx)
        got = mgr.attend(0, q, k, v, rel_bias=bias, ieee=ieee)
        mgr.commit(step, ctx)
        assert torch.isfinite(got.float()).all(), (t, spans)
        torch.testing.assert_close(got.double().cpu(), want, rtol=tol, atol=tol)
        for b, span in enumerate(spans):
            sk, sv = streams[b]
            streams[b] = (torch.cat([sk, k[b, :, :span].double().cpu()], 1),
                          torch.cat([sv, v[b, :, :span].double().cpu()], 1))
            written[b] += span


@pytest.mark.parametrize("per_query", [True, False])
@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32),
    pytest.param("cuda", torch.float32, marks=CUDA),
    pytest.param("cuda", torch.bfloat16, marks=CUDA),
])
def test_stream_matches_brute_force(device, dtype, per_query):
    """Rows at different positions in one batch, over one-token steps and steps
    longer than 2 x BLOCK_N, rows padded past their span, and rows that only read
    (span 0), at head_dim 128."""
    torch.manual_seed(0)
    mgr = _manager(WINDOW, per_query, dtype, device, heads=2, head_dim=128)
    schedule = [(1, [1, 1, 0]), (5, [5, 2, 0]), (150, [150, 0, 150]), (3, [3, 3, 1]),
                (1, [1, 0, 1]), (200, [200, 137, 0]), (2, [2, 2, 2])]
    _check_stream(mgr, schedule, ieee=True, tol=2e-5 if dtype == torch.float32 else 3e-2)


@CUDA
@pytest.mark.parametrize("t", [1, 150])
@pytest.mark.parametrize("per_query", [True, False])
@pytest.mark.parametrize("dtype,ieee,tol", [
    (torch.float32, True, 2e-5), (torch.float32, False, 2e-3),
    (torch.bfloat16, True, 3e-2), (torch.bfloat16, False, 3e-2),
])
def test_stream_at_many_rows(dtype, ieee, tol, per_query, t):
    """32 rows x 8 heads (enough programs for the kernel's wide query tiles) at
    head_dim 128: every tile config a real model reaches compiles and is right."""
    torch.manual_seed(0)
    mgr = _manager(WINDOW, per_query, dtype, "cuda", heads=8, head_dim=128, slots=32)
    spans = [1, 1, 0, 1] if t == 1 else [150, 0, 99, 150]
    _check_stream(mgr, [(150, [150, 37, 0, 150] * 8), (t, spans * 8)], ieee=ieee, tol=tol)


def test_commit_skips_read_only_segments():
    """An outer step that only reads a row commits after the region that wrote it,
    and must not put the row's count back."""
    mgr = _manager(WINDOW, True, torch.float32, "cpu", heads=1, head_dim=4)
    region, region_ctx = _step([1], [0], [1])
    outer, outer_ctx = _step([1], [0], [0])
    assert mgr.admit(outer, outer_ctx).ok
    mgr.plan(outer, outer_ctx)
    assert mgr.admit(region, region_ctx).ok
    mgr.plan(region, region_ctx)
    mgr.commit(region, region_ctx)
    mgr.commit(outer, outer_ctx)
    after, ctx = _step([1], [1], [1])
    assert mgr.admit(after, ctx).ok


def test_admit_refusal_names_the_request():
    mgr = _manager(WINDOW, True, torch.float32, "cpu", heads=1, head_dim=4)
    first, ctx = _step([1, 2], [0, 0], [1, 1])
    assert mgr.admit(first, ctx).ok
    mgr.plan(first, ctx)
    mgr.commit(first, ctx)
    # request 2 declares a stale position; request 1's is right
    stale, ctx = _step([1, 2], [1, 0], [1, 1])
    outcome = mgr.admit(stale, ctx)
    assert not outcome.ok and outcome.reason.request_ids == frozenset({2})


def test_rollback_admit_frees_only_fresh_slots():
    """A refused step gives back the slots its admit took, not one a request
    already held."""
    from mstar.engine.resources.kv.bounded.config import StreamPosition

    mgr = _manager(WINDOW, True, torch.float32, "cpu", heads=1, head_dim=4, slots=3)
    first, ctx = _step([1], [0], [1])
    assert mgr.admit(first, ctx).ok
    mgr.plan(first, ctx)
    mgr.commit(first, ctx)
    # 2 takes a slot, then 3 (a fresh request declaring written tokens) is refused
    bad, ctx = _step([1, 2, 3], [1, 0, 0], [1, 1, 1])
    bad = type(bad)(segments=bad.segments, positions={**bad.positions, 3: StreamPosition(0, 5)})
    assert not mgr.admit(bad, ctx).ok
    mgr.rollback_admit(bad, ctx)
    # both slots are free again, and 1 kept its slot and count
    fresh, ctx = _step([4, 5], [0, 0], [1, 1])
    assert mgr.admit(fresh, ctx).ok
    again, ctx = _step([1], [1], [1])
    assert mgr.admit(again, ctx).ok


def test_attend_refuses_misuse():
    """Arguments the kernel would read out of bounds or reinterpret are refused
    on the host, before a launch."""
    from mstar.engine.resources.kv.bounded.config import BoundedKVConfig, BoundedKVStep, StreamPosition
    from mstar.engine.resources.kv.bounded.manager import BoundedKVManager
    from mstar.engine.resources.step import Segment, StepContext

    h, d, t = 1, 16, 2
    config = BoundedKVConfig(num_layers=1, num_heads=h, head_dim=d, rows_per_request=1, min_source_len=10,
                             max_source_len=10, retention=SinkWindow(4, 8), max_slots=2)
    mgr = BoundedKVManager(config, torch.device("cpu"))
    # row 1 starts from a shared source, row 2 from its own row_source entry
    step = BoundedKVStep(segments=(Segment(1, "main", t), Segment(2, "main", t)),
                         positions={1: StreamPosition(10, 0, 0), 2: StreamPosition(10, 0, -1)})
    ctx = StepContext(request_ids=(1, 2), graph_walk="w", slot=0, capture=False)
    assert mgr.admit(step, ctx).ok
    mgr.plan(step, ctx)
    q, k, v = (torch.randn(2, h, t, d) for _ in range(3))
    source = torch.randn(1, 1, h, 10, 2 * d)
    row_source = (torch.randn(2, 1, h, 10, 2 * d), torch.tensor([1, 0], dtype=torch.int32))
    # 10 retained (4 sink + 6 window, all source) + 2T - 1
    band = torch.zeros(2, h, t, 10 + 2 * t - 1)
    mgr.attend(0, q, k, v, source, rel_bias=band, row_source=row_source)
    for kwargs in (dict(row_source=row_source), dict(source=source),
                   dict(source=source, row_source=row_source, rel_bias=band[..., :-1])):
        with pytest.raises(ValueError):
            mgr.attend(0, q, k, v, **kwargs)
    with pytest.raises(ValueError):
        mgr.attend(0, q.double(), k, v, source, row_source=row_source)


def test_kernel_refuses_a_head_dim_it_cannot_tile():
    pytest.importorskip("triton")
    from mstar.engine.resources.kv.bounded.kernels import bounded_attention
    from mstar.engine.resources.kv.bounded.layout import ROW_INTS

    q = torch.randn(1, 1, 1, 24)
    cache = torch.zeros(1, 1, 1, 4, 48)
    with pytest.raises(ValueError):
        bounded_attention(q, q, q, cache, None, torch.zeros(1, ROW_INTS, dtype=torch.int32))


def test_per_query_window_config():
    from mstar.engine.resources.kv.bounded.config import BoundedKVConfig

    def config(retention, **kw):
        return BoundedKVConfig(num_layers=1, num_heads=1, head_dim=4, rows_per_request=1,
                               max_source_len=4, retention=retention, **kw)

    assert config(lambda n: SinkWindow(0, 70), query_window=70).query_window == 70
    # a longer retained window than the query's is allowed (its far keys are masked)
    config(lambda n: SinkWindow(0, 70 + n), query_window=70)
    for bad in (SinkWindow(2, 70), SinkWindow(0, 70, window_includes_step=True), SinkWindow(0, 69)):
        with pytest.raises(ValueError):
            config(bad, query_window=70)
    with pytest.raises(ValueError):
        config(SinkWindow(0, 70), query_window=70, reverse_step_order=True)


def test_constant_retention_compares_by_value():
    """A constant policy works like the callable form, and two configs built from
    equal ones are equal (a model's spec compared against another's)."""
    from mstar.engine.resources.kv.bounded.config import BoundedKVConfig

    def config(retention):
        return BoundedKVConfig(num_layers=1, num_heads=1, head_dim=4, rows_per_request=1,
                               max_source_len=4, retention=retention, query_window=70)

    assert config(SinkWindow(0, 70)) == config(SinkWindow(0, 70))
    assert config(SinkWindow(0, 70)) != config(SinkWindow(0, 71))
    assert config(SinkWindow(0, 70)).retention_for(3) == config(lambda n: SinkWindow(0, 70)).retention_for(3)


def test_plan_resolves_retention_once_per_source_len():
    """The policy is resolved when the config is built, not per row per step."""
    from mstar.engine.resources.kv.bounded.config import BoundedKVConfig
    from mstar.engine.resources.kv.bounded.manager import BoundedKVManager

    calls = []

    def retention(n):
        calls.append(n)
        return SinkWindow(0, WINDOW)

    config = BoundedKVConfig(num_layers=1, num_heads=1, head_dim=4, rows_per_request=1, max_source_len=0,
                             retention=retention, max_slots=4)
    mgr = BoundedKVManager(config, torch.device("cpu"))
    built = len(calls)
    for t in range(3):
        step, ctx = _step([1, 2], [t, t], [1, 1])
        assert mgr.admit(step, ctx).ok
        mgr.plan(step, ctx)
        mgr.commit(step, ctx)
    assert len(calls) == built


def test_dtype_none_is_the_engines_kv_dtype():
    from mstar.engine.resources.base import EngineResourceInfo, Resource
    from mstar.engine.resources.kv.bounded.config import BoundedKVConfig, BoundedKVSpec
    from mstar.engine.resources.kv.bounded.manager import BoundedKVManager

    config = BoundedKVConfig(num_layers=1, num_heads=1, head_dim=4, rows_per_request=1, max_source_len=0,
                             retention=SinkWindow(0, 8), dtype=None)
    spec = BoundedKVSpec(resource_key="kv", nodes={"node"}, config=config)
    mgr = BoundedKVManager.build(spec, EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.bfloat16))
    assert mgr.layer(0).dtype == torch.bfloat16
    # left as declared, so the spec still equals one built from the same config
    assert config.dtype is None
    # the engine sizes graph tables through main's hook name
    assert BoundedKVManager.build_accelerator_graph_buffers is not Resource.build_accelerator_graph_buffers
