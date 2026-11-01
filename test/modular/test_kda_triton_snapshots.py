"""The fused KDA prefill's prefix snapshots on CUDA: the state and conv tail it stores at a span
offset are what a span cut there leaves, and storing them changes nothing else, eager or
replayed from a captured graph with new targets.
"""
import sys

sys.path.insert(0, ".")

import pytest
import torch
from test_kda_triton import HEAD_DIM, HEADS, HIDDEN, SLOTS, _kda, _Kda, _Pool, _pools, _spans

from mstar.engine.resources.convenience import KDACallable
from mstar.engine.resources.linear_attn import kda_triton
from mstar.engine.resources.linear_attn.kda import KDAPlan

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and kda_triton._HAS_TRITON),
    reason="fused kernels need CUDA + triton",
)

C = kda_triton.CONFIG.chunk


def _plan(spans, snapshots=None, num_tokens=None) -> KDAPlan:
    lens = [s.q_len for s in spans]
    slots = torch.tensor([s.slot for s in spans], dtype=torch.int32, device="cuda")
    return KDAPlan(
        slot_ids=slots, has_state=torch.ones_like(slots, dtype=torch.bool), spans=tuple(lens),
        num_tokens=num_tokens or sum(lens), is_decode=False,
        layout=torch.tensor(kda_triton.varlen_layout(lens, num_tokens, snapshots=snapshots),
                            dtype=torch.int32, device="cuda"),
        snapshots=tuple(snapshots or ()), snapshot_section=snapshots is not None,
    )


def _run(kda, x, plan, rec, conv):
    kda.mix = KDACallable(_Pool(rec, conv), _Kda(plan, kda_triton.TritonKDAKernels()))
    with torch.no_grad():
        return kda(x, 0)


@pytest.mark.parametrize("fresh", [True, False])
def test_snapshots_are_the_state_a_span_cut_there_leaves(fresh):
    kda = _kda()
    lengths = (300, 2 * C, 5, 3 * C + 7)
    spans, total = _spans(lengths)
    x = torch.randn(total, HIDDEN, device="cuda", dtype=torch.bfloat16)
    rec, conv = _pools(fresh)
    # free slots for the snapshots: the spans' slots are 1..SLOTS-1, so take two more pools' worth
    snap_rec, snap_conv = torch.zeros(8, *rec.shape[1:], device="cuda"), torch.zeros(8, *conv.shape[1:],
                                                                                      device="cuda", dtype=conv.dtype)
    big_rec, big_conv = torch.cat([rec, snap_rec]), torch.cat([conv, snap_conv])
    first = SLOTS
    # (span, offset, slot): two chunk ends, one at a span's end, two in one span
    targets = [(0, 2 * C, first), (0, 4 * C, first + 1), (1, 2 * C, first + 2), (3, 3 * C, first + 3)]
    keep_rec, keep_conv = big_rec.clone(), big_conv.clone()
    out = _run(kda, x, _plan(spans, targets), keep_rec, keep_conv)

    plain_rec, plain_conv = big_rec.clone(), big_conv.clone()
    ref = _run(kda, x, _plan(spans), plain_rec, plain_conv)
    assert torch.equal(out, ref), "keeping snapshots changed the outputs"
    assert torch.equal(keep_rec[:first], plain_rec[:first]) and torch.equal(keep_conv[:first], plain_conv[:first])

    for span, offset, slot in targets:
        s = spans[span]
        cut = type(s)(slot=s.slot, q_start=0, q_len=offset)
        cut_rec, cut_conv = big_rec.clone(), big_conv.clone()
        _run(kda, x[s.q_start:s.q_start + offset], _plan([cut]), cut_rec, cut_conv)
        assert torch.equal(keep_rec[slot], cut_rec[s.slot]), f"span {span}: the state at {offset}"
        assert torch.equal(keep_conv[slot], cut_conv[s.slot]), f"span {span}: the conv tail at {offset}"


def test_a_captured_prefill_replays_with_new_snapshot_targets():
    kda = _kda(seed=3)
    tokens, rows = 512, 2
    spans, total = _spans((300, 200))
    x = torch.zeros(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    x[:total] = torch.randn(total, HIDDEN, device="cuda", dtype=torch.bfloat16)
    rec, conv = _pools(fresh=False)
    rec = torch.cat([rec, torch.zeros(4, *rec.shape[1:], device="cuda")])
    conv = torch.cat([conv, torch.zeros(4, *conv.shape[1:], device="cuda", dtype=conv.dtype)])
    start_rec, start_conv = rec.clone(), conv.clone()

    capture = _plan(spans, [], num_tokens=tokens)  # the section, with no targets
    assert len(capture.spans) == rows
    kda.mix = KDACallable(_Pool(rec, conv), _Kda(capture, kda_triton.TritonKDAKernels()))
    with torch.no_grad():
        kda(x, 0)  # compile outside the capture
        rec.copy_(start_rec), conv.copy_(start_conv)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = kda(x, 0)

    targets = [(0, 4 * C, SLOTS), (1, 3 * C, SLOTS + 1)]
    replay = _plan(spans, targets, num_tokens=tokens)
    capture.layout.copy_(replay.layout)
    rec.copy_(start_rec), conv.copy_(start_conv)
    graph.replay()
    torch.cuda.synchronize()

    eager_rec, eager_conv = start_rec.clone(), start_conv.clone()
    eager = _run(kda, x, _plan(spans, targets, num_tokens=tokens), eager_rec, eager_conv)
    assert torch.equal(graph_out[:total], eager[:total])
    assert torch.equal(rec, eager_rec) and torch.equal(conv, eager_conv), "the replay kept other snapshots"
    assert eager_rec[SLOTS].abs().sum() > 0 and eager_rec[SLOTS + 1].abs().sum() > 0
    assert HEADS * HEAD_DIM == conv.shape[1] // 3
