"""Target taps (``TapSpec``): the residual stream entering a few target layers, every row of a
verify step's block, for a drafter that reads them in the same forward. On CPU, through the
draft-verify toy: each verify step's taps equal the stream a float64 trunk computes over the
request's tokens at the block's positions, rows of several requests of different lengths in one
step, across rejections (a rewound row is tapped again from the tokens that replace it)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import draft_verify_toy as toy  # noqa: E402

from mstar.engine.resources import DraftPolicy, Segment, StepContext, TapStep  # noqa: E402
from mstar.engine.resources.base import CGSlotSpec  # noqa: E402
from mstar.engine.resources.draft_verify.taps import TapResource  # noqa: E402
from mstar.engine.resources.step import BucketKey, SlotLease  # noqa: E402

PROMPTS = {0: [3, 9, 12, 40], 1: [5, 11], 2: [20, 21, 22, 23, 24, 25]}


def _streams(ids: list[int]) -> torch.Tensor:
    """``[len(ids), LAYERS + 1, HIDDEN]``: the float64 trunk's stream entering each block, and
    after the last, at every position."""
    sub = toy.ToySubmodule(k=0).double()
    with torch.no_grad():
        x = sub.embed(torch.tensor(ids)) + sub.pos_emb(torch.arange(len(ids)))
        out = [x]
        for block in sub.blocks:
            q, k, v = block.qkv(x).view(len(ids), 3, toy.HEADS, toy.HEAD_DIM).unbind(1)
            scores = torch.einsum("qhd,khd->hqk", q, k) * toy.HEAD_DIM ** -0.5
            mask = torch.ones(len(ids), len(ids), dtype=torch.bool).tril()
            attn = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            o = torch.einsum("hqk,khd->qhd", attn, v).reshape(len(ids), toy.HIDDEN)
            x = block.mlp(x + block.attn_scale * block.o(o))
            out.append(x)
    return torch.stack(out, 1)


@pytest.mark.parametrize("taps", [(0, 1, 2), (2, 0)], ids=["every_layer", "reordered"])
def test_each_verify_step_taps_the_stream_entering_each_layer_at_every_row_of_its_block(taps):
    k = 3
    sub = toy.ToySubmodule(k=k, corrupt=2, taps=taps)
    eng = toy.engine(sub, torch.device("cpu"))
    drv = toy.Driver(eng, sub)
    drv.add_batch(PROMPTS, 12)
    sub.tap_log = []
    live, steps, rejected = list(PROMPTS), 0, 0
    while live:
        before = {rid: PROMPTS[rid] + drv.emitted[rid] for rid in live}
        drv.decode(live)
        block, got = sub.tap_log[-1]
        assert got.shape == (len(live) * (k + 1), len(taps), toy.HIDDEN)
        for row, rid in enumerate(live):
            assert int(block[row, 0]) == before[rid][-1]
            seq = before[rid] + block[row, 1:].tolist()
            start = len(before[rid]) - 1
            want = _streams(seq)[start:start + k + 1][:, list(taps)]
            assert torch.allclose(got[row * (k + 1):(row + 1) * (k + 1)].double(), want, atol=1e-4)
            kept = len(drv.emitted[rid]) - (len(before[rid]) - len(PROMPTS[rid]))
            rejected += kept < k + 1
        steps += 1
        live = [rid for rid in live if rid not in drv.stops]
    # the walk rejected rows, so later blocks tapped positions a rewind gave back
    assert steps > 2 and rejected


def test_a_walk_that_taps_needs_the_taps_declared():
    with pytest.raises(ValueError, match="TapSpec"):
        toy.engine(toy.ToySubmodule(k=3, taps=(0, 1)), torch.device("cpu"), taps=())
    with pytest.raises(ValueError, match="TapSpec"):
        toy.engine(toy.ToySubmodule(k=3, taps=(0, 1)), torch.device("cpu"), taps=(1, 0))
    with pytest.raises(ValueError, match="distinct"):
        DraftPolicy(max_drafts=2, taps=(1, 1))


def test_a_step_without_taps_writes_none_and_a_captured_step_writes_its_slots_buffer():
    res = TapResource((3, 1), 8, torch.float32, torch.device("cpu"))
    assert res.columns == {3: 0, 1: 1}
    eager = StepContext(request_ids=(0, 1), graph_walk="decode", slot=0, capture=False)
    # unarmed: a target running outside a tapping step writes nothing, and nothing reads
    res.write(3, torch.ones(4, 8))
    with pytest.raises(RuntimeError, match="declared none"):
        res.read()
    res.plan(TapStep(segments=(Segment(0, "main", 2), Segment(1, "main", 3))), eager)
    res.write(3, torch.full((5, 8), 2.0))
    res.write(1, torch.full((6, 8), 3.0))  # a wider tensor: its first rows
    assert res.read().shape == (5, 2, 8)
    assert torch.equal(res.read()[:, 0], torch.full((5, 8), 2.0))
    assert torch.equal(res.read()[:, 1], torch.full((5, 8), 3.0))
    res.commit(TapStep(), eager)
    with pytest.raises(RuntimeError):
        res.read()

    bucket = BucketKey("decode", 2, 8)
    res.build_accelerator_graph_buffers([CGSlotSpec(bucket, 0, None), CGSlotSpec(bucket, 1, None)], 2, 64)
    lease = SlotLease(slot=1, bucket=bucket)
    ctx = StepContext(request_ids=(0, 1), graph_walk="decode", slot=1, capture=False, slot_lease=lease)
    res.plan(TapStep(segments=(Segment(0, "main", 4), Segment(-1, "main", 4))), ctx)
    res.write(1, torch.full((8, 8), 7.0))
    # the slot's own buffer, at the address a graph recorded against
    assert res.read().data_ptr() == res._static[1].data_ptr()
    assert torch.equal(res._static[1][:, 1], torch.full((8, 8), 7.0))
    assert not res._static[0].any()
    with pytest.raises(RuntimeError, match="holds"):
        res.plan(TapStep(segments=(Segment(0, "main", 9),)), ctx)

