"""The page-locked index ring the KV plan hands FlashInfer's plan copies
(MSTAR_KV_PINNED_INDPTRS): values round-trip, a slot comes back after
``depth`` takes, and a longer array grows every buffer."""
from __future__ import annotations

import numpy as np
import torch

from mstar.engine.resources.kv.plan import PinnedIndexRing, SequenceView, build_paged_indptrs


def test_values_round_trip_and_slots_rotate():
    ring = PinnedIndexRing(numel=8, depth=3)
    a = ring.take([1, 2, 3])
    b = ring.take(np.array([4, 5], dtype=np.int64))
    c = ring.take([6])
    assert a.tolist() == [1, 2, 3] and b.tolist() == [4, 5] and c.tolist() == [6]
    assert a.dtype == torch.int32
    # the fourth take reuses the first buffer
    d = ring.take([7, 8, 9, 10])
    assert d.data_ptr() == a.data_ptr() and a.tolist() == [7, 8, 9, 10][:3]
    assert len({t.data_ptr() for t in (a, b, c)}) == 3


def test_a_long_array_grows_the_ring():
    ring = PinnedIndexRing(numel=4, depth=2)
    small = ring.take([1, 2])
    big = ring.take(list(range(300)))  # past the 256 floor every buffer starts with
    assert big.tolist() == list(range(300))
    assert ring.take([3]).tolist() == [3]
    assert small.tolist() == [1, 2]  # the old buffer is still what it was
    assert big.numel() == 300 and ring.take([4]).numel() == 1


def test_build_paged_indptrs_takes_from_the_ring():
    views = [
        SequenceView("a", "main", [3, 4], 6, 1),
        SequenceView("b", "main", [7], 2, 1),
    ]
    plain = build_paged_indptrs(views, 4)
    ring = PinnedIndexRing(numel=16, depth=4)
    ringed = build_paged_indptrs(views, 4, ring=ring)
    for p, q in zip(plain[:4], ringed[:4], strict=True):
        assert torch.equal(p, q) and q.dtype == torch.int32
    assert ringed.kv_lens.tolist() == plain.kv_lens.tolist() == [6, 2]
    assert type(ringed.kv_lens) is type(plain.kv_lens)
