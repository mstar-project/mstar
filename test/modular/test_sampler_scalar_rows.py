"""A sampler's scalar settings reach each cg slot's per-step rows in one
staged copy (``ScalarRows``): every buffer's slot view reads its rows'
values, at its own dtype, and the two slots stay apart. On the pinned path a
gather of bytes the slot already holds issues no copy."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.sampler.utils import SamplerBuffers, SamplingConfig


def _buffers(device):
    bufs = SamplerBuffers.allocate(
        max_batch_size=4, device=device, cg_slots=2, enable_min_p=True, enable_top_p_first=True,
    )
    bufs.register_request("a", SamplingConfig(temperature=0.5, top_k=7, top_p=0.9, min_p=0.1))
    bufs.register_request("b", SamplingConfig(top_p_first=True, top_p_min_keep=3))
    return bufs


def _rows(bufs, bs, slot):
    view = bufs.slice_for_bs(bs, slot)
    return (
        view["temperature_buf"].tolist(), view["top_k_buf"].tolist(), view["top_p_buf"].tolist(),
        view["min_p_buf"].tolist(), view["order"].top_p_first.tolist(), view["order"].min_keep.tolist(),
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))])
def test_each_buffer_reads_its_rows_in_each_slot(device):
    bufs = _buffers(torch.device(device))
    bufs.gather_static(["a", "b"], 3, 0)
    bufs.gather_static(["b"], 2, 1)
    if device == "cuda":
        torch.cuda.synchronize()
    default = SamplingConfig()
    t, k, p, m, first, keep = _rows(bufs, 3, 0)
    assert t == pytest.approx([0.5, default.temperature, 1.0])  # the padding row reads the trash row
    assert k == [7, 0, 0] and p == pytest.approx([0.9, 1.0, 1.0]) and m == pytest.approx([0.1, 0.0, 0.0])
    assert first == [False, True, False] and keep == [1, 3, 1]
    t, k, _, _, first, keep = _rows(bufs, 2, 1)
    assert t == pytest.approx([default.temperature, 1.0]) and k == [0, 0]
    assert first == [True, False] and keep == [3, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the skip is on the pinned path")
def test_a_gather_of_the_bytes_a_slot_holds_issues_no_copy():
    bufs = _buffers(torch.device("cuda"))
    bufs.register_request("c", SamplingConfig(temperature=0.5, top_k=7, top_p=0.9, min_p=0.1))
    stager = bufs._scalar_rows._stager
    bufs.gather_static(["a"], 2, 0)
    ring = stager._next
    # another batch with the same settings: a gather, but no copy
    bufs.gather_static(["c"], 2, 0)
    assert stager._next == ring
    bufs.gather_static(["b"], 2, 0)
    assert stager._next != ring
    torch.cuda.synchronize()
    assert _rows(bufs, 2, 0)[4] == [True, False]
