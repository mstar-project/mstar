"""Strided q/k/v into the ragged attention wrappers, as ``ulysses_attention`` or a split
fused QKV projection hands them over: views with strided rows and a dense head dim.

A wrapper passes such views to its kernel only where the kernel reads their strides
(FlashInfer's FA2/FA3, FA3 varlen) and copies them for every other backend. On a GPU,
those kernels must read the views exactly as they read contiguous copies.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.attn.ragged import dense_wrappers
from mstar.engine.resources.attn.ragged.wrappers import FlashInferRaggedPrefillWrapper

ROWS, HQ, HKV = 37, 4, 2


def _fused(rows=ROWS, head_dim=64, device="cpu"):
    """q, k and v as views of one ``[rows, HQ + 2 HKV, head_dim]`` buffer."""
    gen = torch.Generator(device).manual_seed(0)
    buf = torch.randn(rows, HQ + 2 * HKV, head_dim, dtype=torch.bfloat16, device=device, generator=gen)
    return buf.split([HQ, HKV, HKV], dim=1)


def _as_lain(got, t):
    return got.data_ptr() == t.data_ptr() and got.stride() == t.stride()


class _Planned:
    """What ``_pad_head_dim`` reads of a planned FlashInfer wrapper: the resolved backend."""

    def __init__(self, backend: str):
        self._backend = backend


@pytest.mark.parametrize("backend,reads_strides", [
    ("fa2", True), ("fa3", True), ("cudnn", False), ("cutlass", False), ("auto", False),
])
def test_flashinfer_takes_views_only_where_the_kernel_reads_strides(backend, reads_strides):
    wrapper = FlashInferRaggedPrefillWrapper.__new__(FlashInferRaggedPrefillWrapper)
    wrapper.padded_head_dim = 64
    wrapper.attn_wrapper = _Planned(backend)
    for t in _fused():
        got = wrapper._pad_head_dim(t)
        assert _as_lain(got, t) if reads_strides else got.is_contiguous() and torch.equal(got, t)
    strided_head_dim = torch.randn(ROWS, 64, HQ, dtype=torch.bfloat16).transpose(1, 2)
    got = wrapper._pad_head_dim(strided_head_dim)
    assert got.is_contiguous() and torch.equal(got, strided_head_dim)


@pytest.mark.parametrize("reads_strides", [True, False], ids=["fa3", "aiter"])
def test_dense_wrapper_takes_views_only_on_fa3(monkeypatch, reads_strides):
    seen = {}

    def varlen(q, k, v, *args, **kwargs):
        seen.update(q=q, k=k, v=v)
        return torch.zeros_like(q)

    monkeypatch.setattr(dense_wrappers, "flash_attn_varlen_func", varlen)
    monkeypatch.setattr(dense_wrappers, "VARLEN_READS_STRIDES", reads_strides, raising=False)
    wrapper = dense_wrappers.DenseRaggedPrefillWrapper(HQ, HKV, 64, device=torch.device("cpu"))
    wrapper.plan(torch.tensor([0, 11, ROWS], dtype=torch.int32))
    q, k, v = _fused()
    wrapper.run(q, k, v)
    for name, t in (("q", q), ("k", k), ("v", v)):
        got = seen[name]
        assert _as_lain(got, t) if reads_strides else got.is_contiguous() and torch.equal(got, t)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@cuda
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_flashinfer_reads_the_views_exactly(backend, graph):
    pytest.importorskip("flashinfer")
    if backend == "fa3" and torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FlashInfer's FA3 prefill is Hopper-only")
    device = torch.device("cuda")
    rows = 4099
    extra = dict(max_num_segments=2, max_total_tokens=rows, accelerator_graph=True) if graph else {}
    wrapper = FlashInferRaggedPrefillWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device=device), HQ, HKV, 128, backend=backend,
        device=device, **extra,
    )
    wrapper.plan(torch.tensor([0, 1500, rows], dtype=torch.int32))
    assert wrapper.attn_wrapper._backend == backend
    q, k, v = _fused(rows, 128, device)
    assert all(_as_lain(wrapper._pad_head_dim(t), t) for t in (q, k, v))
    got = wrapper.run(q, k, v)
    assert torch.equal(got, wrapper.run(q.contiguous(), k.contiguous(), v.contiguous()))


@cuda
def test_fa3_varlen_reads_the_views_exactly():
    from mstar.engine.resources.attn.dense import _fa3_unavailable_reason

    device = torch.device("cuda")
    reason = _fa3_unavailable_reason(device)
    if reason is not None:
        pytest.skip(reason)
    rows = 4099
    wrapper = dense_wrappers.DenseRaggedPrefillWrapper(HQ, HKV, 128, device=device)
    wrapper.plan(torch.tensor([0, 1500, rows], dtype=torch.int32))
    q, k, v = _fused(rows, 128, device)
    got = wrapper.run(q, k, v)
    assert torch.equal(got, wrapper.run(q.contiguous(), k.contiguous(), v.contiguous()))
