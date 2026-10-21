"""The Triton KV write must match the two index_put kernels it replaces."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import cache as kv_cache
from mstar.engine.resources.kv.cache import _kv_scatter_nhd_eager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton KV write needs CUDA")


def _expected(cache, layer_idx, k, v, page, slot):
    expected = cache.clone()
    expected[layer_idx][page, 0, slot] = k.to(cache.dtype)
    expected[layer_idx][page, 1, slot] = v.to(cache.dtype)
    return expected


@pytest.fixture
def launches(monkeypatch):
    """The grids the Triton kernel was launched with; empty if index_put ran instead."""
    grids = []
    kernel = kv_cache._kv_scatter_nhd_kernel

    class Spy:
        def __getitem__(self, grid):
            grids.append(grid)
            return kernel[grid]

    monkeypatch.setattr(kv_cache, "_kv_scatter_nhd_kernel", Spy())
    return grids


@pytest.mark.parametrize("contiguous_indices", [True, False])
def test_kv_scatter_matches_index_put(contiguous_indices, launches):
    torch.manual_seed(0)
    # [layers, pages, k/v, page_size, kv_heads, head_dim]
    cache = torch.zeros(3, 10, 2, 16, 2, 128, dtype=torch.bfloat16, device="cuda")
    # K and V as strided views of one fused QKV projection, as attention passes them.
    qkv = torch.randn(5, 6 * 128, device="cuda").to(torch.bfloat16)
    k = qkv[:, 2 * 128:4 * 128].view(5, 2, 128)
    v = qkv[:, 4 * 128:].view(5, 2, 128)
    page = torch.tensor([3, 3, 7, 0, 9], device="cuda")
    slot = torch.tensor([0, 1, 15, 4, 2], device="cuda")
    if not contiguous_indices:  # takes the index_put fallback
        page = torch.stack([page, page], dim=1)[:, 0]
        slot = torch.stack([slot, slot], dim=1)[:, 0]
        assert not page.is_contiguous()
    expected = _expected(cache, 1, k, v, page, slot)
    _kv_scatter_nhd_eager(cache, 1, k, v, page, slot)
    torch.testing.assert_close(cache, expected, atol=0, rtol=0)
    assert launches == ([(5,)] if contiguous_indices else [])


@pytest.mark.parametrize(
    "heads, head_dim, tokens, kv_dtype",
    [
        pytest.param(20, 64, 5, torch.bfloat16, id="masked-tail"),
        pytest.param(5, 96, 4097, torch.bfloat16, id="multi-wave"),
        pytest.param(32, 128, 64, torch.bfloat16, id="8-warps"),
        pytest.param(64, 128, 64, torch.bfloat16, id="16-warps"),
        pytest.param(8, 128, 257, torch.float32, id="fp32-kv"),
    ],
)
def test_kv_scatter_geometries(heads, head_dim, tokens, kv_dtype, launches):
    torch.manual_seed(0)
    page_size = 16
    pages = tokens // page_size + 1
    cache = torch.zeros(3, pages, 2, page_size, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    qkv = torch.randn(tokens, 3, heads, head_dim, device="cuda").to(kv_dtype)
    k, v = qkv[:, 1], qkv[:, 2]
    # Distinct slots: which of two writes to the same slot wins is unspecified.
    flat = torch.randperm(pages * page_size, device="cuda")[:tokens]
    page, slot = flat // page_size, flat % page_size
    expected = _expected(cache, 1, k, v, page, slot)
    _kv_scatter_nhd_eager(cache, 1, k, v, page, slot)
    torch.testing.assert_close(cache, expected, atol=0, rtol=0)
    assert launches == [(tokens,)]


@pytest.mark.parametrize("mismatch", ["v_tokens", "cache_heads", "page_count"])
def test_kv_scatter_rejects_what_index_put_rejects(mismatch):
    # The kernel sizes everything from k, so these must take index_put and fail there.
    cache = torch.zeros(3, 10, 2, 16, 2, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(5, 2, 128, device="cuda").to(torch.bfloat16)
    v = torch.randn(5, 2, 128, device="cuda").to(torch.bfloat16)
    page = torch.tensor([3, 3, 7, 0, 9], device="cuda")
    slot = torch.tensor([0, 1, 15, 4, 2], device="cuda")
    if mismatch == "v_tokens":
        v = v[:4]
    elif mismatch == "cache_heads":
        cache = cache[..., :1, :].contiguous()  # one KV head; k has two
    else:
        page = page[:4]
    with pytest.raises((RuntimeError, IndexError)):
        _kv_scatter_nhd_eager(cache, 1, k, v, page, slot)


def test_kv_scatter_takes_int32_indices_and_no_tokens():
    cache = torch.zeros(3, 10, 2, 16, 2, 128, dtype=torch.bfloat16, device="cuda")
    expected = cache.clone()
    empty = torch.empty(0, 2, 128, device="cuda", dtype=torch.bfloat16)
    none = torch.empty(0, device="cuda", dtype=torch.int32)
    _kv_scatter_nhd_eager(cache, 0, empty, empty, none, none)
    assert torch.equal(cache, expected)
    k = torch.randn(2, 2, 128, device="cuda").to(torch.bfloat16)
    v = torch.randn(2, 2, 128, device="cuda").to(torch.bfloat16)
    page = torch.tensor([8, 3], device="cuda", dtype=torch.int32)
    slot = torch.tensor([15, 0], device="cuda", dtype=torch.int32)
    expected = _expected(cache, 2, k, v, page.long(), slot.long())
    _kv_scatter_nhd_eager(cache, 2, k, v, page, slot)
    torch.testing.assert_close(cache, expected, atol=0, rtol=0)


def test_kv_scatter_replay_follows_its_index_buffers():
    """Captured, the write reads the indices at replay, not at capture."""
    cache = torch.zeros(1, 10, 2, 16, 2, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(4, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    page = torch.zeros(4, device="cuda", dtype=torch.long)
    slot = torch.arange(4, device="cuda", dtype=torch.long)
    _kv_scatter_nhd_eager(cache, 0, k, v, page, slot)  # warm the JIT
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _kv_scatter_nhd_eager(cache, 0, k, v, page, slot)
    k.copy_(torch.randn_like(k))
    v.copy_(torch.randn_like(v))
    page.copy_(torch.tensor([5, 5, 7, 1], device="cuda"))
    slot.copy_(torch.tensor([2, 9, 0, 15], device="cuda"))
    expected = _expected(cache, 0, k, v, page, slot)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(cache, expected, atol=0, rtol=0)
