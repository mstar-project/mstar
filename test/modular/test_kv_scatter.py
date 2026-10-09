"""The Triton KV write must match the two index_put kernels it replaces."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv.cache import _kv_scatter_nhd_eager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton KV write needs CUDA")


def _expected(cache, layer_idx, k, v, page, slot):
    expected = cache.clone()
    expected[layer_idx][page, 0, slot] = k.to(cache.dtype)
    expected[layer_idx][page, 1, slot] = v.to(cache.dtype)
    return expected


@pytest.mark.parametrize("contiguous_indices", [True, False])
def test_kv_scatter_matches_index_put(contiguous_indices):
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
