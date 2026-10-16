"""KVCache.write_tokens' Triton scatter writes exactly what the advanced-index scatter writes: K and V of each
token into its (page, offset-in-page) slot of one layer, from strided slices of a fused projection, with a dtype
cast, and nothing else."""
import pytest
import torch

from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import KVLayout, PagedKVConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")


def _cache(num_kv_heads, head_dim):
    cfg = PagedKVConfig(num_layers=3, max_seq_len=144, max_num_pages=9, page_size=16, num_kv_heads=num_kv_heads,
                        head_dim=head_dim, layout=KVLayout.NHD)
    cache = KVCache(cfg, torch.device("cuda"), dtype=torch.bfloat16)
    cache.tensor.normal_()
    return cache


@pytest.mark.parametrize("num_tokens", [1, 5, 40])
@pytest.mark.parametrize("num_kv_heads,head_dim", [(8, 128), (2, 64), (4, 80)])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
def test_matches_index_put(num_tokens, num_kv_heads, head_dim, src_dtype):
    torch.manual_seed(0)
    cache = _cache(num_kv_heads, head_dim)
    expected = cache.tensor.clone()
    hd = num_kv_heads * head_dim
    # k and v as strided slices of one qkv projection, as the attention layers pass them
    qkv = torch.randn(num_tokens, 3 * hd, device="cuda", dtype=src_dtype)
    k = qkv[:, hd: 2 * hd].view(num_tokens, num_kv_heads, head_dim)
    v = qkv[:, 2 * hd:].view(num_tokens, num_kv_heads, head_dim)
    # distinct slots, page 0 included (the padding sink)
    flat = torch.randperm(9 * 16, device="cuda")[:num_tokens]
    page_idx, cache_idx = flat // 16, flat % 16

    cache.write_tokens(layer_idx=1, k=k, v=v, page_idx=page_idx, cache_idx=cache_idx)

    layer = expected[1]
    layer[page_idx, 0, cache_idx] = k.to(torch.bfloat16)
    layer[page_idx, 1, cache_idx] = v.to(torch.bfloat16)
    assert torch.equal(cache.tensor, expected)


def test_int32_indices_and_empty():
    cache = _cache(8, 128)
    expected = cache.tensor.clone()
    empty = torch.empty(0, 8, 128, device="cuda", dtype=torch.bfloat16)
    idx = torch.empty(0, device="cuda", dtype=torch.int32)
    cache.write_tokens(layer_idx=0, k=empty, v=empty, page_idx=idx, cache_idx=idx)
    assert torch.equal(cache.tensor, expected)

    k = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    page_idx = torch.tensor([8, 3], device="cuda", dtype=torch.int32)
    cache_idx = torch.tensor([15, 0], device="cuda", dtype=torch.int32)
    cache.write_tokens(layer_idx=2, k=k, v=v, page_idx=page_idx, cache_idx=cache_idx)
    expected[2][page_idx.long(), 0, cache_idx.long()] = k
    expected[2][page_idx.long(), 1, cache_idx.long()] = v
    assert torch.equal(cache.tensor, expected)


def test_cuda_graph_replay_follows_index_buffers():
    cache = _cache(8, 128)
    k = torch.zeros(4, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    page_idx = torch.zeros(4, device="cuda", dtype=torch.long)
    cache_idx = torch.arange(4, device="cuda", dtype=torch.long)
    cache.write_tokens(layer_idx=0, k=k, v=v, page_idx=page_idx, cache_idx=cache_idx)  # warm the JIT
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cache.write_tokens(layer_idx=0, k=k, v=v, page_idx=page_idx, cache_idx=cache_idx)
    k.copy_(torch.randn_like(k))
    v.copy_(torch.randn_like(v))
    page_idx.copy_(torch.tensor([5, 5, 7, 1], device="cuda"))
    cache_idx.copy_(torch.tensor([2, 9, 0, 15], device="cuda"))
    expected = cache.tensor.clone()
    expected[0][page_idx, 0, cache_idx] = k
    expected[0][page_idx, 1, cache_idx] = v
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(cache.tensor, expected)
