"""``mstar::qkv_norm_rope_cache`` on AMD GPUs (AITER's fused kernel) against its
torch reference at Qwen3.5's decode shapes: q, gate, k, v and the cache rows it
writes, and a captured replay that writes the next step's slots."""

import pytest
import torch

from mstar.engine.resources.attn import prologue
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = "cuda"
PAGE = 128


def _case(tokens, heads, kv_heads, head_dim=256, half=32, pages=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    qkv = torch.randn(tokens, (2 * heads + 2 * kv_heads) * head_dim, device=DEV, generator=g).to(torch.bfloat16)
    qw = (torch.randn(head_dim, device=DEV, generator=g) * 0.1).to(torch.bfloat16)
    kw = (torch.randn(head_dim, device=DEV, generator=g) * 0.1).to(torch.bfloat16)
    inv = 1.0 / (1e7 ** (torch.arange(0, 2 * half, 2, device=DEV).float() / (2 * half)))
    freqs = inv[None, :] * torch.arange(32768, device=DEV).float()[:, None]
    cos, sin = freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)
    positions = torch.randint(0, 32768, (tokens,), device=DEV, generator=g)
    slots = torch.randperm(pages * PAGE, device=DEV, generator=g)[:tokens]
    cache = torch.zeros(pages, 2, PAGE, kv_heads, head_dim, device=DEV, dtype=torch.bfloat16)
    return qkv, qw, kw, cos, sin, positions, cache, slots


def _run(fn, qkv, qw, kw, cos, sin, positions, cache, slots, heads, kv_heads, head_dim=256):
    out = fn(qkv, qw, kw, cos, sin, positions, cache[:, 0], cache[:, 1], slots, heads, kv_heads, head_dim, 1e-6, True)
    return out, cache


@pytest.mark.parametrize("tokens", [1, 4, 64, 128])
@pytest.mark.parametrize("heads, kv_heads", [(16, 4), (8, 2)])
def test_matches_the_reference(tokens, heads, kv_heads):
    qkv, qw, kw, cos, sin, positions, cache, slots = _case(tokens, heads, kv_heads)
    impl = prologue.QKV_NORM_ROPE_CACHE.implementation()
    got, got_cache = _run(impl, qkv, qw, kw, cos, sin, positions, cache.clone(), slots, heads, kv_heads)
    want, want_cache = _run(prologue._reference, qkv, qw, kw, cos, sin, positions, cache.clone(), slots,
                            heads, kv_heads)
    for name, a, b in zip(("q", "gate", "k", "v"), got, want, strict=True):
        # the kernel rounds the normalised q once before rotating: a bf16 ulp or so
        torch.testing.assert_close(a.float(), b.float(), rtol=1.6e-2, atol=1.6e-2, msg=name)
    torch.testing.assert_close(got_cache.float(), want_cache.float(), rtol=1.6e-2, atol=1.6e-2)
    untouched = torch.ones(cache.shape[0] * PAGE, dtype=torch.bool, device=DEV)
    untouched[slots] = False
    assert got_cache.permute(0, 2, 1, 3, 4).reshape(-1, 2, kv_heads, 256)[untouched].abs().max() == 0


def test_a_captured_call_writes_the_slots_it_is_replayed_with():
    heads, kv_heads, tokens = 16, 4, 8
    qkv, qw, kw, cos, sin, positions, cache, slots = _case(tokens, heads, kv_heads)
    impl = prologue.QKV_NORM_ROPE_CACHE.implementation()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _run(impl, qkv, qw, kw, cos, sin, positions, cache, slots, heads, kv_heads)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _run(impl, qkv, qw, kw, cos, sin, positions, cache, slots, heads, kv_heads)
    cache.zero_()
    fresh = torch.arange(100, 100 + tokens, device=DEV)
    slots.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    want, want_cache = _run(prologue._reference, qkv, qw, kw, cos, sin, positions,
                            torch.zeros_like(cache), fresh, heads, kv_heads)
    torch.testing.assert_close(cache.float(), want_cache.float(), rtol=1.6e-2, atol=1.6e-2)
