"""``mstar::decode_linear`` on AMD GPUs: every shape the device's table names runs its hipBLASLt solution and
matches F.linear within a GEMM's rounding, eagerly and replayed from a captured graph; a shape the table does
not name falls back to F.linear."""

import pytest
import torch
import torch.nn.functional as F

from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")


def _table():
    from mstar.platforms.rocm_gemm import decode_linear_table

    return decode_linear_table(torch.cuda.current_device())


def _shape(key):
    rows, n, k = (int(part[len(tag):]) for part, tag in zip(key.split("-"), ("rows", "n", "k"), strict=True))
    return rows, n, k


def _check(out, want):
    scale = want.abs().max().item()
    assert (out.float() - want.float()).abs().max().item() <= 1.6e-2 * scale + 1e-2 * scale


@pytest.mark.parametrize("key", sorted(_table()) if current_platform.is_rocm() else [])
def test_each_tuned_shape_matches_f_linear(key):
    rows, n, k = _shape(key)
    g = torch.Generator(device="cuda").manual_seed(rows + n + k)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * k ** -0.5).contiguous()
    want = F.linear(x, w)
    _check(torch.ops.mstar.decode_linear(x, w, None), want)
    graph, out = torch.cuda.CUDAGraph(), None
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.ops.mstar.decode_linear(x, w, None)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        out = torch.ops.mstar.decode_linear(x, w, None)
    x.copy_(torch.randn(rows, k, device="cuda", dtype=torch.bfloat16, generator=g))
    graph.replay()
    torch.cuda.synchronize()
    _check(out, F.linear(x, w))


def test_an_untuned_shape_is_f_linear():
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(96, 64, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(torch.ops.mstar.decode_linear(x, w, None), F.linear(x, w))
