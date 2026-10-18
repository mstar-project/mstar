"""``mstar::rms_norm`` and ``mstar::fused_add_rms_norm`` on AMD GPUs.

Both must compute FlashInfer's numerics (fp32 math, one round-to-nearest-even
store), which an fp32 reference rounded once reproduces. A truncating store
(AITER's default on gfx942) misses by about half a bf16 ulp on average, which the
relative-error bound catches.
"""

import pytest
import torch

from mstar.model.components.norm import RMSNorm
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = "cuda"
EPS = 1e-6


def _ref_norm(x, w, gemma=False):
    xf = x.float()
    scale = w.float() + (1.0 if gemma else 0.0)
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * scale


def _assert_rounded_once(out, ref_fp32):
    ref = ref_fp32.to(out.dtype)
    rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel < 1e-4, f"relative error {rel:.2e} (a truncating store gives ~4e-3)"
    # any element off by more than one ulp is a different computation
    torch.testing.assert_close(out, ref, rtol=2 ** -7, atol=1e-3)


@pytest.mark.parametrize("hidden", [128, 1024, 2048, 3072, 5120])
@pytest.mark.parametrize("rows", [1, 7, 64, 513])
def test_rms_norm_2d(hidden, rows):
    torch.manual_seed(rows * hidden)
    x = torch.randn(rows, hidden, device=DEV, dtype=torch.bfloat16) * 3
    w = torch.randn(hidden, device=DEV, dtype=torch.bfloat16)
    _assert_rounded_once(torch.ops.mstar.rms_norm(x, w, EPS, None), _ref_norm(x, w))


def test_rms_norm_per_head_views():
    """Per-head q/k norms arrive 3-D, strided out of a fused QKV projection."""
    torch.manual_seed(0)
    qkv = torch.randn(33, (16 + 2 * 4) * 128, device=DEV, dtype=torch.bfloat16)
    q = qkv[:, : 16 * 128].view(33, 16, 128)
    w = torch.randn(128, device=DEV, dtype=torch.bfloat16)
    out = torch.ops.mstar.rms_norm(q, w, EPS, None)
    assert out.shape == q.shape
    _assert_rounded_once(out, _ref_norm(q, w))
    row_strided = qkv[:, :1000]
    w2 = torch.randn(1000, device=DEV, dtype=torch.bfloat16)
    _assert_rounded_once(torch.ops.mstar.rms_norm(row_strided, w2, EPS, None), _ref_norm(row_strided, w2))


def test_rms_norm_dtypes():
    """fp16 runs as is; fp32 input on a GPU is normalized in bf16 and returned in fp32."""
    torch.manual_seed(0)
    x = torch.randn(9, 2048, device=DEV, dtype=torch.float16)
    w = torch.randn(2048, device=DEV, dtype=torch.float16)
    _assert_rounded_once(torch.ops.mstar.rms_norm(x, w, EPS, None), _ref_norm(x, w))
    x32 = torch.randn(9, 2048, device=DEV)
    out = torch.ops.mstar.rms_norm(x32, w.float(), EPS, None)
    assert out.dtype == torch.float32
    xb = x32.bfloat16()
    torch.testing.assert_close(out, _ref_norm(xb, w.bfloat16()).bfloat16().float(), rtol=2 ** -7, atol=1e-3)


@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("hidden", [1024, 2048, 3072, 5120])
@pytest.mark.parametrize("rows", [1, 7, 64, 513])
def test_fused_add_rms_norm(gemma, hidden, rows):
    torch.manual_seed(rows + hidden)
    h0 = torch.randn(rows, hidden, device=DEV, dtype=torch.bfloat16)
    r0 = torch.randn(rows, hidden, device=DEV, dtype=torch.bfloat16) * 4
    w = torch.randn(hidden, device=DEV, dtype=torch.bfloat16) * 0.5
    h, r = h0.clone(), r0.clone()
    torch.ops.mstar.fused_add_rms_norm(h, r, w, EPS, gemma)
    total = h0.float() + r0.float()
    assert torch.equal(r, total.bfloat16()), "the new residual is the sum rounded once"
    _assert_rounded_once(h, _ref_norm(total, w, gemma))


def _stack(norms, h, r):
    """A layer loop the way Qwen3.5 runs it: add-norm, a matmul, repeat."""
    for norm in norms:
        h, r = norm.forward_residual(h, r)
        h = h @ norm.proj
    return h, r


def _norms(hidden, gemma):
    torch.manual_seed(1)
    norms = []
    for _ in range(3):
        n = RMSNorm(hidden, eps=EPS, gemma_mode=gemma).to(DEV, torch.bfloat16)
        with torch.no_grad():
            n.weight.normal_(0.0 if gemma else 1.0, 0.1)
        n.proj = torch.randn(hidden, hidden, device=DEV, dtype=torch.bfloat16) / hidden ** 0.5
        norms.append(n)
    return norms


@pytest.mark.parametrize("gemma", [False, True])
def test_forward_residual_compiles_into_one_graph(gemma):
    """The op is one node under dynamo: ``fullgraph=True`` fails on any graph
    break, and the compiled stack must match eager bit for bit."""
    hidden = 2048
    norms = _norms(hidden, gemma)
    h0 = torch.randn(8, hidden, device=DEV, dtype=torch.bfloat16)
    r0 = torch.randn(8, hidden, device=DEV, dtype=torch.bfloat16)
    eager = _stack(norms, h0.clone(), r0.clone())
    torch._dynamo.reset()
    compiled = torch.compile(_stack, fullgraph=True, dynamic=False)
    out = compiled(norms, h0.clone(), r0.clone())
    assert torch.equal(out[0], eager[0]) and torch.equal(out[1], eager[1])


def test_forward_residual_under_graph_capture():
    hidden = 3072
    norms = _norms(hidden, False)
    h = torch.randn(4, hidden, device=DEV, dtype=torch.bfloat16)
    r = torch.randn(4, hidden, device=DEV, dtype=torch.bfloat16)
    h_in, r_in = h.clone(), r.clone()
    expected = _stack(norms, h_in.clone(), r_in.clone())
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _stack(norms, h.clone(), r.clone())
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _stack(norms, h, r)
    h.copy_(h_in)
    r.copy_(r_in)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out[0], expected[0]) and torch.equal(out[1], expected[1])
