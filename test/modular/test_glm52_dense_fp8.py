"""GLM-5.2 ``dense_fp8`` kernels at one TP8 rank's shapes: each path (w8a16 FMA and dot,
flashinfer and cuBLAS W8A8) against a float64 oracle from the dequantized weights under a
rounding bound, the dispatch by batch size, graph replay and torch.compile."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.model.glm52.quantization import dequantize_fp8_block_weight  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="dense_fp8 kernels need CUDA")

BLOCK = (128, 128)
# (weight rows, K, glu) at one TP8 rank: fused q_a+kv_a (replicated), q_b (8 heads x 256),
# o_proj (K = 8 heads x 256), shared expert gate+up / down, dense-layer gate+up / down
SHAPES = {
    "qkv_a": (2624, 6144, False), "q_b": (2048, 2048, False), "o_proj": (6144, 2048, False),
    "shared_gu": (512, 6144, True), "shared_down": (6144, 256, False),
    "dense_gu": (3072, 6144, True), "dense_down": (6144, 1536, False),
}


def _random_fp8(rows, cols, gen, block=BLOCK):
    """N(0, 0.02) times a log-uniform 1/4..4 factor per block (scales vary as a
    checkpoint's do), quantized per block; a ragged last row block keeps its own amax."""
    bo, bi = block
    nb = -(-rows // bo)
    w = torch.randn(nb * bo, cols, device="cuda", generator=gen) * 0.02
    f = torch.exp2(torch.rand(nb, 1, cols // bi, 1, device="cuda", generator=gen) * 4 - 2)
    w = (w.view(nb, bo, cols // bi, bi) * f)
    w[-1, rows - (nb - 1) * bo:] = 0.0
    s = w.abs().amax(dim=(1, 3)) / 448.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = (w / s[:, None, :, None]).to(torch.float8_e4m3fn).view(nb * bo, cols)[:rows]
    return q.contiguous().view(torch.uint8), s.contiguous()


def _x(tokens, cols, gen):
    """Normed-hidden-like rows: N(0, 1) with a few 20x outlier channels."""
    x = torch.randn(tokens, cols, device="cuda", generator=gen)
    x[:, torch.randint(0, cols, (max(1, cols // 512),), device="cuda", generator=gen)] *= 20
    return x.bfloat16()


def _oracle(x, q, s, glu):
    """float64 output; per element sum |x_k w_k| (the w8a16 accumulation term); sum
    |w_k| delta_k with delta_k the e4m3 rounding bound of x_k under its 128-group scale (the
    W8A8 activation term); and under glu, what bf16 rounding of the gate and up outputs
    moves silu(g) * u by (the W8A8 path rounds them before its SwiGLU). Under glu the first
    two propagate through silu(g) * u."""
    xd = x.double()
    wd = dequantize_fp8_block_weight(q, s, block_size=BLOCK, out_dtype=torch.float32).double()
    g = xd.abs().view(x.shape[0], -1, 128).amax(-1, keepdim=True) / 448.0
    delta = torch.maximum(xd.abs().view(x.shape[0], -1, 128) * 2.0**-4, g * 2.0**-9)
    y, mag, amag = xd @ wd.T, xd.abs() @ wd.abs().T, delta.view_as(xd) @ wd.abs().T
    imag = torch.zeros_like(y)
    if glu:
        n = y.shape[1] // 2
        gate, up = y[:, :n], y[:, n:]
        act = gate * torch.sigmoid(gate)
        # |d(silu(g) u)| <= 1.1 |u| dg + |silu(g)| du   (|silu'| <= 1.1)
        mag = 1.1 * up.abs() * mag[:, :n] + act.abs() * mag[:, n:]
        amag = 1.1 * up.abs() * amag[:, :n] + act.abs() * amag[:, n:]
        imag = 1.1 * up.abs() * gate.abs() + act.abs() * up.abs()
        y = act * up
    return y, mag, amag, imag


def _bf16_copies(x, q, s, glu):
    """Today's numerics: the weight dequantized to bf16 at load, a bf16 linear."""
    y = F.linear(x, dequantize_fp8_block_weight(q, s, block_size=BLOCK))
    if glu:
        g, u = y.chunk(2, dim=-1)
        y = F.silu(g) * u
    return y


def _worst(got, ref, bound):
    return float(((got.double() - ref).abs() / bound).max())


@pytest.fixture(scope="module")
def dense_fp8():
    from mstar.model.glm52 import dense_fp8

    dense_fp8.reserve("cuda")
    return dense_fp8


@pytest.fixture(scope="module")
def rank_weights(dense_fp8):
    gen = torch.Generator(device="cuda").manual_seed(0)
    return {name: _random_fp8(n, k, gen) for name, (n, k, _) in SHAPES.items()}


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 16, 32, 64, 100])
def test_w8a16_within_the_rounding_bound(dense_fp8, rank_weights, name, tokens):
    """w8a16: exact products, fp32 accumulation over K terms, one bf16 rounding of the
    output (2^-8 |ref|); the bf16 copies add a 2^-8 weight rounding, so fp8 must also beat
    them in rms."""
    q, s = rank_weights[name]
    n, k, glu = SHAPES[name]
    gen = torch.Generator(device="cuda").manual_seed(tokens)
    x = _x(tokens, k, gen)
    got = dense_fp8._w8a16(x, q, s, BLOCK, glu)
    ref, mag, _, _ = _oracle(x, q, s, glu)
    assert got.dtype == torch.bfloat16 and got.shape == ref.shape
    assert _worst(got, ref, 2.0**-8 * ref.abs() + k * 2.0**-24 * mag + 1e-30) <= 1.0
    bf16 = _bf16_copies(x, q, s, glu)
    rms = [float((y.double() - ref).pow(2).mean().sqrt()) for y in (got, bf16)]
    assert rms[0] <= rms[1], f"fp8 rms err {rms[0]:.3e} vs bf16 copies {rms[1]:.3e}"


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("path, tokens", [("fi", 33), ("fi", 300), ("cublas", 65),
                                          ("cublas", 1025), ("cublas", 2048)])
def test_w8a8_within_the_rounding_bound(dense_fp8, rank_weights, name, path, tokens):
    """W8A8: x rounded to e4m3 under its 128-group scale (2^-4 relative, or 2^-9 of the
    scale for subnormals, allowing a power-of-two rounded scale), then exact products; the
    fp8 MMA's partial sums get 2^-11 of sum |x w|; under glu the gate and up outputs round
    to bf16 before the SwiGLU."""
    q, s = rank_weights[name]
    n, k, glu = SHAPES[name]
    device = torch.device("cuda", torch.cuda.current_device())
    if not (dense_fp8._FI_OK if path == "fi" else dense_fp8._W8A8_OK).get(device):
        pytest.skip(f"the {path} block-scaled fp8 GEMM is not available")
    gen = torch.Generator(device="cuda").manual_seed(tokens)
    x = _x(tokens, k, gen)
    got = dense_fp8._fi(x, q, s, glu) if path == "fi" else dense_fp8._w8a8(x, q, s, BLOCK, glu)
    ref, mag, amag, imag = _oracle(x, q, s, glu)
    bound = 2.0**-8 * (ref.abs() + imag) + 1.07 * amag + 2.0**-11 * mag + 1e-30
    assert got.shape == ref.shape and _worst(got, ref, bound) <= 1.0


def test_dispatch_by_tokens(dense_fp8, rank_weights, monkeypatch):
    """w8a16 for decode batches (longer for the shared expert's small GEMMs), then
    flashinfer, then cuBLAS."""
    seen = []
    for fn in ("_w8a16", "_fi", "_w8a8"):
        monkeypatch.setattr(dense_fp8, fn, lambda *a, fn=fn, **k: seen.append(fn))
    monkeypatch.setitem(dense_fp8._FI_OK, torch.device("cuda", torch.cuda.current_device()),
                        True)
    for name, tokens in (("qkv_a", 32), ("qkv_a", 33), ("qkv_a", 512), ("qkv_a", 513),
                         ("shared_gu", 128), ("shared_down", 129)):
        q, s = rank_weights[name]
        dense_fp8._linear(torch.zeros(tokens, SHAPES[name][1], device="cuda",
                                      dtype=torch.bfloat16), q, s, BLOCK, SHAPES[name][2])
    assert seen == ["_w8a16", "_fi", "_fi", "_w8a8", "_w8a16", "_fi"]


def test_graph_replay_is_bitwise_and_leaves_the_counters_at_zero(dense_fp8, rank_weights):
    """Split-K programs finish in any order; the last one sums the partials in split order,
    so every replay gives the same bits."""
    q, s = rank_weights["shared_gu"]
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = _x(4, 6144, gen)
    cfg = dict(BN=16, BK=128, SPLIT=16, DOT=True, STAGES=3, num_warps=4)
    eager = dense_fp8._w8a16(x, q, s, BLOCK, True, cfg)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dense_fp8._w8a16(x, q, s, BLOCK, True, cfg)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = dense_fp8._w8a16(x, q, s, BLOCK, True, cfg)
    for _ in range(20):
        graph.replay()
        assert torch.equal(out, eager)
    torch.cuda.synchronize()
    assert int(dense_fp8._COUNTERS[x.device].abs().sum()) == 0


def test_op_compiles_whole(dense_fp8, rank_weights):
    """An opaque custom op: fullgraph compile succeeds and matches eager bit for bit."""
    q, s = rank_weights["o_proj"]
    x = _x(3, 2048, torch.Generator(device="cuda").manual_seed(2))

    def f(x):
        return dense_fp8.linear(x * 2.0, q, s, BLOCK) + 1.0

    torch._dynamo.reset()
    assert torch.equal(torch.compile(f, fullgraph=True)(x), f(x))


def test_an_unusable_cublas_block_gemm_falls_back(monkeypatch):
    """torch 2.10-2.12 have the API, but blockwise scaled_mm raises off SM90 or on a
    cuBLASLt before 12.9 (the cu128 builds): every prefill step past 512 tokens failed."""
    from mstar.model.glm52 import dense_fp8

    def raises(*args, **kwargs):
        raise NotImplementedError("blockwise scaling needs cuBLASLt >= 12.9")

    monkeypatch.setattr(dense_fp8, "_W8A8_API", True)
    monkeypatch.setattr(dense_fp8, "_w8a8", raises)
    assert dense_fp8._probe_w8a8(torch.device("cpu")) is False
    taken = []
    monkeypatch.setattr(dense_fp8, "_w8a16", lambda *args: taken.append(1) or None)
    monkeypatch.setattr(dense_fp8, "_W8A8_OK", {torch.device("cpu"): False})
    dense_fp8._linear(torch.zeros(600, 256, dtype=torch.bfloat16),
                      torch.zeros(256, 256, dtype=torch.uint8), torch.ones(2, 2), (128, 128), False)
    assert taken == [1]
