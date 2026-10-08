"""The shared small-batch fp8 MoE kernels (fused_moe.decode): the per-GPU tables and the
SwiGLU switch reaching every launch, on any host; on CUDA, the unclamped path at a hidden size
that is not a power of two vs an fp32 oracle, under each GPU's table."""
from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn.functional as F

try:
    from mstar.utils.fused_moe import decode
except ModuleNotFoundError:  # the conftest's triton stand-in has no triton.language.extra
    _cuda = types.ModuleType("triton.language.extra.cuda")
    _cuda.gdc_wait = _cuda.gdc_launch_dependents = None
    sys.modules["triton.language.extra"] = types.ModuleType("triton.language.extra")
    sys.modules["triton.language.extra.cuda"] = _cuda
    try:
        from mstar.utils.fused_moe import decode
    finally:
        del sys.modules["triton.language.extra"], sys.modules["triton.language.extra.cuda"]

BLOCK = (128, 128)
TABLES = ["_H200", "_H100"]
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + triton")


@pytest.fixture
def launches(monkeypatch):
    """Every kernel launch as (name, grid, args, kwargs); nothing runs."""
    calls = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *args, **kw: calls.append((self.name, grid, args, kw))

    for name in ("_router_logits_kernel", "_router_topk_kernel", "_gate_up_kernel",
                 "_down_kernel", "_plan_kernel", "_grouped_gate_up_kernel",
                 "_grouped_down_kernel", "_sum_kernel"):
        monkeypatch.setattr(decode, name, Kernel(name))
    if not hasattr(decode.triton, "next_power_of_2"):
        monkeypatch.setattr(decode.triton, "next_power_of_2", lambda n: 1 << (n - 1).bit_length(),
                            raising=False)
    return calls


def _use(monkeypatch, table):
    tuning = getattr(decode, table)
    monkeypatch.setattr(decode, "_tuning", lambda device: tuning)
    return tuning


def _host_layer(tokens, hidden=256, inter=256, experts=16, top_k=4):
    """Expert and shared weights of the right shapes and dtypes, on the CPU."""
    u8 = dict(dtype=torch.uint8)
    return (torch.randn(tokens, hidden).bfloat16(),
            torch.zeros(experts, 2 * inter, hidden, **u8),
            torch.ones(experts, 2 * inter // 128, hidden // 128),
            torch.zeros(experts, hidden, inter, **u8),
            torch.ones(experts, hidden // 128, inter // 128),
            torch.zeros(2 * inter, hidden).bfloat16(), torch.zeros(hidden, inter).bfloat16(),
            torch.rand(tokens, top_k),
            torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)]))


def test_tuning_follows_the_gpu_name(monkeypatch):
    pick = decode._tuning.__wrapped__
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d=None: "NVIDIA H100 80GB HBM3")
    assert pick(torch.device("cuda", 0)) is decode._H100
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d=None: "NVIDIA H200")
    assert pick(torch.device("cuda", 0)) is decode._H200
    assert pick(torch.device("cpu")) is decode._H200


def test_tile_tables():
    h200, h100 = decode._H200, decode._H100
    assert decode._tiles(3, h200) == (dict(bj=8, bk=1024, bjs=32, bks=256, warps=4),
                                      dict(bn=32, bc=128, warps=4))
    assert decode._tiles(2, h200)[1] == dict(bn=32, bc=256, warps=8)
    assert decode._tiles(4, h200)[0] == dict(bj=16, bk=256, bjs=16, bks=128, warps=4)
    assert decode._tiles(1, h100)[0] == dict(bj=8, bk=512, bjs=32, bks=256, warps=2)
    assert decode._tiles(2, h100)[0] == decode._tiles(2, h200)[0]
    assert decode._tiles(3, h100)[1] == dict(bn=16, bc=256, warps=8)
    assert decode._tiles(4, h100)[1] == decode._tiles(4, h200)[1]


def test_router_split_cuts_hidden_into_whole_steps():
    assert decode._router_split(4096, decode._H200) == (16, 256)
    assert decode._router_split(6144, decode._H200) == (12, 256)
    assert decode._router_split(6144, decode._H100) == (12, 128)
    assert decode._router_split(4096, decode._H100) == (8, 128)
    assert decode._router_split(384, decode._H200) == (3, 128)
    for table in (decode._H200, decode._H100):
        for hidden in (6144, 4096, 512, 384, 128):
            splits, bk = decode._router_split(hidden, table)
            assert hidden % (splits * bk) == 0


@pytest.mark.parametrize("table", TABLES)
def test_pair_bound_picks_the_path(table, launches, monkeypatch):
    tuning = _use(monkeypatch, table)
    for tokens in (tuning.pair_max_tokens, tuning.pair_max_tokens + 1):
        launches.clear()
        decode.experts(*_host_layer(tokens), block_size=BLOCK)
        grouped = ["_plan_kernel", "_grouped_gate_up_kernel", "_grouped_down_kernel",
                   "_sum_kernel"]
        pair = ["_gate_up_kernel", "_down_kernel"]
        assert [c[0] for c in launches] == (grouped if tokens > tuning.pair_max_tokens else pair)


@pytest.mark.parametrize("tokens", [2, 16])
@pytest.mark.parametrize("limit", [None, 7.0])
def test_swiglu_limit_reaches_every_activation(tokens, limit, launches):
    decode.experts(*_host_layer(tokens), block_size=BLOCK, swiglu_limit=limit)
    clamped = {n: kw["CLAMP"] for n, _, _, kw in launches if "CLAMP" in kw}
    expect = (["_grouped_gate_up_kernel", "_grouped_down_kernel"] if tokens > 10
              else ["_gate_up_kernel"])
    assert clamped == dict.fromkeys(expect, limit is not None)
    for name, _, args, _ in launches:
        if "gate_up" in name:
            floats = [a for a in args if isinstance(a, float)]
            assert floats == [7.0 if limit else 0.0]


def test_split_x_rounds_hidden_up_to_a_power_of_two(launches):
    decode.experts(*_host_layer(16, hidden=3072), block_size=BLOCK)
    (plan,) = [kw for name, _, _, kw in launches if name == "_plan_kernel"]
    assert plan["H"] == 3072 and plan["HP"] == 4096


@pytest.mark.parametrize("table", TABLES)
def test_router_launch_follows_the_table(table, launches, monkeypatch):
    tuning = _use(monkeypatch, table)
    x, w = torch.randn(3, 6144).bfloat16(), torch.randn(64, 6144).bfloat16()
    decode.route(x, w, torch.zeros(64), top_k=8, scale=1.0, normalize=True)
    (_, grid, _, logits), (_, _, _, topk) = launches
    splits, bk = decode._router_split(6144, tuning)
    assert grid == (2 * splits,) and logits["S"] == topk["S"] == splits
    assert logits["BK"] == bk and logits["num_warps"] == tuning.router["warps"]


def test_router_topk_op(launches):
    w, ids = decode.router_topk(torch.randn(5, 64), torch.zeros(64), 8, 2.5, True)
    assert w.shape == ids.shape == (5, 8) and ids.dtype == torch.int64
    (_, grid, _, kw), = launches
    assert grid == (5,) and kw["S"] == 1 and kw["NORM"] is True
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        w, ids = decode.router_topk(torch.empty(5, 64), torch.empty(64), 8, 2.5, True)
    assert w.shape == (5, 8) and ids.dtype == torch.int64


# ---------------------------------------------------------------- CUDA


@pytest.fixture
def fp32_matmul():
    """mstar.engine sets TF32 matmuls globally; the oracle needs true fp32."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev)


def _quantize(w):
    """(E, N, K) fp32 -> (e4m3 bytes, fp32 scale per 128 x 128 block)."""
    E, N, K = w.shape
    b = w.view(E, N // 128, 128, K // 128, 128)
    s = b.abs().amax(dim=(2, 4)) / 448.0
    q = (b / s[:, :, None, :, None]).to(torch.float8_e4m3fn).view(E, N, K)
    return q.view(torch.uint8), s


@pytest.fixture(scope="module")
def layer():
    """One TP8 rank of a GLM-5.2-shaped layer (hidden 6144), fewer experts; per-block scales
    vary by 16x, as a checkpoint's do."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    gen = torch.Generator(device="cuda").manual_seed(0)
    E, H, I = 32, 6144, 256

    def fp8(n, k):
        w = torch.randn(E, n, k, device="cuda", generator=gen) * 0.03
        f = torch.exp2(torch.rand(E, n // 128, 1, k // 128, 1, device="cuda", generator=gen) * 4
                       - 2)
        return _quantize((w.view(E, n // 128, 128, k // 128, 128) * f).view(E, n, k))

    w13, s13 = fp8(2 * I, H)
    w2, s2 = fp8(H, I)
    sw13 = (torch.randn(2 * I, H, device="cuda", generator=gen) * 0.02).bfloat16()
    sw2 = (torch.randn(H, I, device="cuda", generator=gen) * 0.02).bfloat16()
    gate = (torch.randn(E, H, device="cuda", generator=gen) * 0.02).bfloat16()
    bias = torch.randn(E, device="cuda", generator=gen) * 0.02
    return dict(w13=w13, s13=s13, w2=w2, s2=s2, sw13=sw13, sw2=sw2, gate=gate, bias=bias)


def _experts(p):
    return p["w13"], p["s13"], p["w2"], p["s2"], p["sw13"], p["sw2"]


def _dequant(q, s):
    E, N, K = q.shape
    w = q.view(torch.float8_e4m3fn).float().view(E, N // 128, 128, K // 128, 128)
    return (w * s[:, :, None, :, None]).view(E, N, K)


def _swiglu(gate, up, limit):
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return F.silu(gate) * up


def _oracle(p, x, topk_w, topk_ids, limit):
    """fp32 routed + shared output, and each output's sum of |terms| of the down contraction
    (what act rounding scales with)."""
    w13, w2 = _dequant(p["w13"], p["s13"]), _dequant(p["w2"], p["s2"])
    xs = x.float()
    out, mag = torch.zeros_like(xs), torch.zeros_like(xs)
    for e in topk_ids.unique().tolist():
        rows, slots = (topk_ids == e).nonzero(as_tuple=True)
        act = _swiglu(*(xs[rows] @ w13[e].T).chunk(2, dim=-1), limit)
        tw = topk_w[rows, slots, None]
        out.index_add_(0, rows, (act @ w2[e].T) * tw)
        mag.index_add_(0, rows, (act.abs() @ w2[e].abs().T) * tw.abs())
    act = _swiglu(*(xs @ p["sw13"].float().T).chunk(2, dim=-1), limit)
    down = p["sw2"].float()
    return out + act @ down.T, mag + act.abs() @ down.abs().T


def _assert_near_oracle(got, oracle, grouped):
    """Off by the bf16 output rounding (2^-8 |ref|) plus, grouped, act rounded for the down
    dot and the fp8 gate/up dot's 32-deep partial sums (2^-7 of the |terms|); per pair, act
    stays fp32 (2^-12 of the |terms| covers fp32 summation)."""
    ref, mag = oracle
    bound = 2**-8 * ref.abs() + (2**-7 if grouped else 2**-12) * mag + 1e-6 * mag.max()
    worst = float(((got.float() - ref).abs() / bound).max())
    assert worst <= 1.0, f"error {worst:.2f}x the rounding bound"


def _routing(tokens, experts=32, top_k=8):
    ids = torch.stack([torch.randperm(experts, device="cuda")[:top_k] for _ in range(tokens)])
    return torch.rand(tokens, top_k, device="cuda"), ids


@gpu
@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("limit", [None, 10.0])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 7, 8, 11, 33, 64])
def test_matches_oracle_at_hidden_6144(layer, table, limit, tokens, monkeypatch, fp32_matmul):
    tuning = _use(monkeypatch, table)
    torch.manual_seed(tokens)
    x = torch.randn(tokens, 6144, device="cuda").bfloat16()
    topk_w, topk_ids = _routing(tokens)
    got = decode.experts(x, *_experts(layer), topk_w, topk_ids, block_size=BLOCK,
                         swiglu_limit=limit)
    assert got.dtype == torch.bfloat16 and got.shape == x.shape
    oracle = _oracle(layer, x, topk_w, topk_ids, limit)
    _assert_near_oracle(got, oracle, grouped=tokens > tuning.pair_max_tokens)


@gpu
@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("tokens", [1, 3, 8, 11, 64])
def test_chain_matches_route_then_experts(layer, table, tokens, monkeypatch):
    _use(monkeypatch, table)
    x = torch.randn(tokens, 6144, device="cuda").bfloat16()
    kw = dict(top_k=8, scale=2.5, normalize=True)
    chained = decode.forward(x, layer["gate"], layer["bias"], *_experts(layer), **kw,
                             block_size=BLOCK)
    topk_w, topk_ids = decode.route(x, layer["gate"], layer["bias"], **kw)
    ref = decode.experts(x, *_experts(layer), topk_w, topk_ids, block_size=BLOCK)
    torch.testing.assert_close(chained, ref, rtol=0, atol=0)


def _torch_routing(logits, bias, top_k, scale):
    scores = logits.sigmoid()
    ids = torch.topk(scores + bias, k=top_k, dim=-1)[1]
    w = scores.gather(1, ids)
    return w / w.sum(-1, keepdim=True) * scale, ids


def _assert_same_routing(got, ref):
    (w, ids), (w_r, ids_r) = got, ref
    ids, order = ids.sort(dim=-1)
    ids_r, order_r = ids_r.sort(dim=-1)
    torch.testing.assert_close(ids, ids_r, rtol=0, atol=0)
    torch.testing.assert_close(w.gather(1, order), w_r.gather(1, order_r), rtol=1e-5, atol=1e-6)


@gpu
@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("tokens", [1, 5, 64])
def test_router_at_hidden_6144(layer, table, tokens, monkeypatch, fp32_matmul):
    _use(monkeypatch, table)
    x = torch.randn(tokens, 6144, device="cuda").bfloat16()
    logits = x.float() @ layer["gate"].float().T
    ref = _torch_routing(logits, layer["bias"], 8, 2.5)
    got = decode.route(x, layer["gate"], layer["bias"], top_k=8, scale=2.5, normalize=True)
    _assert_same_routing(got, ref)
    _assert_same_routing(decode.router_topk(logits, layer["bias"], 8, 2.5, True), ref)


@gpu
@pytest.mark.parametrize("tokens", [8, 16])
def test_grouped_path_at_tp1_expert_width(tokens, monkeypatch, fp32_matmul):
    # one rank holding a whole expert (I = 2048): the down loop must not stage every block
    tuning = _use(monkeypatch, "_H100")
    gen = torch.Generator(device="cuda").manual_seed(1)
    E, H, I = 8, 1024, 2048

    def fp8(n, k):
        return _quantize(torch.randn(E, n, k, device="cuda", generator=gen) * 0.03)

    w13, s13 = fp8(2 * I, H)
    w2, s2 = fp8(H, I)
    p = dict(w13=w13, s13=s13, w2=w2, s2=s2,
             sw13=(torch.randn(2 * I, H, device="cuda", generator=gen) * 0.02).bfloat16(),
             sw2=(torch.randn(H, I, device="cuda", generator=gen) * 0.02).bfloat16())
    x = torch.randn(tokens, H, device="cuda", generator=gen).bfloat16()
    topk_w, topk_ids = _routing(tokens, experts=E, top_k=4)
    got = decode.experts(x, *_experts(p), topk_w, topk_ids, block_size=BLOCK)
    assert tokens > tuning.pair_max_tokens
    _assert_near_oracle(got, _oracle(p, x, topk_w, topk_ids, None), grouped=True)


def test_the_router_refuses_more_tokens_than_its_tile_holds():
    """All T tokens share one router tile; at 129+ on the H200 table the first launch
    raised OutOfResources. The models send at most 64 tokens here."""
    T = decode.MAX_ROUTER_TOKENS + 1
    with pytest.raises(ValueError, match="at most"):
        decode.route(torch.empty(T, 256), torch.empty(8, 256), torch.empty(8), top_k=2,
                     scale=1.0, normalize=True)


def test_strided_block_scales_are_refused():
    """The kernels index the scales as dense blocks past their leading stride: a TP slice
    kept as a view applied other blocks' scales, silently."""
    H, I, E = 256, 128, 2
    x = torch.zeros(1, H, dtype=torch.bfloat16)
    w13, w2 = torch.zeros(E, 2 * I, H, dtype=torch.uint8), torch.zeros(E, H, I, dtype=torch.uint8)
    s13 = torch.ones(E, 2 * I // 128, H // 128)
    s2 = torch.ones(E, H // 128, 2 * I // 128)[:, :, : I // 128]
    sw13, sw2 = torch.zeros(2 * I, H, dtype=torch.bfloat16), torch.zeros(H, I, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="contiguous"):
        decode._check_experts(x, w13, s13, w2, s2, sw13, sw2, (128, 128))
    decode._check_experts(x, w13, s13, w2, s2.contiguous(), sw13, sw2, (128, 128))
