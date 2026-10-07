"""GLM-5.2 small-batch MoE decode kernels (moe_decode): the config flag and block wiring on
CPU; accuracy vs an fp32 dequantized oracle, routing and graph replay on CUDA, at one TP8
rank's shapes (H 6144, 256 experts, top-8, 256 intermediate per rank)."""
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


if "flashinfer" not in sys.modules:
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        sys.modules["flashinfer"] = types.ModuleType("flashinfer")
        sys.modules["flashinfer"].norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)

from mstar.model.glm52.components import moe as moe_module  # noqa: E402
from mstar.model.glm52.components.moe import Glm52MoEGate, Glm52SparseMoeBlock  # noqa: E402
from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.quantization import (  # noqa: E402
    Fp8BlockQuantConfig,
    dequantize_fp8_block_weight,
)

BLOCK = (128, 128)
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="moe_decode kernels need CUDA")


def test_decode_kernel_flag_defaults_off_and_reaches_the_block():
    from mstar.model.glm52.glm52_model import Glm52Model

    assert Glm52ModelConfig().moe_decode_kernel is False
    model = Glm52Model("unused", config_variant="reduced_fp8", moe_decode_kernel=True)
    assert model.config.moe_decode_kernel is True
    cfg = Glm52ModelConfig.reduced_fp8(block=(16, 16))
    assert Glm52SparseMoeBlock(cfg)._decode_kernel is False
    cfg.moe_decode_kernel = True
    assert Glm52SparseMoeBlock(cfg)._decode_kernel is True


def test_use_decode_needs_the_fused_path_and_a_cuda_batch():
    cfg = Glm52ModelConfig.reduced_fp8(block=(16, 16))
    cfg.moe_decode_kernel = True
    block = Glm52SparseMoeBlock(cfg)
    for p in block.parameters():
        if p.dtype == torch.uint8:
            p.data.copy_((torch.randn(p.shape) * 0.1).to(torch.float8_e4m3fn).view(torch.uint8))
        else:
            p.data.normal_(0, 0.05)
    block.process_weights_after_loading("cpu")  # reference: no fused path
    x = torch.randn(2, cfg.hidden_size) * 0.1
    assert not block._use_decode(x.bfloat16())
    assert torch.isfinite(block(x)).all()
    block._use_fused = True
    assert not block._use_decode(x.bfloat16())  # host tensor


def test_prefill_kernel_flag_defaults_off_and_reaches_the_block():
    from mstar.model.glm52.glm52_model import Glm52Model

    assert Glm52ModelConfig().moe_prefill_kernel is False
    model = Glm52Model("unused", config_variant="reduced_fp8", moe_prefill_kernel=True)
    assert model.config.moe_prefill_kernel is True
    cfg = Glm52ModelConfig.reduced_fp8(block=(16, 16))
    cfg.moe_prefill_kernel = True
    assert Glm52SparseMoeBlock(cfg)._prefill_kernel is True
    fn = Glm52SparseMoeBlock._dispatch_prefill
    assert hasattr(fn, "_torchdynamo_disable") or hasattr(fn, "_torchdynamo_orig_callable")


def test_forward_decode_is_compiler_disabled():
    """The PDL launch chain stays outside dynamo (the captured decode is compiled)."""
    fn = Glm52SparseMoeBlock._forward_decode
    assert hasattr(fn, "_torchdynamo_disable") or hasattr(fn, "_torchdynamo_orig_callable")


def test_router_split_fits_glm52_hidden():
    moe_decode = pytest.importorskip("mstar.utils.fused_moe.decode")
    for tuning in (moe_decode._H100, moe_decode._H200):
        for h in (6144, 4096, 512, 384, 128):
            splits, bk = moe_decode._router_split(h, tuning)
            assert h % (splits * bk) == 0 and bk >= 16
    assert moe_decode._router_split(6144, moe_decode._H100) == (12, 128)


# ---------------------------------------------------------------- CUDA


@pytest.fixture
def fp32_matmul():
    """The oracle and the fp32 router reference need true fp32 matmuls, not TF32."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev)


def _quantize(w, block):
    """(E, N, K) fp32 -> (e4m3 bytes, fp32 scale_inv per block), dequant = w8 * scale."""
    E, N, K = w.shape
    bo, bi = block
    b = w.view(E, N // bo, bo, K // bi, bi)
    s = b.abs().amax(dim=(2, 4)) / 448.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = (b / s[:, :, None, :, None]).to(torch.float8_e4m3fn).view(E, N, K)
    return q.view(torch.uint8), s


def _random_fp8(param, scale, block, std, gen):
    """Fill fp8 expert weights with N(0, std) values times a log-uniform 1/4..4 factor per
    128 x 128 block, so scales vary across blocks as a checkpoint's do."""
    E, N, K = param.shape
    bo, bi = block
    for e0 in range(0, E, 32):
        e1 = min(E, e0 + 32)
        w = torch.randn(e1 - e0, N, K, device=param.device, generator=gen) * std
        f = torch.exp2(torch.rand(e1 - e0, N // bo, 1, K // bi, 1, device=param.device,
                                  generator=gen) * 4 - 2)
        w = (w.view(e1 - e0, N // bo, bo, K // bi, bi) * f).view(e1 - e0, N, K)
        param.data[e0:e1], scale.data[e0:e1] = _quantize(w, block)


def _block(*, hidden=6144, inter=256, experts=256, top_k=8, bias_std=0.02, seed=0):
    """One TP8 rank's fp8 MoE block on CUDA (a TP1 block of the per-rank intermediate)."""
    cfg = Glm52ModelConfig.reduced()
    cfg.hidden_size, cfg.moe_intermediate_size = hidden, inter
    cfg.n_routed_experts, cfg.num_experts_per_tok = experts, top_k
    cfg.quantization_config = Fp8BlockQuantConfig(weight_block_size=BLOCK)
    cfg.moe_quant_kernel = "triton"
    cfg.moe_decode_kernel = True
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            block = Glm52SparseMoeBlock(cfg)
    finally:
        torch.set_default_dtype(torch.float32)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    exp = block.experts
    with torch.no_grad():
        _random_fp8(exp.gate_up_proj_fp8, exp.gate_up_proj_scale_inv, BLOCK, 0.02, gen)
        _random_fp8(exp.down_proj_fp8, exp.down_proj_scale_inv, BLOCK, 0.02, gen)
        block.gate.weight.normal_(0.0, 0.02, generator=gen)
        block.gate.e_score_correction_bias.normal_(0.0, bias_std, generator=gen)
        for p in block.shared_expert.parameters():
            p.normal_(0.0, 0.02, generator=gen)
    block.process_weights_after_loading("cuda")
    assert block._use_fused and block._decode_kernel
    return block


@pytest.fixture(scope="module")
def rank_block():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return _block()


def _oracle(block, x, topk_w, topk_ids):
    """fp32 routed + shared expert output from the dequantized weights and fp32 routing, and
    each output's sum of |terms| of the down contraction (what act rounding scales with)."""
    exp, xs = block.experts, x.float()
    out = torch.zeros(x.shape, device=x.device)
    mag = torch.zeros(x.shape, device=x.device)
    for e in topk_ids.unique().tolist():
        gate_up = dequantize_fp8_block_weight(
            exp.gate_up_proj_fp8[e], exp.gate_up_proj_scale_inv[e],
            block_size=block.block_size, out_dtype=torch.float32)
        down = dequantize_fp8_block_weight(
            exp.down_proj_fp8[e], exp.down_proj_scale_inv[e],
            block_size=block.block_size, out_dtype=torch.float32)
        rows, slots = (topk_ids == e).nonzero(as_tuple=True)
        gate, up = (xs[rows] @ gate_up.T).chunk(2, dim=-1)
        act, tw = F.silu(gate) * up, topk_w[rows, slots, None].float()
        out.index_add_(0, rows, (act @ down.T) * tw)
        mag.index_add_(0, rows, (act.abs() @ down.abs().T) * tw.abs())
    shared = block.shared_expert
    gate, up = (xs @ shared.gate_up_proj.weight.float().T).chunk(2, dim=-1)
    act, down = F.silu(gate) * up, shared.down_proj.weight.float()
    return out + act @ down.T, mag + act.abs() @ down.abs().T


def _pair_max():
    from mstar.utils.fused_moe import decode as moe_decode

    return moe_decode._tuning(torch.device("cuda", torch.cuda.current_device())).pair_max_tokens


def _current(block, x):
    """The block's path without the flag: fp32 router, W8A8 runner, bf16 shared expert."""
    block._decode_kernel = False
    try:
        return block(x)
    finally:
        block._decode_kernel = True


def _rel_err(got, ref):
    return float((got.float() - ref).abs().max() / ref.abs().max())


def _assert_near_oracle(got, oracle, grouped):
    """w8a16 with fp32 accumulation: each output is off by its bf16 rounding (at most
    2^-8 |ref|) plus, on the grouped path, act rounded to bf16 for the down dot (at most 2^-8
    of the sum of |terms|) and the fp8 gate/up dot's 32-deep partial sums; 2^-7 covers
    both. The per-pair path keeps act in fp32: 2^-12 of |terms| covers fp32 summation."""
    ref, mag = oracle
    err = (got.float() - ref).abs()
    bound = 2**-8 * ref.abs() + (2**-7 if grouped else 2**-12) * mag + 1e-6 * mag.max()
    worst = float((err / bound).max())
    assert worst <= 1.0, f"error {worst:.2f}x the rounding bound"


@gpu
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 32, 64])
def test_decode_matches_oracle_at_rank_shapes(rank_block, tokens, fp32_matmul):
    block = rank_block
    torch.manual_seed(tokens)
    x = torch.randn(tokens, block.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        topk_w, topk_ids = block.gate(x)
        got = block._dispatch_decode(x, topk_w, topk_ids)
        oracle = _oracle(block, x, topk_w, topk_ids)
        old = _current(block, x)
        assert block._use_decode(x)
        chained = block(x)
    assert got.dtype == torch.bfloat16 and got.shape == x.shape
    grouped = tokens > _pair_max()
    _assert_near_oracle(got, oracle, grouped)
    new_err, old_err = _rel_err(got, oracle[0]), _rel_err(old, oracle[0])
    print(f"tokens {tokens}: max|err|/max|ref| current {old_err:.2e} decode {new_err:.2e}")
    assert new_err * 4 <= old_err
    # the chained forward routes with the kernel router: same experts, weights to ~1e-7
    _assert_near_oracle(chained, oracle, grouped)


@gpu
@pytest.mark.parametrize("tokens", [24, 64])
@pytest.mark.parametrize("skew", ["same8", "zipf"])
def test_grouped_skewed_routing(rank_block, tokens, skew, fp32_matmul):
    # Real routing concentrates: an expert can hold every token, i.e. several row tiles.
    block = rank_block
    torch.manual_seed(7)
    x = torch.randn(tokens, block.hidden_size, device="cuda", dtype=torch.bfloat16)
    if skew == "same8":
        topk_ids = torch.arange(8, device="cuda").expand(tokens, 8)
    else:
        p = 1.0 / torch.arange(1, 33, device="cuda", dtype=torch.float32)
        topk_ids = torch.multinomial(p.expand(tokens, 32), 8)
    topk_w = torch.rand(tokens, 8, device="cuda")
    with torch.no_grad():
        got = block._dispatch_decode(x, topk_w, topk_ids)
        oracle = _oracle(block, x, topk_w, topk_ids)
    _assert_near_oracle(got, oracle, grouped=True)


@gpu
@pytest.mark.parametrize("tokens", [11, 64])
def test_grouped_x_row_scales(rank_block, tokens, fp32_matmul):
    # The grouped gate/up meets x as two e4m3 terms under one power-of-two scale per row:
    # rows four decades apart, a zero row and a row with one large outlier must each stay
    # within the oracle bound; the unclamped act of the outlier row enters the down as bf16.
    block = rank_block
    torch.manual_seed(3)
    x = torch.randn(tokens, block.hidden_size, device="cuda")
    x *= torch.logspace(-2, 2, tokens, device="cuda")[:, None]
    x[1] = 0.0
    x[2, 7] = 3.0e4
    x = x.to(torch.bfloat16)
    with torch.no_grad():
        topk_w, topk_ids = block.gate(x)
        got = block._dispatch_decode(x, topk_w, topk_ids).float()
        ref, mag = _oracle(block, x, topk_w, topk_ids)
    assert torch.equal(got[1], torch.zeros_like(got[1]))
    rows = [r for r in range(tokens) if r != 1]
    _assert_near_oracle(got[rows], (ref[rows], mag[rows]), grouped=True)


@gpu
@pytest.mark.parametrize("tokens", [5, 20])
def test_odd_expert_count_and_top_k(tokens, fp32_matmul):
    block = _block(experts=7, top_k=3, hidden=512, inter=128)
    x = torch.randn(tokens, block.hidden_size, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.stack([torch.randperm(7, device="cuda")[:3] for _ in range(tokens)])
    topk_w = torch.rand(tokens, 3, device="cuda")
    with torch.no_grad():
        got = block._dispatch_decode(x, topk_w, topk_ids)
        oracle = _oracle(block, x, topk_w, topk_ids)
    _assert_near_oracle(got, oracle, grouped=tokens > _pair_max())


@gpu
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 32, 64])
def test_router_matches_the_fp32_gate(rank_block, tokens, fp32_matmul):
    from mstar.utils.fused_moe import decode as moe_decode

    gate = rank_block.gate
    for seed in range(3):
        torch.manual_seed(seed)
        x = torch.randn(tokens, gate.hidden_size, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            w, ids = moe_decode.route(
                x, gate.weight, gate.e_score_correction_bias, top_k=gate.top_k,
                scale=gate.routed_scaling_factor, normalize=gate.norm_topk_prob)
            w_r, ids_r = gate(x)
        assert w.dtype == torch.float32 and ids.dtype == torch.int64
        # same experts per token (the reference's topk order is unspecified)
        ids, order = ids.sort(dim=-1)
        ids_r, order_r = ids_r.sort(dim=-1)
        torch.testing.assert_close(ids, ids_r, rtol=0, atol=0)
        torch.testing.assert_close(w.gather(1, order), w_r.gather(1, order_r),
                                   rtol=1e-5, atol=1e-6)


@gpu
def test_router_fp32_operands_and_odd_sizes(fp32_matmul):
    from mstar.utils.fused_moe import decode as moe_decode

    torch.manual_seed(1)
    gate = Glm52MoEGate(512, 7, 3, routed_scaling_factor=2.5).cuda()
    with torch.no_grad():
        gate.weight.normal_(0.0, 0.05)
        gate.e_score_correction_bias.normal_(0.0, 0.5)
    x = torch.randn(5, 512, device="cuda")
    w, ids = moe_decode.route(x, gate.weight, gate.e_score_correction_bias, top_k=3,
                              scale=2.5, normalize=True)
    w_r, ids_r = gate(x)
    ids, order = ids.sort(dim=-1)
    ids_r, order_r = ids_r.sort(dim=-1)
    torch.testing.assert_close(ids, ids_r, rtol=0, atol=0)
    torch.testing.assert_close(w.gather(1, order), w_r.gather(1, order_r), rtol=1e-5, atol=1e-6)


@gpu
@pytest.mark.parametrize("tokens", [4, 32])
def test_routing_weights_stay_fp32(rank_block, tokens):
    """No path rounds the combine weights: the router emits fp32 and the experts consume them
    as given (bf16-rounded weights give a different output)."""
    from mstar.utils.fused_moe import decode as moe_decode

    block, gate = rank_block, rank_block.gate
    torch.manual_seed(5)
    x = torch.randn(tokens, block.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        topk_w, topk_ids = moe_decode.route(
            x, gate.weight, gate.e_score_correction_bias, top_k=gate.top_k,
            scale=gate.routed_scaling_factor, normalize=gate.norm_topk_prob)
        assert topk_w.dtype == torch.float32
        assert not torch.equal(topk_w, topk_w.bfloat16().float())
        exact = block._dispatch_decode(x, topk_w, topk_ids)
        rounded = block._dispatch_decode(x, topk_w.bfloat16().float(), topk_ids)
        chained = block(x)
    assert not torch.equal(exact, rounded)
    torch.testing.assert_close(chained, exact, rtol=0, atol=0)


@gpu
def test_large_batch_and_flag_off_take_the_current_path(rank_block):
    from mstar.utils.fused_moe import fused_experts_fp8

    block = rank_block
    for tokens in (moe_module._DECODE_MAX_TOKENS + 1, 4):
        x = torch.randn(tokens, block.hidden_size, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            if tokens > moe_module._DECODE_MAX_TOKENS:
                assert not block._use_decode(x)
                got = block(x)
            else:
                got = _current(block, x)
            topk_w, topk_ids = block.gate(x)
            exp = block.experts
            old = fused_experts_fp8(
                x, exp.gate_up_proj_fp8, exp.down_proj_fp8, exp.gate_up_proj_scale_inv,
                exp.down_proj_scale_inv, topk_w, topk_ids,
                block_size=block.block_size) + block.shared_expert(x)
        torch.testing.assert_close(got, old, rtol=0, atol=0)


@gpu
@pytest.mark.parametrize("tokens", [1, 4, 10, 11, 32, 64])
def test_graph_replay_matches_unchained_path(rank_block, tokens):
    # The captured forward runs the PDL chain; route + experts launch without it. Fresh
    # inputs every replay, so a kernel reading its predecessor's output early shows up.
    from mstar.utils.fused_moe import decode as moe_decode

    block, gate = rank_block, rank_block.gate
    x = torch.randn(tokens, block.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            block(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = block(x)
        for _ in range(20):
            x.copy_(torch.randn_like(x))
            graph.replay()
            routing = moe_decode.route(
                x, gate.weight, gate.e_score_correction_bias, top_k=gate.top_k,
                scale=gate.routed_scaling_factor, normalize=gate.norm_topk_prob)
            ref = block._dispatch_decode(x, *routing)
            torch.testing.assert_close(out, ref, rtol=0, atol=0)


@gpu
@pytest.mark.parametrize("tokens", [65, 300, 2048])
def test_prefill_kernel_keeps_the_runner_bits(rank_block, tokens):
    """moe_prefill keeps fused_experts_fp8's tile rows and K order: the block's output is
    the same with and without it."""
    block = rank_block
    torch.manual_seed(tokens)
    x = torch.randn(tokens, block.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        old = block(x)
        block._prefill_kernel = True
        try:
            new = block(x)
        finally:
            block._prefill_kernel = False
    torch.testing.assert_close(new, old, rtol=0, atol=0)
