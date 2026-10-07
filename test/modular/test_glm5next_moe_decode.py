"""glm5_next small-batch MoE decode kernels (moe_decode) vs an fp32 oracle on CUDA."""
import pytest
import torch
import torch.nn.functional as F

from mstar.model.glm5_next import fused_decode
from mstar.model.glm5_next.components import moe as moe_module
from mstar.model.glm5_next.components.moe import Glm5NextMoEGate, Glm5NextSparseMoeBlock
from mstar.model.glm5_next.config import Glm5NextModelConfig
from mstar.model.glm5_next.quantization import Fp8BlockQuantConfig, dequantize_fp8_block_weight

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused_decode._HAS_TRITON),
    reason="moe_decode kernels need CUDA + triton",
)


@pytest.fixture
def fp32_matmul():
    """mstar.engine sets TF32 matmuls globally; the oracle needs true fp32."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev)


def _block(*, hidden=4096, inter=256, experts=32, top_k=8, limit=10.0, seed=0):
    """One TP rank's fp8 MoE block (real per-rank dims, fewer experts)."""
    cfg = Glm5NextModelConfig.reduced()
    cfg.hidden_size, cfg.moe_intermediate_size = hidden, inter
    cfg.n_routed_experts, cfg.num_experts_per_tok = experts, top_k
    cfg.swiglu_limit = limit
    cfg.quantization_config = Fp8BlockQuantConfig(weight_block_size=(128, 128))
    cfg.moe_quant_kernel = "auto"
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.bfloat16)
    try:
        block = Glm5NextSparseMoeBlock(cfg).cuda()
    finally:
        torch.set_default_dtype(torch.float32)
    with torch.no_grad():
        for name, p in block.named_parameters():
            if p.dtype == torch.uint8:
                w = torch.randn(p.shape, device="cuda") * 0.06
                p.copy_(w.to(torch.float8_e4m3fn).view(torch.uint8))
            elif "scale_inv" in name:
                p.uniform_(0.5, 2.0)
            elif name.endswith("e_score_correction_bias"):
                p.normal_(0.0, 0.5)
            else:
                p.normal_(0.0, 0.02)
    block.process_weights_after_loading("cuda")
    assert block._use_fused
    return block


def _oracle(block, x, topk_w, topk_ids):
    """fp32 routed-expert output from the dequantized weights."""
    exp, limit = block.experts, block.swiglu_limit
    xs, out = x.float(), torch.zeros(x.shape, device=x.device)
    for e in topk_ids.unique().tolist():
        gate_up = dequantize_fp8_block_weight(
            exp.gate_up_proj_fp8[e], exp.gate_up_proj_scale_inv[e],
            block_size=block.block_size, out_dtype=torch.float32)
        down = dequantize_fp8_block_weight(
            exp.down_proj_fp8[e], exp.down_proj_scale_inv[e],
            block_size=block.block_size, out_dtype=torch.float32)
        rows, slots = (topk_ids == e).nonzero(as_tuple=True)
        gate, up = (xs[rows] @ gate_up.T).chunk(2, dim=-1)
        act = F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
        out.index_add_(0, rows, (act @ down.T) * topk_w[rows, slots, None])
    return out


def _shared_oracle(block, x):
    shared, limit = block.shared_expert, block.swiglu_limit
    gate, up = (x.float() @ shared.gate_up_proj.weight.float().T).chunk(2, dim=-1)
    act = F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
    return act @ shared.down_proj.weight.float().T


def _runner(block, x, topk_w, topk_ids):
    from mstar.utils.fused_moe import fused_experts_fp8

    exp = block.experts
    return fused_experts_fp8(
        x, exp.gate_up_proj_fp8, exp.down_proj_fp8, exp.gate_up_proj_scale_inv,
        exp.down_proj_scale_inv, topk_w, topk_ids, block_size=block.block_size,
        swiglu_limit=block.swiglu_limit)


def _rel_err(got, ref):
    return float((got.float() - ref).abs().max() / ref.abs().max())


def _assert_near_oracle(got, ref):
    # w8a16 with fp32 accumulation: only the final bf16 rounding separates it from the oracle.
    torch.testing.assert_close(got.float(), ref, rtol=5e-3, atol=1e-3 * float(ref.abs().max()))


@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 8, 10, 11, 16, 33, 64])
@pytest.mark.parametrize("limit", [10.0, 2.0])
def test_decode_matches_oracle(tokens, limit, fp32_matmul):
    block = _block(limit=limit)
    x = torch.randn(tokens, block.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        topk_w, topk_ids = block.gate(x)
        assert topk_w.dtype == torch.float32
        got = block._dispatch_decode(x, topk_w, topk_ids)
        ref = _oracle(block, x, topk_w, topk_ids) + _shared_oracle(block, x)
        old = _runner(block, x, topk_w, topk_ids) + block.shared_expert(x)
        assert block._use_decode(x)
        torch.testing.assert_close(block(x), got, rtol=0, atol=0)
    assert got.dtype == torch.bfloat16 and got.shape == x.shape
    _assert_near_oracle(got, ref)
    assert _rel_err(got, ref) <= _rel_err(old, ref)


@pytest.mark.parametrize("tokens", [5, 20])
def test_odd_expert_count_and_top_k(tokens, fp32_matmul):
    block = _block(experts=7, top_k=3, hidden=512, inter=128)
    x = torch.randn(tokens, block.hidden_size, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.stack([torch.randperm(7, device="cuda")[:3] for _ in range(tokens)])
    topk_w = torch.rand(tokens, 3, device="cuda")
    with torch.no_grad():
        got = block._dispatch_decode(x, topk_w, topk_ids)
        ref = _oracle(block, x, topk_w, topk_ids) + _shared_oracle(block, x)
    _assert_near_oracle(got, ref)


@pytest.mark.parametrize("tokens", [24, 64])
@pytest.mark.parametrize("skew", ["same8", "zipf"])
def test_grouped_skewed_routing(tokens, skew, fp32_matmul):
    # Real routing concentrates: an expert can hold every token, i.e. several row tiles.
    block = _block()
    x = torch.randn(tokens, block.hidden_size, device="cuda", dtype=torch.bfloat16)
    if skew == "same8":
        topk_ids = torch.arange(8, device="cuda").expand(tokens, 8)
    else:
        p = 1.0 / torch.arange(1, 33, device="cuda", dtype=torch.float32)
        topk_ids = torch.multinomial(p.expand(tokens, 32), 8)
    topk_w = torch.rand(tokens, 8, device="cuda")
    with torch.no_grad():
        got = block._dispatch_decode(x, topk_w, topk_ids)
        ref = _oracle(block, x, topk_w, topk_ids) + _shared_oracle(block, x)
        old = _runner(block, x, topk_w, topk_ids) + block.shared_expert(x)
    _assert_near_oracle(got, ref)
    assert _rel_err(got, ref) <= _rel_err(old, ref)


@pytest.mark.parametrize("tokens", [11, 64])
def test_grouped_x_row_scales(tokens, fp32_matmul):
    # The grouped gate/up meets x as two e4m3 terms under one power-of-two scale per row:
    # rows four decades apart, a zero row and a row with one large outlier must each stay
    # within the oracle bound.
    block = _block()
    x = torch.randn(tokens, block.hidden_size, device="cuda")
    x *= torch.logspace(-2, 2, tokens, device="cuda")[:, None]
    x[1] = 0.0
    x[2, 7] = 3.0e4
    x = x.to(torch.bfloat16)
    with torch.no_grad():
        topk_w, topk_ids = block.gate(x)
        got = block._dispatch_decode(x, topk_w, topk_ids).float()
        ref = _oracle(block, x, topk_w, topk_ids) + _shared_oracle(block, x)
    assert torch.equal(got[1], torch.zeros_like(got[1]))
    rows = [r for r in range(tokens) if r != 1]
    err = (got[rows] - ref[rows]).abs().amax(1) / ref[rows].abs().amax(1)
    assert float(err.max()) < 5e-3


def _gate(experts=288, top_k=8, hidden=4096, dtype=torch.bfloat16, seed=1):
    torch.manual_seed(seed)
    gate = Glm5NextMoEGate(hidden, experts, top_k, routed_scaling_factor=2.5).cuda()
    with torch.no_grad():
        gate.weight.normal_(0.0, 0.02)
        gate.e_score_correction_bias.normal_(0.0, 0.5)
    gate.weight.data = gate.weight.data.to(dtype)
    gate.finalize_weights()
    return gate


def _assert_same_routing(gate, x, monkeypatch):
    with torch.no_grad():
        w, ids = gate(x)
        monkeypatch.setattr(fused_decode, "_ENABLED", False)
        w_r, ids_r = gate(x)
        monkeypatch.setattr(fused_decode, "_ENABLED", True)
    assert w.dtype == torch.float32 and ids.dtype == torch.int64
    # Same experts per token (the reference's topk order is unspecified); ties would go to
    # the lower expert id, and the random scores here have none.
    ids, order = ids.sort(dim=-1)
    ids_r, order_r = ids_r.sort(dim=-1)
    torch.testing.assert_close(ids, ids_r, rtol=0, atol=0)
    torch.testing.assert_close(w.gather(1, order), w_r.gather(1, order_r), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 8, 16, 32, 64])
def test_decode_router_matches_reference(tokens, monkeypatch, fp32_matmul):
    gate = _gate()
    for seed in range(3):
        torch.manual_seed(seed)
        x = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
        _assert_same_routing(gate, x, monkeypatch)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_router_fp32_operands_and_odd_sizes(dtype, monkeypatch, fp32_matmul):
    gate = _gate(experts=7, top_k=3, hidden=512, dtype=dtype)
    x = torch.randn(5, 512, device="cuda", dtype=dtype)
    _assert_same_routing(gate, x, monkeypatch)


def test_large_batch_and_disabled_use_the_runner(monkeypatch):
    block = _block()
    for tokens, enabled in ((moe_module._DECODE_MAX_TOKENS + 1, True), (4, False)):
        monkeypatch.setattr(fused_decode, "_ENABLED", enabled)
        x = torch.randn(tokens, block.hidden_size, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            got = block(x)
            topk_w, topk_ids = block.gate(x)
            old = _runner(block, x, topk_w, topk_ids) + block.shared_expert(x)
        torch.testing.assert_close(got, old, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 4, 10, 11, 32, 64])
def test_graph_replay_matches_unchained_path(tokens):
    # The captured forward runs the PDL chain; route + experts launch without it. Fresh
    # inputs every replay, so a kernel reading its predecessor's output early shows up.
    block = _block()
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
            ref = block._dispatch_decode(x, *block.gate(x))
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
