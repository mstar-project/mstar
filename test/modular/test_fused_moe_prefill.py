"""The shared prefill fp8 MoE path (fused_moe.prefill) against the fused_experts_fp8 runner:
same kernels, other tiles, so the outputs must be identical. The SwiGLU switch on any host,
the rest on CUDA + triton."""
import pytest
import torch

from mstar.utils.fused_moe import prefill

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernels need CUDA + triton")


@pytest.mark.parametrize("limit", [None, 7.0])
def test_swiglu_limit_reaches_the_act_quant(limit, monkeypatch):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kw: calls.append((args, kw))

    monkeypatch.setattr(prefill, "_act_quant_kernel", Kernel())
    if not hasattr(prefill.triton, "next_power_of_2"):
        monkeypatch.setattr(prefill.triton, "next_power_of_2",
                            lambda n: 1 << (n - 1).bit_length(), raising=False)
    q, s = prefill._group_quant(torch.zeros(3, 512).bfloat16(), 128, swiglu=True, limit=limit)
    assert q.shape == (3, 256) and s.shape == (3, 2)
    ((args, kw),) = calls
    assert kw["CLAMP"] is (limit is not None) and args[4] == (limit or 0.0)


def _layer(experts, hidden, inter, block, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    w13 = (torch.randn(experts, 2 * inter, hidden, device=dev) * 0.05).to(torch.float8_e4m3fn)
    w2 = (torch.randn(experts, hidden, inter, device=dev) * 0.05).to(torch.float8_e4m3fn)
    s13 = torch.rand(experts, 2 * inter // block, hidden // block, device=dev) + 0.5
    s2 = torch.rand(experts, hidden // block, inter // block, device=dev) + 0.5
    return w13.view(torch.uint8), w2.view(torch.uint8), s13, s2


def _routing(tokens, experts, top_k, device="cuda"):
    ids = torch.stack([torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)])
    w = torch.rand(tokens, top_k, device=device)
    return w / w.sum(-1, keepdim=True), ids


@gpu
@pytest.mark.parametrize("tokens", [17, 64, 65, 300, 1036])
@pytest.mark.parametrize("limit", [10.0, None])
@pytest.mark.parametrize("hidden", [1024, 1536])
def test_matches_the_shared_runner(tokens, limit, hidden):
    from mstar.utils.fused_moe import fused_experts_fp8

    experts, inter, block, top_k = 64, 256, 128, 8
    w13, w2, s13, s2 = _layer(experts, hidden, inter, block)
    x = torch.randn(tokens, hidden, device="cuda").to(torch.bfloat16)
    weights, ids = _routing(tokens, experts, top_k)
    kw = dict(block_size=(block, block), swiglu_limit=limit)
    ref = fused_experts_fp8(x, w13, w2, s13, s2, weights, ids, **kw)
    out = prefill.experts(x, w13, w2, s13, s2, weights, ids, **kw)
    assert torch.equal(out, ref), (out - ref).abs().max()
    # the shared expert's output rides the top-k sum: the block's routed + shared, same bits
    shared = torch.randn(tokens, hidden, device="cuda").to(torch.bfloat16)
    out = prefill.experts(x, w13, w2, s13, s2, weights, ids, shared=shared, **kw)
    assert torch.equal(out, ref + shared)


@gpu
@pytest.mark.parametrize("rows, width", [(1, 4096), (300, 4096), (2401, 256), (77, 384)])
@pytest.mark.parametrize("limit", [10.0, None])
def test_glue_kernels_match_the_shared_ones(rows, width, limit):
    from mstar.utils.fused_moe.kernels import act_and_mul_triton
    from mstar.utils.quant_fp8 import per_token_group_quant_fp8

    x = (torch.randn(rows, width, device="cuda") * 3).to(torch.bfloat16)
    x[0, :128] = 0  # an all-zero group takes the eps floor
    q, s = prefill._group_quant(x, 128)
    q_ref, s_ref = per_token_group_quant_fp8(x, 128)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(s, s_ref)

    h1 = (torch.randn(rows, 2 * width, device="cuda") * 6).to(torch.bfloat16)
    h2 = torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
    act_and_mul_triton(h1, h2, activation="silu", swiglu_limit=limit)
    q_ref, s_ref = per_token_group_quant_fp8(h2, 128)
    q, s = prefill._group_quant(h1, 128, swiglu=True, limit=limit)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)) and torch.equal(s, s_ref)


@gpu
@pytest.mark.parametrize("inter, calls", [(256, 1), (384, 0)])
def test_down_kernel_takes_two_k_groups(inter, calls, monkeypatch):
    """The parity above runs the new down kernel (a TP8 rank's 256 = 2 x 128); other widths
    keep the shared one."""
    seen = []
    orig = prefill._down
    monkeypatch.setattr(prefill, "_down", lambda *a, **k: seen.append(1) or orig(*a, **k))
    w13, w2, s13, s2 = _layer(16, 512, inter, 128)
    x = torch.randn(300, 512, device="cuda").to(torch.bfloat16)
    weights, ids = _routing(300, 16, 4)
    prefill.experts(x, w13, w2, s13, s2, weights, ids, block_size=(128, 128), swiglu_limit=10.0)
    assert len(seen) == calls


def test_tiles_keep_the_runner_rows():
    small, _ = prefill.tiles(288, 288, 128)
    large, down = prefill.tiles(289, 288, 128)
    assert small["BLOCK_SIZE_M"] == 16 and large["BLOCK_SIZE_M"] == 64
    # both GEMMs read one alignment, so they share its tile rows
    assert down["BLOCK_SIZE_M"] == large["BLOCK_SIZE_M"] and large["BLOCK_SIZE_K"] == 128


def test_one_opaque_op_to_dynamo():
    """Like fused_experts_fp8: traced inline, the launch path's host logic broke the graph
    and failed outright on a symbolic token count (torch 2.9.1)."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    assert hasattr(torch.ops.mstar, "fused_moe_prefill_experts")
    with FakeTensorMode():
        x = torch.empty(37, 256, dtype=torch.bfloat16)
        w13 = torch.empty(4, 256, 256, dtype=torch.uint8)
        w2 = torch.empty(4, 256, 128, dtype=torch.uint8)
        s13, s2 = torch.empty(4, 2, 2), torch.empty(4, 2, 1)
        out = prefill.experts(x, w13, w2, s13, s2, torch.empty(37, 2),
                              torch.empty(37, 2, dtype=torch.int64), block_size=(128, 128),
                              shared=torch.empty(37, 256, dtype=torch.bfloat16))
    assert out.shape == (37, 256) and out.dtype == torch.bfloat16


@gpu
def test_compiles_on_a_dynamic_token_count():
    w13, w2, s13, s2 = _layer(16, 512, 256, 128)

    def f(x, weights, ids):
        return prefill.experts(x, w13, w2, s13, s2, weights, ids, block_size=(128, 128))

    compiled = torch.compile(f, dynamic=True, fullgraph=True)
    for tokens in (65, 300):
        x = torch.randn(tokens, 512, device="cuda").to(torch.bfloat16)
        weights, ids = _routing(tokens, 16, 4)
        assert torch.equal(compiled(x, weights, ids), f(x, weights, ids))
