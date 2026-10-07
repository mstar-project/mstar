"""Host-side contract of the fp8 fused-MoE path, runnable without a GPU."""
from __future__ import annotations

import inspect

import pytest
import torch

from mstar.utils.fused_moe import kernels, runner
from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import _grid_rows, get_default_config
from mstar.utils.quant_fp8 import FP8_DTYPE

BLOCK = (128, 128)


class _Recorder:
    """Stand-in for a ``@triton.jit`` kernel: ``kernel[grid](*args, **kw)``."""

    def __init__(self):
        self.calls: list[dict] = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append({"grid": grid, "args": args, "kwargs": kwargs})

        return launch


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


# ----------------------------------------------------------------------------
# get_default_config / _grid_rows
# ----------------------------------------------------------------------------


def test_get_default_config_branches_depend_only_on_m_vs_e():
    small = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}
    large = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}
    assert get_default_config(M=4, E=256, N=512, K=6144, top_k=8) == small
    assert get_default_config(M=256, E=256, N=512, K=6144, top_k=8) == small
    assert get_default_config(M=257, E=256, N=512, K=6144, top_k=8) == large
    # N / K / top_k are ignored: same tiles for the gate/up and down shapes.
    assert get_default_config(M=4, E=256, N=6144, K=256, top_k=1) == small


def test_grid_rows_hand_computation():
    # GLM-5.2 k=3 decode: 8 tokens x 4 experts = 32 slots, E=256, BLOCK_M=16.
    tokens, top_k, E, block_m = 8, 4, 256, 16
    topk_ids = torch.randint(0, E, (tokens, top_k), dtype=torch.int32)
    sorted_ids, _, n_post = moe_align_block_size(topk_ids, block_m, E)
    worst = tokens * top_k + E * (block_m - 1)
    assert sorted_ids.shape[0] == worst == 3872
    assert _grid_rows(sorted_ids, topk_ids, block_m) == tokens * top_k * block_m == 512
    # The clamp never cuts below the alignment's real padded slot count.
    assert int(n_post.item()) <= 512

    # Large batch / few experts: the worst case is already the smaller bound.
    tokens, top_k, E = 64, 8, 4
    topk_ids = torch.randint(0, E, (tokens, top_k), dtype=torch.int32)
    sorted_ids, _, n_post = moe_align_block_size(topk_ids, block_m, E)
    assert sorted_ids.shape[0] == 512 + 4 * 15 == 572
    assert _grid_rows(sorted_ids, topk_ids, block_m) == 572
    assert int(n_post.item()) <= 572


@pytest.mark.parametrize("tokens,top_k,E,block_m", [(1, 8, 256, 16), (5, 3, 128, 16), (100, 8, 16, 64)])
def test_grid_rows_covers_num_tokens_post_padded(tokens, top_k, E, block_m):
    torch.manual_seed(tokens)
    topk_ids = torch.randint(0, E, (tokens, top_k), dtype=torch.int32)
    sorted_ids, _, n_post = moe_align_block_size(topk_ids, block_m, E)
    em = _grid_rows(sorted_ids, topk_ids, block_m)
    assert int(n_post.item()) <= em <= sorted_ids.shape[0]
    assert em == min(sorted_ids.shape[0], tokens * top_k * block_m)


# ----------------------------------------------------------------------------
# Launchers: grid and EM both come from _grid_rows
# ----------------------------------------------------------------------------


def _aligned(tokens: int, top_k: int, E: int, block_m: int):
    topk_ids = torch.randint(0, E, (tokens, top_k), dtype=torch.int32)
    topk_weights = torch.rand(tokens, top_k, dtype=torch.bfloat16)
    sorted_ids, expert_ids, n_post = moe_align_block_size(topk_ids, block_m, E)
    return topk_ids, topk_weights, sorted_ids, expert_ids, n_post


def test_invoke_fused_moe_kernel_em_from_grid_rows(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(kernels, "fused_moe_kernel", rec)
    tokens, top_k, E, hidden, inter = 4, 8, 256, 64, 32
    config = get_default_config(M=tokens, E=E, N=2 * inter, K=hidden, top_k=top_k)
    topk_ids, topk_weights, sorted_ids, expert_ids, n_post = _aligned(tokens, top_k, E, config["BLOCK_SIZE_M"])
    A = torch.zeros(tokens, hidden, dtype=torch.bfloat16)
    B = torch.zeros(E, 2 * inter, hidden, dtype=torch.bfloat16)
    C = torch.zeros(tokens * top_k, 2 * inter, dtype=torch.bfloat16)

    kernels.invoke_fused_moe_kernel(
        A, B, C, topk_weights, topk_ids, sorted_ids, expert_ids, n_post,
        mul_routed_weight=False, top_k=top_k, config=config, compute_type="bf16",
    )

    (call,) = rec.calls
    em = _grid_rows(sorted_ids, topk_ids, config["BLOCK_SIZE_M"])
    assert em == 512 < sorted_ids.shape[0]
    assert call["args"][9] == em  # EM positional
    assert call["args"][10] == topk_ids.numel()  # num_valid_tokens
    assert call["grid"](config) == (_cdiv(em, config["BLOCK_SIZE_M"]) * _cdiv(2 * inter, config["BLOCK_SIZE_N"]),)


def test_invoke_fp8_launcher_grid_em_and_constexprs(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(kernels, "fused_moe_kernel_fp8_w8a8", rec)
    tokens, top_k, E, hidden, inter = 4, 8, 256, 256, 128
    group_n, group_k = BLOCK
    config = get_default_config(M=tokens, E=E, N=2 * inter, K=hidden, top_k=top_k)
    config["BLOCK_SIZE_K"] = group_k
    topk_ids, topk_weights, sorted_ids, expert_ids, n_post = _aligned(tokens, top_k, E, config["BLOCK_SIZE_M"])
    A = torch.zeros(tokens, hidden, dtype=FP8_DTYPE)
    B = torch.zeros(E, 2 * inter, hidden, dtype=FP8_DTYPE)
    C = torch.zeros(tokens * top_k, 2 * inter, dtype=torch.bfloat16)
    A_scale = torch.ones(tokens, hidden // group_k)
    B_scale = torch.ones(E, _cdiv(2 * inter, group_n), hidden // group_k)

    kernels.invoke_fused_moe_kernel_fp8_w8a8(
        A, B, C, A_scale, B_scale, topk_weights, topk_ids, sorted_ids, expert_ids, n_post,
        mul_routed_weight=False, top_k=top_k, config=config, compute_type="bf16", block_shape=BLOCK,
    )

    (call,) = rec.calls
    em = _grid_rows(sorted_ids, topk_ids, config["BLOCK_SIZE_M"])
    assert call["args"][9:13] == (2 * inter, hidden, em, topk_ids.numel())  # N, K, EM, num_valid
    assert call["grid"](config) == (_cdiv(em, config["BLOCK_SIZE_M"]) * _cdiv(2 * inter, config["BLOCK_SIZE_N"]),)
    kw = call["kwargs"]
    assert (kw["group_n"], kw["group_k"], kw["even_Ks"], kw["top_k"], kw["MUL_ROUTED_WEIGHT"]) == (
        group_n, group_k, True, top_k, False,
    )
    assert kw["BLOCK_SIZE_K"] == group_k

    # BLOCK_SIZE_K != group_k is refused on the host: the in-kernel rescale is per K tile.
    bad = dict(config, BLOCK_SIZE_K=64)
    with pytest.raises(AssertionError, match="BLOCK_SIZE_K == group_k"):
        kernels.invoke_fused_moe_kernel_fp8_w8a8(
            A, B, C, A_scale, B_scale, topk_weights, topk_ids, sorted_ids, expert_ids, n_post,
            mul_routed_weight=False, top_k=top_k, config=bad, compute_type="bf16", block_shape=BLOCK,
        )
    # Non-fp8 operands are refused too (the runner is responsible for the uint8 re-view).
    with pytest.raises(AssertionError):
        kernels.invoke_fused_moe_kernel_fp8_w8a8(
            A.view(torch.uint8), B, C, A_scale, B_scale, topk_weights, topk_ids, sorted_ids, expert_ids,
            n_post, mul_routed_weight=False, top_k=top_k, config=config, compute_type="bf16", block_shape=BLOCK,
        )


# ----------------------------------------------------------------------------
# fused_experts_fp8 dispatch: decode tiles per launch, shapes, dtypes
# ----------------------------------------------------------------------------


@pytest.fixture
def fp8_fakes(monkeypatch):
    """Replace the four Triton launchers the runner calls with recorders."""
    launches: list[dict] = []
    quants: list[tuple] = []
    reduces: list[tuple] = []

    def fake_quant(x, group_size, eps=1e-10):
        quants.append((tuple(x.shape), x.dtype, group_size))
        return torch.empty_like(x, dtype=FP8_DTYPE), torch.ones(x.shape[0], x.shape[1] // group_size)

    def fake_gemm(**kw):
        launches.append(kw)

    def fake_act(gateup, down, activation="silu", swiglu_limit=None):
        assert gateup.shape[1] == 2 * down.shape[1]
        assert swiglu_limit is None

    def fake_reduce(inp, out, routed_scaling_factor=1.0):
        reduces.append((tuple(inp.shape), tuple(out.shape)))

    monkeypatch.setattr(runner, "per_token_group_quant_fp8", fake_quant)
    monkeypatch.setattr(runner, "invoke_fused_moe_kernel_fp8_w8a8", fake_gemm)
    monkeypatch.setattr(runner, "act_and_mul_triton", fake_act)
    monkeypatch.setattr(runner, "moe_sum_reduce_triton", fake_reduce)
    monkeypatch.setattr(runner, "_tl_compute_type", lambda dtype: f"tl:{dtype}")
    return launches, quants, reduces


def _fp8_inputs(tokens: int, E: int = 8, hidden: int = 256, inter: int = 128, top_k: int = 2):
    torch.manual_seed(0)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    w1 = torch.zeros(E, 2 * inter, hidden, dtype=torch.uint8)  # byte view, as in the checkpoint
    w2 = torch.zeros(E, hidden, inter, dtype=torch.uint8)
    w1_s = torch.ones(E, _cdiv(2 * inter, BLOCK[0]), hidden // BLOCK[1])
    w2_s = torch.ones(E, _cdiv(hidden, BLOCK[0]), inter // BLOCK[1])
    topk_ids = torch.randint(0, E, (tokens, top_k))  # int64, as torch.topk returns
    topk_weights = torch.rand(tokens, top_k, dtype=torch.bfloat16)
    return x, w1, w2, w1_s, w2_s, topk_weights, topk_ids


def test_fused_experts_fp8_decode_tiles_per_launch(fp8_fakes):
    launches, quants, reduces = fp8_fakes
    x, w1, w2, w1_s, w2_s, topk_weights, topk_ids = _fp8_inputs(tokens=4)
    tokens, hidden, inter, top_k = 4, 256, 128, 2

    out = runner.fused_experts_fp8(x, w1, w2, w1_s, w2_s, topk_weights, topk_ids, block_size=BLOCK)

    assert out.shape == (tokens, hidden) and out.dtype == torch.bfloat16
    assert reduces == [((tokens, top_k, hidden), (tokens, hidden))]
    assert quants == [((tokens, hidden), torch.bfloat16, 128), ((tokens * top_k, inter), torch.bfloat16, 128)]

    up, down = launches
    # Gate/up: N=2*inter, K=hidden, reads source rows via top_k, no routed weight.
    assert up["A"].dtype == FP8_DTYPE and tuple(up["A"].shape) == (tokens, hidden)
    assert up["B"].dtype == FP8_DTYPE and tuple(up["B"].shape) == (8, 2 * inter, hidden)  # uint8 re-viewed
    assert tuple(up["C"].shape) == (tokens * top_k, 2 * inter) and up["C"].dtype == torch.bfloat16
    assert up["B_scale"] is w1_s and tuple(up["A_scale"].shape) == (tokens, hidden // 128)
    assert (up["top_k"], up["mul_routed_weight"], up["block_shape"]) == (top_k, False, BLOCK)
    assert up["config"] == {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8,
        "num_warps": 4, "num_stages": 5,
    }
    # Down: N=hidden, K=inter, top_k=1 (slot-indexed rows), routed weight folded in.
    assert tuple(down["A"].shape) == (tokens * top_k, inter) and down["A"].dtype == FP8_DTYPE
    assert tuple(down["B"].shape) == (8, hidden, inter) and down["B"].dtype == FP8_DTYPE
    assert tuple(down["C"].shape) == (tokens * top_k, hidden)
    assert down["B_scale"] is w2_s and tuple(down["A_scale"].shape) == (tokens * top_k, inter // 128)
    assert (down["top_k"], down["mul_routed_weight"], down["block_shape"]) == (1, True, BLOCK)
    assert down["config"] == {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1,
        "num_warps": 4, "num_stages": 2,
    }
    # Both launches share the alignment built for BLOCK_SIZE_M=16.
    assert up["sorted_token_ids"] is down["sorted_token_ids"]
    assert up["topk_ids"].dtype == torch.int32
    assert up["compute_type"] == down["compute_type"] == "tl:torch.bfloat16"


def test_fused_experts_fp8_above_decode_threshold_uses_block_fp8_tiles(fp8_fakes):
    launches, _, _ = fp8_fakes
    x, w1, w2, w1_s, w2_s, topk_weights, topk_ids = _fp8_inputs(tokens=17)
    runner.fused_experts_fp8(x, w1, w2, w1_s, w2_s, topk_weights, topk_ids, block_size=BLOCK)

    up, down = launches
    # M=17 > E=8 keeps the large-batch BLOCK_SIZE_M; the rest is vLLM's block-fp8 default.
    expected = {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32,
        "num_warps": 4, "num_stages": 3,
    }
    assert up["config"] == expected and down["config"] == expected
    assert up["config"] is not down["config"]


def test_fused_experts_fp8_reduce_results_false(fp8_fakes):
    _, _, reduces = fp8_fakes
    x, w1, w2, w1_s, w2_s, topk_weights, topk_ids = _fp8_inputs(tokens=3)
    out = runner.fused_experts_fp8(x, w1, w2, w1_s, w2_s, topk_weights, topk_ids, block_size=BLOCK,
                                   reduce_results=False)
    assert out.shape == (3, 2, 256) and out.dtype == torch.bfloat16
    assert reduces == []


def test_fused_experts_fp8_forwards_swiglu_limit(fp8_fakes, monkeypatch):
    seen = []

    def fake_act(gateup, down, activation="silu", swiglu_limit=None):
        seen.append(swiglu_limit)

    monkeypatch.setattr(runner, "act_and_mul_triton", fake_act)
    args = _fp8_inputs(tokens=2)
    runner.fused_experts_fp8(*args, block_size=BLOCK, swiglu_limit=10.0)
    assert seen == [10.0]


def test_fused_experts_fp8_rejects_bad_scale_shapes_and_groups(fp8_fakes):
    x, w1, w2, w1_s, w2_s, topk_weights, topk_ids = _fp8_inputs(tokens=2)
    with pytest.raises(AssertionError, match="w1_scale_inv"):
        runner.fused_experts_fp8(x, w1, w2, w1_s[:, :1], w2_s, topk_weights, topk_ids, block_size=BLOCK)
    with pytest.raises(AssertionError, match="w2_scale_inv"):
        runner.fused_experts_fp8(x, w1, w2, w1_s, w2_s[:, :1], topk_weights, topk_ids, block_size=BLOCK)
    # inter=128 is not a multiple of block_k=256: the intermediate cannot be group-quantized.
    with pytest.raises(AssertionError, match="multiples of block_k"):
        runner.fused_experts_fp8(x, w1, w2, w1_s, w2_s, topk_weights, topk_ids, block_size=(128, 256))
    with pytest.raises(AssertionError, match="e4m3"):
        runner.fused_experts_fp8(x, w1.to(torch.int8), w2, w1_s, w2_s, topk_weights, topk_ids, block_size=BLOCK)


def test_bf16_fused_experts_signature_unchanged():
    params = list(inspect.signature(runner.fused_experts).parameters)
    assert params == ["hidden_states", "w1", "w2", "topk_weights", "topk_ids", "activation", "reduce_results"]
    assert list(inspect.signature(get_default_config).parameters) == ["M", "E", "N", "K", "top_k"]
