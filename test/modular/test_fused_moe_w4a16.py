"""The grouped GEMM over MXFP4 experts (``fused_moe.w4a16``): its launches compile for Hopper on
any host, and on a GPU a launch gives each slot its row times its expert's dequantized weight."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton.backends.nvidia")

from mstar.testing_utils.triton_compile import compile_launches, record_launches  # noqa: E402
from mstar.utils.fused_moe import w4a16  # noqa: E402
from mstar.utils.fused_moe.align import moe_align_block_size  # noqa: E402
from mstar.utils.mx_round import dequant_row_fp4, round_fp8  # noqa: E402

GROUP = 32
E8M0 = torch.float8_e8m0fnu


def _experts(experts: int, rows: int, cols: int, device="cpu", seed: int = 0):
    """Random MXFP4 experts as a checkpoint packs them: ``(E, rows, cols / 2)`` bytes of two E2M1
    codes, ``(E, rows, cols / 32)`` E8M0 scale bytes over a few binades."""
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (experts, rows, cols // 2), generator=g, dtype=torch.uint8)
    scale = torch.randint(118, 124, (experts, rows, cols // GROUP), generator=g, dtype=torch.uint8)
    return packed.to(device), scale.to(device)


def _gemm(a, packed, scale, ids, top_k: int, weights=None) -> torch.Tensor:
    """One launch as a caller makes it: ``(slots, rows)``, slot ``s`` from ``a[s // top_k]``."""
    experts, rows, _ = packed.shape
    config = w4a16.moe_config(ids.shape[0], experts, rows, packed.shape[2] * 2, ids.shape[1], GROUP)
    out = torch.zeros(ids.numel(), rows, dtype=a.dtype, device=a.device)
    w4a16._launch(a, packed, scale, None, out, a if weights is None else weights,
                  *moe_align_block_size(ids, config["BLOCK_SIZE_M"], experts), ids, weights is not None, top_k,
                  config, w4a16._compute_type(a.dtype), w4a16.KIND_E2M1, GROUP)
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("tokens", [3, 40], ids=["m16", "m64"])
def test_the_mxfp4_launches_compile_for_hopper(monkeypatch, dtype, tokens):
    """Gate/up (every slot reads its token's row), down over a K that is not whole 128-wide
    steps (288, a TP8 slice: masked tails), and down scaled by the routing weight: on the
    16-row tiles of a few tokens and the 64-row tiles of many."""
    kernel = record_launches(monkeypatch, w4a16, "fused_moe_w4a16_kernel")
    experts, hidden, inter, top_k = 8, 256, 288, 2
    ids = torch.arange(tokens * top_k).reshape(tokens, top_k) % experts
    w1, s1 = _experts(experts, 2 * inter, hidden)
    w2, s2 = _experts(experts, hidden, inter)
    _gemm(torch.zeros(tokens, hidden, dtype=dtype), w1, s1, ids, top_k)
    act = torch.zeros(tokens * top_k, inter, dtype=dtype)
    _gemm(act, w2, s2, ids, 1)
    _gemm(act, w2, s2, ids, 1, weights=torch.ones(tokens, top_k))
    assert [kernel.arguments(i)["K"] for i in range(3)] == [hidden, inter, inter]
    assert kernel.arguments()["BLOCK_SIZE_M"] == (16 if tokens <= experts else 64)
    assert compile_launches(kernel) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton kernel runs on CUDA")
@pytest.mark.parametrize("dtype, tolerance", [(torch.bfloat16, 2.0 ** -7), (torch.float32, 1e-5)],
                         ids=["bf16", "fp32"])
def test_a_launch_gives_each_slot_its_row_times_its_experts_weight(dtype, tolerance):
    """Against the dequantized weights in torch, fp32 sums: within the output dtype's rounding,
    and in fp32 within a summation order (the activations lie on the FP8 grid, which the fp32
    tensor-core dot holds exactly). A slot routed past the last expert is left as it was."""
    experts, hidden, inter, top_k, tokens = 8, 256, 288, 3, 37
    g = torch.Generator().manual_seed(1)
    w, s = _experts(experts, hidden, inter, device="cuda", seed=2)
    a = round_fp8(torch.randn(tokens * top_k, inter, generator=g).to("cuda", dtype), GROUP)
    ids = torch.stack([torch.randperm(experts + 1, generator=g)[:top_k] for _ in range(tokens)]).cuda()
    weights = torch.rand(tokens, top_k, generator=g).cuda()
    dense = torch.stack([dequant_row_fp4(w[e], s[e].view(E8M0), torch.float32) for e in range(experts)])
    for scaled in (None, weights):
        got = _gemm(a, w, s, ids, 1, weights=scaled).float()
        flat = ids.flatten()
        run = flat < experts
        want = torch.einsum("sk,snk->sn", a.float()[run], dense[flat[run]])
        terms = torch.einsum("sk,snk->sn", a.float()[run].abs(), dense[flat[run]].abs())
        if scaled is not None:
            want, terms = (t * weights.flatten()[run, None] for t in (want, terms))
        assert not got[~run].any()
        assert ((got[run] - want).abs() <= tolerance * want.abs() + 1e-6 * terms).all()
