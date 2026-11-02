"""Block-scaled low-precision linears without a GPU: scale layout, the host quantizer, guards, a model's declaration,
and the bf16 fallback. ``test/integration/test_block_scaled_gpu.py`` holds the kernels to these."""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from mstar.model.components import linear as linear_module
from mstar.model.components.linear import (
    BlockScaledLinear,
    FusedColumnLinear,
    LowPrecisionLinears,
    apply_linear_precision,
)
from mstar.model.flux2_klein.components.transformer import LOW_PRECISION_LINEARS, Flux2DiT
from mstar.model.flux2_klein.config import Flux2TransformerConfig
from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinModel
from mstar.platforms import block_gemm
from mstar.platforms.interface import CudaPlatform, Platform, RocmPlatform
from mstar.utils import quant_fp8
from mstar.utils.fp8_block import (
    FP8_DTYPE,
    MXFP8,
    NVFP4,
    check_block_gemm,
    dequantize_block,
    quantize_block,
    swizzle_block_scales,
    swizzled_scale_elements,
    unswizzle_block_scales,
)


def _doc_offset(row: int, block: int, col_tiles: int) -> int:
    """cuBLAS's formula for one scale: which 128x4 tile, then outer % 32, outer // 32, inner."""
    tile = 512 * ((row // 128) * col_tiles + block // 4)
    outer, inner = row % 128, block % 4
    return tile + (outer % 32) * 16 + (outer // 32) * 4 + inner


def test_swizzle_matches_the_cublas_layout():
    scales = torch.randint(0, 255, (300, 24), dtype=torch.uint8)
    flat = swizzle_block_scales(scales)
    assert flat.numel() == swizzled_scale_elements(300, 24) == 384 * 24
    for row in range(300):
        for block in range(24):
            assert flat[_doc_offset(row, block, col_tiles=6)] == scales[row, block]
    assert torch.equal(unswizzle_block_scales(flat, 300, 24), scales)


def test_swizzle_pads_rows_and_blocks_to_whole_tiles():
    scales = torch.randint(1, 255, (5, 6), dtype=torch.uint8)
    flat = swizzle_block_scales(scales)
    assert flat.numel() == 128 * 8
    assert torch.equal(unswizzle_block_scales(flat, 5, 6), scales)


def _activation(rows=64, k=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    # blocks of very different magnitudes: per-block scales must absorb them
    return (torch.randn(rows, k, generator=g) * torch.logspace(-3, 3, k).flip(0)).to(torch.bfloat16)


def test_mxfp8_scales_are_powers_of_two_and_nothing_clips():
    x = _activation()
    q = quantize_block(x, MXFP8)
    assert q.data.dtype == FP8_DTYPE and q.data.shape == x.shape
    assert q.scale.dtype == torch.float8_e8m0fnu and q.tensor_scale is None
    scales = unswizzle_block_scales(q.scale, 64, 256 // 32)
    # no block's largest value needed more than e4m3's top, nor less than half of it
    amax = x.float().view(64, 8, 32).abs().amax(-1)
    ratio = amax / scales.float()
    assert ratio.max() <= 448 and ratio.min() > 224
    assert torch.equal(torch.frexp(scales.float()).mantissa, torch.full_like(scales.float(), 0.5))
    # e4m3's 3-bit mantissa bounds the relative error once the block scale is divided out
    back = dequantize_block(q, MXFP8)
    torch.testing.assert_close(back, x.float(), rtol=1 / 16 + 1e-3, atol=2.0e-3)


def test_nvfp4_rounds_block_scales_up_and_packs_low_nibble_first():
    x = _activation()
    q = quantize_block(x, NVFP4)
    assert q.data.dtype == torch.float4_e2m1fn_x2 and q.data.shape == (64, 128)
    assert q.scale.dtype == FP8_DTYPE and q.tensor_scale.shape == (1,)
    scale = unswizzle_block_scales(q.scale, 64, 16).float() * q.tensor_scale
    amax = x.float().view(64, 16, 16).abs().amax(-1)
    assert (amax / scale).max() <= 6.0 * (1 + 1e-5)
    back = dequantize_block(q, NVFP4)
    # the coarse values round by at most a quarter of their magnitude, the zero bin to a twelfth of the block's top
    bound = torch.maximum(x.float().abs() * 0.34, scale.repeat_interleave(16, dim=1) * 0.25 * (1 + 1e-3))
    assert ((back - x.float()).abs() <= bound).all()

    known = torch.zeros(1, 64, dtype=torch.bfloat16)
    known[0, 0], known[0, 1] = 6.0, -3.0
    packed = quantize_block(known, NVFP4, tensor_scale=1.0)
    # the block's top maps to 1.0 * 6: code 7 for +6, sign bit + code 5 for -3
    assert packed.data.view(torch.uint8)[0, 0].item() == (0b1101 << 4) | 0b0111
    torch.testing.assert_close(dequantize_block(packed, NVFP4)[0, :2], torch.tensor([6.0, -3.0]))


def test_zero_block_stays_finite_and_zero():
    x = torch.zeros(32, 128, dtype=torch.bfloat16)
    for fmt in (MXFP8, NVFP4):
        back = dequantize_block(quantize_block(x, fmt, **({} if fmt is MXFP8 else {"tensor_scale": 1.0})), fmt)
        assert torch.equal(back, torch.zeros_like(back))


@pytest.mark.parametrize(("fmt", "k", "n", "dtype", "match"), [
    (MXFP8, 3000, 3072, torch.bfloat16, "K a multiple of 128"),
    (NVFP4, 96, 3072, torch.bfloat16, "K a multiple of 64"),
    (MXFP8, 3072, 3000, torch.bfloat16, "N a multiple of 16"),
    (MXFP8, 3072, 3072, torch.float32, "replaces a bf16 GEMM"),
])
def test_guards_refuse_what_the_kernels_cant_run(fmt, k, n, dtype, match):
    with pytest.raises(ValueError, match=match):
        check_block_gemm(fmt, k, n, dtype)


def test_guards_accept_klein_and_z_image_widths():
    for k in (3072, 9216, 12288, 4096, 16384, 3840):
        for fmt in (MXFP8, NVFP4):
            check_block_gemm(fmt, k, 3072, torch.bfloat16)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.keep = nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
        self.fused = FusedColumnLinear(128, {"q": 128, "k": 128}, dtype=torch.bfloat16)
        self.plain = nn.Linear(128, 256, bias=False, dtype=torch.bfloat16)
        for param in self.parameters():
            nn.init.normal_(param, std=0.05)


_DECLARED = LowPrecisionLinears(candidates=("fused", "plain"), keep_bf16=("keep",))


@pytest.fixture
def block_gemm_gpu(monkeypatch):
    monkeypatch.setattr(Platform, "has_block_scaled_gemm", lambda self, device=None: True)
    monkeypatch.setattr(linear_module, "current_platform", Platform())
    block_gemm.block_quant.reset()
    block_gemm.block_scaled_gemm.reset()
    yield
    block_gemm.block_quant.reset()
    block_gemm.block_scaled_gemm.reset()


def test_bf16_changes_nothing():
    block = _Block()
    before = dict(block.named_modules())
    assert apply_linear_precision(block, "bf16", _DECLARED) == []
    assert dict(block.named_modules()) == before


def test_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="linear_precision must be one of"):
        apply_linear_precision(_Block(), "fp3", _DECLARED)


def test_a_gpu_without_the_gemm_stays_bf16(monkeypatch, caplog):
    monkeypatch.setattr(linear_module, "current_platform", Platform())
    block = _Block()
    assert apply_linear_precision(block, "mxfp8", _DECLARED) == []
    assert isinstance(block.plain, nn.Linear) and "staying on bf16" in caplog.text


@pytest.mark.usefixtures("block_gemm_gpu")
@pytest.mark.parametrize("precision", ["mxfp8", "nvfp4"])
def test_declared_linears_move_and_kept_ones_dont(precision):
    block = _Block()
    fused_weight = block.fused.weight.data.clone()
    moved = apply_linear_precision(block, precision, _DECLARED)
    assert moved == ["fused", "plain"]
    assert isinstance(block.keep, nn.Linear)
    assert isinstance(block.fused, BlockScaledLinear) and isinstance(block.plain, BlockScaledLinear)
    assert not list(block.fused.parameters()), "the bf16 weight goes with the module it lived on"

    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 64, 128, generator=g).to(torch.bfloat16)
    out = block.fused(x)
    assert out.dtype == torch.bfloat16 and out.shape == (2, 64, 256)
    ref = x.float() @ fused_weight.float().t()
    limit = 0.05 if precision == "mxfp8" else 0.25
    assert (out.float() - ref).norm() / ref.norm() < limit


@pytest.mark.usefixtures("block_gemm_gpu")
def test_keep_patterns_win_for_one_deployment():
    block = _Block()
    assert apply_linear_precision(block, "mxfp8", _DECLARED, keep=("plain",)) == ["fused"]
    assert isinstance(block.plain, nn.Linear)


def test_an_undeclared_linear_is_refused():
    with pytest.raises(ValueError, match="neither a low-precision candidate nor kept"):
        apply_linear_precision(_Block(), "mxfp8", LowPrecisionLinears(candidates=("fused",), keep_bf16=("keep",)))


@pytest.mark.usefixtures("block_gemm_gpu")
def test_a_candidate_with_a_bias_is_refused():
    block = _Block()
    block.plain = nn.Linear(128, 256, bias=True, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="has a bias"):
        apply_linear_precision(block, "mxfp8", _DECLARED)


def _klein_names(num_layers=2, num_single_layers=3):
    config = Flux2TransformerConfig(num_layers=num_layers, num_single_layers=num_single_layers)
    with torch.device("meta"):
        modules = Flux2DiT(config).named_modules()
    return [name for name, module in modules if isinstance(module, (nn.Linear, FusedColumnLinear))]


def test_klein_declaration_classifies_every_linear():
    selected = [name for name in _klein_names() if LOW_PRECISION_LINEARS.selected(name)]
    assert len(selected) == 2 * 8 + 3 * 2
    kept = set(_klein_names()) - set(selected)
    assert kept == {
        "time_embed.linear_in", "time_embed.linear_out", "mod_double_img", "mod_double_txt", "mod_single",
        "img_in", "txt_in", "norm_out_mod", "proj_out",
    }


def test_klein_widths_pass_the_guards():
    for config in (Flux2TransformerConfig(), Flux2TransformerConfig(num_attention_heads=32)):
        d, m = config.hidden_size, config.mlp_hidden_dim
        for k, n in ((d, 3 * d), (d, d), (d, 2 * m), (m, d), (d, 3 * d + 2 * m), (d + m, d)):
            for fmt in (MXFP8, NVFP4):
                check_block_gemm(fmt, k, n, torch.bfloat16)


def test_klein_knob_defaults_to_bf16_and_refuses_a_typo():
    assert Flux2KleinModel(skip_weight_loading=True).linear_precision == "bf16"
    with pytest.raises(ValueError, match="linear_precision must be one of"):
        Flux2KleinModel(skip_weight_loading=True, linear_precision="mxfp4")


@pytest.mark.parametrize(("capability", "expected"), [((9, 0), False), ((10, 0), True), ((10, 3), True)])
def test_cuda_platform_picks_by_capability(monkeypatch, capability, expected):
    monkeypatch.setattr(CudaPlatform, "cuda_capability", lambda self, device=None: capability)
    assert CudaPlatform().has_block_scaled_gemm() is expected


def test_other_platforms_have_no_block_scaled_gemm():
    assert RocmPlatform().has_block_scaled_gemm() is False
    assert Platform().has_block_scaled_gemm() is False


class _Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append({"grid": grid, "args": args, "kwargs": kwargs})

        return launch


@pytest.mark.parametrize(("fmt", "kernel", "block", "stored"), [
    (MXFP8, "_mxfp8_quant_kernel", 32, 3072), (NVFP4, "_nvfp4_quant_kernel", 16, 1536),
])
def test_quant_block_host_contract(monkeypatch, fmt, kernel, block, stored):
    rec = _Recorder()
    monkeypatch.setattr(quant_fp8, kernel, rec)
    monkeypatch.setattr(quant_fp8.triton, "cdiv", lambda a, b: math.ceil(a / b), raising=False)
    rows, k = 70, 3072  # rows don't fill a tile: the grid rounds up, the scales stay tile-padded
    q = quant_fp8.quant_block(torch.zeros(rows, k, dtype=torch.bfloat16), fmt)
    assert q.data.dtype == fmt.data_dtype and q.data.shape == (rows, stored)
    assert q.scale.dtype == fmt.scale_dtype and q.scale.numel() == 128 * (k // block)
    (call,) = rec.calls
    assert call["grid"] == (3, k // block // 4)
    assert call["kwargs"] == {"ROWS": 32, "BLOCKS": 4, "BLOCK": block}
    assert q.tensor_scale is None
