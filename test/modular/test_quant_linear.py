"""Low-precision linears without a GPU: the shared pieces (``quant_linear``: a linear kind's capability, a model's
declaration, the keep list, ``QuantLinear``, the step schedule on the DiT scaffold) and the block-scaled formats'
numerics (``block_scaled``), whose ops run their reference on CPU tensors. ``test/integration/test_block_scaled_gpu.py``
holds the kernels to these."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.components import quant_linear
from mstar.model.components.block_scaled import MXFP8, NVFP4, Mxfp8Format, Nvfp4Format
from mstar.model.components.diffusion.denoise_loop import DenoiseLoopSubmodule, StepBucket
from mstar.model.components.distributed.linear import ColumnParallelLinear, RowParallelLinear
from mstar.model.components.linear import FusedColumnLinear
from mstar.model.components.lora.capable import LoraOutputAdd, lora_capability
from mstar.model.components.quant_linear import (
    QuantCapability,
    QuantizableLinears,
    QuantLinear,
    StepDtypes,
    apply_linear_dtype,
    check_linear_dtype_steps,
    check_linear_keep,
    quant_capability,
    select_linears,
)
from mstar.model.flux2_klein.components.transformer import QUANTIZABLE_LINEARS, Flux2DiT
from mstar.model.flux2_klein.config import Flux2KleinConfig, Flux2TransformerConfig
from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinModel
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.z_image.components.transformer import QUANTIZABLE_LINEARS as Z_QUANTIZABLE_LINEARS
from mstar.model.z_image.components.transformer import ZImageDiT
from mstar.model.z_image.config import ZImageTransformerConfig
from mstar.model.z_image.z_image_model import ZImageModel
from mstar.platforms.interface import CudaPlatform, Platform, RocmPlatform
from mstar.utils import quant_fp8
from mstar.utils.fp8_block import FP8_DTYPE, swizzle_block_scales, swizzled_scale_elements, unswizzle_block_scales

# ---------------------------------------------------------------- the scale layout and the formats' numerics


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
    q = MXFP8.quantize(x)
    assert q.data.dtype == FP8_DTYPE and q.data.shape == x.shape
    assert q.scale.dtype == torch.float8_e8m0fnu and q.tensor_scale is None
    scales = unswizzle_block_scales(q.scale, 64, 256 // 32)
    # no block's largest value needed more than e4m3's top, nor less than half of it
    amax = x.float().view(64, 8, 32).abs().amax(-1)
    ratio = amax / scales.float()
    assert ratio.max() <= 448 and ratio.min() > 224
    assert torch.equal(torch.frexp(scales.float()).mantissa, torch.full_like(scales.float(), 0.5))
    # e4m3's 3-bit mantissa bounds the relative error once the block scale is divided out
    torch.testing.assert_close(MXFP8.dequantize(q), x.float(), rtol=1 / 16 + 1e-3, atol=2.0e-3)


def test_nvfp4_rounds_block_scales_up_and_packs_low_nibble_first():
    x = _activation()
    q = NVFP4.quantize(x)
    assert q.data.dtype == torch.float4_e2m1fn_x2 and q.data.shape == (64, 128)
    assert q.scale.dtype == FP8_DTYPE and q.tensor_scale.shape == (1,)
    scale = unswizzle_block_scales(q.scale, 64, 16).float() * q.tensor_scale
    amax = x.float().view(64, 16, 16).abs().amax(-1)
    assert (amax / scale).max() <= 6.0 * (1 + 1e-5)
    back = NVFP4.dequantize(q)
    # the coarse values round by at most a quarter of their magnitude, the zero bin to a twelfth of the block's top
    bound = torch.maximum(x.float().abs() * 0.34, scale.repeat_interleave(16, dim=1) * 0.25 * (1 + 1e-3))
    assert ((back - x.float()).abs() <= bound).all()

    known = torch.zeros(1, 64, dtype=torch.bfloat16)
    known[0, 0], known[0, 1] = 6.0, -3.0
    packed = NVFP4.quantize(known, tensor_scale=1.0)
    # the block's top maps to 1.0 * 6: code 7 for +6, sign bit + code 5 for -3
    assert packed.data.view(torch.uint8)[0, 0].item() == (0b1101 << 4) | 0b0111
    torch.testing.assert_close(NVFP4.dequantize(packed)[0, :2], torch.tensor([6.0, -3.0]))


def test_zero_block_stays_finite_and_zero():
    x = torch.zeros(32, 128, dtype=torch.bfloat16)
    for fmt in (MXFP8, NVFP4):
        back = fmt.dequantize(fmt.quantize(x, tensor_scale=1.0))
        assert torch.equal(back, torch.zeros_like(back))


@pytest.mark.parametrize(("amax", "scale"), [
    (None, 1.0), (100.0, 1.0), (2688.0, 1.0), (4080.0, 2.0), (5056.0, 2.0), (39700.0, 16.0), (340000.0, 128.0),
])
def test_nvfp4_input_scale_holds_the_declared_amax_and_never_shrinks(amax, scale):
    assert Nvfp4Format.input_scale(amax) == scale
    assert amax is None or amax <= 2688.0 * scale


def test_nvfp4_unit_range_clips_above_2688_and_a_power_of_two_scale_doesnt():
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    x[0, 0], x[0, 20] = 4080.0, 100.0
    unit = NVFP4.dequantize(NVFP4.quantize(x, tensor_scale=1.0))
    assert unit[0, 0] == 2688.0
    two = NVFP4.dequantize(NVFP4.quantize(x, tensor_scale=Nvfp4Format.input_scale(4080.0)))
    assert two[0, 0] >= 4080.0 * 0.75
    # the block that fit under both quantizes the same way
    assert torch.equal(two[0, 16:32], unit[0, 16:32])


@pytest.mark.parametrize(("fmt", "k", "n", "dtype", "match"), [
    (MXFP8, 3000, 3072, torch.bfloat16, "K a multiple of 128"),
    (NVFP4, 96, 3072, torch.bfloat16, "K a multiple of 64"),
    (MXFP8, 3072, 3000, torch.bfloat16, "N a multiple of 16"),
    (MXFP8, 3072, 3072, torch.float32, "replaces a bf16 GEMM"),
])
def test_guards_refuse_what_the_kernels_cant_run(fmt, k, n, dtype, match):
    with pytest.raises(ValueError, match=match):
        fmt.check(n, k, dtype)


def test_guards_accept_klein_and_z_image_widths():
    for k in (3072, 9216, 12288, 4096, 16384, 3840):
        for fmt in (MXFP8, NVFP4):
            fmt.check(3072, k, torch.bfloat16)


@pytest.mark.parametrize("fmt", [MXFP8, NVFP4], ids=["mxfp8", "nvfp4"])
def test_the_ops_fakes_agree_with_the_reference(fmt):
    weights = fmt.quantize_weight(_activation(48, 256, seed=3), input_amax=None)
    x = _activation(70, 256, seed=4)
    op = torch.ops.mstar.mxfp8_linear if fmt is MXFP8 else torch.ops.mstar.nvfp4_linear
    # the schema test compares outputs with allclose, which has no fp8 kernel on CPU
    result = torch.library.opcheck(
        op, (x, *(weights[name] for name in fmt.buffers)), test_utils=("test_faketensor", "test_aot_dispatch_static"),
    )
    assert set(result.values()) == {"SUCCESS"}


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
], ids=["mxfp8", "nvfp4"])
def test_quantizer_host_contract(monkeypatch, fmt, kernel, block, stored):
    rec = _Recorder()
    monkeypatch.setattr(quant_fp8, kernel, rec)
    monkeypatch.setattr(quant_fp8.triton, "cdiv", lambda a, b: math.ceil(a / b), raising=False)
    rows, k = 70, 3072  # rows don't fill a tile: the grid rounds up, the scales stay tile-padded
    x = torch.zeros(rows, k, dtype=torch.bfloat16)
    tensor_scale = torch.tensor([2.0])
    q = quant_fp8.quant_mxfp8(x) if fmt is MXFP8 else quant_fp8.quant_nvfp4(x, tensor_scale)
    assert q.data.dtype == fmt.data_dtype and q.data.shape == (rows, stored)
    assert q.scale.dtype == fmt.scale_dtype and q.scale.numel() == 128 * (k // block)
    (call,) = rec.calls
    assert call["grid"] == (3, k // block // 4)
    assert call["kwargs"] == {"ROWS": 32, "BLOCKS": 4, "BLOCK": block}
    if fmt is NVFP4:
        assert call["args"][3] is tensor_scale and q.tensor_scale is tensor_scale
    else:
        assert q.tensor_scale is None


# ---------------------------------------------------------------- QuantLinear and how a linear kind takes a format


@pytest.mark.parametrize("fmt", [MXFP8, NVFP4], ids=["mxfp8", "nvfp4"])
def test_quant_linear_reads_a_strided_input_as_its_values(fmt):
    torch.manual_seed(0)
    linear = nn.Linear(256, 128, bias=True, dtype=torch.bfloat16)
    quantized = QuantLinear.from_linear(linear, fmt)
    wide = torch.randn(3, 5, 512, dtype=torch.bfloat16)
    x = wide[..., :256]
    assert not x.is_contiguous()
    out = quantized(x)
    assert out.shape == (3, 5, 128) and out.dtype == torch.bfloat16
    assert torch.equal(out, quantized(x.contiguous()))
    ref = linear(x).float()
    assert (out.float() - ref).norm() / ref.norm() < (0.05 if fmt is MXFP8 else 0.25)


def test_quant_linear_is_lora_capable_with_the_replaced_linears_target():
    fused = FusedColumnLinear(128, {"q": 128, "k": 128}, dtype=torch.bfloat16)
    nn.init.normal_(fused.weight, std=0.05)
    quantized = QuantLinear.from_linear(fused, MXFP8)
    assert quantized.lora_target() == fused.lora_target()
    assert quantized.lora_local_shape() == (128, 256)
    capability = lora_capability(quantized)
    assert capability.capable is not None and issubclass(capability.capable, LoraOutputAdd)
    # bound, the delta adds to the quantized product
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    base = quantized(x)
    quantized.__class__ = capability.capable
    quantized.lora_delta = lambda inp, y: y + 1
    assert torch.equal(quantized(x), base + 1)


class _OwnForward(nn.Linear):
    def forward(self, x):
        return super().forward(x) * 2


def test_linear_kinds_declare_whether_they_take_a_format():
    assert quant_capability(nn.Linear(8, 8)) == QuantCapability(True)
    assert quant_capability(FusedColumnLinear(8, {"a": 8})) == QuantCapability(True)
    assert "forward of its own" in quant_capability(_OwnForward(8, 8)).reason
    replaced = nn.Linear(8, 8)
    replaced.forward = lambda x: x
    assert "replaced on the instance" in quant_capability(replaced).reason
    # a TP linear says why not; the declaration reads no state
    for kind in (ColumnParallelLinear, RowParallelLinear):
        capability = quant_capability(kind.__new__(kind))
        assert capability.capable is False and "tensor-parallel" in capability.reason
    assert quant_capability(QuantLinear(torch.zeros(16, 128, dtype=torch.bfloat16), None, MXFP8)).reason == (
        "already mxfp8"
    )
    for not_a_linear in (nn.LayerNorm(8), nn.Conv2d(3, 8, 2, stride=2), nn.Sequential()):
        assert quant_capability(not_a_linear) is None


# ---------------------------------------------------------------- the declaration, the keep list, the swap


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.keep = nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
        self.fused = FusedColumnLinear(128, {"q": 128, "k": 128}, dtype=torch.bfloat16)
        self.plain = nn.Linear(128, 256, bias=True, dtype=torch.bfloat16)
        for param in self.parameters():
            nn.init.normal_(param, std=0.05)


_DECLARED = QuantizableLinears(candidates=("fused", "plain"), keep_bf16=("keep",), input_amax={"plain": 6000.0})


@pytest.fixture
def block_gemm_gpu(monkeypatch):
    """Both formats available; their ops run the reference on these CPU tensors."""
    for kind in (Mxfp8Format, Nvfp4Format):
        monkeypatch.setattr(kind, "unavailable_reason", lambda self, device: None)


def test_bf16_changes_nothing():
    block = _Block()
    before = dict(block.named_modules())
    assert apply_linear_dtype(block, "bf16", _DECLARED) == []
    assert dict(block.named_modules()) == before


def test_unknown_dtype_is_refused():
    with pytest.raises(ValueError, match="linear_dtype must be one of"):
        apply_linear_dtype(_Block(), "fp3", _DECLARED)


def test_a_gpu_without_the_gemm_stays_bf16(monkeypatch, caplog):
    monkeypatch.setattr(Mxfp8Format, "unavailable_reason", lambda self, device: "no block-scaled GEMM here")
    block = _Block()
    with caplog.at_level(logging.WARNING):
        assert apply_linear_dtype(block, "mxfp8", _DECLARED) == []
    assert isinstance(block.plain, nn.Linear) and "no block-scaled GEMM here" in caplog.text


@pytest.mark.usefixtures("block_gemm_gpu")
@pytest.mark.parametrize("dtype", ["mxfp8", "nvfp4"])
def test_declared_linears_move_and_kept_ones_dont(dtype):
    block = _Block()
    fused_weight = block.fused.weight.data.clone()
    moved = apply_linear_dtype(block, dtype, _DECLARED)
    assert moved == ["fused", "plain"]
    assert isinstance(block.keep, nn.Linear)
    assert isinstance(block.fused, QuantLinear) and isinstance(block.plain, QuantLinear)
    assert not list(block.fused.parameters()), "the bf16 weight goes with the module that held it"
    if dtype == "nvfp4":
        assert block.plain.input_scale.item() == 4.0 and block.fused.input_scale.item() == 1.0

    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 64, 128, generator=g).to(torch.bfloat16)
    out = block.fused(x)
    assert out.dtype == torch.bfloat16 and out.shape == (2, 64, 256)
    ref = x.float() @ fused_weight.float().t()
    limit = 0.05 if dtype == "mxfp8" else 0.25
    assert (out.float() - ref).norm() / ref.norm() < limit


@pytest.mark.usefixtures("block_gemm_gpu")
def test_a_linear_is_replaced_before_the_next_is_quantized(monkeypatch):
    """The load holds one weight set: each bf16 linear is gone from the tree before the next is quantized."""
    block = _Block()
    seen = []
    real = QuantLinear.from_linear.__func__

    def spy(cls, linear, fmt, input_amax=None):
        seen.append(sum(isinstance(m, QuantLinear) for m in block.modules()))
        return real(cls, linear, fmt, input_amax)

    monkeypatch.setattr(QuantLinear, "from_linear", classmethod(spy))
    apply_linear_dtype(block, "mxfp8", _DECLARED)
    assert seen == [0, 1]
    selection = select_linears(_Block(), "mxfp8", _DECLARED)
    assert all(isinstance(name, str) for name in selection.names), "a selection holds names, not modules"


@pytest.mark.usefixtures("block_gemm_gpu")
def test_keep_patterns_win_for_one_deployment(caplog):
    block = _Block()
    with caplog.at_level(logging.INFO):
        assert apply_linear_dtype(block, "mxfp8", _DECLARED, keep=("plain",)) == ["fused"]
    assert isinstance(block.plain, nn.Linear)
    assert "'plain' keeps 1 linears bf16: plain" in caplog.text


def test_keep_list_validation():
    with pytest.raises(ValueError, match="is a list of module-name patterns"):
        check_linear_keep("single_blocks.*")
    with pytest.raises(ValueError, match="is a list of module-name patterns"):
        check_linear_keep([3])
    assert check_linear_keep(None) == ()
    with pytest.raises(ValueError, match="'nothing.here' names none of its candidate linears"):
        select_linears(_Block(), "mxfp8", _DECLARED, keep=("nothing.here",))
    # a declared-bf16 linear isn't a candidate, so keeping it again names nothing
    with pytest.raises(ValueError, match="'keep' names none"):
        select_linears(_Block(), "bf16", _DECLARED, keep=("keep",))
    with pytest.raises(ValueError, match="is a list of module-name patterns"):
        Flux2KleinModel(skip_weight_loading=True, linear_keep_bf16="*")
    with pytest.raises(ValueError, match="is a list of module-name patterns"):
        ZImageModel(skip_weight_loading=True, linear_keep_bf16="*")


def test_patterns_match_segment_by_segment_with_index_ranges():
    match = quant_linear.matches
    assert match("single_blocks.3.out", "single_blocks.0-3.*")
    assert not match("single_blocks.4.out", "single_blocks.0-3.*")
    assert not match("layers.0.attn.qkv", "*.qkv"), "a star never spans a dot"


def test_an_undeclared_linear_is_refused():
    with pytest.raises(ValueError, match="neither a low-precision candidate nor kept"):
        apply_linear_dtype(_Block(), "mxfp8", QuantizableLinears(candidates=("fused",), keep_bf16=("keep",)))


@pytest.mark.usefixtures("block_gemm_gpu")
def test_a_candidate_kind_without_the_capability_is_refused_by_name():
    block = _Block()
    block.plain = _OwnForward(128, 256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="'plain' can't run mxfp8: _OwnForward subclasses nn.Linear"):
        apply_linear_dtype(block, "mxfp8", _DECLARED)
    assert isinstance(block.fused, FusedColumnLinear), "nothing is replaced before every candidate is checked"


# ---------------------------------------------------------------- klein's and Z-Image's declarations


def _linear_names(module):
    return [name for name, child in module.named_modules() if quant_capability(child) is not None]


def test_klein_declaration_classifies_every_linear():
    config = Flux2TransformerConfig(num_layers=2, num_single_layers=3)
    with torch.device("meta"):
        names = _linear_names(Flux2DiT(config))
    selected = [name for name in names if QUANTIZABLE_LINEARS.selected(name)]
    assert len(selected) == 2 * 8 + 3 * 2
    assert set(names) - set(selected) == {
        "time_embed.linear_in", "time_embed.linear_out", "mod_double_img", "mod_double_txt", "mod_single",
        "img_in", "txt_in", "norm_out_mod", "proj_out",
    }
    amax = {name: QUANTIZABLE_LINEARS.amax_of(name) for name in selected}
    assert {name for name, value in amax.items() if value} == {f"double_blocks.{i}.txt_ff.linear_out" for i in (0, 1)}


def test_klein_widths_pass_the_guards():
    for config in (Flux2TransformerConfig(), Flux2TransformerConfig(num_attention_heads=32)):
        d, m = config.hidden_size, config.mlp_hidden_dim
        for k, n in ((d, 3 * d), (d, d), (d, 2 * m), (m, d), (d, 3 * d + 2 * m), (d + m, d)):
            for fmt in (MXFP8, NVFP4):
                fmt.check(n, k, torch.bfloat16)


def test_z_image_declaration_classifies_every_linear_and_its_widths_pass_the_guards():
    with torch.device("meta"):
        dit = ZImageDiT(ZImageTransformerConfig())
    linears = dict((name, m) for name, m in dit.named_modules() if quant_capability(m) is not None)
    selected = [name for name in linears if Z_QUANTIZABLE_LINEARS.selected(name)]
    assert selected
    for name in selected:
        for fmt in (MXFP8, NVFP4):
            fmt.check(*linears[name].weight.shape, torch.bfloat16)
    assert set(Z_QUANTIZABLE_LINEARS.input_amax) <= set(selected), "every amax entry names a candidate"


def test_knobs_default_to_bf16_and_refuse_a_typo():
    assert Flux2KleinModel(skip_weight_loading=True).linear_dtype == "bf16"
    assert ZImageModel(skip_weight_loading=True).linear_dtype == "bf16"
    with pytest.raises(ValueError, match="linear_dtype must be one of"):
        Flux2KleinModel(skip_weight_loading=True, linear_dtype="mxfp4")


# ---------------------------------------------------------------- step schedules


def test_linear_dtype_steps_forms_and_refusals():
    assert check_linear_dtype_steps(None, "mxfp8") is None
    assert check_linear_dtype_steps(0, "mxfp8") is None
    assert check_linear_dtype_steps(2, "bf16") is None, "every step is bf16 anyway"
    steps = check_linear_dtype_steps(1, "mxfp8")
    assert steps == StepDtypes(("bf16", "mxfp8")) and steps.dtypes == ("bf16", "mxfp8")
    assert [steps.at(k) for k in range(4)] == ["bf16", "mxfp8", "mxfp8", "mxfp8"]
    assert check_linear_dtype_steps(["bf16", "nvfp4", "bf16"], "nvfp4").at(5) == "bf16"
    for bad in (-1, True, "1", ["bf16", "mxfp8"], []):
        with pytest.raises(ValueError, match="linear_dtype_steps"):
            check_linear_dtype_steps(bad, "nvfp4")
    with pytest.raises(ValueError, match="must be >= 0"):
        Flux2KleinModel(skip_weight_loading=True, linear_dtype="mxfp8", linear_dtype_steps=-1)


class _Loop(DenoiseLoopSubmodule):
    """The smallest denoise loop the base's real methods run on: one parameter for the device, a key per height."""

    def __init__(self, step_dtypes=None):
        super().__init__(loop_name="denoise_loop", capture_buckets=[("gen", 4), ("gen", 8)], step_dtypes=step_dtypes)
        self.anchor = nn.Parameter(torch.zeros(1))
        self.seen = []

    def bucket_key_for(self, fwd_info):
        return int(fwd_info.step_metadata["h"])

    def schedule_for(self, fwd_info, bucket_key):
        n = int(fwd_info.step_metadata["steps"])
        return SimpleNamespace(num_steps=n, sigmas=torch.linspace(1, 0, n + 1), timesteps=torch.linspace(1000, 0, n))

    def seed_latents(self, fwd_info, bucket_key, generator):
        return torch.zeros(bucket_key, 2)

    def request_inputs(self, fwd_info, inputs, bucket_key):
        return {}

    def num_tokens(self, bucket_key):
        return bucket_key

    def capture_request_inputs(self, bucket_key, device):
        return {"latents": torch.zeros(bucket_key, 2, device=device)}

    def denoise(self, engine_inputs, bucket_key, latents, timestep, sigma, sigma_next, **cond):
        self.seen.append((bucket_key, self.step_dtype))
        return latents + 1


def _info(k, h=4, rid=0, metadata=True):
    return CurrentForwardPassInfo(
        request_id=f"r{rid}", graph_walk="gen", fwd_index=k, random_seed=0, max_tokens=0,
        step_metadata={"h": h, "steps": 4} if metadata else {},
        dynamic_loop_iter_counts={"denoise_loop": k}, rid_handle=rid,
    )


def test_a_steps_key_is_the_same_from_prepare_inputs_and_cg_key_info():
    loop = _Loop(StepDtypes(("bf16", "mxfp8")))
    latents = {"latents": [torch.zeros(4, 2)]}
    for k, dtype in ((0, "bf16"), (1, "mxfp8"), (3, "mxfp8")):
        info = _info(k)
        prepared = loop.prepare_inputs("gen", info, {} if k == 0 else latents)
        assert prepared.resource_step_info == StepBucket(4, dtype)
        assert loop.cg_key_info("gen", {0: info}) == prepared.resource_step_info
    # a padding row carries no metadata and doesn't vote; a mix of dtypes shares no graph
    assert loop.cg_key_info("gen", {0: _info(0), -1: _info(0, metadata=False)}) == StepBucket(4, "bf16")
    assert loop.cg_key_info("gen", {0: _info(0), 1: _info(2, rid=1)}) is None
    assert loop.split_batches_by_capture_key("gen")



def test_cg_key_info_takes_the_step_from_the_input_metadata_when_it_has_one():
    """The scheduler groups a speculation chain's rows at the step they run next, while their request info still
    holds the step in flight's; the lease reads the step's own metadata the same way."""
    from mstar.model.submodule_base import InputMetadata

    loop = _Loop(StepDtypes(("bf16", "mxfp8")))
    ahead = {0: InputMetadata(dynamic_loop_iter_counts={"denoise_loop": 1})}
    assert loop.cg_key_info("gen", {0: _info(0)}, per_request_input_metadata=ahead) == StepBucket(4, "mxfp8")
    assert loop.cg_key_info("gen", {0: _info(0)}, per_request_input_metadata={0: InputMetadata()}) == StepBucket(
        4, "bf16",
    )

def test_without_a_schedule_the_step_key_is_the_requests():
    loop = _Loop()
    info = _info(0)
    assert loop.prepare_inputs("gen", info, {}).resource_step_info == 4 == loop.cg_key_info("gen", {0: info})
    assert not loop.split_batches_by_capture_key("gen")
    assert [config.additional_key_info for config in loop.get_accelerator_graph_configs(torch.device("cpu"))] == [4, 8]


def test_a_schedule_captures_each_bucket_per_dtype_and_denoise_sees_the_plain_key():
    loop = _Loop(StepDtypes(("bf16", "mxfp8")))
    keys = [config.additional_key_info for config in loop.get_accelerator_graph_configs(torch.device("cpu"))]
    assert keys == [StepBucket(4, "bf16"), StepBucket(4, "mxfp8"), StepBucket(8, "bf16"), StepBucket(8, "mxfp8")]
    inputs = [loop.prepare_inputs("gen", _info(0, rid=rid), {}) for rid in (0, 1)]
    assert loop.can_batch(None, inputs)
    engine_inputs = ModelInputsFromEngine(request_ids=[0, 1], per_request_info={})
    loop.forward_batched("gen", engine_inputs, **loop.preprocess("gen", engine_inputs, inputs))
    assert loop.seen == [(4, "bf16")]
    later = loop.prepare_inputs("gen", _info(1, rid=2), {"latents": [torch.zeros(4, 2)]})
    assert not loop.can_batch(None, [inputs[0], later])


def _tiny_klein(monkeypatch, available: bool, **kwargs):
    """Flux2KleinModel's real dit build on a two-block transformer with random bf16 weights."""
    tiny = Flux2TransformerConfig(
        num_layers=1, num_single_layers=1, num_attention_heads=1, attention_head_dim=128, joint_attention_dim=128,
        timestep_guidance_channels=128, axes_dims_rope=(32, 32, 32, 32),
    )
    from mstar.model.flux2_klein import weight_loader

    def build(config, snapshot, device):
        torch.manual_seed(0)
        return Flux2DiT(config.transformer).to(torch.bfloat16)

    monkeypatch.setattr(weight_loader, "build_transformer", build)
    reason = None if available else "no block-scaled GEMM here"
    monkeypatch.setattr(Mxfp8Format, "unavailable_reason", lambda self, device: reason)
    model = Flux2KleinModel(attention_backend="sdpa", compile=False, accelerator_graph=False, **kwargs)
    model.set_config(Flux2KleinConfig(transformer=tiny))
    model._snapshot = Path("/nonexistent")
    copies = []
    real_deepcopy = __import__("copy").deepcopy
    monkeypatch.setattr("mstar.model.flux2_klein.flux2_klein_model.copy.deepcopy",
                        lambda module: copies.append(module) or real_deepcopy(module))
    return model._create_submodule("dit", torch.device("cpu")), copies


def test_the_twin_is_a_copy_taken_before_quantizing_and_only_when_something_was(monkeypatch):
    sub, copies = _tiny_klein(monkeypatch, available=True, linear_dtype="mxfp8", linear_dtype_steps=1)
    assert len(copies) == 1
    assert sub.step_dtypes == StepDtypes(("bf16", "mxfp8")) and sub.split_batches_by_capture_key("image_gen")
    assert any(isinstance(m, QuantLinear) for m in sub.transformer.modules())
    assert not any(isinstance(m, QuantLinear) for m in sub.bf16_transformer.modules())
    twin_qkv = sub.bf16_transformer.double_blocks[0].attn.img_qkv
    assert twin_qkv.weight.dtype == torch.bfloat16

    for kwargs in ({"linear_dtype": "mxfp8"}, {"linear_dtype": "bf16", "linear_dtype_steps": 1}):
        sub, copies = _tiny_klein(monkeypatch, available=True, **kwargs)
        assert not copies and sub.bf16_transformer is None and sub.step_dtypes is None
    # no GEMM on this GPU: every step is bf16, so no twin and no schedule
    sub, copies = _tiny_klein(monkeypatch, available=False, linear_dtype="mxfp8", linear_dtype_steps=1)
    assert not copies and sub.bf16_transformer is None and sub.step_dtypes is None
    assert not sub.split_batches_by_capture_key("image_gen")
    # the keep list covering every candidate quantizes nothing either
    keep = ["double_blocks.*.*.*", "double_blocks.*.*_ff.*", "single_blocks.*.*"]
    sub, copies = _tiny_klein(monkeypatch, available=True, linear_dtype="mxfp8", linear_dtype_steps=1,
                              linear_keep_bf16=keep)
    assert not copies and sub.bf16_transformer is None
