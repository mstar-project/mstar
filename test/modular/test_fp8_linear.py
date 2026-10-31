"""fp8 linears (``linear_dtype: fp8``): the platform's e4m3 format, per-token and per-channel
quantisation, ``mstar::fp8_rowwise_mm``'s reference, the module swap, and which linears each
DiT declares. CPU only: the op's reference runs here; the GPU kernels are checked by
test_kernel_ops_parity.py on a GPU."""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mstar import platforms
from mstar.kernels.fp8_gemm import FP8_ROWWISE_MM, quantize_rows
from mstar.model.components import fp8_linear
from mstar.model.components.fp8_linear import (
    Fp8Linear,
    StepDtypes,
    SteppedFp8Linear,
    apply_linear_dtype,
    check_linear_dtype,
    check_linear_dtype_steps,
    quantize_linears,
)
from mstar.model.components.linear import FusedColumnLinear
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform

FP8_DTYPES = (torch.float8_e4m3fnuz, torch.float8_e4m3fn)


class _Rocm(RocmPlatform):
    def __init__(self, arch: str):
        self._arch = arch

    def arch(self, device=None) -> str:
        return self._arch


class _Cuda(CudaPlatform):
    def __init__(self, capability: tuple[int, int]):
        self._capability = capability

    def cuda_capability(self, device=None) -> tuple[int, int]:
        return self._capability


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm())


def test_the_fp8_format_comes_from_the_architecture():
    assert _Rocm("gfx942").fp8_dtype() == torch.float8_e4m3fnuz
    assert _Rocm("gfx950").fp8_dtype() == torch.float8_e4m3fn
    assert _Rocm("gfx90a").fp8_dtype() is None
    assert _Cuda((9, 0)).fp8_dtype() == torch.float8_e4m3fn
    assert _Cuda((8, 9)).fp8_dtype() == torch.float8_e4m3fn
    assert _Cuda((8, 0)).fp8_dtype() is None
    assert CpuPlatform().fp8_dtype() is None


def test_the_gemm_needs_hopper_on_nvidia(monkeypatch):
    from mstar.kernels import fp8_gemm

    monkeypatch.setattr(platforms, "resolve_platform", lambda: _Cuda((8, 9)))
    with pytest.raises(RuntimeError, match="sm_90 or newer"):
        fp8_gemm._scaled_mm()
    monkeypatch.setattr(platforms, "resolve_platform", lambda: _Cuda((9, 0)))
    assert callable(fp8_gemm._scaled_mm())
    monkeypatch.setattr(platforms, "resolve_platform", lambda: _Rocm("gfx90a"))
    with pytest.raises(RuntimeError, match="no fp8 GEMM"):
        fp8_gemm._scaled_mm()


@pytest.mark.parametrize("fp8", FP8_DTYPES, ids=str)
def test_quantize_rows_maps_each_rows_largest_value_to_the_format_max(fp8):
    torch.manual_seed(0)
    t = torch.randn(64, 256) * torch.logspace(-3, 3, 64)[:, None]
    t[5] = 0
    q, scale = quantize_rows(t, fp8)
    assert q.dtype == fp8 and scale.dtype == torch.float32 and scale.shape == (64, 1)
    peak = q.float().abs().amax(dim=-1)
    assert torch.all(peak[torch.arange(64) != 5] == torch.finfo(fp8).max)
    assert torch.all(q[5].float() == 0) and torch.isfinite(scale).all()
    # e4m3 keeps 3 mantissa bits: rounding moves a normal value by at most 2^-4 of itself
    back = q.float() * scale
    normal = t.abs() >= scale * torch.finfo(fp8).smallest_normal
    assert torch.all(((back - t).abs() <= t.abs() * 2 ** -4)[normal])


@pytest.mark.parametrize("bias", [False, True])
def test_the_ops_reference_is_the_dequantised_product(bias):
    torch.manual_seed(0)
    xq, x_scale = quantize_rows(torch.randn(48, 64), torch.float8_e4m3fnuz)
    wq, w_scale = quantize_rows(torch.randn(32, 64), torch.float8_e4m3fnuz)
    b = torch.randn(32).to(torch.bfloat16) if bias else None
    out = FP8_ROWWISE_MM(xq, x_scale, wq, w_scale.T.contiguous(), b, torch.bfloat16)
    expected = (xq.float() * x_scale) @ (wq.float() * w_scale).T + (b.float() if bias else 0)
    assert out.dtype == torch.bfloat16 and out.shape == (48, 32)
    torch.testing.assert_close(out, expected.to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("fp8", FP8_DTYPES, ids=str)
@pytest.mark.parametrize("bias", [False, True])
def test_an_fp8_linear_tracks_the_bf16_linear(fp8, bias):
    torch.manual_seed(0)
    linear = nn.Linear(128, 96, bias=bias).to(torch.bfloat16)
    x = torch.randn(2, 7, 128).to(torch.bfloat16)
    quantized = Fp8Linear.from_linear(linear, fp8)
    out = quantized(x)
    assert out.shape == (2, 7, 96) and out.dtype == torch.bfloat16
    assert quantized.weight.dtype == fp8 and quantized.weight_scale.shape == (1, 96)
    assert _rel(out, F.linear(x, linear.weight, linear.bias)) < 0.05


def test_a_fused_column_linear_quantises_per_output_row():
    torch.manual_seed(0)
    fused = FusedColumnLinear(64, {"q": 32, "k": 32, "v": 48}).to(torch.bfloat16)
    nn.init.normal_(fused.weight)
    with torch.no_grad():
        fused.weight[32:64] *= 100  # one shard far larger than the others
    quantized = Fp8Linear.from_linear(fused, torch.float8_e4m3fnuz)
    x = torch.randn(5, 64).to(torch.bfloat16)
    reference = fused(x)
    out = quantized(x)
    for rows in (slice(0, 32), slice(32, 64), slice(64, 112)):
        assert _rel(out[:, rows], reference[:, rows]) < 0.05


def test_an_fp8_linear_is_one_op_under_dynamo():
    torch.manual_seed(0)
    quantized = Fp8Linear.from_linear(nn.Linear(64, 32).to(torch.bfloat16), torch.float8_e4m3fnuz)
    x = torch.randn(3, 64).to(torch.bfloat16)
    compiled = torch.compile(quantized, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), quantized(x), rtol=0, atol=0)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = FusedColumnLinear(32, {"q": 32, "k": 32, "v": 32})
        self.out = nn.Linear(32, 32)
        self.mod = nn.Linear(32, 64)
        self.odd = nn.Linear(32, 40)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(2))
        self.proj_out = nn.Linear(32, 32)


def test_only_matching_linears_whose_shapes_fit_are_swapped():
    net = _Net()
    replaced = quantize_linears(net, ("blocks.*.qkv", "blocks.*.out", "blocks.*.odd"), torch.float8_e4m3fnuz)
    assert replaced == ["blocks.0.qkv", "blocks.0.out", "blocks.1.qkv", "blocks.1.out"]
    for block in net.blocks:
        assert isinstance(block.qkv, Fp8Linear) and isinstance(block.out, Fp8Linear)
        assert type(block.mod) is nn.Linear and type(block.odd) is nn.Linear  # 40 rows: not a multiple of 16
    assert type(net.proj_out) is nn.Linear


def test_patterns_match_segment_by_segment_with_index_ranges():
    from mstar.model.components.fp8_linear import matches

    assert matches("single_blocks.3.out", "single_blocks.0-3.*")
    assert not matches("single_blocks.4.out", "single_blocks.0-3.*")
    assert matches("single_blocks.22.qkv_mlp", "single_blocks.20-23.qkv_mlp")
    assert matches("double_blocks.1.img_ff.linear_in", "double_blocks.*.img_ff.linear_*")
    assert not matches("double_blocks.1.img_ff.linear_in", "double_blocks.*")  # * stays in one segment
    assert not matches("single_blocks.x.out", "single_blocks.0-3.out")


def test_keep_leaves_the_named_linears_in_bf16(monkeypatch):
    monkeypatch.setattr(platforms, "resolve_platform", lambda: _Rocm("gfx942"))
    monkeypatch.setattr(fp8_linear, "fp8_unavailable_reason", lambda: None)
    net = _Net()
    replaced = apply_linear_dtype(net, "fp8", ("blocks.*.qkv", "blocks.*.out"), "net", keep=("blocks.1-1.*",))
    assert replaced == ["blocks.0.qkv", "blocks.0.out"]
    assert isinstance(net.blocks[1].qkv, FusedColumnLinear) and type(net.blocks[1].out) is nn.Linear
    with pytest.raises(ValueError, match="name none of its fp8-eligible linears"):
        apply_linear_dtype(_Net(), "fp8", ("blocks.*.qkv",), "net", keep=("blocks.*.mod",))


def test_linear_dtype_bf16_keeps_the_weights_and_unknown_values_fail():
    net = _Net()
    assert apply_linear_dtype(net, "bf16", ("blocks.*.qkv",), "net") == []
    assert isinstance(net.blocks[0].qkv, FusedColumnLinear)
    with pytest.raises(ValueError, match="linear_dtype 'fp16'"):
        check_linear_dtype("fp16")
    with pytest.raises(ValueError, match="linear_dtype 'int8'"):
        apply_linear_dtype(net, "int8", ("blocks.*.qkv",), "net")


def test_linear_dtype_fp8_without_an_fp8_gemm_keeps_bf16(monkeypatch, caplog):
    monkeypatch.setattr(platforms, "resolve_platform", lambda: _Rocm("gfx90a"))
    net = _Net()
    with caplog.at_level(logging.WARNING, logger=fp8_linear.__name__):
        assert apply_linear_dtype(net, "fp8", ("blocks.*.qkv",), "net") == []
    assert "rocm gfx90a has no fp8 GEMM" in caplog.text
    assert isinstance(net.blocks[0].qkv, FusedColumnLinear)


def test_linear_dtype_fp8_takes_the_platforms_format(monkeypatch):
    monkeypatch.setattr(platforms, "resolve_platform", lambda: _Rocm("gfx942"))
    monkeypatch.setattr(fp8_linear, "fp8_unavailable_reason", lambda: None)
    net = _Net()
    assert apply_linear_dtype(net, "fp8", ("blocks.*.qkv",), "net") == ["blocks.0.qkv", "blocks.1.qkv"]
    assert net.blocks[1].qkv.weight.dtype == torch.float8_e4m3fnuz
    with pytest.raises(ValueError, match="matched no linear"):
        apply_linear_dtype(_Net(), "fp8", ("nothing.*",), "net")


def _flux_dit():
    from mstar.model.flux2_klein.components.transformer import Flux2DiT
    from mstar.model.flux2_klein.config import Flux2TransformerConfig

    return Flux2DiT(Flux2TransformerConfig()), 5 * 4 + 20 * 2, (
        "time_embed.", "mod_double_img", "mod_double_txt", "mod_single", "img_in", "txt_in", "norm_out_mod",
        "proj_out", ".txt_")


def _z_image_dit():
    from mstar.model.z_image.components.transformer import ZImageDiT
    from mstar.model.z_image.config import ZImageTransformerConfig

    config = ZImageTransformerConfig()
    return ZImageDiT(config), (config.n_refiner_layers + config.n_layers) * 4, (
        "time_in", "time_out", "x_embedder", "cap_proj", "final_mod", "final_proj", "context_refiner.", ".adaln")


def _wan_dit():
    from mstar.model.wan22.components.dit import Wan22DiT
    from mstar.model.wan22.config import Wan22Config

    config = Wan22Config()
    return Wan22DiT(config), config.num_layers * 8, (
        "condition_embedder.", "proj_out", "patch_embedding", "cross_attn.k_proj", "cross_attn.v_proj")


def test_step_dtypes_index_steps_and_past_the_list_take_its_last_entry():
    policy = StepDtypes(["bf16", "bf16", "fp8", "bf16"])
    assert [policy.at(k) for k in range(6)] == ["bf16", "bf16", "fp8", "bf16", "bf16", "bf16"]
    assert policy.dtypes == ("bf16", "fp8")
    assert policy.active == "bf16"


@pytest.mark.parametrize(
    ("steps", "dtype", "message"),
    [
        (["bf16", "fp8"], "bf16", "needs linear_dtype: fp8"),
        (["bf16", "fp16"], "fp8", "are not in"),
        (["fp8", "fp8"], "fp8", "name one dtype"),
        (["bf16"], "fp8", "name one dtype"),
    ],
)
def test_linear_dtype_steps_are_checked_when_the_model_is_built(steps, dtype, message):
    with pytest.raises(ValueError, match=message):
        check_linear_dtype_steps(steps, dtype)


@pytest.mark.parametrize("steps", [None, []])
def test_no_linear_dtype_steps_is_no_policy(steps):
    assert check_linear_dtype_steps(steps, "fp8") is None
    assert check_linear_dtype_steps(steps, "bf16") is None


def _stepped_net(monkeypatch) -> tuple[_Net, StepDtypes]:
    monkeypatch.setattr(platforms, "resolve_platform", lambda: _Rocm("gfx942"))
    monkeypatch.setattr(fp8_linear, "fp8_unavailable_reason", lambda: None)
    net = _Net()
    policy = check_linear_dtype_steps(["bf16", "fp8"], "fp8")
    apply_linear_dtype(net, "fp8", ("blocks.*.out",), "test", policy=policy)
    return net, policy


def test_a_policy_keeps_both_sets_of_weights_and_runs_the_active_one(monkeypatch):
    net, policy = _stepped_net(monkeypatch)
    stepped = net.blocks[0].out
    assert isinstance(stepped, SteppedFp8Linear)
    assert stepped.fp8.weight.dtype == torch.float8_e4m3fnuz
    assert type(stepped.bf16) is nn.Linear
    x = torch.randn(32, 32)
    policy.active = "bf16"
    bf16_out = F.linear(x, stepped.bf16.weight, stepped.bf16.bias)
    assert torch.equal(stepped(x), bf16_out)
    policy.active = "fp8"
    fp8_out = stepped(x)
    assert torch.equal(fp8_out, stepped.fp8(x))
    assert not torch.equal(fp8_out, bf16_out)


def test_a_compiled_forward_follows_the_policy(monkeypatch):
    # dynamo guards on the switch: the graph traced for one dtype must not serve the other
    net, policy = _stepped_net(monkeypatch)
    stepped = net.blocks[0].out
    x = torch.randn(32, 32)
    compiled = torch.compile(stepped.forward, backend="eager")
    for active in ("bf16", "fp8", "bf16", "fp8"):
        policy.active = active
        want = stepped.bf16(x) if active == "bf16" else stepped.fp8(x)
        assert torch.equal(compiled(x), want)


def test_without_an_fp8_gemm_a_policy_changes_nothing(monkeypatch):
    monkeypatch.setattr(fp8_linear, "fp8_unavailable_reason", lambda: "no fp8 GEMM here")
    net = _Net()
    policy = check_linear_dtype_steps(["bf16", "fp8"], "fp8")
    assert apply_linear_dtype(net, "fp8", ("blocks.*.out",), "test", policy=policy) == []
    assert type(net.blocks[0].out) is nn.Linear


@pytest.mark.parametrize("build", [_flux_dit, _z_image_dit, _wan_dit], ids=["flux2_klein", "z_image", "wan22"])
def test_each_dit_declares_its_token_stream_linears(build):
    """At the shipped dimensions (on the meta device): every declared pattern matches, every
    match is a linear the fp8 GEMM takes, and the small projections are not among them."""
    with torch.device("meta"):
        dit, expected, kept = build()
    patterns = type(dit).FP8_LINEARS
    names = {name for name, _ in dit.named_modules()}
    for pattern in patterns:
        assert any(fp8_linear.matches(name, pattern) for name in names), pattern
    replaced = quantize_linears(dit, patterns, torch.float8_e4m3fnuz)
    assert len(replaced) == expected
    for name in replaced:
        assert not any(k in name for k in kept), name
    linears = [n for n, m in dit.named_modules() if isinstance(m, (nn.Linear, FusedColumnLinear))]
    assert linears and all(any(k in n for k in kept) for n in linears), linears


def test_the_models_take_linear_dtype():
    from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinModel
    from mstar.model.z_image.z_image_model import ZImageModel

    assert Flux2KleinModel(model_path_hf="x").linear_dtype == "bf16"
    assert ZImageModel(model_path_hf="x", linear_dtype="fp8").linear_dtype == "fp8"
    keep = Flux2KleinModel(model_path_hf="x", fp8_keep_bf16=["single_blocks.0-1.*"]).fp8_keep_bf16
    assert keep == ["single_blocks.0-1.*"]
    with pytest.raises(ValueError, match="linear_dtype"):
        Flux2KleinModel(model_path_hf="x", linear_dtype="fp16")


# --- on a GPU with fp8 GEMMs -----------------------------------------------------------------------------------------


def _gpu_fp8_dtype():
    if not torch.cuda.is_available():
        return None
    return platforms.current_platform.fp8_dtype()


gpu = pytest.mark.skipif(_gpu_fp8_dtype() is None, reason="needs a GPU with fp8 GEMMs")


@gpu
def test_a_compiled_fp8_linear_replays_in_a_graph():
    torch.manual_seed(0)
    linear = nn.Linear(1024, 3072, bias=True, device="cuda", dtype=torch.bfloat16)
    quantized = Fp8Linear.from_linear(linear, _gpu_fp8_dtype())
    x = torch.randn(2, 300, 1024, device="cuda", dtype=torch.bfloat16)
    eager = quantized(x)
    assert _rel(eager, F.linear(x, linear.weight, linear.bias)) < 0.05
    compiled = torch.compile(quantized, fullgraph=True, dynamic=False)
    out = compiled(x)
    # inductor's fused quantisation may round a scale differently from eager's; nothing more
    assert _rel(out, eager) < 1e-2
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        compiled(x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, out)


@gpu
def test_a_compiled_fp8_flux_dit_tracks_the_bf16_one():
    import copy

    from mstar.model.components.diffusion.compile_utils import compile_transformer_forward
    from mstar.model.flux2_klein.components.transformer import Flux2DiT
    from mstar.model.flux2_klein.config import Flux2TransformerConfig

    config = Flux2TransformerConfig(
        in_channels=64, out_channels=64, num_layers=1, num_single_layers=2, attention_head_dim=64,
        num_attention_heads=4, joint_attention_dim=256, timestep_guidance_channels=256, axes_dims_rope=(16, 16, 16, 16),
    )
    torch.manual_seed(0)
    dit = Flux2DiT(config).to("cuda", torch.bfloat16)
    for p in dit.parameters():
        nn.init.normal_(p, std=0.05)
    fp8 = copy.deepcopy(dit)
    assert len(quantize_linears(fp8, Flux2DiT.FP8_LINEARS, _gpu_fp8_dtype())) == 8
    compile_transformer_forward(fp8, exact_ops=["norms"])
    img = torch.randn(1, 256, 64, device="cuda", dtype=torch.bfloat16)
    txt = torch.randn(1, 32, 256, device="cuda", dtype=torch.bfloat16)
    t = torch.tensor([0.7], device="cuda", dtype=torch.bfloat16)
    rope = (torch.ones(288, 64, device="cuda"), torch.zeros(288, 64, device="cuda"))
    with torch.no_grad():
        assert _rel(fp8(img, txt, t, rope), dit(img, txt, t, rope)) < 0.08
