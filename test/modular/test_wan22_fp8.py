"""Wan2.2 dense FP8 selection and CPU reference checks (no checkpoint or GPU)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.model.components.dense_fp8_format import DENSE_FP8, DenseFp8Format
from mstar.model.components.diffusion.denoise_loop import StepBucket
from mstar.model.components.diffusion.ffn import gelu_tanh_ffn
from mstar.model.components.fp8_linear import reference as fp8_reference
from mstar.model.components.lora.capable import LoraBound, make_lora_capable
from mstar.model.components.mlp import MLP
from mstar.model.components.quant_linear import QuantLinear, StepDtypes, quant_capability
from mstar.model.wan22.components.dit import QUANTIZABLE_LINEARS, Wan22DiT
from mstar.model.wan22.config import Wan22Config
from mstar.model.wan22.submodules import DENOISE_LOOP_NAME
from mstar.model.wan22.wan22_model import Wan22Model
from mstar.utils.fp8_block import FP8_DTYPE, dequantize_fp8_block_weight


def _tiny_config(num_layers: int = 1, compile_dit: bool = False) -> Wan22Config:
    return Wan22Config(
        num_attention_heads=2, attention_head_dim=64, num_layers=num_layers, ffn_dim=256,
        in_channels=8, out_channels=8, text_dim=128, freq_dim=128,
        rope_max_seq_len=16, patch_size=(1, 2, 2), compile_dit=compile_dit,
    )


def _tiny_dit(num_layers: int = 1, compile_dit: bool = False) -> Wan22DiT:
    torch.manual_seed(5)
    dit = Wan22DiT(_tiny_config(num_layers=num_layers, compile_dit=compile_dit)).cast_serving_dtypes().eval()
    # The real loader fills all parameters; a direct constructor leaves its
    # scale-shift tables uninitialized, so give every parameter finite values.
    with torch.no_grad():
        for param in dit.parameters():
            param.normal_(std=0.02)
    return dit


def test_wan_fp8_declaration_covers_every_linear_and_only_block_gemms():
    with torch.device("meta"):
        dit = Wan22DiT(Wan22Config(num_layers=2))
    linears = {name: layer for name, layer in dit.named_modules() if quant_capability(layer) is not None}
    selected = {name for name in linears if QUANTIZABLE_LINEARS.selected(name)}
    assert len(selected) == 20
    assert set(linears) - selected == {
        "condition_embedder.time_embedder.linear_in", "condition_embedder.time_embedder.linear_out",
        "condition_embedder.time_proj", "condition_embedder.text_embedder.linear_in",
        "condition_embedder.text_embedder.linear_out", "proj_out",
    }
    for name in selected:
        layer = linears[name]
        assert layer.bias is not None
        DENSE_FP8.check(*layer.weight.shape, torch.bfloat16)


def test_wan_fp8_knob_is_opt_in_and_rejects_other_formats():
    assert Wan22Model("unused", skip_weight_loading=True).linear_dtype == "bf16"
    for valid in ("bf16", "dense_fp8", "mxfp8", "nvfp4"):
        assert Wan22Model("unused", skip_weight_loading=True, linear_dtype=valid).linear_dtype == valid
    for invalid in ("fp8", "mxfp4", None):
        with pytest.raises(ValueError, match="linear_dtype must be"):
            Wan22Model("unused", skip_weight_loading=True, linear_dtype=invalid)
    with pytest.raises(ValueError, match="linear_keep_bf16"):
        Wan22Model("unused", skip_weight_loading=True, linear_dtype="dense_fp8", linear_keep_bf16="blocks.0.*.*")
    for bad_steps in (-1, True, ["bf16", "mxfp8"]):
        with pytest.raises(ValueError, match="linear_dtype_steps"):
            Wan22Model("unused", skip_weight_loading=True, linear_dtype="dense_fp8", linear_dtype_steps=bad_steps)


def test_wan_dit_loading_keeps_bf16_default_and_quantizes_only_blocks(monkeypatch):
    from mstar.model.wan22 import weight_loader

    monkeypatch.setattr(weight_loader, "build_wan22_dit", lambda *args, **kwargs: _tiny_dit())
    monkeypatch.setattr(Wan22Model, "_refresh_checkpoint_defaults", lambda self: None)
    # Run the existing dense kernel's CPU reference through the model's actual
    # loading path. A real CPU deployment leaves the requested format in bf16.
    monkeypatch.setattr(DenseFp8Format, "unavailable_reason", lambda self, device: None)

    def load(dtype):
        model = Wan22Model("unused", linear_dtype=dtype)
        model.config = _tiny_config()
        return model.get_submodule("dit").transformer

    bf16, fp8 = load("bf16"), load("dense_fp8")
    block = fp8.blocks[0]
    assert all(isinstance(getattr(block.self_attn, name), QuantLinear) for name in
               ("q_proj", "k_proj", "v_proj", "o_proj"))
    assert all(isinstance(getattr(block.cross_attn, name), QuantLinear) for name in
               ("q_proj", "k_proj", "v_proj", "o_proj"))
    assert isinstance(block.ffn.linear_in, QuantLinear)
    assert isinstance(block.ffn.linear_out, QuantLinear)
    assert isinstance(fp8.condition_embedder.time_embedder.linear_in, nn.Linear)
    assert isinstance(fp8.condition_embedder.time_proj, nn.Linear)
    assert isinstance(fp8.proj_out, nn.Linear)
    assert isinstance(bf16.blocks[0].ffn.linear_in, nn.Linear)
    assert block.self_attn.q_proj.bias.dtype == torch.bfloat16
    assert block.self_attn.q_proj.weight.dtype == torch.uint8

    torch.manual_seed(6)
    latent = torch.randn(1, 8, 1, 2, 2).bfloat16()
    text = torch.randn(1, 8, 128).bfloat16()
    timestep = torch.ones(1, 1)
    with torch.no_grad():
        expected = bf16(latent, timestep, text)
        actual = fp8(latent, timestep, text)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    # A small deterministic smoke check, not a checkpoint quality criterion.
    assert (actual.float() - expected.float()).abs().max() < 0.01


def test_dense_fp8_zeros_and_bias_match_the_dequantized_reference():
    weight = torch.zeros(256, 256, dtype=torch.bfloat16)
    weight[:64, :64] = 0.125
    bias = torch.linspace(-1, 1, 256).to(torch.bfloat16)
    q = QuantLinear(weight, bias, DENSE_FP8)
    assert q.weight.dtype == torch.uint8
    assert q.weight_scale_inv.dtype == torch.float32
    assert torch.isfinite(q.weight_scale_inv).all()
    assert q.weight_scale_inv[0, 0] == 0.125 / 448
    assert torch.equal(q.weight_scale_inv[1:, :], torch.ones(1, 2))
    assert q.weight_scale_inv[0, 1] == 1
    x = torch.randn(2, 256).to(torch.bfloat16)
    from mstar.utils.fp8_block import dequantize_fp8_block_weight

    dequant = dequantize_fp8_block_weight(q.weight, q.weight_scale_inv, (128, 128)).to(x.dtype)
    expected = torch.nn.functional.linear(x, dequant) + bias
    torch.testing.assert_close(q(x), expected, rtol=0, atol=0)


def test_dense_fp8_refuses_misaligned_or_non_bf16_weights():
    with pytest.raises(ValueError, match="multiples of 128"):
        DENSE_FP8.check(127, 128, torch.bfloat16)
    with pytest.raises(ValueError, match="bf16 GEMM"):
        DENSE_FP8.check(128, 128, torch.float32)


def test_compiled_ffn_dispatches_quant_linear_through_its_forward(monkeypatch):
    """Exercise the CUDA/compile condition on CPU without claiming a GPU run."""
    class PretendCuda(torch.Tensor):
        @property
        def is_cuda(self) -> bool:
            return True

    torch.manual_seed(7)
    mlp = MLP(128, 256, 128, activation="gelu_tanh", bias=True).bfloat16()
    mlp.linear_in = QuantLinear.from_linear(mlp.linear_in, DENSE_FP8)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(
        DenseFp8Format, "linear",
        lambda self, rows, weight, scale: fp8_reference(rows, weight, scale, self.block_size),
    )
    expected = mlp(x)
    pretend_cuda = torch.Tensor._make_subclass(PretendCuda, x, False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    actual = gelu_tanh_ffn(pretend_cuda, mlp)
    torch.testing.assert_close(actual.as_subclass(torch.Tensor), expected, rtol=0, atol=0)


def test_compiled_ffn_keeps_lora_delta_for_bf16_linear(monkeypatch):
    """A LoRA-bound bf16 nn.Linear subclasses nn.Linear and must not take _addmm_activation."""
    class PretendCuda(torch.Tensor):
        @property
        def is_cuda(self) -> bool:
            return True

    torch.manual_seed(8)
    mlp = MLP(128, 256, 128, activation="gelu_tanh", bias=True).bfloat16()
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    base = mlp(x)
    make_lora_capable(mlp.linear_in)
    assert isinstance(mlp.linear_in, nn.Linear) and isinstance(mlp.linear_in, LoraBound)
    mlp.linear_in.lora_delta = lambda inp, y: y + 1.0
    expected = mlp(x)
    assert not torch.equal(expected, base)

    pretend_cuda = torch.Tensor._make_subclass(PretendCuda, x, False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    actual = gelu_tanh_ffn(pretend_cuda, mlp)
    torch.testing.assert_close(actual.as_subclass(torch.Tensor), expected, rtol=0, atol=0)


def test_dense_fp8_extreme_and_zero_blocks_remain_finite():
    weight = torch.zeros(256, 256, dtype=torch.bfloat16)
    weight[:128, :128] = 1e-30
    weight[:128, 128:] = 480.0
    weight[128:, :128] = -1e30
    q = QuantLinear(weight, None, DENSE_FP8)
    fp8_vals = q.weight.view(FP8_DTYPE).float()
    assert not torch.isnan(fp8_vals).any() and torch.isfinite(fp8_vals).all()
    assert torch.isfinite(q.weight_scale_inv).all() and (q.weight_scale_inv > 0).all()
    assert q.weight_scale_inv[1, 1] == 1.0
    out = q(torch.randn(2, 256, dtype=torch.bfloat16))
    assert torch.isfinite(out).all()


def _quantize_activations_1x128(x: torch.Tensor) -> torch.Tensor:
    """CPU simulation of GPU _group_quant(x, 128): 1x128 dynamic e4m3 activation quant."""
    rows, cols = x.shape
    groups = x.float().reshape(rows, cols // 128, 128)
    scale = groups.abs().amax(dim=-1).clamp(min=1e-10).div_(448.0)
    q = (groups / scale[:, :, None]).clamp(-448.0, 448.0).to(FP8_DTYPE)
    return (q.float() * scale[:, :, None]).reshape(rows, cols).to(x.dtype)


def test_dense_fp8_w8a8_cpu_simulation_vs_w8a16_and_outliers(monkeypatch):
    """Simulate GPU _w8a8 (1x128 activation e4m3 + 128x128 weight e4m3) on CPU."""
    torch.manual_seed(9)
    weight = (torch.randn(256, 256) * 0.05).bfloat16()
    bias = (torch.randn(256) * 0.02).bfloat16()
    q = QuantLinear(weight, bias, DENSE_FP8)
    x_normal = torch.randn(64, 256, dtype=torch.bfloat16)
    exact = torch.nn.functional.linear(x_normal, weight, bias)
    w_deq = dequantize_fp8_block_weight(q.weight, q.weight_scale_inv, (128, 128)).to(torch.bfloat16)
    out_w8a8 = torch.nn.functional.linear(_quantize_activations_1x128(x_normal), w_deq, bias)
    rel_l2_normal = (out_w8a8.float() - exact.float()).norm() / exact.float().norm()
    assert torch.isfinite(out_w8a8).all() and rel_l2_normal < 0.08

    # A single large channel outlier per 1x128 group expands the shared scale for 127 neighbors.
    # Zeroing the outlier columns isolates the distortion on the 127 co-grouped channels.
    w_no_col0 = weight.clone()
    w_no_col0[:, ::128] = 0
    q_no_col0 = QuantLinear(w_no_col0, bias, DENSE_FP8)
    w_deq_no_col0 = dequantize_fp8_block_weight(
        q_no_col0.weight, q_no_col0.weight_scale_inv, (128, 128),
    ).to(torch.bfloat16)
    x_outlier = x_normal.clone()
    x_outlier[:, ::128] = 500000.0
    exact_outlier = torch.nn.functional.linear(x_outlier, w_no_col0, bias)
    out_w8a16_outlier = fp8_reference(x_outlier, q_no_col0.weight, q_no_col0.weight_scale_inv, (128, 128)) + bias
    out_w8a8_outlier = torch.nn.functional.linear(_quantize_activations_1x128(x_outlier), w_deq_no_col0, bias)
    err_w8a16 = (out_w8a16_outlier.float() - exact_outlier.float()).norm() / exact_outlier.float().norm()
    err_w8a8 = (out_w8a8_outlier.float() - exact_outlier.float()).norm() / exact_outlier.float().norm()
    assert err_w8a8 > 5 * err_w8a16

    # Multi-block tiny Wan22DiT simulation under W8A8 vs BF16.
    monkeypatch.setattr(DenseFp8Format, "unavailable_reason", lambda self, device: None)
    bf16_dit = _tiny_dit(num_layers=4)
    w8a8_dit = _tiny_dit(num_layers=4)
    from mstar.model.components.quant_linear import apply_linear_dtype

    apply_linear_dtype(w8a8_dit, "dense_fp8", QUANTIZABLE_LINEARS, what="Wan2.2 DiT")
    monkeypatch.setattr(
        DenseFp8Format, "linear",
        lambda self, rows, w, s: fp8_reference(_quantize_activations_1x128(rows), w, s, self.block_size),
    )
    latent = torch.randn(1, 8, 1, 2, 2).bfloat16()
    text = torch.randn(1, 8, 128).bfloat16()
    timestep = torch.ones(1, 1)
    with torch.no_grad():
        y_bf16 = bf16_dit(latent, timestep, text)
        y_w8a8 = w8a8_dit(latent, timestep, text)
    rel_dit = (y_w8a8.float() - y_bf16.float()).norm() / y_bf16.float().norm()
    assert torch.isfinite(y_w8a8).all() and rel_dit < 0.15


def test_wan_model_linear_keep_bf16_and_dtype_steps(monkeypatch):
    from mstar.model.wan22 import weight_loader

    monkeypatch.setattr(weight_loader, "build_wan22_dit", lambda *args, **kwargs: _tiny_dit(num_layers=2))
    monkeypatch.setattr(Wan22Model, "_refresh_checkpoint_defaults", lambda self: None)
    reason = None
    monkeypatch.setattr(DenseFp8Format, "unavailable_reason", lambda self, device: reason)

    copies = []
    real_deepcopy = __import__("copy").deepcopy

    def spy(x, memo=None):
        if memo is None:
            copies.append(x)
        return real_deepcopy(x, memo)

    monkeypatch.setattr("mstar.model.wan22.wan22_model.copy.deepcopy", spy)

    def build_sub(compile_dit: bool = False, **kwargs):
        copies.clear()
        model = Wan22Model("unused", **kwargs)
        model.config = _tiny_config(num_layers=2, compile_dit=compile_dit)
        return model._create_submodule("dit", torch.device("cpu")), len(copies)

    # Keep block 0 in bf16 while quantizing block 1 and taking a bf16 twin for step 0.
    sub, n_copies = build_sub(
        linear_dtype="dense_fp8",
        linear_keep_bf16=["blocks.0.*.*"],
        linear_dtype_steps=1,
    )
    assert n_copies == 1
    assert sub.step_dtypes == StepDtypes(("bf16", "dense_fp8"))
    assert sub.split_batches_by_capture_key("video_gen")
    assert isinstance(sub.transformer.blocks[0].self_attn.q_proj, nn.Linear)
    assert isinstance(sub.transformer.blocks[1].self_attn.q_proj, QuantLinear)
    assert isinstance(sub.bf16_transformer.blocks[1].self_attn.q_proj, nn.Linear)

    # No twin when steps aren't scheduled, dtype is bf16, keep covers all candidates, or GPU unavailable.
    for kwargs in (
        {"linear_dtype": "dense_fp8"},
        {"linear_dtype": "bf16", "linear_dtype_steps": 1},
        {"linear_dtype": "dense_fp8", "linear_keep_bf16": ["blocks.*.*.*"], "linear_dtype_steps": 1},
    ):
        s, n = build_sub(**kwargs)
        assert n == 0 and s.bf16_transformer is None and s.step_dtypes is None
    reason = "no SM90"
    s, n = build_sub(linear_dtype="dense_fp8", linear_dtype_steps=1)
    assert n == 0 and s.bf16_transformer is None and s.step_dtypes is None
    reason = None

    with pytest.raises(ValueError, match="names none of its candidate linears"):
        build_sub(linear_dtype="dense_fp8", linear_keep_bf16=["nonexistent.*"])

    # StepBucket dispatch + _eager_forwards selection across both transformers when compile_dit=True.
    compiles = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: compiles.append(kw) or fn)
    monkeypatch.setattr(
        weight_loader, "build_wan22_dit", lambda *args, **kwargs: _tiny_dit(num_layers=2, compile_dit=True),
    )
    sub_c, _ = build_sub(
        compile_dit=True,
        linear_dtype="dense_fp8",
        linear_dtype_steps=1,
        compile_dit_sizes=("64x32x5",),
    )
    assert len(compiles) == 2

    seen = []
    bf16_eager = sub_c._eager_forwards[id(sub_c.bf16_transformer)][1]
    fp8_eager = sub_c._eager_forwards[id(sub_c.transformer)][1]
    sub_c._eager_forwards[id(sub_c.bf16_transformer)] = (
        sub_c.bf16_transformer,
        lambda **kw: seen.append("bf16") or bf16_eager(**kw),
    )
    sub_c._eager_forwards[id(sub_c.transformer)] = (
        sub_c.transformer,
        lambda **kw: seen.append("dense_fp8") or fp8_eager(**kw),
    )
    sub_c.get_device = lambda: torch.device("cpu")

    def run_step(k: int):
        info = SimpleNamespace(
            request_id="r", rid_handle="r", graph_walk="video_gen", random_seed=0,
            step_metadata={"num_inference_steps": 4, "guidance_scale": 1.0, "height": 32, "width": 32, "num_frames": 1},
            dynamic_loop_iter_counts={DENOISE_LOOP_NAME: k},
        )
        prepared = sub_c.prepare_inputs("video_gen", info, {
            "latents": [torch.randn(1, 8, 1, 2, 2)],
            "unipc_model_outputs": [torch.randn(2, 1, 8, 1, 2, 2)],
            "unipc_last_sample": [torch.randn(1, 8, 1, 2, 2)],
            "text_embeds_pos": [torch.randn(1, 8, 128)],
            "text_embeds_neg": [torch.randn(1, 8, 128)],
        })
        assert prepared.resource_step_info.dtype == ("bf16" if k == 0 else "dense_fp8")
        assert isinstance(prepared.resource_step_info, StepBucket)
        engine_inputs = SimpleNamespace(request_ids=["r"], per_request_info={"r": info})
        stacked = sub_c.preprocess("video_gen", engine_inputs, [prepared])
        return sub_c.forward("video_gen", engine_inputs=engine_inputs, **stacked)["latents"][0]

    run_step(0)
    run_step(1)
    assert seen == ["bf16", "dense_fp8"]
