"""Wan2.2 dense FP8 selection and CPU reference checks (no checkpoint or GPU)."""

import pytest
import torch
from torch import nn

from mstar.model.components.dense_fp8_format import DENSE_FP8, DenseFp8Format
from mstar.model.components.quant_linear import QuantLinear, quant_capability
from mstar.model.wan22.components.dit import QUANTIZABLE_LINEARS, Wan22DiT
from mstar.model.wan22.config import Wan22Config
from mstar.model.wan22.wan22_model import Wan22Model


def _tiny_config() -> Wan22Config:
    return Wan22Config(
        num_attention_heads=2, attention_head_dim=64, num_layers=1, ffn_dim=256,
        in_channels=8, out_channels=8, text_dim=128, freq_dim=128,
        rope_max_seq_len=16, patch_size=(1, 2, 2), compile_dit=False,
    )


def _tiny_dit() -> Wan22DiT:
    torch.manual_seed(5)
    dit = Wan22DiT(_tiny_config()).cast_serving_dtypes().eval()
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
    assert Wan22Model("unused", skip_weight_loading=True, linear_dtype="dense_fp8").linear_dtype == "dense_fp8"
    for invalid in ("mxfp8", "nvfp4", "fp8", None):
        with pytest.raises(ValueError, match="linear_dtype must be"):
            Wan22Model("unused", skip_weight_loading=True, linear_dtype=invalid)


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
