"""Native FP8 text storage/loading and projection wiring; no pretrained payloads.

CPU arithmetic substitutes only for the GPU FP8 matmul in wiring tests. These
checks are not real-weight or GPU kernel parity, and add no serving fallback.
"""

import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

import mstar.model.components.scalar_fp8_linear as fp8_component
from mstar.model.components.scalar_fp8_linear import FP8_DTYPE, ScalarFp8Linear, _scaled_fp8_mm
from mstar.model.ministral3.components.language_model import Ministral3LMConfig
from mstar.model.ministral3.config import Ministral3TextConfig
from mstar.model.ministral3.fp8 import FP8_TEXT_RULES, build_fp8_text_model, load_fp8_text_weights

FIXTURES = Path(__file__).parent / "fixtures" / "ministral3"
QUANTIZATION = json.loads((FIXTURES / "config.json").read_text())["quantization_config"]
PREFIX = "language_model.model."
TINY = Ministral3LMConfig(hidden_size=96, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=32, intermediate_size=192, vocab_size=128)


def build(config=TINY, device="cpu"):
    return build_fp8_text_model(config, QUANTIZATION, device=device, attn_key="attn", kv_key="kv", pos_key="rope")


def checkpoint():
    generator = torch.Generator().manual_seed(51)
    weights = {
        PREFIX + "embed_tokens.weight": torch.randn(128, 96, generator=generator).to(torch.bfloat16),
        PREFIX + "norm.weight": torch.ones(96, dtype=torch.bfloat16),
    }
    for i in range(2):
        root = f"{PREFIX}layers.{i}."
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            weights[root + name] = torch.ones(96, dtype=torch.bfloat16)
        for index, (name, shape) in enumerate({
            "self_attn.q_proj": (128, 96), "self_attn.k_proj": (64, 96),
            "self_attn.v_proj": (64, 96), "self_attn.o_proj": (96, 128),
            "mlp.gate_proj": (192, 96), "mlp.up_proj": (192, 96), "mlp.down_proj": (96, 192),
        }.items()):
            weights[root + name + ".weight"] = torch.randn(*shape, generator=generator).to(FP8_DTYPE)
            # Distinct, non-power-of-two scalar scales expose accidental sharing.
            weights[root + name + ".weight_scale_inv"] = torch.tensor(0.017 + index * .003, dtype=torch.bfloat16)
            weights[root + name + ".activation_scale"] = torch.tensor(0.0137 + index * .012, dtype=torch.bfloat16)
    return weights


def test_real_header_matches_every_fp8_native_parameter_on_meta():
    config = Ministral3TextConfig.from_hf(FIXTURES, tokenizer_eos_token_id=2)
    model = build(config, device="meta")
    params = dict(model.named_parameters())
    header = json.loads((FIXTURES / "model.safetensors.header.json").read_text())
    mapped = set()
    for name, entry in header.items():
        if not name.startswith(PREFIX):
            continue
        target = FP8_TEXT_RULES.native(name)
        assert target not in mapped
        mapped.add(target)
        param = params[target]
        assert list(param.shape) == entry["shape"]
        assert param.is_meta
        expected_dtype = (torch.uint8 if entry["dtype"] == "F8_E4M3" else
                          torch.float32 if name.endswith(("weight_scale_inv", "activation_scale")) else torch.bfloat16)
        assert param.dtype == expected_dtype
    assert mapped == set(params)
    assert len(mapped) == 600
    assert sum(isinstance(m, ScalarFp8Linear) for m in model.modules()) == 182
    assert model.lm_head.weight is model.embed_tokens.weight


def test_strict_load_preserves_every_byte_scale_and_tied_embedding():
    model, weights = build(), checkpoint()
    weights["vision_tower.some_weight"] = torch.ones(1, dtype=torch.bfloat16)
    weights["multi_modal_projector.some_weight"] = torch.ones(1, dtype=torch.bfloat16)
    report = load_fp8_text_weights(model, iter(weights.items()), QUANTIZATION)
    assert report.loaded == frozenset(dict(model.named_parameters()))
    assert report.excluded_vision == ("multi_modal_projector.some_weight", "vision_tower.some_weight")
    model.to(dtype=torch.bfloat16)  # Same cast the current worker performs after loading.
    params = dict(model.named_parameters())
    for name, tensor in weights.items():
        if not name.startswith(PREFIX):
            continue
        param = params[FP8_TEXT_RULES.native(name)]
        expected = tensor.view(torch.uint8) if tensor.dtype == FP8_DTYPE else tensor.to(param.dtype)
        torch.testing.assert_close(param, expected, rtol=0, atol=0)
    assert model.lm_head.weight is model.embed_tokens.weight
    ids = torch.tensor([3, 19])
    torch.testing.assert_close(model.embed(ids), weights[PREFIX + "embed_tokens.weight"][ids], rtol=0, atol=0)


@pytest.mark.parametrize("suffix", ["weight", "weight_scale_inv", "activation_scale"])
def test_missing_projection_tensor_fails(suffix):
    weights = checkpoint()
    del weights[PREFIX + "layers.1.self_attn.k_proj." + suffix]
    with pytest.raises(ValueError, match="got no weight"):
        load_fp8_text_weights(build(), weights.items(), QUANTIZATION)


@pytest.mark.parametrize("bad", ["duplicate", "unknown", "dtype", "shape", "scale", "norm_dtype"])
def test_malformed_checkpoint_fails(bad):
    weights = checkpoint()
    q = PREFIX + "layers.0.self_attn.q_proj.weight"
    if bad == "unknown":
        weights["surprise.weight"] = torch.ones(1)
    elif bad == "dtype":
        weights[q] = weights[q].to(torch.bfloat16)
    elif bad == "shape":
        weights[q] = weights[q][:96]  # Query width is 128, not residual width 96.
    elif bad == "scale":
        weights[q.removesuffix("weight") + "activation_scale"] = torch.tensor(0., dtype=torch.bfloat16)
    elif bad == "norm_dtype":
        weights[PREFIX + "norm.weight"] = weights[PREFIX + "norm.weight"].float()
    stream = list(weights.items())
    if bad == "duplicate":
        stream.append(stream[0])
    with pytest.raises(ValueError):
        load_fp8_text_weights(build(), stream, QUANTIZATION)


@pytest.mark.parametrize("update", [
    {"activation_scheme": "dynamic"}, {"weight_block_size": [128, 128]},
    {"quant_method": "int8"}, {"dequantize": True},
])
def test_incompatible_metadata_fails_before_build(update):
    with pytest.raises(ValueError, match="static per-tensor FP8"):
        build_fp8_text_model(TINY, QUANTIZATION | update, device="meta",
                             attn_key="attn", kv_key="kv", pos_key="rope")


def oracle_linear(x, weights, stem):
    scale = weights[stem + ".activation_scale"].double()
    a = (x.double() / scale).clamp(-448, 448).to(FP8_DTYPE).double() * scale
    w = weights[stem + ".weight"].double() * weights[stem + ".weight_scale_inv"].double()
    return (a @ w.T).to(torch.bfloat16)


@torch.no_grad()
def test_attention_and_swiglu_projection_wiring_uses_each_checkpoint_scale(monkeypatch):
    # Exercise the production _scaled_fp8_mm operand/scale routing on CPU;
    # substitute only the device op and its matmul, not the projections.
    monkeypatch.setattr(fp8_component, "scalar_fp8_linear", _scaled_fp8_mm)

    def cpu_mm(a, b, *, scale_a, scale_b, out_dtype, use_fast_accum):
        assert a.dtype == b.dtype == FP8_DTYPE and not use_fast_accum
        return ((a.double() * scale_a.double()) @ (b.double() * scale_b.double())).to(out_dtype)

    monkeypatch.setattr(torch, "_scaled_mm", cpu_mm)
    model, weights = build(), checkpoint()
    load_fp8_text_weights(model, weights.items(), QUANTIZATION)
    x = torch.linspace(-1, 1, 3 * 96).reshape(3, 96).to(torch.bfloat16)
    for i, layer in enumerate(model.layers):
        root = f"{PREFIX}layers.{i}."
        q, k, v = layer.self_attn._project_qkv(x)
        for actual, projection, heads in ((q, "q", 4), (k, "k", 2), (v, "v", 2)):
            expected = oracle_linear(x, weights, root + f"self_attn.{projection}_proj").reshape(3, heads, 32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(layer.self_attn.o_proj(q.flatten(1)),
                                   oracle_linear(q.flatten(1), weights, root + "self_attn.o_proj"), rtol=0, atol=0)
        gate = oracle_linear(x, weights, root + "mlp.gate_proj")
        up = oracle_linear(x, weights, root + "mlp.up_proj")
        expected = oracle_linear(F.silu(gate) * up, weights, root + "mlp.down_proj")
        torch.testing.assert_close(layer.mlp(x), expected, rtol=0, atol=0)
        # Negative control: sharing Q's activation scale with K changes the result.
        stem = root + "self_attn.k_proj"
        wrong = weights | {stem + ".activation_scale": weights[root + "self_attn.q_proj.activation_scale"]}
        assert not torch.equal(k.flatten(1), oracle_linear(x, wrong, stem))


def test_native_fp8_decoder_does_not_execute_a_cpu_bf16_fallback():
    model = build()
    load_fp8_text_weights(model, checkpoint().items(), QUANTIZATION)
    with pytest.raises(RuntimeError, match="requires NVIDIA"):
        model.layers[0].self_attn._project_qkv(model.embed(torch.tensor([3])))
