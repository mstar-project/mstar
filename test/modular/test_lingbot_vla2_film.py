"""Tiny FiLM parity with Robbyant/lingbot-vla-v2 at be969b8.

The expected rows come from the pinned reference's AdaRMSNorm class, extracted
without its other dependencies and loaded with the same explicit parameters.
"""
import json

import pytest
import torch
from safetensors.torch import save_file

from mstar.model.components.lingbot_vla2_film import LingBotVLAAdaRMSNorm, load_action_film_weights

PREFIX = "model.qwenvl_with_expert.qwen_expert.model.layers.0.input_layernorm."


def _checkpoint(tmp_path, kind="input_layernorm"):
    norm = LingBotVLAAdaRMSNorm(4, 4)
    prefix = PREFIX.replace("input_layernorm.", f"{kind}.")
    weights = {
        prefix + name: (
            torch.linspace(-0.2, 0.3, param.numel()).reshape_as(param) + i * 0.017
        ).contiguous()
        for i, (name, param) in enumerate(norm.named_parameters())
    }
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in weights},
    }))
    return weights


def _inputs(dtype):
    x = torch.tensor([
        [[0.2, -0.1, 0.3, 0.4], [-0.7, 0.8, 0.5, -0.2]],
        [[0.9, -0.3, 0.6, -0.4], [0.1, 0.7, -0.8, 0.2]],
    ], dtype=dtype)
    cond = torch.tensor([[0.25, -0.5, 0.75, 0.1], [-0.3, 0.8, 0.2, -0.6]], dtype=dtype)
    return x, cond


@pytest.mark.parametrize("dtype,expected", [
    (torch.float32, [
        [[-0.2901713252, 0.0809443817, 0.4994268417, 1.2224969864],
         [0.00228099525, 0.0232082009, 0.4563459456, 0.4101222754]],
        [[-0.4072848260, 0.0294757634, 0.3482745290, 0.1040857136],
         [-0.1962000132, -0.0288161431, -0.0338666886, 0.5206531882]],
    ]),
    (torch.bfloat16, [
        [[-0.2890625, 0.0810546875, 0.5, 1.2265625],
         [0.00257873535, 0.0233154297, 0.455078125, 0.41015625]],
        [[-0.408203125, 0.0294189453, 0.349609375, 0.10107421875],
         [-0.1962890625, -0.0290527344, -0.034423828125, 0.51953125]],
    ]),
])
def test_film_matches_pinned_reference(tmp_path, dtype, expected):
    weights = _checkpoint(tmp_path)
    norm = LingBotVLAAdaRMSNorm(4, 4)
    loaded = load_action_film_weights(norm, tmp_path, layer=0, kind="input_layernorm")
    assert loaded == set(dict(norm.named_parameters()))
    for name, parameter in norm.named_parameters():
        torch.testing.assert_close(parameter, weights[PREFIX + name], rtol=0, atol=0)
    norm = norm.to(dtype)
    x, cond = _inputs(dtype)
    actual = norm(x, cond)
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=dtype), rtol=0,
        atol=0 if dtype == torch.bfloat16 else 1e-6,
    )


def test_film_checks_index_and_shapes_before_loading(tmp_path):
    _checkpoint(tmp_path)
    index = tmp_path / "model.safetensors.index.json"
    original = json.loads(index.read_text())
    for key, pattern in (
        (PREFIX + "gamma.weight", "1 missing"),
        (PREFIX + "unknown.weight", "1 extra"),
    ):
        changed = json.loads(json.dumps(original))
        if "unknown" in key:
            changed["weight_map"][key] = "missing.safetensors"
        else:
            del changed["weight_map"][key]
        index.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=pattern):
            load_action_film_weights(LingBotVLAAdaRMSNorm(4, 4), tmp_path, 0, "input_layernorm")
    index.write_text(json.dumps(original))
    with pytest.raises(AssertionError, match="shape mismatch"):
        load_action_film_weights(LingBotVLAAdaRMSNorm(8, 8), tmp_path, 0, "input_layernorm")
    with pytest.raises(ValueError, match="supported norm kind"):
        load_action_film_weights(LingBotVLAAdaRMSNorm(4, 4), tmp_path, 0, "q_layernorm")
    index.unlink()
    with pytest.raises(FileNotFoundError, match="LingBot-VLA FiLM requires"):
        load_action_film_weights(LingBotVLAAdaRMSNorm(4, 4), tmp_path, 0, "input_layernorm")


def test_film_loads_post_attention_norm(tmp_path):
    weights = _checkpoint(tmp_path, kind="post_attention_layernorm")
    norm = LingBotVLAAdaRMSNorm(4, 4)
    loaded = load_action_film_weights(norm, tmp_path, 0, "post_attention_layernorm")
    assert loaded == set(dict(norm.named_parameters()))
    for name, parameter in norm.named_parameters():
        torch.testing.assert_close(
            parameter, weights[PREFIX.replace("input_layernorm.", "post_attention_layernorm.") + name],
            rtol=0, atol=0,
        )


def test_film_rejects_broadcasting_and_casts_condition_to_parameter_dtype():
    norm = LingBotVLAAdaRMSNorm(6, 4)
    x = torch.randn(2, 3, 6)
    cond = torch.randn(2, 4)
    torch.testing.assert_close(norm(x, cond.to(torch.bfloat16)), norm(x, cond.to(torch.bfloat16).float()))
    with pytest.raises(ValueError, match="input must"):
        norm(x[:, 0], cond)
    with pytest.raises(ValueError, match="condition must"):
        norm(x, cond[:, None].expand(-1, 3, -1))
    with pytest.raises(ValueError, match="condition must"):
        norm(x, cond[:1])
