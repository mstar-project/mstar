"""CPU suffix-input checks against LingBot V2's pinned embed_suffix output.

The golden rows below were generated with the unmodified
``FlowMatching.embed_suffix`` method from Robbyant/lingbot-vla-v2 at
``be969b8fd117fb70550c5d4bf4bc328211b5b1b6``, using the explicit
projection weights and inputs in this test. The V2 flow model inherits it.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from mstar.model.components.lingbot_vla2_suffix import (
    LingBotVLASuffixConfig,
    LingBotVLASuffixInputs,
    load_suffix_input_weights,
)


def _checkpoint(tmp_path):
    config = LingBotVLASuffixConfig(max_state_dim=3, max_action_dim=2, expert_hidden_size=4)
    source = LingBotVLASuffixInputs(config)
    weights = {}
    for i, name in enumerate((
        "state_proj", "action_in_proj", "action_time_mlp_in", "action_time_mlp_out", "action_out_proj",
    )):
        layer = getattr(source, name)
        weights[f"model.{name}.weight"] = (
            torch.linspace(-0.2, 0.3, layer.weight.numel()).reshape_as(layer.weight) + i * 0.017
        ).contiguous()
        weights[f"model.{name}.bias"] = (
            torch.linspace(-0.12, 0.12, layer.bias.numel()) - i * 0.009
        ).contiguous()
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in weights},
    }))
    return config, weights


def test_suffix_inputs_match_pinned_reference_rows_and_masks(tmp_path):
    config, weights = _checkpoint(tmp_path)
    suffix = LingBotVLASuffixInputs(config)
    loaded = load_suffix_input_weights(suffix, tmp_path)
    assert loaded == set(dict(suffix.named_parameters()))
    for name, parameter in suffix.named_parameters():
        torch.testing.assert_close(parameter, weights[f"model.{name}"], rtol=0, atol=0)

    state = torch.tensor([[0.2, -0.1, 0.7], [-0.5, 0.4, 0.9]])
    actions = torch.tensor([
        [[0.1, -0.3], [0.5, 0.2], [-0.7, 0.4]],
        [[0.3, 0.8], [-0.2, -0.1], [0.6, -0.5]],
    ])
    timestep = torch.tensor([0.25, 0.75])
    time_emb, rows, pad_mask, attention_mask = suffix(state, actions, timestep)

    expected_time = torch.tensor([
        [0.000015536614, 0.382683455944, -1.0, 0.923879504204],
        [0.000077127421, 0.923879563808, -1.0, 0.382683426142],
    ])
    expected_rows = torch.tensor([
        [[-0.220909088850, -0.031818177551, 0.157272726297, 0.346363633871],
         [-0.144745513797, -0.056202679873, 0.032340150326, 0.120882973075],
         [-0.147461533546, -0.051116973162, 0.045227602124, 0.141572177410],
         [-0.145004719496, -0.053210537881, 0.038583651185, 0.130377829075]],
        [[-0.179999992251, 0.009090915322, 0.198181837797, 0.387272745371],
         [-0.145733997226, -0.048379015177, 0.048975959420, 0.146330952644],
         [-0.141262218356, -0.056941036135, 0.027380153537, 0.111701339483],
         [-0.141996517777, -0.057610794902, 0.026774937287, 0.111160665751]],
    ])
    torch.testing.assert_close(time_emb, expected_time, rtol=0, atol=1e-6)
    torch.testing.assert_close(rows, expected_rows, rtol=0, atol=1e-6)
    assert pad_mask.tolist() == [[True] * 4, [True] * 4]
    assert attention_mask.tolist() == [[True, True, False, False]] * 2


def test_suffix_refuses_incomplete_or_unknown_projection_index(tmp_path):
    config, _ = _checkpoint(tmp_path)
    index = tmp_path / "model.safetensors.index.json"
    original = json.loads(index.read_text())
    for key, pattern in (
        ("model.state_proj.weight", "1 missing"),
        ("model.action_out_proj.bias", "1 missing"),
        ("model.state_proj.unknown", "1 extra"),
    ):
        changed = json.loads(json.dumps(original))
        if key.endswith("unknown"):
            changed["weight_map"][key] = "missing.safetensors"
        else:
            del changed["weight_map"][key]
        index.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=pattern):
            load_suffix_input_weights(LingBotVLASuffixInputs(config), tmp_path)
    index.unlink()
    with pytest.raises(FileNotFoundError, match="LingBot-VLA suffix requires"):
        load_suffix_input_weights(LingBotVLASuffixInputs(config), tmp_path)


def test_suffix_fp32_projection_mode_matches_pinned_bf16_reference(tmp_path):
    config, _ = _checkpoint(tmp_path)
    config = LingBotVLASuffixConfig(
        config.max_state_dim, config.max_action_dim, config.expert_hidden_size, action_fp32=True,
    )
    suffix = LingBotVLASuffixInputs(config)
    load_suffix_input_weights(suffix, tmp_path)
    suffix = suffix.to(torch.bfloat16)
    state = torch.tensor([[0.2, -0.1, 0.7], [-0.5, 0.4, 0.9]], dtype=torch.bfloat16)
    actions = torch.tensor([
        [[0.1, -0.3], [0.5, 0.2], [-0.7, 0.4]],
        [[0.3, 0.8], [-0.2, -0.1], [0.6, -0.5]],
    ], dtype=torch.bfloat16)
    time, rows, pad, attention = suffix(state, actions, torch.tensor([0.25, 0.75]))
    assert time.dtype == torch.bfloat16
    assert rows.dtype == torch.float32
    torch.testing.assert_close(
        time[0], torch.tensor([0.000015497208, 0.3828125, -1.0, 0.92578125], dtype=torch.bfloat16),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        rows, torch.tensor([
            [[-0.220886230469, -0.031892240047, 0.157491683960, 0.346851348877],
             [-0.145265907049, -0.056072138250, 0.032452531159, 0.120852828026],
             [-0.147987082601, -0.050972640514, 0.045368172228, 0.141601145267],
             [-0.145532265306, -0.053070239723, 0.038721524179, 0.130392938852]],
            [[-0.179626464844, 0.008873701096, 0.198373794556, 0.387519836426],
             [-0.146218717098, -0.048259757459, 0.049026336521, 0.146200120449],
             [-0.141738474369, -0.056839127094, 0.027395296842, 0.111489534378],
             [-0.142476767302, -0.057499568909, 0.026812041178, 0.110986016691]],
        ]),
        rtol=0, atol=1e-7,
    )
    assert pad.all() and attention[:, :2].all() and not attention[:, 2:].any()


def test_suffix_requires_matching_input_shapes():
    suffix = LingBotVLASuffixInputs(LingBotVLASuffixConfig(3, 2, 4))
    state, actions, timestep = torch.zeros(2, 3), torch.zeros(2, 3, 2), torch.zeros(2)
    with pytest.raises(ValueError, match="state must"):
        suffix(state[:, :2], actions, timestep)
    with pytest.raises(ValueError, match="actions must"):
        suffix(state, actions[:, :, :1], timestep)
    with pytest.raises(ValueError, match="timestep must"):
        suffix(state, actions, timestep[:1])
    with pytest.raises(ValueError, match="same device"):
        suffix(state.to("meta"), actions, timestep)


def test_action_output_projection_matches_reference_dtype_rules(tmp_path):
    config, _ = _checkpoint(tmp_path)
    decoded = torch.tensor([
        [[0.2, -0.1, 0.3, 0.4], [-0.7, 0.8, 0.5, -0.2]],
        [[0.9, -0.3, 0.6, -0.4], [0.1, 0.7, -0.8, 0.2]],
    ])
    suffix = LingBotVLASuffixInputs(config)
    load_suffix_input_weights(suffix, tmp_path)
    # FlowMatchingV2.sample_actions takes the last action rows before this
    # projection and calls action_out_proj on them.
    # Golden from the pinned reference's action_out_proj with the fixture weights.
    expected = torch.tensor([
        [[-0.1401714087, 0.3284000158], [-0.1230856925, 0.2312000096]],
        [[-0.2830285430, 0.1855428517], [-0.2038285434, 0.0933142826]],
    ])
    torch.testing.assert_close(suffix.project_actions(decoded), expected, rtol=0, atol=1e-7)

    default_bf16 = suffix.to(torch.bfloat16)
    actual_bf16 = default_bf16.project_actions(decoded)
    expected_bf16 = torch.nn.functional.linear(
        decoded.to(torch.bfloat16), default_bf16.action_out_proj.weight,
        default_bf16.action_out_proj.bias,
    )
    assert actual_bf16.dtype == torch.bfloat16
    torch.testing.assert_close(actual_bf16, expected_bf16, rtol=0, atol=0)

    suffix = LingBotVLASuffixInputs(LingBotVLASuffixConfig(3, 2, 4, action_fp32=True))
    load_suffix_input_weights(suffix, tmp_path)
    suffix = suffix.to(torch.bfloat16)
    actual = suffix.project_actions(decoded.to(torch.bfloat16))
    expected = torch.nn.functional.linear(
        decoded.to(torch.bfloat16).float(), suffix.action_out_proj.weight.float(),
        suffix.action_out_proj.bias.float(),
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="decoded actions must"):
        suffix.project_actions(decoded[..., :3])
