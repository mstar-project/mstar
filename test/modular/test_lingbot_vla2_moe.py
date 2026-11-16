"""Tiny action-MoE parity with Robbyant/lingbot-vla-v2 at be969b8.

The golden routing and output values were generated from the pinned eager
``Qwen2TokenMoeBlock`` with the explicit weights and input below. The released
RoboTwin config uses the same sigmoid, top-4, correction-bias and shared-expert
path at larger dimensions; this test uses top-2 of three experts on CPU.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from mstar.model.components.lingbot_vla2_moe import (
    LingBotVLAActionMoe,
    LingBotVLAActionMoeConfig,
    load_action_moe_weights,
)

PREFIX = "model.qwenvl_with_expert.qwen_expert.model.layers.0.mlp."


def _weights(tmp_path):
    config = LingBotVLAActionMoeConfig(
        hidden_size=8, num_experts=3, top_k=2,
        routed_intermediate_size=5, shared_intermediate_size=6,
        routed_scaling_factor=4.0,
    )

    def tensor(shape, index):
        return (torch.linspace(-0.18, 0.23, torch.Size(shape).numel()).reshape(shape) + index * 0.013).contiguous()

    weights = {
        PREFIX + "gate.weight": tensor((3, 8), 0),
        PREFIX + "e_score_correction_bias": torch.tensor([0.08, -0.03, 0.02]),
        PREFIX + "experts.gate_proj": torch.stack([tensor((5, 8), i) for i in (1, 4, 7)]),
        PREFIX + "experts.up_proj": torch.stack([tensor((5, 8), i) for i in (2, 5, 8)]),
        PREFIX + "experts.down_proj": torch.stack([tensor((8, 5), i) for i in (3, 6, 9)]),
        PREFIX + "shared_expert.gate_proj.weight": tensor((6, 8), 10),
        PREFIX + "shared_expert.up_proj.weight": tensor((6, 8), 11),
        PREFIX + "shared_expert.down_proj.weight": tensor((8, 6), 12),
        "model.qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8),
    }
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in weights},
    }))
    return config, weights


def _input():
    return torch.tensor([
        [[0.2, -0.1, 0.3, 0.4, -0.7, 0.8, 0.5, -0.2],
         [0.9, -0.3, 0.6, -0.4, 0.1, 0.7, -0.8, 0.2]],
        [[0.4, 0.3, -0.2, 0.5, -0.6, 0.1, 0.2, -0.7],
         [-0.5, 0.8, 0.1, 0.6, -0.3, 0.4, -0.9, 0.7]],
    ])


def test_action_moe_matches_pinned_reference_router_and_outputs(tmp_path):
    config, weights = _weights(tmp_path)
    moe = LingBotVLAActionMoe(config).eval()
    loaded = load_action_moe_weights(moe, tmp_path, layer=0)
    assert loaded == {name.removeprefix(PREFIX) for name in weights if name.startswith(PREFIX)}

    x = _input()
    with torch.no_grad():
        routing, expert_ids, _ = moe.gate(x.reshape(-1, 8))
        routed = moe.experts(x.reshape(-1, 8), routing, expert_ids)
        shared = moe.shared_expert(x.reshape(-1, 8))
        actual = moe(x)
    assert expert_ids.tolist() == [[2, 0], [2, 0], [0, 2], [2, 0]]
    torch.testing.assert_close(routing, torch.tensor([
        [2.167718887329, 1.832281231880],
        [2.144762754440, 1.855237364769],
        [2.0, 2.0],
        [2.126792669296, 1.873207092285],
    ]), rtol=0, atol=1e-6)

    expected_shared = torch.tensor([
        [0.002528580371, 0.013277471066, 0.024026360363, 0.034775249660,
         0.045524135232, 0.056273020804, 0.067021913826, 0.077770799398],
        [0.001379726222, 0.006976349279, 0.012572972104, 0.018169596791,
         0.023766215891, 0.029362838715, 0.034959457815, 0.040556084365],
        [-0.000008663673, 0.000198255409, 0.000405174476, 0.000612093601,
         0.000819012523, 0.001025931677, 0.001232850715, 0.001439769752],
        [0.001343686716, 0.007055259775, 0.012766834348, 0.018478406593,
         0.024189976975, 0.029901549220, 0.035613123327, 0.041324693710],
    ])
    expected_final = torch.tensor([
        [[-0.017129780725, 0.013626929373, 0.044383637607, 0.075140357018,
          0.105897039175, 0.136653751135, 0.167410463095, 0.198167160153],
         [-0.009856896475, 0.005986819975, 0.021830536425, 0.037674255669,
          0.053517963737, 0.069361679256, 0.085205391049, 0.101049110293]],
        [[-0.001552463276, -0.000343394408, 0.000865674403, 0.002074743155,
          0.003283811733, 0.004492880311, 0.005701949354, 0.006911017932],
         [-0.009353061207, 0.006966724060, 0.023286506534, 0.039606295526,
          0.055926073343, 0.072245858610, 0.088565647602, 0.104885429144]],
    ])
    torch.testing.assert_close(shared, expected_shared, rtol=0, atol=1e-6)
    torch.testing.assert_close(routed, expected_final.reshape(-1, 8) - expected_shared, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual, expected_final, rtol=0, atol=1e-6)


def test_action_moe_checks_index_before_reading(tmp_path):
    config, _ = _weights(tmp_path)
    index = tmp_path / "model.safetensors.index.json"
    original = json.loads(index.read_text())
    for key, pattern in (
        (PREFIX + "gate.weight", "1 missing"),
        (PREFIX + "unknown.weight", "1 extra"),
    ):
        changed = json.loads(json.dumps(original))
        if "unknown" in key:
            changed["weight_map"][key] = "missing.safetensors"
        else:
            del changed["weight_map"][key]
        index.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=pattern):
            load_action_moe_weights(LingBotVLAActionMoe(config), tmp_path, layer=0)
    index.unlink()
    with pytest.raises(FileNotFoundError, match="LingBot-VLA MoE requires"):
        load_action_moe_weights(LingBotVLAActionMoe(config), tmp_path, layer=0)


def test_action_moe_bf16_stays_within_one_ulp_of_pinned_reference(tmp_path):
    config, _ = _weights(tmp_path)
    moe = LingBotVLAActionMoe(config)
    load_action_moe_weights(moe, tmp_path, layer=0)
    moe = moe.to(torch.bfloat16).eval()
    with torch.no_grad():
        actual = moe(_input().to(torch.bfloat16)).float()
    expected = torch.tensor([
        [[-0.01708984375, 0.013671875, 0.04443359375, 0.0751953125,
          0.1064453125, 0.13671875, 0.16796875, 0.19921875],
         [-0.009765625, 0.005889892578125, 0.021728515625, 0.037353515625,
          0.05322265625, 0.06884765625, 0.08447265625, 0.10009765625]],
        [[-0.001556396484375, -0.000347137451172, 0.000862121582031, 0.002059936523438,
          0.003280639648438, 0.004486083984375, 0.00567626953125, 0.00689697265625],
         [-0.00946044921875, 0.006927490234375, 0.0233154296875, 0.03955078125,
          0.05615234375, 0.072265625, 0.0888671875, 0.10546875]],
    ])
    # The shared fused projections round a little differently from the
    # reference's separate BF16 linears on CPU; the max gap is 0.0009765625.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.001)


def test_action_moe_rejects_wrong_input_shape():
    moe = LingBotVLAActionMoe(LingBotVLAActionMoeConfig(8, 3, 2, 5, 6, 4.0))
    with pytest.raises(ValueError, match=r"\[batch, tokens, hidden_size\]"):
        moe(torch.zeros(4, 8))
