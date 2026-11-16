"""Action projection and suffix visibility checks against LingBot V2 at be969b8.

The reference uses Qwen2Attention's separate Q/K/V linears and an output
linear. The mask rows below come from its pinned ``make_att_2d_masks`` and
``predict_velocity`` with three prefix and three suffix tokens.
"""
import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from mstar.model.components.lingbot_vla2_attention import (
    SUFFIX_READS,
    LingBotVLAActionAttentionConfig,
    LingBotVLAActionProjections,
    load_action_attention_weights,
)

PREFIX = "model.qwenvl_with_expert.qwen_expert.model.layers.0.self_attn."
CONFIG = LingBotVLAActionAttentionConfig(hidden_size=8, num_heads=4, num_kv_heads=2, head_dim=4)


def _checkpoint(tmp_path):
    weights = {}
    for i, (name, shape) in enumerate((
        ("q_proj.weight", (16, 8)), ("q_proj.bias", (16,)),
        ("k_proj.weight", (8, 8)), ("k_proj.bias", (8,)),
        ("v_proj.weight", (8, 8)), ("v_proj.bias", (8,)),
        ("o_proj.weight", (8, 16)),
    )):
        weights[PREFIX + name] = (
            torch.linspace(-0.18, 0.23, torch.Size(shape).numel()).reshape(shape) + i * 0.013
        ).contiguous()
    weights["model.qwenvl_with_expert.qwen_expert.model.layers.1.self_attn.q_proj.weight"] = (
        torch.zeros(16, 8)
    )
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in weights},
    }))
    return weights


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_action_qkv_and_output_match_reference_linears(tmp_path, dtype):
    weights = _checkpoint(tmp_path)
    projection = LingBotVLAActionProjections(CONFIG)
    loaded = load_action_attention_weights(projection, tmp_path, layer=0)
    assert loaded == set(dict(projection.named_parameters()))
    projection = projection.to(dtype)
    x = torch.tensor([
        [[0.2, -0.1, 0.3, 0.4, -0.7, 0.8, 0.5, -0.2],
         [0.9, -0.3, 0.6, -0.4, 0.1, 0.7, -0.8, 0.2]],
        [[0.4, 0.3, -0.2, 0.5, -0.6, 0.1, 0.2, -0.7],
         [-0.5, 0.8, 0.1, 0.6, -0.3, 0.4, -0.9, 0.7]],
    ], dtype=dtype)
    q, k, v = projection.project_qkv(x)
    for name, actual, heads in (("q", q, 4), ("k", k, 2), ("v", v, 2)):
        expected = F.linear(
            x, weights[PREFIX + f"{name}_proj.weight"].to(dtype),
            weights[PREFIX + f"{name}_proj.bias"].to(dtype),
        ).reshape(2, 2, heads, 4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=3e-7 if dtype == torch.float32 else 0)
    if dtype == torch.bfloat16:
        mixed = projection.project_qkv(x.float())
        for actual, expected in zip(mixed, (q, k, v), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_out = F.linear(
        q.reshape(2, 2, 16), weights[PREFIX + "o_proj.weight"].to(dtype),
    )
    torch.testing.assert_close(
        projection.project_output(q), expected_out, rtol=0,
        atol=3e-7 if dtype == torch.float32 else 0,
    )
    if dtype == torch.bfloat16:
        # The reference casts FP32 attended rows back to o_proj's weight dtype.
        torch.testing.assert_close(projection.project_output(q.float()), expected_out, rtol=0, atol=0)


def test_action_attention_checks_index_and_weight_shapes(tmp_path):
    _checkpoint(tmp_path)
    index = tmp_path / "model.safetensors.index.json"
    original = json.loads(index.read_text())
    for key, pattern in (
        (PREFIX + "k_proj.bias", "1 missing"),
        (PREFIX + "unexpected.weight", "1 extra"),
    ):
        changed = json.loads(json.dumps(original))
        if "unexpected" in key:
            changed["weight_map"][key] = "missing.safetensors"
        else:
            del changed["weight_map"][key]
        index.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=pattern):
            load_action_attention_weights(LingBotVLAActionProjections(CONFIG), tmp_path, 0)
    index.write_text(json.dumps(original))
    wrong_width = LingBotVLAActionAttentionConfig(hidden_size=16, num_heads=4, num_kv_heads=2, head_dim=4)
    with pytest.raises(ValueError, match="has shape"):
        load_action_attention_weights(LingBotVLAActionProjections(wrong_width), tmp_path, 0)
    wrong_heads = LingBotVLAActionAttentionConfig(hidden_size=8, num_heads=2, num_kv_heads=2, head_dim=4)
    with pytest.raises(ValueError, match="has shape"):
        load_action_attention_weights(LingBotVLAActionProjections(wrong_heads), tmp_path, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        load_action_attention_weights(LingBotVLAActionProjections(CONFIG), tmp_path, -1)
    projection = LingBotVLAActionProjections(CONFIG)
    with pytest.raises(ValueError, match="input must"):
        projection.project_qkv(torch.zeros(2, 3, 7))
    with pytest.raises(ValueError, match="attended rows must"):
        projection.project_output(torch.zeros(2, 3, 8))


def test_declared_suffix_reads_match_pinned_reference_mask_rows():
    # Reference prefix padding [T, F, T] and suffix markers [T, T, F].
    # Its suffix queries are [state, action, action]. The prefix was already
    # cached, so each valid prefix key is visible even to the state query.
    regions = ("prefix", "prefix", "prefix", "state", "actions", "actions")
    expected = [
        [[1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
    ]
    actual = []
    for prefix_valid in ((True, False, True), (True, True, True)):
        key_valid = (*prefix_valid, True, True, True)
        rows = []
        for reads in (SUFFIX_READS.state, SUFFIX_READS.actions, SUFFIX_READS.actions):
            rows.append([int(valid and region in reads) for valid, region in zip(key_valid, regions, strict=True)])
        actual.append(rows)
    assert actual == expected
    assert "actions" not in SUFFIX_READS.state
    assert "actions" in SUFFIX_READS.actions
