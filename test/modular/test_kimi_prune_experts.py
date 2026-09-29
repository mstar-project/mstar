"""CPU test for the expert pruner against a small fake checkpoint that mimics
the real Kimi-K2.7 key layout, split across two shards so each MoE layer's
gate sits in a different shard from its experts."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "integration"))

import torch
from kimi_prune_experts import main
from safetensors import safe_open
from safetensors.torch import save_file

PROJS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight_packed", "weight_scale", "weight_shape")
LAYER_PREFIX = "language_model.model.layers.{}"
EXPERT_INDEX_RE = re.compile(r"\.mlp\.experts\.(\d+)\.")


def _expert_tensor(layer, expert, proj, part):
    base = layer * 1000 + expert * 10 + PROJS.index(proj)
    if part == "weight_packed":
        return torch.full((2, 2), base, dtype=torch.int32)
    if part == "weight_scale":
        return torch.full((2,), float(base), dtype=torch.bfloat16)
    return torch.tensor([base, base], dtype=torch.int64)


def _expert_items(layer):
    return {
        f"{LAYER_PREFIX.format(layer)}.mlp.experts.{expert}.{proj}.{part}": _expert_tensor(layer, expert, proj, part)
        for expert in range(4)
        for proj in PROJS
        for part in PARTS
    }


def _build_checkpoint(src):
    src.mkdir()
    (src / "tiktoken.model").write_bytes(b"fake-tokenizer-bytes")
    config = {
        "architectures": ["KimiK25ForConditionalGeneration"],
        "text_config": {"n_routed_experts": 4, "hidden_size": 8},
    }
    (src / "config.json").write_text(json.dumps(config))

    gate1 = torch.tensor(
        [
            [3, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    gate2 = gate1 * 10
    bias1 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    bias2 = torch.tensor([1.1, 1.2, 1.3, 1.4])

    # Layer 1's gate/bias live with layer 2's experts, and vice versa, so a
    # layer's router and its experts never share a shard.
    shard_a = {
        "language_model.model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
        f"{LAYER_PREFIX.format(0)}.mlp.down_proj.weight": torch.ones(4, 4),
        f"{LAYER_PREFIX.format(0)}.mlp.gate_proj.weight": torch.ones(4, 4) * 2,
        f"{LAYER_PREFIX.format(0)}.mlp.up_proj.weight": torch.ones(4, 4) * 3,
        "vision_tower.x.weight": torch.zeros(2, 3),
        f"{LAYER_PREFIX.format(1)}.mlp.gate.weight": gate1,
        f"{LAYER_PREFIX.format(1)}.mlp.gate.e_score_correction_bias": bias1,
        **_expert_items(2),
    }
    shard_b = {
        f"{LAYER_PREFIX.format(2)}.mlp.gate.weight": gate2,
        f"{LAYER_PREFIX.format(2)}.mlp.gate.e_score_correction_bias": bias2,
        **_expert_items(1),
    }

    shard_a_name = "model-00001-of-00002.safetensors"
    shard_b_name = "model-00002-of-00002.safetensors"
    save_file(shard_a, str(src / shard_a_name))
    save_file(shard_b, str(src / shard_b_name))

    weight_map = dict.fromkeys(shard_a, shard_a_name)
    weight_map.update(dict.fromkeys(shard_b, shard_b_name))
    total_size = sum(t.numel() * t.element_size() for t in (*shard_a.values(), *shard_b.values()))
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (src / "model.safetensors.index.json").write_text(json.dumps(index))

    return {**shard_a, **shard_b}, gate1, gate2, bias1, bias2


def _read_output_tensors(out, weight_map):
    tensors = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(out / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                assert weight_map[key] == shard
                tensors[key] = f.get_tensor(key)
    return tensors


def test_prune_experts_on_cpu(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    originals, gate1, gate2, bias1, bias2 = _build_checkpoint(src)

    main(["--src", str(src), "--out", str(out), "--keep", "2"])

    index = json.loads((out / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    tensors = _read_output_tensors(out, weight_map)
    assert set(tensors) == set(weight_map)

    keep_idx = [2, 0]
    for layer in (1, 2):
        experts_present = {
            int(EXPERT_INDEX_RE.search(k).group(1)) for k in tensors if f"layers.{layer}.mlp.experts." in k
        }
        assert experts_present == {0, 1}
        for new_idx, orig_idx in enumerate(keep_idx):
            for proj in PROJS:
                for part in PARTS:
                    new_key = f"{LAYER_PREFIX.format(layer)}.mlp.experts.{new_idx}.{proj}.{part}"
                    assert torch.equal(tensors[new_key], _expert_tensor(layer, orig_idx, proj, part))

    assert torch.equal(tensors[f"{LAYER_PREFIX.format(1)}.mlp.gate.weight"], gate1[keep_idx])
    assert torch.equal(tensors[f"{LAYER_PREFIX.format(1)}.mlp.gate.e_score_correction_bias"], bias1[keep_idx])
    assert torch.equal(tensors[f"{LAYER_PREFIX.format(2)}.mlp.gate.weight"], gate2[keep_idx])
    assert torch.equal(tensors[f"{LAYER_PREFIX.format(2)}.mlp.gate.e_score_correction_bias"], bias2[keep_idx])

    for key in (
        "language_model.model.embed_tokens.weight",
        "vision_tower.x.weight",
        f"{LAYER_PREFIX.format(0)}.mlp.down_proj.weight",
        f"{LAYER_PREFIX.format(0)}.mlp.gate_proj.weight",
        f"{LAYER_PREFIX.format(0)}.mlp.up_proj.weight",
    ):
        assert torch.equal(tensors[key], originals[key])

    assert (out / "tiktoken.model").read_bytes() == b"fake-tokenizer-bytes"

    out_config = json.loads((out / "config.json").read_text())
    assert out_config["text_config"]["n_routed_experts"] == 2

    expected_total = sum(t.numel() * t.element_size() for t in tensors.values())
    assert index["metadata"]["total_size"] == expected_total

    manifest = json.loads((out / "pruning_manifest.json").read_text())
    assert manifest["layers"]["1"] == keep_idx
    assert manifest["layers"]["2"] == keep_idx
