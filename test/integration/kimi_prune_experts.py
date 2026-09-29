"""Prune routed experts from a Kimi-K2.7 checkpoint, keeping every other
tensor byte-identical. Mirrors the mgoin/Kimi-K3-pruned75 recipe: rank each
MoE layer's experts by descending signed sum(router.weight[i]), keep the
prefix of that ranking, and compact the expert and router indices.

Streams shards through safetensors so no shard is ever fully materialized in
memory: only one tensor at a time is read off the source mmap.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"

EXPERT_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)\.(?P<part>weight_packed|weight_scale|weight_shape)$"
)
GATE_RE = re.compile(r"^language_model\.model\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$")
BIAS_RE = re.compile(r"^language_model\.model\.layers\.(?P<layer>\d+)\.mlp\.gate\.e_score_correction_bias$")


def gate_key(layer):
    return f"language_model.model.layers.{layer}.mlp.gate.weight"


def find_moe_layers(weight_map):
    layers = {int(m.group("layer")) for k in weight_map if (m := EXPERT_RE.match(k))}
    return sorted(layers)


def rank_experts(src, weight_map, layers, keep):
    """Per MoE layer: the kept original expert indices, in descending
    router-score order, and the layer's original expert count."""
    keep_idx = {}
    num_experts = {}
    for layer in layers:
        key = gate_key(layer)
        if key not in weight_map:
            raise ValueError(f"layer {layer} has expert tensors but no {key}")
        with safe_open(str(src / weight_map[key]), framework="pt", device="cpu") as f:
            weight = f.get_tensor(key)
        score = weight.float().sum(dim=1)
        order = torch.argsort(score, descending=True, stable=True)
        keep_idx[layer] = order[:keep].tolist()
        num_experts[layer] = weight.shape[0]
    return keep_idx, num_experts


def _prune_tensor(f, key, keep_idx, old_to_new, num_experts):
    m = EXPERT_RE.match(key)
    if m:
        layer, expert = int(m.group("layer")), int(m.group("expert"))
        if expert >= num_experts[layer]:
            raise ValueError(f"expert {expert} exceeds {num_experts[layer]} router rows for layer {layer}")
        new_expert = old_to_new[layer].get(expert)
        if new_expert is None:
            return None, None
        new_key = f"language_model.model.layers.{layer}.mlp.experts.{new_expert}.{m.group('proj')}.{m.group('part')}"
        return new_key, f.get_tensor(key)

    m = GATE_RE.match(key) or BIAS_RE.match(key)
    if m:
        order = torch.tensor(keep_idx[int(m.group("layer"))], dtype=torch.long)
        return key, f.get_tensor(key)[order]

    return key, f.get_tensor(key)


def prune_shards(src, out, weight_map, keep_idx, num_experts, shard_size_bytes):
    old_to_new = {layer: {old: new for new, old in enumerate(order)} for layer, order in keep_idx.items()}
    shard_to_keys = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    key_to_shard = {}
    total_size = 0
    buffer = {}
    buffer_bytes = 0
    shard_idx = 0

    def flush():
        nonlocal buffer, buffer_bytes, shard_idx
        if not buffer:
            return
        shard_idx += 1
        name = f"model-{shard_idx:05d}.safetensors"
        save_file(buffer, str(out / name), metadata={"format": "pt"})
        key_to_shard.update(dict.fromkeys(buffer, name))
        buffer = {}
        buffer_bytes = 0

    for shard_name in sorted(shard_to_keys):
        with safe_open(str(src / shard_name), framework="pt", device="cpu") as f:
            for key in sorted(shard_to_keys[shard_name]):
                new_key, tensor = _prune_tensor(f, key, keep_idx, old_to_new, num_experts)
                if new_key is None:
                    continue
                buffer[new_key] = tensor
                nbytes = tensor.numel() * tensor.element_size()
                total_size += nbytes
                buffer_bytes += nbytes
                if buffer_bytes >= shard_size_bytes:
                    flush()
    flush()
    return key_to_shard, total_size, shard_idx


def finalize_shard_names(out, key_to_shard, n_shards):
    renamed = {}
    for i in range(1, n_shards + 1):
        old_name = f"model-{i:05d}.safetensors"
        new_name = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        (out / old_name).rename(out / new_name)
        renamed[old_name] = new_name
    return {key: renamed[shard] for key, shard in key_to_shard.items()}


def copy_passthrough(src, out):
    skip = {INDEX_NAME, CONFIG_NAME, ".cache"}
    for item in sorted(src.iterdir()):
        if item.name in skip or item.suffix == ".safetensors":
            continue
        if item.is_dir():
            shutil.copytree(item, out / item.name)
        else:
            shutil.copy2(item, out / item.name)


def set_n_routed_experts(config, keep):
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and "n_routed_experts" in text_config:
        text_config["n_routed_experts"] = keep
    elif "n_routed_experts" in config:
        config["n_routed_experts"] = keep
    else:
        raise ValueError("n_routed_experts not found in config.json")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--keep", type=int, default=96)
    parser.add_argument("--shard-size-gb", type=float, default=5)
    args = parser.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    weight_map = json.loads((src / INDEX_NAME).read_text())["weight_map"]
    moe_layers = find_moe_layers(weight_map)
    keep_idx, num_experts = rank_experts(src, weight_map, moe_layers, args.keep)

    shard_size_bytes = int(args.shard_size_gb * 1024**3)
    key_to_shard, total_size, n_shards = prune_shards(src, out, weight_map, keep_idx, num_experts, shard_size_bytes)
    key_to_shard = finalize_shard_names(out, key_to_shard, n_shards)
    index = {"metadata": {"total_size": total_size}, "weight_map": key_to_shard}
    (out / INDEX_NAME).write_text(json.dumps(index))

    copy_passthrough(src, out)

    config = json.loads((src / CONFIG_NAME).read_text())
    set_n_routed_experts(config, args.keep)
    (out / CONFIG_NAME).write_text(json.dumps(config, indent=2))

    manifest = {
        "src": str(src),
        "keep": args.keep,
        "ranking_method": (
            "descending signed sum(router.weight[i]), independently per layer; "
            "retained set is the prefix of that ranking"
        ),
        "layers": {str(layer): keep_idx[layer] for layer in moe_layers},
        "num_tensors": len(key_to_shard),
        "total_size": total_size,
    }
    (out / "pruning_manifest.json").write_text(json.dumps(manifest, indent=2))

    gb = total_size / 1024**3
    print(
        f"pruned {len(moe_layers)} MoE layers to {args.keep} experts, "
        f"{len(key_to_shard)} tensors, {gb:.1f} GB written to {out}"
    )


if __name__ == "__main__":
    main()
