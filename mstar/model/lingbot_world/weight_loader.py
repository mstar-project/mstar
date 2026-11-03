"""Build the LingBot-World causal DiT from its safetensors shards.

The checkpoint is fp32 in the original Wan layout; the module keeps those
names, so loading is a strict key match plus a cast to bf16 (round to nearest,
as the reference's ``model.to(dtype=torch.bfloat16)``).
"""

import json
import os

import torch

from mstar.model.lingbot_world.components.dit import LingBotCausalDiT
from mstar.model.lingbot_world.config import LingBotWorldConfig


def resolve_dit_dir(repo_or_dir: str, cache_dir: str | None = None) -> str:
    if os.path.isdir(repo_or_dir):
        root = repo_or_dir
    else:
        from huggingface_hub import snapshot_download

        root = snapshot_download(repo_or_dir, cache_dir=cache_dir, allow_patterns=["*.safetensors", "*.json"])
    return root


def build_lingbot_dit(
    config: LingBotWorldConfig, repo_or_dir: str, device: str = "cuda", cache_dir: str | None = None,
) -> LingBotCausalDiT:
    from safetensors import safe_open

    dit_dir = resolve_dit_dir(repo_or_dir, cache_dir)
    index_name = next(
        name for name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json")
        if os.path.isfile(os.path.join(dit_dir, name))
    )
    with open(os.path.join(dit_dir, index_name)) as f:
        weight_map = json.load(f)["weight_map"]

    with torch.device("meta"):
        dit = LingBotCausalDiT(config)
    dit = dit.to(torch.bfloat16).to_empty(device=device).eval().requires_grad_(False)
    params = dict(dit.named_parameters())
    missing = set(params) - set(weight_map)
    unexpected = set(weight_map) - set(params)
    if missing or unexpected:
        raise ValueError(
            f"LingBot DiT checkpoint does not match the module: missing {sorted(missing)[:8]}, "
            f"unexpected {sorted(unexpected)[:8]}"
        )
    loaded = set()
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(dit_dir, shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name not in params:
                    raise ValueError(f"LingBot DiT shard {shard} holds {name!r}, which the module does not have")
                if weight_map[name] != shard:
                    raise ValueError(
                        f"LingBot DiT shard {shard} holds {name!r}, which the index puts in {weight_map[name]}"
                    )
                tensor = f.get_tensor(name)
                if tuple(tensor.shape) != tuple(params[name].shape):
                    raise ValueError(f"{name}: checkpoint {tuple(tensor.shape)} vs module {tuple(params[name].shape)}")
                params[name].data.copy_(tensor.to(torch.bfloat16))
                loaded.add(name)
    # the index can list a tensor its shard lacks; ``to_empty`` left that one uninitialized
    if unloaded := set(params) - loaded:
        raise ValueError(f"LingBot DiT checkpoint shards are missing {sorted(unloaded)[:8]}")
    return dit
