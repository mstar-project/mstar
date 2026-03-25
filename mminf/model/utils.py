
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open


def load_weights(
    state_dict: dict[str, Any],
    module: torch.nn.Module,
    prefix: str = None,
    enforce_missing_keys: bool = True,
):
    if prefix is not None:
        if not prefix.endswith("."):
            prefix += "."
        state_dict = {
            k.removeprefix(prefix): v for k, v in state_dict.items() \
                if k.startswith(prefix)
        }

    missing_keys, _ = module.load_state_dict(state_dict, strict=False)
    if enforce_missing_keys and missing_keys:
        raise KeyError(f"Missing keys when loading state_dict with prefix {prefix!r}: {missing_keys}")


@dataclass
class ModuleAndPrefix:
    module: torch.nn.Module
    prefix: str = None
    enforce_missing_keys: bool = True


def load_weights_from_file(
    safetensors_file: str,
    modules: list[ModuleAndPrefix],
    device: str = "cpu",
):
    # Precompute expected keys for each module
    module_key_maps = []
    for mod in modules:
        prefix = mod.prefix or ""
        if prefix and not prefix.endswith("."):
            prefix += "."

        key_map = {
            prefix + k: k
            for k in mod.module.state_dict().keys()
        }
        module_key_maps.append((mod, prefix, key_map))

    # Temporary per-module state dicts
    state_dicts = [dict() for _ in modules]

    # safetensors can't take cuda:0 etc
    st_device = "cuda" if str(device).startswith("cuda") else device

    with safe_open(safetensors_file, framework="pt", device=st_device) as f:
        for k in f.keys():
            for i, (_mod, _prefix, key_map) in enumerate(module_key_maps):
                if k in key_map:
                    tensor = f.get_tensor(k)
                    if device != st_device:
                        tensor = tensor.to(device, non_blocking=True)
                    state_dicts[i][key_map[k]] = tensor
                    break

    # Load modules
    for (mod, state_dict) in zip(modules, state_dicts, strict=True):
        missing_keys, _ = mod.module.load_state_dict(state_dict, strict=False, assign=True)
        if mod.enforce_missing_keys and missing_keys:
            raise KeyError(
                f"Missing keys when loading state_dict with prefix {mod.prefix!r}: {missing_keys}"
            )


def load_weights_from_hf_shards(
    repo_dir: str | Path,
    modules: list[ModuleAndPrefix],
    device: str = "cpu",
):
    """Load weights from a sharded HuggingFace checkpoint (multiple safetensors files).

    Reads model.safetensors.index.json to find which shard each key lives in,
    then loads from each shard file.
    """
    repo_dir = Path(repo_dir)
    index_path = repo_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Precompute expected keys for each module
    module_key_maps = []
    for mod in modules:
        prefix = mod.prefix or ""
        if prefix and not prefix.endswith("."):
            prefix += "."
        key_map = {prefix + k: k for k in mod.module.state_dict().keys()}
        module_key_maps.append((mod, prefix, key_map))

    # Group checkpoint keys by shard file
    shard_to_keys: dict[str, list[str]] = {}
    for ck in weight_map:
        for _i, (_mod, _prefix, key_map) in enumerate(module_key_maps):
            if ck in key_map:
                shard = weight_map[ck]
                shard_to_keys.setdefault(shard, []).append(ck)
                break

    # Temporary per-module state dicts
    state_dicts: list[dict[str, torch.Tensor]] = [dict() for _ in modules]

    st_device = "cuda" if str(device).startswith("cuda") else device

    for shard_file, keys_in_shard in shard_to_keys.items():
        shard_path = str(repo_dir / shard_file)
        keys_set = set(keys_in_shard)
        with safe_open(shard_path, framework="pt", device=st_device) as f:
            for k in f.keys():
                if k not in keys_set:
                    continue
                for i, (_mod, _prefix, key_map) in enumerate(module_key_maps):
                    if k in key_map:
                        tensor = f.get_tensor(k)
                        if device != st_device:
                            tensor = tensor.to(device, non_blocking=True)
                        state_dicts[i][key_map[k]] = tensor
                        break

    # Load modules
    for mod, state_dict in zip(modules, state_dicts, strict=True):
        missing_keys, _ = mod.module.load_state_dict(state_dict, strict=False, assign=True)
        if mod.enforce_missing_keys and missing_keys:
            raise KeyError(f"Missing keys when loading state_dict with prefix {mod.prefix!r}: {missing_keys}")
