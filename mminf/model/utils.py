
from dataclasses import dataclass
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
