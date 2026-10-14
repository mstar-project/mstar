"""Loading weights into the DiT scaffold's shared components.

Model-agnostic machinery: walking a diffusers or transformers component
directory, streaming it into a module under the completeness contract, and
materialising a meta-device module. Shared by every model on the scaffold, so a
model never imports a sibling model's loader. The Qwen3 prompt encoder's own
rules live in ``qwen3.weight_loading``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
from safetensors import safe_open

from mstar.model.loader.base import StackedParamRule, load_weights_into
from mstar.model.loader.iterators import iter_safetensors_file

logger = logging.getLogger(__name__)





# Checkpoint keys that intentionally load nothing.
_IGNORED_KEYS = {"bn.num_batches_tracked"}


def _iter_component(
    component_dir: Path, device, index_name: str, single_name: str, skip: Callable[[str], bool] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream the safetensors of one pipeline component (sharded or single file).

    ``skip`` names the checkpoint keys the caller will not load (the encoder's unused LM
    layers, say); they are filtered before the read, so they never touch the device.
    """
    def wanted(names) -> set[str]:
        return {name for name in names if skip is None or not skip(name)}

    index_path = component_dir / index_name
    if index_path.exists():
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
        for shard in sorted(set(weight_map.values())):
            keys = wanted(name for name, in_shard in weight_map.items() if in_shard == shard)
            if keys:
                yield from iter_safetensors_file(component_dir / shard, device=device, keys=keys)
        return
    single = component_dir / single_name
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            keys = wanted(f.keys())
        yield from iter_safetensors_file(single, device=device, keys=keys)
        return
    raise FileNotFoundError(f"no safetensors checkpoint in {component_dir}")


def iter_diffusers_component(component_dir: Path, device) -> Iterator[tuple[str, torch.Tensor]]:
    return _iter_component(
        component_dir, device, "diffusion_pytorch_model.safetensors.index.json", "diffusion_pytorch_model.safetensors",
    )


def iter_transformers_component(
    component_dir: Path, device, skip: Callable[[str], bool] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    return _iter_component(component_dir, device, "model.safetensors.index.json", "model.safetensors", skip=skip)


def load_native(
    module: torch.nn.Module,
    weights: Iterator[tuple[str, torch.Tensor]],
    remap,
    what: str,
    stacked_params: list[StackedParamRule] | None = None,
    skip=None,
) -> torch.nn.Module:
    """Stream ``weights`` into ``module`` through ``remap`` and enforce the completeness contract."""
    targets = dict(module.named_parameters())
    targets.update(dict(module.named_buffers()))
    unexpected: list[str] = []
    skipped: list[str] = []

    def remapper(name: str) -> str | None:
        if name in _IGNORED_KEYS or (skip is not None and skip(name)):
            skipped.append(name)
            return None
        mapped = remap(name)
        # a stacked rule's source name maps to the fused target; check that instead
        for rule in stacked_params or ():
            if rule.source_suffix in mapped:
                mapped_target = mapped.replace(rule.source_suffix, rule.target_suffix)
                if mapped_target in targets:
                    return mapped
                unexpected.append(name)
                return None
        if mapped not in targets:
            unexpected.append(name)
            return None
        return mapped

    loaded = load_weights_into(module, weights, stacked_params=stacked_params, name_remapper=remapper)
    missing = sorted(set(targets) - loaded)
    if unexpected or missing:
        raise RuntimeError(
            f"{what}: checkpoint/module mismatch — {len(unexpected)} unexpected checkpoint keys "
            f"{unexpected[:5]}, {len(missing)} unloaded parameters/buffers {missing[:5]}; refusing to serve a "
            "partially loaded module."
        )
    if skipped:
        logger.info("%s: skipped %d checkpoint keys by design (%s...)", what, len(skipped), skipped[:2])
    return module.eval()


def materialize(module: torch.nn.Module, dtype: torch.dtype, device) -> torch.nn.Module:
    module.to(dtype)  # on meta: storage is allocated directly in the serving dtype below
    return module.to_empty(device=device)
