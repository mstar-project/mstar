"""Static LoRA merging for the DiT scaffold.

Serving-time LoRA here means folding the adapter into the base weights once at load
(``W += scale * alpha / r * B @ A``): zero cost per step, CUDA graphs and compiled kernels
see plain Linear weights, and the merged model behaves exactly like the reference pipeline
after ``load_lora_weights`` + ``fuse_lora``. Per-request adapter switching is out of scope.

Accepted checkpoint layouts (``.safetensors``):

* diffusers / PEFT: ``[transformer.]<module>.lora_A.weight`` + ``.lora_B.weight`` (also the
  ``base_model.model.`` prefix), optionally ``<module>.alpha``; without an alpha the scaling
  is ``1`` (diffusers sets ``lora_alpha = r``).
* Kohya-style naming ``lora_down`` / ``lora_up`` for A / B.
* model-native layouts (e.g. the BFL ``double_blocks.N.img_attn.qkv`` keys of FLUX.2) through a
  model-supplied ``convert_keys`` hook that rewrites them to the diffusers module paths.

``<module>`` is a *diffusers* module path; the model's checkpoint remap (the same one the
weight loader uses) and its stacked-shard rules turn it into the native parameter and, for
fused projections, the shard to update.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch
from torch import nn

from mstar.model.components.linear import FusedColumnLinear
from mstar.model.loader.base import StackedParamRule, _apply_stacked

logger = logging.getLogger(__name__)

_PREFIXES = ("transformer.", "base_model.model.", "diffusion_model.")


@dataclass(frozen=True)
class LoraSpec:
    """One adapter to fold in: a safetensors path and its strength."""

    path: str
    scale: float = 1.0

    @classmethod
    def parse(cls, raw) -> "LoraSpec":
        if isinstance(raw, str):
            return cls(path=raw)
        return cls(path=str(raw["path"]), scale=float(raw.get("scale", 1.0)))


@dataclass
class LoraAdapter:
    """``module path -> (A [r, in], B [out, r], alpha)`` in diffusers naming."""

    layers: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.layers)


def load_lora_file(path: str, device="cpu") -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path, device=str(device))


KeyConverter = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]


def normalize_lora_state_dict(
    state_dict: dict[str, torch.Tensor], convert_keys: KeyConverter | None = None,
) -> LoraAdapter:
    """Group raw LoRA tensors into per-module ``(A, B, alpha)`` triples in diffusers naming."""
    sd = {}
    for key, value in state_dict.items():
        if "dora_scale" in key:
            continue
        for prefix in _PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        sd[key.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")] = value
    if convert_keys is not None:
        sd = convert_keys(sd)
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    alphas: dict[str, float] = {}
    for key, value in sd.items():
        m = re.match(r"^(.*)\.lora_([AB])\.weight$", key)
        if m:
            pairs.setdefault(m.group(1), {})[m.group(2)] = value
            continue
        m = re.match(r"^(.*)\.alpha$", key)
        if m:
            alphas[m.group(1)] = float(value.item())
            continue
        raise ValueError(f"unrecognized LoRA key {key!r}")
    adapter = LoraAdapter()
    for module, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            raise ValueError(f"LoRA module {module!r} is missing lora_{'B' if 'A' in ab else 'A'}")
        a, b = ab["A"], ab["B"]
        rank = a.shape[0]
        if b.shape[1] != rank:
            raise ValueError(f"LoRA module {module!r}: A is rank {rank} but B has {b.shape[1]} columns")
        adapter.layers[module] = (a, b, alphas.get(module, float(rank)))
    return adapter


def _merge_delta(param: nn.Parameter, delta: torch.Tensor, owner: nn.Module, shard_id) -> None:
    """``param[rows] += delta`` in fp32 with one final rounding to the parameter dtype."""
    if shard_id is None:
        target = param.data
    else:
        if not isinstance(owner, FusedColumnLinear):
            raise TypeError(f"stacked LoRA target {type(owner).__name__} is not a FusedColumnLinear")
        offset, size = owner.shard_slice(shard_id)
        target = param.data.narrow(0, offset, size)
    if target.shape != delta.shape:
        raise ValueError(f"LoRA delta {tuple(delta.shape)} does not match weight rows {tuple(target.shape)}")
    target.copy_((target.float() + delta.to(target.device, torch.float32)).to(target.dtype))


def merge_lora(
    module: nn.Module,
    adapter: LoraAdapter,
    remap: Callable[[str], str],
    stacked_params: Iterable[StackedParamRule] = (),
    scale: float = 1.0,
) -> list[str]:
    """Fold ``adapter`` into ``module``'s weights; returns the native parameters it touched.

    Every adapter layer must land on a parameter: an unmatched module path is a mapping bug,
    not something to skip silently (the reference would have raised as well).
    """
    params = dict(module.named_parameters())
    rules = list(stacked_params)
    touched = []
    for module_path, (a, b, alpha) in adapter.layers.items():
        native = remap(module_path + ".weight")
        target, shard_id = _apply_stacked(native, rules)
        if target not in params:
            raise KeyError(f"LoRA layer {module_path!r} maps to {target!r}, which is not a parameter of the module")
        factor = scale * alpha / a.shape[0]
        delta = (b.float() @ a.float()) * factor
        owner_path = target.rsplit(".", 1)[0]
        _merge_delta(params[target], delta, module.get_submodule(owner_path), shard_id)
        touched.append(target)
    logger.info("merged LoRA into %d parameters (scale %.3f)", len(touched), scale)
    return touched


def apply_loras(
    module: nn.Module,
    specs: Iterable[LoraSpec],
    remap: Callable[[str], str],
    stacked_params: Iterable[StackedParamRule] = (),
    convert_keys=None,
) -> None:
    """Load and merge each adapter in order (later adapters see the earlier merges)."""
    for spec in specs:
        adapter = normalize_lora_state_dict(load_lora_file(spec.path), convert_keys=convert_keys)
        merge_lora(module, adapter, remap, stacked_params, scale=spec.scale)
        logger.info("LoRA %s: %d layers merged at scale %.3f", spec.path, len(adapter), spec.scale)
