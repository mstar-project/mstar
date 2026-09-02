"""HF DeepSeek-V3 checkpoint loading for the Kimi-K2.7 module tree."""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from mstar.model.loader.base import StackedParamRule

if TYPE_CHECKING:
    from mstar.model.kimi_k2_7.quantization import CompressedTensorsQuantConfig

# Keep the expert index attached while remapping both bf16 and packed sub-keys.
_EXPERT_RE = re.compile(
    r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)"
    r"\.(weight|weight_packed|weight_scale|weight_zero_point)$"
)

_EXPERT_BASE_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)$")


def _is_routed_expert_base(base: str) -> bool:
    return _EXPERT_BASE_RE.search(base) is not None


def kimi_name_remapper(name: str) -> str | None:
    if "rotary_emb" in name:
        return None
    if name.startswith("language_model."):
        name = name[len("language_model."):]
    name = name.replace(".shared_experts.", ".shared_expert.")
    m = _EXPERT_RE.match(name)
    if m:
        prefix, expert_idx, proj, suffix = m.groups()
        return f"{prefix}.experts.{proj}.__expert{expert_idx}__.{suffix}"
    return name


def build_kimi_stacked_params(
    n_routed_experts: int, packed_experts: bool = False,
) -> list[StackedParamRule]:
    rules: list[StackedParamRule] = []
    for i in range(n_routed_experts):
        if packed_experts:
            for proj, sid in (("gate_proj", f"gate:{i}"), ("up_proj", f"up:{i}")):
                rules.append(StackedParamRule(
                    target_suffix=".experts.gate_up_proj_packed",
                    source_suffix=f".experts.{proj}.__expert{i}__.weight_packed",
                    shard_id=sid,
                ))
                rules.append(StackedParamRule(
                    target_suffix=".experts.gate_up_proj_scale",
                    source_suffix=f".experts.{proj}.__expert{i}__.weight_scale",
                    shard_id=sid,
                ))
            rules.append(StackedParamRule(
                target_suffix=".experts.down_proj_packed",
                source_suffix=f".experts.down_proj.__expert{i}__.weight_packed",
                shard_id=f"down:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.down_proj_scale",
                source_suffix=f".experts.down_proj.__expert{i}__.weight_scale",
                shard_id=f"down:{i}",
            ))
        else:
            rules.append(StackedParamRule(
                target_suffix=".experts.gate_up_proj",
                source_suffix=f".experts.gate_proj.__expert{i}__.weight",
                shard_id=f"gate:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.gate_up_proj",
                source_suffix=f".experts.up_proj.__expert{i}__.weight",
                shard_id=f"up:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.down_proj",
                source_suffix=f".experts.down_proj.__expert{i}__.weight",
                shard_id=f"down:{i}",
            ))
    # Dense/shared gate-up rules must follow expert rules because matching is first-win.
    rules.append(StackedParamRule(".gate_up_proj", ".gate_proj", 0))
    rules.append(StackedParamRule(".gate_up_proj", ".up_proj", 1))
    return rules


def restore_router_bias_fp32(module: nn.Module) -> None:
    """Force DeepSeek's router selection bias back to fp32 before loading."""
    for sub in module.modules():
        bias = getattr(sub, "e_score_correction_bias", None)
        if isinstance(bias, nn.Parameter) and bias.dtype != torch.float32:
            bias.data = bias.data.float()


def load_kimi_hf_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    n_routed_experts: int,
    quant_config: "CompressedTensorsQuantConfig | None" = None,
    packed_experts: bool = False,
) -> set[str]:
    from mstar.model.loader import load_hf_weights

    if quant_config is not None:
        from mstar.model.kimi_k2_7.quantization import (
            dequant_compressed_tensors_stream,
        )

        keep_packed = _is_routed_expert_base if packed_experts else None
        weights = dequant_compressed_tensors_stream(
            weights, quant_config, keep_packed=keep_packed,
        )
    elif packed_experts:
        raise ValueError("packed_experts=True requires a quant_config")

    restore_router_bias_fp32(module)
    return load_hf_weights(
        module,
        weights,
        stacked_params=build_kimi_stacked_params(
            n_routed_experts, packed_experts=packed_experts,
        ),
        name_remapper=kimi_name_remapper,
    )


def load_weights(
    module: nn.Module,
    source: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    from mstar.model.loader import load_weights as _driver

    return _driver(module, source, device=device)
