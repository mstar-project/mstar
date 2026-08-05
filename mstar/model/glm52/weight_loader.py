"""HF GLM-5.2 checkpoint loading for the glm52 module tree.

The checkpoint documents its own map: ``modules_to_not_convert`` names every
bf16 module, ``self_attn.indexer.*`` appears on every 4th layer (DSA), and
layer index 78 (== num_hidden_layers) is the MTP module in DeepSeek-V3
naming (enorm/hnorm/eh_proj/shared_head + a full decoder layer).

M1 scope: indexer and MTP keys are skipped up front — with a count logged,
not silently — because the M1 model instantiates neither (ctx <= 2048 makes
DSA identical to dense attention; MTP is Phase D). Everything else
dequantizes to bf16 except routed experts, which load FP8-resident (see
``quantization.py`` for why).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from mstar.model.glm52.quantization import dequant_fp8_block_stream
from mstar.model.loader.base import StackedParamRule

if TYPE_CHECKING:
    from mstar.model.glm52.quantization import Fp8BlockQuantConfig

logger = logging.getLogger(__name__)

# Keep the expert index attached while remapping both fp8 and bf16 sub-keys.
_EXPERT_RE = re.compile(
    r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)"
    r"\.(weight|weight_scale_inv)$"
)

_EXPERT_BASE_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)$")

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _is_routed_expert_base(base: str) -> bool:
    return _EXPERT_BASE_RE.search(base) is not None


def glm52_name_remapper(name: str) -> str | None:
    name = name.replace(".shared_experts.", ".shared_expert.")
    m = _EXPERT_RE.match(name)
    if m:
        prefix, expert_idx, proj, suffix = m.groups()
        return f"{prefix}.experts.{proj}.__expert{expert_idx}__.{suffix}"
    return name


def skip_phase_b_keys(
    weights: Iterable[tuple[str, torch.Tensor]],
    num_hidden_layers: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Drop DSA-indexer and MTP-layer keys before any dequant buffering.

    Runs upstream of the fp8 stream so skipped fp8 pairs are never buffered
    or dequantized (the MTP layer alone carries ~9.7B expert params).
    """
    skipped_indexer = 0
    skipped_mtp = 0
    for name, tensor in weights:
        if ".self_attn.indexer." in name:
            skipped_indexer += 1
            continue
        m = _LAYER_RE.match(name)
        if m and int(m.group(1)) >= num_hidden_layers:
            skipped_mtp += 1
            continue
        yield name, tensor
    if skipped_indexer or skipped_mtp:
        logger.info(
            "GLM-5.2 M1 load: skipped %d DSA-indexer keys (Phase C) and %d "
            "MTP-layer keys (Phase D).", skipped_indexer, skipped_mtp,
        )


def build_glm52_stacked_params(
    n_routed_experts: int, fp8_experts: bool = False,
) -> list[StackedParamRule]:
    rules: list[StackedParamRule] = []
    for i in range(n_routed_experts):
        if fp8_experts:
            # scale_inv rules MUST precede weight rules: matching is
            # first-win substring, and ".weight" is a prefix of
            # ".weight_scale_inv".
            for proj, sid in (("gate_proj", f"gate:{i}"), ("up_proj", f"up:{i}")):
                rules.append(StackedParamRule(
                    target_suffix=".experts.gate_up_proj_scale_inv",
                    source_suffix=f".experts.{proj}.__expert{i}__.weight_scale_inv",
                    shard_id=sid,
                ))
                rules.append(StackedParamRule(
                    target_suffix=".experts.gate_up_proj_fp8",
                    source_suffix=f".experts.{proj}.__expert{i}__.weight",
                    shard_id=sid,
                ))
            rules.append(StackedParamRule(
                target_suffix=".experts.down_proj_scale_inv",
                source_suffix=f".experts.down_proj.__expert{i}__.weight_scale_inv",
                shard_id=f"down:{i}",
            ))
            rules.append(StackedParamRule(
                target_suffix=".experts.down_proj_fp8",
                source_suffix=f".experts.down_proj.__expert{i}__.weight",
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


def restore_fp32_params(module: nn.Module) -> None:
    """Re-widen params the checkpoint stores fp32 before loading into them.

    ``model.to(autocast_dtype)`` narrows every floating param to bf16;
    the router selection bias and the fp8 block scales must stay fp32.
    (The fp8 expert bytes live in uint8 containers and are immune.)
    """
    for name, param in module.named_parameters():
        if name.endswith("e_score_correction_bias") or name.endswith("_scale_inv"):
            if param.dtype != torch.float32:
                param.data = param.data.float()


def load_glm52_hf_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    n_routed_experts: int,
    quant_config: "Fp8BlockQuantConfig | None" = None,
    fp8_experts: bool = False,
    num_hidden_layers: int = 78,
) -> set[str]:
    from mstar.model.loader import load_hf_weights

    weights = skip_phase_b_keys(weights, num_hidden_layers)
    if quant_config is not None:
        keep = _is_routed_expert_base if fp8_experts else None
        weights = dequant_fp8_block_stream(weights, quant_config, keep_fp8=keep)
    elif fp8_experts:
        raise ValueError("fp8_experts=True requires a quant_config")

    restore_fp32_params(module)
    return load_hf_weights(
        module,
        weights,
        stacked_params=build_glm52_stacked_params(
            n_routed_experts, fp8_experts=fp8_experts,
        ),
        name_remapper=glm52_name_remapper,
    )


def load_weights(
    module: nn.Module,
    source: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    from mstar.model.loader import load_weights as _driver

    return _driver(module, source, device=device)
