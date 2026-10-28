"""HF GLM-5.2 checkpoint loading."""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from mstar.model.components.quantization import Fp8Block, expert_param_rules
from mstar.model.loader.base import StackedParamRule

if TYPE_CHECKING:
    from mstar.utils.fp8_block import Fp8BlockQuantConfig

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


def _role_of(base: str) -> str:
    """A quantized checkpoint weight's role: routed experts or everything else."""
    return "routed_experts" if _is_routed_expert_base(base) else "linear"


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
    load_indexer: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Drop MTP-layer keys — and, with ``load_indexer=False``, indexer keys —
    before any dequant buffering.
    """
    skipped_indexer = 0
    skipped_mtp = 0
    for name, tensor in weights:
        if not load_indexer and ".self_attn.indexer." in name:
            skipped_indexer += 1
            continue
        m = _LAYER_RE.match(name)
        if m and int(m.group(1)) >= num_hidden_layers:
            skipped_mtp += 1
            continue
        yield name, tensor
    if skipped_indexer or skipped_mtp:
        logger.info(
            "GLM-5.2 load: skipped %d DSA-indexer keys and %d MTP-layer keys.",
            skipped_indexer, skipped_mtp,
        )


def build_glm52_stacked_params(
    n_routed_experts: int, fp8_experts: bool = False,
) -> list[StackedParamRule]:
    rules: list[StackedParamRule] = []
    if fp8_experts:
        # the rules don't depend on the scale block
        rules += expert_param_rules(
            Fp8Block(), n_routed_experts, lambda proj, i: f".experts.{proj}.__expert{i}__")
    for i in range(n_routed_experts):
        if not fp8_experts:
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
    """Re-widen params the checkpoint stores fp32 before loading into them."""
    for name, param in module.named_parameters():
        if name.endswith("e_score_correction_bias") or name.endswith("_scale_inv"):
            if param.dtype != torch.float32:
                param.data = param.data.float()


def build_glm52_read_plan(
    checkpoint_keys: Iterable[str],
    config,
    tp_rank: int,
    tp_size: int,
    load_indexer: bool | None = None,
) -> tuple[set[str], "dict[str, tuple[int, int, int]]"]:
    """Keys-to-read + per-key slice specs for the TP fast read path.

    ``load_indexer`` defaults to ``config.dsa_long_context``: flag-off the
    model builds no indexer to fill, so its keys are never read.
    """
    from mstar.model.glm52.components.indexer import is_full_indexer_layer
    from mstar.model.glm52.config import glm52_quant_method

    if load_indexer is None:
        load_indexer = bool(config.dsa_long_context)
    fp8_experts = config.quantization_config is not None and config.moe_fp8_resident
    shard_inter = config.moe_intermediate_size // tp_size
    keys: set[str] = set()
    specs: dict[str, tuple[int, int, int]] = {}
    if fp8_experts:
        quant = glm52_quant_method(config.quantization_config, fp8_resident=True)
        bo, bi = config.quantization_config.weight_block_size
        assert shard_inter % bo == 0 and shard_inter % bi == 0, (
            f"per-rank intermediate {shard_inter} must be a multiple of the "
            f"scale block ({bo}, {bi}) for sliced reads"
        )

    for key in checkpoint_keys:
        m_layer = _LAYER_RE.match(key)
        layer = int(m_layer.group(1)) if m_layer else None
        if layer is not None and layer >= config.num_hidden_layers:
            continue  # the MTP layer: never read, never transfer
        if ".self_attn.indexer." in key and (
            not load_indexer
            or layer is None
            or not is_full_indexer_layer(config, layer)
        ):
            continue
        keys.add(key)

        if not fp8_experts:
            continue
        m = _EXPERT_RE.match(key)
        if not m:
            continue
        _, _, proj, suffix = m.groups()
        # gate / up rows, down's contraction columns
        specs[key] = quant.expert_read_slice(
            suffix, proj, config.moe_intermediate_size, tp_rank, tp_size)

    return keys, specs


def load_glm52_hf_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    n_routed_experts: int,
    quant_config: "Fp8BlockQuantConfig | None" = None,
    fp8_experts: bool = False,
    num_hidden_layers: int = 78,
    load_indexer: bool = True,
) -> set[str]:
    from mstar.model.loader import load_hf_weights

    weights = skip_phase_b_keys(weights, num_hidden_layers, load_indexer=load_indexer)
    if quant_config is not None:
        from mstar.model.glm52.config import glm52_quant_method

        quant = glm52_quant_method(quant_config, fp8_experts)
        weights = quant.checkpoint_stream(weights, _role_of)
    elif fp8_experts:
        raise ValueError("fp8_experts=True requires a quant_config")
    else:
        weights = _refuse_fp8(weights)

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


def _refuse_fp8(weights: Iterable[tuple[str, torch.Tensor]]):
    """Without a quant config an fp8 checkpoint loaded clean and served garbage: its e4m3
    tensors were copied unscaled and its block scales dropped as unmatched."""
    for name, tensor in weights:
        if name.endswith("_scale_inv") or tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError(
                f"{name} is fp8 or a block scale, but no quantization_config was found "
                "(config.json's, at its top level)"
            )
        yield name, tensor
