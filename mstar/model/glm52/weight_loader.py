"""HF GLM-5.2 checkpoint loading for the glm52 module tree."""
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
    load_indexer: bool = True,
    load_mtp: bool = False,
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
        if not load_mtp and m and int(m.group(1)) >= num_hidden_layers:
            skipped_mtp += 1
            continue
        yield name, tensor
    if skipped_indexer or skipped_mtp:
        logger.info(
            "GLM-5.2 load: skipped %d DSA-indexer keys and %d MTP-layer keys "
            "(Phase D).", skipped_indexer, skipped_mtp,
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
    load_indexer: bool = True,
    load_mtp: bool = False,
) -> tuple[set[str], "dict[str, tuple[int, int, int]]"]:
    """Keys-to-read + per-key slice specs for the TP fast read path."""
    from mstar.model.glm52.components.indexer import is_full_indexer_layer

    fp8_experts = config.quantization_config is not None and config.moe_fp8_resident
    shard_inter = config.moe_intermediate_size // tp_size
    keys: set[str] = set()
    specs: dict[str, tuple[int, int, int]] = {}
    if fp8_experts:
        bo, bi = config.quantization_config.weight_block_size
        assert shard_inter % bo == 0 and shard_inter % bi == 0, (
            f"per-rank intermediate {shard_inter} must be a multiple of the "
            f"scale block ({bo}, {bi}) for sliced reads"
        )

    for key in checkpoint_keys:
        m_layer = _LAYER_RE.match(key)
        layer = int(m_layer.group(1)) if m_layer else None
        if (
            layer is not None
            and layer >= config.num_hidden_layers
            and not load_mtp
        ):
            continue  # MTP module off: never read, never transfer
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
        is_scale = suffix == "weight_scale_inv"
        if proj in ("gate_proj", "up_proj"):
            unit = bo if is_scale else 1
            rows = shard_inter // unit
            specs[key] = (0, tp_rank * rows, (tp_rank + 1) * rows)
        else:  # down_proj: contraction dim is sharded -> column slice
            unit = bi if is_scale else 1
            cols = shard_inter // unit
            specs[key] = (1, tp_rank * cols, (tp_rank + 1) * cols)

    return keys, specs


def _make_glm52_name_remapper(num_hidden_layers: int, load_mtp: bool):
    """Trunk remapping, plus (M3) layer-78 routing onto the ``mtp.``
    submodule: strip the layer prefix, apply ``remap_mtp_key`` (glue keys
    direct, the rest under ``transformer_layer.``), then the trunk naming
    conventions — the expert/shared-expert regexes are prefix-agnostic, so
    the fused stacked-param rules apply to the MTP MoE unchanged.
    """
    if not load_mtp:
        return glm52_name_remapper

    from mstar.model.glm52.components.mtp import remap_mtp_key

    def remap(name: str) -> str | None:
        m = _LAYER_RE.match(name)
        if m and int(m.group(1)) >= num_hidden_layers:
            return glm52_name_remapper("mtp." + remap_mtp_key(name[m.end():]))
        return glm52_name_remapper(name)

    return remap


def load_glm52_hf_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    n_routed_experts: int,
    quant_config: "Fp8BlockQuantConfig | None" = None,
    fp8_experts: bool = False,
    num_hidden_layers: int = 78,
    load_indexer: bool = True,
    load_mtp: bool = False,
) -> set[str]:
    from mstar.model.loader import load_hf_weights

    weights = skip_phase_b_keys(
        weights, num_hidden_layers, load_indexer=load_indexer, load_mtp=load_mtp,
    )
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
        name_remapper=_make_glm52_name_remapper(num_hidden_layers, load_mtp),
    )


def load_weights(
    module: nn.Module,
    source: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    from mstar.model.loader import load_weights as _driver

    return _driver(module, source, device=device)
