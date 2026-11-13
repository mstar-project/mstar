"""HF GLM-5.2 checkpoint loading."""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from torch import nn

from mstar.model.loader.base import StackedParamRule

logger = logging.getLogger(__name__)

# Keep the expert index attached while remapping both fp8 and bf16 sub-keys.
_EXPERT_RE = re.compile(
    r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)"
    r"\.(weight|weight_scale_inv)$"
)

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


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
            "GLM-5.2 load: skipped %d DSA-indexer keys and %d MTP-layer keys.",
            skipped_indexer, skipped_mtp,
        )


def build_glm52_stacked_params(n_routed_experts: int) -> list[StackedParamRule]:
    """Per-expert gate/up/down onto the stacked expert params; a block-fp8 checkpoint's
    ``weight_scale_inv`` rides its weight's rule onto ``<param>_scale_inv`` (substring match)."""
    rules: list[StackedParamRule] = []
    for i in range(n_routed_experts):
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
    """Re-widen the params the checkpoint stores fp32 (the router's correction bias) after a
    bf16 cast. Block scales live in int32 containers (the quantization plan's storage), which
    no cast touches: an integer param is never widened, which would change its bytes."""
    for name, param in module.named_parameters():
        if name.endswith("e_score_correction_bias") and param.dtype.is_floating_point:
            if param.dtype != torch.float32:
                param.data = param.data.float()


def build_glm52_read_plan(
    checkpoint_keys: Iterable[str],
    config,
    tp_rank: int,
    tp_size: int,
    load_indexer: bool | None = None,
    load_mtp: bool = False,
) -> tuple[set[str], "dict[str, tuple[int, int, int]]"]:
    """Keys-to-read + per-key slice specs for the TP fast read path.

    ``load_indexer`` defaults to ``config.dsa_long_context``: flag-off the
    model builds no indexer to fill, so its keys are never read.
    """
    from mstar.model.glm52.components.indexer import is_full_indexer_layer

    if load_indexer is None:
        load_indexer = bool(config.dsa_long_context)
    # a checkpoint's fp8 experts read only this rank's slice; every expert loader takes one
    fp8_experts = config.quantization_config is not None
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
        if layer is not None and layer >= config.num_hidden_layers and not load_mtp:
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
    """The trunk's remapper, plus the layer-78 keys onto ``mtp.`` through ``remap_mtp_key``;
    the expert regexes are prefix-agnostic, so the stacked-param rules cover the MTP MoE."""
    if not load_mtp:
        return glm52_name_remapper

    from mstar.model.glm52.components.mtp import remap_mtp_key

    def remap(name: str) -> str | None:
        m = _LAYER_RE.match(name)
        if m and int(m.group(1)) >= num_hidden_layers:
            if int(m.group(1)) > num_hidden_layers:
                # one draft module: a second next-token layer would overwrite the first
                raise ValueError(f"{name}: MTP serves one next-token layer, "
                                 f"{num_hidden_layers}; the checkpoint has more")
            return glm52_name_remapper("mtp." + remap_mtp_key(name[m.end():]))
        return glm52_name_remapper(name)

    return remap


def load_glm52_hf_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    n_routed_experts: int,
    num_hidden_layers: int = 78,
    load_indexer: bool = True,
    load_mtp: bool = False,
) -> set[str]:
    """Load an HF GLM-5.2 stream. Its fp8 linears go where the node's quantization plan says
    (``mstar.quantization``: stored, dequantized, or quantized); with no plan active, an fp8
    tensor is refused rather than copied unscaled into a bf16 parameter."""
    from mstar.model.loader import load_hf_weights
    from mstar.quantization.scope import active_scope

    weights = skip_phase_b_keys(
        weights, num_hidden_layers, load_indexer=load_indexer, load_mtp=load_mtp,
    )
    if active_scope() is None:
        weights = _refuse_fp8(weights)
    restore_fp32_params(module)
    return load_hf_weights(
        module,
        weights,
        stacked_params=build_glm52_stacked_params(n_routed_experts),
        name_remapper=_make_glm52_name_remapper(num_hidden_layers, load_mtp),
    )


def load_weights(
    module: nn.Module,
    source: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    from mstar.model.loader import load_weights as _driver

    return _driver(module, source, device=device)


def _refuse_fp8(weights: Iterable[tuple[str, torch.Tensor]]):
    """Outside a quantization plan an fp8 checkpoint would load clean and serve garbage: its
    e4m3 tensors copied unscaled and its block scales dropped as unmatched."""
    for name, tensor in weights:
        if name.endswith("_scale_inv") or tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError(
                f"{name} is fp8 or a block scale, but no quantization plan is active (load "
                "through the engine, or mstar.quantization.scope around the load)"
            )
        yield name, tensor
