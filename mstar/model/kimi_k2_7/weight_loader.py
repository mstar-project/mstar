"""Kimi-K2.7 / DeepSeek-V3 weight loading.

Maps an HF ``DeepseekV3ForCausalLM`` checkpoint onto the Kimi module tree via the
shared ``load_hf_weights`` machinery (name remap + stacked-shard rules), mirroring
``qwen3_omni_model.py``'s thinker remap/stacked params.

- :func:`kimi_name_remapper`: strip a ``language_model.`` prefix if present (the
  multimodal K2.7-Code repo carries it on its text weights);
  ``shared_experts`` -> ``shared_expert``; tag per-routed-expert projections with
  an ``__expert{i}__`` marker so one ``shard_id`` carries projection + expert slot.
  Vision (``vision_tower.*`` / ``mm_projector.*``) and ``weight_shape`` keys fall
  through and are dropped by the base loader's unknown-key skip.
- :func:`build_kimi_stacked_params`: fuse per-expert gate/up -> ``gate_up_proj``
  (w13) and down -> ``down_proj`` (w2); dense/shared gate+up -> ``gate_up_proj``.
  Dense rules MUST come after the expert rules — ``_apply_stacked`` returns on
  first match and a remapped expert key also contains ``.gate_proj``.
- Router bias (``e_score_correction_bias``) is forced fp32 before load so the
  whole-model ``.to(bf16)`` cast can't downcast this fp32 selection bias.

MLA loads strictly by name — the naive path keeps separate ``q_a_proj`` /
``kv_a_proj_with_mqa``, so no ``fused_qkv_a_proj`` fusion is needed.

Compressed-tensors INT4 checkpoints: with a ``quant_config`` the stream is
dequantized to bf16 on load (see ``quantization.py``); routed experts can instead
stay packed (``packed_experts=True``) and dequantize inside the fused-expert
kernel. Both are additive — the remap + stacked rules are unchanged.

Ref: HF key -> param authority is vLLM
``model_executor/models/deepseek_v2.py::DeepseekV2ForCausalLM.load_weights``.
"""
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

# HF suffixes for the per-routed-expert projections. The trailing alternation
# covers a native-bf16 ``.weight`` AND the compressed-tensors sub-keys
# (``.weight_packed`` / ``.weight_scale`` / ``.weight_zero_point``) so a
# packed-expert stream (which passes those sub-keys through raw) is remapped with
# its expert index preserved. A dequant-on-load stream only ever carries ``.weight``.
_EXPERT_RE = re.compile(
    r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)"
    r"\.(weight|weight_packed|weight_scale|weight_zero_point)$"
)

# Base-name matcher (no suffix) for the routed-expert weights kept packed. Used to
# build the ``keep_packed`` predicate handed to the dequant stream.
_EXPERT_BASE_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)$")


def _is_routed_expert_base(base: str) -> bool:
    """True for a routed-expert weight base (``...experts.<i>.<proj>``).

    ``shared_experts`` does not match — there is no ``.experts.<digit>.`` (the HF
    key is ``mlp.shared_experts.gate_proj``, an underscore not a dotted index), so
    the shared expert still dequantizes on load while the routed experts stay packed.
    """
    return _EXPERT_BASE_RE.search(base) is not None


def kimi_name_remapper(name: str) -> str | None:
    """HF DeepSeek-V3 checkpoint key -> Kimi module param path.

    Returns ``None`` to drop a key (precomputed ``rotary_emb`` buffers). See the
    module docstring for the full mapping; MLA / norms / embed / lm_head are all
    identity. Vision (``vision_tower.*`` / ``mm_projector.*``) and ``.weight_shape``
    sub-keys are left unmapped and fall through the base loader's unknown-key skip.
    """
    if "rotary_emb" in name:
        return None
    # Multimodal K2.7-Code text keys carry a ``language_model.`` prefix; strip it
    # only-if-present (a bare ``model.*`` key is left unchanged).
    if name.startswith("language_model."):
        name = name[len("language_model."):]
    # HF names the shared expert plural; our module has one ``shared_expert``.
    name = name.replace(".shared_experts.", ".shared_expert.")
    # Per-expert fusion marker so the stacked rules can pick up expert index. The
    # suffix (``weight`` for bf16, ``weight_packed``/``weight_scale`` for packed
    # experts) is carried through so the packed vs bf16 stacked rules can route it.
    m = _EXPERT_RE.match(name)
    if m:
        prefix, expert_idx, proj, suffix = m.groups()
        return f"{prefix}.experts.{proj}.__expert{expert_idx}__.{suffix}"
    return name


def build_kimi_stacked_params(
    n_routed_experts: int, packed_experts: bool = False,
) -> list[StackedParamRule]:
    """Fused-shard routing for Kimi-K2.7 (mirrors the Qwen3-MoE thinker rules).

    ``packed_experts=False`` (native / dequantized bf16): per-expert ``gate``/``up``
    -> ``experts.gate_up_proj`` (w13) and ``down`` -> ``experts.down_proj`` (w2).

    ``packed_experts=True``: the per-expert ``.weight_packed`` / ``.weight_scale``
    sub-keys route to the FOUR packed params
    (``experts.{gate_up_proj,down_proj}_{packed,scale}``), and the bf16 ``.weight``
    expert rules are OMITTED — their ``...__expert{i}__.weight`` source substring
    would spuriously match ``...__expert{i}__.weight_packed`` (first-match wins).

    The dense/shared SwiGLU gate/up merge is appended last in both cases (expert
    rules precede it so ``.gate_proj`` inside an expert key can't hijack it).
    """
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
    # Dense MLP + shared-expert gate/up fusion — AFTER the expert rules.
    rules.append(StackedParamRule(".gate_up_proj", ".gate_proj", 0))
    rules.append(StackedParamRule(".gate_up_proj", ".up_proj", 1))
    return rules


def restore_router_bias_fp32(module: nn.Module) -> None:
    """Force every ``e_score_correction_bias`` back to fp32 in place.

    DeepSeek keeps this selection bias fp32; a whole-model ``.to(bfloat16)`` would
    downcast it. Call immediately before loading so the source (fp32) copies into
    an fp32 destination.
    """
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
    """Load an HF DeepSeek-V3 weight stream into ``module``.

    Thin wrapper: restore the fp32 router bias, optionally wrap the stream with the
    dequant-on-load parser (when ``quant_config`` is set — the checkpoint is
    compressed-tensors quantized), then dispatch through ``load_hf_weights`` with
    the Kimi remap + stacked rules. The dequant wrapper emits bf16 ``*.weight``
    keys, so the remap + stacked rules see the same stream as a native-bf16
    checkpoint. Returns the set of param paths that received a tensor (callers can
    diff against ``named_parameters()`` to assert completeness).

    ``packed_experts=True``: the routed experts stay PACKED. A ``keep_packed``
    predicate matching routed-expert bases is handed to the dequant stream so those
    sub-keys pass through raw (int32 + scale), and the stacked rules route them to
    the packed params; every other quantized weight (MLA, dense FFN, shared expert)
    still dequantizes to bf16. Requires ``quant_config``.
    """
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
    """``(module, source, device)`` entrypoint mirroring Orpheus.

    ``source`` is a safetensors file or an HF-style checkpoint directory. Picks
    the right streaming iterator and drives ``module.load_weights`` (which calls
    :func:`load_kimi_hf_weights`).
    """
    from mstar.model.loader import load_weights as _driver

    return _driver(module, source, device=device)
