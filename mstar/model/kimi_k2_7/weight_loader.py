"""Kimi-K2.7 / DeepSeek-V3 weight loading (M5).

Maps an HF ``DeepseekV3ForCausalLM`` checkpoint onto the mstar Kimi module tree
using the shared ``load_hf_weights`` machinery — a name remap plus stacked-shard
rules, exactly mirroring ``qwen3_omni_model.py::_get_thinker_stacked_params`` /
``_thinker_remap`` (same fused-expert layout). No shared abstraction is modified.

Two transforms take the checkpoint keys to the module's ``named_parameters``:

1. **Name remap** (:func:`kimi_name_remapper`):
     - ``mlp.shared_experts.*`` -> ``mlp.shared_expert.*`` (HF plural -> our
       singular ``ParallelGatedMLP`` submodule name);
     - per-routed-expert ``mlp.experts.{i}.{gate,up,down}_proj.weight`` ->
       ``mlp.experts.{gate,up,down}_proj.__expert{i}__.weight`` — the
       ``__expert{i}__`` marker lets the stacked rules carry both the projection
       *and* the expert slot in one ``shard_id``;
     - everything else is identity (the checkpoint prefixes ``model.``,
       ``self_attn.``, ``input_layernorm``, ``lm_head`` … already line up).

2. **Stacked rules** (:func:`build_kimi_stacked_params`):
     - routed experts: ``.experts.gate_proj.__expert{i}__.weight`` /
       ``.up_proj.__expert{i}__.weight`` -> fused ``.experts.gate_up_proj``
       ("w13", gate then up) with ``shard_id="gate:i"/"up:i"``;
       ``.down_proj.__expert{i}__.weight`` -> ``.experts.down_proj`` ("w2",
       ``shard_id="down:i"``). The fused params get their per-shard
       ``weight_loader`` in :class:`KimiSparseMoeBlock`.
     - dense + shared SwiGLU ``.gate_proj`` / ``.up_proj`` -> merged
       ``.gate_up_proj`` (shard 0 / 1). These MUST come *after* the expert rules:
       ``_apply_stacked`` returns on first match and the remapped expert key
       ``…experts.gate_proj.__expert{i}__.weight`` also contains ``.gate_proj``.

**MLA loads strictly by name — NO q_a/kv_a fusion.** M3 built *separate*
``q_a_proj`` and ``kv_a_proj_with_mqa`` (the naive/materialized path), so their
checkpoint keys map straight to the identically-named params. Fusing them into a
single ``fused_qkv_a_proj`` is only needed for the deferred weight-absorbed MLA
class (``DeepseekV2MLAAttention``); the naive path needs no such fusion. This is
a deliberate simplification of the earlier plan note.

**Router bias stays fp32.** ``KimiMoEGate.e_score_correction_bias`` is a
selection-only bias DeepSeek keeps in fp32 for router stability. A whole-model
``.to(bfloat16)`` would downcast it, so :func:`restore_router_bias_fp32` forces
every such param back to fp32 immediately before the load (the copy then lands
fp32 -> fp32). The checkpoint stores this tensor as fp32.

The dense-vs-MoE split follows ``is_moe_layer`` (``first_k_dense_replace`` /
``moe_layer_freq``): early dense layers carry ``mlp.{gate,up,down}_proj`` into a
``ParallelGatedMLP``; MoE layers carry the expert-stacked + shared + gate params.
The routing here is layer-agnostic — a dense layer simply never emits
``mlp.experts.*`` / ``mlp.gate.*`` keys, and a MoE layer never emits a bare
``mlp.gate_proj``.

Refs: HF key -> param authority is vLLM
``model_executor/models/deepseek_v2.py::DeepseekV2ForCausalLM.load_weights``
(the ``stacked_params_mapping`` + per-expert ``expert_params_mapping`` there);
the fused ``w13``/``w2`` naming is vLLM's.

----------------------------------------------------------------------------
DESIGN NOTE — compressed-tensors INT4 / fp8 dequant-on-load (DEFERRED)
----------------------------------------------------------------------------
The real Kimi-K2.7 checkpoint ships **compressed-tensors INT4** (fp8 variants
also exist); ``fused_experts`` is bf16/fp16-only and a full bf16 dequant of the
1T model is ~2 TB > 8xH200 (see the perf memo). So the real quantized checkpoint
is **out of scope for M5** (no checkpoint present, would not fit) — this loader is
the clean **bf16 path**, validated on a synthetic ``reduced()`` checkpoint.

When the checkpoint + a quantized kernel land, dequant-on-load slots in **without
touching the routing above**, because ``load_hf_weights`` dispatches per parameter
through each param's ``weight_loader``:

  * compressed-tensors stores, per quantized tensor, a packed ``*.weight_packed``
    (INT4 nibbles / fp8 bytes) plus ``*.weight_scale`` (+ optional
    ``*.weight_zero_point``) at a group granularity from the checkpoint's
    ``quantization_config``.
  * Hook point A (streaming): wrap the ``(name, tensor)`` iterator with a
    dequantizer that consumes the ``weight_packed``/``weight_scale`` group, emits
    a single bf16 ``*.weight`` (unpack nibble -> int -> ``(q - zp) * scale`` per
    group), and drops the scale/zp/packed keys. Downstream routing is unchanged.
  * Hook point B (per-param, memory-lean): keep the packed tensor in VRAM and
    give the *destination* fused param a quant-aware ``weight_loader`` that stores
    packed shards + scales (extend :class:`KimiSparseMoeBlock` to hold
    ``gate_up_proj_packed`` / ``_scale``) and swap ``_dispatch`` for a quantized
    grouped-GEMM. This is the only way to actually *serve* the 1T model and is
    tracked as the top memory item in the perf backlog — a separate effort from
    this DeepSeek-V3 port.

Either hook is additive: the bf16 routing (remap + stacked rules) below is the
substrate both build on.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

from mstar.model.loader.base import StackedParamRule

# HF checkpoint suffixes for the per-routed-expert projections.
_EXPERT_RE = re.compile(
    r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def kimi_name_remapper(name: str) -> str | None:
    """HF DeepSeek-V3 checkpoint key -> Kimi module param path.

    Returns ``None`` to drop a key (precomputed ``rotary_emb`` buffers). See the
    module docstring for the full mapping; MLA / norms / embed / lm_head are all
    identity.
    """
    if "rotary_emb" in name:
        return None
    # HF names the shared expert plural; our module has one ``shared_expert``.
    name = name.replace(".shared_experts.", ".shared_expert.")
    # Per-expert fusion marker so the stacked rules can pick up expert index.
    m = _EXPERT_RE.match(name)
    if m:
        prefix, expert_idx, proj = m.groups()
        return f"{prefix}.experts.{proj}.__expert{expert_idx}__.weight"
    return name


def build_kimi_stacked_params(n_routed_experts: int) -> list[StackedParamRule]:
    """Fused-shard routing for Kimi-K2.7 (mirrors the Qwen3-MoE thinker rules).

    Per-expert ``gate``/``up`` -> ``experts.gate_up_proj`` (w13) and ``down`` ->
    ``experts.down_proj`` (w2), then the dense/shared SwiGLU gate/up merge.
    Expert rules precede the dense rules (first-match wins in ``_apply_stacked``).
    """
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
) -> set[str]:
    """Load an HF DeepSeek-V3 weight stream into ``module``.

    Thin wrapper: restore the fp32 router bias, then dispatch through
    ``load_hf_weights`` with the Kimi remap + stacked rules. Returns the set of
    param paths that received a tensor (callers can diff against
    ``named_parameters()`` to assert completeness).
    """
    from mstar.model.loader import load_hf_weights

    restore_router_bias_fp32(module)
    return load_hf_weights(
        module,
        weights,
        stacked_params=build_kimi_stacked_params(n_routed_experts),
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
