"""Kimi K3 / K3.1 checkpoint names onto ``KimiK3ForCausalLM``.

The checkpoint names the text model ``language_model.{model.*, lm_head.weight}`` beside the
vision tower and projector (``vision_tower.*``, ``mm_projector.*``, skipped). Within a layer:

* KDA: ``q/k/v_conv1d`` fill the fused ``conv1d`` (shards q, k, v); ``f_a_proj``, ``f_b_proj``,
  ``dt_bias`` and ``A_log`` sit under ``forget_gate``. ``A_log`` is stored padded to 128 entries
  for 96 heads (the tail is zeros); the stream keeps the first ``linear_num_heads``.
* routed experts: ``experts.<e>.{w1,w3,w2}`` fill the stacked ``experts.gate_up_proj`` (gate:e,
  up:e) and ``experts.down_proj`` (down:e). bf16 checkpoints name them ``.weight``; the MXFP4
  release names ``.weight_packed`` / ``.weight_scale``, which the same rules route to
  ``gate_up_proj_packed`` / ``gate_up_proj_scale`` (the suffix after ``weight`` is kept), the
  storage the quantization API's mxfp4 experts format gives the block when it loads as stored.
* dense and shared MLPs: ``gate_proj`` / ``up_proj`` fill ``gate_up_proj`` (0, 1).
* the indexer (K3.1) loads only under ``dsa_long_context``.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator

import torch
from torch import nn

from mstar.model.kimi_k3.config import KimiK3ModelConfig
from mstar.model.loader.base import CheckpointRules, StackedParamRule

logger = logging.getLogger(__name__)

PREFIX = "language_model."
_SKIPPED_PREFIXES = ("vision_tower.", "mm_projector.")

_EXPERT_RE = re.compile(r"^(.*\.experts)\.(\d+)\.(w1|w2|w3)\.(weight|weight_packed|weight_scale)$")
_KDA_FORGET_RE = re.compile(r"\.self_attn\.(f_a_proj\.weight|f_b_proj\.weight|dt_bias|A_log)$")
_A_LOG_RE = re.compile(r"\.self_attn\.A_log$")

# parameters the checkpoint stores fp32 and the forward reads fp32; restored after the
# module-wide cast to the serving dtype, before loading
FP32_PARAM_SUFFIXES = (
    "e_score_correction_bias",   # the router's selection bias
    ".dt_bias", ".A_log",        # KDA's forget gate
    "conv1d.weight",             # KDA's short conv
)


def kimi_k3_name_remapper(dsa_long_context: bool):
    """A ``language_model.``-stripped checkpoint name -> the module's, before stacked rules;
    None drops it."""

    def remap(name: str) -> str | None:
        if ".self_attn.indexer." in name and not dsa_long_context:
            return None
        name = _KDA_FORGET_RE.sub(r".self_attn.forget_gate.\1", name)
        m = _EXPERT_RE.match(name)
        if m:
            prefix, expert, proj, suffix = m.groups()
            return f"{prefix}.{proj}.__expert{expert}__.{suffix}"
        return name

    return remap


def build_stacked_params(num_experts: int) -> list[StackedParamRule]:
    rules: list[StackedParamRule] = []
    for e in range(num_experts):
        # a rule replaces ``.experts.w1.__expert<e>__.weight`` and keeps what follows it, so
        # ``weight_packed`` / ``weight_scale`` reach the quantized storage's ``_packed`` / ``_scale``
        rules += [
            StackedParamRule(".experts.gate_up_proj", f".experts.w1.__expert{e}__.weight", f"gate:{e}"),
            StackedParamRule(".experts.gate_up_proj", f".experts.w3.__expert{e}__.weight", f"up:{e}"),
            StackedParamRule(".experts.down_proj", f".experts.w2.__expert{e}__.weight", f"down:{e}"),
        ]
    # KDA's three checkpoint convs row-concat into one [3 * qkv, 1, K] parameter
    rules += [StackedParamRule(".conv1d.weight", f".{b}_conv1d.weight", b) for b in ("q", "k", "v")]
    # specific targets, so strict loading's shard check never pairs them with the experts'
    for owner in (".mlp", ".shared_experts"):
        rules += [StackedParamRule(f"{owner}.gate_up_proj", f"{owner}.gate_proj", 0),
                  StackedParamRule(f"{owner}.gate_up_proj", f"{owner}.up_proj", 1)]
    return rules


def checkpoint_rules(config: KimiK3ModelConfig) -> CheckpointRules:
    """How the checkpoint's names land on ``KimiK3ForCausalLM``: for the loader, the
    quantization API and the index check alike."""
    return CheckpointRules(
        stacked_params=tuple(build_stacked_params(config.num_experts)),
        remap=kimi_k3_name_remapper(config.dsa_long_context),
        prefix=PREFIX,
    )


def text_weights(
    weights: Iterable[tuple[str, torch.Tensor]], config: KimiK3ModelConfig,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The text model's tensors, ``language_model.`` stripped, with ``A_log`` cut to its heads."""
    skipped = 0
    for name, tensor in weights:
        if name.startswith(_SKIPPED_PREFIXES):
            skipped += 1
            continue
        if not name.startswith(PREFIX):
            raise ValueError(f"unexpected checkpoint tensor {name!r} (not language_model.*)")
        name = name[len(PREFIX):]
        if _A_LOG_RE.search(name):
            tensor = _heads_of_a_log(name, tensor, config.linear_num_heads)
        yield name, tensor
    if skipped:
        logger.info("Kimi K3: skipped %d vision tower / projector tensors (text only).", skipped)


def _heads_of_a_log(name: str, tensor: torch.Tensor, heads: int) -> torch.Tensor:
    """The checkpoint pads ``A_log`` to 128 entries; the padding must be zeros, or the layout is
    not the one this loader knows."""
    flat = tensor.reshape(-1)
    if flat.numel() == heads:
        return flat
    if flat.numel() < heads or (not flat.is_meta and bool(flat[heads:].any())):
        raise ValueError(f"{name}: {flat.numel()} entries for {heads} heads, and the tail is not "
                         "zero padding")
    return flat[:heads]


def restore_fp32_params(module: nn.Module) -> None:
    """Re-widen the parameters the checkpoint stores fp32 before loading into them."""
    for name, param in module.named_parameters():
        if name.endswith(FP32_PARAM_SUFFIXES) and param.dtype != torch.float32:
            param.data = param.data.float()


def load_kimi_k3_weights(
    module: nn.Module, weights: Iterable[tuple[str, torch.Tensor]], config: KimiK3ModelConfig,
) -> set[str]:
    """Load the checkpoint stream into ``module`` (a ``KimiK3ForCausalLM``); strict: every
    text tensor must land, and every parameter (each stacked shard) must get one."""
    from mstar.model.loader import load_weights_into

    def guard(stream):
        packed = None  # whether the experts hold packed storage, read once the scope prepared them
        for name, tensor in stream:
            if name.endswith((".weight_packed", ".weight_scale")):
                if packed is None:
                    packed = any(n.endswith(".experts.gate_up_proj_packed") for n, _ in module.named_parameters())
                if not packed:
                    raise ValueError(
                        f"{name}: the routed experts are MXFP4 (compressed-tensors) in this checkpoint, "
                        "but this load holds them bf16: load the node under its quantization scope "
                        "(mstar.quantization.scope.load_submodule, as the engine does), or use a "
                        "checkpoint with bf16 experts")
            yield name, tensor

    restore_fp32_params(module)
    rules = checkpoint_rules(config)
    return load_weights_into(
        module, guard(text_weights(weights, config)), stacked_params=list(rules.stacked_params),
        name_remapper=rules.remap, strict=True,
    )


# ---------------------------------------------------------------------------
# Name-level dry run of the load against a checkpoint's index


def resolve_index_names(
    index_names: Iterable[str], config: KimiK3ModelConfig, module_params: Iterable[str],
) -> dict[str, list[str]]:
    """Every checkpoint name through the loader's stages, by outcome: ``skipped`` (vision,
    projector), ``dropped`` (the indexer with DSA off), ``loaded`` (``name -> param``), and
    ``unmapped`` (lands on no parameter of ``module_params``). The real rule matcher, not a copy."""
    from mstar.model.loader.base import _StackedMatcher

    params = set(module_params)
    rules = checkpoint_rules(config)
    match = _StackedMatcher(list(rules.stacked_params))
    out: dict[str, list[str]] = {"skipped": [], "dropped": [], "loaded": [], "unmapped": []}
    for raw in index_names:
        if raw.startswith(_SKIPPED_PREFIXES):
            out["skipped"].append(raw)
            continue
        native = rules.native(raw)
        if native is None:
            out["dropped"].append(raw)
            continue
        target, _ = match(native)
        out["loaded" if target in params else "unmapped"].append(f"{raw} -> {target}")
    return out
