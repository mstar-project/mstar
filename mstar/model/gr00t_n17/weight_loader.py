"""Where each tensor of a GR00T N1.7 checkpoint goes.

The checkpoint holds one model under two roots: ``backbone.`` (the Qwen3-VL,
already cut to ``select_layer`` layers) and ``action_head.``. Each node loads
the tensors under its own prefixes, with the root stripped and the rest kept as
the reference names them. The LLM's separate q/k/v and gate/up load into fused
parameters.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mstar.model.loader.base import StackedParamRule, load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards

_VISION = "backbone.model.model.visual."
_LLM = "backbone.model.model.language_model."
_ACTION_HEAD = "action_head."

# node -> (root it strips, checkpoint prefixes it owns)
NODE_PREFIXES: dict[str, tuple[str, tuple[str, ...]]] = {
    "vision_encoder": (_VISION, (_VISION,)),
    "LLM": (_LLM, (_LLM,)),
    "vl_bridge": (_ACTION_HEAD, ("action_head.vlln.", "action_head.vl_self_attention.")),
    "state_encoder": (_ACTION_HEAD, ("action_head.state_encoder.",)),
    "denoise_step": (_ACTION_HEAD, (
        "action_head.action_encoder.",
        "action_head.model.",
        "action_head.action_decoder.",
        "action_head.position_embedding.",
    )),
}

# the one part of `visual.` that is GR00T's and not the reused tower's
_DEEPSTACK = "deepstack_merger_list."

# GR00T reads `hidden_states[-1]` off `Qwen3VLForConditionalGeneration`, which
# is the last decoder layer's output *before* the final norm -- checked at that
# level, because `Qwen3VLTextModel` called directly reports the normed state
# there instead. So neither the norm nor the logits run.
SKIPPED = frozenset({
    "backbone.model.model.language_model.norm.weight",
    "backbone.model.lm_head.weight",
})

_LLM_STACKED: list[StackedParamRule] = [
    StackedParamRule(".qkv_proj", ".q_proj", "q"),
    StackedParamRule(".qkv_proj", ".k_proj", "k"),
    StackedParamRule(".qkv_proj", ".v_proj", "v"),
    StackedParamRule(".gate_up_proj", ".gate_proj", 0),
    StackedParamRule(".gate_up_proj", ".up_proj", 1),
]


def node_of(name: str) -> str | None:
    """The node a checkpoint tensor loads into; None for a named skip."""
    if name in SKIPPED:
        return None
    for node, (_, prefixes) in NODE_PREFIXES.items():
        if name.startswith(prefixes):
            return node
    raise ValueError(f"checkpoint tensor {name!r} belongs to no GR00T node")


def _remapper(node: str):
    root, prefixes = NODE_PREFIXES[node]

    def remap(name: str) -> str | None:
        if name in SKIPPED or not name.startswith(prefixes):
            return None
        name = name[len(root):]
        # `vision_encoder` keeps Qwen3.5's tower under `.tower`, with GR00T's
        # own DeepStack mergers beside it, so the tower's keys gain that prefix
        if node == "vision_encoder" and not name.startswith(_DEEPSTACK):
            name = "tower." + name
        return name

    return remap


def load_gr00t_n17_weights(
    module: nn.Module,
    node: str,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    """Load one node's tensors, strictly: every parameter gets one, every tensor lands."""
    root, _ = NODE_PREFIXES[node]
    remap, arrived = _remapper(node), set()

    def tracked(tensors):
        for name, tensor in tensors:
            arrived.add(remap(name))
            yield name, tensor

    # the root only narrows which shards open; the remapper picks the node's keys
    loaded = load_weights_into(
        module,
        tracked(iter_safetensors_shards(Path(path), device=device, prefix=root)),
        name_remapper=remap,
        stacked_params=_LLM_STACKED if node == "LLM" else None,
    )
    # A fused parameter counts as loaded once *any* of its parts lands (`load_weights_into`),
    # so a checkpoint missing one of q/k/v or gate/up would serve those rows at random init.
    missing = sorted({n for n, _ in module.named_parameters()} - loaded) + sorted(
        name.replace(rule.target_suffix, rule.source_suffix)
        for name, _ in module.named_parameters() if node == "LLM"
        for rule in _LLM_STACKED
        if rule.target_suffix in name and name.replace(rule.target_suffix, rule.source_suffix) not in arrived
    )
    if missing:
        raise ValueError(f"{node}: {len(missing)} parameter(s) got no weight, e.g. {missing[:5]}")
    # `load_weights_into` only logs a key that matches no parameter; under this node's
    # prefixes that is a renamed or missing module whose trained weights go unused.
    stacked = _LLM_STACKED if node == "LLM" else []
    unexpected = sorted(
        name for name in arrived - {None}
        if next((name.replace(r.source_suffix, r.target_suffix) for r in stacked if r.source_suffix in name), name)
        not in loaded
    )
    if unexpected:
        raise ValueError(f"{node}: {len(unexpected)} tensor(s) matched no parameter, e.g. {unexpected[:5]}")
    return loaded
