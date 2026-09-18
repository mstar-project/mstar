"""Streaming loader for the Kokoro ``.pth`` checkpoint.

The upstream file is a dict of five state dicts (``bert``, ``bert_encoder``,
``predictor``, ``decoder``, ``text_encoder``) saved from ``DataParallel``
wrappers (``module.`` prefixes) with pre-parametrization weight norm
(``weight_g`` / ``weight_v`` pairs). Loading folds the weight norm into plain
convolution weights, renames the few paths where the M* modules differ from
the reference, and fails on any parameter left uninitialized or any checkpoint
tensor that found no home.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import nn

from mstar.model.loader import StackedParamRule, load_weights_into

# ALBERT's three attention projections load into one fused parameter.
QKV_STACKED_PARAMS: list[StackedParamRule] = [
    StackedParamRule(".attention.qkv", ".attention.query", "q"),
    StackedParamRule(".attention.qkv", ".attention.key", "k"),
    StackedParamRule(".attention.qkv", ".attention.value", "v"),
]

_ALBERT_LAYER = "bert.encoder.albert_layer_groups.0.albert_layers.0."
# Prefix renames, most specific first.
_RENAMES: tuple[tuple[str, str], ...] = (
    ("bert.encoder.embedding_hidden_mapping_in.", "bert.mapping_in."),
    (_ALBERT_LAYER + "ffn_output.", "bert.layer.ffn.linear_out."),
    (_ALBERT_LAYER + "ffn.", "bert.layer.ffn.linear_in."),
    (_ALBERT_LAYER, "bert.layer."),
    ("predictor.duration_proj.linear_layer.", "predictor.duration_proj."),
)
# The ALBERT pooler is trained but never used by the TTS graph.
_DROP_PREFIXES = ("bert.pooler.",)
# ``nn.LSTM(bidirectional=True)`` keys -> the two unidirectional LSTMs of MaskedBiLSTM.
_LSTM_KEY = re.compile(r"^(?P<module>.*)\.(?P<param>weight_ih|weight_hh|bias_ih|bias_hh)_l0(?P<reverse>_reverse)?$")


def fold_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``weight = g * v / ||v||`` with the norm over all dims but the first
    (``torch.nn.utils.weight_norm`` with ``dim=0``)."""
    norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
    return v * (g / norm)


def remap_name(name: str) -> str | None:
    """Checkpoint path -> M* parameter path; ``None`` drops the tensor."""
    if name.startswith(_DROP_PREFIXES):
        return None
    for old, new in _RENAMES:
        if name.startswith(old):
            name = new + name[len(old) :]
            break
    match = _LSTM_KEY.match(name)
    if match is not None:
        direction = "bwd" if match.group("reverse") else "fwd"
        name = f"{match.group('module')}.{direction}.{match.group('param')}_l0"
    return name


def iter_kokoro_weights(path: str | Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(checkpoint_path, tensor)`` with weight norm folded."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    for section, section_state in state.items():
        pending: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in section_state.items():
            name = f"{section}.{key.removeprefix('module.')}"
            base, _, suffix = name.rpartition(".")
            if suffix in ("weight_g", "weight_v"):
                pending.setdefault(base, {})[suffix] = tensor
                continue
            yield name, tensor
        for base, parts in pending.items():
            if set(parts) != {"weight_g", "weight_v"}:
                raise ValueError(f"Incomplete weight-norm pair for {base}: {sorted(parts)}")
            yield f"{base}.weight", fold_weight_norm(parts["weight_g"], parts["weight_v"])


def _stacked_target(name: str) -> str:
    for rule in QKV_STACKED_PARAMS:
        if rule.source_suffix in name:
            return name.replace(rule.source_suffix, rule.target_suffix)
    return name


def load_kokoro_weights(module: nn.Module, path: str | Path) -> None:
    """Load the checkpoint at ``path`` into ``module`` and verify completeness
    in both directions."""
    params = dict(module.named_parameters())
    mapped_names: list[str] = []

    def remap(name: str) -> str | None:
        mapped = remap_name(name)
        if mapped is not None:
            mapped_names.append(mapped)
        return mapped

    loaded = load_weights_into(
        module, iter_kokoro_weights(path), stacked_params=QKV_STACKED_PARAMS, name_remapper=remap
    )
    missing = sorted(set(params) - loaded)
    if missing:
        raise RuntimeError(f"Kokoro checkpoint left {len(missing)} parameters uninitialized: {missing[:8]}")
    unmatched = sorted(name for name in mapped_names if _stacked_target(name) not in params)
    if unmatched:
        raise RuntimeError(f"Kokoro checkpoint has {len(unmatched)} tensors with no parameter: {unmatched[:8]}")
