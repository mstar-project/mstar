"""Loading a Qwen3.5 checkpoint into this stack.

A pure rename in both towers: every tensor matches ours in shape, so nothing
is split, fused or transposed on the way in. Two rules cover all 426 text
tensors — the checkpoint nests the text stack under ``language_model`` and
names the gated-delta-net mixer ``linear_attn`` where ours calls both mixers
``self_attn``. The 297 vision tensors need only their ``model.visual.``
prefix stripped.

The two towers load separately because they are separate nodes: a worker
holding only ``LLM`` never builds the ViT, and vice versa. The MTP head
(``mtp.*``) is skipped; it isn't built.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mstar.model.loader.base import load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards

_TEXT_PREFIX = "model.language_model."
_VISION_PREFIX = "model.visual."


def qwen3_5_name_remapper(name: str) -> str | None:
    if not name.startswith(_TEXT_PREFIX):
        return None
    return name.replace(_TEXT_PREFIX, "model.").replace(
        ".linear_attn.", ".self_attn."
    )


def qwen3_5_vision_name_remapper(name: str) -> str | None:
    if not name.startswith(_VISION_PREFIX):
        return None
    return name[len(_VISION_PREFIX):]


def _load(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str,
    remapper,
    prefix: str,
    expected: set[str],
) -> set[str]:
    """Fill ``expected`` from the shards under ``prefix``, or raise.

    A silently half-loaded tower produces plausible-looking garbage rather
    than an error, so an unfilled parameter has to be fatal.
    """
    loaded = load_weights_into(
        model,
        # `prefix` here only narrows which shards get opened; `remapper` is
        # what actually decides what loads.
        iter_safetensors_shards(Path(path), device=device, prefix=prefix),
        name_remapper=remapper,
    )
    missing = sorted(expected - loaded)
    if missing:
        raise ValueError(
            f"{len(missing)} parameter(s) got no weight, e.g. {missing[:5]}"
        )
    return loaded


def load_qwen3_5_weights(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    """Load the text tower. Returns the parameter paths that were filled."""
    return _load(
        model, path, device, qwen3_5_name_remapper, _TEXT_PREFIX,
        expected={
            n for n, _ in model.named_parameters()
            # tied to embed_tokens, so the checkpoint carries no tensor for it
            if not (model.config.tie_word_embeddings and n == "lm_head.weight")
        },
    )


def load_qwen3_5_vision_weights(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    """Load the ViT. Returns the parameter paths that were filled."""
    return _load(
        model, path, device, qwen3_5_vision_name_remapper, _VISION_PREFIX,
        expected={n for n, _ in model.named_parameters()},
    )
