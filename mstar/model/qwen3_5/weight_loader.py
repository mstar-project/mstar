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
# Sits at the checkpoint's root rather than under the text prefix, and only
# exists at all when the embeddings are untied — 9B and 27B, not 0.8B/2B/4B.
_LM_HEAD = "lm_head.weight"


def qwen3_5_name_remapper(name: str) -> str | None:
    if name == _LM_HEAD:
        # top level in the checkpoint and in ours, so it passes through
        return name
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
    selectors: list[dict],
    expected: set[str],
) -> set[str]:
    """Fill ``expected`` from the shards each selector picks out, or raise.

    A selector is ``prefix=`` or ``keys=`` for ``iter_safetensors_shards``, and
    only narrows *which shards get opened* — ``remapper`` is what decides what
    loads. More than one because a tower's tensors are not always under a
    single prefix: an untied ``lm_head`` sits at the checkpoint's root.

    A silently half-loaded tower produces plausible-looking garbage rather
    than an error, so an unfilled parameter has to be fatal.
    """
    loaded: set[str] = set()
    for selector in selectors:
        loaded |= load_weights_into(
            model,
            iter_safetensors_shards(Path(path), device=device, **selector),
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
    tied = model.config.tie_word_embeddings
    selectors = [{"prefix": _TEXT_PREFIX}]
    if not tied:
        # 9B and 27B untie the head, and then the checkpoint carries it at the
        # root — outside the text prefix, so it needs a pass of its own.
        selectors.append({"keys": {_LM_HEAD}})
    return _load(
        model, path, device, qwen3_5_name_remapper, selectors,
        expected={
            n for n, _ in model.named_parameters()
            # tied to embed_tokens, so the checkpoint carries no tensor for it
            # (and `named_parameters` dedupes it away besides)
            if not (tied and n == _LM_HEAD)
        },
    )


def load_qwen3_5_vision_weights(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    """Load the ViT. Returns the parameter paths that were filled."""
    return _load(
        model, path, device, qwen3_5_vision_name_remapper,
        [{"prefix": _VISION_PREFIX}],
        expected={n for n, _ in model.named_parameters()},
    )
