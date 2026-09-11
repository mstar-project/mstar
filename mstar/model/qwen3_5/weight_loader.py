"""Loading a Qwen3.5 checkpoint into this stack.

A pure rename: every text-tower tensor matches ours in shape, so nothing is
split, fused or transposed on the way in. Two rules cover all 426 of them —
the checkpoint nests the text stack under ``language_model`` and names the
gated-delta-net mixer ``linear_attn`` where ours calls both mixers
``self_attn``.

Vision (``model.visual.*``) and the MTP head (``mtp.*``) are skipped: neither
is built yet.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mstar.model.loader.base import load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards

_TEXT_PREFIX = "model.language_model."


def qwen3_5_name_remapper(name: str) -> str | None:
    if not name.startswith(_TEXT_PREFIX):
        return None
    return name.replace(_TEXT_PREFIX, "model.").replace(
        ".linear_attn.", ".self_attn."
    )


def load_qwen3_5_weights(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    """Load the text tower. Returns the parameter paths that were filled.

    Raises if any parameter went unfilled, since a silently half-loaded model
    produces plausible-looking garbage rather than an error.
    """
    loaded = load_weights_into(
        model,
        iter_safetensors_shards(Path(path), device=device),
        name_remapper=qwen3_5_name_remapper,
    )
    expected = {
        n for n, _ in model.named_parameters()
        # tied to embed_tokens, so the checkpoint carries no tensor for it
        if not (model.config.tie_word_embeddings and n == "lm_head.weight")
    }
    missing = sorted(expected - loaded)
    if missing:
        raise ValueError(
            f"{len(missing)} parameter(s) got no weight, e.g. {missing[:5]}"
        )
    return loaded
