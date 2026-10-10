"""Resolving a Hugging Face repo id to a local snapshot directory.

Not model-specific: every model that reads a pipeline snapshot's component
configs needs it, so it lives here rather than in one model's config module.
"""

from __future__ import annotations

from pathlib import Path


def resolve_snapshot_dir(model_path_hf: str, cache_dir: str | None = None) -> Path:
    """Local snapshot directory of a pipeline repo: a local path is used as-is,
    a hub id resolves through the HF cache (configs + weights of every
    component, offline when ``HF_HUB_OFFLINE`` is set)."""
    local = Path(model_path_hf)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path_hf, cache_dir=cache_dir))
