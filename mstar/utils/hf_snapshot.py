"""Resolving a Hugging Face repo id to a local snapshot directory.

Not model-specific: every model that reads a pipeline snapshot's component
configs needs it, so it lives here rather than in one model's config module.
"""

from __future__ import annotations

from pathlib import Path

# Weights in formats mstar never loads, and training state. A download that skips only these keeps every
# small file (README, processor files, assets): an offline snapshot_download refuses a cached snapshot
# missing any file its patterns select, so a narrow allow list breaks later loads that ask for more.
OTHER_WEIGHT_FORMATS = ["*.bin", "*.pt", "*.pth", "*.msgpack", "*.h5", "*.ot", "*.onnx_data", "optimizer*"]


def resolve_snapshot_dir(
    model_path_hf: str,
    cache_dir: str | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> Path:
    """Local snapshot directory of a pipeline repo: a local path is used as-is,
    a hub id resolves through the HF cache (configs + weights of every
    component, offline when ``HF_HUB_OFFLINE`` is set).

    ``allow_patterns`` limits a download to the components the model serves, for
    repos that ship alternatives (a second transformer, a prompt enhancer) the
    deployment never loads; ``ignore_patterns`` skips files instead (``OTHER_WEIGHT_FORMATS``)."""
    local = Path(model_path_hf)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        model_path_hf, cache_dir=cache_dir, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns,
    ))
