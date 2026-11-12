"""Resolving a Hugging Face repo id to a local snapshot directory.

Not model-specific: every model that reads a pipeline snapshot's component
configs needs it, so it lives here rather than in one model's config module.
"""

from __future__ import annotations

from pathlib import Path

# What a download without allow_patterns skips, by file name, not size: weights in formats mstar never loads
# (mstar loads safetensors), and training state. Every file not matching these patterns is kept (README,
# processor files, assets): an offline snapshot_download refuses a cached snapshot missing any file its patterns
# select, so a narrow allow list breaks later loads that ask for more. A model whose assets match one (a .pt
# speaker embedding, say) passes allow_patterns.
UNSERVED_FILES = ["*.bin", "*.pt", "*.pth", "*.msgpack", "*.h5", "*.ot", "*.onnx_data", "optimizer*"]


def resolve_snapshot_dir(
    model_path_hf: str,
    cache_dir: str | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Local snapshot directory of a pipeline repo: a local path is used as-is,
    a hub id resolves through the HF cache (configs + weights of every
    component, offline when ``HF_HUB_OFFLINE`` is set).

    ``allow_patterns`` limits a download to the components the model serves, for
    repos that ship alternatives (a second transformer, a prompt enhancer) the
    deployment never loads. Without it, everything but :data:`UNSERVED_FILES` is fetched."""
    local = Path(model_path_hf)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        model_path_hf, cache_dir=cache_dir, allow_patterns=allow_patterns,
        ignore_patterns=UNSERVED_FILES if allow_patterns is None else None,
    ))
