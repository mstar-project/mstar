"""Where the Phi-4-multimodal tests find the checkpoint and the reference recordings.

``snapshot()`` is the cached checkpoint (None when it isn't cached: the weight tests
skip). ``ORACLE`` is the reference recordings' directory, ``$PHI4MM_ORACLE`` (written by
``test/phi4mm/record*.py``; None when unset: the parity tests skip).
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

REPO = "microsoft/Phi-4-multimodal-instruct"
ORACLE = Path(os.environ["PHI4MM_ORACLE"]) if os.environ.get("PHI4MM_ORACLE") else None


@cache
def snapshot(full: bool = False) -> Path | None:
    """The cached snapshot; with ``full``, only if its weights are cached too."""
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(REPO, allow_patterns=["config.json"], local_files_only=True))
    except Exception:  # noqa: BLE001 — not cached
        return None
    if full and not (path / "model.safetensors.index.json").exists():
        return None
    return path
