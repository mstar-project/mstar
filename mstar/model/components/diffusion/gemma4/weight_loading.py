"""Checkpoint rules for the Gemma-4 prompt encoder.

A ``transformers`` Gemma-4 checkpoint keeps the text tower under
``model.language_model.``; the vision and audio towers beside it are never loaded.
The model-agnostic streaming/materialising half is ``components.diffusion.weight_loading``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mstar.model.components.diffusion.gemma4.encoder import Gemma4TextEncoder, GemmaTextConfig
from mstar.model.components.diffusion.weight_loading import iter_transformers_component, load_native, materialize

TEXT_TOWER_PREFIX = "model.language_model."


def build_text_encoder(
    cfg: GemmaTextConfig, component_dir: Path, device, dtype: torch.dtype = torch.bfloat16, *, name: str = "Gemma-4",
) -> Gemma4TextEncoder:
    """Build the encoder on ``device`` and load its weights from ``component_dir``."""
    with torch.device("meta"):
        module = Gemma4TextEncoder(cfg)
    materialize(module, dtype, device)

    def skip(key: str) -> bool:
        return not key.startswith(TEXT_TOWER_PREFIX)

    weights = iter_transformers_component(component_dir, device, skip=skip)
    return load_native(module, weights, lambda key: key[len(TEXT_TOWER_PREFIX):], f"{name} text encoder")
