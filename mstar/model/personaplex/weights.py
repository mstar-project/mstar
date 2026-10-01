"""PersonaPlex LM weight map and strict loading.

The native modules keep NVIDIA's ``LMModel`` parameter names and shapes
([PP] models/lm.py:279-352), so the map is a prefix route: depformer keys go
to ``depth.``, everything else to ``temporal.``. The checkpoint already holds
16 depth slots (worklog §2 R8), so NVIDIA's slot-copy patches
([PP] models/loaders.py:218-251) are not reproduced; any missing, extra,
duplicate or misshapen key is an error.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from torch import nn

from .components.depformer import Depformer
from .components.temporal import TemporalTransformer
from .config import LMArchConfig

DEPTH_PREFIXES = ("depformer_in.", "depformer_emb.", "depformer_text_emb.", "depformer.", "linears.")


def remap_lm_key(key: str) -> str:
    """NVIDIA ``LMModel`` state-dict key -> ``PersonaPlexLM`` parameter path."""
    return ("depth." if key.startswith(DEPTH_PREFIXES) else "temporal.") + key


def reference_lm_key(name: str) -> str:
    """Inverse of ``remap_lm_key``."""
    for prefix in ("temporal.", "depth."):
        if name.startswith(prefix):
            key = name.removeprefix(prefix)
            if remap_lm_key(key) == name:
                return key
    raise ValueError(f"{name} is not a PersonaPlexLM parameter path")


class PersonaPlexLM(nn.Module):
    """The temporal transformer and depformer, as one loadable unit."""

    def __init__(self, config: LMArchConfig):
        super().__init__()
        self.config = config
        self.temporal = TemporalTransformer(config)
        self.depth = Depformer(config)


def reference_manifest(lm: nn.Module) -> dict[str, list[int]]:
    """NVIDIA key -> shape for every parameter (T5 compares this to the reference)."""
    return {reference_lm_key(name): list(param.shape) for name, param in lm.named_parameters()}


def load_lm_weights(lm: PersonaPlexLM, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Load a NVIDIA-named stream strictly (``load_weights_into`` alone ignores extras)."""
    # mstar.model.loader imports safetensors eagerly; keep this module importable without it.
    from mstar.model.loader.base import load_weights_into

    params = dict(lm.named_parameters())
    seen: set[str] = set()
    unexpected: list[str] = []
    duplicates: list[str] = []

    def checked() -> Iterator[tuple[str, torch.Tensor]]:
        for key, tensor in weights:
            target = remap_lm_key(key)
            if target not in params:
                unexpected.append(key)
                continue
            if target in seen:
                duplicates.append(key)
                continue
            seen.add(target)
            if tuple(tensor.shape) != tuple(params[target].shape):
                raise ValueError(f"{key}: checkpoint shape {tuple(tensor.shape)} != {tuple(params[target].shape)}")
            yield key, tensor

    loaded = load_weights_into(lm, checked(), name_remapper=remap_lm_key)
    missing = sorted(reference_lm_key(name) for name in params.keys() - loaded)
    if unexpected or duplicates or missing:
        raise ValueError(
            f"PersonaPlex LM weights do not match: {len(missing)} missing {missing[:8]}, "
            f"{len(unexpected)} unexpected {sorted(unexpected)[:8]}, {len(duplicates)} duplicate {duplicates[:8]}"
        )


def iter_lm_checkpoint(path: str, device: torch.device | str = "cpu") -> Iterator[tuple[str, torch.Tensor]]:
    """Stream ``model.safetensors`` (lazy import keeps safetensors optional)."""
    from mstar.model.loader.iterators import iter_safetensors_file

    yield from iter_safetensors_file(path, device=device)


def build_lm(
    config: LMArchConfig, weights: Iterable[tuple[str, torch.Tensor]], *,
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16,
) -> PersonaPlexLM:
    """Materialise the LM directly in ``dtype`` on ``device`` and load it strictly.

    Built on meta and cast before ``to_empty`` so storage is allocated once in
    the target dtype (``Model.get_submodule`` contract). NVIDIA also casts every
    tensor to one dtype, norm scales included ([PP] models/loaders.py:253-260).
    """
    with torch.device("meta"):
        lm = PersonaPlexLM(config)
    lm = lm.to(dtype=dtype).to_empty(device=device)
    load_lm_weights(lm, ((key, tensor.to(device=device, dtype=dtype)) for key, tensor in weights))
    return lm.eval()
