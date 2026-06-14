"""Weight loading for the Cosmos3 generator backbone.

The published checkpoint is the diffusers ``transformer/`` layout: flat
``layers.N.*`` keys with unfused attention projections (``to_q/to_k/to_v`` for
the understanding pathway, ``add_q_proj/add_k_proj/add_v_proj`` for the
generation pathway) and ``_moe_gen``-suffixed GEN MLP/norms. Our backbone
module mirrors that layout one-to-one, so loading needs no key remapping and
no stacked-parameter fusion — only the unused text ``lm_head`` is dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

# Checkpoint keys deliberately not loaded into the generator backbone. The
# text ``lm_head`` exists in the checkpoint (the understanding tower descends
# from a text LM) but is never used: generation emits flow velocity via
# ``proj_out``, so we do not build or load it.
DROP_KEYS: frozenset[str] = frozenset({"lm_head.weight"})


def cosmos3_name_remapper(name: str) -> str | None:
    """Map a checkpoint key to a backbone parameter path, or ``None`` to drop.

    Identity for every key the backbone owns; ``None`` for the intentional
    drop-list. Kept explicit so an unexpected checkpoint key surfaces as a
    coverage failure rather than being silently ignored.
    """
    if name in DROP_KEYS:
        return None
    return name


def read_transformer_weight_keys(checkpoint_dir: str | Path) -> set[str]:
    """Return every tensor key declared by the ``transformer/`` shard index."""
    tdir = Path(checkpoint_dir) / "transformer"
    index = tdir / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            return set(json.load(f)["weight_map"].keys())
    # Single-shard fallback: read tensor names from the safetensors header.
    shards = list(tdir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no transformer weights found under {tdir}")
    from safetensors import safe_open

    keys: set[str] = set()
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            keys.update(handle.keys())
    return keys


def load_transformer_weights(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    device: str = "cpu",
) -> set[str]:
    """Stream the ``transformer/`` shards into ``model`` and return loaded keys.

    Mirrors the meta-device + ``load_hf_weights`` path the other model packages
    use. No stacked-parameter rules: the checkpoint's projections are unfused
    and match the backbone parameter names directly.
    """
    from mstar.model.loader import load_hf_weights
    from mstar.model.loader.iterators import iter_safetensors_shards

    weights = iter_safetensors_shards(
        Path(checkpoint_dir) / "transformer", device=device
    )
    return load_hf_weights(model, weights, name_remapper=cosmos3_name_remapper)
