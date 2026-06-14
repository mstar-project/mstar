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


def _transformer_shard_names(tdir: Path) -> list[str]:
    """Resolve the ``transformer/`` shard filenames.

    The diffusers checkpoint indexes its shards under
    ``diffusion_pytorch_model.safetensors.index.json`` (not the
    ``model.safetensors`` name the generic shard iterator assumes), so the
    shard list is read from that index; a single-file checkpoint is the
    fallback.
    """
    index = tdir / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            return sorted(set(json.load(f)["weight_map"].values()))
    shards = sorted(p.name for p in tdir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no transformer weights found under {tdir}")
    return shards


def read_transformer_weight_shapes(checkpoint_dir: str | Path) -> dict[str, tuple[int, ...]]:
    """Return ``{key: shape}`` for every ``transformer/`` tensor by reading only
    the safetensors headers — no tensor data is materialized. Enables CPU-side
    shape verification of the meta-built backbone against the checkpoint.
    """
    from safetensors import safe_open

    tdir = Path(checkpoint_dir) / "transformer"
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in _transformer_shard_names(tdir):
        with safe_open(tdir / shard, framework="pt") as handle:
            for key in handle.keys():
                shapes[key] = tuple(handle.get_slice(key).get_shape())
    return shapes


def load_transformer_weights(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    device: str = "cpu",
) -> set[str]:
    """Stream the ``transformer/`` shards into ``model`` and return loaded keys.

    Mirrors the meta-device + ``load_hf_weights`` path the other model packages
    use, but resolves the shard list from the diffusers ``diffusion_pytorch_model``
    index (the generic iterator only knows the ``model.safetensors`` name). No
    stacked-parameter rules: the checkpoint's projections are unfused and match
    the backbone parameter names directly. Raises if any backbone parameter is
    left unfilled — the completeness guarantee bagel's loader also enforces.
    """
    from mstar.model.loader import iter_safetensors_file, load_hf_weights

    tdir = Path(checkpoint_dir) / "transformer"
    shard_names = _transformer_shard_names(tdir)

    def _weights():
        for shard in shard_names:
            yield from iter_safetensors_file(tdir / shard, device=device)

    loaded = load_hf_weights(model, _weights(), name_remapper=cosmos3_name_remapper)

    expected = set(dict(model.named_parameters()).keys())
    missing = expected - loaded
    if missing:
        sample = sorted(missing)[:10]
        more = "…" if len(missing) > 10 else ""
        raise KeyError(
            f"Cosmos3 transformer load left {len(missing)} parameter(s) unfilled "
            f"from {tdir}: {sample}{more}"
        )
    return loaded
