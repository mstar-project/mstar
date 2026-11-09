"""Weight loading for the Cosmos3 generator backbone and the Edge reasoner's
vision tower.

The published checkpoint is the diffusers ``transformer/`` layout: flat
``layers.N.*`` keys with unfused attention projections (``to_q/to_k/to_v`` for
the understanding pathway, ``add_q_proj/add_k_proj/add_v_proj`` for the
generation pathway) and ``_moe_gen``-suffixed GEN MLP/norms. Our backbone
module mirrors that layout one-to-one, except that the understanding
pathway's q, k and v load into one fused ``to_qkv`` (``TRANSFORMER_STACKED``).
The text ``lm_head`` is dropped for generators that never decode text and
loaded for checkpoints served as a reasoner.

Edge checkpoints add ``vision_encoder/model.safetensors`` (the SigLIP2-style
tower under ``model.visual.*`` and the patch-merger projector under
``model.projector.*``); ``load_vision_encoder_weights`` streams it into the
reasoner's vision module, whose submodules are named ``visual`` and
``projector`` so only the ``model.`` prefix is stripped.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mstar.model.loader.base import StackedParamRule

# Checkpoint keys deliberately not loaded into a generator-only backbone. The
# text ``lm_head`` exists in every checkpoint (the understanding tower
# descends from a text LM) but generation emits flow velocity via
# ``proj_out``; only a backbone that also serves the reasoner builds and loads
# it (see ``cosmos3_name_remapper``).
DROP_KEYS: frozenset[str] = frozenset({"lm_head.weight"})

# The understanding pathway's projections, fused into one GEMM.
TRANSFORMER_STACKED = [
    StackedParamRule(".to_qkv", ".to_q", "q"),
    StackedParamRule(".to_qkv", ".to_k", "k"),
    StackedParamRule(".to_qkv", ".to_v", "v"),
]


# The vision tower's q / k / v load into one fused ``qkv_proj`` a layer, as
# the transformer's do.
VISION_STACKED = [
    StackedParamRule(".qkv_proj", ".q_proj", "q"),
    StackedParamRule(".qkv_proj", ".k_proj", "k"),
    StackedParamRule(".qkv_proj", ".v_proj", "v"),
]


def stacked_param_name(name: str, rules: list[StackedParamRule] = TRANSFORMER_STACKED) -> str:
    """The parameter a checkpoint key loads into: its fused one for a q / k /
    v projection, else the key itself."""
    for rule in rules:
        marker = rule.source_suffix + "."
        if marker in name:
            return name.replace(marker, rule.target_suffix + ".")
    return name


def stacked_param_shapes(
    shapes: dict[str, tuple[int, ...]], rules: list[StackedParamRule] = TRANSFORMER_STACKED,
) -> dict[str, tuple[int, ...]]:
    """Checkpoint tensor shapes as the module's parameters hold them: a fused
    projection's rows are its shards' rows together."""
    out: dict[str, tuple[int, ...]] = {}
    for name, shape in shapes.items():
        target = stacked_param_name(name, rules)
        if target == name:
            out[name] = tuple(shape)
        elif target in out:
            out[target] = (out[target][0] + shape[0], *shape[1:])
        else:
            out[target] = tuple(shape)
    return out


# The Edge ``vision_encoder/`` shard's key prefix; the reasoner's vision
# module keeps the ``visual`` / ``projector`` names below it.
VISION_ENCODER_PREFIX = "model."


def cosmos3_name_remapper(name: str, with_lm_head: bool = False) -> str | None:
    """Map a checkpoint key to a backbone parameter path, or ``None`` to drop.

    Identity for every key the backbone owns; ``None`` for the intentional
    drop-list (``lm_head.weight`` unless the backbone serves the reasoner).
    Kept explicit so an unexpected checkpoint key surfaces as a coverage
    failure rather than being silently ignored.
    """
    if name in DROP_KEYS and not with_lm_head:
        return None
    return name


def vision_encoder_name_remapper(name: str) -> str | None:
    """``model.visual.*`` / ``model.projector.*`` -> ``visual.*`` / ``projector.*``."""
    if name.startswith(VISION_ENCODER_PREFIX):
        return name[len(VISION_ENCODER_PREFIX):]
    return None


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
    index (the generic iterator only knows the ``model.safetensors`` name). The
    checkpoint's q / k / v projections load into the fused ``to_qkv``
    (``TRANSFORMER_STACKED``); every other key matches its backbone parameter
    directly. Raises if any backbone parameter is
    left unfilled — the completeness guarantee bagel's loader also enforces.
    """
    from mstar.model.loader import iter_safetensors_file, load_hf_weights

    tdir = Path(checkpoint_dir) / "transformer"
    shard_names = _transformer_shard_names(tdir)

    def _weights():
        for shard in shard_names:
            yield from iter_safetensors_file(tdir / shard, device=device)

    with_lm_head = hasattr(model, "lm_head")
    loaded = load_hf_weights(
        model, _weights(), stacked_params=TRANSFORMER_STACKED,
        name_remapper=lambda name: cosmos3_name_remapper(name, with_lm_head=with_lm_head),
    )

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


def read_vision_encoder_weight_shapes(checkpoint_dir: str | Path) -> dict[str, tuple[int, ...]]:
    """``{parameter: shape}`` of ``vision_encoder/model.safetensors`` as the
    vision module holds it (q / k / v fused), from the safetensors header only
    (CPU-side shape verification of the meta-built vision module)."""
    from safetensors import safe_open

    path = Path(checkpoint_dir) / "vision_encoder" / "model.safetensors"
    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(path, framework="pt") as handle:
        for key in handle.keys():
            mapped = vision_encoder_name_remapper(key)
            if mapped is not None:
                shapes[mapped] = tuple(handle.get_slice(key).get_shape())
    return stacked_param_shapes(shapes, VISION_STACKED)


def load_vision_encoder_weights(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    device: str = "cpu",
) -> set[str]:
    """Stream ``vision_encoder/model.safetensors`` into the reasoner's vision
    module (tower + projector) and return the loaded keys, raising if any
    parameter is left unfilled."""
    from mstar.model.loader import iter_safetensors_file, load_hf_weights

    path = Path(checkpoint_dir) / "vision_encoder" / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"no vision encoder weights under {path}")
    loaded = load_hf_weights(
        model, iter_safetensors_file(path, device=device), stacked_params=VISION_STACKED,
        name_remapper=vision_encoder_name_remapper,
    )
    expected = set(dict(model.named_parameters()).keys())
    missing = expected - loaded
    if missing:
        sample = sorted(missing)[:10]
        more = "…" if len(missing) > 10 else ""
        raise KeyError(
            f"Cosmos3 vision encoder load left {len(missing)} parameter(s) unfilled "
            f"from {path}: {sample}{more}"
        )
    return loaded
