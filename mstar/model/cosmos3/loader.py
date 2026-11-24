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


def _resolve_shard_files(path: str | Path, *, allow_glob_fallback: bool = False) -> list[Path]:
    from mstar.model.loader.reload import safetensors_files
    from mstar.model.reload_errors import WeightReloadRefusedError

    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"no weights found under {p}")
    try:
        return safetensors_files(p)
    except WeightReloadRefusedError as exc:
        if "no model.safetensors" in str(exc):
            if allow_glob_fallback:
                shards = sorted(p.glob("*.safetensors"))
                if shards:
                    return shards
            raise FileNotFoundError(f"no weights found under {p}") from exc
        raise


def _verify_complete_load(
    expected: set[str],
    loaded: set[str],
    stacked_rules: list[StackedParamRule],
    got_shards: dict[str, set[str | int]],
    component: str,
    source: str | Path,
) -> None:
    missing = set(expected - loaded)
    for target, seen in got_shards.items():
        needed = {r.shard_id for r in stacked_rules if r.target_suffix in target}
        missing_shards = needed - seen
        if missing_shards:
            missing.add(f"{target} (missing shards {sorted(missing_shards)})")
    if missing:
        sample = sorted(missing)[:10]
        more = "…" if len(missing) > 10 else ""
        raise KeyError(
            f"Cosmos3 {component} load left {len(missing)} parameter(s) unfilled "
            f"from {source}: {sample}{more}"
        )


def load_transformer_path(
    model: torch.nn.Module,
    path: str | Path,
    device: str = "cpu",
) -> set[str]:
    """Stream transformer weights from ``path`` (a ``transformer/`` directory,
    HF/diffusers shard directory, or direct ``.safetensors`` file) into
    ``model`` and verify full parameter + ``TRANSFORMER_STACKED`` shard
    completeness."""
    from mstar.model.loader import StackedMatcher, iter_safetensors_file, load_hf_weights

    shards = _resolve_shard_files(path, allow_glob_fallback=True)
    with_lm_head = hasattr(model, "lm_head")
    expected = set(dict(model.named_parameters()).keys())
    match_stacked = StackedMatcher(TRANSFORMER_STACKED)
    got_shards: dict[str, set[str | int]] = {}

    def _weights():
        for shard in shards:
            for key, tensor in iter_safetensors_file(shard, device=device):
                mapped = cosmos3_name_remapper(key, with_lm_head=with_lm_head)
                if mapped is not None:
                    target, shard_id = match_stacked(mapped)
                    if shard_id is not None and target in expected:
                        got_shards.setdefault(target, set()).add(shard_id)
                yield key, tensor

    loaded = load_hf_weights(
        model, _weights(), stacked_params=TRANSFORMER_STACKED,
        name_remapper=lambda name: cosmos3_name_remapper(name, with_lm_head=with_lm_head),
    )
    _verify_complete_load(expected, loaded, TRANSFORMER_STACKED, got_shards, "transformer", path)
    return loaded


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
    return load_transformer_path(model, Path(checkpoint_dir) / "transformer", device=device)


def read_vision_encoder_weight_shapes(checkpoint_dir: str | Path) -> dict[str, tuple[int, ...]]:
    """``{parameter: shape}`` of ``vision_encoder/`` as the vision module
    holds it (q / k / v fused), from the safetensors header only (CPU-side
    shape verification of the meta-built vision module)."""
    from safetensors import safe_open

    shards = _resolve_shard_files(Path(checkpoint_dir) / "vision_encoder")
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                mapped = vision_encoder_name_remapper(key)
                if mapped is not None:
                    shapes[mapped] = tuple(handle.get_slice(key).get_shape())
    return stacked_param_shapes(shapes, VISION_STACKED)


def load_vision_encoder_path(
    model: torch.nn.Module,
    path: str | Path,
    device: str = "cpu",
) -> set[str]:
    """Stream ``vision_encoder`` weights from ``path`` into the reasoner's
    vision module (tower + projector) and return the loaded keys, raising if
    any parameter or stacked shard is left unfilled."""
    from mstar.model.loader import StackedMatcher, iter_safetensors_file, load_hf_weights

    shards = _resolve_shard_files(path)
    expected = set(dict(model.named_parameters()).keys())
    match_stacked = StackedMatcher(VISION_STACKED)
    got_shards: dict[str, set[str | int]] = {}

    def _weights():
        for shard in shards:
            for key, tensor in iter_safetensors_file(shard, device=device):
                mapped = vision_encoder_name_remapper(key)
                if mapped is not None:
                    target, shard_id = match_stacked(mapped)
                    if shard_id is not None and target in expected:
                        got_shards.setdefault(target, set()).add(shard_id)
                yield key, tensor

    loaded = load_hf_weights(
        model, _weights(), stacked_params=VISION_STACKED,
        name_remapper=vision_encoder_name_remapper,
    )
    _verify_complete_load(expected, loaded, VISION_STACKED, got_shards, "vision encoder", path)
    return loaded


def load_vision_encoder_weights(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    device: str = "cpu",
) -> set[str]:
    """Stream ``vision_encoder/model.safetensors`` into the reasoner's vision
    module (tower + projector) and return the loaded keys, raising if any
    parameter is left unfilled."""
    return load_vision_encoder_path(model, Path(checkpoint_dir) / "vision_encoder", device=device)


def load_vae_path(
    vae: torch.nn.Module,
    path: str | Path,
    device: str = "cpu",
) -> set[str]:
    """Stream Wan VAE weights from ``path`` into ``vae`` in place, preserving
    ``channels_last_3d`` / ``channels_last`` layouts and decode dtypes."""
    from mstar.model.loader import iter_safetensors_file, load_weights_into

    shards = _resolve_shard_files(path, allow_glob_fallback=True)
    drop_encoder = getattr(vae, "encoder", None) is None

    def _weights():
        for shard in shards:
            yield from iter_safetensors_file(shard, device=device)

    remapper = (lambda k: None if k.startswith("encoder.") else k) if drop_encoder else None
    loaded = load_weights_into(vae, _weights(), name_remapper=remapper)
    expected = set(dict(vae.named_parameters()).keys())
    _verify_complete_load(expected, loaded, [], {}, "VAE", path)
    return loaded


def load_vae_weights(
    vae: torch.nn.Module,
    checkpoint_dir: str | Path,
    device: str = "cpu",
) -> set[str]:
    """Stream ``vae/`` weights into ``vae`` in place."""
    return load_vae_path(vae, Path(checkpoint_dir) / "vae", device=device)


def load_sound_tokenizer_path(
    model: torch.nn.Module,
    path: str | Path,
    device: str = "cpu",
) -> set[str]:
    """Stream ``sound_tokenizer`` decoder weights from ``path`` into ``model``
    in place, ignoring ``encoder.*`` keys and preserving ``weight_norm``
    ``weight_g``/``weight_v`` storage."""
    from mstar.model.loader import iter_safetensors_file, load_weights_into

    shards = _resolve_shard_files(path, allow_glob_fallback=True)
    expected = set(dict(model.named_parameters()).keys())

    def _weights():
        for shard in shards:
            yield from iter_safetensors_file(shard, device=device)

    loaded = load_weights_into(
        model, _weights(), name_remapper=lambda k: k if k in expected else None,
    )
    _verify_complete_load(expected, loaded, [], {}, "sound tokenizer", path)
    return loaded


def load_sound_tokenizer_weights(
    model: torch.nn.Module,
    checkpoint_dir: str | Path,
    device: str = "cpu",
) -> set[str]:
    """Stream ``sound_tokenizer/`` weights into ``model`` in place."""
    return load_sound_tokenizer_path(model, Path(checkpoint_dir) / "sound_tokenizer", device=device)
