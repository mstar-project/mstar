"""Load V-JEPA 2 checkpoints into our component modules.

Two checkpoint families are supported:

* **HF safetensors** (``facebook/vjepa2-{vitl,vith,vitg,vitg-384}-fpc64-*``) —
  top-level ``encoder.*`` / ``predictor.*`` key layout; our component classes
  match, so :func:`load_vjepa2_hf_weights` just streams each prefix slice
  through :func:`mminf.model.loader.load_hf_weights` with a thin remapper
  that strips the outer ``encoder.``/``predictor.`` prefix.

* **Upstream ``.pt``** (``facebook/vjepa2-ac-vitg/original/model.pth`` or the
  raw S3 mirror at ``https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt``)
  — a single ``torch.save`` blob ``{"encoder": ..., "predictor": ...}`` where
  the encoder uses the upstream ``module.backbone.*`` layout (fused ``qkv``,
  ``blocks.X.``, top-level ``norm.``) and the AC predictor uses the upstream
  ``module.*`` layout (fused ``qkv``, ``predictor_blocks.X.``).

  The AC predictor's class in this repo was ported *to* the upstream key
  layout deliberately, so its side of the rename is a trivial
  ``module.``/``backbone.`` strip.  The encoder is HF-keyed in our tree, so
  we apply the same renames the HF conversion script does
  (``transformers/.../convert_vjepa2_to_hf.py::convert_encoder_keys``).  The
  encoder rename also splits the fused ``qkv`` weight/bias into separate
  ``query`` / ``key`` / ``value`` projections (our :class:`VJEPA2Encoder`
  uses the HF-style separated layout).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import torch

from mminf.model.loader import (
    iter_safetensors_shards,
    load_hf_weights,
)

logger = logging.getLogger(__name__)


# Public S3 mirror of the upstream V-JEPA 2-AC checkpoint (same file HF's
# ``upload_original_ckpts`` in ``convert_vjepa2_to_hf.py`` pushes under
# ``original/model.pth``).  ~11.7 GB, no auth required.
VJEPA2_AC_VITG_S3_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"


def _strip_prefix_remapper(prefix: str) -> Callable[[str], str | None]:
    """Return a ``load_hf_weights`` ``name_remapper`` that strips ``prefix.``
    from each key (returning ``None`` for non-matching keys so they're
    dropped).  Used to peel the outer ``encoder.``/``predictor.`` envelope
    off HF V-JEPA 2 safetensors keys.
    """
    pref = prefix if prefix.endswith(".") else prefix + "."

    def remap(name: str) -> str | None:
        if name.startswith(pref):
            return name[len(pref):]
        return None

    return remap


def _assert_no_missing(
    expected: set[str], loaded: set[str], context: str,
) -> None:
    missing = expected - loaded
    if missing:
        sample = sorted(missing)[:10]
        more = "…" if len(missing) > 10 else ""
        raise KeyError(
            f"Missing {len(missing)} keys when loading {context}: "
            f"{sample}{more}"
        )


def load_vjepa2_hf_weights(
    repo_dir: str | Path,
    encoder_module: torch.nn.Module | None,
    predictor_module: torch.nn.Module | None,
    device: str = "cpu",
) -> None:
    """Load HF V-JEPA 2 weights into the supplied encoder / predictor modules.

    Pass ``None`` for a module you don't want to load (e.g. when only the
    encoder lives on this worker's GPU).  ``encoder_module.named_parameters()``
    is expected to have keys matching the HF layout *after* the ``encoder.``
    prefix is stripped; same for ``predictor_module`` and ``predictor.``.

    Raises :class:`KeyError` if any parameter of the supplied module(s) is
    missing from the checkpoint — the equivalent of the old
    ``enforce_missing_keys=True`` safety net.
    """
    repo_dir = Path(repo_dir)
    if encoder_module is None and predictor_module is None:
        logger.warning("load_vjepa2_hf_weights called with no modules to load")
        return

    for module, prefix, label in (
        (encoder_module, "encoder.", "HF V-JEPA 2 encoder"),
        (predictor_module, "predictor.", "HF V-JEPA 2 predictor"),
    ):
        if module is None:
            continue
        expected = set(dict(module.named_parameters()).keys())
        # iter_safetensors_shards transparently handles both the sharded
        # (``model.safetensors.index.json``) and single-file
        # (``model.safetensors``) layouts.
        loaded = load_hf_weights(
            module,
            iter_safetensors_shards(repo_dir, device=device, prefix=prefix),
            name_remapper=_strip_prefix_remapper(prefix),
        )
        _assert_no_missing(expected, loaded, label)


def download_vjepa2_snapshot(model_path_hf: str, cache_dir: str | None = None) -> Path:
    """Resolve the local directory for an HF V-JEPA 2 snapshot, downloading
    if absent.  Thin wrapper over :func:`huggingface_hub.snapshot_download`."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=model_path_hf, cache_dir=cache_dir)
    return Path(local)


# ----------------------------------------------------------------------
# V-JEPA 2-AC — upstream ``.pt`` loader
# ----------------------------------------------------------------------


def _rename_upstream_encoder_keys(
    state_dict: Mapping[str, torch.Tensor],
    hidden_size: int,
) -> dict[str, torch.Tensor]:
    """Translate the upstream V-JEPA 2 encoder state_dict onto our HF-keyed
    :class:`VJEPA2Encoder`.

    Mirrors the transform in
    ``transformers/src/transformers/models/vjepa2/convert_vjepa2_to_hf.py::convert_encoder_keys``,
    minus the outer ``encoder.`` prefix the HF model uses (our
    ``VJEPA2Encoder`` is already the ``encoder`` sub-module).

    Transforms applied, in order, per key:
      1. strip ``module.`` and ``backbone.`` independently (matches upstream
         ``_clean_backbone_key`` — real checkpoints use either or both
         depending on the DDP / parent-wrapper situation at save time)
      2. ``blocks.X.``               → ``layer.X.``
      3. ``attn.``                   → ``attention.``
      4. ``patch_embed.``            → ``embeddings.patch_embeddings.``
      5. top-level ``norm.``         → ``layernorm.``  (NOT block-internal norm1/norm2)
      6. split fused ``attention.qkv.{weight,bias}`` into
         ``attention.{query,key,value}.{weight,bias}`` along dim 0
      7. drop ``pos_embed`` (AC encoder is RoPE-only; upstream may still ship
         a zero-init pos_embed for compatibility but our port has no slot
         for it)

    Args:
        state_dict: upstream encoder state_dict (``state["encoder"]`` from
            the ``.pt``).
        hidden_size: encoder embedding dim — needed to slice fused qkv.

    Returns:
        A new dict with keys matching ``VJEPA2Encoder.state_dict()``.
    """
    out: dict[str, torch.Tensor] = {}
    for raw_key, val in state_dict.items():
        # Match upstream ``_clean_backbone_key`` exactly: strip both wrapping
        # prefixes independently, not as a combined ``module.backbone.`` token.
        # Real checkpoints may expose either order (``module.backbone.X``,
        # ``backbone.module.X``, or just one of them) depending on whether
        # the ``.pt`` was saved after DDP-wrap, backbone-wrap, or both.
        key = raw_key.replace("module.", "").replace("backbone.", "")
        # Drop RoPE-encoder pos_embed (our VJEPA2Encoder has no position
        # embedding slot — it uses 3D RoPE inside attention instead).
        if key == "pos_embed":
            continue
        if key.startswith("blocks."):
            key = key.replace("blocks.", "layer.", 1)
        if ".attn." in key:
            key = key.replace(".attn.", ".attention.")
        if key.startswith("patch_embed."):
            key = key.replace("patch_embed.", "embeddings.patch_embeddings.", 1)
        if key.startswith("norm."):
            key = key.replace("norm.", "layernorm.", 1)

        # Split fused qkv (upstream stacks Q/K/V along dim 0).
        if key.endswith("attention.qkv.weight"):
            prefix = key[: -len("qkv.weight")]
            q = val[0:hidden_size, :]
            k = val[hidden_size : 2 * hidden_size, :]
            v = val[2 * hidden_size : 3 * hidden_size, :]
            out[prefix + "query.weight"] = q
            out[prefix + "key.weight"] = k
            out[prefix + "value.weight"] = v
            continue
        if key.endswith("attention.qkv.bias"):
            prefix = key[: -len("qkv.bias")]
            q = val[0:hidden_size]
            k = val[hidden_size : 2 * hidden_size]
            v = val[2 * hidden_size : 3 * hidden_size]
            out[prefix + "query.bias"] = q
            out[prefix + "key.bias"] = k
            out[prefix + "value.bias"] = v
            continue

        out[key] = val
    return out


def _rename_upstream_ac_predictor_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Translate the upstream V-JEPA 2-AC predictor state_dict onto our
    :class:`VisionTransformerPredictorAC`.

    Upstream wraps the module under DDP (``module.``) and sometimes under an
    extra ``backbone.`` attribute depending on how the checkpoint was saved.
    Since our AC predictor class was deliberately ported to preserve upstream
    internal naming (fused ``qkv``, ``predictor_blocks.X.``,
    ``predictor_embed``, etc.), the translation is just a prefix strip —
    matches ``_clean_backbone_key`` in ``vjepa2/src/hub/backbones.py``.
    """
    out: dict[str, torch.Tensor] = {}
    for raw_key, val in state_dict.items():
        key = raw_key.replace("module.", "").replace("backbone.", "")
        out[key] = val
    return out


def download_vjepa2_ac_upstream_pt(
    model_path_hf: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Fetch the upstream V-JEPA 2-AC ``.pt`` from the public S3 mirror.

    The HuggingFace V-JEPA 2 collection does not include an
    AC-variant repo (only the base + SSv2 / Diving-48 classification
    checkpoints for vitl/h/g are published there).  So we go straight to
    :data:`VJEPA2_AC_VITG_S3_URL` — the public
    ``dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`` artifact that
    upstream ``vjepa2_ac_vit_giant(pretrained=True)`` pulls from.

    Cached at ``{cache_dir}/vjepa2-ac-vitg.pt`` so re-launches skip the
    ~11.7 GB download.  The ``model_path_hf`` argument is kept in the
    signature for symmetry with :func:`download_vjepa2_snapshot` but is
    ignored (we don't need an HF repo ID for the S3 path).
    """
    del model_path_hf  # not used — S3 path is unique, not keyed by repo ID.
    cache_root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "mminf_vjepa2"
    cache_root.mkdir(parents=True, exist_ok=True)

    pt_path = cache_root / "vjepa2-ac-vitg.pt"
    if pt_path.exists() and pt_path.stat().st_size > 0:
        logger.info("Using cached V-JEPA 2-AC checkpoint at %s", pt_path)
        return pt_path

    logger.info(
        "Downloading V-JEPA 2-AC checkpoint from %s to %s (~11.7 GB, be patient)",
        VJEPA2_AC_VITG_S3_URL,
        pt_path,
    )
    # ``torch.hub.download_url_to_file`` atomically writes + shows tqdm progress.
    torch.hub.download_url_to_file(VJEPA2_AC_VITG_S3_URL, str(pt_path), progress=True)
    return pt_path


def _to_device_items(
    renamed: Mapping[str, torch.Tensor], device: torch.device,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` with each tensor moved to ``device``.

    The upstream ``.pt`` is loaded into CPU RAM via ``torch.load``; we move
    tensors to the worker's GPU lazily as the loader pulls them, matching
    what the previous ``state_dict = {k: v.to(device) ...}`` bulk-copy did.
    """
    is_cpu = device.type == "cpu"
    for key, tensor in renamed.items():
        yield key, tensor if is_cpu else tensor.to(device, non_blocking=True)


def load_vjepa2_ac_upstream_weights(
    pt_path: str | Path,
    encoder_module: torch.nn.Module | None,
    predictor_module: torch.nn.Module | None,
    device: str = "cpu",
    hidden_size: int | None = None,
) -> None:
    """Load weights from an upstream V-JEPA 2-AC ``.pt`` into the supplied modules.

    Args:
        pt_path: path to ``vjepa2-ac-vitg.pt`` (or the HF-hosted copy under
            ``original/model.pth``).
        encoder_module: instance of :class:`VJEPA2Encoder` to populate, or
            ``None`` if the encoder lives on a different worker.
        predictor_module: instance of :class:`VisionTransformerPredictorAC`
            to populate, or ``None``.
        device: target device for tensors after the load.  Weights are first
            loaded into CPU RAM (via ``torch.load``) and then moved; pinning
            each rank's load to its own ``cuda:X`` here lets the OS disk
            cache amortize multi-rank reads.
        hidden_size: encoder embedding dim.  Required when
            ``encoder_module`` is not None (used to split fused qkv); ignored
            otherwise.  If None, falls back to
            ``encoder_module.config.hidden_size``.
    """
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"V-JEPA 2-AC checkpoint not found: {pt_path}")

    if encoder_module is None and predictor_module is None:
        logger.warning("load_vjepa2_ac_upstream_weights called with no modules to load")
        return

    # ``weights_only=True`` is both the modern default and protects against
    # arbitrary pickle code in the ``.pt`` (the upstream checkpoint is
    # trusted — Meta publishes it — but the flag is still best practice).
    logger.info("Loading upstream V-JEPA 2-AC checkpoint from %s", pt_path)
    blob = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or "encoder" not in blob or "predictor" not in blob:
        raise ValueError(
            f"Unexpected checkpoint structure at {pt_path}: expected a dict with "
            "'encoder' and 'predictor' keys (upstream V-JEPA 2-AC format)."
        )

    target_device = torch.device(device) if isinstance(device, str) else device

    if encoder_module is not None:
        if hidden_size is None:
            hidden_size = getattr(encoder_module.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "hidden_size must be provided (or encoder_module.config.hidden_size must exist) "
                "to split the upstream fused qkv weights."
            )
        # Debug aid for key-rename mismatches: log a few source-side sample
        # keys so a future failure shows exactly what prefixes the blob uses
        # (seen at least: ``module.backbone.X``, ``backbone.X``, ``module.X``).
        src_sample = list(blob["encoder"].keys())[:4]
        logger.info("AC encoder: %d source keys; sample: %s", len(blob["encoder"]), src_sample)
        renamed = _rename_upstream_encoder_keys(blob["encoder"], hidden_size=hidden_size)
        logger.info("AC encoder: %d renamed keys; sample: %s", len(renamed), list(renamed.keys())[:4])
        expected = set(dict(encoder_module.named_parameters()).keys())
        loaded = load_hf_weights(
            encoder_module,
            _to_device_items(renamed, target_device),
        )
        _assert_no_missing(expected, loaded, "AC V-JEPA 2 encoder")
        # Free upstream-layout encoder state once it's been remapped +
        # assigned — otherwise the ~5.8 GB original CPU blob (and the
        # ``renamed`` views into it) stay pinned in memory while we go on
        # to load the predictor.
        blob["encoder"] = None
        del renamed

    if predictor_module is not None:
        src_sample = list(blob["predictor"].keys())[:4]
        logger.info("AC predictor: %d source keys; sample: %s", len(blob["predictor"]), src_sample)
        renamed = _rename_upstream_ac_predictor_keys(blob["predictor"])
        logger.info("AC predictor: %d renamed keys; sample: %s", len(renamed), list(renamed.keys())[:4])
        expected = set(dict(predictor_module.named_parameters()).keys())
        loaded = load_hf_weights(
            predictor_module,
            _to_device_items(renamed, target_device),
        )
        _assert_no_missing(expected, loaded, "AC V-JEPA 2 predictor")
        blob["predictor"] = None
