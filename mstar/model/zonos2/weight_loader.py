"""Checkpoint loading for Zonos2 (``params.json`` and ``model.pth``).

A Zonos2 release checkpoint is a directory. It holds:

* ``params.json`` — the model and training config. See
  :func:`mstar.model.zonos2.config.load_zonos2_config`.
* ``model.pth`` (or ``model.pt``, or ``consolidated/consolidated.pth``) — a
  torch ``state_dict``. It can be nested under a ``"model"`` key.

This is not the HuggingFace safetensors layout that the generic
``mstar.model.loader`` handles, so Zonos2 loads its own checkpoint here. The
code is adapted from ``../ZONOS2/python/zonos2/models/weight.py`` and
``utils/hf.py``. The ``weight_loader`` hook of each parameter handles the
tensor-parallel sharding, and this module gives those hooks the full checkpoint
tensors.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import torch

from mstar.model.zonos2.config import Zonos2Config, load_zonos2_config

logger = logging.getLogger(__name__)

_HF_REPO_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
_STATE_DICT_NAMES = ("model.pth", "model.pt", "consolidated/consolidated.pth")


@lru_cache()
def resolve_zonos2_checkpoint(model_path: str, cache_dir: str | None = None) -> str:
    """Return a local checkpoint directory.

    The function downloads an HF repo id if necessary.
    """
    if Path(model_path).expanduser().exists():
        return str(Path(model_path).expanduser())
    if _HF_REPO_ID_RE.match(model_path):
        from huggingface_hub import snapshot_download

        logger.info("Zonos2: downloading checkpoint from Hugging Face: %s", model_path)
        return snapshot_download(
            model_path,
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.pth", "*.pt", "*.yaml"],
        )
    return model_path


def _read_params_json(checkpoint_dir: str) -> dict | None:
    params_json = Path(checkpoint_dir) / "params.json"
    if params_json.exists():
        with open(params_json) as f:
            return json.load(f)
    return None


def _find_state_dict_file(checkpoint_dir: str) -> Path | None:
    path = Path(checkpoint_dir)
    if path.is_file() and path.suffix in (".pt", ".pth"):
        return path
    for name in _STATE_DICT_NAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def load_zonos2_state_dict(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    """Load the raw torch ``state_dict`` on the CPU. Unwrap a ``"model"`` key."""
    sd_file = _find_state_dict_file(checkpoint_dir)
    if sd_file is None:
        raise FileNotFoundError(
            f"No Zonos2 checkpoint ({' / '.join(_STATE_DICT_NAMES)}) under {checkpoint_dir!r}."
        )
    logger.info("Zonos2: loading weights from %s", sd_file)
    state = torch.load(str(sd_file), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return state


def normalize_zonos2_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove the training keys that inference does not need.

    This agrees with the reference:

    * Unwrap the weight-norm reparametrization.
      ``x.parametrizations.w.original`` becomes ``x.w``.
    * Drop the training-only router stats: ``router.ent_denom`` and
      ``router.normalized_entropy``.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if ".router.ent_denom" in key or ".router.normalized_entropy" in key:
            continue
        if ".parametrizations." in key and ".original" in key:
            key = key.replace(".parametrizations.", ".").replace(".original", "")
        out[key] = value
    return out


def load_zonos2_config_from_checkpoint(
    checkpoint_dir: str, **overrides,
) -> Zonos2Config:
    """Read ``params.json`` into a :class:`Zonos2Config`. ``overrides`` win.

    Some training-run checkpoints have no ``text_vocab`` in ``params.json``.
    The code then infers it from the row count of the text embedder.
    """
    params = _read_params_json(checkpoint_dir)
    if params is None:
        raise FileNotFoundError(f"No params.json under {checkpoint_dir!r}.")
    cfg = load_zonos2_config(params, **overrides)

    if cfg.text_vocab is None:
        inferred = _infer_text_vocab(checkpoint_dir, cfg.n_codebooks)
        if inferred is not None:
            logger.info("Zonos2: inferred text_vocab=%d from checkpoint", inferred)
            cfg.text_vocab = inferred
        else:
            logger.warning("Zonos2: text_vocab unknown; text embedding disabled.")
    return cfg


def _infer_text_vocab(checkpoint_dir: str, n_codebooks: int) -> int | None:
    """Infer ``text_vocab`` from the text embedder.

    The text embedder is ``multi_embedder.embedders.{n_codebooks}``, and it has
    ``text_vocab + 1`` rows.
    """
    try:
        sd = load_zonos2_state_dict(checkpoint_dir)
    except FileNotFoundError:
        return None
    w = sd.get(f"multi_embedder.embedders.{n_codebooks}.weight")
    return int(w.shape[0]) - 1 if w is not None else None


def load_zonos2_weights(
    model: torch.nn.Module, checkpoint_dir: str, device: torch.device | str = "cpu",
) -> set[str]:
    """Load and normalize the checkpoint into ``model`` through ``load_weights``.

    The code streams the tensors from the CPU. The ``load_weights`` method of
    the model copies each one into its parameters, which are already on the
    device. Peak VRAM therefore stays at about 1x.
    """
    state_dict = normalize_zonos2_state_dict(load_zonos2_state_dict(checkpoint_dir))
    loaded = model.load_weights(state_dict.items())

    total = sum(1 for _ in model.named_parameters()) + sum(1 for _ in model.named_buffers())
    logger.info("Zonos2: loaded %d/%d model tensors from checkpoint", len(loaded), total)
    if len(loaded) < total:
        missing = (
            set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
        ) - loaded
        logger.warning(
            "Zonos2: %d model tensors not found in checkpoint (first few: %s)",
            len(missing), sorted(missing)[:8],
        )
    return loaded
