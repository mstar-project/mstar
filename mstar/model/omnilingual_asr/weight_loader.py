"""Fetch and load Meta's Omnilingual ASR CTC checkpoints, with no fairseq2.

A checkpoint is a plain ``{"model": state_dict, "fs2": True}`` pickle holding
only fp32 tensors, so ``torch.load(weights_only=True)`` reads it. Its names
already match :class:`Wav2Vec2CTC`; the one conversion is the positional
convolution's weight norm (``weight_g``, ``weight_v``), folded into a plain
weight with the same ``torch._weight_norm`` op the reference's hook runs on
every forward, in the serving dtype, as the hook does.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping
from pathlib import Path

import torch

from mstar.model.omnilingual_asr.config import (
    CDN_CHECKPOINTS,
    CDN_TOKENIZER,
    HUB_CHECKPOINTS,
    HUB_TOKENIZER,
    Artifact,
)

logger = logging.getLogger(__name__)

POS_CONV = "encoder_frontend.pos_encoder.conv."


def default_cache_dir() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "mstar" / "omnilingual_asr"


def sha256_of(path: str | Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _cache_root(cache_dir: str | Path | None) -> Path:
    root = Path(cache_dir) if cache_dir else default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def verify_sha256(path: str | Path, expected: str, cache_dir: str | Path | None) -> None:
    """Raise unless ``path`` hashes to ``expected``. The hash of a multi-GB
    checkpoint takes a while, so a match is remembered in a marker keyed by
    the hash and the file's inode, size and times; a rewritten or replaced
    file is hashed again."""
    path = Path(path).resolve()
    stat = path.stat()
    stamp = f"{stat.st_ino} {stat.st_size} {stat.st_mtime_ns} {stat.st_ctime_ns} {path}"
    marker = _cache_root(cache_dir) / "verified" / expected
    if marker.exists() and stamp in marker.read_text().splitlines():
        return
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(
            f"{path} has sha256 {actual}, not the pinned {expected}; "
            "delete it to download it again, or pass a local file with checkpoint_path/tokenizer_path"
        )
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        with open(marker, "a") as f:
            f.write(stamp + "\n")
    except OSError as exc:
        # a read-only cache (a mounted volume) only costs the hash on each start
        logger.warning("cannot record the verified hash of %s (%s); it is hashed again next time", path, exc)


def fetch_cdn(artifact: Artifact, cache_dir: str | Path | None) -> Path:
    """The artifact's cached path, downloading it first if needed, verified
    against its pinned sha256 either way. ``download_url_to_file`` writes to a
    temporary file and renames it only once the hash matches."""
    path = _cache_root(cache_dir) / artifact.filename
    if not path.exists():
        logger.info("Downloading %s to %s", artifact.url, path)
        torch.hub.download_url_to_file(artifact.url, str(path), hash_prefix=artifact.sha256, progress=False)
    verify_sha256(path, artifact.sha256, cache_dir)
    return path


def fetch_hub(repo_id: str, filename: str, revision: str, sha256: str, cache_dir: str | Path | None) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id, filename, revision=revision, cache_dir=cache_dir))
    verify_sha256(path, sha256, cache_dir)
    return path


def resolve_tokenizer(version: str, cache_dir: str | Path | None, tokenizer_path: str | None = None) -> Path:
    """A local ``tokenizer_path`` is used as given; the shipped tokenizers are verified."""
    if tokenizer_path:
        return Path(tokenizer_path)
    if version == "v2":
        return fetch_cdn(CDN_TOKENIZER, cache_dir)
    hub = HUB_CHECKPOINTS["300m"]
    return fetch_hub(hub.repo_id, HUB_TOKENIZER[0], hub.revision, HUB_TOKENIZER[1], cache_dir)


def resolve_checkpoint(
    size: str,
    version: str,
    cache_dir: str | Path | None,
    checkpoint_path: str | None = None,
) -> Path:
    """A local ``checkpoint_path`` is used as given; the shipped checkpoints are verified."""
    if checkpoint_path:
        return Path(checkpoint_path)
    if version == "v2":
        return fetch_cdn(CDN_CHECKPOINTS[size], cache_dir)
    hub = HUB_CHECKPOINTS[size]
    return fetch_hub(hub.repo_id, hub.filename, hub.revision, hub.sha256, cache_dir)


def read_checkpoint(path: str | Path) -> Mapping[str, torch.Tensor]:
    """The checkpoint's state dict, memory-mapped (nothing is read until used)."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    state = ckpt.get("model") if isinstance(ckpt, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError(f"{path} is not an Omnilingual ASR checkpoint (no 'model' state dict)")
    return state


def convert_state_dict(
    state: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Checkpoint names -> :class:`Wav2Vec2CTC` names and tensors in ``dtype``
    on ``device``, with the positional conv's weight norm folded."""
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name in (POS_CONV + "weight_g", POS_CONV + "weight_v"):
            continue
        out[name] = tensor.to(device=device, dtype=dtype)
    if POS_CONV + "weight_v" in state:
        g = state[POS_CONV + "weight_g"].to(device=device, dtype=dtype)
        v = state[POS_CONV + "weight_v"].to(device=device, dtype=dtype)
        # dim=2: the legacy weight_norm(conv, dim=2) fairseq2 registers
        out[POS_CONV + "weight"] = torch._weight_norm(v, g, 2)
    return out


def load_into(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Install ``state`` as ``model``'s parameters (the model may be on the meta
    device: the tensors are assigned, not copied). Every parameter must be
    covered and every tensor used, with matching shapes."""
    params = dict(model.named_parameters())
    missing = sorted(set(params) - set(state))
    unexpected = sorted(set(state) - set(params))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match the model: missing {missing[:5]}{'...' if len(missing) > 5 else ''}, "
            f"unexpected {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    for name, param in params.items():
        if state[name].shape != param.shape:
            raise ValueError(f"{name}: checkpoint shape {tuple(state[name].shape)} != model shape {tuple(param.shape)}")
    model.load_state_dict(state, strict=True, assign=True)
