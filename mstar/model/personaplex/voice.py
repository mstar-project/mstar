"""PersonaPlex ``.pt`` voice presets ([PP] models/lm.py:977-983, 1031-1061).

A preset holds ``embeddings``, the stacked ``embed_codes`` outputs of every
voice frame after the first (``[N, 1, 1, 4096]``), and ``cache``, the token
cache saved after the raw-voice pass (``[1, 17, 4]``). It holds no KV:
replaying it re-runs the temporal transformer on each embedding, then restores
the cache. Presets ship in the gated repo's ``voices.tgz``: operators either
point ``voice_dir`` at extracted files or leave it unset, in which case the
archive is fetched and extracted like NVIDIA's ``offline.py`` does.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import AUDIO_CARD, CACHE_SIZE, NUM_CODEBOOKS, TEMPORAL_DIM, TEXT_CARD, UNGENERATED_TOKEN_ID


@dataclass(frozen=True)
class VoicePreset:
    embeddings: torch.Tensor  # [N, 1, 1, D] float
    cache: torch.Tensor  # [1, 17, 4] int64

    @property
    def num_embeddings(self) -> int:
        return self.embeddings.shape[0]


VOICES_ARCHIVE = "voices.tgz"  # [PP] offline.py:125-150 (_get_voice_prompt_dir)


def extract_voice_presets(archive: str | Path) -> Path:
    """Extract ``voices.tgz`` once into ``voices/`` beside it, as [PP] offline.py:125-150 does.

    Unlike the reference, extraction uses tarfile's ``data`` filter (no absolute
    or parent paths, no links out of the tree, no devices) and goes into a
    temporary sibling that is renamed into place, so concurrent workers never
    read a partly extracted directory.
    """
    archive = Path(archive)
    target = archive.parent / "voices"
    if target.is_dir():
        return target
    staging = Path(tempfile.mkdtemp(prefix=".voices-", dir=archive.parent))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(staging, filter="data")
        extracted = staging / "voices"
        if not extracted.is_dir():
            raise RuntimeError(f"{archive.name} has no voices/ directory")
        try:
            extracted.rename(target)
        except OSError:
            if not target.is_dir():  # not merely another worker finishing first
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def resolve_voice_path(voice_dir: str | Path | None, voice: object) -> Path:
    """Map a request's ``voice`` name to a ``.pt`` file inside ``voice_dir``.

    Only bare names are accepted (``"NATF2"`` or ``"NATF2.pt"``), so a request
    cannot read outside the operator's directory.
    """
    if voice_dir is None:
        raise ValueError("PersonaPlex needs a voice preset, but this deployment has no voice_dir configured")
    if not isinstance(voice, str) or not voice:
        raise ValueError("PersonaPlex requires model_kwargs['voice'], the name of a .pt voice preset")
    name = voice if voice.endswith(".pt") else f"{voice}.pt"
    if Path(name).name != name or name.startswith("."):
        raise ValueError(f"voice must be a preset file name, not a path: {voice!r}")
    root = Path(voice_dir).resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        available = sorted(p.stem for p in root.glob("*.pt")) if root.is_dir() else []
        raise ValueError(f"unknown voice {voice!r}; available presets: {available}")
    return path


def load_voice_preset(path: str | Path, *, dim: int = TEMPORAL_DIM) -> VoicePreset:
    """Load and validate a preset without unpickling arbitrary objects."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as a 400
        raise ValueError(f"cannot load voice preset {Path(path).name}: {exc}") from exc
    if not isinstance(state, dict) or set(state) != {"embeddings", "cache"}:
        keys = sorted(state) if isinstance(state, dict) else type(state).__name__
        raise ValueError(f"voice preset must hold exactly 'embeddings' and 'cache', got {keys}")
    embeddings, cache = state["embeddings"], state["cache"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.dim() != 4 or tuple(embeddings.shape[1:]) != (1, 1, dim):
        shape = tuple(embeddings.shape) if isinstance(embeddings, torch.Tensor) else type(embeddings).__name__
        raise ValueError(f"voice embeddings must be [N, 1, 1, {dim}], got {shape}")
    if embeddings.shape[0] < 1 or not embeddings.is_floating_point() or not torch.isfinite(embeddings).all():
        raise ValueError("voice embeddings must be a non-empty finite floating-point tensor")
    if not isinstance(cache, torch.Tensor) or tuple(cache.shape) != (1, NUM_CODEBOOKS, CACHE_SIZE):
        shape = tuple(cache.shape) if isinstance(cache, torch.Tensor) else type(cache).__name__
        raise ValueError(f"voice cache must be [1, {NUM_CODEBOOKS}, {CACHE_SIZE}], got {shape}")
    if cache.dtype != torch.long:
        raise ValueError(f"voice cache must be int64, got {cache.dtype}")
    # Tokens or the -1/-2 sentinels only; the initial tokens are the cards themselves.
    if cache.min() < UNGENERATED_TOKEN_ID or cache[:, 0].max() > TEXT_CARD or cache[:, 1:].max() > AUDIO_CARD:
        raise ValueError("voice cache holds token ids outside the PersonaPlex vocabularies")
    return VoicePreset(embeddings=embeddings, cache=cache)
