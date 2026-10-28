"""The user's cache root, which M*'s on-disk caches (compiled kernels,
downloaded checkpoints, eval work dirs) live under.

Standard library only: :mod:`mstar.engine` calls it before importing torch.
"""
import os
from pathlib import Path


def user_cache_dir() -> Path:
    """``$XDG_CACHE_HOME``, else ``~/.cache``. A relative ``XDG_CACHE_HOME`` is
    ignored, as the XDG spec says. Raises ``RuntimeError`` when there is no home
    directory to fall back to."""
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    return Path(xdg) if os.path.isabs(xdg) else Path.home() / ".cache"
