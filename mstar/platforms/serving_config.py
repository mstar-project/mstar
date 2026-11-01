"""Reading a serving YAML, which may differ per platform.

A deployment that should run differently on one platform lists the difference
under ``platform_overrides``, keyed by ``Platform.name``; an entry's keys
replace the file's top-level keys. Keeping the block inside the file means a
copy of the config (a benchmark arm's, say) still means the same thing.

Everything that reads a deployment goes through ``load_serving_config``, so
the API server, the conductor and the model's worker graphs see one layout.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import yaml

from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform, XpuPlatform, _built_for

logger = logging.getLogger(__name__)

OVERRIDES_KEY = "platform_overrides"
PLATFORM_NAMES = frozenset(p.name for p in (CpuPlatform, CudaPlatform, RocmPlatform, XpuPlatform))


def apply_platform_overrides(config: Any, platform: str) -> Any:
    """``config`` as ``platform`` serves it, without the overrides block.

    Raises on a block that names no platform, so a misspelled key fails at
    startup rather than silently serving the base layout."""
    if not isinstance(config, Mapping) or OVERRIDES_KEY not in config:
        return config
    resolved = dict(config)
    overrides = resolved.pop(OVERRIDES_KEY)
    if not isinstance(overrides, Mapping):
        raise ValueError(f"`{OVERRIDES_KEY}:` must be a mapping, got {type(overrides).__name__}")
    unknown = sorted(set(overrides) - PLATFORM_NAMES)
    if unknown:
        raise ValueError(f"`{OVERRIDES_KEY}:` names {unknown}; platforms are {sorted(PLATFORM_NAMES)}")
    for name, entry in overrides.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"`{OVERRIDES_KEY}: {name}` must be a mapping, got {type(entry).__name__}")
    entry = overrides.get(platform) or {}
    if entry:
        logger.info("%s: %s replaces %s", OVERRIDES_KEY, platform, sorted(entry))
    return {**resolved, **entry}


def load_serving_config(path: str) -> Any:
    """The parsed YAML at ``path`` for the platform torch was built for. The
    API server reads its config before the first device call, and asking
    ``current_platform`` would start the GPU runtime in that process."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return apply_platform_overrides(config, _built_for().name)
