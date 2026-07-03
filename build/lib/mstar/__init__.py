"""mstar — disaggregated multimodal inference engine.

The Python SDK is exposed lazily so ``import mstar`` stays cheap and the
server code paths never pull in the client's HTTP dependencies:

    from mstar import MStarClient
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers / IDEs only — no runtime import cost
    from mstar.client import AudioBuffer, GenerateResult, MStarClient  # noqa: F401

_LAZY: dict[str, tuple[str, str]] = {
    "MStarClient": ("mstar.client", "MStarClient"),
    "GenerateResult": ("mstar.client", "GenerateResult"),
    "AudioBuffer": ("mstar.client", "AudioBuffer"),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'mstar' has no attribute {name!r}")
