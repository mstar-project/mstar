"""mminf — disaggregated multimodal inference engine.

The Python SDK is exposed lazily so ``import mminf`` stays cheap and the
server code paths never pull in the client's HTTP dependencies:

    from mminf import MMInfClient
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers / IDEs only — no runtime import cost
    from mminf.client import AudioBuffer, GenerateResult, MMInfClient  # noqa: F401

_LAZY: dict[str, tuple[str, str]] = {
    "MMInfClient": ("mminf.client", "MMInfClient"),
    "GenerateResult": ("mminf.client", "GenerateResult"),
    "AudioBuffer": ("mminf.client", "AudioBuffer"),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'mminf' has no attribute {name!r}")
