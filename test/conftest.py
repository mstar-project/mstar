"""Shared pytest hooks for the entire test tree.

No need to add per-file ``pytest.mark.skipif(not torch.cuda.is_available())``, 
we skip all ``gpu`` marked tests when CUDA is unavailable here. 

Mark things at necessary granularity: module-level when an entire file is 
subject to certain requirements; test-level otherwise. 
"""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    if torch is not None and torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="requires CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
