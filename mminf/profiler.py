"""Utilities for NVTX range annotations for profiling with nsys."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def _sync_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def range_push(name: str, *, synchronize: bool = True) -> None:
    """Push an NVTX range and optionally synchronize before the marker."""
    if synchronize:
        _sync_if_available()

    torch.cuda.nvtx.range_push(name)


def range_pop(*, synchronize: bool = True) -> None:
    """Pop the current NVTX range and optionally synchronize before the marker."""
    if synchronize:
        _sync_if_available()

    torch.cuda.nvtx.range_pop()


@contextmanager
def nvtx_range(name: str, *, synchronize: bool = True) -> Iterator[None]:
    """Convenience context manager for `range_push`/`range_pop`."""
    range_push(name, synchronize=synchronize)
    try:
        yield
    finally:
        range_pop(synchronize=synchronize)
