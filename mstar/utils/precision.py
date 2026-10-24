"""Scoped fp32 matmul precision."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def ieee_fp32_matmul() -> Iterator[None]:
    """Full fp32 matmuls inside the block, where the engine otherwise runs fp32
    GEMMs in TF32; a graph captured inside keeps this precision. The setting is
    process-wide, so callers stay on the worker's one GPU thread."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
