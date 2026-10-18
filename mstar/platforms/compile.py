"""Platform settings applied around compiled callables and kernel calls."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from mstar.platforms import current_platform


def with_inductor_overrides(
    fn: Callable[..., Any], num_tokens: int | None = None,
) -> Callable[..., Any]:
    """Wrap ``fn``, a ``torch.compile``'d callable, so it runs under the
    platform's ``inductor_overrides(num_tokens)``.

    torch.compile compiles on call, not when wrapped, so the config is patched
    around every call. With no overrides ``fn`` is returned as is and costs
    nothing extra.

    This resolves ``current_platform``, so call it once the process has its
    device, not at import time.
    """
    overrides = current_platform.inductor_overrides(num_tokens)
    if not overrides:
        return fn

    import torch._inductor.config as inductor_config

    overrides = dict(overrides)

    def forward(*args, **kwargs):
        with inductor_config.patch(**overrides):
            return fn(*args, **kwargs)

    return forward


@contextlib.contextmanager
def deterministic_convs():
    """Deterministic conv kernels inside the block on ROCm, a no-op elsewhere.

    MIOpen's split-K implicit-GEMM conv solver is not deterministic, so the
    same input can give a different result on each call.
    """
    if not current_platform.is_rocm():
        yield
        return
    import torch

    previous = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = previous
