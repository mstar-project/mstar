"""The platform's inductor settings, applied to one compiled callable or to the compile pool."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mstar.platforms import current_platform


def with_inductor_overrides(
    fn: Callable[..., Any], num_tokens: int | None = None, rows: int | None = None,
    decode_rows: bool = False,
) -> Callable[..., Any]:
    """Wrap ``fn``, a ``torch.compile``'d callable, so it runs under the
    platform's ``inductor_overrides(num_tokens, rows, decode_rows)``.

    torch.compile compiles on call, not when wrapped, so the config is patched
    around every call. With no overrides ``fn`` is returned as is and costs
    nothing extra.

    This resolves ``current_platform``, so call it once the process has its
    device, not at import time.
    """
    overrides = current_platform.inductor_overrides(num_tokens, rows, decode_rows)
    if not overrides:
        return fn

    import torch._inductor.config as inductor_config

    overrides = dict(overrides)

    def forward(*args, **kwargs):
        with inductor_config.patch(**overrides):
            return fn(*args, **kwargs)

    return forward


def start_compile_pool() -> None:
    """Build Inductor's compile pool under the platform's ``compile_pool_overrides``.

    Call before the process's first compile, since the first caller builds the
    one pool. The pool reads these when built, and keeps them after the patch.
    A value set in the environment would stay in the config instead, and so in
    every graph's cache key.
    """
    overrides = current_platform.compile_pool_overrides()
    if not overrides:
        return

    import torch._inductor.config as inductor_config
    from torch._inductor.async_compile import AsyncCompile

    with inductor_config.patch(**overrides):
        AsyncCompile.warm_pool()
