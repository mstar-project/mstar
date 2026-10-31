"""The platform's inductor settings, applied to one compiled callable."""

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
