"""The platform's inductor settings, applied to one compiled callable."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mstar.platforms import current_platform


def with_inductor_overrides(
    fn: Callable[..., Any], num_tokens: int | None = None,
) -> Callable[..., Any]:
    """``fn``, a ``torch.compile``'d callable, run under the platform's
    ``inductor_overrides(num_tokens)``.

    torch.compile traces lazily, so patching the config around each call is
    what puts the overrides in effect for the traces ``fn`` builds, and only
    for those. With no overrides ``fn`` itself comes back, so a platform that
    keeps inductor's defaults pays nothing per call.

    Asks ``current_platform`` now: wrap where the process already drives its
    device, as the engine's compile sites do, not at import.
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
