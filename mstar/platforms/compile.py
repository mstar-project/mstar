"""The platform's inductor settings, applied to one compiled callable."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

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


@functools.cache
def compiled(fn: Callable[..., Any], dynamic: bool | None = None) -> Callable[..., Any]:
    """``torch.compile(fn)`` in Inductor's deterministic mode under the
    platform's inductor overrides, built on first call and cached, for a
    module-level helper that a resource runs eagerly (``torch.compiler.disable``)
    inside a captured forward. Deterministic as Qwen3.5's captures are, so the
    helpers keep their output the same across server processes too."""
    from mstar.engine.accelerator_graph_runner import deterministic_inductor_options

    options = deterministic_inductor_options()
    return with_inductor_overrides(torch.compile(fn, dynamic=dynamic, options=options or None))
