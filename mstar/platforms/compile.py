"""The platform's inductor settings, applied to one compiled callable."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

import torch

from mstar.platforms import current_platform

logger = logging.getLogger(__name__)


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
def _has_deterministic_mode() -> bool:
    import torch._inductor.config as inductor_config

    if hasattr(inductor_config, "deterministic"):
        return True
    logger.warning(
        "torch %s has no Inductor deterministic mode; compiles that ask for it time "
        "their kernels instead, so their output can differ between server processes",
        torch.__version__,
    )
    return False


def deterministic_inductor_options() -> dict[str, Any]:
    """Inductor options for its deterministic mode (the engine's
    ``DETERMINISTIC_COMPILE_MODE``), for a compile that sets its own options;
    empty, with a warning, on a torch without the mode."""
    return {"deterministic": True} if _has_deterministic_mode() else {}


@functools.cache
def compiled(fn: Callable[..., Any], dynamic_args: tuple[int, ...] = ()) -> Callable[..., Any]:
    """``torch.compile(fn)`` in Inductor's deterministic mode under the
    platform's inductor overrides, built on first call and cached, for a
    module-level helper that a resource runs eagerly (``torch.compiler.disable``)
    inside a captured forward. Deterministic, so the helpers keep their output
    the same across server processes.

    ``dynamic_args`` are the positional tensor args whose dim 0 (a step's token
    count) is marked dynamic on every call, so ``fn`` compiles once for every
    count with its other dims static. Marked per call, since a call inside a
    captured forward inherits the runner's ``dynamic=False``, which would
    otherwise compile it once per bucket."""
    options = deterministic_inductor_options()
    run = with_inductor_overrides(torch.compile(fn, options=options or None))
    if not dynamic_args:
        return run

    def call(*args):
        for i in dynamic_args:
            torch._dynamo.maybe_mark_dynamic(args[i], 0)
        return run(*args)

    return call
