"""Torch settings the engine depends on, applied one thread at a time.

Since torch 2.12, ``torch._dynamo.config`` keeps each override in a
``ContextVar``, so an assignment only reaches the thread that made it. A thread
started later begins with an empty context and reads torch's defaults. The
worker compiles on two threads: the main thread captures CUDA graphs during
warmup, and the GPU executor thread runs every request after that. Each has to
call ``apply_torch_config`` itself. Importing ``mstar.engine`` covers the main
thread, and the worker's executor thread initializer covers the rest (#167).
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

RECOMPILE_LIMIT_ENV = "MSTAR_RECOMPILE_LIMIT"
# How many compiled variants dynamo keeps for one function before it runs that
# function eagerly. CUDA graph capture compiles one variant per captured shape.
DEFAULT_RECOMPILE_LIMIT = 84
# Bounds on the env override. The minimum is torch's own default.
RECOMPILE_LIMIT_MIN = 8
RECOMPILE_LIMIT_MAX = 1024
# torch's default for its second cap, over every variant of one function.
_TORCH_ACCUMULATED_RECOMPILE_LIMIT = 256


def recompile_limit() -> int:
    """The recompile limit: ``MSTAR_RECOMPILE_LIMIT`` if set, clamped to
    ``[RECOMPILE_LIMIT_MIN, RECOMPILE_LIMIT_MAX]``, else the default."""
    raw = os.environ.get(RECOMPILE_LIMIT_ENV, "").strip()
    if not raw:
        return DEFAULT_RECOMPILE_LIMIT
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring %s=%r: not an integer; using %d",
            RECOMPILE_LIMIT_ENV, raw, DEFAULT_RECOMPILE_LIMIT,
        )
        return DEFAULT_RECOMPILE_LIMIT
    clamped = min(max(value, RECOMPILE_LIMIT_MIN), RECOMPILE_LIMIT_MAX)
    if clamped != value:
        logger.warning(
            "%s=%d is outside [%d, %d]; using %d",
            RECOMPILE_LIMIT_ENV, value, RECOMPILE_LIMIT_MIN, RECOMPILE_LIMIT_MAX,
            clamped,
        )
    return clamped


def apply_torch_config() -> None:
    """Apply mstar's torch settings to the calling thread.

    Call it on every thread that may compile.
    """
    limit = recompile_limit()
    config = torch._dynamo.config
    config.recompile_limit = limit
    # The second cap counts a function's variants across all module instances.
    # An override above torch's default for it would otherwise do nothing.
    config.accumulated_recompile_limit = max(limit, _TORCH_ACCUMULATED_RECOMPILE_LIMIT)
    # Let int inputs, and int attributes on modules, turn symbolic once their
    # value changes, instead of compiling a new variant for every value.
    config.allow_unspec_int_on_nn_module = True
    config.specialize_int = False
    # This one is process-wide, not per thread. It lives here so every torch
    # setting the engine needs is in one place.
    torch.set_float32_matmul_precision("high")
