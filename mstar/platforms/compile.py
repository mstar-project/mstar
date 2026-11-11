"""The platform's Inductor settings for one compiled callable, and a process-wide fix to
Inductor's autotuning."""

from __future__ import annotations

import copy
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


def _below_split_scan_minimum(meta: dict, name: str, val: int) -> bool:
    return (
        meta.get("grid_type") == "SplitScanGrid"
        and name.startswith("R") and name.endswith("BLOCK")
        and val < meta.get("min_split_scan_rblock", 256)
    )


def _raise_split_scan_blocks(tuner: Any) -> None:
    """Recompile, at the minimum block size, any of a split scan's configs below it."""
    meta = tuner.inductor_meta
    results = []
    for result in tuner.compile_results:
        cfg = result.config
        low = [k for k, v in cfg.kwargs.items() if _below_split_scan_minimum(meta, k, v)]
        if not low:
            results.append(result)
            continue
        raised = copy.copy(cfg)
        raised.kwargs = {**cfg.kwargs, **dict.fromkeys(low, meta.get("min_split_scan_rblock", 256))}
        raised.found_by_coordesc = False  # retune from here, and cache what it finds
        if tuner.fn.fn is None:  # compiled in a worker, which dropped the source
            tuner.fn = tuner._reload_kernel().fn
        results.append(tuner._precompile_config(raised))
        logger.info("%s: raised a split scan's %s below the workspace minimum", meta.get("kernel_name"), low)
    tuner.compile_results = results


@functools.cache
def keep_split_scans_in_their_workspace() -> None:
    """Never launch a split scan with a block size below the one its workspace is sized for.
    Process-wide; the engine installs it before anything compiles.

    A split scan (a long 1-D ``cumsum``) sizes its look-back workspace for blocks of at
    least ``config.triton.min_split_scan_rblock``. Coordinate descent's only lower bound is
    zero, so it can tune below that, and the scan then writes past its workspace: an
    illegal memory access, or a silent overwrite of what follows. Such a pick is then
    cached, and comes back through compile workers, the autotune cache and the FX graph
    cache. So the tuner refuses to step below the minimum, and every config is raised to
    it where the parent process turns configs into launchers, which every launch and every
    tuning baseline passes through. Upstream: pytorch/pytorch#197493.
    """
    try:
        from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    except ImportError:
        CoordescTuner = CachingAutotuner = None
    if not (hasattr(CoordescTuner, "value_too_small") and hasattr(CachingAutotuner, "_make_launchers")):
        logger.warning("Inductor's autotuner changed: split scans are not kept inside their workspace")
        return
    below_any = CoordescTuner.value_too_small

    def value_too_small(self, name: str, val: int) -> bool:
        return _below_split_scan_minimum(self.inductor_meta, name, val) or below_any(self, name, val)

    make_launchers = CachingAutotuner._make_launchers

    def _make_launchers(self) -> None:
        if self.inductor_meta.get("grid_type") == "SplitScanGrid" and len(self.launchers) != len(self.compile_results):
            _raise_split_scan_blocks(self)
        make_launchers(self)

    CoordescTuner.value_too_small = value_too_small
    CachingAutotuner._make_launchers = _make_launchers


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
