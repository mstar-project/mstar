"""Keeps Inductor's pointwise kernels off launch configs that give each thread
two elements, which ptxas before 13.1.115 can miscompile on Blackwell.

The miscompile needs a floor division of a periodic index under a range
select, as in an interleaved mRoPE table. Pointwise blocks go wrong only at
exactly two elements per thread, so such a config becomes the same block with
twice or half the warps, and coordinate descent and the autotune cache are kept
off them. Reductions can go wrong at other tilings and are not covered.

Inductor builds a kernel's configs in its compile workers, so the patch is
installed there too, by the function each worker runs per kernel. A cache key
tag keeps FX graph cache entries autotuned without the patch apart.
"""

from __future__ import annotations

import copy
import functools
import inspect
import re
import subprocess
import threading

CACHE_KEY_TAG = "mstar-two-element-guard"
FIXED_PTXAS = (13, 1, 115)

_installed = False


def elems_per_thread(config) -> float:
    """Elements of a pointwise block each thread holds under ``config``."""
    block = 1
    for key in ("XBLOCK", "YBLOCK", "ZBLOCK"):
        block *= config.kwargs.get(key, 1)
    return block / (32 * config.num_warps)


def without_two_per_thread(config):
    """``config``, or the same block at one or four elements per thread.

    Doubling the warps below 8 keeps Inductor's minimum of 4 warps for blocks
    of 128 elements and up; halving above keeps the grid and vector width.
    """
    if elems_per_thread(config) != 2:
        return config
    config = copy.copy(config)
    config.num_warps = config.num_warps // 2 if config.num_warps >= 8 else config.num_warps * 2
    return config


@functools.cache
def blackwell_ptxas_version() -> tuple[int, int, int] | None:
    """(major, minor, build) of the ptxas Triton runs for Blackwell, None if unknown."""
    try:
        from triton import knobs

        out = subprocess.run(
            [knobs.nvidia.ptxas_blackwell.path, "--version"],
            capture_output=True, text=True, check=True,
        ).stdout
    except Exception:
        return None
    match = re.search(r"V(\d+)\.(\d+)\.(\d+)", out)
    return tuple(int(part) for part in match.groups()) if match else None


def ptxas_affected() -> bool:
    """Whether Triton's Blackwell ptxas predates the fix; an unknown version counts."""
    version = blackwell_ptxas_version()
    return version is None or version < FIXED_PTXAS


def install() -> None:
    """Patch Inductor in this process. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True

    from torch._inductor import async_compile
    from torch._inductor.runtime import coordinate_descent_tuner, triton_heuristics
    from torch._inductor.runtime.hints import HeuristicType
    from torch.compiler import config as compiler_config

    cached_autotune = triton_heuristics.cached_autotune
    check_autotune_cache = triton_heuristics.check_autotune_cache
    is_valid_config = coordinate_descent_tuner.CoordescTuner.is_valid_config
    signature = inspect.signature(cached_autotune)
    pointwise = threading.local()

    @functools.wraps(cached_autotune)
    def guarded_cached_autotune(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        if bound.arguments["heuristic_type"] != HeuristicType.POINTWISE or bound.arguments.get("custom_kernel"):
            return cached_autotune(*args, **kwargs)
        bound.arguments["configs"] = [without_two_per_thread(c) for c in bound.arguments["configs"]]
        pointwise.active = True
        try:
            return cached_autotune(*bound.args, **bound.kwargs)
        finally:
            pointwise.active = False

    @functools.wraps(check_autotune_cache)
    def guarded_check_autotune_cache(*args, **kwargs):
        # a best config cached without the guard may have two elements per thread
        configs, cache, info = check_autotune_cache(*args, **kwargs)
        if getattr(pointwise, "active", False):
            configs = [without_two_per_thread(c) for c in configs]
        return configs, cache, info

    @functools.wraps(is_valid_config)
    def guarded_is_valid_config(self, config):
        if (
            not self.is_mm
            and self.size_hints is not None
            and set(self.size_hints) <= {"x", "y", "z"}
            and elems_per_thread(config) == 2
        ):
            return False
        return is_valid_config(self, config)

    triton_heuristics.cached_autotune = guarded_cached_autotune
    triton_heuristics.check_autotune_cache = guarded_check_autotune_cache
    coordinate_descent_tuner.CoordescTuner.is_valid_config = guarded_is_valid_config
    async_compile._worker_compile_triton = _worker_compile_triton
    tag = compiler_config.cache_key_tag
    compiler_config.cache_key_tag = f"{tag}+{CACHE_KEY_TAG}" if tag else CACHE_KEY_TAG


def _worker_compile_triton(load_kernel, extra_env, extra_config):
    """Inductor's per-kernel compile task, run with the guard installed in the
    compile worker; pickled by reference, so the worker imports this module."""
    from torch._inductor.runtime.compile_tasks import _worker_compile_triton as compile_kernel

    install()
    return compile_kernel(load_kernel, extra_env, extra_config)
