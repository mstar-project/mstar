"""Candidates and the knob spaces they are drawn from.

A ``Candidate`` is one way to compute an op: the shipped registry op, a torch
expression, or a Triton or HIP kernel at one point of its knob space. Knob
spaces list the values worth trying per knob; ``valid_for`` drops the points a
device cannot run or ignores (three-plus pipeline stages need gfx950's 160 KB
LDS, ``kpack`` is a gfx942 knob, direct-to-LDS loads exist on gfx950 only),
and ``search_order`` puts earlier winners first and samples the rest under a
budget.
"""

from __future__ import annotations

import dataclasses
import itertools
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from mstar.platforms.facts import DeviceFacts

KiB = 1024

# Triton on ROCm and CUDA. num_stages is the number of tiles in flight through
# LDS; waves_per_eu, matrix_instr_nonkdim and kpack are AMD backend options.
TRITON_KNOBS: Mapping[str, tuple] = {
    "BLOCK_M": (16, 32, 64, 128, 256),
    "BLOCK_N": (16, 32, 64, 128, 256),
    "BLOCK_K": (32, 64, 128, 256),
    "num_warps": (1, 2, 4, 8, 16),
    "num_stages": (1, 2, 3, 4),
    "waves_per_eu": (0, 1, 2, 4),
    "matrix_instr_nonkdim": (0, 16, 32),
    "kpack": (1, 2),
    "xcd_remap": (False, True),
    "schedule_hint": ("none", "attention", "memory-bound-attention"),
    "split_k": (1, 2, 4, 8),
}

# HIP C++, compiled per point: workgroup shape, the split of K across
# workgroups and how workgroups map onto XCDs, LDS layout, unroll and prefetch
# distance, and instruction-scheduling hints as short integer vectors
# (s_setprio levels per loop phase, sched_group_barrier (mask, count) pairs).
HIP_KNOBS: Mapping[str, tuple] = {
    "tile": ((16, 16), (32, 32), (64, 64), (128, 128)),
    "waves_per_wg": (1, 2, 4, 8, 16),
    "split_k": (1, 2, 4, 8),
    "xcd_map": ("round_robin", "xcd_local"),
    "lds_layout": ("linear", "padded", "xor"),
    "unroll": (1, 2, 4, 8),
    "prefetch": (0, 1, 2),
    "setprio": ((), (1, 0), (2, 0), (3, 1)),
    "sched_group_barrier": ((), ((0x008, 1), (0x002, 2)), ((0x008, 2), (0x100, 1))),
    "async_lds": (False, True),
}

# AMD knobs, at the value that means "off" elsewhere
_AMD_NEUTRAL: Mapping[str, Any] = {
    "waves_per_eu": 0, "matrix_instr_nonkdim": 0, "kpack": 1, "xcd_remap": False, "xcd_map": "round_robin",
    "setprio": (), "sched_group_barrier": (), "async_lds": False,
}


def is_amd(facts: DeviceFacts) -> bool:
    return bool(facts.arch and facts.arch.startswith("gfx"))


def valid_for(knobs: Mapping[str, Any], facts: DeviceFacts) -> str | None:
    """Why ``knobs`` is not worth running on this device, or None."""
    amd = is_amd(facts)
    for name, neutral in _AMD_NEUTRAL.items():
        value = knobs.get(name, neutral)
        if not amd and (type(value) is not type(neutral) or value != neutral):
            return f"{name} is an AMD knob"
    lds = facts.lds_bytes or 64 * KiB
    if knobs.get("num_stages", 1) > 2 and lds < 160 * KiB:
        return "more than two stages need 160 KB of LDS"
    if knobs.get("kpack", 1) != 1 and facts.arch != "gfx942":
        return "kpack is a gfx942 knob"
    if knobs.get("async_lds") and facts.arch != "gfx950":
        return "direct-to-LDS loads are gfx950's"
    warp = facts.warp_size or (64 if amd else 32)
    threads = knobs.get("waves_per_wg", 1) * warp
    if threads > 1024 or knobs.get("num_warps", 1) * warp > 1024:
        return "more than 1024 threads in a workgroup"
    return None


def space(
    knobs: Mapping[str, Sequence[Any]], facts: DeviceFacts, keep: Callable[[dict], bool] | None = None,
) -> list[dict[str, Any]]:
    """Every point of the product of ``knobs`` this device runs, in order."""
    names = list(knobs)
    points = []
    for values in itertools.product(*(knobs[n] for n in names)):
        point = dict(zip(names, values, strict=True))
        if valid_for(point, facts) is None and (keep is None or keep(point)):
            points.append(point)
    return points


# The shuffle a budget is drawn from; a search sets another (``--shuffle-seed``)
# to reach points an earlier search with the same budget did not try.
SHUFFLE_SEED = 0


def search_order(
    points: Sequence[Mapping[str, Any]], seeds: Iterable[Mapping[str, Any]] = (), budget: int | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """``seeds`` that are in ``points`` first (earlier winners), then the rest
    in a seeded random order, at most ``budget`` in all."""
    seed = SHUFFLE_SEED if seed is None else seed
    pool = [dict(p) for p in points]
    first = [dict(s) for s in seeds if dict(s) in pool]
    rest = [p for p in pool if p not in first]
    random.Random(seed).shuffle(rest)
    ordered = first + rest
    return ordered if budget is None else ordered[:budget]


def label(family: str, knobs: Mapping[str, Any]) -> str:
    """A candidate's name: ``triton[num_warps=4,waves_per_eu=2]``."""
    if not knobs:
        return family
    return f"{family}[{','.join(f'{k}={v}' for k, v in sorted(knobs.items()))}]"


@dataclasses.dataclass(frozen=True, eq=False)
class Candidate:
    """One way to compute an op. ``build()`` returns a callable with the op's
    signature (compiling, if it must); ``isa(fn, args)`` returns register and
    LDS statistics once ``fn`` has run, where the family can read them."""
    family: str
    knobs: Mapping[str, Any]
    build: Callable[[], Callable[..., Any]]
    isa: Callable[[Callable[..., Any], tuple], dict] | None = None

    @property
    def name(self) -> str:
        return label(self.family, self.knobs)
