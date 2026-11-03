"""Load generation: closed loop (fixed in-flight count) and open loop (Poisson).

A suite supplies one coroutine per request; the driver decides when each one
starts and records its start offset. Both modes return results in input
order so repeats line up request by request.

* ``closed``: ``concurrency`` requests are in flight at all times; the next
  starts as soon as one finishes (sglang-omni's and benchmark/runner.py's
  ``closed_loop``).
* ``open``: requests arrive as a Poisson process at ``rate`` per second,
  seeded so every system sees the same arrival times. ``concurrency`` caps
  in-flight requests when > 0 (arrivals then queue client side, and the wait
  counts toward their latency since the clock starts at the arrival).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")
RequestFn = Callable[[int, Any, float], Awaitable[T]]  # (index, item, t0 of the request) -> result

MODES = ("closed", "open")


@dataclass(frozen=True)
class LoadSpec:
    mode: str = "closed"
    concurrency: int = 1
    rate: float | None = None
    seed: int = 0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"load mode must be one of {MODES}, got {self.mode!r}")
        if self.mode == "closed" and self.concurrency < 1:
            raise ValueError("closed loop needs concurrency >= 1")
        if self.mode == "open" and not (self.rate and self.rate > 0):
            raise ValueError("open loop needs rate > 0")

    def describe(self) -> str:
        if self.mode == "closed":
            return f"c={self.concurrency}"
        cap = f",cap={self.concurrency}" if self.concurrency > 0 else ""
        return f"rate={self.rate:g}/s{cap}"


def poisson_offsets(n: int, rate: float, seed: int) -> list[float]:
    """Arrival offsets (seconds from the start) of ``n`` Poisson arrivals; the first is 0."""
    rng = random.Random(seed)
    t, out = 0.0, []
    for i in range(n):
        if i:
            t += rng.expovariate(rate)
        out.append(t)
    return out


async def run_load(items: Sequence[Any], fn: RequestFn, spec: LoadSpec) -> tuple[list[Any], float]:
    """Run ``fn`` over ``items`` under ``spec``; (results in item order, wall seconds)."""
    results: list[Any] = [None] * len(items)
    start = time.perf_counter()
    if spec.mode == "closed":
        sem = asyncio.Semaphore(spec.concurrency)

        async def closed(i: int, item: Any) -> None:
            async with sem:
                results[i] = await fn(i, item, time.perf_counter())

        await asyncio.gather(*(closed(i, it) for i, it in enumerate(items)))
    else:
        sem = asyncio.Semaphore(spec.concurrency) if spec.concurrency > 0 else None
        offsets = poisson_offsets(len(items), spec.rate, spec.seed)

        async def arrive(i: int, item: Any) -> None:
            delay = start + offsets[i] - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            t0 = time.perf_counter()
            if sem is None:
                results[i] = await fn(i, item, t0)
            else:
                async with sem:
                    results[i] = await fn(i, item, t0)

        await asyncio.gather(*(arrive(i, it) for i, it in enumerate(items)))
    return results, time.perf_counter() - start


async def warmup(items: Sequence[Any], fn: RequestFn, spec: LoadSpec, count: int) -> None:
    """``count`` untimed requests at the measured concurrency, so the first measured
    wave does not hit cold shapes (accelerator graphs, KV pages, scheduler queues)."""
    if count <= 0 or not items:
        return
    warm = [items[i % len(items)] for i in range(count)]
    closed = LoadSpec("closed", max(1, spec.concurrency))
    await run_load(warm, fn, closed)
