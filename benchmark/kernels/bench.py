"""Captured, cache-rotated kernel timing.

Each candidate is captured in its own graph of ``calls`` calls that cycle
through enough copies of the operands to exceed the device's memory-side
cache (twice the Infinity Cache on AMD, so every pass reads HBM; capped by
``max_copies`` and ``max_bytes``), the way a decode step streams its weights.
Graphs are built first, then replayed in turn for ``rounds`` rounds, so a
drifting clock or a co-tenant affects every candidate alike, and the median
replay is kept. A candidate that cannot be captured is timed eagerly and
marked so. On the CPU (the harness's own tests) a wall clock stands in.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import torch

MiB = 1 << 20


@dataclasses.dataclass
class Timing:
    us: float
    runs_us: list[float]
    captured: bool
    calls: int
    copies: int
    note: str = ""


class _Runner(NamedTuple):
    """One candidate, ready to time: ``run()`` times a pass in ms."""
    run: Callable[[], float]
    captured: bool
    note: str


def tensor_bytes(args: Sequence[Any]) -> int:
    return sum(a.numel() * a.element_size() for a in args if isinstance(a, torch.Tensor))


def copies_needed(set_bytes: int, rotate_bytes: int, max_copies: int, max_bytes: int) -> int:
    """Operand sets one pass needs to touch more than ``rotate_bytes``."""
    if set_bytes <= 0:
        return 1
    wanted = math.ceil(rotate_bytes / set_bytes) + 1 if set_bytes <= rotate_bytes else 1
    return max(1, min(wanted, max_copies, max(1, max_bytes // set_bytes)))


class Bench:
    """Times callables on one device.

    ``rotate_bytes`` is the traffic a pass over the operand copies must
    exceed (default: twice the device's memory-side cache, or 512 MB).
    ``timer`` replaces the clock on the CPU, for tests.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        calls: int = 256,
        rounds: int = 3,
        rotate_bytes: int | None = None,
        max_copies: int = 64,
        max_bytes: int = 4 << 30,
        timer: Callable[[], float] | None = None,
    ):
        self.device = torch.device(device)
        self.calls = calls
        self.rounds = rounds
        self.rotate_bytes = 512 * MiB if rotate_bytes is None else rotate_bytes
        self.max_copies = max_copies
        self.max_bytes = max_bytes
        self.timer = timer or time.perf_counter

    def operand_sets(self, make: Callable[[int], tuple]) -> list[tuple]:
        """Copies of the operands, ``make(i)`` building the i-th."""
        first = make(0)
        count = copies_needed(tensor_bytes(first), self.rotate_bytes, self.max_copies, self.max_bytes)
        return [first, *(make(i) for i in range(1, count))]

    def time(
        self, fns: Mapping[str, Callable[..., Any]], arg_sets: Sequence[tuple], capture: bool = True,
    ) -> dict[str, Timing | str]:
        """Each of ``fns`` over ``arg_sets``: its Timing, or why it could not run."""
        calls = max(len(arg_sets), self.calls // len(arg_sets) * len(arg_sets))
        if self.device.type != "cuda":
            return {name: self._time_cpu(fn, arg_sets, calls) for name, fn in fns.items()}
        runners: dict[str, _Runner] = {}
        results: dict[str, Timing | str] = {}
        for name, fn in fns.items():
            try:
                runners[name] = self._runner(fn, arg_sets, calls, capture)
            except Exception as exc:  # noqa: BLE001  recorded per candidate
                results[name] = f"{type(exc).__name__}: {exc}"[:500]
        runs: dict[str, list[float]] = {name: [] for name in runners}
        for _ in range(self.rounds):
            for name, runner in list(runners.items()):
                try:
                    runs[name].append(runner.run() * 1e3 / calls)
                except Exception as exc:  # noqa: BLE001  the others keep their rounds
                    results[name] = f"{type(exc).__name__}: {exc}"[:500]
                    del runners[name]
        for name, runner in runners.items():
            results[name] = Timing(
                us=statistics.median(runs[name]), runs_us=runs[name], captured=runner.captured, calls=calls,
                copies=len(arg_sets), note=runner.note,
            )
        return results

    def _runner(self, fn, arg_sets, calls, capture) -> _Runner:
        """``fn`` captured (or else eager) over ``calls`` calls, after a warm-up
        that runs every operand set once, outside any capture."""
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for args in arg_sets:
                fn(*args)
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        note = ""
        if capture:
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for i in range(calls):
                        fn(*arg_sets[i % len(arg_sets)])
                graph.replay()
                torch.cuda.synchronize(self.device)
                return _Runner(lambda: self._elapsed_ms(graph.replay), True, note)
            except Exception as exc:  # noqa: BLE001  timed eagerly instead
                note = f"not capturable ({type(exc).__name__}: {str(exc)[:200]})"
                torch.cuda.synchronize(self.device)

        def eager():
            for i in range(calls):
                fn(*arg_sets[i % len(arg_sets)])

        eager()
        return _Runner(lambda: self._elapsed_ms(eager), False, note)

    def _elapsed_ms(self, run: Callable[[], Any]) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    def _time_cpu(self, fn, arg_sets, calls) -> Timing | str:
        try:
            for args in arg_sets:
                fn(*args)
            runs = []
            for _ in range(self.rounds):
                t0 = self.timer()
                for i in range(calls):
                    fn(*arg_sets[i % len(arg_sets)])
                runs.append((self.timer() - t0) * 1e6 / calls)
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"[:500]
        return Timing(statistics.median(runs), runs, captured=False, calls=calls, copies=len(arg_sets),
                      note="wall clock on the CPU")

    def node_floor_us(self) -> float | None:
        """Per-call cost of the smallest kernel in a captured graph: the floor
        a launch adds whatever the kernel does. None off the GPU."""
        if self.device.type != "cuda":
            return None
        cell = torch.zeros(1, device=self.device)
        timing = self.time({"fill": lambda t: t.fill_(1.0)}, [(cell,)])["fill"]
        return timing.us if isinstance(timing, Timing) else None
