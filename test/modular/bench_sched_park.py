"""What one free scan costs over a queue of waiting requests, with the park off and on, on CPU.

Not a test (pytest does not collect it). Run from the repository root:

    .venv/bin/python test/modular/bench_sched_park.py                  # the table
    .venv/bin/python test/modular/bench_sched_park.py --scans 400 --every 8

The scan is a real ``MicroScheduler.get_next_batch`` over a real ``KVManager`` behind a real
``StepRunner``, as ``test_sched_park.py`` builds it. What is stubbed is the graph runtime: it
holds the ready rids in a dict and hands them over as one list each scan (the list is part of
every figure, parked or not). There is no worker, no engine thread, no forward pass, so what
is timed is the scan loop and the asks it makes, and nothing else.

The pool is pressed: two requests run on 30 of its 31 pages and ``N`` waiting ones need 20
each, so none fits. A scan that finds nothing to admit is what the park is for. Three cases:

  pressed    nothing moves between scans: every waiter is parked after the first one
  finishing  one admitted request finishes every ``--every`` scans, and the next waiter in
             line is let in and runs (its prompt pages move the pool); one request arrives
             as another is let in, so the queue holds ``N``
  growing    the above, and a running request is granted a page on every scan, as decode
             asks for one: the pool moves each scan, so the front of the queue is asked
             each time and what is behind it is not

``scan`` is the whole ``get_next_batch``; ``asks`` is the engine's ``check_ready`` calls in it.
"""

from __future__ import annotations

import argparse
import collections
import os
import statistics
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_kv_admission_peak import _StubTransfer
from test_sched_park import MODES, _pressed, _World

from mstar.engine.resources.kv import manager as manager_mod

SIZES = [100, 400, 900]
GATE_US = 1000.0  # a scan over 900 waiters with the park on, in microseconds


def _percentile(sorted_us: list[float], q: float) -> float:
    return sorted_us[min(len(sorted_us) - 1, int(q * len(sorted_us)))]


def _timed_scan(world: _World) -> tuple[float, int]:
    """One free scan: its microseconds and the asks it made. Whom it let in then runs."""
    world.engine.asks.clear()
    start = time.perf_counter_ns()
    batch = world.sched.get_next_batch(world.state)
    elapsed = (time.perf_counter_ns() - start) / 1e3
    asked = len(world.engine.asks)
    for rid in [] if batch is None else list(batch.request_to_worker_graph):
        world.start(rid)
        world.admitted.append(rid)
    return elapsed, asked


def run(fit: str, order: str, n: int, park: bool, case: str, scans: int, every: int):
    world = _pressed(fit, order, n=n, park=park)
    world.admitted = collections.deque(["run", "small"])
    spare = iter(range(10**9))
    scan_us: list[float] = []
    asks: list[int] = []
    for at in range(scans):
        if case != "pressed" and at % every == every - 1 and world.admitted:
            world.finish(world.admitted.popleft())
        if case == "growing" and world.admitted:
            world.grant(world.admitted[-1])
        us, asked = _timed_scan(world)
        scan_us.append(us)
        asks.append(asked)
        for _ in range(n - len(world.waiting())):
            world.arrive(f"x{next(spare)}")
    return scan_us, asks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scans", type=int, default=300, help="free scans timed per row")
    parser.add_argument("--every", type=int, default=10, help="scans between two finishes")
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--cases", nargs="+", default=["pressed", "finishing", "growing"])
    args = parser.parse_args()
    manager_mod.KVTransferManager = _StubTransfer
    for name in (
        "MSTAR_KV_ADMISSION_FIT", "MSTAR_KV_ADMISSION_ORDER", "MSTAR_KV_BACKFILL_WINDOW",
        "MSTAR_STEP_TELEMETRY_DIR", "MSTAR_KV_DEBUG_ASSERTS",
    ):
        os.environ.pop(name, None)
    raise SystemExit(report(args))


def report(args) -> int:
    print(
        f"{'fit-order':<14}{'case':<11}{'N':>5}  {'park':<4}"
        f"{'p50 us':>10}{'p99 us':>10}{'mean us':>10}{'asks/scan':>11}"
    )
    gate_us = None
    for fit, order in MODES:
        for case in args.cases:
            for n in args.sizes:
                for park in (False, True):
                    scan_us, asks = run(fit, order, n, park, case, args.scans, args.every)
                    ordered = sorted(scan_us)
                    p50, p99 = _percentile(ordered, 0.5), _percentile(ordered, 0.99)
                    print(
                        f"{fit + '-' + order:<14}{case:<11}{n:>5}  {'on' if park else 'off':<4}"
                        f"{p50:>10.1f}{p99:>10.1f}{statistics.fmean(scan_us):>10.1f}"
                        f"{statistics.fmean(asks):>11.1f}"
                    )
                    if park and n == 900 and case == "pressed":
                        gate_us = max(gate_us or 0.0, p99)
    if gate_us is None:
        print("\n(no 900-waiter pressed row, so the gate is not checked)")
        return 0
    verdict = "met" if gate_us < GATE_US else "NOT met"
    print(f"\ngate: 900 waiters, park on, pressed, worst p99 {gate_us:.0f} us < {GATE_US:.0f} us: {verdict}")
    return 0 if gate_us < GATE_US else 1


if __name__ == "__main__":
    main()
