"""Workload generation for the simulator.

The arrival patterns mirror ``benchmark/runner.py`` so a simulated run and a
measured run can be driven the same way — that is what makes their metrics
comparable at all:

* ``offline``     — strict waves of ``batch_size`` requests; the next wave
                    starts when the previous one has fully drained.
* ``closed_loop`` — a fixed number of in-flight requests; a new one is
                    admitted whenever one completes.
* ``online``      — Poisson arrivals at a fixed rate.

The simulator cannot inspect token values, so output lengths come from the
spec rather than from EOS — the same situation as a measured run with
``--ignore-eos``, which is how the benchmark pins output length for
cross-system comparison.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from mstar.sim.des import SimRequest


@dataclass
class WorkloadSpec:
    """What to send, and how fast."""

    num_requests: int = 32
    mode: str = "online"  # online | closed_loop | offline
    rate: float = 4.0     # requests/second, online only
    concurrency: int = 8  # closed_loop / offline batch size
    prompt_tokens: int = 64
    output_tokens: int = 128
    #: Per-request jitter on the lengths, as a fraction of the mean. 0 keeps
    #: every request identical, which makes a first comparison easier to read.
    length_jitter: float = 0.0
    seed: int = 0

    def describe(self) -> str:
        if self.mode == "online":
            pace = f"poisson {self.rate} req/s"
        elif self.mode == "closed_loop":
            pace = f"closed loop, {self.concurrency} in flight"
        else:
            pace = f"offline waves of {self.concurrency}"
        return (
            f"{self.num_requests} requests, {pace}, "
            f"prompt~{self.prompt_tokens} tok, output~{self.output_tokens} tok"
        )


def build_requests(spec: WorkloadSpec) -> list[SimRequest]:
    """Materialize the request list with arrival times already assigned.

    Only ``online`` can have all its arrivals precomputed. Closed-loop and
    offline arrivals depend on completions, so those are given arrival time 0
    and released by the driver as slots free up (see :func:`drive`).
    """
    rng = random.Random(spec.seed)
    reqs: list[SimRequest] = []
    t = 0.0
    for i in range(spec.num_requests):
        if spec.mode == "online":
            t += rng.expovariate(spec.rate) if spec.rate > 0 else 0.0
            arrival = t
        else:
            arrival = 0.0

        def jitter(mean: int) -> int:
            if spec.length_jitter <= 0:
                return mean
            lo = int(mean * (1 - spec.length_jitter))
            hi = int(mean * (1 + spec.length_jitter))
            return max(1, rng.randint(lo, hi))

        reqs.append(SimRequest(
            rid=f"req-{i:05d}",
            arrival_s=arrival,
            prompt_tokens=jitter(spec.prompt_tokens),
            target_output_tokens=jitter(spec.output_tokens),
        ))
    return reqs


def drive(sim, spec: WorkloadSpec) -> None:
    """Feed a workload into a simulator and run it to completion.

    Online mode submits everything up front — arrivals are already timed.
    The closed-loop and offline modes are load-following, so they submit in
    rounds, running the simulator between them.
    """
    reqs = build_requests(spec)

    if spec.mode == "online":
        for r in reqs:
            sim.submit(r)
        sim.run()
        return

    if spec.mode == "closed_loop":
        pending = list(reqs)
        in_flight = 0
        # Prime the pipe, then replace each completion with a new request.
        while pending or in_flight:
            while pending and in_flight < spec.concurrency:
                r = pending.pop(0)
                r.arrival_s = sim.cal.now
                sim.submit(r)
                in_flight += 1
            before = len(sim.finished)
            sim.run()
            done = len(sim.finished) - before
            if done == 0 and not pending:
                break
            in_flight = max(0, in_flight - done)
        return

    # offline: strict waves — the next wave starts only once this one drained
    pending = list(reqs)
    while pending:
        wave, pending = pending[:spec.concurrency], pending[spec.concurrency:]
        for r in wave:
            r.arrival_s = sim.cal.now
            sim.submit(r)
        sim.run()
