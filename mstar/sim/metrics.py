"""Aggregate a simulated run into the metrics the benchmark harness reports.

Definitions follow ``benchmark/request.py`` so simulated and measured numbers
mean the same thing:

* **TTFT** — first chunk of a modality, relative to request send.
* **ITL** — inter-token latency for text: gaps between chunk arrivals,
  token-weighted.
* **E2E** — send to last chunk.
* **RTF** — end-to-end latency divided by the duration of audio produced
  (24 kHz mono, matching the harness's assumption).
* **throughput** — completed requests and output tokens per second of
  wall (here, simulated) time.

Every report carries the simulation's coverage flags. A number computed from
extrapolated or missing step costs is not the same kind of claim as one
computed from measured costs, and the report says which it is instead of
leaving the reader to assume.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from mstar.sim.des import Simulator
from mstar.sim.stepdb import Coverage

#: The benchmark harness assumes 24 kHz mono when converting audio bytes to
#: seconds; keep the same constant so RTF is comparable.
AUDIO_SAMPLE_RATE = 24000


def _pcts(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    def pct(p: float) -> float:
        if len(s) == 1:
            return s[0]
        k = (len(s) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)
    return {
        "mean": statistics.fmean(s),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
    }


@dataclass
class SimReport:
    """Aggregate results of one simulated run."""

    num_requests: int = 0
    num_completed: int = 0
    sim_duration_s: float = 0.0
    ttft_ms: dict[str, float] = field(default_factory=dict)
    itl_ms: dict[str, float] = field(default_factory=dict)
    e2e_ms: dict[str, float] = field(default_factory=dict)
    request_throughput: float = 0.0
    output_token_throughput: float = 0.0
    #: rank -> fraction of simulated time the GPU was busy
    gpu_utilization: dict[int, float] = field(default_factory=dict)
    cpu_utilization: dict[int, float] = field(default_factory=dict)
    steps: int = 0
    coverage: str = "exact"
    missing: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "num_requests": self.num_requests,
            "num_completed": self.num_completed,
            "sim_duration_s": self.sim_duration_s,
            "ttft_ms": self.ttft_ms,
            "itl_ms": self.itl_ms,
            "e2e_ms": self.e2e_ms,
            "request_throughput": self.request_throughput,
            "output_token_throughput": self.output_token_throughput,
            "gpu_utilization": self.gpu_utilization,
            "cpu_utilization": self.cpu_utilization,
            "steps": self.steps,
            "coverage": self.coverage,
            "missing": self.missing,
        }

    def render(self) -> str:
        lines = [
            "─" * 62,
            f" Simulated run — {self.num_completed}/{self.num_requests} completed",
            "─" * 62,
        ]
        if self.coverage != "exact":
            lines.append(f" ⚠ cost coverage: {self.coverage}")
            for key, n in sorted(self.missing.items(), key=lambda kv: -kv[1]):
                lines.append(f"     {n:6d} steps with no measured cost: {key}")
            lines.append("")

        def block(title: str, d: dict[str, float], unit: str = "ms") -> None:
            if not d:
                return
            lines.append(f" {title}")
            lines.append(
                f"   mean {d['mean']:8.2f} {unit}   p50 {d['p50']:8.2f}   "
                f"p95 {d['p95']:8.2f}   p99 {d['p99']:8.2f}"
            )

        block("TTFT", self.ttft_ms)
        block("ITL (per output token)", self.itl_ms)
        block("E2E", self.e2e_ms)
        lines.append("")
        lines.append(f" throughput  {self.request_throughput:8.3f} req/s   "
                     f"{self.output_token_throughput:8.1f} tok/s")
        lines.append(f" duration    {self.sim_duration_s:8.3f} s simulated, "
                     f"{self.steps} engine steps")
        if self.gpu_utilization:
            util = "  ".join(
                f"w{r}: gpu {g * 100:.1f}% cpu {self.cpu_utilization.get(r, 0) * 100:.1f}%"
                for r, g in sorted(self.gpu_utilization.items())
            )
            lines.append(f" utilization {util}")
        lines.append("─" * 62)
        return "\n".join(lines)


def summarize(sim: Simulator, num_submitted: int | None = None) -> SimReport:
    """Aggregate a finished simulation."""
    done = [r for r in sim.finished if r.finish_s is not None]
    rep = SimReport(
        num_requests=num_submitted if num_submitted is not None else len(sim.requests),
        num_completed=len(done),
        sim_duration_s=sim.cal.now,
        steps=sim.step_count,
        coverage=sim.coverage.describe(),
        missing=dict(sim.missing_keys),
    )
    if not done:
        return rep

    rep.ttft_ms = _pcts([
        r.ttft_s() * 1e3 for r in done if r.ttft_s() is not None
    ])
    rep.e2e_ms = _pcts([
        r.e2e_s() * 1e3 for r in done if r.e2e_s() is not None
    ])

    # ITL: gaps between consecutive client-visible chunks, per request,
    # pooled across requests (the harness's token-weighted definition
    # reduces to this when every chunk carries one token).
    gaps: list[float] = []
    for r in done:
        times = sorted(t for _, t in r.chunks)
        gaps.extend((b - a) * 1e3 for a, b in zip(times, times[1:]))
    rep.itl_ms = _pcts(gaps)

    span = max((r.finish_s for r in done), default=0.0)
    if span > 0:
        rep.request_throughput = len(done) / span
        rep.output_token_throughput = sum(r.decode_steps for r in done) / span

    for rank, w in sim.workers.items():
        if sim.cal.now > 0:
            rep.gpu_utilization[rank] = w.gpu_busy_s / sim.cal.now
            rep.cpu_utilization[rank] = w.cpu_busy_s / sim.cal.now
    return rep


def rtf(report: SimReport, audio_seconds: float) -> float | None:
    """Real-time factor: E2E divided by the audio duration produced."""
    if not report.e2e_ms or audio_seconds <= 0:
        return None
    return (report.e2e_ms["mean"] / 1e3) / audio_seconds
