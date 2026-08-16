"""Shared problem shapes + honest timing for the fused-MoE grouped GEMM.

Timing follows the same discipline as the rest of the perf work here: an
iteration-count warmup is meaningless for a 10--100 us kernel because the
SM clock has not moved yet.  We warm for a wall-clock duration under
continuous load, then measure several windows and report the median, along
with the SM clock and throttle state at the time of the measurement so a
number can be checked against the power budget it was taken under.
"""

from __future__ import annotations

import statistics
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

# ---------------------------------------------------------------------------
# Problem shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoEShape:
    """One MoE block's routed-expert geometry."""

    name: str
    hidden: int
    inter: int
    num_experts: int
    top_k: int

    @property
    def gemm1(self) -> tuple[int, int]:
        """(N, K) of the gate+up GEMM: cache1 = hidden @ w1[e].T."""
        return (2 * self.inter, self.hidden)

    @property
    def gemm2(self) -> tuple[int, int]:
        """(N, K) of the down GEMM: cache3 = cache2 @ w2[e].T."""
        return (self.hidden, self.inter)


# Live Qwen3-Omni configs (mstar/model/qwen3_omni/config.py).
SHAPES = {
    "thinker": MoEShape("thinker", hidden=2048, inter=768, num_experts=128, top_k=8),
    "talker": MoEShape("talker", hidden=1024, inter=384, num_experts=128, top_k=8),
}

# Decode-heavy first, then prefill-sized batches.
DEFAULT_M = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _gpu_state(index: int = 0) -> dict[str, Any]:
    """SM clock / power / throttle reasons, so a number carries its context."""
    q = "clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        sm, pw, tmp, thr = (f.strip() for f in out.split(","))
        return {"sm_mhz": int(sm), "power_w": float(pw), "temp_c": int(tmp), "throttle": thr}
    except Exception:  # noqa: BLE001 -- telemetry is best-effort
        return {}


@dataclass
class Timing:
    """Result of :func:`bench`, in microseconds."""

    us: float
    windows: list[float] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover -- human output
        sm = self.state.get("sm_mhz", "?")
        return f"{self.us:8.2f} us  (sm {sm} MHz)"


def bench(
    fn: Callable[[], Any],
    warm_s: float = 0.25,
    windows: int = 5,
    iters: int = 20,
    index: int = 0,
    graph: bool = True,
) -> Timing:
    """Time ``fn`` under a settled clock.

    ``warm_s`` seconds of continuous back-to-back launches first, then
    ``windows`` measurement windows of ``iters`` launches each.  The first
    window is dropped (it still carries some ramp) and the median of the
    rest is reported.

    ``graph=True`` (the default) captures ``iters`` invocations into a CUDA
    graph and times replays of it.  This is not an optimization of the
    measurement, it is the only way to get the number we want: Triton's
    Python-side launch path costs tens of microseconds of *CPU* time, which
    for a decode-sized MoE GEMM is larger than the kernel itself, so an
    eager loop measures the launcher and reports the same time for every
    tile config.  Graph replay also matches how the fused MoE actually runs
    in mstar (``mstar/engine/cuda_graph_runner.py``).  Pass ``graph=False``
    only to deliberately measure the eager launch path.
    """
    fn()
    torch.cuda.synchronize()

    if graph:
        # Capture on a side stream, as torch requires.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        replay: Callable[[], Any] = g.replay
    else:
        def replay() -> None:
            for _ in range(iters):
                fn()

    replay()
    torch.cuda.synchronize()

    deadline = time.perf_counter() + warm_s
    while time.perf_counter() < deadline:
        replay()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(windows):
        start.record()
        replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)

    state = _gpu_state(index)
    body = samples[1:] or samples
    return Timing(us=statistics.median(body), windows=samples, state=state)


def tflops(us: float, m: int, n: int, k: int) -> float:
    """Effective TFLOP/s for one ``m x n x k`` GEMM taking ``us`` micro-seconds."""
    return 2.0 * m * n * k / (us * 1e-6) / 1e12


# H100 SXM HBM3 spec sheet peak.  Real streaming kernels top out near 85% of it.
HBM_PEAK_GBS = 3350.0


def touched_experts(topk_ids: torch.Tensor, num_experts: int) -> int:
    """How many experts receive at least one row for this routing."""
    counts = torch.bincount(topk_ids.reshape(-1).to(torch.int64), minlength=num_experts)
    return int(counts.gt(0).sum())


def weight_bytes(shape, touched: int, elem: int = 2) -> int:
    """Bytes of expert weight that must cross HBM at least once.

    Both GEMMs of a dispatch read every touched expert's ``w1`` and ``w2``
    exactly once, and at decode those weights dwarf the activations -- which
    is why the roofline here is a bandwidth roofline, not a FLOP roofline.
    """
    per_expert = (2 * shape.inter * shape.hidden + shape.hidden * shape.inter) * elem
    return touched * per_expert


def roofline_us(shape, touched: int, elem: int = 2, peak_gbs: float = HBM_PEAK_GBS) -> float:
    """Lower bound, in microseconds, on one dispatch's two GEMMs."""
    return weight_bytes(shape, touched, elem) / (peak_gbs * 1e9) * 1e6
