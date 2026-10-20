"""Size sweeps, percentiles and the latency(S) = fixed + S / bandwidth fit.

Pure functions, so the tool's numbers can be re-derived from its JSON.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

_UNITS = {"": 1, "B": 1, "K": 1 << 10, "KB": 1 << 10, "KIB": 1 << 10,
          "M": 1 << 20, "MB": 1 << 20, "MIB": 1 << 20,
          "G": 1 << 30, "GB": 1 << 30, "GIB": 1 << 30}


def parse_size(text: str) -> int:
    """``64``, ``4K``, ``16KiB``, ``4M`` -> bytes (binary units)."""
    m = re.fullmatch(r"\s*(\d+)\s*([A-Za-z]*)\s*", text)
    if m is None or m.group(2).upper() not in _UNITS:
        raise ValueError(f"not a size: {text!r} (use e.g. 64, 4K, 1M)")
    return int(m.group(1)) * _UNITS[m.group(2).upper()]


def parse_sizes(spec: str) -> list[int]:
    """``64:4M:x4`` (log-spaced, factor 4) or ``64,1K,1M`` (explicit)."""
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3 or not parts[2].lower().startswith("x"):
            raise ValueError(f"range sizes look like 64:4M:x4, got {spec!r}")
        lo, hi = parse_size(parts[0]), parse_size(parts[1])
        factor = float(parts[2][1:])
        if lo <= 0 or hi < lo or factor <= 1:
            raise ValueError(f"need 0 < min <= max and factor > 1, got {spec!r}")
        sizes, s = [], float(lo)
        while s <= hi * (1 + 1e-9):
            sizes.append(int(round(s)))
            s *= factor
        return sizes
    sizes = [parse_size(p) for p in spec.split(",") if p.strip()]
    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError(f"sizes must be positive, got {spec!r}")
    return sorted(set(sizes))


def human_size(n: float) -> str:
    for unit, scale in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= scale:
            v = n / scale
            return f"{v:.0f} {unit}" if v == int(v) else f"{v:.1f} {unit}"
    return f"{n:.0f} B"


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


@dataclass
class LinearFit:
    fixed_s: float
    bandwidth_bps: float  # inf when latency does not grow with size
    r2: float

    def predict(self, size: float) -> float:
        return self.fixed_s + (size / self.bandwidth_bps if math.isfinite(self.bandwidth_bps) else 0.0)

    def to_json(self) -> dict:
        d = asdict(self)
        if not math.isfinite(d["bandwidth_bps"]):
            d["bandwidth_bps"] = None
        return d


def fit_latency(sizes: list[float], latencies_s: list[float]) -> LinearFit | None:
    """Least squares for ``latency = fixed + size / bandwidth``, weighted by
    1 / latency^2 so small sizes (which set the fixed cost) count as much as
    large ones (which set the bandwidth). None for fewer than two points."""
    pts = [(s, y) for s, y in zip(sizes, latencies_s, strict=True)
           if y > 0 and math.isfinite(y)]
    if len(pts) < 2:
        return None
    w = [1.0 / (y * y) for _, y in pts]
    sw = sum(w)
    mx = sum(wi * s for wi, (s, _) in zip(w, pts, strict=True)) / sw
    my = sum(wi * y for wi, (_, y) in zip(w, pts, strict=True)) / sw
    sxx = sum(wi * (s - mx) ** 2 for wi, (s, _) in zip(w, pts, strict=True))
    sxy = sum(wi * (s - mx) * (y - my) for wi, (s, y) in zip(w, pts, strict=True))
    slope = sxy / sxx if sxx > 0 else 0.0
    if slope <= 0:
        # Latency that does not grow with size: the flat model, so the fixed
        # cost is not an intercept extrapolated from a negative slope.
        slope = 0.0
    fixed = my - slope * mx
    syy = sum(wi * (y - my) ** 2 for wi, (_, y) in zip(w, pts, strict=True))
    ss_res = sum(wi * (y - (fixed + slope * s)) ** 2 for wi, (s, y) in zip(w, pts, strict=True))
    r2 = 1.0 - ss_res / syy if syy > 0 else 1.0
    bandwidth = 1.0 / slope if slope > 0 else math.inf
    return LinearFit(fixed_s=fixed, bandwidth_bps=bandwidth, r2=r2)


def fit_crossover(a: LinearFit, b: LinearFit) -> float | None:
    """The size where the two fitted lines meet, or None if they don't at a
    positive size (one path wins everywhere)."""
    inv_a = 0.0 if not math.isfinite(a.bandwidth_bps) else 1.0 / a.bandwidth_bps
    inv_b = 0.0 if not math.isfinite(b.bandwidth_bps) else 1.0 / b.bandwidth_bps
    if inv_a == inv_b:
        return None
    size = (b.fixed_s - a.fixed_s) / (inv_a - inv_b)
    return size if size > 0 else None


def measured_crossover(sizes: list[int], lat_a: list[float], lat_b: list[float]) -> int | None:
    """The first measured size at which the other path becomes strictly
    cheaper, or None if it never does. Ties change nothing."""
    first = None
    for s, x, y in zip(sizes, lat_a, lat_b, strict=True):
        if x == y:
            continue
        a_cheaper = x < y
        if first is None:
            first = a_cheaper
        elif a_cheaper != first:
            return s
    return None
