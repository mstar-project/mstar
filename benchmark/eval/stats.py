"""Summary statistics shared by every suite.

Per-request values inside one repeat are summarized with percentiles; one
number per repeat is then summarized across repeats as mean +- sample std.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def finite(values: Iterable[float | None]) -> list[float]:
    """The values that are real numbers (drops None, NaN and inf)."""
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def percentile(values: Iterable[float | None], q: float) -> float | None:
    """Linear-interpolated percentile, ``q`` in [0, 100]; None when empty."""
    xs = sorted(finite(values))
    if not xs:
        return None
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    k = (len(xs) - 1) * q / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(values: Iterable[float | None]) -> float | None:
    xs = finite(values)
    return sum(xs) / len(xs) if xs else None


def summarize(values: Iterable[float | None]) -> dict:
    """mean, sample std (0 for a single value), min, max and n of the finite values."""
    xs = finite(values)
    if not xs:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    m = sum(xs) / len(xs)
    std = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return {"mean": m, "std": std, "min": min(xs), "max": max(xs), "n": len(xs)}


def label_agreement(a: list, b: list) -> dict:
    """Raw agreement and Cohen's kappa of two judges' labels on the same items
    (how sglang-omni reports Qwen vs GPT-4o behavior labels)."""
    if len(a) != len(b):
        raise ValueError(f"{len(a)} vs {len(b)} labels")
    n = len(a)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None}
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    labels = set(a) | set(b)
    p_e = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    kappa = 1.0 if p_e == 1.0 else (p_o - p_e) / (1.0 - p_e)
    return {"n": n, "agreement": p_o, "kappa": kappa}


def confusion_matrix(a: list, b: list, labels: list | None = None) -> dict:
    """Counts of each (``a`` label, ``b`` label) over the same items: one row per ``a``
    label, one column per ``b`` label, in ``labels`` order (default: sorted union)."""
    if len(a) != len(b):
        raise ValueError(f"{len(a)} vs {len(b)} labels")
    labels = list(labels) if labels is not None else sorted(set(a) | set(b))
    index = {lab: i for i, lab in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b, strict=True):
        matrix[index[x]][index[y]] += 1
    return {"labels": labels, "matrix": matrix}


def fmt_mean_std(summary: dict | None, digits: int = 2, scale: float = 1.0) -> str:
    """``"12.30 +- 0.40"`` (or ``"12.30"`` for one repeat, ``"n/a"`` for none)."""
    if not summary or summary.get("mean") is None:
        return "n/a"
    m = summary["mean"] * scale
    if summary.get("n", 0) <= 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} +- {summary['std'] * scale:.{digits}f}"
