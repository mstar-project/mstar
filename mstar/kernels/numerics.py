"""The parity bar a kernel meets against its op's reference.

The reference runs on the CPU in fp64 (``reference_outputs``), and a kernel's
output is compared with it under the op's ``Tolerance``:

- ``rounded_once``: the kernel computes in fp32 or wider and rounds each
  output once, to nearest even. Every element is within one ulp of the fp64
  result rounded to the output dtype, and the relative norm distance from
  that rounded result stays under ``rel_norm``. A truncating store, half an
  ulp off on average, fails the norm bound though each element passes.
- ``accumulated``: a long fp32 reduction rounded once (a GEMM), where the
  summation order moves results by more than an ulp: each element within
  ``rtol`` of the reference plus ``atol_scale`` times its largest magnitude.
- ``exact``: bit for bit, for kernels that claim it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import torch


@dataclasses.dataclass(frozen=True)
class Tolerance:
    kind: str = "rounded_once"
    # rounded_once only
    rel_norm: float | None = 1e-4
    rtol: float = 0.0
    atol_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("rounded_once", "accumulated", "exact"):
            raise ValueError(f"unknown tolerance kind {self.kind!r}")

    @classmethod
    def rounded_once(cls, rel_norm: float = 1e-4) -> Tolerance:
        return cls("rounded_once", rel_norm=rel_norm)

    @classmethod
    def accumulated(cls, rtol: float = 1.6e-2, atol_scale: float = 1e-2) -> Tolerance:
        return cls("accumulated", rel_norm=None, rtol=rtol, atol_scale=atol_scale)

    @classmethod
    def exact(cls) -> Tolerance:
        return cls("exact", rel_norm=None)


@dataclasses.dataclass(frozen=True)
class Comparison:
    """How one output compares with the reference."""
    ok: bool
    numel: int
    # elements outside the elementwise bound
    mismatched: int
    max_abs: float
    # in ulps of the output dtype, for rounded_once
    max_ulps: float
    rel_norm: float

    def summary(self) -> str:
        verdict = "ok" if self.ok else "FAIL"
        return (f"{verdict}: {self.mismatched}/{self.numel} outside the bound, max abs {self.max_abs:.3g}, "
                f"max {self.max_ulps:.2f} ulp, rel norm {self.rel_norm:.2e}")


def ulp(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The spacing of ``dtype`` at each of ``values`` (fp64), the gap one
    rounding can open."""
    finfo = torch.finfo(dtype)
    _, exponent = torch.frexp(values.abs())
    spacing = torch.ldexp(torch.full_like(values, finfo.eps), exponent - 1)
    # zero and the subnormal range share the smallest subnormal's spacing
    subnormal = finfo.eps * finfo.tiny
    return torch.where(values == 0, torch.full_like(values, subnormal), spacing.clamp_min(subnormal))


def compare(out: torch.Tensor, ref: torch.Tensor, tol: Tolerance) -> Comparison:
    """``out``, a kernel's output, against ``ref``, the reference's in fp64."""
    if out.shape != ref.shape:
        raise ValueError(f"output shape {tuple(out.shape)} != reference {tuple(ref.shape)}")
    got = out.detach().to("cpu", torch.float64)
    ref = ref.detach().to("cpu", torch.float64)
    numel = got.numel()
    if numel == 0:
        return Comparison(True, 0, 0, 0.0, 0.0, 0.0)
    if not out.dtype.is_floating_point:
        mismatched = int((got != ref).sum())
        max_abs = float((got - ref).abs().max())
        return Comparison(mismatched == 0, numel, mismatched, max_abs, 0.0, 0.0)
    # the best the output dtype can hold
    rounded = ref.to(out.dtype).to(torch.float64)
    err = (got - rounded).abs()
    nonfinite = int((torch.isfinite(got) != torch.isfinite(rounded)).sum())
    finite = torch.isfinite(rounded) & torch.isfinite(got)
    err = torch.where(finite, err, torch.zeros_like(err))
    max_abs = float(err.max())
    ulps = err / ulp(rounded, out.dtype)
    max_ulps = float(ulps.max())
    # against the rounded reference: a correctly rounded output sits at zero
    # and a truncating store near half an ulp per element
    zero = torch.zeros_like(rounded)
    ref_norm = float(torch.linalg.vector_norm(torch.where(finite, rounded, zero)))
    rel_norm = float(torch.linalg.vector_norm(torch.where(finite, got - rounded, zero))) / max(
        ref_norm, torch.finfo(torch.float64).tiny)
    if tol.kind == "exact":
        bad = (got != rounded) & finite
    elif tol.kind == "rounded_once":
        bad = ulps > 1.0
    else:
        bound = tol.rtol * rounded.abs() + tol.atol_scale * float(ref.abs().max())
        bad = err > bound
    mismatched = int(bad.sum()) + nonfinite
    ok = mismatched == 0 and (tol.kind != "rounded_once" or rel_norm <= tol.rel_norm)
    return Comparison(ok, numel, mismatched, max_abs, max_ulps, rel_norm)


def _to_fp64_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        moved = value.detach().to("cpu")
        return moved.to(torch.float64) if moved.dtype.is_floating_point else moved.clone()
    if isinstance(value, (list, tuple)):
        return type(value)(_to_fp64_cpu(v) for v in value)
    return value


def reference_outputs(
    reference: Callable[..., Any], args: Sequence[Any], mutated: Sequence[int] = (),
) -> list[torch.Tensor]:
    """Run ``reference`` on fp64 CPU copies of ``args``: its returned tensors,
    then the arguments at the ``mutated`` positions after the call."""
    args64 = [_to_fp64_cpu(a) for a in args]
    result = reference(*args64)
    return [*as_tensors(result), *(args64[i] for i in mutated)]


def as_tensors(result: Any) -> list[torch.Tensor]:
    if result is None:
        return []
    if isinstance(result, torch.Tensor):
        return [result]
    return [t for t in result if isinstance(t, torch.Tensor)]
