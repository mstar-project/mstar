"""Model-agnostic kernel-level headroom analysis for torch.compile'd models.

Pipeline: parse an Inductor debug dump (exact per-kernel tensor traffic +
source provenance), join with measured per-kernel times from a profiler run,
score each kernel against a GPU spec (speed-of-light roofline), and bucket the
gap into kernel-quality / data-movement-elimination / fusion headroom.

Nothing in this package imports mstar or knows about a specific model — a
driver script supplies the dump dir, the measured times, and (optionally) an
eager-kernel attribution map captured during warmup.
"""

from .eager_attr import profile_eager_attribution
from .hardware import GpuSpec, resolve_gpu
from .inductor_dump import Subgraph, parse_dump
from .report import Report, build_report, render_text

__all__ = [
    "GpuSpec",
    "Report",
    "Subgraph",
    "build_report",
    "parse_dump",
    "profile_eager_attribution",
    "render_text",
    "resolve_gpu",
]
