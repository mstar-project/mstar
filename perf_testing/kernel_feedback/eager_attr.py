"""Attribute eager (uncompiled) CUDA kernels to source lines and shapes.

Kernels outside the Inductor graphs (the eager glue around graph breaks) have
no provenance in the debug dump, and under CUDA-graph *replay* the profiler
loses CPU<->kernel correlation entirely. The trick: profile the *warmup* pass
— the same code runs eagerly there, correlation works, and ``with_stack`` +
``record_shapes`` give each launching aten op a Python stack and input shapes.
We join to replay-time kernels by kernel name.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import prod


@dataclass
class EagerOpAttribution:
    op: str                                  # e.g. "aten::copy_"
    stack: list[str] = field(default_factory=list)   # project frames, innermost first
    input_shapes: list[list[int]] = field(default_factory=list)
    count: int = 0

    def traffic_bytes(self, elem_bytes: int = 2) -> int | None:
        """Read+write bytes assuming one pass over inputs and an equal-size
        output. Shapes come without dtypes, so ``elem_bytes`` is an assumption
        (default bf16) — callers should surface that caveat."""
        numels = [prod(s) for s in self.input_shapes if s]
        if not numels:
            return None
        return 2 * max(numels) * elem_bytes


def profile_eager_attribution(fn, project_markers: tuple[str, ...] = ("mstar",),
                              max_stack: int = 4) -> dict[str, list[EagerOpAttribution]]:
    """Run ``fn`` under torch.profiler and map each CUDA kernel name to the
    aten ops that launched it (with stacks filtered to project frames)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    try:
        # without verbose=True this torch build leaves FunctionEvent.stack empty
        experimental_config = torch._C._profiler._ExperimentalConfig(verbose=True)
    except Exception:
        experimental_config = None
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        experimental_config=experimental_config,
    ) as p:
        fn()

    per_kernel: dict[str, dict[tuple, EagerOpAttribution]] = defaultdict(dict)
    for ev in p.events():
        kernels = getattr(ev, "kernels", None)
        if not kernels or not ev.name.startswith("aten::"):
            continue
        raw = ev.stack or []

        def shorten(frame: str) -> str:
            for m in project_markers:
                idx = frame.rfind(f"/{m}/")
                if idx >= 0:
                    return frame[idx + 1:]
            return frame

        stack = [shorten(f) for f in raw if any(m in f for m in project_markers)][:max_stack]
        if not stack:   # keep the innermost real frames so the op is still locatable
            stack = [f for f in raw if not f.startswith("<built-in")][:3]
        shapes = [list(s) for s in (ev.input_shapes or []) if s]
        sig = (ev.name, tuple(tuple(s) for s in shapes), tuple(stack))
        for kern in kernels:
            slot = per_kernel[kern.name]
            if sig not in slot:
                slot[sig] = EagerOpAttribution(op=ev.name, stack=stack, input_shapes=shapes)
            slot[sig].count += 1
    return {name: sorted(attrs.values(), key=lambda a: -a.count) for name, attrs in per_kernel.items()}
