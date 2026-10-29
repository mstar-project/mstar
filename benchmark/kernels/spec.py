"""What the harness measures for one op.

A ``KernelSpec`` names the op, the shapes and dtypes to run (harvested from
served steps, ``shapes/*.json``), how to build inputs for a shape, the
reference and the bar a candidate must meet against it (``mstar.kernels.
numerics``), the bytes and matrix FLOPs a shape costs (for its roofline
floor), whether the op must be graph-capturable, and the candidates to try on
a device. A spec for a registry op takes its inputs, reference and tolerance
from the ``KernelOp`` (``KernelSpec.for_op``).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from benchmark.kernels.candidates import Candidate
from mstar.kernels import registry
from mstar.kernels.numerics import Tolerance

Shape = Mapping[str, int]
SHAPES_DIR = Path(__file__).resolve().parent / "shapes"


@dataclasses.dataclass(frozen=True, eq=False)
class KernelSpec:
    name: str
    shapes: tuple[Shape, ...]
    dtypes: tuple[torch.dtype, ...]
    # (shape, dtype, device, generator) -> the op's arguments
    inputs: Callable[[Shape, torch.dtype, torch.device, torch.Generator | None], tuple]
    # the op's semantics; the oracle evaluates it in fp64 on the CPU, or in
    # fp32 on the device where `oracle_on_device` (operands too big for the host)
    reference: Callable[..., Any]
    tolerance: Tolerance
    bytes_moved: Callable[[Shape, torch.dtype], float]
    flops: Callable[[Shape, torch.dtype], float]
    # (facts, budget per family, earlier winners' knobs by family) -> candidates
    candidates: Callable[..., list[Candidate]]
    mutated: tuple[int, ...] = ()
    capture: bool = True
    oracle_on_device: bool = False
    # names a shape's class (what a tuned config is keyed by)
    shape_class: Callable[[Shape], str | None] = lambda shape: None

    @classmethod
    def for_op(cls, op_name: str, **fields) -> KernelSpec:
        """A spec for registry op ``op_name``, its inputs, reference, tolerance,
        mutated arguments, dtypes and shape classes taken from the op."""
        registry.load_all()
        op = registry.get(op_name)
        if op.shapes is None:
            raise ValueError(f"{op.qualname} declares no ShapePolicy")
        fields.setdefault("name", op.name)
        fields.setdefault("shapes", tuple(op.shapes.warm))
        fields.setdefault("dtypes", op.shapes.dtypes)
        fields.setdefault("inputs", op.shapes.inputs)
        fields.setdefault("reference", op.reference)
        fields.setdefault("tolerance", op.tolerance)
        fields.setdefault("mutated", op.mutated_args())
        fields.setdefault("shape_class", op.shapes.shape_class)
        return cls(**fields)


def load_shapes(path: str | Path, op: str) -> tuple[Shape, ...]:
    """The shapes for ``op`` in a harvested shape file (``trace/harvest.py``),
    deduplicated in their order."""
    with open(path) as f:
        data = json.load(f)
    seen: list[dict[str, int]] = []
    for entry in data["ops"].get(op, []):
        shape = {k: int(v) for k, v in entry.items() if k != "count"}
        if shape not in seen:
            seen.append(shape)
    return tuple(seen)


def shape_product(**dims: Sequence[int]) -> tuple[Shape, ...]:
    """Every combination of the given dimension values, in order."""
    shapes: list[dict[str, int]] = [{}]
    for name, values in dims.items():
        shapes = [{**s, name: v} for s in shapes for v in values]
    return tuple(shapes)
