"""What a node takes on each input signal, applied once at ingestion.

A node declares ``input_specs: {signal: TensorSpec}``. The engine resolves
them against the node's ``NodePrecision`` at load and applies them to each
request's routed tensors before ``prepare_inputs``: after transport, so a
producer's tensor keeps the dtype it was written in until the consumer that
wants another one casts it, and the node itself casts nothing by hand.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.engine.precision import NodePrecision
from mstar.engine.resources.spec import require_unscaled_dtype

logger = logging.getLogger(__name__)

OnMismatch = Literal["cast", "warn", "raise"]
_ON_MISMATCH = ("cast", "warn", "raise")


@dataclass(frozen=True)
class TensorSpec:
    # "param": the node's param_dtype, so a spec follows a precision change;
    # None: no dtype is checked
    dtype: torch.dtype | Literal["param"] | None
    # None: cast a floating dtype, raise on any other (token ids are checked,
    # never cast)
    on_mismatch: OnMismatch | None = None

    def __post_init__(self):
        if isinstance(self.dtype, str) and self.dtype != "param":
            raise ValueError(f"TensorSpec dtype is a torch dtype, 'param' or None, not {self.dtype!r}")
        if self.on_mismatch is not None and self.on_mismatch not in _ON_MISMATCH:
            raise ValueError(f"TensorSpec on_mismatch is one of {_ON_MISMATCH}, not {self.on_mismatch!r}")
        if isinstance(self.dtype, torch.dtype):
            if self.dtype.is_floating_point:
                require_unscaled_dtype(self.dtype, "an input spec's dtype")
            elif self.on_mismatch == "cast":
                raise ValueError(f"a {self.dtype} input is checked, not cast; use 'raise' or 'warn'")


@dataclass(frozen=True)
class ResolvedSpec:
    dtype: torch.dtype
    on_mismatch: OnMismatch


def resolve_input_specs(
    node_name: str, specs: Mapping[str, TensorSpec], precision: NodePrecision,
) -> dict[str, ResolvedSpec]:
    """``specs`` with ``"param"`` replaced by the node's param dtype. A spec
    without a dtype checks nothing and is left out."""
    resolved: dict[str, ResolvedSpec] = {}
    for signal, spec in specs.items():
        dtype = spec.dtype
        if dtype is None:
            continue
        if dtype == "param":
            if precision.param_dtype is None:
                raise ValueError(
                    f"node {node_name!r}: input {signal!r} takes the param dtype, but the "
                    "node keeps the dtype it was built in (param_dtype None); declare the dtype"
                )
            dtype = precision.param_dtype
        on_mismatch = spec.on_mismatch or ("cast" if dtype.is_floating_point else "raise")
        resolved[signal] = ResolvedSpec(dtype, on_mismatch)
    return resolved


_warned: set[tuple[str, str, torch.dtype]] = set()


def apply_input_specs(
    node_name: str, inputs: NameToTensorList, specs: Mapping[str, ResolvedSpec],
) -> NameToTensorList:
    """One request's routed tensors as the node's specs want them. ``inputs``
    is returned as is when nothing is cast, else a copy: the batch's own dict
    is left alone. Only floating dtypes are cast into one another; any other
    mismatch is a TypeError, or a one-time warning under ``"warn"``."""
    out = inputs
    for signal, spec in specs.items():
        tensors = inputs.get(signal)
        if not tensors or all(t.dtype == spec.dtype for t in tensors):
            continue
        cast = []
        for tensor in tensors:
            if tensor.dtype == spec.dtype:
                cast.append(tensor)
                continue
            if spec.on_mismatch == "warn":
                key = (node_name, signal, tensor.dtype)
                if key not in _warned:
                    _warned.add(key)
                    logger.warning(
                        "node %r: input %r is %s, its spec wants %s; passed through as is",
                        node_name, signal, tensor.dtype, spec.dtype,
                    )
                cast.append(tensor)
            elif spec.on_mismatch == "cast" and tensor.dtype.is_floating_point:
                cast.append(tensor.to(spec.dtype))
            else:
                raise TypeError(
                    f"node {node_name!r}: input {signal!r} is {tensor.dtype}, its spec wants {spec.dtype}"
                )
        if out is inputs:
            out = dict(inputs)
        out[signal] = cast
    return out


def ingest(
    submodule, inputs: NameToTensorList, precision: NodePrecision = NodePrecision(None), node_name: str = "node",
) -> NameToTensorList:
    """``inputs`` as the engine hands them to ``submodule.prepare_inputs``, for a
    caller that drives a node by hand (a test, an offline harness)."""
    return apply_input_specs(node_name, inputs, resolve_input_specs(node_name, submodule.input_specs, precision))
