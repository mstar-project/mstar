"""``QuantConfig``: one node's linear quantization, as a deployment sets it."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal

# activation schemes QuantLinear runs; None is weight-only
ACTIVATION_SCHEMES = ("dynamic_token", None)
# in the plan but not built yet; named so the error says so
_PLANNED_ACTIVATIONS = ("dynamic_block128", "static")


@dataclass(frozen=True)
class QuantConfig:
    # format key (see formats.FORMATS)
    weights: str
    weight_scale: str = "amax"
    # "dynamic_token" or None (weight-only)
    activations: str | None = None
    activation_scale: str = "amax"
    # fnmatch globs on module names, added to the node's quant_skip
    skip: tuple[str, ...] = ()
    source: Literal["online", "checkpoint"] = "online"
    scales_path: str | None = None
    # pin a kernel; default: the first the device supports
    kernel: str | None = None

    def __post_init__(self):
        from mstar.engine.quant.formats import FORMATS
        from mstar.engine.quant.kernels import KERNELS
        from mstar.engine.quant.scale_rules import get_scale_rule

        if isinstance(self.skip, str):
            raise TypeError(f"skip is a tuple of globs, not the string {self.skip!r}")
        object.__setattr__(self, "skip", tuple(self.skip))
        if self.weights not in FORMATS:
            raise ValueError(f"unknown quant format {self.weights!r}; registered: {sorted(FORMATS)}")
        get_scale_rule(self.weight_scale)
        get_scale_rule(self.activation_scale)
        if self.activations in _PLANNED_ACTIVATIONS:
            raise NotImplementedError(f"activations {self.activations!r} is not implemented yet")
        if self.activations not in ACTIVATION_SCHEMES:
            raise ValueError(f"unknown activations {self.activations!r}; accepted: {list(ACTIVATION_SCHEMES)}")
        if self.activation_scale != "amax":
            # dynamic scales are computed per call in the kernel, always amax
            raise ValueError("activation_scale other than amax needs static activations")
        if self.source == "checkpoint":
            raise NotImplementedError("source 'checkpoint' needs a checkpoint reader (step 7)")
        if self.source != "online":
            raise ValueError(f"unknown quant source {self.source!r}; accepted: ['online', 'checkpoint']")
        if self.scales_path is not None:
            raise NotImplementedError("scales_path needs static activations, not implemented yet")
        if self.kernel is not None and self.kernel not in KERNELS:
            raise ValueError(f"unknown quant kernel {self.kernel!r}; registered: {sorted(KERNELS)}")


_YAML_KEYS = {f.name for f in fields(QuantConfig)}


def parse_yaml_quant(node_name: str, block: Mapping[str, Any] | None) -> QuantConfig | None:
    """``nodes.<name>.precision.quant``; null removes the model's."""
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"nodes.{node_name}.precision.quant is a mapping or null, not {block!r}")
    unknown = sorted(block.keys() - _YAML_KEYS)
    if unknown:
        raise ValueError(
            f"nodes.{node_name}.precision.quant: unknown key(s) {unknown}; accepted: {sorted(_YAML_KEYS)}"
        )
    values = dict(block)
    if not values.get("weights"):
        raise ValueError(f"nodes.{node_name}.precision.quant: weights (a format) is required")
    if "skip" in values:
        values["skip"] = tuple(values["skip"] or ())
    try:
        return QuantConfig(**values)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise type(exc)(f"nodes.{node_name}.precision.quant: {exc}") from exc
