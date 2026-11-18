"""Tap declarations: which layers' residual stream a node keeps for the step
(`TapSpec`), and which steps write it (`TapStep`). See `taps.resource`."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mstar.engine.resources.base import Resource
from mstar.engine.resources.spec import NodeResourceSpec, ResourceCapabilities
from mstar.engine.resources.step import ResourceStep


@dataclass(frozen=True)
class TapStep(ResourceStep):
    """A step that writes the node's taps: one segment a row, its span the row's tokens."""


@dataclass
class TapSpec(NodeResourceSpec):
    """The layers whose input a node taps: the residual stream entering each of
    ``layers``, ``hidden_size`` wide, in ``dtype`` (None: the node's compute
    dtype, or its KV dtype outside autocast)."""
    layers: tuple[int, ...] = ()
    hidden_size: int = 0
    dtype: torch.dtype | None = None

    @property
    def resource_class(self) -> type[Resource]:
        from mstar.engine.resources.taps.resource import TapResource

        return TapResource

    def resource_capabilities(self, ctx) -> ResourceCapabilities:
        # rewritten every step: a rejected row's taps die with the step
        return ResourceCapabilities.granting_all()
