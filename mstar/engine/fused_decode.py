"""Captured decode buckets that replay a submodule's fused forward instead of its compiled graph.

A submodule declares ``FusedDecodeConfig`` for a walk and batch sizes and supplies the fused forward
(``NodeSubmodule.get_fused_decode_configs``). The engine captures that forward for each declared
bucket right after the bucket's graph, from the same static inputs, dummy rows and plan, so the
replay reads the same buffers and the worker's stream and event protocol doesn't change. A bucket
keeps its graph when the capture fails or the parity step below disagrees with the modules.

Fused kernels sync their stages through counters and accumulators in global memory. Those are
``FusedScratch``, one set per captured (bucket, slot), engine-owned so a submodule holds no
counters; the kernels leave them zero for the next launch.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from mstar.engine.resources.base import Resource

logger = logging.getLogger(__name__)

# Final hidden state and every state write: one 4-layer block of Qwen3.5's kernels is 1.3% from
# the eager modules, since the kernels sum fp32 partials in a free order.
PARITY_REL_L2 = 0.02


class FusedScratch:
    """Zero-filled device buffers by name, for one captured (bucket, slot).

    Allocated on a warm-up's first call, never under capture, so the graph records their addresses
    and no fill; the kernels return them to zero themselves."""

    def __init__(self, device: torch.device):
        self._device = device
        self._buffers: dict[str, torch.Tensor] = {}

    def zeros(self, name: str, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        found = self._buffers.get(name)
        if found is not None and found.shape == torch.Size(shape) and found.dtype == dtype:
            return found
        if self._device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"fused scratch {name!r} {tuple(shape)} first asked for under capture")
        found = self._buffers[name] = torch.zeros(shape, dtype=dtype, device=self._device)
        return found

    def buffers(self) -> Mapping[str, torch.Tensor]:
        return self._buffers


def rel_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    """``|value - reference| / |reference|`` in fp32; 0 for two zero tensors."""
    ref = reference.float()
    diff = (value.float() - ref).norm().item()
    scale = ref.norm().item()
    if scale == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / scale


def _row_state(resources: Mapping[str, Resource], rids: Sequence[int]) -> dict[str, dict[str, torch.Tensor]]:
    states = {key: resource.row_state(rids) for key, resource in resources.items()}
    return {key: state for key, state in states.items() if state}


def _load(resources: Mapping[str, Resource], rids: Sequence[int], states) -> None:
    for key, state in states.items():
        resources[key].load_row_state(rids, state)


def fused_parity(
    submodule, graph_walk: str, engine_inputs: Any, static_inputs: Mapping[str, Any],
    resources: Mapping[str, Resource], rids: Sequence[int],
) -> dict[str, float]:
    """Rel L2 of the fused path against the modules over one planned step of ``rids``: the final
    hidden state, and each resource's state of the rows after the step.

    Eager, on the plan the capture left staged. One step through the modules first, so the rows
    hold the state a step leaves rather than a fresh slot's zeros; then both paths run from a
    snapshot of that, and the rows get back what they held before."""
    before = _row_state(resources, rids)

    def hidden(fused: bool) -> torch.Tensor:
        return submodule.decode_hidden(graph_walk, engine_inputs, fused=fused, **static_inputs)

    try:
        hidden(False)
        start = _row_state(resources, rids)
        reference = hidden(False).clone()
        written = _row_state(resources, rids)
        _load(resources, rids, start)
        fused = hidden(True).clone()
        fused_written = _row_state(resources, rids)
    finally:
        _load(resources, rids, before)
    errors = {"hidden": rel_l2(fused, reference)}
    for key, state in written.items():
        for name, tensor in state.items():
            errors[f"{key}.{name}"] = rel_l2(fused_written[key][name], tensor)
    return errors


def parity_failure(errors: Mapping[str, float]) -> str | None:
    """Which comparison is past ``PARITY_REL_L2``, or None."""
    worst = max(errors, key=errors.get)
    if errors[worst] < PARITY_REL_L2:
        return None
    return f"{worst} rel L2 {errors[worst]:.4f} (limit {PARITY_REL_L2})"


def format_errors(errors: Mapping[str, float]) -> str:
    return ", ".join(f"{name} {value:.4f}" for name, value in errors.items())
