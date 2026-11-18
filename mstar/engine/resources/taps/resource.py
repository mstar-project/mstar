"""Taps: a node's residual stream at a few of its layers, read by another
component of the same node in the same forward (a drafter's context features,
say).

``TapResource`` holds, for every row the step computes, the residual stream
entering each listed layer: ``run_layers`` writes it (``write``; layer
``num_layers`` is the stream leaving the last layer, before the final norm,
which is not transformers' post-norm ``hidden_states[-1]``), and the reader
takes it back later in the same forward (``read``: ``[rows, len(layers),
hidden]``, row i the step's packed row i). Only the rows a step computes are
tapped and nothing is kept across steps, so a prefix hit or a verify result
corrects nothing here.

A captured step writes into a buffer of its capture slot, sized for the slot's
widest bucket, so a replay writes and reads the addresses its graph recorded;
each slot has its own, so a step planned ahead never shares one with the step in
flight. An eager step's buffer is made by its first write, on the forward's
stream, and lives until the step after it. The resource plans nothing on the
device and stages nothing from the host. A target writes through `step_taps`,
since the resource has no abort hook: a tap step that failed before its commit
leaves it armed.
"""

from __future__ import annotations

import torch

from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo, Resource
from mstar.engine.resources.step import StepContext
from mstar.engine.resources.taps.config import TapSpec, TapStep


class TapResource(Resource):
    # keeps nothing per request: a prefix hit's step writes and reads only the rows it computes
    prefix_skip_safe = True

    def __init__(self, layers: tuple[int, ...], hidden_size: int, dtype: torch.dtype, device: torch.device):
        if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
            raise ValueError(f"TapSpec.layers must name distinct layers >= 0; got {layers!r}")
        if hidden_size <= 0:
            raise ValueError("TapSpec.hidden_size must be set")
        self.layers = tuple(int(layer) for layer in layers)
        # layer -> its column, for the target's per-layer check
        self.columns = {layer: i for i, layer in enumerate(self.layers)}
        self.hidden_size = int(hidden_size)
        self.dtype = dtype
        self._device = torch.device(device)
        # capture slot -> its buffer, [rows, taps, hidden]
        self._static: dict[int, torch.Tensor] = {}
        self._sealed = False
        self._rows = 0
        self._current: torch.Tensor | None = None
        # a step that declared taps is between its plan and its commit: a target that runs in
        # a step with none (a plain prefill on the same node) writes nothing
        self._armed = False

    def resident_refusal(self) -> str | None:
        # rewritten every step and keyed by nothing per request: a resident step plans it as any other
        return None

    @classmethod
    def build(cls, spec: TapSpec, info: EngineResourceInfo) -> TapResource:
        dtype = spec.dtype or info.compute_dtype or info.kv_dtype
        return cls(spec.layers, spec.hidden_size, dtype, info.device)

    def build_accelerator_graph_buffers(self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int) -> None:
        del max_bs, max_seq_len
        widest: dict[int, int] = {}
        for spec in slots:
            widest[spec.slot] = max(widest.get(spec.slot, 0), spec.num_tokens)
        for slot, rows in widest.items():
            have = self._static.get(slot)
            if have is not None and have.shape[0] >= rows:
                continue
            if have is not None and self._sealed:
                # a graph captured against the old one would read it in place
                raise RuntimeError(
                    f"taps: slot {slot}'s buffer holds {have.shape[0]} rows; a capture after sealing asks {rows}"
                )
            # every runner sizes its buffers before any captures: grow until then
            self._static[slot] = torch.zeros(
                rows, len(self.layers), self.hidden_size, dtype=self.dtype, device=self._device,
            )

    def seal_capture_buffers(self) -> None:
        self._sealed = True

    def plan(self, step: TapStep, ctx: StepContext) -> None:
        rows = sum(segment.span for segment in step.segments or ())
        lease = ctx.slot_lease
        if lease is None:
            self._current = None
        else:
            buffer = self._static.get(lease.slot)
            if buffer is None or buffer.shape[0] < rows:
                raise RuntimeError(
                    f"taps: a captured step of {rows} rows on slot {lease.slot}, which holds "
                    f"{0 if buffer is None else buffer.shape[0]}"
                )
            self._current = buffer[:rows]
        self._rows = rows
        self._armed = True

    def commit(self, step: TapStep, ctx: StepContext) -> None:
        del step, ctx
        self._armed = False
        self._current = None

    @torch.compiler.disable
    def write(self, layer: int, hidden: torch.Tensor) -> None:
        """The stream entering ``layer`` for the step's rows (``[rows, hidden]``, or the
        first rows of a wider tensor). The target calls it for each layer in ``columns``; a
        step that declared no taps writes nothing."""
        if not self._armed:
            return
        current = self._current
        if current is None:
            current = self._current = torch.empty(
                self._rows, len(self.layers), self.hidden_size, dtype=self.dtype, device=hidden.device,
            )
        rows = current.shape[0]
        if hidden.shape[0] < rows:
            raise ValueError(f"taps: layer {layer}'s stream has {hidden.shape[0]} rows for a step of {rows}")
        current[:, self.columns[layer]].copy_(hidden[:rows])

    @torch.compiler.disable
    def read(self) -> torch.Tensor:
        """``[rows, len(layers), hidden]``: this step's taps, once the target has run."""
        if not self._armed or self._current is None:
            raise RuntimeError("taps: read in a step that declared none, or before the target wrote them")
        return self._current

    def cleanup(self):
        self._static.clear()
        self._current = None


def step_taps(engine_inputs, key: str) -> TapResource | None:
    """The node's taps under ``key`` when this step declared a `TapStep` for
    them, else None. A target writes through this: a tap step whose forward
    failed before its commit leaves the resource armed (`Resource` has no
    abort hook), and a later step that declared no taps must not write into
    it. A drafter reads them only in the steps that declare them."""
    step = engine_inputs.step
    if step is None or not isinstance(step.get(key), TapStep):
        return None
    return engine_inputs.resources.get(key)
