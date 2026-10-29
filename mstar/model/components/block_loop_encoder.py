"""A packed encoder node whose block loop replays as a captured region.

`PackedBlockLoopEncoder` takes one prompt per step, every image of it packed
into one call, and replays the tower's block loop as a piecewise region per
patch bucket; patch embed and the projection run eagerly around it (the
projection usually changes the row count, which a region's output view
cannot express). Eager, a ViT tower is launch-bound (~500 kernels, ~4 ms of
GPU work in a ~20 ms forward on Qwen3.5 9B), and one prompt per replay beats
batching them.

A model writes ``prepare_inputs``: the grid maths over `grids` (host values
from the walk's ``MEDIA_LAYOUT``, so no device tensor is read), returning ``NodeInputs``
with the forward's tensors in ``tensor_inputs``, the patch count as
``input_seq_len``, and ``resource_step_info`` mapping each of
``SEGMENT_LABELS`` to the spans that attend together under it (frames,
windows, ...), all over the same patches.

The tower (``self.model``) provides:

- ``embed(**tensors) -> (hidden, cos, sin)``: everything before the block
  loop, per patch
- ``encode(hidden, cos, sin) -> hidden``: the block loop
- ``project(encoded, **tensors) -> embeds``: everything after it

Each takes every tensor of ``tensor_inputs`` and ignores those it does not
use. ``config`` has ``hidden_size`` and ``head_dim``; the attention is
declared under ``mstar.model.keys.VISION_ATTN``. The node sends its
`MediaEncoder`'s ``output``.

A replay's attention plan holds at most the resource's
``max_segments_per_request`` segments under each label; a prompt with more
takes no lease and runs the block loop eagerly, however few its patches.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewisePackedConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.step import Segment, SlotLease, SubmoduleStep
from mstar.model.keys import VISION_ATTN
from mstar.model.multimodal import MEDIA_LAYOUT, MediaEncoder, media_grids
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

# The tower's block loop, as a piecewise capture region
VISION_BLOCK_LOOP = "vision_block_loop"


class PackedBlockLoopEncoder(NodeSubmodule):
    # Patch-count buckets, one prompt per replay; a larger prompt runs eagerly.
    BLOCK_LOOP_PATCH_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]
    # The ragged layouts a step plans over its patches, one label each.
    SEGMENT_LABELS: tuple[str, ...] = ("main",)

    def __init__(self, model: torch.nn.Module, config: Any, encoder: MediaEncoder):
        super().__init__()
        self.model = model
        self.config = config
        self.output = encoder.output

    def grids(self, fwd_info: CurrentForwardPassInfo) -> list[tuple[int, int, int]]:
        """The patch grid ``(t, h, w)`` of each run this node fills, in prompt
        order, off the walk's ``MEDIA_LAYOUT``."""
        return media_grids(fwd_info.step_metadata.get(MEDIA_LAYOUT) or {}, self.output)

    def _declare(
        self, rows: Sequence[tuple[Any, Mapping[str, Sequence[int]]]],
    ) -> SubmoduleStep:
        """The node's step: each row's spans under each label, all
        non-causal on the one vision attention resource."""
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label=label, span=span)
                for rid, lengths in rows
                for label in self.SEGMENT_LABELS
                for span in lengths[label]
            ],
            steps={VISION_ATTN: AttentionStep(causal=False)},
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep | None:
        # Leased, the captured block loop declares, plans and commits the
        # attention per replay, and that is the node's whole step.
        if (piecewise_leases or {}).get(VISION_BLOCK_LOOP):
            return None
        return self._declare([
            (rid, inp.resource_step_info) for rid, inp in zip(request_ids, inputs, strict=True)
        ])

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        # one request a step, so it can take the captured block loop
        return 1

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype,
        tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """The block loop, one prompt per replay, by patch-count bucket."""
        hidden, head_dim = self.config.hidden_size, self.config.head_dim

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            n = shape.total_tokens
            return {
                "hidden": torch.zeros(n, hidden, dtype=autocast_dtype, device=device),
                "cos": torch.zeros(n, head_dim, dtype=autocast_dtype, device=device),
                "sin": torch.zeros(n, head_dim, dtype=autocast_dtype, device=device),
            }

        def declare_step(
            request_ids: list[str], seq_lens: list[int],
            label_seq_lens: Mapping[str, list[int]] | None = None,
        ) -> SubmoduleStep:
            (rid,) = request_ids
            # capture passes no labels: each plans the bucket's one span
            return self._declare([
                (rid, label_seq_lens or {label: seq_lens for label in self.SEGMENT_LABELS}),
            ])

        return {
            VISION_BLOCK_LOOP: PiecewisePackedConfig(
                capture_fn=self._capture_block_loop,
                make_static_inputs=make_static_inputs,
                declare_step=declare_step,
                lease_before_step=True,
                can_lease=self._fits_capture,
                total_tokens=self.BLOCK_LOOP_PATCH_BUCKETS,
                capture_batch_sizes=[1],
            )
        }

    def _fits_capture(self, inputs: list[NodeInputs]) -> bool:
        """Whether the batch's segments fit a replay's attention plan under
        every label: the resource's per-request ceiling, read from it so a
        YAML override holds."""
        per_request = self.node_resources[VISION_ATTN].max_segments_per_request
        return all(
            sum(len(inp.resource_step_info[label]) for inp in inputs) <= per_request * len(inputs)
            for label in self.SEGMENT_LABELS
        )

    def _capture_block_loop(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        """The captured region: reads the runner's buffers, never rebinds."""
        s = inp.static_inputs
        return {"hidden": self.model.encode(s["hidden"], s["cos"], s["sin"])}

    @torch.compiler.disable
    def _replay_block_loop(
        self,
        engine_inputs: ModelInputsFromEngine,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        segment_lengths: Mapping[str, Sequence[int]],
    ) -> torch.Tensor | None:
        """The block loop replayed once per prompt, or None if not leased.

        A leased step declares nothing of its own (`declare_step`), so
        ``engine_inputs.step`` is None; a bucket that fits is not enough, as
        ``can_lease`` may have refused the batch. The bucket check keeps a
        forward driven outside the engine (no declaration either) eager.
        """
        runner = engine_inputs.piecewise_runners.get(VISION_BLOCK_LOOP)
        if runner is None or engine_inputs.step is not None or not runner.can_run(
            len(engine_inputs.request_ids), int(hidden.shape[0]),
        ):
            return None
        spans = {label: list(segment_lengths[label]) for label in self.SEGMENT_LABELS}
        # one replay for every image, not one each: a second plan overwrites
        # the schedule a queued replay still reads
        return runner.run(
            static_inputs={"hidden": hidden, "cos": cos, "sin": sin},
            request_ids=[engine_inputs.request_ids[0]],
            seq_lens=spans[self.SEGMENT_LABELS[-1]],
            real_bs=1,
            label_seq_lens=spans,
        ).get_view("hidden")

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        """The batch's one request (see `max_batch_size`), its images packed
        in `declare_step`'s segment order."""
        (inp,) = inputs
        device = self.get_device()

        def to_device(t: torch.Tensor) -> torch.Tensor:
            # from pinned memory, so the copy does not block the host
            if t.device.type == "cpu" and device.type != "cpu":
                t = t.pin_memory()
            return t.to(device, non_blocking=True)

        out: dict[str, Any] = {name: to_device(t) for name, t in inp.tensor_inputs.items()}
        out["segment_lengths"] = inp.resource_step_info
        return out

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        rid = engine_inputs.request_ids[0]
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[rid]

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        segment_lengths: Mapping[str, Sequence[int]],
        **tensors: torch.Tensor,
    ) -> dict[str, NameToTensorList]:
        # one call for every image; the ragged kernel's segments keep them apart
        hidden, cos, sin = self.model.embed(**tensors)
        encoded = self._replay_block_loop(engine_inputs, hidden, cos, sin, segment_lengths)
        if encoded is None:
            encoded = self.model.encode(hidden, cos, sin)
        embeds = self.model.project(encoded, **tensors)
        return {engine_inputs.request_ids[0]: {self.output: [embeds]}}
