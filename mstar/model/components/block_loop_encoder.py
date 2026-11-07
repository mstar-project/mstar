"""A packed encoder node whose block loop replays as a captured region.

`PackedBlockLoopEncoder` packs a step's requests, every image of each, into
one call, and replays the tower's block loop as a piecewise region per patch
bucket, the whole step as one row; patch embed and the projection run eagerly
around it (the projection usually changes the row count, which a region's
output view cannot express). Eager, a ViT tower is launch-bound (~500
kernels, ~4 ms of GPU work in a ~20 ms forward on Qwen3.5 9B).

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
``output`` (on the VLM scaffold, its `MediaEncoder`'s). A tower whose block
loop returns more than the hidden states (Qwen3-Omni's deepstack taps) adds
them to the region's outputs in `_capture_block_loop` and reads them off the
replay in its own `forward_batched`.

A replay's attention plan holds at most the resource's
``max_segments_per_step`` segments under each label. A batch that would
overflow it or the top patch bucket runs its requests one at a time instead,
each replaying if it fits alone; a request too large alone runs the block
loop eagerly, however few its patches.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

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
from mstar.model.multimodal import MEDIA_LAYOUT, media_grids
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

if TYPE_CHECKING:
    from mstar.engine.accelerator_graph_runner import PiecewiseOutput

# The tower's block loop, as a piecewise capture region
VISION_BLOCK_LOOP = "vision_block_loop"


class PackedBlockLoopEncoder(NodeSubmodule):
    # Patch-count buckets of one replay; a larger step runs eagerly.
    BLOCK_LOOP_PATCH_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]
    # Requests per encoder step. Above one, every tensor of ``tensor_inputs``
    # must pack by concatenation (an index into a request's own patches does
    # not), and ``config`` has ``spatial_merge_size``, which cuts the output
    # back apart.
    MAX_BATCH_SIZE = 1
    # The ragged layouts a step plans over its patches, one label each.
    SEGMENT_LABELS: tuple[str, ...] = ("main",)
    # The dtype of what ``embed`` hands the block loop under the node's
    # autocast, which the replay's input buffers hold. None: autocast's.
    BLOCK_LOOP_DTYPE: torch.dtype | None = None

    def __init__(self, model: torch.nn.Module, config: Any, output: str):
        super().__init__()
        self.model = model
        self.config = config
        # the node's output signal; a MediaEncoder's ``output`` on the VLM scaffold
        self.output = output

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
        # packed, the requests must fit one replay together. Applied with the
        # block loop uncaptured too (capture off, or a node without the
        # runner), where it only splits a large step into per-request runs.
        return self._fits_capture(model_inputs)

    def max_batch_size(self, graph_walk: str) -> int | None:
        return self.MAX_BATCH_SIZE

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype,
        tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """The block loop, one replay per step, by patch-count bucket. Batch
        size 1: a replay is the whole step as one row, every request's
        segments in it."""
        hidden, head_dim = self.config.hidden_size, self.config.head_dim
        dtype = self.BLOCK_LOOP_DTYPE or autocast_dtype

        def make_static_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            n = shape.total_tokens
            return {
                "hidden": torch.zeros(n, hidden, dtype=dtype, device=device),
                "cos": torch.zeros(n, head_dim, dtype=dtype, device=device),
                "sin": torch.zeros(n, head_dim, dtype=dtype, device=device),
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
                one_row_per_step=True,
                can_lease=self._fits_capture,
                total_tokens=self.BLOCK_LOOP_PATCH_BUCKETS,
                capture_batch_sizes=[1],
            )
        }

    def _fits_capture(self, inputs: list[NodeInputs]) -> bool:
        """Whether the batch fits one replay (`_fits_replay`), under every
        label."""
        return self._fits_replay(
            sum(inp.input_seq_len for inp in inputs),
            max(sum(len(inp.resource_step_info[label]) for inp in inputs) for label in self.SEGMENT_LABELS),
        )

    def _fits_replay(self, patches: int, segments: int) -> bool:
        """Whether a step of this many patches and attention segments fits one
        replay: the top patch bucket, and the attention plan's per-step
        segment ceiling, read from the resource so a YAML override holds."""
        # the whole step replays as one row (`one_row_per_step`)
        return (
            patches <= self.BLOCK_LOOP_PATCH_BUCKETS[-1]
            and segments <= self.node_resources[VISION_ATTN].max_segments_per_step
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
    ) -> "PiecewiseOutput | None":
        """The block loop replayed once for the whole step, or None if not
        leased: views of the region's outputs (``_capture_block_loop``'s
        names), valid within this step."""
        if VISION_BLOCK_LOOP not in engine_inputs.piecewise_leases:
            return None
        spans = {label: list(segment_lengths[label]) for label in self.SEGMENT_LABELS}
        # one replay for every image, not one each: a second plan overwrites
        # the schedule a queued replay still reads
        return engine_inputs.piecewise_runners[VISION_BLOCK_LOOP].run(
            static_inputs={"hidden": hidden, "cos": cos, "sin": sin},
            request_ids=engine_inputs.request_ids,
            seq_lens=spans[self.SEGMENT_LABELS[-1]],
            label_seq_lens=spans,
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        """Every request's images packed end to end, in `declare_step`'s
        segment order."""
        device = self.get_device()

        def packed(name: str) -> torch.Tensor:
            parts = [inp.tensor_inputs[name] for inp in inputs]
            t = parts[0] if len(parts) == 1 else torch.cat(parts)
            # from pinned memory, so the copy does not block the host
            if t.device.type == "cpu" and device.type != "cpu":
                t = t.pin_memory()
            return t.to(device, non_blocking=True)

        out: dict[str, Any] = {name: packed(name) for name in inputs[0].tensor_inputs}
        out["segment_lengths"] = {
            label: [n for inp in inputs for n in inp.resource_step_info[label]]
            for label in self.SEGMENT_LABELS
        }
        # patches per request, to cut the output back apart
        out["row_patches"] = tuple(inp.input_seq_len for inp in inputs)
        return out

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        rid = engine_inputs.request_ids[0]
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[rid]

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        segment_lengths: Mapping[str, Sequence[int]],
        row_patches: tuple[int, ...] = (),
        **tensors: torch.Tensor,
    ) -> dict[str, NameToTensorList]:
        # one call for every image; the ragged kernel's segments keep them apart
        hidden, cos, sin = self.model.embed(**tensors)
        replayed = self._replay_block_loop(engine_inputs, hidden, cos, sin, segment_lengths)
        encoded = self.model.encode(hidden, cos, sin) if replayed is None else replayed.get_view("hidden")
        embeds = self.model.project(encoded, **tensors)
        return self._split_rows(engine_inputs.request_ids, {self.output: [embeds]}, row_patches)

    def _split_rows(
        self, request_ids: list[str], outputs: NameToTensorList, row_patches: tuple[int, ...],
    ) -> dict[str, NameToTensorList]:
        """Each request's slice of every packed output, one merged row per
        ``spatial_merge_size ** 2`` patches."""
        if len(request_ids) == 1:
            return {request_ids[0]: outputs}
        out, start = {}, 0
        for rid, patches in zip(request_ids, row_patches, strict=True):
            n = patches // self.config.spatial_merge_size ** 2
            out[rid] = {name: [t[start:start + n] for t in tensors] for name, tensors in outputs.items()}
            start += n
        return out
