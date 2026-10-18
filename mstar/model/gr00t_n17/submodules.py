"""GR00T N1.7's node submodules. The vision encoder for now; the LLM, the VL
bridge, the state encoder and the denoise step land in later cycles."""
from __future__ import annotations

from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.model.components.qwen3_vit import vision_interpolation, vision_position_ids, vision_seq_lengths
from mstar.model.gr00t_n17.components.vision import Gr00tN17VisionEncoder
from mstar.model.qwen3_5.config import VISION_ATTN, Qwen3_5VisionConfig
from mstar.model.submodule_base import (
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.utils.h2d import to_device_nonblocking


class Gr00tN17VisionEncoderSubmodule(NodeSubmodule):
    """The ``vision_encoder`` node: one request's camera frames, packed, become
    the embeddings the LLM scatters onto its image rows plus the three
    DeepStack features it adds after its first three layers.

    Shaped after ``qwen3_5``'s ``VisionEncoderSubmodule``, which drives the same
    tower: the grid maths runs here rather than in the forward, where a
    ``.tolist()`` would break the graph and leave h, w as symints inductor
    cannot codegen from. Capture is a later cycle, so there is no piecewise
    config and the node takes one request a step (``can_batch`` stays off).
    """

    def __init__(
        self, model: Gr00tN17VisionEncoder, config: Qwen3_5VisionConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        device = self.get_device()
        merge = self.config.spatial_merge_size
        # the reshape takes a bare [t, h, w] row as one image; iterating it
        # as given unpacks scalars and raises
        grid = [
            (int(t), int(h), int(w))
            for g in inputs["image_grid_thw"]
            for t, h, w in g.reshape(-1, 3).tolist()
        ]
        indices, weights = vision_interpolation(grid, self.config.num_grid_per_side, merge)
        return NodeInputs(
            # what the engine sizes the batch by
            input_seq_len=sum(t * h * w for t, h, w in grid),
            tensor_inputs={
                "pixel_values": torch.cat(inputs["pixel_values"], dim=0),
                "indices": to_device_nonblocking(indices, device),
                "weights": to_device_nonblocking(weights, device),
                "position_ids": to_device_nonblocking(vision_position_ids(grid, merge), device),
            },
            # not a forward argument: `declare_step` makes the segments
            kwargs={"seq_lengths": vision_seq_lengths(grid)},
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        **kwargs: Any,
    ) -> SubmoduleStep | None:
        # one segment per frame: a frame attends to itself alone
        segments = [
            Segment(request_id=rid, label="main", span=span)
            for rid, inp in zip(request_ids, inputs, strict=True)
            for span in inp.kwargs["seq_lengths"]
        ]
        return SubmoduleStep(
            segments=segments,
            steps={VISION_ATTN: AttentionStep(causal=False)},
        )

    def max_batch_size(self, graph_walk: str) -> int | None:
        # there is no batched forward here, so a wider batch is the engine's
        # per-request fallback in disguise: have the scheduler split instead
        return 1

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        """The batch's one request. Only the pixels cross over: the rest is
        built on the device in `prepare_inputs`, and `seq_lengths` is spent."""
        (inp,) = inputs
        tensors = dict(inp.tensor_inputs)
        tensors["pixel_values"] = tensors["pixel_values"].to(self.get_device())
        return tensors

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        # one call for every image; the ragged kernel's per-frame segments
        # keep them apart
        embeds, deepstack = self.model(pixel_values, indices, weights, position_ids)
        return {"vision_embeds": [embeds], "deepstack_embeds": deepstack}
