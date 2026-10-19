"""GR00T N1.7's node submodules. The vision encoder, the LLM and the state
encoder for now; the VL bridge and the denoise step land in later cycles."""
from __future__ import annotations

from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.model.components.qwen3_vit import vision_interpolation, vision_position_ids, vision_seq_lengths
from mstar.model.gr00t_n17.components.embodiment import CategorySpecificMLP
from mstar.model.gr00t_n17.components.llm import Gr00tN17LanguageModel
from mstar.model.gr00t_n17.components.rope import mrope_position_ids
from mstar.model.gr00t_n17.components.vision import Gr00tN17VisionEncoder
from mstar.model.gr00t_n17.config import LLM_ATTN, Gr00tN17Config
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


class Gr00tN17LLMSubmodule(NodeSubmodule):
    """The ``LLM`` node: the prompt's tokens, with the vision encoder's output
    on their image rows, become the features the action head conditions on.

    One shot and one request a step: the whole prompt in a single causal
    ragged segment, no KV cache, nothing kept between steps. The grid and
    position maths runs in ``prepare_inputs`` for the reason the vision node
    does it there -- a ``.tolist()`` in the forward would break the graph.
    """

    def __init__(self, model: Gr00tN17LanguageModel, config: Gr00tN17Config):
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
        input_ids = inputs["input_ids"][0].reshape(-1)
        grid = torch.cat([g.reshape(-1, 3) for g in inputs["image_grid_thw"]])
        image_token_id = self.config.text.image_token_id
        features = inputs["deepstack_embeds"]
        # the checkpoint says how many taps the vision tower has; a short or
        # long list from that node would otherwise inject into the wrong
        # layers and read as a plausible answer
        if len(features) != len(self.config.deepstack_visual_indexes):
            raise ValueError(
                f"vision_encoder sent {len(features)} DeepStack feature(s), "
                f"the checkpoint taps {len(self.config.deepstack_visual_indexes)}"
            )
        return NodeInputs(
            input_seq_len=input_ids.shape[0],
            tensor_inputs={
                "input_ids": input_ids,
                # where the vision rows go, and where DeepStack is added
                "image_rows": (input_ids == image_token_id)
                .nonzero(as_tuple=True)[0]
                .to(device),
                "position_ids": mrope_position_ids(
                    input_ids,
                    grid,
                    image_token_id,
                    self.config.vision.spatial_merge_size,
                    device,
                ),
                "vision_embeds": inputs["vision_embeds"][0],
                # [taps, rows, hidden]: one tensor to stage, unbound in forward
                "deepstack_embeds": torch.stack(features),
            },
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        **kwargs: Any,
    ) -> SubmoduleStep | None:
        # one segment per request, so a prompt never attends to its neighbours
        segments = [
            Segment(request_id=rid, label="main", span=inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        return SubmoduleStep(
            segments=segments,
            steps={LLM_ATTN: AttentionStep(causal=True)},
        )

    def max_batch_size(self, graph_walk: str) -> int | None:
        # as the vision node: no batched forward, so a wider batch would only
        # be the engine's per-request fallback in disguise
        return 1

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        (inp,) = inputs
        device = self.get_device()
        return {k: v.to(device) for k, v in inp.tensor_inputs.items()}

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        image_rows: torch.Tensor,
        position_ids: torch.Tensor,
        vision_embeds: torch.Tensor,
        deepstack_embeds: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        hidden = self.model.embed(input_ids)
        # the reference `masked_scatter`s the merged tokens onto the image
        # rows, in the order the vision encoder packed them
        hidden = hidden.index_copy(0, image_rows, vision_embeds.to(hidden.dtype))
        cos, sin = self.model.build_cos_sin(position_ids, hidden.dtype)
        features = self.model(
            hidden, cos, sin, deepstack_embeds.unbind(0), image_rows,
        )
        return {"backbone_features": [features]}


class Gr00tN17StateEncoderSubmodule(NodeSubmodule):
    """The ``state_encoder`` node: a request's raw state becomes the single
    conditioning token the denoise loop carries beside the action tokens.

    No attention and no resource, so there is no step to declare -- the base
    class's ``None`` is the whole declaration. The child is named
    ``state_encoder`` because every action-head node strips ``action_head.``
    and keeps the reference's attribute names below it.
    """

    def __init__(self, state_encoder: CategorySpecificMLP):
        super().__init__()
        self.state_encoder = state_encoder

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        state = inputs["state"][0].reshape(1, -1, inputs["state"][0].shape[-1])
        embodiment_id = inputs["embodiment_id"][0].reshape(-1)
        # `W[cat_ids]` would read -1 as the last embodiment and serve the
        # wrong robot in silence, which is the failure D06 exists to stop.
        # The request carries this id, so it is checked where it arrives.
        categories = self.state_encoder.layer1.W.shape[0]
        if int(embodiment_id.min()) < 0 or int(embodiment_id.max()) >= categories:
            raise ValueError(
                f"embodiment id {embodiment_id.tolist()} outside the "
                f"checkpoint's {categories} embodiments"
            )
        return NodeInputs(
            # one token per state row, which is one while the checkpoint's
            # `state_history_length` is 1
            input_seq_len=state.shape[1],
            tensor_inputs={"state": state, "embodiment_id": embodiment_id},
        )

    def max_batch_size(self, graph_walk: str) -> int | None:
        # as the other nodes: no batched forward, so a wider batch would only
        # be the engine's per-request fallback in disguise
        return 1

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        (inp,) = inputs
        device = self.get_device()
        return {k: v.to(device) for k, v in inp.tensor_inputs.items()}

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        return {"state_features": [self.state_encoder(state, embodiment_id)]}
