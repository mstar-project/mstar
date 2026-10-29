"""GR00T N1.7's node submodules: the `prefill` walk's four, and the denoise
step that `action_gen` loops over."""
from __future__ import annotations

import functools
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.ragged.config import RaggedCrossAttentionStep, cross_label
from mstar.engine.resources.convenience import RaggedAttentionCallable
from mstar.engine.resources.step import Segment, SlotLease, SubmoduleStep
from mstar.model.components.diffusion.attention import joint_attention
from mstar.model.components.diffusion.denoise_loop import DenoiseLoopSubmodule
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule, euler_step
from mstar.model.gr00t_n17.components.bridge import Gr00tN17VLBridge
from mstar.model.gr00t_n17.components.dit import ActionPositionEmbedding, DiT
from mstar.model.gr00t_n17.components.embodiment import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from mstar.model.gr00t_n17.components.llm import Gr00tN17LanguageModel
from mstar.model.gr00t_n17.components.rope import mrope_position_ids
from mstar.model.gr00t_n17.components.vision import Gr00tN17VisionEncoder
from mstar.model.gr00t_n17.config import (
    BRIDGE_ATTN,
    DT_ATTN,
    DT_XATTN,
    DT_XATTN_PAIRS,
    LLM_ATTN,
    SA_SPAN,
    VL_IMAGE_SPAN,
    VL_TEXT_SPAN,
    Gr00tN17Config,
)
from mstar.model.qwen3_5.components.vision import (
    vision_interpolation,
    vision_position_ids,
    vision_seq_lengths,
)
from mstar.model.qwen3_5.config import VISION_ATTN, Qwen3_5VisionConfig
from mstar.model.submodule_base import (
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)


class Gr00tN17VisionEncoderSubmodule(NodeSubmodule):
    """The ``vision_encoder`` node: one request's camera frames, packed, become
    the embeddings the LLM scatters onto its image rows plus the three
    DeepStack features it adds after its first three layers.

    Shaped after ``qwen3_5``'s ``VisionEncoderSubmodule``, which drives the same
    tower: the grid maths runs here rather than in the forward, where a
    ``.tolist()`` would break the graph and leave h, w as symints inductor
    cannot codegen from. This node is not captured, so there is no piecewise
    config, and it takes one request a step (``can_batch`` stays off).
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
        indices, weights = vision_interpolation(
            grid, self.config.num_grid_per_side, merge, device,
        )
        return NodeInputs(
            # what the engine sizes the batch by
            input_seq_len=sum(t * h * w for t, h, w in grid),
            tensor_inputs={
                "pixel_values": torch.cat(inputs["pixel_values"], dim=0),
                "indices": indices,
                "weights": weights,
                "position_ids": vision_position_ids(grid, merge, device),
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
        # the bridge splits on these, and this node already derived them;
        # deriving them twice is two chances to disagree
        return {"backbone_features": [features], "image_rows": [image_rows]}


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
        # `[T, dim]` -> `[1, 1, T * dim]`: the reference's `state.view(B, 1, -1)`, one
        # token whatever the history; `process_prompt` has checked T
        state = inputs["state"][0].reshape(1, 1, -1)
        embodiment_id = inputs["embodiment_id"][0].reshape(-1)
        return NodeInputs(
            input_seq_len=1,
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


class Gr00tN17VLBridgeSubmodule(NodeSubmodule):
    """The ``vl_bridge`` node: the backbone's features become the two spans
    the denoise loop cross-attends.

    The split is the point. The reference masks every DT cross-attention
    against the image rows or the non-image rows, alternating by block; here
    the rows are separated **once**, into ``vl_text`` and ``vl_image``, and the
    loop attends a whole span with no mask at all. Softmax over a span does
    not care in what order the keys arrive, so the two are the same
    computation.

    Unmasked in the reference, so one ragged segment per request is the same
    thing at B = 1 and keeps a request independent of its neighbours.
    """

    # Under autocast `vlln` runs in fp32 and its output *is* the residual
    # stream, so all four blocks and both spans come out fp32. NVIDIA runs the
    # bridge in plain bf16; so does this node, which then matches it bitwise
    # on SDPA and hands the denoise loop bf16 rows.
    disable_autocast = True
    # Compiled in plain bf16, a request's spans changed once a prompt of
    # another length had been served (the recompile for dynamic shapes rounds
    # differently), so the same request stopped returning the same actions.
    # Eager is also NVIDIA's arithmetic; a captured graph keeps it.
    disable_torch_compile = True

    def __init__(self, model: Gr00tN17VLBridge, config: Gr00tN17Config):
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
        features = inputs["backbone_features"][0]
        image_rows = inputs["image_rows"][0]
        if int(image_rows.numel()) > features.shape[0]:
            raise ValueError(
                f"{int(image_rows.numel())} image row(s) for "
                f"{features.shape[0]} backbone feature row(s)"
            )
        return NodeInputs(
            input_seq_len=features.shape[0],
            tensor_inputs={"backbone_features": features, "image_rows": image_rows},
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        **kwargs: Any,
    ) -> SubmoduleStep | None:
        # one segment per request, unmasked: every row of a prompt sees every
        # other row of that prompt, and none of anyone else's
        segments = [
            Segment(request_id=rid, label="main", span=inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        return SubmoduleStep(
            segments=segments,
            steps={BRIDGE_ATTN: AttentionStep(causal=False)},
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
        backbone_features: torch.Tensor,
        image_rows: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        rows = self.model(backbone_features)
        is_image = torch.zeros(
            rows.shape[0], dtype=torch.bool, device=rows.device,
        ).index_fill_(0, image_rows, True)
        return {
            "vl_text": [rows[~is_image]],
            "vl_image": [rows[is_image]],
        }


@dataclass(frozen=True)
class Gr00tN17Shape:
    """What has to match for two requests to share a denoise step.

    The scaffold stacks a step's conditioning rectangularly, so a batch's requests
    have to agree on both span lengths; ``can_batch`` keys on this and so do the
    graph buckets. Measured over the 20 recorded requests: the image span is
    constant (448: two cameras at two timesteps, four frames) and only the text span varies (13 or 21), so
    in practice this buckets on the prompt. The image count is carried all the same,
    because nothing makes it a constant -- another camera set is another number.
    """

    text_rows: int
    image_rows: int


class Gr00tN17DenoiseStepSubmodule(DenoiseLoopSubmodule):
    """The ``denoise_step`` node: one flow-matching step of the action head.

    The loop is the scaffold's, not ours: it owns the iteration counter, the
    seeded noise, the per-request schedule, the graph buckets, the overshoot veto and
    the step declaration's plumbing. What is here is the hooks -- what GR00T's step
    *is*, rather than how a step is driven.

    The children are named for the checkpoint: ``action_encoder``, ``model``,
    ``action_decoder`` and ``position_embedding`` are what ``action_head.`` strips to.
    """

    def __init__(
        self,
        config: Gr00tN17Config,
        action_encoder: MultiEmbodimentActionEncoder,
        position_embedding: ActionPositionEmbedding,
        model: DiT,
        action_decoder: CategorySpecificMLP,
        *,
        loop_name: str,
        max_batch_size: int = 8,
        capture_buckets: Sequence[tuple[str, Gr00tN17Shape]] = (),
        capture_batch_sizes: Sequence[int] = (1, 2, 4, 8),
    ):
        # `attn_resource_key=None`: this node declares its own two resources rather
        # than the base's single one, as `ltx2_5` does for its four.
        super().__init__(
            loop_name=loop_name, max_batch_size=max_batch_size, attn_resource_key=None,
            capture_buckets=capture_buckets, capture_batch_sizes=capture_batch_sizes,
        )
        self.config = config
        self.action_encoder = action_encoder
        self.position_embedding = position_embedding
        self.model = model
        self.action_decoder = action_decoder
        self._attends: dict[str, Any] | None = None

    # ------------------------------------------------------------------ shape
    def bucket_key_for(self, fwd_info: CurrentForwardPassInfo) -> Gr00tN17Shape:
        """The spans are needed before their tensors arrive, so they ride in the step
        metadata: the model reads them off the persisted bridge tensors' `dims`
        (`Gr00tN17Model.get_partition_forward_pass_args`)."""
        meta = fwd_info.step_metadata
        missing = [k for k in ("vl_text_rows", "vl_image_rows") if k not in meta]
        if missing:
            raise KeyError(
                f"denoise_step has no {missing} in step_metadata; the model reads "
                "the bridge's two spans off their persisted tensors and puts them "
                "there, because the key is needed before the tensors arrive"
            )
        return Gr00tN17Shape(
            text_rows=int(meta["vl_text_rows"]), image_rows=int(meta["vl_image_rows"]),
        )

    def num_tokens(self, bucket_key: Gr00tN17Shape) -> int:
        """The state row, then the action rows: 1 + 40. Not the latents' 40."""
        return 1 + self.config.action_horizon

    # --------------------------------------------------------------- schedule
    def schedule_for(self, fwd_info, bucket_key: Gr00tN17Shape) -> FlowMatchSchedule:
        """GR00T's four steps, written out rather than built.

        ``FlowMatchSchedule.build`` would give the image convention -- sigmas
        descending from 1 and timesteps ``[1000, 750, 500, 250]``. GR00T's time runs
        the other way: `t = k / 4`, conditioned on `int(t * 1000)`, with
        `actions += dt * velocity`. So sigma *ascends* and `sigma_next - sigma` is
        the reference's `dt`, which is what `euler_step` multiplies by. Our "sigma"
        is therefore GR00T's time, not a noise level -- same arithmetic, read the
        other way round, and not something to "fix" into descending.
        """
        steps = self.config.num_inference_timesteps
        sigmas = torch.arange(steps + 1, dtype=torch.float32) / steps
        timesteps = (sigmas[:-1] * self.config.num_timestep_buckets).floor()
        return FlowMatchSchedule(sigmas=sigmas, timesteps=timesteps, mu=None)

    def seed_latents(self, fwd_info, bucket_key: Gr00tN17Shape, generator) -> torch.Tensor:
        """The noisy trajectory the loop carries, ``[40, 132]`` without a batch dim.

        Drawn at the **weights'** dtype, as the reference draws it at
        ``vl_embeds.dtype`` (`gr00t_n1d7.py:349-353`). That keeps the carried dtype
        the same at every iteration, which `denoise` returns and which a captured
        graph records -- an fp32 seed feeding a bf16 step would change dtype after
        iteration 0 and need two graphs.
        """
        return torch.randn(
            (self.config.action_horizon, self.config.max_action_dim),
            generator=generator, dtype=next(self.model.parameters()).dtype,
        )

    # ----------------------------------------------------------- conditioning
    def request_inputs(self, fwd_info, inputs: NameToTensorList, bucket_key) -> dict:
        """One request's frozen conditioning, each **without** a batch dim: the base
        stacks them, so a leading 1 here would arrive as a spurious axis.

        Both checks below live here rather than in ``denoise`` because this runs
        host-side in ``prepare_inputs``; ``denoise`` is the region a graph captures,
        and reading a tensor's values there (``.min()``, ``int(...)``) would sync the
        device and make it uncapturable.
        """
        text, image = inputs["vl_text"][0], inputs["vl_image"][0]
        # The bucket key comes from `step_metadata` and the spans come from the
        # bridge. Nothing else compares them, and they are declared to the engine:
        # a miscount plans spans the forward does not use, and a real ragged kernel
        # reads the *declared* length. Inside a batch `torch.stack` forces the rows
        # to agree and `can_batch` forces the keys to agree, so a preprocessing
        # miscount would be uniform and silent.
        if (text.shape[0], image.shape[0]) != (bucket_key.text_rows, bucket_key.image_rows):
            raise ValueError(
                f"denoise_step was told {bucket_key.text_rows} text and "
                f"{bucket_key.image_rows} image rows by step_metadata, but the "
                f"bridge sent {text.shape[0]} and {image.shape[0]}; the counts the "
                "model put in step_metadata must be the spans the bridge emits"
            )
        embodiment_id = inputs["embodiment_id"][0].reshape(())
        return {
            "vl_text": text,
            "vl_image": image,
            "state": inputs["state_features"][0].reshape(1, -1),
            "embodiment_id": embodiment_id,
        }

    def split_batches_by_capture_key(self, graph_walk: str) -> bool:
        """A batch mixing prompt lengths matches no capture and runs eager one request at a
        time, so batch each length apart."""
        del graph_walk
        return True

    def capture_request_inputs(self, bucket_key: Gr00tN17Shape, device) -> dict:
        dtype = next(self.model.parameters()).dtype
        width = self.config.backbone_embedding_dim
        return {
            self.LATENTS: torch.zeros(
                self.config.action_horizon, self.config.max_action_dim,
                dtype=dtype, device=device,
            ),
            "vl_text": torch.zeros(bucket_key.text_rows, width, dtype=dtype, device=device),
            "vl_image": torch.zeros(bucket_key.image_rows, width, dtype=dtype, device=device),
            "state": torch.zeros(1, self.config.input_embedding_dim, dtype=dtype, device=device),
            "embodiment_id": torch.zeros((), dtype=torch.long, device=device),
        }

    # -------------------------------------------------------------- attention
    def bind_node_resources(self, resources) -> None:
        super().bind_node_resources(resources)
        self._attends = None  # the memoised callables hold the old resources

    def attends(self) -> dict[str, Any]:
        """One callable per span, built once per binding.

        A compiled region guards on the *identity* of the callables it is handed, so
        a fresh one per step retraces it every step. A span whose resource is not
        bound is left out, and the DT reads it with SDPA -- the parity backend.
        """
        if self._attends is None:
            self_attn = self.node_resources.get(DT_ATTN)
            cross = self.node_resources.get(DT_XATTN)

            def on(resource, label):
                if resource is None:
                    return None
                return functools.partial(
                    joint_attention, ragged=RaggedAttentionCallable(resource, label),
                )

            # All or nothing. Nothing bound is the parity backend -- every span on
            # SDPA, which is how the CPU tests run; a deployment always binds both,
            # since `get_node_resources` declares every resource unconditionally.
            # *Partly* bound is incoherent: `declare_step` declares both resources,
            # so the engine plans a step whose kernel the forward would not use, and
            # a captured graph would record SDPA where the plan says ragged.
            if (self_attn is None) != (cross is None):
                raise RuntimeError(
                    f"denoise_step has {DT_ATTN if self_attn is not None else DT_XATTN} "
                    f"but not {DT_XATTN if self_attn is not None else DT_ATTN}; it "
                    "declares a step for both, so declare both specs or neither"
                )
            built = {
                SA_SPAN: on(self_attn, SA_SPAN),
                VL_TEXT_SPAN: on(cross, cross_label(SA_SPAN, VL_TEXT_SPAN)),
                VL_IMAGE_SPAN: on(cross, cross_label(SA_SPAN, VL_IMAGE_SPAN)),
            }
            self._attends = {k: v for k, v in built.items() if v is not None}
        return self._attends

    def attention_segments(self, bucket_key: Gr00tN17Shape) -> Sequence[tuple[str, int]]:
        return (
            (SA_SPAN, self.num_tokens(bucket_key)),
            (VL_TEXT_SPAN, bucket_key.text_rows),
            (VL_IMAGE_SPAN, bucket_key.image_rows),
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Any = None,
        **kwargs: Any,
    ) -> SubmoduleStep | None:
        # `attention_segments` is the one description of the layout; building the
        # segments here from anything else would say it twice and let the two disagree.
        by_label: dict[str, list[Segment]] = {SA_SPAN: [], VL_TEXT_SPAN: [], VL_IMAGE_SPAN: []}
        for rid, inp in zip(request_ids, inputs, strict=True):
            for label, span in self.attention_segments(inp.resource_step_info):
                by_label[label].append(Segment(request_id=rid, label=label, span=span))
        sa, text, image = by_label[SA_SPAN], by_label[VL_TEXT_SPAN], by_label[VL_IMAGE_SPAN]
        return SubmoduleStep(
            steps={
                DT_ATTN: AttentionStep(segments=tuple(sa), causal=False),
                DT_XATTN: RaggedCrossAttentionStep(
                    segments=tuple(sa + text + image), pairs=DT_XATTN_PAIRS,
                ),
            },
            cg_key_info=self._uniform_key(inp.resource_step_info for inp in inputs),
        )

    # ----------------------------------------------------------------- the step
    def denoise(
        self,
        engine_inputs: ModelInputsFromEngine,
        bucket_key: Hashable,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        vl_text: torch.Tensor,
        vl_image: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        """``[B, 40, 132] -> [B, 40, 132]``: the reference's one iteration.

        The position table lands on the action rows **before** the state row joins
        them, so of the DT's 41 rows the 40 action rows carry a position and the
        state row carries none.
        """
        dtype = next(self.model.parameters()).dtype
        cat_ids = embodiment_id.reshape(-1)
        actions = self.action_encoder(latents.to(dtype), timestep, cat_ids)
        rows = torch.cat([state, self.position_embedding(actions)], dim=1)
        out = self.model(rows, vl_text, vl_image, timestep, self.attends())
        velocity = self.action_decoder(out, cat_ids)[:, -self.config.action_horizon:]
        # The carried trajectory stays at the weights' dtype, which `euler_step`
        # gives us by returning in the velocity's. That is what the reference does:
        # it is served in bf16 (`gr00t_policy.py:111`), draws `actions` at
        # `vl_embeds.dtype`, and updates with a Python-float `dt`, so its trajectory
        # is bf16 at every step.
        #
        # Carrying fp32 instead is the better *integrator* -- one increment is about
        # 3 bf16 ulp, so the reference quantises each update -- but it costs the only
        # end-to-end bar that catches a wiring bug: over 4 steps an fp32 carry moves
        # the trajectory by 0.0243, while an off-by-one in the DT's text/image
        # alternation moves it by 0.0391. A tolerance wide enough to admit our own
        # "improvement" is within 1.6x of admitting a real defect. Accuracy we were
        # not asked for, bought with the test that matters.
        #
        # `euler_step` already sums in fp32 before rounding back, and for two bf16
        # operands and an exact `dt` that is bit-identical to bf16 arithmetic
        # (measured over 200k samples), so matching the reference costs nothing.
        return euler_step(latents, velocity, sigma, sigma_next)
