"""Praxis-1 compute nodes.

mstar owns the graph: the walks, the loop, the state a session carries and the
batching. The heavy forwards here call the reference ``praxis`` modules, the
way the OmniVoice node reuses its reference package, so the first landing is
numerically the published policy rather than a re-derivation of it. The native
port of the backbone and the expert follows as its own change, with a
reference-equivalence test in the shape of ``test_pi05_reference_equivalence``.

Two parts do not come from the reference package. The Wan2.2 VAE is the one
mstar already serves for Wan2.2-TI2V, so the observation is encoded through
that rather than through a second copy, and the text encoder is mstar's
Qwen3.5 reading the hidden layer the checkpoint asks for.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.state import ArrayStep, Segment, SubmoduleStep
from mstar.model.components.diffusion.decode_utils import latent_affine
from mstar.model.praxis1.components.action_rows import (
    GRIP_DIMS,
    Q99_DIMS,
    clip_grip,
    denormalize_rows,
    unpack_action_tokens,
)
from mstar.model.praxis1.components.canvas import (
    build_canvas,
    tile_view_single_arm,
    tile_views_bimanual,
)
from mstar.model.praxis1.components.proprio import normalized_proprio
from mstar.model.praxis1.config import Praxis1Config
from mstar.model.praxis1.episode_history import (
    BOUNDARY_STATE_NAME,
    STATE_NAME,
    boundary_offsets,
    clean_depth,
)
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)


class _Praxis1Node(NodeSubmodule):
    """Shared plumbing: every node here is per-request and uncompiled.

    One policy call is one request's chunk. Rows of different requests share
    neither a window nor a canvas, so nothing batches across them, and the
    data-dependent denoise loop is the case ``disable_torch_compile`` exists
    for.
    """

    disable_torch_compile = True

    def __init__(self, config: Praxis1Config) -> None:
        super().__init__()
        self.config = config

    def max_batch_size(self, graph_walk: str) -> int:
        return 1


class Praxis1TextEncoderSubmodule(_Praxis1Node):
    """Qwen3.5-2B, read at the hidden layer the checkpoint names.

    Emits the task embedding and the blank-caption embedding beside it: the
    video guidance pass needs the unconditional branch, and the caption is
    fixed for the checkpoint, so both are encoded once and persist between
    walks of this request.
    """

    def __init__(self, config: Praxis1Config, encoder: nn.Module) -> None:
        super().__init__(config)
        self.encoder = encoder
        self._check(encoder)

    def _check(self, encoder: nn.Module) -> None:
        """Refuse a text encoder the checkpoint cannot have been trained on.

        Both facts are cheap to read and expensive to find later. A wrong
        hidden size makes the backbone's text projection reject its input,
        which at least fails; too few layers is the quiet one, because
        ``hidden_states[-3]`` still indexes and just returns a different
        layer's states.
        """
        hidden = getattr(encoder.config, "hidden_size", None)
        if hidden is not None and hidden != self.config.text_hidden_dim:
            raise ValueError(
                f"the text encoder is {hidden}-wide, the checkpoint wants "
                f"{self.config.text_hidden_dim}"
            )
        layers = getattr(encoder.config, "num_hidden_layers", None)
        needed = abs(self.config.text_hidden_layer)
        if layers is not None and layers < needed:
            raise ValueError(
                f"the checkpoint reads hidden layer {self.config.text_hidden_layer}, "
                f"but this encoder has only {layers} layers"
            )

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={
                "text_ids": inputs["text_ids"][0],
                "text_mask": inputs["text_mask"][0],
                "blank_ids": inputs["blank_ids"][0],
                "blank_mask": inputs["blank_mask"][0],
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        blank_ids: torch.Tensor,
        blank_mask: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        with torch.inference_mode():
            task = self._encode(text_ids, text_mask)
            blank = self._encode(blank_ids, blank_mask)
        return {"text_embeds": [task], "text_embeds_blank": [blank]}

    def _encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The hidden states of one prompt, ``(1, S, 2048)``, padding removed.

        Two things here are easy to invert. The padding is on the *tokens*,
        up to a multiple of 32, and it is there for the kernel rather than
        for the model: the attention mask hides those positions, and the
        states handed on are trimmed back to the prompt's own length. Padding
        the output instead would hand the backbone 32-aligned rows the
        checkpoint never attends to.

        And the layer is ``-3``, the output of decoder layer 21 of 24, before
        the final norm. The last hidden state is a different tensor.
        """
        device = self.get_device()
        ids, mask = ids.to(device), mask.to(device)
        if ids.dim() == 1:
            ids, mask = ids[None], mask[None]
        states = self.encoder(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states[self.config.text_hidden_layer]
        length = int(mask.sum(dim=-1)[0])
        return states[:1, :length]


class Praxis1VaeEncoderSubmodule(_Praxis1Node):
    """The observation canvas to latents, through mstar's Wan2.2 VAE.

    The canvas is one tile per camera view, side by side, at the tile size the
    checkpoint names. The client sends raw camera frames; the fit and the trim
    happen here so the API server stages uint8 rather than floats.
    """

    def __init__(self, config: Praxis1Config, vae: nn.Module, *, max_context_chunks: int = 0) -> None:
        super().__init__(config)
        self.vae = vae
        self.max_context_chunks = max_context_chunks
        self.episode_history = max_context_chunks > 0
        latent = getattr(vae.config, "z_dim", None)
        if latent is not None and latent != self.config.latent_channels:
            raise ValueError(
                f"the VAE has {latent} latent channels, the checkpoint wants "
                f"{self.config.latent_channels}"
            )

    def vae_encode(self, canvas: torch.Tensor) -> torch.Tensor:
        """The canvas to the policy's latent, ``(1, 192, T', h, w)`` float32.

        Three steps the reference keeps separate and a port tends to fuse.
        The canvas arrives holding integer values in ``[0, 255]``, so the
        unit conversion happens here and exactly once. The per-channel
        statistics are those of the Wan2.2 VAE rather than of this
        checkpoint, so they come off the VAE the same way Wan2.2-TI2V's own
        node takes them. Only then does the 2x2 spatial patchify fold the 48
        latent channels into the 192 the backbone reads.
        """
        device = self.get_device()
        video = (canvas.to(device=device) / 127.5 - 1.0).to(self.vae.dtype)
        latent = self.vae.encode(video).latent_dist.mode().to(torch.float32)
        mean, inv_std = latent_affine(self.vae, latent.dtype, latent.device)
        return self._patchify((latent - mean) * inv_std)

    @staticmethod
    def _patchify(latent: torch.Tensor) -> torch.Tensor:
        """Fold each 2x2 spatial block into the channel axis."""
        b, c, t, h, w = latent.shape
        if h % 2 or w % 2:
            raise ValueError(f"a {h}x{w} latent does not divide into 2x2 patches")
        x = latent.reshape(b, c, t, h // 2, 2, w // 2, 2)
        return x.permute(0, 1, 4, 6, 2, 3, 5).reshape(b, c * 4, t, h // 2, w // 2)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        frames = inputs["image_inputs"]
        if len(frames) != self.config.num_cameras:
            raise ValueError(
                f"the {self.config.embodiment} rig has {self.config.num_cameras} views "
                f"({', '.join(self.config.views)}); got {len(frames)} images"
            )
        tensors = {f"view_{i}": f for i, f in enumerate(frames)}
        append = 0
        if self.episode_history:
            pose = inputs["executed_pose_inputs"][0]
            pose_valid = inputs["executed_pose_valid"][0]
            append = int(fwd_info.step_metadata["history_append"])
            if (pose.ndim != 2 or pose.shape[0] != frames[0].shape[0]
                    or pose_valid.shape != (pose.shape[0], 1)):
                raise ValueError("executed poses and camera frames must have matching rows")
            if frames[0].shape[0] < append:
                raise ValueError("executed camera stack is shorter than the planned append")
            tensors["history_pose"] = pose
            tensors["history_pose_valid"] = pose_valid
        return NodeInputs(tensor_inputs=tensors, resource_step_info=append,
                          kwargs={"history_append": append})

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs], **kwargs,
    ) -> SubmoduleStep | None:
        if not self.episode_history:
            return None
        return SubmoduleStep(
            segments=[Segment(rid, "main", int(inp.resource_step_info))
                      for rid, inp in zip(request_ids, inputs, strict=True)],
            state={
                STATE_NAME: ArrayStep(),
                BOUNDARY_STATE_NAME: ArrayStep(segments=tuple(
                    Segment(rid, "main", 1) for rid in request_ids
                )),
            },
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        history_append: int = 0,
        history_pose: torch.Tensor | None = None,
        history_pose_valid: torch.Tensor | None = None,
        **views: torch.Tensor,
    ) -> NameToTensorList:
        device = self.get_device()
        with torch.inference_mode():
            tiles = [views[f"view_{i}"].to(device) for i in range(self.config.num_cameras)]
            if self.episode_history:
                assert history_pose is not None and history_pose_valid is not None
                if any(view.shape[0] != tiles[0].shape[0] for view in tiles):
                    raise ValueError("executed camera stacks have different lengths")
                # Select only rows new to the episode. On first warm the
                # seed already contains the overlap's first frame.
                selected = [view[-(history_append + 1):-1] if history_append > 1
                            else view[-1:] for view in tiles]
                added = build_canvas(self._tiles(selected)).squeeze(0).permute(1, 0, 2, 3).to(torch.uint8)
                poses = history_pose[-(history_append + 1):-1] if history_append > 1 else history_pose[-1:]
                pose_valid = (history_pose_valid[-(history_append + 1):-1]
                              if history_append > 1 else history_pose_valid[-1:])
                handle = self.state_handles[STATE_NAME]
                boundary = self.state_handles[BOUNDARY_STATE_NAME]
                previous = boundary.read(0).fields
                previously_valid = bool(previous["history_valid"][-1, 0]) if len(previous["pose"]) else True
                history_valid = previously_valid and bool(history_pose_valid[:-1].all())
                boundary.add(
                    0, pose=history_pose[-1:].to(device=device, dtype=torch.float32),
                    history_valid=torch.tensor([[int(history_valid)]], dtype=torch.uint8, device=device),
                )
                handle.add(0, frame=added, pose=poses.to(device=device, dtype=torch.float32),
                           pose_valid=pose_valid.to(device=device, dtype=torch.uint8))
                observed = handle.read(0).fields
                boundaries = boundary.read(0).fields
                tiles = [view[-1:] for view in tiles]
            canvas = build_canvas(self._tiles(tiles))
            latent = self.vae_encode(canvas)
            if not self.episode_history:
                return {"obs_latents": [latent]}
            depth = clean_depth(len(observed["frame"]), self.max_context_chunks)
            if depth:
                clean_canvas = observed["frame"].permute(1, 0, 2, 3).unsqueeze(0).float()
                clean = self.vae_encode(clean_canvas)
                if self.config.embodiment == "single_arm":
                    anchor = clean[:, :, :1].contiguous()
                else:
                    anchor = self.vae_encode(clean_canvas[:, :, :1])
                offsets = boundary_offsets(depth)
                from_stream = history_valid and bool(observed["pose_valid"].all())
                if from_stream:
                    poses_all = torch.cat((observed["pose"][list(offsets)], boundaries["pose"][-1:]))
                else:
                    poses_all = boundaries["pose"][-(depth + 1):]
            else:
                anchor = latent
                clean = latent[:, :, :0]
                poses_all = boundaries["pose"][-1:]
            proprio_all = torch.cat([
                normalized_proprio(row.cpu().tolist(), self.config.embodiment, self.config.normalization)
                for row in poses_all
            ], dim=1).to(device=device)
        return {"obs_latents": [latent], "anchor_latents": [anchor],
                "clean_latents": [clean], "proprio_all": [proprio_all]}

    def _tiles(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        """Each view at the checkpoint's tile size, by that rig's own rule.

        The two rigs do not share one: single-arm fits every view against a
        single native size, bimanual fits each against its own and refuses a
        view that does not land on the tile.
        """
        tile_h, tile_w = self.config.tile_height, self.config.tile_width
        if self.config.embodiment == "bimanual":
            return tile_views_bimanual(views, tile_h, tile_w)
        return [
            tile_view_single_arm(v, tile_h, tile_w, antialias=self.config.antialias)
            for v in views
        ]


class Praxis1BackboneSubmodule(_Praxis1Node):
    """One Euler step of the video backbone, and the taps it leaves.

    The taps are the encoder features the action expert reads. They stay
    inside the iteration that made them: the loop carries latents, not taps,
    so nothing holds a step's hidden states past the step.
    """

    def __init__(self, config: Praxis1Config, backbone: nn.Module, *, episode_text: bool = False) -> None:
        super().__init__(config)
        self.backbone = backbone
        self.episode_text = episode_text

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        tensors = {
            "obs_latents": inputs["obs_latents"][0],
        }
        text = inputs.get("text_embeds")
        blank = inputs.get("text_embeds_blank")
        if bool(text) != bool(blank) or (not text and not self.episode_text):
            raise ValueError("Praxis backbone needs both text embedding streams")
        if text:
            tensors["text_embeds"] = text[0]
            tensors["text_embeds_blank"] = blank[0]
        anchor = inputs.get("anchor_latents")
        clean = inputs.get("clean_latents")
        if bool(anchor) != bool(clean):
            raise ValueError("Praxis backbone needs anchor and clean latents together")
        if anchor:
            tensors["anchor_latents"] = anchor[0]
            tensors["clean_latents"] = clean[0]
        # First iteration: the loop has not carried anything yet, so the
        # latents are the noise this request's seed draws.
        for name in ("video_latents", "action_latents"):
            carried = inputs.get(name)
            if carried:
                tensors[name] = carried[0]
        metadata = fwd_info.step_metadata
        write_text = bool(self.episode_text and metadata.get("text_write") and
                          fwd_info.dynamic_loop_iter_counts.get("praxis_denoise", 0) == 0)
        if write_text and not text:
            raise ValueError("cold Praxis text episode needs both encoder outputs")
        if write_text and (int(metadata.get("text_tokens", 0)) < 1 or
                           int(metadata.get("blank_tokens", 0)) < 1):
            raise ValueError("cold Praxis text episode needs tokenized stream lengths")
        return NodeInputs(
            tensor_inputs=tensors,
            kwargs={"write_text": write_text,
                    "text_tokens": int(metadata.get("text_tokens", 0)),
                    "blank_tokens": int(metadata.get("blank_tokens", 0))},
        )

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs) -> SubmoduleStep | None:
        if not self.episode_text:
            return None
        segments = []
        for rid, inp in zip(request_ids, inputs, strict=True):
            write = inp.kwargs["write_text"]
            segments.extend((
                Segment(rid, "cond", inp.kwargs["text_tokens"] if write else 0),
                Segment(rid, "blank", inp.kwargs["blank_tokens"] if write else 0),
            ))
        return SubmoduleStep(state={"text_embeds": ArrayStep(segments=tuple(segments))})

    def episode_text_embeddings(
        self, text_embeds: torch.Tensor | None, text_embeds_blank: torch.Tensor | None,
        *, write_text: bool, text_tokens: int, blank_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.episode_text:
            if text_embeds is None or text_embeds_blank is None:
                raise ValueError("cold Praxis call needs text embeddings")
            return text_embeds, text_embeds_blank
        handle = self.state_handles["text_embeds"]
        if write_text:
            if text_embeds is None or text_embeds_blank is None:
                raise ValueError("cold Praxis text episode needs both encoder outputs")
            if (text_embeds.shape != (1, text_tokens, self.config.text_hidden_dim) or
                    text_embeds_blank.shape != (1, blank_tokens, self.config.text_hidden_dim)):
                raise ValueError("Praxis text embeddings do not match tokenized lengths")
            handle.add(0, h=torch.cat((text_embeds[0], text_embeds_blank[0])))
        reached = handle.read(0).fields["h"]
        if reached.shape != (text_tokens + blank_tokens, self.config.text_hidden_dim):
            raise ValueError("retained Praxis text lengths differ from this prompt")
        return reached[:text_tokens][None], reached[text_tokens:][None]

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        obs_latents: torch.Tensor,
        text_embeds: torch.Tensor | None = None,
        text_embeds_blank: torch.Tensor | None = None,
        video_latents: torch.Tensor | None = None,
        action_latents: torch.Tensor | None = None,
        anchor_latents: torch.Tensor | None = None,
        clean_latents: torch.Tensor | None = None,
        write_text: bool = False,
        text_tokens: int = 0,
        blank_tokens: int = 0,
        **kwargs,
    ) -> NameToTensorList:
        text_embeds, text_embeds_blank = self.episode_text_embeddings(
            text_embeds, text_embeds_blank, write_text=write_text,
            text_tokens=text_tokens, blank_tokens=blank_tokens,
        )
        raise NotImplementedError(
            "Praxis-1 backbone step: wires praxis.model.PraxisPolicy's encoder pass"
        )


class Praxis1ActionExpertSubmodule(_Praxis1Node):
    """One Euler step of the action expert, over the backbone's taps."""

    def __init__(self, config: Praxis1Config, expert: nn.Module) -> None:
        super().__init__(config)
        self.expert = expert

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        tensors = {
                "taps": inputs["taps"][0],
                "video_latents": inputs["video_latents"][0],
                "action_latents": inputs["action_latents"][0],
                "proprio": inputs["proprio"][0],
        }
        if proprio_all := inputs.get("proprio_all"):
            tensors["proprio_all"] = proprio_all[0]
        return NodeInputs(tensor_inputs=tensors)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        taps: torch.Tensor,
        video_latents: torch.Tensor,
        action_latents: torch.Tensor,
        proprio: torch.Tensor,
        proprio_all: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        raise NotImplementedError(
            "Praxis-1 expert step: wires praxis.action_expert.run_expert_on_taps_k"
        )


class Praxis1ActionHeadSubmodule(_Praxis1Node):
    """The settled action tokens as relative rows for cold openpi calls.

    The Positronic session wire composes these relative rows with the current
    pose into absolute commands in its own serving adapter. The openpi action
    array keeps the checkpoint's relative xyz and rotation convention.
    """

    # The engine manager casts floating buffers to the node dtype. Keep q99
    # statistics in float32 so the inverse map matches the checkpoint.
    disable_autocast = True

    def __init__(self, config: Praxis1Config) -> None:
        super().__init__(config)
        stats = config.normalization
        try:
            q01 = torch.as_tensor(stats["action_q01"], dtype=torch.float32)
            q99 = torch.as_tensor(stats["action_q99"], dtype=torch.float32)
            raw_dims = stats.get("q99_normed_dims", Q99_DIMS[config.embodiment])
            if not isinstance(raw_dims, (list, tuple)) or any(type(d) is not int for d in raw_dims):
                raise ValueError("q99_normed_dims must be integer indices")
            dims = torch.as_tensor(raw_dims, dtype=torch.long)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("checkpoint lacks valid action normalization statistics") from exc
        dim = config.action_dim
        if (q01.shape != (dim,) or q99.shape != (dim,) or
                not torch.isfinite(q01).all() or not torch.isfinite(q99).all()):
            raise ValueError(f"checkpoint action quantiles need {dim} finite values")
        if (dims.ndim != 1 or dims.numel() == 0 or (dims < 0).any() or (dims >= dim).any() or
                dims.unique().numel() != dims.numel()):
            raise ValueError(f"checkpoint q99 dims must be unique indices in [0, {dim})")
        if not torch.all(q99[dims] > q01[dims]):
            raise ValueError("checkpoint action_q99 must exceed action_q01 on normalized dims")
        self.register_buffer("action_q01", q01, persistent=False)
        self.register_buffer("action_q99", q99, persistent=False)
        self.register_buffer("q99_dims", dims, persistent=False)
        self.register_buffer(
            "grip_dims", torch.tensor(GRIP_DIMS[config.embodiment], dtype=torch.long), persistent=False,
        )

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        return NodeInputs(tensor_inputs={"action_latents": inputs["action_latents"][0]})

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        action_latents: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        rows = unpack_action_tokens(
            action_latents, depth=0, action_dim=self.config.action_dim,
            frames_per_token=self.config.action_frames_per_token,
        )
        if rows.shape[0] != self.config.action_horizon:
            raise ValueError(
                f"cold Praxis-1 action head expected {self.config.action_horizon} rows, got {rows.shape[0]}"
            )
        relative = denormalize_rows(rows, self.action_q01, self.action_q99, self.q99_dims)
        return {"actions": [clip_grip(relative, self.grip_dims)]}
