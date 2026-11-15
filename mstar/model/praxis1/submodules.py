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
from mstar.model.components.diffusion.decode_utils import latent_affine
from mstar.model.praxis1.components.canvas import (
    build_canvas,
    tile_view_single_arm,
    tile_views_bimanual,
)
from mstar.model.praxis1.config import Praxis1Config
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

    def __init__(self, config: Praxis1Config, vae: nn.Module) -> None:
        super().__init__(config)
        self.vae = vae
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
        return NodeInputs(tensor_inputs={f"view_{i}": f for i, f in enumerate(frames)})

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **views: torch.Tensor,
    ) -> NameToTensorList:
        device = self.get_device()
        with torch.inference_mode():
            tiles = [
                views[f"view_{i}"].to(device) for i in range(self.config.num_cameras)
            ]
            canvas = build_canvas(self._tiles(tiles))
            latent = self.vae_encode(canvas)
        return {"obs_latents": [latent]}

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

    def __init__(self, config: Praxis1Config, backbone: nn.Module) -> None:
        super().__init__(config)
        self.backbone = backbone

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        tensors = {
            "obs_latents": inputs["obs_latents"][0],
            "text_embeds": inputs["text_embeds"][0],
            "text_embeds_blank": inputs["text_embeds_blank"][0],
        }
        # First iteration: the loop has not carried anything yet, so the
        # latents are the noise this request's seed draws.
        for name in ("video_latents", "action_latents"):
            carried = inputs.get(name)
            if carried:
                tensors[name] = carried[0]
        return NodeInputs(tensor_inputs=tensors)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        obs_latents: torch.Tensor,
        text_embeds: torch.Tensor,
        text_embeds_blank: torch.Tensor,
        video_latents: torch.Tensor | None = None,
        action_latents: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
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
        return NodeInputs(
            tensor_inputs={
                "taps": inputs["taps"][0],
                "video_latents": inputs["video_latents"][0],
                "action_latents": inputs["action_latents"][0],
                "proprio": inputs["proprio"][0],
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        taps: torch.Tensor,
        video_latents: torch.Tensor,
        action_latents: torch.Tensor,
        proprio: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        raise NotImplementedError(
            "Praxis-1 expert step: wires praxis.action_expert.run_expert_on_taps_k"
        )


class Praxis1ActionHeadSubmodule(_Praxis1Node):
    """The settled action latents as command rows.

    It will emit float32 commands in the rig's own units. Predicted video
    latents end at the denoise loop; a later call uses the executed camera
    frames as context.
    """

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        action_latents: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        raise NotImplementedError(
            "Praxis-1 action head: unpatches the action tokens and unnormalizes the rows"
        )
