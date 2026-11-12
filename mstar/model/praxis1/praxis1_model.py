"""Praxis-1: a video-action policy for robot arms.

One policy call answers one chunk of the future: 6 video latents and 6 action
tokens (24 command rows at 15 Hz). The backbone denoises the video over a few
Euler steps; each step hands the action expert its hidden states ("taps"), and
the expert denoises the actions from them. The robot executes the rows and
asks again.

The graph follows that shape:

* ``encode_text`` runs once per episode. The task text is fixed for the run,
  so its embedding (and the blank-caption embedding the video guidance needs)
  persists at the conductor rather than being recomputed per chunk.
* ``chunk`` is one policy call: VAE-encode the observation, run the denoise
  loop, emit the action rows. The loop carries the video and action latents;
  the expert reads the backbone's taps inside the same iteration, so the two
  nodes sit in one loop body rather than in separate walks.

The video of a chunk is predicted whether or not anyone wants to look at it:
it is what conditions the next call. Decoding it back to pixels is not, so the
VAE decoder only runs for a request that asked for video.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from mstar.engine.resources import NodeResourceSpec
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import (
    Model,
    ProcessPromptOutput,
    RobotIO,
    TensorAndMetadata,
    WalkStep,
)
from mstar.model.media_limits import read_image
from mstar.model.praxis1.config import Praxis1Config, Praxis1ConfigError

logger = logging.getLogger(__name__)

DENOISE_LOOP_NAME = "praxis_denoise"

# The state kwarg the openpi WebSocket protocol fills from a client's request.
STATE_KWARG = "robot_state"


class Praxis1Model(Model):
    """The served Praxis-1 policy.

    Serves the single-arm (DROID) and bimanual (YAM) checkpoints; the rig and
    every width come from the checkpoint's own config, so the two differ in
    numbers rather than in code.
    """

    ENCODE_TEXT_WALK = "encode_text"
    CHUNK_WALK = "chunk"

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        **kwargs,
    ) -> None:
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.skip_weight_loading = skip_weight_loading
        # The checkpoint needs two public components beside it, neither of
        # which it ships: the Wan2.2 VAE and the Qwen3.5-2B text encoder.
        self.vae_weights_hf: str = kwargs.pop("vae_weights_hf", "Wan-AI/Wan2.2-TI2V-5B")
        self.text_encoder_hf: str = kwargs.pop("text_encoder_hf", "Qwen/Qwen3.5-2B")
        # Sampler defaults of the deployment; a request may ask for fewer
        # steps, never for more than the loop's cap.
        self.sampler_defaults: dict = {
            "num_steps": kwargs.pop("num_steps", 2),
            "t_shift": kwargs.pop("t_shift", 12.5),
            "cfg_scale": kwargs.pop("cfg_scale", 1.0),
            "cfg_video_scale": kwargs.pop("cfg_video_scale", 4.0),
            "max_context_chunks": kwargs.pop("max_context_chunks", 1),
        }
        self._yaml_config_overrides: dict = dict(kwargs)

        self._repo_dir: Path | None = None
        self._submodule_cache: dict = {}
        self.config: Praxis1Config = self._load_config()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _ensure_repo(self) -> Path:
        if self._repo_dir is not None:
            return self._repo_dir
        from huggingface_hub import snapshot_download

        # config.json is all the data worker needs to build the graph; the
        # weights come later, per submodule, on the worker that holds them.
        local = snapshot_download(
            repo_id=self.model_path_hf,
            cache_dir=self.cache_dir,
            allow_patterns=["config.json"] if self.skip_weight_loading else None,
        )
        self._repo_dir = Path(local)
        return self._repo_dir

    def _rig(self) -> str:
        """The rig this registry entry serves, from the checkpoint it names."""
        rig = self._yaml_config_overrides.get("rig")
        if rig:
            return str(rig)
        name = self.model_path_hf.rsplit("/", 1)[-1].lower()
        for suffix, embodiment in (("yam", "bimanual"), ("droid", "single_arm")):
            if suffix in name:
                return embodiment
        raise Praxis1ConfigError(
            f"cannot tell the rig from {self.model_path_hf!r}; pass rig= in model_kwargs"
        )

    def _load_config(self) -> Praxis1Config:
        """Read the checkpoint's own config.

        Every width belongs to the checkpoint, so a served model reads it: a
        default would build a graph whose shapes disagree with the weights it
        is about to load. Dummy mode has no checkpoint to read, and takes the
        published shape of the rig instead.
        """
        if self.skip_weight_loading:
            return Praxis1Config.for_rig(self._rig())
        return Praxis1Config.from_checkpoint(self._ensure_repo())

    # ------------------------------------------------------------------
    # Model ABC: the graph
    # ------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # The task text is constant for an episode and the blank caption is
        # constant for the checkpoint, so both embeddings persist and no
        # policy call pays for the text encoder.
        encode_text = GraphNode(
            name="text_encoder",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(next_node=EMPTY_DESTINATION, name="text_embeds", persist=True),
                GraphEdge(next_node=EMPTY_DESTINATION, name="text_embeds_blank", persist=True),
            ],
        )

        # The observation: one tile per camera view on a single canvas, taken
        # to latents once per call. The executed past arrives as clean latents
        # the client already has, so only the new frames are encoded here.
        vae_encoder = GraphNode(
            name="vae_encoder",
            input_names=["image_inputs"],
            outputs=[GraphEdge(next_node="backbone", name="obs_latents")],
        )

        # One Euler step: the backbone denoises the video window and leaves
        # its taps; the expert denoises the actions from them. The two carry
        # their own latents round the loop.
        backbone = GraphNode(
            name="backbone",
            input_names=[
                "obs_latents",
                "text_embeds",
                "text_embeds_blank",
                "video_latents",
                "action_latents",
            ],
            outputs=[
                GraphEdge(next_node="action_expert", name="taps"),
                GraphEdge(next_node="action_expert", name="video_latents"),
            ],
        )
        action_expert = GraphNode(
            name="action_expert",
            input_names=["taps", "video_latents", "action_latents"],
            outputs=[
                GraphEdge(next_node="backbone", name="video_latents"),
                GraphEdge(next_node="backbone", name="action_latents"),
            ],
            # the loop's cap keeps speculation off a request's last step
            enable_async_scheduling=True,
        )

        denoise_loop = Loop(
            name=DENOISE_LOOP_NAME,
            section=Sequential([backbone, action_expert]),
            # Ceiling only; the request's step count is the loop's cap.
            max_iters=self.config.raw.get("max_denoise_steps", 8),
            outputs=[
                GraphEdge(next_node="action_head", name="action_latents"),
                GraphEdge(next_node="action_head", name="video_latents"),
            ],
        )

        # The head turns the settled action latents into command rows. It is
        # its own node because the video latents leave the walk here too: a
        # request that wants the predicted video takes them to the decoder,
        # and one that does not pays nothing for it.
        action_head = GraphNode(
            name="action_head",
            input_names=["action_latents", "video_latents"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="actions",
                    output_modality="action",
                ),
                GraphEdge(next_node=EMPTY_DESTINATION, name="chunk_latents", persist=True),
            ],
        )

        return {
            self.ENCODE_TEXT_WALK: encode_text,
            self.CHUNK_WALK: Sequential([vae_encoder, denoise_loop, action_head]),
        }

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """No KV cache and no sampler.

        Nothing here decodes tokens: the backbone re-reads its whole window
        every step, the expert's attention is over one chunk, and the actions
        come from flow matching rather than from sampling. The context a
        session keeps between calls is the executed chunks' latents, which
        ride the request as the client's frame stack rather than living in a
        cache the engine owns.
        """
        return []

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------

    def robot_io(self) -> RobotIO:
        return RobotIO(STATE_KWARG, self.config.action_horizon, self.config.action_dim)

    def request_kwargs(self) -> frozenset[str]:
        # The sampler knobs come from the deployment YAML. What a request
        # carries is the arm's pose and how much of the last chunk it ran.
        return frozenset({STATE_KWARG, "executed_rows"})

    def max_input_files(self) -> dict[str, int]:
        return {"image": self.config.num_cameras}

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """Camera frames stay uint8 ``(C, H, W)``.

        The canvas is built on the GPU, so the API server stages a quarter of
        the bytes and does no float pass per camera.
        """
        return self.load_decoded_image(read_image(filepath, rgb=True), device)

    def load_decoded_image(self, image: torch.Tensor, device: str) -> TensorAndMetadata:
        return TensorAndMetadata(image.to(device))

    def postprocess(
        self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None
    ) -> bytes:
        if modality == "action":
            # float32 rows, as the openpi wire wants them.
            return output.detach().cpu().to(torch.float32).numpy().tobytes()
        raise ValueError(f"Unsupported modality for Praxis-1: {modality!r}")

    def process_prompt(self, prompt: str | None, *args, **kwargs) -> ProcessPromptOutput:
        raise NotImplementedError(
            "Praxis-1 prompt handling lands with the text encoder port"
        )

    # ------------------------------------------------------------------
    # Model ABC: forward pass orchestration
    # ------------------------------------------------------------------

    def walk_plan(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict,
    ) -> list[WalkStep]:
        """Encode the text once per episode, then one walk per policy call."""
        if not input_signals.get("image_inputs"):
            raise ValueError("Praxis-1 requires camera images (image_inputs)")
        steps: list[WalkStep] = []
        if not input_signals.get("text_embeds"):
            steps.append(WalkStep(self.ENCODE_TEXT_WALK))
        steps.append(WalkStep(self.CHUNK_WALK))
        return steps

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> torch.nn.Module | None:
        raise NotImplementedError(
            f"Praxis-1 submodule {node_name!r} lands with the compute port"
        )
