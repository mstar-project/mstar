"""Praxis-1: a video-action policy for robot arms.

One cold policy call answers one chunk of the future: 6 video latents and 6
action tokens (21 relative action rows at 15 Hz). Later calls can return 24
rows once episode state is available. The backbone denoises the video over a few
Euler steps; each step hands the action expert its hidden states ("taps"), and
the expert denoises the actions from them. The robot executes the rows and
asks again.

The graph follows that shape:

* ``encode_text`` produces the task and blank-caption embeddings for a
  request. Its output persists between walks of that request, but a later
  policy call re-encodes the text until episode state can retain it.
* ``chunk`` is one policy call: VAE-encode the observation, run the denoise
  loop, emit the action rows. The loop carries the video and action latents;
  the expert reads the backbone's taps inside the same iteration, so the two
  nodes sit in one loop body rather than in separate walks.

The video a chunk predicts is denoised alongside the actions: the action
expert reads the backbone's hidden states as the video settles. It does not
condition the next call. An opt-in episode array retains the real camera
frames and aligned poses from executed rows, with no predicted frames in the
window. The VAE node writes and reads that window; a later trunk port must
consume its clean encoding. The predicted video latents end at the denoise
loop, and nothing predicted leaves the walk except the action rows.

Nothing here serves yet. The production trunk weight loader and session
validation still need to land, so a served model is refused at construction
before it fetches anything. Dummy mode (``skip_weight_loading``) builds the
graph for startup checks and CPU tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

from mstar.engine.state import (
    Array,
    Attend,
    ChunkCausal,
    ChunkCausalLayout,
    Entry,
    ExpertCross,
    ExpertCrossLayout,
    ExpertSelf,
    ExpertSelfLayout,
    KVEntry,
    Layout,
    Read,
    TensorType,
    Window,
)
from mstar.engine.state.adapters import (
    CHUNK_CAUSAL_LAYOUT_KWARG,
    EXPERT_LAYOUT_KWARG,
    chunk_causal_layouts,
    expert_layouts,
)
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
    EpisodeView,
    Model,
    ProcessPromptOutput,
    RobotIO,
    TensorAndMetadata,
    WalkStep,
)
from mstar.model.media_limits import read_image
from mstar.model.praxis1.components.commands import compose_commands
from mstar.model.praxis1.components.pose_frames import pose_to_model
from mstar.model.praxis1.components.proprio import normalized_proprio
from mstar.model.praxis1.config import Praxis1Config, Praxis1ConfigError
from mstar.model.praxis1.episode_history import BOUNDARY_STATE_NAME, STATE_NAME, append_count, clean_depth, window_size
from mstar.model.praxis1.submodules import (
    ATTN_READS,
    DECODER_BLANK_KV,
    DECODER_COND_KV,
    ENCODER_BLANK_KV,
    ENCODER_COND_KV,
    EXPERT_QUERIES,
    EXPERT_TRUNK_KEYS,
    Praxis1ActionExpertSubmodule,
    Praxis1ActionHeadSubmodule,
    Praxis1BackboneSubmodule,
    Praxis1TextEncoderSubmodule,
    Praxis1VaeEncoderSubmodule,
)

logger = logging.getLogger(__name__)

DENOISE_LOOP_NAME = "praxis_denoise"
TEXT_STATE_NAME = "text_embeds"

# The state kwarg the openpi WebSocket protocol fills from a client's request.
STATE_KWARG = "robot_state"
MODEL_POSE_KWARG = "_praxis_model_pose"

# False until get_submodule can build the nodes. Until then a served model is
# refused in the constructor, the first model code every serving process runs,
# rather than in get_submodule once every process has started. The PR that
# ports the compute deletes this.
_SERVABLE = False

# The denoise loop's ceiling when the checkpoint names none.
DEFAULT_MAX_DENOISE_STEPS = 8


class Praxis1Model(Model):
    """The served Praxis-1 policy.

    Serves the single-arm (DROID) and bimanual (YAM) checkpoints; the rig and
    every width come from the checkpoint's own config, so the two differ in
    numbers rather than in code.
    """

    ENCODE_TEXT_WALK = "encode_text"
    CHUNK_WALK = "chunk"
    CHUNK_WARM_WALK = "chunk_warm"
    INTERNAL_KWARGS = (MODEL_POSE_KWARG,)
    PREPROCESS_ONLY_KWARGS = (STATE_KWARG, MODEL_POSE_KWARG, "pose_descriptor", "control_mode")

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
        # steps than num_steps, never for more (get_loop_caps).
        self.sampler_defaults: dict = {
            "num_steps": kwargs.pop("num_steps", 2),
            "t_shift": kwargs.pop("t_shift", 12.5),
            "cfg_scale": kwargs.pop("cfg_scale", 1.0),
            "cfg_video_scale": kwargs.pop("cfg_video_scale", 4.0),
            # A cold-only call has no earlier executed-frame context.
            "max_context_chunks": kwargs.pop("max_context_chunks", 0),
        }
        self.max_text_tokens = kwargs.pop("max_text_tokens", None)
        self._yaml_config_overrides: dict = dict(kwargs)

        if not skip_weight_loading and not _SERVABLE:
            raise NotImplementedError(
                f"Praxis-1 ({model_path_hf}) is not servable yet: this build has its "
                "graph and config reader but no node computation, which lands with the "
                "compute port. Nothing was downloaded."
            )

        self._repo_dir: Path | None = None
        self._submodule_cache: dict = {}
        self._tokenizer_cache = None
        self.config: Praxis1Config = self._load_config()
        context_chunks = self.sampler_defaults["max_context_chunks"]
        if type(context_chunks) is not int or not 0 <= context_chunks < self.config.n_chunks:
            raise Praxis1ConfigError(
                f"max_context_chunks must be in [0, {self.config.n_chunks - 1}]; got {context_chunks!r}"
            )
        if context_chunks and (type(self.max_text_tokens) is not int or self.max_text_tokens < 1):
            raise Praxis1ConfigError("episode text reuse needs a positive max_text_tokens deployment cap")
        # A recipe past the loop's ceiling would have every request refused at
        # the conductor, so it is refused once, here.
        num_steps = self.sampler_defaults["num_steps"]
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or not (
            1 <= num_steps <= self.max_denoise_steps
        ):
            raise Praxis1ConfigError(
                f"num_steps must be an integer in [1, {self.max_denoise_steps}]; got {num_steps!r}"
            )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _ensure_repo(self) -> Path:
        if self._repo_dir is not None:
            return self._repo_dir
        from huggingface_hub import snapshot_download

        # config.json is all the graph needs, and every process that builds
        # the model builds the graph: the API server, the conductor and the
        # data worker read nothing else. The weights are fetched per submodule,
        # by get_submodule on the worker that holds them, so no other process
        # pays for a download it never reads.
        local = snapshot_download(
            repo_id=self.model_path_hf,
            cache_dir=self.cache_dir,
            allow_patterns=["config.json"],
        )
        self._repo_dir = Path(local)
        return self._repo_dir

    def _ensure_component_weights(self, prefix: str) -> Path:
        """Fetch only indexed shards needed by one tower, pinned to the config revision."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        from mstar.model.praxis1.weight_loader import INDEX, SINGLE, required_shards

        root = self._ensure_repo()
        revision = root.name if root.parent.name == "snapshots" else None
        if not (root / INDEX).is_file() and not (root / SINGLE).is_file():
            if revision is None:
                raise FileNotFoundError(f"Praxis weights lack {INDEX} or {SINGLE} in {root}")
            try:
                hf_hub_download(self.model_path_hf, INDEX, revision=revision, cache_dir=self.cache_dir)
            except EntryNotFoundError:
                hf_hub_download(self.model_path_hf, SINGLE, revision=revision, cache_dir=self.cache_dir)
        for filename in required_shards(root, prefix):
            if (root / filename).is_file():
                continue
            if revision is None:
                raise FileNotFoundError(f"Praxis weights lack {filename} in {root}")
            hf_hub_download(self.model_path_hf, filename, revision=revision, cache_dir=self.cache_dir)
        return root

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

    @property
    def max_denoise_steps(self) -> int:
        """The denoise loop's ``max_iters``: a ceiling, not the step count."""
        return int(self.config.raw.get("max_denoise_steps", DEFAULT_MAX_DENOISE_STEPS))

    # ------------------------------------------------------------------
    # Model ABC: the graph
    # ------------------------------------------------------------------

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # Both embeddings persist between walks of this request. A later
        # policy call re-encodes them until episode state can retain them.
        encode_text = GraphNode(
            name="text_encoder",
            input_names=["text_ids", "text_mask", "blank_ids", "blank_mask"],
            outputs=[
                GraphEdge(next_node=EMPTY_DESTINATION, name="text_embeds", persist=True),
                GraphEdge(next_node=EMPTY_DESTINATION, name="text_embeds_blank", persist=True),
            ],
        )

        def chunk_graph(needs_text: bool) -> Sequential:
            # One canvas per observation. When episode history is enabled, the
            # VAE node also stores the newly executed real frames with their poses.
            vae_encoder = GraphNode(
                name="vae_encoder",
                input_names=["image_inputs", "executed_pose_inputs", "executed_pose_valid"]
                if self.sampler_defaults["max_context_chunks"] else ["image_inputs"],
                outputs=[GraphEdge(next_node="backbone", name="obs_latents"),
                         GraphEdge(next_node="backbone", name="video_latents"),
                         GraphEdge(next_node="backbone", name="action_latents"), *(
                    [GraphEdge(next_node="backbone", name="anchor_latents"),
                     GraphEdge(next_node="backbone", name="clean_latents"),
                     GraphEdge(next_node="action_expert", name="proprio_all")]
                    if self.sampler_defaults["max_context_chunks"] else []
                )],
            )

            # One Euler step: the backbone denoises the video window and leaves
            # its taps; the expert denoises the actions from them. The two carry
            # their own latents round the loop.
            backbone = GraphNode(
                name="backbone",
                input_names=[
                    "obs_latents",
                    *(["anchor_latents", "clean_latents"] if self.sampler_defaults["max_context_chunks"] else []),
                    *(["text_embeds", "text_embeds_blank"] if needs_text else []),
                    "video_latents",
                    "action_latents",
                ],
                outputs=[
                    GraphEdge(next_node="action_expert", name="taps"),
                    GraphEdge(next_node="action_expert", name="tap_embeddings"),
                    GraphEdge(next_node="action_expert", name="tap_rope_ids"),
                    GraphEdge(next_node="action_expert", name="tap_geometry"),
                    GraphEdge(next_node="action_expert", name="expert_video_latents"),
                    # Distinct from the loop output: only the expert's updated
                    # action_latents may be cached for the action head.
                    GraphEdge(next_node="action_expert", name="expert_action_latents"),
                ],
            )
            action_expert = GraphNode(
                name="action_expert",
                input_names=["taps", "tap_embeddings", "tap_rope_ids", "tap_geometry",
                             "expert_video_latents", "expert_action_latents", "proprio", *(
                    ["proprio_all"] if self.sampler_defaults["max_context_chunks"] else []
                )],
                outputs=[
                    GraphEdge(next_node="backbone", name="video_latents"),
                    GraphEdge(next_node="backbone", name="action_latents"),
                ],
                # get_loop_caps ends the loop on a request's last step, so nothing
                # speculates past it
                enable_async_scheduling=True,
            )

            denoise_loop = Loop(
                name=DENOISE_LOOP_NAME,
                section=Sequential([backbone, action_expert]),
                # Ceiling only; get_loop_caps gives each request its step count.
                max_iters=self.max_denoise_steps,
                # Only the actions leave the loop. The predicted video has no
                # reader after the last step: the reply is command rows, and the
                # next call is conditioned on real frames, not on this video.
                outputs=[
                    GraphEdge(next_node="action_head", name="action_latents"),
                ],
            )

            # The head turns the settled action latents into command rows. That
            # happens once per call, after the last Euler step, so it sits after
            # the loop rather than in its body.
            action_head = GraphNode(
                name="action_head",
                input_names=["action_latents"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="actions",
                        output_modality="action",
                    ),
                ],
            )

            return Sequential([vae_encoder, denoise_loop, action_head])

        graphs = {self.ENCODE_TEXT_WALK: encode_text, self.CHUNK_WALK: chunk_graph(True)}
        if self.sampler_defaults["max_context_chunks"]:
            graphs[self.CHUNK_WARM_WALK] = chunk_graph(False)
        return graphs

    def declare_state(self) -> list:
        """Declare within-step attention and optional executed-frame episode state."""
        depth = self.sampler_defaults["max_context_chunks"]
        c = self.config
        p = (c.latent_height // c.patch_size[1]) * (c.latent_width // c.patch_size[2])
        f = c.latents_per_chunk
        future = int(c.raw.get("action_expert", {}).get("num_future_per_chunk", 8))
        clean_only = bool(c.raw.get("action_expert", {}).get("clean_only_conditioning", False))
        trunk = [
            Array(name, nodes={"backbone"}, labels={name},
                  layers=set(range(c.encoder_depth if "encoder" in name else c.decoder_depth)),
                  entry=KVEntry(heads=c.num_kv_heads, head_dim=c.head_dim), max_len=0,
                  reads={ATTN_READS[name]: Read(mask=ChunkCausal(f * p if "encoder" in name else f * p * 4),
                                                op=Attend(q_heads=c.num_heads))})
            for name in (ENCODER_COND_KV, ENCODER_BLANK_KV, DECODER_COND_KV, DECODER_BLANK_KV)
        ]
        expert = [
            Array(EXPERT_QUERIES, nodes={"action_expert"}, labels={EXPERT_QUERIES},
                  layers=set(range(1, c.expert_layers, 2)),
                  entry=KVEntry(heads=c.expert_heads, head_dim=c.expert_head_dim), max_len=0,
                  reads={ATTN_READS[EXPERT_QUERIES]: Read(mask=ExpertSelf(f, future),
                                                        op=Attend(q_heads=c.expert_heads))}),
            Array(EXPERT_TRUNK_KEYS, nodes={"action_expert"}, labels={EXPERT_TRUNK_KEYS},
                  layers=set(range(0, c.expert_layers, 2)),
                  entry=KVEntry(heads=c.expert_heads, head_dim=c.expert_head_dim), max_len=0,
                  reads={ATTN_READS[EXPERT_TRUNK_KEYS]: Read(
                      mask=ExpertCross(f, future, p, c.window_frames, include_noised=not clean_only),
                      op=Attend(q_heads=c.expert_heads), queries=EXPERT_QUERIES)}),
        ]
        if depth == 0:
            return [*trunk, *expert]
        pose_width = 8 if self.config.embodiment == "single_arm" else 16
        return [*trunk, *expert, Array(
            STATE_NAME,
            nodes={"vae_encoder"}, layers={0},
            entry=Entry({
                "frame": TensorType(
                    (3, self.config.tile_height, self.config.tile_width * self.config.num_cameras),
                    dtype=torch.uint8,
                ),
                "pose": TensorType((pose_width,), dtype=torch.float32),
                # A missing historical pose leaves the camera history valid.
                # The dense pool's fresh-row gather promotes bool to int64,
                # so validity is stored as uint8 0/1.
                "pose_valid": TensorType((1,), dtype=torch.uint8),
            }),
            max_len=None,
            reads={"history": Read(mask=Window(window_size(depth)))},
            lifetime="episode",
        ), Array(
            BOUNDARY_STATE_NAME,
            nodes={"vae_encoder"}, layers={0},
            entry=Entry({
                "pose": TensorType((pose_width,), dtype=torch.float32),
                "history_valid": TensorType((1,), dtype=torch.uint8),
            }),
            max_len=None,
            reads={"recent": Read(mask=Window(depth + 1))},
            lifetime="episode",
        ), Array(
            TEXT_STATE_NAME,
            nodes={"backbone"}, labels={"cond", "blank"}, layers={0},
            entry=Entry({"h": TensorType((self.config.text_hidden_dim,), dtype=torch.bfloat16)}),
            max_len=None,
            reads={"praxis_text_history": Read(mask=Window(self.max_text_tokens))},
            lifetime="episode",
        )]

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------

    def robot_io(self) -> RobotIO:
        return RobotIO(STATE_KWARG, self.config.action_horizon, self.config.action_dim)

    def request_kwargs(self) -> frozenset[str]:
        # The sampler knobs come from the deployment YAML. What a request
        # carries is the arm's pose, how much of the last chunk it ran, and
        # optionally fewer Euler steps than the deployment's recipe.
        return frozenset({
            STATE_KWARG, "executed_rows", "num_steps", "action_format", "pose_descriptor", "control_mode",
        })

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        """The denoise loop runs the deployment's ``num_steps``, or the request's if fewer.

        The cap is what ends the loop on a request's last Euler step: without
        it the loop runs its ``max_iters`` ceiling, and the expert's async
        scheduling would speculate a step past the last one. A request may ask
        for fewer steps than the recipe, never more: the recipe is what the
        checkpoint was tuned for, and the constructor already holds it under
        the loop's ceiling.
        """
        recipe = self.sampler_defaults["num_steps"]
        steps = model_kwargs.get("num_steps")
        if steps is None:
            return {DENOISE_LOOP_NAME: recipe}
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(f"num_steps must be an integer >= 1; got {steps!r}")
        if steps > recipe:
            raise ValueError(
                f"num_steps {steps} is more than this deployment's {recipe}; "
                "a request may ask for fewer steps, never more"
            )
        return {DENOISE_LOOP_NAME: steps}

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
            rows = output.detach().cpu().to(torch.float32).numpy()
            kwargs = request_kwargs or {}
            output_format = kwargs.get("action_format", "openpi")
            if output_format == "openpi":
                return rows.tobytes()
            if output_format == "positronic":
                from mstar.api_server.openpi import pack

                model_pose = kwargs.get(MODEL_POSE_KWARG)
                if model_pose is None:
                    raise ValueError("Positronic commands need a model-frame pose from process_prompt")
                commands = compose_commands(
                    rows, model_pose, self.config.embodiment,
                    fps=self.config.grid_hz, descriptor=kwargs.get("pose_descriptor", ""),
                    control_mode=kwargs.get("control_mode", "none"),
                )
                return pack(commands)
            raise ValueError(f"unsupported Praxis-1 action format {output_format!r}")
        raise ValueError(f"Unsupported modality for Praxis-1: {modality!r}")

    def process_prompt(
        self, prompt: str | None, *args, robot_state: object = None, **kwargs
    ) -> ProcessPromptOutput:
        """Tokenize the task text and the blank caption, once per request.

        Both are tokenized here rather than in the node, so the node's
        forward stays tensor in and tensor out. The padding is the
        reference's: up to a multiple of 32, with the mask carried beside the
        ids so the node can trim the states back afterwards.

        A blank prompt becomes a single space rather than the empty string,
        which is what the reference does and what leaves the unconditional
        branch a token to read.
        """
        tok = self._tokenizer()
        task = prompt if (prompt and prompt.strip()) else " "
        blank = self._blank_caption()
        pad = self.config.text_pad_to_multiple_of

        def encode(text: str) -> tuple:
            batch = tok(
                text=[text],
                padding=True,
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True,
                pad_to_multiple_of=pad,
                return_tensors="pt",
            )
            return batch["input_ids"][0], batch["attention_mask"][0]

        task_ids, task_mask = encode(task)
        blank_ids, blank_mask = encode(blank)
        episode_keys = None
        if self.sampler_defaults["max_context_chunks"]:
            if max(int(task_mask.sum()), int(blank_mask.sum())) > self.max_text_tokens:
                raise ValueError(f"Praxis prompt exceeds max_text_tokens={self.max_text_tokens}")
            key_payload = json.dumps({
                "task_ids": task_ids.tolist(), "task_mask": task_mask.tolist(),
                "blank_ids": blank_ids.tolist(), "blank_mask": blank_mask.tolist(),
                "encoder": self.text_encoder_hf, "hidden_layer": self.config.text_hidden_layer,
            }, sort_keys=True).encode()
            episode_keys = {TEXT_STATE_NAME: hashlib.sha256(key_payload).hexdigest()}
        metadata = {"text_tokens": int(task_mask.sum()), "blank_tokens": int(blank_mask.sum())}
        descriptor = kwargs.get("pose_descriptor", "")
        model_pose = pose_to_model(robot_state, self.config.embodiment, descriptor)
        if kwargs.get("action_format") == "positronic":
            metadata[MODEL_POSE_KWARG] = model_pose.tolist()
        robot_state = model_pose.tolist()
        proprio = normalized_proprio(robot_state, self.config.embodiment, self.config.normalization)
        new_tensors = {
            "text_ids": [task_ids],
            "text_mask": [task_mask],
            "blank_ids": [blank_ids],
            "blank_mask": [blank_mask],
            "proprio": [proprio],
        }
        if self.sampler_defaults["max_context_chunks"]:
            tensors = kwargs.get("tensors") or {}
            pose_width = 8 if self.config.embodiment == "single_arm" else 16
            current = tensors.get("image_inputs") or []
            previous = tensors.get("executed_frame_inputs") or []
            prior_poses = tensors.get("executed_pose_inputs") or []
            if len(current) != self.config.num_cameras:
                raise ValueError("Praxis episode needs one current image per camera")
            if previous and len(previous) != len(current):
                raise ValueError("Praxis executed camera stacks must match the current views")
            if len(prior_poses) > 1:
                raise ValueError("Praxis executed poses need at most one stack")
            views = []
            n_prior = 0
            for index, frame in enumerate(current):
                if frame.ndim != 3 or frame.shape[0] != 3:
                    raise ValueError("Praxis current images must have shape (3,H,W)")
                history = previous[index] if previous else frame.new_empty((0, *frame.shape))
                if history.ndim != 4 or tuple(history.shape[1:]) != tuple(frame.shape):
                    raise ValueError("Praxis executed images must have shape (T,3,H,W) matching current")
                if index and history.shape[0] != n_prior:
                    raise ValueError("Praxis executed camera stacks have different lengths")
                n_prior = history.shape[0]
                views.append(torch.cat((history, frame.unsqueeze(0)), dim=0))
            if prior_poses:
                raw_pose = prior_poses[0]
                if raw_pose.ndim != 2 or raw_pose.shape[1] != pose_width or raw_pose.shape[0] not in (0, n_prior):
                    raise ValueError("Praxis executed poses must be absent or align with executed camera rows")
                poses_available = raw_pose.shape[0] == n_prior
                if poses_available:
                    pose_history = torch.stack([
                        torch.from_numpy(pose_to_model(row.cpu().numpy(), self.config.embodiment, descriptor))
                        for row in raw_pose
                    ]) if n_prior else torch.empty((0, raw_pose.shape[1]), dtype=torch.float32)
                else:
                    pose_history = torch.zeros((n_prior, pose_width), dtype=torch.float32)
            else:
                poses_available = n_prior == 0
                pose_history = torch.zeros((n_prior, pose_width), dtype=torch.float32)
            new_tensors["image_inputs"] = views
            new_tensors["executed_pose_inputs"] = [torch.cat((
                pose_history, torch.as_tensor(robot_state, dtype=torch.float32).reshape(1, -1)
            ), dim=0)]
            new_tensors["executed_pose_valid"] = [torch.cat((
                torch.full((n_prior, 1), int(poses_available), dtype=torch.uint8),
                torch.ones((1, 1), dtype=torch.uint8),
            ), dim=0)]
            # The data worker merges this output into its tensor map. The
            # original stack has been folded into image_inputs and must not
            # travel to the conductor as a second unused signal.
            tensors.pop("executed_frame_inputs", None)
        elif kwargs.get("tensors") is not None:
            # Cold-only serving never declares an episode window.
            kwargs["tensors"].pop("executed_frame_inputs", None)
            kwargs["tensors"].pop("executed_pose_inputs", None)
        return ProcessPromptOutput(
            new_input_tensors=new_tensors,
            metadata=metadata,
            episode_keys=episode_keys,
        )

    def _blank_caption(self) -> str:
        """The unconditional caption the checkpoint names, or a single space."""
        text = str(self.config.raw.get("text_encoder", {}).get("blank_caption", " "))
        return text if text.strip() else " "

    def _tokenizer(self):
        if self._tokenizer_cache is None:
            from transformers import AutoProcessor

            self._tokenizer_cache = AutoProcessor.from_pretrained(self.text_encoder_hf)
        return self._tokenizer_cache

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
        episode: EpisodeView | None = None,
    ) -> list[WalkStep]:
        """Encode text if this request lacks embeddings, then run its chunk."""
        if not input_signals.get("image_inputs"):
            raise ValueError("Praxis-1 requires camera images (image_inputs)")
        steps: list[WalkStep] = []
        has_episode_text = episode is not None and episode.held.get(TEXT_STATE_NAME, 0) > 0
        if not has_episode_text and not input_signals.get("text_embeds"):
            steps.append(WalkStep(self.ENCODE_TEXT_WALK))
        metadata = {
            "num_steps": self.get_loop_caps(**model_kwargs)[DENOISE_LOOP_NAME],
            "t_shift": self.sampler_defaults["t_shift"],
            "cfg_scale": self.sampler_defaults["cfg_scale"],
            "cfg_video_scale": self.sampler_defaults["cfg_video_scale"],
            "action_format": model_kwargs.get("action_format", "openpi"),
            "text_tokens": model_kwargs.get("text_tokens", 0),
            "blank_tokens": model_kwargs.get("blank_tokens", 0),
        }
        if metadata["cfg_scale"] != 1.0:
            raise ValueError("Praxis expert CFG needs a blank tap stream, which this graph has not declared")
        if self.sampler_defaults["max_context_chunks"]:
            metadata["text_write"] = not has_episode_text
            held = 0 if episode is None else episode.held.get(STATE_NAME, 0)
            if held and metadata["action_format"] != "positronic":
                raise ValueError("warm Praxis episodes require Positronic action_format")
            rows = model_kwargs.get("executed_rows", 0)
            metadata["history_append"] = append_count(held, rows)
            metadata["history_held"] = held
        else:
            held = 0
        depth = clean_depth(held + metadata.get("history_append", 0), self.sampler_defaults["max_context_chunks"])
        k = depth + 1
        c = self.config
        p = (c.latent_height // c.patch_size[1]) * (c.latent_width // c.patch_size[2])
        f = c.latents_per_chunk
        g = c.num_registers + p
        decoder_tokens = k * f * p * c.patch_size[1] * c.patch_size[2]
        text = int(model_kwargs.get("text_tokens", 0))
        blank = int(model_kwargs.get("blank_tokens", 0))
        layout = {
            ENCODER_COND_KV: {ENCODER_COND_KV: Layout(chunk_causal={
                ATTN_READS[ENCODER_COND_KV]: ChunkCausalLayout(
                    global_len=g + text, noised_len=k * f * p, clean_len=depth * f * p,
                ),
            })},
            ENCODER_BLANK_KV: {ENCODER_BLANK_KV: Layout(chunk_causal={
                ATTN_READS[ENCODER_BLANK_KV]: ChunkCausalLayout(
                    global_len=g + blank, noised_len=k * f * p, clean_len=depth * f * p,
                ),
            })},
            DECODER_COND_KV: {DECODER_COND_KV: Layout(chunk_causal={
                ATTN_READS[DECODER_COND_KV]: ChunkCausalLayout(
                    global_len=c.num_registers, noised_len=decoder_tokens,
                ),
            })},
            DECODER_BLANK_KV: {DECODER_BLANK_KV: Layout(chunk_causal={
                ATTN_READS[DECODER_BLANK_KV]: ChunkCausalLayout(
                    global_len=c.num_registers, noised_len=decoder_tokens,
                ),
            })},
            EXPERT_QUERIES: {EXPERT_QUERIES: Layout(expert_self={
                ATTN_READS[EXPERT_QUERIES]: ExpertSelfLayout(k),
            })},
            EXPERT_TRUNK_KEYS: {EXPERT_TRUNK_KEYS: Layout(expert_cross={
                ATTN_READS[EXPERT_TRUNK_KEYS]: ExpertCrossLayout(k, g + text, depth * f, k * f),
            })},
        }
        declared = self.declare_state()
        metadata[CHUNK_CAUSAL_LAYOUT_KWARG] = chunk_causal_layouts(declared, layout)
        metadata[EXPERT_LAYOUT_KWARG] = expert_layouts(declared, layout)
        steps.append(
            WalkStep(
                self.CHUNK_WARM_WALK if has_episode_text else self.CHUNK_WALK,
                step_metadata=metadata,
            )
        )
        return steps

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> torch.nn.Module | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(self, node_name: str, device: str) -> torch.nn.Module | None:
        """Build one node from the checkpoint and the two public components.

        Dummy mode returns None, which is what lets the graph, the walks and
        the request layouts be tested on a machine with no weights.

        The trunk nodes are not here yet. They raise rather than returning
        None, because None means "run this node without real computation" and
        a policy that silently emits nothing would look like a working server
        that commands an arm badly.
        """
        if self.skip_weight_loading:
            return None
        if node_name == "text_encoder":
            from transformers import AutoModel

            encoder = (
                AutoModel.from_pretrained(
                    self.text_encoder_hf,
                    cache_dir=self.cache_dir,
                    dtype=torch.bfloat16,
                )
                .eval()
                .requires_grad_(False)
            )
            return Praxis1TextEncoderSubmodule(self.config, encoder)
        if node_name == "vae_encoder":
            from diffusers import AutoencoderKLWan

            vae = (
                AutoencoderKLWan.from_pretrained(
                    self.vae_weights_hf,
                    subfolder="vae",
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.bfloat16,
                )
                .eval()
                .requires_grad_(False)
            )
            return Praxis1VaeEncoderSubmodule(
                self.config, vae, max_context_chunks=self.sampler_defaults["max_context_chunks"]
            )
        if node_name == "action_head":
            return Praxis1ActionHeadSubmodule(self.config)
        if node_name == "backbone":
            from mstar.model.praxis1.native_towers import build_backbone
            from mstar.model.praxis1.weight_loader import load_component

            with torch.device("meta"):
                backbone = build_backbone(self.config)
            for component in ("encoder", "decoder"):
                prefix = f"backbone.{component}."
                root = self._ensure_component_weights(prefix)
                load_component(getattr(backbone, component), root, prefix, assign=True)
            return Praxis1BackboneSubmodule(
                self.config, backbone, episode_text=self.sampler_defaults["max_context_chunks"] > 0,
            ).to(device)
        if node_name == "action_expert":
            from mstar.model.praxis1.native_towers import build_expert
            from mstar.model.praxis1.weight_loader import load_component

            with torch.device("meta"):
                expert = build_expert(self.config)
            root = self._ensure_component_weights("action_expert.")
            load_component(expert, root, "action_expert.", assign=True)
            return Praxis1ActionExpertSubmodule(self.config, expert).to(device)
        raise ValueError(f"Praxis-1 has no node {node_name!r}")
