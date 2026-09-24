"""WaypointModel: Waypoint-1.5-1B interactive video world model.

Architecture (two nodes):
    vae_encoder - TAEHV encode: the seed clip (``temporal_compression`` raw
                  frames) into the one latent frame it stands for.
    dit         - the 1.28B world DiT. One engine step is one latent frame:
                  four frozen Euler denoise passes plus one committing cache
                  pass, then a TAEHV decode of that frame, fused into a single
                  ``forward`` so a same-worker speculative N+1 can start while
                  N's frame is still going out.

Graph walks (2):
    prime   - vae_encoder -> dit.append_frame(+decode). Seeds the world and
              decoder state from a real frame and emits nothing.
    rollout - Loop("rollout_loop") over a single dit.generate_frame(+decode)
              node with a self loop-back ("clock") and
              ``enable_async_scheduling=True``; one latent frame decoded and
              emitted per iteration.

Prime also decodes (not just encodes): the functional decoder's first call
spends ``frames_to_trim`` of temporal memory priming itself, so an
encode-only prime would leave every rollout frame quietly drifting from the
reference. The reconstructed seed frames are never sent to the client.

The world state is the ring KV cache, an engine resource
(``get_node_resources`` below) rather than a model-owned buffer, since a
per-request ring cannot survive CUDA-graph capture.

``num_sessions`` (resident worlds) and ``max_batch_size`` (rows sharing one
forward step, set by ``step_batch_size <= num_sessions``) are separate knobs.
"""

import logging
import math
from dataclasses import replace

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    KVSpec,
    NodeResourceSpec,
    RingKVConfig,
    RingKVLayerConfig,
)
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.waypoint.config import (
    WAYPOINT_VARIANT_360P,
    WAYPOINT_VARIANT_720P,
    WAYPOINT_VARIANT_HF_REPOS,
    WaypointConfig,
    waypoint_1_5_1b_360p,
    waypoint_1_5_1b_720p,
)
from mstar.model.waypoint.ring_geometry import describe_ring_memory
from mstar.model.waypoint.submodules import (
    ATTN_RESOURCE,
    KV_RESOURCE,
    PRIME_WALK,
    ROLLOUT_LOOP_NAME,
    ROLLOUT_WALK,
    WaypointDitSubmodule,
    WaypointVaeEncoderSubmodule,
)

logger = logging.getLogger(__name__)

DIT_NODE = "dit"
VAE_ENCODER_NODE = "vae_encoder"

# The scripted action stream, one row per frame. Named once since walk
# declarations, arg validation and the edge builder must all agree with
# WaypointDitSubmodule._controller_slice on these names.
_CONTROLLER_STREAMS = ("mouse", "button", "scroll")

_VARIANT_FACTORIES = {
    WAYPOINT_VARIANT_720P: waypoint_1_5_1b_720p,
    WAYPOINT_VARIANT_360P: waypoint_1_5_1b_360p,
}


class WaypointModel(Model):
    """Waypoint-1.5-1B (720P by default; the 360P sibling shares the class)."""

    PRIME_WALK = PRIME_WALK
    ROLLOUT_WALK = ROLLOUT_WALK

    # Referenced by ``WaypointDitSubmodule.check_stop`` via
    # ``request_info.dynamic_loop_iter_counts[...]``.
    ROLLOUT_LOOP_NAME = ROLLOUT_LOOP_NAME

    def __init__(
        self,
        model_path_hf: str | None = None,
        cache_dir: str | None = None,
        variant: str = WAYPOINT_VARIANT_720P,
        skip_weight_loading: bool = False,
        checkpoint_dir: str | None = None,
        ae_path: str | None = None,
        reference_compat: bool | None = None,
        compile_dit: bool | None = None,
        cuda_graph: bool | None = None,
        capture_dit_prime: bool | None = None,
        full_global_ring: bool | None = None,
        step_batch_size: int | None = None,
        checkpoint_revision: str | None = None,
        ae_revision: str | None = None,
    ):
        if variant not in _VARIANT_FACTORIES:
            raise NotImplementedError(
                f"Waypoint variant {variant!r} is not implemented; known variants "
                f"are {sorted(_VARIANT_FACTORIES)}."
            )
        # None picks the variant's published repository; an explicit source
        # (local path or cross-variant Hub ID) is authoritative and the
        # manifest preflight rejects it if its geometry is incompatible.
        self.model_path_hf = (
            WAYPOINT_VARIANT_HF_REPOS[variant]
            if model_path_hf is None
            else model_path_hf
        )
        self.cache_dir = cache_dir
        config = _VARIANT_FACTORIES[variant]()
        overrides = {
            key: value for key, value in {
                "reference_compat": reference_compat,
                "compile_dit": compile_dit,
                "cuda_graph": cuda_graph,
                "capture_dit_prime": capture_dit_prime,
                "full_global_ring": full_global_ring,
                "step_batch_size": step_batch_size,
            }.items() if value is not None
        }
        self.config: WaypointConfig = replace(config, **overrides)
        # ``build_waypoint_dit`` never downloads: the caller resolves the local
        # directory holding model.safetensors. ``cache_dir`` is only where an
        # out-of-band snapshot lands.
        self.checkpoint_dir = checkpoint_dir or self.model_path_hf
        # TAEHV weights ship in their own repo, so ``ae_path`` overrides
        # ``config.ae_uri``, not ``checkpoint_dir``.
        self.ae_uri = ae_path or self.config.ae_uri
        self.checkpoint_revision = checkpoint_revision
        self.ae_revision = ae_revision
        # Dummy mode: get_submodule returns None for every node.
        self.skip_weight_loading = skip_weight_loading

        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._taehv: torch.nn.Module | None = None
        self._checkpoints_resolved = False

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """The ring KV cache holding the world, and the FlexAttention over it.

        Ring geometry is copied out of ``WaypointConfig`` layer by layer since
        Waypoint's layers are not alike (six global layers hold 16 frames
        spaced 8 apart, the other eighteen hold 16 consecutive frames), so
        ``ring_frames`` and ``ring_buckets`` cannot be derived from one another.

        FlexAttention rather than the paged FlashInfer default: a paged kernel
        changes the KV accumulation order, and a mask/position bug here would
        not raise, it would just drift the video. Bit-exactness against the
        reference is the only check there is.
        """
        ring_config = RingKVConfig(
            num_layers=self.config.n_layers,
            num_kv_heads=self.config.n_kv_heads,
            num_qo_heads=self.config.n_heads,
            head_dim=self.config.d_head,
            tokens_per_frame=self.config.tokens_per_frame,
            layers=tuple(
                RingKVLayerConfig(
                    ring_frames=self.config.ring_frames(i),
                    ring_buckets=self.config.ring_buckets(i),
                    pinned_dilation=self.config.pinned_dilation(i),
                )
                for i in range(self.config.n_layers)
            ),
            # Resident session count, not the step batch (that's
            # ``max_batch_size``/``step_batch_size``). Default 1; a deployment
            # raises it via ``resources: {kv: {num_sessions: N}}`` along with
            # ``max_concurrent_requests`` (see ``validate_config_yaml``).
            num_sessions=1,
        )
        # Logged since nothing downstream prints this ~816 MiB/world cost.
        # Uses the DECLARED num_sessions: `apply_yaml_overrides` may still
        # raise it after this hook returns.
        logger.info(
            "%s", describe_ring_memory(self.config, num_sessions=ring_config.num_sessions)
        )
        return [
            KVSpec(
                resource_key=KV_RESOURCE, nodes={DIT_NODE}, config=ring_config,
            ),
            AttentionSpec(
                resource_key=ATTN_RESOURCE,
                nodes={DIT_NODE},
                config=AttentionConfig(
                    kv_cache=KV_RESOURCE, backend=AttnBackend.FLEX,
                ),
            ),
        ]

    def _emit_frames(self) -> GraphEdge:
        return GraphEdge(
            next_node=EMIT_TO_CLIENT,
            name="video_output",
            output_modality="video_frame",
        )

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # prime: encode the seed clip, commit it to the world, decode it to
        # init the fused decoder's state, and discard the reconstructed seed
        # frames (outputs=[] below).
        prime = Sequential([
            GraphNode(
                name=VAE_ENCODER_NODE,
                input_names={"image_inputs"},
                outputs=[GraphEdge(next_node=DIT_NODE, name="latent")],
            ),
            GraphNode(
                name=DIT_NODE,
                input_names={"latent", *_CONTROLLER_STREAMS},
                outputs=[],
            ),
        ])

        # rollout: one frame decoded and emitted per iteration, from a single
        # node. The "clock" self loop-back carries no data (frame_pos and
        # rollout_step live in host state); it only makes the dit a same-node
        # speculation target (GraphNode.is_ready_for_speculation). Under
        # enable_async_scheduling=True the worker can build iteration N+1
        # before check_stop(N) registers the loop's finish signal; that
        # overshoot is vetoed host-side in
        # WaypointDitSubmodule.prepare_inputs before it can commit into the
        # ring, which has no undo.
        rollout = Loop(
            name=ROLLOUT_LOOP_NAME,
            section=GraphNode(
                name=DIT_NODE,
                input_names=set(_CONTROLLER_STREAMS) | {"clock"},
                outputs=[
                    GraphEdge(next_node=DIT_NODE, name="clock"),
                    self._emit_frames(),
                ],
                enable_async_scheduling=True,
            ),
            # Ceiling only; num_steps stops the loop early via
            # WaypointDitSubmodule.check_stop.
            max_iters=self.config.max_frames,
            outputs=[],
            accumulated_outputs=[],
        )

        return {PRIME_WALK: prime, ROLLOUT_WALK: rollout}

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuse to serve unless the deployment caps concurrency at the number
        of worlds the ring was sized for.

        The primary gate on the world pool: a world is claimed at ``admit``,
        and the conductor's FIFO admit queue (only active when
        ``max_concurrent_requests`` is set) is what keeps arrivals inside the
        pool. ``max_batch_size``/``step_batch_size`` caps a step's row count,
        not how many worlds may exist, so it does not substitute for this
        check.
        """
        # Same block ``EngineManager.build`` feeds to
        # ``apply_yaml_overrides``, so the gate and the allocation agree.
        overrides = (config.get("resources") or {}).get(KV_RESOURCE) or {}
        num_sessions = overrides.get("num_sessions", 1)
        if (
            not isinstance(num_sessions, int)
            or isinstance(num_sessions, bool)
            or num_sessions < 1
        ):
            raise ValueError(
                f"Waypoint requires `resources.{KV_RESOURCE}.num_sessions` in "
                f"{config_path} to be a positive int; got {num_sessions!r}."
            )
        limit = config.get("max_concurrent_requests")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(
                f"Waypoint requires `max_concurrent_requests` in {config_path} to be "
                f"a positive int; got {limit!r}. The DiT node holds "
                f"{num_sessions} live world(s) in a fixed ring buffer, and the "
                "conductor's FIFO admit queue — which exists only when this key "
                "is set — is what keeps arrivals inside that pool. max_batch_size "
                "alone does not: it caps a step, not the number of requests on "
                "the node."
            )
        if limit > num_sessions:
            raise ValueError(
                f"`max_concurrent_requests: {limit}` in {config_path} exceeds the "
                f"{num_sessions} world(s) the ring is sized for. Every request past "
                "the pool fails terminally at admit — there is nothing to evict. "
                f"Set `resources.{KV_RESOURCE}.num_sessions` to {limit} to match, at "
                "the cost of ~816 MiB of ring per world at 720P."
            )
        if limit < num_sessions:
            logger.warning(
                "Waypoint ring is sized for %d worlds but max_concurrent_requests "
                "is %d: %d world(s) of ring (~816 MiB each at 720P) are allocated "
                "and can never be filled.",
                num_sessions, limit, num_sessions - limit,
            )
        if self.config.step_batch_size > num_sessions:
            raise ValueError(
                f"`step_batch_size: {self.config.step_batch_size}` exceeds "
                f"`resources.{KV_RESOURCE}.num_sessions: {num_sessions}` in "
                f"{config_path}. A step cannot batch more rows than there are "
                "resident worlds to supply them."
            )

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Materialize the request's scripted controller stream (and its seed
        clip, if any) as the edges the walk's first nodes consume.

        ``prompt`` is ignored: this checkpoint has ``prompt_conditioning=None``
        and carries no cross-attention.

        ``actions`` is a list of per-step dicts, one per generated latent:
        ``{"mouse": [dx, dy], "buttons": [id, ...], "scroll": s}``; buttons are
        one-hot scattered into ``n_buttons`` columns and an omitted field is
        that control's neutral value. Prime never consumes action zero.
        """
        del prompt, input_modalities
        if output_modalities != ["video_frame"]:
            raise ValueError(
                "Waypoint requires exactly one output modality, 'video_frame'; "
                f"got {output_modalities!r}. Raw RGB frames are served only by "
                "the native streaming endpoint, not as encoded video."
            )
        num_steps = self._resolve_num_steps(kwargs)
        actions = kwargs.get("actions")
        if not isinstance(actions, list) or len(actions) != num_steps:
            actual = len(actions) if isinstance(actions, list) else 0
            raise ValueError(
                "Waypoint requires exactly one action object per generated latent "
                f"step; got {actual} actions for "
                f"num_steps={num_steps}."
            )

        mouse = torch.zeros((1, num_steps, 2), dtype=torch.float32)
        button = torch.zeros((1, num_steps, self.config.n_buttons), dtype=torch.float32)
        scroll = torch.zeros((1, num_steps, 1), dtype=torch.float32)
        float32_max = torch.finfo(torch.float32).max
        for step, action in enumerate(actions):
            if not isinstance(action, dict):
                raise ValueError(f"action {step} must be an object; got {type(action).__name__}")
            unknown = set(action) - {"mouse", "buttons", "scroll"}
            if unknown:
                raise ValueError(f"action {step} has unknown field(s) {sorted(unknown)}")
            motion = action.get("mouse", (0.0, 0.0))
            if not isinstance(motion, (list, tuple)) or len(motion) != 2:
                raise ValueError(f"action {step} mouse must contain exactly [dx, dy]")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in motion
            ):
                raise ValueError(f"action {step} mouse values must be numbers")
            try:
                dx, dy = map(float, motion)
            except OverflowError:
                raise ValueError(
                    f"action {step} mouse values must be finite and representable "
                    "as float32"
                ) from None
            if (
                not math.isfinite(dx)
                or not math.isfinite(dy)
                or abs(dx) > float32_max
                or abs(dy) > float32_max
            ):
                raise ValueError(
                    f"action {step} mouse values must be finite and representable "
                    "as float32"
                )
            mouse[0, step, 0] = dx
            mouse[0, step, 1] = dy
            pressed_ids = action.get("buttons", ())
            if not isinstance(pressed_ids, (list, tuple)):
                raise ValueError(f"action {step} buttons must be a list of ids")
            seen: set[int] = set()
            for pressed in pressed_ids:
                if isinstance(pressed, bool) or not isinstance(pressed, int):
                    raise ValueError(f"action {step} button ids must be integers")
                if not 0 <= pressed < self.config.n_buttons:
                    raise ValueError(
                        f"button id {pressed} out of range for n_buttons="
                        f"{self.config.n_buttons} (action {step})."
                    )
                if pressed in seen:
                    raise ValueError(f"action {step} repeats button id {pressed}")
                seen.add(pressed)
                button[0, step, pressed] = 1.0
            raw_scroll = action.get("scroll", 0.0)
            if isinstance(raw_scroll, bool) or not isinstance(raw_scroll, (int, float)):
                raise ValueError(f"action {step} scroll must be a number")
            try:
                scroll_value = float(raw_scroll)
            except OverflowError:
                raise ValueError(
                    f"action {step} scroll must be finite and representable as float32"
                ) from None
            if not math.isfinite(scroll_value) or abs(scroll_value) > float32_max:
                raise ValueError(
                    f"action {step} scroll must be finite and representable as float32"
                )
            scroll[0, step, 0] = scroll_value

        out: NameToTensorList = {"mouse": [mouse], "button": [button], "scroll": [scroll]}

        if not tensors or not tensors.get("image_inputs"):
            raise ValueError("Waypoint requires one RGB seed image or four-frame seed clip.")
        out["image_inputs"] = [self._seed_clip(tensors["image_inputs"][0])]
        return out

    def _seed_clip(self, image: torch.Tensor) -> torch.Tensor:
        """The prime walk's ``[temporal_compression, H, W, 3]`` uint8 clip.

        A single frame is repeated to fill it (seeding from a still). Fewer or
        more frames is refused since the streaming encoder emits one latent
        per ``t_downscale``. Only the aspect ratio is checked, not resolution.
        """
        frames = image if image.dim() == 4 else image.unsqueeze(0)
        if frames.dtype != torch.uint8 or frames.shape[-1] != 3:
            raise ValueError(
                "seed frames must be uint8 RGB shaped [H, W, 3] or [T, H, W, 3]; "
                f"got {tuple(image.shape)} of {image.dtype}."
            )
        n = self.config.temporal_compression
        if frames.shape[0] == 1:
            frames = frames.expand(n, -1, -1, -1)
        if frames.shape[0] != n:
            raise ValueError(
                f"the seed clip must be 1 or {n} frames (one latent frame); got "
                f"{frames.shape[0]}."
            )
        height, width = int(frames.shape[1]), int(frames.shape[2])
        if self.config.auto_aspect_ratio and height * 16 != width * 9:
            raise ValueError(f"seed frames must be 16:9; got {height}x{width}.")
        return frames.contiguous()

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """The seed frame as uint8 ``[H, W, 3]``, not the base loader's float
        ``[C, H, W]``: the AE scales it itself, in its own dtype, and a float
        round trip through the request would round twice on the way there."""
        import torchvision

        img = torchvision.io.decode_image(filepath).to(device)  # uint8 [C, H, W]
        return TensorAndMetadata(img.permute(1, 2, 0).contiguous())

    def postprocess(
        self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None,
    ) -> bytes:
        """One step's frames as raw uint8 RGB bytes,
        ``[temporal_compression, H, W, 3]`` in C order. No container: the emit
        is per engine step."""
        del request_kwargs
        if modality != "video_frame":
            raise ValueError(f"Unsupported modality for Waypoint: {modality!r}")
        if output.dtype != torch.uint8:
            raise ValueError(
                f"the dit's fused decode emits uint8 RGB frames; got {output.dtype}."
            )
        return output.detach().cpu().contiguous().numpy().tobytes()

    def get_output_frame_rate(
        self,
        modality: str = "video_frame",
        request_kwargs: dict | None = None,
    ) -> float:
        del request_kwargs
        if modality != "video_frame":
            raise ValueError(f"Unsupported frame modality for Waypoint: {modality!r}")
        return float(self.config.inference_fps)

    # ------------------------------------------------------------------
    # Model ABC: forward pass orchestration
    # ------------------------------------------------------------------

    def _resolve_num_steps(self, model_kwargs: dict | None) -> int:
        """Validate the generated-latent count against the trained horizon."""
        model_kwargs = model_kwargs or {}
        requested = model_kwargs.get("num_steps")
        if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
            raise ValueError(f"Waypoint requires num_steps > 0; got {requested!r}.")
        if requested > self.config.max_frames:
            raise ValueError(
                f"num_steps={requested} exceeds the checkpoint horizon "
                f"({self.config.max_frames})."
            )
        return requested

    def _get_step_metadata(self, metadata: CurrentForwardConductorMetadata) -> dict:
        """Per-pass metadata the submodule reads off ``request_info``.

        ``num_steps`` is what ``check_stop`` counts the rollout loop against;
        nothing else in the shell is per-request.
        """
        return {
            "is_prefill": metadata.is_prefill,
            "num_steps": metadata.kwargs["num_steps"],
        }

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        del partition_name, input_modalities
        if output_modalities != ["video_frame"]:
            raise ValueError(
                "Waypoint requires exactly one output modality, 'video_frame'; "
                f"got {output_modalities!r}."
            )
        # Backstop: process_prompt already rejects a malformed request (400).
        # A raise here runs at the conductor, whose main loop swallows it, so
        # the client would hang instead.
        for name in _CONTROLLER_STREAMS:
            if not input_signals.get(name):
                raise ValueError(
                    f"Waypoint needs the {name!r} controller stream; "
                    "process_prompt emits all three."
                )

        if not input_signals.get("image_inputs"):
            raise ValueError("Waypoint cannot start without its required seed clip.")
        schedule = [PRIME_WALK, ROLLOUT_WALK]

        kwargs = {
            "walk_schedule": schedule,
            "walk_step": 0,
            "num_steps": self._resolve_num_steps(model_kwargs),
        }
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=["tensor"],
            output_modalities=output_modalities,
            graph_walk=schedule[0],
            is_prefill=schedule[0] == PRIME_WALK,
            kwargs=kwargs,
        )
        inputs = self._walk_inputs(schedule[0], input_signals)
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            # Nothing is released after the first pass: the controller streams
            # are re-read by every frame of the rollout, and the seed clip is
            # dropped with the request. Unpersisting either here would strand
            # the rollout walk with no inputs and the node would never become
            # ready.
            unpersist_tensors=[],
            step_metadata=self._get_step_metadata(full_metadata),
        )

    def _walk_inputs(
        self, walk: str, signals: dict[str, list[TensorPointerInfo]],
    ) -> list[GraphEdge]:
        """The external edges seeding one walk. Both walks read the controller
        streams at the dit; only ``prime`` reads the seed clip, and it reads it
        at the vae_encoder; only ``rollout`` seeds the dit's self loop-back."""
        inputs = [
            GraphEdge(next_node=DIT_NODE, name=name, persist=True)
            for name in _CONTROLLER_STREAMS
        ]
        if walk == PRIME_WALK:
            inputs.insert(
                0,
                GraphEdge(
                    next_node=VAE_ENCODER_NODE, name="image_inputs", persist=True
                ),
            )
        else:
            # Sent empty, like wan22's denoise loop-back edges: the dit
            # submodule never reads its value, only its name (F4 pattern).
            inputs.append(GraphEdge(next_node=DIT_NODE, name="clock"))
        for edge in inputs:
            edge.tensor_info = signals.get(edge.name, [])
        return inputs

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Step through the request's walk schedule; done after the rollout."""
        del partition_name, incoming_connections
        metadata = partition_metadata
        schedule = metadata.kwargs["walk_schedule"]
        step = metadata.kwargs["walk_step"] + 1
        if step >= len(schedule):
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                step_metadata=self._get_step_metadata(metadata),
                request_done=True,
            )

        metadata.kwargs["walk_step"] = step
        walk = schedule[step]
        metadata.graph_walk = walk
        metadata.is_prefill = walk == PRIME_WALK
        inputs = self._walk_inputs(walk, persist_signals)
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            # The rollout is the last walk, so its inputs are consumed for the
            # last time here.
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
            step_metadata=self._get_step_metadata(metadata),
        )

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def get_autocast_dtype(self):
        """Allocate BF16 resources while every node disables autocast.

        The dtype layout is settled at build time (``cast_serving_dtypes()``
        pins fp32 islands after casting the rest to bf16); this is also the
        ring-KV allocation dtype, and returning None would silently allocate
        the ring in fp32.
        """
        return torch.bfloat16

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None, sp_group=None,
    ) -> torch.nn.Module | None:
        # autocast_dtype/tp_group/sp_group exist for interface parity only:
        # weights load in the checkpoint's own dtypes and nothing shards yet.
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("Loaded Waypoint submodule for node %s", node_name)
        return submodule

    def _create_submodule(
        self, node_name: str, device: str = "cpu",
    ) -> NodeSubmodule | None:
        """Construct one node's submodule. None in dummy mode and for unknown
        nodes, which makes the engine run that node without real computation."""
        if self.skip_weight_loading:
            return None
        if node_name in {DIT_NODE, VAE_ENCODER_NODE}:
            self._resolve_checkpoints()
        if node_name == DIT_NODE:
            from mstar.model.waypoint.weight_loader import build_waypoint_dit

            dit = build_waypoint_dit(
                self.config, self.checkpoint_dir, device=device,
            )
            return WaypointDitSubmodule(dit, self._taehv_weights(device), self.config)
        if node_name == VAE_ENCODER_NODE:
            return WaypointVaeEncoderSubmodule(self._taehv_weights(device), self.config)
        logger.warning("Waypoint has no submodule for node %r; running it dummy.", node_name)
        return None

    def _resolve_checkpoints(self) -> None:
        """Resolve both artifacts and validate the DiT manifest before allocation.

        Triggered once by the first requested node. Resolved together so
        startup fails on a missing AE before the multi-gigabyte DiT has been
        allocated, even if the DiT node is requested first.
        """
        if self._checkpoints_resolved:
            return
        if not self.checkpoint_dir:
            raise ValueError(
                "Waypoint requires `checkpoint_dir` or `model_path_hf` when "
                "weight loading is enabled."
            )
        from mstar.model.waypoint.checkpoint import (
            require_taehv_runtime,
            resolve_taehv_checkpoint,
            resolve_waypoint_checkpoint,
        )

        # Must precede the DiT resolver: a missing TAEHV install shouldn't
        # trigger a multi-GiB download first.
        require_taehv_runtime()
        self.checkpoint_dir = str(resolve_waypoint_checkpoint(
            self.checkpoint_dir,
            self.config,
            cache_dir=self.cache_dir,
            revision=self.checkpoint_revision,
        ))
        self.ae_uri = str(resolve_taehv_checkpoint(
            self.ae_uri,
            cache_dir=self.cache_dir,
            revision=self.ae_revision,
        ))
        self._checkpoints_resolved = True

    def _taehv_weights(self, device: str) -> torch.nn.Module:
        """The AE weights, built once and shared by both VAE nodes: they differ
        only in per-request streaming state, not held here."""
        if self._taehv is None:
            from mstar.model.waypoint.components.taehv import load_taehv

            self._taehv = load_taehv(self.ae_uri, cache_dir=self.cache_dir).to(
                device=device, dtype=torch.bfloat16
            )
        return self._taehv
