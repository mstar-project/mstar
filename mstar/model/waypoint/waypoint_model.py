"""WaypointModel: Waypoint-1.5-1B interactive video world model.

Architecture (three nodes):
    vae_encoder - TAEHV encode. The seed clip (``temporal_compression`` raw
                  frames) into the one latent frame it stands for.
    dit         - the 1.28B world DiT. One engine step is one latent frame:
                  four frozen Euler denoise passes plus one committing cache
                  pass, all inside a single ``forward``.
    vae_decoder - TAEHV decode. One latent frame back into its raw frames.

Graph walks (2):
    prime   - vae_encoder -> dit.append_frame -> vae_decoder. Seeds the world
              and decoder state from a real frame, advances the ring clock by
              one, and emits nothing.
    rollout - Loop("rollout_loop") over dit.generate_frame -> vae_decoder ->
              client; one latent frame per iteration, emitted as it lands.

**Prime decodes as well as encodes**, and not for symmetry: the functional
decoder's first call spends ``frames_to_trim`` of temporal memory priming
itself, so a prime that only encoded would leave the first *rollout* frame
paying for it and every frame after that shifted against the world it came
from. Silently — drifting video, no exception. The reconstructed seed frames
are internal initialization output and are never sent to the client.

The world state is the ring KV cache. It is an engine resource
(``get_node_resources`` below), not a model-owned buffer: a per-request ring in
``PerRequestState`` cannot survive CUDA-graph capture, and the resource
lifecycle is the only place that can hand a request one of the node's worlds —
or refuse it when they are all taken. That refusal is a backstop, though — see
``get_worker_graphs``.

``num_worlds`` and ``max_batch_size`` are separate numbers and stay separate.
``num_worlds`` is how many sessions are *resident* (one ring span each, folded
into the token dimension by ``LayerRingCache``); ``max_batch_size`` is how many
share one *forward step*, and is still 1, so resident worlds take turns across
steps rather than batching. Raising the second is the next cut and does not
change the layout chosen for the first.
"""

import logging
import math
from dataclasses import replace

import torch
import yaml

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
from mstar.model.base import ForwardPassArgs, Model, TensorAndMetadata, WorkerGraph
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
    WaypointVaeDecoderSubmodule,
    WaypointVaeEncoderSubmodule,
)

logger = logging.getLogger(__name__)

DIT_NODE = "dit"
VAE_ENCODER_NODE = "vae_encoder"
VAE_DECODER_NODE = "vae_decoder"

# The scripted action stream, one row per frame. Named once because the walk
# declarations, the initial-args validation and the per-walk edge builder all
# have to agree with WaypointDitSubmodule._controller_slice, which reads these
# names off the request's inputs dict.
_CONTROLLER_STREAMS = ("mouse", "button", "scroll")

_VARIANT_FACTORIES = {
    WAYPOINT_VARIANT_720P: waypoint_1_5_1b_720p,
    WAYPOINT_VARIANT_360P: waypoint_1_5_1b_360p,
}


class WaypointModel(Model):
    """Waypoint-1.5-1B (720P by default; the 360P sibling shares the class)."""

    PRIME_WALK = PRIME_WALK
    ROLLOUT_WALK = ROLLOUT_WALK

    # Loop name — referenced by ``WaypointDitSubmodule.check_stop`` through
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
        full_global_ring: bool | None = None,
        checkpoint_revision: str | None = None,
        ae_revision: str | None = None,
    ):
        if variant not in _VARIANT_FACTORIES:
            raise NotImplementedError(
                f"Waypoint variant {variant!r} is not implemented; known variants "
                f"are {sorted(_VARIANT_FACTORIES)}."
            )
        # The generic registry passes None so the selected variant chooses its
        # published repository. Any explicit source remains authoritative,
        # including a local path or a deliberately cross-variant Hub ID; the
        # manifest preflight will reject it if its geometry is incompatible.
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
                "full_global_ring": full_global_ring,
            }.items() if value is not None
        }
        self.config: WaypointConfig = replace(config, **overrides)
        # ``build_waypoint_dit`` never downloads: the caller resolves the local
        # directory holding model.safetensors. ``cache_dir`` is where a
        # snapshot lands if one is fetched out of band; the two are not the same
        # thing and conflating them is how a half-downloaded repo gets loaded.
        self.checkpoint_dir = checkpoint_dir or self.model_path_hf
        # The TAEHV weights ship in their own repo, so ``ae_path`` is a local
        # override of ``config.ae_uri`` and not of ``checkpoint_dir``.
        self.ae_uri = ae_path or self.config.ae_uri
        self.checkpoint_revision = checkpoint_revision
        self.ae_revision = ae_revision
        # Dummy mode: get_submodule returns None for every node, so engines and
        # tests run without weights, GPU or network.
        self.skip_weight_loading = skip_weight_loading

        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._taehv: torch.nn.Module | None = None
        self._checkpoints_resolved = False

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """The ring KV cache holding the world, and the FlexAttention over it.

        The ring geometry is copied out of ``WaypointConfig`` layer by layer
        rather than summarized: Waypoint's layers are not alike (the six global
        layers hold 16 frames spaced 8 apart, the other eighteen hold 16
        consecutive frames), and ``ring_frames`` and ``ring_buckets`` are two
        separate questions. They happen to agree on every layer of the
        *compacted* 720P ring, which is exactly what makes deriving one from the
        other look safe — flip ``full_global_ring`` back to the reference's
        sizing and a global layer is 128 frames indexed by 16 buckets.

        FlexAttention rather than the paged FlashInfer default: a paged
        kernel changes the accumulation order over the KV blocks, and a
        mask-or-position bug in this model does not raise, it produces
        plausible, smoothly drifting video. Bit-exactness against the reference
        is the only check there is, so the kernel has to be the reference's.

        The attention spec names the cache by key, so ``depends_on`` orders the
        two: the ring is built first and the attention resource resolves it.
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
            # How many sessions this node holds resident. One by default
            # because a world is ~816 MiB of ring at 720P and a model has no
            # business assuming the box; a deployment raises it under
            # ``resources: {kv: {num_worlds: N}}`` and raises
            # ``max_concurrent_requests`` with it (see ``get_worker_graphs``).
            # Not the step batch — that is ``max_batch_size``, still 1.
            num_worlds=1,
        )
        # Logged, not merely allocated: this declaration is worth ~816 MiB per
        # world and nothing downstream prints it. The report carries the
        # counterfactual under the other ``full_global_ring`` setting, which is
        # the number you want *before* the engine commits to one of them.
        #
        # `ring_config.num_worlds` is the DECLARED count, which is what this
        # line can honestly report: `EngineManager.build` calls
        # `apply_yaml_overrides` on the specs after this hook returns, so a
        # deployment's `num_worlds` has not landed yet. The reported total
        # scales linearly with it -- the world dim is folded into the token
        # axis -- so N worlds is N times the number below.
        logger.info(
            "%s", describe_ring_memory(self.config, num_worlds=ring_config.num_worlds)
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
        # -- prime: encode the seed clip, commit it to the world, initialize the
        # -- decoder state, and discard the reconstructed seed frames.
        # --
        # -- Both `latent` edges carry that name because both endpoints call it
        # -- that; a section keys edges on (name, next_node), so they are two
        # -- edges and not one.
        prime = Sequential([
            GraphNode(
                name=VAE_ENCODER_NODE,
                input_names={"image_inputs"},
                outputs=[GraphEdge(next_node=DIT_NODE, name="latent")],
            ),
            GraphNode(
                name=DIT_NODE,
                input_names={"latent", *_CONTROLLER_STREAMS},
                outputs=[GraphEdge(next_node=VAE_DECODER_NODE, name="latent")],
            ),
            # Advance the decoder's nine temporal histories, but do not expose
            # reconstructed seed frames. Client frame zero is generated.
            GraphNode(
                name=VAE_DECODER_NODE,
                input_names={"latent"},
                outputs=[],
            ),
        ])

        # -- rollout: one frame per iteration, emitted as it lands.
        # --
        # -- No loop-back edges, and that is not an omission: everything that
        # -- crosses a frame boundary is the ring (an engine resource at a fixed
        # -- address), the host-side frame_pos, or the decoder's streaming
        # -- state. The controller streams are loop-external, re-injected every
        # -- iteration, and the submodule slices the current frame's row out.
        # --
        # -- EMIT_TO_CLIENT sits on the decoder node, one message per iteration,
        # -- rather than on Loop.accumulated_outputs: a client that only sees
        # -- frames after the rollout ends has no world to interact with.
        # --
        # -- check_stop ends the loop, but the loop's registry calls
        # -- complete_iter only once *every* entity in the section is done, so
        # -- the final frame is decoded before the loop closes.
        rollout = Loop(
            name=ROLLOUT_LOOP_NAME,
            section=Sequential([
                GraphNode(
                    name=DIT_NODE,
                    input_names=set(_CONTROLLER_STREAMS),
                    outputs=[GraphEdge(next_node=VAE_DECODER_NODE, name="latent")],
                    # Speculation would dispatch iteration N+1 before
                    # check_stop's decision on N landed. The overshoot forward
                    # *commits a frame into the ring* and there is no undo; the
                    # next real rollout would inherit it.
                    enable_async_scheduling=False,
                ),
                GraphNode(
                    name=VAE_DECODER_NODE,
                    input_names={"latent"},
                    outputs=[self._emit_frames()],
                    # And here for the decoder's own reason: it is streaming, so
                    # frames must be decoded exactly once in emission order, and
                    # a speculative decode of a frame that may not stand is a
                    # reorder of a stream that cannot be reordered.
                    enable_async_scheduling=False,
                ),
            ]),
            # Ceiling only; the request's num_steps stops the loop early via
            # WaypointDitSubmodule.check_stop.
            max_iters=self.config.max_frames,
            outputs=[],
            accumulated_outputs=[],
        )

        return {PRIME_WALK: prime, ROLLOUT_WALK: rollout}

    def get_worker_graphs(self, config_path: str) -> list[WorkerGraph]:
        """Refuse to build unless the deployment caps concurrency at the number
        of worlds the ring was sized for.

        This is the **primary** gate on the world pool, not a nicety. A world is
        claimed at ``admit``, i.e. once a batch has already been formed — by
        then the only thing ``RingKVManager.admit`` can do about a request the
        pool cannot hold is fail it terminally. What actually keeps arrivals
        inside the pool is the conductor's FIFO admit queue, and that queue only
        exists when ``max_concurrent_requests`` is set: the conductor drains
        ``waiting_queue`` while ``len(self.requests) < max_concurrent_requests``,
        so an unset value admits everything on arrival and every request past
        the Nth dies at admit. Unset therefore stays fatal, exactly as before —
        what changed is that the accepted value is a range rather than the
        single number 1.

        ``max_batch_size = 1`` does NOT cover this, and that is still true with
        N worlds. It caps how many requests share one *step*; N admitted
        rollouts alternate steps, which is now the intended shape — each holds
        its own world and the BlockMask keeps them apart — but it says nothing
        about how many may exist, which is the thing the pool bounds.

        A limit *below* ``num_worlds`` is legal and only wasteful: it allocates
        rings (~816 MiB each at 720P) for worlds no request can ever reach, so
        it is warned about rather than refused.

        Checked here because this hook is the only place a model sees the key:
        the Conductor reads it out of the YAML itself and
        ``api_server/entrypoint.py`` forwards only ``model_kwargs`` to
        ``Model.__init__``.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        # The same block ``EngineManager.build`` feeds to
        # ``apply_yaml_overrides``, read here for the same key, so the gate and
        # the allocation cannot disagree about how many worlds exist.
        overrides = (config.get("resources") or {}).get(KV_RESOURCE) or {}
        num_worlds = overrides.get("num_worlds", 1)
        if (
            not isinstance(num_worlds, int)
            or isinstance(num_worlds, bool)
            or num_worlds < 1
        ):
            raise ValueError(
                f"Waypoint requires `resources.{KV_RESOURCE}.num_worlds` in "
                f"{config_path} to be a positive int; got {num_worlds!r}."
            )
        limit = config.get("max_concurrent_requests")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(
                f"Waypoint requires `max_concurrent_requests` in {config_path} to be "
                f"a positive int; got {limit!r}. The DiT node holds "
                f"{num_worlds} live world(s) in a fixed ring buffer, and the "
                "conductor's FIFO admit queue — which exists only when this key "
                "is set — is what keeps arrivals inside that pool. max_batch_size "
                "alone does not: it caps a step, not the number of requests on "
                "the node."
            )
        if limit > num_worlds:
            raise ValueError(
                f"`max_concurrent_requests: {limit}` in {config_path} exceeds the "
                f"{num_worlds} world(s) the ring is sized for. Every request past "
                "the pool fails terminally at admit — there is nothing to evict. "
                f"Set `resources.{KV_RESOURCE}.num_worlds` to {limit} to match, at "
                "the cost of ~816 MiB of ring per world at 720P."
            )
        if limit < num_worlds:
            logger.warning(
                "Waypoint ring is sized for %d worlds but max_concurrent_requests "
                "is %d: %d world(s) of ring (~816 MiB each at 720P) are allocated "
                "and can never be filled.",
                num_worlds, limit, num_worlds - limit,
            )
        return super().get_worker_graphs(config_path)

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
        and carries no cross-attention, so a text prompt would have nowhere to
        go. Raising on one would break clients that send an empty default.

        ``actions`` is a list of per-step dicts, exactly one per generated latent:
        ``{"mouse": [dx, dy], "buttons": [id, ...], "scroll": s}``. The button
        field is a set of pressed ids that gets one-hot scattered into
        ``n_buttons`` columns, matching the reference's ``CtrlInput``; an
        omitted field is that control's neutral value. Prime uses a separate
        internal idle action and never consumes action zero.
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

        One frame is repeated to fill it, the reference's way of seeding from a
        still (``gen_sample.py``'s ``seed_frame_x4``). Fewer or more frames is
        refused: the streaming encoder emits one latent per ``t_downscale``, so a
        short clip buffers silently and a long one encodes twice.

        Resolution is not checked, only the aspect ratio -- the AE resizes 16:9
        input onto its own grid and decodes back to the variant's resolution.
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
        ``[temporal_compression, H, W, 3]`` in C order. No container: the emit is
        per engine step, and a per-step mp4 is a fragment nothing plays."""
        del request_kwargs
        if modality != "video_frame":
            raise ValueError(f"Unsupported modality for Waypoint: {modality!r}")
        if output.dtype != torch.uint8:
            raise ValueError(
                f"the vae_decoder emits uint8 RGB frames; got {output.dtype}."
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
        # A backstop, not the primary guard: process_prompt already rejected a
        # malformed request on the data worker, where a ValueError becomes a
        # 400. A raise here runs at the conductor, whose main loop swallows it,
        # so the client would hang instead.
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
        at the vae_encoder."""
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

        The dtype layout is settled at build time — ``cast_serving_dtypes()``
        takes the meta module to bf16 and pins the fp32 islands back, and the AE
        is built bf16 whole. The submodules' ``disable_autocast`` flags preserve
        that mixed layout. Returning BF16 here is also the explicit ring-KV
        allocation dtype; returning None silently allocated the ring in fp32.
        """
        return torch.bfloat16

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None, sp_group=None,
    ) -> torch.nn.Module | None:
        # ``autocast_dtype``/``tp_group``/``sp_group`` exist for interface
        # parity: weights load in the checkpoint's own dtypes and neither the
        # ring nor the BlockMask shards yet.
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
        if node_name in {DIT_NODE, VAE_ENCODER_NODE, VAE_DECODER_NODE}:
            self._resolve_checkpoints()
        if node_name == DIT_NODE:
            from mstar.model.waypoint.weight_loader import build_waypoint_dit

            dit = build_waypoint_dit(
                self.config, self.checkpoint_dir, device=device,
            )
            return WaypointDitSubmodule(dit, self.config)
        if node_name == VAE_ENCODER_NODE:
            return WaypointVaeEncoderSubmodule(self._taehv_weights(device), self.config)
        if node_name == VAE_DECODER_NODE:
            return WaypointVaeDecoderSubmodule(self._taehv_weights(device), self.config)
        logger.warning("Waypoint has no submodule for node %r; running it dummy.", node_name)
        return None

    def _resolve_checkpoints(self) -> None:
        """Resolve both artifacts and validate the DiT manifest before allocation.

        The first requested node triggers this once. Resolving both together is
        intentional: startup must fail on a missing AE before a multi-gigabyte
        DiT has been allocated, even when the engine happens to ask for the DiT
        node first. Tensor completeness and the pinned TAEHV runtime architecture
        are then validated while loading, before request admission.
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

        # Dependency validation is part of the same preflight as both weight
        # sources. In particular it must precede the DiT resolver: a missing or
        # empty TAEHV install should not trigger a multi-GiB download first.
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
        only in streaming state, which is per request and not held here. bf16 at
        build is the reference's serving dtype, and with ``disable_autocast`` on
        both nodes it is the dtype the convs actually run in."""
        if self._taehv is None:
            from mstar.model.waypoint.components.taehv import load_taehv

            self._taehv = load_taehv(self.ae_uri, cache_dir=self.cache_dir).to(
                device=device, dtype=torch.bfloat16
            )
        return self._taehv
