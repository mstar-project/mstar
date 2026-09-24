import logging

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig
from mstar.engine.resources import AttentionStep, RingKVStep, SubmoduleStep
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.model.waypoint.components.dit import WaypointDiT
from mstar.model.waypoint.components.taehv import (
    DECODER_HISTORY_PREFIX,
    decode_latent,
    encode_seed_clip,
    encoded_size_for_latent,
    initial_decoder_histories,
    pixel_size_for_latent,
    validate_taehv_architecture,
)
from mstar.model.waypoint.config import WaypointConfig

logger = logging.getLogger(__name__)

PRIME_WALK = "prime"
ROLLOUT_WALK = "rollout"
ROLLOUT_LOOP_NAME = "rollout_loop"

# Resource labels this node declares.
KV_RESOURCE = "kv"
ATTN_RESOURCE = "attn"

# splitmix64 constants, used to derive a per-frame seed. See _frame_seed.
_U64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_MIX1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MIX2 = 0x94D049BB133111EB


def _frame_seed(request_seed: int, frame_pos: int) -> int:
    """A reproducible seed for one ``(request, frame)`` pair.

    Stateless by construction: a pure function of the request's seed and the
    frame position, so a resumed or re-run frame draws the identical tensor.
    A splitmix64 finalizer rather than ``seed + frame_pos`` avoids nearby
    seeds sharing correlated noise.
    """
    z = (request_seed + (frame_pos + 1) * _SPLITMIX_GAMMA) & _U64
    z = ((z ^ (z >> 30)) * _SPLITMIX_MIX1) & _U64
    z = ((z ^ (z >> 27)) * _SPLITMIX_MIX2) & _U64
    z ^= z >> 31
    # manual_seed takes a signed 64-bit; keep it non-negative.
    return z & (_U64 >> 1)


def _rollout_capture_batch_sizes(step_batch_size: int) -> list[int]:
    """Powers of two up to ``step_batch_size``, with ``step_batch_size`` itself.

    A geometric bucket set makes capture startup grow with ``log B`` instead
    of linearly in B; a step of ``n`` rows replays the smallest bucket
    ``>= n`` and pads the tail with dummy rows (see ``RingKVManager.plan``).
    """
    sizes = []
    bs = 1
    while bs < step_batch_size:
        sizes.append(bs)
        bs *= 2
    sizes.append(step_batch_size)
    return sizes


class _SingleRequestMixin:
    """Serve one request per step, through the engine's batched entry point.

    The v1 engine always dispatches to ``forward_batched``, so a submodule
    that only defines ``forward`` never runs. Copied from the wan22 idiom
    rather than imported, since the two models share no other code.
    """

    def max_batch_size(self, graph_walk: str):
        return 1

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        request_ids = engine_inputs.request_ids
        assert len(request_ids) == 1, (
            f"{type(self).__name__} does not batch a step; got "
            f"{len(request_ids)} requests in one step (max_batch_size should "
            "have capped it at 1)"
        )
        return {
            request_ids[0]: self.forward(
                graph_walk, engine_inputs=engine_inputs, **kwargs
            )
        }


class _FunctionalAeMixin:
    """Shared fixed-shape facts for the captured functional AE paths."""

    @property
    def ae_dtype(self) -> torch.dtype:
        """The dtype the weights are in, read live: an input scaled into a dtype
        the convs are not in faults on the first layer."""
        return next(self.taehv.parameters()).dtype

    @property
    def encoded_size(self) -> tuple[int, int]:
        return encoded_size_for_latent(
            self.config.latent_height, self.config.latent_width
        )

    @property
    def pixel_size(self) -> tuple[int, int]:
        return pixel_size_for_latent(
            self.config.latent_height, self.config.latent_width
        )


class WaypointDitSubmodule(_FunctionalAeMixin, NodeSubmodule):
    """The world DiT: one latent frame per engine step, TAEHV-decoded in the
    same forward.

    The decode is fused in rather than left on its own node so a same-worker
    speculative N+1 (``GraphNode.enable_async_scheduling``) can start while
    N's frame is still going out. See ``WaypointModel``'s module docstring
    for the resulting two-node graph.
    """

    # Do not let the engine independently compile this wrapper and fuse
    # across the DiT's own full-graph regions.
    disable_torch_compile = True

    # Waypoint pins an explicit fp32 island list at build time; this flag
    # prevents EngineManager from blanket-casting those islands to bf16.
    disable_autocast = True

    def __init__(self, dit: WaypointDiT, taehv: torch.nn.Module, config: WaypointConfig):
        super().__init__()
        validate_taehv_architecture(taehv)
        self.dit = dit
        self.taehv = taehv
        self.config = config
        # Matches the engine's own capture options (``compile=True`` ->
        # ``torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=False,
        # dynamic=False)``) so Inductor picks the same kernels bit-for-bit.
        # Wrapping just the decode call keeps the engine from compiling across
        # the denoise/decode boundary.
        self._decode_latent = (
            torch.compile(
                decode_latent, mode="max-autotune-no-cudagraphs",
                fullgraph=False, dynamic=False,
            )
            if config.cuda_graph else decode_latent
        )

    def bind_node_resources(self, resources: dict) -> None:
        """Require both resources before letting the bind reach the layers."""
        missing = {KV_RESOURCE, ATTN_RESOURCE} - set(resources)
        if missing:
            raise KeyError(
                f"WaypointDitSubmodule was bound without {sorted(missing)}; the "
                "dit node declares both in WaypointModel.get_node_resources() "
                "and all 24 attention layers call them."
            )
        super().bind_node_resources(resources)

    def max_batch_size(self, graph_walk: str) -> int:
        """Both walks carry up to ``step_batch_size`` rows: one per resident
        world sharing the forward."""
        return self.config.step_batch_size

    def can_batch(self, batch, model_inputs) -> bool:
        """Rows are independent (``preprocess`` concatenates on the batch dim
        and every resource is scoped per row), so any admitted set is
        batchable."""
        del batch, model_inputs
        return True

    # ------------------------------------------------------------------
    # prepare_inputs / preprocess
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs | None:
        """This frame's row: the ring clock, its controller slice, and either
        the noise to denoise from (rollout) or the latent to prime with. Also
        the nine decoder histories the fused decode reads and writes.

        Runs on the host, outside any captured region, which is why the noise
        is drawn here rather than in ``forward``: a captured region cannot
        call the RNG.

        ``inputs.get("clock")`` is never read; it only has to be present in
        ``input_names`` for ``GraphNode.is_ready_for_speculation`` to propose
        the next iteration as a same-node speculation target.
        """
        device = self.get_device()
        dtype = self.dit.dtype
        # The clock is per request and lives on the host.
        state = self.request_state(fwd_info.request_id)

        if graph_walk == ROLLOUT_WALK:
            requested = int(fwd_info.step_metadata.get("num_steps", 0) or 0)
            rollout_step = int(state.get("rollout_step", 0))
            if requested and rollout_step >= requested:
                # Async scheduling can dispatch iteration N+1 before
                # check_stop(N) registers the loop's finish signal. Veto here,
                # before any tensor work: None skips the forward, so the
                # overshoot never commits a frame into the ring.
                logger.info(
                    "Waypoint dit: skipping async-overshoot rollout step %d "
                    "(request %s runs %d steps)",
                    rollout_step, fwd_info.request_id, requested,
                )
                return None

        frame_pos = int(state.get("frame_pos", 0))

        if graph_walk == PRIME_WALK:
            # Prime has its own idle action and must not consume the
            # client's action zero.
            mouse, button, scroll = self._idle_controller(device, dtype)
        else:
            action_index = int(state.get("rollout_step", 0))
            mouse, button, scroll = self._controller_slice(
                inputs, action_index, device, dtype
            )
        tensor_inputs = {
            # [1], never []: see the class docstring.
            "frame_pos": torch.full((1,), frame_pos, dtype=torch.int64, device=device),
            "mouse": mouse,
            "button": button,
            "scroll": scroll,
        }
        if graph_walk == PRIME_WALK:
            # Already-settled x0 for a real frame, off the vae_encoder node.
            tensor_inputs["latent"] = inputs["latent"][0].to(device=device, dtype=dtype)
        elif graph_walk == ROLLOUT_WALK:
            tensor_inputs["noise"] = self._frame_noise(
                fwd_info.random_seed, frame_pos, device, dtype
            )
        else:
            raise ValueError(f"Unknown Waypoint graph walk: {graph_walk!r}")

        tensor_inputs.update({
            f"{DECODER_HISTORY_PREFIX}{idx}": value
            for idx, value in enumerate(self._history_state(fwd_info.request_id))
        })

        return NodeInputs(
            tensor_inputs=tensor_inputs,
            input_seq_len=self.config.tokens_per_frame,
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict:
        """Concatenate every row's tensors along the batch dim, in
        ``engine_inputs.request_ids`` order (``inputs[i]`` pairs positionally
        with row ``i`` -- the row-order invariant every batched resource below
        this node relies on). ``len(inputs) == 1`` returns the row unchanged,
        no copy.
        """
        if len(inputs) == 1:
            return inputs[0].tensor_inputs
        return {
            key: torch.cat([row.tensor_inputs[key] for row in inputs], dim=0)
            for key in inputs[0].tensor_inputs
        }

    def _frame_noise(
        self,
        request_seed: int,
        frame_pos: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """``[1, 1, C, H, W]`` of fresh noise for this frame.

        Drawn straight onto the device. Determinism is per-GPU: a CUDA
        generator reproduces run-to-run on the same arch + torch build, not
        against a CPU draw or another arch.
        """
        shape = (1, 1, *self.config.latent_shape)
        if device.type == "meta":
            # Shape-only builds (meta-device shell tests) have no RNG to seed.
            return torch.empty(shape, device=device, dtype=dtype)
        generator = torch.Generator(device=device).manual_seed(
            _frame_seed(request_seed, frame_pos)
        )
        return torch.randn(
            shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def _controller_slice(
        self,
        inputs: NameToTensorList,
        action_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This frame's ``(mouse, button, scroll)``, each ``[1, 1, *]``.

        The request carries the whole scripted action stream as ``[1, F, *]``
        and one frame is sliced out per step; the conductor re-injects the
        same loop-external tensor every iteration while the rollout counter
        advances through it. Prime never calls this method.
        """
        widths = {"mouse": 2, "button": self.config.n_buttons, "scroll": 1}
        out = []
        for name, width in widths.items():
            supplied = inputs.get(name) if inputs is not None else None
            if not supplied:
                raise ValueError(f"Waypoint rollout is missing its {name!r} action stream.")
            stream = supplied[0]
            if stream.ndim != 3 or stream.shape[0] != 1 or stream.shape[2] != width:
                raise ValueError(
                    f"Waypoint {name!r} stream must have shape [1, steps, {width}]; "
                    f"got {tuple(stream.shape)}."
                )
            if not 0 <= action_index < stream.shape[1]:
                raise IndexError(
                    f"Waypoint action index {action_index} is outside the {name!r} "
                    f"stream of length {stream.shape[1]}."
                )
            out.append(
                stream[:, action_index : action_index + 1].to(device=device, dtype=dtype)
            )
        return tuple(out)

    def _idle_controller(
        self, device: torch.device, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((1, 1, 2), dtype=dtype, device=device),
            torch.zeros((1, 1, self.config.n_buttons), dtype=dtype, device=device),
            torch.zeros((1, 1, 1), dtype=dtype, device=device),
        )

    def _zero_histories(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Fresh zero-valued histories, shaped for this config's latent grid.

        Takes a device rather than a real latent since only the shape
        matters; used both from a request's first ``prepare_inputs`` and
        from the capture template.
        """
        seed_latent = torch.zeros(
            (1, *self.config.latent_shape), dtype=self.ae_dtype, device=device,
        )
        return initial_decoder_histories(self.taehv, seed_latent)

    def _history_state(self, request_id: str) -> tuple[torch.Tensor, ...]:
        """The request's nine decoder histories, seeded to zero on first use
        (prime, since prime always runs first)."""
        state = self.request_state(request_id)
        histories = tuple(
            state.get(f"{DECODER_HISTORY_PREFIX}{idx}") for idx in range(9)
        )
        if any(value is None for value in histories):
            histories = self._zero_histories(self.get_device())
            for idx, value in enumerate(histories):
                state.add(f"{DECODER_HISTORY_PREFIX}{idx}", value)
        return histories

    # ------------------------------------------------------------------
    # declare_step
    # ------------------------------------------------------------------

    def _declared_frames(self, request_ids: list[str]) -> tuple[tuple[str, int], ...]:
        """Every request's ring clock, for ``RingKVStep``.

        Read off the same host ``state["frame_pos"]`` that ``prepare_inputs``
        derives its device tensor from and ``postprocess`` advances, never off
        ``inputs`` (a device tensor by then, so an ``.item()`` would sync per
        step). One pair per rid, with no ``None`` in the return type: a batch
        this submodule cannot describe is refused rather than silently
        skipping ``RingKVManager``'s continuity check.
        """
        return tuple(
            (rid, int(self.request_state(rid).get("frame_pos", 0))) for rid in request_ids
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease=None,
        piecewise_leases=None,
        **kwargs,
    ) -> SubmoduleStep:
        """Name both resources so the runner drives their lifecycle.

        Neither step carries segments or reserves anything: a ring overwrites
        in place, so there is no span to admit. Declaring them is still
        required for ``admit`` to hand a request one of the node's worlds. The
        KV step carries the ring clock via ``RingKVStep``, letting
        ``RingKVManager.admit`` check it against the frame its last ``commit``
        recorded for that rid.
        """
        del graph_walk, inputs, slot_lease, piecewise_leases, kwargs
        return SubmoduleStep(
            steps={
                KV_RESOURCE: RingKVStep(frames=self._declared_frames(request_ids)),
                ATTN_RESOURCE: AttentionStep(),
            },
        )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        frame_pos: torch.Tensor,
        mouse: torch.Tensor,
        button: torch.Tensor,
        scroll: torch.Tensor,
        noise: torch.Tensor | None = None,
        latent: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """One frame. ``rollout`` denoises it from noise, ``prime`` appends a
        real one; both commit to the ring, decode it with TAEHV in the same
        call, and return the decoded frame, the updated histories, and the
        clock passthrough.

        ``engine_inputs`` is unused here, on purpose: under capture it is the
        dummy request's forever. The ring and attention backend come off
        ``self.node_resources``, resolved once at ``bind_node_resources``.
        """
        del engine_inputs
        histories = tuple(kwargs.pop(f"{DECODER_HISTORY_PREFIX}{idx}") for idx in range(9))
        if kwargs:
            raise TypeError(f"unexpected dit inputs: {sorted(kwargs)}")
        pos = frame_pos

        if graph_walk == ROLLOUT_WALK:
            out = self.dit.generate_frame(
                noise, pos, mouse=mouse, button=button, scroll=scroll
            )
        elif graph_walk == PRIME_WALK:
            out = self.dit.append_frame(
                latent, pos, mouse=mouse, button=button, scroll=scroll
            )
        else:
            raise ValueError(f"Unknown Waypoint graph walk: {graph_walk!r}")

        frames, updated = self._decode_latent(
            self.taehv,
            out.squeeze(1),
            histories,
            output_size=self.pixel_size,
            initialize=graph_walk == PRIME_WALK,
        )
        result: NameToTensorList = {
            "video_output": [frames],
            # Loop-back edge for next iteration's "clock" input; see
            # prepare_inputs. Harmless on prime, whose node declares no
            # outputs at all.
            "clock": [frame_pos],
        }
        result.update({
            f"{DECODER_HISTORY_PREFIX}{idx}": [value]
            for idx, value in enumerate(updated)
        })
        return result

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Run the batched ``forward`` once, then split its rows back out.

        Row ``i`` of every output tensor belongs to
        ``engine_inputs.request_ids[i]``, the same row-order invariant
        ``preprocess`` used going in.
        """
        out = self.forward(graph_walk, engine_inputs=engine_inputs, **kwargs)
        # Each value is a one-element list holding the batched tensor; index
        # the tensor's rows, not the list.
        return {
            rid: {
                key: [value[0][i]] if key == "video_output" else [value[0][i : i + 1]]
                for key, value in out.items()
            }
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[CudaGraphConfig]:
        """Both walks, as optional captures.

        Prime's inputs are the rollout template with ``noise`` renamed, so it
        reuses the pool the rollout graph already sized.
        """
        del tp_world_size  # no sharded nodes; the ring and the mask do not shard
        if not self.config.cuda_graph:
            return []
        dtype = self.dit.dtype
        frame = (1, 1, *self.config.latent_shape)

        def template(latent_key: str) -> NodeInputs:
            tensor_inputs = {
                latent_key: torch.zeros(frame, dtype=dtype, device=device),
                "frame_pos": torch.zeros(1, dtype=torch.int64, device=device),
                "mouse": torch.zeros((1, 1, 2), dtype=dtype, device=device),
                "button": torch.zeros(
                    (1, 1, self.config.n_buttons), dtype=dtype, device=device
                ),
                "scroll": torch.zeros((1, 1, 1), dtype=dtype, device=device),
            }
            # The fused decode's histories, exactly as prepare_inputs builds
            # them. "clock" is absent: it is never read as an input, only
            # produced as an output.
            tensor_inputs.update({
                f"{DECODER_HISTORY_PREFIX}{idx}": value
                for idx, value in enumerate(self._zero_histories(device))
            })
            return NodeInputs(
                tensor_inputs=tensor_inputs,
                input_seq_len=self.config.tokens_per_frame,
            )

        # Rollout listed first: ``prepare_for_capture``'s ``(bs, num_tokens)``
        # sort is descending and stable, so rollout captures first at every
        # size tie with prime and sizes the shared graph pool at its biggest
        # allocation; prime reuses freed blocks going down.
        batch_sizes = _rollout_capture_batch_sizes(self.config.step_batch_size)
        walks = [(ROLLOUT_WALK, "noise")]
        if self.config.capture_dit_prime:
            # Prime captures the same geometric buckets as rollout, so a
            # multi-request prime batch replays a captured graph instead of
            # falling back to a runtime eager re-trace (see
            # ``RingKVManager.plan``).
            walks.append((PRIME_WALK, "latent"))
        configs = []
        for walk, latent_key in walks:
            single_request_inputs = template(latent_key)
            configs.append(BatchedCudaGraphConfig(
                capture_graph_walk=walk,
                single_request_inputs=single_request_inputs,
                capture_batch_sizes=batch_sizes,
                capture_forward_method="forward_batched",
                # The DiT compiles its two reference-shaped fullgraph regions
                # itself. Compiling this wrapper would fuse across their boundary.
                compile=False,
                # Every tensor here is a per-request row that ``preprocess``
                # concatenates on dim 0 -- including ``button``, whose
                # ``n_buttons`` can coincidentally equal a bucket's token
                # count (e.g. 360p at bs=2) and fool the runner's size-based
                # guess.
                input_seq_dims={key: 0 for key in single_request_inputs.tensor_inputs},
            ))
        return configs

    # ------------------------------------------------------------------
    # step tail
    # ------------------------------------------------------------------

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: NodeInputs | None = None,
        **kwargs,
    ):
        """Advance the ring clock by exactly one committed frame, and copy the
        fused decode's updated histories into the request's stable tensors.

        Both walks commit, so both advance the clock. The clock advance is
        metadata only, and the history copy is a device ``copy_`` into the
        same fixed-address tensors ``prepare_inputs`` reads back next call --
        neither syncs.
        """
        del inputs, kwargs
        state = self.request_state(request_id)
        state.add("frame_pos", int(state.get("frame_pos", 0)) + 1)
        if request_info.graph_walk == ROLLOUT_WALK:
            state.add("rollout_step", int(state.get("rollout_step", 0)) + 1)
        for idx in range(9):
            key = f"{DECODER_HISTORY_PREFIX}{idx}"
            values = outputs.get(key)
            if not values:
                raise RuntimeError(f"fused dit+decode returned no {key}")
            target = state.get(key)
            if target is None:
                state.add(key, values[0].clone())
            else:
                target.copy_(values[0])

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        """Stop the rollout loop after exactly ``num_steps`` iterations.

        Iteration k (0-based) is still being postprocessed when the loop
        counter reads k, so N frames means firing at ``k + 1 >= N``. Under
        async scheduling an overshoot iteration this signal is too late to
        stop is vetoed instead in ``prepare_inputs``.
        """
        del request_id, outputs
        if request_info.graph_walk != ROLLOUT_WALK:
            return set()
        iter_idx = request_info.dynamic_loop_iter_counts.get(ROLLOUT_LOOP_NAME, 0)
        requested = int(request_info.step_metadata.get("num_steps", 0) or 0)
        if requested > 0 and iter_idx + 1 >= requested:
            return {ROLLOUT_LOOP_NAME}
        return set()

    def cleanup_request(self, request_id: str):
        """Drop the request's clock. The ring itself is released by the
        engine's ``remove_request`` sweep (``RingKVManager.remove_request``),
        not here."""
        super().cleanup_request(request_id)


class WaypointVaeEncoderSubmodule(_SingleRequestMixin, _FunctionalAeMixin, NodeSubmodule):
    """TAEHV encoder: ``temporal_compression`` raw frames -> one latent frame.

    ``image_inputs`` ``[4, 720, 1280, 3]`` uint8 -> ``latent``
    ``[1, 1, 32, 32, 64]``, the dit's priming input unchanged. Prime walk only.
    """

    disable_torch_compile = True
    # Dtype layout is settled at load; an engine-level cast would round it
    # underneath the AE.
    disable_autocast = True

    def __init__(self, taehv: torch.nn.Module, config: WaypointConfig):
        super().__init__()
        validate_taehv_architecture(taehv)
        self.taehv = taehv
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        """The seed clip, scaled into the AE's dtype. Cast then divide, the
        reference's order: 0-255 is exact in bf16, so the divide rounds once."""
        del graph_walk, fwd_info, kwargs
        frames = inputs["image_inputs"][0]
        scale = frames.dtype == torch.uint8
        frames = frames.to(device=self.get_device(), dtype=self.ae_dtype)
        return NodeInputs(tensor_inputs={"image": frames.div(255) if scale else frames})

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        image: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        """Encode the seed clip into the latent the dit primes the world on.
        Request ids are never read here: forward deletes ``engine_inputs``
        outright, so it never needs a real request's ids, captured or not."""
        del graph_walk, kwargs
        del engine_inputs
        latent = encode_seed_clip(
            self.taehv, image, output_size=self.encoded_size
        )
        # [B, C, h, w] -> [B, 1, C, h, w]: the dit's frame axis, added where the
        # reference adds it (``WorldEngine.append_frame``).
        return {"latent": [latent.unsqueeze(1)]}

    @property
    def _captured_image_shape(self) -> tuple[int, int, int, int]:
        """The one ``image`` shape ``get_cuda_graph_configs`` captures."""
        return (self.config.temporal_compression, *self.pixel_size, 3)

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[CudaGraphConfig]:
        del tp_world_size
        if not self.config.cuda_graph:
            return []
        image = torch.zeros(
            self._captured_image_shape, dtype=self.ae_dtype, device=device,
        )
        return [BatchedCudaGraphConfig(
            capture_graph_walk=PRIME_WALK,
            single_request_inputs=NodeInputs(
                tensor_inputs={"image": image},
                input_seq_len=self.config.temporal_compression,
            ),
            capture_batch_sizes=[1],
            capture_forward_method="forward_batched",
            compile=True,
        )]

    def can_use_cuda_graphs(self, batch, model_inputs) -> bool:
        """``_seed_clip`` accepts any 16:9 size, but the graph's static
        buffer is sized for exactly one H/W (``pixel_size``). A batch with
        any other image shape must run eager, where ``encode_seed_clip``
        resizes it to ``encoded_size``."""
        if not super().can_use_cuda_graphs(batch, model_inputs):
            return False
        return all(
            node_inputs.tensor_inputs["image"].shape == self._captured_image_shape
            for node_inputs in model_inputs
        )
