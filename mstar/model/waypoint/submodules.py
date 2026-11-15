from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    geometric_batch_sizes,
)
from mstar.engine.accelerator_graph_runner import compile_kwargs
from mstar.engine.state import ArrayStep, Segment, SubmoduleStep, position
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.model.waypoint.components.dit import WaypointDiT
from mstar.model.waypoint.components.taehv import (
    decode_latent,
    encode_seed_clip,
    encoded_size_for_latent,
    pixel_size_for_latent,
    validate_decoder_histories,
    validate_taehv_architecture,
)
from mstar.model.waypoint.config import WaypointConfig

PRIME_WALK = "prime"
ROLLOUT_WALK = "rollout"
ROLLOUT_LOOP_NAME = "rollout_loop"
# the client stream of per-frame controller rows (see WaypointModel._action_rows)
ACTIONS_STREAM = "actions"

# The state the dit node declares: the world's ring of frames, and the decoder's histories.
KV = "kv"
DECODER = "decoder"

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


# The prime-walk capture and the decode compile share this mode. max-autotune
# until the default compile level (mstar serve -O) is measured on Waypoint.
_COMPILE_MODE = "max-autotune-no-cudagraphs"


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
        validate_decoder_histories(taehv, config.latent_height, config.latent_width)
        self.dit = dit
        self.taehv = taehv
        self.config = config
        # bound at load by `bind_state`
        self.kv = self.decoder = None
        # The prime-walk capture's own arguments, so Inductor picks the same
        # kernels bit-for-bit. Wrapping just the decode call keeps the engine
        # from compiling across the denoise/decode boundary.
        self._decode_latent = (
            torch.compile(decode_latent, **compile_kwargs(_COMPILE_MODE))
            if config.accelerator_graph else decode_latent
        )

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv, self.decoder = handles[KV], handles[DECODER]

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
        """This frame's row: its index, the frames the ring has counted for
        the request, its controller row, and either the noise to denoise from
        (rollout) or the latent to prime with. A rollout step on the actions
        stream's empty final chunk is declined, which ends the loop.

        Runs on the host, outside any captured region, which is why the noise
        is drawn here rather than in ``forward``: a captured region cannot
        call the RNG.

        ``inputs.get("clock")`` is never read; it only has to be present in
        ``input_names`` for ``GraphNode.is_ready_for_speculation`` to propose
        the next iteration as a same-node speculation target.
        """
        device = self.get_device()
        dtype = self.dit.dtype
        frame_pos = position(self.kv, fwd_info.rid_handle, "main") // self.config.tokens_per_frame

        if graph_walk == PRIME_WALK:
            # Prime has its own idle action; the client's first action is the
            # first rollout frame's.
            mouse, button, scroll = self._idle_controller(device, dtype)
        else:
            rows = inputs.get(ACTIONS_STREAM)
            if not rows:
                assert kwargs.get("is_final_stream_chunk"), "an empty chunk before the end of the stream"
                return None
            mouse, button, scroll = self._controller_row(rows[0], device, dtype)
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

    def _controller_row(
        self, row: torch.Tensor, device: torch.device, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One action row, ``[mouse dx, dy | buttons | scroll]``, as this
        frame's ``(mouse, button, scroll)``, each ``[1, 1, *]``."""
        row = row.to(device=device, dtype=dtype).reshape(1, 1, -1)
        return tuple(row.split((2, self.config.n_buttons, 1), dim=-1))

    def _idle_controller(
        self, device: torch.device, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((1, 1, 2), dtype=dtype, device=device),
            torch.zeros((1, 1, self.config.n_buttons), dtype=dtype, device=device),
            torch.zeros((1, 1, 1), dtype=dtype, device=device),
        )

    # ------------------------------------------------------------------
    # declare_step
    # ------------------------------------------------------------------

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease=None,
        piecewise_leases=None,
        **kwargs,
    ) -> SubmoduleStep:
        """One frame per row, which prime and rollout both keep in the ring,
        and each row's decoder histories."""
        del graph_walk, inputs, slot_lease, piecewise_leases, kwargs
        return SubmoduleStep(
            segments=[Segment(rid, "main", self.config.tokens_per_frame) for rid in request_ids],
            state={KV: ArrayStep(), DECODER: ArrayStep()},
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
        call, write the decoder's histories back, and return the decoded frame
        and the clock passthrough.

        ``engine_inputs`` is unused here, on purpose: under capture it is the
        dummy request's forever.
        """
        del engine_inputs
        if kwargs:
            raise TypeError(f"unexpected dit inputs: {sorted(kwargs)}")
        stored = self.decoder.read(0).fields
        histories = tuple(stored[f"history_{idx}"] for idx in range(len(stored)))
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
        self.decoder.write(0, **{f"history_{idx}": value for idx, value in enumerate(updated)})
        return {
            "video_output": [frames],
            # Loop-back edge for next iteration's "clock" input; see
            # prepare_inputs. Harmless on prime, whose node declares no
            # outputs at all.
            "clock": [frame_pos],
        }

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

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[AcceleratorGraphConfig]:
        """Both walks, as optional captures.

        Prime's inputs are the rollout template with ``noise`` renamed, so it
        reuses the pool the rollout graph already sized.
        """
        del tp_world_size  # no sharded nodes; the ring and the mask do not shard
        if not self.config.accelerator_graph:
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
            return NodeInputs(
                tensor_inputs=tensor_inputs,
                input_seq_len=self.config.tokens_per_frame,
            )

        # Rollout listed first: ``prepare_for_capture``'s ``(bs, num_tokens)``
        # sort is descending and stable, so rollout captures first at every
        # size tie with prime and sizes the shared graph pool at its biggest
        # allocation; prime reuses freed blocks going down.
        # a step of n rows replays the smallest bucket >= n, the tail padded with
        # dummy rows (see RingKVManager.plan)
        batch_sizes = geometric_batch_sizes(self.config.step_batch_size)
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
            configs.append(BatchedAcceleratorGraphConfig(
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


class WaypointVaeEncoderSubmodule(_FunctionalAeMixin, NodeSubmodule):
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

    def max_batch_size(self, graph_walk: str):
        # one seed clip per step: the capture buckets hold a single row
        return 1

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

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        # what the capture config runs: a replayed graph's outputs are mapped
        # back to requests by row id
        (rid,) = engine_inputs.request_ids
        return {rid: self.forward(graph_walk, engine_inputs=engine_inputs, **kwargs)}

    @property
    def _capture_hw(self) -> tuple[tuple[int, int], ...]:
        """Native ``pixel_size`` plus ``seed_capture_sizes``; one bucket each."""
        return tuple(sorted({self.pixel_size, *self.config.seed_capture_sizes}))

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[AcceleratorGraphConfig]:
        del tp_world_size
        if not self.config.accelerator_graph:
            return []
        configs = []
        for h, w in self._capture_hw:
            image = torch.zeros(
                (self.config.temporal_compression, h, w, 3),
                dtype=self.ae_dtype, device=device,
            )
            configs.append(BatchedAcceleratorGraphConfig(
                compile_mode=_COMPILE_MODE,
                capture_graph_walk=PRIME_WALK,
                single_request_inputs=NodeInputs(
                    tensor_inputs={"image": image},
                    input_seq_len=self.config.temporal_compression,
                ),
                capture_batch_sizes=[1],
                capture_forward_method="forward_batched",
                compile=True,
                additional_key_info=(h, w),
            ))
        return configs

    def cg_key_info(
        self, graph_walk: str, per_request_info: dict[str, CurrentForwardPassInfo],
        per_request_input_metadata=None, **kwargs,
    ) -> tuple[int, int] | None:
        """The captured (H, W) bucket for this batch's seed, or None for eager.

        The seed shape isn't on ``model_inputs`` at lease time, so it is read
        off ``step_metadata["seed_hw"]``, set by ``WaypointModel.walk_plan``."""
        del per_request_input_metadata, kwargs
        if graph_walk != PRIME_WALK:
            return None
        sizes = set()
        for info in per_request_info.values():
            seed_hw = info.step_metadata.get("seed_hw")
            if seed_hw is None:
                return None
            sizes.add(tuple(seed_hw))
        if len(sizes) != 1:
            return None
        size = sizes.pop()
        return size if size in self._capture_hw else None
