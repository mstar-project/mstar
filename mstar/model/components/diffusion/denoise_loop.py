"""The DiT scaffold's Loop body: one flow-matching denoise step per iteration.

``DenoiseLoopSubmodule`` is the node behind a model's ``Loop("denoise_loop", dit)``.
It owns everything that is the same for every flow/diffusion transformer and
leaves the model-specific parts to a handful of hooks:

  engine contract (base)                          model hooks (subclass)
  ───────────────────────────────────────────     ──────────────────────────────────────
  step index from the engine's loop counter       shape_key_for(fwd_info)       -> hashable
  per-request schedule + seeded initial noise     schedule_for(fwd_info, key)   -> FlowMatchSchedule
  overshoot veto (async scheduling)               seed_latents(fwd_info, key, generator) -> [L, C]
  equal-shape request batching (can_batch)        request_inputs(fwd_info, inputs, key) -> {name: [..]}
  stacked preprocess / per-row outputs            num_tokens(key)               -> int
  stop after the request's own step count         denoise(engine_inputs, key, latents, timestep,
  per-shape CUDA-graph buckets (Euler inside)              sigma, sigma_next, **cond) -> [B, L, C]
  ragged-attention step declaration               capture_request_inputs(key, device) -> {name: [..]}

Conventions the base fixes:

* The only loop-back edge is ``latents``; every other input is re-injected by the
  Loop each iteration. The step index ``k`` is ``fwd_info.dynamic_loop_iter_counts``
  for the loop, refreshed by the worker before each batch — no host sync per step.
* A request's schedule and its device-resident ``sigmas`` / ``timesteps`` live in
  the engine-owned ``PerRequestState`` (freed with the request).
* Per-step per-request scalars (``sigma``, ``sigma_next``, ``timestep``) travel as
  ``[1]`` device tensors, so a batch of requests at different steps (or step
  counts) shares one forward, and a captured graph reads them from staged
  buffers.
* Rows are batched only at an identical ``shape_key`` (latent grid, text
  length, conditioning layout); the key is also the CUDA-graph bucket key.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, Segment, SlotLease, SubmoduleStep
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)

LATENTS = "latents"
_STEP_KEYS = ("sigma", "sigma_next", "timestep")


class DenoiseLoopSubmodule(NodeSubmodule):
    """Base class for the ``dit`` node of an image/video flow model (see module docstring)."""

    # The engine's blanket compile is off: the model compiles its transformer region
    # itself (per shape) when asked, and the captured graphs record those kernels.
    disable_torch_compile = True

    def __init__(
        self,
        *,
        loop_name: str,
        max_batch_size: int = 8,
        attn_resource_key: str | None = None,
        capture_shapes: Sequence[tuple[str, Hashable]] = (),
        capture_batch_sizes: Sequence[int] = (1, 2, 4, 8),
        replay_walks: Mapping[str, Sequence[str]] | None = None,
    ):
        super().__init__()
        self.loop_name = loop_name
        self._max_batch_size = int(max_batch_size)
        self.attn_resource_key = attn_resource_key
        # (graph_walk, shape_key) pairs to capture; the same key on another walk
        # replays only if ``replay_walks`` says so.
        self.capture_shapes = list(capture_shapes)
        self.capture_batch_sizes = sorted(int(b) for b in capture_batch_sizes)
        self.replay_walks = dict(replay_walks or {})
        # Per-shape derived tensors (rotary tables, ...) built on first use and never
        # evicted: a captured graph reads them at fixed addresses.
        self._layouts: dict[Hashable, Any] = {}

    # ------------------------------------------------------------------ hooks
    def shape_key_for(self, fwd_info: CurrentForwardPassInfo) -> Hashable:
        raise NotImplementedError

    def schedule_for(self, fwd_info: CurrentForwardPassInfo, shape_key: Hashable) -> FlowMatchSchedule:
        raise NotImplementedError

    def seed_latents(
        self, fwd_info: CurrentForwardPassInfo, shape_key: Hashable, generator: torch.Generator,
    ) -> torch.Tensor:
        """Initial noise for one request, ``[L, C]`` on the CPU generator's device."""
        raise NotImplementedError

    def request_inputs(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, shape_key: Hashable,
    ) -> dict[str, torch.Tensor]:
        """The request's conditioning tensors for this step, without a batch dim."""
        raise NotImplementedError

    def num_tokens(self, shape_key: Hashable) -> int:
        """Tokens one request contributes to the step (sizes the attention segment / buckets)."""
        raise NotImplementedError

    def capture_request_inputs(self, shape_key: Hashable, device: torch.device) -> dict[str, torch.Tensor]:
        """Placeholder ``latents`` + conditioning for one request of this shape (capture time)."""
        raise NotImplementedError

    def denoise(
        self,
        engine_inputs: ModelInputsFromEngine,
        shape_key: Hashable,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        **cond: torch.Tensor,
    ) -> torch.Tensor:
        """One Euler step for a stacked batch: ``[B, L, C] -> [B, L, C]``."""
        raise NotImplementedError

    def build_layout(self, shape_key: Hashable, device: torch.device) -> Any:
        """Derived per-shape state (e.g. rotary tables); cached by :meth:`layout`."""
        return None

    # --------------------------------------------------------------- helpers
    def layout(self, shape_key: Hashable, device: torch.device) -> Any:
        entry = self._layouts.get(shape_key)
        if entry is None:
            entry = self._layouts[shape_key] = self.build_layout(shape_key, device)
        return entry

    def step_index(self, fwd_info: CurrentForwardPassInfo) -> int:
        return int(fwd_info.dynamic_loop_iter_counts.get(self.loop_name, 0))

    def _ragged(self):
        if self.attn_resource_key is None:
            return None
        resource = self.node_resources.get(self.attn_resource_key)
        return None if resource is None else resource.run

    # --------------------------------------------------------- engine contract
    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> NodeInputs | None:
        state = self.request_state(fwd_info.request_id)
        k = self.step_index(fwd_info)
        device = self.get_device()
        if "schedule" not in state:
            shape_key = self.shape_key_for(fwd_info)
            schedule = self.schedule_for(fwd_info, shape_key)
            state.add_all(
                schedule=schedule, shape_key=shape_key, num_steps=schedule.num_steps,
                sigmas=schedule.sigmas.to(device), timesteps=schedule.timesteps.to(device),
            )
        shape_key, num_steps = state["shape_key"], state["num_steps"]
        if k >= num_steps:
            # Async scheduling dispatched an iteration past this request's stop; None
            # makes the engine skip the forward (the cosmos3 / wan22 veto).
            logger.info("%s: skipping overshoot iteration %d (request %s runs %d steps)",
                        type(self).__name__, k, fwd_info.request_id, num_steps)
            return None
        if k == 0 or not inputs.get(LATENTS):
            generator = torch.Generator(device="cpu").manual_seed(fwd_info.random_seed)
            latents = self.seed_latents(fwd_info, shape_key, generator).to(device)
        else:
            latents = inputs[LATENTS][0]
        sigmas, timesteps = state["sigmas"], state["timesteps"]
        tensors = {
            LATENTS: latents,
            "sigma": sigmas[k:k + 1],
            "sigma_next": sigmas[k + 1:k + 2],
            "timestep": timesteps[k:k + 1],
            **self.request_inputs(fwd_info, inputs, shape_key),
        }
        return NodeInputs(
            tensor_inputs=tensors, input_seq_len=self.num_tokens(shape_key), resource_step_info=shape_key,
        )

    def can_batch(self, batch, model_inputs: list[NodeInputs]) -> bool:
        if len(model_inputs) < 2:
            return False
        return len({inp.resource_step_info for inp in model_inputs}) == 1

    def max_batch_size(self, graph_walk: str):
        return self._max_batch_size

    def preprocess(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs]) -> dict:
        keys = inputs[0].tensor_inputs.keys()
        stacked = {key: torch.stack([inp.tensor_inputs[key] for inp in inputs]) for key in keys}
        for key in _STEP_KEYS:
            # [B, 1] -> [B] for the timestep, [B, 1, 1] for the sigmas (broadcast over [B, L, C])
            stacked[key] = stacked[key].view(-1) if key == "timestep" else stacked[key].view(-1, 1, 1)
        stacked["shape_key"] = inputs[0].resource_step_info
        return stacked

    def _run(self, engine_inputs: ModelInputsFromEngine, kwargs: dict) -> torch.Tensor:
        shape_key = kwargs.pop("shape_key")
        latents = kwargs.pop(LATENTS)
        sigma, sigma_next, timestep = (kwargs.pop(key) for key in ("sigma", "sigma_next", "timestep"))
        return self.denoise(engine_inputs, shape_key, latents, timestep, sigma, sigma_next, **kwargs)

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return {LATENTS: [self._run(engine_inputs, kwargs)[0]]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs,
    ) -> dict[str, NameToTensorList]:
        new_latents = self._run(engine_inputs, kwargs)
        return {rid: {LATENTS: [new_latents[i]]} for i, rid in enumerate(engine_inputs.request_ids)}

    def check_stop(self, request_id: str, request_info: CurrentForwardPassInfo, outputs) -> set[str]:
        state = self.request_states.get(request_id)
        if state is None or "num_steps" not in state:
            return set()
        # iteration k is being postprocessed while the counter reads k, so N steps stop at k == N - 1
        if self.step_index(request_info) + 1 >= int(state["num_steps"]):
            return {self.loop_name}
        return set()

    # ------------------------------------------------------------ resources
    def _uniform_key(self, keys) -> Hashable | None:
        keys = set(keys)
        return keys.pop() if len(keys) == 1 else None

    def cg_key_info(self, graph_walk: str, per_request_info: dict[str, CurrentForwardPassInfo]):
        # Padding rows (a captured bucket's dummy requests) carry no step metadata; the
        # real rows decide the bucket.
        keys = [self.shape_key_for(info) for info in per_request_info.values() if info.step_metadata]
        return self._uniform_key(keys) if keys else None

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None, piecewise_leases=None, **kwargs,
    ) -> SubmoduleStep | None:
        if self.attn_resource_key is None:
            return None
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={self.attn_resource_key: AttentionStep(causal=False)},
            cg_key_info=self._uniform_key(inp.resource_step_info for inp in inputs),
        )

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1):
        from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig

        configs = []
        for walk, shape_key in self.capture_shapes:
            tensors = dict(self.capture_request_inputs(shape_key, device))
            tensors["sigma"] = torch.ones(1, dtype=torch.float32, device=device)
            tensors["sigma_next"] = torch.zeros(1, dtype=torch.float32, device=device)
            tensors["timestep"] = torch.full((1,), 1000.0, dtype=torch.float32, device=device)
            single = NodeInputs(
                tensor_inputs=tensors, input_seq_len=self.num_tokens(shape_key), resource_step_info=shape_key,
            )
            configs.append(BatchedCudaGraphConfig(
                capture_graph_walk=walk,
                replay_graph_walks=[walk, *self.replay_walks.get(walk, ())],
                single_request_inputs=single,
                additional_key_info=shape_key,
                # the model compiles its own transformer region; the graph records those kernels
                compile=False,
                capture_batch_sizes=list(self.capture_batch_sizes),
                # uncaptured sizes / shapes run the eager batched path, so don't cap the batch
                caps_eager_batch_size=False,
            ))
        return configs
