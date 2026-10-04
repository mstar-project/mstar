"""The DiT scaffold's Loop body: one flow-matching denoise step per iteration.

``DenoiseLoopSubmodule`` is the node behind a model's ``Loop("denoise_loop", dit)``.
It owns everything that is the same for every flow/diffusion transformer and
leaves the model-specific parts to a handful of hooks:

  engine contract (base)                          model hooks (subclass)
  ───────────────────────────────────────────     ──────────────────────────────────────
  step index from the engine's loop counter       bucket_key_for(fwd_info)      -> hashable
  per-request schedule + seeded initial noise     schedule_for(fwd_info, key)   -> FlowMatchSchedule
  overshoot veto (async scheduling)               seed_latents(fwd_info, key, generator) -> [L, C]
  equal-key request batching (can_batch)          request_inputs(fwd_info, inputs, key) -> {name: [..]}
  stacked preprocess / per-row outputs            num_tokens(key)               -> int
  stop after the request's own step count         denoise(engine_inputs, key, latents, timestep,
  per-key CUDA-graph buckets (Euler inside)               sigma, sigma_next, **cond) -> [B, L, C]
  ragged-attention step declaration               capture_request_inputs(key, device) -> {name: [..]}
                                                  attention_segments(key) -> ((label, span), ...)

Conventions the base fixes:

* The loop-back edges are :attr:`~DenoiseLoopSubmodule.loop_back_names`
  (``latents`` plus whatever ``SOLVER_STATE`` a multi-step scheduler carries);
  every other input is re-injected by the Loop each iteration. The step index
  ``k`` is ``fwd_info.dynamic_loop_iter_counts`` for the loop, refreshed by the
  worker before each batch — no host sync per step.
* A request's schedule and its device-resident ``sigmas`` / ``timesteps`` live in
  the engine-owned ``PerRequestState`` (freed with the request).
* Per-step per-request scalars (``STEP_SCALARS``: ``sigma``, ``sigma_next``,
  ``timestep``) are sliced out of the schedule ``schedule_for`` returns and
  travel as ``[1]`` device tensors, so a batch of requests at different steps
  (or step counts) shares one forward, and a captured graph reads them from
  staged buffers. Declaring one is enough: the base stages every source the
  table names on the device and slices it per step.
* Rows are batched only at an identical ``bucket_key``, which is also the
  CUDA-graph bucket key. It need not be shape alone: anything that has to match
  for two rows to share a forward belongs in it (the latent grid, text length
  and conditioning layout, but equally whether CFG is on).
* Every span a forward attends over must be declared: the default is one
  ``"main"`` segment of all the request's tokens; a model whose layers also
  attend over a sub-span (a refiner over the image tokens alone, say) lists
  each span with its own label in :meth:`attention_segments` and threads
  :meth:`ragged_for` that label to those layers.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.convenience import RaggedAttentionCallable
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)


class StepScalar(NamedTuple):
    """One per-step per-request scalar, sliced out of the request's schedule.

    Travels as a ``[1]`` device tensor so a batch of requests at different steps
    shares one forward (see the module docstring)."""

    # Attribute of the FlowMatchSchedule ``schedule_for`` returns ("sigmas").
    # The base stages it on the device and slices it per step, so declaring a
    # scalar whose source the schedule does not carry is an error.
    source: str
    offset: int = 0  # 0 reads step k; 1 reads k + 1, the step's target
    # viewed as [B, 1, 1] to broadcast over [B, L, C]; a plain [B] otherwise
    broadcast: bool = True
    capture_fill: float = 1.0  # placeholder value for a graph capture's dummy rows


class DenoiseLoopSubmodule(NodeSubmodule):
    """Base class for the ``dit`` node of an image/video flow model (see module docstring)."""

    # The engine's blanket compile is off: the model compiles its transformer region
    # itself (per shape) when asked, and the captured graphs record those kernels.
    disable_torch_compile = True

    # The sample the scheduler advances: seeded on iteration 0, routed back to
    # this node every iteration after. A subclass that renames it must name
    # :meth:`denoise`'s parameter to match — the carried tensors are passed by
    # keyword.
    LATENTS = "latents"
    # The span every forward attends over; see :meth:`attention_segments`.
    MAIN_SPAN = "main"

    # Loop-back edges beyond LATENTS: solver state a multi-step scheduler carries
    # between iterations, e.g. wan22's UniPC ("unipc_model_outputs",
    # "unipc_last_sample"). :meth:`denoise` returns these alongside the new
    # sample and :meth:`seed_loop_back` supplies iteration 0's values.
    SOLVER_STATE: tuple[str, ...] = ()

    # name -> where in the request's schedule this step's value comes from. A
    # scheduler needing another scalar (a per-step guidance scale, say) adds an
    # entry here instead of overriding prepare_inputs / preprocess / _run.
    STEP_SCALARS: Mapping[str, StepScalar] = {
        "sigma": StepScalar("sigmas"),
        "sigma_next": StepScalar("sigmas", offset=1, capture_fill=0.0),
        "timestep": StepScalar("timesteps", broadcast=False, capture_fill=1000.0),
    }

    def __init__(
        self,
        *,
        loop_name: str,
        max_batch_size: int = 8,
        attn_resource_key: str | None = None,
        capture_buckets: Sequence[tuple[str, Hashable]] = (),
        capture_batch_sizes: Sequence[int] = (1, 2, 4, 8),
        replay_walks: Mapping[str, Sequence[str]] | None = None,
    ):
        super().__init__()
        self.loop_name = loop_name
        self._max_batch_size = int(max_batch_size)
        self.attn_resource_key = attn_resource_key
        # (graph_walk, bucket_key) pairs to capture; the same key on another walk
        # replays only if ``replay_walks`` says so.
        self.capture_buckets = list(capture_buckets)
        self.capture_batch_sizes = sorted(int(b) for b in capture_batch_sizes)
        self.replay_walks = dict(replay_walks or {})
        # Per-key derived tensors (rotary tables, ...) built on first use and never
        # evicted: a captured graph reads them at fixed addresses.
        self._layouts: dict[Hashable, Any] = {}
        # One callable per attention label for the life of a binding; see
        # :meth:`ragged_for`.
        self._ragged_fns: dict[str, RaggedAttentionCallable] = {}

    @property
    def loop_back_names(self) -> tuple[str, ...]:
        """The node's loop-back edges, the sample first."""
        return (self.LATENTS, *self.SOLVER_STATE)

    # ------------------------------------------------------------------ hooks
    def bucket_key_for(self, fwd_info: CurrentForwardPassInfo) -> Hashable:
        """What has to match for two rows to share a forward (and a captured graph)."""
        raise NotImplementedError

    def schedule_for(self, fwd_info: CurrentForwardPassInfo, bucket_key: Hashable) -> FlowMatchSchedule:
        raise NotImplementedError

    def seed_latents(
        self, fwd_info: CurrentForwardPassInfo, bucket_key: Hashable, generator: torch.Generator,
    ) -> torch.Tensor:
        """Initial noise for one request, ``[L, C]`` on the CPU generator's device."""
        raise NotImplementedError

    def seed_loop_back(
        self, fwd_info: CurrentForwardPassInfo, bucket_key: Hashable, generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Iteration 0's value for every loop-back edge, without a batch dim.

        Default: seeded noise for the sample and nothing else. A model with
        ``SOLVER_STATE`` adds that scheduler's initial state here."""
        return {self.LATENTS: self.seed_latents(fwd_info, bucket_key, generator)}

    def request_inputs(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, bucket_key: Hashable,
    ) -> dict[str, torch.Tensor]:
        """The request's conditioning tensors for this step, without a batch dim."""
        raise NotImplementedError

    def num_tokens(self, bucket_key: Hashable) -> int:
        """Tokens one request contributes to the step (sizes the attention segment / buckets)."""
        raise NotImplementedError

    def capture_request_inputs(self, bucket_key: Hashable, device: torch.device) -> dict[str, torch.Tensor]:
        """Placeholder loop-back tensors + conditioning for one row of this bucket (capture time)."""
        raise NotImplementedError

    def denoise(
        self,
        engine_inputs: ModelInputsFromEngine,
        bucket_key: Hashable,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        **cond: torch.Tensor,
    ) -> torch.Tensor | Mapping[str, torch.Tensor]:
        """One scheduler step for a stacked batch: ``[B, L, C] -> [B, L, C]``.

        A model with ``SOLVER_STATE`` returns ``{name: stacked tensor}`` covering
        every loop-back edge instead of the new sample alone."""
        raise NotImplementedError

    def build_layout(self, bucket_key: Hashable, device: torch.device) -> Any:
        """Derived per-key state (e.g. rotary tables); cached by :meth:`layout`."""
        return None

    def attention_segments(self, bucket_key: Hashable) -> Sequence[tuple[str, int]]:
        """``(label, span)`` for every span one request's layers attend over, in
        declaration order. Default: one ``"main"`` segment of all its tokens."""
        return ((self.MAIN_SPAN, self.num_tokens(bucket_key)),)

    # --------------------------------------------------------------- helpers
    def layout(self, bucket_key: Hashable, device: torch.device) -> Any:
        entry = self._layouts.get(bucket_key)
        if entry is None:
            entry = self._layouts[bucket_key] = self.build_layout(bucket_key, device)
        return entry

    def step_index(self, fwd_info: CurrentForwardPassInfo) -> int:
        return int(fwd_info.dynamic_loop_iter_counts.get(self.loop_name, 0))

    def _ragged(self) -> RaggedAttentionCallable | None:
        """The attention callable for the ``"main"`` span, or None (SDPA)."""
        return self.ragged_for(self.MAIN_SPAN)

    def bind_node_resources(self, resources) -> None:
        super().bind_node_resources(resources)
        self._ragged_fns.clear()  # the memoized callables hold the old resource

    def ragged_for(self, label: str) -> RaggedAttentionCallable | None:
        """``(q, k, v) -> out`` over the segments declared under ``label``, or None
        when no ragged resource is bound (layers then fall back to SDPA).

        One callable per label for the life of the binding: the compiled transformer
        region guards on the identity of the callables it is handed, and a fresh one
        per step would recompile it every step until dynamo gave up."""
        if self.attn_resource_key is None:
            return None
        if label not in self._ragged_fns:
            resource = self.node_resources.get(self.attn_resource_key)
            if resource is None:
                return None
            self._ragged_fns[label] = RaggedAttentionCallable(resource, label)
        return self._ragged_fns[label]

    # --------------------------------------------------------- engine contract
    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> NodeInputs | None:
        # Keyed by the integer rid_handle, NOT by request_id. The handle is what
        # the engine keys batches by and what it hands check_stop
        state = self.request_state(fwd_info.rid_handle)
        k = self.step_index(fwd_info)
        device = self.get_device()
        if "schedule" not in state:
            bucket_key = self.bucket_key_for(fwd_info)
            schedule = self.schedule_for(fwd_info, bucket_key)
            state.add_all(
                schedule=schedule, bucket_key=bucket_key, num_steps=schedule.num_steps,
                **self._step_scalar_sources(schedule, device),
            )
        bucket_key, num_steps = state["bucket_key"], state["num_steps"]
        if k >= num_steps:
            # Async scheduling dispatched an iteration past this request's stop; None
            # makes the engine skip the forward (the cosmos3 / wan22 veto).
            logger.info("%s: skipping overshoot iteration %d (request %s runs %d steps)",
                        type(self).__name__, k, fwd_info.request_id, num_steps)
            return None
        tensors = {
            **self._carried_inputs(fwd_info, inputs, bucket_key, k, device),
            **{
                name: state[scalar.source][k + scalar.offset:k + scalar.offset + 1]
                for name, scalar in self.STEP_SCALARS.items()
            },
            **self.request_inputs(fwd_info, inputs, bucket_key),
        }
        # The step the FORWARD actually runs, which is not what check_stop logs: that
        # reads the in-flight batch's info, this reads the per-pass one, and under
        # speculation the two can disagree. A repeated k here is the loop-counter bug.
        # key_present distinguishes the two ways k can read 0: an fwd_info whose
        # counters were never seeded (key absent -- a freshly built object), versus
        # one that was seeded and never advanced (key present, value 0).
        _counts = fwd_info.dynamic_loop_iter_counts
        logger.info("%s: prepare request %s k=%d/%d walk=%s key_present=%s counts=%s info=%x",
                    type(self).__name__, fwd_info.request_id, k, num_steps, fwd_info.graph_walk,
                    self.loop_name in _counts, dict(_counts), id(_counts))
        return NodeInputs(
            tensor_inputs=tensors, input_seq_len=self.num_tokens(bucket_key), resource_step_info=bucket_key,
        )

    def _step_scalar_sources(
        self, schedule: FlowMatchSchedule, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """The schedule tensors ``STEP_SCALARS`` slices, device-resident, by source
        name: what ``prepare_inputs`` puts in the request's state.

        Override if you need custom logic here.
        """
        staged: dict[str, torch.Tensor] = {}
        for name, scalar in self.STEP_SCALARS.items():
            tensor = getattr(schedule, scalar.source, None)
            if tensor is None:
                raise AttributeError(
                    f"{type(self).__name__}.STEP_SCALARS[{name!r}] reads "
                    f"{scalar.source!r}, which {type(schedule).__name__} does not "
                    f"carry. A declared per-step scalar comes off the schedule "
                    f"{type(self).__name__}.schedule_for returns."
                )
            # Step k reads index k + offset, so an offset of 1 needs one entry
            # more than there are steps (which is why sigmas carries a terminal 0
            # and timesteps does not).
            needed = schedule.num_steps + scalar.offset
            if tensor.shape[0] < needed:
                raise ValueError(
                    f"{type(self).__name__}.STEP_SCALARS[{name!r}] reads "
                    f"{scalar.source}[k + {scalar.offset}], needing {needed} "
                    f"entries for {schedule.num_steps} steps, but "
                    f"{scalar.source} has {tensor.shape[0]}."
                )
            staged.setdefault(scalar.source, tensor.to(device))
        return staged

    def _carried_inputs(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList,
        bucket_key: Hashable, k: int, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """This iteration's loop-back tensors: the previous iteration's outputs, or
        seeds on the first one (where the Loop has nothing to route back yet).

        Past iteration 0 a missing edge is a routing bug: raise rather than reseed."""
        if k == 0:
            generator = torch.Generator(device="cpu").manual_seed(fwd_info.random_seed)
            return {
                name: tensor.to(device)
                for name, tensor in self.seed_loop_back(fwd_info, bucket_key, generator).items()
            }
        missing = [name for name in self.loop_back_names if not inputs.get(name)]
        if missing:
            raise RuntimeError(
                f"{type(self).__name__}: loop-back edge(s) {missing} missing at iteration {k} "
                f"of request {fwd_info.request_id}"
            )
        return {name: inputs[name][0] for name in self.loop_back_names}

    def can_batch(self, batch, model_inputs: list[NodeInputs]) -> bool:
        if len(model_inputs) < 2:
            return False
        return len({inp.resource_step_info for inp in model_inputs}) == 1

    def max_batch_size(self, graph_walk: str):
        return self._max_batch_size

    def preprocess(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs]) -> dict:
        keys = inputs[0].tensor_inputs.keys()
        stacked = {key: torch.stack([inp.tensor_inputs[key] for inp in inputs]) for key in keys}
        for name, scalar in self.STEP_SCALARS.items():
            # [B, 1] -> [B, 1, 1] to broadcast over [B, L, C], or a plain [B]
            stacked[name] = stacked[name].view(-1, 1, 1) if scalar.broadcast else stacked[name].view(-1)
        stacked["bucket_key"] = inputs[0].resource_step_info
        return stacked

    def _run(
        self, engine_inputs: ModelInputsFromEngine, kwargs: dict,
    ) -> dict[str, torch.Tensor]:
        """One stacked step, as ``{loop-back edge: [B, ...]}``."""
        bucket_key = kwargs.pop("bucket_key")
        carried = {name: kwargs.pop(name) for name in self.loop_back_names}
        scalars = {name: kwargs.pop(name) for name in self.STEP_SCALARS}
        out = self.denoise(engine_inputs, bucket_key, **carried, **scalars, **kwargs)
        return {self.LATENTS: out} if isinstance(out, torch.Tensor) else dict(out)

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return {name: [value[0]] for name, value in self._run(engine_inputs, kwargs).items()}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs,
    ) -> dict[str, NameToTensorList]:
        outputs = self._run(engine_inputs, kwargs)
        return {
            rid: {name: [value[i]] for name, value in outputs.items()}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def check_stop(self, request_id: int, request_info: CurrentForwardPassInfo, outputs) -> set[str]:
        # ``request_id`` is the engine's integer rid handle, the same key
        # prepare_inputs stored under.
        state = self.request_states.get(request_id)
        if state is None or "num_steps" not in state:
            logger.warning(
                "%s: check_stop found no schedule for request %r (state=%s); the loop "
                "will not be stopped by this pass",
                type(self).__name__, request_id, "absent" if state is None else "no num_steps",
            )
            return set()
        # iteration k is being postprocessed while the counter reads k, so N steps stop at k == N - 1
        k, num_steps = self.step_index(request_info), int(state["num_steps"])
        stop = k + 1 >= num_steps
        logger.info("%s: check_stop request %s k=%d/%d -> %s", type(self).__name__,
                    request_id, k, num_steps, "STOP" if stop else "continue")
        return {self.loop_name} if stop else set()

    # ------------------------------------------------------------ resources
    def _uniform_key(self, keys) -> Hashable | None:
        keys = set(keys)
        return keys.pop() if len(keys) == 1 else None

    def cg_key_info(
        self, graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
        per_request_input_metadata=None,
        **kwargs,
    ):
        del per_request_input_metadata, kwargs
        # Padding rows (a captured bucket's dummy requests) carry no step metadata; the
        # real rows decide the bucket.
        keys = [self.bucket_key_for(info) for info in per_request_info.values() if info.step_metadata]
        return self._uniform_key(keys) if keys else None

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None, piecewise_leases=None, **kwargs,
    ) -> SubmoduleStep | None:
        if self.attn_resource_key is None:
            return None
        # every row (padding rows included) carries its bucket key, so the spans are
        # identical between a bucket's capture and its replays
        segments = [
            Segment(request_id=rid, label=label, span=span)
            for rid, inp in zip(request_ids, inputs, strict=True)
            for label, span in self.attention_segments(inp.resource_step_info)
        ]
        return SubmoduleStep(
            segments=segments,
            steps={self.attn_resource_key: AttentionStep(causal=False)},
            cg_key_info=self._uniform_key(inp.resource_step_info for inp in inputs),
        )

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1):
        from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig

        configs = []
        for walk, bucket_key in self.capture_buckets:
            tensors = dict(self.capture_request_inputs(bucket_key, device))
            for name, scalar in self.STEP_SCALARS.items():
                tensors[name] = torch.full(
                    (1,), scalar.capture_fill, dtype=torch.float32, device=device,
                )
            single = NodeInputs(
                tensor_inputs=tensors, input_seq_len=self.num_tokens(bucket_key), resource_step_info=bucket_key,
            )
            configs.append(BatchedCudaGraphConfig(
                capture_graph_walk=walk,
                replay_graph_walks=[walk, *self.replay_walks.get(walk, ())],
                single_request_inputs=single,
                additional_key_info=bucket_key,
                # the model compiles its own transformer region; the graph records those kernels
                compile=False,
                capture_batch_sizes=list(self.capture_batch_sizes),
                # uncaptured sizes / shapes run the eager batched path, so don't cap the batch
                caps_eager_batch_size=False,
                # every static input is row-leading ([tokens, features] or a per-batch scalar); a text
                # embedding's hidden size can equal an edit bucket's token count (7680 for klein), which
                # would otherwise be mistaken for the token axis
                input_seq_dims={key: 0 for key in tensors},
            ))
        return configs


# The default names, for callers that only need those (the module's own graph
# wiring, tests). A subclass overrides the class attributes, not these.
LATENTS = DenoiseLoopSubmodule.LATENTS
MAIN_SPAN = DenoiseLoopSubmodule.MAIN_SPAN
