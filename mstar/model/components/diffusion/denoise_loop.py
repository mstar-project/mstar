"""The DiT scaffold's Loop body: one flow-matching denoise step per iteration.

``DenoiseLoopSubmodule`` is the node behind a model's ``Loop("denoise_loop", dit)``.
It owns everything that is the same for every flow/diffusion transformer and
leaves the model-specific parts to a handful of hooks:

  engine contract (base)                          model hooks (subclass)
  ───────────────────────────────────────────     ──────────────────────────────────────
  step index from the engine's loop counter       bucket_key_for(fwd_info)      -> hashable
  memoized schedule + seeded initial noise,       schedule_settings(fwd_info)   -> hashable
    drawn at ingest under PREFETCH_SEEDS          schedule_for(walk, settings, key) -> FlowMatchSchedule
  equal-key request batching (can_batch)          seed_latents(key, generator)  -> [L, C]
  stacked preprocess / per-row outputs            request_inputs(fwd_info, inputs, key) -> {name: [..]}
  per-key CUDA-graph buckets (Euler inside)       num_tokens(key)               -> int
  guidance branches, their split over the         model_output(engine_inputs, key, latents, timestep,
    cfg ranks and their combine (GUIDANCE)                     **cond) -> [B, ...] prediction
                                                  (or denoise(...), a model that steps itself)
  the step's arrays and their segments            capture_request_inputs(key, device) -> {name: [..]}
                                                  attention_segments(key) -> {array: ((label, span), ...)}

``bucket`` is the model's own key (``bucket_key_for``). ``key`` is the step's
:class:`StepKey`: the bucket, plus what the base adds per step (the linears'
dtype under a step schedule, and the guidance branches this rank runs), so the
per-step hooks read their own fields as ``key.bucket.<field>``.

Conventions the base fixes:

* The loop ends on its iteration count: a model whose request picks its own
  step count declares it as the loop's cap (``Model.get_loop_caps``), so the
  loop completes after that iteration and the async worker never speculates
  past it.

* The loop-back edges are :attr:`~DenoiseLoopSubmodule.loop_back_names`
  (``latents`` plus whatever ``SOLVER_STATE`` a multi-step scheduler carries);
  every other input is re-injected by the Loop each iteration. The step index
  ``k`` is ``fwd_info.dynamic_loop_iter_counts`` for the loop, refreshed by the
  worker before each batch — no host sync per step.
* A schedule and its device-resident ``sigmas`` / ``timesteps`` are a memo of the hook's
  inputs, so requests with the same settings share one on the device, and losing an
  entry costs a rebuild and changes no output. So are the seeds drawn at ingest and a
  guidance scale's per-step values (keyed by ``(num_steps, knobs)``).
* Per-step per-request scalars (``STEP_SCALARS``: ``sigma``, ``sigma_next``,
  ``timestep``) are sliced out of the schedule ``schedule_for`` returns and
  travel as ``[1]`` device tensors, so a batch of requests at different steps
  (or step counts) shares one forward, and a captured graph reads them from
  staged buffers. Declaring one is enough: the base stages every source the
  table names on the device and slices it per step.
* Rows are batched only at an identical step key, which is also the
  CUDA-graph bucket key. The bucket need not be shape alone: anything that has
  to match for two rows to share a forward belongs in it (the latent grid, text
  length and conditioning layout). Whether guidance is on is the step key's.
* Every span a forward attends over is a segment of a declared array: a model
  whose layers also attend over a sub-span (a refiner over the image tokens
  alone, say) declares an array for it and lists it in :meth:`attention_segments`.
* A step schedule (``step_dtypes``, a model's ``linear_dtype_steps``) is the
  base's alone: the step key carries the step's dtype (:meth:`step_key`), so rows
  batch and capture per dtype, the scheduler batches each dtype apart, and the
  forward reads it as ``key.dtype``.
* Guidance is a declaration (``GUIDANCE``, see ``guidance.py``). A guided step runs
  each branch this rank owns as a row (``key.branches``: ``my_rows`` of the
  ``cfg_group``), gathers the branches' predictions over the group and combines
  them, with the request's scales as step scalars. An unguided step runs branch 0
  alone. A request's rows are its local branches in order. Whether a request is
  guided is decided at intake (``Guidance.active`` of its scales) and is the step
  key's.
"""

from __future__ import annotations

import functools
import threading
from collections import OrderedDict, deque
from collections.abc import Hashable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup
from mstar.engine.state import ArrayStep, Segment, SlotLease, SubmoduleStep
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule, euler_step
from mstar.model.components.diffusion.guidance import Guidance
from mstar.model.components.diffusion.unipc import (
    LAST_SAMPLE,
    MODEL_OUTPUTS,
    SOLVER_STATE,
    UniPCState,
    initial_state,
    unipc_step,
)
from mstar.model.components.distributed.sequence_parallel import my_rows
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.utils.h2d import to_device_nonblocking
from mstar.utils.threads import ThreadPool

if TYPE_CHECKING:
    from mstar.model.components.quant_linear import StepDtypes

# Entries in each memo. A schedule is a few floats per step and a seed entry a latent, and a miss only rebuilds it.
_MEMO_ENTRIES = 256
# What a request's step count is called in its step metadata. A walk whose schedule fixes it has no such key.
NUM_STEPS = "num_inference_steps"


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


@dataclass(frozen=True)
class StepKey:
    """What a step batches and captures on: the model's bucket, plus what the base decides per step. Two keys that
    differ in any field share neither a forward nor a graph.

    ``branches`` is a function of ``guided`` and the rank, so on one rank two keys are equal exactly when the
    rank-invariant fields are, and every rank of a cfg group batches the same rows."""

    bucket: Hashable
    dtype: str | None = None             # the linears' dtype under a step schedule
    guided: bool = False                 # whether the step runs its guidance branches
    branches: tuple[int, ...] = (0,)     # the branches this rank runs, in row order


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
    # between iterations, e.g. UniPC's (:class:`UniPCDenoiseLoopSubmodule`).
    # :meth:`denoise` returns these alongside the new sample and
    # :meth:`seed_loop_back` supplies iteration 0's values.
    SOLVER_STATE: tuple[str, ...] = ()

    # name -> where in the request's schedule this step's value comes from. A
    # scheduler needing another scalar adds an entry here instead of overriding
    # prepare_inputs / preprocess / _run (a guided model's scales come from GUIDANCE).
    STEP_SCALARS: Mapping[str, StepScalar] = {
        "sigma": StepScalar("sigmas"),
        "sigma_next": StepScalar("sigmas", offset=1, capture_fill=0.0),
        "timestep": StepScalar("timesteps", broadcast=False, capture_fill=1000.0),
    }

    # The model's guidance branches and their combine, or None for an unguided model
    # (a distilled checkpoint). See ``guidance.py``.
    GUIDANCE: Guidance | None = None

    # True when :meth:`seed_loop_back` depends on nothing but the bucket key and the
    # request's seed. The base then draws the seeds when the request reaches the
    # worker, on a thread of its own and from the ingest-time request info, and
    # iteration 0 only stages them. Drawn at iteration 0, a CPU randn of a 1024^2
    # latent holds the GPU thread for milliseconds per joining row.
    PREFETCH_SEEDS = False

    def __init__(
        self,
        *,
        loop_name: str,
        max_batch_size: int = 8,
        capture_buckets: Sequence[tuple[str, Hashable]] = (),
        capture_batch_sizes: Sequence[int] = (1, 2, 4, 8),
        replay_walks: Mapping[str, Sequence[str]] | None = None,
        max_uncaptured_layouts: int = 4,
        step_dtypes: StepDtypes | None = None,
        cfg_group: CommGroup | None = None,
    ):
        super().__init__()
        self.loop_name = loop_name
        self._max_batch_size = int(max_batch_size)
        # (graph_walk, bucket_key) pairs to capture; the same key on another walk
        # replays only if ``replay_walks`` says so.
        self.capture_buckets = list(capture_buckets)
        self._captured_keys = frozenset(key for _, key in self.capture_buckets)
        self.capture_batch_sizes = sorted(int(b) for b in capture_batch_sizes)
        self.replay_walks = dict(replay_walks or {})
        # Per-key derived tensors (rotary tables, ...) built on first use. A captured
        # key's are kept for good, since its graphs read them at fixed addresses; the
        # rest are LRU-bounded, as a stream of distinct request shapes (any fps a
        # client asks for, say) would otherwise grow them without limit.
        self._layouts: dict[Hashable, Any] = {}
        self._uncaptured_layouts: OrderedDict[Hashable, Any] = OrderedDict()
        self.max_uncaptured_layouts = int(max_uncaptured_layouts)
        # Draws seeds ahead under PREFETCH_SEEDS; built on first use.
        self._seed_pool: ThreadPool | None = None
        # One draw per request, as ((bucket key, seed), future), oldest first. An entry goes when taken or evicted.
        # Under a lock, since a request's arrival adds to it on the main thread and iteration 0 takes on the GPU's.
        self._seed_draws: deque[tuple[tuple[Hashable, int], Future]] = deque(maxlen=_MEMO_ENTRIES)
        self._seed_lock = threading.Lock()
        # Keyed by `schedule_for`'s inputs: the schedule, and the tensors `prepare_inputs` and the solver read on the
        # device.
        self._schedules = functools.lru_cache(_MEMO_ENTRIES)(self.schedule_for)
        self._staged_schedules = functools.lru_cache(_MEMO_ENTRIES)(self._stage_schedule)
        # The dtype each step runs (``step_key``): None without a schedule.
        self.step_dtypes = step_dtypes
        # The node's classifier-free-guidance group (cfg_size), across which the guidance branches split; trivial
        # on one GPU. Each rank's share of the branches, guided or not, is fixed for the life of the node.
        self.cfg_group = cfg_group or CommGroup.trivial()
        guidance = self.GUIDANCE
        if guidance is not None:
            arguments = {frozenset(branch) for branch in guidance.branches}
            if len(guidance.branches) < 2 or len(arguments) != 1 or not next(iter(arguments)):
                raise ValueError(
                    f"{type(self).__name__}: GUIDANCE needs at least two branches, each mapping the same "
                    f"model_output arguments (got {[sorted(branch) for branch in guidance.branches]})"
                )
        self._branch_shares = {
            False: (0,),
            True: tuple(range(len(guidance.branches)))[my_rows(self.cfg_group, len(guidance.branches))]
            if guidance is not None else (0,),
        }
        # The request's guidance scales, one value per step and on the device, memoized by the step count and the
        # knobs' values, as the schedule is.
        self._staged_scales = functools.lru_cache(_MEMO_ENTRIES)(self._stage_scales)
        # The step scalars the base slices per step: the scheduler's, and a guided model's scales. A scale is an
        # input under its own name, beside the step scalars, the loop-back edges and the branches' request inputs;
        # its values are staged under ``guidance:<name>`` (``_stage_scales``), apart from the schedule's sources. Shared
        # request inputs no branch names are the model's own: prepare_inputs checks those per row.
        scales = guidance.scales if guidance is not None else ()
        branch_inputs = {edge for branch in guidance.branches for edge in branch.values()} if guidance else set()
        shadowed = set(scales) & {*self.STEP_SCALARS, *self.loop_back_names, *branch_inputs}
        if shadowed:
            raise ValueError(
                f"{type(self).__name__}: guidance scale(s) {sorted(shadowed)} share a name with a step scalar, "
                f"loop-back input or guidance branch input"
            )
        self.step_scalars: Mapping[str, StepScalar] = {
            **self.STEP_SCALARS, **{name: StepScalar(f"guidance:{name}", broadcast=False) for name in scales},
        }
        # What the scaffold stages itself, which a model's request_inputs must not name
        self._staged_names = frozenset({*self.step_scalars, *self.loop_back_names})

    @property
    def loop_back_names(self) -> tuple[str, ...]:
        """The node's loop-back edges, the sample first."""
        return (self.LATENTS, *self.SOLVER_STATE)

    # ------------------------------------------------------------------ hooks
    def bucket_key_for(self, fwd_info: CurrentForwardPassInfo) -> Hashable:
        """What has to match for two rows to share a forward (and a captured graph)."""
        raise NotImplementedError

    def schedule_settings(self, fwd_info: CurrentForwardPassInfo) -> Hashable:
        """What the request sets of its schedule beyond the bucket key: by default its step count, None where the walk
        fixes it."""
        return fwd_info.step_metadata.get(NUM_STEPS)

    def schedule_for(self, walk: str, settings: Hashable, bucket_key: Hashable) -> FlowMatchSchedule:
        """The schedule of ``walk`` at ``bucket_key`` under the request's ``settings`` (`schedule_settings`). A pure
        function of the three, since the base memoizes it across requests."""
        raise NotImplementedError

    def seed_latents(self, bucket_key: Hashable, generator: torch.Generator) -> torch.Tensor:
        """Initial noise for one request, ``[L, C]`` on the CPU generator's device."""
        raise NotImplementedError

    def stage_seed(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Seed tensor on ``device`` without stalling the GPU thread, where this
        runs: a pageable copy would hold up the pre-plan and the overlap with the
        previous step's postprocess (:func:`mstar.utils.h2d.to_device_nonblocking`)."""
        return to_device_nonblocking(tensor, device)

    def seed_loop_back(self, bucket_key: Hashable, generator: torch.Generator) -> dict[str, torch.Tensor]:
        """Iteration 0's value for every loop-back edge, without a batch dim. A pure function of the two, since the base
        draws it ahead of the request's walk.

        Default: seeded noise for the sample and nothing else. A model with
        ``SOLVER_STATE`` adds that scheduler's initial state here."""
        return {self.LATENTS: self.seed_latents(bucket_key, generator)}

    def initial_loop_back(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, bucket_key: Hashable,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Iteration 0's loop-back values, with the step's inputs in view.

        Default: :meth:`seed_loop_back`, i.e. a loop that starts from noise, taken from
        the draw made at ingest when there is one for this bucket key. A loop that
        starts from given latents instead (a refinement stage, image-to-image) overrides
        this and reads them from ``inputs``, the edges the walk routed to this node."""
        del inputs
        drawn = self._take_drawn_seeds(bucket_key, fwd_info.random_seed)
        if drawn is not None:
            return drawn
        return self.seed_loop_back(bucket_key, generator)

    def on_request_added(self, request_id: int, request_info: CurrentForwardPassInfo) -> None:
        if not self.PREFETCH_SEEDS:
            return
        bucket_key, seed = self.bucket_key_for(request_info), request_info.random_seed
        if self._seed_pool is None:
            # a CPU randn is single-threaded and releases the GIL
            self._seed_pool = ThreadPool(max_workers=2, thread_name_prefix="mstar-seed")
        draw = self._seed_pool.submit(self._draw_seeds, bucket_key, seed)
        with self._seed_lock:
            self._seed_draws.append(((bucket_key, seed), draw))

    def _draw_seeds(self, bucket_key: Hashable, seed: int) -> dict[str, torch.Tensor]:
        """:meth:`seed_loop_back` as iteration 0 would draw it, made contiguous here so
        staging it is one memcpy. Host-only: this runs beside the GPU thread."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return {name: tensor.contiguous() for name, tensor in self.seed_loop_back(bucket_key, generator).items()}

    def _take_drawn_seeds(self, bucket_key: Hashable, seed: int) -> dict[str, torch.Tensor] | None:
        """A draw made at ingest for these inputs, once. None when there is none, and iteration 0 then draws inline."""
        with self._seed_lock:
            at = next((i for i, (key, _) in enumerate(self._seed_draws) if key == (bucket_key, seed)), None)
            if at is None:
                return None
            draw = self._seed_draws[at][1]
            del self._seed_draws[at]
        # usually done long before; waiting releases the GIL and syncs no device
        return draw.result()

    def request_inputs(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, key: StepKey,
    ) -> dict[str, torch.Tensor]:
        """The request's conditioning tensors for this step, without a batch dim. A guided model returns every
        branch input its ``key.branches`` read (the contract is on ``Guidance.branches``); the base maps them onto
        each branch's row. No name may be a step scalar's or a loop-back edge's (``prepare_inputs`` refuses it)."""
        raise NotImplementedError

    def num_tokens(self, key: StepKey) -> int:
        """Tokens one guidance branch of one request contributes to the step (sizes the attention segment / buckets)."""
        raise NotImplementedError

    def capture_request_inputs(self, key: StepKey, device: torch.device) -> dict[str, torch.Tensor]:
        """Placeholder loop-back tensors + conditioning for one row of this bucket (capture time)."""
        raise NotImplementedError

    def model_output(
        self, engine_inputs: ModelInputsFromEngine, key: StepKey, latents: torch.Tensor,
        timestep: torch.Tensor, **cond: torch.Tensor,
    ) -> torch.Tensor:
        """The model's prediction for a stacked batch of rows, ``[R, ...]`` over the rows' samples (a velocity for
        the Euler step, what the solver takes for UniPC). Under guidance the rows are each request's local branches
        (``key.branches``), with ``latents``, ``timestep`` and the shared inputs repeated per branch and each branch's
        inputs mapped onto its row."""
        raise NotImplementedError

    def denoise(
        self,
        engine_inputs: ModelInputsFromEngine,
        key: StepKey,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        **cond: torch.Tensor,
    ) -> torch.Tensor | Mapping[str, torch.Tensor]:
        """One scheduler step for a stacked batch: ``[B, L, C] -> [B, L, C]``. Default: the Euler step of the
        (guided) :meth:`model_output`.

        A model that steps itself (several streams, a model-specific update) overrides this; one with
        ``SOLVER_STATE`` returns ``{name: stacked tensor}`` covering every loop-back edge instead of the new sample
        alone."""
        return euler_step(latents, self.guided_output(engine_inputs, key, latents, timestep, **cond), sigma, sigma_next)

    def build_layout(self, key: StepKey, device: torch.device) -> Any:
        """Derived per-key state (e.g. rotary tables); cached by :meth:`layout` per bucket and branch share."""
        return None

    def attention_segments(self, key: StepKey) -> Mapping[str, Sequence[tuple[str, int]]]:
        """``(label, span)`` for each segment one request brings to each array its layers attend through, in row order
        (each local branch's segments, ``key.branches`` in order). Whole on every rank, since the engine cuts each
        read's share. Default: none, for a loop with no declared arrays."""
        return {}

    def guided_output(
        self, engine_inputs: ModelInputsFromEngine, key: StepKey, latents: torch.Tensor, timestep: torch.Tensor,
        **inputs: torch.Tensor,
    ) -> torch.Tensor:
        """:meth:`model_output` under the model's ``GUIDANCE``: this rank's branches as rows, request-major, then
        the predictions gathered over the cfg group and combined. Unguided, branch 0 alone."""
        guidance = self.GUIDANCE
        if guidance is None:
            return self.model_output(engine_inputs, key, latents, timestep, **inputs)
        values = {name: inputs.pop(name) for name in guidance.scales}
        named = {edge for branch in guidance.branches for edge in branch.values()}
        shared = {name: tensor for name, tensor in inputs.items() if name not in named}
        rows = [
            {**shared, **{arg: inputs[edge] for arg, edge in guidance.branches[b].items()}} for b in key.branches
        ]
        local = len(rows)
        pred = self.model_output(
            engine_inputs, key, _branch_rows([latents] * local), _branch_rows([timestep] * local),
            **{name: _branch_rows([row[name] for row in rows]) for name in rows[0]},
        )
        if not key.guided:
            return pred
        # [B * local, ...] -> [B, local, ...] -> every branch, gathered over the group in branch order
        pred = self.cfg_group.all_gather(pred.reshape(-1, local, *pred.shape[1:]).contiguous(), dim=1)
        rank = [1] * (pred.ndim - 2)
        return guidance.combine(pred.unbind(1), {name: s.reshape(-1, *rank) for name, s in values.items()})

    # --------------------------------------------------------------- helpers
    def layout(self, key: StepKey, device: torch.device) -> Any:
        """:meth:`build_layout` for ``key``, cached by its bucket and branch share (the step's dtype changes no
        layout)."""
        cache_key = (key.bucket, key.branches)
        entry = self._layouts.get(cache_key)
        if entry is not None:
            return entry
        if key.bucket in self._captured_keys:
            entry = self._layouts[cache_key] = self.build_layout(key, device)
            return entry
        cache = self._uncaptured_layouts
        entry = cache.get(cache_key)
        if entry is None:
            entry = cache[cache_key] = self.build_layout(key, device)
            while len(cache) > self.max_uncaptured_layouts:
                cache.popitem(last=False)
        else:
            cache.move_to_end(cache_key)
        return entry

    def step_index(self, fwd_info: CurrentForwardPassInfo) -> int:
        """The denoise iteration this step runs, from the engine's loop counter.

        A value carried on a loop-back edge would be the previous step's device
        output, and reading it on the host waits for that step to finish, which
        stalls the speculative preparation of this one. A missing counter raises:
        read as 0, it would reseed from noise every iteration.
        """
        return int(fwd_info.dynamic_loop_iter_counts[self.loop_name])

    def step_key(self, fwd_info: CurrentForwardPassInfo, bucket_key: Hashable, step: int | None = None) -> StepKey:
        """What step ``step`` (default: ``fwd_info``'s) batches and captures on: ``bucket_key``, with the step's dtype
        under a step schedule and whether the request is guided. ``prepare_inputs`` and ``cg_key_info`` both take it
        from here, so a step's rows and its graph agree."""
        guidance = self.GUIDANCE
        guided = guidance is not None and guidance.active(fwd_info.step_metadata)
        if self.step_dtypes is None:
            return self._key(bucket_key, None, guided)
        k = self.step_index(fwd_info) if step is None else step
        return self._key(bucket_key, self.step_dtypes.at(k), guided)

    def _key(self, bucket_key: Hashable, dtype: str | None, guided: bool) -> StepKey:
        return StepKey(bucket_key, dtype, guided, self._branch_shares[guided])

    def _knobs(self, fwd_info: CurrentForwardPassInfo) -> dict[str, float]:
        """The request's guidance scales as it set them."""
        return {name: float(fwd_info.step_metadata[name]) for name in self.GUIDANCE.scales}

    # --------------------------------------------------------- engine contract
    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> NodeInputs | None:
        k = self.step_index(fwd_info)
        device = self.get_device()
        # A request whose walks run this node in successive stages (a draft loop, then a
        # refinement loop) gets a fresh shape and schedule for each, through the key.
        bucket_key = self.bucket_key_for(fwd_info)
        schedule_key = self._schedule_key(fwd_info, bucket_key)
        num_steps = self._schedules(*schedule_key).num_steps
        if k >= num_steps:
            raise RuntimeError(
                f"{type(self).__name__}: iteration {k} is past request {fwd_info.request_id}'s "
                f"{num_steps} steps; declare the step count as the loop's cap (Model.get_loop_caps)"
            )
        staged = self._staged_schedules(*schedule_key)
        if self.GUIDANCE is not None:
            staged = {**staged, **self._staged_scales(num_steps, tuple(self._knobs(fwd_info).items()))}
        key = self.step_key(fwd_info, bucket_key)
        conditioning = self.request_inputs(fwd_info, inputs, key)
        shadowed = conditioning.keys() & self._staged_names
        if shadowed:
            raise ValueError(
                f"{type(self).__name__}.request_inputs returned {sorted(shadowed)}, the name of a step scalar or "
                f"loop-back input"
            )
        tensors = {
            **self._carried_inputs(fwd_info, inputs, bucket_key, k, device),
            **{
                name: staged[scalar.source][k + scalar.offset:k + scalar.offset + 1]
                for name, scalar in self.step_scalars.items()
            },
            **conditioning,
        }
        return NodeInputs(tensor_inputs=tensors, input_seq_len=self.num_tokens(key), resource_step_info=key)

    def _schedule_key(self, fwd_info: CurrentForwardPassInfo, bucket_key: Hashable) -> tuple[str, Hashable, Hashable]:
        return fwd_info.graph_walk, self.schedule_settings(fwd_info), bucket_key

    def schedule_of(self, fwd_info: CurrentForwardPassInfo, bucket_key: Hashable) -> FlowMatchSchedule:
        """The memoized schedule of the walk ``fwd_info`` is on, at its ``bucket_key``."""
        return self._schedules(*self._schedule_key(fwd_info, bucket_key))

    def _stage_schedule(self, walk: str, settings: Hashable, bucket_key: Hashable) -> dict[str, torch.Tensor]:
        return self._step_scalar_sources(self._schedules(walk, settings, bucket_key), self.get_device())

    def _stage_scales(self, num_steps: int, knobs: tuple[tuple[str, float], ...]) -> dict[str, torch.Tensor]:
        """Each of the request's guidance scales, one value per step, on the device under its step scalar's source
        name."""
        return {
            self.step_scalars[name].source:
                to_device_nonblocking(torch.full((num_steps,), value, dtype=torch.float32), self.get_device())
            for name, value in knobs
        }

    def _step_scalar_sources(
        self, schedule: FlowMatchSchedule, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """The schedule tensors ``STEP_SCALARS`` slices, device-resident, by source
        name: the memo ``prepare_inputs`` reads.

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
            staged.setdefault(scalar.source, to_device_nonblocking(tensor, device))
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
            seeds = self.initial_loop_back(fwd_info, inputs, bucket_key, generator)
            return {name: self.stage_seed(tensor, device) for name, tensor in seeds.items()}
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
        for name, scalar in self.step_scalars.items():
            # [B, 1] -> [B, 1, 1] to broadcast over [B, L, C], or a plain [B]
            stacked[name] = stacked[name].view(-1, 1, 1) if scalar.broadcast else stacked[name].view(-1)
        stacked["step_key"] = inputs[0].resource_step_info
        return stacked

    def _run(
        self, engine_inputs: ModelInputsFromEngine, kwargs: dict,
    ) -> dict[str, torch.Tensor]:
        """One stacked step, as ``{loop-back edge: [B, ...]}``."""
        key = kwargs.pop("step_key")
        carried = {name: kwargs.pop(name) for name in self.loop_back_names}
        scalars = {name: kwargs.pop(name) for name in self.step_scalars}
        out = self.denoise(engine_inputs, key, **carried, **scalars, **kwargs)
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
        del kwargs
        # Padding rows (a captured bucket's dummy requests) carry no step metadata; the
        # real rows decide the bucket.
        keys = [
            self.step_key(info, self.bucket_key_for(info), self._step_of(rid, per_request_input_metadata))
            for rid, info in per_request_info.items() if info.step_metadata
        ]
        return self._uniform_key(keys) if keys else None

    def _step_of(self, rid, per_request_input_metadata) -> int | None:
        """The row's step from its input metadata, which the scheduler advances for a speculated step whose request
        info still holds the step in flight's; None when it has none."""
        if self.step_dtypes is None or not per_request_input_metadata:
            return None
        metadata = per_request_input_metadata.get(rid)
        loops = metadata.dynamic_loop_iter_counts if metadata is not None else {}
        return int(loops[self.loop_name]) if self.loop_name in loops else None

    def split_batches_by_capture_key(self, graph_walk: str) -> bool:
        # a step mixing dtypes, or guided and unguided requests, matches no capture and would run a row at a time
        del graph_walk
        return self.step_dtypes is not None or self.GUIDANCE is not None

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None, piecewise_leases=None, **kwargs,
    ) -> SubmoduleStep | None:
        """One array step per array :meth:`attention_segments` names. An override that declares a step to a
        resource that journals (a paged KV, which every cfg rank holds a copy of) must not read ``key.branches``,
        the rank's own share, or the ranks would take different pages."""
        # every row (padding rows included) carries its bucket key, so the spans are
        # identical between a bucket's capture and its replays
        rows = [
            (rid, self.attention_segments(inp.resource_step_info))
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        if not rows[0][1]:
            return None
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label=label, span=span)
                for rid, arrays in rows for spans in arrays.values() for label, span in spans
            ],
            state={
                name: ArrayStep(segments=tuple(
                    Segment(request_id=rid, label=label, span=span)
                    for rid, arrays in rows for label, span in arrays[name]
                ), fixed_layout=True)
                for name in rows[0][1]
            },
            cg_key_info=self._uniform_key(inp.resource_step_info for inp in inputs),
        )

    def get_accelerator_graph_configs(self, device: torch.device, tp_world_size: int = 1):
        from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig

        configs = []
        dtypes = [None] if self.step_dtypes is None else list(self.step_dtypes.dtypes)
        guided = [False] if self.GUIDANCE is None else [False, True]
        for walk, bucket_key in self.capture_buckets:
            for step_key in (self._key(bucket_key, dtype, on) for dtype in dtypes for on in guided):
                tensors = dict(self.capture_request_inputs(step_key, device))
                for name, scalar in self.step_scalars.items():
                    tensors[name] = torch.full(
                        (1,), scalar.capture_fill, dtype=torch.float32, device=device,
                    )
                single = NodeInputs(
                    tensor_inputs=tensors, input_seq_len=self.num_tokens(step_key), resource_step_info=step_key,
                )
                configs.append(BatchedAcceleratorGraphConfig(
                    capture_graph_walk=walk,
                    replay_graph_walks=[walk, *self.replay_walks.get(walk, ())],
                    single_request_inputs=single,
                    additional_key_info=step_key,
                    # the model compiles its own transformer region; the graph records those kernels
                    compile=False,
                    capture_batch_sizes=list(self.capture_batch_sizes),
                    # a row's input_seq_len is one branch's tokens; the step plans every local branch's
                    total_tokens_multiplier=len(step_key.branches),
                    # uncaptured sizes / shapes run the eager batched path, so don't cap the batch
                    caps_eager_batch_size=False,
                    # every static input is row-leading ([tokens, features] or a per-batch scalar); a text
                    # embedding's hidden size can equal an edit bucket's token count (7680 for klein), which
                    # would otherwise be mistaken for the token axis
                    input_seq_dims={key: 0 for key in tensors},
                ))
        return configs


class UniPCDenoiseLoopSubmodule(DenoiseLoopSubmodule):
    """A denoise loop stepped by the scaffold's UniPC (``components/diffusion/unipc.py``).

    The subclass predicts (:meth:`model_output`); the base carries the solver state
    in ``SOLVER_STATE`` and steps each row from its own tables and the loop counter,
    so rows at different steps share one forward. The solver is per row because the
    order ramp is, and because it holds a host sigma, this path has no captured graph.
    """

    SOLVER_STATE = SOLVER_STATE
    STEP_SCALARS = {"timestep": StepScalar("timesteps", broadcast=False, capture_fill=1000.0)}

    def seed_loop_back(self, bucket_key: Hashable, generator: torch.Generator) -> dict[str, torch.Tensor]:
        latents = self.seed_latents(bucket_key, generator)
        return {self.LATENTS: latents, **initial_state(latents)}

    def _step_scalar_sources(
        self, schedule: FlowMatchSchedule, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        # the solver reads its sigmas from the table, by the row's step index
        return {
            **super()._step_scalar_sources(schedule, device),
            "sigmas": to_device_nonblocking(schedule.sigmas, device),
        }

    def denoise(
        self,
        engine_inputs: ModelInputsFromEngine,
        key: StepKey,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        unipc_model_outputs: torch.Tensor,
        unipc_last_sample: torch.Tensor,
        **cond: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self.guided_output(engine_inputs, key, latents, timestep, **cond)
        steps = []
        for i, rid in enumerate(engine_inputs.request_ids):
            info = engine_inputs.per_request_info[rid]
            schedule_key = self._schedule_key(info, key.bucket)
            schedule = self._schedules(*schedule_key)
            steps.append(unipc_step(
                output[i], latents[i],
                UniPCState(model_outputs=unipc_model_outputs[i], last_sample=unipc_last_sample[i]),
                schedule.sigmas, self._staged_schedules(*schedule_key)["sigmas"], self.step_index(info),
                schedule.num_steps,
            ))
        return {
            self.LATENTS: torch.stack([step.sample for step in steps]),
            MODEL_OUTPUTS: torch.stack([step.model_outputs for step in steps]),
            LAST_SAMPLE: torch.stack([step.last_sample for step in steps]),
        }


def _branch_rows(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Each branch's ``[B, ...]`` stacked request-major, ``[B * branches, ...]``: a request's rows are adjacent, in
    branch order, which is the order its attention segments are declared in."""
    return tensors[0] if len(tensors) == 1 else torch.stack(tensors, dim=1).flatten(0, 1)


# The default names, for callers that only need those (the module's own graph
# wiring, tests). A subclass overrides the class attributes, not these.
LATENTS = DenoiseLoopSubmodule.LATENTS
MAIN_SPAN = DenoiseLoopSubmodule.MAIN_SPAN
