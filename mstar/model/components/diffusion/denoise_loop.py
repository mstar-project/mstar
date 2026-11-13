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
                                                  denoise(engine_inputs, key, latents, timestep,
                                                          sigma, sigma_next, **cond) -> [B, L, C]
  the step's arrays and their segments            capture_request_inputs(key, device) -> {name: [..]}
                                                  attention_segments(key) -> {array: ((label, span), ...)}

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
  entry costs a rebuild and changes no output. So are the seeds drawn at ingest.
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
* Every span a forward attends over is a segment of a declared array: a model
  whose layers also attend over a sub-span (a refiner over the image tokens
  alone, say) declares an array for it and lists it in :meth:`attention_segments`.
* A captured graph is keyed by :meth:`graph_key` of the bucket key, the key itself
  by default. A model whose ``denoise`` reads each request's layout off its step
  inputs (``LAYOUT_FROM_INPUTS``) maps every key one graph can replay to one key,
  and its graphs then plan each replay's attention spans.
* A step schedule (``step_dtypes``, a model's ``linear_dtype_steps``) is the
  base's alone: a step's key is then a :class:`StepBucket` (:meth:`step_key`),
  so rows batch and capture per dtype, the scheduler batches each dtype apart,
  and ``denoise`` reads the step's dtype off :attr:`step_dtype` while the hooks
  keep seeing the model's own key.
* Step caching (``step_cache``, off by default) is the base's too: a step's key
  then also says whether it reuses the cached residual (:class:`StepBucket`), so
  rows batch only with rows making the same decision, and ``denoise`` gets the
  step's :meth:`block_cache` to route its block stack through
  (``components/diffusion/step_cache.py``).
"""

from __future__ import annotations

import functools
import threading
from collections import OrderedDict, deque
from collections.abc import Hashable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.state import ArrayStep, Segment, SlotLease, SubmoduleStep
from mstar.model.components.diffusion.flow_match import FlowMatchSchedule
from mstar.model.components.diffusion.unipc import (
    LAST_SAMPLE,
    MODEL_OUTPUTS,
    SOLVER_STATE,
    UniPCState,
    initial_state,
    unipc_step,
)
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.utils.h2d import to_device_nonblocking
from mstar.utils.threads import ThreadPool

if TYPE_CHECKING:
    from mstar.model.components.diffusion.step_cache import BlockCache, StepCachePolicy
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
class StepBucket:
    """A step's key under a step schedule: the request's bucket, the dtype the step's linears run in (None without a
    dtype schedule) and whether it reuses the step cache's residual. Two steps differing in either share neither a
    forward nor a graph."""

    bucket: Hashable
    dtype: str | None
    reuse: bool = False


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
    # scheduler needing another scalar (a per-step guidance scale, say) adds an
    # entry here instead of overriding prepare_inputs / preprocess / _run.
    STEP_SCALARS: Mapping[str, StepScalar] = {
        "sigma": StepScalar("sigmas"),
        "sigma_next": StepScalar("sigmas", offset=1, capture_fill=0.0),
        "timestep": StepScalar("timesteps", broadcast=False, capture_fill=1000.0),
    }

    # True when :meth:`seed_loop_back` depends on nothing but the bucket key and the
    # request's seed. The base then draws the seeds when the request reaches the
    # worker, on a thread of its own and from the ingest-time request info, and
    # iteration 0 only stages them. Drawn at iteration 0, a CPU randn of a 1024^2
    # latent holds the GPU thread for milliseconds per joining row.
    PREFETCH_SEEDS = False

    # True when ``denoise`` reads everything that varies between the bucket keys one graph
    # replays (positions, prompt rows, ...) off its step inputs (``request_inputs``), so a
    # replay reads it from the graph's staged buffers rather than from the capture's key.
    # Only then may :meth:`graph_key` be coarser than the bucket key.
    LAYOUT_FROM_INPUTS = False

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
        step_cache: StepCachePolicy | None = None,
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
        # The dtype each step runs (``step_key``), and the one the step in ``denoise`` runs: None without a schedule.
        self.step_dtypes = step_dtypes
        self.step_dtype: str | None = None
        # Which steps reuse the last computed block residual (off: None). Cached residuals are per-request state the
        # engine owns, at no fixed address, so no captured graph could read them.
        if step_cache is not None and self.capture_buckets:
            raise ValueError(f"{type(self).__name__}: step_cache does not support captured buckets yet")
        self.step_cache = step_cache
        self._step_reuse = False

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

    def graph_key(self, bucket_key: Hashable) -> Hashable:
        """The captured graph a step of ``bucket_key`` replays: the key itself by default.
        Coarser only under ``LAYOUT_FROM_INPUTS``; ``capture_buckets`` then lists graph keys."""
        return bucket_key

    def build_layout(self, bucket_key: Hashable, device: torch.device) -> Any:
        """Derived per-key state (e.g. rotary tables); cached by :meth:`layout`."""
        return None

    def attention_segments(self, bucket_key: Hashable) -> Mapping[str, Sequence[tuple[str, int]]]:
        """``(label, span)`` for each segment one request brings to each array its layers attend through, or
        ``(label, span, layout)`` for a segment whose attention reads how its tokens are arranged (``Segment.layout``).
        Whole on every rank, since the engine cuts each read's share. Default: none, for a loop with no declared
        arrays."""
        return {}

    # --------------------------------------------------------------- helpers
    def layout(self, bucket_key: Hashable, device: torch.device) -> Any:
        entry = self._layouts.get(bucket_key)
        if entry is not None:
            return entry
        if bucket_key in self._captured_keys:
            entry = self._layouts[bucket_key] = self.build_layout(bucket_key, device)
            return entry
        cache = self._uncaptured_layouts
        entry = cache.get(bucket_key)
        if entry is None:
            entry = cache[bucket_key] = self.build_layout(bucket_key, device)
            while len(cache) > self.max_uncaptured_layouts:
                cache.popitem(last=False)
        else:
            cache.move_to_end(bucket_key)
        return entry

    def step_index(self, fwd_info: CurrentForwardPassInfo) -> int:
        """The denoise iteration this step runs, from the engine's loop counter.

        A value carried on a loop-back edge would be the previous step's device
        output, and reading it on the host waits for that step to finish, which
        stalls the speculative preparation of this one. A missing counter raises:
        read as 0, it would reseed from noise every iteration.
        """
        return int(fwd_info.dynamic_loop_iter_counts[self.loop_name])

    def step_key(self, fwd_info: CurrentForwardPassInfo, bucket_key: Hashable, step: int | None = None) -> Hashable:
        """What step ``step`` (default: ``fwd_info``'s) batches and captures on: ``bucket_key``, with the step's dtype
        under a step schedule and its reuse decision under a step cache. ``prepare_inputs`` and ``cg_key_info`` both
        take it from here, so a step's rows and its graph agree."""
        if self.step_dtypes is None and self.step_cache is None:
            return bucket_key
        k = self.step_index(fwd_info) if step is None else step
        dtype = None if self.step_dtypes is None else self.step_dtypes.at(k)
        reuse = self.step_cache is not None and self.step_cache.reuses(
            k, self.schedule_of(fwd_info, bucket_key).num_steps,
        )
        return StepBucket(bucket_key, dtype, reuse)

    def _bucket_of(self, step_key: Hashable) -> Hashable:
        return step_key.bucket if isinstance(step_key, StepBucket) else step_key

    def _graph_step_key(self, step_key: Hashable) -> Hashable:
        """``step_key`` with its bucket replaced by that bucket's :meth:`graph_key`."""
        if isinstance(step_key, StepBucket):
            return replace(step_key, bucket=self.graph_key(step_key.bucket))
        return self.graph_key(step_key)

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
        tensors = {
            **self._carried_inputs(fwd_info, inputs, bucket_key, k, device),
            **{
                name: staged[scalar.source][k + scalar.offset:k + scalar.offset + 1]
                for name, scalar in self.STEP_SCALARS.items()
            },
            **self.request_inputs(fwd_info, inputs, bucket_key),
        }
        return NodeInputs(
            tensor_inputs=tensors, input_seq_len=self.num_tokens(bucket_key),
            resource_step_info=self.step_key(fwd_info, bucket_key),
        )

    def _schedule_key(self, fwd_info: CurrentForwardPassInfo, bucket_key: Hashable) -> tuple[str, Hashable, Hashable]:
        return fwd_info.graph_walk, self.schedule_settings(fwd_info), bucket_key

    def schedule_of(self, fwd_info: CurrentForwardPassInfo, bucket_key: Hashable) -> FlowMatchSchedule:
        """The memoized schedule of the walk ``fwd_info`` is on, at its ``bucket_key``."""
        return self._schedules(*self._schedule_key(fwd_info, bucket_key))

    def _stage_schedule(self, walk: str, settings: Hashable, bucket_key: Hashable) -> dict[str, torch.Tensor]:
        return self._step_scalar_sources(self._schedules(walk, settings, bucket_key), self.get_device())

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
        if isinstance(bucket_key, StepBucket):
            self.step_dtype, self._step_reuse, bucket_key = bucket_key.dtype, bucket_key.reuse, bucket_key.bucket
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
            self._graph_step_key(
                self.step_key(info, self.bucket_key_for(info), self._step_of(rid, per_request_input_metadata)))
            for rid, info in per_request_info.items() if info.step_metadata
        ]
        return self._uniform_key(keys) if keys else None

    def _step_of(self, rid, per_request_input_metadata) -> int | None:
        """The row's step from its input metadata, which the scheduler advances for a speculated step whose request
        info still holds the step in flight's; None when it has none."""
        if (self.step_dtypes is None and self.step_cache is None) or not per_request_input_metadata:
            return None
        metadata = per_request_input_metadata.get(rid)
        loops = metadata.dynamic_loop_iter_counts if metadata is not None else {}
        return int(loops[self.loop_name]) if self.loop_name in loops else None

    def split_batches_by_capture_key(self, graph_walk: str) -> bool:
        # a step mixing dtypes matches no capture and would run a row at a time; one mixing reuse decisions cannot run
        del graph_walk
        return self.step_dtypes is not None or self.step_cache is not None

    def block_cache(self, engine_inputs: ModelInputsFromEngine) -> BlockCache | None:
        """The step's view of its rows' cached block residuals, for ``denoise`` to route its block stack through; None
        without a step cache. Call it from ``denoise``, where the step's decision is set."""
        if self.step_cache is None:
            return None
        from mstar.model.components.diffusion.step_cache import BlockCache

        infos = [engine_inputs.per_request_info[rid] for rid in engine_inputs.request_ids]
        final = [
            self.step_index(info) + 1 >= self.schedule_of(info, self.bucket_key_for(info)).num_steps
            for info in infos
        ] if self.step_cache.records else [False] * len(infos)
        return BlockCache(
            reuse=self._step_reuse,
            _states=[self.request_state(rid) for rid in engine_inputs.request_ids],
            _final=final, _records=self.step_cache.records,
        )

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None, piecewise_leases=None, per_request_info=None, **kwargs,
    ) -> SubmoduleStep | None:
        # every row (padding rows included) carries its bucket key, so the spans are
        # identical between a bucket's capture and its replays
        rows = [
            (rid, self.attention_segments(self._row_bucket(rid, inp, per_request_info)))
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        if not rows[0][1]:
            return None
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label=label, span=span)
                for rid, arrays in rows for spans in _by_label(arrays).values() for label, span in spans
            ],
            state={
                name: ArrayStep(segments=tuple(
                    Segment(rid, *entry) for rid, arrays in rows for entry in arrays[name]
                ), fixed_layout=not self.LAYOUT_FROM_INPUTS)
                for name in rows[0][1]
            },
            cg_key_info=self._uniform_key(self._graph_step_key(inp.resource_step_info) for inp in inputs),
        )

    def _row_bucket(self, rid, inp: NodeInputs, per_request_info) -> Hashable:
        """A step row's bucket key. A pre-plan declares a graph's step over its template rows, whose
        spans are the request's only where the graph's key is the request's: under
        ``LAYOUT_FROM_INPUTS`` a real row takes its key from its request info."""
        info = per_request_info.get(rid) if self.LAYOUT_FROM_INPUTS and per_request_info else None
        if info is not None and info.step_metadata:
            return self.bucket_key_for(info)
        return self._bucket_of(inp.resource_step_info)

    def get_accelerator_graph_configs(self, device: torch.device, tp_world_size: int = 1):
        from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig

        configs = []
        for walk, bucket_key in self.capture_buckets:
            if self.graph_key(bucket_key) != bucket_key:
                # the capture runs this key's own layout, so it has to be the graph's key
                raise ValueError(f"{type(self).__name__}: capture bucket {bucket_key!r} is not its own graph key "
                                 f"{self.graph_key(bucket_key)!r}")
            step_keys = (
                [bucket_key] if self.step_dtypes is None
                else [StepBucket(bucket_key, dtype) for dtype in self.step_dtypes.dtypes]
            )
            for step_key in step_keys:
                tensors = dict(self.capture_request_inputs(bucket_key, device))
                for name, scalar in self.STEP_SCALARS.items():
                    tensors[name] = torch.full(
                        (1,), scalar.capture_fill, dtype=torch.float32, device=device,
                    )
                single = NodeInputs(
                    tensor_inputs=tensors, input_seq_len=self.num_tokens(bucket_key), resource_step_info=step_key,
                )
                configs.append(BatchedAcceleratorGraphConfig(
                    capture_graph_walk=walk,
                    replay_graph_walks=[walk, *self.replay_walks.get(walk, ())],
                    single_request_inputs=single,
                    additional_key_info=self._graph_step_key(step_key),
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


class UniPCDenoiseLoopSubmodule(DenoiseLoopSubmodule):
    """A denoise loop stepped by the scaffold's UniPC (``components/diffusion/unipc.py``).

    The subclass predicts (:meth:`model_output`); the base carries the solver state
    in ``SOLVER_STATE`` and steps each row from its own tables and the loop counter,
    so rows at different steps share one forward. The solver is per row because the
    order ramp is, and because it holds a host sigma, this path has no captured graph.
    """

    SOLVER_STATE = SOLVER_STATE
    STEP_SCALARS = {"timestep": StepScalar("timesteps", broadcast=False, capture_fill=1000.0)}

    def model_output(
        self, engine_inputs: ModelInputsFromEngine, bucket_key: Hashable, latents: torch.Tensor,
        timestep: torch.Tensor, **cond: torch.Tensor,
    ) -> torch.Tensor:
        """What the solver takes for a stacked batch, ``[B, ...]`` over the rows'
        samples; ``latents`` and ``cond`` are stacked over the rows' tensors."""
        raise NotImplementedError

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
        bucket_key: Hashable,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        unipc_model_outputs: torch.Tensor,
        unipc_last_sample: torch.Tensor,
        **cond: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self.model_output(engine_inputs, bucket_key, latents, timestep, **cond)
        steps = []
        for i, rid in enumerate(engine_inputs.request_ids):
            info = engine_inputs.per_request_info[rid]
            schedule_key = self._schedule_key(info, bucket_key)
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


# The default names, for callers that only need those (the module's own graph
# wiring, tests). A subclass overrides the class attributes, not these.
LATENTS = DenoiseLoopSubmodule.LATENTS
MAIN_SPAN = DenoiseLoopSubmodule.MAIN_SPAN


def _by_label(arrays: Mapping[str, Sequence[tuple[str, int]]]) -> dict[str, list[tuple[str, int]]]:
    """One request's ``(label, span)`` segments, each label's once: the node's step (which the sequence shard and the
    LoRA pools plan from) holds a request's tokens once, however many arrays its layers attend them through. Two arrays
    of one label lay out the same tokens, so they must give it the same segments."""
    by_label: dict[str, list[tuple[str, int]]] = {}
    for name, spans in arrays.items():
        mine: dict[str, list[tuple[str, int]]] = {}
        for label, span, *_ in spans:
            mine.setdefault(label, []).append((label, span))
        for label, segments in mine.items():
            if by_label.setdefault(label, segments) != segments:
                raise ValueError(f"array {name!r} lays out label {label!r} as {segments}, and another array as "
                                 f"{by_label[label]}: arrays of one label hold the same tokens")
    return by_label
