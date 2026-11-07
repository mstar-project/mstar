import logging
from dataclasses import asdict

import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import CachedPrefix, EngineResourceInfo, Resource
from mstar.engine.resources.resident import CapturedResidentStep, ResidentPlan, ResidentPlanner, ResidentWrites
from mstar.engine.resources.resident_hooks import (
    ANY_REPEATS,
    ResidentBinding,
    no_resident_effect,
)
from mstar.engine.resources.sampler.config import (
    SAMPLED_LOGPROB,
    STOP_ODDS,
    SamplerSpec,
    SamplerStep,
    SamplingReqConfig,
)
from mstar.engine.resources.sampler.resident import SamplerCells
from mstar.engine.resources.sampler.utils import (
    AcceleratorGraphableSampler,
    GenerationHistory,
    Sampler,
    SamplerBuffers,
    SamplerRows,
    SamplingConfig,
)
from mstar.engine.resources.step import ASKS, AdmitOutcome, SlotLease, StepContext
from mstar.utils.device import DeviceCapability
from mstar.utils.knobs import sampler_ingraph_scatter
from mstar.utils.profiler import PhaseLaps

logger = logging.getLogger(__name__)


class SamplerResource(Resource):
    prefix_skip_safe = True

    def can_capture(self) -> bool:
        return DeviceCapability.of(self._device).graph_backend is not None

    # TODO: this is  a light wrapper around mstar/engine/resources/sampler/utils.py. In the future,
    # we should rip out the parts we need from sampling.py and discard the rest.
    def __init__(
        self,
        vocab_size: int | None,
        enable_repetion_penalty: bool,
        device: torch.device,
        comm_group: JointGroups | None=None,
        enable_min_p: bool = False,
        max_repetition_window: int = 0,
        min_tokens_stop_ids: tuple[int, ...] = (),
        enable_top_p_first: bool = False,
        enable_logprobs: bool = False,
        enable_processed_logprobs: bool = False,
    ):
        if enable_repetion_penalty and (
            isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0
        ):
            raise ValueError("a presence-capable SamplerSpec needs a positive vocab_size")
        self._track_seen_tokens = enable_repetion_penalty
        self._enable_min_p = enable_min_p
        self._enable_logprobs = enable_logprobs
        self._enable_processed_logprobs = enable_processed_logprobs
        self._vocab_size = vocab_size if self._track_seen_tokens else None
        self._max_repetition_window = max_repetition_window
        self._enable_top_p_first = enable_top_p_first
        if max_repetition_window < 0:
            raise ValueError(f"max_repetition_window={max_repetition_window} must be >= 0")
        # What each request has generated, for the windowed penalty and the
        # min-tokens floor; shared by the eager and graph samplers, and built
        # only when the spec enables one of them. The filter order needs none.
        self._history: GenerationHistory | None = None
        self._stop_ids: torch.Tensor | None = None
        if max_repetition_window or min_tokens_stop_ids:
            self._history = GenerationHistory(max_repetition_window, device)
        if min_tokens_stop_ids:
            self._stop_ids = torch.tensor(
                sorted(set(min_tokens_stop_ids)), dtype=torch.long, device=device,
            )
        self._rows = SamplerRows(device, self._history, vocab_size=self._vocab_size)
        self._sampler = Sampler(
            device=device,
            rows=self._rows,
            tp_group=comm_group,
            stop_ids=self._stop_ids,
            enable_top_p_first=enable_top_p_first,
        )
        # Captures bake the step capability, independent of current multipliers.
        self._apply_penalty_this_step: bool = enable_repetion_penalty
        self._apply_filters_this_step: bool = True
        # Resident requests that asked for log-probs: with none, no step's
        # log-prob rows are copied to the host.
        self._logprob_rids: set[str] = set()
        self._sampled_this_step = False
        self._device = device
        self._comm_group = comm_group
        self._cg_buffers: SamplerBuffers | None = None
        self._cg_max_bs = 0
        self._cg_slots = 1
        self._capture_buffers_sealed = False
        # ``ask_stop_odds``: the walks and stop ids it scores, the rows a
        # capture writes, an eager step's own rows, and whether this step scores
        self._odds_walks: set[str] = set()
        self._odds_ids: torch.Tensor | None = None
        self._odds: torch.Tensor | None = None
        self._odds_eager: torch.Tensor | None = None
        self._odds_this_step = False

        # This is set during plan in the cuda graph case
        self._cg_sampler: AcceleratorGraphableSampler | None = None
        # pre-planned a step ahead, promoted by the next non-preplan plan
        self._preplan_cg_sampler: AcceleratorGraphableSampler | None = None
        self._preplan_key = None
        self._preplanned = False
        # rid -> the prompt tokens a cache hit kept out of each walk's inputs. Keyed by the
        # position after them, so a repeated probe replaces its own entry.
        self._cached_prefix: dict[str, dict[int, torch.Tensor]] = {}
        # this step's `SamplerStep.kept_rids`, for the eager draw and the in-graph scatter index
        self._kept_rids: frozenset[int] | None = None
        # this node's resident rows, once its planner binds this sampler
        self._cells: SamplerCells | None = None

    @classmethod
    def build(cls, spec: SamplerSpec, info: EngineResourceInfo):
        return cls(
            vocab_size=spec.vocab_size,
            enable_repetion_penalty=spec.enable_repetion_penalty,
            device=info.device,
            comm_group=info.joint_comm_group,
            enable_min_p=spec.enable_min_p,
            max_repetition_window=spec.max_repetition_window,
            min_tokens_stop_ids=tuple(spec.min_tokens_stop_ids),
            enable_top_p_first=spec.enable_top_p_first,
            enable_logprobs=spec.enable_logprobs,
            enable_processed_logprobs=spec.enable_processed_logprobs,
        )

    def build_accelerator_graph_buffers(
        self, slots, max_bs: int, max_seq_len: int
    ):
        del max_seq_len
        cg_slots = max((s.slot for s in slots), default=0) + 1
        # Every runner sizes before capture; redundant sizing keeps its views.
        if (
            self._cg_buffers is not None
            and max_bs <= self._cg_max_bs
            and cg_slots <= self._cg_slots
        ):
            return
        if self._capture_buffers_sealed:
            raise ValueError("sampler capture buffers cannot grow after capture sizing is sealed")
        self._cg_max_bs = max(max_bs, self._cg_max_bs)
        self._cg_slots = max(cg_slots, self._cg_slots)
        if self._odds_ids is not None:
            # before any capture: every one writes these rows
            self._odds = torch.zeros(self._cg_max_bs, dtype=torch.float32, device=self._device)
        self._cg_buffers = SamplerBuffers.allocate(
            max_batch_size=self._cg_max_bs, device=self._device,
            rows=self._rows,
            tp_group=self._comm_group,
            vocab_size=self._vocab_size,
            cg_slots=self._cg_slots,
            # every deciding sampler moves its own RNG offset rows in the graph,
            # a node reading its tokens back or not
            # every rank of a lockstep instance runs the leader's steps, so each
            # admits the same rows and moves its offsets in the graph alike
            ingraph_scatter=sampler_ingraph_scatter(),
            **self._scalar_options(),
        )
        self._cg_buffers.cells = self._cells
        for rid, config in self._sampler._sampling_config.items():
            if self._rows.has_slot(rid):
                self._cg_buffers.register_request(rid, config)

    def seal_capture_buffers(self) -> None:
        self._capture_buffers_sealed = True

    def _scalar_options(self) -> dict:
        """What decides the scalars a captured sampler reads per row."""
        options = dict(enable_min_p=self._enable_min_p, enable_logprobs=self._enable_logprobs)
        if self._device.type == "cuda":
            # CUDA-only (requests asking for them are refused elsewhere), so
            # another device's captured sampler is left exactly as it was
            options.update(
                history=self._history,
                stop_ids=self._stop_ids,
                enable_top_p_first=self._enable_top_p_first,
                enable_processed_logprobs=self._enable_processed_logprobs,
            )
        return options

    def ingest_request(self, rid: str, overrides: SamplingReqConfig | None=None):
        extra_kwargs = asdict(overrides) if overrides is not None else {}
        self._sampler.add_request(request_id=rid)
        self._sampler.set_config(
            request_id=rid,
            vocab_size=self._vocab_size,
            **extra_kwargs
        )
        # Read off the resolved config rather than `overrides`, so a request
        # that leaves the penalty unset takes the same default the sampler will.
        resolved = self._sampler._sampling_config[rid]
        # written so NaN fails too; NaN would also slip past the `> 0` check below
        if not 0.0 <= resolved.min_p <= 1.0:
            self._sampler.remove_request(rid)
            raise ValueError(
                f"request {rid!r} asks for min_p={resolved.min_p}; it must be in [0, 1]"
            )
        if resolved.min_p > 0 and not self._enable_min_p:
            # the graph-captured sampler has no min-p buffer, so honouring it
            # eagerly but not in graph would make sampling depend on the path
            self._sampler.remove_request(rid)
            raise ValueError(
                f"request {rid!r} asks for min_p={resolved.min_p} but the node's "
                "SamplerSpec has enable_min_p=False"
            )
        problem = self._generation_controls_problem(resolved)
        if problem is not None:
            self._sampler.remove_request(rid)
            raise ValueError(f"request {rid!r} {problem}")
        # after the refusals, so a refused request leaves no stale entry
        if resolved.return_logprobs:
            self._logprob_rids.add(rid)
        if self._rows.limit is None:
            if self._cg_buffers is not None:
                self._cg_buffers.register_request(rid, sampling_config=resolved)
            else:
                self._rows.register_request(rid)

    def _generation_controls_problem(self, cfg: SamplingConfig) -> str | None:
        """Why this node cannot serve ``cfg``'s generation-aware settings, if
        it cannot. Each needs its capability on the spec: the graph-captured
        sampler carries only those, so honouring one eagerly but not in graph
        would make sampling depend on the path."""
        if cfg.presence_penalty != 1.0 and not self._track_seen_tokens:
            return "asks for a presence repetition_penalty but the SamplerSpec disables it"
        if cfg.repetition_window < 0 or cfg.min_tokens < 0 or cfg.top_p_min_keep < 1:
            return (
                f"asks for repetition_window={cfg.repetition_window}, "
                f"min_tokens={cfg.min_tokens}, top_p_min_keep={cfg.top_p_min_keep}; "
                "the first two must be >= 0 and the last >= 1"
            )
        if cfg.repetition_window > self._max_repetition_window:
            return (
                f"asks for repetition_window={cfg.repetition_window} but the node's "
                f"SamplerSpec has max_repetition_window={self._max_repetition_window}"
            )
        if cfg.repetition_window > 0 and not cfg.repetition_penalty > 0:
            return f"asks for a windowed repetition_penalty={cfg.repetition_penalty}; it must be > 0"
        if cfg.min_tokens > 0 and self._stop_ids is None:
            return (
                f"asks for min_tokens={cfg.min_tokens} but the node's SamplerSpec "
                "has no min_tokens_stop_ids"
            )
        if (cfg.top_p_first or cfg.top_p_min_keep > 1) and not self._enable_top_p_first:
            return (
                f"asks for top_p_first={cfg.top_p_first}, top_p_min_keep="
                f"{cfg.top_p_min_keep} but the node's SamplerSpec has "
                "enable_top_p_first=False"
            )
        if cfg.return_logprobs and not self._enable_logprobs:
            return "asks for return_logprobs but the node's SamplerSpec has enable_logprobs=False"
        processed = cfg.return_logprobs and cfg.logprobs_mode == "processed"
        if processed and not self._enable_processed_logprobs:
            return "asks for processed logprobs but the node's SamplerSpec has enable_processed_logprobs=False"
        if (cfg.uses_history or cfg.uses_filter_order or processed) and self._sampler.device.type != "cuda":
            return (
                "asks for repetition_window, min_tokens, top_p_first, top_p_min_keep or "
                f"processed logprobs, which are CUDA-only (device {self._sampler.device})"
            )
        return None

    def ingest_capture_row(self, rid: str):
        """A capture padding row: default config, no seen-token mask, and the
        trash row in the graph buffers rather than a slot of its own."""
        self._sampler.add_request(request_id=rid)
        if self._cg_buffers is not None:
            self._cg_buffers.register_trash_row(rid)
        else:
            self._rows.register_trash_row(rid)

    def apply_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str,
        inputs, prefix: CachedPrefix | None,
    ) -> None:
        """Keep the tokens the cache is about to skip, for `plan` to fold in.

        A step declares its tracked tokens from inputs the prefix has already
        been cut out of, so without these the mask would hold the prompt's tail
        alone and the penalty would let the model repeat the rest. A walk the
        cache serves whole runs no plan, so its tokens are kept until the next
        walk that does.

        The generation history needs nothing here: it counts generated tokens
        only, and a cached prefix is prompt.
        """
        del node_name, graph_walk
        # a walk that prefills from embeddings has no ids to keep
        if prefix is not None and prefix.tokens > 0 and inputs.input_ids is not None:
            self._cached_prefix.setdefault(rid, {})[prefix.position] = (
                inputs.input_ids[:prefix.tokens]
            )

    def remove_request(self, rid: str):
        self._sampler.remove_request(rid)
        self._logprob_rids.discard(rid)
        self._cached_prefix.pop(rid, None)
        if self._cg_buffers is not None:
            self._cg_buffers.unregister_request(rid)
        else:
            self._rows.unregister_request(rid)

    # Admission: with the in-graph scatter, a request's row in the masters the
    # captured graphs index, of which there are `sampler_slots`. A request takes
    # it before its first step and holds it until it is removed, which waits for
    # the steps still running it, so a row is never handed on while one of them
    # can still write it.

    def ask(self, rid: str, node_name: str, graph_walk: str) -> AdmitOutcome | None:
        if self._rows.limit is None or self._rows.has_slot(rid):
            return None
        return ASKS

    def reserve(self, rid: str, node_name: str, graph_walk: str, published: None) -> int | None:
        room = self._rows.free_rows
        if room <= 0:
            return room
        if self._cg_buffers is not None:
            self._cg_buffers.register_request(rid, sampling_config=self._sampler._sampling_config[rid])
        else:
            self._rows.register_request(rid)
        return None

    def unreserve(self, rid: str) -> None:
        if self._cg_buffers is not None:
            self._cg_buffers.unregister_request(rid)
        else:
            self._rows.unregister_request(rid)

    def grew_since(self, mark: int) -> bool:
        return self._rows.free_rows > mark

    def _set_step_flags(self, step: SamplerStep, ctx: StepContext):
        if not ctx.is_preplan:
            self._apply_filters_this_step = step.apply_filters
            self._apply_penalty_this_step = self._track_seen_tokens and step.apply_penalty

    # Resident planning: the scalars as cells of the planner's rows, by its
    # slot; the rest by this sampler's own rows (`sampler/resident.py`).

    def repeat_refusal(self, walk: str, host_step=None) -> str | None:
        """Replays need the RNG offset to move in the graph.
        Fixed at load, as `_cg_buffers` decides it. What else a draw reads
        moves in the graph too: history and token presence persist each draw."""
        if not sampler_ingraph_scatter():
            return "it scatters its RNG offsets outside the graph"
        return None

    def max_repeats(self, step, ctx) -> int:
        """1 for a step with a row that wants log-probs (one a step), or whose
        rows don't all keep their draw. From the step's own rows only, which
        are the same on every rank of a lockstep instance: the pool-wide set
        depends on when each rank ingested its requests."""
        logprobs = self._logprob_rids
        if (logprobs and not logprobs.isdisjoint(ctx.request_ids)) or self._kept_rids_of(step, ctx) is not None:
            return 1
        return ANY_REPEATS

    def _check_resident_horizon(self, step, ctx) -> None:
        if step is None:
            raise ValueError("captured state updates need their declared row step")
        if (self.repeat_refusal(ctx.graph_walk) is not None or self._cg_buffers is None
                or not self._cg_buffers.ingraph_scatter):
            raise ValueError("needs captured sampler state updates")
        if not self._logprob_rids.isdisjoint(ctx.request_ids) or self._kept_rids_of(step, ctx) is not None:
            raise ValueError("new rows cannot require logprobs or discard a sampled draw")

    def resident_binding(self) -> ResidentBinding:
        return ResidentBinding(
            check_streams=no_resident_effect,
            bind=self._bind_resident,
            row=self._resident_row,
            stage=no_resident_effect,
            gather=self._resident_gather,
            run_gather=no_resident_effect,
            commit_in_graph=no_resident_effect,
            advance=no_resident_effect,
            check_horizon=self._check_resident_horizon,
            max_repeats=self.max_repeats,
            rewind_run=no_resident_effect,
        )

    def _bind_resident(self, planner: ResidentPlanner, key: str) -> None:
        if not sampler_ingraph_scatter():
            return
        self._cells = SamplerCells(planner, SamplerBuffers.fields_for(**self._scalar_options()))
        if self._cg_buffers is not None:
            self._cg_buffers.attach_cells(self._cells)
        logger.info(
            "resident planning on %s: sampler %r keeps %s as cells; the RNG offset, "
            "seen-token mask and history stay by its own rows",
            planner.node, key, ", ".join(f.name for f in self._cells.fields),
        )

    def _resident_row(self, rid: int, slot: int, row: list[int], writes: ResidentWrites) -> None:
        if self._cells is not None:
            self._cells.row(rid, slot, self._rows.slot_of(rid), row)

    def _resident_gather(self, step: CapturedResidentStep) -> None:
        if self._cells is None or self._cg_buffers is None:
            return
        self._cells.gather(step)
        self._cg_sampler = self._cells.sampler_for(self._cg_buffers, step.bucket, step.slot)

    def _resident_rows(self, ctx: StepContext) -> tuple | None:
        """(bucket, cg slot) of a resident step, which reads its scalars off the cells."""
        if self._cells is None or ctx.resident is None:
            return None
        return ctx.slot_lease.bucket, ctx.slot_lease.slot

    @property
    def supports_preplan(self):
        return True

    @property
    def force_double_buffer(self):
        # Per-step gather indices are staged into a reused pinned buffer
        # (``_slot_idx_cpu``) behind an async H2D, so a single-buffered region
        # (a piecewise runner with no pre-plan) races when the host runs ahead.
        return True

    def clear_preplan(self):
        self._preplanned = False
        self._preplan_cg_sampler = None
        self._preplan_key = None

    def _plan_key(self, ctx: StepContext):
        """What identifies the step a pre-plan was staged for: its padded
        request rows and the slot they were leased on, or on a resident step
        the planner's stage, which it promotes only for the step it was built
        for (compared by identity: see ``_preplan_matches``)."""
        if self._resident_rows(ctx) is not None:
            return ctx.resident_plan
        lease = ctx.slot_lease
        return (tuple(ctx.padded_request_ids), lease.slot if lease is not None else None)

    def _preplan_matches(self, ctx: StepContext) -> bool:
        key = self._plan_key(ctx)
        if isinstance(key, ResidentPlan) or isinstance(self._preplan_key, ResidentPlan):
            return key is self._preplan_key
        return key == self._preplan_key

    @staticmethod
    def _kept_rids_of(step: SamplerStep, ctx: StepContext) -> frozenset[int] | None:
        """The rows whose draw commits: ``step.kept_rids`` less any row mid-prefill.
        A non-final chunk's token is not a real sample, so it advances no RNG
        offset and marks no token seen."""
        mid_prefill = ctx.chunk_tokens_after
        if not mid_prefill:
            return step.kept_rids
        kept = frozenset(ctx.request_ids) if step.kept_rids is None else step.kept_rids
        return kept - mid_prefill.keys()

    def plan(self, step: SamplerStep, ctx: StepContext):
        laps = PhaseLaps(f"{ctx.plan_phase}.sampler.")
        self._set_step_flags(step, ctx)
        if not ctx.is_preplan:
            self._kept_rids = self._kept_rids_of(step, ctx)
            self._sampled_this_step = False
            self._odds_this_step = ctx.graph_walk in self._odds_walks
        if not ctx.is_preplan and self._track_seen_tokens:
            configs = self._sampler._sampling_config
            for rid, tokens in step.prefill_tracked_tokens.items():
                # a replay's padding rows are declared too, and hold no mask
                if ctx.is_padding_row(rid):
                    continue
                if configs[rid].penalize_prompt:
                    self._rows.seed_seen(rid, tokens)
            # dropped as they are used: `plan` runs on every step of a resident
            # request, and these belong to the one prefill that was cut
            for rid in ctx.request_ids:
                skipped = self._cached_prefix.pop(rid, None)
                if skipped is not None and configs[rid].penalize_prompt:
                    for tokens in skipped.values():
                        self._rows.seed_seen(rid, tokens)

        # A step planned ahead promotes here. Its static config was gathered in
        # the preplan; dynamic state must reflect the previous step, so its
        # gather runs inline now or inside the graph, after those writes.
        # Only the leased step the plan was staged for may promote it: a
        # different batch reaching the GPU thread first (a new request's
        # eager prefill while a decode step sits pre-planned) plans inline
        # and the staged plan is dropped, exactly as `reset_pre_plan_for_batch`
        # would have done.
        laps.lap("penalty")
        if self._preplanned and not ctx.is_preplan:
            if ctx.slot_lease is None or not self._preplan_matches(ctx):
                self.clear_preplan()
            else:
                self._gather_dynamic(ctx, ctx.slot_lease)
                laps.lap("dynamic")
                self._cg_sampler = self._preplan_cg_sampler
                self._preplan_cg_sampler = None
                self._preplanned = False
                return

        # invalidated on the inline path here (not in commit, which now runs
        # before output collection); a preplan must leave the in-flight one be
        if not ctx.is_preplan:
            self._cg_sampler = None
            self._sampler.logprobs = None
        lease = ctx.slot_lease
        if self._cg_buffers is None or lease is None:
            return

        cg_slot = lease.slot
        padded_bs = lease.bucket.bs
        rows = self._resident_rows(ctx)
        if rows is not None:
            # The graph gathers scalars and owner identity from resident cells.
            sampler = self._cells.sampler_for(self._cg_buffers, *rows)
            laps.lap("resident")
        else:
            # static config only: safe to pre-plan (unchanged step to step). A
            # preplan leases a different slot, so this is disjoint from the in-flight
            # forward's buffers.
            # Real rids: gather_static pads to padded_bs itself (the tail reads
            # slot 0, a defaults row) and gates the scatter-back on the real count.
            self._cg_buffers.gather_static(ctx.request_ids, padded_bs, cg_slot)
            sampler = self._cg_buffers.sampler_for(padded_bs, cg_slot)
            laps.lap("static")
        if ctx.is_preplan:
            self._preplan_cg_sampler = sampler
            self._preplan_key = self._plan_key(ctx)
            self._preplanned = True
        else:
            # fresh inline (capture / no preplan): gather the per-step state too
            self._gather_dynamic(ctx, lease)
            laps.lap("dynamic")
            self._cg_sampler = sampler

    def _gather_dynamic(self, ctx: StepContext, lease: SlotLease):
        """Prepare indices and drain resets inline. Direct captures gather
        the owner's RNG/history/presence after earlier device writes; host
        scatter mode gathers the same state here before replay."""
        cg_slot = lease.slot
        padded_bs = lease.bucket.bs
        rows = self._resident_rows(ctx)
        if rows is None:
            self._cg_buffers.gather_dynamic(
                ctx.request_ids, padded_bs, cg_slot,
                gather_seen_tokens=self._apply_penalty_this_step, kept=self._kept_rids,
            )
        else:
            # No resident identity is read before flush/gather. The capture
            # resolves its owner rows and gathers all dynamic data itself.
            self._rows.sync()
            self._cells.stage_keep(*rows, ctx.request_ids, self._kept_rids)

    def commit(self, step: SamplerStep, ctx: StepContext):
        if self._cg_buffers is None or self._cg_sampler is None or self._cg_buffers.ingraph_scatter:
            return
        kept = self._kept_rids_of(step, ctx)
        keep = None if kept is None else [rid in kept for rid in ctx.request_ids]
        self._cg_buffers.scatter_state(
            ctx.slot_lease.slot, keep=self._cg_buffers.keep_mask(keep),
            seen_tokens=self._apply_penalty_this_step,
        )

    ### Submodule-level functionality

    # Out of the caller's compiled frame: dynamo would guard on the ids it passes on.
    @torch.compiler.disable
    def sample(
        self, request_ids: list[str], logits: torch.Tensor, **kwargs
    ):
        """One token per row of ``logits``, in ``request_ids`` order.

        Called from the forward, so the batch it ran on is the batch to hand
        here: under capture that is the slot's padded rows and its dummy ids,
        which is what the graph sampler's per-row params were gathered for.
        The eager sampler keys off request ids and sees the real batch only.
        """
        if self._odds_this_step:
            self._score_stops(logits)
        if self._enable_logprobs:
            # under capture this fires once, at capture; a replay never calls in
            if self._sampled_this_step:
                raise RuntimeError("log-probs need one sample() per step, and this node's forward samples again")
            self._sampled_this_step = True
        if self._cg_sampler is not None:
            # The step capability is baked at capture and must match every
            # replay, independently of the current per-request multiplier.
            return self._cg_sampler.sample(
                request_ids, logits,
                apply_penalty=self._apply_penalty_this_step,
                apply_filters=self._apply_filters_this_step,
            )
        return self._sampler.sample(
            request_ids, logits, apply_penalty=self._apply_penalty_this_step,
            apply_filters=self._apply_filters_this_step, kept_rids=self._kept_rids,
        )

    def ask_stop_odds(self, walk: str, stop_ids) -> None:
        """Before capture: score each row's stop ids at every sample of ``walk``
        (``STOP_ODDS``)."""
        self._odds_walks.add(walk)
        self._odds_ids = torch.tensor(sorted(stop_ids), dtype=torch.long, device=self._device)

    def _score_stops(self, logits: torch.Tensor) -> None:
        """Each row's probability of its stop ids, off the raw logits, so a
        greedy request has one too; into the rows a capture writes."""
        logits = logits.float()
        odds = (logits.index_select(-1, self._odds_ids) - logits.logsumexp(-1, keepdim=True)).exp().sum(-1)
        rows = odds.shape[0]
        if self._odds is not None and rows <= self._odds.shape[0]:
            self._odds[:rows].copy_(odds)
            self._odds_eager = None
        else:
            self._odds_eager = odds

    def step_host_rows(self, repeats: int = 1) -> dict[str, torch.Tensor]:
        """The step's log-prob rows, and its stop odds where asked. The
        log-probs assume one ``sample`` per step whose token is the row's
        emitted one, which ``sample`` enforces while log-probs are on; a node
        that samples more than once (a code predictor) needs its own path. A
        step that repeats has none: no row of it asked, or it would have
        settled at 1. Its odds are its last iteration's."""
        out = {}
        if self._odds_this_step:
            # an eager step's rows are its own; a replay wrote the capture's
            out[STOP_ODDS] = self._odds_eager if self._odds_eager is not None else self._odds
            self._odds_this_step, self._odds_eager = False, None
        if not self._logprob_rids or repeats > 1:
            return out
        # a replay never calls `sample`: its rows are the slot's static ones
        rows = self._cg_sampler.logprob_buf if self._cg_sampler is not None else self._sampler.logprobs
        return out if rows is None else {**out, SAMPLED_LOGPROB: rows}

