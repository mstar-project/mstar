

from dataclasses import asdict
from time import perf_counter

import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import CachedPrefix, EngineResourceInfo, Resource
from mstar.engine.resources.sampler.config import (
    SamplerSpec,
    SamplerStep,
    SamplingReqConfig,
)
from mstar.engine.resources.sampler.utils import (
    AcceleratorGraphableSampler,
    Sampler,
    SamplerBuffers,
    fused_temperature_softmax,
)
from mstar.engine.resources.step import SlotLease, StepContext
from mstar.utils.device import DeviceCapability
from mstar.utils.knobs import device_loopback_enabled, sampler_ingraph_scatter
from mstar.utils.profiler import PHASE_PERIOD, phase_record


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
    ):
        self._track_seen_tokens = enable_repetion_penalty
        self._enable_min_p = enable_min_p
        self._vocab_size = vocab_size if self._track_seen_tokens else None
        self._sampler = Sampler(
            device=device,
            tp_group=comm_group
        )
        # Two flags, because they have different lifetimes.
        #
        # `_apply_penalty_this_step` is the CAPABILITY: whether the sampling
        # kernel this step runs contains the penalty at all. `APPLY_PENALTY` is
        # a Triton constexpr baked when the graph is captured (see
        # `fused_temperature_softmax`), so a replay cannot toggle it — this has
        # to be the value capture saw, and it is what `sample` forwards.
        #
        # `_penalty_needed_this_step` is the LIVENESS: whether the penalty can
        # change a logit at all right now. With every resident request at
        # `repetition_penalty == 1.0` the baked kernel is
        # `where(x > 0, x / 1.0, x * 1.0)` — an identity map WHATEVER the
        # seen-token mask holds — so the mask may be left stale and every bit of
        # staging/gathering/syncing around it is skippable, bit-exactly. That
        # traffic is bs x [vocab] copies plus a [bs, vocab] gather per step, all
        # off-graph on the GPU thread's critical path, so dropping it when it
        # cannot matter is the whole point of the split.
        self._apply_penalty_this_step: bool = enable_repetion_penalty
        self._penalty_needed_this_step: bool = enable_repetion_penalty
        # Resident requests that asked for a penalty. Held as a set rather than
        # recomputed per step: `admit` is on the per-step path and this only
        # moves on ingest/remove.
        self._penalty_rids: set[str] = set()
        self._device = device
        self._comm_group = comm_group
        self._cg_buffers: SamplerBuffers | None = None
        self._cg_max_bs = 0
        self._cg_slots = 1

        # This is set during plan in the cuda graph case
        self._cg_sampler: AcceleratorGraphableSampler | None = None
        # pre-planned a step ahead, promoted by the next non-preplan plan
        self._preplan_cg_sampler: AcceleratorGraphableSampler | None = None
        self._preplan_key = None
        self._preplanned = False
        # (cg slot, padded bs) of the step planned last, for the last-token
        # read in ``loopback_tokens``; None on an eager step
        self._step_slot: tuple[int, int] | None = None
        # whether the last sampled token is persisted by slot at all: off until
        # a node that reads the masters registers (``enable_device_loopback_reader``)
        self._keep_last_token = False
        # rid -> the prompt tokens a cache hit kept out of each walk's inputs. Keyed by the
        # position after them, so a repeated probe replaces its own entry.
        self._cached_prefix: dict[str, dict[int, torch.Tensor]] = {}
        # this step's `SamplerStep.kept_rids`, for the eager draw
        self._kept_rids: frozenset[int] | None = None

    @property
    def _penalty_live(self) -> bool:
        """Whether any resident request wants a repetition penalty.

        Resource-level rather than per-step, deliberately. A per-step answer
        would let a request decode for a stretch with its mask un-synced and
        then land in a batch beside a penalised one, at which point the penalty
        would be applied against a mask missing everything generated meanwhile.
        Keyed on residency, the masks are maintained from the moment anyone
        needs them; a request sitting at 1.0 accumulates a stale mask that only
        its own (inert) row ever reads.
        """
        return self._track_seen_tokens and bool(self._penalty_rids)

    @classmethod
    def build(cls, spec: SamplerSpec, info: EngineResourceInfo):
        return cls(
            vocab_size=spec.vocab_size,
            enable_repetion_penalty=spec.enable_repetion_penalty,
            device=info.device,
            comm_group=info.joint_comm_group,
            enable_min_p=spec.enable_min_p,
        )

    def build_accelerator_graph_buffers(
        self, slots, max_bs: int, max_seq_len: int
    ):
        del max_seq_len
        cg_slots = max((s.slot for s in slots), default=0) + 1
        # every runner capturing against this node calls in; reallocating would
        # drop the rows already registered, so only grow
        if (
            self._cg_buffers is not None
            and max_bs <= self._cg_max_bs
            and cg_slots <= self._cg_slots
        ):
            return
        self._cg_max_bs = max(max_bs, self._cg_max_bs)
        self._cg_slots = max(cg_slots, self._cg_slots)
        self._cg_buffers = SamplerBuffers.allocate(
            max_batch_size=self._cg_max_bs, device=self._device,
            tp_group=self._comm_group,
            vocab_size=self._vocab_size,
            cg_slots=self._cg_slots,
            enable_min_p=self._enable_min_p,
            ingraph_scatter=self._keep_last_token and sampler_ingraph_scatter(),
        )

    def ingest_request(self, rid: str, overrides: SamplingReqConfig | None=None):
        if rid in self._sampler._sampling_config:
            # a worker serving two partitions ingests the rid twice; adding it
            # again would reset its seen-token mask and RNG offset mid-request
            return
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
        # after the refusals, so a refused request leaves no stale entry
        if resolved.repetition_penalty != 1.0:
            self._penalty_rids.add(rid)
        if self._cg_buffers is not None:
            self._cg_buffers.register_request(
                rid, sampling_config=self._sampler._sampling_config[rid]
            )

    def ingest_capture_row(self, rid: str):
        """A capture padding row: default config, no seen-token mask, and the
        trash row in the graph buffers rather than a slot of its own."""
        self._sampler.add_request(request_id=rid)
        if self._cg_buffers is not None:
            self._cg_buffers.register_trash_row(rid)

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
        """
        del node_name, graph_walk
        # a walk that prefills from embeddings has no ids to keep
        if prefix is not None and prefix.tokens > 0 and inputs.input_ids is not None:
            self._cached_prefix.setdefault(rid, {})[prefix.position] = (
                inputs.input_ids[:prefix.tokens]
            )

    def remove_request(self, rid: str):
        self._sampler.remove_request(rid)
        self._penalty_rids.discard(rid)
        self._cached_prefix.pop(rid, None)
        if self._cg_buffers is not None:
            self._cg_buffers.unregister_request(rid)

    def _set_penalty_flags(self, step: SamplerStep, ctx: StepContext):
        """Settle this step's two penalty flags.
        """
        if not ctx.is_preplan:
            self._apply_penalty_this_step = (
                self._track_seen_tokens and step.apply_penalty
            )
            self._penalty_needed_this_step = (
                self._apply_penalty_this_step and self._penalty_live
            )
            if __debug__ and self._apply_penalty_this_step \
                    and not self._penalty_needed_this_step:
                self._assert_penalty_inert(ctx)

    def _assert_penalty_inert(self, ctx: StepContext) -> None:
        """Skipping the mask is sound only while every penalty in the step is 1.0.

        An O(bs) dict walk on the skip path — a rounding error next to the mask
        traffic it guards, and this invariant is subtle enough to be worth
        checking live. `python -O` drops it.
        """
        configs = self._sampler._sampling_config
        penalised = [
            rid for rid in ctx.request_ids
            if rid in configs and configs[rid].repetition_penalty != 1.0
        ]
        assert not penalised, (
            "repetition-penalty bookkeeping was skipped for a step carrying "
            f"penalised requests {penalised}; `_penalty_rids` is out of sync "
            "with the sampler's configs (a mid-request config update?)"
        )

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

    @staticmethod
    def _plan_key(ctx: StepContext):
        """What identifies the step a pre-plan was staged for: its padded
        request rows and the slot they were leased on."""
        lease = ctx.slot_lease
        return (tuple(ctx.padded_request_ids), lease.slot if lease is not None else None)

    def plan(self, step: SamplerStep, ctx: StepContext):
        self._set_penalty_flags(step, ctx)
        if not ctx.is_preplan:
            self._kept_rids = step.kept_rids
        if not ctx.is_preplan and self._penalty_live:
            configs = self._sampler._sampling_config
            for rid, tokens in step.prefill_tracked_tokens.items():
                # a replay's padding rows are declared too, and hold no mask
                if ctx.is_padding_row(rid):
                    continue
                if configs[rid].penalize_prompt:
                    self._sampler.get_token_mask(rid).add_tokens(tokens)
            # dropped as they are used: `plan` runs on every step of a resident
            # request, and these belong to the one prefill that was cut
            for rid in ctx.request_ids:
                skipped = self._cached_prefix.pop(rid, None)
                if skipped is not None and configs[rid].penalize_prompt:
                    for tokens in skipped.values():
                        self._sampler.get_token_mask(rid).add_tokens(tokens)

        # A step planned ahead promotes here. Its static config was gathered in
        # the preplan; the per-step state (RNG offset + seen-token mask) is NOT
        # double-buffered — it must reflect the previous step's commit, so
        # gather it inline now, on the default stream, after that commit.
        # Only the leased step the plan was staged for may promote it: a
        # different batch reaching the GPU thread first (a new request's
        # eager prefill while a decode step sits pre-planned) plans inline
        # and the staged plan is dropped, exactly as `reset_pre_plan_for_batch`
        # would have done.
        if self._preplanned and not ctx.is_preplan:
            if ctx.slot_lease is None or self._preplan_key != self._plan_key(ctx):
                self.clear_preplan()
            else:
                self._gather_dynamic(ctx, ctx.slot_lease)
                self._cg_sampler = self._preplan_cg_sampler
                self._preplan_cg_sampler = None
                self._preplanned = False
                self._step_slot = (ctx.slot_lease.slot, ctx.slot_lease.bucket.bs)
                return

        # invalidated on the inline path here (not in commit, which now runs
        # before output collection); a preplan must leave the in-flight one be
        if not ctx.is_preplan:
            self._cg_sampler = None
            self._step_slot = None
        lease = ctx.slot_lease
        if self._cg_buffers is None or lease is None:
            return

        cg_slot = lease.slot
        padded_bs = lease.bucket.bs
        # static config only: safe to pre-plan (unchanged step to step). A
        # preplan leases a different slot, so this is disjoint from the in-flight
        # forward's buffers.
        # Real rids: gather_static pads to padded_bs itself (the tail reads
        # slot 0, a defaults row) and gates the scatter-back on the real count.
        self._cg_buffers.gather_static(ctx.request_ids, padded_bs, cg_slot)
        sampler = self._cg_buffers.sampler_for(padded_bs, cg_slot)
        if ctx.is_preplan:
            self._preplan_cg_sampler = sampler
            self._preplan_key = self._plan_key(ctx)
            self._preplanned = True
        else:
            # fresh inline (capture / no preplan): gather the per-step state too
            self._gather_dynamic(ctx, lease)
            self._cg_sampler = sampler
            self._step_slot = (cg_slot, padded_bs)

    def _gather_dynamic(self, ctx: StepContext, lease: SlotLease):
        """Gather the per-step RNG offset + seen-token mask into this step's
        slot, inline so they reflect the previous step's committed tokens.

        The mask half is the expensive half and is skipped whenever the penalty
        is inert this step — see the flags in `__init__`. The RNG offset is
        always gathered: it advances in-graph every step regardless."""
        cg_slot = lease.slot
        padded_bs = lease.bucket.bs
        if self._penalty_needed_this_step:
            self._cg_buffers.stage_seen_token_masks(
                request_ids=ctx.request_ids,
                seen_masks=[self._sampler.get_token_mask(rid) for rid in ctx.request_ids]
            )
        self._cg_buffers.gather_dynamic(
            ctx.request_ids, padded_bs, cg_slot,
            gather_seen_tokens=self._penalty_needed_this_step,
        )

    def commit(self, step: SamplerStep, ctx: StepContext):
        # None on an eager step, which never gathered one
        if self._cg_buffers is None or self._cg_sampler is None:
            return
        kept = step.kept_rids
        keep = None if kept is None else [rid in kept for rid in ctx.request_ids]
        if self._cg_buffers.ingraph_scatter:
            # the captured sample scattered both rows itself
            if self._penalty_needed_this_step:
                self._cg_sampler.sync_seen_token_masks(
                    [self._sampler.get_token_mask(rid) for rid in ctx.request_ids],
                    keep=keep,
                )
            return
        t0 = perf_counter() if PHASE_PERIOD else 0.0
        keep_gpu = self._cg_buffers.keep_mask(keep)
        self._cg_buffers.scatter_offset(ctx.slot_lease.slot, keep=keep_gpu)
        if PHASE_PERIOD:
            phase_record("res.commit.sampler.offset", perf_counter() - t0)
            t0 = perf_counter()
        if self._keep_last_token:
            self._cg_buffers.scatter_last_token(ctx.slot_lease.slot, keep=keep_gpu)
            if PHASE_PERIOD:
                phase_record("res.commit.sampler.last_token", perf_counter() - t0)
        # Skipped when nothing read the mask this step: there is then nothing to
        # copy back, and the rows go stale only for requests at penalty 1.0,
        # which never read them. See `_penalty_live` for why that stays sound.
        if self._penalty_needed_this_step:
            self._cg_sampler.sync_seen_token_masks(
                [self._sampler.get_token_mask(rid) for rid in ctx.request_ids],
                keep=keep,
            )

    ### Submodule-level functionality

    def speculative_sample(
        self, request_ids: list[str], logits: torch.Tensor, drafts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Speculative sampling of one verify block per row, the drafts taken as certain.

        ``logits`` ``[n, k + 1, V]`` are the target's after the last token and after each
        draft, ``drafts`` ``[n, k]``. Draft i stays with probability p_i(draft); the first one
        refused is replaced by a draw from p_i without it, and with every draft kept one more
        comes from p_k, so the tokens are distributed as sampling the target one at a time. A
        greedy row (temperature 0) keeps its argmax's prefix, as greedy verification does.
        Returns the tokens ``[n, k + 1]`` (a row's first refused draft replaced; later entries
        unspecified) and the drafts kept per row. Neither a repetition penalty nor min_p
        applies here: callers refuse requests that set them.
        """
        n, block, vocab = logits.shape
        # A row draws up to block + 1 numbers from its Philox stream at `offset` (each draft,
        # the refused one again, then the replacement); the next step starts past them all
        stride = block + 1
        temperature, top_k, top_p, seed, offset = self._spec_params(
            request_ids, n, logits.device, stride)
        rows = logits.reshape(n * block, vocab)
        expand = lambda t: t.repeat_interleave(block)  # noqa: E731
        top_k = torch.where(top_k > 0, top_k, vocab)
        if logits.is_cuda:
            import flashinfer

            with torch.cuda.device(logits.device):
                probs = fused_temperature_softmax(rows, expand(temperature), include_greedy=True)
                probs = flashinfer.sampling.top_k_renorm_probs(probs, expand(top_k))
                probs = flashinfer.sampling.top_p_renorm_probs(probs, expand(top_p))
                draft_probs = probs.new_zeros(n, block - 1, vocab)
                draft_probs.scatter_(2, drafts[..., None].long(), 1.0)
                accepted = torch.zeros(n, dtype=torch.int32, device=logits.device)
                kept = torch.zeros(n, dtype=torch.int32, device=logits.device)
                # (tokens, accepted per position, drafts kept); the counts are added in place
                tokens, _, kept = flashinfer.sampling.chain_speculative_sampling(
                    draft_probs, drafts.int(), probs.view(n, block, vocab), accepted, kept,
                    deterministic=True, seed=seed, offset=offset,
                )
                # A draft refused at u == 1 under a target one-hot at it leaves nothing to
                # draw from, and the kernel emits an id it never set: the draft stands there
                at = kept.long()[:, None]
                q = probs.view(n, block, vocab).gather(
                    1, at[..., None].expand(n, 1, vocab)).squeeze(1)
                emitted = tokens.long().gather(1, at)
                drawn = (emitted >= 0) & (emitted < vocab) & (
                    q.gather(1, emitted.clamp(0, vocab - 1)) > 0)
                tokens = tokens.long().scatter(
                    1, at, torch.where(drawn, emitted, q.argmax(-1, keepdim=True)))
        else:
            tokens, kept = _speculative_sample_torch(
                rows.float(), expand(temperature), expand(top_k), expand(top_p), drafts, block)
        if self._cg_sampler is not None:
            self._cg_sampler.offset_buf += stride
        return tokens.long().clamp_min(0), kept.long()

    def _spec_params(self, request_ids, n, device, stride):
        """Per-row temperature, top_k, top_p, seed and RNG offset: the slot's gathered buffers
        under a graph, else the requests' configs and the eager sampler's offsets, which
        this step moves ``stride`` on."""
        cg = self._cg_sampler
        if cg is not None:
            return (cg.temperature_buf[:n], cg.top_k_buf[:n], cg.top_p_buf[:n],
                    cg.seed_buf[:n], cg.offset_buf[:n])
        configs = [self._sampler._sampling_config[rid] for rid in request_ids]
        steps = self._sampler._step_offset
        offsets = []
        for rid in request_ids:
            offsets.append(steps.get(rid, 0))
            steps[rid] = offsets[-1] + stride
        as_tensor = lambda values, dtype: torch.tensor(values, dtype=dtype, device=device)  # noqa: E731
        return (as_tensor([c.temperature for c in configs], torch.float32),
                as_tensor([c.top_k for c in configs], torch.int32),
                as_tensor([c.top_p for c in configs], torch.float32),
                as_tensor([c._seed for c in configs], torch.int64),
                as_tensor(offsets, torch.int64))

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
        if self._cg_sampler is not None:
            # The capability flag, not the liveness one. Under a replay this
            # argument is inert (the kernel variant was baked at capture), and
            # under capture it has to match what every later replay will run.
            # `_penalty_needed_this_step` only ever suppresses work that cannot
            # change the result, so leaving the kernel on costs nothing.
            return self._cg_sampler.sample(
                request_ids, logits,
                apply_penalty=self._apply_penalty_this_step
            )
        tokens = self._sampler.sample(
            request_ids, logits, apply_penalty=self._apply_penalty_this_step,
            kept_rids=self._kept_rids,
        )
        # An eager step has no per-step row to scatter from: write the masters
        # by request, so the next step can still read its token on the device.
        bufs = self._cg_buffers
        if (
            self._keep_last_token and bufs is not None
            and all(bufs.has_slot(rid) for rid in request_ids)
        ):
            bufs.write_last_tokens(request_ids, tokens)
        return tokens

    def enable_device_loopback_reader(self) -> None:
        """A node of this sampler declares it reads its last sampled tokens
        back on the device (``NodeSubmodule.reads_device_loopback``). Only then
        are the slot masters kept: the eager-step write behind them is a small
        synchronous host-to-device copy per step, and a node that takes its
        tokens on the host would pay it for nothing. Honours the knob."""
        self._keep_last_token = device_loopback_enabled()

    @property
    def has_slot_masters(self) -> bool:
        """Whether the last sampled token of every request is kept on the
        device by slot (the graph-buffer path): what the device-side
        loop-back needs. False on an eager-only node or with the knob off."""
        return self._cg_buffers is not None and self._keep_last_token

    def loopback_tokens(self, request_ids: list[str]) -> torch.Tensor:
        """The last sampled token of each row of this step, as its input ids,
        read on the device. Under a slot lease the planned slot's index row
        orders them (padding rows read slot 0); an eager step looks its
        requests up by slot. Call after ``plan``."""
        bufs = self._cg_buffers
        if bufs is None:
            raise RuntimeError("device loop-back needs the sampler's slot masters")
        if self._cg_sampler is not None and self._step_slot is not None:
            cg_slot, padded_bs = self._step_slot
            return bufs.gather_last_tokens(cg_slot, padded_bs)
        return bufs.last_tokens_for(request_ids)


def _speculative_sample_torch(probs_logits, temperature, top_k, top_p, drafts, block):
    """``SamplerResource.speculative_sample`` without a GPU: the same rule, torch's RNG."""
    rows, vocab = probs_logits.shape
    n, k = rows // block, block - 1
    greedy = temperature <= 0
    probs = torch.softmax(probs_logits / torch.where(greedy, 1.0, temperature)[:, None], -1)
    probs = torch.where(greedy[:, None], torch.nn.functional.one_hot(
        probs_logits.argmax(-1), vocab).float(), probs)
    order = probs.argsort(-1, descending=True)
    sorted_probs = probs.gather(-1, order)
    rank = torch.arange(vocab, device=probs.device)[None, :]
    keep = (rank < top_k[:, None]) & (sorted_probs.cumsum(-1) - sorted_probs < top_p[:, None])
    keep[:, 0] = True
    probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs * keep)
    probs = (probs / probs.sum(-1, keepdim=True)).view(n, block, vocab)
    tokens = torch.zeros(n, block, dtype=torch.long, device=probs.device)
    kept = torch.zeros(n, dtype=torch.long, device=probs.device)
    for r in range(n):
        for i in range(block):
            p = probs[r, i]
            if i < k:
                d = int(drafts[r, i])
                if torch.rand(()) < p[d]:
                    tokens[r, i] = d
                    kept[r] += 1
                    continue
                p = p.clone()
                p[d] = 0.0
            tokens[r, i] = torch.multinomial(p / p.sum(), 1)[0]
            break
    return tokens, kept
