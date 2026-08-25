

from dataclasses import asdict, dataclass

import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import Resource
from mstar.engine.resources.sampler.utils import CudaGraphableSampler, Sampler, SamplerBuffers
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig, ResourceType
from mstar.engine.resources.step import ADMIT_OK, AdmitOutcome, SamplerStep, StepContext


@dataclass
class SamplerSpec(NodeResourceSpec):
    vocab_size: int | None # must be set for enabled_repetion_penalty
    # Capability, not intent: whether this node's sampling kernel is *able* to
    # apply a repetition penalty, which decides both the seen-token buffers and
    # the kernel variant baked into the captured graph. Whether the penalty
    # actually runs on a given step is settled per step from the resident
    # requests' `repetition_penalty` — see `SamplerResource.admit`.
    enable_repetion_penalty: bool = True

    @property
    def resource_type(self):
        return ResourceType.SAMPLER


@dataclass
class SamplingReqConfig(ResourceReqConfig):
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 1
    ignore_eos: bool = False # used for benchmark parity
    repetition_penalty: float = 1
    _seed: int = 0 # set by the conductor

    @property
    def resource_type(self):
        return ResourceType.SAMPLER

    def apply_conductor_config(
        self, seed: int=0,
        **kwargs
    ):
        self._seed = seed

    @property
    def seed(self):
        return self._seed


class SamplerResource(Resource):
    # TODO: this is  a light wrapper around mstar/engine/resources/sampler/utils.py. In the future,
    # we should rip out the parts we need from sampling.py and discard the rest.
    def __init__(
        self,
        vocab_size: int | None,
        enable_repetion_penalty: bool,
        device: torch.device,
        comm_group: JointGroups | None=None
    ):
        self._track_seen_tokens = enable_repetion_penalty
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
        self._cg_sampler: CudaGraphableSampler | None = None
        # pre-planned a step ahead, promoted by the next non-preplan plan
        self._preplan_cg_sampler: CudaGraphableSampler | None = None
        self._preplanned = False

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
    def build(
        cls, spec: SamplerSpec,
        device: torch.device,
        joint_comm_group: JointGroups | None,
        **engine_kwargs
    ):
        return cls(
            vocab_size=spec.vocab_size,
            enable_repetion_penalty=spec.enable_repetion_penalty,
            device=device,
            comm_group=joint_comm_group
        )

    def build_cuda_graph_buffers(
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
        self._cg_max_bs = max_bs
        self._cg_slots = max(cg_slots, self._cg_slots)
        self._cg_buffers = SamplerBuffers.allocate(
            max_batch_size=max_bs, device=self._device,
            tp_group=self._comm_group,
            vocab_size=self._vocab_size,
            cg_slots=self._cg_slots,
        )

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
        if self._sampler._sampling_config[rid].repetition_penalty != 1.0:
            self._penalty_rids.add(rid)
        if self._cg_buffers is not None:
            self._cg_buffers.register_request(
                rid, sampling_config=self._sampler._sampling_config[rid]
            )

    def remove_request(self, rid: str):
        self._sampler.remove_request(rid)
        self._penalty_rids.discard(rid)
        if self._cg_buffers is not None:
            self._cg_buffers.unregister_request(rid)

    def admit(self, step: SamplerStep, ctx: StepContext) -> AdmitOutcome:
        """Settle this step's two penalty flags. Reserves nothing.

        Both are read by `plan`, `commit` and `sample`, all of which run on the
        non-preplan path — and `_drive_step` orders `admit` ahead of them, after
        the previous step's `commit`. So one pair of flags is always the right
        one and needs no preplan twin the way `_cg_sampler` does; the preplan
        path only reaches `gather_static`, which never touches the mask.

        Latching here rather than reading `_penalty_live` at each use also keeps
        one step internally consistent when a request is removed mid-step.
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
        return ADMIT_OK

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

    def clear_preplan(self):
        self._preplanned = False
        self._preplan_cg_sampler = None

    def plan(self, step: SamplerStep, ctx: StepContext):
        # The flags themselves are settled in `admit`. Recording prompt tokens
        # is only worth it while something reads the mask: a request at penalty
        # 1.0 whose prompt went unrecorded is unaffected, and a penalised
        # request is resident (so `_penalty_live` is set) before its own prefill
        # declares these tokens.
        if not ctx.is_preplan and self._penalty_live:
            for rid, tokens in step.prefill_tracked_tokens.items():
                self._sampler.get_token_mask(rid).add_tokens(tokens)

        # A step planned ahead promotes here. Its static config was gathered in
        # the preplan; the per-step state (RNG offset + seen-token mask) is NOT
        # double-buffered — it must reflect the previous step's commit, so
        # gather it inline now, on the default stream, after that commit.
        if self._preplanned and not ctx.is_preplan:
            self._gather_dynamic(ctx)
            self._cg_sampler = self._preplan_cg_sampler
            self._preplan_cg_sampler = None
            self._preplanned = False
            return

        # invalidated on the inline path here (not in commit, which now runs
        # before output collection); a preplan must leave the in-flight one be
        if not ctx.is_preplan:
            self._cg_sampler = None
        if self._cg_buffers is None or ctx.slot_lease is None:
            return

        cg_slot = ctx.slot_lease.slot
        padded_bs = ctx.slot_lease.bucket.bs
        # static config only: safe to pre-plan (unchanged step to step). A
        # preplan leases a different slot, so this is disjoint from the in-flight
        # forward's buffers.
        self._cg_buffers.gather_static(ctx.request_ids, padded_bs, cg_slot)
        sampler = self._cg_buffers.sampler_for(padded_bs, cg_slot)
        if ctx.is_preplan:
            self._preplan_cg_sampler = sampler
            self._preplanned = True
        else:
            # fresh inline (capture / no preplan): gather the per-step state too
            self._gather_dynamic(ctx)
            self._cg_sampler = sampler

    def _gather_dynamic(self, ctx: StepContext):
        """Gather the per-step RNG offset + seen-token mask into this step's
        slot, inline so they reflect the previous step's committed tokens.

        The mask half is the expensive half and is skipped whenever the penalty
        is inert this step — see the flags in `__init__`. The RNG offset is
        always gathered: it advances in-graph every step regardless."""
        cg_slot = ctx.slot_lease.slot
        padded_bs = ctx.slot_lease.bucket.bs
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
        self._cg_buffers.scatter_offset(ctx.slot_lease.slot)
        # Skipped when nothing read the mask this step: there is then nothing to
        # copy back, and the rows go stale only for requests at penalty 1.0,
        # which never read them. See `_penalty_live` for why that stays sound.
        if self._penalty_needed_this_step:
            self._cg_sampler.sync_seen_token_masks(
                [self._sampler.get_token_mask(rid) for rid in ctx.request_ids]
            )

    ### Submodule-level functionality

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
        return self._sampler.sample(request_ids, logits)
