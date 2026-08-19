

from dataclasses import asdict, dataclass

import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import Resource
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig, ResourceType
from mstar.engine.resources.step import AdmitOutcome, SamplerStep, StepContext
from mstar.utils.sampling import CudaGraphableSampler, Sampler, SamplerBuffers


@dataclass
class SamplerSpec(NodeResourceSpec):
    vocab_size: int | None # must be set for enabled_repetion_penalty
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
    # TODO: this is  a light wrapper around mstar/utils/sampling.py. In the future,
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
        self._apply_penalty_this_step: bool = enable_repetion_penalty
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
        if self._cg_buffers is not None:
            self._cg_buffers.register_request(
                rid, sampling_config=self._sampler._sampling_config[rid]
            )

    def remove_request(self, rid: str):
        self._sampler.remove_request(rid)
        if self._cg_buffers is not None:
            self._cg_buffers.unregister_request(rid)

    def admit(self, step, ctx):
        del ctx, step
        return AdmitOutcome(ok=True)

    @property
    def supports_preplan(self):
        return True

    def clear_preplan(self):
        self._preplanned = False
        self._preplan_cg_sampler = None

    def plan(self, step: SamplerStep, ctx: StepContext):
        for rid, tokens in step.prefill_tracked_tokens.items():
            self._sampler.get_token_mask(rid).add_tokens(tokens)
        self._apply_penalty_this_step = step.apply_penalty

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
        slot, inline so they reflect the previous step's committed tokens."""
        cg_slot = ctx.slot_lease.slot
        padded_bs = ctx.slot_lease.bucket.bs
        if self._apply_penalty_this_step:
            self._cg_buffers.stage_seen_token_masks(
                request_ids=ctx.request_ids,
                seen_masks=[self._sampler.get_token_mask(rid) for rid in ctx.request_ids]
            )
        self._cg_buffers.gather_dynamic(
            ctx.request_ids, padded_bs, cg_slot,
            gather_seen_tokens=self._track_seen_tokens,
        )

    def commit(self, step: SamplerStep, ctx: StepContext):
        # None on an eager step, which never gathered one
        if self._cg_buffers is None or self._cg_sampler is None:
            return
        self._cg_buffers.scatter_offset(ctx.slot_lease.slot)
        self._cg_sampler.sync_seen_token_masks(
            [self._sampler.get_token_mask(rid) for rid in ctx.request_ids]
        )

    ### Submodule-level functionality

    def sample(
        self, request_ids: list[str], logits: torch.Tensor, **kwargs
    ):
        if self._cg_sampler is not None:
            return self._cg_sampler.sample(
                request_ids, logits,
                apply_penalty=self._apply_penalty_this_step
            )
        return self._sampler.sample(request_ids, logits)
