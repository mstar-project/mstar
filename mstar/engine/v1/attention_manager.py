

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import torch

from mstar.engine.resources.base import CGSlotKey, Resource
from mstar.engine.resources.spec import NodeResourceSpec, ResourceType
from mstar.engine.resources.step import AttentionStep, SlotLease, StepContext
from mstar.engine.v1.attention_wrappers import FlashInferDecodeWrapper, FlashInferPrefillWrapper
from mstar.engine.v1.kv_cache import KVConfig
from mstar.engine.v1.kv_manager import KVPlanOutput, PagedIndptrs, SequenceView


class AttnBackend(Enum):
    FLASHINFER = "flashinfer"

@dataclass
class AttentionConfig:
    kv_cache: str # name of the KV cache
    backend: AttnBackend=AttnBackend.FLASHINFER
    flashinfer_backend: str = "auto"


@dataclass
class AttentionSpec(NodeResourceSpec):
    config: AttentionConfig
    kv_config: KVConfig

    @property
    def resource_type(self):
        return ResourceType.ATTENTION


class AttentionManager(Resource):
    # Remains abstract except for build; will build based
    # on the attention backend

    @classmethod
    def build(
        cls, spec: AttentionSpec,
        device: torch.device,
        dtype=torch.bfloat16,
        **engine_kwargs
    ):
        if spec.config.backend == AttnBackend.FLASHINFER:
            return FlashInferManager(
                kv_cache=spec.config.kv_cache,
                device=device,
                dtype=dtype,
                kv_config=spec.kv_config,
                backend=spec.config.flashinfer_backend,
            )


class PlanCacheKey(NamedTuple):
    """Fingerprint of a wrapper ``plan`` call's inputs. When it is unchanged
    between steps the re-plan is skippable."""
    q_seq_lens: tuple
    page_indices: tuple
    last_page_lens: tuple


AttentionWrapper = FlashInferPrefillWrapper | FlashInferDecodeWrapper


class FlashInferManager(AttentionManager):
    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: KVConfig,
        backend: str="auto",
    ):
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype

        # label to plan state
        self._current_plan_states: dict[str, AttentionWrapper] = {}

        self._cg_plan_states: dict[CGSlotKey, AttentionWrapper] = {}

        self._preplan_states: dict[str, AttentionWrapper] = {}
        self._preplanned = False

        self._kv_config = kv_config
        self._wrapper_kv_kwargs = dict(
            num_qo_heads=kv_config.num_qo_heads,
            num_kv_heads=kv_config.num_kv_heads,
            head_dim=kv_config.head_dim,
            page_size=kv_config.page_size,
            max_num_pages=kv_config.max_num_pages,
            device=device,
            backend=backend
        )

        self._buffer_size = int(
            os.environ.get("MSTAR_WORKSPACE_BUFFER_MB", "512")
        ) * 1024 * 1024
        self._workspace_buffers: dict[str, torch.Tensor] = {}

    def depends_on(self):
        return {self._kv_cache_name}

    def _get_workspace_buffer(
        self, label: str, cg_slot: int | None=None
    ):
        key = label if cg_slot is None else f"{label}_{cg_slot}"
        if key not in self._workspace_buffers:
            self._workspace_buffers[key] = torch.empty(
                self._buffer_size, dtype=torch.uint8, device=self._device
            )
        return self._workspace_buffers[key]

    def _cg_wrapper(self, lease: SlotLease, label: str) -> AttentionWrapper:
        """The captured-graph wrapper for one (bucket, slot, label).

        Built on the first plan for that key rather than up front: which labels
        a walk plans under is the step's to declare, and the first plan for a
        bucket runs during capture, before the graph is recorded, so the wrapper
        is just as persistent either way.
        """
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        wrapper = self._cg_plan_states.get(key)
        if wrapper is not None:
            return wrapper

        bucket = lease.bucket
        buffer = self._get_workspace_buffer(label, lease.slot)
        if bucket.bs == bucket.num_tokens:
            wrapper = FlashInferDecodeWrapper(
                workspace_buffer=buffer,
                batch_size=bucket.bs,
                use_cuda_graph=True,
                **self._wrapper_kv_kwargs,
            )
        else:
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=buffer,
                batch_size=bucket.bs,
                max_total_tokens=bucket.num_tokens,
                use_cuda_graph=True,
                **self._wrapper_kv_kwargs,
            )
        self._cg_plan_states[key] = wrapper
        return wrapper

    @property
    def supports_preplan(self):
        return True

    def plan(self, step: AttentionStep, ctx: StepContext):
        assert not ctx.is_preplan or ctx.slot_lease is not None, (
            "preplan requires a cuda graph step: eager wrappers share one "
            "workspace per label with the forward still in flight"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "attention preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        if self._preplanned:
            # the wrappers were planned a step early against this same step's
            # KV plan; nothing left to do but promote them
            self._current_plan_states = self._preplan_states
            self._preplan_states = {}
            self._preplanned = False
            return

        plan_outputs: dict[str, KVPlanOutput] = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, f"Attention Manager expected plan result from {self._kv_cache_name}"

        # a preplan leases a different cg slot, so its wrappers and workspaces
        # are disjoint from the ones the in-flight forward reads
        plan_states = self._preplan_states if ctx.is_preplan else self._current_plan_states

        plan_states.clear()
        for label, kv_out in plan_outputs.items():
            indptrs = kv_out.cpu_indptrs
            if ctx.slot_lease is not None:
                wrapper = self._cg_wrapper(ctx.slot_lease, label)
            else:
                is_decode = indptrs.qo_indptr[-1] == len(indptrs.qo_indptr) - 1
                buffer = self._get_workspace_buffer(label)
                if is_decode:
                    wrapper = FlashInferDecodeWrapper(
                        workspace_buffer=buffer,
                        **self._wrapper_kv_kwargs,
                    )
                else:
                    wrapper = FlashInferPrefillWrapper(
                        workspace_buffer=buffer,
                        **self._wrapper_kv_kwargs,
                    )

            # TODO: cache the latest plan state
            wrapper.plan(
                causal=step.causal,
                dtype=self._dtype,
                **indptrs.to_kwargs_dict()
            )
            plan_states[label] = wrapper

        self._preplanned = ctx.is_preplan

    def clear_preplan(self):
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}

    ### Submodule-level functionality
    def qo_indptr_buf(self, label: str) -> torch.Tensor | None:
        if label not in self._current_plan_states:
            return
        # decode wrappers never set one
        return getattr(self._current_plan_states[label], "_qo_indptr_buf", None)

    def run(
        self, q: torch.Tensor, label: str,
        kv_cache_layer: torch.Tensor
    ) -> torch.Tensor:
        return self._current_plan_states[label].run(q, kv_cache_layer)


@dataclass
class CrossAttentionConfig:
    """Cross-attention against a context written once and never extended.

    ``kv_cache`` names the KV resource holding the encoder context;
    ``query_kv_cache`` names the decoder's KV resource, whose plan defines
    this step's query packing. They may be the same resource when the
    context shares the decoder's head config — the context then lives in it
    under its own ``context_label``. They differ when it does not, which is
    the usual case (an encoder's head count rarely matches the decoder's).
    """
    kv_cache: str  # name of the KV cache holding the context
    query_kv_cache: str  # name of the KV cache driving the queries
    context_label: str = "context"
    backend: AttnBackend = AttnBackend.FLASHINFER
    flashinfer_backend: str = "auto"


@dataclass
class CrossAttentionSpec(NodeResourceSpec):
    config: CrossAttentionConfig
    # head config of the *context* cache, which need not match the decoder's
    kv_config: KVConfig

    @property
    def resource_type(self):
        return ResourceType.CROSS_ATTENTION


class CrossAttentionManager(Resource):
    # Remains abstract except for build; will build based
    # on the attention backend

    @classmethod
    def build(
        cls, spec: CrossAttentionSpec,
        device: torch.device,
        dtype=torch.bfloat16,
        **engine_kwargs
    ):
        if spec.config.backend == AttnBackend.FLASHINFER:
            return FlashInferCrossManager(
                kv_cache=spec.config.kv_cache,
                query_kv_cache=spec.config.query_kv_cache,
                context_label=spec.config.context_label,
                device=device,
                dtype=dtype,
                kv_config=spec.kv_config,
                backend=spec.config.flashinfer_backend,
            )


class FlashInferCrossManager(CrossAttentionManager):
    """non-causal paged attention over fixed encoder context

    context is an ordinary kv stream so requires no additional storage
    or allocation on part of attention manager

    encode step writes context via normal `Segment` on cache, and later
    steps use `span=0` on that label

    wrapper is planned per decoder plan label and run is keyed on that label
    a la `FlashInferManager`"""

    def __init__(
        self,
        kv_cache: str,
        query_kv_cache: str,
        context_label: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: KVConfig,
        backend: str = "auto",
    ):
        self._kv_cache_name = kv_cache
        self._query_kv_cache_name = query_kv_cache
        self._context_label = context_label
        self._device = device
        self._dtype = dtype

        # decoder plan label -> wrapper
        self._current_plan_states: dict[str, AttentionWrapper] = {}
        # persistent wrappers [we can keep eager ones]
        self._eager_plan_states: dict[str, AttentionWrapper] = {}
        self._cg_plan_states: dict[CGSlotKey, AttentionWrapper] = {}

        self._preplan_states: dict[str, AttentionWrapper] = {}
        self._preplanned = False
        # wrapper key (plan label, or CGSlotKey under capture) -> fingerprint
        # of the last plan call made on that wrapper
        # NOTE: a little bit (very) sketchy should be refined; goal is to avoid
        # redundant plan comp. how to do the mapping to track is another issue
        self._plan_keys: dict[Any, PlanCacheKey] = {}

        self._kv_config = kv_config
        self._wrapper_kv_kwargs = dict(
            num_qo_heads=kv_config.num_qo_heads,
            num_kv_heads=kv_config.num_kv_heads,
            head_dim=kv_config.head_dim,
            page_size=kv_config.page_size,
            max_num_pages=kv_config.max_num_pages,
            device=device,
            backend=backend
        )

        self._buffer_size = int(
            os.environ.get("MSTAR_WORKSPACE_BUFFER_MB", "512")
        ) * 1024 * 1024
        self._workspace_buffers: dict[str, torch.Tensor] = {}

    def depends_on(self):
        return {self._kv_cache_name, self._query_kv_cache_name}

    def _get_workspace_buffer(
        self, label: str, cg_slot: int | None=None
    ):
        key = label if cg_slot is None else f"{label}_{cg_slot}"
        if key not in self._workspace_buffers:
            self._workspace_buffers[key] = torch.empty(
                self._buffer_size, dtype=torch.uint8, device=self._device
            )
        return self._workspace_buffers[key]

    @property
    def supports_preplan(self):
        return True

    def plan(self, step: AttentionStep, ctx: StepContext):
        del step  # non-causal
        assert not ctx.is_preplan or ctx.slot_lease is not None, (
            "preplan requires a cuda graph step: the eager wrapper for a label "
            "persists and would be replanned under the in-flight forward"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "cross attention preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        if self._preplanned:
            self._current_plan_states = self._preplan_states
            self._preplan_states = {}
            self._preplanned = False
            return

        context_views = self._context_views(ctx)
        # a preplan leases a different cg slot, so its wrappers and workspaces
        # are disjoint from the ones the in-flight forward reads
        plan_states = self._preplan_states if ctx.is_preplan else self._current_plan_states

        plan_states.clear()
        for plan_label, kv_out in self._query_plan_outputs(ctx).items():
            indptrs = self._build_indptrs(kv_out, context_views, plan_label)
            state_key, wrapper = self._wrapper_for(plan_label, ctx)

            # The context pages are immutable once written, so between steps
            # only the query side moves; when neither moved the whole plan is
            # skippable. Every tensor here is on CPU, so `.tolist()` costs no
            # sync.

            # context is immutable; between step only q will change; when nothing
            # changes we can skip plan (use old)
            plan_key = PlanCacheKey(
                q_seq_lens=tuple(indptrs.qo_indptr.tolist()),
                page_indices=tuple(indptrs.paged_kv_indices.tolist()),
                last_page_lens=tuple(indptrs.paged_kv_last_page_len.tolist()),
            )
            if self._plan_keys.get(state_key) != plan_key:
                wrapper.plan(
                    causal=False,
                    dtype=self._dtype,
                    **indptrs.to_kwargs_dict()
                )
                self._plan_keys[state_key] = plan_key
            plan_states[plan_label] = wrapper

        self._preplanned = ctx.is_preplan

    def clear_preplan(self):
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}

    def _context_views(self, ctx: StepContext) -> dict[str, SequenceView]:
        """context stream of each request in batch by request id

        key by request rather than zip with query view because not
        necessarily same length"""
        plan_outputs: dict[str, KVPlanOutput] = ctx.plan_results.get(
            self._kv_cache_name
        )
        assert plan_outputs is not None, (
            f"Cross Attention Manager expected plan result from {self._kv_cache_name}"
        )
        kv_out = plan_outputs.get(self._context_label)
        assert kv_out is not None, (
            f"Cross Attention Manager found no plan for context label "
            f"{self._context_label!r} in {self._kv_cache_name!r}; every step "
            "that attends the context must declare a zero-span segment on "
            f"that label. planned: {sorted(plan_outputs)}"
        )
        return {view.request_id: view for view in kv_out.views}

    def _query_plan_outputs(self, ctx: StepContext) -> dict[str, KVPlanOutput]:
        plan_outputs: dict[str, KVPlanOutput] = ctx.plan_results.get(
            self._query_kv_cache_name
        )
        assert plan_outputs is not None, (
            f"Cross Attention Manager expected plan result from "
            f"{self._query_kv_cache_name}"
        )
        if self._query_kv_cache_name != self._kv_cache_name:
            return plan_outputs

        # cache holds both q, k. context is in k, not in own q stream
        return {
            label: kv_out for label, kv_out in plan_outputs.items()
            if label != self._context_label
        }

    def _build_indptrs(
        self,
        query_out: KVPlanOutput,
        context_views: dict[str, SequenceView],
        plan_label: str,
    ) -> PagedIndptrs:
        """key off context, query off decoder

        reuse of decoder `qo_indpter` keeps the label major order produced by
        combined CFG plan."""
        page_size = self._kv_config.page_size
        kv_indptr = [0]
        all_pages: list[int] = []
        last_page_lens: list[int] = []

        for view in query_out.views:
            context = context_views.get(view.request_id)
            assert context is not None, (
                f"no cross attention context for request {view.request_id!r} "
                f"under label {self._context_label!r}"
            )
            assert context.length > 0, (
                f"cross attention context for request {view.request_id!r} is "
                "empty; it must be written before the step that attends it"
            )
            # stream can have more pages than context needs takes only the prefix
            # written length covers
            n_pages = (context.length + page_size - 1) // page_size
            all_pages.extend(context.page_idxs[:n_pages])
            kv_indptr.append(kv_indptr[-1] + n_pages)
            last_page_lens.append(context.last_page_len(page_size) or page_size)

        # fork repeats pages, so batch can index more than held in cache.
        assert len(all_pages) <= self._kv_config.max_num_pages, (
            f"cross attention plan {plan_label!r} indexes {len(all_pages)} "
            f"pages but the context cache holds {self._kv_config.max_num_pages}; "
            "raise max_num_pages on the context cache's KVConfig"
        )

        return PagedIndptrs(
            qo_indptr=query_out.cpu_indptrs.qo_indptr,
            paged_kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32),
            paged_kv_indices=torch.tensor(all_pages, dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor(last_page_lens, dtype=torch.int32),
        )

    def _wrapper_for(
        self, plan_label: str, ctx: StepContext
    ) -> tuple[Any, AttentionWrapper]:
        if ctx.slot_lease is not None:
            lease = ctx.slot_lease
            key = CGSlotKey(
                bucket=lease.bucket, slot=lease.slot, label=plan_label
            )
            wrapper = self._cg_plan_states.get(key)
            if wrapper is None:
                # built on first plan for the key; see FlashInferManager._cg_wrapper
                wrapper = self._cg_plan_states[key] = FlashInferPrefillWrapper(
                    workspace_buffer=self._get_workspace_buffer(
                        plan_label, lease.slot
                    ),
                    batch_size=lease.bucket.bs,
                    max_total_tokens=lease.bucket.num_tokens,
                    use_cuda_graph=True,
                    **self._wrapper_kv_kwargs,
                )
            return key, wrapper

        wrapper = self._eager_plan_states.get(plan_label)
        if wrapper is None:
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=self._get_workspace_buffer(plan_label),
                **self._wrapper_kv_kwargs,
            )
            self._eager_plan_states[plan_label] = wrapper
        return plan_label, wrapper



    def run(
        self, q: torch.Tensor, label: str,
        kv_cache_layer: torch.Tensor
    ) -> torch.Tensor:
        """One layer's cross attention. ``kv_cache_layer`` is a layer of the
        *context* cache; nothing is written to it here."""
        return self._current_plan_states[label].run(q, kv_cache_layer)


# TODO: dense attention
