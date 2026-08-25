

import functools
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import AttentionResource, CGSlotKey
from mstar.engine.resources.spec import NodeResourceSpec, ResourceType
from mstar.engine.resources.step import AttentionStep, SlotLease, StepContext
from mstar.engine.resources.attn.wrappers import FlashInferDecodeWrapper, FlashInferPrefillWrapper
from mstar.engine.resources.kv.cache import KVConfig
from mstar.engine.resources.kv.manager import (
    SINK_PAGE,
    KVPlanOutput,
    KVPlanOutputs,
    PagedIndptrs,
    SequenceView,
)

logger = logging.getLogger(__name__)


class AttnBackend(Enum):
    FLASHINFER = "flashinfer"
    DENSE = "dense"

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

    def apply_yaml_overrides(
        self,
        max_num_pages: int | None = None,
        page_size: int | None = None,
        **kwargs,
    ):
        """Track the cache's geometry: the wrappers are planned against it, so
        a deployment that resizes the cache resizes these too."""
        del kwargs  # keys meant for other resources
        if max_num_pages is not None:
            self.kv_config.max_num_pages = max_num_pages
        if page_size is not None:
            self.kv_config.page_size = page_size


class AttentionManager(AttentionResource):
    # Remains abstract except for build; will build based
    # on the attention backend

    # Label / layer cursors come from `AttentionResource`; `run` resolves them.

    @classmethod
    def build(
        cls, spec: AttentionSpec,
        device: torch.device,
        joint_comm_group: JointGroups | None = None,
        dtype=torch.bfloat16,
        **engine_kwargs
    ):
        # the wrappers are planned against per-rank head counts; idempotent, so
        # sharing one KVConfig with the KV resource is fine (KVConfig.shard)
        if joint_comm_group is not None:
            spec.kv_config.shard(joint_comm_group.world_size)
        backend = spec.config.backend
        if backend == AttnBackend.DENSE:
            # A dense backend needs the FlashAttention-3 kernel; where the
            # wheel does not match the installed torch/CUDA build, degrade to
            # the paged backend rather than failing to serve. The two are
            # drop-in for each other: the step declaration is identical
            # (see DenseAttentionManager) and `requires_kv_write` tells the
            # layer which one it is talking to.
            reason = _fa3_unavailable_reason()
            if reason is None:
                return DenseAttentionManager(
                    kv_cache=spec.config.kv_cache,
                    device=device,
                    dtype=dtype,
                    kv_config=spec.kv_config,
                )
            _warn_dense_fallback(reason)
            backend = AttnBackend.FLASHINFER

        if backend == AttnBackend.FLASHINFER:
            return FlashInferManager(
                kv_cache=spec.config.kv_cache,
                device=device,
                dtype=dtype,
                kv_config=spec.kv_config,
                backend=spec.config.flashinfer_backend,
            )
        raise ValueError(f"Unknown attention backend {backend!r}")

    @property
    def requires_kv_write(self) -> bool:
        """Whether a layer must write this step's K/V through the KV resource
        before calling ``run``.

        False only for backends that take the fresh K/V straight into the
        kernel (the dense one). A layer therefore reads

            if self.attn.requires_kv_write:
                self.kv.write_kv(k, v, layer_idx=i, label=label)
            out = self.attn.run(q, label, kv.layer_view(i), k=k, v=v, layer_idx=i)

        and stays correct whichever backend the spec named.
        """
        return True


class PlanCacheKey(NamedTuple):
    """Fingerprint of a wrapper ``plan`` call's inputs. When it is unchanged
    between steps the re-plan is skippable."""
    q_seq_lens: tuple
    page_indices: tuple
    last_page_lens: tuple


AttentionWrapper = FlashInferPrefillWrapper | FlashInferDecodeWrapper


# Attention resources reachable from a custom op. A traced forward can't pass
# the resource itself into an op (a schema takes tensors and scalars), so it
# passes this handle. One entry per resource, fixed for the process, so dynamo
# specializing on it costs nothing — a handle that varied per step or per slot
# would reintroduce the recompiles this exists to avoid.
_ATTENDERS: dict[int, "AttentionManager"] = {}


def _register_attender(manager: "AttentionManager") -> int:
    """Give an attention resource a handle its layers can pass into the op."""
    handle = len(_ATTENDERS)
    _ATTENDERS[handle] = manager
    return handle


@torch.library.custom_op("mstar::flashinfer_attend", mutates_args=())
def flashinfer_attend(
    handle: int, label: str, q: torch.Tensor, kv_cache_layer: torch.Tensor,
) -> torch.Tensor:
    """One layer's attention, planned by the resource behind ``handle``.

    Behind an op because FlashInfer's kernel is a TVM-FFI call dynamo can't
    trace and can't run on fake tensors: called directly it breaks the graph
    once per layer, and each break makes the layer body a frame dynamo
    recompiles per ``layer_idx``.
    """
    out = _ATTENDERS[handle].attend(q, label, kv_cache_layer)
    # the planned wrapper runs in its own dtype; the fake below promises q's,
    # and a mismatch there is silent corruption
    return out.to(q.dtype)


@flashinfer_attend.register_fake
def _flashinfer_attend_fake(
    handle: int, label: str, q: torch.Tensor, kv_cache_layer: torch.Tensor,
) -> torch.Tensor:
    # attention is shape-preserving on the query
    return torch.empty_like(q)


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
        self._attend_handle = _register_attender(self)

        # label to plan state
        self._current_plan_states: dict[str, AttentionWrapper] = {}

        # (label, is_decode) -> wrapper; persistent, because constructing one
        # allocates FlashInfer's own workspaces and is far from free per step
        self._eager_plan_states: dict[tuple[str, bool], AttentionWrapper] = {}
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

    def _cg_wrapper(self, lease: SlotLease, label: str, num_rows: int) -> AttentionWrapper:
        """The captured-graph wrapper for one (bucket, slot, label).

        Built on the first plan for that key rather than up front: which labels
        a walk plans under is the step's to declare, and the first plan for a
        bucket runs during capture, before the graph is recorded, so the wrapper
        is just as persistent either way.

        ``num_rows`` is the qo_indptr row count this label actually plans —
        the FlashInfer batch_size for CUDA-graph mode, which must stay fixed
        for the wrapper's lifetime. It equals ``bucket.bs`` (one row per
        request) for an ordinary label, but a label that combines several
        source labels into one plan (e.g. cond+uncond batched CFG) always
        contributes several rows per request; using ``bucket.bs`` there
        under-allocates the wrapper's fixed buffers and every later plan()
        call overruns them with a "runtime batch size" mismatch.
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
                batch_size=num_rows,
                use_cuda_graph=True,
                **self._wrapper_kv_kwargs,
            )
        else:
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=buffer,
                batch_size=num_rows,
                max_total_tokens=bucket.num_tokens,
                use_cuda_graph=True,
                **self._wrapper_kv_kwargs,
            )
        self._cg_plan_states[key] = wrapper
        return wrapper

    def _eager_wrapper(self, label: str, is_decode: bool) -> AttentionWrapper:
        """The persistent eager wrapper for one (label, kind)."""
        key = (label, is_decode)
        wrapper = self._eager_plan_states.get(key)
        if wrapper is None:
            cls = FlashInferDecodeWrapper if is_decode else FlashInferPrefillWrapper
            wrapper = self._eager_plan_states[key] = cls(
                workspace_buffer=self._get_workspace_buffer(label),
                **self._wrapper_kv_kwargs,
            )
        return wrapper

    @property
    def supports_preplan(self):
        return True

    def plan(self, step: AttentionStep, ctx: StepContext):
        self.reset_default_cursors()
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

        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, f"Attention Manager expected plan result from {self._kv_cache_name}"

        # a preplan leases a different cg slot, so its wrappers and workspaces
        # are disjoint from the ones the in-flight forward reads
        plan_states = self._preplan_states if ctx.is_preplan else self._current_plan_states

        plan_states.clear()
        for label, kv_out in plan_outputs.items():
            indptrs = kv_out.cpu_indptrs
            if ctx.slot_lease is not None:
                num_rows = indptrs.qo_indptr.shape[0] - 1
                wrapper = self._cg_wrapper(ctx.slot_lease, label, num_rows)
            else:
                is_decode = bool(
                    indptrs.qo_indptr[-1] == len(indptrs.qo_indptr) - 1
                )
                wrapper = self._eager_wrapper(label, is_decode)

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
    def qo_indptr_buf(self, label: str="main") -> torch.Tensor | None:
        if label not in self._current_plan_states:
            return
        # decode wrappers never set one
        return getattr(self._current_plan_states[label], "_qo_indptr_buf", None)

    def select_last_hidden(
        self, hidden: torch.Tensor, label: str="main"
    ) -> torch.Tensor:
        """
        Select last token of the hidden vector per request, used for sampling
        from prefill.
        """
        qo_indptr_buf = self.qo_indptr_buf(label)
        last_token_indices = (qo_indptr_buf[1:] - 1).long()
        return hidden.index_select(0, last_token_indices)

    def run(
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        # `k`/`v`/`layer_idx` are the dense backend's; here the K/V is already
        # in the pages (the layer wrote it through the KV resource) and the
        # planned wrapper carries the layout. Accepted and ignored so one
        # layer body serves either backend — see `requires_kv_write`.
        del k, v, layer_idx
        if label is None:
            label = self._default_label
        return torch.ops.mstar.flashinfer_attend(
            self._attend_handle, label, q, kv_cache_layer,
        )

    def attend(
        self, q: torch.Tensor, label: str, kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        """The eager body of ``run``, called from inside the op."""
        return self._current_plan_states[label].run(q, kv_cache_layer)


class QueryPacking(NamedTuple):
    """One plan label's query side: the requests in packed order, and the
    cumulative query lengths over them."""
    request_ids: list[str]
    qo_indptr: torch.Tensor  # int32, CPU


@dataclass
class CrossAttentionConfig:
    """Cross-attention against a context written once and never extended.

    ``kv_cache`` names the KV resource holding the encoder context;
    ``query_kv_cache`` names the decoder's KV resource, whose plan defines
    this step's query packing. They may be the same resource when the
    context shares the decoder's head config — the context then lives in it
    under its own ``context_label``. They differ when it does not, which is
    the usual case (an encoder's head count rarely matches the decoder's).

    ``query_kv_cache=None`` covers the query side having no KV cache at all
    (nothing is cached across steps on it): the packing then comes off the
    cross-attention step's own segments, one qo entry per segment in
    declared order.
    """
    kv_cache: str  # name of the KV cache holding the context
    query_kv_cache: str | None = None  # KV cache driving the queries, if any
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

    def apply_yaml_overrides(
        self,
        max_num_pages: int | None = None,
        page_size: int | None = None,
        **kwargs,
    ):
        """Track the context cache's geometry; see ``AttentionSpec``."""
        del kwargs  # keys meant for other resources
        if max_num_pages is not None:
            self.kv_config.max_num_pages = max_num_pages
        if page_size is not None:
            self.kv_config.page_size = page_size


class CrossAttentionManager(AttentionResource):
    # Remains abstract except for build; will build based
    # on the attention backend

    @classmethod
    def build(
        cls, spec: CrossAttentionSpec,
        device: torch.device,
        joint_comm_group: JointGroups | None = None,
        dtype=torch.bfloat16,
        **engine_kwargs
    ):
        # the context cache's own head counts; see AttentionManager.build
        if joint_comm_group is not None:
            spec.kv_config.shard(joint_comm_group.world_size)
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
        query_kv_cache: str | None,
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
        self._attend_handle = _register_attender(self)

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
        # a query side with no cache contributes no dependency: its packing
        # comes off the step, not another resource's plan
        if self._query_kv_cache_name is None:
            return {self._kv_cache_name}
        return {self._kv_cache_name, self._query_kv_cache_name}

    @property
    def context_cache_key(self) -> str:
        """label of kv resource holding context so x-attention can find cache to read"""
        return self._kv_cache_name

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
        self.reset_default_cursors()
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
        for plan_label, packing in self._query_packings(step, ctx).items():
            indptrs = self._build_indptrs(packing, context_views, plan_label)
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
        plan_outputs: KVPlanOutputs = ctx.plan_results.get(
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

    def _query_packings(
        self, step: AttentionStep, ctx: StepContext
    ) -> dict[str, QueryPacking]:
        if self._query_kv_cache_name is None:
            return self._step_query_packings(step)

        plan_outputs: KVPlanOutputs = ctx.plan_results.get(
            self._query_kv_cache_name
        )
        assert plan_outputs is not None, (
            f"Cross Attention Manager expected plan result from "
            f"{self._query_kv_cache_name}"
        )
        if self._query_kv_cache_name == self._kv_cache_name:
            # cache holds both q, k. context is in k, not in own q stream
            plan_outputs = {
                label: kv_out for label, kv_out in plan_outputs.items()
                if label != self._context_label
            }
        return {
            label: QueryPacking(
                request_ids=[view.request_id for view in kv_out.views],
                qo_indptr=kv_out.cpu_indptrs.qo_indptr,
            )
            for label, kv_out in plan_outputs.items()
        }

    def _step_query_packings(self, step: AttentionStep) -> dict[str, QueryPacking]:
        """Query packing off this step's own segments, for a query side with no
        KV cache: one entry per segment, in declared order.

        No ``combined_labels`` here — that grouping lives on ``KVStep``, so a
        cacheless query side plans one label per segment label.
        """
        rids: dict[str, list[str]] = {}
        qo_indptrs: dict[str, list[int]] = {}
        for segment in step.segments or ():
            rids.setdefault(segment.label, []).append(segment.request_id)
            qo = qo_indptrs.setdefault(segment.label, [0])
            qo.append(qo[-1] + segment.span)

        return {
            label: QueryPacking(
                request_ids=label_rids,
                qo_indptr=torch.tensor(qo_indptrs[label], dtype=torch.int32),
            )
            for label, label_rids in rids.items()
        }

    def _build_indptrs(
        self,
        packing: QueryPacking,
        context_views: dict[str, SequenceView],
        plan_label: str,
    ) -> PagedIndptrs:
        """key off context, query off the packing

        reuse of the query `qo_indptr` keeps the label major order produced by
        combined CFG plan."""
        page_size = self._kv_config.page_size
        kv_indptr = [0]
        all_pages: list[int] = []
        last_page_lens: list[int] = []

        for request_id in packing.request_ids:
            context = context_views.get(request_id)
            assert context is not None, (
                f"no cross attention context for request {request_id!r} "
                f"under label {self._context_label!r}"
            )
            if context.length == 0:
                # A capture's padding row never had an encoder context written,
                # and `release` resets the streams after every replay, so this
                # is the normal state for the tail of a padded batch. Plan it
                # against SINK_PAGE: a well-formed row whose output the step
                # discards, rather than failing the whole batch.
                all_pages.append(SINK_PAGE)
                kv_indptr.append(kv_indptr[-1] + 1)
                last_page_lens.append(1)
                continue
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
            qo_indptr=packing.qo_indptr,
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
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One layer's cross attention. ``kv_cache_layer`` is a layer of the
        *context* cache; nothing is written to it here."""
        if label is None:
            label = self._default_label
        return torch.ops.mstar.flashinfer_attend(
            self._attend_handle, label, q, kv_cache_layer,
        )

    def attend(
        self, q: torch.Tensor, label: str, kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        """The eager body of ``run``, called from inside the op."""
        return self._current_plan_states[label].run(q, kv_cache_layer)


@functools.cache
def _fa3_unavailable_reason() -> str | None:
    """verify flash attention 3 fwd kernel imports"""
    try:
        import fa3_fwd_interface  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


@functools.cache
def _warn_dense_fallback(reason: str) -> None:
    logger.warning(
        "Attention backend 'dense' requested but the fa3-fwd kernel is "
        "unavailable (%s); using the paged 'flashinfer' backend instead.",
        reason,
    )


@dataclass(frozen=True)
class DenseSegment:
    """(request, label) of dense plan; frozen prefix and fresh tokens"""
    request_id: str
    label: str
    pages: torch.Tensor  # the prefix's pages, in stream order
    prefix_len: int
    q_len: int
    generation: int  # of the stream, at plan time; see CacheStream.generation


@dataclass(frozen=True)
class DensePlan:
    """dense plan layout. once per step and read by each layer `run`"""
    segments: tuple[DenseSegment, ...]
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int
    causal: bool


@dataclass
class _GatheredPrefix:
    """One stream's gathered prefix, per layer, tagged with what it was read
    from so a stale gather cannot be served."""
    """gathered prefix of one stream per-layer. tagged with what read from so stale cannot serve"""
    generation: int
    length: int
    layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


class DenseAttentionManager(AttentionManager):
    """Attention as one dense FlashAttention-3 varlen pass over a contiguous
    [frozen prefix | fresh tokens] sequence.

    For a workload that recomputes every one of its K/V every step and only
    reuses a small frozen prefix — diffusion denoise, where the prefix is the
    text conditioning — the paged path's per-step full-buffer K/V write and
    ``wrapper.plan`` are pure overhead. This gathers the prefix once per
    (stream, layer), concatenates it with the freshly projected K/V, and runs
    one kernel. The gathered prefix is reused across steps: it is keyed on the
    stream's ``generation``, so a fork, a reset, or a page-table move
    invalidates it (``CacheStream.generation``).

    The layer does not write its K/V to the pages here (``requires_kv_write``
    is False) — nothing would ever read it back.

    Eager-only. The gather and the concatenation are shape-dependent and
    allocate, so there is nothing to capture; a node that runs this walk under
    a graph should name the paged backend for it.

    NOTE on the step declaration: the fresh tokens are declared as ordinary
    spans on ``KVStep`` with ``commit=False``, exactly as they would be for the
    paged backend, so the query lengths arrive as the KV plan's ``to_compute``
    and the prefix as ``length - to_compute``. That is what keeps the backend a
    spec-time choice: the same declaration runs either way, and positions still
    take their packing off the KV plan output. The alternative — declaring the
    fresh tokens zero-span and carrying their lengths on ``AttentionStep``'s
    own segments — is what the v0 dense path did and would save the pages that
    are reserved here and never written; it costs the swappability, since a
    model would then have to declare its step differently per backend, and a
    zero-span label yields no packing for any resource that derives from it
    (``PositionManager`` would only work for labels supplying explicit pos_ids).
    """

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: KVConfig,
    ):
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        self._on_cuda = device.type == "cuda" and torch.cuda.is_available()

        # plan label -> this step's layout
        self._current_plans: dict[str, DensePlan] = {}
        # (request, label) -> the prefix gathered out of the pages
        self._prefix_cache: dict[tuple[str, str], _GatheredPrefix] = {}

    def depends_on(self):
        return {self._kv_cache_name}

    @property
    def requires_kv_write(self) -> bool:
        return False

    # the gathered prefix is per-request state, so unlike the paged backend
    # these are not no-ops
    def ingest_request(self, rid: str, overrides=None):
        del overrides
        self._drop_prefixes(rid)

    def remove_request(self, rid: str):
        self._drop_prefixes(rid)

    def reset_request(self, rid: str, free: bool=False):
        del free
        self._drop_prefixes(rid)

    def _drop_prefixes(self, rid: str) -> None:
        for key in [k for k in self._prefix_cache if k[0] == rid]:
            del self._prefix_cache[key]

    def plan(self, step: AttentionStep, ctx: StepContext):
        self.reset_default_cursors()
        assert ctx.slot_lease is None and not ctx.is_preplan, (
            "dense attention is eager-only: the prefix gather and the "
            "concatenation allocate and are shaped by the step, so there is "
            "nothing for a captured graph to replay"
        )
        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, (
            f"Dense Attention Manager expected plan result from {self._kv_cache_name}"
        )

        self._current_plans = {
            plan_label: self._build_plan(kv_out, step.causal)
            for plan_label, kv_out in plan_outputs.items()
        }

    def _build_plan(self, kv_out: KVPlanOutput, causal: bool) -> DensePlan:
        """The per-segment gather layout plus the cumulative lengths one varlen
        kernel needs, in the packed order the KV plan defined."""
        page_size = self._kv_config.page_size
        segments: list[DenseSegment] = []
        cu_q = [0]
        cu_k = [0]
        max_q = 0
        max_k = 0
        for view in kv_out.views:
            assert view.start == 0, (
                f"dense attention reads a stream from its first page; "
                f"{view.label!r} of {view.request_id!r} starts at {view.start}"
            )
            prefix_len = view.length - view.to_compute
            n_pages = (prefix_len + page_size - 1) // page_size
            segments.append(DenseSegment(
                request_id=view.request_id,
                label=view.label,
                pages=torch.tensor(
                    view.page_idxs[:n_pages], dtype=torch.long, device=self._device
                ),
                prefix_len=prefix_len,
                q_len=view.to_compute,
                generation=view.generation,
            ))
            cu_q.append(cu_q[-1] + view.to_compute)
            cu_k.append(cu_k[-1] + view.length)
            max_q = max(max_q, view.to_compute)
            max_k = max(max_k, view.length)

        return DensePlan(
            segments=tuple(segments),
            cu_q=torch.tensor(cu_q, dtype=torch.int32, device=self._device),
            cu_k=torch.tensor(cu_k, dtype=torch.int32, device=self._device),
            max_q=max_q,
            max_k=max_k,
            causal=causal,
        )

    def _prefix_kv(
        self, segment: DenseSegment, layer_idx: int, kv_cache_layer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This segment's frozen prefix for one layer, gathered out of the
        pages on first use and held until the stream moves under it."""
        key = (segment.request_id, segment.label)
        entry = self._prefix_cache.get(key)
        if (
            entry is None
            or entry.generation != segment.generation
            or entry.length != segment.prefix_len
        ):
            entry = self._prefix_cache[key] = _GatheredPrefix(
                generation=segment.generation, length=segment.prefix_len
            )

        cached = entry.layers.get(layer_idx)
        if cached is not None:
            return cached

        cfg = self._kv_config
        # [n_pages, 2, page_size, num_kv_heads, head_dim] -> the prefix's rows
        pages = kv_cache_layer[segment.pages]
        k_pref = pages[:, 0].reshape(-1, cfg.num_kv_heads, cfg.head_dim)
        v_pref = pages[:, 1].reshape(-1, cfg.num_kv_heads, cfg.head_dim)
        # the gather already copied; the slice keeps the trailing slots of the
        # last page out, and `clone` keeps the whole page span from being held
        cached = (
            k_pref[:segment.prefix_len].clone(), v_pref[:segment.prefix_len].clone()
        )
        entry.layers[layer_idx] = cached
        return cached

    @torch.compiler.disable
    def run(
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """One layer's dense attention. ``k``/``v`` are this step's freshly
        projected K/V, packed in the plan's segment order; they are never
        written to the pages."""
        if label is None:
            label = self._default_label
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        assert k is not None and v is not None and layer_idx is not None, (
            "dense attention runs on the step's own K/V: pass k, v and "
            "layer_idx (the layer writes nothing through the KV resource "
            "under this backend — see `requires_kv_write`)"
        )
        # `plan` refuses a leased slot, which covers every path that plans. A
        # piecewise region declaring `reuses_outer_plan` never plans, so its
        # capture would reach here and bake this step's gather, its shapes and
        # the address of a cached prefix into the graph; the first invalidation
        # re-gathers into a new tensor the replay does not read. Silent, so
        # raise rather than assert.
        if self._on_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "dense attention cannot be captured: the prefix gather and "
                "the concatenation are shaped by the step and allocate. A "
                "piecewise region with `reuses_outer_plan` is the path that "
                "gets here — capture that region against the paged backend."
            )
        from fa3_fwd_interface import flash_attn_varlen_func

        plan = self._current_plans[label]
        q = q.to(self._dtype)
        k = k.to(self._dtype)
        v = v.to(self._dtype)

        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        offset = 0
        for segment in plan.segments:
            if segment.prefix_len:
                k_pref, v_pref = self._prefix_kv(segment, layer_idx, kv_cache_layer)
                k_parts.append(k_pref)
                v_parts.append(v_pref)
            k_parts.append(k[offset:offset + segment.q_len])
            v_parts.append(v[offset:offset + segment.q_len])
            offset += segment.q_len

        out = flash_attn_varlen_func(
            q, torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0),
            plan.cu_q, plan.cu_k, plan.max_q, plan.max_k,
            # a causal plan aligns the query block to the end of the key
            # sequence, so the fresh tokens see the whole prefix; only
            # non-causal generation attention is exercised today
            causal=plan.causal,
        )
        return out[0] if isinstance(out, tuple) else out
