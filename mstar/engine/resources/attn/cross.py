"""Cross-attention over a context written once and never extended."""

from typing import Any, NamedTuple

import torch

from mstar.engine.resources.attn.backends import flashinfer_kwargs
from mstar.engine.resources.attn.base import (
    AttentionWrapper,
    EagerSlotKey,
    PlanCacheKey,
    WorkspacePool,
    resolve_attn_backend,
)
from mstar.engine.resources.attn.config import (
    AttentionStep,
    AttnBackend,
    CrossAttentionSpec,
    refuse_context_only,
)
from mstar.engine.resources.attn.wrappers import (
    FlashInferDecodeWrapper,
    FlashInferPrefillWrapper,
)
from mstar.engine.resources.base import (
    AttentionResource,
    CGSlotKey,
    EngineResourceInfo,
)
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import (
    SINK_PAGE,
    KVPlanOutputs,
    PagedIndptrs,
    SequenceView,
)
from mstar.engine.resources.step import SlotLease, StepContext


class _LastPlan(NamedTuple):
    """What a wrapper was last planned with. The context's KV manager hands
    back the same plan object while the context streams are untouched, so the
    same object over the same query rows skips building the arrays; a new
    object over the same pages still skips FlashInfer's plan."""
    context: Any  # the context label's KVPlanOutput
    request_ids: tuple
    qo_indptr: tuple
    plan_key: PlanCacheKey


class QueryPacking(NamedTuple):
    """One plan label's query side: the requests in packed order, and the
    cumulative query lengths over them."""
    request_ids: list[str]
    qo_indptr: torch.Tensor  # int32, CPU


class CrossAttentionManager(AttentionResource):
    prefix_skip_safe = True

    # Remains abstract except for build; will build based
    # on the attention backend

    @classmethod
    def build(cls, spec: CrossAttentionSpec, info: EngineResourceInfo):
        # the context cache's own head counts; see AttentionManager.build
        kv_config = info.dependency(spec.config.kv_cache).config
        backend = resolve_attn_backend(
            spec.config.backend, info.device, kv_config.page_size,
        )
        if info.joint_comm_group is not None:
            kv_config.shard(info.joint_comm_group.world_size)
        if backend == AttnBackend.FLASHINFER:
            return FlashInferCrossManager(
                kv_cache=spec.config.kv_cache,
                query_kv_cache=spec.config.query_kv_cache,
                context_label=spec.config.context_label,
                device=info.device,
                dtype=info.kv_dtype,
                kv_config=kv_config,
                **flashinfer_kwargs(
                    kv_config, spec.config.flashinfer_backend, spec.config.decode_tensor_cores,
                ),
            )
        if backend == AttnBackend.AITER:
            from mstar.engine.resources.attn.aiter import AiterCrossManager

            return AiterCrossManager(
                kv_cache=spec.config.kv_cache,
                query_kv_cache=spec.config.query_kv_cache,
                context_label=spec.config.context_label,
                device=info.device,
                dtype=info.kv_dtype,
                kv_config=kv_config,
            )
        raise ValueError(f"Unknown cross attention backend {backend!r}")

    # Context views and query packing are the same for every kernel. Each
    # backend sets ``_kv_cache_name``, ``_query_kv_cache_name``,
    # ``_context_label`` and ``_kv_config``.

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

    def _context_output(self, ctx: StepContext):
        """The context label's KV plan output (its views are each request's
        context stream; key by request rather than zip with the query rows,
        which need not line up)."""
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
        return kv_out

    def _context_views(self, ctx: StepContext) -> dict[str, SequenceView]:
        """The context stream of each request in the batch, by request id."""
        return {view.request_id: view for view in self._context_output(ctx).views}

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
            "raise max_num_pages on the context cache's PagedKVConfig"
        )

        return PagedIndptrs(
            qo_indptr=packing.qo_indptr,
            paged_kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32),
            paged_kv_indices=torch.tensor(all_pages, dtype=torch.int32),
            paged_kv_last_page_len=torch.tensor(last_page_lens, dtype=torch.int32),
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
        kv_config: PagedKVConfig,
        backend: str = "auto",
        decode_tensor_cores: bool = True,
    ):
        self._kv_cache_name = kv_cache
        self._query_kv_cache_name = query_kv_cache
        self._decode_tensor_cores = decode_tensor_cores
        self._context_label = context_label
        self._device = device
        self._dtype = dtype

        # decoder plan label -> wrapper
        self._current_plan_states: dict[str, AttentionWrapper] = {}
        # persistent wrappers [we can keep eager ones]
        self._eager_plan_states: dict[EagerSlotKey, AttentionWrapper] = {}
        self._cg_plan_states: dict[CGSlotKey, AttentionWrapper] = {}

        self._preplan_states: dict[str, AttentionWrapper] = {}
        self._preplanned = False
        # wrapper key (EagerSlotKey, or CGSlotKey under capture) -> its last plan
        self._last_plans: dict[Any, _LastPlan] = {}

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

        self._workspaces = WorkspacePool(device)

    @property
    def supports_preplan(self):
        return True

    @property
    def force_double_buffer(self):
        # same pinned staging inside FlashInfer's plan as the self-attention
        # manager; see `FlashInferManager.force_double_buffer`
        return True

    def plan(self, step: AttentionStep, ctx: StepContext):
        refuse_context_only(step, type(self).__name__)
        self.reset_default_cursors()
        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a captured graph step: an eager wrapper shares its "
            "workspace with the captured one on the same slot"
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

        context_out = self._context_output(ctx)
        context_views = None
        # a preplan leases a different cg slot, so its wrappers and workspaces
        # are disjoint from the ones the in-flight forward reads
        plan_states = self._preplan_states if ctx.is_preplan else self._current_plan_states

        plan_states.clear()
        for plan_label, packing in self._query_packings(step, ctx).items():
            qo_indptr = tuple(packing.qo_indptr.tolist())
            num_rows = len(qo_indptr) - 1
            # every row queries one token (a span-0 row does not)
            decode = not ctx.force_prefill and qo_indptr == tuple(range(num_rows + 1))
            state_key, wrapper = self._wrapper_for(
                plan_label, lease, num_rows=num_rows, slot=ctx.slot, decode=decode,
            )
            request_ids = tuple(packing.request_ids)
            last = self._last_plans.get(state_key)
            if (
                last is not None and last.context is context_out
                and last.request_ids == request_ids and last.qo_indptr == qo_indptr
            ):
                plan_states[plan_label] = wrapper
                continue
            if context_views is None:
                context_views = {view.request_id: view for view in context_out.views}
            indptrs = self._build_indptrs(packing, context_views, plan_label)

            # The context pages are immutable once written, so only the query
            # side moves between steps; when neither moved, FlashInfer's plan
            # is skippable. Every tensor here is on CPU: `.tolist()` costs no sync.
            plan_key = PlanCacheKey(
                q_seq_lens=qo_indptr,
                page_indices=tuple(indptrs.paged_kv_indices.tolist()),
                last_page_lens=tuple(indptrs.paged_kv_last_page_len.tolist()),
            )
            if last is None or last.plan_key != plan_key:
                wrapper.plan(
                    causal=False,
                    dtype=self._dtype,
                    **indptrs.to_kwargs_dict()
                )
            self._last_plans[state_key] = _LastPlan(context_out, request_ids, qo_indptr, plan_key)
            plan_states[plan_label] = wrapper

        self._preplanned = ctx.is_preplan

    def clear_preplan(self):
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}

    def _wrapper_for(
        self, plan_label: str, lease: SlotLease | None, num_rows: int, slot: int,
        decode: bool = False,
    ) -> tuple[Any, AttentionWrapper]:
        """``num_rows`` is this label's qo_indptr row count — bucket.bs for an
        ordinary label, more when the query plan combines labels. See
        ``FlashInferManager._cg_wrapper``.

        ``decode``: every row queries one token, as a decoder's cross
        attention does each step against the whole encoder context, which the
        decode kernel serves."""
        if lease is not None:
            key = CGSlotKey(
                bucket=lease.bucket, slot=lease.slot, label=plan_label
            )
            wrapper = self._cg_plan_states.get(key)
            if wrapper is not None:
                # a captured graph replays the kernel it was recorded with
                assert isinstance(wrapper, FlashInferDecodeWrapper) == decode, (
                    f"cross attention {plan_label!r} in {lease.bucket} was captured as "
                    f"{'decode' if not decode else 'prefill'} but this step plans "
                    f"{'decode' if decode else 'prefill'} rows"
                )
            else:
                # built on first plan for the key; see FlashInferManager._cg_wrapper
                workspace = self._workspaces.get(plan_label, lease.slot)
                if decode:
                    wrapper = self._decode_wrapper(workspace, batch_size=num_rows, accelerator_graph=True)
                else:
                    wrapper = FlashInferPrefillWrapper(
                        workspace_buffer=workspace,
                        batch_size=num_rows,
                        max_total_tokens=lease.bucket.num_tokens,
                        accelerator_graph=True,
                        **self._wrapper_kv_kwargs,
                    )
                self._cg_plan_states[key] = wrapper
            return key, wrapper

        # slotted like the captured one, and on the same workspace; see
        # `FlashInferManager._eager_wrapper`
        key = EagerSlotKey(label=plan_label, slot=slot, is_decode=decode)
        wrapper = self._eager_plan_states.get(key)
        if wrapper is None:
            workspace = self._workspaces.get(plan_label, slot)
            if decode:
                wrapper = self._decode_wrapper(workspace)
            else:
                wrapper = FlashInferPrefillWrapper(workspace_buffer=workspace, **self._wrapper_kv_kwargs)
            self._eager_plan_states[key] = wrapper
        return key, wrapper

    def _decode_wrapper(self, workspace: torch.Tensor, **kwargs) -> FlashInferDecodeWrapper:
        kv = self._wrapper_kv_kwargs
        return FlashInferDecodeWrapper(
            workspace_buffer=workspace,
            num_qo_heads=kv["num_qo_heads"], num_kv_heads=kv["num_kv_heads"],
            head_dim=kv["head_dim"], page_size=kv["page_size"],
            max_num_pages=kv["max_num_pages"], device=kv["device"], backend=kv["backend"],
            use_tensor_cores=self._decode_tensor_cores,
            **kwargs,
        )

    def run(
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One layer's cross attention. ``kv_cache_layer`` is a layer of the
        *context* cache; nothing is written to it here."""
        if label is None:
            label = self._default_label
        return self._attend(q, label, kv_cache_layer)

    @torch.compiler.disable
    def _attend(
        self, q: torch.Tensor, label: str, kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        o = self._current_plan_states[label].run(q, kv_cache_layer)
        if o.dtype != q.dtype:
            o = o.to(q.dtype)
        return o
