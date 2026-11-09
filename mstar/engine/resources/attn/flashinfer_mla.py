"""Multi-head Latent Attention through FlashInfer's ``BatchMLAPagedAttentionWrapper``.

The cache (``KVLayout.MLA``) holds one ``[ckv | kpe]`` latent per token; the layer writes
it through the KV resource and attends with the *absorbed* query
(``q_nope @ W_UK`` -> ``kv_lora_rank`` wide, plus the un-rotated ``q_pe``). The kernel's
output is in the latent space (``[tokens, heads, kv_lora_rank]``); the layer applies
``W_UV``. One wrapper class serves prefill and decode (query lengths come from
``qo_indptr``), so the slot/label bookkeeping is the same as ``FlashInferManager``'s.
"""
from __future__ import annotations

import functools
import logging

import torch

from mstar.engine.resources.attn.base import (
    AttentionManager,
    EagerSlotKey,
    captured_page_capacity,
)
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.workspace import Workspace, WorkspacePool
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.kv.config import KVConfig, KVLayout
from mstar.engine.resources.kv.plan import KVPlanOutputs, PagedIndptrs, build_paged_indptrs, context_only_views
from mstar.engine.resources.step import SlotLease, StepContext

logger = logging.getLogger(__name__)


def flashinfer_mla_supports(kv_lora_rank: int, qk_rope_head_dim: int) -> bool:
    """FlashInfer's Hopper MLA kernel is specialised for the DeepSeek/Kimi latent shape
    (``ckv`` 512, ``kpe`` 0 or 64); other shapes are JIT-compiled but read out of bounds.
    Anything else runs the torch fallback (eager, not CUDA-graph safe): small test
    checkpoints, not production models."""
    if kv_lora_rank != 512:
        return False
    if qk_rope_head_dim == 64:
        return True
    # kpe 0 (a NoPE cache with no rope slot) plans from FlashInfer 0.6.18 on
    return qk_rope_head_dim == 0 and _flashinfer_at_least("0.6.18")


@functools.cache
def _flashinfer_at_least(version: str) -> bool:
    try:
        import flashinfer
        from packaging.version import Version
    except ImportError:
        return False
    return Version(flashinfer.__version__) >= Version(version)


class FlashInferMLAWrapper:
    """Thin wrapper over ``flashinfer.mla.BatchMLAPagedAttentionWrapper``.

    Args:
        workspace_buffer: FlashInfer float workspace.
        num_qo_heads: local query heads.
        kv_lora_rank / qk_rope_head_dim: latent split (``ckv`` / ``kpe`` widths).
        page_size, max_num_pages: cache geometry.
        batch_size: rows (requests) for CUDA-graph mode; fixed for the wrapper's life.
        sm_scale: softmax scale of the original ``qk_head_dim`` (e.g. ``192 ** -0.5``).
    """

    def __init__(
        self,
        workspace_buffer: Workspace,
        num_qo_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        page_size: int,
        sm_scale: float,
        batch_size: int | None = None,
        max_num_pages: int | None = None,
        device: torch.device = torch.device("cuda"),
        use_cuda_graph: bool = False,
        backend: str = "auto",
        num_requests: int | None = None,
    ):
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.batch_size = batch_size
        self.num_qo_heads = num_qo_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.page_size = page_size
        self.sm_scale = sm_scale
        self.dtype = None
        self.kv_dtype = None
        self._qo_indptr_buf: torch.Tensor | None = None
        self._kv_len_buf: torch.Tensor | None = None  # the rows' planned kv lengths, on device
        # the step's requests' query offsets and kv lengths, as the consumers read them
        # (``select_last_hidden``); the kernel's rows differ when short causal rows are split
        # (``rowwise_causal_indptrs``), and then a captured wrapper keeps them in their own buffers
        self.request_qo_indptr: torch.Tensor | None = None
        self.request_kv_len: torch.Tensor | None = None
        self._req_bufs: tuple[torch.Tensor, torch.Tensor] | None = None
        on_gpu = torch.device(device).type == "cuda"
        self.fallback = not on_gpu or not flashinfer_mla_supports(kv_lora_rank, qk_rope_head_dim)
        if self.fallback:
            logger.warning(
                "MLA latent shape (%d, %d) is outside FlashInfer's kernel; using the torch fallback",
                kv_lora_rank, qk_rope_head_dim,
            )
            self._fb: tuple | None = None
            self.attn_wrapper = None
            return
        import flashinfer

        self._workspace = workspace_buffer
        if use_cuda_graph:
            assert batch_size is not None and max_num_pages is not None
            self._qo_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            self._kv_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            # every page a request can hold, plus the sink page each padding row repeats
            self._kv_indices_buf = torch.zeros(
                max_num_pages + batch_size, dtype=torch.int32, device=device,
            )
            self._kv_len_buf = torch.zeros(batch_size, dtype=torch.int32, device=device)
            if num_requests is not None and num_requests != batch_size:
                self._req_bufs = (torch.zeros(num_requests + 1, dtype=torch.int32, device=device),
                                  torch.zeros(num_requests, dtype=torch.int32, device=device))
            self.attn_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                self._workspace.tensor, use_cuda_graph=True,
                qo_indptr=self._qo_indptr_buf, kv_indptr=self._kv_indptr_buf,
                kv_indices=self._kv_indices_buf, kv_len_arr=self._kv_len_buf,
                backend=backend,
            )
        else:
            self.attn_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                self._workspace.tensor, backend=backend,
            )

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        causal: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        kv_dtype: torch.dtype | None = None,
        requests: PagedIndptrs | None = None,
        **kwargs,
    ):
        """Inputs may be CPU tensors (the KV plan builds them on the host). ``requests``: the
        step's own rows when the kernel's are split (``rowwise_causal_indptrs``), for the
        consumers of ``request_qo_indptr`` / ``request_kv_len``."""
        num_pages = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
        kv_len_arr = (num_pages - 1).clamp_min(0) * self.page_size + paged_kv_last_page_len
        kv_len_arr = torch.where(num_pages > 0, kv_len_arr, torch.zeros_like(kv_len_arr)).to(torch.int32)
        if not self.use_cuda_graph or self.fallback:
            # eager: keep the qo_indptr on device for select_last_hidden
            self._qo_indptr_buf = qo_indptr.to(self.device, dtype=torch.int32, non_blocking=True)
            self._kv_len_buf = kv_len_arr.to(self.device, dtype=torch.int32, non_blocking=True)
        elif not self.fallback:
            self._kv_len_buf[: kv_len_arr.numel()].copy_(kv_len_arr.to(torch.int32), non_blocking=True)
        self.dtype = dtype
        self.kv_dtype = kv_dtype or dtype
        self._stage_requests(requests)
        if self.fallback:
            self._fb = (
                qo_indptr.tolist(), paged_kv_indptr.tolist(), paged_kv_indices.to(self.device, dtype=torch.long),
                kv_len_arr.tolist(), bool(causal),
            )
            return
        self._workspace.plan(
            self.attn_wrapper.plan,
            qo_indptr=qo_indptr.to(torch.int32),
            kv_indptr=paged_kv_indptr.to(torch.int32),
            kv_indices=paged_kv_indices.to(torch.int32),
            kv_len_arr=kv_len_arr,
            num_heads=self.num_qo_heads,
            head_dim_ckv=self.kv_lora_rank,
            head_dim_kpe=self.qk_rope_head_dim,
            page_size=self.page_size,
            causal=causal,
            sm_scale=self.sm_scale,
            q_data_type=dtype,
            kv_data_type=kv_dtype or dtype,
        )
        self.dtype = dtype
        self.kv_dtype = kv_dtype or dtype

    def _stage_requests(self, requests: PagedIndptrs | None) -> None:
        if requests is None:
            self.request_qo_indptr, self.request_kv_len = self._qo_indptr_buf, self._kv_len_buf
            return
        qo = requests.qo_indptr.to(torch.int32)
        lens = requests.kv_lens.to(torch.int32)
        if self._req_bufs is None:
            self.request_qo_indptr = qo.to(self.device, non_blocking=True)
            self.request_kv_len = lens.to(self.device, non_blocking=True)
            return
        qo_buf, len_buf = self._req_bufs
        qo_buf[: qo.numel()].copy_(qo, non_blocking=True)
        len_buf[: lens.numel()].copy_(lens, non_blocking=True)
        self.request_qo_indptr, self.request_kv_len = qo_buf, len_buf

    @torch.compiler.disable
    def run(self, q_nope: torch.Tensor, q_pe: torch.Tensor, kv_cache_layer: torch.Tensor, return_lse: bool = False):
        """``q_nope [T, H, kv_lora_rank]`` (absorbed), ``q_pe [T, H, qk_rope_head_dim]``,
        ``kv_cache_layer [pages, page_size, latent]`` -> ``[T, H, kv_lora_rank]`` (and, with
        ``return_lse``, the natural log-sum-exp of the scores ``[T, H]`` fp32)."""
        ckv = kv_cache_layer[..., : self.kv_lora_rank]
        kpe = kv_cache_layer[..., self.kv_lora_rank :]
        if self.fallback:
            return self._run_fallback(q_nope, q_pe, kv_cache_layer, return_lse)
        return self.attn_wrapper.run(q_nope.to(self.dtype), q_pe.to(self.dtype), ckv, kpe,
                                     return_lse=return_lse, return_lse_base_on_e=return_lse)

    def _run_fallback(self, q_nope: torch.Tensor, q_pe: torch.Tensor, kv_cache_layer: torch.Tensor,
                      return_lse: bool = False):
        """Per-request gather of the paged latents and a dense fp32 softmax attention:
        the reference semantics of the kernel, for latent shapes it does not support."""
        assert self._fb is not None, "plan() before run()"
        qo, kv_indptr, kv_indices, kv_len, causal = self._fb
        out = torch.empty(q_nope.shape[0], q_nope.shape[1], self.kv_lora_rank, dtype=q_nope.dtype, device=q_nope.device)
        lse = torch.full(q_nope.shape[:2], float("-inf"), dtype=torch.float32, device=q_nope.device)
        for i in range(len(qo) - 1):
            n = qo[i + 1] - qo[i]
            m = kv_len[i]
            if n == 0:
                continue
            if m == 0:
                out[qo[i]:qo[i + 1]] = 0
                continue
            pages = kv_indices[kv_indptr[i]:kv_indptr[i + 1]]
            kv = kv_cache_layer.index_select(0, pages).reshape(-1, kv_cache_layer.shape[-1])[:m].float()
            ckv, kpe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            qn = q_nope[qo[i]:qo[i + 1]].float()
            qp = q_pe[qo[i]:qo[i + 1]].float()
            scores = (torch.einsum("qhl,kl->hqk", qn, ckv) + torch.einsum("qhr,kr->hqk", qp, kpe)) * self.sm_scale
            if causal:
                allowed = torch.arange(m, device=kv.device)[None, :] <= (
                    torch.arange(n, device=kv.device)[:, None] + (m - n)
                )
                scores = scores.masked_fill(~allowed[None], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[qo[i]:qo[i + 1]] = torch.einsum("hqk,kl->qhl", probs, ckv).to(out.dtype)
            lse[qo[i]:qo[i + 1]] = torch.logsumexp(scores, dim=-1).transpose(0, 1)
        return (out, lse) if return_lse else out


# Where the head count is one FlashInfer's 64/128-query tile doesn't divide (Kimi K3's 96, 12 a rank at
# TP8), its planned MLA backends (fa2, fa3) refuse a multi-query causal row whose tile boundary falls
# inside a query at a key that starts a KV tile (``_validate_fa_causal_tile_bound``; GPU job 1529:
# "causal tile boundary ... query=2, key=64"). Which rows: those whose first query sits at an offset off
# a 64-key boundary (at 12 heads, offsets 11, 22, 43 and 54 mod 64): a verify block, a drafter's block,
# and a prompt the scheduler cut mid-way, whose next chunk starts anywhere (B300 pre-flight audit: ~90%
# of c=64 waves of ~1K-token prompts). A one-query row, and a row starting on a 64-key boundary, pass for
# every head count and KV tile (16, 32, 64; checked exhaustively against the rule in the tests).
ROW_ALIGN = 64


def rowwise_causal_indptrs(indptrs: PagedIndptrs, page_size: int, align: int = ROW_ALIGN) -> PagedIndptrs:
    """``indptrs`` with each causal row of 2+ queries whose first query is off an ``align``-key boundary
    planned as single-query rows up to that boundary, then the rest of the row as one causal row: query
    ``j`` of a row of ``q`` over ``m`` keys becomes a row of one query over the row's first
    ``m - q + 1 + j`` keys (its pages cut to them). The queries keep their order, so the step's query
    tensor and the output need no reshuffle. Unchanged when no row qualifies."""
    qo = indptrs.qo_indptr.tolist()
    kv = indptrs.paged_kv_indptr.tolist()
    last = indptrs.paged_kv_last_page_len.tolist()

    def keys(i):
        n = kv[i + 1] - kv[i]
        return (n - 1) * page_size + last[i] if n else 0

    def peel(i):
        q = qo[i + 1] - qo[i]
        return min(q, -(keys(i) - q) % align) if q > 1 else 0

    if not any(peel(i) for i in range(len(qo) - 1)):
        return indptrs
    pages = indptrs.paged_kv_indices.tolist()
    new_qo, new_kv, new_pages, new_last, new_lens = [0], [0], [], [], []

    def row(queries, row_pages, length):
        n = -(-length // page_size)
        new_qo.append(new_qo[-1] + queries)
        new_pages.extend(row_pages[:n])
        new_kv.append(new_kv[-1] + n)
        new_last.append(length - (n - 1) * page_size if n else page_size)
        new_lens.append(length)

    for i in range(len(qo) - 1):
        q, row_pages, m, k = qo[i + 1] - qo[i], pages[kv[i]:kv[i + 1]], keys(i), peel(i)
        for j in range(k):
            row(1, row_pages, max(m - (q - 1 - j), 0))
        if q > k:
            row(q - k, row_pages, m)

    def ints(values):
        return torch.tensor(values, dtype=torch.int32)

    return PagedIndptrs(qo_indptr=ints(new_qo), paged_kv_indptr=ints(new_kv), paged_kv_indices=ints(new_pages),
                        paged_kv_last_page_len=ints(new_last), kv_lens=ints(new_lens))


def _kv_lens(indptrs: PagedIndptrs, page_size: int) -> torch.Tensor:
    pages = indptrs.paged_kv_indptr[1:] - indptrs.paged_kv_indptr[:-1]
    lens = (pages - 1).clamp_min(0) * page_size + indptrs.paged_kv_last_page_len
    return torch.where(pages > 0, lens, torch.zeros_like(lens)).to(torch.int32)


class FlashInferMLAManager(AttentionManager):
    supports_drop = True

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: KVConfig,
        backend: str = "auto",
        sm_scale: float | None = None,
    ):
        assert kv_config.layout == KVLayout.MLA, (
            f"FlashInferMLAManager needs a KVLayout.MLA cache; {kv_cache!r} is {kv_config.layout}"
        )
        if not kv_config.qo_heads_given:
            # it defaulted to the one latent KV head, and the kernel planned a single head
            raise ValueError(f"MLA attention over {kv_cache!r} needs the cache's num_qo_heads, "
                             "the model's query heads")
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        self._current_plan_states: dict[str, FlashInferMLAWrapper] = {}
        self._eager_plan_states: dict[EagerSlotKey, FlashInferMLAWrapper] = {}
        self._cg_plan_states: dict[CGSlotKey, FlashInferMLAWrapper] = {}
        self._preplan_states: dict[str, FlashInferMLAWrapper] = {}
        self._preplanned = False
        if sm_scale is None:
            raise ValueError(
                "AttentionConfig.sm_scale must be set for MLA: the kernel scores the absorbed "
                "latent, so it needs the original qk head dim's scale"
            )
        self._wrapper_kwargs = dict(
            num_qo_heads=kv_config.num_qo_heads,
            kv_lora_rank=kv_config.kv_lora_rank,
            qk_rope_head_dim=kv_config.qk_rope_head_dim,
            page_size=kv_config.tokens_per_page,
            max_num_pages=kv_config.max_num_pages,
            sm_scale=sm_scale,
            device=device,
            backend=backend,
        )
        self._workspaces = WorkspacePool(device, owner=kv_cache)

    def depends_on(self):
        return {self._kv_cache_name}

    @property
    def _splits_short_rows(self) -> bool:
        """Whether causal rows are aligned to 64 keys by single-query rows: on the kernel, for a
        head count FlashInfer's query tile doesn't divide (Kimi K3's 96, 12 a rank at TP8), where its
        planned backends refuse some offsets of a multi-query causal row (``rowwise_causal_indptrs``)."""
        return self.uses_kernel and 128 % self._kv_config.num_qo_heads != 0

    @property
    def uses_kernel(self) -> bool:
        """False when the torch fallback serves the plans, which cannot be captured."""
        return torch.device(self._device).type == "cuda" and flashinfer_mla_supports(
            self._kv_config.kv_lora_rank, self._kv_config.qk_rope_head_dim)

    def _cg_wrapper(self, lease: SlotLease, label: str, num_rows: int, num_requests: int | None = None):
        if not self.uses_kernel:
            # the fallback reads its plan from host lists, so a replay would
            # attend the capture's pages
            raise RuntimeError(
                "MLA torch fallback cannot run under a CUDA graph; declare no "
                "capture configs when uses_kernel is False"
            )
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        wrapper = self._cg_plan_states.get(key)
        if wrapper is None:
            kwargs = {**self._wrapper_kwargs, "max_num_pages": captured_page_capacity(
                num_rows, lease.bucket.num_tokens, self._kv_config)}
            wrapper = FlashInferMLAWrapper(
                workspace_buffer=self._workspaces.get(label, lease.slot),
                batch_size=num_rows, use_cuda_graph=True, num_requests=num_requests, **kwargs,
            )
            self._cg_plan_states[key] = wrapper
        return wrapper

    def _eager_wrapper(self, label: str, slot: int):
        """The persistent eager wrapper for one (label, slot): its plan stages into a
        pinned buffer the wrapper holds, so one per slot keeps plan(N+1) off step N's
        queued copy. Takes the workspace this slot's captured wrappers hold; the two
        never run at once."""
        key = EagerSlotKey(label=label, slot=slot)
        wrapper = self._eager_plan_states.get(key)
        if wrapper is None:
            wrapper = FlashInferMLAWrapper(
                workspace_buffer=self._workspaces.get(label, slot), **self._wrapper_kwargs)
            self._eager_plan_states[key] = wrapper
        return wrapper

    @property
    def supports_preplan(self):
        return True

    @property
    def force_double_buffer(self):
        # FlashInfer's MLA plan stages into pinned buffers per wrapper and copies
        # them to the device without blocking
        return True

    def plan(self, step: AttentionStep, ctx: StepContext):
        self.reset_default_cursors()
        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, "preplan requires a cuda graph step"
        assert not (self._preplanned and ctx.is_preplan), "MLA preplan already pending"
        if self._preplanned:
            self._current_plan_states = self._preplan_states
            self._preplan_states = {}
            self._preplanned = False
            return
        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, f"expected plan result from {self._kv_cache_name}"
        plan_states = self._preplan_states if ctx.is_preplan else self._current_plan_states
        plan_states.clear()
        for label, kv_out in plan_outputs.items():
            indptrs = kv_out.cpu_indptrs
            if step.context_only:
                # the rows' query counts come from this step's own segments, the kv from the
                # stored context (the cache's segments describe what the step appends)
                spans = [int(seg.span) for seg in (step.segments or ()) if seg.label == label]
                views = context_only_views(kv_out.views, spans, self._kv_config.tokens_per_page)
                indptrs = build_paged_indptrs(views, self._kv_config.tokens_per_page)
            requests = None
            if step.causal and self._splits_short_rows:
                rows = rowwise_causal_indptrs(indptrs, self._kv_config.tokens_per_page)
                if rows is not indptrs:
                    requests = indptrs if indptrs.kv_lens is not None else indptrs._replace(
                        kv_lens=_kv_lens(indptrs, self._kv_config.tokens_per_page))
                    indptrs = rows
            if lease is not None:
                wrapper = self._cg_wrapper(lease, label, indptrs.qo_indptr.shape[0] - 1,
                                           None if requests is None else requests.qo_indptr.shape[0] - 1)
            else:
                wrapper = self._eager_wrapper(label, ctx.slot)
            wrapper.plan(causal=step.causal, dtype=self._dtype, requests=requests, **indptrs.to_kwargs_dict())
            plan_states[label] = wrapper
        self._preplanned = ctx.is_preplan

    def clear_preplan(self):
        self._preplanned = False
        self._preplan_states = {}

    @torch.compiler.disable
    def qo_indptr_buf(self, label: str = "main") -> torch.Tensor | None:
        wrapper = self._current_plan_states.get(label)
        if wrapper is None:
            return None
        # the step's requests' offsets (the kernel's rows differ when short causal rows split)
        qo = getattr(wrapper, "request_qo_indptr", None)
        return wrapper._qo_indptr_buf if qo is None else qo

    @torch.compiler.disable
    def kv_len_buf(self, label: str = "main") -> torch.Tensor:
        """The rows' planned kv lengths, int32 on device (static under a lease): for a
        ``context_only`` plan the length of each row's stored context, which is where the row's
        queries (and the entries the step appends) start."""
        wrapper = self._current_plan_states[label]
        lens = getattr(wrapper, "request_kv_len", None)
        return wrapper._kv_len_buf if lens is None else lens

    def run(
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
        k: torch.Tensor | None = None, v: torch.Tensor | None = None,
        layer_idx: int | None = None, q_pe: torch.Tensor | None = None,
        return_lse: bool = False,
    ):
        """``q`` is the absorbed ``q_nope`` (``[T, H, kv_lora_rank]``); ``q_pe`` the
        rope-part query (``[T, H, qk_rope_head_dim]``). With ``return_lse`` also the natural
        log-sum-exp of the scores ``[T, H]`` fp32, for merging with another key segment."""
        del k, v, layer_idx
        if label is None:
            label = self._default_label
        assert q_pe is not None, "MLA needs the rope-part query"
        return self._attend(q, q_pe, label, kv_cache_layer, return_lse)

    @torch.compiler.disable
    def _attend(self, q_nope, q_pe, label, kv_cache_layer, return_lse=False):
        out = self._current_plan_states[label].run(q_nope, q_pe, kv_cache_layer, return_lse=return_lse)
        o, lse = out if return_lse else (out, None)
        if o.dtype != q_nope.dtype:
            o = o.to(q_nope.dtype)
        return (o, lse) if return_lse else o
