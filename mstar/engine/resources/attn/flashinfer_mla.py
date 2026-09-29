"""Multi-head Latent Attention through FlashInfer's ``BatchMLAPagedAttentionWrapper``.

The cache (``KVLayout.MLA``) holds one ``[ckv | kpe]`` latent per token; the layer writes
it through the KV resource and attends with the *absorbed* query
(``q_nope @ W_UK`` -> ``kv_lora_rank`` wide, plus the un-rotated ``q_pe``). The kernel's
output is in the latent space (``[tokens, heads, kv_lora_rank]``); the layer applies
``W_UV``. One wrapper class serves prefill and decode (query lengths come from
``qo_indptr``), so the slot/label bookkeeping is the same as ``FlashInferManager``'s.
"""
from __future__ import annotations

import logging

import torch

from mstar.engine.resources.attn.base import AttentionManager, WorkspacePool
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.flashmla import FlashMLAWrapper, flashmla_available, flashmla_supports, flashmla_wanted
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.kv.config import KVConfig, KVLayout
from mstar.engine.resources.kv.plan import KVPlanOutputs, build_paged_indptrs, context_only_views
from mstar.engine.resources.step import SlotLease, StepContext

logger = logging.getLogger(__name__)


def flashinfer_mla_supports(kv_lora_rank: int, qk_rope_head_dim: int) -> bool:
    """FlashInfer's Hopper MLA kernel is specialised for the DeepSeek/Kimi latent shape
    (``ckv`` 512, ``kpe`` 0 or 64); other shapes are JIT-compiled but read out of bounds.
    Anything else runs the torch fallback (eager, not CUDA-graph safe): small test
    checkpoints, not production models."""
    return kv_lora_rank == 512 and qk_rope_head_dim in (0, 64)


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
        workspace_buffer: torch.Tensor,
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

        if use_cuda_graph:
            assert batch_size is not None and max_num_pages is not None
            self._qo_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            self._kv_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            self._kv_indices_buf = torch.zeros(max_num_pages, dtype=torch.int32, device=device)
            self._kv_len_buf = torch.zeros(batch_size, dtype=torch.int32, device=device)
            self.attn_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                workspace_buffer, use_cuda_graph=True,
                qo_indptr=self._qo_indptr_buf, kv_indptr=self._kv_indptr_buf,
                kv_indices=self._kv_indices_buf, kv_len_arr=self._kv_len_buf,
                backend=backend,
            )
        else:
            self.attn_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                workspace_buffer, backend=backend,
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
        **kwargs,
    ):
        """Inputs may be CPU tensors (the KV plan builds them on the host)."""
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
        if self.fallback:
            self._fb = (
                qo_indptr.tolist(), paged_kv_indptr.tolist(), paged_kv_indices.to(self.device, dtype=torch.long),
                kv_len_arr.tolist(), bool(causal),
            )
            return
        self.attn_wrapper.plan(
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


class FlashInferMLAManager(AttentionManager):
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
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        self._current_plan_states: dict[str, FlashInferMLAWrapper | FlashMLAWrapper] = {}
        self._eager_plan_states: dict[tuple[str, bool], FlashInferMLAWrapper | FlashMLAWrapper] = {}
        self._cg_plan_states: dict[tuple[CGSlotKey, bool], FlashInferMLAWrapper | FlashMLAWrapper] = {}
        self._preplan_states: dict[str, FlashInferMLAWrapper | FlashMLAWrapper] = {}
        self._preplanned = False
        # DeepSeek's FlashMLA for the plans it takes (causal, one query per row) when
        # MSTAR_MLA_DECODE_BACKEND=flashmla asks for it; measured no faster than FlashInfer's Hopper kernel
        # on an H100 at 1 to 64 rows, so FlashInfer stays the default
        self._flashmla = (flashmla_wanted() and flashmla_available()
                          and flashmla_supports(kv_config.kv_lora_rank, kv_config.qk_rope_head_dim,
                                                kv_config.page_size))
        if self._flashmla:
            logger.info("MLA attention: FlashMLA serves the decode plans of %r", kv_cache)
        if sm_scale is None:
            raise ValueError(
                "AttentionConfig.sm_scale must be set for MLA: the kernel scores the absorbed "
                "latent, so it needs the original qk head dim's scale"
            )
        self._wrapper_kwargs = dict(
            num_qo_heads=kv_config.num_qo_heads,
            kv_lora_rank=kv_config.kv_lora_rank,
            qk_rope_head_dim=kv_config.qk_rope_head_dim,
            page_size=kv_config.page_size,
            max_num_pages=kv_config.max_num_pages,
            sm_scale=sm_scale,
            device=device,
            backend=backend,
        )
        self._workspaces = WorkspacePool(device)

    def depends_on(self):
        return {self._kv_cache_name}

    def _flashmla_kwargs(self) -> dict:
        k = self._wrapper_kwargs
        return dict(num_qo_heads=k["num_qo_heads"], kv_lora_rank=k["kv_lora_rank"],
                    qk_rope_head_dim=k["qk_rope_head_dim"],
                    page_size=k["page_size"], sm_scale=k["sm_scale"], device=k["device"],
                    max_pages_per_row=-(-int(self._kv_config.max_seq_len) // int(k["page_size"])))

    def _cg_wrapper(self, lease: SlotLease, label: str, num_rows: int, flashmla: bool = False):
        key = (CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label), flashmla)
        wrapper = self._cg_plan_states.get(key)
        if wrapper is None:
            if flashmla:
                wrapper = FlashMLAWrapper(batch_size=num_rows, use_cuda_graph=True, **self._flashmla_kwargs())
            else:
                wrapper = FlashInferMLAWrapper(
                    workspace_buffer=self._workspaces.get(label, lease.slot),
                    batch_size=num_rows, use_cuda_graph=True, **self._wrapper_kwargs,
                )
            self._cg_plan_states[key] = wrapper
        return wrapper

    def _eager_wrapper(self, label: str, flashmla: bool = False):
        key = (label, flashmla)
        wrapper = self._eager_plan_states.get(key)
        if wrapper is None:
            if flashmla:
                wrapper = FlashMLAWrapper(**self._flashmla_kwargs())
            else:
                wrapper = FlashInferMLAWrapper(workspace_buffer=self._workspaces.get(label), **self._wrapper_kwargs)
            self._eager_plan_states[key] = wrapper
        return wrapper

    @property
    def supports_preplan(self):
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
                views = context_only_views(kv_out.views, spans, self._kv_config.page_size)
                indptrs = build_paged_indptrs(views, self._kv_config.page_size)
            # FlashMLA for a plain decode plan (one causal query per row, no context-only read)
            fmla = self._flashmla and FlashMLAWrapper.plan_fits(indptrs.qo_indptr, step.causal, step.context_only) == 1
            if lease is not None:
                wrapper = self._cg_wrapper(lease, label, indptrs.qo_indptr.shape[0] - 1, fmla)
            else:
                wrapper = self._eager_wrapper(label, fmla)
            wrapper.plan(causal=step.causal, dtype=self._dtype, **indptrs.to_kwargs_dict())
            plan_states[label] = wrapper
        self._preplanned = ctx.is_preplan

    def clear_preplan(self):
        self._preplanned = False
        self._preplan_states = {}

    @torch.compiler.disable
    def qo_indptr_buf(self, label: str = "main") -> torch.Tensor | None:
        wrapper = self._current_plan_states.get(label)
        return None if wrapper is None else wrapper._qo_indptr_buf

    @torch.compiler.disable
    def kv_len_buf(self, label: str = "main") -> torch.Tensor:
        """The rows' planned kv lengths, int32 on device (static under a lease): for a
        ``context_only`` plan the length of each row's stored context, which is where the row's
        queries (and the entries the step appends) start."""
        return self._current_plan_states[label]._kv_len_buf

    def select_last_hidden(self, hidden: torch.Tensor, label: str = "main") -> torch.Tensor:
        last = (self.qo_indptr_buf(label)[1:] - 1).long()
        return hidden.index_select(0, last)

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
