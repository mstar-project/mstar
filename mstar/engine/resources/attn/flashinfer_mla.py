"""Multi-head Latent Attention through FlashInfer's ``BatchMLAPagedAttentionWrapper``.

The cache (``KVLayout.MLA``) holds one ``[ckv | kpe]`` latent per token; the layer writes
it through the KV resource and attends with the *absorbed* query
(``q_nope @ W_UK`` -> ``kv_lora_rank`` wide, plus the un-rotated ``q_pe``). The kernel's
output is in the latent space (``[tokens, heads, kv_lora_rank]``); the layer applies
``W_UV``. One wrapper class serves prefill and decode (query lengths come from
``qo_indptr``), so the slot/label bookkeeping is the same as ``FlashInferManager``'s, and so is
attention over a context-parallel pool (``ContextParallelWrapper``).
"""
from __future__ import annotations

import functools
import logging
import math

import numpy as np
import torch

from mstar.engine.resources.attn.base import (
    AttentionManager,
    EagerSlotKey,
    WorkspacePool,
    captured_page_capacity,
    one_token_rows,
)
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.context_parallel import ContextParallel
from mstar.engine.resources.kv.config import KVConfig, KVLayout
from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitOutcome,
    AdmitRuntimeError,
    AllocationFailed,
    SlotLease,
    StepContext,
)

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


# the FlashInfer whose FA MLA planner (fa2, fa3) rules `fa_work_items_bound` copies: its cap on a
# plan's work items, its SM cluster and query tile, and its KV split limit. In that release they are
# `_fa_plan_work_count` and `_FA_MAX_WORK_ITEMS` in flashinfer/mla/_batch_mla/_backends/_fa_common.py,
# which mirror `MLAPlan` in include/flashinfer/attention/scheduler.cuh; the `64 % heads` refusal
# at bind is that file's `_validate_fa_causal_tile_bound`
_FA_PLANNER_VERSION = "0.7.2"
_FA_MAX_WORK_ITEMS = 16384


def fa_work_items_bound(qo_indptr: np.ndarray | torch.Tensor, num_heads: int, num_sm: int) -> int:
    """At least the work items FlashInfer's FA MLA planner counts for a step, KV splits included,
    from its query lengths alone: one per (row, query tile), and at most one split per SM
    cluster past those, since a split is at least the average KV per cluster. Each rank of a
    context-parallel pool plans its own KV lengths, so the planner's count can differ by rank;
    this can't."""
    q = np.diff(np.asarray(qo_indptr)).astype(np.int64)
    if not len(q):
        return 0
    # the planner's cluster and query tile
    cluster = 2 if q.sum() * num_heads // len(q) > 64 else 1
    tile = 64 * cluster
    return int(((q * num_heads + tile - 1) // tile).sum()) + num_sm // cluster


def _sm_count(device: torch.device) -> int | None:
    """The device's SMs; None off CUDA, where the torch fallback plans."""
    device = torch.device(device)
    return torch.cuda.get_device_properties(device).multi_processor_count if device.type == "cuda" else None


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
            # every page a request can hold, plus the sink page each padding row repeats
            self._kv_indices_buf = torch.zeros(
                max_num_pages + batch_size, dtype=torch.int32, device=device,
            )
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
    def run(self, q_nope: torch.Tensor, q_pe: torch.Tensor, kv_cache_layer: torch.Tensor, return_lse: bool = False,
            out: torch.Tensor | None = None, lse: torch.Tensor | None = None):
        """``q_nope [T, H, kv_lora_rank]`` (absorbed), ``q_pe [T, H, qk_rope_head_dim]``,
        ``kv_cache_layer [pages, page_size, latent]`` -> ``[T, H, kv_lora_rank]`` (and, with
        ``return_lse``, the natural log-sum-exp of the scores ``[T, H]`` fp32), in ``out`` and
        ``lse`` when given (the kernel path; the fallback returns its own)."""
        ckv = kv_cache_layer[..., : self.kv_lora_rank]
        kpe = kv_cache_layer[..., self.kv_lora_rank :]
        if self.fallback:
            return self._run_fallback(q_nope, q_pe, kv_cache_layer, return_lse)
        return self.attn_wrapper.run(q_nope.to(self.dtype), q_pe.to(self.dtype), ckv, kpe, out=out, lse=lse,
                                     return_lse=return_lse, return_lse_base_on_e=return_lse)

    def _run_fallback(self, q_nope: torch.Tensor, q_pe: torch.Tensor, kv_cache_layer: torch.Tensor,
                      return_lse: bool = False):
        """Per-request gather of the paged latents and a dense fp32 softmax attention:
        the reference semantics of the kernel, for latent shapes it does not support."""
        assert self._fb is not None, "plan() before run()"
        assert q_nope.shape[1] == self.num_qo_heads, "the kernel runs the query heads it was planned for"
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


class FlashInferMLAOwnTokensWrapper(FlashInferMLAWrapper):
    """A step's own tokens attending each other from the latents ``run`` is passed (``[T, latent]``),
    with no copy: the MLA kernel over pages of one slot, page ``i`` the step's token ``i``, so each
    row's keys are its own queries' tokens."""

    def __init__(self, workspace_buffer: torch.Tensor, dtype: torch.dtype, **kwargs):
        super().__init__(workspace_buffer, page_size=1, **kwargs)
        self._plan_dtype = dtype
        self._out_buf = self._lse_buf = None
        if self.use_cuda_graph and not self.fallback:
            # as the ragged wrapper's (`_FlashInferRaggedPrefillBase`): the kernel writes only the
            # planned rows, so ``plan`` resets a captured bucket's padding to out 0, lse -inf, which
            # merges to 0. ``run`` returns these buffers themselves (no clone, unlike the ragged out):
            # read them before the next ``run``. A page a token, so the pages are the bucket's tokens
            rows = (kwargs["max_num_pages"], self.num_qo_heads)
            self._out_buf = torch.zeros(*rows, self.kv_lora_rank, dtype=dtype, device=self.device)
            self._lse_buf = torch.full(rows, -math.inf, dtype=torch.float32, device=self.device)

    def plan(self, qo_indptr: torch.Tensor, causal: bool = True):
        tokens = int(qo_indptr[-1])
        super().plan(
            qo_indptr, qo_indptr, torch.arange(tokens, dtype=torch.int32),
            torch.ones(qo_indptr.shape[0] - 1, dtype=torch.int32), causal=causal, dtype=self._plan_dtype,
        )
        if self._out_buf is not None:
            self._out_buf[tokens:].zero_()
            self._lse_buf[tokens:].fill_(-math.inf)

    def forget_plan(self) -> None:
        """Every ``plan`` plans its kernel, so there is nothing to forget."""

    def run(self, q_nope: torch.Tensor, q_pe: torch.Tensor, latent: torch.Tensor, return_lse: bool = False):
        owned = {}
        if self._out_buf is not None:
            n = q_nope.shape[0]
            assert n <= self._out_buf.shape[0], (
                f"{type(self).__name__}: {n} rows exceeds the "
                f"{self._out_buf.shape[0]} this graph-mode wrapper was built for"
            )
            owned = dict(out=self._out_buf[:n], lse=self._lse_buf[:n] if return_lse else None)
        return super().run(q_nope, q_pe, latent.unsqueeze(1), return_lse=return_lse, **owned)


class FlashInferMLAManager(AttentionManager):
    supports_context_parallel = True

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
        self._workspaces = WorkspacePool(device)
        # over a placed pool on the FA planner (the kernel in use), the device's SMs, for its work-item cap
        self._fa_sm_count: int | None = None

    def _bind_placement(self, placement: ContextParallel) -> None:
        # the latent is every TP rank's, so under TP placement it is attended for every partner's heads
        cfg = self._kv_config
        self._wrapper_kwargs.update(placement.local_geometry(cfg.tokens_per_page, cfg.num_qo_heads))
        # the merge reads the natural-log lse fa2 and fa3 are known to return (`AttentionManager.build`
        # refuses the others); under auto a plan they refuse would fall through to another backend
        if self._wrapper_kwargs["backend"] == "auto" and torch.device(self._device).type == "cuda":
            from mstar.platforms import current_platform

            self._wrapper_kwargs["backend"] = "fa3" if current_platform.arch(self._device) == "sm_90" else "fa2"
            logger.info(
                "MLA attention over context-parallel KV pool %r plans on %s for flashinfer_backend 'auto'",
                self._kv_cache_name, self._wrapper_kwargs["backend"],
            )
        if self._wrapper_kwargs["backend"] in ("fa2", "fa3") and self.uses_kernel:
            import flashinfer
            from packaging.version import Version

            if Version(flashinfer.__version__).public != _FA_PLANNER_VERSION:
                raise ValueError(
                    f"MLA attention over context-parallel KV pool {self._kv_cache_name!r} refuses, on every "
                    f"rank alike, the steps FlashInfer {_FA_PLANNER_VERSION}'s FA planner may refuse on one "
                    f"rank alone, from a copy of that planner's rules; FlashInfer {flashinfer.__version__} "
                    f"is installed. Install flashinfer-python=={_FA_PLANNER_VERSION}"
                )
            # FA's causal tile-bound check can refuse a causal plan at a head count that doesn't divide
            # its query tile, which is 64 on a plan averaging few queries a row (a captured bucket's
            # padding rows make one). The backend is pinned, so such a step would fail mid-serving
            # rather than fall through: refused here instead, under TP placement too
            heads = self._kv_config.num_qo_heads
            if 64 % heads:
                raise ValueError(
                    f"MLA attention over context-parallel KV pool {self._kv_cache_name!r} plans on FlashInfer's "
                    f"FA planner, whose causal tile-bound check can refuse a step at {heads} query heads a rank: "
                    "use a TP size that gives each rank a head count dividing 64, or turn context_parallel off"
                )
            self._fa_sm_count = _sm_count(self._device)

    @property
    def uses_kernel(self) -> bool:
        """False when the torch fallback serves the plans, which cannot be captured."""
        return torch.device(self._device).type == "cuda" and flashinfer_mla_supports(
            self._kv_config.kv_lora_rank, self._kv_config.qk_rope_head_dim)

    def _cg_wrapper(self, lease: SlotLease, label: str, num_rows: int, force_prefill: bool = False):
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
            bucket = lease.bucket
            kwargs = {**self._wrapper_kwargs, "max_num_pages": captured_page_capacity(
                num_rows, bucket.num_tokens, self._kv_config), "batch_size": num_rows, "use_cuda_graph": True}
            if self._placement is not None:
                wrapper = self._context_parallel_wrapper(
                    label, lease.slot, one_token_rows(bucket.bs, bucket.num_tokens, force_prefill), kwargs,
                    bucket.num_tokens,
                )
            else:
                wrapper = FlashInferMLAWrapper(workspace_buffer=self._workspaces.get(label, lease.slot), **kwargs)
            self._cg_plan_states[key] = wrapper
        return wrapper

    def _eager_wrapper(self, label: str, slot: int, is_decode: bool = False):
        """The persistent eager wrapper for one (label, slot), and over a placed pool kind: its
        plan stages into a pinned buffer the wrapper holds, so one per slot keeps plan(N+1) off
        step N's queued copy. Takes the workspace this slot's captured wrappers hold; the two
        never run at once."""
        key = EagerSlotKey(label=label, slot=slot, is_decode=is_decode)
        wrapper = self._eager_plan_states.get(key)
        if wrapper is None and self._placement is not None:
            wrapper = self._context_parallel_wrapper(label, slot, is_decode, dict(self._wrapper_kwargs))
        elif wrapper is None:
            wrapper = FlashInferMLAWrapper(
                workspace_buffer=self._workspaces.get(label, slot), **self._wrapper_kwargs)
        self._eager_plan_states[key] = wrapper
        return wrapper

    def _context_wrapper(self, workspace, decode, kwargs, max_total_tokens):
        # one wrapper takes either kind of step
        return FlashInferMLAWrapper(workspace_buffer=workspace, **kwargs)

    def _own_tokens_wrapper(self, workspace, kwargs, max_total_tokens):
        own = {**kwargs, "num_qo_heads": self._kv_config.num_qo_heads}
        del own["page_size"]
        if max_total_tokens is not None:
            own["max_num_pages"] = max_total_tokens
        return FlashInferMLAOwnTokensWrapper(workspace, self._dtype, **own)

    @property
    def supports_preplan(self):
        return True

    @property
    def force_double_buffer(self):
        # FlashInfer's MLA plan stages into pinned buffers per wrapper and copies
        # them to the device without blocking
        return True

    def admit(self, step: AttentionStep, ctx: StepContext) -> AdmitOutcome:
        """Over a placed pool on the FA planner, refuses a step its planner may not hold
        (``_refuse_past_work_item_cap``) before it is planned, so the worker retries it smaller:
        sized by how many of its requests, in its order, fit, or terminal for a request that
        doesn't fit alone (or alone in its capture's bucket, whose padding counts). The bound reads
        query lengths alone, so every rank refuses alike."""
        # a step of one query a row (decode) is about a work item a row, far under the cap
        if self._fa_sm_count is None or all(segment.span == 1 for segment in step.segments):
            return ADMIT_OK
        # request -> label -> its queries this step
        rows: dict[str, dict[str, int]] = {}
        for segment in step.segments:
            rows.setdefault(segment.request_id, {})[segment.label] = segment.span
        spans = list(rows.values())
        if self._fits(spans):
            return ADMIT_OK
        where = f"FlashInfer's FA MLA planner past its cap of {_FA_MAX_WORK_ITEMS} work items over " \
                f"context-parallel KV pool {self._kv_cache_name!r}"
        # sized by the real rows: a capture's padding rows (no queries, but they move the planner's
        # tile) go with the bucket, so each retry is strictly smaller
        real = [rid for rid in rows if not ctx.is_padding_row(rid)]
        alone = frozenset(rid for rid in real if not self._fits([rows[rid]]))
        fit = next((n for n in range(len(real) - 1, 0, -1) if self._fits([rows[rid] for rid in real[:n]])), None)
        if alone or fit is None:
            # (no fit: one real row, which its capture's padding takes past the cap)
            return AdmitOutcome(ok=False, reason=AdmitRuntimeError(
                message=f"The request's prompt takes {where} in one step; shorten it.{self._cap_hint()}",
                request_ids=alone or frozenset(real),
            ))
        return AdmitOutcome(ok=False, reason=AllocationFailed(
            message=f"{len(real)} requests' queries take {where}; {fit} of them fit",
            # named for the KV cache's unit; work items are what is short here
            pages_short=1,
            label=step.segments[0].label,
            request_id=real[fit],
            rows_fit=fit,
            rows_fit_alone=fit,
            rows_sized=len(real),
        ))

    def _fits(self, rows: list[dict[str, int]]) -> bool:
        """Whether each label's plan over ``rows`` (label -> queries, a request each) stays within
        the FA planner's work-item cap."""
        labels = {label for row in rows for label in row}
        heads = self._wrapper_kwargs["num_qo_heads"]
        return all(
            fa_work_items_bound(np.cumsum([0, *(row[label] for row in rows if label in row)]), heads,
                                self._fa_sm_count) <= _FA_MAX_WORK_ITEMS
            for label in labels
        )

    def _cap_hint(self) -> str:
        if self._placement.over_sp:
            return ""
        return (f" Under TP placement each rank plans every partner's query heads, so a step reaches the cap "
                f"{self._placement.size}x sooner; placing positions over SP ranks avoids that.")

    def plan(self, step: AttentionStep, ctx: StepContext):
        if step.context_only and self._placement is not None:
            raise ValueError(
                "FlashInferMLAManager does not plan a context_only step over a context-parallel KV pool"
            )
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
                indptrs = indptrs.context_only(
                    self._kv_config.tokens_per_page, torch.from_numpy(np.cumsum([0, *spans], dtype=np.int32)),
                )
            rows = indptrs.qo_indptr.shape[0] - 1
            if lease is not None:
                wrapper = self._cg_wrapper(lease, label, rows, ctx.force_prefill)
            else:
                # one wrapper takes either kind of step, but over a placed pool they differ
                is_decode = self._placement is not None and one_token_rows(
                    rows, int(indptrs.qo_indptr[-1]), ctx.force_prefill,
                )
                wrapper = self._eager_wrapper(label, ctx.slot, is_decode)
            if self._placement is not None:
                self._refuse_past_work_item_cap(indptrs.qo_indptr)
                wrapper.plan(kv_out, causal=step.causal)
            else:
                wrapper.plan(causal=step.causal, dtype=self._dtype, **indptrs.to_kwargs_dict())
            plan_states[label] = wrapper
        self._preplanned = ctx.is_preplan

    def _refuse_past_work_item_cap(self, qo_indptr: torch.Tensor) -> None:
        """Refuses, before FlashInfer can, a step its FA planner may not hold. The planner counts
        this rank's own KV lengths, so its refusal could come on one rank of the lockstep instance
        alone; the bound reads the step's query lengths, the same on every rank. A backstop: ``admit``
        bounds the same plan, a capture's padding rows included, and refuses such a step first."""
        if self._fa_sm_count is None:
            return
        heads = self._wrapper_kwargs["num_qo_heads"]
        bound = fa_work_items_bound(qo_indptr, heads, self._fa_sm_count)
        if bound > _FA_MAX_WORK_ITEMS:
            raise ValueError(
                f"a step of {int(qo_indptr[-1])} query tokens at {heads} heads can take FlashInfer's FA MLA "
                f"planner up to {bound} work items over context-parallel KV pool {self._kv_cache_name!r}, past "
                f"its cap of {_FA_MAX_WORK_ITEMS}.{self._cap_hint()}"
            )

    def clear_preplan(self):
        self._preplanned = False
        self._preplan_states = {}

    @torch.compiler.disable
    def qo_indptr_buf(self, label: str = "main") -> torch.Tensor | None:
        wrapper = self._current_plan_states.get(label)
        return None if wrapper is None else wrapper._qo_indptr_buf

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
        log-sum-exp of the scores ``[T, H]`` fp32, for merging with another key segment.
        ``k`` is the step's latents (``[T, latent]``), read only over a placed pool, where the
        step's own tokens attend each other from them."""
        del v, layer_idx
        if label is None:
            label = self._default_label
        assert q_pe is not None, "MLA needs the rope-part query"
        if self._placement is not None:
            if return_lse:
                raise ValueError("FlashInferMLAManager returns no lse over a context-parallel KV pool")
            return self._attend_placed(label, (q, q_pe), kv_cache_layer, (k,))
        return self._attend(q, q_pe, label, kv_cache_layer, return_lse)

    @torch.compiler.disable
    def _attend(self, q_nope, q_pe, label, kv_cache_layer, return_lse=False):
        out = self._current_plan_states[label].run(q_nope, q_pe, kv_cache_layer, return_lse=return_lse)
        o, lse = out if return_lse else (out, None)
        if o.dtype != q_nope.dtype:
            o = o.to(q_nope.dtype)
        return (o, lse) if return_lse else o
