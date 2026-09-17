"""Weight-absorbed MLA attention over a ``KVLayout.MLA`` latent cache."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.engine.resources.attn.base import AttentionManager, WorkspacePool
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.kv.plan import SINK_PAGE, KVPlanOutputs, SequenceView
from mstar.engine.resources.step import ResourceStep, SlotLease, StepContext
from mstar.utils.pinned_staging import pinned, to_device_async

if TYPE_CHECKING:
    from mstar.engine.resources.kv.config import KVConfig

logger = logging.getLogger(__name__)


# ── Spec / step ─────────────────────────────────────────────────────────


class MlaSubPlan(NamedTuple):
    """One attention pass of a step, per request positionally with the step's segments. Row
    ``j`` of request ``i`` lands at slot ``kv_lens[i] - q_lens[i] + j`` and attends
    ``[0, kv_lens[i])``.
    """
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]


@dataclass(frozen=True)
class MlaAttentionStep(ResourceStep):
    causal: bool = True
    # Several passes over the step's streams inside one forward; None means
    # the one pass the KV segments describe (q = span, kv = stored + span).
    sub_plans: tuple[MlaSubPlan, ...] | None = None
    # Which slot the first entry of ``sub_plans`` occupies, and how many the
    # region has in all. A region can plan its passes in two steps — the
    # ones a readback does not decide before it, the rest after — and the
    # later step must land beside, not over, the earlier one's plans.
    first_sub_plan: int = 0
    num_sub_plans: int | None = None


@functools.cache
def _mla_kernel_available(ckv: int, kpe: int, sm_major: int) -> bool:
    """Whether the FlashInfer MLA kernel can serve these latent dims."""
    if not (ckv == 512 and kpe == 64):
        return False
    if sm_major != 9:
        return False
    try:
        import flashinfer.mla  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def paged_scatter_map_host(
    qo_indptr: list[int],
    kv_indptr: list[int],
    kv_indices: list[int],
    kv_len_arr: list[int],
    page_size: int,
) -> tuple[list[int], list[int]]:
    """Where each new token of a planned batch lands, on the host."""
    t2p: list[int] = []
    t2c: list[int] = []
    for i in range(len(qo_indptr) - 1):
        length = qo_indptr[i + 1] - qo_indptr[i]
        if length <= 0:
            continue
        base = kv_len_arr[i] - length
        ptr = kv_indptr[i]
        for j in range(length):
            g = base + j
            t2p.append(kv_indices[ptr + g // page_size])
            t2c.append(g % page_size)
    return t2p, t2c


class _HostPlan(NamedTuple):
    """One pass's index lists, as the wrapper and the scatter map consume them."""
    qo_indptr: list[int]
    kv_indptr: list[int]
    kv_indices: list[int]
    kv_len_arr: list[int]
    page_tables: list[list[int]]  # per request, the pages its kv_len spans

    @property
    def total_tokens(self) -> int:
        return self.qo_indptr[-1]


def build_host_plan(
    views: list[SequenceView], sub_plan: MlaSubPlan, page_size: int,
) -> _HostPlan:
    """The pass's indptrs over the streams' page tables."""
    if len(sub_plan.q_lens) != len(views) or len(sub_plan.kv_lens) != len(views):
        raise ValueError(
            f"sub-plan over {len(sub_plan.q_lens)}/{len(sub_plan.kv_lens)} "
            f"requests for a step of {len(views)}"
        )
    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices: list[int] = []
    kv_len_arr: list[int] = []
    page_tables: list[list[int]] = []
    for view, q_len, kv_len in zip(views, sub_plan.q_lens, sub_plan.kv_lens, strict=True):
        if q_len < 0 or kv_len < q_len:
            raise ValueError(
                f"{view.request_id}: sub-plan q_len={q_len} kv_len={kv_len}"
            )
        if kv_len > view.length:
            raise ValueError(
                f"{view.request_id}: sub-plan attends {kv_len} tokens but the "
                f"step declared {view.length}; widen the segment's span"
            )
        num_pages = -(-kv_len // page_size)
        pages = list(view.page_idxs[:num_pages])
        qo_indptr.append(qo_indptr[-1] + q_len)
        kv_indices.extend(pages)
        kv_indptr.append(kv_indptr[-1] + len(pages))
        kv_len_arr.append(kv_len)
        page_tables.append(pages)
    return _HostPlan(qo_indptr, kv_indptr, kv_indices, kv_len_arr, page_tables)


# ── Wrapper: FlashInfer kernel ──────────────────────────────────────────


class FlashInferMLAWrapper:
    """``flashinfer.mla.BatchMLAPagedAttentionWrapper`` plus the latent scatter
    map, with static buffers under a CUDA graph.
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        *,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        sm_scale: float,
        device: torch.device,
        batch_size: int | None = None,
        max_num_pages: int | None = None,
        max_total_tokens: int | None = None,
        use_cuda_graph: bool = False,
        backend: str = "auto",
    ):
        import flashinfer

        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.num_heads = num_heads
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe
        self.page_size = page_size
        self.sm_scale = sm_scale
        self.max_total_tokens = max_total_tokens
        self.dtype: torch.dtype | None = None
        self._total_tokens = 0
        self._qo_indptr_buf: torch.Tensor | None = None

        if use_cuda_graph:
            assert batch_size is not None and max_num_pages is not None
            assert max_total_tokens is not None
            # stable addresses for graph replay
            self._qo_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            self._kv_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            self._kv_indices_buf = torch.zeros(max_num_pages, dtype=torch.int32, device=device)
            self._kv_len_arr_buf = torch.zeros(batch_size, dtype=torch.int32, device=device)
            self.attn_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self._qo_indptr_buf,
                kv_indptr=self._kv_indptr_buf,
                kv_indices=self._kv_indices_buf,
                kv_len_arr=self._kv_len_arr_buf,
                backend=backend,
            )
            self.token_to_page = torch.full(
                (max_total_tokens,), SINK_PAGE, dtype=torch.long, device=device
            )
            self.token_to_cache = torch.zeros(max_total_tokens, dtype=torch.long, device=device)
        else:
            self.attn_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                workspace_buffer, backend=backend,
            )
            self.token_to_page = None
            self.token_to_cache = None

        # Fence between consecutive plans on this wrapper: FlashInfer's plan
        # copies its own pinned int workspace to the device non_blocking with
        # no guard, so the next plan must not overwrite that source before the
        # DMA ran. Normally complete already; it only bites when the host is
        # more than one replay ahead of the GPU.
        self._plan_event: torch.cuda.Event | None = None

    @torch.compiler.disable
    def plan(self, host: _HostPlan, *, causal: bool, dtype: torch.dtype) -> None:
        self.dtype = dtype
        if self._plan_event is not None and not self._plan_event.query():
            self._plan_event.synchronize()

        qo_indptr = pinned(host.qo_indptr)
        kv_indptr = pinned(host.kv_indptr)
        kv_indices = pinned(host.kv_indices)
        kv_len_arr = pinned(host.kv_len_arr)
        self.attn_wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_len_arr,
            self.num_heads, self.head_dim_ckv, self.head_dim_kpe,
            self.page_size, causal, self.sm_scale, dtype, dtype,
        )

        t2p, t2c = paged_scatter_map_host(
            host.qo_indptr, host.kv_indptr, host.kv_indices, host.kv_len_arr,
            self.page_size,
        )
        n = len(t2p)
        self._total_tokens = n
        if self.use_cuda_graph:
            if n > self.max_total_tokens:
                raise ValueError(
                    f"plan of {n} tokens on a wrapper captured for {self.max_total_tokens}"
                )
            if n:
                self.token_to_page[:n].copy_(pinned(t2p, torch.long), non_blocking=True)
                self.token_to_cache[:n].copy_(pinned(t2c, torch.long), non_blocking=True)
            if n < self.max_total_tokens:
                # the captured scatter writes max_total_tokens rows every
                # replay; the tail aims at the sink page
                self.token_to_page[n:].fill_(SINK_PAGE)
                self.token_to_cache[n:].fill_(0)
        else:
            self.token_to_page = to_device_async(t2p, torch.long, self.device)
            self.token_to_cache = to_device_async(t2c, torch.long, self.device)
            self._qo_indptr_buf = to_device_async(host.qo_indptr, torch.int32, self.device)

        if self.device.type == "cuda":
            if self._plan_event is None:
                self._plan_event = torch.cuda.Event()
            self._plan_event.record(torch.cuda.current_stream())

    def _rows(self, given: int) -> int:
        """How many rows a call touches: the plan's real count eagerly; under
        a graph every row the caller hands over (the capture bakes it in,
        and the scatter map's tail past the plan aims at the sink page)."""
        if not self.use_cuda_graph:
            return self._total_tokens
        return min(given, self.max_total_tokens)

    @torch.compiler.disable
    def write_latent(self, latent_layer: torch.Tensor, latent: torch.Tensor) -> None:
        """Scatter one latent row per new token into the layer's pages."""
        n = self._rows(latent.shape[0])
        latent_layer[self.token_to_page[:n], self.token_to_cache[:n]] = (
            latent[:n].to(latent_layer.dtype)
        )

    @torch.compiler.disable
    def run(
        self, q_nope: torch.Tensor, q_pe: torch.Tensor, latent_layer: torch.Tensor,
    ) -> torch.Tensor:
        ckv = latent_layer[..., :self.head_dim_ckv]
        kpe = latent_layer[..., self.head_dim_ckv:]
        return self.attn_wrapper.run(
            q_nope.to(self.dtype), q_pe.to(self.dtype), ckv, kpe, return_lse=False,
        )


# ── Wrapper: shape-static SDPA fallback ─────────────────────────────────


class SdpaMLAWrapper:
    """Reference MLA attention over the same plan, in plain torch."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim_ckv: int,
        page_size: int,
        sm_scale: float,
        device: torch.device,
        batch_size: int,
        max_pages_per_request: int,
        max_total_tokens: int,
        use_cuda_graph: bool = False,
    ):
        del num_heads
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.head_dim_ckv = head_dim_ckv
        self.page_size = page_size
        self.sm_scale = sm_scale
        self.batch_size = batch_size
        self.max_pages = max_pages_per_request
        self.max_total_tokens = max_total_tokens
        self.dtype: torch.dtype | None = None
        self._total_tokens = 0
        # static like every other buffer here: a captured replay reads it at a
        # fixed address, so plan() stages into it rather than rebinding it
        self._qo_indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        self.token_to_page = torch.full(
            (max_total_tokens,), SINK_PAGE, dtype=torch.long, device=device
        )
        self.token_to_cache = torch.zeros(max_total_tokens, dtype=torch.long, device=device)
        # per token: which request, absolute query position
        self.token_to_req = torch.zeros(max_total_tokens, dtype=torch.long, device=device)
        self.token_to_pos = torch.zeros(max_total_tokens, dtype=torch.long, device=device)
        # per request: page table (sink-padded) and visible length
        self.page_table = torch.full(
            (batch_size, max_pages_per_request), SINK_PAGE,
            dtype=torch.long, device=device,
        )
        self.kv_lens = torch.zeros(batch_size, dtype=torch.long, device=device)

    def _stage(self, buf: torch.Tensor, values: list[int], fill: int) -> None:
        n = len(values)
        if n:
            buf[:n].copy_(pinned(values, torch.long), non_blocking=True)
        if n < buf.numel():
            buf[n:].fill_(fill)

    @torch.compiler.disable
    def plan(self, host: _HostPlan, *, causal: bool, dtype: torch.dtype) -> None:
        if not causal:
            raise NotImplementedError("the MLA fallback is causal only")
        self.dtype = dtype
        t2p, t2c = paged_scatter_map_host(
            host.qo_indptr, host.kv_indptr, host.kv_indices, host.kv_len_arr,
            self.page_size,
        )
        n = len(t2p)
        if n > self.max_total_tokens or len(host.kv_len_arr) > self.batch_size:
            raise ValueError("plan larger than the fallback's static buffers")
        self._total_tokens = n
        t2r: list[int] = []
        t2q: list[int] = []
        for i in range(len(host.qo_indptr) - 1):
            length = host.qo_indptr[i + 1] - host.qo_indptr[i]
            base = host.kv_len_arr[i] - length
            t2r.extend([i] * length)
            t2q.extend(range(base, base + length))
        self._stage(self.token_to_page, t2p, SINK_PAGE)
        self._stage(self.token_to_cache, t2c, 0)
        self._stage(self.token_to_req, t2r, 0)
        # a padding token attends nothing: position -1 masks every key
        self._stage(self.token_to_pos, t2q, -1)
        table = torch.full((self.batch_size, self.max_pages), SINK_PAGE, dtype=torch.long)
        for i, pages in enumerate(host.page_tables):
            if len(pages) > self.max_pages:
                raise ValueError(
                    f"request {i} spans {len(pages)} pages, fallback holds {self.max_pages}"
                )
            table[i, :len(pages)] = torch.as_tensor(pages, dtype=torch.long)
        self.page_table.copy_(pinned(table, torch.long), non_blocking=True)
        self._stage(self.kv_lens, host.kv_len_arr, 0)
        qo = host.qo_indptr
        self._qo_indptr_buf[:len(qo)].copy_(pinned(qo, torch.int32), non_blocking=True)
        if len(qo) < self._qo_indptr_buf.numel():
            self._qo_indptr_buf[len(qo):].fill_(qo[-1] if qo else 0)

    _rows = FlashInferMLAWrapper._rows

    @torch.compiler.disable
    def write_latent(self, latent_layer: torch.Tensor, latent: torch.Tensor) -> None:
        n = self._rows(latent.shape[0])
        latent_layer[self.token_to_page[:n], self.token_to_cache[:n]] = (
            latent[:n].to(latent_layer.dtype)
        )

    @torch.compiler.disable
    def run(
        self, q_nope: torch.Tensor, q_pe: torch.Tensor, latent_layer: torch.Tensor,
    ) -> torch.Tensor:
        n = self._rows(q_nope.shape[0])
        q = torch.cat([q_nope[:n], q_pe[:n]], dim=-1).float()  # (n, H, D)
        # (bs, max_pages * page_size, D): every request's visible pages
        gathered = latent_layer[self.page_table].reshape(self.batch_size, -1, latent_layer.shape[-1]).float()
        keys = gathered[self.token_to_req[:n]]  # (n, L, D)
        scores = torch.einsum("nhd,nld->nhl", q, keys) * self.sm_scale
        k_pos = torch.arange(keys.shape[1], device=q.device)
        visible = (k_pos[None, :] <= self.token_to_pos[:n, None]) & (
            k_pos[None, :] < self.kv_lens[self.token_to_req[:n]][:, None]
        )
        scores = scores.masked_fill(~visible[:, None, :], float("-inf"))
        attn = scores.softmax(-1)
        # a fully masked (padding) row softmaxes to NaN; it is discarded anyway
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.einsum("nhl,nld->nhd", attn, keys[..., :self.head_dim_ckv])
        return out.to(q_nope.dtype)


MLAWrapper = FlashInferMLAWrapper | SdpaMLAWrapper


@dataclass
class _LabelPlan:
    wrappers: list[MLAWrapper]
    host_plans: list[_HostPlan]
    sub: int = 0

    @property
    def current(self) -> MLAWrapper:
        return self.wrappers[self.sub]


# ── The resource ────────────────────────────────────────────────────────


class MlaAttentionManager(AttentionManager):
    """Built from an ``AttentionSpec`` whose backend is ``AttnBackend.MLA``;
    ``AttentionManager.build`` constructs it.
    """

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: "KVConfig",
        softmax_scale: float,
        ckv_dim: int,
        backend: str = "auto",
    ):
        from mstar.engine.resources.kv.config import KVLayout

        if kv_config.layout != KVLayout.MLA:
            raise ValueError(
                f"MLA attention needs a KVLayout.MLA cache, {kv_cache!r} is {kv_config.layout}"
            )
        if not 0 < ckv_dim < kv_config.head_dim:
            raise ValueError(f"ckv_dim={ckv_dim} must split head_dim={kv_config.head_dim}")
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        self._scale = softmax_scale
        self._ckv = ckv_dim
        self._kpe = kv_config.head_dim - ckv_dim
        self._backend = backend
        sm_major = (
            torch.cuda.get_device_capability(device)[0]
            if device.type == "cuda" and torch.cuda.is_available() else 0
        )
        self._use_kernel = _mla_kernel_available(self._ckv, self._kpe, sm_major)
        if not self._use_kernel:
            logger.warning(
                "MLA attention %s: FlashInfer MLA kernel unavailable (ckv=%d, kpe=%d, "
                "sm%d); using the SDPA fallback", kv_cache, self._ckv, self._kpe, sm_major,
            )
        self._workspaces = WorkspacePool(device)

        self._current: dict[str, _LabelPlan] = {}
        self._eager: dict[str, _LabelPlan] = {}  # persistent eager wrappers per label
        self._cg: dict[CGSlotKey, _LabelPlan] = {}
        self._preplan: dict[str, _LabelPlan] = {}
        self._preplanned = False

    @property
    def requires_kv_write(self) -> bool:
        """False: this resource owns the latent write (``write_latent``), so a
        layer must not also write through the KV resource.
        """
        return False

    def depends_on(self):
        return {self._kv_cache_name}

    @property
    def uses_kernel(self) -> bool:
        return self._use_kernel

    @property
    def ckv_dim(self) -> int:
        return self._ckv

    # ── wrappers ──

    def _new_wrapper(
        self, workspace: torch.Tensor, *, batch_size: int, max_total_tokens: int,
        use_cuda_graph: bool,
    ) -> MLAWrapper:
        if self._use_kernel:
            return FlashInferMLAWrapper(
                workspace,
                num_heads=self._kv_config.num_qo_heads,
                head_dim_ckv=self._ckv, head_dim_kpe=self._kpe,
                page_size=self._kv_config.page_size, sm_scale=self._scale,
                device=self._device, batch_size=batch_size,
                max_num_pages=self._kv_config.max_num_pages,
                max_total_tokens=max_total_tokens,
                use_cuda_graph=use_cuda_graph, backend=self._backend,
            )
        page_size = self._kv_config.page_size
        max_pages = min(
            -(-self._kv_config.max_seq_len // page_size), self._kv_config.max_num_pages,
        )
        return SdpaMLAWrapper(
            num_heads=self._kv_config.num_qo_heads, head_dim_ckv=self._ckv,
            page_size=page_size, sm_scale=self._scale, device=self._device,
            batch_size=batch_size, max_pages_per_request=max_pages,
            max_total_tokens=max_total_tokens, use_cuda_graph=use_cuda_graph,
        )

    def _wrappers(
        self, workspace: torch.Tensor, n_sub: int, *, batch_size: int,
        max_total_tokens: int, use_cuda_graph: bool,
    ) -> list[MLAWrapper]:
        """One wrapper per sub-plan, each on its own 256-byte-aligned slice of
        the workspace: every sub-plan is live inside one replay."""
        if n_sub == 1:
            slices = [workspace]
        else:
            slice_len = (workspace.numel() // n_sub) // 256 * 256
            slices = [workspace.narrow(0, i * slice_len, slice_len) for i in range(n_sub)]
        return [
            self._new_wrapper(
                ws, batch_size=batch_size, max_total_tokens=max_total_tokens,
                use_cuda_graph=use_cuda_graph,
            )
            for ws in slices
        ]

    def _cg_plan(self, lease: SlotLease, label: str, num_rows: int, n_sub: int) -> _LabelPlan:
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        plan = self._cg.get(key)
        if plan is None:
            plan = self._cg[key] = _LabelPlan(
                wrappers=self._wrappers(
                    self._workspaces.get(label, lease.slot), n_sub,
                    batch_size=num_rows, max_total_tokens=lease.bucket.num_tokens,
                    use_cuda_graph=True,
                ),
                host_plans=[],
            )
        elif len(plan.wrappers) != n_sub:
            raise ValueError(
                f"{key}: captured with {len(plan.wrappers)} sub-plans, planned with {n_sub}"
            )
        return plan

    def _eager_plan(self, label: str, num_rows: int, total_tokens: int, n_sub: int) -> _LabelPlan:
        plan = self._eager.get(label)
        # the fallback's buffers are static per wrapper; regrow when a batch
        # outgrows them (never on the kernel path, whose eager wrapper has none)
        if plan is not None and not self._use_kernel:
            w = plan.wrappers[0]
            if num_rows > w.batch_size or total_tokens > w.max_total_tokens:
                plan = None
        if plan is None or len(plan.wrappers) < n_sub:
            plan = self._eager[label] = _LabelPlan(
                wrappers=self._wrappers(
                    self._workspaces.get(label), max(n_sub, 1),
                    batch_size=max(num_rows, 1), max_total_tokens=max(total_tokens, 1),
                    use_cuda_graph=False,
                ),
                host_plans=[],
            )
        return plan

    # ── step lifecycle ──

    @property
    def supports_preplan(self):
        return True

    def clear_preplan(self):
        self._preplanned = False
        self._preplan = {}

    def plan(self, step: MlaAttentionStep, ctx: StepContext) -> None:
        self.reset_default_cursors()
        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a cuda graph step: eager wrappers are shared "
            "with the forward still in flight"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "MLA preplan already pending; clear_preplan before planning ahead"
        )
        if self._preplanned:
            self._current = self._preplan
            self._preplan = {}
            self._preplanned = False
            return

        kv_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert kv_outputs is not None, (
            f"MLA attention expected the plan of {self._kv_cache_name}"
        )
        target = self._preplan if ctx.is_preplan else self._current
        target.clear()
        page_size = self._kv_config.page_size
        first = step.first_sub_plan
        for label, kv_out in kv_outputs.items():
            views = kv_out.views
            # None: the one pass the segments describe; () : nothing to plan
            # in this phase (a second phase with no chain rows)
            sub_plans = step.sub_plans if step.sub_plans is not None else (MlaSubPlan(
                q_lens=tuple(v.to_compute for v in views),
                kv_lens=tuple(v.length for v in views),
            ),)
            n_sub = step.num_sub_plans or (first + len(sub_plans))
            if first + len(sub_plans) > n_sub:
                raise ValueError(
                    f"sub-plans {first}..{first + len(sub_plans) - 1} of {n_sub}"
                )
            hosts = [build_host_plan(views, sp, page_size) for sp in sub_plans]
            if lease is not None:
                plan = self._cg_plan(lease, label, len(views), n_sub)
            else:
                plan = self._eager_plan(
                    label, len(views), max((h.total_tokens for h in hosts), default=0), n_sub,
                )
            for i, host in enumerate(hosts):
                plan.wrappers[first + i].plan(host, causal=step.causal, dtype=self._dtype)
            if len(plan.host_plans) != n_sub:
                plan.host_plans = [None] * n_sub
            plan.host_plans[first:first + len(hosts)] = hosts
            plan.sub = 0
            target[label] = plan
        self._preplanned = ctx.is_preplan

    # ── what the layers call ──

    def _plan_for(self, label: str | None) -> _LabelPlan:
        return self._current[self._default_label if label is None else label]

    @torch.compiler.disable
    def select_plan_slot(self, sub: int, label: str | None = None) -> None:
        """Make sub-plan ``sub`` the one ``write_latent``/``run`` use. Host
        state only: inside a captured region the choice is baked into the
        graph, which is the point."""
        plan = self._plan_for(label)
        if not 0 <= sub < len(plan.wrappers):
            raise IndexError(f"sub-plan {sub} of {len(plan.wrappers)}")
        plan.sub = sub

    @torch.compiler.disable
    def write_latent(
        self, latent: torch.Tensor, latent_layer: torch.Tensor, label: str | None = None,
    ) -> None:
        """Scatter this pass's latent rows ([tokens, ckv + kpe]) into the
        layer's pages (``kv.layer_view(layer_idx)``)."""
        self._plan_for(label).current.write_latent(latent_layer, latent)

    @torch.compiler.disable
    def run(
        self, q_nope: torch.Tensor, q_pe: torch.Tensor, latent_layer: torch.Tensor,
        label: str | None = None,
    ) -> torch.Tensor:
        """[tokens, H, ckv] x [tokens, H, kpe] -> [tokens, H, ckv]."""
        out = self._plan_for(label).current.run(q_nope, q_pe, latent_layer)
        return out if out.dtype == q_nope.dtype else out.to(q_nope.dtype)

    @torch.compiler.disable
    def qo_indptr_buf(self, label: str = "main") -> torch.Tensor | None:
        plan = self._current.get(label)
        return None if plan is None else plan.current._qo_indptr_buf

    def select_last_hidden(self, hidden: torch.Tensor, label: str = "main") -> torch.Tensor:
        """The last row of every request, for sampling after a prefill."""
        qo_indptr = self.qo_indptr_buf(label)
        return hidden.index_select(0, (qo_indptr[1:] - 1).long())

    def page_tables(self, label: str = "main") -> list[list[int]]:
        """The current sub-plan's per-request page tables (host), e.g. for a
        sparse-attention gather over the raw pages."""
        plan = self._plan_for(label)
        return plan.host_plans[plan.sub].page_tables

    def host_plan(self, label: str = "main") -> _HostPlan:
        plan = self._plan_for(label)
        return plan.host_plans[plan.sub]

    def cleanup(self):
        self._current = {}
        self._eager = {}
        self._cg = {}
        self._preplan = {}
