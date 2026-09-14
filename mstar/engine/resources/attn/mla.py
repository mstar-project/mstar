"""Weight-absorbed multi-head latent attention over a compressed latent cache.

Not drop-in with the other backends: ``run_mla`` replaces ``run``, and the KV
side writes one latent per token rather than a K/V pair. See
``AttentionCallable.run_mla``, which pairs the two.
"""

import functools
from dataclasses import dataclass

import torch

from mstar.engine.resources.attn.base import (
    AttentionManager,
    EagerSlotKey,
    WorkspacePool,
)
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.wrappers import FlashInferMLAWrapper
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.kv.config import KVConfig
from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs
from mstar.engine.resources.step import SlotLease, StepContext


@dataclass(frozen=True)
class MlaRequestSlice:
    """One request's slice of an absorbed-MLA SDPA plan.

    ``q_start``/``seq_len`` locate this request's rows in the packed query
    batch; ``total_len`` is its context length after this step (so the causal
    mask knows how many cached tokens precede the new ones); ``page_indices``
    gathers its latent pages.
    """
    q_start: int
    seq_len: int
    total_len: int
    page_indices: torch.Tensor


@dataclass(frozen=True)
class MlaSdpaPlan:
    """Absorbed-MLA fallback plan: the per-request gather layout.

    Built only when the FlashInfer MLA kernel cannot serve the configured
    latent dims (see :func:`mla_kernel_available_for`); the kernel path plans a
    :class:`FlashInferMLAWrapper` instead. Exactly one of
    ``PlanState.wrapper`` / ``PlanState.sdpa`` is set — ``run_mla`` picks its
    path off that, so leaving a stale value in either is a bug.
    """
    requests: tuple[MlaRequestSlice, ...]


@dataclass
class PlanState:
    """One label's planned attention: the wrapper or the fallback's layout."""
    # device-side, for select_last_hidden
    qo_indptr: torch.Tensor
    wrapper: FlashInferMLAWrapper | None = None
    sdpa: MlaSdpaPlan | None = None


@functools.cache
def _mla_kernel_available(ckv: int, kpe: int, sm_major: int) -> bool:
    """Whether the FlashInfer MLA kernel fast path can serve these latent dims.

    Gated conservatively: ``flashinfer.mla.BatchMLAPagedAttentionWrapper`` is
    hard-locked to the real Kimi dims (ckv=512, kpe=64). Off-dim calls trigger an
    *uncatchable* illegal memory access (it corrupts the CUDA context — not a
    catchable exception), so this decision MUST be made BEFORE any kernel
    construction/call. Requires the flashinfer MLA module, exactly ckv=512/kpe=64,
    and a Hopper (sm90) GPU (``backend="auto"`` -> fa3). Everything else — reduced
    configs, pre-sm90, Blackwell (sm100, which wants the trtllm MLA path), or a
    build without flashinfer — returns False and the manager uses the all-dims
    SDPA fallback.
    """
    if not (ckv == 512 and kpe == 64):
        return False
    if sm_major != 9:
        return False
    try:
        import flashinfer.mla  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def mla_kernel_available_for(
    kv_config: KVConfig,
    mla_ckv_dim: int | None,
    device: torch.device,
) -> bool:
    """Whether these latent dims can use the FlashInfer MLA kernel.

    Total by construction — a non-CUDA device returns False rather than raising,
    so the absorbed-SDPA fallback is reachable on CPU. ``_mla_kernel_available``
    explains why this must be decided *before* any kernel is constructed.
    """
    if mla_ckv_dim is None:
        return False
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return False
    sm_major = torch.cuda.get_device_capability(device)[0]
    return _mla_kernel_available(
        mla_ckv_dim, kv_config.head_dim - mla_ckv_dim, sm_major
    )


class MlaAbsorbManager(AttentionManager):
    """Paged-cache backend for weight-absorbed MLA.

    Reads a 4-D latent cache ``[pages, page_size, ckv + kpe]`` per layer — one
    compressed latent per token, no K/V axis and no KV-head axis. Real Kimi dims
    on sm90 run the FlashInfer MLA kernel; any other dims fall back to eager
    SDPA, which serves every config at the cost of being uncapturable (see
    ``capture_blocked``).
    """

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: KVConfig,
        softmax_scale: float | None = None,
        mla_ckv_dim: int | None = None,
        backend: str = "auto",
        enable_nvtx: bool = False,
    ):
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        self._backend = backend
        self._enable_nvtx = enable_nvtx

        self._ckv_dim = mla_ckv_dim
        # MLA scales by the unabsorbed qk_head_dim, which the latent width does
        # not carry; the model passes it in
        self._softmax_scale = (
            softmax_scale if softmax_scale is not None
            else kv_config.head_dim ** -0.5
        )
        self._has_mla_kernel = mla_kernel_available_for(
            kv_config=kv_config, mla_ckv_dim=mla_ckv_dim, device=device,
        )

        self._cg_wrappers: dict[CGSlotKey, FlashInferMLAWrapper] = {}
        self._eager_wrappers: dict[EagerSlotKey, FlashInferMLAWrapper] = {}
        self._current_plan_states: dict[str, PlanState] = {}
        self._preplan_states: dict[str, PlanState] = {}
        self._preplanned = False

        self._workspaces = WorkspacePool(device)

    def depends_on(self):
        return {self._kv_cache_name}

    @property
    def requires_kv_write(self) -> bool:
        # True, but through `KVManager.write_latent` rather than `write_kv`:
        # there is one compressed latent per token and no K/V pair. See
        # `AttentionCallable.run_mla`, which is what a layer calls.
        return True

    def mla_kernel_available(self) -> bool:
        return self._has_mla_kernel

    @property
    def capture_blocked(self) -> str | None:
        """Only the kernel path is capturable; absorbed SDPA loops over requests
        in Python. ``plan`` asserts the same, so the two cannot drift."""
        if self._has_mla_kernel:
            return None
        kpe = None if self._ckv_dim is None else self._kv_config.head_dim - self._ckv_dim
        return (
            "absorbed MLA without the FlashInfer MLA kernel "
            f"(ckv={self._ckv_dim}, kpe={kpe}, device={self._device}); the "
            "absorbed-SDPA fallback runs eager and plans no wrapper"
        )

    @property
    def supports_preplan(self):
        return True

    @property
    def force_double_buffer(self):
        # same pinned staging inside FlashInfer's plan as the self-attention
        # manager; see `FlashInferManager.force_double_buffer`
        return True

    def _new_wrapper(
        self, label: str, slot: int | None = None,
        batch_size: int | None = None,
    ) -> FlashInferMLAWrapper:
        return FlashInferMLAWrapper(
            workspace_buffer=self._workspaces.get(label, slot),
            num_heads=self._kv_config.num_qo_heads,
            head_dim_ckv=self._ckv_dim,
            head_dim_kpe=self._kv_config.head_dim - self._ckv_dim,
            page_size=self._kv_config.page_size,
            sm_scale=self._softmax_scale,
            batch_size=batch_size,
            max_num_pages=self._kv_config.max_num_pages,
            device=self._device,
            use_cuda_graph=batch_size is not None,
            backend=self._backend,
            enable_nvtx=self._enable_nvtx,
        )

    def _cg_wrapper(
        self, lease: SlotLease, label: str, num_rows: int
    ) -> FlashInferMLAWrapper:
        """The captured-graph wrapper for one (bucket, slot, label). ``num_rows``
        is not ``bucket.bs``; see ``FlashInferManager._cg_wrapper``."""
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        wrapper = self._cg_wrappers.get(key)
        if wrapper is None:
            wrapper = self._cg_wrappers[key] = self._new_wrapper(
                label, lease.slot, batch_size=num_rows,
            )
        return wrapper

    def _eager_wrapper(self, label: str, slot: int) -> FlashInferMLAWrapper:
        """The persistent eager wrapper for one (label, slot). One class covers
        prefill and decode, so there is nothing to key on the step kind."""
        key = EagerSlotKey(label=label, slot=slot)
        wrapper = self._eager_wrappers.get(key)
        if wrapper is None:
            wrapper = self._eager_wrappers[key] = self._new_wrapper(label, slot)
        return wrapper

    def plan(self, step: AttentionStep, ctx: StepContext):
        self.reset_default_cursors()
        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a cuda graph step: an eager wrapper shares its "
            "workspace with the captured one on the same slot"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "attention preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        assert lease is None or self._has_mla_kernel, self.capture_blocked
        if self._preplanned:
            # the wrappers were planned a step early against this same step's
            # KV plan; nothing left to do but promote them
            self._current_plan_states = self._preplan_states
            self._preplan_states = {}
            self._preplanned = False
            return

        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, (
            f"MLA attention manager expected plan result from {self._kv_cache_name}"
        )

        # a preplan leases a different cg slot, so its wrappers and workspaces
        # are disjoint from the ones the in-flight forward reads
        plan_states = (
            self._preplan_states if ctx.is_preplan else self._current_plan_states
        )
        plan_states.clear()
        for label, kv_out in plan_outputs.items():
            plan_states[label] = self._plan_label(label, kv_out, step, ctx, lease)

        self._preplanned = ctx.is_preplan

    def clear_preplan(self):
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}

    def _plan_label(
        self, label: str, kv_out: KVPlanOutput,
        step: AttentionStep, ctx: StepContext, lease: SlotLease | None,
    ) -> PlanState:
        indptrs = kv_out.device_indptrs(self._device)
        state = PlanState(qo_indptr=indptrs.qo_indptr)
        if not self._has_mla_kernel:
            state.sdpa = self._sdpa_plan(kv_out)
            return state

        if lease is not None:
            wrapper = self._cg_wrapper(
                lease, label, num_rows=indptrs.qo_indptr.shape[0] - 1,
            )
        else:
            wrapper = self._eager_wrapper(label, ctx.slot)
        wrapper.plan(
            qo_indptr=indptrs.qo_indptr,
            kv_indptr=indptrs.paged_kv_indptr,
            kv_indices=indptrs.paged_kv_indices,
            # the kernel wants each request's KV length after this step's
            # tokens land, which is what the view's `length` already is
            kv_len_arr=torch.tensor(
                [view.length for view in kv_out.views],
                dtype=torch.int32,
            ).to(self._device, non_blocking=True),
            causal=step.causal,
            dtype=self._dtype,
        )
        state.wrapper = wrapper
        return state

    def _sdpa_plan(self, kv_out: KVPlanOutput) -> MlaSdpaPlan:
        """The eager fallback's gather layout, entirely off the KV plan's
        views, in the packed order they define."""
        requests: list[MlaRequestSlice] = []
        q_start = 0
        for view in kv_out.views:
            assert view.start == 0, (
                f"absorbed MLA reads a stream from its first page; "
                f"{view.label!r} of {view.request_id!r} starts at {view.start}"
            )
            requests.append(MlaRequestSlice(
                q_start=q_start,
                seq_len=view.to_compute,
                total_len=view.length,
                page_indices=torch.tensor(
                    view.page_idxs, dtype=torch.long, device=self._device,
                ),
            ))
            q_start += view.to_compute
        return MlaSdpaPlan(requests=tuple(requests))

    ### Submodule-level functionality

    @torch.compiler.disable
    def qo_indptr_buf(self, label: str = "main") -> torch.Tensor | None:
        state = self._current_plan_states.get(label)
        return None if state is None else state.qo_indptr

    def select_last_hidden(
        self, hidden: torch.Tensor, label: str = "main"
    ) -> torch.Tensor:
        """Last token of each request's hidden states, for sampling off prefill."""
        qo_indptr = self.qo_indptr_buf(label)
        return hidden.index_select(0, (qo_indptr[1:] - 1).long())

    @torch.compiler.disable
    def run_mla(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache_layer: torch.Tensor,
        label: str | None = None,
    ) -> torch.Tensor:
        """Compressed-latent MLA attention over one layer's latent pages. This
        step's latents must already be written through the KV resource.

        Args:
            q_nope: [T, H, L]      query, no-rope part (L = kv_lora_rank).
            q_pe:   [T, H, Drope]  query, rope part.
            latent_cache_layer: [pages, page_size, L + Drope] for this layer.
            label: plan label; defaults to the cursor the stack set.
        Returns:
            [T, H, L] attention output (the ``value`` = ``kv_c`` slice width).
        """
        if label is None:
            label = self._default_label
        state = self._current_plan_states[label]

        if state.wrapper is not None:
            ckv = q_nope.shape[-1]  # ckv width (post-w_kc absorption)
            return state.wrapper.run(
                q_nope, q_pe,
                latent_cache_layer[..., :ckv], latent_cache_layer[..., ckv:],
            ).to(q_nope.dtype)

        assert state.sdpa is not None
        return self._run_sdpa(state.sdpa, q_nope, q_pe, latent_cache_layer)

    def _run_sdpa(
        self, plan: MlaSdpaPlan,
        q_nope: torch.Tensor, q_pe: torch.Tensor,
        latent_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        """Per-request eager attention over the shared latent head."""
        num_tokens, num_heads, ckv = q_nope.shape
        query = torch.cat([q_nope, q_pe], dim=-1)
        out = torch.empty(
            num_tokens, num_heads, ckv,
            dtype=q_nope.dtype, device=q_nope.device,
        )
        for req in plan.requests:
            # gather this request's pages, then drop the last page's tail
            gathered = latent_cache_layer[req.page_indices].reshape(
                -1, latent_cache_layer.shape[-1]
            )[:req.total_len]
            rows = slice(req.q_start, req.q_start + req.seq_len)
            out[rows] = self._sdpa_mla(
                query[rows], key=gathered, value=gathered[:, :ckv],
                old_len=req.total_len - req.seq_len, scale=self._softmax_scale,
            )
        return out

    @staticmethod
    def _sdpa_mla(
        q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        old_len: int,
        scale: float,
    ) -> torch.Tensor:
        """Causal SDPA for one request over the shared latent head."""
        sl = q.shape[0]
        total = key.shape[0]
        qt = q.transpose(0, 1).float()
        scores = torch.einsum("hqd,kd->hqk", qt, key.float()) * scale
        q_pos = old_len + torch.arange(sl, device=q.device)
        k_pos = torch.arange(total, device=q.device)
        mask = torch.where(
            k_pos[None, :] <= q_pos[:, None],
            0.0,
            torch.tensor(float("-inf"), device=q.device),
        )
        scores = scores + mask
        attn = scores.softmax(-1)
        out = torch.einsum("hqk,kd->hqd", attn, value.float())
        return out.transpose(0, 1).to(q.dtype)
