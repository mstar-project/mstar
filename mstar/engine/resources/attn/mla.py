"""Absorbed multi-head latent attention (DeepSeek/Kimi/GLM MLA) over a
``KVLayout.MLA`` cache.
"""

import logging

import torch

from mstar.engine.resources.attn.base import AttentionManager, WorkspacePool
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.mla_wrapper import (
    FlashInferMLAWrapper,
    mla_kernel_available,
)
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.kv.config import KVConfig, KVLayout
from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs, SequenceView
from mstar.engine.resources.step import SlotLease, StepContext

logger = logging.getLogger(__name__)


class _EagerMlaPlan:
    """What the SDPA fallback needs for one label's step: the packed views
    (page tables + lengths) and the qo layout."""

    def __init__(self, views: list[SequenceView], page_size: int):
        self.views = views
        self.page_size = page_size


class MLAAttentionManager(AttentionManager):
    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: KVConfig,
        ckv_dim: int | None,
        softmax_scale: float | None,
        backend: str = "auto",
    ):
        if kv_config.layout != KVLayout.MLA:
            raise ValueError(
                f"the MLA attention backend needs a KVLayout.MLA cache; "
                f"{kv_cache!r} is {kv_config.layout}"
            )
        if ckv_dim is None or softmax_scale is None:
            raise ValueError(
                "AttentionConfig(backend=MLA) needs mla_ckv_dim and softmax_scale"
            )
        if not 0 < ckv_dim <= kv_config.head_dim:
            raise ValueError(
                f"mla_ckv_dim={ckv_dim} must lie in (0, head_dim={kv_config.head_dim}]"
            )
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        self.ckv_dim = ckv_dim
        self.kpe_dim = kv_config.head_dim - ckv_dim
        self.num_heads = kv_config.num_qo_heads  # already sharded per rank
        self.page_size = kv_config.page_size
        self.softmax_scale = float(softmax_scale)
        self._backend = backend

        self.use_kernel = mla_kernel_available(self.ckv_dim, self.kpe_dim, device)
        if not self.use_kernel:
            logger.info(
                "MLA attention %r: FlashInfer MLA kernel unavailable for "
                "ckv=%d kpe=%d on %s; using the fp32 SDPA fallback",
                kv_cache, self.ckv_dim, self.kpe_dim, device,
            )

        # label -> planned wrapper (kernel) or _EagerMlaPlan (fallback)
        self._current_plan_states: dict[str, object] = {}
        # label -> device last-token indices for select_last_hidden
        self._current_last_token: dict[str, torch.Tensor] = {}
        self._eager_wrappers: dict[str, FlashInferMLAWrapper] = {}
        self._cg_wrappers: dict[CGSlotKey, FlashInferMLAWrapper] = {}
        self._preplan_states: dict[str, object] = {}
        self._preplan_last_token: dict[str, torch.Tensor] = {}
        self._preplanned = False
        self._workspaces = WorkspacePool(device)

    def depends_on(self):
        return {self._kv_cache_name}

    # -- wrappers ---------------------------------------------------------

    def _wrapper_kwargs(self):
        return dict(
            num_heads=self.num_heads,
            head_dim_ckv=self.ckv_dim,
            head_dim_kpe=self.kpe_dim,
            page_size=self.page_size,
            sm_scale=self.softmax_scale,
            device=self._device,
            backend=self._backend,
        )

    def _cg_wrapper(self, lease: SlotLease, label: str, num_rows: int) -> FlashInferMLAWrapper:
        """The captured-graph wrapper for one (bucket, slot, label), built on
        the first plan for that key (during capture, so the graph records its
        static buffers). ``num_rows`` is the planned qo row count."""
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        wrapper = self._cg_wrappers.get(key)
        if wrapper is None:
            wrapper = self._cg_wrappers[key] = FlashInferMLAWrapper(
                workspace_buffer=self._workspaces.get(label, lease.slot),
                batch_size=num_rows,
                max_num_pages=self._kv_config.max_num_pages,
                use_cuda_graph=True,
                **self._wrapper_kwargs(),
            )
        return wrapper

    def _eager_wrapper(self, label: str) -> FlashInferMLAWrapper:
        wrapper = self._eager_wrappers.get(label)
        if wrapper is None:
            wrapper = self._eager_wrappers[label] = FlashInferMLAWrapper(
                workspace_buffer=self._workspaces.get(label),
                **self._wrapper_kwargs(),
            )
        return wrapper

    # -- plan / preplan ---------------------------------------------------

    @property
    def supports_preplan(self):
        return True

    def clear_preplan(self):
        self._preplanned = False
        self._preplan_states = {}
        self._preplan_last_token = {}

    @staticmethod
    def kv_len_arr(kv_out: KVPlanOutput) -> torch.Tensor:
        """Total resident length per request (this step's tokens included),
        the ``kv_len_arr`` the MLA kernel plans against."""
        return torch.tensor(
            [view.start + view.length for view in kv_out.views], dtype=torch.int32
        )

    def plan(self, step: AttentionStep, ctx: StepContext):
        self.reset_default_cursors()
        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a cuda graph step: eager wrappers share one "
            "workspace per label with the forward still in flight"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "attention preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        if self._preplanned:
            self._current_plan_states = self._preplan_states
            self._current_last_token = self._preplan_last_token
            self._preplan_states = {}
            self._preplan_last_token = {}
            self._preplanned = False
            return

        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, (
            f"MLA attention expected a plan result from {self._kv_cache_name}"
        )
        plan_states = self._preplan_states if ctx.is_preplan else self._current_plan_states
        last_token = self._preplan_last_token if ctx.is_preplan else self._current_last_token
        plan_states.clear()
        last_token.clear()

        for label, kv_out in plan_outputs.items():
            indptrs = kv_out.cpu_indptrs
            qo_indptr = indptrs.qo_indptr
            # eager prefill samples the last token of every request; stage the
            # gather index once here (H2D off the critical path), not per call
            last_token[label] = (qo_indptr[1:] - 1).to(
                torch.long).to(self._device, non_blocking=True)
            if not self.use_kernel:
                plan_states[label] = _EagerMlaPlan(kv_out.views, self.page_size)
                continue
            if lease is not None:
                wrapper = self._cg_wrapper(lease, label, qo_indptr.shape[0] - 1)
            else:
                wrapper = self._eager_wrapper(label)
            wrapper.plan(
                qo_indptr,
                indptrs.paged_kv_indptr,
                indptrs.paged_kv_indices,
                self.kv_len_arr(kv_out),
                causal=step.causal,
                dtype=self._dtype,
            )
            plan_states[label] = wrapper

        self._preplanned = ctx.is_preplan

    # -- submodule-facing -------------------------------------------------

    @torch.compiler.disable
    def select_last_hidden(self, hidden: torch.Tensor, label: str = "main") -> torch.Tensor:
        """The last token's row per request, in plan order — what prefill
        samples from."""
        return hidden.index_select(0, self._current_last_token[label])

    def run(
        self, q: torch.Tensor, label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """``q`` is the concatenated [T, H, ckv + kpe] query; the latent rows
        were written through the KV resource before this call (`k`/`v` are
        accepted for the shared layer body and ignored)."""
        del k, v, layer_idx
        return self.run_mla(
            q[..., : self.ckv_dim], q[..., self.ckv_dim:],
            label=label, kv_cache_layer=kv_cache_layer,
        )

    @torch.compiler.disable
    def run_mla(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend ``q_nope`` [T, H, ckv] + ``q_pe`` [T, H, kpe] over one
        layer's latent plane ([pages, page_size, ckv + kpe], from
        ``KVManager.layer_view``). Returns [T, H, ckv] in ``q_nope``'s dtype."""
        if label is None:
            label = self._default_label
        if kv_cache_layer is None:
            raise ValueError("run_mla needs the layer's latent plane (kv.layer_view)")
        state = self._current_plan_states[label]
        if isinstance(state, FlashInferMLAWrapper):
            out = state.run(
                q_nope, q_pe,
                kv_cache_layer[..., : self.ckv_dim],
                kv_cache_layer[..., self.ckv_dim:],
            )
            return out if out.dtype == q_nope.dtype else out.to(q_nope.dtype)
        return self._sdpa_fallback(state, q_nope, q_pe, kv_cache_layer)

    def _sdpa_fallback(
        self, plan: _EagerMlaPlan,
        q_nope: torch.Tensor, q_pe: torch.Tensor,
        latent_layer: torch.Tensor,
    ) -> torch.Tensor:
        """Causal fp32 attention per request over its gathered pages —
        the reference the kernel path is held to."""
        T, H, L = q_nope.shape
        query = torch.cat([q_nope, q_pe], dim=-1)
        out = torch.empty(T, H, L, dtype=q_nope.dtype, device=q_nope.device)
        q_start = 0
        for view in plan.views:
            sl = view.to_compute
            total = view.start + view.length
            if sl == 0:
                continue
            pages = torch.as_tensor(view.page_idxs, dtype=torch.long, device=latent_layer.device)
            gathered = latent_layer[pages].reshape(-1, latent_layer.shape[-1])[:total]
            out[q_start:q_start + sl] = _sdpa_mla(
                query[q_start:q_start + sl], gathered, gathered[:, :L],
                old_len=total - sl, scale=self.softmax_scale,
            )
            q_start += sl
        return out


def _sdpa_mla(
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    old_len: int,
    scale: float,
) -> torch.Tensor:
    """Causal SDPA for one request over the shared latent head.
    ``q`` [sl, H, ckv+kpe], ``key`` [total, ckv+kpe], ``value`` [total, ckv]."""
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
