"""Paged attention through FlashAttention-3's kvcache kernel.

The point of this backend is what it does *not* do: FlashInfer splits
scheduling out into a `plan()` that runs on the host every step (measured at
about 2 ms per step on Whisper turbo, against roughly 5 ms of GPU work per
iteration — see `FlashInferManager`). FlashAttention partitions the work
inside the kernel instead, so the step's only host cost is filling in the
index tensors the kernel reads.

One kernel serves both regimes. `flash_attn_with_kvcache` takes a packed
varlen query (`cu_seqlens_q`) alongside the paged KV (`page_table`), so a
decode step is just the q_len=1 case of a prefill and there is no
prefill/decode wrapper split to pick between.

`num_splits` stays at 1 deliberately. FlashAttention's split-KV heuristic
reads the sequence lengths on the host to decide a partition, which is both a
sync and uncapturable; at one split the kernel schedules itself. Our shapes
have query rows times heads well above the SM count, so there is nothing for
split-KV to recover.
"""

import functools
import logging
from dataclasses import dataclass

import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.base import CGSlotKey
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import SINK_PAGE, KVPlanOutput, KVPlanOutputs
from mstar.engine.resources.step import StepContext

logger = logging.getLogger(__name__)

# imported here, not per `run`: the package may be absent, so keep the failure
# at build time where `AttentionManager.build` can name it
try:
    from flash_attn_interface import flash_attn_with_kvcache
except Exception as _fa3_exc:  # noqa: BLE001
    flash_attn_with_kvcache = None
    _FA3_IMPORT_ERROR: str | None = f"{type(_fa3_exc).__name__}: {_fa3_exc}"
else:
    _FA3_IMPORT_ERROR = None

# The head dims the `flash_attn_3` wheel in use is compiled for. A build
# narrowed with FLASH_ATTENTION_DISABLE_HDIM* raises from inside the kernel
# rather than returning wrong values, but refusing up front names the fix
# (rebuild with that flag cleared) instead of surfacing a kernel assert.
FA3_HEAD_DIMS = frozenset({64, 128})


@functools.cache
def _fa3_paged_unavailable_reason() -> str | None:
    """Why this backend cannot run, or None. The import is the only thing
    checkable without a device; a wheel built with
    FLASHATTENTION_DISABLE_PAGEDKV imports fine and raises "This flash
    attention build does not support paged KV" on the first call."""
    return _FA3_IMPORT_ERROR


def check_fa3_head_dim(head_dim: int, device: torch.device | None = None) -> None:
    if device is not None and device.type != "cuda":
        return
    if head_dim not in FA3_HEAD_DIMS:
        raise ValueError(
            f"FlashAttention-3 paged attention: head_dim {head_dim} is not one "
            f"of {sorted(FA3_HEAD_DIMS)}. The installed flash_attn_3 wheel was "
            "built for those only; rebuild it with the matching "
            "FLASH_ATTENTION_DISABLE_HDIM* flag cleared, or pad the model's "
            "q/k/v to a supported size and declare that as the KV cache's "
            "head_dim."
        )


@dataclass
class _IndexBuffers:
    """The three index tensors the kernel reads, one set per plan target.

    Under a captured graph these addresses are baked into the replay, so a
    plan writes values into them with `copy_` and never reallocates. The eager
    path uses the same objects for the same reason it is cheap to: one
    allocation per (label, slot) rather than per step.
    """

    page_table: torch.Tensor      # [rows, max_blocks] int32
    cache_seqlens: torch.Tensor   # [rows] int32
    cu_seqlens_q: torch.Tensor    # [rows + 1] int32

    @classmethod
    def alloc(
        cls, rows: int, max_blocks: int, device: torch.device,
    ) -> "_IndexBuffers":
        return cls(
            # SINK_PAGE, not zeros-as-garbage: a padded row has to name a real
            # page, and length 1 below keeps its one-token read in bounds
            page_table=torch.full(
                (rows, max_blocks), SINK_PAGE, dtype=torch.int32, device=device,
            ),
            cache_seqlens=torch.ones(rows, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.zeros(rows + 1, dtype=torch.int32, device=device),
        )


@dataclass(frozen=True)
class FlashAttnPlan:
    """One label's layout for this step.

    The tensors are SLICES of the buffers, cut to `rows` — the width the
    kernel will actually be handed. A captured step's `rows` is the bucket's,
    so the slice is the whole buffer and the replay sees the address it was
    captured with; an eager step's is the live batch's, which matters because
    the eager buffers are reused and may be wider than this step.

    `total_q` is the query row count those indices describe. It exists to be
    checked against `q` in `run`: a mismatch there is the kernel reading off
    the end of `q`, which surfaces as an async illegal access somewhere else
    entirely.
    """

    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    total_q: int
    max_q: int
    causal: bool


class FlashAttnManager(AttentionManager):
    """Paged KV attention on FlashAttention-3, with no per-step plan call."""

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: PagedKVConfig,
    ):
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        check_fa3_head_dim(kv_config.head_dim, device)

        # per-request page ceiling, not the pool size: a row of the page table
        # covers one sequence
        self._max_blocks = -(
            -kv_config.max_seq_len // kv_config.page_size
        )

        self._current_plans: dict[str, FlashAttnPlan] = {}
        # staged a step ahead on the plan thread, promoted by the next `plan`
        self._preplan_plans: dict[str, FlashAttnPlan] = {}
        self._preplanned = False
        # (bucket, slot, label) -> buffers, so a captured graph replays against
        # the addresses it was captured with
        self._cg_buffers: dict[CGSlotKey, _IndexBuffers] = {}
        # (label, slot) -> buffers for the uncaptured path
        self._eager_buffers: dict[tuple[str, int], _IndexBuffers] = {}

    def depends_on(self):
        return {self._kv_cache_name}

    @property
    def supports_preplan(self) -> bool:
        """Yes. There is no kernel-side schedule to hoist, but building the page
        table and the two indptrs is still host work, and the plan thread can do
        it for step N+1 while step N runs. A preplan leases a different cg slot,
        so it writes buffers the in-flight replay is not reading."""
        return True

    @property
    def force_double_buffer(self) -> bool:
        """Yes, and not only for the pre-plan path.

        The page table and the indptrs live in device buffers a captured replay
        reads, and `plan` writes them through a host->device copy. With one
        buffer set, `plan(N+1)` can land before step N's kernels have consumed
        step N's values and the replay attends with the wrong layout — the
        hazard `Resource.force_double_buffer` describes. Two consecutive steps
        race that way on the async worker whether or not anything pre-plans.
        """
        return True

    def plan(self, step: AttentionStep, ctx: StepContext):
        self.reset_default_cursors()
        lease = ctx.slot_lease
        assert not ctx.is_preplan or lease is not None, (
            "preplan requires a cuda graph step: the buffers are leased per "
            "slot, and an eager step shares its slot with the captured one"
        )
        assert not (self._preplanned and ctx.is_preplan), (
            "flash_attn preplan is already pending; clear_preplan before "
            "planning a different step ahead"
        )
        if self._preplanned:
            # staged a step early against this same step; nothing to do but
            # promote it to live
            self._current_plans = self._preplan_plans
            self._preplan_plans = {}
            self._preplanned = False
            return

        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, (
            f"FlashAttn attention expected plan result from {self._kv_cache_name}"
        )

        plans = {
            label: self._build_plan(label, kv_out, step.causal, ctx)
            for label, kv_out in plan_outputs.items()
        }
        if ctx.is_preplan:
            self._preplan_plans = plans
        else:
            self._current_plans = plans
        self._preplanned = ctx.is_preplan

    def _buffers_for(
        self, label: str, rows: int, ctx: StepContext,
    ) -> _IndexBuffers:
        """The buffer set this step writes into.

        Captured steps key on the lease so the addresses are stable per
        (bucket, slot); eager steps key on (label, slot) only, since nothing
        holds those addresses past the call.
        """
        lease = ctx.slot_lease
        if lease is not None:
            key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
            buffers = self._cg_buffers.get(key)
            if buffers is None:
                # the bucket's row count, not the live batch's: a replay always
                # runs the full padded width
                buffers = self._cg_buffers[key] = _IndexBuffers.alloc(
                    rows=lease.bucket.bs,
                    max_blocks=self._max_blocks,
                    device=self._device,
                )
            return buffers

        key = (label, ctx.slot)
        buffers = self._eager_buffers.get(key)
        if buffers is None or buffers.cache_seqlens.shape[0] < rows:
            buffers = self._eager_buffers[key] = _IndexBuffers.alloc(
                rows=rows, max_blocks=self._max_blocks, device=self._device,
            )
        return buffers

    def _build_plan(
        self, label: str, kv_out: KVPlanOutput, causal: bool, ctx: StepContext,
    ) -> FlashAttnPlan:
        views = kv_out.views
        buffers = self._buffers_for(label, len(views), ctx)
        # The width the kernel gets. A captured replay always runs the bucket's
        # full width, so the padded tail is part of the step; an eager step runs
        # exactly its own rows, and padding out to the (reused, possibly wider)
        # buffer would describe query rows that `q` does not have.
        lease = ctx.slot_lease
        rows = lease.bucket.bs if lease is not None else len(views)
        if len(views) > rows:
            raise RuntimeError(
                f"flash_attn plan for {label!r} has {len(views)} rows but its "
                f"bucket holds {rows}"
            )
        if rows > buffers.cache_seqlens.shape[0]:
            raise RuntimeError(
                f"flash_attn plan for {label!r} needs {rows} rows but its "
                f"buffers hold {buffers.cache_seqlens.shape[0]}"
            )

        # Built as host lists and copied in one shot per tensor: a per-row
        # `copy_` would be one H2D launch each, which is the cost this backend
        # exists to avoid.
        page_rows: list[list[int]] = []
        kv_lens: list[int] = []
        cu_q: list[int] = [0]
        max_q = 0
        for view in views:
            pages = list(view.page_idxs)
            # Truncating would leave `cache_seqlens` describing tokens whose
            # pages are not in the table — silently wrong attention, or an
            # out-of-bounds row read. Name it instead.
            if len(pages) > self._max_blocks:
                raise RuntimeError(
                    f"flash_attn plan for {label!r}: request "
                    f"{view.request_id!r} names {len(pages)} pages but a page "
                    f"table row holds {self._max_blocks} "
                    f"(max_seq_len {self._kv_config.max_seq_len} / page_size "
                    f"{self._kv_config.page_size})"
                )
            page_rows.append(
                pages + [SINK_PAGE] * (self._max_blocks - len(pages))
            )
            kv_lens.append(view.length)
            cu_q.append(cu_q[-1] + view.to_compute)
            max_q = max(max_q, view.to_compute)

        # The padded tail of a captured batch: a well-formed one-token row
        # whose output the step discards, matching the cross-attention plan's
        # treatment of a row with no context.
        for _ in range(rows - len(views)):
            page_rows.append([SINK_PAGE] * self._max_blocks)
            kv_lens.append(1)
            cu_q.append(cu_q[-1] + 1)

        page_table = buffers.page_table[:rows]
        cache_seqlens = buffers.cache_seqlens[:rows]
        cu_seqlens_q = buffers.cu_seqlens_q[:rows + 1]
        # Blocking, deliberately: the sources are temporaries, and a genuinely
        # async copy could outlive them. Pageable host memory gets nothing from
        # `non_blocking` anyway — buying that back means staging through a
        # pinned buffer, which is the block-table builder's job to own.
        page_table.copy_(torch.tensor(page_rows, dtype=torch.int32))
        cache_seqlens.copy_(torch.tensor(kv_lens, dtype=torch.int32))
        cu_seqlens_q.copy_(torch.tensor(cu_q, dtype=torch.int32))
        return FlashAttnPlan(
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            total_q=cu_q[-1],
            max_q=max(max_q, 1),
            causal=causal,
        )

    def clear_preplan(self):
        self._preplanned = False
        self._preplan_plans = {}

    ### Submodule-level functionality
    @torch.compiler.disable
    def qo_indptr_buf(self, label: str = "main") -> torch.Tensor | None:
        plan = self._current_plans.get(label)
        return None if plan is None else plan.cu_seqlens_q

    def select_last_hidden(
        self, hidden: torch.Tensor, label: str = "main",
    ) -> torch.Tensor:
        """The last hidden state per request, for sampling off a prefill."""
        qo_indptr_buf = self.qo_indptr_buf(label)
        last_token_indices = (qo_indptr_buf[1:] - 1).long()
        return hidden.index_select(0, last_token_indices)

    def run(
        self,
        q: torch.Tensor,
        label: str | None = None,
        kv_cache_layer: tuple[torch.Tensor, torch.Tensor] | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        # `k`/`v`/`layer_idx` belong to the dense backend; here the layer has
        # already written its K/V through the KV resource. Accepted and ignored
        # so one layer body serves either backend — see `requires_kv_write`.
        del k, v, layer_idx
        if label is None:
            label = self._default_label
        return self._attend(q, label, kv_cache_layer)

    @torch.compiler.disable
    def _attend(
        self,
        q: torch.Tensor,
        label: str,
        kv_cache_layer: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        plan = self._current_plans[label]
        k_cache, v_cache = kv_cache_layer
        # Shape check, not a sync: the kernel indexes `q` through
        # `cu_seqlens_q`, so a disagreement is an out-of-bounds read that
        # reports as an illegal access from whatever runs next.
        if q.shape[0] != plan.total_q:
            raise RuntimeError(
                f"flash_attn {label!r}: q has {q.shape[0]} rows but the plan "
                f"describes {plan.total_q}"
            )
        out = flash_attn_with_kvcache(
            q.to(self._dtype),
            k_cache,
            v_cache,
            cache_seqlens=plan.cache_seqlens,
            page_table=plan.page_table,
            cu_seqlens_q=plan.cu_seqlens_q,
            max_seqlen_q=plan.max_q,
            causal=plan.causal,
            num_splits=1,  # see the module docstring
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.to(q.dtype)
