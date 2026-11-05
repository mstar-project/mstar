"""Context parallelism: a paged KV pool's token positions placed over a group of ranks
that would otherwise hold the same KV heads.

Page ids stay logical and are the same on every rank, so allocation, prefix keys,
offload and admission need nothing new. Within a page, logical slot ``s`` belongs to
partner ``s % P`` and sits at its physical slot ``s // P``, so each rank stores
``page_size // P`` slots a page. ``PagedKVConfig.place`` resolves the placement.

Attention over such a pool (``ContextParallelWrapper``) runs on each rank over its own slots and
returns the partial with its log-sum-exp; the partners exchange partials and ``merge`` them.
"""

import functools
import logging
import math
from collections.abc import Iterable, Sequence
from typing import NamedTuple

import numpy as np
import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.kv.plan import SINK_PAGE, HostLens, PagedIndptrs, first_pages
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.platforms import current_platform
from mstar.utils.h2d import PinnedStager

logger = logging.getLogger(__name__)


class ContextParallel(NamedTuple):
    """This rank's place in its KV pool's placement.

    Under SP placement the partners are the whole SP group. Under TP placement they are a block of
    the TP group (the ranks sharing one KV head), and the exchange is an all-to-all over the whole
    TP group with nothing sent to the ranks outside the block, so no group of its own is needed."""

    # P: how many partners each page's slots stripe over
    size: int
    # r: this rank owns the logical slots s with s % P == r
    index: int
    # what the partners exchange over: the SP group in SP mode, the TP group in TP mode
    group: CommGroup
    over_sp: bool
    # the partners' ranks in ``group``, in partner-index order
    partners: tuple[int, ...]

    def slot_range(self, start: int, end: int) -> tuple[int, int]:
        """The physical slots of a page holding this rank's share of logical slots ``[start, end)``."""
        p, r = self.size, self.index
        return (start - r + p - 1) // p, (end - r + p - 1) // p

    def write_slots(self, pages, offsets):
        """Each token's write address on this rank, from its logical ``(page, offset)``: an owned
        token goes to ``(page, offset // P)``, the rest to ``(SINK_PAGE, 0)``. Device tensors stay
        on the device; anything else comes back as numpy arrays."""
        if isinstance(pages, torch.Tensor):
            where = torch.where
        else:
            where, pages, offsets = np.where, np.asarray(pages), np.asarray(offsets)
        owned = offsets % self.size == self.index
        return where(owned, pages, SINK_PAGE), where(owned, offsets // self.size, 0)


def _pack(out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """A partial's ``out`` ``[..., D]`` in its own dtype and its lse ``[...]`` in fp32 as one byte
    tensor ``[..., D * itemsize + 4]``, so one collective carries both at the out's width."""
    lse = lse.float().unsqueeze(-1).contiguous()
    return torch.cat([out.contiguous().view(torch.uint8), lse.view(torch.uint8)], dim=-1)


def _unpack(packed: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """``_pack``'s ``(out, lse)`` back, as views of ``packed``."""
    return packed[..., :-4].view(dtype), packed[..., -4:].view(torch.float32).squeeze(-1)


def _gather_partials(group: CommGroup, out: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Every rank of ``group``'s ``(out, lse)`` for the same queries, stacked on a new first dim."""
    return _unpack(group.all_gather(_pack(out, lse).unsqueeze(0), dim=0), out.dtype)


class ContextParallelWrapper:
    """A step's attention over a placed pool from a backend's two wrappers: each rank attends its
    own slots, and the partners exchange the partials and merge them. A one-token-a-row step
    attends the pages after its write (``new`` None); any other step attends its stored context
    (not causal) on ``context`` and its own tokens, from the keys the layer passes to ``run``, on
    ``new``. ``page_size`` is the logical one."""

    @staticmethod
    def local_geometry(placement: ContextParallel, page_size: int, num_qo_heads: int) -> dict[str, int]:
        """A paged attention wrapper's ``page_size`` and ``num_qo_heads`` over this rank's slots: the
        slots a page it stores, and under TP placement every partner's query heads, which it attends
        for them (``_exchange_queries``)."""
        return dict(
            page_size=page_size // placement.size,
            num_qo_heads=num_qo_heads * (1 if placement.over_sp else placement.size),
        )

    @staticmethod
    def local_indptrs(
        placement: ContextParallel, indptrs: PagedIndptrs, page_size: int,
    ) -> tuple[PagedIndptrs, np.ndarray]:
        """A plan's logical paged indptrs as this rank's (``page_size`` the logical one), and per row
        whether it holds any of this rank's slots.

        Every row keeps a length of at least 1 and a last page holding at least one slot, as on one
        rank: FlashInfer's split-KV planner counts a row's pages where its kernel counts its tokens,
        and a last page with none of this rank's slots puts the two out of step. Such a page is
        dropped from its row. A row with no slot here attends one slot of SINK_PAGE instead, and its
        partial is masked before the exchange (``_exchange_partials``)."""
        size, index = placement.size, placement.index
        kvp = indptrs.paged_kv_indptr.numpy().astype(np.int64)
        npages = np.diff(kvp)
        # this rank's slots in each row's last page, and its KV length
        slots = page_size // size
        last = np.clip((indptrs.paged_kv_last_page_len.numpy() - index + size - 1) // size, 0, slots)
        lens = np.where(npages > 0, (npages - 1) * slots + last, 0)
        valid = lens > 0
        drop = valid & (last == 0)
        kv_indptr, indices = indptrs.paged_kv_indptr, indptrs.paged_kv_indices
        if drop.any() or not valid.all():
            new_kvp, out = first_pages(kvp, indices.numpy(), np.where(valid, npages - drop, 0), fill=SINK_PAGE)
            last = np.where(valid, np.where(drop, page_size // size, last), 1)
            lens = np.where(valid, lens, 1)
            kv_indptr, indices = torch.from_numpy(new_kvp.astype(np.int32)), torch.from_numpy(out)
        return PagedIndptrs(
            qo_indptr=indptrs.qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=indices,
            paged_kv_last_page_len=torch.from_numpy(last.astype(np.int32)),
            kv_lens=torch.from_numpy(lens.astype(np.int32)).as_subclass(HostLens),
        ), valid

    def __init__(
        self,
        placement: ContextParallel,
        page_size: int,
        dtype: torch.dtype,
        context,
        new,
        device: torch.device,
        batch_size: int | None = None,
        max_total_tokens: int | None = None,
    ):
        self._placement, self._page_size, self._dtype = placement, page_size, dtype
        self.context = context
        # None for a one-token-a-row step
        self.new = new
        self._device = device
        self._qo_indptr_buf = None
        # the planned step's rows hold no stored context, so it attends its own tokens alone (`plan`)
        self._own_only = False
        # per query token, whether its row holds any of this rank's slots. A captured wrapper's
        # holds its bucket's every token (a row's at least, and the bucket's) and is never
        # reallocated, so a replay reads the address it was captured with
        captured_tokens = max(batch_size or 0, max_total_tokens or 0)
        self._captured = captured_tokens > 0
        self._valid = torch.zeros(captured_tokens, dtype=torch.bool, device=device) if self._captured else None
        self._stager = PinnedStager(torch.bool)
        # page-locked copy of the trimmed paged indices; one per wrapper, so per slot. The event
        # follows the last plan's copies out of it
        self._pinned_indices: torch.Tensor | None = None
        self._pinned_done: torch.Event | None = None

    @torch.compiler.disable
    def plan(self, kv_out, causal: bool = True, real_rows: frozenset | None = None):
        """``kv_out``: the step's logical ``KVPlanOutput`` for this label. The context attention reads
        its indptrs for a one-token-a-row step, its stored context alone otherwise.

        ``real_rows``: a replay's real requests when it has padding rows (``StepContext.real_rows``),
        which follow the real rows in each label. The own-token part then plans the real rows alone,
        so the padding's keys lie past its plan, where ``run`` zeroes them: the kernel reads keys
        past a row's end to its KV tile's, and a NaN the model's layers left in a padding row's
        keys would reach the real rows before it."""
        context = kv_out.cpu_indptrs
        if self.new is not None:
            context = context.context_only(self._page_size)
        # a step whose rows store no context (a first prefill) has no partial to exchange; from the
        # logical indptrs, so every rank alike. A captured step exchanges whatever it stores, as its
        # replays issue its capture's collectives
        self._own_only = (
            self.new is not None and not self._captured and int(context.paged_kv_indptr[-1]) == 0
        )
        local, valid_rows = self.local_indptrs(self._placement, context, self._page_size)
        if kv_out.cpu_indptrs.paged_kv_indices.is_pinned() and not local.paged_kv_indices.is_pinned():
            # page-locked plan indices (MSTAR_KV_PINNED_INDPTRS) stay page-locked once trimmed, so
            # FlashInfer copies them asynchronously on every rank alike, whichever rows a rank trims
            local = local._replace(paged_kv_indices=self._pin(local.paged_kv_indices))
        # not causal whatever the step: every query sees all of it (a one-token row's own key included)
        try:
            self.context.plan(causal=False, dtype=self._dtype, **local.to_kwargs_dict())
        finally:
            # also on a raise: copies already issued still read the pinned buffer
            if self._pinned_done is not None:
                self._pinned_done.record()
        if self.new is not None:
            qo = context.qo_indptr
            own = None if real_rows is None else _leading_rows(kv_out, real_rows)
            self.new.plan(qo if own is None else qo[: own + 1], causal=causal)
        self._qo_indptr_buf = getattr(self.context, "_qo_indptr_buf", None)
        valid = np.repeat(valid_rows, np.diff(context.qo_indptr.numpy()))
        n = len(valid)
        if self._captured:
            assert n <= self._valid.shape[0], f"{n} query tokens on a wrapper captured for {self._valid.shape[0]}"
        elif self._valid is None or self._valid.shape[0] < n:
            self._valid = torch.zeros(max(n, 256), dtype=torch.bool, device=self._device)
        # a captured bucket's whole buffer, so its padding tokens read False rather than a stale flag;
        # an eager step reads its own tokens' alone
        self._stager.copy_(self._valid, valid, pad_value=False if self._captured else None)

    def _pin(self, indices: torch.Tensor) -> torch.Tensor:
        """``indices`` in this wrapper's page-locked buffer, once the last plan's non-blocking copies
        out of it have landed: a slot's step has retired before its next plan, but a dropped pre-plan
        is replanned on the same slot at once."""
        if self._pinned_done is None:
            self._pinned_done = torch.Event(device=self._device)
        else:
            self._pinned_done.synchronize()
        n = len(indices)
        if self._pinned_indices is None or len(self._pinned_indices) < n:
            grown = 2 * (0 if self._pinned_indices is None else len(self._pinned_indices))
            self._pinned_indices = torch.empty(max(n, grown, 256), dtype=torch.int32, pin_memory=True)
        return self._pinned_indices[:n].copy_(indices)

    def attend(
        self, q: torch.Tensor, kv_cache_layer: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None,
    ) -> torch.Tensor:
        """Attention over the whole placed pool, in the query's dtype: ``q``
        ``[N, H, D]``, and the step's own keys as this backend's ``run`` takes them (read only on a
        step that isn't one token a row). Every rank runs the same collectives."""
        if self.new is not None and (k is None or v is None):
            raise ValueError(
                "attention over a context-parallel KV pool attends a step's own tokens from the k and v "
                "run is passed; this step has more than one token a row and got none"
            )
        new = None if self.new is None else self.new.run(q, k, v, return_lse=True)
        if self._own_only:
            # every partial would be masked (a first prefill): no exchange, no merge
            return new[0].to(q.dtype)
        queries = self._exchange_queries(q)
        # over this rank's slots, with the rows holding any of them
        out, lse = self.context.run(queries, kv_cache_layer, return_lse=True)
        outs, lses = self._exchange_partials(out, lse, self._valid[: queries.shape[0]])
        if new is not None:
            outs, lses = [*outs, new[0].to(outs.dtype)], torch.cat([lses, new[1].float()[None]])
        return merge(outs, lses).to(q.dtype)

    def _exchange_queries(self, q: torch.Tensor) -> torch.Tensor:
        """The queries this rank attends its slots for, from its own ``[N, H, D]``. SP partners hold
        the same heads, so they are its own; TP partners hold different query heads of one KV head,
        so each gets every partner's, ``[N, P * H, D]`` partner-major."""
        placement = self._placement
        if placement.over_sp:
            return q
        q = q.unsqueeze(0)
        if placement.size == placement.group.world_size:
            # every rank of the group is a partner: the same block goes to each
            q = placement.group.all_gather(q.contiguous(), dim=0)
        else:
            q = placement.group.to_partners(q.expand(placement.size, *q.shape[1:]), placement.partners)
        return q.transpose(0, 1).flatten(1, 2)

    def _exchange_partials(
        self, out: torch.Tensor, lse: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Every partner's partial for this rank's heads, ``[P, N, H, D]`` in ``out``'s dtype and
        ``[P, N, H]`` fp32, to ``merge``; from this rank's ``out`` ``[N, H', D]`` and natural-log
        ``lse`` ``[N, H']`` over its slots, for the heads ``_exchange_queries`` returned. A query row
        ``valid`` (``[N]``) marks as holding none of this rank's slots goes out as out 0, lse -inf,
        whatever the kernel left there (a select, so a NaN cannot survive)."""
        placement = self._placement
        out = torch.where(valid[:, None, None], out, 0)
        lse = torch.where(valid[:, None], lse.float(), -math.inf)
        if placement.over_sp:
            return _gather_partials(placement.group, out, lse)
        heads = out.shape[1] // placement.size
        blocks = _pack(out, lse).unflatten(1, (placement.size, heads)).transpose(0, 1)
        return _unpack(placement.group.to_partners(blocks, placement.partners), out.dtype)

    def forget_plan(self) -> None:
        """Replan the own-token wrapper on the next ``plan`` whatever its layout, for a dropped
        pre-plan: a ragged one skips a layout it already holds (`_RaggedPrefillBase.forget_plan`)."""
        if self.new is not None:
            self.new.forget_plan()

    @property
    def reads_step_kv(self) -> bool:
        """Whether the step attends the keys ``run`` is passed (``AttentionManager.reads_step_kv``):
        any step but one token a row."""
        return self.new is not None


def _leading_rows(kv_out, real_rows: frozenset) -> int | None:
    """How many of ``kv_out``'s rows are ``real_rows``, which lead it; None when all are, or none."""
    rids = [view.request_id for view in kv_out.views] if kv_out.rows is None else [row[0] for row in kv_out.rows]
    n = sum(rid in real_rows for rid in rids)
    if not all(rid in real_rows for rid in rids[:n]):
        raise ValueError(f"a padding row comes before a real one in a plan of rows {rids}")
    return n if 0 < n < len(rids) else None


def merge(outs: Sequence[torch.Tensor], lses: torch.Tensor) -> torch.Tensor:
    """Attention over the union of disjoint key sets from each set's partial: each out ``[..., H, D]``
    (a ``[K, ..., H, D]`` stack is a sequence too) and their natural-log lses stacked, ``[K, ..., H]``.
    One ``[..., H, D]`` in the outs' dtype, merged in fp32 but for ``_merge_states``'s one exception. A
    partial with lse -inf and a finite out adds nothing, and a query no partial has a key for (a
    captured bucket's padding) comes out 0. Runs in FlashInfer's kernel where it fits, and otherwise
    here, a part at a time."""
    first = outs[0]
    if first.is_cuda and _merge_states_fits(first, len(outs)) and (kernel := _merge_states_kernel()):
        return _merge_states(kernel, outs, lses)
    lses = lses.float()
    # finite on a row with no key, so its every weight is 0
    top = lses.amax(0).clamp_min(torch.finfo(torch.float32).min)
    weights = (lses - top).exp_()
    out = weights[0].unsqueeze(-1) * first
    for weight, part in zip(weights[1:], outs[1:], strict=True):
        out.addcmul_(weight.unsqueeze(-1), part)
    return out.div_(weights.sum(0).clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)).to(first.dtype)


def _merge_states_fits(first: torch.Tensor, k: int) -> bool:
    """Whether FlashInfer's ``merge_states`` takes ``k`` parts like ``first``: half precision at a head
    dim it builds, a query's heads in one thread block, and every offset within its 32-bit indexing,
    past which it silently reads another query's part. A prompt can get there: SP mode never chunks."""
    heads, head_dim = first.shape[-2:]
    threads_a_head = head_dim // max(8, head_dim // 32)
    return (
        first.dtype in (torch.float16, torch.bfloat16) and head_dim in (64, 128, 256, 512)
        and heads * threads_a_head <= 1024 and 0 < first.numel() * k < 2**32
    )


@functools.cache
def _merge_states_kernel():
    """FlashInfer's ``merge_states`` with its module built, which a first use compiles, or None where
    it can't run: off NVIDIA (ROCm's torch says ``cuda`` too), or a FlashInfer that doesn't import or
    build. A model can run without one, as LTX-2.5 does on SDPA."""
    if not current_platform.is_cuda():
        return None
    try:
        from flashinfer.cascade import get_cascade_module, merge_states

        get_cascade_module()
    except Exception as exc:  # noqa: BLE001
        logger.warning("attention partials merge in torch, FlashInfer's kernel being unavailable: %s", exc)
        return None
    return merge_states


def _merge_states(kernel, outs: Sequence[torch.Tensor], lses: torch.Tensor) -> torch.Tensor:
    """``merge`` in ``kernel``, which takes contiguous ``[N, K, H, D]`` parts and base-2 ``[N, K, H]``
    fp32 lses: one copy of each into that layout, then the kernel. It merges in fp32, except that for
    at most K queries with K over 16, 8, 4 and 4 at D 64, 128, 256 and 512, it rounds partial merges
    to the parts' dtype before the last one."""
    v = torch.stack(tuple(outs), dim=-3)
    s = lses.new_empty((*lses.shape[1:-1], lses.shape[0], lses.shape[-1]), dtype=torch.float32)
    torch.mul(lses.movedim(0, -2), 1 / math.log(2), out=s)
    out, _ = kernel(v.flatten(0, -4), s.flatten(0, -3))
    return out.view(outs[0].shape)


@torch.compiler.disable
def gather_merge(group: CommGroup, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """A query replicated on every rank of ``group`` attending keys sharded across them, from this
    rank's ``out`` ``[..., H, D]`` and natural-log ``lse`` ``[..., H]`` over its share: one gather
    of both, then ``merge``. Moves query-sized results rather than the keys."""
    if group.world_size == 1:
        return out
    return merge(*_gather_partials(group, out, lse))


def _position_pools(specs: Iterable[NodeResourceSpec], node: str) -> dict[str, bool]:
    """``node``'s resources that hold token positions, each to whether it places them
    (``NodeResourceSpec.places_positions``). Raises when one is placed and another isn't."""
    pools = {
        spec.resource_key: placed for spec in specs
        if node in spec.nodes and (placed := spec.places_positions) is not None
    }
    if any(pools.values()) and not all(pools.values()):
        raise ValueError(
            f"node {node!r} places positions over its ranks, so every pool of them it uses must: "
            f"set context_parallel on {sorted(key for key, placed in pools.items() if not placed)} too"
        )
    return pools


def positions_over_sp(specs: Iterable[NodeResourceSpec], node: str, sp_size: int, sp_enabled_nodes) -> bool:
    """Whether ``node``, on ``sp_size`` SP ranks, places its pools' positions over them, sharding
    heads over TP alone and running replicated across SP, so the model loads without its SP group;
    otherwise it runs the model's own sequence parallelism (it is in ``sp_enabled_nodes``). False on
    one SP rank. Raises when it does neither, or both, or places one pool and not another."""
    if sp_size == 1:
        return False
    pools = _position_pools(specs, node)
    placed = any(pools.values())
    if placed and node in sp_enabled_nodes:
        raise ValueError(
            f"node {node!r} runs its own sequence parallelism (sp_enabled_nodes), and a paged KV pool over "
            "it sets context_parallel, which would replace it: turn context_parallel off, or use sp_size 1 "
            "to place positions over TP ranks"
        )
    if not placed and node not in sp_enabled_nodes:
        raise ValueError(
            f"node {node!r} on {sp_size} SP ranks is not in sp_enabled_nodes {sorted(sp_enabled_nodes)}"
            + (": set context_parallel on its paged KV pools to place positions over the SP ranks instead"
               if pools else ", and holds no paged KV pool to place over them: put it in a node group of "
               "its own with sp_size 1")
        )
    return placed
