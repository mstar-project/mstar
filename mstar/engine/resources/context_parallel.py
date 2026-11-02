"""Context parallelism: a paged KV pool's token positions placed over a group of ranks
that would otherwise hold the same KV heads.

Page ids stay logical and are the same on every rank, so allocation, prefix keys,
offload and admission need nothing new. Within a page, logical slot ``s`` belongs to
partner ``s % P`` and sits at its physical slot ``s // P``, so each rank stores
``page_size // P`` slots a page. ``PagedKVConfig.place`` resolves the placement.

Attention over such a pool runs on each rank over its own slots and returns the partial with
its log-sum-exp; the partners exchange partials and ``merge`` them, exactly.
"""

import math
from collections.abc import Callable, Iterable
from typing import NamedTuple

import numpy as np
import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.kv.plan import SINK_PAGE, HostLens, PagedIndptrs, first_pages
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.utils.h2d import PinnedStager


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

    def local_geometry(self, page_size: int, num_qo_heads: int) -> dict[str, int]:
        """A paged attention wrapper's ``page_size`` and ``num_qo_heads`` over this rank's slots: the
        slots a page it stores, and under TP placement every partner's query heads, which it attends
        for them (``exchange_queries``)."""
        return dict(page_size=page_size // self.size, num_qo_heads=num_qo_heads * (1 if self.over_sp else self.size))

    def local_indptrs(self, indptrs: PagedIndptrs, page_size: int) -> tuple[PagedIndptrs, np.ndarray]:
        """A plan's logical paged indptrs as this rank's (``page_size`` the logical one), and per row
        whether it holds any of this rank's slots.

        Every row keeps a length of at least 1 and a last page holding at least one slot, as on one
        rank: FlashInfer's split-KV planner counts a row's pages where its kernel counts its tokens,
        and a last page with none of this rank's slots puts the two out of step. Such a page is
        dropped from its row. A row with no slot here attends one slot of SINK_PAGE instead, and its
        partial is masked before the exchange (``exchange_partials``)."""
        kvp = indptrs.paged_kv_indptr.numpy().astype(np.int64)
        npages = np.diff(kvp)
        # this rank's slots in each row's last page, and its KV length
        slots = page_size // self.size
        last = np.clip((indptrs.paged_kv_last_page_len.numpy() - self.index + self.size - 1) // self.size, 0, slots)
        lens = np.where(npages > 0, (npages - 1) * slots + last, 0)
        valid = lens > 0
        drop = valid & (last == 0)
        kv_indptr, indices = indptrs.paged_kv_indptr, indptrs.paged_kv_indices
        if drop.any() or not valid.all():
            new_kvp, out = first_pages(kvp, indices.numpy(), np.where(valid, npages - drop, 0), fill=SINK_PAGE)
            last = np.where(valid, np.where(drop, page_size // self.size, last), 1)
            lens = np.where(valid, lens, 1)
            kv_indptr, indices = torch.from_numpy(new_kvp.astype(np.int32)), torch.from_numpy(out)
        return PagedIndptrs(
            qo_indptr=indptrs.qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=indices,
            paged_kv_last_page_len=torch.from_numpy(last.astype(np.int32)),
            kv_lens=torch.from_numpy(lens.astype(np.int32)).as_subclass(HostLens),
        ), valid

    def attend(
        self,
        queries: tuple[torch.Tensor, ...],
        partial: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        new: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Attention over the whole pool for this rank's query parts (each ``[N, H, *]``), fp32.
        ``partial`` gives ``(out, lse, valid)`` over this rank's slots; ``new`` is the step's own
        tokens' ``(out, lse)``. Every rank runs the same collectives."""
        outs, lses = self.exchange_partials(*partial(*self.exchange_queries(queries)))
        if new is not None:
            outs = torch.cat([outs, new[0].to(outs.dtype).unsqueeze(0)])
            lses = torch.cat([lses, new[1].float().unsqueeze(0)])
        return merge(outs, lses)

    def exchange_queries(self, queries: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """The query parts this rank attends its slots for, from its own (each ``[N, H, *]``). SP
        partners hold the same heads, so they are its own; TP partners hold different query heads
        of one KV head, so each gets every partner's, ``[N, P * H, *]`` partner-major, the parts
        sent as one tensor."""
        if self.over_sp:
            return queries
        q = (queries[0] if len(queries) == 1 else torch.cat(queries, dim=-1)).unsqueeze(0)
        if self.size == self.group.world_size:
            # every rank of the group is a partner (MLA's always are): the same block goes to each
            q = self.group.all_gather(q.contiguous(), dim=0)
        else:
            q = self.group.to_partners(q.expand(self.size, *q.shape[1:]), self.partners)
        return q.transpose(0, 1).flatten(1, 2).split([part.shape[-1] for part in queries], dim=-1)

    def exchange_partials(
        self, out: torch.Tensor, lse: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Every partner's partial for this rank's heads, ``[P, N, H, D]`` in ``out``'s dtype and
        ``[P, N, H]`` fp32, to ``merge``; from this rank's ``out`` ``[N, H', D]`` and natural-log
        ``lse`` ``[N, H']`` over its slots, for the heads ``exchange_queries`` returned. A query row
        ``valid`` (``[N]``) marks as holding none of this rank's slots goes out as out 0, lse -inf,
        whatever the kernel left there (a select, so a NaN cannot survive)."""
        out = torch.where(valid[:, None, None], out, 0)
        lse = torch.where(valid[:, None], lse.float(), -math.inf)
        if self.over_sp:
            return _gather_partials(self.group, out, lse)
        heads = out.shape[1] // self.size
        blocks = _pack(out, lse).unflatten(1, (self.size, heads)).transpose(0, 1)
        return _unpack(self.group.to_partners(blocks, self.partners), out.dtype)


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
    """A step's attention over a placed pool from a backend's two wrappers, exchanged and merged
    by ``ContextParallel.attend``. A one-token-a-row step attends the pages after its write
    (``new`` None); any other step attends its stored context (not causal) on ``context`` and its
    own tokens, from the keys the layer passes to ``run``, on ``new``. ``page_size`` is the
    logical one."""

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
    def plan(self, kv_out, causal: bool = True):
        """``kv_out``: the step's logical ``KVPlanOutput`` for this label. The context attention reads
        its indptrs for a one-token-a-row step, its stored context alone otherwise."""
        context = kv_out.cpu_indptrs
        if self.new is not None:
            context = context.context_only(self._page_size)
        local, valid_rows = self._placement.local_indptrs(context, self._page_size)
        if kv_out.cpu_indptrs.paged_kv_indices.is_pinned() and not local.paged_kv_indices.is_pinned():
            # page-locked plan indices (MSTAR_KV_PINNED_INDPTRS) stay page-locked once trimmed, so
            # FlashInfer copies them asynchronously on every rank alike, whichever rows a rank trims
            local = local._replace(paged_kv_indices=self._pin(local.paged_kv_indices))
        # not causal whatever the step: every query sees all of it (a one-token row's own key
        # included), and FA MLA's causal tile check reads each rank's KV lengths
        try:
            self.context.plan(causal=False, dtype=self._dtype, **local.to_kwargs_dict())
        finally:
            # also on a raise: copies already issued still read the pinned buffer
            if self._pinned_done is not None:
                self._pinned_done.record()
        if self.new is not None:
            self.new.plan(context.qo_indptr, causal=causal)
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
        self, queries: tuple[torch.Tensor, ...], kv_cache_layer: torch.Tensor,
        new_keys: tuple[torch.Tensor | None, ...],
    ) -> torch.Tensor:
        """Attention over the whole placed pool (``ContextParallel.attend``), in the query's dtype:
        ``queries`` and ``new_keys`` are the query parts and the step's own keys as this backend's
        ``run`` takes them (``new_keys`` read only on a step that isn't one token a row)."""
        if self.new is not None and any(key is None for key in new_keys):
            raise ValueError(
                "attention over a context-parallel KV pool attends a step's own tokens from the keys "
                "run is passed (k and v, or MLA's latents as k); this step has more than one token "
                "a row and got none"
            )

        def context(*parts: torch.Tensor):
            # over this rank's slots, with the rows holding any of them
            out, lse = self.context.run(*parts, kv_cache_layer, return_lse=True)
            return out, lse, self._valid[: parts[0].shape[0]]

        new = None if self.new is None else self.new.run(*queries, *new_keys, return_lse=True)
        return self._placement.attend(queries, context, new).to(queries[0].dtype)

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


def merge(outs: torch.Tensor, lses: torch.Tensor) -> torch.Tensor:
    """Attention over the union of disjoint key sets from each set's partial: ``outs`` ``[K, ..., D]``
    and natural-log ``lses`` ``[K, ...]``; fp32, one ``[..., D]`` accumulated a part at a time. A
    partial with lse -inf and a finite out adds nothing, and a query no partial has a key for (a
    captured bucket's padding) comes out 0."""
    lses = lses.float()
    # finite on a row with no key, so its every weight is 0
    top = lses.amax(0).clamp_min(torch.finfo(torch.float32).min)
    weights = (lses - top).exp_()
    out = weights[0].unsqueeze(-1) * outs[0]
    for weight, part in zip(weights[1:], outs[1:], strict=True):
        out.addcmul_(weight.unsqueeze(-1), part)
    return out.div_(weights.sum(0).clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1))


@torch.compiler.disable
def gather_merge(group: CommGroup, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """A query replicated on every rank of ``group`` attending keys sharded across them, from this
    rank's ``out`` ``[..., H, D]`` and natural-log ``lse`` ``[..., H]`` over its share: one gather
    of both, then ``merge``. Moves query-sized results rather than the keys."""
    if group.world_size == 1:
        return out
    return merge(*_gather_partials(group, out, lse)).to(out.dtype)


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
