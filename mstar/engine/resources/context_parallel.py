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
from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

import numpy as np
import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.kv.plan import SINK_PAGE, HostLens, PagedIndptrs, first_pages, token_runs
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.utils.h2d import PinnedStager, to_device_nonblocking


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
        partial is masked before the merge (``_masked``)."""
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
        """Attention over the whole pool for this rank's query parts (each ``[N, H, *]``): fp32
        under TP placement, the partial's dtype under SP placement. ``partial`` gives ``(out, lse,
        valid)`` over this rank's slots; ``new`` is the step's own tokens' ``(out, lse)`` this rank
        computes (under SP placement its zigzag chunks, lse -inf on the rest). Every rank runs the
        same collectives: under SP placement one all-gather, under TP placement an exchange of queries
        and one of partials."""
        if self.over_sp:
            return gather_merge(self.group, *self.merged_share(queries, partial, new))
        outs, lses = self.exchange_partials(*partial(*self.exchange_queries(queries)))
        if new is None:
            return merge(outs, lses)
        return merge([*outs, new[0]], torch.cat([lses, new[1][None]]))

    def merged_share(
        self,
        queries: tuple[torch.Tensor, ...],
        partial: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        new: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Under SP placement, this rank's ``(out, lse)`` as ``attend`` gathers it: its slots' partial
        merged with its share of the step's own tokens, rounded back to the partial's dtype for the
        gather. Each own token's keys are attended by one rank, so merging every rank's share is the
        whole attention. The rounding is one more than TP placement's, at the floor a bf16 partial
        carries already; gathering the fp32 merge instead would double the gather's bytes."""
        parts = [_masked(*partial(*queries))]
        if new is not None:
            parts.append(new)
        if len(parts) == 1:
            return parts[0][0], parts[0][1].float()
        (a, lse_a), (b, lse_b) = parts
        out, lse = _merged((a, b), torch.stack([lse_a, lse_b]), with_lse=True)
        return out.to(a.dtype), lse

    def exchange_queries(self, queries: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Under TP placement, the query parts this rank attends its slots for, from its own (each
        ``[N, H, *]``): its partners hold different query heads of one KV head, so it gets every
        partner's, ``[N, P * H, *]`` partner-major, the parts sent as one tensor."""
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
        """Under TP placement, every partner's partial for this rank's heads, ``[P, N, H, D]`` in
        ``out``'s dtype and ``[P, N, H]`` fp32, to ``merge``; from this rank's ``out`` ``[N, H', D]``
        and natural-log ``lse`` ``[N, H']`` over its slots, for the heads ``exchange_queries``
        returned, masked by ``valid`` (``_masked``)."""
        out, lse = _masked(out, lse, valid)
        heads = out.shape[1] // self.size
        blocks = _pack(out, lse).unflatten(1, (self.size, heads)).transpose(0, 1)
        return _unpack(self.group.to_partners(blocks, self.partners), out.dtype)


def _masked(out: torch.Tensor, lse: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A partial over this rank's slots with each query row ``valid`` (``[N]``) marks as holding none
    of them set to out 0, lse -inf, whatever the kernel left there (a select, so a NaN cannot
    survive); lse fp32."""
    return torch.where(valid[:, None, None], out, 0), torch.where(valid[:, None], lse.float(), -math.inf)


def _pack(out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """A partial's ``out`` ``[..., D]`` in its own dtype and its lse ``[...]`` in fp32 as one byte
    tensor ``[..., D * itemsize + 4]``, so one collective carries both at the out's width."""
    lse = lse.float().unsqueeze(-1).contiguous()
    return torch.cat([out.contiguous().view(torch.uint8), lse.view(torch.uint8)], dim=-1)


def _unpack(packed: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """``_pack``'s ``(out, lse)`` back, as views of ``packed``."""
    return packed[..., :-4].view(dtype), packed[..., -4:].view(torch.float32).squeeze(-1)



class ContextParallelWrapper:
    """A step's attention over a placed pool from a backend's two wrappers, exchanged and merged
    by ``ContextParallel.attend``. A one-token-a-row step attends the pages after its write
    (``new`` None); any other step attends its stored context (not causal) on ``context`` and its
    own tokens, from the keys the layer passes to ``run``, on ``new``: under TP placement all of
    them for this rank's heads, under SP placement this rank's zigzag chunks (``ZigzagWrapper``).
    ``page_size`` is the logical one.

    ``check(what, qo_indptr)``, before anything is planned, is the backend's refusal of a plan its
    kernel's planner may refuse on one rank of the lockstep instance alone. It sees every plan whose
    lengths differ by rank: the context's here, and under SP placement every partner's zigzag share
    of the own tokens (``ZigzagWrapper``). TP mode's own-token plan has the same lengths on every
    rank, so a refusal of it is symmetric already."""

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
        check: Callable[[str, np.ndarray], None] | None = None,
    ):
        self._placement, self._page_size, self._dtype = placement, page_size, dtype
        self._check = check
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
        if self._check is not None:
            self._check("the context", context.qo_indptr.numpy())
        if self.new is not None:
            # before the context, so a share's refusal (``ZigzagWrapper``) comes before anything is planned
            self.new.plan(context.qo_indptr, causal=causal)
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


class ZigzagRows(NamedTuple):
    """This rank's zigzag chunks of a step's own tokens under SP placement (chunks ``r`` and
    ``2P - 1 - r`` of each row's ``2P``; see the context-parallel docs). Arrays run over the
    non-empty chunks, row by row; token indices are the step's."""

    # [M] the queries this rank attends, chunk by chunk (``ZigzagWrapper.share`` counts them)
    tokens: np.ndarray
    # [S + 1] each chunk's keys in ``kv_tokens``
    kv_indptr: np.ndarray
    # [K] each chunk's keys: its row's first token to the chunk's last (or the row's)
    kv_tokens: np.ndarray


def zigzag_rows(qo_indptr: np.ndarray, size: int, index: int, causal: bool = True) -> ZigzagRows:
    """Rank ``index`` of ``size``'s zigzag chunks of the rows ``qo_indptr`` lays out."""
    row, first, end, kv_end = _zigzag_chunks(qo_indptr, size, index, causal)
    _, tokens = token_runs(row + first, end - first)
    kv, kv_tokens = token_runs(row, kv_end)
    return ZigzagRows(tokens, kv, kv_tokens)


def zigzag_share(qo_indptr: np.ndarray, size: int, index: int) -> np.ndarray:
    """The ``qo_indptr`` of rank ``index`` of ``size``'s zigzag chunks of the rows ``qo_indptr`` lays
    out, as its own-token plan has them before a capture's padding."""
    _, first, end, _ = _zigzag_chunks(qo_indptr, size, index, causal=True)
    return np.concatenate([[0], np.cumsum(end - first)])


def zigzag_capture(size: int, rows: int, tokens: int) -> tuple[int, int]:
    """A ``ZigzagWrapper`` captured for a bucket of ``rows`` rows and ``tokens`` tokens: its kernel's
    fixed rows (at most 2 chunks a row, and one more for the padding queries) and queries (at most
    ``ceil(T / 2P)`` a chunk)."""
    return 2 * rows + 1, min(tokens, -(-(tokens + rows * (2 * size - 1)) // size))


def padded_share(share: np.ndarray, rows: int, queries: int) -> np.ndarray:
    """A share's ``qo_indptr`` (``zigzag_share``) as a captured wrapper's planner reads it: padded to
    its ``rows`` and ``queries`` (``zigzag_capture``), the spare queries one row and the rows left
    over empty."""
    return np.concatenate([share, np.full(rows + 1 - len(share), queries)])


def _zigzag_chunks(
    qo_indptr: np.ndarray, size: int, index: int, causal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``zigzag_rows``' non-empty chunks as counts: each one's row start, its first and end token
    in the row, and its keys' end in the row."""
    starts, lens = qo_indptr[:-1].astype(np.int64), np.diff(qo_indptr).astype(np.int64)
    chunks = 2 * size
    base, rem = lens // chunks, lens % chunks

    def bounds(chunk: int) -> tuple[np.ndarray, np.ndarray]:
        first = chunk * base + np.minimum(chunk, rem)
        return first, first + base + (chunk < rem)

    (lo, lo_end), (hi, hi_end) = bounds(index), bounds(chunks - 1 - index)
    first, end = np.stack([lo, hi], 1).ravel(), np.stack([lo_end, hi_end], 1).ravel()
    row, row_len = np.repeat(starts, 2), np.repeat(lens, 2)
    keep = end > first
    first, end, row, row_len = first[keep], end[keep], row[keep], row_len[keep]
    return row, first, end, end if causal else row_len


class ZigzagWrapper:
    """A step's own tokens under SP placement, attended for this rank's zigzag chunks alone, out 0
    and lse -inf off them. The kernel is the backend's paged prefill wrapper over pages of one
    slot (page ``i`` the step's token ``i``), so a rank's two chunks, whose key prefixes overlap,
    plan as two rows of one call with no copy of the keys.

    ``paged`` builds the kernel, with ``batch_size``, ``max_total_tokens`` and ``max_num_pages``
    for a captured one; the first ``num_queries`` of ``run``'s arguments are query parts, the rest
    keys (``one_slot_pages``). ``check`` is ``ContextParallelWrapper``'s, which ``plan`` applies to
    every partner's share, as each rank can lay out every partner's."""

    def __init__(
        self,
        placement: ContextParallel,
        paged: Callable,
        num_queries: int,
        device: torch.device,
        dtype: torch.dtype,
        captured_rows: int = 0,
        captured_tokens: int = 0,
        check: Callable[[str, np.ndarray], None] | None = None,
    ):
        self._placement = placement
        self._num_queries = num_queries
        self._check = check
        self._device, self._dtype = device, dtype
        self._stager = PinnedStager(torch.int64)
        self._captured = captured_tokens > 0
        # the step's token count the last plan was for
        self._step_tokens = 0
        # each planned query's token to gather and row to scatter to; None when the last plan gave
        # this rank none
        self._gather = self._scatter = None
        if not self._captured:
            self.kernel = paged()
            return
        # fixed for the bucket: at most 2 chunks a row, of at most ceil(T / 2P) queries each, whose
        # keys are at most T each; and one more row for the padding queries. The padding and
        # leftover rows' one key each fit in the batch_size spare indices the kernel adds
        self._segments, self._queries = zigzag_capture(placement.size, captured_rows, captured_tokens)
        self.kernel = paged(
            batch_size=self._segments, max_total_tokens=self._queries, max_num_pages=2 * captured_tokens,
        )
        # never reallocated, so a replay reads the addresses it was captured with
        self._tokens = torch.zeros(2, self._queries, dtype=torch.int64, device=device)
        self._gather, self._scatter = self._tokens
        self._bucket_tokens = captured_tokens

    def share(self, qo_indptr: np.ndarray, index: int) -> np.ndarray:
        """Partner ``index``'s plan of a step as this wrapper makes it, which ``plan`` hands its kernel:
        its chunks' ``qo_indptr`` (``zigzag_share``), padded as its kernel's planner reads it."""
        return self._padded(zigzag_share(qo_indptr, self._placement.size, index))

    def _padded(self, qo_indptr: np.ndarray) -> np.ndarray:
        """A share's chunks' ``qo_indptr`` on a captured wrapper, padded to the bucket's queries and
        rows. Its padding queries are one row over one key (``plan``), so they cost O(pad): under
        causal alignment all but the last see none, and their outputs land in the row ``run``
        drops. The rows left over hold no query and one key."""
        if not self._captured:
            return qo_indptr
        return padded_share(qo_indptr, self._segments, self._queries)

    @torch.compiler.disable
    def plan(self, qo_indptr: torch.Tensor, causal: bool = True) -> None:
        step = qo_indptr.numpy()
        if self._check is not None:
            for index in range(self._placement.size):
                share = self.share(step, index)
                if len(share) > 1:
                    self._check(f"partner {index}'s own tokens", share)
        rows = zigzag_rows(step, self._placement.size, self._placement.index, causal)
        qo = self.share(step, self._placement.index)
        # a padding or leftover row's one key is token 0
        extra = len(qo) - len(rows.kv_indptr)
        kv_indptr = np.concatenate([rows.kv_indptr, rows.kv_indptr[-1] + np.arange(1, extra + 1)])
        kv_tokens = np.concatenate([rows.kv_tokens, np.zeros(extra, dtype=np.int64)])
        self._step_tokens = int(qo_indptr[-1])
        if self._captured:
            # a padding query gathers token 0 and scatters to the row past the bucket's, dropped
            host = np.zeros((2, self._queries), dtype=np.int64)
            host[:, : len(rows.tokens)] = rows.tokens
            host[1, len(rows.tokens):] = self._bucket_tokens
            self._stager.copy_(self._tokens, host.reshape(-1))
        elif len(rows.tokens):
            self._gather = self._scatter = to_device_nonblocking(rows.tokens, self._device, torch.int64)
        else:
            # none of the step's tokens is this rank's to attend
            self._gather = self._scatter = None
            return
        self.kernel.plan(
            qo_indptr=torch.from_numpy(qo.astype(np.int32)),
            paged_kv_indptr=torch.from_numpy(kv_indptr.astype(np.int32)),
            paged_kv_indices=torch.from_numpy(kv_tokens.astype(np.int32)),
            paged_kv_last_page_len=torch.ones(len(qo) - 1, dtype=torch.int32),
            kv_lens=torch.from_numpy(np.diff(kv_indptr).astype(np.int32)).as_subclass(HostLens),
            causal=causal, dtype=self._dtype,
        )

    def forget_plan(self) -> None:
        """Every ``plan`` plans its kernel, so there is nothing to forget."""

    def run(self, *queries_and_keys: torch.Tensor, return_lse: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        assert return_lse, "a rank's share of the own tokens is only ever merged"
        queries, keys = queries_and_keys[: self._num_queries], queries_and_keys[self._num_queries:]
        n, heads = queries[0].shape[:2]
        assert n >= self._step_tokens, f"{n} query tokens for a plan of {self._step_tokens}"
        # one row past the bucket's (an eager step's), for padding queries to land in
        rows = (self._bucket_tokens if self._captured else n) + 1
        out = queries[0].new_zeros(rows, heads, queries[0].shape[-1])
        lse = torch.full((rows, heads), -math.inf, dtype=torch.float32, device=queries[0].device)
        if self._gather is not None:
            own = [q.index_select(0, self._gather) for q in queries]
            got, got_lse = self.kernel.run(*own, one_slot_pages(*keys), return_lse=True)
            out.index_copy_(0, self._scatter, got.to(out.dtype))
            lse.index_copy_(0, self._scatter, got_lse.float())
        return out[:n], lse[:n]


def one_slot_pages(*keys: torch.Tensor):
    """A step's own keys as a paged kernel's layer of one-slot pages, page ``i`` token ``i``: MLA's
    latents ``[N, latent]``, or K and V ``[N, H, D]`` as FlashInfer's paged ``(k, v)``. Views,
    unless K's and V's strides differ, for which FA2 runs a kernel of its own that a capture would
    have to prewarm."""
    if len(keys) == 1:
        return keys[0].unsqueeze(1)
    k, v = keys
    if k.stride() != v.stride():
        k, v = k.contiguous(), v.contiguous()
    return k.unsqueeze(1), v.unsqueeze(1)


def merge(outs: Sequence[torch.Tensor], lses: torch.Tensor) -> torch.Tensor:
    """Attention over the union of disjoint key sets from each set's partial: each out ``[..., D]``
    (a ``[K, ..., D]`` stack is a sequence too) and their natural-log lses stacked, ``[K, ...]``;
    fp32."""
    return _merged(outs, lses)[0]


def _merged(
    outs: Sequence[torch.Tensor], lses: torch.Tensor, with_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``merge``, and with ``with_lse`` the union's lse: every part's weight relative to its row's
    largest lse, then one fp32 ``[..., D]`` accumulated a part at a time. A partial with lse -inf and
    a finite out adds nothing, and a query no partial has a key for (a captured bucket's padding)
    comes out 0, -inf."""
    lses = lses.float()
    # finite on a row with no key, so its every weight is 0
    top = lses.amax(0).clamp_min(torch.finfo(torch.float32).min)
    weights = (lses - top).exp_()
    total = weights.sum(0)
    out = weights[0].unsqueeze(-1) * outs[0]
    for weight, part in zip(weights[1:], outs[1:], strict=True):
        out.addcmul_(weight.unsqueeze(-1), part)
    out.div_(total.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1))
    return out, total.log_().add_(top) if with_lse else None


@torch.compiler.disable
def gather_merge(group: CommGroup, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """A query replicated on every rank of ``group`` attending keys sharded across them, from this
    rank's ``out`` ``[..., H, D]`` and natural-log ``lse`` ``[..., H]`` over its share: one gather
    of both, then ``merge``. Moves query-sized results rather than the keys."""
    if group.world_size == 1:
        return out
    gathered = _unpack(group.all_gather(_pack(out, lse).unsqueeze(0), dim=0), out.dtype)
    return merge(*gathered).to(out.dtype)


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
