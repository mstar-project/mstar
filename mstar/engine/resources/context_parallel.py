"""Context parallelism: a paged KV pool's token positions placed over a group of ranks
that would otherwise hold the same KV heads.

Page ids stay logical and are the same on every rank, so allocation, prefix keys,
offload and admission need nothing new. Within a page, logical slot ``s`` belongs to
partner ``s % P`` and sits at its physical slot ``s // P``, so each rank stores
``page_size // P`` slots a page. ``PagedKVConfig.place`` resolves the placement.

Attention over such a pool (``ContextParallelWrapper``) runs on each rank over its own slots and
returns the partial with its log-sum-exp; the partners exchange partials and ``merge`` them,
exactly.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.kv.plan import SINK_PAGE, HostLens, PagedIndptrs, first_pages, token_runs
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.utils.h2d import PinnedStager, to_device_nonblocking

if TYPE_CHECKING:
    from mstar.engine.resources.attn.base import PlacedAttention
    from mstar.engine.resources.attn.wrappers import PagedKernel


@dataclass(frozen=True)
class ContextParallel:
    """This rank's place in its KV pool's placement: which of a page's logical slots it stores.
    ``SPPlacement`` or ``TPPlacement`` says over which ranks."""

    # P: how many partners each page's slots stripe over
    size: int
    # r: this rank owns the logical slots s with s % P == r
    index: int
    # what the partners exchange over
    group: CommGroup

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


@dataclass(frozen=True)
class SPPlacement(ContextParallel):
    """Placed over the node's SP group, whose ranks hold the same heads: the partners are the whole
    group, in rank order (``SPContextParallelWrapper``)."""


@dataclass(frozen=True)
class TPPlacement(ContextParallel):
    """Placed over the block of the node's TP ranks that hold one KV head, each with different query
    heads of it (``TPContextParallelWrapper``). The exchange is an all-to-all over the whole TP
    group with nothing sent to the ranks outside the block, so no group of its own is needed."""

    # the partners' ranks in ``group``, in partner-index order
    partners: tuple[int, ...]


def local_indptrs(placement: ContextParallel, indptrs: PagedIndptrs, page_size: int) -> tuple[PagedIndptrs, np.ndarray]:
    """A plan's logical paged indptrs as this rank's (``page_size`` the logical one), and per row
    whether it holds any of this rank's slots.

    Every row keeps a length of at least 1 and a last page holding at least one slot, as on one
    rank: FlashInfer's split-KV planner counts a row's pages where its kernel counts its tokens,
    and a last page with none of this rank's slots puts the two out of step. Such a page is
    dropped from its row. A row with no slot here attends one slot of SINK_PAGE instead, and its
    partial is masked before the merge (``_masked``)."""
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
        last = np.where(valid, np.where(drop, slots, last), 1)
        lens = np.where(valid, lens, 1)
        kv_indptr, indices = torch.from_numpy(new_kvp.astype(np.int32)), torch.from_numpy(out)
    return PagedIndptrs(
        qo_indptr=indptrs.qo_indptr,
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=indices,
        paged_kv_last_page_len=torch.from_numpy(last.astype(np.int32)),
        kv_lens=torch.from_numpy(lens.astype(np.int32)).as_subclass(HostLens),
    ), valid


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


class KeyRows(NamedTuple):
    """Rows of a step's own queries, each over a list of the step's own tokens as its keys: what an
    ``OwnTokensKernel`` plans. Token ids, not pages."""

    # [R + 1] each row's queries, in the order ``run`` gets them
    qo_indptr: np.ndarray
    # [R + 1] each row's keys in ``kv_tokens``
    kv_indptr: np.ndarray
    # [K] each row's keys, as the step's token ids
    kv_tokens: np.ndarray

    @classmethod
    def whole(cls, qo_indptr: np.ndarray) -> "KeyRows":
        """Each row of ``qo_indptr`` over its own tokens: TP placement's own tokens."""
        return cls(qo_indptr, qo_indptr, np.arange(int(qo_indptr[-1]), dtype=np.int64))


class ShareCapture(NamedTuple):
    """What the kernel of a captured step's own tokens is sized for."""

    # the kernel's fixed rows and queries
    rows: int
    queries: int
    # the bucket's tokens, which the share's keys are drawn from
    tokens: int


class OwnTokensKernel(ABC):
    """Attention of rows of a step's own queries over lists of its own tokens as keys
    (``KeyRows``), from the keys the layer passes ``run``: a placed pool's own-token part, TP's
    whole rows and SP's zigzag share alike (``PlacedAttention.own_tokens_kernel`` supplies it).
    Nothing here assumes pages: a paged kernel reads the keys in place (``OneSlotPages``); a
    ragged one reads contiguous runs (FlashInfer's for whole rows), or would gather the keys by
    ``kv_tokens``.

    A captured kernel is run on ``captured.queries`` queries every replay, however few its plan has
    (``ZigzagWrapper`` gathers that many; TP passes the bucket's). A kernel that declares
    ``whole_rows_only`` is refused where any other rows would be planned on it."""

    # what a captured kernel is sized for; None on an eager one
    captured: ShareCapture | None = None
    # a kernel that takes only whole rows, each over its own tokens (``KeyRows.whole``)
    whole_rows_only: bool = False

    @abstractmethod
    def plan(self, rows: KeyRows, causal: bool) -> None:
        """Plans ``rows``; ``causal`` aligns each row's queries to the end of its keys."""

    @abstractmethod
    def run(
        self, queries: tuple[torch.Tensor, ...], keys: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(out, lse)`` (lse natural-log fp32) of the planned rows' ``queries``, in plan order,
        over the step's ``keys``: query parts and keys as the backend's ``run`` takes them. A row for
        every query passed; one past the plan (a captured bucket's padding) holds whatever the
        kernel left, which reaches no real row."""

    def forget_plan(self) -> None:  # noqa: B027 - a kernel that skips a layout it holds replans
        """Every ``plan`` plans the kernel, so there is nothing to forget."""


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


class OneSlotPages(OwnTokensKernel):
    """An ``OwnTokensKernel`` on a backend's paged kernel over pages of one slot, page ``i`` the
    step's token ``i``: rows whose key lists overlap (a zigzag share's two chunks) plan as rows of
    one call with no copy of the keys. Captured, it pads each plan as the paged planner's fixed
    buffers read it: the spare queries one row, the rows left over empty, each over token 0. The
    padding row's one key costs O(pad): under causal alignment all but its last query see none."""

    def __init__(self, kernel: "PagedKernel", dtype: torch.dtype, captured: ShareCapture | None = None):
        self.kernel, self._dtype, self.captured = kernel, dtype, captured

    def plan(self, rows: KeyRows, causal: bool) -> None:
        qo, kv_indptr, kv_tokens = rows
        if self.captured is not None:
            qo = np.concatenate([qo, np.full(self.captured.rows + 1 - len(qo), self.captured.queries)])
            # a padding or leftover row's one key is token 0
            extra = len(qo) - len(kv_indptr)
            kv_indptr = np.concatenate([kv_indptr, kv_indptr[-1] + np.arange(1, extra + 1)])
            kv_tokens = np.concatenate([kv_tokens, np.zeros(extra, dtype=np.int64)])
        self.kernel.plan(
            qo_indptr=torch.from_numpy(qo.astype(np.int32)),
            paged_kv_indptr=torch.from_numpy(kv_indptr.astype(np.int32)),
            paged_kv_indices=torch.from_numpy(kv_tokens.astype(np.int32)),
            paged_kv_last_page_len=torch.ones(len(qo) - 1, dtype=torch.int32),
            kv_lens=torch.from_numpy(np.diff(kv_indptr).astype(np.int32)).as_subclass(HostLens),
            causal=causal, dtype=self._dtype,
        )

    def run(self, queries, keys):
        return self.kernel.run(*queries, one_slot_pages(*keys), return_lse=True)


class ContextParallelWrapper(ABC):
    """A step's attention over a placed pool from a backend's two wrappers: each rank attends its
    own slots, and the partners exchange the partials and merge them, as its placement's subclass
    does (``TPContextParallelWrapper``, ``SPContextParallelWrapper``). A one-token-a-row step
    attends the pages after its write (``new`` None); any other step attends its stored context
    (not causal) on ``context`` and its own tokens, from the keys the layer passes to ``run``, on
    ``new``. ``page_size`` is the logical one.

    A plan's lengths differ by rank, so a planner that refuses on lengths could refuse on one rank
    of the lockstep instance alone. The backend refuses, on every rank alike and before anything is
    planned, an eager step whose plans it may (admission, ``rank_plans``) and a captured bucket whose
    every plan it may (at capture, from the kernels' fixed sizes)."""

    def __init__(
        self,
        placement: ContextParallel,
        page_size: int,
        dtype: torch.dtype,
        context: "PagedKernel",
        new: "OwnTokensKernel | ZigzagWrapper | None",
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
    def plan(self, kv_out, causal: bool = True, real_rows: frozenset | None = None):
        """``kv_out``: the step's logical ``KVPlanOutput`` for this label. The context attention reads
        its indptrs for a one-token-a-row step, its stored context alone otherwise.

        ``real_rows``: a replay's real requests when it has padding rows (``StepContext.real_rows``),
        which follow the real rows in each label. The own-token part then plans the real rows alone,
        so the padding's queries come out 0, lse -inf, and its keys lie past the plan: FA3's ragged
        kernel (FlashInfer's TP own tokens) reads keys past a row's end to its KV tile's, where
        ``run`` zeroes them, so a NaN the model's layers left in a padding row's keys would
        otherwise reach the real rows before it. (FlashInfer passes them; MLA's own tokens read
        each row's own pages alone.)"""
        context = kv_out.cpu_indptrs
        if self.new is not None:
            context = context.context_only(self._page_size)
            qo = context.qo_indptr
            own = None if real_rows is None else _leading_rows(kv_out, real_rows)
            self._plan_own(qo if own is None else qo[: own + 1], causal)
        local, valid_rows = local_indptrs(self._placement, context, self._page_size)
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
        self._qo_indptr_buf = self.context._qo_indptr_buf
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

    @staticmethod
    def local_geometry(placement: ContextParallel, page_size: int, num_qo_heads: int) -> dict[str, int]:
        """A paged attention wrapper's ``page_size`` and ``num_qo_heads`` over this rank's slots: the
        slots a page it stores, and its own query heads."""
        return dict(page_size=page_size // placement.size, num_qo_heads=num_qo_heads)

    @classmethod
    @abstractmethod
    def own_tokens(
        cls, kernels: "PlacedAttention", placement: ContextParallel, device: torch.device, workspace: torch.Tensor,
        kwargs: dict, max_total_tokens: int | None,
    ) -> "OwnTokensKernel | ZigzagWrapper":
        """The part a multi-token step attends its own tokens on, from the backend's ``kernels``
        (``_plan_own`` plans it, ``_merged_over_ranks`` runs it)."""

    @staticmethod
    @abstractmethod
    def rank_plans(
        placement: ContextParallel, qo_indptr: np.ndarray, num_qo_heads: int,
    ) -> list[tuple[np.ndarray, int]]:
        """Every plan a multi-token step of ``qo_indptr``'s rows makes on any rank of the placement,
        as ``(qo_indptr, query heads)``: the context's at the local geometry's ``num_qo_heads``, and
        the own tokens'. What admission bounds an eager step's planner refusals by, the same on every
        rank; a captured bucket's plans are bounded at capture."""

    @abstractmethod
    def _plan_own(self, qo_indptr: torch.Tensor, causal: bool) -> None:
        """Plans the own-token part for the rows ``qo_indptr`` lays out."""

    @staticmethod
    def cap_hint(placement: ContextParallel) -> str:
        """Why a planner's work cap comes sooner under this placement, for a refusal's message."""
        return ""

    def attend(
        self, queries: tuple[torch.Tensor, ...], kv_cache_layer: torch.Tensor,
        new_keys: tuple[torch.Tensor | None, ...],
    ) -> torch.Tensor:
        """Attention over the whole placed pool, in the query's dtype (merged in fp32): ``queries``
        and ``new_keys`` are the query parts (each ``[N, H, *]``) and the step's own keys as this
        backend's ``run`` takes them (``new_keys`` read only on a step that isn't one token a row).
        Every rank runs the same collectives."""
        if self.new is not None and any(key is None for key in new_keys):
            raise ValueError(
                "attention over a context-parallel KV pool attends a step's own tokens from the keys "
                "run is passed (k and v, or MLA's latents as k); this step has more than one token "
                "a row and got none"
            )
        keys = None if self.new is None else new_keys
        return self._merged_over_ranks(queries, kv_cache_layer, keys).to(queries[0].dtype)

    @abstractmethod
    def _merged_over_ranks(
        self, queries: tuple[torch.Tensor, ...], kv_cache_layer: torch.Tensor,
        keys: tuple[torch.Tensor, ...] | None,
    ) -> torch.Tensor:
        """This rank's slots' partial for ``queries``, exchanged with the partners' and merged with
        the own tokens' attention from ``keys`` (None for a one-token-a-row step)."""

    def forget_plan(self) -> None:
        """Replan the own-token wrapper on the next ``plan`` whatever its layout, for a dropped
        pre-plan: a ragged one skips a layout it already holds (`_RaggedPrefillBase.forget_plan`)."""
        if self.new is not None:
            self.new.forget_plan()

    @property
    def reads_step_kv(self) -> bool:
        """Whether the step attends the keys ``run`` is passed (``PlacedAttention.reads_step_kv``):
        any step but one token a row."""
        return self.new is not None


def _leading_rows(kv_out, real_rows: frozenset) -> int | None:
    """How many of ``kv_out``'s rows are ``real_rows``, which lead it; None when all are, or none."""
    rids = [view.request_id for view in kv_out.views] if kv_out.rows is None else [row[0] for row in kv_out.rows]
    n = sum(rid in real_rows for rid in rids)
    if not all(rid in real_rows for rid in rids[:n]):
        raise ValueError(f"a padding row comes before a real one in a plan of rows {rids}")
    return n if 0 < n < len(rids) else None


class TPContextParallelWrapper(ContextParallelWrapper):
    """Attention over a ``TPPlacement``: the partners hold different query heads of one KV head, so
    each attends its slots for every partner's heads (``_exchange_queries``), sends each partner its
    heads' partial back (``_exchange_partials``), and merges them with the own tokens, which it
    attends whole for its own heads."""

    _placement: TPPlacement
    new: "OwnTokensKernel | None"

    @staticmethod
    def local_geometry(placement: TPPlacement, page_size: int, num_qo_heads: int) -> dict[str, int]:
        """And every partner's query heads, which it attends for them."""
        geometry = ContextParallelWrapper.local_geometry(placement, page_size, num_qo_heads)
        return {**geometry, "num_qo_heads": num_qo_heads * placement.size}

    @classmethod
    def own_tokens(cls, kernels, placement, device, workspace, kwargs, max_total_tokens):
        # captured: every row whole, and one more row for the queries past a plan
        captured = None
        if max_total_tokens:
            captured = ShareCapture((kwargs.get("batch_size") or 0) + 1, max_total_tokens, max_total_tokens)
        return _own_tokens_kernel(kernels, workspace, kwargs, captured, whole=True)

    @staticmethod
    def rank_plans(placement, qo_indptr, num_qo_heads):
        # the own tokens, whole, at this rank's own heads: the same lengths on every rank
        return [(qo_indptr, num_qo_heads), (qo_indptr, num_qo_heads // placement.size)]

    def _plan_own(self, qo_indptr, causal):
        self.new.plan(KeyRows.whole(qo_indptr.numpy()), causal)

    @staticmethod
    def cap_hint(placement: TPPlacement) -> str:
        return (f" Under TP placement each rank plans every partner's query heads, so a step reaches the cap "
                f"{placement.size}x sooner; placing positions over SP ranks avoids that.")

    def _merged_over_ranks(self, queries, kv_cache_layer, keys):
        new = None if keys is None else self.new.run(queries, keys)
        parts = self._exchange_queries(queries)
        # over this rank's slots, with the rows holding any of them
        out, lse = self.context.run(*parts, kv_cache_layer, return_lse=True)
        outs, lses = self._exchange_partials(out, lse, self._valid[: parts[0].shape[0]])
        if new is None:
            return merge(outs, lses)
        # the own tokens' part beside the partners', with no copy of theirs
        return merge([*outs, new[0]], torch.cat([lses, new[1].float()[None]]))

    def _exchange_queries(self, queries: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """The query parts this rank attends its slots for, from its own (each ``[N, H, *]``): every
        partner's, ``[N, P * H, *]`` partner-major, the parts sent as one tensor."""
        placement = self._placement
        q = (queries[0] if len(queries) == 1 else torch.cat(queries, dim=-1)).unsqueeze(0)
        if placement.size == placement.group.world_size:
            # every rank of the group is a partner (MLA's always are): the same block goes to each
            q = placement.group.all_gather(q.contiguous(), dim=0)
        else:
            q = placement.group.to_partners(q.expand(placement.size, *q.shape[1:]), placement.partners)
        return q.transpose(0, 1).flatten(1, 2).split([part.shape[-1] for part in queries], dim=-1)

    def _exchange_partials(
        self, out: torch.Tensor, lse: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Every partner's partial for this rank's heads, ``[P, N, H, D]`` in ``out``'s dtype and
        ``[P, N, H]`` fp32, to ``merge``; from this rank's ``out`` ``[N, H', D]`` and natural-log
        ``lse`` ``[N, H']`` over its slots, for the heads ``_exchange_queries`` returned, masked by
        ``valid`` (``_masked``)."""
        placement = self._placement
        out, lse = _masked(out, lse, valid)
        heads = out.shape[1] // placement.size
        blocks = _pack(out, lse).unflatten(1, (placement.size, heads)).transpose(0, 1)
        return _unpack(placement.group.to_partners(blocks, placement.partners), out.dtype)


class SPContextParallelWrapper(ContextParallelWrapper):
    """Attention over an ``SPPlacement``: the partners hold the same heads, so each attends its own
    slots and its zigzag share of the own tokens (``ZigzagWrapper``) for its own queries, merges the
    two, and one gather merges every partner's."""

    _placement: SPPlacement
    new: "ZigzagWrapper | None"

    @classmethod
    def own_tokens(cls, kernels, placement, device, workspace, kwargs, max_total_tokens):
        # its zigzag share of the own tokens, whose two chunks' keys overlap
        captured = None
        if max_total_tokens:
            captured = zigzag_capture(placement.size, kwargs.get("batch_size") or 0, max_total_tokens)
        return ZigzagWrapper(placement, _own_tokens_kernel(kernels, workspace, kwargs, captured, whole=False), device)

    @staticmethod
    def rank_plans(placement, qo_indptr, num_qo_heads):
        # and every partner's zigzag share of the own tokens, which differ by rank, at the same heads
        shares = [zigzag_share(qo_indptr, placement.size, index) for index in range(placement.size)]
        return [(plan, num_qo_heads) for plan in (qo_indptr, *shares)]

    def _plan_own(self, qo_indptr, causal):
        self.new.plan(qo_indptr, causal=causal)

    def _merged_over_ranks(self, queries, kv_cache_layer, keys):
        new = None if keys is None else self.new.run(queries, keys)
        # this rank's slots' partial merged with its share of the own tokens, rounded back to the
        # partial's dtype for the one gather: one rounding more than TP placement's, at the floor a
        # bf16 partial carries already; gathering the fp32 merge would double the gather's bytes
        out, lse = self.context.run(*queries, kv_cache_layer, return_lse=True)
        out, lse = _masked(out, lse, self._valid[: queries[0].shape[0]])
        if new is not None:
            merged, lse = merge_lse((out, new[0]), torch.stack([lse, new[1]]))
            out = merged.to(out.dtype)
        return gather_merge(self._placement.group, out, lse)


def _own_tokens_kernel(kernels, workspace, kwargs, captured: ShareCapture | None, whole: bool) -> OwnTokensKernel:
    """The backend's own-token kernel (``PlacedAttention.own_tokens_kernel``), refused at build unless
    it is an ``OwnTokensKernel``, one built for a capture saying so."""
    kernel = kernels.own_tokens_kernel(workspace, kwargs, captured, whole)
    if not isinstance(kernel, OwnTokensKernel):
        raise ValueError(f"a placed pool's own tokens run on an OwnTokensKernel, not a {type(kernel).__name__}")
    if captured is not None and kernel.captured is None:
        raise ValueError(f"{type(kernel).__name__} was built for a capture and leaves its captured shape unset")
    if kernel.whole_rows_only and not whole:
        raise ValueError(f"{type(kernel).__name__} takes whole rows alone, and this placement's own tokens aren't")
    return kernel


def zigzag_rows(qo_indptr: np.ndarray, size: int, index: int, causal: bool = True) -> tuple[np.ndarray, KeyRows]:
    """Rank ``index`` of ``size``'s zigzag chunks of the rows ``qo_indptr`` lays out (chunks ``r``
    and ``2P - 1 - r`` of each row's ``2P``; see the context-parallel docs): the step's tokens it
    attends, chunk by chunk, and its chunks as rows over their keys, each row's first token to the
    chunk's last (or the row's, not causal)."""
    row, first, end, kv_end = _zigzag_chunks(qo_indptr, size, index, causal)
    _, tokens = token_runs(row + first, end - first)
    kv, kv_tokens = token_runs(row, kv_end)
    return tokens, KeyRows(np.concatenate([[0], np.cumsum(end - first)]), kv, kv_tokens)


def zigzag_share(qo_indptr: np.ndarray, size: int, index: int) -> np.ndarray:
    """The ``qo_indptr`` of rank ``index`` of ``size``'s zigzag chunks of the rows ``qo_indptr`` lays
    out, before a capture's padding."""
    _, first, end, _ = _zigzag_chunks(qo_indptr, size, index, causal=True)
    return np.concatenate([[0], np.cumsum(end - first)])


def zigzag_capture(size: int, rows: int, tokens: int) -> ShareCapture:
    """A zigzag share's kernel captured for a bucket of ``rows`` rows and ``tokens`` tokens: its
    fixed rows (at most 2 chunks a row, and one more for the padding queries) and queries (at most
    ``ceil(T / 2P)`` a chunk)."""
    return ShareCapture(2 * rows + 1, min(tokens, -(-(tokens + rows * (2 * size - 1)) // size)), tokens)


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
    """A step's own tokens under SP placement, attended for this rank's zigzag chunks alone on
    ``kernel``, out 0 and lse -inf off them. On a captured kernel it gathers the same number of
    queries every replay: its share, then padding, which gathers token 0 and scatters to a row
    ``run`` drops."""

    def __init__(self, placement: SPPlacement, kernel: OwnTokensKernel, device: torch.device):
        self._placement, self.kernel, self._device = placement, kernel, device
        self._stager = PinnedStager(torch.int64)
        self._bucket_tokens = 0 if kernel.captured is None else kernel.captured.tokens
        # the step's token count the last plan was for
        self._step_tokens = 0
        # each planned query's token to gather and row to scatter to; None when the last plan gave
        # this rank none
        self._gather = self._scatter = None
        if kernel.captured is not None:
            # never reallocated, so a replay reads the addresses it was captured with
            self._tokens = torch.zeros(2, kernel.captured.queries, dtype=torch.int64, device=device)
            self._gather, self._scatter = self._tokens

    @torch.compiler.disable
    def plan(self, qo_indptr: torch.Tensor, causal: bool = True) -> None:
        step = qo_indptr.numpy()
        tokens, rows = zigzag_rows(step, self._placement.size, self._placement.index, causal)
        self._step_tokens = int(qo_indptr[-1])
        if self._bucket_tokens:
            # a padding query gathers token 0 and scatters to the row past the bucket's, dropped
            host = np.zeros(tuple(self._tokens.shape), dtype=np.int64)
            host[:, : len(tokens)] = tokens
            host[1, len(tokens):] = self._bucket_tokens
            self._stager.copy_(self._tokens, host.reshape(-1))
        elif len(tokens):
            self._gather = self._scatter = to_device_nonblocking(tokens, self._device, torch.int64)
        else:
            # none of the step's tokens is this rank's to attend
            self._gather = self._scatter = None
            return
        self.kernel.plan(rows, causal)

    def forget_plan(self) -> None:
        self.kernel.forget_plan()

    @property
    def captured(self) -> ShareCapture | None:
        """What its kernel is sized for."""
        return self.kernel.captured

    def run(
        self, queries: tuple[torch.Tensor, ...], keys: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's share of the own tokens' ``(out, lse)`` for every token of the step."""
        n, heads = queries[0].shape[:2]
        assert n >= self._step_tokens, f"{n} query tokens for a plan of {self._step_tokens}"
        # one row past the bucket's (an eager step's), for padding queries to land in
        rows = (self._bucket_tokens or n) + 1
        out = queries[0].new_zeros(rows, heads, queries[0].shape[-1])
        lse = torch.full((rows, heads), -math.inf, dtype=torch.float32, device=queries[0].device)
        if self._gather is not None:
            got, got_lse = self.kernel.run(tuple(q.index_select(0, self._gather) for q in queries), keys)
            out.index_copy_(0, self._scatter, got.to(out.dtype))
            lse.index_copy_(0, self._scatter, got_lse.float())
        return out[:n], lse[:n]


def merge(outs: Sequence[torch.Tensor], lses: torch.Tensor) -> torch.Tensor:
    """Attention over the union of disjoint key sets from each set's partial: each out ``[..., D]``
    (a ``[K, ..., D]`` stack is a sequence too) and their natural-log lses stacked, ``[K, ...]``;
    fp32."""
    return _weighted(outs, lses)[0]


def merge_lse(outs: Sequence[torch.Tensor], lses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``merge``, and the union's natural-log lse, fp32, to merge again."""
    out, total, top = _weighted(outs, lses)
    return out, total.log_().add_(top)


def _weighted(outs: Sequence[torch.Tensor], lses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``merge``'s out, from every part's weight relative to its row's largest lse, one fp32
    ``[..., D]`` accumulated a part at a time; and the weights' sum and that largest lse. A partial
    with lse -inf and a finite out adds nothing, and a query no partial has a key for (a captured
    bucket's padding) comes out 0 (lse -inf)."""
    lses = lses.float()
    # finite on a row with no key, so its every weight is 0
    top = lses.amax(0).clamp_min(torch.finfo(torch.float32).min)
    weights = (lses - top).exp_()
    total = weights.sum(0)
    out = weights[0].unsqueeze(-1) * outs[0]
    for weight, part in zip(weights[1:], outs[1:], strict=True):
        out.addcmul_(weight.unsqueeze(-1), part)
    out.div_(total.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1))
    return out, total, top


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
