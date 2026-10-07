"""GLM-5.2 DSA past ``index_topk`` (``dsa_long_context``): paged index keys, slot selection,
sparse MLA.

The index keys live in a second KV resource (``INDEX_KV_RESOURCE``, one ``index_head_dim``
latent per token per FULL layer). Each query row scores its request's keys and keeps its top
``index_topk`` as slots of the MLA latent cache, and attention runs over just those slots. Every
query row selects over its own causal prefix, so a prefill beyond ``index_topk`` is the same
computation as prefilling ``index_topk`` tokens and decoding the rest one at a time.

On CUDA: ``dsa_kernels`` for the scores (decode rows in one launch, prefill rows in chunks of
one request), flashinfer's fused top-k + page-table transform for the slots, and
``sparse_mla``. Elsewhere, torch versions of the same math (the CPU tests' reference).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import torch

from mstar.engine.resources.attn import sparse_mla
from mstar.model.glm52.components.indexer import select_topk_causal

# Prefill rows scored per launch: a chunk's scores are [rows, its last row's keys] fp32, so a
# chunk takes at most this many rows and this many bytes of scores (fewer rows as keys grow).
PREFILL_CHUNK_ROWS = 1024
SCORE_BUDGET_BYTES = 256 << 20


@dataclass
class Glm52DsaPagedContext:
    """Per-forward DSA state, built by the submodule's preprocess.

    Per query row (device): ``row_req`` its request's page-table row, ``lens`` the keys it
    sees (position + 1); ``host_lens`` the same on the host. Per request (device, int32,
    zero-padded): ``kv_table`` pages of the MLA latent cache, ``index_table`` pages of the
    index keys. ``spans`` (host): (first row, rows, request) per request, in row order.
    ``width``: the decode rows' score width, the longest context in the step, or under a
    CUDA graph the serving window (every replay reuses the captured shapes). ``step_rows``:
    set on a subset of a step's rows (``rows``, ``pick``), where they sit in the step, so the
    subset writes the cache at its own rows of the step's plan. ``decode_rows``: score every
    row on its own (a decode step's, an MTP verify block's), the capturable launch."""

    row_req: torch.Tensor
    lens: torch.Tensor
    host_lens: list[int]
    kv_table: torch.Tensor
    index_table: torch.Tensor
    spans: list[tuple[int, int, int]]
    width: int
    page_size: int
    topk: int
    needs_selection: bool
    step_rows: torch.Tensor | None = None
    decode_rows: bool = False
    last_selection: torch.Tensor | None = None
    last_selection_layer: int | None = None
    sparse_plan: sparse_mla.SparseGraphPlan | sparse_mla.EagerSparsePlan | None = None

    @property
    def attn_lens(self) -> list[int]:
        """Keys each row attends: its top ``topk``, or all it sees below that."""
        return [min(n, self.topk) for n in self.host_lens]

    def rows(self, c0: int, c1: int) -> Glm52DsaPagedContext:
        """The context of rows ``[c0, c1)`` alone, for a pass over them after the rows before
        them have written the cache. Every row takes the sparse path (its own selection)."""
        spans = []
        for r0, n, req in self.spans:
            a, b = max(r0, c0), min(r0 + n, c1)
            if a < b:
                spans.append((a - c0, b - a, req))
        idx = torch.arange(c0, c1, device=self.lens.device)
        return self._subset(idx, self.host_lens[c0:c1], spans)

    def pick(self, rows: list[int]) -> Glm52DsaPagedContext:
        """The context of the given rows (ascending) alone, one row per span."""
        starts = [r0 for r0, _, _ in self.spans]
        req = [self.spans[bisect.bisect_right(starts, r) - 1][2] for r in rows]
        idx = torch.tensor(rows, device=self.lens.device)
        return self._subset(idx, [self.host_lens[r] for r in rows],
                            [(i, 1, q) for i, q in enumerate(req)])

    def _subset(self, idx: torch.Tensor, host_lens: list[int],
                spans: list[tuple[int, int, int]]) -> Glm52DsaPagedContext:
        return Glm52DsaPagedContext(
            row_req=self.row_req[idx], lens=self.lens[idx], host_lens=host_lens,
            kv_table=self.kv_table, index_table=self.index_table, spans=spans,
            width=max(host_lens), page_size=self.page_size, topk=self.topk,
            needs_selection=True,
            step_rows=idx if self.step_rows is None else self.step_rows[idx])


def write_rows(kv, layer_idx: int, rows: torch.Tensor, ctx: Glm52DsaPagedContext) -> None:
    """``kv.write_kv`` of a forward's rows (per-token latents): the whole step's, or under
    ``ctx.step_rows`` a subset's, at its rows of the step's planned slots."""
    if ctx.step_rows is None:
        kv.write_kv(rows, None, layer_idx=layer_idx)
        return
    pages, offsets = kv.write_slots()
    layer = kv.layer_view(layer_idx)
    layer[pages[ctx.step_rows], offsets[ctx.step_rows]] = rows.to(layer.dtype)


def select(q: torch.Tensor, w: torch.Tensor, index_layer: torch.Tensor,
           ctx: Glm52DsaPagedContext, group=None) -> torch.Tensor:
    """``[rows, topk]`` int32 latent-cache slots of every row's top keys (entries past a row's
    ``attn_lens`` are unused). Single-row spans (decode) score in one launch; longer spans
    (prefill) in chunks of one request's rows, each chunk as wide as its last row's context.
    A TP ``group`` splits a prefill's rows across ranks (``sparse_mla.select_rows``)."""
    if ctx.decode_rows or all(n == 1 for _, n, _ in ctx.spans):  # capturable: no host index
        scores = _decode_scores(q, w, index_layer, ctx.index_table, ctx.row_req, ctx.lens,
                                ctx.width)
        return select_slots(scores, ctx.kv_table, ctx.row_req, ctx.lens, ctx.page_size,
                            ctx.topk)
    out = torch.empty(q.shape[0], ctx.topk, dtype=torch.int32, device=q.device)
    single = [r0 for r0, n, _ in ctx.spans if n == 1]
    if single:
        idx = torch.tensor(single, device=q.device)
        width = max(ctx.host_lens[r] for r in single)
        scores = _decode_scores(q[idx], w[idx], index_layer, ctx.index_table, ctx.row_req[idx],
                                ctx.lens[idx], width)
        out[idx] = select_slots(scores, ctx.kv_table, ctx.row_req[idx], ctx.lens[idx],
                                ctx.page_size, ctx.topk)
    for r0, n, req in ctx.spans:
        if n > 1:
            out[r0:r0 + n] = sparse_mla.select_rows(
                lambda a, b, req=req: _select_rows(q, w, index_layer, ctx, req, a, b),
                r0, r0 + n, ctx.topk, q.device, group)
    return out


def _select_rows(q, w, index_layer, ctx, req, r0, r1):
    """Rows ``[r0, r1)`` of request ``req``'s prefill, in chunks as wide as their last row's
    context and at most SCORE_BUDGET_BYTES of scores."""
    out = torch.empty(r1 - r0, ctx.topk, dtype=torch.int32, device=q.device)
    c0 = r0
    while c0 < r1:
        c1 = _chunk_end(ctx.host_lens, c0, r1)
        width = ctx.host_lens[c1 - 1]  # causal: the chunk's last row sees the most
        scores = _prefill_scores(q[c0:c1], w[c0:c1], index_layer, ctx.index_table[req],
                                 ctx.lens[c0:c1], width)
        out[c0 - r0:c1 - r0] = select_slots(scores, ctx.kv_table, ctx.row_req[c0:c1],
                                            ctx.lens[c0:c1], ctx.page_size, ctx.topk)
        c0 = c1
    return out


def _chunk_end(host_lens: list[int], c0: int, end: int) -> int:
    """End of the prefill chunk that starts at row ``c0``: the most rows, up to
    PREFILL_CHUNK_ROWS, whose [rows, last row's keys] fp32 scores fit SCORE_BUDGET_BYTES, and
    at least one."""
    lo, hi = 1, min(PREFILL_CHUNK_ROWS, end - c0)
    while lo < hi:  # the bytes grow with the rows, so the largest fitting count is a bisection
        mid = (lo + hi + 1) // 2
        if 4 * mid * host_lens[c0 + mid - 1] <= SCORE_BUDGET_BYTES:
            lo = mid
        else:
            hi = mid - 1
    return c0 + lo


def _gathered_scores(q, w, keys, lens, width):
    """The indexer's fp32 score math on gathered keys ``(rows, width, D)``; -inf past lens."""
    with torch.autocast(q.device.type, enabled=False):  # fp32 under the engine's autocast too
        dots = torch.einsum("rhd,rsd->rhs", q.float(), keys.float()).relu()
        scores = torch.einsum("rh,rhs->rs", w.float(), dots)
    visible = torch.arange(width, device=q.device).expand(q.shape[0], -1) < lens[:, None]
    return scores.masked_fill(~visible, float("-inf"))


def _decode_scores(q, w, index_layer, index_table, row_req, lens, width):
    if q.is_cuda:
        from mstar.model.glm52.dsa_kernels import decode_scores

        return decode_scores(q.to(torch.bfloat16).contiguous(), w.float().contiguous(),
                             index_layer.contiguous(), index_table, row_req, lens, width)
    keys = index_layer[index_table[row_req.long()].long()].flatten(1, 2)[:, :width]
    return _gathered_scores(q, w, keys, lens, width)


def _prefill_scores(q, w, index_layer, table_row, lens, width):
    if q.is_cuda:
        from mstar.model.glm52.dsa_kernels import prefill_scores

        return prefill_scores(q.to(torch.bfloat16).contiguous(), w.float().contiguous(),
                              index_layer.contiguous(), table_row.contiguous(),
                              (lens - 1).contiguous(), width)
    keys = index_layer[table_row.long()].flatten(0, 1)[:width]
    return _gathered_scores(q, w, keys.expand(q.shape[0], -1, -1), lens, width)


def select_slots(scores: torch.Tensor, kv_table: torch.Tensor, row_req: torch.Tensor,
                 lens: torch.Tensor, page_size: int, topk: int) -> torch.Tensor:
    """``[rows, topk]`` int32 slots of the MLA latent cache (``page * page_size + offset``)
    of each row's top ``topk`` scores; entries past ``min(lens, topk)`` are unused. Ties go
    to the earlier position, so a row's choice does not depend on the batch."""
    if scores.is_cuda:
        import flashinfer

        return flashinfer.top_k_page_table_transform(
            scores, kv_table, lens, topk, row_to_batch=row_req,
            deterministic=True, tie_break=flashinfer.TopKTieBreak.SMALL, dsa_graph_safe=True,
            page_size=page_size)
    positions = select_topk_causal(scores, lens.long() - 1, topk).long()        # (rows, topk)
    live = positions >= 0
    positions = positions.clamp_min(0)
    pages = kv_table[row_req.long()].long().gather(1, positions // page_size)
    slots = pages * page_size + positions % page_size
    return torch.where(live, slots, -1).to(torch.int32)


def sparse_attend(q_nope: torch.Tensor, q_pe: torch.Tensor, latent_layer: torch.Tensor,
                  slots: torch.Tensor, ctx: Glm52DsaPagedContext, sm_scale: float) -> torch.Tensor:
    """Absorbed MLA of each row over its selected latents: ``[rows, heads, kv_lora_rank]``.
    ``latent_layer`` is the MLA cache's layer view ``[pages, page_size, ckv + kpe]``."""
    if not sparse_mla.uses_kernel(q_nope, latent_layer):
        return sparse_mla.reference(q_nope, q_pe, latent_layer, slots, ctx.attn_lens, sm_scale)
    if ctx.sparse_plan is None:  # eager: every layer of the forward shares one plan
        ctx.sparse_plan = sparse_mla.EagerSparsePlan(
            ctx.attn_lens, ctx.topk, q_nope.shape[1], q_nope.shape[-1], q_pe.shape[-1],
            sm_scale, q_nope.device)
    return ctx.sparse_plan.attend(q_nope, q_pe, latent_layer, slots)
