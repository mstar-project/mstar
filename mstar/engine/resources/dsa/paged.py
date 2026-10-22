"""DeepSeek sparse attention (DSA) past ``topk`` over the paged MLA cache: index keys, each
query row's selection, sparse MLA over it.

Dense MLA is exact while a query row sees at most ``topk`` keys. Past that, each row attends
only the tokens the indexer scores highest. ``kpool`` tokens share a key:

- ``kpool == 1`` (GLM-5.2, DeepSeek-V3.2): every token has its own key, and a row attends its
  top ``topk`` tokens.
- ``kpool > 1`` (GLM-5.3-Flash's k-pool indexer): pools of ``kpool`` consecutive tokens from
  position 0 share one key, the gated sum of their members'. A row attends the members of its
  top ``topk / kpool`` complete pools plus the incomplete pool at its end, which the reference
  indexer always selects (``Glm5NextTextIndexer``).

The index keys live on the MLA cache's pages, in its aux planes ``INDEX_PLANES`` (one plane per
indexer layer, a row per token at the token's latent slot): ``[k_t]`` with ``kpool == 1``,
``[k_t | gate_t | key of the pool ending at t]`` with ``kpool > 1``. So one flat page list
serves the key writes, the scores, the top-k and the sparse gather. Every row selects over its
own causal prefix, so a prefill past ``topk`` is the same computation as prefilling ``topk``
tokens and decoding the rest one at a time.

On CUDA: ``kernels`` for the pool keys and the scores, flashinfer's fused top-k + page-table
transform (with a page of ``page_size / kpool`` pools, a pool's slot times ``kpool`` is the
slot of its first token) and ``sparse_mla``. Elsewhere, torch versions of the same math, which
the CPU tests hold to the references.
"""
from __future__ import annotations

import bisect
import dataclasses
from dataclasses import dataclass

import torch

from mstar.engine.resources.dsa import sparse_mla
from mstar.engine.resources.dsa.config import INDEX_PLANES

# Bytes of fp32 prefill scores per launch: a chunk of rows is as wide as its last row's pools.
PREFILL_SCORE_BYTES = 256 << 20


@dataclass
class DSAContext:
    """Per-forward DSA state, from the inputs the resource staged (``DSAManager.step_inputs``).

    Device, per step row: ``pos`` its position (keys it sees are ``[0, pos]``), ``row_req`` its
    request. Per request: ``page_start`` its first entry in ``pages``, the flat page list of
    the step's requests. Host, when the step runs eager: ``host_pos``, ``spans`` ((first row,
    rows, request) per request, in row order) and ``host_page_start``. ``max_pools`` is the
    score width (the step's most pools, or a CUDA graph's fixed width); ``sparse_plan`` the
    graph slot's plan, or the eager one the first layer builds.

    ``needs_selection``: some row sees more than ``topk`` keys; below, every row attends all it
    sees and dense MLA is the same computation. ``decode_rows``: every row scores on its own (a
    decode step's, an MTP verify block's, a captured step's), the capturable launch.
    ``last_selection``: the last indexer layer's pick, which a layer without an indexer
    (IndexShare's shared layers) reuses."""

    pos: torch.Tensor
    row_req: torch.Tensor
    pages: torch.Tensor
    page_start: torch.Tensor
    host_pos: list[int] | None
    spans: list[tuple[int, int, int]] | None
    host_page_start: list[int] | None
    max_pools: int
    page_size: int
    topk: int
    kpool: int
    needs_selection: bool = True
    decode_rows: bool = False
    sparse_plan: sparse_mla.SparseGraphPlan | sparse_mla.EagerSparsePlan | None = None
    # a graph plan made for longer rows than the step's (MTP trimmed on device): each row's
    # true length and the masked slot its spare entries attend
    valid: torch.Tensor | None = None
    pad_slot: int | None = None
    last_selection: torch.Tensor | None = None
    last_selection_layer: int | None = None

    @property
    def width(self) -> int:
        """Slots per row: the selected pools' tokens, then the tail."""
        return self.topk + self.kpool - 1

    def attn_lens(self) -> list[int]:
        return attn_lens(self.host_pos, self.topk, self.kpool)

    def rows(self, c0: int, c1: int) -> DSAContext:
        """This eager step's rows ``[c0, c1)`` alone, for a pass over them after the rows
        before them have written the cache (under the KV resource's ``token_window(c0,
        c1)``). Every row takes the sparse path: dense attention would plan the whole step."""
        spans = []
        for r0, n, req in self.spans:
            a, b = max(r0, c0), min(r0 + n, c1)
            if a < b:
                spans.append((a - c0, b - a, req))
        return self._subset(slice(c0, c1), self.host_pos[c0:c1], spans)

    def pick(self, rows: list[int]) -> DSAContext:
        """This eager step's given rows (ascending) alone, one row per span (under the KV
        resource's ``token_rows``)."""
        starts = [r0 for r0, _, _ in self.spans]
        req = [self.spans[bisect.bisect_right(starts, r) - 1][2] for r in rows]
        idx = torch.tensor(rows, device=self.pos.device)
        return self._subset(idx, [self.host_pos[r] for r in rows],
                            [(i, 1, q) for i, q in enumerate(req)])

    def _subset(self, idx, host_pos: list[int], spans: list[tuple[int, int, int]]) -> DSAContext:
        return dataclasses.replace(
            self, pos=self.pos[idx], row_req=self.row_req[idx], host_pos=host_pos, spans=spans,
            max_pools=max(1, (max(host_pos) + 1) // self.kpool), needs_selection=True,
            decode_rows=False, sparse_plan=None, last_selection=None, last_selection_layer=None)


def attn_lens(positions: list[int], topk: int, kpool: int) -> list[int]:
    """Keys each row attends: its selected pools' tokens plus its tail."""
    cap = topk // kpool
    return [kpool * min((p + 1) // kpool, cap) + (p + 1) % kpool for p in positions]


def attn_lens_device(positions: torch.Tensor, topk: int, kpool: int) -> torch.Tensor:
    """``attn_lens`` of device positions, without a sync."""
    p = positions.long() + 1
    return kpool * torch.clamp(p // kpool, max=topk // kpool) + p % kpool


# -- the step's index keys --------------------------------------------------------------------

def write_index(kv, plane: int, label: str | None, k: torch.Tensor, ctx: DSAContext,
                gate: torch.Tensor | None = None, ape: torch.Tensor | None = None) -> None:
    """Write the step's index keys ``k`` to index plane ``plane`` at the step's slots. With
    ``kpool > 1`` the rows' ``gate`` too, then the key of every pool a row completes (its
    members may sit in earlier steps)."""
    if ctx.kpool == 1:
        kv.write_aux(INDEX_PLANES, k, layer_idx=plane, label=label)
        return
    view = kv.aux_view(INDEX_PLANES, plane)
    d = k.shape[-1]
    rows = torch.cat([k, gate.to(k.dtype), k.new_zeros(k.shape[0], view.shape[-1] - 2 * d)],
                     dim=-1)
    kv.write_aux(INDEX_PLANES, rows, layer_idx=plane, label=label)
    if view.is_cuda and view.dtype == torch.bfloat16:
        from mstar.engine.resources.dsa.kernels import pool_keys

        pool_keys(view, ctx.pages, ctx.page_start, ctx.row_req, ctx.pos, ape, d)
        return
    _pool_keys_torch(view, ctx, ape, d)


def _token_slots(ctx: DSAContext, req: torch.Tensor, positions: torch.Tensor,
                 page_size: int) -> torch.Tensor:
    """Cache slots (``page * page_size + offset``) of ``positions`` of requests ``req``."""
    page = ctx.pages[(ctx.page_start[req] + positions // page_size).long()].long()
    return page * page_size + positions % page_size


def _pool_keys_torch(view: torch.Tensor, ctx: DSAContext, ape: torch.Tensor,
                     d: int) -> None:
    kp, flat = ape.shape[0], view.view(-1, view.shape[-1])
    done = (ctx.pos % kp) == kp - 1
    if not bool(done.any()):
        return
    req, last = ctx.row_req[done].long(), ctx.pos[done].long()
    members = last[:, None] - (kp - 1) + torch.arange(kp, device=last.device)
    slots = _token_slots(ctx, req[:, None], members, view.shape[1])          # (pools, KP)
    rows = flat[slots]                                                        # (pools, KP, W)
    keys, gates = rows[..., :d], rows[..., d:2 * d]
    # the reference's arithmetic: softmax in fp32, then bf16 probabilities x bf16 keys
    prob = (gates.float() + ape.float()[None]).softmax(dim=1).to(keys.dtype)
    flat[slots[:, -1], 2 * d:3 * d] = (prob * keys).sum(dim=1)


# -- selection ---------------------------------------------------------------------------------

def select(q: torch.Tensor, w: torch.Tensor, view: torch.Tensor,
           ctx: DSAContext, group=None) -> torch.Tensor:
    """``[rows, width]`` int32 MLA-cache slots each row attends: the tokens of its top ``topk /
    kpool`` pools, then its tail; entries past ``attn_lens`` are unused. ``q (rows, NH, D)``,
    ``w (rows, NH)`` fp32 with the scales folded in, ``view`` the layer's index plane
    (``kv.aux_view(INDEX_PLANES, plane)``). Single-row spans (decode) score in one launch,
    longer spans (prefill) in chunks of one request's rows; a TP ``group`` splits a prefill's
    rows across ranks (``sparse_mla.select_rows``)."""
    if ctx.decode_rows or ctx.spans is None or all(n == 1 for _, n, _ in ctx.spans):
        scores = _decode_scores(q, w, view, ctx, ctx.pos, ctx.row_req, max(ctx.max_pools, 1))
        return _slots(scores, ctx)
    out = torch.empty(q.shape[0], ctx.width, dtype=torch.int32, device=q.device)
    single = [r0 for r0, n, _ in ctx.spans if n == 1]
    if single:  # a mixed step's decode rows, in one launch
        idx = torch.tensor(single, device=q.device)
        pools = max(1, *((ctx.host_pos[r] + 1) // ctx.kpool for r in single))
        scores = _decode_scores(q[idx], w[idx], view, ctx, ctx.pos[idx], ctx.row_req[idx], pools)
        out[idx] = _slots(scores, ctx, idx)
    for r0, n, req in ctx.spans:
        if n > 1:
            out[r0:r0 + n] = sparse_mla.select_rows(
                lambda a, b, req=req: _select_rows(q, w, view, ctx, req, a, b),
                r0, r0 + n, ctx.width, q.device, group)
    return out


def _select_rows(q, w, view, ctx, req, r0, r1):
    """Rows ``[r0, r1)`` of request ``req``'s prefill, in chunks as wide as their last row's
    pools and at most ``PREFILL_SCORE_BYTES`` of scores."""
    out = torch.empty(r1 - r0, ctx.width, dtype=torch.int32, device=q.device)
    last = ctx.host_pos[r1 - 1]
    chunk = max(1, PREFILL_SCORE_BYTES // (4 * max(1, (last + 1) // ctx.kpool)))
    for c0 in range(r0, r1, chunk):
        c1 = min(c0 + chunk, r1)
        pools = max(1, (ctx.host_pos[c1 - 1] + 1) // ctx.kpool)  # causal: the last row's
        scores = _prefill_scores(q[c0:c1], w[c0:c1], view, ctx, req, c0, c1, pools)
        out[c0 - r0:c1 - r0] = _slots(scores, ctx, slice(c0, c1))
    return out


def _slots(scores: torch.Tensor, ctx: DSAContext, rows=slice(None)) -> torch.Tensor:
    """Rows ``rows`` of the step: their top pools' slots, expanded to their tokens."""
    first = top_pools(scores, ctx, rows)
    return first if ctx.kpool == 1 else expand_slots(first, ctx, rows)


def _kernel_fits(q, view) -> bool:
    # tl.dot takes 16+ rows and columns: a reduced config's 4 index heads take the torch math
    return q.is_cuda and view.dtype == torch.bfloat16 and q.shape[1] >= 16 and q.shape[2] >= 16


def _decode_scores(q, w, view, ctx, pos, row_req, pools):
    if _kernel_fits(q, view):
        from mstar.engine.resources.dsa.kernels import decode_scores

        return decode_scores(q.to(torch.bfloat16).contiguous(), w.float().contiguous(), view,
                             ctx.pages, ctx.page_start, row_req, pos, pools, ctx.kpool)
    return _scores_torch(q, w, view, ctx, row_req.long(), pos, pools)


def _prefill_scores(q, w, view, ctx, req, c0, c1, pools):
    if _kernel_fits(q, view):
        from mstar.engine.resources.dsa.kernels import prefill_scores

        return prefill_scores(q.to(torch.bfloat16).contiguous(), w.float().contiguous(), view,
                              ctx.pages, ctx.host_page_start[req], ctx.pos[c0:c1], pools,
                              ctx.kpool)
    return _scores_torch(q, w, view, ctx, ctx.row_req[c0:c1].long(), ctx.pos[c0:c1], pools)


def _scores_torch(q, w, view, ctx, req, pos, pools):
    """The indexer's score math on gathered pool keys, -inf past each row's pools."""
    d = q.shape[-1]
    key = 0 if ctx.kpool == 1 else 2 * d  # where a pool's key sits in its row (kernels.py too)
    j = torch.arange(pools, device=q.device)
    last = j * ctx.kpool + ctx.kpool - 1
    visible = j[None, :] < ((pos.long() + 1) // ctx.kpool)[:, None]
    slots = _token_slots(ctx, req[:, None], torch.where(visible, last[None, :], 0),
                         view.shape[1])
    keys = view.view(-1, view.shape[-1])[slots][..., key:key + d].float()     # (rows, pools, D)
    with torch.autocast(q.device.type, enabled=False):  # fp32 under the engine's autocast too
        dots = torch.einsum("rhd,rjd->rhj", q.float(), keys).relu()
        scores = torch.einsum("rh,rhj->rj", w.float(), dots)
    return scores.masked_fill(~visible, float("-inf"))


def top_pools(scores: torch.Tensor, ctx: DSAContext, rows=slice(None)) -> torch.Tensor:
    """``[rows, topk / kpool]`` int32: the slot of the first token of each row's top pools
    (ties to the earlier pool), -1 past ``min(pools, topk / kpool)``. With ``kpool == 1``,
    each row's top ``topk`` token slots."""
    pos, req = ctx.pos[rows], ctx.row_req[rows]
    n = ((pos + 1) // ctx.kpool).to(torch.int32)
    k = ctx.topk // ctx.kpool
    if scores.is_cuda:
        import flashinfer

        pool_slots = flashinfer.top_k_page_table_transform(
            scores, ctx.pages[None, :], n, k,
            row_to_batch=torch.zeros_like(req), deterministic=True,
            tie_break=flashinfer.TopKTieBreak.SMALL, dsa_graph_safe=True,
            page_table_row_starts=ctx.page_start[req.long()].to(torch.int32),
            page_size=ctx.page_size // ctx.kpool)
        if ctx.kpool == 1:
            return pool_slots
        return torch.where(pool_slots >= 0, pool_slots * ctx.kpool, -1)
    # stable sort: equal scores keep pool order, so ties go to the earlier pool
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices[:, :k]
    if order.shape[1] < k:
        order = torch.cat([order, order.new_zeros(order.shape[0], k - order.shape[1])], 1)
    live = torch.arange(k, device=scores.device)[None, :] < n[:, None]
    first = _token_slots(ctx, req[:, None].long(), torch.where(live, order * ctx.kpool, 0),
                         ctx.page_size)
    return torch.where(live, first, -1).to(torch.int32)


def expand_slots(pool_first: torch.Tensor, ctx: DSAContext, rows=slice(None)) -> torch.Tensor:
    """Each row's slot list: its pools' tokens, then the tail ``[kpool * pools, pos]``, packed
    from entry 0 (``attn_lens`` long). Device ops only, so a captured decode replays it."""
    if pool_first.is_cuda:
        from mstar.engine.resources.dsa.kernels import expand_slots as expand

        return expand(pool_first.contiguous(), ctx.pos[rows], ctx.row_req[rows], ctx.pages,
                      ctx.page_start, ctx.kpool, ctx.page_size, ctx.width)
    pos, req = ctx.pos[rows].long(), ctx.row_req[rows].long()
    kp, k = ctx.kpool, ctx.topk // ctx.kpool
    m = torch.arange(kp, device=pos.device)
    tokens = torch.where(pool_first[..., None] >= 0, pool_first[..., None] + m, -1)
    out = torch.full((pos.shape[0], ctx.width + 1), -1, dtype=torch.int32, device=pos.device)
    out[:, :k * kp] = tokens.flatten(1)
    pools = (pos + 1) // kp
    tail = m[:kp - 1]
    tail_pos = pools[:, None] * kp + tail
    live = tail[None, :] < ((pos + 1) % kp)[:, None]
    slots = _token_slots(ctx, req[:, None], torch.where(live, tail_pos, 0), ctx.page_size)
    col = torch.where(live, torch.clamp(pools, max=k)[:, None] * kp + tail, ctx.width)
    out.scatter_(1, col, torch.where(live, slots, -1).to(torch.int32))
    return out[:, :ctx.width]


# -- sparse MLA --------------------------------------------------------------------------------

def sparse_attend(q_nope: torch.Tensor, q_pe: torch.Tensor, latent: torch.Tensor,
                  slots: torch.Tensor, ctx: DSAContext, sm_scale: float) -> torch.Tensor:
    """Absorbed MLA of each row over its slots: ``[rows, heads, kv_lora_rank]``. ``latent`` is
    the layer's latent plane ``[pages, page_size, kv_lora_rank + kpe]``."""
    if not sparse_mla.uses_kernel(q_nope, latent):
        lens = ctx.attn_lens() if ctx.host_pos is not None else attn_lens(
            ctx.pos.tolist(), ctx.topk, ctx.kpool)
        return sparse_mla.reference(q_nope, q_pe, latent, slots, lens, sm_scale)
    if ctx.sparse_plan is None:  # eager: every layer of the forward shares one plan
        ctx.sparse_plan = sparse_mla.EagerSparsePlan(
            ctx.attn_lens(), ctx.width, q_nope.shape[1], q_nope.shape[-1], q_pe.shape[-1],
            sm_scale, q_nope.device)
    if ctx.valid is not None:
        return ctx.sparse_plan.attend(q_nope, q_pe, latent, slots, ctx.valid, ctx.pad_slot)
    return ctx.sparse_plan.attend(q_nope, q_pe, latent, slots)
