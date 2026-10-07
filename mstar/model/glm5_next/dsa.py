"""GLM-5.3-Flash DSA past index_topk: the k-pool indexer on device and sparse MLA.

Dense MLA is exact while a query sees at most ``index_topk / index_kpool`` complete pools. Past
that, each query attends to the members of its top ``index_topk / index_kpool`` complete pools
(``index_kpool`` consecutive tokens each, from position 0) plus the incomplete pool at its end,
which the reference indexer always selects (``Glm5NextTextIndexer``).

The indexer's state lives in the MLA cache, in one index plane per full-attention layer next to
the latent planes: row ``t`` holds ``[k_t | gate_t | key of the pool ending at t | unused]``.
Both kinds of plane share the request's pages, so one flat page list serves the key writes,
the scores, the top-k and the sparse gather.

On CUDA: ``dsa_kernels`` for the pool keys and the scores, flashinfer's fused top-k + page-table
transform (with a page of ``page_size / index_kpool`` pools, a pool's slot times ``index_kpool``
is the slot of its first token) and ``sparse_mla``. Elsewhere, torch versions of the same math,
which the CPU tests hold to the reference.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from mstar.engine.resources.attn import sparse_mla

# Bytes of fp32 prefill scores per launch: a chunk of rows is as wide as its last row's pools.
PREFILL_SCORE_BYTES = 256 << 20


@dataclass
class Glm5NextDsaContext:
    """Per-forward DSA state, built from the submodule's preprocess outputs.

    Device, per step row: ``pos`` its position (keys it sees are ``[0, pos]``), ``row_req`` its
    request. Per request: ``page_start`` its first entry in ``pages``, the flat page list of
    the step's requests. Host, when the step runs eager: ``host_pos``, ``spans`` ((first row,
    rows, request) per request) and ``host_page_start``. ``max_pools`` is the score width (the
    step's most pools, or a CUDA graph's fixed width); ``sparse_plan`` the graph slot's plan, or
    the eager one the first layer builds."""

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
    sparse_plan: sparse_mla.SparseGraphPlan | sparse_mla.EagerSparsePlan | None = None

    @property
    def width(self) -> int:
        """Slots per row: the selected pools' tokens, then the tail."""
        return self.topk + self.kpool - 1

    def attn_lens(self) -> list[int]:
        return attn_lens(self.host_pos, self.topk, self.kpool)

    def window(self, start: int, end: int,
               pieces: list[tuple[int, int, int]]) -> Glm5NextDsaContext:
        """This eager step's rows ``[start, end)``: ``pieces`` are (first row, end row,
        request) of each request's rows in the window."""
        host_pos = self.host_pos[start:end]
        return dataclasses.replace(
            self, pos=self.pos[start:end], row_req=self.row_req[start:end], host_pos=host_pos,
            spans=[(a - start, b - a, req) for a, b, req in pieces],
            max_pools=(max(host_pos) + 1) // self.kpool, sparse_plan=None)


class DsaState:
    """The submodule's handle on the forward's DSA context, shared by the MLA layers."""

    def __init__(self) -> None:
        self.ctx: Glm5NextDsaContext | None = None


def attn_lens(positions: list[int], topk: int, kpool: int) -> list[int]:
    """Keys each row attends: its selected pools' tokens plus its tail."""
    cap = topk // kpool
    return [kpool * min((p + 1) // kpool, cap) + (p + 1) % kpool for p in positions]


# -- the step's index keys --------------------------------------------------------------------

def write_index(kv, plane: int, label: str, k: torch.Tensor, gate: torch.Tensor,
                ape: torch.Tensor, ctx: Glm5NextDsaContext) -> None:
    """Write the step's ``k`` and ``gate`` rows to the index plane, then the key of every pool
    a row completes (its members may sit in earlier steps)."""
    width = kv.layer_view(plane).shape[-1]
    d = k.shape[-1]
    rows = torch.cat([k, gate.to(k.dtype), k.new_zeros(k.shape[0], width - 2 * d)], dim=-1)
    kv.write_kv(rows, None, layer_idx=plane, label=label)
    view = kv.layer_view(plane)
    if view.is_cuda and view.dtype == torch.bfloat16:
        from mstar.model.glm5_next.dsa_kernels import pool_keys

        pool_keys(view, ctx.pages, ctx.page_start, ctx.row_req, ctx.pos, ape, d)
        return
    _pool_keys_torch(view, ctx, ape, d)


def _token_slots(ctx: Glm5NextDsaContext, req: torch.Tensor, positions: torch.Tensor,
                 page_size: int) -> torch.Tensor:
    """Cache slots (``page * page_size + offset``) of ``positions`` of requests ``req``."""
    page = ctx.pages[(ctx.page_start[req] + positions // page_size).long()].long()
    return page * page_size + positions % page_size


def _pool_keys_torch(view: torch.Tensor, ctx: Glm5NextDsaContext, ape: torch.Tensor,
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
           ctx: Glm5NextDsaContext, group=None) -> torch.Tensor:
    """``[rows, width]`` int32 MLA-cache slots each row attends: the tokens of its top ``topk /
    kpool`` pools, then its tail; entries past ``attn_lens`` are unused. ``q (rows, NH, D)``,
    ``w (rows, NH)`` fp32 with the scales folded in, ``view`` the layer's index plane. A TP
    ``group`` splits a prefill's rows across ranks (``sparse_mla.select_rows``)."""
    if ctx.spans is None or all(n == 1 for _, n, _ in ctx.spans):  # decode: capturable
        scores = _decode_scores(q, w, view, ctx)
        return expand_slots(top_pools(scores, ctx), ctx)
    out = torch.empty(q.shape[0], ctx.width, dtype=torch.int32, device=q.device)
    for r0, n, req in ctx.spans:
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
        out[c0 - r0:c1 - r0] = expand_slots(top_pools(scores, ctx, c0, c1), ctx, c0, c1)
    return out


def _decode_scores(q, w, view, ctx):
    if q.is_cuda and view.dtype == torch.bfloat16:
        from mstar.model.glm5_next.dsa_kernels import decode_scores

        return decode_scores(q.to(torch.bfloat16).contiguous(), w.float().contiguous(), view,
                             ctx.pages, ctx.page_start, ctx.row_req, ctx.pos,
                             max(ctx.max_pools, 1), ctx.kpool)
    return _scores_torch(q, w, view, ctx, ctx.row_req.long(), ctx.pos, max(ctx.max_pools, 1))


def _prefill_scores(q, w, view, ctx, req, c0, c1, pools):
    if q.is_cuda and view.dtype == torch.bfloat16:
        from mstar.model.glm5_next.dsa_kernels import prefill_scores

        return prefill_scores(q.to(torch.bfloat16).contiguous(), w.float().contiguous(), view,
                              ctx.pages, ctx.host_page_start[req], ctx.pos[c0:c1], pools,
                              ctx.kpool)
    return _scores_torch(q, w, view, ctx, ctx.row_req[c0:c1].long(), ctx.pos[c0:c1], pools)


def _scores_torch(q, w, view, ctx, req, pos, pools):
    """The indexer's score math on gathered pool keys, -inf past each row's pools."""
    d = q.shape[-1]
    j = torch.arange(pools, device=q.device)
    last = j * ctx.kpool + ctx.kpool - 1
    visible = j[None, :] < ((pos.long() + 1) // ctx.kpool)[:, None]
    slots = _token_slots(ctx, req[:, None], torch.where(visible, last[None, :], 0),
                         view.shape[1])
    keys = view.view(-1, view.shape[-1])[slots][..., 2 * d:3 * d].float()      # (rows, pools, D)
    with torch.autocast(q.device.type, enabled=False):  # fp32 under the engine's autocast too
        dots = torch.einsum("rhd,rjd->rhj", q.float(), keys).relu()
        scores = torch.einsum("rh,rhj->rj", w.float(), dots)
    return scores.masked_fill(~visible, float("-inf"))


def top_pools(scores: torch.Tensor, ctx: Glm5NextDsaContext, c0: int = 0,
              c1: int | None = None) -> torch.Tensor:
    """``[rows, topk / kpool]`` int32: the slot of the first token of each row's top pools
    (ties to the earlier pool), -1 past ``min(pools, topk / kpool)``."""
    rows = slice(c0, c1)
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
        return torch.where(pool_slots >= 0, pool_slots * ctx.kpool, -1)
    # stable sort: equal scores keep pool order, so ties go to the earlier pool
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices[:, :k]
    if order.shape[1] < k:
        order = torch.cat([order, order.new_zeros(order.shape[0], k - order.shape[1])], 1)
    live = torch.arange(k, device=scores.device)[None, :] < n[:, None]
    first = _token_slots(ctx, req[:, None].long(), torch.where(live, order * ctx.kpool, 0),
                         ctx.page_size)
    return torch.where(live, first, -1).to(torch.int32)


def expand_slots(pool_first: torch.Tensor, ctx: Glm5NextDsaContext, c0: int = 0,
                 c1: int | None = None) -> torch.Tensor:
    """Each row's slot list: its pools' tokens, then the tail ``[kpool * pools, pos]``, packed
    from entry 0 (``attn_lens`` long). Device ops only, so a captured decode replays it."""
    rows = slice(c0, c1)
    if pool_first.is_cuda:
        from mstar.model.glm5_next.dsa_kernels import expand_slots as expand

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
                  slots: torch.Tensor, ctx: Glm5NextDsaContext, sm_scale: float) -> torch.Tensor:
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
    return ctx.sparse_plan.attend(q_nope, q_pe, latent, slots)
