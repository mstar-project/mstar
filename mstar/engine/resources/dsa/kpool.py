"""Pooled-key DSA (GLM-5.3-Flash's k-pool indexer) past index_topk, and its resource.

Dense MLA is exact while a query sees at most ``index_topk / index_kpool`` complete pools. Past
that, each query attends to the members of its top ``index_topk / index_kpool`` complete pools
(``index_kpool`` consecutive tokens each, from position 0) plus the incomplete pool at its end,
which the reference indexer always selects (``Glm5NextTextIndexer``).

The indexer's state lives in the MLA cache, in one index plane per full-attention layer next to
the latent planes: row ``t`` holds ``[k_t | gate_t | key of the pool ending at t | unused]``.
Both kinds of plane share the request's pages, so one flat page list serves the key writes,
the scores, the top-k and the sparse gather.

On CUDA: ``kpool_kernels`` for the pool keys and the scores, flashinfer's fused top-k + page-table
transform (with a page of ``page_size / index_kpool`` pools, a pool's slot times ``index_kpool``
is the slot of its first token) and ``sparse_mla``. Elsewhere, torch versions of the same math,
which the CPU tests hold to the reference.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.dsa import sparse_mla
from mstar.engine.resources.dsa.config import DSAConfig, DSASpec
from mstar.engine.resources.step import StepContext
from mstar.utils.h2d import PinnedStager

# Bytes of fp32 prefill scores per launch: a chunk of rows is as wide as its last row's pools.
PREFILL_SCORE_BYTES = 256 << 20


@dataclass
class KPoolContext:
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
    # a graph plan made for longer rows than the step's (MTP trimmed on device): each row's
    # true length and the masked slot its spare entries attend
    valid: torch.Tensor | None = None
    pad_slot: int | None = None

    @property
    def width(self) -> int:
        """Slots per row: the selected pools' tokens, then the tail."""
        return self.topk + self.kpool - 1

    def attn_lens(self) -> list[int]:
        return attn_lens(self.host_pos, self.topk, self.kpool)

    def window(self, start: int, end: int,
               pieces: list[tuple[int, int, int]]) -> KPoolContext:
        """This eager step's rows ``[start, end)``: ``pieces`` are (first row, end row,
        request) of each request's rows in the window."""
        host_pos = self.host_pos[start:end]
        return dataclasses.replace(
            self, pos=self.pos[start:end], row_req=self.row_req[start:end], host_pos=host_pos,
            spans=[(a - start, b - a, req) for a, b, req in pieces],
            max_pools=(max(host_pos) + 1) // self.kpool, sparse_plan=None)


def attn_lens(positions: list[int], topk: int, kpool: int) -> list[int]:
    """Keys each row attends: its selected pools' tokens plus its tail."""
    cap = topk // kpool
    return [kpool * min((p + 1) // kpool, cap) + (p + 1) % kpool for p in positions]


def attn_lens_device(positions: torch.Tensor, topk: int, kpool: int) -> torch.Tensor:
    """``attn_lens`` of device positions, without a sync."""
    p = positions.long() + 1
    return kpool * torch.clamp(p // kpool, max=topk // kpool) + p % kpool


# -- the step's index keys --------------------------------------------------------------------

def write_index(kv, plane: int, label: str, k: torch.Tensor, gate: torch.Tensor,
                ape: torch.Tensor, ctx: KPoolContext) -> None:
    """Write the step's ``k`` and ``gate`` rows to the index plane, then the key of every pool
    a row completes (its members may sit in earlier steps)."""
    width = kv.layer_view(plane).shape[-1]
    d = k.shape[-1]
    rows = torch.cat([k, gate.to(k.dtype), k.new_zeros(k.shape[0], width - 2 * d)], dim=-1)
    kv.write_kv(rows, None, layer_idx=plane, label=label)
    view = kv.layer_view(plane)
    if view.is_cuda and view.dtype == torch.bfloat16:
        from mstar.engine.resources.dsa.kpool_kernels import pool_keys

        pool_keys(view, ctx.pages, ctx.page_start, ctx.row_req, ctx.pos, ape, d)
        return
    _pool_keys_torch(view, ctx, ape, d)


def _token_slots(ctx: KPoolContext, req: torch.Tensor, positions: torch.Tensor,
                 page_size: int) -> torch.Tensor:
    """Cache slots (``page * page_size + offset``) of ``positions`` of requests ``req``."""
    page = ctx.pages[(ctx.page_start[req] + positions // page_size).long()].long()
    return page * page_size + positions % page_size


def _pool_keys_torch(view: torch.Tensor, ctx: KPoolContext, ape: torch.Tensor,
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
           ctx: KPoolContext, group=None) -> torch.Tensor:
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


def _kernel_fits(q, view) -> bool:
    # tl.dot takes 16+ rows and columns: a reduced config's 4 index heads take the torch math
    return q.is_cuda and view.dtype == torch.bfloat16 and q.shape[1] >= 16 and q.shape[2] >= 16


def _decode_scores(q, w, view, ctx):
    if _kernel_fits(q, view):
        from mstar.engine.resources.dsa.kpool_kernels import decode_scores

        return decode_scores(q.to(torch.bfloat16).contiguous(), w.float().contiguous(), view,
                             ctx.pages, ctx.page_start, ctx.row_req, ctx.pos,
                             max(ctx.max_pools, 1), ctx.kpool)
    return _scores_torch(q, w, view, ctx, ctx.row_req.long(), ctx.pos, max(ctx.max_pools, 1))


def _prefill_scores(q, w, view, ctx, req, c0, c1, pools):
    if _kernel_fits(q, view):
        from mstar.engine.resources.dsa.kpool_kernels import prefill_scores

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


def top_pools(scores: torch.Tensor, ctx: KPoolContext, c0: int = 0,
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


def expand_slots(pool_first: torch.Tensor, ctx: KPoolContext, c0: int = 0,
                 c1: int | None = None) -> torch.Tensor:
    """Each row's slot list: its pools' tokens, then the tail ``[kpool * pools, pos]``, packed
    from entry 0 (``attn_lens`` long). Device ops only, so a captured decode replays it."""
    rows = slice(c0, c1)
    if pool_first.is_cuda:
        from mstar.engine.resources.dsa.kpool_kernels import expand_slots as expand

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
                  slots: torch.Tensor, ctx: KPoolContext, sm_scale: float) -> torch.Tensor:
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


# -- the resource ------------------------------------------------------------------------------

class KPoolDSA(Resource):
    """Pooled-key DSA over a node's MLA cache (``config.kv_cache``), whose index planes hold the
    indexer's keys. Each step the submodule stages its rows (``step_inputs``: preprocess outputs,
    which a CUDA graph interns and every replay restages) and sets the forward's context built
    from them (``context``) as ``ctx``, which the MLA layers read; under CUDA graphs the resource
    keeps each capture slot's sparse-attention plans."""

    @classmethod
    def build(cls, spec: DSASpec, info: EngineResourceInfo):
        # `dependency` hands back the SPEC; its `.config` is the KV config, sharded once
        kv_config = info.dependency(spec.config.kv_cache).config
        if info.joint_comm_group is not None:
            kv_config.shard(info.joint_comm_group.world_size)
        return cls(spec.config, kv_config, info.device)

    def __init__(self, config: DSAConfig, kv_config, device: torch.device):
        self.config = config
        self.page_size = kv_config.page_size
        self.kv_rows = kv_config.max_seq_len  # cache rows a request may hold
        self.num_heads = kv_config.num_qo_heads  # this rank's
        self.kv_lora_rank = kv_config.kv_lora_rank
        self.kpe = kv_config.qk_rope_head_dim
        self._device = device
        # the forward's context, open while its layers run
        self.ctx: KPoolContext | None = None
        self._int_stager = PinnedStager(torch.int32)
        # under CUDA graphs one sparse-attention plan per (walk, capture slot, rows), and per
        # (walk, capture slot) the workspace and index buffer its plans share
        self._sparse_plans: dict[tuple[str, int, int], sparse_mla.SparseGraphPlan] = {}
        self._sparse_buffers: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

    def step_inputs(self, step_ctx: StepContext, captured: bool, label: str,
                    mtp_trims: bool = False) -> dict[str, Any]:
        """The step's DSA state from the KV plan, as preprocess outputs. The tensors are top
        level, so a CUDA graph interns them and each replay stages the step's values: per row
        its position and request, per request its first entry in the flat page list. Under a
        graph the score width spans the serving window and the sparse plan is the slot's own,
        re-planned here every step; the host half rides in ``dsa_meta``. ``mtp_trims``: a decode
        step's rows may sit past their true positions (the submodule trims rejected drafts on
        device)."""
        graph = bool(captured)
        walk = step_ctx.graph_walk
        kv_out = step_ctx.plan_results[self.config.kv_cache][label]
        topk, kpool = self.config.topk, self.config.kpool
        pos: list[int] = []
        row_req: list[int] = []
        spans: list[tuple[int, int, int]] = []
        for i, view in enumerate(kv_out.views):
            spans.append((len(pos), view.to_compute, i))
            pos.extend(range(view.length - view.to_compute, view.length))
            row_req.extend([i] * view.to_compute)
        n_real = len(pos)
        rows = n_real
        if graph and step_ctx.slot_lease is not None:
            # a packed prefill bucket's padding rows: position 0 of the first request. Staged, so
            # they don't keep what a larger bucket left in the shared static buffers
            rows = step_ctx.slot_lease.bucket.num_tokens
            pos += [0] * (rows - n_real)
            row_req += [0] * (rows - n_real)
        indptr = kv_out.cpu_indptrs.paged_kv_indptr
        parts = [torch.tensor(pos + row_req, dtype=torch.int32), indptr[:-1].to(torch.int32),
                 kv_out.cpu_indptrs.paged_kv_indices.to(torch.int32)]
        if step_ctx.capture:
            # the static buffer's size: every row's request at the serving window (a replay
            # stages just its own pages)
            pad = len(kv_out.views) * -(-self.kv_rows // self.page_size) - parts[-1].numel()
            parts.append(torch.zeros(pad, dtype=torch.int32))
        n_pos, n_req = len(pos), indptr.numel() - 1
        staged = self._int_stager.to_device(torch.cat(parts), self._device)
        lens = attn_lens(pos, topk, kpool)  # a padding row's is 1
        if mtp_trims and walk == "decode":
            # these rows sit at or past their true ones (_trim_on_device): plan each for the
            # longest it can be there, its spare entries masked
            lens = [min(p + 1, topk + kpool - 1) for p in pos]
        # score width: the serving window's pools for a decode graph; a captured prefill holds
        # whole prompts no longer than its bucket
        max_pools = (max(pos) + 1) // kpool if pos else 1
        sparse_plan = None
        if graph:
            if walk == "prefill":
                if pos and max(pos) >= rows:
                    raise RuntimeError(f"a captured prefill row sits at {max(pos)}, past its "
                                       f"{rows}-token bucket")
                max_pools = rows // kpool
            else:
                max_pools = self.kv_rows // kpool
        if graph and self._device.type == "cuda":
            width = topk + kpool - 1
            key = (walk, step_ctx.slot, rows)
            sparse_plan = self._sparse_plans.get(key)
            if sparse_plan is None:
                buffers = self._sparse_buffers.get(key[:2])
                if buffers is None or buffers[1].numel() < rows * width + 1:
                    # captures run largest first, so the first is the slot's size; +1: the
                    # plan's spare index entry
                    buffers = self._sparse_buffers[key[:2]] = (
                        torch.empty(sparse_mla.WORKSPACE_BYTES, dtype=torch.uint8,
                                    device=self._device),
                        torch.zeros(rows * width + 1, dtype=torch.int32, device=self._device))
                sparse_plan = self._sparse_plans[key] = sparse_mla.SparseGraphPlan(
                    rows, width, buffers[0], buffers[1])
            sparse_plan.plan(lens, self.num_heads, self.kv_lora_rank, self.kpe,
                             self.config.sm_scale)
        return {
            "dsa_pos": staged[:n_pos],
            "dsa_row_req": staged[n_pos:2 * n_pos],
            "dsa_page_start": staged[2 * n_pos:2 * n_pos + n_req],
            "dsa_pages": staged[2 * n_pos + n_req:],
            "dsa_meta": dict(
                host_pos=None if graph else pos[:n_real],
                spans=None if graph else spans,
                host_page_start=None if graph else indptr[:-1].tolist(),
                max_pools=max(max_pools, 1),
                page_size=self.page_size, sparse_plan=sparse_plan,
            ),
        }

    def context(self, inputs: dict[str, Any]) -> KPoolContext | None:
        """The forward's context from ``step_inputs``' outputs, or None without them; the
        submodule sets it as ``ctx`` while the layers run."""
        meta = inputs.get("dsa_meta")
        if meta is None:
            return None
        return KPoolContext(
            pos=inputs["dsa_pos"], row_req=inputs["dsa_row_req"], pages=inputs["dsa_pages"],
            page_start=inputs["dsa_page_start"], host_pos=meta["host_pos"],
            spans=meta["spans"], host_page_start=meta["host_page_start"],
            max_pools=meta["max_pools"], page_size=meta["page_size"],
            topk=self.config.topk, kpool=self.config.kpool,
            sparse_plan=meta["sparse_plan"])
