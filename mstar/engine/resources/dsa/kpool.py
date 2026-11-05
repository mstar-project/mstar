"""Pooled-key DSA (GLM-5.3-Flash's k-pool indexer) and its resource.

Each query attends to the members of its top ``topk / kpool`` complete pools (``kpool``
consecutive tokens from position 0) plus the incomplete pool at its end, which the reference
indexer always selects. A query that sees no more pools than that attends everything, where
dense MLA is the same computation.

Row ``t`` of the cache's aux planes ``INDEX_PLANES`` holds ``[k_t | gate_t | key of the pool
ending at t]`` at token t's latent slot, so one page list addresses the key writes, the scores,
the top-k and the sparse gather. On CUDA the top-k is FlashInfer's fused top-k + page-table
transform over pages of ``page_size / kpool`` pools: a pool's slot times ``kpool`` is its first
token's. Elsewhere torch versions of the same math run.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo, Resource
from mstar.engine.resources.dsa import sparse_mla
from mstar.engine.resources.dsa.config import INDEX_PLANES, DSAConfig, DSASpec, DSAStep
from mstar.engine.resources.step import BucketKey, StepContext
from mstar.utils.h2d import PinnedStager

_MAIN = "main"

# Bytes of fp32 prefill scores per launch: a chunk of rows is as wide as its last row's pools.
PREFILL_SCORE_BYTES = 256 << 20


@dataclass
class KPoolContext:
    """One step's DSA rows, as ``KPoolDSA.plan`` builds them from the KV plan.

    Per row: ``pos`` (it sees keys ``[0, pos]``) and ``row_req``. Per request: ``page_start``, its
    first entry in the flat page list ``pages``. The host fields are set only for an eager step;
    ``spans`` is (first row, rows, request) per request. ``max_pools`` is the score width, fixed
    under a graph; ``sparse_plan`` is the graph slot's, or the eager one the first layer builds."""

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


# -- the step's index keys --------------------------------------------------------------------

def write_index(kv, plane: int, label: str, k: torch.Tensor, gate: torch.Tensor,
                ape: torch.Tensor, ctx: KPoolContext) -> None:
    """Write the step's ``k`` and ``gate`` rows to index plane ``plane``, then the key of every
    pool a row completes (its members may sit in earlier steps)."""
    view = kv.aux_view(INDEX_PLANES, plane)
    d = k.shape[-1]
    rows = torch.cat([k, gate.to(k.dtype), k.new_zeros(k.shape[0], view.shape[-1] - 2 * d)],
                     dim=-1)
    kv.write_aux(INDEX_PLANES, rows, layer_idx=plane, label=label)
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
    """``[rows, width]`` int32 cache slots each row attends: its top pools' tokens, then its
    tail; entries past ``attn_lens`` are unused. ``w (rows, NH)`` is fp32 with the scales
    folded in, ``view`` the layer's index plane. A TP ``group`` splits a prefill's rows."""
    if ctx.spans is None or all(n == 1 for _, n, _ in ctx.spans):  # decode: capturable
        scores = _decode_scores(q, w, view, ctx)
        return expand_slots(top_pools(scores, ctx), ctx)
    out = torch.empty(q.shape[0], ctx.width, dtype=torch.int32, device=q.device)
    for r0, n, req in ctx.spans:
        out[r0:r0 + n] = sparse_mla.select_rows(
            lambda a, b, req=req: _select_rows(q, w, view, ctx, req, a, b),
            r0, r0 + n, ctx.width, q.device, group, _request_pages(ctx, req))
    return out


def _request_pages(ctx: KPoolContext, req: int) -> sparse_mla.RequestPages:
    starts = ctx.host_page_start
    end = starts[req + 1] if req + 1 < len(starts) else ctx.pages.numel()
    return sparse_mla.RequestPages(ctx.pages[starts[req]:end], ctx.page_size)


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
    return ctx.sparse_plan.attend(q_nope, q_pe, latent, slots)


# -- the resource ------------------------------------------------------------------------------

class KPoolDSA(Resource):
    """Pooled-key DSA over the MLA cache ``config.kv_cache``, whose aux planes ``INDEX_PLANES``
    hold the index keys. A step declares ``DSAStep()``; ``plan`` builds each label's rows from
    the KV plan, and the MLA layers write keys, select and attend through the resource
    (``write_index``, ``select``, ``attend``), which reads the current plan. Under a capture
    lease the rows sit in buffers of the (bucket, slot) that the graph reads and every plan
    refills, beside the slot's sparse-attention plan."""

    @classmethod
    def build(cls, spec: DSASpec, info: EngineResourceInfo):
        # `dependency` hands back the SPEC; its `.config` is the KV config (shard is idempotent)
        kv_config = info.dependency(spec.config.kv_cache).config
        if info.joint_comm_group is not None:
            kv_config.shard(info.joint_comm_group.world_size)
        return cls(spec.config, kv_config, info.device)

    def __init__(self, config: DSAConfig, kv_config, device: torch.device):
        self.config = config
        self.page_size = kv_config.tokens_per_page
        self.kv_rows = kv_config.max_seq_len  # cache rows a request may hold
        self.num_heads = kv_config.num_qo_heads  # this rank's
        self.kv_lora_rank = kv_config.kv_lora_rank
        self.kpe = kv_config.qk_rope_head_dim
        self._device = device
        self._kv = None  # the KV resource whose pages it reads (`bind_dependencies`)
        # label -> this step's plan, which the layers read
        self._current: dict[str, KPoolContext] = {}
        # pre-planning: a plan made a step ahead, which the next `plan` promotes
        self._preplanned: dict[str, KPoolContext] | None = None
        self._int_stager = PinnedStager(torch.int32)
        # (bucket, slot) -> the int32 rows a captured graph reads: pos, row_req, page_start, pages
        self._cg_rows: dict[tuple[BucketKey, int], torch.Tensor] = {}
        # (walk, capture slot, rows) -> its sparse-attention plan under CUDA graphs
        self._sparse_plans: dict[tuple[str, int, int], sparse_mla.SparseGraphPlan] = {}
        # (walk, capture slot) -> the workspace and index buffer its plans share
        self._sparse_buffers: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

    def bind_dependencies(self, resources: Mapping[str, Resource]) -> None:
        self._kv = resources[self.config.kv_cache]

    @property
    def force_double_buffer(self) -> bool:
        # a captured slot's sparse plan is re-planned in `plan` through FlashInfer's pinned
        # staging, which the replay before it may still be reading
        return True

    @property
    def supports_preplan(self) -> bool:
        return True

    def clear_preplan(self) -> None:
        self._preplanned = None

    # -- capture buffers ---------------------------------------------------------------------

    def _rows_layout(self, bucket: BucketKey) -> tuple[int, int, int]:
        """A captured (bucket)'s rows, requests and page entries: every request at the window."""
        return bucket.num_tokens, bucket.bs, bucket.bs * -(-self.kv_rows // self.page_size)

    def _cg_buffer(self, bucket: BucketKey, slot: int) -> torch.Tensor:
        key = (bucket, slot)
        buf = self._cg_rows.get(key)
        if buf is None:
            rows, reqs, pages = self._rows_layout(bucket)
            buf = self._cg_rows[key] = torch.zeros(
                2 * rows + reqs + pages, dtype=torch.int32, device=self._device)
        return buf

    def build_accelerator_graph_buffers(self, slots: list[CGSlotSpec], max_bs: int,
                                        max_seq_len: int) -> None:
        # a captured step plans its bucket's rows as one sparse plan, which FlashInfer takes up
        # to sparse_mla.MAX_ROWS; refused here whichever ladder the buckets came from
        past = sorted({spec.num_tokens for spec in slots if spec.num_tokens > sparse_mla.MAX_ROWS})
        if past:
            raise ValueError(f"DSA: captured buckets of {past} tokens pass one sparse plan's "
                             f"{sparse_mla.MAX_ROWS} rows; cap the node's prefill ladder there")
        for spec in slots:
            self._cg_buffer(spec.bucket, spec.slot)

    # -- the step ----------------------------------------------------------------------------

    def plan(self, step: DSAStep, ctx: StepContext) -> dict[str, KPoolContext]:
        """Each label's rows from the KV plan (``ctx.plan_results``), the plan the layers read
        until the next. A pre-plan is kept and the next call promotes it."""
        del step
        if self._preplanned is not None and not ctx.is_preplan:
            self._current, self._preplanned = self._preplanned, None
            return self._current
        plans = {label: self._plan_label(ctx, label, kv_out)
                 for label, kv_out in ctx.plan_results[self.config.kv_cache].items()}
        if ctx.is_preplan:
            self._preplanned = plans
        else:
            self._current = plans
        return plans

    def _plan_label(self, step_ctx: StepContext, label: str, kv_out) -> KPoolContext:
        lease = step_ctx.slot_lease
        graph = lease is not None and lease.bucket is not None
        walk = step_ctx.graph_walk
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
        indptr = kv_out.cpu_indptrs.paged_kv_indptr
        n_req = indptr.numel() - 1
        page_list = kv_out.cpu_indptrs.paged_kv_indices.to(torch.int32)
        if graph:
            rows, reqs, page_cap = self._rows_layout(lease.bucket)
            if n_real > rows or n_req > reqs or page_list.numel() > page_cap:
                raise RuntimeError(f"dsa: {n_real} rows of {n_req} requests ({page_list.numel()} "
                                   f"pages) overflow the {lease.bucket} capture's buffers")
            # a packed bucket's padding rows: position 0 of the first request
            pos += [0] * (rows - n_real)
            row_req += [0] * (rows - n_real)
        else:
            reqs = n_req
        host = torch.cat([torch.tensor(pos + row_req, dtype=torch.int32),
                          indptr[:-1].to(torch.int32),
                          torch.zeros(reqs - n_req, dtype=torch.int32), page_list]).numpy()
        if graph:
            staged = self._cg_buffer(lease.bucket, lease.slot)
            self._int_stager.copy_(staged, host)
        else:
            staged = torch.empty(host.size, dtype=torch.int32, device=self._device)
            self._int_stager.copy_(staged, host)
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
            if self._device.type == "cuda":
                sparse_plan = self._graph_plan(walk, step_ctx.slot, rows,
                                               attn_lens(pos, topk, kpool))
        return KPoolContext(
            pos=staged[:rows], row_req=staged[rows:2 * rows],
            page_start=staged[2 * rows:2 * rows + n_req], pages=staged[2 * rows + reqs:],
            host_pos=None if graph else pos[:n_real], spans=None if graph else spans,
            host_page_start=None if graph else indptr[:-1].tolist(),
            max_pools=max(max_pools, 1), page_size=self.page_size, topk=topk, kpool=kpool,
            sparse_plan=sparse_plan)

    def _graph_plan(self, walk: str, slot: int, rows: int,
                    lens: list[int]) -> sparse_mla.SparseGraphPlan:
        """The capture slot's sparse plan for ``rows`` rows, re-planned with ``lens``."""
        width = self.config.topk + self.config.kpool - 1
        key = (walk, slot, rows)
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
        sparse_plan.plan(lens, self.num_heads, self.kv_lora_rank, self.kpe, self.config.sm_scale)
        return sparse_plan

    # -- what the layers call ----------------------------------------------------------------

    def current(self, label: str = _MAIN) -> KPoolContext:
        """This step's plan for ``label``."""
        found = self._current.get(label)
        if found is None:
            raise KeyError(f"dsa: no plan for label {label!r}; the step declares DSAStep()")
        return found

    @contextmanager
    def rows(self, start: int, end: int, pieces: list[tuple[int, int, int]],
             label: str = _MAIN) -> Iterator[None]:
        """Narrow this eager step's plan to its rows ``[start, end)`` (``pieces``: each
        request's (first row, end row, request) in them), for a forward run in windows."""
        full = self.current(label)
        self._current[label] = full.window(start, end, pieces)
        try:
            yield
        finally:
            self._current[label] = full

    def _need_kv(self):
        if self._kv is None:
            raise RuntimeError(f"dsa: bind_dependencies never handed it {self.config.kv_cache}")
        return self._kv

    def write_index(self, plane: int, k: torch.Tensor, gate: torch.Tensor, ape: torch.Tensor,
                    label: str = _MAIN) -> None:
        """Write the step's index keys and gates to ``plane``, and the keys of the pools they
        complete."""
        write_index(self._need_kv(), plane, label, k, gate, ape, self.current(label))

    def select(self, plane: int, q: torch.Tensor, w: torch.Tensor, group=None,
               label: str = _MAIN) -> torch.Tensor:
        """``[rows, width]`` cache slots each row attends, scored over ``plane``'s keys."""
        return select(q, w, self._need_kv().aux_view(INDEX_PLANES, plane), self.current(label),
                      group)

    def attend(self, plane: int, q_nope: torch.Tensor, q_pe: torch.Tensor, slots: torch.Tensor,
               sm_scale: float, label: str = _MAIN) -> torch.Tensor:
        """Absorbed MLA of each row over its ``slots`` of ``plane``'s latents."""
        return sparse_attend(q_nope, q_pe, self._need_kv().layer_view(plane), slots,
                             self.current(label), sm_scale)
