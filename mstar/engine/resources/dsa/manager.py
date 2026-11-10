"""DeepSeek sparse attention (DSA) over the paged MLA cache: the index keys, each query's top-k
and the attention over it.

A model's FULL indexer layers hand this resource their tokens' index keys (``store_keys``) and
their queries and head weights (``select``): each query row scores its request's keys,
``sum_h w[h] * relu(q[h] . k)`` in fp32, and keeps its top ``topk`` positions. SHARED layers
reuse the last FULL layer's pick (``shared_selection``), and ``attend`` runs absorbed MLA over
just the picked latents of the MLA cache. A step's rows come from the KV plan: each request's
rows, the position they start at and its pages.

The keys live in per-request buffers that grow with the context and the selection runs per
request on the host, so the path is eager only; past ``topk`` it decodes (one row per request).
The follow-up moves the keys onto the MLA cache's pages, under its page table, allocator and
eviction, and gives selection and attention fixed-shape buffers so a decode can capture.
``KPoolDSA`` below is that resource for a pooled-key indexer (GLM-5.3-Flash), over ``paged.py``.
"""

import itertools
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from mstar.engine.resources.attn.workspace import WorkspacePool
from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo, Resource
from mstar.engine.resources.dsa import paged, sparse_mla
from mstar.engine.resources.dsa.config import INDEX_PLANES, DSAConfig, DSASpec, DSAStep
from mstar.engine.resources.dsa.paged import KPoolContext, attn_lens
from mstar.engine.resources.kv.plan import KVPlanOutputs
from mstar.engine.resources.step import BucketKey, StepContext
from mstar.utils.h2d import PinnedStager, to_device_nonblocking

_INITIAL_CAPACITY = 64
_MAIN = "main"


class IndexKeyStore:
    """Per-request, per-layer growing buffers of the indexer's keys, in position order."""

    def __init__(self) -> None:
        # request_id -> layer_idx -> [buffer (capacity, head_dim), filled rows]
        self._buffers: dict[str, dict[int, list]] = {}

    def append(
        self,
        request_id: str,
        layer_idx: int,
        keys: torch.Tensor,
        start_pos: int,
    ) -> None:
        """Append one chunk of ``(n, head_dim)`` keys at positions ``start_pos ..``."""
        per_layer = self._buffers.setdefault(request_id, {})
        num_new, head_dim = keys.shape
        entry = per_layer.get(layer_idx)
        if entry is None:
            capacity = max(_INITIAL_CAPACITY, num_new)
            entry = per_layer[layer_idx] = [keys.new_empty((capacity, head_dim)), 0]
        buffer, filled = entry
        if start_pos != filled:
            raise RuntimeError(
                f"DSA key store desync for request {request_id!r} layer {layer_idx}: "
                f"append at position {start_pos} but {filled} keys are stored"
            )
        needed = filled + num_new
        if needed > buffer.shape[0]:
            grown = buffer.new_empty((max(needed, 2 * buffer.shape[0]), head_dim))
            grown[:filled] = buffer[:filled]
            entry[0] = buffer = grown
        buffer[filled:needed] = keys.detach()
        entry[1] = needed

    def history(self, request_id: str, layer_idx: int, length: int) -> torch.Tensor:
        """The first ``length`` stored keys — a view, not a copy. Raises if
        fewer are stored (a scoring window must never exceed the appended
        history: the current chunk is appended before selection)."""
        entry = self._buffers[request_id][layer_idx]
        buffer, filled = entry
        if length > filled:
            raise RuntimeError(
                f"DSA key store for request {request_id!r} layer {layer_idx} holds "
                f"{filled} keys; selection window wants {length}"
            )
        return buffer[:length]

    def evict(self, request_id: str) -> None:
        """Drop all of a request's buffers. Idempotent — the engine may retire
        a request that never reached a DSA forward."""
        self._buffers.pop(request_id, None)

    # -- introspection (tests + leak asserts) ---------------------------------

    def tracked_requests(self) -> set[str]:
        return set(self._buffers)

    def tokens(self, request_id: str, layer_idx: int) -> int:
        """Stored rows for one (request, layer); 0 when absent."""
        entry = self._buffers.get(request_id, {}).get(layer_idx)
        return 0 if entry is None else entry[1]


def select_topk_causal(
    scores: torch.Tensor, positions: torch.Tensor, topk: int
) -> torch.Tensor:
    """Per-row top-``topk`` key positions of causally masked ``scores``.

    ``scores`` is ``(T, num_keys)`` with every key past a row's position
    already ``-inf``; the result is ``(T, topk)`` int32, -1 padded where a
    row sees fewer than ``topk`` keys. One stable sort for the whole batch:
    no per-token topk launch, no host sync. Ties break toward the earlier
    key position, so a row's selection does not depend on how the batch
    is packed or on which device ran it (torch.topk's tie order is neither).
    """
    num_tokens, num_keys = scores.shape
    k_eff = min(topk, num_keys)
    order = scores.sort(dim=1, descending=True, stable=True).indices[:, :k_eff]
    # the -inf tail sorts last (stable: by ascending index), so rank >= p_t+1
    # is exactly the invisible remainder
    rank = torch.arange(k_eff, device=scores.device)
    invisible = rank.unsqueeze(0) >= positions.unsqueeze(1) + 1
    selection = torch.full(
        (num_tokens, topk), -1, dtype=torch.int32, device=scores.device)
    selection[:, :k_eff] = order.to(torch.int32).masked_fill(invisible, -1)
    return selection


def index_topk(
    q: torch.Tensor, weights: torch.Tensor, keys: torch.Tensor,
    positions: torch.Tensor, topk: int,
) -> torch.Tensor:
    """Top-``topk`` key positions per query row, ``(T, topk)`` int32, -1 padded.

    ``q`` ``(T, H, D)`` and ``weights`` ``(T, H)`` are the indexer's queries and head weights
    (any scales folded in), ``keys`` ``(S, D)`` a request's keys by position and ``positions``
    ``(T,)`` the rows' own: a row sees keys ``0 .. position``, itself included."""
    # score[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s]), in fp32: per-head ReLU BEFORE the
    # weighted sum; the raw weights get no softmax/sigmoid. The engine's bf16 autocast would run
    # both einsums in bf16.
    with torch.autocast(q.device.type, enabled=False):
        dots = torch.einsum("thd,sd->ths", q.float(), keys.float()).relu()
        scores = torch.einsum("th,ths->ts", weights.float(), dots)
    key_pos = torch.arange(keys.shape[0], device=scores.device)
    scores = scores.masked_fill(key_pos.unsqueeze(0) > positions.unsqueeze(1), float("-inf"))
    return select_topk_causal(scores, positions, topk)


@dataclass
class DSASpan:
    """One request's rows of a step: ``q_len`` rows from row ``q_start``, the first at position
    ``ctx_start``; ``pages`` its MLA cache pages, ``page_start`` its first token's offset."""

    request_id: str
    q_start: int
    q_len: int
    ctx_start: int
    pages: list[int]
    page_start: int = 0


@dataclass
class DSAPlan:
    """A label's rows this step. ``needs_selection``: some row sees more than ``topk`` keys;
    ``last_selection``: the last FULL layer's pick this forward, which SHARED layers reuse.

    Under selection every row is its request's one decode row, and ``pages`` holds the rows'
    page tables back to back on the device: row ``r``'s starts at ``row_page_offset[r]``, and
    its first token sits ``row_start[r]`` slots into it."""

    spans: list[DSASpan]
    needs_selection: bool
    last_selection: torch.Tensor | None = None
    pages: torch.Tensor | None = None
    row_page_offset: torch.Tensor | None = None
    row_start: torch.Tensor | None = None


class DSAManager(Resource):
    """DSA for one node's layers over its MLA cache (``config.kv_cache``)."""

    @classmethod
    def build(cls, spec: DSASpec, info: EngineResourceInfo):
        # `dependency` hands back the SPEC; its `.config` is the KV config
        kv_spec = info.dependency(spec.config.kv_cache)
        return cls(spec.config, page_size=kv_spec.config.tokens_per_page, device=info.device)

    def __init__(self, config: DSAConfig, page_size: int, device: torch.device):
        self.config = config
        self.page_size = page_size
        self._device = device
        self.keys = IndexKeyStore()
        self._plans: dict[str, DSAPlan] = {}

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

    def remove_request(self, rid: str) -> None:
        self.keys.evict(rid)

    def reset_request(self, rid: str, free: bool = False) -> None:
        self.keys.evict(rid)

    def plan(self, step: DSAStep, ctx: StepContext) -> dict[str, DSAPlan]:
        """Each label's rows, as the KV plan placed them: a request's rows follow the one
        before it, and its first sits at its length before the step."""
        del step
        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self.config.kv_cache)
        assert plan_outputs is not None, f"DSA expected a plan result from {self.config.kv_cache}"
        self._plans = {}
        for label, kv_out in plan_outputs.items():
            spans, q_start = [], 0
            for view in kv_out.views:
                spans.append(DSASpan(
                    request_id=view.request_id, q_start=q_start, q_len=view.to_compute,
                    ctx_start=view.length - view.to_compute, pages=list(view.page_idxs),
                    page_start=view.start,
                ))
                q_start += view.to_compute
            needs = any(view.length > self.config.topk for view in kv_out.views)
            plan = self._plans[label] = DSAPlan(spans=spans, needs_selection=needs)
            if needs:
                self._stage_page_tables(plan)
        return self._plans

    def _stage_page_tables(self, plan: DSAPlan) -> None:
        """Every layer of the step attends through the same page tables, so they go to the
        device once, here, rather than once per layer."""
        for span in plan.spans:
            if span.q_len != 1:
                raise RuntimeError(
                    "sparse attention beyond topk is decode-only (one row per request); a "
                    "prefill past topk is refused before the step"
                )
        pages = [page for span in plan.spans for page in span.pages]
        offsets = list(itertools.accumulate((len(span.pages) for span in plan.spans[:-1]), initial=0))
        starts = [span.page_start for span in plan.spans]
        staged = to_device_nonblocking(pages + offsets + starts, self._device, dtype=torch.long)
        n, rows = len(pages), len(plan.spans)
        plan.pages = staged[:n]
        plan.row_page_offset = staged[n:n + rows]
        plan.row_start = staged[n + rows:]

    def needs_selection(self, label: str = _MAIN) -> bool:
        """Whether this step's rows attend a pick rather than every key."""
        return self._plans[label].needs_selection

    # Host-side per-request work (dict mutation, python loops over spans): keep dynamo out of
    # it, as the cache managers' plan / run do, so the engine's warmup torch.compile runs these
    # eagerly instead of graph-breaking token by token.
    @torch.compiler.disable
    def store_keys(self, layer_idx: int, keys: torch.Tensor, label: str = _MAIN) -> None:
        """Store the step's ``(T, D)`` index keys of one layer, each request's at its positions."""
        for span in self._plans[label].spans:
            self.keys.append(
                span.request_id, layer_idx, keys[span.q_start:span.q_start + span.q_len],
                start_pos=span.ctx_start,
            )

    @torch.compiler.disable
    def select(
        self, layer_idx: int, q: torch.Tensor, weights: torch.Tensor, label: str = _MAIN,
    ) -> torch.Tensor:
        """Each row's top ``topk`` positions of its request (``index_topk``) over this layer's
        stored keys, ``(T, topk)`` int32, -1 padded; SHARED layers reuse it this forward."""
        plan = self._plans[label]
        rows = []
        for span in plan.spans:
            rows_of = slice(span.q_start, span.q_start + span.q_len)
            keys = self.keys.history(span.request_id, layer_idx, span.ctx_start + span.q_len)
            positions = torch.arange(
                span.ctx_start, span.ctx_start + span.q_len, device=q.device)
            rows.append(index_topk(q[rows_of], weights[rows_of], keys, positions,
                                   self.config.topk))
        selection = torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]
        plan.last_selection = selection
        return selection

    def shared_selection(self, layer_idx: int, label: str = _MAIN) -> torch.Tensor:
        """The last FULL layer's pick this forward, for a SHARED layer (IndexShare)."""
        selection = self._plans[label].last_selection
        if selection is None:
            raise RuntimeError(
                f"SHARED layer {layer_idx} needs a DSA selection but no FULL layer ran before "
                "it this forward: the model's first indexer layer must be FULL"
            )
        return selection

    @torch.compiler.disable
    def attend(
        self,
        selection: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        latent_layer: torch.Tensor,
        sm_scale: float,
        label: str = _MAIN,
    ) -> torch.Tensor:
        """Absorbed MLA of each row over its picked latents, ``(T, H, kv_lora_rank)``.

        ``latent_layer`` is the MLA cache's layer view ``[pages, page_size, kv_lora_rank +
        rope]``, this step's latents already written; ``selection`` holds request-local
        positions (-1: none). Past ``topk`` a request decodes, so row ``r`` is request ``r``'s."""
        plan = self._plans[label]
        rank = q_nope.shape[-1]  # the ckv width, after w_kc absorption
        query = torch.cat([q_nope, q_pe], dim=-1).float()  # (T, H, rank + rope)
        live = selection >= 0
        slots = selection.clamp_min(0).long() + plan.row_start[:, None]  # (T, topk)
        pages = plan.pages[plan.row_page_offset[:, None] + slots // self.page_size]
        picked = latent_layer[pages, slots % self.page_size].float()  # (T, topk, rank + rope)
        # fp32 outside the engine's bf16 autocast, which would run both einsums in bf16
        with torch.autocast(query.device.type, enabled=False):
            scores = torch.einsum("thd,tkd->thk", query, picked) * sm_scale
            attn = scores.masked_fill(~live[:, None, :], float("-inf")).softmax(dim=-1)
            out = torch.einsum("thk,tkd->thd", attn, picked[..., :rank])
        return out.to(q_nope.dtype)


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
        planes = kv_config.aux_planes.get(INDEX_PLANES)
        if planes is None:
            raise ValueError(f"DSA keeps its index keys on {spec.config.kv_cache}'s pages: "
                             f"declare its aux_planes[{INDEX_PLANES!r}]")
        d = spec.config.index_head_dim
        if d is None or planes.width != 3 * d:
            # a row is [key | gate | key of the pool ending there]
            raise ValueError(f"DSA's {INDEX_PLANES} rows are 3 x index_head_dim ({d}) wide; "
                             f"{spec.config.kv_cache} declares {planes.width}")
        page, kpool = kv_config.tokens_per_page, spec.config.kpool
        if page % kpool:
            # a pool's key sits in its last token's row: pools must not straddle pages
            raise ValueError(f"{spec.config.kv_cache} page_size={page} must be a multiple of "
                             f"index_kpool={kpool}")
        if (page // kpool) & (page // kpool - 1):
            # flashinfer's page-table transform takes power-of-two pools per page
            raise ValueError(f"{spec.config.kv_cache} page_size={page} over index_kpool={kpool} "
                             "must be a power of two")
        if info.device.type == "cuda":
            from mstar.engine.resources.attn.flashinfer_mla import _flashinfer_at_least

            if info.kv_dtype != torch.bfloat16:
                # the kernels take bf16; the torch paths sync with the host and capture nothing
                raise ValueError(f"DSA serves a bf16 KV cache on CUDA, not {info.kv_dtype}: set "
                                 "the node's dtype to bfloat16")
            if not _flashinfer_at_least("0.6.18"):
                raise ValueError("DSA needs FlashInfer >= 0.6.18, whose "
                                 "top_k_page_table_transform takes a page_size")
        if info.joint_comm_group is not None:
            kv_config.shard(info.joint_comm_group.world_size)
        return cls(spec.config, kv_config, info.device, spec.resource_key)

    def __init__(self, config: DSAConfig, kv_config, device: torch.device, name: str = "dsa"):
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
        # FlashInfer's workspaces: one per (walk, capture slot), one for eager steps
        self._workspaces = WorkspacePool(device, config.workspace_mb, owner=name)
        # (walk, capture slot, rows) -> its sparse-attention plan under CUDA graphs
        self._sparse_plans: dict[tuple[str, int, int], sparse_mla.SparseGraphPlan] = {}
        # (walk, capture slot) -> the index buffer its plans share
        self._sparse_indices: dict[tuple[str, int], torch.Tensor] = {}
        # an eager step's sparse plans, reused
        self._eager = (sparse_mla.EagerSparsePlans(self._workspaces)
                       if device.type == "cuda" else None)

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

    def seal_capture_buffers(self) -> None:
        # a workspace a plan outgrew would move, and a captured graph holds its address
        self._workspaces.seal()

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
        spans: list[paged.DSASpan] = []
        for i, view in enumerate(kv_out.views):
            spans.append(paged.DSASpan(len(pos), view.to_compute, i))
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
            indices = self._sparse_indices.get(key[:2])
            if indices is None or indices.numel() < rows * width + 1:
                # captures run largest first, so the first is the slot's size; +1: the
                # plan's spare index entry
                indices = self._sparse_indices[key[:2]] = torch.zeros(
                    rows * width + 1, dtype=torch.int32, device=self._device)
            sparse_plan = self._sparse_plans[key] = sparse_mla.SparseGraphPlan(
                rows, width, self._workspaces.get(walk, slot), indices)
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
        paged.write_index(self._need_kv(), plane, label, k, gate, ape, self.current(label))

    def select(self, plane: int, q: torch.Tensor, w: torch.Tensor, group=None,
               label: str = _MAIN) -> torch.Tensor:
        """``[rows, width]`` cache slots each row attends, scored over ``plane``'s keys."""
        return paged.select(q, w, self._need_kv().aux_view(INDEX_PLANES, plane),
                            self.current(label), group)

    def attend(self, plane: int, q_nope: torch.Tensor, q_pe: torch.Tensor, slots: torch.Tensor,
               sm_scale: float, label: str = _MAIN) -> torch.Tensor:
        """Absorbed MLA of each row over its ``slots`` of ``plane``'s latents."""
        return paged.sparse_attend(q_nope, q_pe, self._need_kv().layer_view(plane), slots,
                                   self.current(label), sm_scale, self._eager)
