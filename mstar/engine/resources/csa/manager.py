"""Compressed sparse attention (CSA) over a node's paged latent caches: the resource.

DeepSeek-V4.1's layers attend two kinds of rows in one softmax: a short window of their own
recent rows, which the model passes in, and a few rows of a compressed cache shared by several
layers, picked per query by an indexer. A compressed source pools ``ratio`` tokens into one
latent. Its cache keeps one row per group, so a request's cache of ratio r advances a row every
r tokens, and a group is visible to a query once the query has passed its last token. A learned
sink logit per head joins the softmax's denominator.

The caches count rows, so the resource keeps each request's token count: a step declares
``CSAStep()`` over its tokens, ``kv_segments`` gives the rows each cache advances by, ``plan``
lays the step's token positions over the caches' page tables (``CSARows``) and ``commit``
advances the count. The layers write latents and keys, score, select and attend through the
resource. This is the reference in torch: eager only (``can_capture`` is False), every row
scored against its request's keys.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.csa.config import INDEX_PLANES, CSAConfig, CSASpec, CSAStep, Fp4Rows
from mstar.engine.resources.step import Segment, StepContext
from mstar.utils.h2d import PinnedStager
from mstar.utils.mx_round import pack_fp4_rows, unpack_fp4_rows

_MAIN = "main"


@dataclass
class CSACache:
    """One cache's addressing for a step. Device: ``pages``, the flat page list of the step's
    requests and one spare entry (a request may hold no page yet), and ``page_start``, each
    request's first entry in it."""

    pages: torch.Tensor
    page_start: torch.Tensor
    page_size: int


@dataclass
class CSARows:
    """A step's rows under one label. Device, per row: ``pos`` its token position, ``row_req``
    its request (the step's order), ``first`` the step row its request starts at, ``end`` the
    position after its request's last row. Per request: ``req_first`` and ``req_rows`` (its
    rows in the step). ``caches`` by pooling ratio. Host: ``spans``, each request's ``(first
    row, rows, first position)``."""

    pos: torch.Tensor
    row_req: torch.Tensor
    first: torch.Tensor
    end: torch.Tensor
    req_first: torch.Tensor
    req_rows: torch.Tensor
    caches: dict[int, CSACache]
    spans: list[tuple[int, int, int]]

    def slots(self, ratio: int, req: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """Slots (``page * page_size + offset``) of ``groups`` of requests ``req`` in the cache
        of ``ratio``. A group its request does not hold gives some slot of the cache, for the
        caller to mask."""
        cache = self.caches[ratio]
        entry = cache.page_start[req] + groups // cache.page_size
        page = cache.pages[entry.clamp_max(cache.pages.numel() - 1)]
        return page * cache.page_size + groups % cache.page_size

    def groups(self, ratio: int) -> torch.Tensor:
        """Per row, the groups of a ``ratio`` cache it sees: those whose last token is at or
        before it."""
        return (self.pos + 1) // ratio


def top_scores(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's ``k`` best scores and their columns, equal scores to the earlier column: a
    row's pick then depends neither on the step's width nor on the device, as ``torch.topk``'s
    order among ties does."""
    order = scores.sort(dim=-1, descending=True, stable=True)
    return order.values[:, :k], order.indices[:, :k]


def candidate_blocks(scores: torch.Tensor, groups: torch.Tensor, topk_blocks: int,
                     block_size: int) -> torch.Tensor:
    """The first level of a two-level selection: per row, a mask over ``scores``' groups that
    keeps the ``topk_blocks`` best blocks of ``block_size`` consecutive groups (a block scores
    as its best group), always with the block that holds the row's newest group, which is only
    partly filled. ``scores`` are -inf past each row's ``groups``."""
    width = scores.shape[-1]
    block = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    block = block.unflatten(-1, (-1, block_size)).amax(-1)
    last = (groups - 1) // block_size
    ids = torch.arange(block.shape[-1], device=scores.device)
    block = block.masked_fill(ids[None, :] == last[:, None], torch.inf)
    values, indices = top_scores(block, min(topk_blocks, block.shape[-1]))
    keep = torch.zeros_like(block, dtype=torch.bool).scatter_(-1, indices, values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _store(values: torch.Tensor, fmt: Fp4Rows | None, like: torch.Tensor) -> torch.Tensor:
    """``values [n, width]`` as rows of the cache plane ``like`` (``[..., row width]`` in the
    cache's dtype): as they are, or with ``fmt`` rounded to its grid, as FP4 bytes, the row's
    spare bytes zero."""
    if fmt is None:
        return values
    packed = pack_fp4_rows(values, fmt.block, fmt.e4m3_scale)
    rows = packed.new_zeros(values.shape[0], like.shape[-1] * like.dtype.itemsize)
    rows[:, :packed.shape[1]] = packed
    return rows.view(like.dtype)


def _load(rows: torch.Tensor, fmt: Fp4Rows | None, width: int,
          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Gathered cache rows ``[..., row width]`` as ``dtype`` values ``[..., width]``: fp32, or
    bf16, which holds every FP4 value exactly."""
    if fmt is None:
        return rows[..., :width].to(dtype)
    return unpack_fp4_rows(rows.contiguous().view(torch.uint8), width, fmt.block, fmt.e4m3_scale, dtype)


class CSAManager(Resource):
    """CSA for one node's layers over their compressed caches (``config.caches``)."""

    @classmethod
    def build(cls, spec: CSASpec, info: EngineResourceInfo):
        config, page_sizes = spec.config, {}
        itemsize = info.kv_dtype.itemsize
        latent_bytes, key_bytes = config.row_bytes(itemsize)

        def holds(width: int, nbytes: int, fmt: Fp4Rows | None) -> bool:
            # values fill their row exactly; packed bytes may leave part of a last element spare
            return width * itemsize == nbytes if fmt is None else width * itemsize >= nbytes

        for ratio, key in config.caches.items():
            kv_config = info.dependency(key).config
            planes = kv_config.aux_planes.get(INDEX_PLANES)
            if planes is None or not holds(planes.width, key_bytes, config.key_rows):
                raise ValueError(
                    f"CSA keeps its index keys, {key_bytes} bytes a row, on {key}'s pages: declare its "
                    f"aux_planes[{INDEX_PLANES!r}] {-(-key_bytes // itemsize)} {info.kv_dtype} values wide")
            if kv_config.qk_rope_head_dim:
                raise ValueError(f"CSA attends whole latents: {key} must declare no rope slot, "
                                 f"not qk_rope_head_dim={kv_config.qk_rope_head_dim}")
            if not holds(kv_config.kv_lora_rank, latent_bytes, config.latent_rows):
                raise ValueError(
                    f"CSA keeps its latents, {latent_bytes} bytes a row, on {key}: declare its "
                    f"kv_lora_rank as {-(-latent_bytes // itemsize)} {info.kv_dtype} values")
            page_sizes[ratio] = kv_config.tokens_per_page
        return cls(config, page_sizes, info.device)

    def __init__(self, config: CSAConfig, page_sizes: Mapping[int, int], device: torch.device):
        self.config = config
        self.page_sizes = dict(page_sizes)
        self._device = device
        self._kv: dict[int, Resource] = {}  # by ratio, the caches it reads (`bind_dependencies`)
        self._current: dict[str, CSARows] = {}
        # request -> the tokens its committed steps ran: where its next step's rows stand
        self._tokens: dict = {}
        self._stager = PinnedStager(torch.long)

    def depends_on(self) -> set[str]:
        return set(self.config.caches.values())

    def bind_dependencies(self, resources: Mapping[str, Resource]) -> None:
        self._kv = {ratio: resources[key] for ratio, key in self.config.caches.items()}

    def can_capture(self) -> bool:
        # the torch reference sizes its launches on the host
        return False

    # -- the request and the step ------------------------------------------------------------

    def remove_request(self, rid) -> None:
        self._tokens.pop(rid, None)

    def reset_request(self, rid, free: bool = False) -> None:
        self._tokens.pop(rid, None)

    def kv_segments(self, segments: Sequence[Segment]) -> dict[str, tuple[Segment, ...]]:
        """What a step of ``segments`` (spans in tokens) writes to each cache, by its key: per
        request, the groups of that cache's ratio the step's tokens complete. The submodule
        declares each cache's ``KVStep`` over these."""
        tokens = self._tokens
        rows: dict[str, list[Segment]] = {key: [] for key in self.config.caches.values()}
        for rid, label, span in segments:
            start = tokens.get(rid, 0)
            for ratio, key in self.config.caches.items():
                rows[key].append(Segment(rid, label, (start + span) // ratio - start // ratio))
        return {key: tuple(segs) for key, segs in rows.items()}

    def plan(self, step: CSAStep, ctx: StepContext) -> dict[str, CSARows]:
        labels: dict[str, list[Segment]] = {}
        for segment in step.segments:
            labels.setdefault(segment.label, []).append(segment)
        self._current = {label: self._rows(label, segments, ctx) for label, segments in labels.items()}
        return self._current

    def _rows(self, label: str, segments: list[Segment], ctx: StepContext) -> CSARows:
        pos: list[int] = []
        row_req: list[int] = []
        first: list[int] = []
        end: list[int] = []
        spans: list[tuple[int, int, int]] = []
        for i, (rid, _, n) in enumerate(segments):
            row0, start = len(pos), self._tokens.get(rid, 0)
            spans.append((row0, n, start))
            pos.extend(range(start, start + n))
            row_req.extend([i] * n)
            first.extend([row0] * n)
            end.extend([start + n] * n)

        def host(values) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.long)

        parts = [host(p) for p in (pos, row_req, first, end, [s for s, _, _ in spans], [n for _, n, _ in spans])]
        for ratio, key in self.config.caches.items():
            kv_out = ctx.plan_results[key][label]
            for view, (_, n, start) in zip(kv_out.views, spans, strict=True):
                have = (start + n) // ratio
                if (view.to_compute, view.length) != (have - start // ratio, have):
                    raise RuntimeError(
                        f"csa: {key} plans {view.to_compute} rows to a length of {view.length} for a "
                        f"request at token {start} running {n}: its step is not kv_segments' "
                        f"({have - start // ratio} to {have})")
            indptr = kv_out.cpu_indptrs
            # the page lists stay tensors: a long context's are the bulk of the plan
            parts += [torch.cat([host(indptr.paged_kv_indices), host([0])]), host(indptr.paged_kv_indptr[:-1])]

        # one copy from pinned memory, which waits for nothing on the stream
        staged = torch.empty(sum(p.numel() for p in parts), dtype=torch.long, device=self._device)
        self._stager.copy_(staged, torch.cat(parts).numpy())
        cuts = iter(staged.split([p.numel() for p in parts]))
        rows = CSARows(*(next(cuts) for _ in range(6)), caches={}, spans=spans)
        for ratio in self.config.caches:
            rows.caches[ratio] = CSACache(next(cuts), next(cuts), self.page_sizes[ratio])
        return rows

    def commit(self, step: CSAStep, ctx: StepContext) -> None:
        tokens = self._tokens
        for rid, _, span in step.segments:
            if span and not ctx.is_padding_row(rid):
                tokens[rid] = tokens.get(rid, 0) + span

    # -- what the layers and the submodule call ----------------------------------------------

    def current(self, label: str | None = None) -> CSARows:
        label = label or _MAIN
        if label not in self._current:
            raise KeyError(f"csa: no plan for label {label!r}; the step declares CSAStep()")
        return self._current[label]

    def _cache(self, ratio: int) -> Resource:
        if ratio not in self._kv:
            raise RuntimeError(f"csa: no cache of ratio {ratio} bound (config.caches, bind_dependencies)")
        return self._kv[ratio]

    def write(self, ratio: int, plane: int, latent: torch.Tensor | None = None,
              keys: torch.Tensor | None = None, label: str | None = None) -> None:
        """Write the step's ``latent`` rows ``[rows, latent_dim]`` and its index ``keys``
        ``[rows, index_head_dim]`` (either may be left for a later call) to ``plane`` of the
        ``ratio`` cache. A row whose token completes a group goes to that group's slot, which
        the cache's step reserved; the others go to the cache's spare row, so the write has the
        step's shape whichever rows complete one. A cache of ``Fp4Rows`` rounds them to its grid."""
        rows, kv = self.current(label), self._cache(ratio)
        seen = rows.groups(ratio)
        spare, _ = kv.spare_row()
        slots = torch.where((rows.pos + 1) % ratio == 0, rows.slots(ratio, rows.row_req, seen - 1), spare)
        for values, fmt, view in ((latent, self.config.latent_rows, kv.layer_view(plane)),
                                  (keys, self.config.key_rows, kv.aux_view(INDEX_PLANES, plane))):
            if values is not None:
                view.view(-1, view.shape[-1])[slots] = _store(values, fmt, view).to(view.dtype)

    def scores(self, ratio: int, plane: int, q: torch.Tensor, w: torch.Tensor,
               label: str | None = None) -> torch.Tensor:
        """``[rows, groups]`` indexer scores of each row against its request's keys in ``plane``
        of the ``ratio`` cache, ``sum_h w[h] * relu(q[h] . key)``; -inf past the groups the row
        sees. ``q (rows, heads, index_head_dim)``, ``w (rows, heads)`` with every scale folded
        in. Computed in ``q``'s dtype, as the model's reference scores, and returned in fp32:
        in bf16 near-equal scores round together, and which rows a query picks follows."""
        rows = self.current(label)
        view = self._cache(ratio).aux_view(INDEX_PLANES, plane)
        flat = view.view(-1, view.shape[-1])
        seen = rows.groups(ratio)
        width = max(1, max(start + n for _, n, start in rows.spans) // ratio)
        out = torch.full((q.shape[0], width), -torch.inf, dtype=torch.float32, device=q.device)
        group = torch.arange(width, device=q.device)
        heads = q.shape[1]
        with torch.autocast(q.device.type, enabled=False):
            if all(n == 1 for _, n, _ in rows.spans):
                # a decode step: every row against its own request's keys, a launch per chunk
                # of rows instead of one per request
                chunk = max(1, self.config.score_bytes // (4 * width * max(heads, q.shape[-1])))
                for c0 in range(0, q.shape[0], chunk):
                    c1 = min(c0 + chunk, q.shape[0])
                    slots = rows.slots(ratio, rows.row_req[c0:c1, None], group[None, :])
                    keys = _load(flat[slots], self.config.key_rows, q.shape[-1], q.dtype)
                    dots = torch.einsum("rhd,rjd->rhj", q[c0:c1], keys).relu_()
                    out[c0:c1] = (dots * w[c0:c1].unsqueeze(-1)).sum(dim=1)
                return out.masked_fill_(group[None, :] >= seen[:, None], -torch.inf)
            for req, (r0, n, start) in enumerate(rows.spans):
                count = (start + n) // ratio  # groups the span's last row sees
                if count == 0:
                    continue
                req_ids = torch.full((count,), req, dtype=torch.long, device=q.device)
                keys = _load(flat[rows.slots(ratio, req_ids, group[:count])], self.config.key_rows, q.shape[-1],
                             q.dtype)
                chunk = max(1, self.config.score_bytes // (4 * heads * count))
                for c0 in range(r0, r0 + n, chunk):
                    c1 = min(c0 + chunk, r0 + n)
                    dots = torch.einsum("rhd,jd->rhj", q[c0:c1], keys).relu_()
                    out[c0:c1, :count] = (dots * w[c0:c1].unsqueeze(-1)).sum(dim=1)
        return out.masked_fill_(group[None, :] >= seen[:, None], -torch.inf)

    def candidates(self, ratio: int, scores: torch.Tensor, topk_blocks: int, block_size: int,
                   label: str | None = None) -> torch.Tensor:
        """``candidate_blocks`` of this step's ``scores``: the mask a later ``select`` keeps to."""
        return candidate_blocks(scores, self.current(label).groups(ratio), topk_blocks, block_size)

    def select(self, ratio: int, scores: torch.Tensor, keep: torch.Tensor | None = None,
               label: str | None = None) -> torch.Tensor:
        """``[rows, k]`` slots, in the ``ratio`` cache, of each row's top ``topk`` groups by
        ``scores`` (within ``keep``, a ``candidates`` mask; equal scores to the earlier group),
        in the groups' order, then -1 where a row sees fewer."""
        rows = self.current(label)
        if keep is not None:
            scores = scores.masked_fill(~keep[:, :scores.shape[1]], -torch.inf)
        values, indices = top_scores(scores, min(self.config.topk, scores.shape[1]))
        # by position, as the softmax's blocks take them; the groups a row does not see last
        indices = torch.where(values > -torch.inf, indices, scores.shape[1]).sort(dim=-1).values
        live = indices < scores.shape[1]
        return torch.where(live, rows.slots(ratio, rows.row_req[:, None], torch.where(live, indices, 0)), -1)

    def attend(self, q: torch.Tensor, sink: torch.Tensor, window: torch.Tensor, window_idx: torch.Tensor,
               window_cols: torch.Tensor, ratio: int | None = None, plane: int | None = None,
               slots: torch.Tensor | None = None) -> torch.Tensor:
        """``[rows, heads, D]``: each row's heads over its entries, one softmax at
        ``config.sm_scale`` with ``sink (heads,)`` in its denominator. A row's entries are its
        window rows ``window[window_idx[:window_cols]]`` (``window (m, D)``, ``window_idx (rows,
        w)``, -1 an empty column) and then its ``slots`` of ``plane``'s latents in the ``ratio``
        cache (``select``'s, -1 none). The softmax accumulates over them ``config.softmax_block``
        at a time, so where a row's entries sit decides which share a block. A row with nothing
        to attend gets zeros."""
        count, heads, width = q.shape
        latent = None
        picks = 0 if slots is None else slots.shape[1]
        if slots is not None:
            view = self._cache(ratio).layer_view(plane)
            latent = view.view(-1, view.shape[-1])
        # each row's entries packed: its window columns, then its picks
        col = torch.arange(window_idx.shape[1] + picks, device=q.device)[None, :]
        at = col.clamp_max(window_idx.shape[1] - 1).expand(count, -1)
        from_window = torch.where(col < window_cols[:, None], window_idx.gather(1, at), -1)
        from_cache = None
        if slots is not None:
            pick = col - window_cols[:, None]
            from_cache = torch.where((pick >= 0) & (pick < picks), slots.gather(1, pick.clamp(0, picks - 1)), -1)
        # the softmax runs a block of entries at a time: each block's scores against the running
        # maximum, its probabilities rounded to the queries' dtype, the sums so far rescaled. The
        # whole blocks' scores and weighted entries are computed at once, a last partial block at
        # its own width (a GEMM over zero padding sums in another order), then folded in block order.
        block = self.config.softmax_block
        whole = col.shape[1] // block
        full = whole * block
        blocks = whole + (full < col.shape[1])
        chunk = max(1, self.config.attend_bytes // (8 * width * (col.shape[1] + blocks * heads)))
        out = torch.empty_like(q)
        sink = sink.float()[None, :]
        with torch.autocast(q.device.type, enabled=False):
            for c0 in range(0, count, chunk):
                c1 = min(c0 + chunk, count)
                query = q[c0:c1].float()
                idx = from_window[c0:c1]
                valid = idx >= 0
                entries = window[idx.clamp_min(0)].float() * valid[..., None]
                if from_cache is not None:
                    picked = from_cache[c0:c1]
                    live = picked >= 0
                    entries = entries + _load(
                        latent[picked.clamp_min(0)], self.config.latent_rows, width) * live[..., None]
                    valid = valid | live
                scores = torch.einsum("rhd,rkd->rhk", query, entries) * self.config.sm_scale
                scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
                # (rows, heads, blocks, block) for the whole blocks, (rows, heads, 1, rest) for the last
                parts = []
                if whole:
                    parts.append((scores[..., :full].unflatten(-1, (whole, block)),
                                  entries[:, :full].unflatten(1, (whole, block))))
                if blocks > whole:
                    parts.append((scores[..., None, full:], entries[:, None, full:]))
                peak = torch.cat([part.amax(-1) for part, _ in parts], dim=-1).cummax(-1).values.clamp_min(-1e30)
                decay = torch.exp(F.pad(peak[..., :-1], (1, 0), value=-1e30) - peak)
                mass, weighted, at = [], [], 0
                for part, rows_ in parts:
                    prob = torch.exp(part - peak[..., at:at + part.shape[2], None])
                    mass.append(prob.sum(-1))
                    weighted.append(torch.einsum("rhbk,rbkd->rhbd", prob.to(q.dtype).float(), rows_))
                    at += part.shape[2]
                mass, weighted = torch.cat(mass, dim=-1), torch.cat(weighted, dim=2)
                acc = torch.zeros(c1 - c0, heads, width, dtype=torch.float32, device=q.device)
                total = torch.zeros(c1 - c0, heads, dtype=torch.float32, device=q.device)
                for b in range(blocks):
                    total = total * decay[..., b] + mass[..., b]
                    acc = acc * decay[..., b, None] + weighted[:, :, b]
                out[c0:c1] = (acc / (total + torch.exp(sink - peak[..., -1]))[..., None]).to(out.dtype)
        return out
