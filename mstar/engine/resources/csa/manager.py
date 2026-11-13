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
from mstar.engine.resources.csa.config import INDEX_PLANES, CSAConfig, CSASpec, CSAStep
from mstar.engine.resources.step import Segment, StepContext
from mstar.utils.h2d import PinnedStager

_MAIN = "main"


@dataclass
class CSACache:
    """One cache's addressing for a step. Device: ``pages``, the flat page list of the step's
    requests and one spare entry (a request may hold no page yet), ``page_start`` each
    request's first entry in it, and ``written``, the step's rows whose token completes a
    group, in the order the cache's step writes them."""

    pages: torch.Tensor
    page_start: torch.Tensor
    written: torch.Tensor
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


class CSAManager(Resource):
    """CSA for one node's layers over their compressed caches (``config.caches``)."""

    @classmethod
    def build(cls, spec: CSASpec, info: EngineResourceInfo):
        page_sizes = {}
        for ratio, key in spec.config.caches.items():
            kv_config = info.dependency(key).config
            planes = kv_config.aux_planes.get(INDEX_PLANES)
            if planes is None or planes.width != spec.config.index_head_dim:
                raise ValueError(
                    f"CSA keeps its {spec.config.index_head_dim}-wide index keys on {key}'s pages: "
                    f"declare its aux_planes[{INDEX_PLANES!r}] that wide")
            if kv_config.qk_rope_head_dim:
                raise ValueError(f"CSA attends whole latents: {key} must declare no rope slot, "
                                 f"not qk_rope_head_dim={kv_config.qk_rope_head_dim}")
            page_sizes[ratio] = kv_config.tokens_per_page
        return cls(spec.config, page_sizes, info.device)

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
        written: dict[int, list[int]] = {ratio: [] for ratio in self.config.caches}
        for i, (rid, _, n) in enumerate(segments):
            row0, start = len(pos), self._tokens.get(rid, 0)
            spans.append((row0, n, start))
            pos.extend(range(start, start + n))
            row_req.extend([i] * n)
            first.extend([row0] * n)
            end.extend([start + n] * n)
            for ratio, rows in written.items():
                # the tokens at the end of a group: positions p with (p + 1) % ratio == 0
                rows.extend(range(row0 + -(start + 1) % ratio, row0 + n, ratio))

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
            parts += [torch.cat([host(indptr.paged_kv_indices), host([0])]), host(indptr.paged_kv_indptr[:-1]),
                      host(written[ratio])]

        # one copy from pinned memory, which waits for nothing on the stream
        staged = torch.empty(sum(p.numel() for p in parts), dtype=torch.long, device=self._device)
        self._stager.copy_(staged, torch.cat(parts).numpy())
        cuts = iter(staged.split([p.numel() for p in parts]))
        rows = CSARows(*(next(cuts) for _ in range(6)), caches={}, spans=spans)
        for ratio in self.config.caches:
            rows.caches[ratio] = CSACache(next(cuts), next(cuts), next(cuts), self.page_sizes[ratio])
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
        """Write the step's ``latent`` rows ``[rows, width]`` and its index ``keys`` ``[rows,
        index_head_dim]`` (either may be left for a later call) to ``plane`` of the ``ratio``
        cache: the rows that complete a group, at the slots the cache's step planned. The other
        rows' are dropped."""
        written = self.current(label).caches[ratio].written
        if not written.numel():
            return
        kv = self._cache(ratio)
        if latent is not None:
            kv.write_kv(latent.index_select(0, written), None, layer_idx=plane, label=label)
        if keys is not None:
            kv.write_aux(INDEX_PLANES, keys.index_select(0, written), layer_idx=plane, label=label)

    def scores(self, ratio: int, plane: int, q: torch.Tensor, w: torch.Tensor,
               label: str | None = None) -> torch.Tensor:
        """``[rows, groups]`` fp32 indexer scores of each row against its request's keys in
        ``plane`` of the ``ratio`` cache, ``sum_h w[h] * relu(q[h] . key)``; -inf past the groups
        the row sees. ``q (rows, heads, index_head_dim)``, ``w (rows, heads)`` with every scale
        folded in."""
        rows = self.current(label)
        view = self._cache(ratio).aux_view(INDEX_PLANES, plane)
        flat = view.view(-1, view.shape[-1])
        seen = rows.groups(ratio)
        width = max(1, max(start + n for _, n, start in rows.spans) // ratio)
        out = torch.full((q.shape[0], width), -torch.inf, dtype=torch.float32, device=q.device)
        group = torch.arange(width, device=q.device)
        heads = q.shape[1]
        with torch.autocast(q.device.type, enabled=False):
            for req, (r0, n, start) in enumerate(rows.spans):
                count = (start + n) // ratio  # groups the span's last row sees
                if count == 0:
                    continue
                req_ids = torch.full((count,), req, dtype=torch.long, device=q.device)
                keys = flat[rows.slots(ratio, req_ids, group[:count])].float()
                chunk = max(1, self.config.score_bytes // (4 * heads * count))
                for c0 in range(r0, r0 + n, chunk):
                    c1 = min(c0 + chunk, r0 + n)
                    dots = torch.einsum("rhd,jd->rhj", q[c0:c1].float(), keys).relu_()
                    out[c0:c1, :count] = torch.einsum("rh,rhj->rj", w[c0:c1].float(), dots)
        return out.masked_fill_(group[None, :] >= seen[:, None], -torch.inf)

    def candidates(self, ratio: int, scores: torch.Tensor, topk_blocks: int, block_size: int,
                   label: str | None = None) -> torch.Tensor:
        """``candidate_blocks`` of this step's ``scores``: the mask a later ``select`` keeps to."""
        return candidate_blocks(scores, self.current(label).groups(ratio), topk_blocks, block_size)

    def select(self, ratio: int, scores: torch.Tensor, keep: torch.Tensor | None = None,
               label: str | None = None) -> torch.Tensor:
        """``[rows, k]`` slots, in the ``ratio`` cache, of each row's top ``topk`` groups by
        ``scores`` (within ``keep``, a ``candidates`` mask; equal scores to the earlier group),
        -1 where a row sees fewer."""
        rows = self.current(label)
        if keep is not None:
            scores = scores.masked_fill(~keep[:, :scores.shape[1]], -torch.inf)
        values, indices = top_scores(scores, min(self.config.topk, scores.shape[1]))
        live = values > -torch.inf
        return torch.where(live, rows.slots(ratio, rows.row_req[:, None], torch.where(live, indices, 0)), -1)

    def attend(self, q: torch.Tensor, sink: torch.Tensor, window: torch.Tensor,
               window_idx: torch.Tensor, ratio: int | None = None, plane: int | None = None,
               slots: torch.Tensor | None = None) -> torch.Tensor:
        """``[rows, heads, D]``: each row's heads over its window rows ``window[window_idx]``
        (``window (m, D)``, ``window_idx (rows, w)``, -1 none) and its ``slots`` of ``plane``'s
        latents in the ``ratio`` cache (``select``'s, -1 none), one softmax at
        ``config.sm_scale`` with ``sink (heads,)`` in its denominator. A row with nothing to
        attend gets zeros."""
        width = q.shape[-1]
        latent = None
        if slots is not None:
            view = self._cache(ratio).layer_view(plane)
            latent = view.view(-1, view.shape[-1])
        entries = window_idx.shape[1] + (0 if slots is None else slots.shape[1])
        chunk = max(1, self.config.attend_bytes // (8 * max(entries, 1) * width))
        out = torch.empty_like(q)
        sink = sink.float()[None, :, None]
        with torch.autocast(q.device.type, enabled=False):
            for c0 in range(0, q.shape[0], chunk):
                c1 = min(c0 + chunk, q.shape[0])
                idx = window_idx[c0:c1]
                rows, valid = window[idx.clamp_min(0)].float(), idx >= 0
                if latent is not None:
                    picked = slots[c0:c1]
                    rows = torch.cat([rows, latent[picked.clamp_min(0)][..., :width].float()], dim=1)
                    valid = torch.cat([valid, picked >= 0], dim=1)
                logits = torch.einsum("rhd,rkd->rhk", q[c0:c1].float(), rows) * self.config.sm_scale
                logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
                prob = torch.cat([logits, sink.expand(c1 - c0, -1, 1)], dim=-1).softmax(-1)[..., :-1]
                out[c0:c1] = torch.einsum("rhk,rkd->rhd", prob, rows).to(out.dtype)
        return out
