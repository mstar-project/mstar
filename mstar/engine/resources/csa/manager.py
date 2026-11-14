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
resource. This is the reference in torch, every row scored against its request's keys.

A decode step can be captured: under a capture lease ``plan`` lays the bucket's rows, its padding
rows among them, into the (bucket, slot)'s own buffers, and every row scores each cache at one
width (``CSAStep.context`` tokens, else all the cache holds), so the forward's shapes follow from
the bucket alone and its values from the buffers a later plan refills.

A decode step may run drafts of a request's next tokens behind its token
(``CSAStep(decode_rows=True)``), each row as a one-token step. ``take_back`` then takes the
rejected ones out of the request's count and its caches' lengths, between steps: their cache
rows lie past the lengths, where no row sees them and the next step's writes land.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo, Resource
from mstar.engine.resources.csa.config import INDEX_PLANES, CSAConfig, CSASpec, CSAStep, Fp4Rows
from mstar.engine.resources.step import BucketKey, Segment, StepContext
from mstar.utils.h2d import PinnedStager
from mstar.utils.mx_round import pack_fp4_rows, unpack_fp4_rows

_MAIN = "main"


@dataclass
class CSACache:
    """One cache's addressing for a step. Device: ``pages``, the flat page lists of the step's
    requests in every cache, and ``page_start``, each request's first entry of this cache in it.
    ``num_pages`` is the cache's size: an entry read past a request's own pages (a group it does
    not hold, which the caller masks) is held inside the cache."""

    pages: torch.Tensor
    page_start: torch.Tensor
    page_size: int
    num_pages: int


@dataclass
class CSARows:
    """A step's rows under one label. Device, per row: ``pos`` its token position, ``row_req``
    its request (the step's order), ``first`` the step row its request starts at, ``end`` the
    position after its request's last row. Per request: ``req_first`` and ``req_rows`` (its
    rows in the step). ``caches`` by pooling ratio. Host: ``spans``, each request's ``(first
    row, rows, first position)``; ``one_row_each``, whether every request runs one row; ``decode``:
    every row runs as a one-token step would (``CSAStep.decode_rows``), whatever its request's
    rows in the step; and under a capture lease ``widths``, the groups every row scores each cache
    at (the most a request can hold), by ratio."""

    pos: torch.Tensor
    row_req: torch.Tensor
    first: torch.Tensor
    end: torch.Tensor
    req_first: torch.Tensor
    req_rows: torch.Tensor
    caches: dict[int, CSACache]
    spans: list[tuple[int, int, int]]
    one_row_each: bool = False
    widths: dict[int, int] | None = None
    decode: bool = False

    def slots(self, ratio: int, req: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """Slots (``page * page_size + offset``) of ``groups`` of requests ``req`` in the cache
        of ``ratio``. A group its request does not hold gives some slot of the cache, for the
        caller to mask."""
        cache = self.caches[ratio]
        entry = cache.page_start[req] + groups // cache.page_size
        page = cache.pages[entry.clamp(0, cache.pages.numel() - 1)].clamp_max(cache.num_pages - 1)
        return page * cache.page_size + groups % cache.page_size

    def score_width(self, ratio: int) -> int:
        """The groups a step's rows score the ``ratio`` cache at: its requests' most, or a
        captured step's fixed width."""
        if self.widths is not None:
            return max(1, self.widths[ratio])
        return max(1, max(start + n for _, n, start in self.spans) // ratio)

    def tail(self, count: int):
        """An index of the step's rows that are among their request's last ``count``: every row
        of a step that runs one row a request, as a slice, so that what a layer keeps of a
        captured step for the next has a shape that does not follow from the rows' positions."""
        if self.one_row_each:
            return slice(None)
        return self.pos >= self.end - count

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
        config, page_sizes, capacity = spec.config, {}, {}
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
            # a cache is sized in the node's tokens, of which a group takes ``ratio``
            capacity[ratio] = kv_config.max_seq_len // ratio
        return cls(config, page_sizes, info.device, capacity)

    def __init__(self, config: CSAConfig, page_sizes: Mapping[int, int], device: torch.device,
                 capacity: Mapping[int, int] | None = None):
        self.config = config
        self.page_sizes = dict(page_sizes)
        # by ratio, the groups a request's cache can hold: a captured step's score width
        self.capacity = dict(capacity or {})
        self._device = device
        self._kv: dict[int, Resource] = {}  # by ratio, the caches it reads (`bind_dependencies`)
        self._current: dict[str, CSARows] = {}
        # request -> the tokens its committed steps ran: where its next step's rows stand
        self._tokens: dict = {}
        self._stager = PinnedStager(torch.long)
        # (bucket, slot) -> the rows a captured graph reads, which every plan under that lease refills
        self._cg_rows: dict[tuple[BucketKey, int], torch.Tensor] = {}

    def depends_on(self) -> set[str]:
        return set(self.config.caches.values())

    def bind_dependencies(self, resources: Mapping[str, Resource]) -> None:
        self._kv = {ratio: resources[key] for ratio, key in self.config.caches.items()}

    @property
    def force_double_buffer(self) -> bool:
        # a captured slot's rows are refilled in `plan` from pinned memory, which the replay
        # before it may still be reading
        return True

    # -- capture buffers ---------------------------------------------------------------------

    def _layout(self, rows: int, requests: int, pages: int) -> tuple[list[int], int]:
        """The sizes of a step's staged parts, in order, and their sum: four values a row, two
        a request, each cache's first page entry a request, then ``pages`` page entries and a
        spare one."""
        sizes = [rows] * 4 + [requests] * (2 + len(self.config.caches)) + [pages + 1]
        return sizes, sum(sizes)

    def _bucket_pages(self, bucket: BucketKey) -> int:
        """Page entries a captured bucket's requests can hold over every cache."""
        return bucket.bs * sum(-(-self._capacity(ratio) // size) for ratio, size in self.page_sizes.items())

    def _capacity(self, ratio: int) -> int:
        if ratio not in self.capacity:
            raise RuntimeError(f"csa: no capacity known for the cache of ratio {ratio}: a capture needs "
                               "the groups a request can hold (CSAManager.build)")
        return self.capacity[ratio]

    def _cg_buffer(self, bucket: BucketKey, slot: int) -> torch.Tensor:
        key = (bucket, slot)
        buf = self._cg_rows.get(key)
        if buf is None:
            _, total = self._layout(bucket.num_tokens, bucket.bs, self._bucket_pages(bucket))
            buf = self._cg_rows[key] = torch.zeros(total, dtype=torch.long, device=self._device)
        return buf

    def build_accelerator_graph_buffers(self, slots: list[CGSlotSpec], max_bs: int,
                                        max_seq_len: int) -> None:
        for spec in slots:
            self._cg_buffer(spec.bucket, spec.slot)

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
        self._current = {
            label: self._rows(label, segments, ctx, step.context, step.decode_rows)
            for label, segments in labels.items()}
        return self._current

    def _rows(self, label: str, segments: list[Segment], ctx: StepContext,
              context: int | None = None, decode: bool = False) -> CSARows:
        lease = ctx.slot_lease
        graph = lease is not None and lease.bucket is not None
        tokens = self._tokens
        pos: list[int] = []
        row_req: list[int] = []
        first: list[int] = []
        end: list[int] = []
        spans: list[tuple[int, int, int]] = []
        for i, (rid, _, n) in enumerate(segments):
            # a capture's rows and a replay's padding rows hold nothing: they stand at token 0
            start = 0 if ctx.capture or ctx.is_padding_row(rid) else tokens.get(rid, 0)
            row0 = len(pos)
            spans.append((row0, n, start))
            pos.extend(range(start, start + n))
            row_req.extend([i] * n)
            first.extend([row0] * n)
            end.extend([start + n] * n)
        one_row_each = all(n == 1 for _, n, _ in spans)
        if graph:
            bucket = lease.bucket
            if not one_row_each or (len(pos), len(spans)) != (bucket.num_tokens, bucket.bs):
                raise RuntimeError(
                    f"csa: a captured step runs one row a request at its bucket's shape; {bucket} got "
                    f"{len(pos)} rows of {len(spans)} requests")
            if context is not None and max(end) > context:
                raise RuntimeError(
                    f"csa: a request reaches token {max(end)}, past the {context} tokens of context "
                    f"the {bucket} capture scores")

        def host(values) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.long)

        parts = [host(p) for p in (pos, row_req, first, end, [s for s, _, _ in spans], [n for _, n, _ in spans])]
        # every cache's page lists end to end in one list, each request's first entry by cache
        page_lists: list[torch.Tensor] = []
        entries = 0
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
            parts.append(host(indptr.paged_kv_indptr[:-1]) + entries)
            page_lists.append(host(indptr.paged_kv_indices))
            entries += page_lists[-1].numel()
        parts += [*page_lists, host([0])]

        used = sum(p.numel() for p in parts)
        if graph:
            # the slot's buffer, laid out for the most pages its requests can hold: the graph
            # recorded these addresses, and what lies past this step's page lists is not read
            sizes, _ = self._layout(len(pos), len(spans), self._bucket_pages(lease.bucket))
            if entries + 1 > sizes[-1]:
                raise RuntimeError(f"csa: {entries} pages overflow the {lease.bucket} capture's buffers")
            staged = self._cg_buffer(lease.bucket, lease.slot)
        else:
            sizes, _ = self._layout(len(pos), len(spans), entries)
            staged = torch.empty(used, dtype=torch.long, device=self._device)
        # one copy from pinned memory, which waits for nothing on the stream
        self._stager.copy_(staged[:used], torch.cat(parts).numpy())
        cuts = iter(staged.split(sizes))
        widths = None
        if graph:
            widths = {ratio: self._capacity(ratio) if context is None else min(self._capacity(ratio), context // ratio)
                      for ratio in self.config.caches}
        rows = CSARows(*(next(cuts) for _ in range(6)), caches={}, spans=spans, one_row_each=one_row_each,
                       widths=widths, decode=decode)
        starts = [next(cuts) for _ in self.config.caches]
        pages = next(cuts)
        for (ratio, page_start) in zip(self.config.caches, starts, strict=True):
            rows.caches[ratio] = CSACache(pages, page_start, self.page_sizes[ratio], self._num_pages(ratio))
        return rows

    def _num_pages(self, ratio: int) -> int:
        """The pages the ``ratio`` cache was built with."""
        return self._cache(ratio).layer_view(0).shape[0]

    def commit(self, step: CSAStep, ctx: StepContext) -> None:
        if ctx.capture:
            return  # a capture's rows are padding rows, which stay at token 0
        tokens = self._tokens
        for rid, _, span in step.segments:
            if span and not ctx.is_padding_row(rid):
                tokens[rid] = tokens.get(rid, 0) + span

    def take_back(self, rid, tokens: int, label: str | None = None) -> None:
        """Take request ``rid``'s last ``tokens`` tokens out of its context, between steps:
        drafts a decode step ran (``CSAStep.decode_rows``) and its caller rejected. Its count
        moves back, and each cache's length by the groups those tokens completed. The rows
        stay on their pages, past the lengths: no later row sees them, and the request's next
        tokens overwrite them. State a model keeps by position beside the caches (a window,
        an open group) is the model's to keep whole across a take-back."""
        have = self._tokens.get(rid, 0)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 <= tokens <= have:
            raise ValueError(f"csa: cannot take {tokens!r} tokens back from a request at {have}")
        for ratio, kv in self._kv.items():
            rows = have // ratio - (have - tokens) // ratio
            if rows:
                kv.correct_len(rid, label or _MAIN, -rows)
        if tokens:
            self._tokens[rid] = have - tokens

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
        width = rows.score_width(ratio)
        out = torch.full((q.shape[0], width), -torch.inf, dtype=torch.float32, device=q.device)
        group = torch.arange(width, device=q.device)
        heads = q.shape[1]
        with torch.autocast(q.device.type, enabled=False):
            if rows.one_row_each and self._score_kernel(q):
                # a decode step in bf16 on CUDA: one launch reads the keys' bytes in place
                from mstar.engine.resources.csa.kernels import decode_scores

                cache = rows.caches[ratio]
                out = decode_scores(q.contiguous(), w.to(q.dtype).contiguous(), view, cache.pages,
                                    cache.page_start, rows.row_req, seen, width)
                return out.masked_fill_(group[None, :] >= seen[:, None], -torch.inf)
            if rows.decode or rows.one_row_each:
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

    def _score_kernel(self, q: torch.Tensor) -> bool:
        """Whether the Triton kernel scores these queries: bf16 on CUDA, against keys held as
        FP4 bytes with a power-of-two scale per 32 (its arithmetic is the torch path's in bf16;
        in fp32 the heads' sum has an order, and the torch path keeps its own)."""
        fmt, dim = self.config.key_rows, q.shape[-1]
        return (self.config.score_kernel and q.is_cuda and q.dtype == torch.bfloat16 and fmt is not None
                and fmt.block == 32 and not fmt.e4m3_scale and dim >= 16 and dim & (dim - 1) == 0)

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
        if self.config.attend_kernel and q.is_cuda and q.dtype == torch.bfloat16 and width >= 16 \
                and width & (width - 1) == 0:
            return self._attend_on_kernel(q, sink, window, from_window, from_cache, latent)
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

    def _attend_on_kernel(self, q: torch.Tensor, sink: torch.Tensor, window: torch.Tensor,
                          from_window: torch.Tensor, from_cache: torch.Tensor | None,
                          latent: torch.Tensor | None) -> torch.Tensor:
        """``attend`` with the softmax on the Triton kernel: each row's entries gathered to bf16,
        which holds every window row and every FP4 latent exactly, in their places."""
        from mstar.engine.resources.csa.kernels import attend

        count, heads, width = q.shape
        block = self.config.softmax_block
        spare = -from_window.shape[1] % block
        chunk = max(1, self.config.attend_bytes // (2 * width * (from_window.shape[1] + spare)))
        out = torch.empty_like(q)
        for c0 in range(0, count, chunk):
            c1 = min(c0 + chunk, count)
            idx = from_window[c0:c1]
            valid = idx >= 0
            entries = window[idx.clamp_min(0)]
            if from_cache is not None:
                picked = from_cache[c0:c1]
                live = picked >= 0
                entries = torch.where(live[..., None], _load(
                    latent[picked.clamp_min(0)], self.config.latent_rows, width, q.dtype), entries)
                valid = valid | live
            if spare:
                entries, valid = F.pad(entries, (0, 0, 0, spare)), F.pad(valid, (0, spare))
            out[c0:c1] = attend(q[c0:c1].contiguous(), entries.contiguous(), valid.contiguous(), sink.float(),
                                self.config.sm_scale, block)
        return out
