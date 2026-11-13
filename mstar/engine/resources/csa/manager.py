"""Compressed sparse attention (CSA) over a node's paged latent cache: the resource.

DeepSeek-V4.1's layers attend two kinds of rows in one softmax: a short window of their own
recent rows, which the model passes in, and a few rows of a compressed cache shared by several
layers, picked per query by an indexer. A compressed source pools ``ratio`` tokens into one
latent, and its latent and index key live in the cache row of the group's last token, so the
cache keeps the token page table and a group is visible to a query once the query has passed
that token. A learned sink logit per head joins the softmax's denominator.

A step declares ``CSAStep()``; ``plan`` reads the step's rows off the KV plan (``CSARows``).
The layers write keys, score, select and attend through the resource. This is the reference in
torch: eager only (``can_capture`` is False), every row scored against its request's keys.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.csa.config import INDEX_PLANES, CSAConfig, CSASpec, CSAStep
from mstar.engine.resources.step import StepContext

_MAIN = "main"


@dataclass
class CSARows:
    """A step's rows under one label, from the KV plan. Device, per row: ``pos`` its token
    position, ``row_req`` its request (the step's order), ``first`` the step row its request
    starts at, ``end`` the position after its request's last row. Per request: ``req_first``
    and ``req_rows`` (its rows in the step), ``page_start`` its first entry in ``pages``, the
    flat page list. Host: ``spans``, each request's ``(first row, rows, first position)``."""

    pos: torch.Tensor
    row_req: torch.Tensor
    first: torch.Tensor
    end: torch.Tensor
    req_first: torch.Tensor
    req_rows: torch.Tensor
    pages: torch.Tensor
    page_start: torch.Tensor
    page_size: int
    spans: list[tuple[int, int, int]]

    def slots(self, req: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Cache slots (``page * page_size + offset``) of ``positions`` of requests ``req``."""
        page = self.pages[self.page_start[req] + positions // self.page_size]
        return page * self.page_size + positions % self.page_size

    def groups(self, ratio: int) -> torch.Tensor:
        """Per row, the compressed groups of a ``ratio`` source it sees: those whose last
        token is at or before it."""
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
    """CSA for one node's layers over their compressed cache (``config.kv_cache``)."""

    @classmethod
    def build(cls, spec: CSASpec, info: EngineResourceInfo):
        kv_config = info.dependency(spec.config.kv_cache).config
        planes = kv_config.aux_planes.get(INDEX_PLANES)
        if planes is None or planes.width != spec.config.index_head_dim:
            raise ValueError(
                f"CSA keeps its {spec.config.index_head_dim}-wide index keys on "
                f"{spec.config.kv_cache}'s pages: declare its aux_planes[{INDEX_PLANES!r}] that wide")
        if kv_config.qk_rope_head_dim:
            raise ValueError(f"CSA attends whole latents: {spec.config.kv_cache} must declare no "
                             f"rope slot, not qk_rope_head_dim={kv_config.qk_rope_head_dim}")
        return cls(spec.config, kv_config.tokens_per_page, info.device)

    def __init__(self, config: CSAConfig, page_size: int, device: torch.device):
        self.config = config
        self.page_size = page_size
        self._device = device
        self._kv = None  # the KV resource whose pages it reads (`bind_dependencies`)
        self._current: dict[str, CSARows] = {}

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

    def bind_dependencies(self, resources: Mapping[str, Resource]) -> None:
        self._kv = resources[self.config.kv_cache]

    def can_capture(self) -> bool:
        # the torch reference sizes its launches on the host
        return False

    # -- the step ----------------------------------------------------------------------------

    def plan(self, step: CSAStep, ctx: StepContext) -> dict[str, CSARows]:
        self._current = {label: self._rows(kv_out)
                         for label, kv_out in ctx.plan_results[self.config.kv_cache].items()}
        return self._current

    def _rows(self, kv_out) -> CSARows:
        pos: list[int] = []
        row_req: list[int] = []
        first: list[int] = []
        end: list[int] = []
        spans: list[tuple[int, int, int]] = []
        for i, view in enumerate(kv_out.views):
            n, start = view.to_compute, len(pos)
            spans.append((start, n, view.length - n))
            pos.extend(range(view.length - n, view.length))
            row_req.extend([i] * n)
            first.extend([start] * n)
            end.extend([view.length] * n)
        indptr = kv_out.cpu_indptrs.paged_kv_indptr

        def dev(values) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.long).to(self._device)

        return CSARows(
            pos=dev(pos), row_req=dev(row_req), first=dev(first), end=dev(end),
            req_first=dev([s for s, _, _ in spans]), req_rows=dev([n for _, n, _ in spans]),
            pages=dev(kv_out.cpu_indptrs.paged_kv_indices), page_start=dev(indptr[:-1]),
            page_size=self.page_size, spans=spans)

    # -- what the layers and the submodule call ----------------------------------------------

    def current(self, label: str | None = None) -> CSARows:
        label = label or _MAIN
        if label not in self._current:
            raise KeyError(f"csa: no plan for label {label!r}; the step declares CSAStep()")
        return self._current[label]

    def _need_kv(self):
        if self._kv is None:
            raise RuntimeError(f"csa: bind_dependencies never handed it {self.config.kv_cache}")
        return self._kv

    def write_index(self, plane: int, keys: torch.Tensor, label: str | None = None) -> None:
        """Write the step's index keys ``[rows, index_head_dim]`` to ``plane`` at the step's
        slots. A row that completes no group writes a key nothing reads."""
        self._need_kv().write_aux(INDEX_PLANES, keys, layer_idx=plane, label=label)

    def scores(self, plane: int, q: torch.Tensor, w: torch.Tensor, ratio: int,
               label: str | None = None) -> torch.Tensor:
        """``[rows, groups]`` fp32 indexer scores of each row against its request's keys in
        ``plane``, ``sum_h w[h] * relu(q[h] . key)``; -inf past the groups the row sees.
        ``q (rows, heads, index_head_dim)``, ``w (rows, heads)`` with every scale folded in."""
        rows = self.current(label)
        view = self._need_kv().aux_view(INDEX_PLANES, plane)
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
                keys = flat[rows.slots(req_ids, group[:count] * ratio + ratio - 1)].float()
                chunk = max(1, self.config.score_bytes // (4 * heads * count))
                for c0 in range(r0, r0 + n, chunk):
                    c1 = min(c0 + chunk, r0 + n)
                    dots = torch.einsum("rhd,jd->rhj", q[c0:c1].float(), keys).relu_()
                    out[c0:c1, :count] = torch.einsum("rh,rhj->rj", w[c0:c1].float(), dots)
        return out.masked_fill_(group[None, :] >= seen[:, None], -torch.inf)

    def candidates(self, scores: torch.Tensor, ratio: int, topk_blocks: int, block_size: int,
                   label: str | None = None) -> torch.Tensor:
        """``candidate_blocks`` of this step's ``scores``: the mask a later ``select`` keeps to."""
        return candidate_blocks(scores, self.current(label).groups(ratio), topk_blocks, block_size)

    def select(self, scores: torch.Tensor, ratio: int, keep: torch.Tensor | None = None,
               label: str | None = None) -> torch.Tensor:
        """``[rows, k]`` cache slots of each row's top ``topk`` groups by ``scores`` (within
        ``keep``, a ``candidates`` mask; equal scores to the earlier group), -1 where a row
        sees fewer."""
        rows = self.current(label)
        if keep is not None:
            scores = scores.masked_fill(~keep[:, :scores.shape[1]], -torch.inf)
        values, indices = top_scores(scores, min(self.config.topk, scores.shape[1]))
        live = values > -torch.inf
        token = torch.where(live, indices * ratio + ratio - 1, 0)
        return torch.where(live, rows.slots(rows.row_req[:, None], token), -1)

    def attend(self, q: torch.Tensor, sink: torch.Tensor, window: torch.Tensor,
               window_idx: torch.Tensor, plane: int | None = None,
               slots: torch.Tensor | None = None) -> torch.Tensor:
        """``[rows, heads, D]``: each row's heads over its window rows ``window[window_idx]``
        (``window (m, D)``, ``window_idx (rows, w)``, -1 none) and its ``slots`` of ``plane``'s
        latents (``select``'s, -1 none), one softmax at ``config.sm_scale`` with ``sink
        (heads,)`` in its denominator. A row with nothing to attend gets zeros."""
        width = q.shape[-1]
        latent = None
        if slots is not None:
            view = self._need_kv().layer_view(plane)
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
