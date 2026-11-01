"""The KV plan: what `KVManager.plan` hands its dependents.

Attention and positions read this and nothing else of the KV resource — the
packed ordering, one view per (request, label) stream, and the index tensors a
paged-attention wrapper plans against. Kept apart from the manager so a
consumer depends on the contract rather than on the cache.
"""

import itertools
from typing import NamedTuple

import numpy as np
import torch

from mstar.engine.resources.step import Segment


def group_by_plan_label(
    segments: tuple[Segment, ...],
    combined_labels: dict[tuple[str, ...], str],
) -> dict[str, list[Segment]]:
    """Segments per plan label in order of packed forward view

    combined plan concats source labels in label major order. standalone keeps
    og batch order. KV should be sole producer of this ordering and eveyrone else
    will read `plan` output of KV

    NOTE: combined key with a source label with no segments in step will cause KeyError
    """
    label_to_segments: dict[str, list[Segment]] = {}
    for segment in segments:
        label_to_segments.setdefault(segment.label, []).append(segment)

    sources = set(itertools.chain.from_iterable(combined_labels))
    grouped: dict[str, list[Segment]] = {
        plan_label: [
            segment
            for label in source_labels
            for segment in label_to_segments[label]
        ]
        for source_labels, plan_label in combined_labels.items()
    }

    grouped.update(
        (label, label_segments)
        for label, label_segments in label_to_segments.items()
        if label not in sources
    )
    return grouped


class SequenceView(NamedTuple):
    """One (request, label) stream as this step sees it. A NamedTuple because
    it is built per segment per step and never mutated."""
    request_id: str
    label: str
    page_idxs: list[int]
    length: int # already resident + to_compute
    to_compute: int # new for this step
    start: int = 0
    generation: int = 0

    def last_page_len(self, page_size: int) -> int:
        return (self.start + self.length)  % page_size


class PagedIndptrs(NamedTuple):
    """The four int32 index tensors a FlashInfer prefill/decode wrapper's
    ``plan`` consumes, built on CPU (so wrapper.plan's ``.to("cpu")`` is a
    no-op — see ``FlashInferAttentionManager.plan``)."""

    # TODO: qo_indptr is only needed for prefill; we should avoid computing it
    # (+ doing the H2D) if possible during decode
    qo_indptr: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    # Each row's KV length, host-only, so FlashInfer's decode plan need not
    # rebuild it with torch ops.
    kv_lens: torch.Tensor | None = None

    def to_kwargs_dict(self):
        return dict(
            qo_indptr=self.qo_indptr,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_last_page_len=self.paged_kv_last_page_len,
            kv_lens=self.kv_lens,
        )


def context_only_views(
    views: list[SequenceView], query_spans: list[int], page_size: int,
) -> list[SequenceView]:
    """The views of a step's streams without the tokens the step appends: ``length`` shrinks by
    ``to_compute`` (the pages beyond it dropped) and ``to_compute`` becomes the row's query count.
    For attention over the stored context alone (``AttentionStep.context_only``)."""
    out = []
    for view, queries in zip(views, query_spans, strict=True):
        length = view.length - view.to_compute
        pages = -(-(view.start + length) // page_size) if length > 0 else 0
        out.append(view._replace(page_idxs=view.page_idxs[:pages], length=length, to_compute=queries))
    return out


def build_paged_indptrs(
    segments: list[SequenceView],
    page_size: int,
    ring: "PinnedIndexRing | None" = None,
) -> PagedIndptrs:
    # Most of this is the five list-to-tensor conversions; they go through
    # numpy, which builds an int32 array from a list of ints in about half
    # the time `torch.tensor(list)` takes (42 -> 29 us at bs=32, 10 pages a
    # row). Holding `CacheStream.page_indices` as a numpy array would cut the
    # pages copy further, but it also crosses publish/ZMQ, offload/reload,
    # forks and the CPU pool.
    qo_indptr = [0]
    kv_indptr = [0]
    all_pages: list[int] = []
    last_page_lens: list[int] = []
    kv_lens: list[int] = []
    for s in segments:
        qo_indptr.append(qo_indptr[-1] + s.to_compute)
        all_pages.extend(s.page_idxs)
        num_pages = len(s.page_idxs)
        kv_indptr.append(kv_indptr[-1] + num_pages)
        last = s.last_page_len(page_size) or page_size
        last_page_lens.append(last)
        # FlashInfer's `get_seq_lens`, on the ints already here
        kv_lens.append(max(num_pages - 1, 0) * page_size + last)
    make = ring.take if ring is not None else _int32_tensor
    return PagedIndptrs(
        qo_indptr=make(qo_indptr),
        paged_kv_indptr=make(kv_indptr),
        paged_kv_indices=make(all_pages),
        paged_kv_last_page_len=make(last_page_lens),
        kv_lens=_host_lens(kv_lens),
    )


def _int32_tensor(values: list[int]) -> torch.Tensor:
    return torch.from_numpy(np.array(values, dtype=np.int32))


class PinnedIndexRing:
    """A ring of page-locked int32 host buffers for the index arrays a plan
    hands FlashInfer, so its ``copy_(non_blocking=True)`` calls are really
    asynchronous (from pageable memory the driver stages each one).

    The ring holds ``depth`` buffer sets, one per plan. ``begin_plan`` moves to
    the next set, and each ``take`` until the next ``begin_plan`` fills a buffer
    of that set no other take uses (the set grows if needed), so every label
    of a plan keeps its arrays (four takes each) however many labels there
    are. A set is rewritten only after the copies out of it have completed:
    the next ``begin_plan`` records an event on the stream the plan ran on (by
    then the plan's copies, and its consumers', were issued), and a set waits
    on its event before reuse. Plans do not overlap, so one pending set
    suffices. Without CUDA the buffers are plain tensors and nothing is
    recorded.
    """

    def __init__(self, numel: int = 4096, depth: int = 2, per_plan: int = 4):
        self._depth = depth
        self._numel = max(numel, 256)
        self._pinned = torch.cuda.is_available()
        self._sets: list[list[torch.Tensor]] = [
            [self._alloc(self._numel) for _ in range(per_plan)] for _ in range(depth)
        ]
        self._events: list = [None] * depth
        self._streams: list = [None] * depth
        self._set = 0
        self._used = 0

    def _alloc(self, numel: int) -> torch.Tensor:
        return torch.empty(numel, dtype=torch.int32, pin_memory=self._pinned)

    def begin_plan(self) -> None:
        cur = self._set
        nxt = (cur + 1) % self._depth
        if self._pinned:
            if self._streams[cur] is not None:
                ev = self._events[cur] or torch.cuda.Event()
                self._events[cur] = ev
                ev.record(self._streams[cur])
            ev = self._events[nxt]
            if ev is not None:
                ev.synchronize()
            self._streams[nxt] = torch.cuda.current_stream()
        self._set = nxt
        self._used = 0

    def take(self, values) -> torch.Tensor:
        n = len(values)
        bufs = self._sets[self._set]
        i = self._used
        if i == len(bufs):
            bufs.append(self._alloc(max(n, self._numel)))
        elif n > bufs[i].numel():
            bufs[i] = self._alloc(max(n, bufs[i].numel() * 2))
        self._used = i + 1
        out = bufs[i][:n]
        out.numpy()[:] = values
        return out


class HostLens(torch.Tensor):
    """A host int32 row whose iteration yields numpy scalars.

    FlashInfer's decode ``plan`` takes the longest row with the Python builtin
    ``max(seq_lens)`` and ``.item()`` on the result; iterating a tensor hands
    out one 0-d tensor per row, about 2 us each (0.35 ms at 128 rows, on the
    plan thread every decode step). Iterating the numpy view costs 0.1 us a
    row and the numpy scalar has ``.item()`` too. Everything else FlashInfer
    does with the row (``.cpu()``, ``.min()``, ``len``, a ``copy_`` into its
    device buffer) is plain tensor behaviour, so torch ops skip the subclass
    dispatch (about 6 us a call) and treat it as a plain tensor.
    """

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __iter__(self):
        return iter(self.numpy())


def _host_lens(values: list[int]) -> torch.Tensor:
    return _int32_tensor(values).as_subclass(HostLens)


class KVPlanOutput:
    """
    Output of KVManager.plan for a single label
    """

    __slots__ = (
        "cpu_indptrs", "_views", "_view_builder", "decode_pages", "decode_offsets",
        "rows", "_decode", "columns",
    )

    def __init__(
        self, cpu_indptrs: PagedIndptrs, views: "list[SequenceView] | None" = None,
        decode_pages=None, decode_offsets=None,
        rows=None, view_builder=None, decode: bool | None = None, columns=None,
    ):
        self.cpu_indptrs = cpu_indptrs
        # packing in plan order w/h 1 view per segment covered by plan; a plan
        # off the KV manager's cache hands a builder instead and the views are
        # made on first use (the FlashInfer path never asks for them)
        self._views = views
        self._view_builder = view_builder
        # a decode step planned off the previous one (KVManager's plan cache):
        # each row's write page and offset in it, so staging need not walk the views
        self.decode_pages = decode_pages
        self.decode_offsets = decode_offsets
        # (request_id, label, to_compute) per view, for consumers that need no
        # pages or lengths (the position resource); None means read the views
        self.rows = rows
        # whether every row computes one token, when the plan already knows;
        # None reads the views
        self._decode = decode
        # the plan's host arrays (``PlanColumns``), when it was built from
        # columns. Unlike ``cpu_indptrs``, which may sit in a pinned ring a
        # later plan rewrites, they stay valid as long as the output does.
        self.columns = columns

    @property
    def views(self) -> "list[SequenceView]":
        if self._views is None:
            self._views = self._view_builder()
        return self._views

    @views.setter
    def views(self, value):
        self._views = value

    def get_total_len(self):
        return int(self.cpu_indptrs.qo_indptr[-1])

    @property
    def is_decode(self) -> bool:
        if self.decode_pages is not None:
            return True
        if self._decode is not None:
            return self._decode
        return all(view.to_compute == 1 for view in self.views)


class KVPlanOutputs(dict[str, KVPlanOutput]):
    """Plan label -> output, plus the label forks of the step it came from.

    A dict subclass so consumers that only want the per-label plans keep
    reading it as the mapping it is. Positions need the forks too — a fork
    target inherits the source's counter the way it inherits its pages — and
    a fork target is not necessarily a segment of the step, so it has
    nowhere else to ride.
    """

    __slots__ = ("pre_forks", "post_forks")

    def __init__(
        self,
        mapping: dict[str, KVPlanOutput] | None = None,
        pre_forks: tuple[tuple[str, str], ...] = (),
        post_forks: tuple[tuple[str, str], ...] = (),
    ):
        super().__init__(mapping or {})
        # applied by `plan`; the consumer mirrors them at plan time
        self.pre_forks = pre_forks
        # applied by `commit`, after this step's spans land
        self.post_forks = post_forks


# Page the padding tail of a captured step scatters into. Reserved at
# construction so no request is ever handed it; see KVPlanState.copy_.
SINK_PAGE = 0
