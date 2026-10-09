"""The KV plan: what `KVManager.plan` hands its dependents.

Attention and positions read this and nothing else of the KV resource — the
packed ordering, one view per (request, label) stream, and the index tensors a
paged-attention wrapper plans against. Kept apart from the manager so a
consumer depends on the contract rather than on the cache.
"""

import itertools
from dataclasses import dataclass
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

    def to_device(self, device: torch.device):
        return PagedIndptrs(
            qo_indptr=self.qo_indptr.to(device, non_blocking=True),
            paged_kv_indptr=self.paged_kv_indptr.to(device, non_blocking=True),
            paged_kv_indices=self.paged_kv_indices.to(device, non_blocking=True),
            paged_kv_last_page_len=self.paged_kv_last_page_len.to(device, non_blocking=True),
            kv_lens=self.kv_lens,
        )

    def to_kwargs_dict(self):
        return dict(
            qo_indptr=self.qo_indptr,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_last_page_len=self.paged_kv_last_page_len,
            kv_lens=self.kv_lens,
        )


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

    ``take`` fills the next buffer and returns a view of it. A buffer is
    rewritten only after the copies out of it have completed: the stream a
    plan ran on records an event when the next ``take`` happens on that
    stream (by then the plan's copies were issued), and a slot waits on its
    event before reuse. Plans do not overlap, so one pending slot suffices.
    Without CUDA the buffers are plain tensors and nothing is recorded.
    """

    def __init__(self, numel: int = 4096, depth: int = 8):
        self._depth = depth
        self._numel = 0
        self._pinned = torch.cuda.is_available()
        self._bufs: list[torch.Tensor] = []
        self._events: list = [None] * depth
        self._streams: list = [None] * depth
        self._next = 0
        self._last: int | None = None
        self._grow(numel)

    def _grow(self, numel: int) -> None:
        numel = max(numel, self._numel * 2, 256)
        self._bufs = [
            torch.empty(numel, dtype=torch.int32, pin_memory=self._pinned)
            for _ in range(self._depth)
        ]
        self._numel = numel

    def take(self, values) -> torch.Tensor:
        n = len(values)
        if n > self._numel:
            self._grow(n)
        if self._pinned:
            stream = torch.cuda.current_stream()
            if self._last is not None and self._streams[self._last] is not stream:
                # the previous plan's copies were issued on another stream
                # (an inline plan on the gpu thread after a pre-plan): record
                # there, where they are
                ev = self._events[self._last] or torch.cuda.Event()
                self._events[self._last] = ev
                ev.record(self._streams[self._last])
            elif self._last is not None:
                ev = self._events[self._last] or torch.cuda.Event()
                self._events[self._last] = ev
                ev.record(stream)
            i = self._next
            ev = self._events[i]
            if ev is not None:
                ev.synchronize()
            self._streams[i] = stream
        else:
            i = self._next
        self._next = (i + 1) % self._depth
        self._last = i
        out = self._bufs[i][:n]
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
    device buffer) is plain tensor behaviour.
    """

    def __iter__(self):
        return iter(self.numpy())


def _host_lens(values: list[int]) -> torch.Tensor:
    return _int32_tensor(values).as_subclass(HostLens)


@dataclass
class KVPlanOutput:
    """
    Output of KVManager.plan for a single label
    """
    cpu_indptrs: PagedIndptrs
    # packing in plan order w/h 1 view per segment covered by plan
    views: list[SequenceView]
    # only the packed write addressing needs these on device, and only the
    # resource that builds them reads them; see KVManager._setup_plan_states
    cuda_indptrs: PagedIndptrs | None = None
    # a decode step planned off the previous one (KVManager's plan cache):
    # each row's write page and offset in it, so staging need not walk the views
    decode_pages: np.ndarray | None = None
    decode_offsets: np.ndarray | None = None

    def get_total_len(self):
        return int(self.cpu_indptrs.qo_indptr[-1])

    @property
    def is_decode(self) -> bool:
        if self.decode_pages is not None:
            return True
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
