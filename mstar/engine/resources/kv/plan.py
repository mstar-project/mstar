"""The KV plan: what `KVManager.plan` hands its dependents.

Attention and positions read this and nothing else of the KV resource — the
packed ordering, one view per (request, label) stream, and the index tensors a
paged-attention wrapper plans against. Kept apart from the manager so a
consumer depends on the contract rather than on the cache.
"""

import itertools
from dataclasses import dataclass
from typing import NamedTuple

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

    def to_device(self, device: torch.device):
        return PagedIndptrs(
            qo_indptr=self.qo_indptr.to(device, non_blocking=True),
            paged_kv_indptr=self.paged_kv_indptr.to(device, non_blocking=True),
            paged_kv_indices=self.paged_kv_indices.to(device, non_blocking=True),
            paged_kv_last_page_len=self.paged_kv_last_page_len.to(device, non_blocking=True),
        )

    def to_kwargs_dict(self):
        return dict(
            qo_indptr=self.qo_indptr,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_last_page_len=self.paged_kv_last_page_len
        )


def build_paged_indptrs(
    segments: list[SequenceView],
    page_size: int,
) -> PagedIndptrs:
    # TODO: ~90% of this is the four `torch.tensor(list)` conversions. Holding
    # `CacheStream.page_indices` as a doubling int32 numpy array instead makes
    # the pages copy a memcpy: measured 52us -> 13us at bs=16/1024 pages, for
    # +0.3us on the per-step append. Not done because page_indices also crosses
    # publish/ZMQ, offload/reload, forks and the CPU pool.
    qo_indptr = [0]
    kv_indptr = [0]
    all_pages: list[int] = []
    last_page_lens: list[int] = []
    for s in segments:
        qo_indptr.append(qo_indptr[-1] + s.to_compute)
        all_pages.extend(s.page_idxs)
        kv_indptr.append(kv_indptr[-1] + len(s.page_idxs))
        last_page_lens.append(s.last_page_len(page_size) or page_size)
    return PagedIndptrs(
        qo_indptr=torch.tensor(qo_indptr, dtype=torch.int32),
        paged_kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32),
        paged_kv_indices=torch.tensor(all_pages, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor(last_page_lens, dtype=torch.int32),
    )


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

    def get_total_len(self):
        return int(self.cpu_indptrs.qo_indptr[-1])

    @property
    def is_decode(self) -> bool:
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
