"""KV cache storage resources.

Storage divides in two. The arena is the physical pool: the backing tensor
and the page allocator over it. The pool is per-request accounting against
an arena: which pages a stream holds, its stored length, its position
counter. Several pools may share one arena; nothing above a pool sees the
arena.
"""

import torch

from mstar.conductor.request_info import SequenceInfo
from mstar.engine.cpu_page_pool import CPUPagePool
from mstar.engine.kv_store import (
    PageAllocator,
    PagedAllocationManager,
    PositionInfo,
    StoreWritePolicy,
)
from mstar.engine.resources.base import Reservation, Segment, SequenceView


class PageArena:
    """Physical page storage: a backing K/V tensor and the allocator over
    its pages. Pools draw pages from here and return them here."""

    def __init__(
        self,
        tensor: torch.Tensor | None,
        allocator: PageAllocator,
        page_size: int,
    ):
        self.tensor = tensor
        self.allocator = allocator
        self.page_size = page_size

    def allocate(self, n: int) -> list[int]:
        return self.allocator.allocate(n)

    def try_allocate(self, n: int) -> list[int] | None:
        return self.allocator.try_allocate(n)

    def free(self, pages: list[int]) -> None:
        self.allocator.free(pages)

    @property
    def num_free(self) -> int:
        return self.allocator.num_free

    @property
    def total_pages(self) -> int:
        return self.allocator.max_num_pages


class ScratchKVPool:
    """Fixed-shape scratch KV storage with a trivial lifecycle: no admit,
    no publish, no per-request lifetime. The tensor is overwritten every
    step and slot-indexed by batch position. Models that need a depth
    loop's per-step cache take it from here instead of allocating their
    own tensors on the side."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor


class KVCachePool:
    """Per-request cache accounting behind the segment lifecycle.

    ``admit`` reserves capacity for a segment and reports what is already
    resident; ``view`` describes the stream the segment extends; ``commit``
    advances the stream's stored length and position counter. The
    ``PagedAllocationManager`` this pool fronts remains the storage owner
    (its lock, its request states, its transfer machinery); the pool is the
    surface planning code goes through, so callers stop reaching into
    request-state internals.

    ``cpu_pool`` is the pool's optional CPU tier: with one attached, whole
    requests can be offloaded there and reloaded later, and the pool
    answers the offload questions eviction policy asks.
    """

    def __init__(
        self,
        manager: PagedAllocationManager,
        cpu_pool: CPUPagePool | None = None,
    ):
        self._manager = manager
        self._cpu_pool = cpu_pool
        self._arena = PageArena(
            tensor=manager.kv_cache,
            allocator=manager.page_allocator,
            page_size=manager.config.page_size,
        )

    @property
    def page_size(self) -> int:
        return self._arena.page_size

    @property
    def num_free_pages(self) -> int:
        return self._arena.num_free

    @property
    def total_pages(self) -> int:
        return self._arena.total_pages

    def admit(self, segment: Segment) -> Reservation:
        """Reserve pages so the segment's stream can hold its history plus
        this segment's span. Raises ``AllocationFailedError`` when the arena
        cannot supply the pages; a zero-span segment reserves nothing."""
        state = self._manager.get_state(segment.request_id, segment.label)
        resident = state.seq_len
        self._manager.alloc(
            segment.request_id, segment.label, resident + segment.span
        )
        return Reservation(
            resident=resident,
            to_compute=segment.span,
            pending=state.read_in_progress,
        )

    def view(self, segment: Segment) -> SequenceView:
        """The stream as this step's plans must see it: every page backing
        it and the extent those pages cover once the segment's span lands.
        Call after ``admit`` for spans that need new pages."""
        state = self._manager.get_state(segment.request_id, segment.label)
        return SequenceView(
            pool=self,
            page_indices=tuple(state.page_indices),
            start=0,
            length=state.seq_len + segment.span,
        )

    def commit(self, segment: Segment, pos_advance: int | None = None) -> None:
        """Record that the segment's span was computed: stored length grows
        by the span, the position counter by ``pos_advance`` (defaults to
        the span)."""
        state = self._manager.get_state(segment.request_id, segment.label)
        state.seq_len += segment.span
        state.position_id_start += (
            segment.span if pos_advance is None else pos_advance
        )

    def positions(self, request_id: str, label: str) -> int:
        """Current position counter for one stream, read-only."""
        return self._manager.get_state(request_id, label).position_id_start

    def pos_info(self, request_id: str, label: str) -> PositionInfo:
        """One stream's stored length and position counter, read-only."""
        return self._manager.get_state(request_id, label).get_pos_info()

    def labels(self, request_id: str) -> list[str]:
        """The cache streams currently existing for one request."""
        return self._manager.get_labels(request_id)

    def add_request(self, request_id: str, labels: list[str]) -> None:
        """Open per-request accounting with the given initial streams."""
        self._manager.add_request(request_id, labels)

    def remove_request(self, request_id: str) -> None:
        """Drop a request's accounting on every tier and free its pages."""
        if self._cpu_pool is not None:
            self._cpu_pool.remove_request(request_id)
        self._manager.remove_request(request_id)

    def set_write_policy(self, policy: StoreWritePolicy) -> None:
        """Set whether committed pages are pushed to the distributed store."""
        self._manager.write_policy = policy

    @property
    def supports_offload(self) -> bool:
        """Whether a CPU tier is attached to offload requests into."""
        return self._cpu_pool is not None

    def offload_candidates(self) -> list[tuple[str, int]]:
        """``(request_id, gpu_pages_held)`` for every request holding GPU
        pages, for eviction-victim selection. Empty without a CPU tier."""
        if self._cpu_pool is None:
            return []
        out: list[tuple[str, int]] = []
        for request_id, states in self._manager.request_states.items():
            total_pages = sum(len(s.page_indices) for s in states.values())
            if total_pages > 0:
                out.append((request_id, total_pages))
        return out

    def offload(self, request_id: str) -> int:
        """Move all of one request's streams to the CPU tier. Returns the
        number of GPU pages freed (0 without a CPU tier)."""
        if self._cpu_pool is None:
            return 0
        return self._manager.offload_request(request_id, self._cpu_pool)

    def reload(self, request_id: str) -> bool:
        """Bring an offloaded request back to GPU. False when the request
        is not offloaded or GPU pages are insufficient right now."""
        if self._cpu_pool is None or not self._cpu_pool.is_offloaded(request_id):
            return False
        try:
            self._manager.reload_request(request_id, self._cpu_pool)
            return True
        except RuntimeError:
            return False

    def is_offloaded(self, request_id: str) -> bool:
        """Whether the request currently lives on the CPU tier."""
        return self._cpu_pool is not None and self._cpu_pool.is_offloaded(request_id)

    def fork(
        self,
        request_id: str,
        from_label: str,
        to_label: str,
        realloc: bool = False,
    ) -> None:
        """Make one stream's content the state of another stream of the same
        request: reserve the destination's pages, mirror the stored length
        and position counter, and copy the page data. With ``realloc`` the
        destination is reset first; otherwise only the pages past its
        current length are copied (an incremental top-up). Allocates, so it
        can fail like any admit."""
        manager = self._manager
        from_state = manager.get_state(request_id, from_label)

        if realloc:
            manager.reset_label(request_id, to_label)

        to_state = manager.get_state(request_id, to_label)
        start_pos = to_state.seq_len // self.page_size
        manager.alloc(request_id, to_label, seq_len=from_state.seq_len)

        to_state.seq_len = from_state.seq_len
        to_state.position_id_start = from_state.position_id_start

        tensor = self._arena.tensor
        for src_page, dst_page in zip(
            from_state.page_indices[start_pos:],
            to_state.page_indices[start_pos:],
            strict=True,
        ):
            tensor[:, dst_page] = tensor[:, src_page]

    def retrieve(self, request_id: str, label: str, seq_info: SequenceInfo) -> None:
        """Bring one stream's published state into residency, synchronously.
        Allocates the pages the published length needs, so it can fail like
        any admit."""
        self._manager.sync_retrieve(request_id, label, seq_info)

    def admit_retrieve(
        self, request_id: str, label: str, seq_info: SequenceInfo,
    ) -> Reservation:
        """Start bringing one stream's published state into residency and
        report the admit outcome: ``pending`` stays True while pages are
        in flight, and polling again re-checks without restarting the
        transfer. Allocates on first call, so it can fail like any
        admit."""
        self._manager.start_async_retrieve(request_id, label, seq_info)
        ready = self._manager.check_retrieve_ready(request_id, label)
        return Reservation(
            resident=self._manager.get_state(request_id, label).seq_len,
            to_compute=0,
            pending=not ready,
        )

    def publish(self, request_id: str) -> dict[str, SequenceInfo]:
        """Describe the request's durable streams to another process, one
        ``SequenceInfo`` per label. Waits out any in-flight retrieves first
        so the description never covers half-arrived pages."""
        manager = self._manager
        transfer_info = manager.get_kv_transfer_info()
        per_label_seq_info: dict[str, SequenceInfo] = {}
        for label in list(manager.request_states.get(request_id, {})):
            manager.wait_for_retrieves(request_id, label)

            state = manager.get_state(request_id, label)
            per_label_seq_info[label] = SequenceInfo(
                seq_len=state.seq_len,
                pos_id=state.position_id_start,
                latest_kv_transfer_info=transfer_info,
                page_indices=state.page_indices,
            )
        return per_label_seq_info
