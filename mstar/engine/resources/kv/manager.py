import itertools
import logging
import os
import threading
from collections import Counter
from collections.abc import Callable, Collection
from concurrent.futures import Future, wait
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

import numpy as np
import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import (
    AttentionResource,
    CachedPrefix,
    CGSlotSpec,
    EngineResourceInfo,
    KeyedWalks,
    PublishedInfo,
    Rebuild,
)
from mstar.engine.resources.kv.cache import KVCache, PageAllocator
from mstar.engine.resources.kv.config import (
    KVLayout,
    KVReqConfig,
    KVSpec,
    KVStep,
    PagedKVConfig,
    PrefixSpan,
    RetentionPolicy,
    StreamLayout,
    rebuild_refusal,
)
from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool, PageCopies
from mstar.engine.resources.kv.host_writes import HostWrites
from mstar.engine.resources.kv.keys import PageItem, chain, fingerprint, page_items, page_key
from mstar.engine.resources.kv.plan import (
    SINK_PAGE,
    HostLens,
    KVPlanOutput,
    KVPlanOutputs,
    PagedIndptrs,
    PinnedIndexRing,
    SequenceView,
    build_paged_indptrs,
    group_by_plan_label,
)
from mstar.engine.resources.kv.prefix_index import PrefixIndex
from mstar.engine.resources.kv.transfer import KVTransferInfo, KVTransferManager, TransferEngineInfo
from mstar.engine.resources.step import (
    ADMIT_OK,
    ASKS,
    AdmitFailedReason,
    AdmitOutcome,
    AdmitRuntimeError,
    AllocationFailed,
    RequestOffloading,
    Segment,
    StepContext,
)
from mstar.utils.client_errors import client_error_text
from mstar.utils.h2d import PinnedStager
from mstar.utils.knobs import kv_chain_lazy_steps, kv_pinned_indptrs, kv_plan_cache
from mstar.utils.profiler import PhaseLaps

logger = logging.getLogger(__name__)

# Off by default: `assert_pages_conserved` walks every stream of every live
# request after each admit, commit, reset and remove, and after each change to
# an admitted request's pages.
_DEBUG_ASSERTS = os.environ.get("MSTAR_KV_DEBUG_ASSERTS", "0") == "1"

_NOT_READY = AdmitOutcome(ok=True, ready=False)


class RoomMark(NamedTuple):
    """A request `reserve` couldn't fit, and what it was measured against: the room, and the pages the
    index had filed and requests had taken out of the evictable set."""
    rid: str
    node_name: str
    graph_walk: str
    published: "PublishedKVInfo | None"
    room: int
    filed: int
    taken: int


@dataclass
class PageArena:
    """physical storage, free list, and per-page ownership

    A page goes back to the allocator when its last owner releases it, not its
    first. A sealed page is never written again, so a second owner may read it;
    freeing clears the seal. Every caller holds the manager's `_lock`, so the
    counts and seals need none of their own.
    """
    kv_cache: KVCache | None
    allocator: PageAllocator
    num_owners: list[int] = field(init=False, repr=False)
    sealed: list[bool] = field(init=False, repr=False)
    index: PrefixIndex | None = field(default=None, init=False, repr=False)
    trail: tuple[int, int] | None = field(default=None, init=False, repr=False)
    # queues each page a release leaves to the index alone, for a copy to the host
    writes: HostWrites | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.num_owners = [0] * self.allocator.max_num_pages
        self.sealed = [False] * self.allocator.max_num_pages

    def acquire(self, n: int) -> list[int] | None:
        pages = self.allocator.try_allocate(n)
        if pages is not None:
            for page in pages:
                self.num_owners[page] = 1
            if self.trail is not None:
                self.trail = (self.trail[0] + 1, hash((self.trail, *pages)))
        return pages

    def retain(self, pages: list[int]) -> None:
        for page in pages:
            self.num_owners[page] += 1
            if self.num_owners[page] == 2 and self.index is not None:
                self.index.set_shared(page, True)

    def seal(self, pages: list[int]) -> None:
        for page in pages:
            self.sealed[page] = True

    def any_sealed(self, pages: list[int]) -> bool:
        return any(self.sealed[page] for page in pages)

    def release(self, pages: list[int]) -> None:
        freed = []
        for page in pages:
            assert self.num_owners[page] > 0, f"page {page} released with no owner"
            self.num_owners[page] -= 1
            if self.num_owners[page] == 0:
                self.sealed[page] = False
                freed.append(page)
            elif self.num_owners[page] == 1 and self.index is not None:
                self.index.set_shared(page, False)
                if self.writes is not None:
                    self.writes.queue(self.index.key_of(page))
        self.allocator.free(freed)

    def copy_pages(self, src: list[int], dst: list[int]) -> None:
        self.kv_cache.copy_pages(src, dst)

    @property
    def num_free(self):
        return self.allocator.num_free


@dataclass
class PrefixChain:
    """The keys that name a stream's pages, and how far the index holds them."""
    # one key per page of this stream's prompt, from the preprocess worker
    keys: list[bytes]
    # tokens not yet keyed: the prompt tail, then sampled ids, until a page fills
    unkeyed: list[int] | None
    # keys naming a whole page; the prompt's last key names a partial one until generation fills it
    keyed_pages: int
    # tokens the chain accounts for: its prompt, and every token sampled since
    covered_len: int
    # `spans` are the layout's keyed spans, one per write.
    # `unkeyed_items` are the items on the page `unkeyed` fills.
    spans: list[PrefixSpan]
    unkeyed_items: list[PageItem]
    # The prompt's keys hashed under the root, kept from whichever call hashes them first, so a repeated
    # probe hashes nothing.
    # `rooted` has one key per page.
    # `rooted_ends` has one key per image that ends inside a page, with the token the image ends at.
    rooted: list[bytes] | None = None
    rooted_ends: list[tuple[int, bytes]] | None = None
    # set once the copy from the host or the probe has looked the prompt up, so the copy is tried once
    looked_up: bool = False
    # how many of this stream's pages the index already holds
    cursor: int = 0
    # set once this stream has been reported, so a request that is admitted
    # again (a refused admit, a second partition) is still one line
    reported: bool = False
    from_host: int = 0

    @classmethod
    def seed(
        cls, keys: list[bytes], tail: list[int], spans: list[PrefixSpan], page_size: int,
    ) -> "PrefixChain":
        total = sum(span.length for span in spans)
        whole = total // page_size
        items = page_items(((span.length, span.digest) for span in spans), page_size)
        return cls(
            keys=list(keys), unkeyed=tail, keyed_pages=whole, covered_len=total,
            spans=list(spans), unkeyed_items=items.get(whole, []),
        )

    def extend(self, tokens: list[int], page_size: int) -> None:
        self.unkeyed.extend(tokens)
        self.covered_len += len(tokens)
        # Every item on the prompt's partial page ends there, so it takes length - offset of the page's tokens.
        room = page_size - sum(item.length - item.offset for item in self.unkeyed_items)
        while len(self.unkeyed) >= room:
            whole = self.keyed_pages
            key = page_key(
                self.keys[whole - 1] if whole else b"",
                self.unkeyed[:room], self.unkeyed_items,
            )
            # overwrites the partial key the prompt left here, if any
            if whole < len(self.keys):
                self.keys[whole] = key
            else:
                self.keys.append(key)
            self.keyed_pages = whole + 1
            del self.unkeyed[:room]
            self.unkeyed_items = []
            room = page_size

    def pages_filled(self, stored_len: int, page_size: int) -> int:
        return min(stored_len // page_size, self.keyed_pages)


@dataclass
class SpanCursor:
    """How far a stream is through its request's layout."""
    spans: tuple[PrefixSpan, ...]
    # `done` counts the spans the stream holds, whether written, read in or served from the cache.
    # `consumed` counts the tokens of those spans.
    # `probed` is the span the latest probe answered for.
    done: int = 0
    consumed: int = 0
    probed: int | None = None

    def advance(self, start: int, end: int) -> list[tuple[int, int]]:
        """Return ``(index, end)`` for each span a write over ``[start, end)`` touched. A span
        before ``start`` that a probe served or a retrieve read in is passed, not touched."""
        touched = []
        while self.done < len(self.spans) and self.consumed < end:
            span_end = self.consumed + self.spans[self.done].length
            if span_end > start:
                touched.append((self.done, span_end))
            if end < span_end:
                break
            self.consumed = span_end
            self.done += 1
        return touched


@dataclass
class CacheStream:
    """(request, label) cache stream metadata"""
    page_indices: list[int] = field(default_factory=list)
    stored_len: int = 0
    position: int = 0
    # tokens compacted out of the front by `release_oldest` so far; the
    # stream's committed content is then `[0, protected_prefix) + the newest
    # (stored_len - protected_prefix)` tokens of what was written
    released: int = 0
    # the first `protected_prefix` committed tokens (a text prefix, say) are
    # never released; set once, after they commit (`protect_prefix`)
    protected_prefix: int = 0
    # the policy the last committing step declared for this stream
    # (`KVStep.retention`): the index caps at its prefix, `reset` clears it
    retention: RetentionPolicy | None = None
    read_pending: bool = False
    read_future: Future | None = None
    # a failed retrieve, latched: the future is consumed once, but every later
    # readiness check has to keep reporting the stream as unusable
    read_error: BaseException | None = None
    # what a client may see of it (no server paths or internal text)
    read_error_text: str = ""
    # None while the stream is unkeyed, and once its keys stop describing it
    chain: PrefixChain | None = None
    # pages the index matched for this stream and is holding for it, from the
    # probe until `admit` converts them onto `page_indices`. A cached image end
    # stays leased until `plan` copies it. `lease_len` counts the tokens the lease holds.
    lease: list[int] | None = None
    lease_len: int = 0
    # pages held for this stream's probe until its copy from the host lands
    hold: list[int] | None = None
    # set at conversion, cleared at `commit`, so a refused admit's re-probe answers the same
    converted: bool = False
    offloaded: bool = False
    # General mutation epoch used by plans and offload claims. Appends move it.
    generation: int = 0
    # Logical-content epoch exported to remote readers. Unlike generation, an
    # append does not move it; rewinds and whole-stream replacements do.
    reset_generation: int = 0
    # Last remote logical-content epoch this cache observed. None keeps
    # transfer descriptors from before reset_generation compatible.
    remote_reset_generation: int | None = None

    # set from a successful admit until commit: an admitted step already holds
    # addressing into these pages, so an offload in that window must not claim
    # them. read by `_claim_for_offload`
    step_in_flight: bool = False
    # its prefix lookup is in `KVManager.prefix_tokens`, so a re-probe after a
    # miss (a refused admit prepares the step again) is not counted twice
    prefix_counted: bool = False
    footprint: int | None = None
    span_cursor: SpanCursor | None = None
    # Pages taken from the index while not evictable, because another request held the page or a page
    # below it. A lent page takes no room, so it doesn't count against the allotment.
    lent: set[int] = field(default_factory=set)
    # kept only under `_DEBUG_ASSERTS`
    peak: int = 0
    host_copy: "HostCopy | None" = None

    def reset(self, freed: bool=False, *, content_reset: bool=True):
        self.stored_len = 0
        self.position = 0
        self.released = 0
        self.protected_prefix = 0
        self.retention = None
        self.generation += 1
        if content_reset:
            self.reset_generation += 1
            self.remote_reset_generation = None
        self.step_in_flight = False

        if freed:
            self.page_indices.clear()

    def forget_chain(self):
        """Drop what the chain knows of this stream, and its place in the layout, for a run that starts over.

        Not part of `reset`, which an offload calls too: a reload brings the
        same tokens back, so the chain that describes them has to survive it.
        """
        self.chain = None
        if self.span_cursor is not None:
            self.span_cursor = SpanCursor(self.span_cursor.spans)


class KVPageCounts(NamedTuple):
    used: int
    total: int
    # pages the prefix index names; some may also be in ``used`` by a request
    cached: int


@dataclass(slots=True)
class Allotment:
    """Pages an admitted request may take from free and evictable pages: the pages it asked for, less the
    cache hits another request holds. Nothing is taken off for a guaranteed request. ``remaining`` is its
    share of `KVManager._unused`, and ``prompt`` is the pages it asked for, which `unreserve` puts back in
    `_asks`. An allotment for a request on the host covers its reload, and ``returning`` is set while the
    request returns from the host: the pages it may take beyond its reload."""
    pages: int
    remaining: int = 0
    prompt: int = 0
    returning: int | None = None


class Dropped(NamedTuple):
    """A dropped request until its rebuild lands: the pages it asks to return with, and the tokens its
    laid-out label held, which its rebuild must write again."""
    returns: int
    held: int


@dataclass(slots=True)
class Earmark:
    """The pages ``rid`` ran short of, which the moves of ``victims`` free for it. ``freed`` counts the room
    made by the moves applied so far."""
    rid: str
    pages: int
    victims: set[str]
    freed: int = 0


@dataclass
class ClaimedStream:
    """A stream an in-progress offload has taken ownership of, and the state
    its host copy was made from."""
    label: str
    pages: list[int]
    generation: int
    stored_len: int
    position: int
    released: int
    protected_prefix: int
    # the chain's keys for the pages below its cursor, not yet rooted
    keys: list[bytes]


@dataclass
class HostCopy:
    """A copy from the host for a readiness check, not yet seen to land.

    No `sync()` follows this copy, as one follows a reload's, so ``pages`` enter
    the index under ``keys`` only once ``event`` completes. Until then the record
    holds a reference on ``pages`` and ``host_pages``.
    """
    event: torch.cuda.Event | None
    pages: list[int]
    keys: list[bytes]
    parent: bytes | None
    host_pages: list[int]
    rid: str
    stream: CacheStream
    landed: bool = False


LabelToStream = dict[str, CacheStream]

@dataclass
class KVSequenceInfo:
    seq_len: int
    # for tracking KV cache
    latest_kv_transfer_info: KVTransferInfo | None
    page_indices: list[int] = field(default_factory=list)
    reset_generation: int | None = None


@dataclass
class PublishedKVInfo(PublishedInfo):
    # {rank -> {label: SequenceInfo}}
    info: dict[int, dict[str, KVSequenceInfo]] = field(default_factory=dict)
    world_size: int = 1

    @classmethod
    def build_for_rank(
        cls, rank: int, world_size: int,
        seq_info: dict[str, KVSequenceInfo]
    ):
        return cls(
            info={rank: seq_info},
            world_size=world_size
        )

    def update(self, other: "PublishedKVInfo"):
        for key, val in other.info.items():
            if key not in self.info:
                self.info[key] = val
                continue
            self.info[key] = {
                **self.info[key],
                **val
            }

    def clone(self) -> "PublishedKVInfo":
        return PublishedKVInfo(
            info={
                rank: {
                    label: KVSequenceInfo(
                        seq_len=seq.seq_len,
                        latest_kv_transfer_info=seq.latest_kv_transfer_info,
                        page_indices=list(seq.page_indices),
                        reset_generation=seq.reset_generation,
                    )
                    for label, seq in labels.items()
                }
                for rank, labels in self.info.items()
            },
            world_size=self.world_size,
        )

    def get(self, rank: int) -> dict[str, KVSequenceInfo]:
        return self.info.get(rank, {})


@dataclass
class AllocResult:
    success: bool = True
    error: AdmitFailedReason | None = None
    new_pages: list[int] | None = None


@dataclass
class KVPlanState:
    token_to_page: torch.Tensor
    token_to_cache: torch.Tensor
    total_tokens: int | None = None
    # an eager-write state a pre-plan built on its own stream: the writer waits
    # on it (the replay's plan-done wait comes after an eager write)
    ready: Any = None

    def copy_(self, other: "KVPlanState", capture_len: int):
        """Stage a step's addressing into this captured state.

        Neutralize only ``[n:capture_len]`` — the slots the graph scatters
        beyond the real tokens (SINK_PAGE, else they hit another request's KV).
        Decode fills its bucket exactly (n == capture_len), so no-op there; only
        packed prefill pays it, over the real gap not the whole buffer.
        """
        assert other.total_tokens is not None
        n = other.total_tokens
        self.token_to_cache[:n].copy_(other.token_to_cache)
        self.token_to_page[:n].copy_(other.token_to_page)
        self.token_to_page[n:capture_len].fill_(SINK_PAGE)
        self.token_to_cache[n:capture_len].fill_(0)
        self.total_tokens = n


class _LabelRows:
    """One plan label's real rows of a cached decode plan, derived from the
    full plan's views at the first hit: streams, lengths, page lists and the
    numpy index arrays FlashInfer takes. Padding rows are not here; they are
    rebuilt per step from the step's own padding segments (same layout, other
    dummy ids)."""

    __slots__ = (
        "rows", "streams", "lens", "npages", "row_pages", "last_page",
        "indices", "gens", "n_pad", "pad_span", "pad_pages", "built",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])


class _DecodePlanCache:
    """The last leased decode plan, as `_fast_decode_plan` continues it."""

    __slots__ = (
        "step_segments", "real_segments", "n_real", "pad_layout", "combined_labels", "views", "epoch", "pending",
        "derived", "_rows",
    )

    def __init__(self, step_segments, real_segments, n_real, pad_layout, combined_labels, views, epoch):
        # the planned step's own segments tuple: a commit of that object is the
        # one commit the next plan may assume
        self.step_segments = step_segments
        # the real rows' segments, the key the next step must match
        self.real_segments = real_segments
        self.n_real = n_real
        # (count, span) of the padding rows, which change dummy ids per slot
        self.pad_layout = pad_layout
        # the plan labels the views were built for
        self.combined_labels = combined_labels
        # the full plan's views per label, the source of `derived`
        self.views = views
        self.epoch = epoch
        self.pending = 0
        self.derived: dict | None = None
        self._rows = None

    @property
    def rows(self) -> frozenset:
        """The real (request, label) pairs, for the commit check of a step
        that is not the cached batch."""
        if self._rows is None:
            self._rows = frozenset(seg[:2] for seg in self.real_segments)
        return self._rows


class KVManager(AttentionResource):
    prefix_skip_safe = True

    def __init__(
        self,
        cfg: PagedKVConfig,
        name: str,
        joint_comm_group: JointGroups | None,
        transfer_engine_info: TransferEngineInfo,
        device: torch.device,
        dtype=torch.bfloat16,
        needs_remote_transfer: bool = False,
        leads: bool = True,
        nodes: Collection[str] | None = None,
    ):
        self.config = cfg
        tp_cache = False
        if joint_comm_group is not None:
            # before the cache is allocated: it is sized off the head counts
            cfg.shard(joint_comm_group.world_size)
            tp_cache = joint_comm_group.world_size > 1 and cfg.prefix_cache and cfg.prefix_cache_tp
            if tp_cache and cfg.cpu_offload_pages > 0:
                raise ValueError(
                    f"KV {name}: prefix_cache_tp can't be combined with cpu_offload_pages="
                    f"{cfg.cpu_offload_pages}. Every rank has to take and evict pages in rank 0's "
                    "order, and a reload isn't ordered against cache hits. Set cpu_offload_pages "
                    "to 0 or prefix_cache_tp to false."
                )
            if tp_cache and needs_remote_transfer:
                raise ValueError(
                    f"KV {name}: prefix_cache_tp needs this pool's nodes on one TP group. With more "
                    "than one (replicas, or walks placed on other ranks) the pool counts as shared, "
                    "and a read from another group would take pages outside rank 0's order. Serve "
                    "the nodes from one group or set prefix_cache_tp to false."
                )
        self.kv_cache = KVCache(
            cfg, device, dtype
        )
        self.name = name

        self._arena = PageArena(
            kv_cache=self.kv_cache,
            allocator=PageAllocator(cfg.max_num_pages)
        )
        # take SINK_PAGE out of circulation; the allocator is FIFO from 0
        sink = self._arena.acquire(1)
        assert sink == [SINK_PAGE], f"expected page {SINK_PAGE} first, got {sink}"
        self._transfer = KVTransferManager(
            transfer_engine_info,
            self.kv_cache,
            resource_key=name,
            needs_remote_transfer=needs_remote_transfer,
        )
        self._cpu_pool: CPUPagePool | None = None
        # requests whose every stream has been looked up for a copy from the host, with none arriving, so
        # `_fetch_from_host` returns before reading their streams. Giving one of a request's streams a new chain
        # takes the request out of the set.
        self._looked_up: set[str] = set()
        # A copy is listed with `event` None until it is queued, since an event
        # that was never recorded reads as done.
        self._host_copies: list[HostCopy] = []
        self._arriving: dict[bytes, HostCopy] = {}
        # set once both the prefix cache and the host pool are on
        self._host_writes: HostWrites | None = None
        if cfg.cpu_offload_pages > 0:
            self._cpu_pool = CPUPagePool(
                config=cfg, kv_cache=self.kv_cache,
                arena=PageArena(kv_cache=None, allocator=PageAllocator(cfg.cpu_offload_pages)),
            )
        self._streams: dict[str, LabelToStream] = {}
        self._overrides: dict[str, KVReqConfig] = {}
        # dropped request -> what it returns with, until its rebuild lands
        self._dropped: dict[str, Dropped] = {}
        # (node, walk) -> (label, stop buffer name) per request, for the
        # decode-step chain extension; dropped with the request
        self._chain_lookup: dict[str, dict[tuple[str, str], tuple]] = {}
        # decode steps of sampled tokens waiting to extend the chains (lazy)
        self._chain_backlog: list[tuple] = []
        self._chain_lazy_steps = kv_chain_lazy_steps()
        # steps waiting per request, and the widest row among them: a
        # commit tolerates a chain that lags by that many tokens
        self._chain_pending: Counter = Counter()
        self._chain_backlog_width = 0
        # requests whose waiting tokens went out early (a removal): the next
        # flush leaves them out
        self._chain_drained: set[str] = set()
        # its entity id is the worker id, and one worker is one copy of a node
        self._replica = (
            transfer_engine_info.my_entity_id
            if transfer_engine_info is not None else None
        )
        self._prefix_root: bytes | None = None
        self._index: PrefixIndex | None = None
        # node -> (prompt tokens looked up in the index, tokens it held), one
        # lookup per request stream; read by the worker's metrics
        self.prefix_tokens: dict[str, tuple[int, int]] = {}
        # node -> pages a dropped request's rebuild found in the index; read by the worker's metrics
        self.rebuild_pages: dict[str, int] = {}
        # label -> the walks its keys describe; a label not here is not gated
        self._keyed_walks: dict[str, frozenset[str]] = {}
        # nodes already warned that a declared stream reached them unkeyed
        self._warned_unkeyed: set[str] = set()
        self._overran: set[tuple[str, bool]] = set()
        # streams whose lease still holds a cached image end for `plan` to copy
        self._image_ends: list[tuple[str, CacheStream]] = []
        self._rank = joint_comm_group.rank if joint_comm_group is not None else 0
        self._world_size = joint_comm_group.world_size if joint_comm_group is not None else 1
        self._comm_group = joint_comm_group
        # Only a deciding pool asks and keeps allotments. With more than one rank none decides.
        # Rank 0 would allot at readiness, after the step goes to the other ranks and before its
        # admit, so a grant could pass on one rank and fail on another.
        self._decides = self._world_size == 1 and leads
        # An allotment covers the labels `_nodes` open.
        # `_asks` holds the pages `ask` counted for each request that asked and has no allotment yet, which its
        #   allotment is sized from.
        # `local_loops` holds the loops that grow KV on `_nodes` in a walk this worker runs, set by the worker.
        # `_unused` counts the pages admitted requests were allotted and haven't used yet.
        self._nodes = nodes
        self.local_loops: Collection[str] = frozenset()
        self._asks: dict[str, int] = {}
        self._allotments: dict[str, Allotment] = {}
        self._unused = 0
        # Preemption state.
        # `_earmarks`: request -> the earmark it holds until its victims have moved.
        # `_victims`: a victim not yet moved -> its earmark, or None when the victim is the request that ran short.
        # `_hosted`: the victims that move to the host, decided when they are chosen.
        # `_short`: a request whose shortage no move could cover -> the room at that time.
        # `_ran_short`: request -> the room when its last grant ran short.
        self._earmarks: dict[str, Earmark] = {}
        self._victims: dict[str, Earmark | None] = {}
        self._hosted: set[str] = set()
        self._short: dict[str, int] = {}
        self._ran_short: dict[str, int] = {}
        if _DEBUG_ASSERTS and tp_cache:
            self._arena.trail = (0, 0)
        self._device = device
        self._lock = threading.RLock()

        # (slot, label) -> KVPlanState, sized for the largest capture bucket
        self._static_plan_states: dict[tuple[int, str], KVPlanState] = {}
        # copy-only H2D into those, see `_stage_decode_plan_state`
        self._plan_stager = PinnedStager(torch.long)
        self._cg_max_seq_len = 0
        # the plan cache (`_fast_decode_plan`): bumped by everything that
        # changes a stream's pages or length other than a step's commit
        self._plan_epoch = 0
        self._decode_plan_cache: _DecodePlanCache | None = None
        self._plan_cache_mode = kv_plan_cache()
        # FlashInfer's index arrays from page-locked memory (see PinnedIndexRing)
        self._indptr_ring = (
            PinnedIndexRing() if kv_pinned_indptrs() and torch.cuda.is_available() else None
        )
        self.plan_cache_stats = {"hit": 0, "miss": 0, "mismatch": 0}
        # A read-only step (every segment span 0, e.g. a decoder attending an
        # encoder context it wrote at prefill) plans the same as the last one
        # on its slot with the same segments and bucket while nothing touched
        # their streams: slot -> ((segments, bucket), epochs, plan). One entry
        # per replay slot, as `_staged_outputs` keeps, so the slot count bounds
        # the cache; replay slots pad with rows of their own, and an unpadded
        # batch builds its plan once per slot. The epoch moves with
        # `_plan_epoch`, and when a stream of a cached step's real row changes
        # (`_readonly_rows`).
        self._readonly_epoch = 0
        self._readonly_plans: dict[int | None, tuple] = {}
        self._readonly_rows: frozenset = frozenset()
        # (slot, label) -> the plan output whose state that slot's static
        # buffers hold, so a reused plan skips re-staging them
        self._staged_outputs: dict[tuple, KVPlanOutput] = {}
        self._empty_locations: torch.Tensor | None = None
        self._current_plan_states: dict[str, KVPlanState] = {}
        self.reset_default_cursors()

        self._preplan_states: dict[str, KVPlanState] = {}
        self._preplanned = False
        self._preplan_key = None
        self._cached_plan_output: dict[str, KVPlanOutput] | None = None

        # (rid, to_label) reservations a staged step made; clear_preplan
        # releases them. recorded in admit, not plan: forks are not staged.
        self._preplan_new_labels: list[tuple[str, str]] = []
        # (rid, label) marked step_in_flight by a staged step, so an abandoned
        # one does not leave its streams unevictable
        self._preplan_marked: list[tuple[str, str]] = []

        # (rid, label) -> pages the live reservation freshly acquired, for
        # ``rollback_admit``. One record, not one per pass: only the plan thread
        # or the GPU thread holds a reservation at a time, and the unwind has to
        # find it whichever it was. Admit, commit and unwind each touch only
        # their own step's rows: the unbatched fallback admits every request
        # before any runs, and a refusal unwinds the ones already admitted.
        #
        # Handle-keyed, so it MUST be purged when the handle is released: handles
        # are recycled, and a stale entry would resolve against whichever request
        # gets that handle next. ``admit`` clears its rows' entries, but a
        # pre-planned step's admit returns first -- see ``remove_request`` /
        # ``reset_request``.
        self._admit_reserved_pages: dict[tuple[int, str], list[int]] = {}

    @classmethod
    def build(cls, spec: KVSpec, info: EngineResourceInfo):
        return cls(
            cfg=spec.config,
            name=spec.resource_key,
            device=info.device,
            joint_comm_group=info.joint_comm_group,
            transfer_engine_info=info.transfer_engine_info,
            dtype=info.kv_dtype,
            needs_remote_transfer=info.needs_remote_transfer,
            leads=spec.leader is None or spec.leader in info.nodes,
            nodes=info.nodes,
        )

    def build_accelerator_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ):
        del slots, max_bs
        # the per-(slot, label) buffers themselves are built on first plan for
        # that key (which labels a walk plans under is the step's to declare),
        # all at this one max length so they outlive any single bucket. Every
        # runner capturing against this node calls in, so keep the largest
        self._cg_max_seq_len = max(self._cg_max_seq_len, max_seq_len)

    def _static_plan_state(self, slot: int, label: str) -> KVPlanState:
        state = self._static_plan_states.get((slot, label))
        if state is None:
            state = self._static_plan_states[(slot, label)] = KVPlanState(
                token_to_cache=torch.zeros(
                    self._cg_max_seq_len, dtype=torch.long, device=self._device
                ),
                token_to_page=torch.full(
                    (self._cg_max_seq_len,), SINK_PAGE,
                    dtype=torch.long, device=self._device
                ),
            )
        return state

    def enable_prefix_cache(self, root: bytes, walks: dict[str, KeyedWalks] | None = None) -> bool:
        """Open the index under ``root``, the identity every key hangs from."""
        self._keyed_walks = {}
        for label, named in (walks or {}).items():
            keyed = set(named.writes)
            # Above one rank a growth walk isn't keyed, since each rank reads its sampled ids at a
            # different step.
            if self._world_size == 1:
                keyed.update(named.growth)
            self._keyed_walks[label] = frozenset(keyed)
        if not self.config.prefix_cache:
            return False
        if self._world_size > 1 and not self.config.prefix_cache_tp:
            logger.info(
                "KV %s: prefix cache off at world size %d (prefix_cache_tp opts in)",
                self.name, self._world_size,
            )
            return False
        self._prefix_root = root
        self._index = PrefixIndex(self._arena)
        if self._world_size == 1:
            # The evictable set is counted only at one rank. When the arena reports to the index, `insert`
            # reads owner counts to decide a page's parent, and those counts can differ between ranks.
            self._arena.index = self._index
        if self._cpu_pool is not None:
            self._open_host_writes()
        return True

    def _open_host_writes(self) -> None:
        self._host_writes = HostWrites(self._cpu_pool, self._index, self.kv_cache.tensor[:, 0].nbytes)
        self._arena.writes = self._host_writes

    def close_prefix_cache(self) -> None:
        self._index = None

    def page_counts(self) -> KVPageCounts:
        """The pool's pages right now, without the manager lock: each count is
        read whole, though the three may straddle an allocation."""
        total = self.config.request_pages
        index = self._index
        return KVPageCounts(
            used=total - self._arena.num_free, total=total,
            cached=0 if index is None else index.num_pages(),
        )

    def resolve_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, inputs,
    ) -> CachedPrefix | None:
        """Match this request's prefix against the index and hold what matched.

        The first probe takes one lease for the whole layout, and each walk's
        probe answers its own share of it. A probe is repeated whenever a step
        is prepared again, so it answers the same every time it is asked and
        takes one reference however often that is.
        """
        with self._lock:
            label = self._keyed_label(rid, node_name, graph_walk)
            if label is None:
                self._warn_unkeyed(rid, node_name, graph_walk)
                return None
            stream = self._ensure_label(rid, label)
            chain, cursor = stream.chain, stream.span_cursor
            if chain is None or cursor.done >= len(chain.spans):
                return None
            span = chain.spans[cursor.done]
            self._check_positions(span, graph_walk, inputs)
            if span.walk != graph_walk or span.length != inputs.input_seq_len:
                self._drop_layout(rid, label, stream, (
                    f"{graph_walk} writes {inputs.input_seq_len} slots where its "
                    f"layout's span {cursor.done} has {span.walk} write {span.length}"
                ))
                return None
            if stream.converted:
                return self._answer(label, cursor, stream.stored_len)
            if stream.lease is not None:
                return self._answer(label, cursor, stream.lease_len)
            if stream.stored_len or stream.offloaded or stream.read_pending:
                return None
            if self._world_size > 1:
                # Above one rank, use only the lease taken when rank 0 sent the step. A lookup now
                # would also see this rank's later commits.
                return self._answer(label, cursor, 0)
        if not self._lease(rid, node_name, label, stream, chain):
            return None
        with self._lock:
            if stream.lease is None:
                return None
            return self._answer(label, cursor, stream.lease_len)

    def _lease(self, rid: str, node_name: str, label: str, stream: CacheStream, chain: PrefixChain) -> bool:
        """Lease what the index holds of the stream's prompt, unless the stream holds a lease already.
        False if the stream was removed while its keys were hashed."""
        self._root_chain(chain)
        with self._lock:
            if self._streams.get(rid, {}).get(label) is not stream:
                return False
            chain.looked_up = True
            self._note_looked_up(rid)
            rooted, ends = chain.rooted, chain.rooted_ends
            total = sum(span.length for span in chain.spans)
            # The lock was dropped to hash, so another thread's probe may have taken the lease already.
            if stream.lease is None:
                # Matching the whole prompt would leave the walk an empty input, since it skips every matched page.
                walk = self._index.lookup if self._world_size == 1 else self._index.pages_for
                matched = self._retain_cached(rid, stream, walk(rooted)[:len(rooted) - 1])
                held = len(matched) * self.config.page_size
                # If an image ends inside the next page, look up its cached partial page.
                image_end = self._image_end(ends, held, total)
                if image_end is not None:
                    end, key = image_end
                    cached = self._retain_cached(rid, stream, self._index.lookup([key]), counted=False)
                    if cached:
                        # The lease isn't on the stream yet, so ``remaining`` still counts its pages as unused.
                        # Check this page against the whole allowance rather than ``remaining``.
                        own = self._grant(rid, label, 1, 0) if self._allowance(rid) > 0 else None
                        if own is None:
                            self._release(rid, stream, cached)
                        else:
                            matched, held = matched + cached + own, end
                if not stream.prefix_counted:
                    stream.prefix_counted = True
                    # one assignment, so a reader on another thread never sees half of it
                    queried, hit = self.prefix_tokens.get(node_name, (0, 0))
                    self.prefix_tokens[node_name] = (queried + total, hit + held)
                if rid in self._dropped:
                    self.rebuild_pages[node_name] = self.rebuild_pages.get(node_name, 0) + len(matched)
                if matched:
                    stream.lease, stream.lease_len = matched, held
                    self._recount(rid)
            return True

    def _image_end(self, ends: list[tuple[int, bytes]], held: int, total: int) -> tuple[int, bytes] | None:
        """The last image in ``ends`` that finishes inside the page after ``held`` slots, if any."""
        for end, key in reversed(ends):
            if held < end < min(held + self.config.page_size, total):
                return end, key
        return None

    def _root(
        self, keys: list[bytes], spans: list[PrefixSpan],
    ) -> tuple[list[bytes], list[tuple[int, bytes]]]:
        """The prompt's keys, and the keys of images ending inside a page, hashed
        under the root. Called outside the lock: one SHA-256 per page, and
        admit, commit and remove would wait on it."""
        rooted = [fingerprint(self._prefix_root, key) for key in keys]
        ends, at = [], 0
        for span in spans:
            at += span.length
            if span.end_key is not None:
                ends.append((at, fingerprint(self._prefix_root, span.end_key)))
        return rooted, ends

    def _root_chain(self, chain: PrefixChain) -> None:
        """Fill `chain.rooted` and `chain.rooted_ends` once. Called without the lock held."""
        with self._lock:
            if chain.rooted is not None:
                return
            keys, spans = list(chain.keys), chain.spans
        rooted, ends = self._root(keys, spans)
        with self._lock:
            chain.rooted, chain.rooted_ends = rooted, ends

    def lease_cached_prefix(self, rid: str, node_name: str, graph_walk: str) -> list[int] | None:
        """Returns the same lease with every step until `admit` converts it."""
        with self._lock:
            label = self._keyed_label(rid, node_name, graph_walk)
            if label is None:
                return None
            stream = self._ensure_label(rid, label)
            chain, cursor = stream.chain, stream.span_cursor
            if stream.lease is not None:
                return stream.lease
            if (
                chain is None or cursor.done == len(chain.spans)
                or chain.spans[cursor.done].walk != graph_walk
                or stream.converted or stream.stored_len or stream.offloaded or stream.read_pending
            ):
                return None
        if not self._lease(rid, node_name, label, stream, chain):
            return None
        with self._lock:
            return stream.lease

    def adopt_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, pages: list[int], caught_up: bool,
    ) -> bool:
        """False while some pages are missing and ``caught_up`` is False, since this rank's commit of
        the step that wrote them can lag rank 0's. Once ``caught_up`` nothing will index a missing
        page, so it raises. It raises too if the index holds other pages under those keys,
        since a sharded prefill gathers cache slots across ranks."""
        with self._lock:
            label = self._keyed_label(rid, node_name, graph_walk)
            if label is None:
                return True
            stream = self._ensure_label(rid, label)
            chain = stream.chain
            if stream.converted or stream.lease is not None:
                return True
            if chain is None or stream.stored_len:
                raise RuntimeError(
                    f"KV {self.name}: rank 0 leased {len(pages)} cached pages for {rid}/{label}, "
                    "whose stream here already holds tokens or has no keys"
                )
        self._root_chain(chain)
        with self._lock:
            if self._streams.get(rid, {}).get(label) is not stream:
                return True
            matched = self._index.pages_for(chain.rooted[:len(pages)])
            if len(matched) < len(pages) and not caught_up:
                return False
            if matched != pages:
                raise RuntimeError(
                    f"KV {self.name}: prefix index diverged from rank 0 for {rid}/{label}: "
                    f"pages {matched[:4]}... here ({len(matched)}), {pages[:4]}... there ({len(pages)})"
                )
            self._arena.retain(matched)
            stream.lease, stream.lease_len = matched, len(matched) * self.config.page_size
            return True

    def release_cached_prefix(self, rid: str, node_name: str, graph_walk: str) -> None:
        """Does nothing at one rank, where preparing the step again reuses the lease. A lease also
        stays once the cache has served one of the stream's walks, whose KV lives only in the lease."""
        if self._world_size == 1:
            return
        with self._lock:
            stream = self._streams.get(rid, {}).get(self._keyed_label(rid, node_name, graph_walk))
            if stream is not None and stream.span_cursor is not None and not stream.span_cursor.done:
                self._release_lease(rid, stream)

    def page_trail(self) -> tuple[int, int] | None:
        with self._lock:
            return self._arena.trail

    def _check_positions(self, span: PrefixSpan, graph_walk: str, inputs) -> None:
        assert span.digest is not None or inputs.custom_pos_ids is None, (
            f"KV {self.name}: {graph_walk} places positions of its own, but "
            "the layout keys it by ids, whose positions are their count"
        )

    @staticmethod
    def _answer(label: str, cursor: SpanCursor, present: int) -> CachedPrefix:
        """This walk's share of the ``present`` tokens the stream holds, with the
        position past that share. Records the span as the one the probe answered for."""
        span = cursor.spans[cursor.done]
        tokens = min(max(present - cursor.consumed, 0), span.length)
        position = sum(earlier.advance for earlier in cursor.spans[:cursor.done])
        if span.digest is None:
            position += tokens
        elif tokens == span.length:
            # A partly held item doesn't move the position. Its walk still runs and advances it.
            position += span.advance
        cursor.probed = cursor.done
        return CachedPrefix(label, tokens, position)

    def _drop_layout(
        self, rid: str, label: str, stream: CacheStream, disagreement: str,
    ) -> None:
        """End the chain of a stream whose walks disagree with its layout.

        The KV of the walks already served from the cache lives only in the lease,
        because a served walk never admits. Releasing the lease would have the next
        walk write from token 0 with that KV missing. So the served part is kept
        when it ends on a page boundary. When it ends inside a page the request
        fails, because that page is shared and this stream may not write on it.
        """
        logger.warning(
            "KV %s: %s/%s %s; its chain ends here", self.name, rid, label, disagreement,
        )
        cursor = stream.span_cursor
        if cursor.done and stream.lease is not None:
            keep, partial = divmod(cursor.consumed, self.config.page_size)
            if partial:
                raise RuntimeError(
                    f"KV {self.name}: {rid}/{label}: the walks the cache served end "
                    f"inside a page ({cursor.consumed} slots), so their KV cannot be kept"
                )
            lease, stream.lease = stream.lease, None
            self._adopt(rid, stream, lease[:keep])
            self._release(rid, stream, lease[keep:])
        else:
            self._release_lease(rid, stream)
        stream.chain = None
        self._recount(rid)

    def apply_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, inputs,
        prefix: CachedPrefix | None,
    ) -> None:
        """Cut this stream's lease down to the agreed prefix.

        A walk the cache serves whole leaves the lease alone, since the walks
        after it use the rest. A walk that arrives with no prefix was not probed
        and is about to write over what the lease holds, so its layout is dropped.
        """
        with self._lock:
            label = self._keyed_label(rid, node_name, graph_walk)
            if label is None:
                return
            if prefix is None:
                # an unprobed walk never reached the position check in `resolve_cached_prefix`
                stream = self._ensure_label(rid, label)
                chain, cursor = stream.chain, stream.span_cursor
                if chain is not None and cursor.done < len(chain.spans):
                    self._check_positions(chain.spans[cursor.done], graph_walk, inputs)
            stream = self._streams.get(rid, {}).get(label)
            if stream is None or stream.lease is None or stream.stored_len:
                return
            if prefix is None:
                self._drop_layout(rid, label, stream, f"{graph_walk} writes unprobed")
                return
            cursor = stream.span_cursor
            if prefix.tokens == cursor.spans[cursor.done].length:
                return
            held = cursor.consumed + prefix.tokens
            if held < stream.lease_len:
                keep = held // self.config.page_size
                cut = stream.lease[keep:]
                stream.lease, stream.lease_len = stream.lease[:keep] or None, keep * self.config.page_size
                self._release(rid, stream, cut)
                self._recount(rid)

    def complete_cached_walk(self, rid: str, node_name: str, graph_walk: str) -> None:
        with self._lock:
            label = self._keyed_label(rid, node_name, graph_walk)
            if label is None:
                return
            stream = self._streams.get(rid, {}).get(label)
            if stream is None or stream.chain is None:
                return
            cursor = stream.span_cursor
            cursor.consumed += cursor.spans[cursor.done].length
            cursor.done += 1

    def _index_filled_pages(
        self, segment: Segment, stream: CacheStream, ctx: StepContext,
        touched: list[tuple[int, int]],
    ) -> None:
        """Offer every page this commit filled to the index. ``touched`` holds
        ``(index, end)`` for each span of the layout this commit wrote into.

        Only whole pages, because the tail is still being written into and a
        second owner would be reading bytes that are still moving. A dummy or
        padded row is left out: its pages hold whatever the capture wrote.
        """
        chain = stream.chain
        if (
            self._index is None
            or ctx.capture
            or not ctx.is_real_row(segment.request_id)
            or chain is None
        ):
            return
        walk = ctx.walk_of(segment.request_id) if ctx.request_walks else ctx.graph_walk
        walks = self._keyed_walks.get(segment.label)
        if walks is not None and walk not in walks:
            # another walk wrote this span (an edit's image, say), which the keys do not describe
            stream.chain = None
            return
        # what was here before this write, not after: a decode step can commit
        # before the token it writes is read back and counted. the lazy chain
        # extension holds back a few steps more (`extend_prefix_chains_batch`)
        lag = (
            self._chain_pending.get(segment.request_id, 0) * self._chain_backlog_width
            if self._chain_backlog else 0
        )
        if stream.released or stream.stored_len - segment.span > chain.covered_len + lag:
            stream.chain = None
            return
        ended = None
        for index, end in touched:
            if index >= len(chain.spans):
                break
            span = chain.spans[index]
            if span.walk != walk:
                stream.chain = None
                return
            if end <= stream.stored_len and span.end_key is not None:
                ended = end, span.end_key
        filled = chain.pages_filled(stream.stored_len, self.config.page_size)
        if stream.retention is not None:
            filled = min(
                filled, stream.retention.protected_prefix // self.config.page_size
            )
        # the parent by key: after a lost race or a reload this stream holds a copy
        parent = (
            self._index.page_for(
                fingerprint(self._prefix_root, chain.keys[chain.cursor - 1])
            )
            if 0 < chain.cursor < filled else None
        )
        while chain.cursor < filled:
            key = fingerprint(self._prefix_root, chain.keys[chain.cursor])
            page = stream.page_indices[chain.cursor]
            if not self._index.insert(key, page, parent):
                # another request filled this page first and its copy is the
                # one the index names; ours stays private to this stream
                page = self._index.page_for(key)
            parent = page
            chain.cursor += 1
        if ended is not None:
            self._cache_image_end(stream, chain, *ended)

    def _evict(self, n: int) -> int:
        """`PrefixIndex.evict`, counting the pages it drops that have no copy on the host. Pages still waiting
        for their copy to the host go last."""
        writes = self._host_writes
        dropped = self._index.evict(n, None if writes is None else writes.awaits)
        if writes is not None:
            writes.evicted(dropped)
        if self._cpu_pool is not None:
            self._cpu_pool.count_dropped(dropped)
        return len(dropped)

    def after_output_copy(self, step: KVStep) -> None:
        """Copy up to a step's share of the queued pages to the host. The worker has seen ``step`` finish on the
        device, so every page queued by its commit or an earlier one is final. Nothing waits on the copy."""
        writes = self._host_writes
        if writes is None or not writes.queued:
            return
        with self._lock:
            write = writes.take(step.commits)
        if write is not None:
            writes.launch(write, self.kv_cache.tensor, self._lock)

    def _cache_image_end(self, stream: CacheStream, chain: PrefixChain, end: int, key: bytes) -> None:
        """Cache the page up to the image's end. When the text after the image changes,
        a repeat can still reuse it instead of running the walk again."""
        if self._world_size > 1:
            # Serving a cached image end takes a page on rank 0 alone, so the ranks' pages would differ.
            return
        page = end // self.config.page_size
        rooted = fingerprint(self._prefix_root, key)
        parent = self._index.page_for(fingerprint(self._prefix_root, chain.keys[page - 1])) if page else None
        if (page and parent is None) or self._index.page_for(rooted) is not None:
            return
        # only from free pages. A partial copy isn't worth evicting a full cached page for.
        copy = self._arena.acquire(1)
        if copy is None:
            logger.debug("KV %s: no free page for an image end, so it is not cached", self.name)
            return
        self.kv_cache.copy_slots(stream.page_indices[page], copy[0], end % self.config.page_size)
        self._index.insert(rooted, copy[0], parent)
        if self._host_writes is not None:
            self._host_writes.add_image_end(rooted)
        # drop our reference, so the index is the page's one owner and can evict it
        self._arena.release(copy)

    def _report_admission(self, segment: Segment, stream: CacheStream) -> None:
        """One line per request that a declared stream admitted."""
        # silent where the cache is closed: the keys are still on the stream,
        # and a line saying nothing matched would read as a miss
        chain = stream.chain
        if self._index is None or chain is None or chain.reported:
            return
        chain.reported = True
        matched = chain.cursor
        logger.info(
            "KV %s: %s/%s matched %d of its %d pages, %d from the host, whole "
            "prompt already cached %s, replica %s",
            self.name, segment.request_id, segment.label,
            matched, chain.keyed_pages, chain.from_host,
            bool(matched) and matched == chain.keyed_pages,
            self._replica,
        )

    def _take_local_match(
        self, rid: str, stream: CacheStream, published_len: int, rooted: list[bytes] | None,
    ) -> None:
        """Take what this cache already holds of a stream about to be read in.

        The pages a prefill rank published are often pages this rank wrote for
        an earlier request, and moving them again costs a transfer to arrive at
        bytes that are already here. Converted onto the stream rather than
        leased: the read is issued under this same lock, so nothing can come
        between the two.
        """
        self._touch_plan_cache()
        if (
            self._index is None
            or not rooted
            or stream.stored_len
            or stream.offloaded
            or stream.chain is None
            # the probe's lease is adopted at its walk's admit
            or stream.lease is not None
        ):
            return
        # never past what the other side published, and only whole pages
        matched = self._index.lookup(rooted)[
            :published_len // self.config.page_size
        ]
        matched = self._retain_cached(rid, stream, matched)
        if not matched:
            return
        self._adopt(rid, stream, matched)
        stream.chain.cursor = len(matched)
        self._recount(rid)

    def _warn_unkeyed(self, rid: str, node_name: str, graph_walk: str) -> None:
        """Say once per node that a stream the model declared arrived with no keys or no layout.

        Nothing else would: an unkeyed request is served in full and never
        reported, so the cache stays empty while every other line looks healthy.
        """
        overrides = self._overrides.get(rid)
        if (
            self._index is None
            or node_name in self._warned_unkeyed
            or overrides is None
            or not overrides.prefix_cache
        ):
            return
        keys = overrides.prefix_keys or {}
        layouts = overrides.kv_layout or {}
        for label in overrides.get_labels(node_name, graph_walk):
            layout = layouts.get(label)
            # a layout that keys no span leaves its request unkeyed on purpose
            if graph_walk in self._keyed_walks.get(label, ()) and not keys.get(label) and (
                layout is None or layout.keyed_spans
            ):
                self._warned_unkeyed.add(node_name)
                logger.warning(
                    "KV %s: requests reach %s with no %s for %s, which the "
                    "model declared for prefix reuse, so nothing will be cached",
                    self.name, node_name, "layout" if layout is None else "keys", label,
                )
                return

    def _keyed_label(
        self, rid: str, node_name: str, graph_walk: str,
    ) -> str | None:
        """The one label this request keyed on this node and walk, if any.

        Under `_lock`: `remove_request` drops ``rid``'s overrides on another
        thread, and a label read outside it can name a stream already gone.
        """
        if self._index is None:
            return None
        overrides = self._overrides.get(rid)
        if overrides is None or not overrides.prefix_cache:
            return None
        keys = overrides.prefix_keys or {}
        layouts = overrides.kv_layout or {}
        for label in overrides.get_labels(node_name, graph_walk):
            if keys.get(label) and label in layouts:
                return label
        return None

    def keyed_label(self, rid: str, node_name: str, graph_walk: str) -> str | None:
        with self._lock:
            return self._keyed_label(rid, node_name, graph_walk)

    def keyed_walks(self, label: str) -> frozenset[str]:
        """The walks whose writes under ``label`` the prefix cache keys."""
        return self._keyed_walks.get(label, frozenset())

    def chain_key(self, rid: str, label: str, pages: int) -> bytes | None:
        """The index key of the stream's first ``pages`` pages, or None until
        its chain keys that many whole pages."""
        with self._lock:
            stream = self._streams.get(rid, {}).get(label)
            chain = None if stream is None else stream.chain
            if chain is None or not 0 < pages <= chain.keyed_pages:
                return None
            # The probe rooted the prompt's keys. The prompt's last key can name a
            # page that was partial at the probe, so that key is hashed again.
            if chain.rooted is not None and pages < len(chain.rooted):
                return chain.rooted[pages - 1]
            key = chain.keys[pages - 1]
        return fingerprint(self._prefix_root, key)

    def chain_keys(self, rid: str, label: str, pages: int) -> list[bytes]:
        """`chain_key` for each of the stream's first ``pages`` pages, as many as
        its chain keys whole, in one call."""
        with self._lock:
            stream = self._streams.get(rid, {}).get(label)
            chain = None if stream is None else stream.chain
            if chain is None:
                return []
            pages = min(pages, chain.keyed_pages)
            rooted = [] if chain.rooted is None else list(chain.rooted[:min(pages, len(chain.rooted) - 1)])
            rest = chain.keys[len(rooted):pages]
        return rooted + [fingerprint(self._prefix_root, key) for key in rest]

    def _seed_keys(self, rid: str, label: str, stream: CacheStream) -> None:
        """Put this request's chain for ``label`` on its stream."""
        overrides = self._overrides.get(rid)
        if overrides is None or not overrides.prefix_cache:
            return
        keys = (overrides.prefix_keys or {}).get(label)
        # the chain reads the span cursor, which only a label the request opens has
        layout = self._held_layout(rid, label)
        if not keys or layout is None:
            return
        tail = list((overrides.prefix_tail or {}).get(label) or ())
        stream.chain = PrefixChain.seed(
            keys, tail, layout.spans[:layout.keyed_spans], self.config.page_size,
        )
        self._looked_up.discard(rid)

    @property
    def keys_prefix_chains(self) -> bool:
        return self._index is not None

    def extend_prefix_chain(
        self, rid: str, node_name: str, graph_walk: str, outputs,
    ) -> None:
        """Key what this request generated, once a page of it exists.

        The ids come from the stop check's host copy, which can land after their
        step commits, so a later commit indexes the page.
        """
        self.flush_chain_backlog()
        with self._lock:
            label = self._keyed_label(rid, node_name, graph_walk)
            if label is None:
                return
            tensor = (self._overrides[rid].prefix_decode or {}).get(label)
            sampled = outputs.get(tensor) if tensor else None
            if not sampled:
                return
            stream = self._streams.get(rid, {}).get(label)
            if stream is None or stream.chain is None or stream.chain.unkeyed is None:
                return
            if stream.released:
                # past a front release `page_indices[k]` is no longer page k of
                # the chain, and nothing here can say which page a key names
                stream.chain = None
                return
            stream.chain.extend(sampled[0].flatten().tolist(), self.config.page_size)

    def extend_prefix_chains_batch(
        self, request_ids: list[str], node_name: str,
        walk_of: Callable[[str], str] | str, host_rows,
    ) -> None:
        """``extend_prefix_chain`` for a step, from the stop check's row
        buffers (``HostRows``: row i of every buffer belongs to
        ``host_rows.request_ids[i]``).

        Lazy by default: the step's rows are copied aside and the chains are
        extended once every ``MSTAR_KV_CHAIN_LAZY_STEPS`` steps, one extend per
        request with all its tokens since the last flush, so the per-row Python
        runs once per batch of steps instead of every step. A key can thus
        appear a few steps late and is indexed by a later commit, as the
        single-request path allows. Removals and resets flush first.

        The walk is per row, so a step mixing walks keys each row under its
        own: a step carries its rows' walks, in ``request_ids`` order. A step
        of one walk passes that walk rather than a function."""
        if self._chain_lazy_steps <= 1:
            self._extend_chains_eager(request_ids, node_name, walk_of, host_rows)
            return
        step = (
            tuple(host_rows.request_ids), node_name,
            (walk_of,) * len(request_ids) if isinstance(walk_of, str)
            else tuple(walk_of(rid) for rid in request_ids),
            tuple(request_ids),
            {
                name: buf.reshape(buf.shape[0], -1).cpu().numpy().copy()
                for name, buf in host_rows.buffers.items()
                if torch.is_tensor(buf) and buf.dim() != 0
            },
        )
        width = max((a.shape[1] for a in step[4].values()), default=0)
        with self._lock:
            self._chain_backlog.append(step)
            self._chain_pending.update(step[0])
            self._chain_backlog_width = max(self._chain_backlog_width, width)
            if len(self._chain_backlog) >= self._chain_lazy_steps:
                self.flush_chain_backlog()

    def _extend_chains_eager(
        self, request_ids: list[str], node_name: str,
        walk_of: Callable[[str], str] | str, host_rows,
    ) -> None:
        """The per-step extension (MSTAR_KV_CHAIN_LAZY_STEPS=1): one lock,
        one ``tolist`` per buffer instead of one per request. The round-2
        body, kept as it was: at c32 the step is GIL-bound and even the
        few extra numpy calls of the lazy path's bookkeeping showed."""
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
        buffers = host_rows.buffers
        rows_of: dict[str, list] = {}
        page_size = self.config.page_size
        lookup = self._chain_lookup
        # a step of one walk keys every row alike
        step_key = (node_name, walk_of) if isinstance(walk_of, str) else None
        with self._lock:
            for rid in request_ids:
                # the label and buffer name do not change over a request's
                # life (its overrides are set at ingest), so each is looked up
                # once instead of on every decode step
                per_rid = lookup.get(rid)
                if per_rid is None:
                    per_rid = lookup[rid] = {}
                key = step_key or (node_name, walk_of(rid))
                entry = per_rid.get(key)
                if entry is None:
                    walk = key[1]
                    label = self._keyed_label(rid, node_name, walk)
                    tensor = (
                        (self._overrides[rid].prefix_decode or {}).get(label)
                        if label is not None else None
                    )
                    entry = per_rid[key] = (label, tensor)
                label, tensor = entry
                if label is None or not tensor:
                    continue
                rows = rows_of.get(tensor)
                if rows is None:
                    buf = buffers.get(tensor)
                    if not torch.is_tensor(buf) or buf.dim() == 0:
                        continue
                    rows = rows_of[tensor] = buf.reshape(buf.shape[0], -1).tolist()
                i = row_of.get(rid)
                if i is None or i >= len(rows):
                    continue
                stream = self._streams.get(rid, {}).get(label)
                if stream is None or stream.chain is None or stream.chain.unkeyed is None:
                    continue
                if stream.released:
                    stream.chain = None
                    continue
                stream.chain.extend(rows[i], page_size)

    def flush_chain_backlog(self) -> None:
        """Extend the chains with every step waiting in the backlog."""
        if not self._chain_backlog:
            return
        with self._lock:
            # the swap under the lock, so two flushing threads (the stop check
            # and a removal) cannot both take the same steps
            steps, self._chain_backlog = self._chain_backlog, []
            self._chain_pending = Counter()
            self._chain_backlog_width = 0
            drained, self._chain_drained = self._chain_drained, set()
            # runs of steps with the same rows and batch go out as one extend
            # per request; a change of batch composition starts a new run
            run: list = []
            for st in steps:
                if run and run[0][:4] != st[:4]:
                    self._extend_chains_from(run, drained)
                    run = []
                run.append(st)
            if run:
                self._extend_chains_from(run, drained)

    def _drain_chain_backlog(self, rid: str) -> None:
        """Under the lock: extend one request's chain with its tokens waiting
        in the backlog, and leave them out of the next flush. A removal does
        this instead of a flush, which at high concurrency would come every
        step or two."""
        if not self._chain_backlog or rid in self._chain_drained:
            return
        self._chain_drained.add(rid)
        self._chain_pending.pop(rid, None)
        parts: list[tuple] = []
        for st in self._chain_backlog:
            try:
                i = st[0].index(rid)
            except ValueError:
                continue
            if rid in st[3]:
                parts.append((st, i))
        if not parts:
            return
        st0 = parts[0][0]
        walk = st0[2][st0[3].index(rid)]  # the rid's own walk in that step
        entry = self._chain_lookup.get(rid, {}).get((st0[1], walk))
        if entry is None:
            label = self._keyed_label(rid, st0[1], walk)
            tensor = (
                (self._overrides[rid].prefix_decode or {}).get(label)
                if label is not None else None
            )
        else:
            label, tensor = entry
        if label is None or not tensor:
            return
        stream = self._streams.get(rid, {}).get(label)
        if stream is None or stream.chain is None or stream.chain.unkeyed is None:
            return
        if stream.released:
            stream.chain = None
            return
        tokens: list[int] = []
        for st, i in parts:
            arr = st[4].get(tensor)
            if arr is not None and i < arr.shape[0]:
                tokens.extend(arr[i].tolist())
        if tokens:
            stream.chain.extend(tokens, self.config.page_size)

    def _extend_chains_from(self, steps: list, drained: set[str] = frozenset()) -> None:
        """Under the lock: one extend per request with the tokens of
        ``steps`` (same rows and batch) in step order."""
        rows, node_name, walks, request_ids, _ = steps[0]
        row_of = {rid: i for i, rid in enumerate(rows)}
        page_size = self.config.page_size
        lookup = self._chain_lookup
        cols: dict[str, object] = {}
        for rid, walk in zip(request_ids, walks, strict=True):
            if rid in drained:
                continue
            # the label and buffer name do not change over a request's
            # life (its overrides are set at ingest), so each is looked up
            # once instead of on every decode step
            per_rid = lookup.get(rid)
            if per_rid is None:
                per_rid = lookup[rid] = {}
            key = (node_name, walk)
            entry = per_rid.get(key)
            if entry is None:
                label = self._keyed_label(rid, node_name, walk)
                tensor = (
                    (self._overrides[rid].prefix_decode or {}).get(label)
                    if label is not None else None
                )
                entry = per_rid[key] = (label, tensor)
            label, tensor = entry
            if label is None or not tensor:
                continue
            i = row_of.get(rid)
            if i is None:
                continue
            col = cols.get(tensor)
            if col is None:
                parts = [st[4][tensor] for st in steps if tensor in st[4]]
                if not parts or i >= parts[0].shape[0]:
                    continue
                # (steps * values per row) tokens for every row, in step order
                col = cols[tensor] = np.concatenate(parts, axis=1)
            if i >= col.shape[0]:
                continue
            stream = self._streams.get(rid, {}).get(label)
            if stream is None or stream.chain is None or stream.chain.unkeyed is None:
                continue
            if stream.released:
                stream.chain = None
                continue
            stream.chain.extend(col[i].tolist(), page_size)

    def _adopt(self, rid: str, stream: CacheStream, pages: list[int]) -> None:
        """Make ``pages`` the whole of a stream that has written nothing yet.

        Any pages the stream already holds are empty ones a refused admit left, so they are released.
        """
        empty, stream.page_indices = stream.page_indices, list(pages)
        self._release(rid, stream, empty)
        self._plan_epoch += 1
        stream.stored_len = len(pages) * self.config.page_size

    def _release_lease(self, rid: str, stream: CacheStream, refund: bool = True) -> None:
        """Give back a lease `admit` never converted; a converted one is released
        with `page_indices`."""
        lease, stream.lease = stream.lease, None
        if lease is not None:
            self._release(rid, stream, lease, refund)

    def _release_hold(self, rid: str, stream: CacheStream) -> None:
        hold, stream.hold = stream.hold, None
        if hold is not None:
            self._release(rid, stream, hold)

    def fingerprint(self) -> bytes:
        return fingerprint(
            self.config.layout.value, self.config.page_size,
            self.kv_cache.tensor.dtype, self._world_size,
            self.config.prefix_cache_salt,
        )

    def ingest_request(self, rid, overrides: KVReqConfig | None=None):
        self._touch_plan_cache(rid)
        if rid in self._chain_drained:
            # the handle comes back while steps under its old name still wait
            self.flush_chain_backlog()
        if overrides is None:
            overrides = KVReqConfig()
        # guards `_streams`/`_overrides` against a concurrent admit/plan/commit
        # or reset/remove on another thread (see `_lock`)
        with self._lock:
            # Idempotent: the conductor sends one NewRequest per partition, all
            # carrying the same rid, so a worker serving two partitions ingests
            # twice. Replacing the streams here reset `stored_len` under a node
            # that had already filled them, and the request's publish info then
            # named more tokens than the stream held.
            self._streams.setdefault(rid, {"main": CacheStream()})
            self._overrides.setdefault(rid, overrides)
            for label, stream in self._streams[rid].items():
                if stream.span_cursor is None:
                    self._lay_out(rid, label, stream)
                if stream.chain is None:
                    self._seed_keys(rid, label, stream)

    def _lay_out(self, rid: str, label: str, stream: CacheStream) -> None:
        """A label the request doesn't open gets a footprint of 0, so the pages a batch writes
        there for it count as an overrun."""
        layout = self._held_layout(rid, label)
        if layout is not None:
            stream.footprint = layout.pages(self._overrides[rid].loop_caps, self.config.page_size)
            stream.span_cursor = SpanCursor(layout.spans)
        elif self._overrides.get(rid) is not None and self._overrides[rid].kv_layout:
            stream.footprint = 0

    def _held_layout(self, rid: str, label: str) -> StreamLayout | None:
        overrides = self._overrides.get(rid)
        return (overrides.kv_layout or {}).get(label) if overrides is not None else None

    def admit_retrieve(
        self, rid: str,
        node_name: str,
        graph_walk: str,
        published: PublishedKVInfo | None
    ) -> AdmitOutcome:
        self._touch_plan_cache(rid)
        if self._host_copies:
            with self._lock:
                self._land_host_copies()
        if rid in self._short and self.waits(rid):
            return _NOT_READY
        fetching = self._fetch_from_host(rid, node_name, graph_walk, published)
        if published is None:
            if fetching is None:
                return ADMIT_OK
            with self._lock:
                return AdmitOutcome(ok=True, ready=self._check_ready(rid, fetching)[0])

        if published.world_size != self._world_size:
            # terminal for this request, not for the worker serving it
            return AdmitOutcome(
                ok=False, ready=False,
                reason=AdmitRuntimeError(
                    "KV cache transfer across TP world size is currently "
                    f"disallowed (published {published.world_size}, "
                    f"local {self._world_size})"
                ),
            )
        overrides = self._overrides[rid]
        needed_labels = overrides.get_labels(node_name, graph_walk)
        rooted: dict[str, list[bytes]] = {}
        # Above one rank each rank runs this check at a different step, so it must not stamp or take
        # cached pages.
        if self._index is not None and overrides.prefix_cache and self._world_size == 1:
            # hashed here, as `resolve_cached_prefix` hashes outside its lock
            for label, keys in (overrides.prefix_keys or {}).items():
                if label in needed_labels and keys:
                    rooted[label] = [
                        fingerprint(self._prefix_root, key) for key in keys
                    ]
        # one critical section: reading stored_len, comparing to published, and
        # firing the retrieve must be atomic against a concurrent commit/reset
        # (both non-blocking inside, so holding the lock is safe)
        with self._lock:
            for label, seq_info in published.get(self._rank).items():
                if label not in needed_labels:
                    continue
                label_ready, failed = self._check_ready(rid, label)
                if failed is not None:
                    return AdmitOutcome(ok=False, ready=False, reason=failed)
                if not label_ready:
                    # read already in progress: admitted, just not ready yet
                    return AdmitOutcome(ok=True, ready=False)

                stream = self._ensure_label(rid, label)
                own = self._transfer.owns_transfer_info(
                    transfer_info=seq_info.latest_kv_transfer_info,
                    request_id=rid,
                    label=label,
                )
                remote_reset_generation = seq_info.reset_generation
                if (
                    not own
                    and remote_reset_generation is not None
                    and stream.remote_reset_generation is not None
                    and remote_reset_generation
                    != stream.remote_reset_generation
                ):
                    # The producer rewound/replaced this logical stream. Its
                    # seq_len may be unchanged, so length alone cannot prove
                    # the receiver still holds the published contents.
                    self._release_lease(rid, stream)
                    drop = self._arena.any_sealed(stream.page_indices)
                    if drop:
                        pages, stream.page_indices = stream.page_indices, []
                        self._release(rid, stream, pages)
                    stream.reset(freed=drop)
                    stream.forget_chain()
                    self._recount(rid)
                if not own and remote_reset_generation is not None:
                    stream.remote_reset_generation = remote_reset_generation
                if not own and stream.chain is not None:
                    # another rank sampled the token after this record, so the pending tail is one behind
                    stream.chain.unkeyed = None
                new_len = seq_info.seq_len
                self._take_local_match(rid, stream, new_len, rooted.get(label))
                old_len = stream.stored_len
                if new_len <= old_len:
                    continue
                if own:
                    # This shouldn't happen: the pages already ARE in this cache;
                    # opening our own IPC handle raises `invalid device context`
                    logger.warning(
                        "KV %s: skipping self-retrieve for %s label %s — "
                        "published %d tokens but the stream holds %d",
                        self.name, rid, label, new_len, old_len,
                    )
                    continue

                # _alloc takes a total length, not a delta
                alloc_res = self._alloc(rid, label, new_len)
                if not alloc_res.success:
                    return AdmitOutcome(ok=False, reason=alloc_res.error)

                fut = self._transfer.start_async_retrieve(
                    start_len=old_len, end_len=new_len,
                    local_page_indices=stream.page_indices,
                    remote_page_indices=seq_info.page_indices,
                    kv_transfer_info=seq_info.latest_kv_transfer_info
                )
                stream.read_future = fut
                stream.read_pending = fut is not None
                stream.stored_len = new_len

            ready = True
            for label in needed_labels:
                label_ready, failed = self._check_ready(rid, label)
                if failed is not None:
                    return AdmitOutcome(ok=False, ready=False, reason=failed)
                ready = ready and label_ready
        return AdmitOutcome(
            ok=True, ready=ready
        )

    def _fetch_from_host(
        self, rid: str, node_name: str, graph_walk: str, published: PublishedKVInfo | None,
    ) -> str | None:
        """If ``rid``'s prompt continues on the host past what the index holds,
        copy those pages back so its probe finds them. Returns the label whose
        stream waits on the copy.

        On a pool that decides admission the copy can take more pages than are free, and `_grant`
        evicts for the rest, since the walk would take as many pages to compute them. Any other pool
        has no allotment to bound the copy, so the copy takes free pages only. The walk computes
        whatever isn't copied.
        """
        if self._cpu_pool is None or self._index is None or not self._cpu_pool.index.num_pages():
            return None
        if rid in self._looked_up:
            return None
        # Readiness is checked for every row on every pass, so return before
        # taking the lock when no stream still needs a lookup or waits on a copy.
        streams = self._streams.get(rid, {})
        read_in = published.get(self._rank) if published is not None else {}
        if not any(
            label not in read_in
            and ((stream.chain is not None and not stream.chain.looked_up) or stream.host_copy is not None)
            for label, stream in list(streams.items())
        ):
            return None
        with self._lock:
            label = self._keyed_label(rid, node_name, graph_walk)
            if label is None or label in read_in:
                return None
            stream = self._ensure_label(rid, label)
            if stream.host_copy is not None and stream.read_pending:
                return label
            chain = stream.chain
            if (
                chain is None or chain.looked_up or stream.span_cursor is not None and stream.span_cursor.done
                or chain.spans[0].walk != graph_walk
                or stream.stored_len or stream.lease is not None or stream.converted
                or stream.offloaded or stream.read_pending
            ):
                return None
            if self._overrides[rid].get_labels(node_name, graph_walk) != [label]:
                # The engine doesn't probe a walk that writes two labels, so no probe would read a copy.
                # The chain is marked looked up, so later readiness checks return before taking the lock.
                chain.looked_up = True
                self._note_looked_up(rid)
                return None
        self._root_chain(chain)
        with self._lock:
            # a probe on the GPU thread may have looked the prompt up or taken a lease while the lock was dropped
            if (
                self._streams.get(rid, {}).get(label) is not stream
                or chain.looked_up or stream.lease is not None
            ):
                return None
            rooted = chain.rooted
            chain.looked_up = True
            # one page short of the whole prompt, as the probe matches
            cap = len(rooted) - 1
            run = self._index.pages_for(rooted[:cap])
            at = len(run)
            if at < cap and rooted[at] in self._arriving:
                stream.host_copy = self._arriving[rooted[at]]
                stream.read_pending = True
                return label
            budget = self._arena.num_free
            if self._decides:
                # the copy and the run's evictable pages come out of the allowance
                evictable = self._index.evictable()
                budget = self._allowance(rid) - sum(page in evictable for page in run)
            host = self._cpu_pool.retain_run(rooted[at:min(cap, at + budget)])
            if not host:
                self._note_looked_up(rid)
                return None
            # retained before the grant, whose eviction could drop the run's index-only pages
            held = self._retain_cached(rid, stream, run)
            fresh = self._grant(rid, label, len(host), 0)
            stream.hold = held + fresh
            stream.read_pending = True
            # listed before the lock drops, so `assert_pages_conserved` on another thread finds a record
            # for every page the copy holds
            copy = HostCopy(
                event=None, pages=fresh, keys=rooted[at:at + len(fresh)],
                parent=rooted[at - 1] if at else None, host_pages=host, rid=rid, stream=stream,
            )
            self._add_host_copy(copy)
            stream.host_copy = copy
            self._recount(rid)
        self._cpu_pool.copy_in(self.kv_cache.tensor, PageCopies(gpu=fresh, cpu=host))
        copy.event = self._cpu_pool.record()
        return label

    def _note_looked_up(self, rid: str) -> None:
        """Add ``rid`` to `_looked_up` once each of its streams has been looked up for a copy from the
        host and none waits on one. Callers hold the lock."""
        streams = self._streams.get(rid)
        if streams and all(
            (stream.chain is None or stream.chain.looked_up) and stream.host_copy is None
            for stream in streams.values()
        ):
            self._looked_up.add(rid)

    def admit(self, step: KVStep, ctx: StepContext) -> AdmitOutcome:
        if self._preplanned and not ctx.is_preplan:
            if self._preplan_key == self._plan_key(step, ctx):
                # pages were already reserved by the preplan pass — and the record
                # of them has to survive this call, because it is what a refusal
                # further down the resource order unwinds. Falling through to the
                # clear below would leak them.
                return ADMIT_OK
            # a different step arrived first (see `plan`): drop the staged
            # plan and reserve for this step normally. The staged step's own
            # span pages stay with its stream, where its re-admit finds them.
            # Its reservation went with ``clear_preplan``, so the record of it is
            # stale and the clear below is right for this path.
            self.clear_preplan()
        with self._lock:
            if self._admit_reserved_pages:
                self._forget_step_reservations(ctx)
            if self._host_copies:
                self._land_host_copies()
            # once a step, never in a readiness check
            if self._host_writes is not None and self._host_writes.landing:
                self._host_writes.land()
        if _DEBUG_ASSERTS and not step.commit:
            self._assert_scratch(step, ctx)
        # forks reserve here and copy later (plan for pre-, commit for post-),
        # so a step that never runs leaves pages resident but no page contents
        # moved — re-admitting it allocates nothing and re-copies nothing.
        # A post-fork copies the source *after* this step's spans land, so its
        # reservation covers them.
        # only a post-fork reads the growth, so a decode step does not build it
        growth = (
            self._label_growth(step) if step.commit and step.post_forks else {}
        )
        forks = [(pre, 0) for pre in step.pre_forks] + [
            (post, growth) for post in step.post_forks
        ]
        # one critical section so the read-of-stored_len then alloc is atomic
        # against a concurrent reset/remove/commit on another thread
        with self._lock:
            # before the fork loop and the segment loop, both of which size
            # off `stored_len`: a pre-fork target has to cover the whole prefix
            for segment in step.segments:
                stream = self._streams.get(
                    segment.request_id, {},
                ).get(segment.label)
                if stream is None:
                    continue
                if (
                    stream.lease is not None
                    and not stream.stored_len
                    and not stream.offloaded
                    and not stream.read_pending
                ):
                    assert stream.span_cursor.probed == stream.span_cursor.done, (
                        f"KV {self.name}: {segment.request_id}/{segment.label} admits a "
                        "walk its lease was never probed for, which would write over "
                        "the pages the lease holds for it"
                    )
                    whole = stream.lease_len // self.config.page_size
                    self._adopt(segment.request_id, stream, stream.lease[:whole] + stream.lease[whole + 1:])
                    stream.stored_len = stream.lease_len
                    # they came out of the index, so they are already in it
                    stream.chain.cursor = whole
                    # A cached image end stays leased until `plan` copies it on the step's stream.
                    stream.lease = stream.lease[whole:whole + 1] or None
                    if stream.lease is not None:
                        self._image_ends.append((segment.request_id, stream))
                    stream.converted = True
                    self._recount(segment.request_id)
                self._report_admission(segment, stream)
            if ctx.is_preplan:
                # a preplan that was promoted or abandoned already cleared
                # these; reset anyway so a refused admit can't leave stale
                # entries for the next clear_preplan to act on
                self._preplan_new_labels = []
                self._preplan_marked = []
            for (from_label, to_label), extra in forks:
                for rid in self._forking_rids(step, ctx):
                    # checked before the reservation, which is what creates it
                    if (
                        ctx.is_preplan
                        and to_label not in self._streams.get(rid, {})
                    ):
                        self._preplan_new_labels.append((rid, to_label))
                    alloc_res = self._reserve_fork(
                        rid, from_label, to_label,
                        extra=0 if not extra else extra.get((rid, from_label), 0),
                    )
                    if not alloc_res.success:
                        return AdmitOutcome(ok=False, reason=alloc_res.error)
                    if alloc_res.new_pages:
                        self._admit_reserved_pages.setdefault(
                            (rid, to_label), []).extend(alloc_res.new_pages)

            page_size = self.config.page_size
            sink = step.sink
            for segment in step.segments:
                # A replay's padding rows reserve nothing: they run against
                # SINK_PAGE (see `_sequence_views`). Pages for them would fail
                # to allocate on a near-full arena and hold the batch.
                if not self._reserves(ctx, sink, segment):
                    continue
                stream = self._ensure_label(segment.request_id, segment.label)
                seq_len = segment.span + stream.stored_len
                # the common decode row: its pages already cover the token it
                # appends, so there is nothing to acquire and nothing to record
                if (
                    not stream.offloaded
                    and len(stream.page_indices) * page_size >= seq_len
                ):
                    continue
                alloc_res = self._alloc(
                    segment.request_id, segment.label, seq_len,
                )
                if not alloc_res.success:
                    return AdmitOutcome(
                        ok=False, reason=self._sized_refusal(step, ctx, alloc_res.error),
                    )
                if alloc_res.new_pages:
                    self._admit_reserved_pages.setdefault(
                        (segment.request_id, segment.label), []
                    ).extend(alloc_res.new_pages)

            # marked here rather than in plan so the mark also covers
            # admit -> plan, where an offload would otherwise release pages
            # this step has already been given. last, and only once every
            # reservation above succeeded, so a refusal here sets none.
            # `.get` because a zero-span segment on a label nothing created
            # reserves no stream (see the loop above)
            for segment in step.segments:
                if self._sinks(ctx, sink, segment):
                    continue
                stream = self._streams.get(
                    segment.request_id, {}
                ).get(segment.label)
                if stream is None:
                    continue
                stream.step_in_flight = True
                if ctx.is_preplan:
                    self._preplan_marked.append(
                        (segment.request_id, segment.label)
                    )
            if _DEBUG_ASSERTS:
                self.assert_pages_conserved()
        # retention is applied at commit (see `_apply_retention`)

        return ADMIT_OK

    def _sized_refusal(
        self, step: KVStep, ctx: StepContext, error: AdmitFailedReason,
    ) -> AdmitFailedReason:
        """``error`` with the step's ``rows_fit`` and ``rows_fit_alone`` filled
        in, or a terminal refusal naming a request of the step that needs more
        pages than the whole pool has, the one ``error`` names first: no
        eviction can make that fit, so retrying it only spins until the client
        gives up. Caller holds ``_lock``.

        A request's footprint is every distinct page it would hold after the
        step: the ones its streams hold already, counted once where labels or
        requests share them, plus the ones `admit` would still reserve for each
        stream. A segment that runs against the sink holds and needs nothing,
        and a span-0 one needs nothing beyond what it holds, as in `admit`. The
        pool is every page but the sink, cached prefixes included, since those
        can be evicted. ``rows_sized`` is how many real rows were counted.
        """
        if not isinstance(error, AllocationFailed):
            return error
        page_size = self.config.page_size
        sink = step.sink
        held: dict[str, set[int]] = {}
        needed: dict[tuple[str, str], int] = {}
        for segment in step.segments:
            rid = segment.request_id
            if self._on_sink(ctx, rid):
                continue
            # a row with every label on the sink still counts as a row that fits
            rows_pages = held.setdefault(rid, set())
            if self._sinks(ctx, sink, segment):
                continue
            stream = self._streams.get(rid, {}).get(segment.label)
            pages = () if stream is None else stream.page_indices
            rows_pages.update(pages)
            if not self._reserves(ctx, sink, segment):
                continue
            stored = 0 if stream is None else stream.stored_len
            key = (rid, segment.label)
            more = -(-(stored + segment.span) // page_size) - len(pages)
            needed[key] = max(needed.get(key, 0), more)
        extra: dict[str, int] = {}
        for (rid, _), more in needed.items():
            extra[rid] = extra.get(rid, 0) + max(more, 0)
        capacity = self.config.max_num_pages - 1
        rows = list(held)
        # every row, not only the named one: a row refused ahead of one that
        # fills the pool names itself, and the eviction then takes the filler,
        # the only row holding pages
        for rid in sorted(rows, key=lambda rid: rid != error.request_id):
            own = len(held[rid]) + extra.get(rid, 0)
            if own > capacity:
                # the client reads this text, and the rid is the worker's
                # handle, not an id it sent
                return AdmitRuntimeError(
                    message=(
                        f"The request needs {own} pages of {self.name} for this "
                        f"step, more than the {capacity} the pool holds; raise "
                        "max_num_pages or shorten the request."
                    ),
                    request_ids=frozenset({rid}),
                )
        if error.request_id in rows:
            error.rows_fit = rows.index(error.request_id)
        resident: set[int] = set()
        new = alone = 0
        for rid in rows:
            resident |= held[rid]
            new += extra.get(rid, 0)
            if len(resident) + new > capacity:
                break
            alone += 1
        error.rows_fit_alone = alone
        error.rows_sized = len(rows)
        return error

    def rollback_admit(self, step: KVStep, ctx: StepContext) -> None:
        """Give back the pages this admit freshly acquired, and clear the
        ``step_in_flight`` marks over the step's segments as ``commit`` does.
        Does not touch pages retained by a shared prefix or a lease converted
        onto the stream.

        ``page_indices`` maps token position to page, so the reserved pages can
        only be dropped off the tail. If something else has extended the stream
        since, the tail can no longer be safely removed.
        """
        self._touch_plan_cache()
        with self._lock:
            # No owner check: every rollback finishes before `commit_done` is
            # released, and the only pre-admit (`Worker._preplan_spec`) waits on
            # it, so no other uncommitted step can hold marks on these streams.
            # `.get` as in `admit`: a refusal can come before a stream exists
            for segment in step.segments:
                stream = self._streams.get(
                    segment.request_id, {}
                ).get(segment.label)
                if stream is not None:
                    stream.step_in_flight = False
            reserved = self._admit_reserved_pages
            mine = [key for key in reserved if ctx.is_real_row(key[0])]
            for rid, label in mine:
                pages = reserved.pop((rid, label))
                if not pages:
                    continue  # `page_indices[:-0]` would empty the whole stream
                stream = self._streams.get(rid, {}).get(label)
                if stream is None:
                    continue  # reset or removed under us; its pages went with it
                if stream.page_indices[-len(pages):] != pages:
                    logger.error(
                        "KV %s: cannot unwind %s/%s's admit — its pages are no "
                        "longer the tail of the stream (%d reserved, stream has "
                        "%d). Leaving them; the free pool is short by that much.",
                        self.name, rid, label, len(pages),
                        len(stream.page_indices),
                    )
                    continue
                del stream.page_indices[-len(pages):]
                self._release(rid, stream, pages)
                self._recount(rid)
                # so an offload that claimed this stream mid-admit notices its
                # host copy no longer describes it and aborts
                stream.generation += 1
            # Record the room again after the rollback. Otherwise `preempt` would count the pages the step's other
            # rows gave back as released since the grant ran short.
            for segment in step.segments:
                if segment.request_id in self._ran_short:
                    self._ran_short[segment.request_id] = self._room()
            if _DEBUG_ASSERTS:
                self.assert_pages_conserved()

    def _forget_reservations(self, rid: int) -> None:
        """Drop ``rid``'s unwind record. Caller holds the lock."""
        for key in [k for k in self._admit_reserved_pages if k[0] == rid]:
            del self._admit_reserved_pages[key]

    def _forget_step_reservations(self, ctx: StepContext) -> None:
        """Drop the unwind record of this step's rows only. Caller holds the lock."""
        reserved = self._admit_reserved_pages
        for key in [k for k in reserved if ctx.is_real_row(k[0])]:
            del reserved[key]

    def _sequence_views(
        self,
        segments: list[Segment],
        pending_forks: dict[tuple[str, str], tuple[int, int]] | None = None,
        ctx: StepContext | None = None,
        sink: frozenset[tuple[str, str]] = frozenset(),
    ) -> list[SequenceView]:
        views = []
        page_size = self.kv_cache.page_size
        pending_forks = pending_forks or {}
        for s in segments:
            if ctx is not None and (self._on_sink(ctx, s.request_id) or (sink and (s.request_id, s.label) in sink)):
                # A padding row reads and writes SINK_PAGE; the engine
                # discards its output.
                views.append(SequenceView(
                    request_id=s.request_id, label=s.label,
                    # a sink page per page of span, not one: fewer and its writes spill into the previous row's pages
                    page_idxs=[SINK_PAGE] * -(-s.span // page_size),
                    length=s.span, to_compute=s.span,
                ))
                continue
            stream = self._streams[s.request_id][s.label]
            stored_len, generation = pending_forks.get(
                (s.request_id, s.label), (stream.stored_len, stream.generation)
            )
            # `page_indices` is a high-water mark, so a stream can hold more
            # pages than its tokens need (a refused admit, a reset that kept
            # its pages). slice to the length or the view addresses token 0
            # into the wrong page and reports the padding as resident context
            length = s.span + stored_len
            num_pages = -(-length // page_size)
            views.append(SequenceView(
                request_id=s.request_id,
                label=s.label, page_idxs=stream.page_indices[:num_pages],
                length=length,
                to_compute=s.span,
                generation=generation,
            ))
        return views

    def _compute_plan_state(
        self, cuda_indptrs: PagedIndptrs,
        total_tokens: int
    ) -> KVPlanState:
        qo_indptr = cuda_indptrs.qo_indptr
        paged_kv_indptr = cuda_indptrs.paged_kv_indptr
        paged_kv_last_page_len = cuda_indptrs.paged_kv_last_page_len
        paged_kv_indices = cuda_indptrs.paged_kv_indices

        # Compute per-token page and offset for vectorized KV writes
        n_req = qo_indptr.shape[0] - 1
        starts = qo_indptr[:-1].to(torch.int32)
        lens = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)

        # Pages/lengths AFTER append
        num_pages_after = (
            paged_kv_indptr[1:] - paged_kv_indptr[:-1]
        ).to(torch.int32)
        kv_len_after = (
            (num_pages_after - 1) * self.kv_cache.page_size + paged_kv_last_page_len
        )

        # Flatten to per-token indices
        # output_size keeps repeat_interleave from syncing to read `lens`
        seg = torch.repeat_interleave(
            torch.arange(n_req, dtype=torch.int32, device=self._device), lens,
            output_size=total_tokens
        )
        intra = torch.arange(
            total_tokens, dtype=torch.int32, device=self._device
        ) - torch.repeat_interleave(starts, lens, output_size=total_tokens)

        # Absolute KV position per token
        start_new = kv_len_after[seg] - lens[seg]
        g = start_new + intra

        # Map to page + offset
        page_off = torch.div(g, self.kv_cache.page_size, rounding_mode="floor").to(
            torch.int32
        )
        off_in_page = (g - page_off * self.kv_cache.page_size).to(torch.int32)
        abs_page_ptr = paged_kv_indptr[:-1][seg] + page_off

        return KVPlanState(
            token_to_page=paged_kv_indices[abs_page_ptr].to(torch.long),
            token_to_cache=off_in_page.to(torch.long),
            total_tokens=total_tokens
        )

    def _empty_plan_state(self) -> KVPlanState:
        empty = self._empty_locations
        if empty is None:
            empty = self._empty_locations = torch.empty(0, dtype=torch.long, device=self._device)
        return KVPlanState(token_to_page=empty, token_to_cache=empty, total_tokens=0)

    def _decode_plan_state(self, views: list[SequenceView]) -> KVPlanState:
        """Write addressing for a step appending one token per request.

        The packed path needs the indptrs on device and ~a dozen kernels to
        unpack them per token. A decode step's slot is just the end of each
        stream, so build it in the same CPU pass the views came from and send
        it over as one H2D.
        """
        pages, offsets = self._decode_locations(views)
        locations = torch.tensor(
            [pages, offsets], dtype=torch.long
        ).to(self._device, non_blocking=True)
        return KVPlanState(
            token_to_page=locations[0],
            token_to_cache=locations[1],
            total_tokens=len(views),
        )

    def _decode_locations(
        self, views: list[SequenceView],
    ) -> tuple[list[int], list[int]]:
        """Each decode row's (page, offset-in-page) for the token it writes.

        Off the stream's page count, not its logical length: that is what
        `build_paged_indptrs` hands attention, so a stream holding more pages
        than its length needs stays self-consistent."""
        page_size = self.kv_cache.page_size
        pages = [view.page_idxs[-1] for view in views]
        offsets = [(view.last_page_len(page_size) or page_size) - 1 for view in views]
        return pages, offsets

    def _stage_decode_plan_state(
        self, views: list[SequenceView] | None, static_state: KVPlanState,
        capture_len: int, pages=None, offsets=None,
    ) -> KVPlanState:
        """``_decode_plan_state`` + ``KVPlanState.copy_`` with no device staging
        tensor; rows past the real tokens get SINK_PAGE / 0."""
        if pages is None or offsets is None:
            pages, offsets = self._decode_locations(views)
        self._plan_stager.copy_(
            static_state.token_to_page[:capture_len], pages, pad_value=SINK_PAGE,
        )
        self._plan_stager.copy_(
            static_state.token_to_cache[:capture_len], offsets, pad_value=0,
        )
        static_state.total_tokens = len(pages)
        return static_state

    def _setup_plan_states(
        self, plan_output: dict[str, KVPlanOutput],
        ctx: StepContext, lease, eager_write: bool = False,
    ):
        if eager_write:
            # written outside the graph (KVStep.eager_write): never staged
            lease = None
        for label, indptrs in plan_output.items():
            if indptrs.is_decode and lease is not None:
                # Straight into the captured buffers in one H2D per buffer:
                # this runs in the pre-plan beside a live graph, where a
                # device-side copy + fill would slow it (see mstar.utils.h2d).
                # a cached plan hands its locations; reading `views` there
                # would build them eagerly and defeat the lazy builder
                cached = indptrs.decode_pages is not None
                plan_state = self._stage_decode_plan_state(
                    None if cached else indptrs.views,
                    self._static_plan_state(lease.slot, label),
                    lease.bucket.num_tokens,
                    pages=indptrs.decode_pages, offsets=indptrs.decode_offsets,
                )
                self._staged_outputs[(lease.slot, label)] = indptrs
                if ctx.is_preplan:
                    self._preplan_states[label] = plan_state
                else:
                    self._current_plan_states[label] = plan_state
                continue
            total_tokens = None if indptrs.is_decode else indptrs.get_total_len()
            if indptrs.is_decode:
                plan_state = self._decode_plan_state(indptrs.views)
            elif total_tokens == 0:
                # every segment span 0 (a decoder reading the encoder context
                # it wrote at prefill): nothing is written, so there is no
                # addressing to upload and unpack
                plan_state = self._empty_plan_state()
            else:
                indptrs.cuda_indptrs = indptrs.cpu_indptrs.to_device(self._device)
                plan_state = self._compute_plan_state(
                    indptrs.cuda_indptrs,
                    total_tokens=total_tokens,
                )
            if lease is not None:
                static_state = self._static_plan_state(lease.slot, label)
                static_state.copy_(plan_state, lease.bucket.num_tokens)
                plan_state = static_state
                self._staged_outputs[(lease.slot, label)] = indptrs
            elif eager_write and ctx.is_preplan and plan_state.token_to_page.is_cuda:
                plan_state.ready = torch.cuda.Event()
                plan_state.ready.record()
            if ctx.is_preplan:
                self._preplan_states[label] = plan_state
            else:
                self._current_plan_states[label] = plan_state


    def _plan_output(self, views: list[SequenceView], ring: PinnedIndexRing | None) -> KVPlanOutput:
        return KVPlanOutput(
            cpu_indptrs=build_paged_indptrs(views, self.kv_cache.page_size, ring=ring),
            views=views,
        )

    @staticmethod
    def _is_readonly(step: KVStep, ctx: StepContext) -> bool:
        # no sink rows: they change the views without changing the segments
        # the reuse key holds; the read-only steps that repeat have none.
        # Pre-planned steps too: a decoder that pre-plans gets most of its
        # steps there.
        return (
            not ctx.capture and bool(step.segments)
            and not step.pre_forks and not step.post_forks and not step.sink
            and all(seg.span == 0 for seg in step.segments)
        )

    def _readonly_key(self, step: KVStep, ctx: StepContext) -> tuple[tuple, tuple]:
        """(segments, bucket), and the epochs a plan under that key holds for."""
        lease = ctx.slot_lease
        return (
            (tuple(step.segments), lease.bucket if lease is not None else None),
            (self._plan_epoch, self._readonly_epoch),
        )

    def _reuse_readonly_plan(self, res: "KVPlanOutputs", step: KVStep, ctx: StepContext) -> "KVPlanOutputs":
        """A cached read-only plan for this step, its views and indptrs as they
        are; a slot whose static buffers already hold it is not re-staged."""
        lease = ctx.slot_lease
        if lease is not None and all(
            self._staged_outputs.get((lease.slot, label)) is out for label, out in res.items()
        ):
            states = self._preplan_states if ctx.is_preplan else self._current_plan_states
            for label in res:
                states[label] = self._static_plan_state(lease.slot, label)
        else:
            self._setup_plan_states(res, ctx, lease)
        self._mark_preplanned(step, ctx, res)
        return res

    def _drop_readonly_plans(self) -> None:
        self._readonly_epoch += 1
        # The epochs are what retire an entry: a plan thread mid-build stores
        # the epochs it read before this bump, which never match again. The
        # dict is rebound rather than cleared only so that thread keeps a
        # consistent one to write into.
        self._readonly_plans = {}
        self._readonly_rows = frozenset()

    def _touch_plan_cache(self, rid: str | None = None) -> None:
        """A stream of ``rid`` (any stream, when None) changed other than by a
        commit: the plan caches no longer describe the streams if they hold
        that request. A padding row's reset after a replay holds nothing."""
        if rid is None or rid in self._readonly_rows:
            self._drop_readonly_plans()
        cache = self._decode_plan_cache
        if cache is not None and (rid is None or any(rid == seg[0] for seg in cache.real_segments)):
            self._plan_epoch += 1

    def _split_padding(self, step: KVStep, ctx: StepContext):
        """``(n_real, (n_pad, span))`` of a step whose real segments come
        first and whose padding segments (a replay's dummy rows) all share
        one span; None when the layout is anything else."""
        segs = step.segments
        n_real = 0
        for seg in segs:
            if not ctx.is_real_row(seg.request_id):
                break
            n_real += 1
        pad = segs[n_real:]
        if not pad:
            return n_real, (0, 0)
        span = pad[0].span
        for seg in pad:
            if seg.span != span or ctx.is_real_row(seg.request_id):
                return None
        return n_real, (len(pad), span)

    def _remember_decode_plan(
        self, step: KVStep, ctx: StepContext, res: "KVPlanOutputs",
    ) -> None:
        """Keep a leased decode step's plan so the next step of the same rows
        can be planned off it (`_fast_decode_plan`). O(1): the full plan's
        views are kept as they are and the per-row arrays are derived at the
        first hit. Any other plan leaves the cache alone; the epoch and the
        commit bookkeeping say whether it still describes the streams."""
        lease = ctx.slot_lease
        if lease is None or ctx.capture or step.pre_forks or step.post_forks or step.sink:
            return
        split = self._split_padding(step, ctx)
        if split is None:
            return
        n_real, pad_layout = split
        for seg in step.segments:
            if seg.span != 1:
                return
        self._decode_plan_cache = _DecodePlanCache(
            step_segments=step.segments, real_segments=step.segments[:n_real],
            n_real=n_real, pad_layout=pad_layout, combined_labels=dict(step.combined_labels),
            views={label: out.views for label, out in res.items()},
            epoch=self._plan_epoch,
        )

    def _derive_rows(self, cache: _DecodePlanCache, ctx: StepContext) -> dict | None:
        """The per-label row arrays of a cached plan, from its views (once,
        at the first hit)."""
        page_size = self.kv_cache.page_size
        derived: dict[str, _LabelRows] = {}
        for label, views in cache.views.items():
            n_real = 0
            for v in views:
                if not ctx.is_real_row(v.request_id) and v.request_id not in self._streams:
                    break
                if not any(v.request_id == seg[0] and v.label == seg[1] for seg in cache.real_segments):
                    break
                n_real += 1
            real, pad = views[:n_real], views[n_real:]
            # a hit rebuilds each label's padding from all the step's padding
            # rows; in a combined label, a later source label's real rows can
            # follow an earlier one's padding
            if [(v.request_id, v.label, v.to_compute) for v in pad] != [
                (seg.request_id, seg.label, seg.span) for seg in cache.step_segments[cache.n_real:]
            ]:
                return None
            streams = []
            for v in real:
                stream = self._streams.get(v.request_id, {}).get(v.label)
                if stream is None or len(v.page_idxs) != -(-v.length // page_size):
                    return None
                streams.append(stream)
            row_pages = [v.page_idxs for v in real]
            derived[label] = _LabelRows(
                rows=tuple((v.request_id, v.label, 1) for v in real),
                streams=streams,
                lens=np.fromiter((v.length for v in real), np.int64, n_real),
                npages=np.fromiter((len(p) for p in row_pages), np.int64, n_real),
                row_pages=row_pages,
                last_page=np.fromiter((p[-1] for p in row_pages), np.int64, n_real),
                indices=np.fromiter(
                    itertools.chain.from_iterable(row_pages), np.int32, sum(len(p) for p in row_pages),
                ),
                gens=np.fromiter((v.generation for v in real), np.int64, n_real),
                n_pad=len(pad), pad_span=pad[0].to_compute if pad else 0,
                pad_pages=len(pad[0].page_idxs) if pad else 0,
                built=None,
            )
        return derived

    def _fast_decode_plan(
        self, step: KVStep, ctx: StepContext,
    ) -> "KVPlanOutputs | None":
        """The plan of a leased decode step off the previous step's, when the
        real rows are the same in the same order and only that step's commit
        touched the streams (`_plan_epoch`, `pending`): every length is one
        more, a page is new only where a row crossed a page boundary, the
        index arrays come from numpy and the views are built only if a
        consumer asks. None means plan from the streams."""
        cache = self._decode_plan_cache
        stats = self.plan_cache_stats
        if (
            cache is None
            or ctx.slot_lease is None
            or ctx.capture
            or step.pre_forks
            or step.post_forks
        ):
            return None
        if cache.epoch != self._plan_epoch or cache.pending != 1:
            stats["miss"] += 1
            return None
        n_real = cache.n_real
        segs = step.segments
        if (
            len(segs) != n_real + cache.pad_layout[0]
            or segs[:n_real] != cache.real_segments
            or step.combined_labels != cache.combined_labels
            or any(seg.span != cache.pad_layout[1] or ctx.is_real_row(seg.request_id) for seg in segs[n_real:])
            # padding keeps its labels: their order places the real rows in a combined label
            or any(
                seg.label != prev.label
                for seg, prev in zip(segs[n_real:], cache.step_segments[n_real:], strict=True)
            )
        ):
            stats["miss"] += 1
            return None
        if cache.derived is None:
            cache.derived = self._derive_rows(cache, ctx)
            if cache.derived is None:
                self._decode_plan_cache = None
                stats["miss"] += 1
                return None
        page_size = self.kv_cache.page_size
        pad_segs = segs[n_real:]
        make = self._indptr_ring.take if self._indptr_ring is not None else torch.from_numpy
        out: dict[str, KVPlanOutput] = {}
        for label, rows in cache.derived.items():
            lens = rows.lens + 1
            last = (lens - 1) % page_size + 1
            npages = (lens + page_size - 1) // page_size
            crossed = np.flatnonzero(npages != rows.npages)
            # the one commit since bumped each row; a new page's alloc bumps it again
            gens = rows.gens + 1
            if crossed.size:
                # copy only while the last plan's unbuilt views still hold the list
                row_pages = rows.row_pages
                if rows.built is not None and not rows.built[0]:
                    row_pages = list(row_pages)
                streams, last_page = rows.streams, rows.last_page
                for i in crossed.tolist():
                    want = int(npages[i])
                    stream = streams[i]
                    held = stream.page_indices
                    if len(held) < want:
                        stats["miss"] += 1
                        return None
                    pages = held[:want]
                    row_pages[i] = pages
                    last_page[i] = pages[-1]
                    gens[i] = stream.generation
                rows.row_pages = row_pages
                rows.indices = np.fromiter(
                    itertools.chain.from_iterable(row_pages), np.int32, int(npages.sum()),
                )
            rows.lens = lens
            rows.npages = npages
            rows.gens = gens
            n_pad, pad_span, pad_pages = rows.n_pad, rows.pad_span, rows.pad_pages
            if n_pad:
                pad_last = (pad_span - 1) % page_size + 1
                last_all = np.concatenate((last, np.full(n_pad, pad_last, np.int64)))
                npages_all = np.concatenate((npages, np.full(n_pad, pad_pages, np.int64)))
                indices = np.concatenate((rows.indices, np.full(n_pad * pad_pages, SINK_PAGE, np.int32)))
                pages_all = np.concatenate((rows.last_page, np.full(n_pad, SINK_PAGE, np.int64)))
            else:
                last_all, npages_all, indices, pages_all = last, npages, rows.indices, rows.last_page.copy()
            kv_indptr = np.concatenate((np.zeros(1, np.int64), np.cumsum(npages_all))).astype(np.int32)
            # this label's rows: `n_real` counts the real rows of every label
            n_rows = len(rows.rows) + n_pad
            qo_indptr = np.arange(n_rows + 1, dtype=np.int32)
            kv_lens = (npages_all - 1) * page_size + last_all
            row_pages_now, lens_now = rows.row_pages, lens.tolist()
            pad_now = [(seg.request_id, seg.label, seg.span) for seg in pad_segs]
            rows.built = built = [False]

            def build_views(
                row_pages_now=row_pages_now, lens_now=lens_now, gens_now=gens, pad_now=pad_now, rows=rows,
                built=built,
            ):
                built[0] = True
                # generations as of this plan, not read off the streams a later commit bumps
                views = [
                    SequenceView(rid, lab, pages, length, 1, 0, gen)
                    for (rid, lab, _), pages, length, gen in zip(
                        rows.rows, row_pages_now, lens_now, gens_now.tolist(), strict=True,
                    )
                ]
                views.extend(
                    SequenceView(rid, lab, [SINK_PAGE] * rows.pad_pages, span, span) for rid, lab, span in pad_now
                )
                return views

            out[label] = KVPlanOutput(
                cpu_indptrs=PagedIndptrs(
                    qo_indptr=make(qo_indptr),
                    paged_kv_indptr=make(kv_indptr),
                    paged_kv_indices=make(indices),
                    paged_kv_last_page_len=make(last_all.astype(np.int32)),
                    kv_lens=torch.from_numpy(kv_lens.astype(np.int32)).as_subclass(HostLens),
                ),
                decode_pages=pages_all,
                decode_offsets=last_all - 1,
                rows=rows.rows + tuple(pad_now),
                view_builder=build_views,
            )
        cache.step_segments = step.segments
        cache.pending = 0
        stats["hit"] += 1
        return KVPlanOutputs(out, pre_forks=step.pre_forks, post_forks=step.post_forks)

    @staticmethod
    def _same_plan(a: "KVPlanOutputs", b: "KVPlanOutputs") -> bool:
        """The cached plan against the full one (MSTAR_KV_PLAN_CACHE=2)."""
        if a.keys() != b.keys():
            return False
        for label, x in a.items():
            y = b[label]
            if len(x.views) != len(y.views):
                return False
            for u, v in zip(x.views, y.views, strict=True):
                if u[:5] != v[:5] or u.generation != v.generation:
                    return False
            for p, q in zip(x.cpu_indptrs[:5], y.cpu_indptrs[:5], strict=True):
                if (p is None) != (q is None) or (
                    p is not None and not torch.equal(p, q)
                ):
                    return False
            if x.decode_pages is not None:
                pages = [v.page_idxs[-1] for v in y.views]
                if x.decode_pages.tolist() != pages:
                    return False
        return True

    def _maybe_apply_forks(self, step: KVStep, ctx: StepContext):
        for (from_label, to_label) in step.pre_forks:
            for rid in self._forking_rids(step, ctx):
                self._apply_fork(rid, from_label, to_label)

    def _copy_image_ends(self) -> None:
        """Copy each cached image end still under lease onto the page its stream writes next."""
        if not self._image_ends:
            return
        with self._lock:
            ends, self._image_ends = self._image_ends, []
            for rid, stream in ends:
                # a stream reset since its admit is no longer converted, and a lease it holds now is a later probe's
                if not stream.converted:
                    continue
                if stream.lease is not None and stream.stored_len:
                    whole, end = divmod(stream.stored_len, self.config.page_size)
                    self.kv_cache.copy_slots(stream.lease[0], stream.page_indices[whole], end)
                # no walk writes the cached image end, so releasing it also shrinks the allotment
                self._release_lease(rid, stream, refund=False)
                self._recount(rid)

    def _pending_fork_state(
        self, step: KVStep, ctx: StepContext
    ) -> dict[tuple[str, str], tuple[int, int]]:
        """``(stored_len, generation)`` each pre-fork will hand its target.

        Staging cannot run the fork (`copy_pages` would race the default
        stream), so the target's length and bumped generation are read ahead
        and the copy happens at promotion.
        """
        pending: dict[tuple[str, str], tuple[int, int]] = {}
        with self._lock:
            for (from_label, to_label) in step.pre_forks:
                for rid in self._forking_rids(step, ctx):
                    labels = self._streams.get(rid, {})
                    src = labels.get(from_label)
                    dst = labels.get(to_label)
                    if src is None or dst is None:
                        continue
                    pending[(rid, to_label)] = (
                        src.stored_len, dst.generation + 1,
                    )
        return pending

    def plan(self, step: KVStep, ctx: StepContext) -> dict[str, KVPlanOutput]:
        """
        Returns list of sequence views per plan label
        """
        assert not (self._preplanned and ctx.is_preplan), (
            "KV preplan is already pending; clear_preplan before planning a "
            "different step ahead"
        )
        laps = PhaseLaps(
            "res.pre_plan.kv_cache." if ctx.is_preplan else "res.plan.kv_cache."
        )
        self.reset_default_cursors()
        if self._preplanned:
            if self._preplan_key == self._plan_key(step, ctx):
                self._current_plan_states = self._preplan_states
                res = self._cached_plan_output

                # Promotion: apply the fork copies staging left undone; see
                # `_pending_fork_state`.
                self._maybe_apply_forks(step, ctx)
                self._copy_image_ends()

                self._preplan_new_labels = []
                self._preplan_marked = []
                # the promoted step runs; the plan cache still describes it
                self.clear_preplan(promoted=True)
                laps.lap("promote")
                return res
            # A different step reached the GPU thread before the one planned
            # ahead (e.g. a new request's prefill while a decode step sits
            # pre-planned): it must not be served the staged plan's pages.
            # Undo the staged plan's side effects and plan inline.
            self.clear_preplan()

        readonly = self._is_readonly(step, ctx)
        if readonly:
            key, epochs = self._readonly_key(step, ctx)
            slot = ctx.slot_lease.slot if ctx.slot_lease is not None else None
            cached = self._readonly_plans.get(slot)
            if cached is not None and cached[0] == key and cached[1] == epochs:
                return self._reuse_readonly_plan(cached[2], step, ctx)
            # The epochs and rows before the views: a touch from here on moves
            # an epoch, so a plan built across it is never reused. A pre-plan
            # builds on the plan thread while the main thread can tear
            # requests down.
            self._readonly_rows = self._readonly_rows | frozenset(
                seg.request_id for seg in step.segments if not ctx.is_padding_row(seg.request_id)
            )

        # Forks run ahead of the views, which record length and generation;
        # staging only reads what the fork will land.
        if ctx.is_preplan:
            pending = self._pending_fork_state(step, ctx)
        else:
            pending = {}
            self._maybe_apply_forks(step, ctx)
            self._copy_image_ends()

        # a cached read-only plan outlives the ring's turn, so it takes none
        ring = None if readonly else self._indptr_ring
        if ring is not None:
            ring.begin_plan()
        laps.lap("forks")
        res = None
        if self._plan_cache_mode and not pending:
            res = self._fast_decode_plan(step, ctx)
            laps.lap("fast_decode" if res is not None else "fast_decode_miss")
        if res is None or self._plan_cache_mode == 2:
            full = KVPlanOutputs(
                {
                    plan_label: self._plan_output(
                        self._sequence_views(segments, pending, ctx, step.sink), ring,
                    )
                    for plan_label, segments in group_by_plan_label(
                        step.segments, step.combined_labels
                    ).items()
                },
                pre_forks=step.pre_forks,
                post_forks=step.post_forks,
            )
            if res is not None and not self._same_plan(res, full):
                self.plan_cache_stats["mismatch"] += 1
                logger.error(
                    "KV %s: the plan cache disagrees with the full plan for "
                    "%d segments; using the full plan (MSTAR_KV_PLAN_CACHE=2)",
                    self.name, len(step.segments),
                )
                res = None
            if res is None:
                res = full
                if self._plan_cache_mode and not pending:
                    self._remember_decode_plan(step, ctx, res)
            laps.lap("full")
        self._setup_plan_states(res, ctx, ctx.slot_lease, eager_write=step.eager_write)
        laps.lap("setup_states")
        if readonly:
            self._readonly_plans[slot] = (key, epochs, res)
        self._mark_preplanned(step, ctx, res)
        return res

    def _mark_preplanned(self, step: KVStep, ctx: StepContext, res: "KVPlanOutputs") -> None:
        if ctx.is_preplan:
            self._preplan_key = self._plan_key(step, ctx)
            self._preplanned = True
            self._cached_plan_output = res

    @property
    def supports_preplan(self):
        return True

    @staticmethod
    def _plan_key(step: KVStep, ctx: StepContext):
        """What identifies the step a pre-plan was staged for: its segments
        and the replay slot it was leased on."""
        lease = ctx.slot_lease
        return tuple(step.segments), (lease.slot if lease is not None else None), step.fork_rids, step.sink

    @staticmethod
    def _forking_rids(step: KVStep, ctx: StepContext):
        """The step's real rows its forks apply to: all of them unless it names some."""
        return [
            rid for rid in ctx.padded_request_ids
            if not KVManager._on_sink(ctx, rid)
            and (step.fork_rids is None or rid in step.fork_rids)
        ]

    @staticmethod
    def _sinks(ctx: StepContext, sink: frozenset, segment: Segment) -> bool:
        """Whether ``segment`` runs against SINK_PAGE: its row does (see
        `_on_sink`), or the step puts its label there."""
        return KVManager._on_sink(ctx, segment.request_id) or bool(
            sink and (segment.request_id, segment.label) in sink
        )

    @staticmethod
    def _reserves(ctx: StepContext, sink: frozenset, segment: Segment) -> bool:
        """Whether `admit` reserves pages for ``segment``: it writes tokens,
        and not against the sink. `_sized_refusal` counts the same ones."""
        return bool(segment.span) and not KVManager._sinks(ctx, sink, segment)

    @staticmethod
    def _on_sink(ctx: StepContext, rid) -> bool:
        """Whether ``rid`` runs against SINK_PAGE and holds no pages: a
        replay's padding row, or any row of a capture. A capture's rows are the
        padding rows its replays will run, so they run the same way, and no
        bucket's sizing or capture depends on how many pages the pool has."""
        return ctx.capture or ctx.is_padding_row(rid)

    def clear_preplan(self, promoted: bool = False):
        if not promoted:
            self._touch_plan_cache()
        # the staged step is not going to run, so undo what it did to live
        # state. the pre-forks need no rewinding — staging only read them —
        # but admit's reservations are real pages and real streams
        with self._lock:
            # labels the staged step invented are removed, not rewound to 0:
            # a stream at 0 that nothing asked for is still a stream, and it
            # holds the pages the reservation took
            for rid, label in reversed(self._preplan_new_labels):
                stream = self._streams.get(rid, {}).pop(label, None)
                if stream is not None:
                    self._release(rid, stream, stream.page_indices)
                    self._recount(rid)
            self._preplan_new_labels = []
            for rid, label in self._preplan_marked:
                stream = self._streams.get(rid, {}).get(label)
                if stream is not None:
                    stream.step_in_flight = False
            self._preplan_marked = []
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_key = None
        self._preplan_states = {}
        self._cached_plan_output = None

    def commit(self, step: KVStep, ctx: StepContext):
        streams = self._streams
        retention = step.retention
        # what the plan cache may assume of the next step: this step's spans
        # landed, nothing else (a step that keeps no tokens is "something else")
        cache = self._decode_plan_cache
        if cache is not None:
            if step.segments is cache.step_segments:
                cache.pending += 1
            elif len(step.segments) >= len(cache.step_segments) or any(
                (seg.request_id, seg.label) in cache.rows for seg in step.segments
            ):
                # another batch that is as big (surely sharing rows) or that
                # names a cached row: the cache no longer describes the streams
                self._plan_epoch += 1
        if not step.commit:
            self._plan_epoch += 1
        rows = self._readonly_rows
        if rows and any(seg.span and seg.request_id in rows for seg in step.segments):
            self._drop_readonly_plans()
        # the index only sees real rows of a non-capture step; checked once
        index_on = self._index is not None and not ctx.capture
        # atomic against admit_retrieve reading stored_len on another thread
        with self._lock:
            writes = self._host_writes
            if writes is not None:
                writes.commits += 1
                # read by `after_output_copy` once the worker has seen this step finish
                object.__setattr__(step, "commits", writes.commits)
            # committed, so there is nothing left to unwind
            if self._admit_reserved_pages:
                self._forget_step_reservations(ctx)
            sink = step.sink
            for segment in step.segments:
                if self._on_sink(ctx, segment.request_id) or (sink and (segment.request_id, segment.label) in sink):
                    continue  # ran against SINK_PAGE, holds nothing
                stream = streams[segment.request_id][segment.label]
                # cleared before the `step.commit` test: a step that keeps no
                # tokens (image_gen, action_gen) still read these pages, and
                # leaving the mark set would make the request unevictable
                stream.step_in_flight = False
                if step.commit and segment.span > 0:
                    # an offload beat the mark (claimed before this step's
                    # admit). the host copy predates the span, so writing the
                    # length here would be lost on reload
                    if stream.offloaded:
                        logger.warning(
                            "KV %s: dropping %d committed tokens for %s label "
                            "%s; the stream was offloaded mid-step",
                            self.name, segment.span,
                            segment.request_id, segment.label,
                        )
                        self._plan_epoch += 1
                        continue
                    stream.stored_len += segment.span
                    if stream.converted and self._world_size > 1 and stream.chain is not None:
                        # Above one rank a hit is stamped at commit, which every rank runs in step order,
                        # so every rank evicts the same pages.
                        self._index.touch(stream.page_indices[:stream.chain.cursor])
                    # committed, so there is no refused admit left for a re-probe to answer
                    stream.converted = False
                    policy = (
                        retention.get((segment.request_id, segment.label))
                        if retention else None
                    )
                    if policy is not None:
                        self._adopt_retention(stream, policy, segment)
                    touched = []
                    if stream.span_cursor is not None and (stream.chain is not None or _DEBUG_ASSERTS):
                        written = stream.stored_len + stream.released
                        touched = stream.span_cursor.advance(written - segment.span, written)
                        if _DEBUG_ASSERTS:
                            self._assert_layout(segment, stream, touched, ctx, policy)
                    if index_on and stream.chain is not None:
                        self._index_filled_pages(segment, stream, ctx, touched)
                    # so a claim taken in a window the mark misses still fails
                    # `_commit_offload`'s generation guard
                    stream.generation += 1
                    # The step's retention, if it declared one: drop what aged
                    # past the context budget now that this step's tokens
                    # count. Here, under the lock and before `commit_done`
                    # lets the next step pre-plan, so no admitted plan
                    # addresses the pages this frees (this step's own kernels
                    # may still be reading them, but every later user of the
                    # pages queues behind them on the node's stream, the host
                    # pool's copies included)
                    if policy is not None:
                        self._apply_retention(segment.request_id, stream, policy)
            # post-forks copy what this step just wrote, so they land after the
            # spans above are counted
            for (from_label, to_label) in step.post_forks:
                for rid in self._forking_rids(step, ctx):
                    self._apply_fork(rid, from_label, to_label)
            if _DEBUG_ASSERTS:
                self.assert_pages_conserved()

    def _assert_layout(
        self, segment: Segment, stream: CacheStream, touched: list[tuple[int, int]],
        ctx: StepContext, policy: RetentionPolicy | None,
    ) -> None:
        """Debug only, since it looks the layout up per segment."""
        overrides = self._overrides[segment.request_id]
        layout = overrides.kv_layout[segment.label]
        walk = ctx.walk_of(segment.request_id) if ctx.request_walks else ctx.graph_walk
        where = f"KV {self.name}: {segment.request_id}/{segment.label}"
        for index, _ in touched:
            assert stream.span_cursor.spans[index].walk == walk, (
                f"{where}: {walk} wrote into span {index}, which its layout gives to "
                f"{stream.span_cursor.spans[index].walk}"
            )
        written, bound = stream.stored_len + stream.released, layout.slots(overrides.loop_caps)
        assert written <= bound, f"{where}: committed {written} slots, past its layout's spans and growth ({bound})"
        assert policy is None or policy == layout.retention, (
            f"{where}: a step declares {policy}, where its layout declares {layout.retention}"
        )

    # Partial release behind a protected prefix (windowed generation): a
    # request that generates in windows commits each window's K/V and, once
    # its context horizon fills, drops the oldest generated pages while the
    # prompt prefix stays. Two routes to the same page-floored front release:
    # a `RetentionPolicy` the committing step declares (`KVStep.retention`),
    # applied inside that commit — the served route, safe under pre-planning —
    # and the explicit `protect_prefix` / `release_oldest` pair for a driver
    # that runs between steps. Ported from #198's PagedAllocationManager
    # (merceod) onto the pool's streams.

    @torch.compiler.disable
    def protect_prefix(
        self, request_id: str, num_tokens: int, label: str | None = None,
    ) -> None:
        """Mark the first ``num_tokens`` committed tokens of the stream as never
        releasable. Set once, after the prefix commits and before any release;
        idempotent at the same value."""
        if label is None:
            label = self._default_label
        with self._lock:
            stream = self._streams[request_id][label]
            if num_tokens < 0 or num_tokens > stream.stored_len:
                raise ValueError(
                    f"protect_prefix({num_tokens}) outside the committed {stream.stored_len} "
                    f"tokens of request {request_id!r} label {label!r}"
                )
            if stream.released:
                raise ValueError(
                    f"protect_prefix must precede any release_oldest for request "
                    f"{request_id!r} label {label!r}"
                )
            if stream.protected_prefix not in (0, num_tokens):
                raise ValueError(
                    f"protected prefix already {stream.protected_prefix} tokens for "
                    f"request {request_id!r} label {label!r}, got {num_tokens}"
                )
            stream.protected_prefix = num_tokens

    def _adopt_retention(
        self, stream: CacheStream, policy: RetentionPolicy, segment: Segment,
    ) -> None:
        """Take the policy a committing step declares for its stream: the
        protected prefix must be committed and agree with any earlier one
        (`protect_prefix`, or the previous window's). The budget may change
        between windows. Under the lock."""
        if policy.protected_prefix > stream.stored_len:
            raise ValueError(
                f"retention for request {segment.request_id!r} label {segment.label!r} "
                f"protects {policy.protected_prefix} tokens but only "
                f"{stream.stored_len} are committed"
            )
        if stream.protected_prefix not in (0, policy.protected_prefix):
            raise ValueError(
                f"protected prefix already {stream.protected_prefix} tokens for "
                f"request {segment.request_id!r} label {segment.label!r}, got "
                f"{policy.protected_prefix}"
            )
        stream.protected_prefix = policy.protected_prefix
        stream.retention = policy

    def _apply_retention(self, rid: str, stream: CacheStream, policy: RetentionPolicy) -> int:
        """Release what the step's policy no longer keeps. Under the lock."""
        excess = stream.stored_len - policy.protected_prefix - policy.context_budget
        if excess <= 0:
            return 0
        return self._release_oldest_locked(rid, stream, excess)

    def _release_oldest_locked(self, rid: str, stream: CacheStream, num_tokens: int) -> int:
        """The page-floored front release shared by ``release_oldest`` and the
        commit-time retention. Under the lock."""
        self._touch_plan_cache()
        page_size = self.config.page_size
        first = (stream.protected_prefix + page_size - 1) // page_size
        releasable = stream.stored_len // page_size - first
        k = min(num_tokens // page_size, releasable)
        if k <= 0:
            return 0
        freed = stream.page_indices[first:first + k]
        del stream.page_indices[first:first + k]
        stream.stored_len -= k * page_size
        stream.released += k * page_size
        stream.generation += 1
        self._release(rid, stream, freed)
        self._recount(rid)
        return k * page_size

    @torch.compiler.disable
    def release_oldest(
        self, request_id: str, num_tokens: int, label: str | None = None,
    ) -> int:
        """Free the oldest unprotected committed tokens of a live stream, whole
        pages only, compacting the page list so the remaining stream stays
        contiguous in page-list order. Returns the tokens actually freed.

        The freed span starts at the first page fully past the protected
        prefix; a page straddling the protection boundary and a partially
        filled tail page are never freed, so the realized release can fall
        short of ``num_tokens`` by up to a page — callers re-offer the
        shortfall next time (see ``WindowedKVSession``). ``stored_len`` drops
        by exactly the freed count and ``generation`` moves, so a prefix a
        backend gathered out of these pages is re-read (the dense backend keys
        its gathered prefix on it). Positions are not touched: the tokens that
        remain keep the absolute positions their K/V was written with.

        Refused under an admitted step (its plan addresses these pages) and
        while the stream is offloaded or being retrieved.
        """
        if label is None:
            label = self._default_label
        with self._lock:
            stream = self._streams[request_id][label]
            if stream.step_in_flight:
                raise RuntimeError(
                    f"release_oldest on request {request_id!r} label {label!r} under an "
                    "admitted step; release between steps"
                )
            if stream.offloaded or stream.read_pending:
                raise RuntimeError(
                    f"release_oldest on request {request_id!r} label {label!r} while its "
                    "pages are offloaded or in transfer"
                )
            return self._release_oldest_locked(request_id, stream, num_tokens)

    # Eviction

    @property
    def supports_eviction(self):
        return self._cpu_pool is not None

    def is_offloaded(self, rid: str) -> bool:
        """True from the moment an offload claims the request, not just once
        its pages are on the host.

        ``check_ready`` gates admission on this, so the window where the copy
        is still in flight must not look schedulable — and the worker's victim
        filter must not pick a request that is already on its way out.
        """
        if self._cpu_pool is None:
            return False
        if self._cpu_pool.is_offloaded(rid):
            return True
        # every writer of `_streams` holds the lock, and the worker calls this
        # from its victim filter while steps are running. reentrant, so callers
        # already under the lock are unaffected
        with self._lock:
            return any(
                stream.offloaded for stream in self._streams.get(rid, {}).values()
            )

    def is_resident(self, rid: str) -> bool:
        """False once ``rid``'s offload is claimed or its drop is applied."""
        if rid in self._dropped:
            return False
        return self._cpu_pool is None or not self.is_offloaded(rid)

    def check_drop(self, rid: str) -> str | None:
        if self._world_size > 1:
            return (
                f"KV {self.name}: dropping {rid} on one of {self._world_size} ranks "
                "would leave them holding different pages"
            )
        if self.is_offloaded(rid):
            return f"KV {self.name}: {rid} has a host copy, which a drop would leave behind"
        with self._lock:
            if any(stream.step_in_flight or stream.read_pending for stream in self._streams[rid].values()):
                return f"KV {self.name}: {rid} has a step or a read in flight on its pages"
            if self._overrides[rid].guaranteed:
                return f"KV {self.name}: {rid} is guaranteed, so it is never preempted"
            why = rebuild_refusal((self._overrides[rid].kv_layout or {}).values())
            if why is not None:
                return f"KV {self.name}: {rid} can't be dropped, since {why}"
        return None

    def drop(self, rid: str) -> int:
        """Take ``rid``'s KV off the device without a host copy, and return the
        number of pages freed. Pages the index also holds stay cached, where they
        can be evicted, and the rest go back to the free list."""
        with self._lock:
            refusal = self.check_drop(rid)
            if refusal is not None:
                raise RuntimeError(refusal)
            # under the lock, so no other step allocates between the two counts
            free, room = self._arena.num_free, self._room()
            # A rebuild that hasn't landed wrote nothing, so a request dropped again keeps what its first drop
            # recorded.
            dropped = self._dropped.get(rid)
            if dropped is None:
                # It returns with its history (the tokens its KV held, released ones included, and the sample its
                # next step writes) and one page more, at most its footprint.
                streams = self._streams[rid].values()
                written = max(stream.stored_len + stream.released for stream in streams)
                history = -(-(written + 1) // self.config.page_size)
                returns = sum(self._one_more(history, stream.footprint) for stream in streams)
                # `check_drop` allows at most one laid-out label
                laid_out = [label for label, layout in (self._overrides[rid].kv_layout or {}).items() if layout.spans]
                dropped = Dropped(returns, sum(self._streams[rid][label].stored_len for label in laid_out))
            self.reset_request(rid, free=True)
            self._dropped[rid] = dropped
            self._vacate(rid, room)
            return self._arena.num_free - free

    def prepare_rebuild(self, rid: str, rebuild: Rebuild) -> None:
        """Lay ``rid``'s label out again as one ids span of the rebuild's walk over its ids without the last
        one, keyed as a prompt of those ids would be.

        The growth loop's cap drops by the iterations the span now covers, so the footprint stays
        the one at intake. The decode step that samples an id keys it, so a label that keys
        generation also extends its chain by the last id.
        """
        overrides = self._overrides[rid]
        laid_out = [label for label, layout in (overrides.kv_layout or {}).items() if layout.spans]
        if not laid_out:
            return
        (label,) = laid_out
        ids, walk = rebuild.ids, rebuild.walk
        prompt, page_size = ids[:-1], self.config.page_size
        length = len(prompt)
        held = self._dropped[rid].held
        if length != held:
            raise RuntimeError(f"KV {self.name}: {rid}'s rebuild writes {length} tokens, but its label held {held}")
        pages = [prompt[start:start + page_size] for start in range(0, length, page_size)]
        keyed = bool((overrides.prefix_keys or {}).get(label))
        # hashed before the lock, as `resolve_cached_prefix` hashes outside it
        keys = chain(pages) if keyed else None
        with self._lock:
            layout = overrides.kv_layout[label]
            ((loop, slots),) = layout.growth
            grown = length - sum(span.length for span in layout.spans)
            changes = {}
            if keyed:
                changes["prefix_keys"] = {**overrides.prefix_keys, label: keys}
                changes["prefix_tail"] = {
                    **(overrides.prefix_tail or {}), label: list(pages[-1]) if length % page_size else [],
                }
            self._overrides[rid] = replace(
                overrides,
                kv_layout={
                    **overrides.kv_layout,
                    label: replace(layout, spans=(PrefixSpan(length, length, walk),), keyed_spans=int(keyed)),
                },
                loop_caps={**overrides.loop_caps, loop: overrides.loop_caps[loop] - grown // slots},
                **changes,
            )
            stream = self._streams[rid][label]
            stream.forget_chain()
            self._lay_out(rid, label, stream)
            self._seed_keys(rid, label, stream)
            if stream.chain is not None and (overrides.prefix_decode or {}).get(label):
                stream.chain.extend([ids[-1]], page_size)

    def land_rebuild(self, rid: str) -> None:
        self._dropped.pop(rid, None)

    def offload(self, rid: str) -> int:
        """Move every stream of ``rid`` to host memory. Returns the device pages
        it made reusable: pages that reach the free list, and pages then held
        only by the index.

        A stream whose pages don't fit on the host keeps them, so a partial
        offload still frees whatever did fit.

        Device pages go back to the arena only once every stream has been
        copied: a step admitted before the claim can still run its fork copy,
        and that copy reads one of these streams.
        """
        if self._cpu_pool is None or rid not in self._streams:
            return 0
        claimed, read_futures = self._claim_for_offload(rid)
        if not claimed:
            with self._lock:
                # a victim with no pages on the device still gives up its lease and allotment
                if rid in self._hosted and not any(stream.page_indices for stream in self._streams[rid].values()):
                    self._vacate(rid, self._room())
            return 0
        released: set[str] = set()
        try:
            # blocking work OUTSIDE the lock: drain the in-flight reads, then
            # copy each claimed stream to host
            if read_futures:
                wait(read_futures)
            moved = [
                claim for claim in claimed
                if self._offload_stream(
                    rid, claim, [fingerprint(self._prefix_root, key) for key in claim.keys],
                )
            ]
            if not moved:
                return 0
            # sync so the release can't precede the copy
            self._cpu_pool.sync()
            freed, released = self._commit_offload(rid, moved)
            return freed
        finally:
            self._abandon_claims(
                rid, [c.label for c in claimed if c.label not in released]
            )

    def _offload_stream(
        self, rid: str, claim: ClaimedStream, rooted: list[bytes],
    ) -> bool:
        """False when the host can't hold ``claim``."""
        with self._lock:
            keyed = self._filed(rooted, claim.pages)
            copies = self._cpu_pool.place(
                rid=rid, label=claim.label, gpu_page_indices=claim.pages,
                keys=rooted[:keyed],
                stored_len=claim.stored_len, position=claim.position,
                released=claim.released, protected_prefix=claim.protected_prefix,
            )
        if copies is None:
            return False
        self._cpu_pool.copy_out(self.kv_cache.tensor, copies)
        return True

    def _filed(self, rooted: list[bytes], pages: list[int]) -> int:
        """How many of ``pages``, counted from the first, are filed in the index under ``rooted``. A page that
        lost the insert race is a private copy, so the count stops at it."""
        filed = 0
        for key, page in zip(rooted, pages, strict=False):
            if self._index.page_for(key) != page:
                break
            filed += 1
        return filed

    def _claim_for_offload(
        self, rid: str
    ) -> tuple[list[ClaimedStream], list[Future]]:
        """Take ownership of every offloadable stream of ``rid``.

        Claiming all of them under one lock is what makes each ``pages``
        complete: `_alloc` refuses a claimed stream, so nothing can extend one
        behind us while the copies run.
        """
        claimed: list[ClaimedStream] = []
        read_futures: list[Future] = []
        with self._lock:
            streams = self._streams.get(rid, {})
            # refuse the whole request, not the marked streams: a step is
            # already admitted against these pages and the caller has other
            # victims. the eviction retries once the step commits
            if any(stream.step_in_flight for stream in streams.values()):
                return [], []
            # A stream offloaded before `plan` copies its cached image end would reload without it.
            if any(stream is pending for stream in streams.values() for _, pending in self._image_ends):
                return [], []
            for label, stream in streams.items():
                if stream.offloaded or not stream.page_indices:
                    continue
                stream.offloaded = True
                chain = stream.chain
                claimed.append(ClaimedStream(
                    label=label,
                    pages=list(stream.page_indices),
                    generation=stream.generation,
                    stored_len=stream.stored_len,
                    position=stream.position,
                    released=stream.released,
                    protected_prefix=stream.protected_prefix,
                    # past a front release `page_indices[k]` is no longer page k of the chain
                    keys=(
                        chain.keys[:chain.cursor]
                        if self._index is not None and chain is not None and not stream.released
                        else []
                    ),
                ))
                if stream.read_future is not None:
                    read_futures.append(stream.read_future)
        return claimed, read_futures

    def _commit_offload(
        self, rid: str, moved: list[ClaimedStream]
    ) -> tuple[int, set[str]]:
        """Free the device pages of streams whose host copy is good.

        All-or-nothing over the request: a stream mutated while the lock was
        down (a fork copy is the one writer the `_alloc` guard can't catch) may
        have a torn host copy, and that fork's source is one of these streams,
        so the whole request stays on device rather than half of it.
        """
        self._touch_plan_cache(rid)
        with self._lock:
            streams = self._streams.get(rid, {})
            for claim in moved:
                stream = streams.get(claim.label)
                if stream is None or stream.generation != claim.generation:
                    # removed, or written to behind us. Releasing nothing here
                    # leaves every claim for `_abandon_claims` to undo.
                    return 0, set()
            # counted under the lock, since a probe on another thread can retain
            # index pages
            freed = sum(self._reusable(page) for claim in moved for page in claim.pages)
            # a victim still moves when it shares its pages with another victim, or frees only its allotment
            if not freed and rid not in self._hosted:
                return 0, set()
            room = self._room()
            for claim in moved:
                stream = streams[claim.label]
                stream.page_indices = []
                # the offload takes its pages off the allotment too, so every page it frees is room
                self._release(rid, stream, claim.pages, refund=False)
                # Offload changes residency, not logical stream contents.
                stream.reset(content_reset=False)
                if claim.keys:
                    self._cpu_pool.insert_keys(rid, claim.label)
            if rid in self._hosted:
                self._vacate(rid, room)
            else:
                self._recount(rid)
            return freed, {claim.label for claim in moved}

    def _reusable(self, page: int) -> bool:
        """Whether ``page`` reaches the free list, or is left to the index alone,
        once the request releases it."""
        owners = self._arena.num_owners[page]
        return owners == 1 or (owners == 2 and self._index is not None and self._index.holds(page))

    def _abandon_claims(self, rid: str, labels: list[str]) -> None:
        """Undo claims that never became an offload.

        Drops any host copy already made for them — a raise mid-copy would
        otherwise leave one behind, and `reload` would then hand the stream
        fresh pages while it still holds its own.
        """
        with self._lock:
            streams = self._streams.get(rid, {})
            for label in labels:
                self._cpu_pool.discard(rid, label)
                stream = streams.get(label)
                if stream is not None:
                    stream.offloaded = False

    @staticmethod
    def _offloading_message(rid: str, label: str) -> RequestOffloading:
        return RequestOffloading(
            message=(
                f"request {rid!r} stream {label!r} is being offloaded to host "
                "memory; retry once it has been reloaded"
            ),
            label=label,
            request_id=rid,
        )

    def can_reload(self, rid: str) -> bool:
        """Whether `_reload_room` covers the pages ``rid``'s reload takes. A request that asks again after
        its move to the host reloads only once admitted."""
        if self._cpu_pool is None or not self._cpu_pool.is_offloaded(rid):
            return False
        with self._lock:
            if self._decides and rid in self._asks:
                return False
            return self._reload_pages(rid) <= self._reload_room(rid)

    def _reload_room(self, rid: str) -> int:
        """The room ``rid``'s reload may take: its allotment as well when it was allotted for the reload, and
        otherwise the room alone, since the offload took the pages off the allotment and the reload puts them
        back."""
        allotment = self._allotments.get(rid)
        if allotment is not None and allotment.returning is not None:
            return self._allowance(rid)
        return self._room()

    def _reload_pages(self, rid: str) -> int:
        """The pages ``rid``'s reload takes: those the index no longer holds, and those it holds evictable."""
        needed = 0
        evictable = () if self._index is None else self._index.evictable()
        for state in self._cpu_pool.offloaded[rid].values():
            run = self._index.pages_for(state.keys) if self._index is not None else []
            # the reload retains these, so an evictable one takes room as a copied page does
            needed += len(state.cpu_page_indices) - len(run) + sum(page in evictable for page in run)
        return needed

    def _return_pages(self, rid: str) -> int:
        """Pages a request returning from the host may take beyond its reload, 0 for any other: the larger of its
        prompt and its reload plus a page per stream (`_one_more`, as a drop counts a history), less the
        reload."""
        prompt = self._asks.get(rid)
        if prompt is None:
            return 0
        reloaded = grown = 0
        for label, state in self._cpu_pool.offloaded[rid].items():
            pages = len(state.cpu_page_indices)
            reloaded += pages
            grown += self._one_more(pages, self._streams[rid][label].footprint)
        return max(prompt, grown, reloaded) - reloaded

    @staticmethod
    def _one_more(pages: int, footprint: int | None) -> int:
        """``pages`` and one page more, at most ``footprint``."""
        return pages + 1 if footprint is None else min(pages + 1, footprint)

    def reload(self, rid: str) -> bool:
        """Bring every offloaded stream of ``rid`` back on device.

        Keyed pages the index still holds are taken from it, and the rest come
        from the host. False when the device can't fit them right now; nothing
        moves in that case, so the caller can evict further and try again.
        A request that asks again after its move to the host reloads only once admitted.
        """
        self._touch_plan_cache(rid)
        if self._cpu_pool is None or not self._cpu_pool.is_offloaded(rid):
            return False
        with self._lock:
            if self._decides and rid in self._asks:
                return False
            states = dict(self._cpu_pool.offloaded[rid])
            # a reload is a hit, so `lookup` marks the pages recently used
            runs = {
                label: self._index.lookup(state.keys) if self._index is not None else []
                for label, state in states.items()
            }
            for run in runs.values():
                self._arena.retain(run)
            needed = sum(len(state.cpu_page_indices) - len(runs[label]) for label, state in states.items())
            # retaining the runs already took their evictable pages out of the room, so the check runs
            # even when the index holds every page
            fits = not self._decides or needed <= self._reload_room(rid)
            pages = self._grant(rid, ", ".join(states), needed, 0) if fits else None
            if pages is None:
                for run in runs.values():
                    self._arena.release(run)
                return False
            allotment = self._allotments.get(rid)
            copies = PageCopies(gpu=pages, cpu=[])
            for label, state in states.items():
                run = runs[label]
                stream = self._ensure_label(rid, label)
                at = len(copies.cpu)
                stream.page_indices = run + pages[at:at + len(state.cpu_page_indices) - len(run)]
                if allotment is not None and allotment.returning is None:
                    # the offload took them off the allotment
                    allotment.pages += len(stream.page_indices)
                stream.stored_len = state.stored_len
                stream.position = state.position
                stream.released = state.released
                stream.protected_prefix = state.protected_prefix
                copies.cpu.extend(state.cpu_page_indices[len(run):])
            if allotment is not None and allotment.returning is not None:
                allotment.pages, allotment.returning = self._held(rid) + allotment.returning, None
                self._recount(rid)
        # The streams stay marked offloaded while the copies are queued outside
        # the lock, so nothing extends or claims their pages.
        self._cpu_pool.copy_in(self.kv_cache.tensor, copies)
        # sync outside the lock: orders the reload H2D copies before attention
        # reads them, but the pages are already assigned so it touches no
        # shared state
        self._cpu_pool.sync()
        with self._lock:
            for label, state in states.items():
                # released with the copy still queued, since the next write to a host page queues behind it
                self._cpu_pool.arena.release(self._cpu_pool.pop(rid, label).cpu_page_indices)
                self._streams[rid][label].offloaded = False
                run = runs[label]
                if len(run) < len(state.keys):
                    copied = self._streams[rid][label].page_indices[len(run):len(state.keys)]
                    self._index.insert_run(state.keys[len(run):], copied, run[-1] if run else None)
        return True

    def _add_host_copy(self, copy: HostCopy) -> None:
        self._arena.retain(copy.pages)
        self._host_copies.append(copy)
        for key in copy.keys:
            self._arriving.setdefault(key, copy)

    def _land_host_copies(self) -> None:
        """Under the lock: index the pages of each copy from the host that has landed,
        and drop that copy's references."""
        pending, landed = [], []
        for copy in self._host_copies:
            if copy.event is None or not copy.event.query():
                pending.append(copy)
                continue
            # the parent may have left the index while the copy was in flight
            parent = None if copy.parent is None else self._index.page_for(copy.parent)
            linked = 0
            if copy.parent is None or parent is not None:
                linked = self._index.insert_run(copy.keys, copy.pages, parent)
            for key in copy.keys:
                if self._arriving.get(key) is copy:
                    del self._arriving[key]
            self._cpu_pool.arena.release(copy.host_pages)
            self._arena.release(copy.pages)
            if copy.stream.host_copy is copy:
                if copy.stream.chain is not None:
                    copy.stream.chain.from_host += linked
                copy.stream.host_copy = None
                copy.stream.read_pending = copy.stream.read_future is not None
                self._note_looked_up(copy.rid)
                landed.append(copy)
            copy.landed = True
        self._host_copies = pending
        for copy in landed:
            self._release_hold(copy.rid, copy.stream)
            self._recount(copy.rid)

    def reclaimable(self, rid: str) -> int:
        """The device pages ``rid`` holds, or 0 when offloading it would free none
        of them, when it is offloaded already, or when it is admitted but hasn't run.
        Also 0 for a guaranteed request in a pool that decides, since a move to the host
        preempts it too."""
        with self._lock:
            if self._decides and self._streams.get(rid) and self._overrides[rid].guaranteed:
                return 0
            return self._freeable(rid)

    def frees_on_remove(self, rid: str) -> int:
        """`reclaimable` for any class of request, since a teardown frees a guaranteed request's pages too."""
        with self._lock:
            return self._freeable(rid)

    def _freeable(self, rid: str) -> int:
        """Under the lock: the device pages ``rid`` holds, or 0 when releasing them would free none."""
        streams = self._streams.get(rid, {})
        for stream in streams.values():
            if any(self._reusable(page) for page in reversed(stream.page_indices)):
                return sum(len(stream.page_indices) for stream in streams.values())
        return 0

    def get_offload_priority(self, rid: str) -> float:
        """What `reclaimable` counts, the most reclaimable first."""
        return float(self.reclaimable(rid))

    def _own_transfer_info(
        self,
        request_id: str,
        label: str,
        stream: CacheStream,
    ):
        """This cache's transfer descriptor, as `publish` stamps it."""
        return self._transfer.get_kv_transfer_info(
            request_id=request_id,
            label=label,
            page_indices=stream.page_indices,
            seq_len=stream.stored_len,
            reset_generation=stream.reset_generation,
        )

    def publish(
        self,
        request_id: str,
        node_name: str | None = None,
        graph_walk: str | None = None,
        *,
        final: bool = False,
    ):
        with self._lock:
            # remove_request can race finalize on another thread. Resolve the
            # request and build its descriptor in this one critical section.
            streams = self._streams.get(request_id)
            overrides = self._overrides.get(request_id)
            if streams is None or overrides is None:
                return None
            labels = overrides.get_publish_labels(
                node_name, graph_walk, list(streams), final=final,
            )
            if not labels:
                return None
            seq_info = {
                label: KVSequenceInfo(
                    seq_len=stream.stored_len,
                    latest_kv_transfer_info=self._own_transfer_info(
                        request_id=request_id,
                        label=label,
                        stream=stream,
                    ),
                    page_indices=list(stream.page_indices),
                    reset_generation=stream.reset_generation,
                ) for label in labels
                if (stream := streams.get(label)) is not None
            }
            if not seq_info:
                return None
        return PublishedKVInfo.build_for_rank(
            rank=self._rank, world_size=self._world_size, seq_info=seq_info,
        )

    def publish_for_step(
        self,
        request_id: str,
        node_name: str | None,
        graph_walk: str | None,
    ):
        return self.publish(
            request_id, node_name=node_name, graph_walk=graph_walk,
        )

    def publish_snapshot_batch(
        self,
        request_ids: list[str],
        node_name: str | None,
        graph_walk: str | None,
    ) -> list:
        """``publish_snapshot_for_step`` for a batch: one lock acquisition and
        a plain loop, which is what the gpu thread can afford per row."""
        out = []
        append = out.append
        streams_of = self._streams.get
        overrides_of = self._overrides.get
        with self._lock:
            for request_id in request_ids:
                streams = streams_of(request_id)
                overrides = overrides_of(request_id)
                if streams is None or overrides is None:
                    append(None)
                    continue
                labels = overrides.get_publish_labels(
                    node_name, graph_walk, list(streams), final=False,
                )
                if not labels:
                    append(None)
                    continue
                snap = []
                for label in labels:
                    stream = streams.get(label)
                    if stream is not None:
                        snap.append((
                            label, stream.stored_len, len(stream.page_indices),
                            stream.reset_generation,
                        ))
                append(tuple(snap) if snap else None)
        return out

    def publish_snapshot_for_step(
        self,
        request_id: str,
        node_name: str | None,
        graph_walk: str | None,
    ):
        """Per label the length, page count and generation ``publish`` would
        export now. The page list itself is copied only when the publication
        is finished, which in a steady decode loop is never: this ran for every
        request on every step and was a quarter of the gpu thread's host time.
        """
        with self._lock:
            streams = self._streams.get(request_id)
            overrides = self._overrides.get(request_id)
            if streams is None or overrides is None:
                return None
            labels = overrides.get_publish_labels(
                node_name, graph_walk, list(streams), final=False,
            )
            if not labels:
                return None
            snap = tuple(
                (label, stream.stored_len, len(stream.page_indices),
                 stream.reset_generation)
                for label in labels
                if (stream := streams.get(label)) is not None
            )
        return snap or None

    def publish_from_snapshot(
        self,
        request_id: str,
        snapshot,
        node_name: str | None,
        graph_walk: str | None,
    ):
        """``publish`` as of the snapshot: the lengths are the snapshot's, so a
        step that committed in between (the next one is already running when
        the worker finishes a publication) does not leak into it. A commit
        only appends pages, so the snapshot's page count still names the
        pages it saw. A reset or a fork in between replaces the list (the
        stream's ``reset_generation`` moves); the snapshot describes pages
        that no longer exist then, and the stream is published as it stands."""
        del node_name, graph_walk
        if not snapshot:
            return None
        with self._lock:
            streams = self._streams.get(request_id)
            if streams is None:
                return None
            seq_info = {}
            for label, seq_len, num_pages, reset_generation in snapshot:
                stream = streams.get(label)
                if stream is None:
                    continue
                if stream.reset_generation != reset_generation:
                    seq_len = stream.stored_len
                    num_pages = len(stream.page_indices)
                    reset_generation = stream.reset_generation
                pages = list(stream.page_indices[:num_pages])
                seq_info[label] = KVSequenceInfo(
                    seq_len=seq_len,
                    latest_kv_transfer_info=self._transfer.get_kv_transfer_info(
                        request_id=request_id,
                        label=label,
                        page_indices=pages,
                        seq_len=seq_len,
                        reset_generation=reset_generation,
                    ),
                    page_indices=pages,
                    reset_generation=reset_generation,
                )
        if not seq_info:
            return None
        return PublishedKVInfo.build_for_rank(
            rank=self._rank, world_size=self._world_size, seq_info=seq_info,
        )

    def publish_after_stop(
        self,
        request_id: str,
        node_name: str | None,
        graph_walk: str | None,
    ):
        return self.publish(
            request_id,
            node_name=node_name,
            graph_walk=graph_walk,
            final=True,
        )

    def reset_request(self, rid: str, free: bool=False):
        self._touch_plan_cache(rid)
        self.flush_chain_backlog()
        streams = self._streams.get(rid)
        if streams is None:
            return
        # drain in-flight reads OUTSIDE the lock (their pages must not be reused
        # until they finish writing); the transfer thread doesn't touch _streams
        for stream in streams.values():
            if stream.read_future is not None:
                wait([stream.read_future])
        with self._lock:
            for stream in self._streams.get(rid, {}).values():
                assert not stream.offloaded and stream.host_copy is None, (
                    f"KV {self.name}: reset {rid}, whose pages are on their way to or from the host"
                )
                # a rewind would put the next write on the stream's first page,
                # over a sealed one its other owners still read. drop the pages
                # and let the next write allocate; `free` asks for the same
                self._release_lease(rid, stream)
                # here, not in `CacheStream.reset`: an offload resets the stream
                # too, and the probe after its reload still owes the same length
                stream.converted = False
                drop = free or self._arena.any_sealed(stream.page_indices)
                if drop:
                    pages, stream.page_indices = stream.page_indices, []
                    self._release(rid, stream, pages)
                stream.reset(freed=drop)
                # a stale cursor or generated key would misfile what the rerun writes
                stream.forget_chain()
            self._recount(rid)
            # the pages above were released, so any record of them is stale
            self._forget_reservations(rid)
            for label, stream in self._streams.get(rid, {}).items():
                self._seed_keys(rid, label, stream)
            if _DEBUG_ASSERTS:
                self.assert_pages_conserved()

    def remove_request(self, rid: str):
        self._touch_plan_cache(rid)
        with self._lock:
            # its waiting tokens first, before the handle can name another request
            self._drain_chain_backlog(rid)
        streams = self._streams.get(rid)
        if streams is not None:
            # drain in-flight reads outside the lock; see reset_request
            for stream in streams.values():
                if stream.read_future is not None:
                    wait([stream.read_future])
        with self._lock:
            if _DEBUG_ASSERTS and rid in self._streams:
                self._report_underuse(rid)
            room = self._room()
            allotment = self._allotments.pop(rid, None)
            if allotment is not None:
                self._unused -= allotment.remaining
            self._asks.pop(rid, None)
            if rid in self._streams:
                for stream in self._streams[rid].values():
                    self._release_lease(rid, stream)
                    self._release_hold(rid, stream)
                    self._arena.release(stream.page_indices)
            if self._cpu_pool is not None:
                self._cpu_pool.remove_request(rid)
            # handle-keyed, and this releases the handle: a survivor would be
            # unwound against whichever request is given this handle next
            self._forget_reservations(rid)
            self._streams.pop(rid, None)
            self._looked_up.discard(rid)
            self._overrides.pop(rid, None)
            self._dropped.pop(rid, None)
            self._chain_lookup.pop(rid, None)
            self._transfer.remove_request(rid)
            self._settle(rid, self._room() - room)
            earmark = self._earmarks.pop(rid, None)
            if earmark is not None:
                # its victims run on without being dropped
                for victim in earmark.victims:
                    del self._victims[victim]
                    self._hosted.discard(victim)
            self._short.pop(rid, None)
            self._ran_short.pop(rid, None)
            if _DEBUG_ASSERTS:
                self.assert_pages_conserved()

    def _report_underuse(self, rid: str) -> None:
        overrides = self._overrides[rid]
        for label, layout in (overrides.kv_layout or {}).items():
            stream = self._streams[rid].get(label)
            footprint = layout.pages(overrides.loop_caps, self.config.page_size)
            peak = 0 if stream is None else stream.peak
            if footprint > peak:
                logger.warning(
                    "KV %s: request %s held at most %d pages on label %s against a footprint of %d",
                    self.name, rid, peak, label, footprint,
                )

    def assert_pages_conserved(self) -> None:
        """Check the owner counts against the streams holding the pages, on the
        device and on the host. Each assertion names the rule it checks."""
        with self._lock:
            arena = self._arena
            owned = self._owned_pages(arena, "device")

            # the sink belongs to no request, so count it by hand. `frontier`
            # is the page each stream is still writing into, plus any it holds
            # past that
            refs: dict[int, int] = {SINK_PAGE: 1}
            frontier: set[int] = set()
            if self._index is not None:
                for page in self._index.pages():
                    refs[page] = refs.get(page, 0) + 1
            for streams in self._streams.values():
                for stream in streams.values():
                    for page in stream.page_indices:
                        refs[page] = refs.get(page, 0) + 1
                    if stream.lease is not None:
                        # held for a stream `admit` has not converted yet
                        for page in stream.lease:
                            refs[page] = refs.get(page, 0) + 1
                    for page in stream.hold or []:
                        refs[page] = refs.get(page, 0) + 1
                    full = stream.stored_len // self.config.page_size
                    frontier.update(stream.page_indices[full:])
            copying = Counter(page for copy in self._host_copies for page in copy.pages)
            for page, n in copying.items():
                refs[page] = refs.get(page, 0) + n

            counts = {page: arena.num_owners[page] for page in owned}
            assert counts == refs, (
                "owner counts disagree with the streams naming the pages: "
                + ", ".join(
                    f"page {page} owned {counts.get(page, 0)}, "
                    f"named {refs.get(page, 0)}"
                    for page in sorted(set(counts) | set(refs))
                    if counts.get(page, 0) != refs.get(page, 0)
                )
            )

            # a copy from the host holds its pages without reading them, so its references don't make a page shared
            unsealed = [
                page for page in owned
                if arena.num_owners[page] - copying[page] > 1 and not arena.sealed[page]
            ]
            assert not unsealed, f"pages shared before they were sealed: {unsealed}"

            crowded = sorted(
                page for page in frontier if arena.num_owners[page] != 1
            )
            assert not crowded, (
                f"pages still being written into, but not owned alone: {crowded}"
            )

            unused = 0
            for rid, allotment in self._allotments.items():
                held = 0
                for label, stream in self._streams[rid].items():
                    pages = [*stream.page_indices, *(stream.lease or ()), *(stream.hold or ())]
                    stale = stream.lent - set(pages)
                    assert not stale, f"{rid}/{label} counts pages it no longer holds as lent: {sorted(stale)}"
                    held += len(pages) - len(stream.lent)
                unused += max(0, allotment.pages - held)
            assert unused == self._unused, (
                f"the streams leave {unused} allotted pages unused, but the pool counts {self._unused}"
            )
            evictable = 0 if self._index is None else len(self._index.evictable())
            assert self._unused <= arena.num_free + evictable, (
                f"admitted requests were allotted {self._unused} pages they haven't used, "
                f"but only {arena.num_free} are free and {evictable} evictable"
            )
            if self._index is not None:
                self._index.assert_evictable()
            for rid in self._asks:
                on_host = self._on_host(rid)
                if rid in self._dropped or on_host:
                    held = [
                        page for stream in self._streams[rid].values()
                        for page in (*stream.page_indices, *(stream.lease or ()))
                    ]
                    assert not held, f"{rid} waits {'on the host' if on_host else 'dropped'}, holding pages {held}"
            for rid in self._victims:
                assert not self._overrides[rid].guaranteed, f"{rid} is guaranteed and was chosen as a victim"
            if self._cpu_pool is not None:
                self._assert_host_pages_conserved()

    @staticmethod
    def _owned_pages(arena: PageArena, pool: str) -> list[int]:
        """The pages ``arena`` holds, checked against its free list."""
        total = arena.allocator.max_num_pages
        free = list(arena.allocator.free_pages.queue)
        owned = [page for page in range(total) if arena.num_owners[page] > 0]
        both = sorted(set(free) & set(owned))
        assert not both, f"pages both free and owned in the {pool} pool: {both}"
        assert len(free) + len(owned) == total, (
            f"{total} pages in the {pool} pool, but {len(free)} free and {len(owned)} owned"
        )
        return owned

    def _assert_host_pages_conserved(self) -> None:
        pool = self._cpu_pool
        host = pool.arena
        owned = self._owned_pages(host, "host")
        refs = Counter(pool.index.pages())
        for streams in pool.offloaded.values():
            for state in streams.values():
                refs.update(state.cpu_page_indices)
        for copy in self._host_copies:
            refs.update(copy.host_pages)
        if self._host_writes is not None:
            refs.update(self._host_writes.host_pages_held())
        counts = {page: host.num_owners[page] for page in owned}
        assert counts == dict(refs), (
            "host owner counts disagree with the index and requests naming the pages: "
            + ", ".join(
                f"page {page} owned {counts.get(page, 0)}, named {refs.get(page, 0)}"
                for page in sorted(set(counts) | set(refs))
                if counts.get(page, 0) != refs.get(page, 0)
            )
        )

    def post_warmup_validate(self):
        """Assert ``num_free_pages`` is identical across every TP rank

        Catches YAML drift (e.g. ``cpu_offload_pages`` set on one rank
        but not another), allocator-init bugs, and any future code path
        that adds requests asymmetrically before ``warmup`` returns. The
        vote itself is synchronizing, so no extra barrier is needed on the
        success path.
        """
        if self._comm_group.world_size == 1:
            return
        free = self._arena.num_free
        lo, neg_hi = self._comm_group.agree([free, -free])
        if lo != -neg_hi:
            raise RuntimeError(
                f"KV cache {self.name!r} has asymmetric num_free_pages "
                f"across TP ranks: min {lo}, max {-neg_hi}. v1 requires symmetric "
                "allocator state; check the YAML for per-rank-divergent "
                "max_num_pages / cpu_offload_pages, and any model code "
                "that calls add_request before warmup completes."
            )

    def cleanup(self):
        self._transfer.cleanup()

    def _ensure_label(self, rid: str, label: str) -> CacheStream:
            if label not in self._streams[rid]:
                stream = self._streams[rid][label] = CacheStream()
                self._lay_out(rid, label, stream)
                self._seed_keys(rid, label, stream)
            return self._streams[rid][label]

    # Admission

    def ask(self, rid: str, node_name: str, graph_walk: str) -> AdmitOutcome | None:
        """`ASKS` until ``rid`` has an allotment, or why it can never be admitted.

        A request that can't fit even in an empty pool fails here, and not after
        waiting behind every request ahead of it.
        """
        if not self._decides or rid in self._allotments:
            return None
        with self._lock:
            if rid not in self._asks:
                overrides = self._overrides[rid]
                caps = dict.fromkeys(overrides.loop_caps or (), 0)
                if overrides.guaranteed:
                    caps.update(
                        (loop, cap) for loop, cap in (overrides.loop_caps or {}).items() if loop in self.local_loops
                    )
                pages = overrides.pages(self._nodes, caps, self.config.page_size)
                capacity = self.config.request_pages
                if pages > capacity:
                    return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                        f"KV {self.name}: request {rid} needs {pages} pages to be admitted, "
                        f"and the pool has {capacity} for requests"
                    ))
                self._asks[rid] = pages
        return ASKS

    def reserve(
        self, rid: str, node_name: str, graph_walk: str, published: PublishedKVInfo | None,
    ) -> RoomMark | None:
        """Allot ``rid`` its prompt, less the cache hits another request holds, or what it returns with
        from a drop or the host."""
        with self._lock:
            if self._host_copies:
                self._land_host_copies()
            chains = [] if self._index is None or self._returning(rid) else [
                stream.chain for stream in self._streams[rid].values()
                if stream.chain is not None and stream.chain.rooted is None
            ]
        # Requests ask and are removed on one thread only, so ``rid`` still asks once the lock is
        # taken again.
        for prefix_chain in chains:
            self._root_chain(prefix_chain)
        with self._lock:
            pages, room = self._need(rid, node_name, graph_walk, published), self._room()
            if pages > room:
                return RoomMark(rid, node_name, graph_walk, published, room, *self._index_counts())
            returning = self._return_pages(rid) if self._on_host(rid) else None
            self._allotments[rid] = Allotment(pages=pages, prompt=self._asks.pop(rid), returning=returning)
            self._recount(rid)
            return None

    def unreserve(self, rid: str) -> None:
        with self._lock:
            allotment = self._allotments.pop(rid)
            self._unused -= allotment.remaining
            self._asks[rid] = allotment.prompt

    def grew_since(self, mark: RoomMark) -> bool:
        """A newly filed page can raise a waiting request's hit, and a hit page another request
        takes from the evictable set comes out of that request's allotment, not this one's. Neither
        moves the room, so the index counts both. The counts only rise, so two changes between passes
        can't cancel out."""
        if not _DEBUG_ASSERTS:
            return self._grew_since(mark)
        # under the lock, so a page freed on the GPU thread between the two reads can't trip it
        with self._lock:
            grew = self._grew_since(mark)
            if not grew and mark.rid in self._asks:
                pages = self._need(mark.rid, mark.node_name, mark.graph_walk, mark.published)
                assert pages > self._room(), (
                    f"KV {self.name}: {mark.rid} fits now, but nothing `grew_since` reads has moved "
                    "since it didn't, so nothing would admit it"
                )
            return grew

    def _grew_since(self, mark: RoomMark) -> bool:
        filed, taken = self._index_counts()
        return self._room() > mark.room or filed != mark.filed or taken != mark.taken

    def _need(
        self, rid: str, node_name: str, graph_walk: str, published: PublishedKVInfo | None,
    ) -> int:
        """The pages ``rid``'s allotment would take now. Caller holds the lock.

        A dropped request is charged every cache hit as a fresh page, since nothing is leased before its
        rebuild. When the rebuild's probe hits a page another request holds, the hit costs nothing, so
        that page of the allotment is left for the request's growth.
        """
        dropped = self._dropped.get(rid)
        if dropped is not None:
            return dropped.returns
        if self._on_host(rid):
            return self._reload_pages(rid) + self._return_pages(rid)
        if self._overrides[rid].guaranteed:
            # A guaranteed request takes no cache hit off its allotment, since the request holding a hit page can
            # release it and the index evict it before the lease.
            return self._asks[rid]
        whole, charged = self._hit(rid, node_name, graph_walk, published)
        return self._asks[rid] - whole + charged

    def _on_host(self, rid: str) -> bool:
        return self._cpu_pool is not None and self._cpu_pool.is_offloaded(rid)

    def _returning(self, rid: str) -> bool:
        """Whether ``rid`` asks again after a drop or a move to the host."""
        return rid in self._dropped or self._on_host(rid)

    def _index_counts(self) -> tuple[int, int]:
        """The pages the index has ever filed, and the pages requests have ever taken out of its
        evictable set."""
        return (0, 0) if self._index is None else (self._index.filed, self._index.taken)

    # Preemption

    def preempt(self, rid: str, pages_short: int, droppable: Callable[[str], bool]) -> list[str] | None:
        """Choose newer requests to move off the device for ``rid``'s shortage of ``pages_short`` pages, newest
        first, and earmark what they free for ``rid``. A victim the host can take whole moves there
        (`moves_to_host`), and any other is dropped, once ``droppable`` says its launched steps allow it.

        Returns the victims, or ``[rid]`` when only moving ``rid`` itself frees pages, or ``[]`` while ``rid``
        waits for room or for its earlier victims to move. None when the pool doesn't decide or ``rid`` has no
        arrival number.
        """
        if not self._decides:
            return None
        with self._lock:
            overrides = self._overrides.get(rid)
            if overrides is None or overrides.arrival is None:
                return None
            arrival = overrides.arrival
            room, then = self._room(), self._ran_short.pop(rid, None)
            if then is not None:
                pages_short -= max(0, room - then)
            if pages_short <= 0 or rid in self._earmarks or rid in self._victims:
                return []
            newer = sorted(
                (other for other, overrides in self._overrides.items()
                 if overrides.arrival is not None and overrides.arrival > arrival
                 and self._holds(other)),
                key=lambda other: self._overrides[other].arrival, reverse=True,
            )
            # a victim not yet moved frees its pages for the request it was chosen for
            moving = any(other in self._victims for other in newer)
            newer = [other for other in newer if other not in self._victims]
            on_host = 0 if self._cpu_pool is None else self._cpu_pool.room()
            left, hosted = on_host, set()
            victims: list[str] = []
            for other in newer:
                pages = self._host_pages(other)
                if pages is not None and pages <= left:
                    left -= pages
                    hosted.add(other)
                elif not droppable(other):
                    continue
                victims.append(other)
                if self._frees(victims) >= pages_short:
                    break
            else:
                pages = self._host_pages(rid)
                to_host = pages is not None and pages <= on_host
                if not moving and (to_host or droppable(rid)) and self._frees([rid]) > 0:
                    self._victims[rid] = None
                    if to_host:
                        self._hosted.add(rid)
                    return [rid]
                self._short[rid] = room
                return []
            freed = self._frees(victims)
            for victim in list(victims):
                rest = [other for other in victims if other != victim]
                if self._frees(rest) == freed:
                    victims = rest
            earmark = Earmark(rid, pages_short, set(victims))
            for victim in victims:
                self._victims[victim] = earmark
            self._hosted.update(hosted.intersection(victims))
            self._earmarks[rid] = earmark
            return victims

    def is_victim(self, rid: str) -> bool:
        return rid in self._victims

    def moves_to_host(self, rid: str) -> bool:
        return rid in self._hosted

    def arrival(self, rid: str) -> int | None:
        overrides = self._overrides.get(rid)
        return None if overrides is None else overrides.arrival

    def waits(self, rid: str) -> bool:
        """Whether ``rid`` waits for room: no move could cover its shortage, and the room hasn't grown since."""
        with self._lock:
            room = self._short.get(rid)
            if room is None:
                return False
            if self._room() <= room:
                return True
            del self._short[rid]
            return False

    def _holds(self, rid: str) -> bool:
        allotment = self._allotments.get(rid)
        return (allotment is not None and allotment.remaining > 0) or any(
            stream.page_indices or stream.lease for stream in self._streams[rid].values()
        )

    def _host_pages(self, rid: str) -> int | None:
        """The host pages that moving every stream of ``rid`` takes, or None when ``rid`` can't move there: the
        pool has no host memory, ``rid`` is guaranteed or isn't resident, or a stream holds a cached image end `plan`
        hasn't copied."""
        if self._cpu_pool is None or self._overrides[rid].guaranteed or not self.is_resident(rid):
            return None
        streams = self._streams[rid].values()
        if any(stream is pending for stream in streams for _, pending in self._image_ends):
            return None
        pages = 0
        for stream in streams:
            chain = stream.chain
            rooted = [] if self._index is None or chain is None or chain.rooted is None or stream.released else (
                chain.rooted[:chain.cursor]
            )
            pages += self._cpu_pool.takes(len(stream.page_indices), rooted[:self._filed(rooted, stream.page_indices)])
        return pages

    def _frees(self, rids: Collection[str]) -> int:
        """The room moving ``rids`` off the device would make: the pages of their streams and leases that would
        reach the free list or the evictable set, and their unused allotments."""
        released: Counter = Counter()
        for rid in rids:
            for stream in self._streams[rid].values():
                released.update(stream.page_indices)
                released.update(stream.lease or ())
        owners = self._arena.num_owners
        pages = sum(1 for page, n in released.items() if owners[page] == n)
        if self._index is not None:
            pages += self._index.evictable_after(released)
        return pages + sum(self._allotments[rid].remaining for rid in rids if rid in self._allotments)

    def _settle(self, rid: str, freed: int) -> None:
        """Take ``rid`` off the earmark it is a victim of, now that its move or removal made ``freed`` room.
        Once no victim is left, the allotment of the request that ran short grows by what the victims
        freed, up to its shortage."""
        self._hosted.discard(rid)
        earmark = self._victims.pop(rid, None)
        if earmark is None:
            return
        earmark.freed += freed
        earmark.victims.discard(rid)
        if not earmark.victims:
            del self._earmarks[earmark.rid]
            self._allot(earmark.rid, min(earmark.pages, earmark.freed))

    def _vacate(self, rid: str, room: int) -> None:
        """Release ``rid``'s leases and give up its allotment, so that ``rid`` asks again, once a drop or a move
        to the host has freed its pages. ``room`` is `_room` as it was before the drop or the move."""
        for stream in self._streams[rid].values():
            self._release_lease(rid, stream)
        allotment = self._allotments.pop(rid, None)
        if allotment is not None:
            self._unused -= allotment.remaining
            self._asks[rid] = allotment.prompt
        self._short.pop(rid, None)
        self._ran_short.pop(rid, None)
        self._settle(rid, self._room() - room)

    def _allot(self, rid: str, pages: int) -> None:
        # A request past its allotment holds more than ``allotment.pages``, so ``pages`` add to what it holds.
        allotment = self._allotments[rid]
        allotment.pages = max(allotment.pages, self._held(rid)) + pages
        self._recount(rid)

    def _hit(
        self, rid: str, node_name: str, graph_walk: str, published: PublishedKVInfo | None,
    ) -> tuple[int, int]:
        """How many whole pages ``rid``'s streams will take from the index, and how many of the
        pages they take are evictable. A stream to be read in takes what `_take_local_match`
        would match, and a stream to be prefilled what the probe would.

        The engine doesn't probe a walk that writes two labels from one input, since a probe can cut
        only one label's input.
        """
        overrides = self._overrides[rid]
        if self._index is None or not overrides.prefix_cache:
            return 0, 0
        page_size = self.config.page_size
        reading = {} if published is None else published.get(self._rank)
        needed = overrides.get_labels(node_name, graph_walk)
        evictable = self._index.evictable()
        whole = charged = 0
        for label, stream in self._streams[rid].items():
            chain, cursor = stream.chain, stream.span_cursor
            if chain is None or stream.stored_len or stream.offloaded or stream.lease is not None:
                continue
            if label in reading and label in needed:
                pages = self._index.pages_for(chain.rooted)[:reading[label].seq_len // page_size]
                whole += len(pages)
            elif (
                cursor.done < len(chain.spans) and not stream.read_pending
                and overrides.get_labels(node_name, chain.spans[cursor.done].walk) == [label]
                and self._keyed_label(rid, node_name, chain.spans[cursor.done].walk) == label
            ):
                pages = self._index.pages_for(chain.rooted)[:len(chain.rooted) - 1]
                whole += len(pages)
                total = sum(span.length for span in chain.spans)
                image_end = self._image_end(chain.rooted_ends, len(pages) * page_size, total)
                if image_end is not None:
                    pages = pages + self._index.pages_for([image_end[1]])
            else:
                continue
            charged += sum(1 for page in pages if page in evictable)
        return whole, charged

    def _room(self) -> int:
        evictable = 0 if self._index is None else len(self._index.evictable())
        return self._arena.num_free + evictable - self._unused

    def _allowance(self, rid: str) -> int:
        """Pages ``rid`` can take from free and evictable pages: what its allotment has left,
        then the room. Until `_recount` runs, a page taken lowers the room instead of
        ``remaining``, so the allowance is the same before and after `_recount`."""
        allotment = self._allotments.get(rid)
        return (0 if allotment is None else allotment.remaining) + self._room()

    def _retain_cached(self, rid: str, stream: CacheStream, pages: list[int], counted: bool = True) -> list[int]:
        """Retain the longest prefix of ``pages`` that ``rid`` can take, and return it.

        A page that isn't evictable is lent and takes no room. An evictable one comes out
        of what ``rid`` was allotted and hasn't used, then out of the room, and the walk
        recomputes the pages neither covers.

        ``counted`` is False for pages ``rid``'s allotment doesn't count, such as an image end. A
        guaranteed request is charged for a page its allotment counts, even one another request holds,
        and its allotment grows by each evictable page it doesn't count.
        """
        if self._decides:
            evictable = self._index.evictable()
            guaranteed = self._overrides[rid].guaranteed
            borrow = guaranteed and not counted
            allowance = self._room() if borrow else self._allowance(rid)
            for at, page in enumerate(pages):
                if page not in evictable and not (guaranteed and counted):
                    stream.lent.add(page)
                elif allowance > 0:
                    allowance -= 1
                    if borrow:
                        self._allotments[rid].pages += 1
                else:
                    pages = pages[:at]
                    break
        self._arena.retain(pages)
        return pages

    def _release(self, rid: str, stream: CacheStream, pages: list[int], refund: bool = True) -> None:
        """Release ``pages``, which the caller has already taken off ``stream``.

        A page not in ``stream.lent`` shrinks the allotment by one when its release leaves
        the page neither free nor evictable, and always when ``refund`` is False. Otherwise
        the allotment keeps the page for the walk that writes it again. The caller recounts.
        """
        self._arena.release(pages)
        allotment = self._allotments.get(rid)
        if allotment is not None:
            evictable = () if self._index is None else self._index.evictable()
            owners = self._arena.num_owners
            gone = sum(
                1 for page in pages
                if page not in stream.lent and (not refund or (owners[page] and page not in evictable))
            )
            allotment.pages = max(0, allotment.pages - gone)
        if stream.lent:
            stream.lent.difference_update(pages)

    def _recount(self, rid: str) -> None:
        """Bring ``rid``'s share of `_unused` up to date after its pages or its allotment change.

        A stale count would let a waiting request be admitted into room an admitted request still needs.
        """
        allotment = self._allotments.get(rid)
        if allotment is None:
            return
        remaining = max(0, allotment.pages - self._held(rid))
        self._unused += remaining - allotment.remaining
        allotment.remaining = remaining
        if _DEBUG_ASSERTS:
            self.assert_pages_conserved()

    def _held(self, rid: str) -> int:
        return sum(
            len(stream.page_indices) + len(stream.lease or ()) + len(stream.hold or ()) - len(stream.lent)
            for stream in self._streams.get(rid, {}).values()
        )

    def _check_ready(
        self, rid: str, label: str
    ) -> tuple[bool, AdmitRuntimeError | None]:
        """(ready, terminal failure). A failed retrieve is latched on the
        stream: the future can only be read once, but every later check has to
        keep reporting the stream as unusable."""
        if label not in self._streams[rid]:
            return True, None
        stream = self._streams[rid][label]
        if stream.read_future is not None and stream.read_future.done():
            future, stream.read_future = stream.read_future, None
            try:
                future.result()
            except Exception as e:
                stream.read_error = e
                stream.read_error_text, _ = client_error_text(
                    e, f"KV retrieve for request {rid} label {label!r} failed", log=logger,
                )
            else:
                stream.read_pending = False
        if stream.host_copy is not None and stream.host_copy.landed:
            stream.host_copy = None
            stream.read_pending = stream.read_future is not None
            self._note_looked_up(rid)
        if stream.read_error is not None:
            return False, AdmitRuntimeError(
                f"KV retrieve for request {rid} label {label!r} failed: {stream.read_error_text}"
            )
        return not stream.read_pending, None

    @staticmethod
    def _label_growth(step: KVStep) -> dict[tuple[str, str], int]:
        """(rid, label) -> what this step's commit adds to that stream."""
        growth: dict[tuple[str, str], int] = {}
        for segment in step.segments:
            key = (segment.request_id, segment.label)
            growth[key] = growth.get(key, 0) + segment.span
        return growth

    def _reserve_fork(
        self, rid: str, from_label: str,
        to_label: str, extra: int = 0, realloc: bool = False
    ) -> AllocResult:
        """Pages for a fork target, without moving anything into them.

        ``extra`` is how much the source still grows before the copy runs.
        """
        # TODO: handle realloc
        if from_label not in self._streams[rid]:
            if extra <= 0:
                # nothing to fork from and nothing this step adds; also the
                # shape a padded request produces during capture
                return AllocResult()
            # the source does not exist *yet*: this step's own segments create
            # it, and `extra` is what they will put in it. reserving nothing
            # here left `_apply_fork` copying onto an unbacked target
            self._ensure_label(rid, to_label)
            return self._alloc(rid, to_label, extra)
        from_stream = self._streams[rid][from_label]
        if from_stream.offloaded:
            # the target is a fresh stream an offload never claimed, so refuse
            # here: `_apply_fork` would copy from a source whose pages are gone
            return AllocResult(success=False, error=self._offloading_message(rid, from_label))
        self._ensure_label(rid, to_label)
        return self._alloc(
            rid, to_label, from_stream.stored_len + extra
        )

    def _apply_fork(
        self, rid: str, from_label: str, to_label: str,
    ) -> None:
        """Copy a stream onto its fork target, over pages `_reserve_fork` took.

        Locked (reentrant): called from plan (pre-forks, else unguarded) and
        from the already-locked commit (post-forks).
        """
        self._touch_plan_cache(rid)
        with self._lock:
            if from_label not in self._streams[rid]:
                return
            from_stream = self._streams[rid][from_label]
            to_stream = self._ensure_label(rid, to_label)
            # sized off the source's length, not either side's page count:
            # both can hold more pages than the fork needs, and a target left
            # over-reserved by a refused admit used to make the copy lopsided
            n = -(-from_stream.stored_len // self.config.page_size)
            assert len(to_stream.page_indices) >= n, (
                f"fork target {rid}/{to_label} holds "
                f"{len(to_stream.page_indices)} pages but its source "
                f"{from_label} needs {n}; _reserve_fork under-reserved"
            )
            shared = [
                page for page in to_stream.page_indices[:n]
                if self._arena.num_owners[page] > 1
            ]
            assert not shared, (
                f"fork target {rid}/{to_label} would be written over pages "
                f"{shared}, which another owner still reads; a fork target "
                "cannot be a stream the index holds"
            )
            self._arena.copy_pages(
                from_stream.page_indices[:n], to_stream.page_indices[:n],
            )
            to_stream.stored_len = from_stream.stored_len
            to_stream.generation += 1
            to_stream.reset_generation += 1

    def _alloc(
        self, request_id: str, label: str, seq_len: int
    ) -> AllocResult:
        with self._lock:
            self._ensure_label(request_id, label)
            stream = self._streams[request_id][label]
            if stream.offloaded:
                # an offload claimed this stream: its pages are on their way to
                # the host, and `reload` is the only path that may re-take them
                return AllocResult(
                    success=False, error=self._offloading_message(request_id, label)
                )
            num_pages_needed = (seq_len + self.config.page_size - 1) // self.config.page_size
            num_new_pages = num_pages_needed - len(stream.page_indices)
            # a stream that already holds enough pages acquires none, and has
            # nothing for ``rollback_admit`` to give back
            new_pages: list[int] = []
            if num_new_pages > 0:
                allotment = self._allotments.get(request_id)
                within = 0 if allotment is None else min(num_new_pages, allotment.remaining)
                new_pages = self._grant(request_id, label, num_new_pages, num_new_pages - within)
                if new_pages is None:
                    available = self._allowance(request_id)
                    pages_short = num_new_pages - available
                    if self._decides:
                        self._ran_short[request_id] = self._room()
                    return AllocResult(
                        success=False,
                        error=AllocationFailed(
                            pages_short=pages_short,
                            request_id=request_id,
                            label=label,
                            message=(
                                f"Not enough free pages: requested {num_new_pages}, "
                                f"available {available} for request {request_id}, "
                                f"label {label}."
                            ),
                        )
                    )
                stream.page_indices.extend(new_pages)
                if within:
                    # what `_recount` would count, without its pass over the streams
                    allotment.remaining -= within
                    self._unused -= within
                stream.generation += 1
                if stream.footprint is not None and len(stream.page_indices) > stream.footprint:
                    self._report_overrun(request_id, label, stream)
                if _DEBUG_ASSERTS:
                    stream.peak = max(stream.peak, len(stream.page_indices))
                    self.assert_pages_conserved()
        return AllocResult(new_pages=new_pages)

    def _grant(self, rid: str, label: str, n: int, past: int) -> list[int] | None:
        """Take ``n`` free pages for ``rid``, of which ``past`` go beyond what its allotment has
        left, evicting cached pages if too few are free. None when the pool is short, which on a deciding
        pool means ``past`` is more than the room. The pages within the allotment need
        no check, since the allotments never count more than the pool has."""
        if past and self._decides and past > self._room():
            return None
        pages = self._arena.acquire(n)
        if pages is None and self._index is not None:
            # cached pages are the only ones that can be given back
            # without failing a request that is already running
            self._evict(n - self._arena.num_free)
            pages = self._arena.acquire(n)
        if pages is None and self._decides:
            raise RuntimeError(
                f"KV {self.name}: {rid}/{label} could not take {n} pages that its allotment and the room "
                f"covered, so the allotments count pages the pool can't reach: {self._arena.num_free} free, "
                f"{len(self._index.evictable()) if self._index else 0} evictable, {self._unused} allotted but unused"
            )
        return pages

    def _assert_scratch(self, step: KVStep, ctx: StepContext) -> None:
        for segment in step.segments:
            layout = self._held_layout(segment.request_id, segment.label)
            if layout is None or self._on_sink(ctx, segment.request_id) or (
                (segment.request_id, segment.label) in step.sink
            ):
                continue
            scratch = max([layout.scratch, *(span.scratch for span in layout.spans)])
            assert segment.span <= scratch, (
                f"KV {self.name}: {segment.request_id}/{segment.label}: a step that commits nothing "
                f"writes {segment.span} slots, past its layout's scratch of {scratch}"
            )

    def _report_overrun(self, rid: str, label: str, stream: CacheStream) -> None:
        """Log once per label that a request took more pages than its layout allows.

        The pool would then promise room it doesn't have, and the request's
        layout is the likely cause, so the line prints it. A label the request
        doesn't open is reported on its own, so the pages a batch writes there
        for every row don't hide a real overrun.
        """
        overrides = self._overrides[rid]
        reported = (label, label in overrides.kv_layout)
        if reported in self._overran:
            return
        self._overran.add(reported)
        logger.error(
            "KV %s: request %s took %d pages on label %s against a footprint of %d: layout %s, loop caps %s",
            self.name, rid, len(stream.page_indices), label, stream.footprint,
            overrides.kv_layout.get(label), overrides.loop_caps,
        )

    ### Submodule-level functionality
    # Label / layer cursors come from `AttentionResource`; the readers resolve
    # them. TODO: rename the `set_layer_idx` call sites and drop this alias.
    set_layer_idx = AttentionResource.set_default_layer_idx

    def reset_default_cursors(self) -> None:
        super().reset_default_cursors()
        # unlike attention, every read here needs a usable index
        self._default_layer_idx = 0

    @torch.compiler.disable
    def write_target(self, layer_idx: int = None, label: str = None):
        """Where this step's K/V go, for a layer that writes them itself in a
        fused kernel instead of calling ``write_kv``: ``(layer cache, page of
        each token, slot in page of each token, tokens to write)``. NHD only."""
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        if label is None:
            label = self._default_label
        if self.kv_cache.layout != KVLayout.NHD:
            raise NotImplementedError(f"write_target is not implemented for layout {self.kv_cache.layout}")
        state = self._current_plan_states[label]
        if state.ready is not None:
            torch.cuda.current_stream().wait_event(state.ready)
        return (
            self.kv_cache.tensor[layer_idx], state.token_to_page, state.token_to_cache,
            int(state.total_tokens),
        )

    @torch.compiler.disable
    def layer_view(self, layer_idx: int=None) -> torch.Tensor:
        """layer pages as needed by attention kernel

        handed to `AttentionManager::run`. in `kv_manager` so storage mechanics
        are opaque to layers"""
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        return self.kv_cache.layer_view(layer_idx)

    @torch.compiler.disable
    def read_kv(self, layer_idx: int=None, plan_label: str=None) -> torch.Tensor:
        """
        The slots this step's plan writes, e.g. for NHD:
        [num_tokens, 2, num_kv_heads, head_dim] (K at index 0, V at 1).
        """
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        if plan_label is None:
            plan_label = self._default_label
        plan_state = self._current_plan_states[plan_label]
        n = plan_state.total_tokens
        return self.kv_cache.read_tokens(
            layer_idx=layer_idx,
            page_idx=plan_state.token_to_page[:n],
            cache_idx=plan_state.token_to_cache[:n],
        )

    @torch.compiler.disable
    def write_slots(self, label: str=None) -> tuple[torch.Tensor, torch.Tensor]:
        """The (page, offset-in-page) slots ``write_kv`` scatters this step's tokens to, as
        device tensors, for a kernel that writes the cache itself. Under a capture lease they
        are the slot's static buffers, so a graph replays with the next step's slots."""
        if label is None:
            label = self._default_label
        plan_state = self._current_plan_states[label]
        n = plan_state.total_tokens
        return plan_state.token_to_page[:n], plan_state.token_to_cache[:n]

    @torch.compiler.disable
    def write_kv(
        self, k: torch.Tensor, v: torch.Tensor | None = None,
        layer_idx: int=None, label: str=None, return_tensor: bool = False,
    ) -> torch.Tensor | None:
        """Write K, V into this step's planned slots (under MLA: the per-token
        latent as ``k`` and ``v=None``).

        Returns nothing by default: reading the slots back is a gather no
        caller wants today, and skipping it keeps the write a pure mutation.
        """
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        if label is None:
            label = self._default_label
        plan_state = self._current_plan_states[label]
        if plan_state.ready is not None:
            torch.cuda.current_stream(k.device).wait_event(plan_state.ready)
        # an eager write may carry only the real rows; the padding rows' tokens
        # come last in the plan (on SINK_PAGE) and are skipped
        n = min(plan_state.total_tokens, k.shape[0])
        return self.kv_cache.write_tokens(
            layer_idx=layer_idx,
            k=k[:n], v=None if v is None else v[:n],
            page_idx=plan_state.token_to_page[:n],
            cache_idx=plan_state.token_to_cache[:n],
            return_tensor=return_tensor,
        )
