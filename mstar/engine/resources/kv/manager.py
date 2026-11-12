import logging
import os
import threading
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import (
    AttentionResource,
    CachedLease,
    CachedPrefix,
    CGSlotSpec,
    EngineResourceInfo,
    KeyedWalks,
    MoveKind,
    PublishedInfo,
    Rebuild,
    Residency,
    StepAllowance,
)
from mstar.engine.resources.kv.arena import PageArena, StreamPages, columns_from_raw, new_page_arena, plan_columns
from mstar.engine.resources.kv.cache import GroupPages, KVCache, held_pages, on_pages
from mstar.engine.resources.kv.config import (
    KVGroup,
    KVLayout,
    KVReqConfig,
    KVSpec,
    KVStep,
    PagedKVConfig,
    PrefixSpan,
    RetentionPolicy,
    StreamLayout,
    step_end,
)
from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool, PageCopies
from mstar.engine.resources.kv.host_writes import HostWrites
from mstar.engine.resources.kv.keys import PageItem, fingerprint, page_items, page_key
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
    SettingValue,
    StepContext,
)
from mstar.utils.client_errors import client_error_text
from mstar.utils.h2d import PinnedStager

try:  # rust/src/kv_plan.rs; an older build has none
    from mstar_rust import kv_packed_addressing as _rust_packed_addressing
except ImportError:
    _rust_packed_addressing = None
from mstar.utils.knobs import kv_chain_lazy_steps, kv_pinned_indptrs
from mstar.utils.profiler import PhaseLaps

if TYPE_CHECKING:
    from mstar.engine.resources.context_parallel import ContextParallel
    from mstar.engine.resources.kv.resident import KVResidentTables
    from mstar.engine.resources.recurrent.pool import RecurrentStatePool
    from mstar.engine.resources.resident import (
        CapturedResidentStep,
        ResidentPlan,
        ResidentPlanner,
        ResidentWrites,
    )

logger = logging.getLogger(__name__)

# Off by default: `assert_pages_conserved` walks every stream of every live
# request after each admit, commit, reset and remove, and after each change to
# an admitted request's pages.
_DEBUG_ASSERTS = os.environ.get("MSTAR_KV_DEBUG_ASSERTS", "0") == "1"


class RoomMark(NamedTuple):
    """A request `reserve` couldn't fit, and what it was measured against: the room, and the pages the
    index had filed and requests had taken out of the evictable set, or None for both when no page the index
    files can lower what the request needs."""
    rid: str
    node_name: str
    graph_walk: str
    published: "PublishedKVInfo | None"
    room: int
    filed: int | None
    taken: int | None


class PassMark(NamedTuple):
    """What `reserve_past` refused a request against: the room less `KVManager._kept`, and the index counts
    when a page the index files could still make the request fit."""
    room: int
    filed: int | None
    taken: int | None


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
    # the parent of the first page's key (`KVReqConfig.prefix_root`)
    root: bytes = b""
    # set once this stream has been reported, so a request that is admitted
    # again (a refused admit, a second partition) is still one line
    reported: bool = False
    from_host: int = 0

    @classmethod
    def seed(
        cls, keys: list[bytes], tail: list[int], spans: list[PrefixSpan], page_size: int,
        root: bytes = b"",
    ) -> "PrefixChain":
        total = sum(span.length for span in spans)
        whole = total // page_size
        items = page_items(((span.length, span.digest) for span in spans), page_size)
        return cls(
            keys=list(keys), unkeyed=tail, keyed_pages=whole, covered_len=total,
            spans=list(spans), unkeyed_items=items.get(whole, []), root=root,
        )

    def extend(self, tokens: list[int], page_size: int) -> None:
        self.unkeyed.extend(tokens)
        self.covered_len += len(tokens)
        # Every item on the prompt's partial page ends there, so it takes length - offset of the page's tokens.
        room = page_size - sum(item.length - item.offset for item in self.unkeyed_items)
        while len(self.unkeyed) >= room:
            whole = self.keyed_pages
            key = page_key(
                self.keys[whole - 1] if whole else self.root,
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

    def next_step(self, walk: str) -> tuple[PrefixSpan, ...]:
        """The spans from ``done`` on that one step of ``walk`` writes (`step_end`), none when the next span
        is another walk's. A span that opens no step is of the walk of the span before it, which
        ``LabelLayout.row`` checks, so a step never ends early at a change of walk."""
        if self.done == len(self.spans) or self.spans[self.done].walk != walk:
            return ()
        return self.spans[self.done:step_end(self.spans, self.done)]

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


@dataclass(slots=True)
class CacheStream:
    """(request, label) cache stream metadata. Slotted: the arena's passes
    over a step's rows (`PageArena.admit_rows` and the like) read and move
    ``stored_len``, ``generation``, ``step_in_flight`` and ``offloaded``."""
    # its list in the manager's `PageArena` (no default: a plan reads it from
    # there by id), read like a list; replace it with `.assign`, never by
    # assigning a list
    page_indices: StreamPages
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
    # Its pages in each group of the pool past the first (`PagedKVConfig.groups`), the same
    # tokens in that group's layers and pages (`KVManager._new_stream`). It holds the first
    # group's itself, so that a pool of one group pays nothing for groups.
    later: "tuple[GroupPages, ...]" = field(default=(), repr=False, compare=False)

    @property
    def groups(self) -> "tuple[CacheStream | GroupPages, ...]":
        """Its pages in each group: itself, then `later`. Not a field, which would make every
        stream a reference cycle."""
        return (self, *self.later)

    def group(self, g: int) -> "CacheStream | GroupPages":
        """Its pages in group ``g``, without building `groups`, for the paths run per row."""
        return self.later[g - 1] if g else self

    def pages(self) -> list[int]:
        """Every page it holds, in each group."""
        return held_pages(self.groups) if self.later else self.page_indices

    def restore(self, groups: "tuple[GroupPages, ...]") -> None:
        """Take back each group's pages and lengths from a copy (`on_pages`), as a reload does."""
        for part, back in zip(self.groups, groups, strict=True):
            part.page_indices.assign(back.page_indices)
            part.stored_len, part.released = back.stored_len, back.released

    def free_lists(self) -> None:
        """Drop its page list in each group from the arena, once its pages are released."""
        for part in self.groups:
            part.page_indices.free()

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
        for part in self.later:
            part.stored_len = part.released = 0
            part.generation += 1
            if freed:
                part.page_indices.clear()

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


class Reservation(NamedTuple):
    """The pages an admit acquired for one stream's group in one grant, and the arena's trail before that admit,
    which `KVManager.rollback_admit` puts back."""
    rid: int
    label: str
    group: int
    pages: list[int]
    trail: tuple[int, int] | None


@dataclass(slots=True)
class Allotment:
    """Pages an admitted request may take from free and evictable pages: the pages it asked for, less the
    cache hits another request holds, and less what a window doesn't hold once its prompt is written. A
    guaranteed request's hits come off when it adopts its lease. ``remaining`` is its share of `KVManager._unused`, and
    ``asks_again`` is the least the request asks for once it gives the allotment back, which `unreserve` puts back in
    `_asks`: the pages it asked for, or what its grant needed once it has waited for a shortage. An allotment for a
    request on the host covers its reload, and ``returning`` is set while the request returns from the host: the
    pages it may take beyond its reload. ``passed`` is set for a request admitted past a head that doesn't fit,
    and its allotment then covers its growth too. ``kept`` is the count of `KVManager._kept` the last reservation
    took, ``grown`` is the shortage `reserve` last granted an allotment its request waited holding, and
    ``grown_from`` the allotment's pages before. `unreserve` puts back the first and the last."""
    pages: int
    remaining: int = 0
    asks_again: int = 0
    returning: int | None = None
    passed: bool = False
    kept: int = 0
    grown: int = 0
    grown_from: int = 0


class Dropped(NamedTuple):
    """A dropped request until its rebuild lands: the pages it asks to return with, and the tokens its
    laid-out label held, which its rebuild must write again."""
    returns: int
    held: int


class Victims(NamedTuple):
    """The requests `KVManager.preempt` chose to move off the device: to the host, dropped, or, holding no page of
    their own, already given back their allotment to ask again. When no move covers the shortage, ``to_host``
    holds the request that ran short if the host takes it whole, and otherwise ``waits`` is the pages it asks
    the pool for instead. ``waits`` is 0 otherwise."""
    to_host: list[str]
    to_drop: list[str]
    to_ask: list[str]
    waits: int = 0


class StalledMoves(NamedTuple):
    """The requests that wait holding pages which a move can take now, the host's room, which decides how many
    of them fit there, and the room moving them all would make, which grows when another request lets go of a page
    they share."""
    rids: frozenset[str]
    host_room: int
    frees: int


class ShortGrant(NamedTuple):
    """A grant that ran short: the room then, and the pages it needed past what its request's allotment had left."""
    room: int
    pages: int


@dataclass(slots=True)
class Earmark:
    """The pages the moves of ``victims`` add to ``rid``'s allotment, past the room its grant counted, which went in
    when it ran short. ``granted`` counts the pages added so far. For a request that waits, ``pages`` is all it asks
    for, which it takes whole, ``granted`` counts the pages kept for it until then, and ``mark`` is the refusal
    `make_room` chose the victims for."""
    rid: str
    pages: int
    victims: set[str]
    granted: int = 0
    mark: RoomMark | None = None


@dataclass
class ClaimedStream:
    """A stream an in-progress offload has taken ownership of, and the state
    its host copy was made from."""
    label: str
    # a copy of the stream's pages and lengths in each group (`CacheStream.groups`)
    groups: tuple[GroupPages, ...]
    position: int
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
    # a captured state: both arrays as one buffer (``token_to_page`` then
    # ``token_to_cache``), staged in one copy, and the slot's buffer it is a
    # view of (``KVManager._stage_static``)
    staging: torch.Tensor | None = None
    slot_buffer: "_SlotBuffer | None" = None


class _SlotBuffer:
    """A slot and label's captured write addressing, which every capture
    length's state views: the device buffer, and how many of its leading
    elements are known to be zero. ``SINK_PAGE`` is 0, so a step that writes
    nothing needs its length's ``2 * capture_len`` prefix zero, whatever the
    length; once the whole buffer is, such a step needs no copy at all."""

    __slots__ = ("buffer", "zero_prefix")

    def __init__(self, buffer: torch.Tensor) -> None:
        self.buffer = buffer
        self.zero_prefix = buffer.numel()  # allocated zeroed


class KVManager(AttentionResource):
    prefix_skip_safe = True
    # written through an exchange (`sequence_shard`): the step's every token, or a ring's own ones
    # (`write_kv(tokens=)`)
    supports_split_steps = True
    # the node's sequence shard, which refuses a write outside an exchange (`check_bypass`)
    _sequence_shard = None
    # this pool's resident tables, once a node's planner bound them
    _resident: "KVResidentTables | None" = None
    # Tokens a commit can run ahead of what has been read back and keyed, past
    # the one a step always has: the step in flight's and the one committing's
    # extra replays, 2 * (repeats - 1) for the most this pool has repeated. It
    # only loosens the chain-drop check, so a pool-wide high mark is safe: a gap
    # that never fills still outgrows it and drops the chain, a few tokens later.
    _run_lag: int = 0

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
        host_allowance: StepAllowance | None = None,
    ):
        self.config = cfg
        self._host_allowance = host_allowance
        if len(cfg.groups) > 1 and needs_remote_transfer:
            raise ValueError(
                f"KV {name}: a node on another instance reads this pool (replicas, or walks "
                "placed on other ranks), and this pool keeps only a window of some layers; serve "
                "its nodes from one TP group"
            )
        tp_cache = False
        # the cache is sized off the placement, and off the head counts `build` sharded
        placement = cfg.place(joint_comm_group, name)
        if joint_comm_group is not None:
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
        self.kv_cache = KVCache(cfg, device, dtype, placement)
        self.name = name
        cp = placement
        logger.info(
            "KV %s: %.2f GiB on this rank, %d pages of %d slots%s", name, self.kv_cache.nbytes / 2**30,
            cfg.max_num_pages, self.kv_cache.page_slots,
            "" if cp is None else f" (context parallel, {cp.size} ranks, {type(cp).__name__})",
        )
        self._needs_remote_transfer = needs_remote_transfer

        self._arena = new_page_arena(cfg.num_units)
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
                arena=new_page_arena(cfg.units(cfg.cpu_offload_pages)),
            )
        self._streams: dict[str, LabelToStream] = {}
        self._overrides: dict[str, KVReqConfig] = {}
        # dropped request -> what it returns with, until its rebuild lands
        self._dropped: dict[str, Dropped] = {}
        # rebuilt request -> the tokens it had written when its rebuild landed
        self._rebuilt: dict[str, int] = {}
        # A stream holds its pages in group g at `CacheStream.groups[g]`, and a plan's at
        # `KVPlanOutputs.groups[g]`; write addressing is keyed (label, g).
        self._groups: tuple[KVGroup, ...] = cfg.groups
        # (index, group) of each group past the first, each a window applied at commit
        self._windows = tuple((g, group) for g, group in enumerate(self._groups) if g)
        # layer -> the index of its group
        self._layer_group = [0] * cfg.num_layers
        for g, group in enumerate(self._groups):
            for layer in group.layers:
                self._layer_group[layer] = g
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
        # Every rank of a leading pool keeps the allotments and earmarks and refuses a grant past the room.
        # Rank 0 alone makes the decisions that read state the other ranks lack: it asks, takes allotments and
        # chooses victims, and the other ranks replay its moves. Every rank works out the rest itself, so each rank
        # admits a step from the same counts.
        self._counts = leads
        self._decides = leads and self._rank == 0
        self._trails_counts = _DEBUG_ASSERTS and leads and self._world_size > 1
        # set by the engine on rank 0 of a TP node
        self.journal: Callable[..., None] | None = None
        # An allotment covers the labels `_nodes` open.
        # `_asks` holds the pages each waiting request asks for: what `ask` counted for one with no allotment yet,
        #   which its allotment is sized from, or the shortage one that holds its allotment waits on (`_wait`).
        # `local_loops` holds the loops that grow KV on `_nodes` in a walk this worker runs, set by the worker.
        # `_unused` counts the pages admitted requests were allotted and haven't used yet.
        # `_guaranteed` holds the requests that reserve what they grow into and are never preempted once they start,
        #   decided at their first `ask`, or later when the host has no room to set aside for their move.
        # `_snapshots` holds the pools that restore a hit's recurrent state. A prefill hit counts only as far as they
        #   restore it.
        # `_footprints` holds each request's `_footprint` as it was at intake.
        # `_kept` counts the room kept for `_keeper`, a waiting request that isn't a passer, up to what it asks for:
        #   the pages passers freed while it waited, and the pages moves made for it freed before they covered it.
        #   Passers can't take them, but admitted requests can, and a request admitted in order takes them first.
        #   They go back to the room when `_keeper` leaves.
        # `_made` names the request whose allotment the last reservation made rather than grew.
        # `_passers` holds the requests admitted past the first one that a drop or a move to the host sent back to
        #   ask again. Each returns as it passed, so it takes none of `_kept`.
        self._nodes = nodes
        self.local_loops: Collection[str] = frozenset()
        self._asks: dict[str, int] = {}
        self._allotments: dict[str, Allotment] = {}
        self._unused = 0
        self._guaranteed: set[str] = set()
        self._snapshots: list["RecurrentStatePool"] = []
        self._footprints: dict[str, int] = {}
        self._kept = 0
        self._keeper: str | None = None
        self._made: str | None = None
        self._passers: set[str] = set()
        # Preemption state.
        # `_earmarks`: request -> the earmark it holds until its victims have moved.
        # `_victims`: a victim not yet moved -> its earmark.
        # `_hosted`: the victims that move to the host, decided when they are chosen.
        # `_ran_short`: request -> its last grant that ran short.
        # `_stalled`: the requests that wait holding their allotment, which `make_room` can move.
        self._earmarks: dict[str, Earmark] = {}
        self._victims: dict[str, Earmark] = {}
        self._hosted: set[str] = set()
        self._ran_short: dict[str, ShortGrant] = {}
        self._stalled: set[str] = set()
        if _DEBUG_ASSERTS and tp_cache:
            self._arena.trail = (0, 0)
        self._device = device
        self._lock = threading.RLock()

        # (slot, (label, group), capture_len) -> KVPlanState, a view of its slot's buffer
        self._static_plan_states: dict[tuple[int, tuple[str, int], int], KVPlanState] = {}
        # (slot, (label, group)) -> the buffer its states share
        self._static_plan_buffers: dict[tuple[int, tuple[str, int]], _SlotBuffer] = {}
        # copy-only H2D into those, see `_stage_decode_plan_state`
        self._plan_stager = PinnedStager(torch.long)
        self._cg_max_seq_len = 0
        # the SINK_PAGE run a plan points its sink rows at
        self._sink_pages = self._arena.new_stream()
        # FlashInfer's index arrays from page-locked memory (see PinnedIndexRing)
        self._indptr_ring = (
            PinnedIndexRing() if kv_pinned_indptrs() and torch.cuda.is_available() else None
        )
        # A read-only step (every segment span 0, e.g. a decoder attending an
        # encoder context it wrote at prefill) plans the same as the last one
        # on its slot with the same segments and bucket while nothing touched
        # their streams: slot -> ((segments, bucket), epoch, plan). One entry
        # per replay slot, as `_staged_outputs` keeps, so the slot count bounds
        # the cache; replay slots pad with rows of their own, and an unpadded
        # batch builds its plan once per slot. The epoch moves when a stream's
        # pages or length change other than by a step's commit, and when a
        # stream of a cached step's real row changes (`_readonly_rows`).
        self._readonly_epoch = 0
        self._readonly_plans: dict[int | None, tuple] = {}
        self._readonly_rows: frozenset = frozenset()
        # (slot, label) -> the plan output whose state that slot's static
        # buffers hold, so a reused plan skips re-staging them
        self._staged_outputs: dict[tuple[int, tuple[str, int]], KVPlanOutput] = {}
        self._empty_locations: torch.Tensor | None = None
        # (label, group) -> its write addressing
        self._current_plan_states: dict[tuple[str, int], KVPlanState] = {}
        self.reset_default_cursors()

        self._preplan_states: dict[tuple[str, int], KVPlanState] = {}
        self._preplanned = False
        self._preplan_key = None
        self._cached_plan_output: dict[str, KVPlanOutput] | None = None
        # (resident plan, KV plan) of the last lengths-only step, for its promotion
        self._lengths_plan: tuple[object, KVPlanOutputs] | None = None

        # (rid, label) of each stream a staged step's admit created, which
        # clear_preplan removes once `rollback_admit` has given back its pages.
        # recorded in admit, not plan: forks are not staged.
        self._preplan_new_labels: list[tuple[str, str]] = []

        # The pages the live reservation freshly acquired, in the order it took them, for
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
        self._admit_reserved_pages: list[Reservation] = []

    @classmethod
    def build(cls, spec: KVSpec, info: EngineResourceInfo):
        # before the cache is allocated: it is sized off the head counts
        spec.config.shard(info.head_shards)
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
            host_allowance=info.host_allowance,
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
        # sink rows' pages for any captured length, so a plan rarely grows them
        self._extend_sink_pages(-(-max_seq_len // min(group.page_size for group in self._groups)))

    def _extend_sink_pages(self, need: int) -> None:
        # a plan on the plan thread reads the sink run, so it grows under the lock
        with self._lock:
            sink_pages = self._sink_pages
            if len(sink_pages) < need:
                sink_pages.extend([SINK_PAGE] * (need - len(sink_pages)))

    def _static_plan_state(self, slot: int, label: tuple[str, int], capture_len: int) -> KVPlanState:
        """The captured write addressing of a slot's ``capture_len``-token
        buckets. A slot and label keep one buffer: each length's state is its
        first ``2 * capture_len`` slots, the pages then the offsets, so a step
        stages both in one copy. Only one step uses a slot at a time."""
        state = self._static_plan_states.get((slot, label, capture_len))
        if state is None:
            if capture_len > self._cg_max_seq_len:
                raise ValueError(
                    f"a {capture_len}-token bucket is past the captured buffers' "
                    f"{self._cg_max_seq_len} tokens"
                )
            base = self._static_plan_buffers.get((slot, label))
            if base is None:
                base = self._static_plan_buffers[(slot, label)] = _SlotBuffer(
                    torch.zeros(2 * self._cg_max_seq_len, dtype=torch.long, device=self._device),
                )
            if 2 * capture_len > base.buffer.numel():
                # sized by an earlier, shorter build_accelerator_graph_buffers
                raise ValueError(
                    f"a {capture_len}-token bucket is past slot {slot}'s "
                    f"{base.buffer.numel() // 2}-token buffer"
                )
            # no fill here: only the stager writes the buffer, and every
            # replay is staged first
            both = base.buffer[:2 * capture_len]
            state = self._static_plan_states[(slot, label, capture_len)] = KVPlanState(
                token_to_page=both[:capture_len], token_to_cache=both[capture_len:],
                staging=both, slot_buffer=base,
            )
        return state

    def _stage_static(self, state: KVPlanState, pages, offsets, total: int) -> KVPlanState:
        """Stage a step's write addressing into a captured state: its tokens'
        pages and offsets, the rest SINK_PAGE / 0 (the slots the graph scatters
        beyond the real tokens, else they hit another request's KV). One copy,
        and none for a step that writes nothing into a buffer already zero
        there (a decoder's read of a context label it wrote eagerly)."""
        capture_len = state.token_to_page.shape[0]
        n = len(pages)
        if n > capture_len:
            raise ValueError(f"{n} tokens staged into a {capture_len}-token bucket")
        state.total_tokens = total
        slot_buffer = state.slot_buffer
        if n == 0:
            if slot_buffer.zero_prefix < 2 * capture_len:
                self._plan_stager.copy_(state.staging, (), pad_value=0)
                slot_buffer.zero_prefix = 2 * capture_len
            return state
        host = np.empty(2 * capture_len, np.int64)
        host[:n] = pages
        host[n:capture_len] = SINK_PAGE
        host[capture_len:capture_len + n] = offsets
        host[capture_len + n:] = 0
        self._plan_stager.copy_(state.staging, host)
        slot_buffer.zero_prefix = 0
        return state

    def enable_prefix_cache(self, root: bytes, walks: dict[str, KeyedWalks] | None = None) -> None:
        """Open the index under ``root``, the identity every key hangs from."""
        self._keyed_walks = {}
        for label, named in (walks or {}).items():
            keyed = set(named.writes)
            # Above one rank a growth walk isn't keyed, since each rank reads its sampled ids at a
            # different step.
            if self._world_size == 1:
                keyed.update(named.growth)
            self._keyed_walks[label] = frozenset(keyed)
        self._prefix_root = root
        self._index = PrefixIndex(self._arena)
        # every rank counts the evictable set, since the ranks admit each step from the same room
        self._arena.index = self._index
        if self._cpu_pool is not None:
            self._open_host_writes()

    def _open_host_writes(self) -> None:
        assert self._host_allowance is not None, f"KV {self.name}: copies to the host spend the engine's StepAllowance"
        self._host_writes = HostWrites(
            self._cpu_pool, self._index, self.kv_cache.tensor[:, 0].nbytes, self._host_allowance,
        )
        self._arena.writes = self._host_writes

    def close_prefix_cache(self) -> None:
        self._index = None

    def page_counts(self) -> KVPageCounts:
        """The pool's pages right now, without the manager lock: each count is
        read whole, though the three may straddle an allocation. In pages of every
        layer, as ``max_num_pages`` counts them, whatever the arena's page holds."""
        total = self.config.request_pages
        index = self._index
        return KVPageCounts(
            used=self.config.pages_of(total - self._arena.num_free), total=self.config.pages_of(total),
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
            step = cursor.next_step(graph_walk)
            self._check_positions(step, chain.spans[cursor.done], graph_walk, inputs)
            if sum(span.length for span in step) != inputs.input_seq_len:
                self._drop_layout(rid, label, stream, (
                    f"{graph_walk} writes {inputs.input_seq_len} slots where its "
                    f"layout's span {cursor.done} has {chain.spans[cursor.done].walk} "
                    f"write {sum(span.length for span in step)}"
                ))
                return None
            if stream.converted:
                return self._answer(label, cursor, step, stream.stored_len)
            if stream.lease is not None:
                return self._answer(label, cursor, step, self._leased(rid, stream))
            if stream.stored_len or stream.offloaded or stream.read_pending:
                return None
            if self._world_size > 1:
                # Above one rank, use only the lease taken when rank 0 sent the step. A lookup now
                # would also see this rank's later commits.
                return self._answer(label, cursor, step, 0)
        if not self._lease(rid, node_name, label, stream, chain):
            return None
        with self._lock:
            if stream.lease is None:
                return None
            return self._answer(label, cursor, step, self._leased(rid, stream))

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
                held = len(matched) * self.config.tokens_per_page
                # If an image ends inside the next page, look up its cached partial page.
                image_end = self._image_end(ends, held, total)
                if image_end is not None:
                    end, key = image_end
                    cached = self._retain_cached(rid, stream, self._index.lookup([key]))
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
            if held < end < min(held + self.config.tokens_per_page, total):
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

    def lease_cached_prefix(self, rid: str, node_name: str, graph_walk: str) -> CachedLease | None:
        """Returns the same lease with every step until `admit` converts it."""
        with self._lock:
            label = self._keyed_label(rid, node_name, graph_walk)
            if label is None:
                return None
            stream = self._ensure_label(rid, label)
            chain, cursor = stream.chain, stream.span_cursor
            if stream.lease is not None:
                return self._lease_of(stream)
            if (
                chain is None or cursor.done == len(chain.spans)
                or chain.spans[cursor.done].walk != graph_walk
                or stream.converted or stream.stored_len or stream.offloaded or stream.read_pending
            ):
                return None
        if not self._lease(rid, node_name, label, stream, chain):
            return None
        with self._lock:
            return self._lease_of(stream)

    @staticmethod
    def _lease_of(stream: CacheStream) -> CachedLease | None:
        if stream.lease is None:
            return None
        return CachedLease(stream.lease, [page for page in stream.lease if page in stream.lent])

    def adopt_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, lease: CachedLease, caught_up: bool,
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
                    f"KV {self.name}: rank 0 leased {len(lease.pages)} cached pages for {rid}/{label}, "
                    "whose stream here already holds tokens or has no keys"
                )
        self._root_chain(chain)
        pages = lease.pages
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
            stream.lent.update(lease.lent)
            self._arena.retain(matched)
            stream.lease, stream.lease_len = matched, len(matched) * self.config.tokens_per_page
            self._recount(rid)
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
                self._recount(rid)

    def page_trail(self) -> tuple[int, ...] | None:
        """The arena's trail when it keeps one, then, when it carries the counts, the requests holding an allotment,
        the pages earmarked and the requests dropped, which every rank admits a step from. A step's admit and commit
        move none of the counts, so a follower that reads them at another point of a step in flight than rank 0 reads
        the same."""
        with self._lock:
            if not self._trails_counts:
                return self._arena.trail
            earmarked = sum(earmark.pages for earmark in self._earmarks.values())
            return (*(self._arena.trail or ()), len(self._allotments), earmarked, len(self._dropped))

    def _check_positions(
        self, step: tuple[PrefixSpan, ...], span: PrefixSpan, graph_walk: str, inputs,
    ) -> None:
        # a step of several spans places its own positions and is only moved past whole
        assert len(step) > 1 or span.digest is not None or inputs.custom_pos_ids is None, (
            f"KV {self.name}: {graph_walk} places positions of its own, but "
            "the layout keys it by ids, whose positions are their count"
        )

    def _leased(self, rid: str, stream: CacheStream) -> int:
        """The tokens ``stream``'s lease holds, or for a dropped request at most a page under the tokens its KV
        held: its rebuild writes only those, and runs at least one."""
        dropped = self._dropped.get(rid)
        if dropped is None:
            return stream.lease_len
        page = self.config.tokens_per_page
        return min(stream.lease_len, max(dropped.held - 1, 0) // page * page)

    @staticmethod
    def _answer(
        label: str, cursor: SpanCursor, step: tuple[PrefixSpan, ...], present: int,
    ) -> CachedPrefix:
        """This walk's share of the ``present`` tokens the stream holds, with the
        position past that share. Records the span as the one the probe answered for."""
        length = sum(span.length for span in step)
        tokens = min(max(present - cursor.consumed, 0), length)
        position = sum(earlier.advance for earlier in cursor.spans[:cursor.done])
        if len(step) == 1 and step[0].digest is None:
            position += tokens
        elif tokens == length:
            # A partly held item doesn't move the position. Its walk still runs and advances it.
            position += sum(span.advance for span in step)
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
            keep, partial = divmod(cursor.consumed, self.config.tokens_per_page)
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
                    self._check_positions(
                        cursor.next_step(graph_walk), chain.spans[cursor.done], graph_walk, inputs,
                    )
            stream = self._streams.get(rid, {}).get(label)
            if stream is None or stream.lease is None or stream.stored_len:
                return
            if prefix is None:
                self._drop_layout(rid, label, stream, f"{graph_walk} writes unprobed")
                return
            cursor = stream.span_cursor
            if prefix.tokens == sum(span.length for span in cursor.next_step(graph_walk)):
                return
            held = cursor.consumed + prefix.tokens
            if held < stream.lease_len:
                keep = held // self.config.tokens_per_page
                cut = stream.lease[keep:]
                stream.lease, stream.lease_len = stream.lease[:keep] or None, keep * self.config.tokens_per_page
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
            step = cursor.next_step(graph_walk)
            cursor.consumed += sum(span.length for span in step)
            cursor.done += len(step)

    def _index_filled_pages(
        self, segment: Segment, stream: CacheStream, ctx: StepContext,
        touched: list[tuple[int, int]], span: int,
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
        lag = self._run_lag + (
            self._chain_pending.get(segment.request_id, 0) * self._chain_backlog_width
            if self._chain_backlog else 0
        )
        if stream.released or stream.stored_len - span > chain.covered_len + lag:
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
        filled = chain.pages_filled(stream.stored_len, self.config.tokens_per_page)
        if stream.retention is not None:
            filled = min(
                filled, stream.retention.protected_prefix // self.config.tokens_per_page
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
            # The index files a page under a parent only something else holds. Above one rank the owner counts can
            # differ between ranks, so only a parent this stream holds is offered there.
            held = chain.cursor > 0 and stream.page_indices[chain.cursor - 1] == parent
            if not self._index.insert(key, page, parent if self._world_size == 1 or held else None):
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
        page = end // self.config.tokens_per_page
        rooted = fingerprint(self._prefix_root, key)
        parent = self._index.page_for(fingerprint(self._prefix_root, chain.keys[page - 1])) if page else None
        if (page and parent is None) or self._index.page_for(rooted) is not None:
            return
        # only from free pages. A partial copy isn't worth evicting a full cached page for.
        copy = self._arena.acquire(1)
        if copy is None:
            logger.debug("KV %s: no free page for an image end, so it is not cached", self.name)
            return
        self.kv_cache.copy_slots(stream.page_indices[page], copy[0], end % self.config.tokens_per_page)
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
            :published_len // self.config.tokens_per_page
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

    def bind_snapshots(self, pool: "RecurrentStatePool") -> None:
        """Count a prefill hit only as far as ``pool`` restores the recurrent state under it."""
        self._snapshots.append(pool)

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
        keys = None if overrides is None else overrides.keys_of(label)
        # the chain reads the span cursor, which only a label the request opens has
        layout = self._held_layout(rid, label)
        if not keys or layout is None:
            return
        tail = list((overrides.prefix_tail or {}).get(label) or ())
        stream.chain = PrefixChain.seed(
            keys, tail, layout.spans[:layout.keyed_spans], self.config.tokens_per_page,
            root=overrides.prefix_root,
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
            stream.chain.extend(sampled[0].flatten().tolist(), self.config.tokens_per_page)

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
        page_size = self.config.tokens_per_page
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
            stream.chain.extend(tokens, self.config.tokens_per_page)

    def _extend_chains_from(self, steps: list, drained: set[str] = frozenset()) -> None:
        """Under the lock: one extend per request with the tokens of
        ``steps`` (same rows and batch) in step order."""
        rows, node_name, walks, request_ids, _ = steps[0]
        row_of = {rid: i for i, rid in enumerate(rows)}
        page_size = self.config.tokens_per_page
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
        """Make ``pages`` the whole of a stream that has written nothing yet. The lent ones come off a
        guaranteed request's allotment, since no cut can release them now.

        Any pages the stream already holds are empty ones a refused admit left, so they are released.
        """
        empty = stream.page_indices.tolist()
        stream.page_indices.assign(pages)
        self._release(rid, stream, empty)
        allotment = self._allotments.get(rid)
        if rid in self._guaranteed and allotment is not None:
            allotment.pages -= len(stream.lent.intersection(pages))
        self._readonly_epoch += 1
        stream.stored_len = len(pages) * self.config.tokens_per_page

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

    @property
    def placement(self) -> "ContextParallel | None":
        """This rank's placement of the pool's positions (``PagedKVConfig.context_parallel``), which
        the resources over it read here; None when it is off."""
        return self.kv_cache.placement

    def fingerprint(self) -> bytes:
        cp = self.placement
        return fingerprint(
            self.config.layout.value, self.config.tokens_per_page,
            self.kv_cache.tensor.dtype, self._world_size,
            self.config.prefix_cache_salt,
            *(() if cp is None else ("context_parallel", cp.size, type(cp).__name__)),
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
            if rid not in self._streams:
                self._streams[rid] = {"main": self._new_stream()}
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
            stream.footprint = layout.pages(self._overrides[rid].loop_caps, self.config.tokens_per_page)
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
        # asked of a waiting row every readiness pass: a pool that retrieves
        # from another node refuses resident planning, so its row stands
        self._touch_plan_cache(rid, row=False)
        if self._host_copies:
            with self._lock:
                self._land_host_copies()
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
                    pages = stream.pages()
                    drop = self._arena.any_sealed(pages)
                    if drop:
                        # before the reset, which clears the lists ``pages`` may alias
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
                # the copy and the run's evictable pages come out of the allowance, and the run is held whole
                evictable = self._index.evictable()
                taken = sum(page in evictable for page in run)
                budget = self._allowance(rid) - taken if taken <= self._hit_allowance(rid) else 0
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
            # plan and reserve for this step normally. A caller that throws a
            # stage away unwinds its reservation first, as `StepRunner.clear_preplan` does.
            self.clear_preplan()
        if (step.pre_forks or step.post_forks) and len(self._groups) > 1:
            # A step declares its forks, so this is the first place to see them. Refused before
            # this resource reserves anything, and terminal, so the runner unwinds the step.
            return AdmitOutcome(ok=False, reason=AdmitRuntimeError(
                f"KV {self.name}: a pool that keeps a window doesn't fork streams yet"
            ))
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
            streams = self._streams
            report = self._index is not None
            trail = self._arena.trail
            undoable = ctx.is_preplan and self._world_size > 1
            # leases come only from the index, so without one this pass does nothing
            for segment in step.segments if report else ():
                labels = streams.get(segment.request_id)
                stream = labels.get(segment.label) if labels else None
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
                    whole = stream.lease_len // self.config.tokens_per_page
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
                if report:
                    self._report_admission(segment, stream)
            if ctx.is_preplan:
                # a preplan that was promoted or abandoned already cleared
                # these; reset anyway so a refused admit can't leave stale
                # entries for the next clear_preplan to act on
                self._preplan_new_labels = []
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
                        extra=0 if not extra else extra.get((rid, from_label), 0), undoable=undoable,
                    )
                    if not alloc_res.success:
                        return AdmitOutcome(ok=False, reason=alloc_res.error)
                    if alloc_res.new_pages:
                        self._admit_reserved_pages.append(Reservation(rid, to_label, 0, alloc_res.new_pages, trail))

            page_size = self.config.tokens_per_page
            sink = step.sink
            # the counts are in tokens; the setting names the rows whose KV
            # grows by them, so they are a span this label can reserve
            reserve_ahead = ctx.reserve_whole_prefill
            ahead = ctx.chunk_tokens_after \
                if reserve_ahead.value is not SettingValue.UNSET else None
            # One pass for the reservations and the streams the in-flight mark
            # below covers. A segment against the sink (`_sinks`) reserves
            # nothing (`_reserves`) and is not marked.
            marked: list[tuple[Segment, CacheStream]] = []
            # In a pool of one group, the rows whose pages already cover their
            # span are found in one call and marked after; the rest reserve
            # below. A pool with a window covers its later groups too, and a
            # chunked row reserves ahead, so those steps take every row here.
            ready: list[CacheStream] = []
            rows = step.segments
            repeats = ctx.repeats
            if not ctx.capture and not ahead and repeats == 1 and len(self._groups) == 1:
                ready, slow = self._arena.admit_rows(streams, rows, ctx.real_rows(), sink or None, page_size)
                rows = [rows[i] for i in slow]
            for segment in rows:
                if self._sinks(ctx, sink, segment):
                    continue
                rid, label = segment.request_id, segment.label
                if segment.span == 0:
                    # reserves nothing, and creates no stream for a label nothing created
                    stream = streams.get(rid, {}).get(label)
                    if stream is not None:
                        marked.append((segment, stream))
                    continue
                labels = streams.get(rid)
                stream = labels.get(label) if labels else None
                if stream is None:
                    if ctx.is_preplan:
                        # as the fork loop above: an abandoned stage removes it
                        self._preplan_new_labels.append((rid, label))
                    stream = self._ensure_label(rid, label)
                marked.append((segment, stream))
                # a chunked row reserves the rest of its walk up front, in every group
                span = segment.span * repeats
                if ahead and reserve_ahead.holds_for(rid):
                    span += ahead.get(rid, 0)
                seq_len = span + stream.stored_len
                # the common decode row: its pages already cover the token it
                # appends, so there is nothing to acquire and nothing to record
                if (
                    not stream.offloaded
                    and stream.page_indices.n * page_size >= seq_len
                    and (not stream.later or self._covers(stream, span))
                ):
                    continue
                for g in range(len(self._groups)):
                    alloc_res = self._alloc(rid, label, span + stream.group(g).stored_len, group=g, undoable=undoable)
                    if not alloc_res.success:
                        return AdmitOutcome(
                            ok=False, reason=self._sized_refusal(step, ctx, alloc_res.error),
                        )
                    if alloc_res.new_pages:
                        self._admit_reserved_pages.append(Reservation(rid, label, g, alloc_res.new_pages, trail))

            # marked here rather than in plan so the mark also covers
            # admit -> plan, where an offload would otherwise release pages
            # this step has already been given. last, and only once every
            # reservation above succeeded, so a refusal here sets none.
            # (`_alloc` adds pages to a stream, never replaces it, and the
            # lock is held throughout, so the streams found above are current)
            self._arena.set_in_flight(ready, True)
            for _, stream in marked:
                stream.step_in_flight = True
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
        stream, a chunked row's whole walk included. A segment that runs against the sink holds and needs nothing,
        and a span-0 one needs nothing beyond what it holds, as in `admit`. The
        pool is every page but the sink, cached prefixes included, since those
        can be evicted. ``rows_sized`` is how many real rows were counted.
        """
        if not isinstance(error, AllocationFailed):
            return error
        sink = step.sink
        reserve_ahead = ctx.reserve_whole_prefill
        ahead = ctx.chunk_tokens_after \
            if reserve_ahead.value is not SettingValue.UNSET else None
        held: dict[str, set[int]] = {}
        needed: dict[tuple[str, str, int], int] = {}
        for segment in step.segments:
            rid = segment.request_id
            if self._on_sink(ctx, rid):
                continue
            # a row with every label on the sink still counts as a row that fits
            rows_pages = held.setdefault(rid, set())
            if self._sinks(ctx, sink, segment):
                continue
            stream = self._streams.get(rid, {}).get(segment.label)
            # a label `admit` hasn't reached yet holds nothing
            if stream is not None:
                rows_pages.update(stream.pages())
            if not self._reserves(ctx, sink, segment):
                continue
            walk_ahead = ahead.get(rid, 0) if ahead and reserve_ahead.holds_for(rid) else 0
            for g, group in enumerate(self._groups):
                part = None if stream is None else stream.group(g)
                pages = () if part is None else part.page_indices
                stored = 0 if part is None else part.stored_len
                # as `admit` reserves it: a chunked row's whole walk
                more = -(-(stored + segment.span + walk_ahead) // group.page_size) - len(pages)
                key = (rid, segment.label, g)
                needed[key] = max(needed.get(key, 0), more)
        extra: dict[str, int] = {}
        for (rid, _, _), more in needed.items():
            extra[rid] = extra.get(rid, 0) + max(more, 0)
        # in arena pages, as the pages above count; the message counts pages of every layer
        capacity = self._arena.max_num_pages - 1
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
                        f"The request needs {self.config.pages_of(own, round_up=True)} pages of {self.name} for "
                        f"this step, more than the {self.config.pages_of(capacity)} the pool "
                        "holds; raise max_num_pages or shorten the request."
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

        The arena ends as the admit found it, its free list in the same order
        and its trail the same, so the next admit takes the pages it takes on a
        TP rank that never ran this one.

        ``page_indices`` maps token position to page, so the reserved pages can
        only be dropped off the tail. If something else has extended the stream
        since, the tail can no longer be safely removed.
        """
        self._touch_plan_cache()
        self._drop_resident_stage()
        with self._lock:
            # No owner check: a refused admit unwinds before `commit_done` is
            # released, and the only pre-admit (`StepExecutor._preplan_spec`) waits on
            # it and is unwound before another step admits, so no other
            # uncommitted step can hold marks on these streams.
            # `.get` as in `admit`: a refusal can come before a stream exists
            for segment in step.segments:
                stream = self._streams.get(
                    segment.request_id, {}
                ).get(segment.label)
                if stream is not None:
                    stream.step_in_flight = False
            mine = [held for held in self._admit_reserved_pages if ctx.is_real_row(held.rid)]
            self._forget_step_reservations(ctx)
            # newest first, since each goes back to the front of the free list
            for rid, label, g, pages, trail in reversed(mine):
                stream = self._streams.get(rid, {}).get(label)
                if stream is None:
                    continue  # reset or removed under us; its pages went with it
                part = stream.group(g)
                if part.page_indices[-len(pages):] != pages:
                    logger.error(
                        "KV %s: cannot unwind %s/%s's admit — its pages are no "
                        "longer the tail of the stream (%d reserved, stream has "
                        "%d). Leaving them; the free pool is short by that much.",
                        self.name, rid, label, len(pages),
                        len(part.page_indices),
                    )
                    continue
                del part.page_indices[-len(pages):]
                self._arena.give_back(pages)
                self._arena.trail = trail
                self._recount(rid)
                # so an offload that claimed this stream mid-admit notices its
                # host copy no longer describes it and aborts
                part.generation += 1
            # Record the room again after the rollback. Otherwise `preempt` would count the pages given back here as
            # released since a grant ran short. Run row by row, that grant's row has already rolled back on its own.
            room = self._room()
            for rid, short in self._ran_short.items():
                self._ran_short[rid] = short._replace(room=room)
            if _DEBUG_ASSERTS:
                self.assert_pages_conserved()

    def _forget_reservations(self, rid: int) -> None:
        """Drop ``rid``'s unwind record. Caller holds the lock."""
        self._admit_reserved_pages = [held for held in self._admit_reserved_pages if held.rid != rid]

    def _forget_step_reservations(self, ctx: StepContext) -> None:
        """Drop the unwind record of this step's rows only. Caller holds the lock."""
        self._admit_reserved_pages = [held for held in self._admit_reserved_pages if not ctx.is_real_row(held.rid)]

    def _sequence_views(
        self,
        segments: list[Segment],
        pending_forks: dict[tuple[str, str], tuple[int, int]] | None = None,
        ctx: StepContext | None = None,
        sink: frozenset[tuple[str, str]] = frozenset(),
        group: int = 0,
    ) -> list[SequenceView]:
        """One view per segment, of its stream's pages in group ``group``."""
        views = []
        page_size = self._groups[group].page_size
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
            stream = self._streams[s.request_id][s.label].group(group)
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

    def _empty_plan_state(self) -> KVPlanState:
        empty = self._empty_locations
        if empty is None:
            empty = self._empty_locations = torch.empty(0, dtype=torch.long, device=self._device)
        return KVPlanState(token_to_page=empty, token_to_cache=empty, total_tokens=0)

    def _decode_plan_state(
        self, views: list[SequenceView] | None, page_size: int, pages=None, offsets=None, *,
        placement: "ContextParallel | None",
    ) -> KVPlanState:
        """Write addressing for a step appending one token per request, at ``placement``'s slots
        of a placed group (None for an unplaced one).

        The packed path needs the indptrs on device and ~a dozen kernels to
        unpack them per token. A decode step's slot is just the end of each
        stream, so build it in the same CPU pass the views came from and send
        it over as one page-locked H2D, as an eager packed write's is.
        """
        if pages is None or offsets is None:
            pages, offsets = self._decode_locations(views, page_size)
        if placement is not None:
            pages, offsets = placement.write_slots(pages, offsets)
        n = len(pages)
        host = np.empty(2 * n, np.int64)
        host[:n], host[n:] = pages, offsets
        both = torch.empty(2 * n, dtype=torch.long, device=self._device)
        self._plan_stager.copy_(both, host)
        return KVPlanState(token_to_page=both[:n], token_to_cache=both[n:], total_tokens=n)

    @staticmethod
    def _decode_locations(
        views: list[SequenceView], page_size: int,
    ) -> tuple[list[int], list[int]]:
        """Each decode row's (page, offset-in-page) for the token it writes.

        Off the stream's page count, not its logical length: that is what
        `build_paged_indptrs` hands attention, so a stream holding more pages
        than its length needs stays self-consistent."""
        pages = [view.page_idxs[-1] for view in views]
        offsets = [(view.last_page_len(page_size) or page_size) - 1 for view in views]
        return pages, offsets

    def _stage_packed_plan_state(
        self, indptrs: PagedIndptrs, static_state: KVPlanState, capture_len: int, page_size: int, *,
        placement: "ContextParallel | None",
    ) -> KVPlanState:
        """``_packed_addressing`` staged into the captured state."""
        total = int(indptrs.qo_indptr[-1])
        if total == 0:
            # every row reads without writing (a zero-span context label)
            return self._stage_static(static_state, (), (), 0)
        both = self._packed_addressing(indptrs, total, page_size, placement=placement)
        return self._stage_static(static_state, both[:total], both[total:], total)

    def _packed_addressing(
        self, indptrs: PagedIndptrs, total: int, page_size: int, *, placement: "ContextParallel | None",
    ) -> np.ndarray:
        """Each new token's page, then each one's offset in it (``2 * total``
        int64), from the plan's host indptrs: tokens are appended after each
        row's stored length. One Rust call when the extension has it."""
        qo = indptrs.qo_indptr.numpy()
        if _rust_packed_addressing is not None:
            both = np.empty(2 * total, np.int64)
            _rust_packed_addressing(
                qo, indptrs.paged_kv_indptr.numpy(), indptrs.paged_kv_indices.numpy(),
                indptrs.paged_kv_last_page_len.numpy(), page_size, both,
            )
            return self._placed(both, total, placement)
        kvp = indptrs.paged_kv_indptr.numpy()
        spans = np.diff(qo)
        seg = np.repeat(np.arange(len(spans)), spans)
        # each token's position in its stream, after the append
        last = indptrs.paged_kv_last_page_len.numpy()
        kv_after = (np.diff(kvp) - 1) * page_size + last
        # a writing row must hold its tokens in its own pages: a negative page
        # would wrap in numpy and address another row's page
        live = spans > 0
        if not ((kv_after[live] >= spans[live]) & (last[live] <= page_size)).all():
            raise RuntimeError(
                "KV packed plan: a row writes past its own pages "
                f"(spans {spans.tolist()}, kv indptr {kvp.tolist()}, last page lens {last.tolist()})"
            )
        pos = (kv_after - spans - qo[:-1])[seg] + np.arange(total)
        both = np.empty(2 * total, np.int64)
        page, both[total:] = np.divmod(pos, page_size)
        both[:total] = indptrs.paged_kv_indices.numpy()[kvp[:-1][seg] + page]
        return self._placed(both, total, placement)

    @staticmethod
    def _placed(both: np.ndarray, total: int, placement: "ContextParallel | None") -> np.ndarray:
        """``both`` (``_packed_addressing``'s layout) at this rank's slots of a placed group
        (``ContextParallel.write_slots``), in place."""
        if placement is not None:
            both[:total], both[total:] = placement.write_slots(both[:total], both[total:])
        return both

    def _host_plan_state(
        self, indptrs: PagedIndptrs, total: int, page_size: int, *, placement: "ContextParallel | None",
    ) -> KVPlanState:
        """An eager write's addressing: computed on the host and uploaded in
        one page-locked copy, rather than unpacked on the device from
        uploaded indptrs (four copies and a dozen small kernels a step)."""
        host = self._packed_addressing(indptrs, total, page_size, placement=placement)
        both = torch.empty(2 * total, dtype=torch.long, device=self._device)
        self._plan_stager.copy_(both, host)
        return KVPlanState(token_to_page=both[:total], token_to_cache=both[total:], total_tokens=total)

    def _stage_decode_plan_state(
        self, views: list[SequenceView] | None, static_state: KVPlanState,
        capture_len: int, page_size: int, pages=None, offsets=None, *, placement: "ContextParallel | None",
    ) -> KVPlanState:
        """``_decode_plan_state`` staged into the captured state, with no device
        staging tensor."""
        if pages is None or offsets is None:
            pages, offsets = self._decode_locations(views, page_size)
        if placement is not None:
            pages, offsets = placement.write_slots(pages, offsets)
        return self._stage_static(static_state, pages, offsets, len(pages))

    def _setup_plan_states(
        self, plan_output: dict[str, KVPlanOutput],
        ctx: StepContext, lease, group: int = 0, eager_write: bool = False,
    ):
        """The write addressing of each plan label in group ``group``, keyed (label, group)."""
        if eager_write:
            # written outside the graph (KVStep.eager_write): never staged
            lease = None
        page_size = self._groups[group].page_size
        # the placement stripes the first group alone; a window group is whole on every partner
        placement = self.placement if group == 0 else None
        for plan_label, indptrs in plan_output.items():
            label = (plan_label, group)
            if indptrs.is_decode and lease is not None:
                # Straight into the captured buffers in one H2D per buffer:
                # this runs in the pre-plan beside a live graph, where a
                # device-side copy + fill would slow it (see mstar.utils.h2d).
                # a plan built from columns hands its locations; reading
                # `views` there would build them eagerly and defeat the lazy builder
                located = indptrs.decode_pages is not None
                plan_state = self._stage_decode_plan_state(
                    None if located else indptrs.views,
                    self._static_plan_state(lease.slot, label, lease.bucket.num_tokens),
                    lease.bucket.num_tokens, page_size,
                    pages=indptrs.decode_pages, offsets=indptrs.decode_offsets, placement=placement,
                )
                self._staged_outputs[(lease.slot, label)] = indptrs
                if ctx.is_preplan:
                    self._preplan_states[label] = plan_state
                else:
                    self._current_plan_states[label] = plan_state
                continue
            if lease is not None:
                # a captured packed step (prefill, mixed): addressing off the
                # host indptrs, staged as for decode, not unpacked on device
                plan_state = self._stage_packed_plan_state(
                    indptrs.cpu_indptrs,
                    self._static_plan_state(lease.slot, label, lease.bucket.num_tokens),
                    lease.bucket.num_tokens, page_size, placement=placement,
                )
                self._staged_outputs[(lease.slot, label)] = indptrs
                if ctx.is_preplan:
                    self._preplan_states[label] = plan_state
                else:
                    self._current_plan_states[label] = plan_state
                continue
            total_tokens = None if indptrs.is_decode else indptrs.get_total_len()
            if indptrs.is_decode:
                plan_state = self._decode_plan_state(
                    None if indptrs.decode_pages is not None else indptrs.views, page_size,
                    indptrs.decode_pages, indptrs.decode_offsets, placement=placement,
                )
            elif total_tokens == 0:
                # every segment span 0 (a decoder reading the encoder context
                # it wrote at prefill): nothing is written, so there is no
                # addressing to upload and unpack
                plan_state = self._empty_plan_state()
            else:
                plan_state = self._host_plan_state(indptrs.cpu_indptrs, total_tokens, page_size, placement=placement)
            # (a leased step staged above: these are eager)
            if eager_write and ctx.is_preplan and plan_state.token_to_page.is_cuda:
                plan_state.ready = torch.cuda.Event()
                plan_state.ready.record()
            if ctx.is_preplan:
                self._preplan_states[label] = plan_state
            else:
                self._current_plan_states[label] = plan_state


    def _plan_output(
        self, views: list[SequenceView], page_size: int, ring: PinnedIndexRing | None,
    ) -> KVPlanOutput:
        return KVPlanOutput(
            cpu_indptrs=build_paged_indptrs(views, page_size, ring=ring),
            views=views,
        )

    def _plan_label(
        self, segments: list[Segment], pending: dict, ctx: StepContext,
        sink: frozenset[tuple[str, str]], ring: PinnedIndexRing | None, group: int = 0,
    ) -> KVPlanOutput:
        """One plan label's output in group ``group``:
        ``_plan_output(_sequence_views(...))``, built from columns. One pass
        over the rows reads each stream's id, length and generation; the arrays
        then come from one call (`arena.plan_columns`), and the views are built
        only if a consumer asks. A step whose pre-forks are pending plans from
        the views."""
        page_size = self._groups[group].page_size
        if pending:
            return self._plan_output(self._sequence_views(segments, pending, ctx, sink, group), page_size, ring)
        if not ctx.capture and group == 0 and ctx.repeats == 1:
            return self._plan_label_rows(segments, ctx, sink, ring, page_size)
        streams = self._streams
        capture = ctx.capture
        is_pad = ctx.is_padding_row
        sids, lens, spans, gens = [], [], [], []
        sink_rows = []
        decode = bool(segments)
        for i, s in enumerate(segments):
            span = s.span
            spans.append(span)
            if span != 1:
                decode = False
            if capture or is_pad(s.request_id) or (sink and (s.request_id, s.label) in sink):
                # a padding row reads and writes SINK_PAGE, a page per page of
                # span; see `_sequence_views`
                sink_rows.append(i)
                sids.append(0)
                lens.append(span)
                gens.append(0)
                continue
            stream = streams[s.request_id][s.label].group(group)
            sids.append(stream.page_indices.sid)
            # a step that repeats plans its last replay: every page it reaches
            lens.append(span * ctx.repeats + stream.stored_len)
            gens.append(stream.generation)
        if sink_rows:
            sink_pages = self._sink_pages
            need = -(-max(spans[i] for i in sink_rows) // page_size)
            if len(sink_pages) < need:
                self._extend_sink_pages(need)
            # a row reads no more pages than its length needs
            for i in sink_rows:
                sids[i] = sink_pages.sid
        cols = plan_columns(self._arena, sids, lens, spans, page_size)
        indices, kv_indptr, last = cols.indices, cols.kv_indptr, cols.last_page_len
        make = ring.take if ring is not None else torch.from_numpy
        # a decode row writes the token after its last: the last page, at its
        # last slot (`_decode_locations`); none without a page to write
        write_at = decode and cols.all_have_pages

        def build_views(segments=segments, indices=indices, kv_indptr=kv_indptr, lens=lens, gens=gens):
            bounds = kv_indptr.tolist()
            return [
                SequenceView(
                    s.request_id, s.label, indices[bounds[i]:bounds[i + 1]].tolist(),
                    lens[i], s.span, 0, gens[i],
                )
                for i, s in enumerate(segments)
            ]

        return KVPlanOutput(
            cpu_indptrs=PagedIndptrs(
                qo_indptr=make(cols.qo_indptr),
                paged_kv_indptr=make(kv_indptr),
                paged_kv_indices=make(indices),
                paged_kv_last_page_len=make(last),
                kv_lens=torch.from_numpy(cols.kv_lens).as_subclass(HostLens),
            ),
            decode_pages=cols.decode_pages if write_at else None,
            decode_offsets=cols.decode_offsets if write_at else None,
            decode=decode,
            rows=segments,
            view_builder=build_views,
            columns=cols,
        )

    def _plan_label_rows(
        self, segments: list[Segment], ctx: StepContext, sink: frozenset[tuple[str, str]],
        ring: PinnedIndexRing | None, page_size: int,
    ) -> KVPlanOutput:
        """`_plan_label` for a step that isn't a capture, in the first group,
        every row read off its stream in one call (`PageArena.plan_rows`)."""
        sink_pages = self._sink_pages
        args = (self._streams, segments, ctx.real_rows(), sink or None, sink_pages.sid, page_size)
        raw, total, all_have, lens_gens, max_sink, decode = self._arena.plan_rows(*args)
        if max_sink and len(sink_pages) * page_size < max_sink:
            self._extend_sink_pages(-(-max_sink // page_size))
            raw, total, all_have, lens_gens, max_sink, decode = self._arena.plan_rows(*args)
        n = len(segments)
        cols = columns_from_raw((raw, total, all_have), n)
        indices, kv_indptr, last = cols.indices, cols.kv_indptr, cols.last_page_len
        make = ring.take if ring is not None else torch.from_numpy
        write_at = decode and all_have

        def build_views(segments=segments, indices=indices, kv_indptr=kv_indptr, lens_gens=lens_gens):
            bounds = kv_indptr.tolist()
            lg = np.frombuffer(lens_gens, np.int64).tolist()
            return [
                SequenceView(
                    s.request_id, s.label, indices[bounds[i]:bounds[i + 1]].tolist(),
                    lg[i], s.span, 0, lg[n + i],
                )
                for i, s in enumerate(segments)
            ]

        return KVPlanOutput(
            cpu_indptrs=PagedIndptrs(
                qo_indptr=make(cols.qo_indptr),
                paged_kv_indptr=make(kv_indptr),
                paged_kv_indices=make(indices),
                paged_kv_last_page_len=make(last),
                kv_lens=torch.from_numpy(cols.kv_lens).as_subclass(HostLens),
            ),
            decode_pages=cols.decode_pages if write_at else None,
            decode_offsets=cols.decode_offsets if write_at else None,
            decode=decode,
            rows=segments,
            view_builder=build_views,
            columns=cols,
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

    def _readonly_key(self, step: KVStep, ctx: StepContext) -> tuple[tuple, int]:
        """(segments, bucket, eager_write), and the epoch a plan under that key
        holds for: an eager-write plan is never staged, so it isn't another's."""
        lease = ctx.slot_lease
        return (
            (tuple(step.segments), lease.bucket if lease is not None else None, step.eager_write),
            self._readonly_epoch,
        )

    def _reuse_readonly_plan(self, res: "KVPlanOutputs", step: KVStep, ctx: StepContext) -> "KVPlanOutputs":
        """A cached read-only plan for this step, its views and indptrs as they
        are; a slot whose static buffers already hold it is not re-staged."""
        lease = ctx.slot_lease
        if lease is not None and all(
            self._staged_outputs.get((lease.slot, (label, g))) is out
            for g, outputs in enumerate(res.groups) for label, out in outputs.items()
        ):
            states = self._preplan_states if ctx.is_preplan else self._current_plan_states
            for g, outputs in enumerate(res.groups):
                for label in outputs:
                    states[label, g] = self._static_plan_state(
                        lease.slot, (label, g), lease.bucket.num_tokens,
                    )
        else:
            for g, outputs in enumerate(res.groups):
                self._setup_plan_states(outputs, ctx, lease, g, eager_write=step.eager_write)
        self._mark_preplanned(step, ctx, res)
        return res

    def _drop_readonly_plans(self) -> None:
        self._readonly_epoch += 1
        # The epoch is what retires an entry: a plan thread mid-build stores
        # the epoch it read before this bump, which never matches again. The
        # dict is rebound rather than cleared only so that thread keeps a
        # consistent one to write into.
        self._readonly_plans = {}
        self._readonly_rows = frozenset()

    def _touch_plan_cache(self, rid: str | None = None, row: bool = True) -> None:
        """A stream of ``rid`` (any stream, when None) changed other than by a
        commit: the read-only plans no longer describe the streams if they
        hold that request. A padding row's reset after a replay holds nothing.

        ``row`` also marks its resident row stale. A touch of every stream
        doesn't: those are refused admits and abandoned pre-plans, which leave
        only pages past the row's committed columns."""
        if rid is None or rid in self._readonly_rows:
            self._drop_readonly_plans()
        if row and rid is not None and self._resident is not None:
            self._resident.planner.mark_stale(rid)

    def _drop_resident_stage(self) -> None:
        """Pages a staged step reserved go back: the resident stage that names
        them goes too, as the runner's own does."""
        if self._resident is not None:
            self._resident.planner.drop_stage()

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
                    whole, end = divmod(stream.stored_len, self.config.tokens_per_page)
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
        laps = PhaseLaps(f"{ctx.plan_phase}.{self.name}.")
        self.reset_default_cursors()
        if ctx.resident is not None:
            bucket = ctx.slot_lease.bucket
            if bucket in ctx.resident.host_plan_free:
                # nothing reads a plan of this step, and its graph fills the addressing
                return KVPlanOutputs({})
            if (bucket, self._resident.key) in ctx.resident.lengths_only:
                # its graph fills the addressing and the attention's page arrays;
                # a promoted step keeps the resident plan its pre-plan staged
                plan = ctx.resident_plan
                held = self._lengths_plan
                if held is not None and held[0] is plan:
                    return held[1]
                out = KVPlanOutputs({
                    plan.writes.streams[self._resident.key].label: KVPlanOutput(
                        self._resident.decode_indptrs(plan, bucket.bs),
                    ),
                })
                self._lengths_plan = (plan, out)
                return out
        if self._preplanned:
            if self._preplan_key == self._plan_key(step, ctx):
                self._current_plan_states = self._preplan_states
                res = self._cached_plan_output

                # Promotion: apply the fork copies staging left undone; see
                # `_pending_fork_state`.
                self._maybe_apply_forks(step, ctx)
                self._copy_image_ends()

                self._preplan_new_labels = []
                # the promoted step runs, so no stream changed under the plans
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
            key, epoch = self._readonly_key(step, ctx)
            slot = ctx.slot_lease.slot if ctx.slot_lease is not None else None
            cached = self._readonly_plans.get(slot)
            if cached is not None and cached[0] == key and cached[1] == epoch:
                return self._reuse_readonly_plan(cached[2], step, ctx)
            # The epoch and rows before the views: a touch from here on moves
            # the epoch, so a plan built across it is never reused. A pre-plan
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
        by_label = group_by_plan_label(step.segments, step.combined_labels)
        outputs = [
            {
                plan_label: self._plan_label(segments, pending, ctx, step.sink, ring, g)
                for plan_label, segments in by_label.items()
            }
            for g in range(len(self._groups))
        ]
        res = KVPlanOutputs(
            outputs[0],
            pre_forks=step.pre_forks,
            post_forks=step.post_forks,
            fork_rids=step.fork_rids,
            later_groups=tuple(outputs[1:]),
        )
        laps.lap("full")
        if ctx.resident is None:
            for g, outputs in enumerate(res.groups):
                self._setup_plan_states(outputs, ctx, ctx.slot_lease, g, eager_write=step.eager_write)
        else:
            # the replay fills the slot's addressing (a stream the step only
            # reads has none), so no reuse may skip re-staging it
            for label in res:
                self._staged_outputs.pop((ctx.slot_lease.slot, (label, 0)), None)
        laps.lap("setup_states")
        if readonly:
            self._readonly_plans[slot] = (key, epoch, res)
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
            self._drop_resident_stage()
        # The staged step is not going to run. Staging only read the
        # pre-forks, and the caller's `rollback_admit` gave back admit's pages,
        # so what is left to undo is the streams admit created.
        with self._lock:
            # labels the staged step invented are removed, not rewound to 0:
            # a stream at 0 that nothing asked for is still a stream
            for rid, label in reversed(self._preplan_new_labels):
                stream = self._streams.get(rid, {}).pop(label, None)
                if stream is not None:
                    self._release(rid, stream, stream.pages())
                    stream.free_lists()
                    self._recount(rid)
            self._preplan_new_labels = []
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_key = None
        self._preplan_states = {}
        self._cached_plan_output = None

    def correct_len(self, rid: str, label: str, delta: int) -> None:
        """Move a stream's stored length by ``delta`` tokens, pages kept: a
        speculative verify step commits its whole block and its rejected tail
        is taken back here once the acceptance is known. The generation moves
        so an offload claimed against the old length fails its guard."""
        with self._lock:
            stream = self._streams.get(rid, {}).get(label)
            if stream is None:
                raise KeyError(f"KV {self.name}: no stream {label!r} for request {rid}")
            if len(self._groups) > 1:
                # each group would rewind its own pages; no grouped cache serves MTP yet
                raise NotImplementedError(f"KV {self.name}: correct_len on a cache with KV groups")
            # an admitted step's addressing was planned from the old length
            assert not stream.step_in_flight, (
                f"KV {self.name}: correct_len on {rid}/{label} between admit and commit"
            )
            if delta == 0:
                return
            stream.stored_len = max(0, stream.stored_len + delta)
            stream.generation += 1
            # the read-only plans no longer describe the stream
            self._touch_plan_cache(rid)
            if delta < 0:
                self._drop_sealed_tail(rid, label, stream)
            if _DEBUG_ASSERTS:
                self.assert_pages_conserved()

    def _drop_sealed_tail(self, rid: str, label: str, stream: CacheStream) -> None:
        """After a rewind the next write must not land on a sealed page other owners
        still read: the pages from the write position on leave through `_release`, and
        the one the length ends inside is replaced by a copy."""
        page_size = self.config.tokens_per_page
        first = stream.stored_len // page_size
        tail = stream.page_indices[first:]
        if not self._arena.any_sealed(tail):
            return
        keep: list[int] = []
        if stream.stored_len % page_size:
            keep = self._arena.acquire(1)
            if keep is None:
                raise RuntimeError(
                    f"KV {self.name}: no free page to copy the sealed page "
                    f"{rid}/{label} rewound into"
                )
            self.kv_cache.copy_pages(tail[:1], keep)
        del stream.page_indices[first:]
        stream.page_indices.extend(keep)
        self._release(rid, stream, tail)
        # its keys and cursor name pages this stream no longer holds
        stream.chain = None
        self._recount(rid)

    def commit(self, step: KVStep, ctx: StepContext):
        streams = self._streams
        retention = step.retention
        if not step.commit:
            self._readonly_epoch += 1
        rows = self._readonly_rows
        if rows and any(seg.span and seg.request_id in rows for seg in step.segments):
            self._drop_readonly_plans()
        # the index only sees real rows of a non-capture step; checked once
        index_on = self._index is not None and not ctx.capture
        windows = self._windows
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
            is_pad = ctx.is_padding_row
            sink = step.sink
            rows = () if ctx.capture else step.segments
            repeats = ctx.repeats
            if repeats != 1:
                # a repeated step's stored spans are `repeats` of what it declared (see `_run_lag`)
                self._run_lag = max(self._run_lag, 2 * (repeats - 1))
            if rows and repeats == 1 and self._index is None and not retention and not windows and not _DEBUG_ASSERTS:
                # nothing to index, adopt, retain or check: the arena moves
                # every row's counts in one call, and hands back the ones it can't
                slow = self._arena.commit_rows(streams, rows, ctx.real_rows(), sink or None, step.commit)
                rows = [rows[i] for i in slow]
            for segment in rows:
                if is_pad(segment.request_id) or (sink and (segment.request_id, segment.label) in sink):
                    continue  # ran against SINK_PAGE, holds nothing
                stream = streams[segment.request_id][segment.label]
                # cleared before the `step.commit` test: a step that keeps no
                # tokens (image_gen, action_gen) still read these pages, and
                # leaving the mark set would make the request unevictable
                stream.step_in_flight = False
                span = segment.span * repeats
                if step.commit and span > 0:
                    # an offload beat the mark (claimed before this step's
                    # admit). the host copy predates the span, so writing the
                    # length here would be lost on reload
                    if stream.offloaded:
                        logger.warning(
                            "KV %s: dropping %d committed tokens for %s label "
                            "%s; the stream was offloaded mid-step",
                            self.name, span,
                            segment.request_id, segment.label,
                        )
                        self._readonly_epoch += 1
                        continue
                    stream.stored_len += span
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
                        touched = stream.span_cursor.advance(written - span, written)
                        if _DEBUG_ASSERTS:
                            self._assert_layout(segment, stream, touched, ctx, policy)
                    if index_on and stream.chain is not None:
                        self._index_filled_pages(segment, stream, ctx, touched, span)
                    # so a claim taken in a window the mark misses still fails
                    # `_commit_offload`'s generation guard
                    stream.generation += 1
                    for g, group in windows:
                        part = stream.group(g)
                        part.stored_len += span
                        part.generation += 1
                        # What the window's layers can no longer read goes back to the pool, as
                        # below. No touch: this commit already dropped the read-only plans of
                        # the rows it writes, and a pool with window groups has no resident rows.
                        self._apply_retention(segment.request_id, stream, group.retention, g, touch=False)
                    if windows and segment.request_id in self._allotments:
                        self._settle_window_allotment(segment, stream)
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

    def _apply_retention(
        self, rid: str, stream: CacheStream, policy: RetentionPolicy, group: int = 0, touch: bool = True,
    ) -> int:
        """Release what ``policy`` no longer keeps of the stream's pages in group ``group``: the
        step's policy, or a group's. ``touch`` runs `_touch_plan_cache`. Under the lock."""
        excess = stream.group(group).stored_len - policy.protected_prefix - policy.context_budget
        if excess <= 0:
            return 0
        if touch:
            self._touch_plan_cache()
            self._touch_plan_cache(rid)
        return self._release_oldest_locked(rid, stream, excess, policy.protected_prefix, group)

    def _settle_window_allotment(self, segment: Segment, stream: CacheStream) -> None:
        """On the commit that writes the last of a stream's prompt, drop its window groups' share of
        the allotment to the pages they hold. The allotment covers the prompt, and the pages a window
        released while the prompt filled stayed in it for the prompt's later steps; past the prompt a
        window's pages only cycle, so what it doesn't hold (or need for the layout's scratch) is room
        again. A growth loop that runs before the last span ends the prompt early here, as it does for
        the first group's allotment. Under the lock."""
        rid = segment.request_id
        layout = self._held_layout(rid, segment.label)
        if layout is None:
            return
        prompt = sum(span.length for span in layout.spans)
        written = stream.stored_len + stream.released
        overrides = self._overrides[rid]
        if not written - segment.span < prompt <= written or segment.label not in overrides.opened_labels(self._nodes):
            return
        caps = dict.fromkeys(overrides.loop_caps or (), 0)
        allotment = self._allotments[rid]
        excess = 0
        for g, group in self._windows:
            part = stream.group(g)
            kept = max(len(part.page_indices), -(-(part.stored_len + layout.scratch) // group.page_size))
            excess += max(0, layout.pages(caps, group.page_size, group.retention) - kept)
        allotment.pages = max(0, allotment.pages - excess)
        self._recount(rid)

    def _release_oldest_locked(
        self, rid: str, stream: CacheStream, num_tokens: int, protected_prefix: int, group: int = 0,
    ) -> int:
        """The page-floored front release shared by ``release_oldest`` and the
        commit-time retention, of the stream's pages in group ``group``, behind
        ``protected_prefix`` tokens. Under the lock."""
        page_size = self._groups[group].page_size
        part = stream.group(group)
        first = (protected_prefix + page_size - 1) // page_size
        releasable = part.stored_len // page_size - first
        k = min(num_tokens // page_size, releasable)
        if k <= 0:
            return 0
        freed = part.page_indices[first:first + k]
        del part.page_indices[first:first + k]
        part.stored_len -= k * page_size
        part.released += k * page_size
        part.generation += 1
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
            self._touch_plan_cache()
            self._touch_plan_cache(request_id)
            return self._release_oldest_locked(request_id, stream, num_tokens, stream.protected_prefix)

    # Eviction

    @property
    def supports_eviction(self):
        return self._cpu_pool is not None

    supports_drop = True

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

    def residency(self, rid: str) -> Residency:
        if rid in self._dropped:
            return Residency.DROPPED
        return Residency.ON_HOST if self.is_offloaded(rid) else Residency.RESIDENT

    def in_step(self, rid: str) -> bool:
        with self._lock:
            return any(stream.step_in_flight or stream.read_pending for stream in self._streams[rid].values())

    def check_drop(self, rid: str, tokens: int) -> str | None:
        if self._world_size > 1 and not self._counts:
            return (
                f"KV {self.name}: dropping {rid} on one of {self._world_size} ranks "
                "would leave them holding different pages"
            )
        residency = self.residency(rid)
        if residency is Residency.ON_HOST:
            return f"KV {self.name}: {rid} has a host copy, which a drop would leave behind"
        if residency is Residency.DROPPED:
            return f"KV {self.name}: {rid} is dropped already, and its rebuild hasn't landed"
        with self._lock:
            if rid in self._guaranteed or not self._movable(rid):
                return (
                    f"KV {self.name}: {rid} is guaranteed, has a step or a read in flight on its pages, or holds a "
                    "lease no walk's admit has taken over"
                )
            if self._rebuilt.get(rid, -1) >= self._written(rid):
                return f"KV {self.name}: {rid} has written nothing since its rebuild landed, which a drop would waste"
            if not self._overrides[rid].rebuildable:
                return f"KV {self.name}: {rid} can't be rebuilt on this worker"
            held = self._laid_out_tokens(rid)
            if any(layout.spans for layout in (self._overrides[rid].kv_layout or {}).values()) and held != tokens:
                return f"KV {self.name}: {rid}'s rebuild would write {tokens} tokens, but its label holds {held}"
        return None

    def _laid_out_tokens(self, rid: str) -> int:
        # `check_drop` allows at most one laid-out label
        laid_out = [label for label, layout in (self._overrides[rid].kv_layout or {}).items() if layout.spans]
        return sum(self._streams[rid][label].stored_len for label in laid_out)

    def drop(self, rid: str, tokens: int) -> int:
        """Take ``rid``'s KV off the device without a host copy, and return the
        number of pages freed. Pages the index also holds stay cached, where they
        can be evicted, and the rest go back to the free list."""
        with self._lock:
            refusal = self.check_drop(rid, tokens)
            if refusal is not None:
                raise RuntimeError(refusal)
            # under the lock, so no other step allocates between the two counts
            free, room = self._arena.num_free, self._room()
            # It returns with the larger of its prompt and its history (the tokens its KV held, released
            # ones included, and the sample its next step writes) and one page more, at most its footprint.
            # A request dropped partway through its prompt still writes the rest.
            streams = self._streams[rid].values()
            written = self._written(rid)
            history = -(-(written + 1) // self.config.tokens_per_page)
            allotment = self._allotments.get(rid)
            returns = max(
                0 if allotment is None else allotment.asks_again,
                sum(self._one_more(stream.groups, history, stream.footprint) for stream in streams),
            )
            held = self._laid_out_tokens(rid)
            self.reset_request(rid, free=True)
            self._dropped[rid] = Dropped(returns, held)
            self._vacate(rid, room)
            return self._arena.num_free - free

    def prepare_rebuild(self, rid: str, rebuild: Rebuild) -> None:
        """Lay ``rid``'s label out again to hold the slots its drop checked, which its rebuild writes: the label's
        spans, the last grown by the ids sampled since but the newest (`StreamLayout.rebuilt`). A keyed label
        extends its keys by those ids, as the decode steps that sampled them did.

        The growth loop's cap and the layout's ``max_growth`` drop by what the last span grew, so the
        footprint stays the one at intake. The decode step that samples an id keys it, so a label that keys
        generation also extends its chain by the newest id.
        """
        overrides = self._overrides[rid]
        laid_out = [label for label, layout in (overrides.kv_layout or {}).items() if layout.spans]
        if not laid_out:
            return
        (label,) = laid_out
        layout, held = overrides.kv_layout[label], self._dropped[rid].held
        rebuilt, grown = layout.rebuilt(held), held - sum(span.length for span in layout.spans)
        assert len(rebuild.sampled) - 1 == grown, (
            f"KV {self.name}: {rid}'s rebuild writes {len(rebuild.sampled) - 1} ids past its layout, but its drop "
            f"checked {grown} slots past it"
        )
        # hashed before the lock, as `resolve_cached_prefix` hashes outside it
        keys = self._rebuilt_keys(overrides, label, layout, rebuild.sampled[:-1])
        with self._lock:
            ((loop, slots),) = layout.growth
            self._overrides[rid] = replace(
                overrides,
                kv_layout={
                    **overrides.kv_layout,
                    label: replace(
                        rebuilt, max_growth=None if layout.max_growth is None else layout.max_growth - grown,
                    ),
                },
                loop_caps={**overrides.loop_caps, loop: overrides.loop_caps[loop] - grown // slots},
                **keys,
            )
            stream = self._streams[rid][label]
            stream.forget_chain()
            self._lay_out(rid, label, stream)
            self._seed_keys(rid, label, stream)
            if stream.chain is not None and (overrides.prefix_decode or {}).get(label):
                stream.chain.extend([rebuild.sampled[-1]], self.config.tokens_per_page)
            prefix_chain = stream.chain
        # `make_room` can measure this request's refusal again before its next `reserve` roots the chain
        if prefix_chain is not None and self._index is not None:
            self._root_chain(prefix_chain)

    def _rebuilt_keys(
        self, overrides: KVReqConfig, label: str, layout: StreamLayout, grown: Sequence[int],
    ) -> dict[str, dict]:
        """``label``'s keys and tail once its last span has grown by the ``grown`` ids, or nothing when the request
        keys nothing there. The page those ids leave partly filled is keyed as a prompt's last page."""
        keys = overrides.keys_of(label)
        if not keys:
            return {}
        page_size, root = self.config.tokens_per_page, overrides.prefix_root
        extended = PrefixChain.seed(
            keys, list((overrides.prefix_tail or {}).get(label) or ()), layout.spans, page_size, root=root,
        )
        extended.extend(list(grown), page_size)
        keys, tail = extended.keys[:extended.keyed_pages], extended.unkeyed
        if tail:
            keys.append(page_key(keys[-1] if keys else root, tail, extended.unkeyed_items))
        return {
            "prefix_keys": {**overrides.prefix_keys, label: keys},
            "prefix_tail": {**(overrides.prefix_tail or {}), label: tail},
        }

    def land_rebuild(self, rid: str) -> None:
        with self._lock:
            if self._dropped.pop(rid, None) is not None:
                self._rebuilt[rid] = self._written(rid)

    def _written(self, rid: str) -> int:
        """The tokens ``rid``'s longest stream has written, released ones included."""
        return max((stream.stored_len + stream.released for stream in self._streams[rid].values()), default=0)

    def offload(self, rid: str) -> int:
        """Move every stream of ``rid`` to host memory. Returns the device pages
        it made reusable: pages that reach the free list, and pages then held
        only by the index.

        A stream whose pages don't fit on the host keeps them, so a partial
        offload still frees whatever did fit.

        Device pages go back to the arena only once every stream has been
        copied: a step admitted before the claim can still run its fork copy,
        and that copy reads one of these streams.

        A victim `preempt` chose that doesn't move comes off its earmark with
        nothing freed.
        """
        if self._cpu_pool is None or rid not in self._streams:
            return 0
        claimed, read_futures = self._claim_for_offload(rid)
        if not claimed:
            with self._lock:
                self._settle(rid, 0)
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
            with self._lock:
                self._settle(rid, 0)

    def _offload_stream(
        self, rid: str, claim: ClaimedStream, rooted: list[bytes],
    ) -> bool:
        """False when the host can't hold ``claim``."""
        with self._lock:
            keyed = self._filed(rooted, claim.groups[0].page_indices)
            copies = self._cpu_pool.place(
                rid=rid, label=claim.label, groups=claim.groups, keys=rooted[:keyed],
                position=claim.position, protected_prefix=claim.protected_prefix,
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
                    groups=on_pages(stream.groups, stream.pages()),
                    position=stream.position,
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
        # its resident slot goes with the offload (`Engine.offload_request`)
        self._touch_plan_cache(rid, row=False)
        with self._lock:
            streams = self._streams.get(rid, {})
            for claim in moved:
                stream = streams.get(claim.label)
                if stream is None or any(
                    part.generation != held.generation for part, held in zip(stream.groups, claim.groups, strict=True)
                ):
                    # removed, or written to behind us. Releasing nothing here
                    # leaves every claim for `_abandon_claims` to undo.
                    return 0, set()
            # counted under the lock, since a probe on another thread can retain
            # index pages
            freed = sum(self._reusable(page) for claim in moved for page in held_pages(claim.groups))
            # a victim still moves when it shares its pages with another victim, or frees only its allotment
            if not freed and rid not in self._hosted:
                return 0, set()
            room = self._room()
            for claim in moved:
                stream = streams[claim.label]
                # the offload takes its pages off the allotment too, so every page it frees is room
                self._release(rid, stream, held_pages(claim.groups), refund=False)
                # Offload changes residency, not logical stream contents.
                stream.reset(freed=True, content_reset=False)
                if claim.keys:
                    self._cpu_pool.insert_keys(rid, claim.label)
            # a stream the host couldn't take keeps its pages, and so its allotment
            if rid in self._hosted and self._unwritten(rid):
                self._vacate(rid, room)
            else:
                self._recount(rid)
                self._settle(rid, self._room() - room, self._allotments.get(rid))
            return freed, {claim.label for claim in moved}

    def _reusable(self, page: int) -> bool:
        """Whether ``page`` reaches the free list, or is left to the index alone,
        once the request releases it."""
        owners = self._arena.owners(page)
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

    def can_offload(self, rid: str) -> bool:
        """False for a guaranteed request, which reserves what it grows into so that nothing preempts it."""
        return rid not in self._guaranteed

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
            needed += len(state.host_pages) - len(run) + sum(page in evictable for page in run)
        return needed

    def _return_pages(self, rid: str) -> int:
        """Pages a request asking to return from the host may take beyond its reload: the larger of its prompt and
        its reload plus a page per stream (`_one_more`, as a drop counts a history), less the reload."""
        reloaded = grown = 0
        for label, state in self._cpu_pool.offloaded[rid].items():
            reloaded += len(state.host_pages)
            first = len(state.groups[0].page_indices)
            grown += self._one_more(state.groups, first, self._streams[rid][label].footprint)
        return max(self._asks[rid], grown, reloaded) - reloaded

    @staticmethod
    def _one_more(groups: Sequence[GroupPages], first: int, footprint: int | None) -> int:
        """``first`` pages in the first of ``groups`` and one page more, at most ``footprint``, and the pages the later
        groups hold, which have no footprint of their own."""
        later = sum(len(part.page_indices) for part in groups[1:])
        return later + (first + 1 if footprint is None else min(first + 1, footprint))

    def reload(self, rid: str) -> bool:
        """Bring every offloaded stream of ``rid`` back on device.

        Keyed pages the index still holds are taken from it, and the rest come
        from the host. False when the device can't fit them right now; nothing
        moves in that case, so the caller can evict further and try again.
        A request that asks again after its move to the host reloads only once admitted.
        """
        # an offload gave its resident slot up, so its next step writes the row whole
        self._touch_plan_cache(rid, row=False)
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
            needed = sum(len(state.host_pages) - len(runs[label]) for label, state in states.items())
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
                host = state.host_pages
                # the run the index holds stands in for the first group's leading pages
                stream.restore(on_pages(state.groups, run + pages[at:at + len(host) - len(run)]))
                if allotment is not None and allotment.returning is None:
                    # the offload took them off the allotment
                    allotment.pages += len(host)
                stream.position = state.position
                stream.protected_prefix = state.protected_prefix
                copies.cpu.extend(host[len(run):])
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
                self._cpu_pool.discard(rid, label)
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
        of them, when it is offloaded already, or when it is admitted but hasn't run."""
        with self._lock:
            streams = self._streams.get(rid, {})
            for stream in streams.values():
                if any(self._reusable(page) for page in reversed(stream.page_indices)):
                    return sum(len(stream.pages()) for stream in streams.values())
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
                pages = stream.pages()
                drop = free or self._arena.any_sealed(pages)
                if drop:
                    # before the reset, which clears the lists ``pages`` may alias
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
            self._stalled.discard(rid)
            if rid in self._streams:
                for stream in self._streams[rid].values():
                    self._release_lease(rid, stream)
                    self._release_hold(rid, stream)
                    self._arena.release(stream.pages())
                    stream.free_lists()
            if self._cpu_pool is not None:
                self._cpu_pool.remove_request(rid)
            # handle-keyed, and this releases the handle: a survivor would be
            # unwound against whichever request is given this handle next
            self._forget_reservations(rid)
            self._streams.pop(rid, None)
            self._looked_up.discard(rid)
            self._overrides.pop(rid, None)
            self._guaranteed.discard(rid)
            self._footprints.pop(rid, None)
            self._dropped.pop(rid, None)
            self._rebuilt.pop(rid, None)
            self._chain_lookup.pop(rid, None)
            self._transfer.remove_request(rid)
            self._settle(rid, self._room() - room, allotment)
            self._forget_earmark(rid)
            self._ran_short.pop(rid, None)
            self._passers.discard(rid)
            if _DEBUG_ASSERTS:
                self.assert_pages_conserved()

    def _report_underuse(self, rid: str) -> None:
        overrides = self._overrides[rid]
        for label, layout in (overrides.kv_layout or {}).items():
            stream = self._streams[rid].get(label)
            footprint = layout.pages(overrides.loop_caps, self.config.tokens_per_page)
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
                    for part, group in zip(stream.groups, self._groups, strict=True):
                        for page in part.page_indices:
                            refs[page] = refs.get(page, 0) + 1
                        full = part.stored_len // group.page_size
                        frontier.update(part.page_indices[full:])
                    if stream.lease is not None:
                        # held for a stream `admit` has not converted yet
                        for page in stream.lease:
                            refs[page] = refs.get(page, 0) + 1
                    for page in stream.hold or []:
                        refs[page] = refs.get(page, 0) + 1
            copying = Counter(page for copy in self._host_copies for page in copy.pages)
            for page, n in copying.items():
                refs[page] = refs.get(page, 0) + n

            counts = dict(zip(owned, arena.owners_of(owned), strict=True))
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
                if counts[page] - copying[page] > 1 and not arena.is_sealed(page)
            ]
            assert not unsealed, f"pages shared before they were sealed: {unsealed}"

            crowded = sorted(
                page for page in frontier if arena.owners(page) != 1
            )
            assert not crowded, (
                f"pages still being written into, but not owned alone: {crowded}"
            )

            unused = 0
            for rid, allotment in self._allotments.items():
                held = 0
                for label, stream in self._streams[rid].items():
                    pages = [*stream.pages(), *(stream.lease or ()), *(stream.hold or ())]
                    stale = stream.lent - set(pages)
                    assert not stale, f"{rid}/{label} counts pages it no longer holds as lent: {sorted(stale)}"
                    held += len(pages) - len(stream.lent)
                unused += max(0, allotment.pages - held)
            assert unused == self._unused, (
                f"the streams leave {unused} allotted pages unused, but the pool counts {self._unused}"
            )
            evictable = 0 if self._index is None else len(self._index.evictable())
            # A follower replays rank 0's moves as they arrive, and its own commit can give the room back later.
            assert not self._decides or self._unused <= arena.num_free + evictable, (
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
                        for page in (*stream.pages(), *(stream.lease or ()))
                    ]
                    assert not held, f"{rid} waits {'on the host' if on_host else 'dropped'}, holding pages {held}"
            for rid in self._victims:
                assert rid not in self._guaranteed, f"{rid} is guaranteed and was chosen as a victim"
            stalled = {rid for rid in self._asks if rid in self._allotments}
            assert stalled <= self._stalled <= self._allotments.keys(), (
                f"`_stalled` lists {sorted(self._stalled)}, but {sorted(stalled)} wait holding "
                f"and {sorted(self._allotments)} hold an allotment"
            )
            assert not self._kept or (self._keeper in self._asks and self._keeper not in self._passers), (
                f"`_kept` keeps {self._kept} pages for {self._keeper}, which doesn't wait in order, so they can stay "
                "kept with no request to take them"
            )
            if self._cpu_pool is not None:
                self._assert_host_pages_conserved()

    @staticmethod
    def _owned_pages(arena: PageArena, pool: str) -> list[int]:
        """The pages ``arena`` holds, checked against its free list."""
        total = arena.max_num_pages
        free = arena.free_pages()
        owned = [page for page, owners in enumerate(arena.owners_of(range(total))) if owners > 0]
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
        pool.index.assert_evictable()
        refs = Counter(pool.index.pages())
        for streams in pool.offloaded.values():
            for state in streams.values():
                refs.update(state.host_pages)
        for copy in self._host_copies:
            refs.update(copy.host_pages)
        if self._host_writes is not None:
            refs.update(self._host_writes.host_pages_held())
        counts = dict(zip(owned, host.owners_of(owned), strict=True))
        assert counts == dict(refs), (
            "host owner counts disagree with the index and requests naming the pages: "
            + ", ".join(
                f"page {page} owned {counts.get(page, 0)}, named {refs.get(page, 0)}"
                for page in sorted(set(counts) | set(refs))
                if counts.get(page, 0) != refs.get(page, 0)
            )
        )
        unmoved = sum(pool._unmoved_by(rid) for rid in pool.reserved)
        assert pool._unmoved == unmoved, f"the host counts {pool._unmoved} pages set aside, but requests hold {unmoved}"

    def post_warmup_validate(self):
        """Assert ``num_free_pages`` is identical across every rank of the
        instance

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
                f"across the ranks of its instance: min {lo}, max {-neg_hi}. v1 requires symmetric "
                "allocator state; check the YAML for per-rank-divergent "
                "max_num_pages / cpu_offload_pages, and any model code "
                "that calls add_request before warmup completes."
            )

    def cleanup(self):
        self._transfer.cleanup()

    def _ensure_label(self, rid: str, label: str) -> CacheStream:
            if label not in self._streams[rid]:
                stream = self._streams[rid][label] = self._new_stream()
                self._lay_out(rid, label, stream)
                self._seed_keys(rid, label, stream)
            return self._streams[rid][label]

    def _new_stream(self) -> CacheStream:
        new = self._arena.new_stream
        return CacheStream(page_indices=new(), later=tuple(GroupPages(new()) for _ in self._windows))

    def _covers(self, stream: CacheStream, span: int) -> bool:
        """Whether the stream's pages in every group past the first already hold ``span`` more
        tokens: `admit` checks the first group's itself, as every pool has one."""
        for g in range(1, len(self._groups)):
            part = stream.later[g - 1]
            if len(part.page_indices) * self._groups[g].page_size < span + part.stored_len:
                return False
        return True

    # Admission

    def refuse_rebuild(self, rid: str) -> None:
        """Mark ``rid`` as a request no drop here can free, as rank 0 found when it took ``rid`` in."""
        with self._lock:
            assert rid not in self._asks and rid not in self._allotments, (
                f"KV {self.name}: rank 0 refused {rid} a rebuild after this rank counted its pages as rebuildable"
            )
            self._overrides[rid] = replace(self._overrides[rid], rebuildable=False)

    def ask(self, rid: str, node_name: str, graph_walk: str) -> AdmitOutcome | None:
        """`ASKS` until ``rid`` has an allotment, or while it waits on a shortage `preempt` asked for, or why
        it can never be admitted.

        A request that can't fit even in an empty pool fails here, and not after
        waiting behind every request ahead of it.
        """
        if not self._decides or (rid in self._allotments and rid not in self._asks):
            return None
        with self._lock:
            failed = self._ask(rid)
        return ASKS if failed is None else failed

    def _ask(self, rid: str) -> AdmitOutcome | None:
        """Count the pages ``rid`` asks for into `_asks`, deciding at its first ask whether it is guaranteed, or
        return why it can never be admitted. Caller holds the lock."""
        if rid in self._asks:
            return None
        overrides = self._overrides[rid]
        prompt = overrides.pages(self._nodes, dict.fromkeys(overrides.loop_caps or (), 0), self.config)
        # A request that no drop here can free, and that no host can take, would wait holding its pages once
        # it ran short, so it reserves what it grows into.
        if overrides.unpausable or not (overrides.rebuildable or self._cpu_pool is not None):
            self._guaranteed.add(rid)
        pages = self._footprint(rid) if rid in self._guaranteed else prompt
        capacity = self.config.request_pages
        if pages > capacity:
            return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                f"KV {self.name}: request {rid} needs {self.config.pages_of(pages, round_up=True)} pages "
                f"to be admitted, and the pool has {self.config.pages_of(capacity)} for requests"
            ))
        self._asks[rid] = pages
        return None

    def _sets_aside(self, rid: str) -> bool:
        """Whether the host has the room ``rid`` needs to be admitted. A request that can move only there, and has
        neither an allotment nor room set aside, gets room for its prompt set aside now."""
        return (
            rid in self._allotments or self._overrides[rid].rebuildable or rid in self._guaranteed
            or rid in self._cpu_pool.reserved or self._cpu_pool.reserve(rid, self._asks[rid])
        )

    def _footprint(self, rid: str) -> int:
        """The pages ``rid`` would hold once the loops this worker runs reach their caps."""
        if rid not in self._footprints:
            overrides = self._overrides[rid]
            caps = {loop: cap if loop in self.local_loops else 0 for loop, cap in (overrides.loop_caps or {}).items()}
            self._footprints[rid] = overrides.pages(self._nodes, caps, self.config)
        return self._footprints[rid]

    def reserve(
        self, rid: str, node_name: str, graph_walk: str, published: PublishedKVInfo | None,
    ) -> RoomMark | None:
        """Allot ``rid`` its prompt, less the cache hits another request holds, or what it returns with
        from a drop or the host, or grow its allotment by the shortage it waits on."""
        with self._lock:
            if self._host_copies:
                self._land_host_copies()
            chains = [] if self._index is None or self._on_host(rid) else [
                stream.chain for stream in self._streams[rid].values()
                if stream.chain is not None and stream.chain.rooted is None
            ]
        # Requests ask and are removed on one thread only, so ``rid`` still asks once the lock is
        # taken again.
        for prefix_chain in chains:
            self._root_chain(prefix_chain)
        with self._lock:
            if rid not in self._asks:
                # the moves `make_room` chose for ``rid`` admitted it (`_settle`)
                self._stalled.discard(rid)
                return None
            # read before `_need`, so that a snapshot filed between the two moves the counts past the mark
            counts = self._index_counts()
            pages, room = self._need(rid, node_name, graph_walk, published), self._room_for(rid)
            if pages > room or not self._admit(rid, pages, passed=rid in self._passers):
                if self._fixed_need(rid) is not None:
                    counts = None, None
                if self._keeper not in self._asks and rid not in self._passers:
                    self._keeper = rid
                return RoomMark(rid, node_name, graph_walk, published, room, *counts)
            return None

    def reserve_past(
        self, rid: str, node_name: str, graph_walk: str, published: PublishedKVInfo | None,
    ) -> PassMark | None:
        """Allot ``rid`` past the first request in the queue, when what its own admission would take and the
        growth that admission leaves to the room fit in the room less `_kept`. A request that holds an allotment
        waits for its shortage in order."""
        with self._lock:
            room = self._room() - self._kept
            # When even the fewest pages ``rid`` can take don't fit, no page the index files helps, so only the room
            # moves the mark.
            least = (self._fixed_need(rid) or 0) + self._footprint(rid) - self._asks[rid]
            if rid in self._allotments or least > room:
                return PassMark(room, None, None)
            chains = [] if self._index is None or self._on_host(rid) else [
                stream.chain for stream in self._streams[rid].values()
                if stream.chain is not None and stream.chain.rooted is None
            ]
        for prefix_chain in chains:
            self._root_chain(prefix_chain)
        with self._lock:
            # before `_need`, as in `reserve`
            counts = self._index_counts()
            pages = self._need(rid, node_name, graph_walk, published) + self._footprint(rid) - self._asks[rid]
            room = self._room() - self._kept
            if pages > room:
                return PassMark(room, *counts)
            # The room kept for the head is never lent. A passer that runs short past this allotment is the newest
            # request holding pages, so preemption takes it first.
            if not self._admit(rid, pages, passed=True):
                return PassMark(room, None, None)
            return None

    def _admit(self, rid: str, pages: int, passed: bool = False) -> bool:
        """Allot ``rid`` the ``pages`` it asks for, which the room covers, or return False when ``rid`` turns
        guaranteed here and the room doesn't cover its footprint. Rank 0 journals the flip and the reservation for
        the other ranks to replay, and ``passed`` is set for a request admitted past the first one in the queue,
        whose room leaves out `_kept`. Caller holds the lock."""
        # The host sets room aside for the prompt of a request that can move only there, and `_cover_move`
        # grows it page by page. With no room for even that, the request reserves what it grows into.
        if not self._sets_aside(rid):
            pages = self._footprint(rid)
            self._flip(rid, pages)
            # the other ranks replay the flip, since it can come passes before the reservation they replay
            if self.journal is not None:
                self.journal(MoveKind.GUARANTEE, rid, pages)
            if pages > self._room() - (self._kept if passed else 0):
                return False
        self._reserve(rid, pages, passed)
        if self.journal is not None:
            self.journal(MoveKind.RESERVE, rid, pages)
        return True

    def _reserve(self, rid: str, pages: int, passed: bool = False) -> None:
        """Allot ``rid`` ``pages``, or grow the allotment it holds by them, so the moves still to land for an
        earmark ``rid`` holds add nothing. A request on the host is allotted its reload as well, which then leaves
        `_return_pages` on top of what it holds. A request admitted in order takes `_kept` first, and `_keeper` takes
        all of it."""
        kept = self._kept if rid == self._keeper else 0 if passed else min(self._kept, pages)
        self._kept -= kept
        if rid in self._allotments:
            del self._asks[rid]
            self._stalled.discard(rid)
            allotment = self._allotments[rid]
            allotment.kept = kept
            allotment.grown, allotment.grown_from = pages, allotment.pages
            self._allot(rid, pages)
            self._made = None
        else:
            returning = self._return_pages(rid) if self._on_host(rid) else None
            self._allotments[rid] = Allotment(
                pages=pages, asks_again=self._asks.pop(rid), returning=returning, passed=passed, kept=kept,
            )
            self._recount(rid)
            self._made = rid
        self._passers.discard(rid)
        earmark = self._earmarks.get(rid)
        if earmark is not None:
            earmark.granted, earmark.mark = earmark.pages, None

    def unreserve(self, rid: str) -> None:
        """`Resource.unreserve`. When moves that `make_room` chose for ``rid`` admitted it (`_settle`), `reserve`
        took nothing, but this still gives back the allotment those moves made. ``rid`` then asks for that room again,
        as any other request does."""
        with self._lock:
            made = self._made == rid
            allotment = self._allotments[rid]
            # the room kept for a waiting request goes back to it, or to ``rid`` once that request doesn't wait
            self._kept += allotment.kept
            if allotment.kept and self._keeper not in self._asks:
                self._keeper = rid
            if allotment.grown:
                # it waits holding its allotment again, for the same shortage
                allotment.pages = allotment.grown_from
                self._asks[rid], allotment.grown, allotment.kept = allotment.grown, 0, 0
                self._stalled.add(rid)
                self._recount(rid)
            else:
                self._unreserve(rid)
            if self.journal is not None:
                # 1 when the last reservation made the allotment, which `Engine._journal` then cancels
                self.journal(MoveKind.UNRESERVE, rid, int(made))

    def _unreserve(self, rid: str) -> None:
        allotment = self._allotments.pop(rid, None)
        if allotment is not None:
            self._unused -= allotment.remaining
            self._asks[rid] = allotment.asks_again
        self._stalled.discard(rid)
        # a request that holds nothing waits with no room set aside on the host
        if self._cpu_pool is not None and self._unwritten(rid) and not self._on_host(rid):
            self._cpu_pool.unreserve(rid)

    def refuse_drops(self, rid: str) -> None:
        """Mark ``rid``, admitted as a request a drop here can free, as one no drop can free from now on. As for a
        request marked so at its arrival, the host sets aside room for the pages it holds, or it turns guaranteed
        when the host has none (`_cover_move`)."""
        assert self._world_size == 1, (
            f"KV {self.name}: {rid} lost its log above one rank, where no move tells the other ranks and no node keeps "
            "a log of a step that read more than its ids"
        )
        with self._lock:
            self._overrides[rid] = replace(self._overrides[rid], rebuildable=False)
            if rid in self._allotments:
                self._cover_move(rid, 0, undoable=False)

    def _guarantee(self, rid: str) -> None:
        """Make ``rid`` guaranteed, since no host has room to set aside for its move."""
        self._guaranteed.add(rid)
        if self._cpu_pool is not None:
            self._cpu_pool.unreserve(rid)
            if self._decides:
                self._cpu_pool.count_guaranteed(rid)

    def _flip(self, rid: str, pages: int) -> None:
        """Guarantee ``rid``, which waits for room, so that it asks for its footprint of ``pages``."""
        self._guarantee(rid)
        self._asks[rid] = pages

    def _cover_move(self, rid: str, n: int, undoable: bool) -> int:
        """Grow the room the host sets aside for ``rid``, a request no drop here can free, by the ``n`` pages it is
        about to take, or guarantee ``rid`` when the host has no room for them. Returns the pages its guarantee is
        short of, before which ``rid`` takes nothing.

        ``undoable`` changes neither, as `_grant` with it evicts nothing, and returns ``n`` when one was due."""
        if not self._counts or self._overrides[rid].rebuildable or rid in self._guaranteed:
            return 0
        # a lease, and the pages held for a copy from the host, become a stream's pages without a grant
        held = sum(
            len(stream.pages()) + len(stream.lease or ()) + len(stream.hold or ())
            for stream in self._streams[rid].values()
        )
        reserved = None if self._cpu_pool is None else self._cpu_pool.reserved.get(rid)
        if reserved is not None and held + n <= reserved:
            return 0
        if undoable:
            return n
        if self._cpu_pool is not None and self._cpu_pool.reserve(rid, held + n):
            return 0
        allotment = self._allotments[rid]
        # a lease's lent pages stay in a guaranteed allotment until `_adopt` takes them off
        grows = max(0, self._footprint(rid) - held) + sum(
            len(stream.lent.intersection(stream.lease or ())) for stream in self._streams[rid].values()
        )
        short = grows - allotment.remaining - self._room()
        if short > 0:
            return short
        self._guarantee(rid)
        allotment.pages = self._held(rid) + grows
        self._recount(rid)
        return 0

    def grew_since(self, mark: RoomMark | PassMark) -> bool:
        """A newly filed page can raise a waiting request's hit, and a hit page another request
        takes from the evictable set comes out of that request's allotment, not this one's. Neither
        moves the room, so the index counts both. The counts only rise, so two changes between passes
        can't cancel out. A request that moves admitted (`_settle`) asks for nothing more."""
        if not _DEBUG_ASSERTS:
            return self._grew_since(mark)
        # under the lock, so a page freed on the GPU thread between the two reads can't trip it
        with self._lock:
            grew = self._grew_since(mark)
            if not grew and isinstance(mark, RoomMark) and mark.rid in self._asks:
                pages = self._need(mark.rid, mark.node_name, mark.graph_walk, mark.published)
                assert pages > self._room_for(mark.rid), (
                    f"KV {self.name}: {mark.rid} fits now, but nothing `grew_since` reads has moved "
                    "since it didn't, so nothing would admit it"
                )
            return grew

    def _grew_since(self, mark: RoomMark | PassMark) -> bool:
        if isinstance(mark, PassMark):
            # pages kept for a waiting request are no room for a passer
            room = self._room() - self._kept
        elif mark.rid not in self._asks:
            return True
        else:
            room = self._room_for(mark.rid)
        return room > mark.room or (mark.filed is not None and (mark.filed, mark.taken) != self._index_counts())

    def _room_for(self, rid: str) -> int:
        """The room waiting ``rid`` may take, which leaves out `_kept` when ``rid`` returns past the first request."""
        return self._room() - (self._kept if rid in self._passers else 0)

    def _need(
        self, rid: str, node_name: str, graph_walk: str, published: PublishedKVInfo | None,
    ) -> int:
        """The pages ``rid``'s allotment would take now. Caller holds the lock. No hit covers the
        shortage a request that holds an allotment asks for."""
        fixed = self._fixed_need(rid)
        if fixed is not None:
            return fixed
        dropped = self._dropped.get(rid)
        if dropped is not None:
            return dropped.returns - self._lent_hits(rid, dropped)
        if self._on_host(rid):
            return self._reload_pages(rid) + self._return_pages(rid)
        whole, charged = self._hit(rid, node_name, graph_walk, published)
        return self._asks[rid] - whole + charged

    def _fixed_need(self, rid: str) -> int | None:
        """`_need` for a request whose need no page the index files can lower, or None."""
        if rid in self._allotments:
            return self._asks[rid]
        if rid in self._guaranteed and not self._on_host(rid):
            # A guaranteed request takes no cache hit off its allotment, since the request holding a hit page can
            # release it and the index evict it before the lease.
            return self._asks[rid]
        return None

    def _on_host(self, rid: str) -> bool:
        return self._cpu_pool is not None and self._cpu_pool.is_offloaded(rid)

    def _lent_hits(self, rid: str, dropped: Dropped) -> int:
        """The pages of dropped ``rid``'s prompt that another request holds, up to the lease its rebuild's
        probe can take (`_leased`). They can't be evicted, so the probe takes them for nothing, as a
        prompt's does (`_hit`)."""
        if self._index is None:
            return 0
        pages = max(dropped.held - 1, 0) // self.config.tokens_per_page
        evictable = self._index.evictable()
        return sum(
            page not in evictable
            for stream in self._streams[rid].values() if stream.chain is not None
            for page in self._index.pages_for(stream.chain.rooted[:min(stream.chain.keyed_pages, pages)])
        )

    def _index_counts(self) -> tuple[int, int]:
        """The pages the index has ever filed with the snapshots ever filed under its keys, and the pages
        requests have ever taken out of its evictable set."""
        if self._index is None:
            return 0, 0
        return self._index.filed + sum(pool.filed for pool in self._snapshots), self._index.taken

    # Preemption

    def preempt(
        self, rid: str, pages_short: int, newer: Collection[str] | None, droppable: Callable[[str], bool],
    ) -> Victims | None:
        """Choose requests from ``newer``, the requests that arrived after ``rid``, newest first, to move off the
        device for ``rid``'s shortage of ``pages_short`` pages, and earmark what they free for ``rid``. The room
        its grant counted goes into its allotment with them, so its next grant can't fail. A victim the host can
        take whole moves there, and any other is dropped, once ``droppable`` says its launched steps allow it.

        Returns the victims, or none once the room covers its grant. When no move covers the shortage, ``rid`` asks for
        every page its grant needed past its allotment. It moves itself to the host when the host takes it whole, as
        main's offload-and-hold does, and returns once its history and those pages fit. Otherwise it keeps its
        allotment and waits for them once the runner asks again.
        None when the pool doesn't decide or ``newer`` is None, as it is for a request the conductor gave no number.
        """
        if not self._decides or newer is None:
            return None
        with self._lock:
            room = self._room()
            short = self._ran_short.pop(rid, ShortGrant(room, pages_short))
            # the room the grant counted when it ran short, as the room has grown or shrunk since
            counted = short.pages - pages_short + room - short.room
            claim = max(0, min(short.pages, counted, room))
            pages_short = short.pages - claim
            if pages_short <= 0:
                self._claim(rid, claim)
                return Victims([], [], [])
            victims = self._choose(pages_short, newer, droppable)
            if victims is None:
                self._wait(rid, short.pages)
                if self.journal is not None:
                    self.journal(MoveKind.WAIT, rid, short.pages)
                pages = self._host_pages(rid)
                if pages is not None and pages <= self._cpu_pool.room() and self._movable(rid) and self._frees([rid]):
                    self._host(rid, [rid])
                    return Victims([rid], [], [])
                return Victims([], [], [], waits=short.pages)
            self._claim(rid, claim)
            return self._earmark(Earmark(rid, pages_short, set()), victims)

    def _wait(self, rid: str, pages: int) -> None:
        """Make ``rid``, which holds its allotment, ask for the ``pages`` its grant needed past it. Any move of ``rid``
        from here on returns it with them, since back with a page alone it would run short again at once. A return
        counts every page its streams hold, lent ones included."""
        allotment = self._allotments[rid]
        lent = sum(len(stream.lent) for stream in self._streams[rid].values())
        allotment.asks_again = max(allotment.asks_again, max(allotment.pages, self._held(rid)) + lent + pages)
        self._asks[rid] = pages
        self._stalled.add(rid)

    def _claim(self, rid: str, pages: int) -> None:
        """Add the ``pages`` of room a short grant counted to ``rid``'s allotment."""
        # `_allot` of none would still raise the allotment to what ``rid`` holds, with no move for followers to replay
        if not pages:
            return
        self._allot(rid, pages)
        if self.journal is not None:
            self.journal(MoveKind.ALLOT, rid, pages)

    def any_stalled(self) -> bool:
        """Whether a request waits holding its allotment in a pool that decides, as `make_room` needs. Read without
        the lock."""
        return self._decides and bool(self._stalled)

    def stalled_moves(self, droppable: Callable[[str], bool]) -> StalledMoves:
        """What `make_room` can move. It chooses no differently for a refusal whose mark is the same until this
        changes."""
        with self._lock:
            host = 0 if self._cpu_pool is None else self._cpu_pool.room()
            rids = set()
            for rid in self._stalled:
                if not (self._holds(rid) and self._movable(rid)):
                    continue
                pages = self._host_pages(rid)
                if self._unwritten(rid) or (pages is not None and pages <= host) or droppable(rid):
                    rids.add(rid)
            return StalledMoves(frozenset(rids), host, self._frees(rids))

    def make_room(self, mark: RoomMark, newer: Collection[str], droppable: Callable[[str], bool]) -> Victims | None:
        """Choose requests from ``newer`` that wait holding pages, newest first, to move off the device for
        ``mark.rid``, a waiting request the room doesn't fit, and earmark what they free for it. It takes that room
        as their moves land, once the room covers all it asks for. No victims once the room fits ``mark.rid``, and
        None when the pool doesn't decide, ``mark.rid`` no longer asks, or no such moves cover it."""
        if not self._decides:
            return None
        with self._lock:
            stalled = [other for other in newer if other in self._stalled]
            if not stalled or mark.rid not in self._asks:
                return None
            pages = self._need(mark.rid, mark.node_name, mark.graph_walk, mark.published)
            room = self._room_for(mark.rid)
            if pages <= room:
                return Victims([], [], [])
            victims = self._choose(pages - room, stalled, droppable)
            return None if victims is None else self._earmark(Earmark(mark.rid, pages, set(), mark=mark), victims)

    def _choose(self, pages_short: int, newer: Collection[str], droppable: Callable[[str], bool]) -> Victims | None:
        """The fewest leading requests of ``newer`` whose moves free ``pages_short`` pages, or None when moving every
        one a move can take frees less. Caller holds the lock."""
        left = 0 if self._cpu_pool is None else self._cpu_pool.room()
        hosted, asking = set(), set()
        victims: list[str] = []
        for other in newer:
            if not self._holds(other):
                continue
            if not self._movable(other):
                continue
            if self._unwritten(other):
                asking.add(other)
            elif (pages := self._host_pages(other)) is not None and pages <= left:
                left -= pages
                hosted.add(other)
            elif not droppable(other):
                continue
            victims.append(other)
            if self._frees(victims) >= pages_short:
                break
        else:
            return None
        freed = self._frees(victims)
        for victim in list(victims):
            rest = [other for other in victims if other != victim]
            if self._frees(rest) == freed:
                victims = rest
        return Victims(
            [victim for victim in victims if victim in hosted],
            [victim for victim in victims if victim not in hosted and victim not in asking],
            [victim for victim in victims if victim in asking],
        )

    def _earmark(self, earmark: Earmark, victims: Victims) -> Victims:
        """Set aside what ``victims`` free for ``earmark.rid``. Caller holds the lock."""
        moving = [*victims.to_host, *victims.to_drop, *victims.to_ask]
        if self.journal is not None:
            # A waiting request takes the room whole once it covers all it asks for (`_settle`), so the other ranks
            # replay its earmark with what it asks for, and then rank 0's reservation.
            kind = MoveKind.RESERVE if earmark.rid in self._asks else MoveKind.EARMARK
            self.journal(kind, earmark.rid, earmark.pages, moving)
        self._mark_victims(earmark, moving)
        self._host(earmark.rid, victims.to_host)
        return victims

    def _mark_victims(self, earmark: Earmark, victims: list[str]) -> None:
        """Earmark what ``victims`` free for ``earmark.rid``. A victim that holds no page of its own gives back its
        allotment at once and asks again."""
        earmark.victims = set(victims)
        for victim in victims:
            self._victims[victim] = earmark
        previous = self._earmarks.get(earmark.rid)
        if previous is not None and earmark.mark is not None:
            # what `_kept` already keeps for the request counts toward the new earmark too
            earmark.granted = previous.granted
        self._earmarks[earmark.rid] = earmark
        for victim in victims:
            if self._unwritten(victim):
                self._vacate(victim, self._room())

    def _host(self, rid: str, victims: list[str]) -> None:
        self._hosted.update(victims)
        if self.journal is not None and victims:
            self.journal(MoveKind.OFFLOAD, rid, 0, victims)

    def moves_to_host(self, rid: str) -> bool:
        return rid in self._hosted

    def _forget_earmark(self, rid: str) -> None:
        """Take down ``rid``'s earmark, and the room `_kept` keeps for it, once ``rid`` is removed or gives back its
        allotment. A request that asks again then asks for what its move recorded, not the shortage the earmark
        holds. Caller holds the lock."""
        if rid == self._keeper:
            self._keeper, self._kept = None, 0
        earmark = self._earmarks.pop(rid, None)
        if earmark is not None:
            # its victims run on without being dropped
            for victim in earmark.victims:
                del self._victims[victim]
                self._hosted.discard(victim)

    def _holds(self, rid: str) -> bool:
        allotment = self._allotments.get(rid)
        return (allotment is not None and allotment.remaining > 0) or any(
            stream.pages() or stream.lease for stream in self._streams[rid].values()
        )

    def _movable(self, rid: str) -> bool:
        """Whether a move can take ``rid`` as soon as it is chosen: no step or read is in flight on its pages, it isn't
        dropped with a rebuild still to land, since the rebuild runs on its allotment, and it isn't guaranteed, unless
        it waits holding its allotment with no page written, when moving it only undoes its admission. ``rid``
        also stays while it holds a lease, until a walk's admit takes the lease over. The probe that took the lease
        moved the position counter, which a smaller hit after a move couldn't move back, and a walk the cache served
        keeps its KV only in the lease."""
        if rid in self._guaranteed and not (rid in self._stalled and self._unwritten(rid)):
            return False
        return rid not in self._dropped and not any(
            stream.step_in_flight or stream.read_pending or stream.lease is not None
            for stream in self._streams[rid].values()
        )

    def _unwritten(self, rid: str) -> bool:
        return not any(stream.pages() for stream in self._streams[rid].values())

    def _host_pages(self, rid: str) -> int | None:
        """The host pages that moving every stream of ``rid`` takes out of the host's room, past the pages the host
        set aside for it, or None when ``rid`` can't move there: the pool has no host memory, ``rid`` isn't resident
        or holds no page, or a stream holds a cached image end `plan` hasn't copied."""
        if self._cpu_pool is None or self.residency(rid) is not Residency.RESIDENT or self._unwritten(rid):
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
            pages += self._cpu_pool.takes(len(stream.pages()), rooted[:self._filed(rooted, stream.page_indices)])
        return max(0, pages - self._cpu_pool.reserved.get(rid, 0))

    def _frees(self, rids: Collection[str]) -> int:
        """The room moving ``rids`` off the device would make: the pages of their streams that would reach the free
        list or the evictable set, and their unused allotments."""
        released: Counter = Counter()
        for rid in rids:
            for stream in self._streams[rid].values():
                released.update(stream.pages())
        held = list(released)
        pages = sum(owners == released[page] for page, owners in zip(held, self._arena.owners_of(held), strict=True))
        if self._index is not None:
            pages += self._index.evictable_after(released)
        return pages + sum(self._allotments[rid].remaining for rid in rids if rid in self._allotments)

    def _settle(self, rid: str, freed: int, allotment: Allotment | None = None) -> None:
        """Take ``rid`` off the earmark it is a victim of, now that its move or removal made ``freed`` room, and
        grow the allotment of the request that ran short by that room, up to what its shortage still lacks. A
        victim that asks again at once is then measured against the room without it. A waiting request the moves
        were for takes the room as soon as it covers all it asks for, before a victim can take it back, and the moves
        that land after that add nothing. Until then, `_kept` keeps the room for it, up to what it asks for. It waits
        holding that room until it starts, so an older request can move it.

        When ``rid``'s ``allotment`` passed the first request in the queue, `_kept` also keeps what the earmark
        doesn't take for `_keeper`."""
        self._hosted.discard(rid)
        earmark = self._victims.pop(rid, None)
        if earmark is not None:
            earmark.victims.discard(rid)
            if not earmark.victims:
                del self._earmarks[earmark.rid]
            if earmark.rid in self._asks:
                # measured again, as a victim's move can make the pages a hit took for nothing evictable
                mark = earmark.mark
                pages = earmark.pages if mark is None else self._need(
                    mark.rid, mark.node_name, mark.graph_walk, mark.published,
                )
                passed = earmark.rid in self._passers
                kept = self._keep(earmark.rid, min(pages - earmark.granted, freed))
                earmark.granted += kept
                freed -= kept
                # Rank 0 alone admits it, since another rank can hold other room as the move lands there, as when a
                # window's commit frees pages at another point of the step. The engine journals the move ahead of the
                # admit.
                if self._decides and pages <= self._room_for(earmark.rid) and self._admit(earmark.rid, pages, passed):
                    self._stalled.add(earmark.rid)
            else:
                granted = min(earmark.pages - earmark.granted, freed)
                if granted:
                    earmark.granted += granted
                    self._allot(earmark.rid, granted)
                freed -= granted
        if allotment is not None and allotment.passed:
            self._keep(self._keeper, freed)

    def _keep(self, rid: str | None, pages: int) -> int:
        """Keep up to ``pages`` more of the room for waiting ``rid``, up to what it asks for, and return how many.
        ``rid`` takes over the room kept for another request. A request returning past the first one takes none of
        `_kept`, so none is kept for it."""
        if pages <= 0 or rid not in self._asks or rid in self._passers:
            return 0
        if rid != self._keeper:
            self._keeper, self._kept = rid, min(self._kept, self._asks[rid])
        kept = max(0, min(pages, self._asks[rid] - self._kept))
        self._kept += kept
        return kept

    def _vacate(self, rid: str, room: int) -> None:
        """Give up ``rid``'s allotment, so that ``rid`` asks again, once a drop or a move to the host has freed its
        pages. ``room`` is `_room` as it was before the drop or the move."""
        allotment = self._allotments.get(rid)
        self._unreserve(rid)
        if allotment is not None and allotment.passed:
            self._passers.add(rid)
        self._ran_short.pop(rid, None)
        self._settle(rid, self._room() - room, allotment)
        self._forget_earmark(rid)

    def _allot(self, rid: str, pages: int) -> None:
        # A request past its allotment holds more than ``allotment.pages``, so ``pages`` add to what it holds.
        allotment = self._allotments[rid]
        allotment.pages = max(allotment.pages, self._held(rid)) + pages
        self._recount(rid)

    def replay(self, kind: MoveKind, rid: str, count: int, victims: list[str]) -> None:
        """Make rank 0's ``kind`` move on ``rid`` in this rank's counts, or raise if the move doesn't fit them."""
        with self._lock:
            if kind in (MoveKind.EARMARK, MoveKind.RESERVE):
                # The same filter as rank 0's `_choose`, which can choose a dropped request again once it is
                # allotted pages.
                stale = [victim for victim in victims if victim not in self._streams or not self._holds(victim)]
                if stale or (kind == MoveKind.EARMARK and rid not in self._allotments):
                    raise RuntimeError(
                        f"KV {self.name}: rank 0 earmarked {count} pages for {rid} on victims {victims}, but here "
                        f"{stale} aren't live or hold nothing, or {rid} holds no allotment"
                    )
            if kind == MoveKind.EARMARK:
                self._mark_victims(Earmark(rid, count, set()), victims)
            elif kind == MoveKind.RESERVE:
                # a request that holds an allotment asks only for the shortage its WAIT set
                if rid not in self._allotments and (failed := self._ask(rid)) is not None:
                    raise RuntimeError(failed.reason.message)
                if victims:
                    # rank 0's `make_room`, whose reservation rank 0 journals once the moves of ``victims`` cover it
                    self._mark_victims(Earmark(rid, count, set()), victims)
                elif self._sets_aside(rid):
                    self._reserve(rid, count)
                else:
                    raise RuntimeError(
                        f"KV {self.name}: rank 0 reserved {rid} with host room set aside, and this rank's host has "
                        "no room for it"
                    )
            elif kind == MoveKind.OFFLOAD:
                stale = [victim for victim in victims if victim not in self._streams]
                if stale:
                    raise RuntimeError(f"KV {self.name}: rank 0 moves {victims} to the host, and {stale} left here")
                self._host(rid, victims)
            elif kind == MoveKind.UNRESERVE:
                if rid not in self._allotments:
                    raise RuntimeError(
                        f"KV {self.name}: rank 0 gave back {rid}'s allotment, which this rank doesn't hold"
                    )
                self.unreserve(rid)
            elif kind == MoveKind.GUARANTEE:
                self._flip(rid, count)
            elif kind == MoveKind.ALLOT:
                self._claim(rid, count)
            elif kind == MoveKind.WAIT:
                if rid not in self._allotments:
                    raise RuntimeError(
                        f"KV {self.name}: rank 0 made {rid} wait holding an allotment, which this rank doesn't hold"
                    )
                self._wait(rid, count)

    def _hit(
        self, rid: str, node_name: str, graph_walk: str, published: PublishedKVInfo | None,
    ) -> tuple[int, int]:
        """How many whole pages ``rid``'s streams will take from the index, and how many of the
        pages they take are evictable. A stream to be read in takes what `_take_local_match`
        would match, and a stream to be prefilled what the probe would, up to the newest
        recurrent snapshot under it.

        The engine doesn't probe a walk that writes two labels from one input, since a probe can cut
        only one label's input.
        """
        overrides = self._overrides[rid]
        if self._index is None or not overrides.prefix_cache:
            return 0, 0
        page_size = self.config.tokens_per_page
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
                for pool in self._snapshots:
                    pages = pages[:pool.restorable(chain.rooted[:len(pages)])]
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

    def _hit_allowance(self, rid: str) -> int:
        """The pages ``rid``'s cache hits on evictable pages may take: `_allowance`, or the room alone for a
        guaranteed request. Its allotment keeps every page the request's walks write until the request adopts its
        lease, since a cut leaves the walk to write the cut pages again."""
        return self._room() if rid in self._guaranteed else self._allowance(rid)

    def _retain_cached(self, rid: str, stream: CacheStream, pages: list[int]) -> list[int]:
        """Retain the longest prefix of ``pages`` that ``rid`` can take, and return it.

        A page that isn't evictable is lent and takes no room. An evictable one comes out
        of what ``rid`` was allotted and hasn't used, then out of the room, and the walk
        recomputes the pages neither covers.

        A guaranteed request's evictable pages come out of `_hit_allowance`, and every page is lent.
        """
        if self._decides:
            evictable = self._index.evictable()
            guaranteed = rid in self._guaranteed
            allowance = self._hit_allowance(rid)
            for at, page in enumerate(pages):
                if page in evictable:
                    if allowance <= 0:
                        pages = pages[:at]
                        break
                    allowance -= 1
                if guaranteed or page not in evictable:
                    stream.lent.add(page)
        self._arena.retain(pages)
        return pages

    def _release(self, rid: str, stream: CacheStream, pages: list[int], refund: bool = True) -> None:
        """Release ``pages`` of ``stream``, which the caller takes off the stream before it recounts.

        A page not in ``stream.lent`` shrinks the allotment by one when its release leaves
        the page neither free nor evictable, and always when ``refund`` is False. Otherwise
        the allotment keeps the page for the walk that writes it again. The caller recounts.
        """
        self._arena.release(pages)
        allotment = self._allotments.get(rid)
        if allotment is not None:
            evictable = () if self._index is None else self._index.evictable()
            gone = sum(
                1 for page, owners in zip(pages, self._arena.owners_of(pages), strict=True)
                if page not in stream.lent and (not refund or (owners and page not in evictable))
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
            len(stream.pages()) + len(stream.lease or ()) + len(stream.hold or ()) - len(stream.lent)
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
        to_label: str, extra: int = 0, realloc: bool = False, undoable: bool = False,
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
            return self._alloc(rid, to_label, extra, undoable=undoable)
        from_stream = self._streams[rid][from_label]
        if from_stream.offloaded:
            # the target is a fresh stream an offload never claimed, so refuse
            # here: `_apply_fork` would copy from a source whose pages are gone
            return AllocResult(success=False, error=self._offloading_message(rid, from_label))
        self._ensure_label(rid, to_label)
        return self._alloc(
            rid, to_label, from_stream.stored_len + extra, undoable=undoable,
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
            n = -(-from_stream.stored_len // self.config.tokens_per_page)
            assert len(to_stream.page_indices) >= n, (
                f"fork target {rid}/{to_label} holds "
                f"{len(to_stream.page_indices)} pages but its source "
                f"{from_label} needs {n}; _reserve_fork under-reserved"
            )
            fork = to_stream.page_indices[:n]
            shared = [page for page, owners in zip(fork, self._arena.owners_of(fork), strict=True) if owners > 1]
            assert not shared, (
                f"fork target {rid}/{to_label} would be written over pages "
                f"{shared}, which another owner still reads; a fork target "
                "cannot be a stream the index holds"
            )
            self.kv_cache.copy_pages(
                from_stream.page_indices[:n], to_stream.page_indices[:n],
            )
            to_stream.stored_len = from_stream.stored_len
            to_stream.generation += 1
            to_stream.reset_generation += 1

    def _alloc(
        self, request_id: str, label: str, seq_len: int, group: int = 0, undoable: bool = False,
    ) -> AllocResult:
        """Pages for ``seq_len`` tokens of the stream's pages in group ``group``."""
        with self._lock:
            self._ensure_label(request_id, label)
            stream = self._streams[request_id][label]
            if stream.offloaded:
                # an offload claimed this stream: its pages are on their way to
                # the host, and `reload` is the only path that may re-take them
                return AllocResult(
                    success=False, error=self._offloading_message(request_id, label)
                )
            page_size = self._groups[group].page_size
            stream = stream.group(group)
            num_pages_needed = (seq_len + page_size - 1) // page_size
            num_new_pages = num_pages_needed - stream.page_indices.n
            # a stream that already holds enough pages acquires none, and has
            # nothing for ``rollback_admit`` to give back
            new_pages: list[int] = []
            if num_new_pages > 0:
                short = self._cover_move(request_id, num_new_pages, undoable)
                allotment = self._allotments.get(request_id)
                within = 0 if allotment is None else min(num_new_pages, allotment.remaining)
                new_pages = None if short else self._grant(
                    request_id, label, num_new_pages, num_new_pages - within, undoable,
                )
                if new_pages is None:
                    available = self._allowance(request_id)
                    pages_short = short or num_new_pages - available
                    if self._decides and not undoable:
                        # The grant needed the room and `pages_short` past the allotment, which for a guarantee
                        # `_cover_move` found short is every page the request grows into. A refused pre-admit ran
                        # short of nothing, since exec admits the step again and may evict.
                        room = self._room()
                        self._ran_short[request_id] = ShortGrant(room, room + pages_short)
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
                # a later group's pages have no footprint of their own
                if not group and stream.footprint is not None and len(stream.page_indices) > stream.footprint:
                    self._report_overrun(request_id, label, stream)
                if _DEBUG_ASSERTS:
                    if not group:
                        stream.peak = max(stream.peak, len(stream.page_indices))
                    self.assert_pages_conserved()
        return AllocResult(new_pages=new_pages)

    def _grant(self, rid: str, label: str, n: int, past: int, undoable: bool = False) -> list[int] | None:
        """Take ``n`` free pages for ``rid``, of which ``past`` go beyond what its allotment has
        left, evicting cached pages if too few are free. None when the pool is short, which on a leading
        pool means ``past`` is more than the room. The pages within the allotment need
        no check, since the allotments never count more than the pool has.

        ``undoable`` evicts nothing, for a lockstep pool's pre-admit. Rank 0 can't put back a page it evicted for a
        stage it throws away, which a follower may never have planned."""
        if past and self._counts and past > self._room():
            return None
        pages = self._arena.acquire(n)
        if undoable:
            return pages
        if pages is None and self._index is not None:
            # cached pages are the only ones that can be given back
            # without failing a request that is already running
            self._evict(n - self._arena.num_free)
            pages = self._arena.acquire(n)
        if pages is None and self._counts:
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

    # Resident planning

    def max_repeats(self, step: KVStep, ctx: StepContext) -> int:
        """What the pool can hand over covers, one page a row at most: a refused
        admit would break the chain for pages a shorter run doesn't need. Past
        a row's reach (its pages, or ``max_seq_len``) there is no table row."""
        if ctx.resident is None:
            return 1
        step_rows = {(s.request_id, s.label) for s in step.segments if s.span and ctx.is_real_row(s.request_id)}
        with self._lock:
            # the room a refused admit would count: pages other requests' allotments hold aren't ours
            if self._room() < max(len(step_rows), 1):
                return 1
            longest = max((
                self._streams[rid][label].stored_len for rid, label in step_rows
                if label in self._streams.get(rid, ())
            ), default=0)
        page_size = self._groups[0].page_size
        reach = min(self.config.max_seq_len, self.config.request_pages * page_size)
        return max(1, min(page_size, reach - longest))

    def resident_refusal(self) -> str | None:
        if len(self._groups) > 1:
            return "a pool with sliding-window layers holds its pages per group, which a block table row can't"
        if self.kv_cache.layout != KVLayout.NHD:
            return f"the {self.kv_cache.layout.value} layout has no resident write path"
        if self._needs_remote_transfer:
            return "a pool read across nodes takes pages a resident row doesn't track"
        if self.placement is not None:
            return (
                "a context-parallel pool writes each token on one rank, at that rank's slot of the page "
                "(ContextParallel.write_slots), which the resident write path doesn't address"
            )
        return None

    def bind_resident(self, planner: "ResidentPlanner", key: str) -> None:
        from mstar.engine.resources.kv.resident import KVResidentTables

        page_size = self._groups[0].page_size
        # the pool's own context or override (not the serving config's top-level
        # `max_seq_len`), and a stream's pages are distinct pages of the pool
        width = min(-(-self.config.max_seq_len // page_size), self.config.request_pages)
        self._resident = KVResidentTables(planner, key, page_size, width)

    def resident_row(self, rid: int, slot: int, row: list[int], writes: "ResidentWrites") -> None:
        tables = self._resident
        label = writes.streams[tables.key].label
        stream = self._streams[rid][label]
        if stream.released:
            raise ValueError(
                f"KV {self.name}: {rid}/{label} released {stream.released} tokens, "
                "which a block table row can't address"
            )
        tables.row(stream.stored_len, stream.page_indices, slot, row, writes)

    def resident_stage(self, plan: "ResidentPlan") -> None:
        tables, streams = self._resident, self._streams
        stream = plan.writes.streams[tables.key]
        label, planner = stream.label, tables.planner
        if not stream.reads:
            for rid in plan.moved:
                tables.refresh_len(streams[rid][label].stored_len, planner.slot_of(rid), plan.writes)
        tables.stage_step(
            plan, lambda rid: streams[rid][label].stored_len, lambda rid: streams[rid][label].page_indices,
            planner.rewrite,
        )
        if _DEBUG_ASSERTS:
            for rid, slot in zip(plan.rids, plan.slots.tolist(), strict=True):
                held, stored = int(tables.host_len[slot]), streams[rid][label].stored_len
                assert held == stored, f"KV {self.name}: resident row of {rid} at {held}, its stream at {stored}"

    def resident_gather(self, step: "CapturedResidentStep") -> None:
        tables = self._resident
        stream = step.streams[tables.key]
        if stream.reads:
            tables.gather(step, None, None)
            return
        state = self._static_plan_state(step.slot, (stream.label, 0), step.bucket.num_tokens)
        tables.gather(step, state.token_to_page, state.token_to_cache)

    def resident_commit(self, step: "CapturedResidentStep") -> None:
        self._resident.commit(step)

    def resident_advance(self, plan: "ResidentPlan") -> None:
        self._resident.advance(plan)

    def reset_default_cursors(self) -> None:
        super().reset_default_cursors()
        # unlike attention, every read here needs a usable index
        self._default_layer_idx = 0

    def bind_sequence_shard(self, shard) -> None:
        """The sequence shard an attention over this pool exchanges split steps through
        (`AttentionManager.bind_dependencies`): a write outside an exchange raises (`check_bypass`).
        One per pool, since the writes check this one shard whichever node's step is running."""
        if self._sequence_shard is not None and self._sequence_shard is not shard:
            raise ValueError(
                f"KV {self.name!r} serves nodes under two sequence shards: declare one SeqShardSpec "
                "over every node of the pool"
            )
        self._sequence_shard = shard

    def opens_label(self, rid: int, label: str) -> bool:
        """Whether request ``rid`` opened ``label``. A padding row has no config, and opens any."""
        overrides = self._overrides.get(rid)
        return overrides is None or label in overrides.opened_labels(self._nodes)

    @torch.compiler.disable
    def write_target(self, layer_idx: int = None, label: str = None):
        """Where this step's K/V go, for a layer that writes them itself in a
        fused kernel instead of calling ``write_kv``: ``(layer cache, page of
        each token, slot in page of each token, tokens to write)``. NHD only."""
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        if label is None:
            label = self._default_label
        if self._sequence_shard is not None:
            self._sequence_shard.check_bypass(label, "KV write_target", by_head=self.placement is None)
        if self.kv_cache.layout != KVLayout.NHD:
            raise NotImplementedError(f"write_target is not implemented for layout {self.kv_cache.layout}")
        state = self._current_plan_states[label, self._layer_group[layer_idx]]
        if state.ready is not None:
            torch.cuda.current_stream().wait_event(state.ready)
        return (
            self.kv_cache.layer_view(layer_idx), state.token_to_page, state.token_to_cache,
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
    def write_slots(self, label: str=None) -> tuple[torch.Tensor, torch.Tensor]:
        """The (page, offset-in-page) slots ``write_kv`` scatters this step's tokens to, as
        device tensors, for a kernel that writes the cache itself. Under a capture lease they
        are the slot's static buffers, so a graph replays with the next step's slots."""
        if len(self._groups) > 1:
            raise ValueError(f"KV {self.name}: a pool with full and sliding layers writes through write_kv only")
        if label is None:
            label = self._default_label
        if self._sequence_shard is not None:
            self._sequence_shard.check_bypass(label, "KV write_slots", by_head=self.placement is None)
        plan_state = self._current_plan_states[label, 0]
        n = plan_state.total_tokens
        return plan_state.token_to_page[:n], plan_state.token_to_cache[:n]

    @torch.compiler.disable
    def write_kv(
        self, k: torch.Tensor, v: torch.Tensor | None = None,
        layer_idx: int=None, label: str=None, tokens: torch.Tensor | None = None,
    ) -> None:
        """Write K, V into this step's planned slots (under MLA: the per-token
        latent as ``k`` and ``v=None``). ``tokens``: which of the step's tokens
        ``k`` and ``v`` hold, by index (a ring's own share); by default its first."""
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        if label is None:
            label = self._default_label
        if self._sequence_shard is not None:
            self._sequence_shard.check_bypass(label, "KV write_kv", by_head=self.placement is None)
        plan_state = self._current_plan_states[label, self._layer_group[layer_idx]]
        if plan_state.ready is not None:
            torch.cuda.current_stream(k.device).wait_event(plan_state.ready)
        if tokens is not None:
            self.kv_cache.write_tokens(
                layer_idx=layer_idx, k=k, v=v,
                page_idx=plan_state.token_to_page[tokens], cache_idx=plan_state.token_to_cache[tokens],
            )
            return
        # an eager write may carry only the real rows; the padding rows' tokens
        # come last in the plan (on SINK_PAGE) and are skipped
        n = min(plan_state.total_tokens, k.shape[0])
        self.kv_cache.write_tokens(
            layer_idx=layer_idx,
            k=k[:n], v=None if v is None else v[:n],
            page_idx=plan_state.token_to_page[:n],
            cache_idx=plan_state.token_to_cache[:n],
        )
