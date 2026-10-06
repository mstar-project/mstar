import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

import numpy as np
import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import (
    AttentionResource,
    CGSlotSpec,
    EngineResourceInfo,
    PublishedInfo,
)
from mstar.engine.resources.kv.admission_log import open_log
from mstar.engine.resources.kv.cache import KVCache, PageAllocator
from mstar.engine.resources.kv.config import (
    KVReqConfig,
    KVSpec,
    KVStep,
    PagedKVConfig,
    RetentionPolicy,
)
from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool
from mstar.engine.resources.kv.keys import fingerprint, page_key
from mstar.engine.resources.kv.peak_plan import (
    PeakPlanner,
    PlanEntry,
    banker_safe,
    easy_allows,
    footprint,
    shadow_sum,
)
from mstar.engine.resources.kv.plan import (
    SINK_PAGE,
    KVPlanOutput,
    KVPlanOutputs,
    PagedIndptrs,
    SequenceView,
    build_paged_indptrs,
    group_by_plan_label,
)
from mstar.engine.resources.kv.plan_table import ABSENT, PlanTable, plan_entries
from mstar.engine.resources.kv.prefix_index import PrefixIndex
from mstar.engine.resources.kv.transfer import KVTransferManager, TransferEngineInfo
from mstar.engine.resources.step import (
    ADMIT_OK,
    ADMIT_WAIT,
    ADMIT_WAIT_BEHIND,
    AdmitFailedReason,
    AdmitOutcome,
    AdmitRuntimeError,
    AllocationFailed,
    GrantDeferred,
    RequestOffloading,
    Segment,
    StepContext,
)

logger = logging.getLogger(__name__)

# Off by default: `assert_pages_conserved` walks every stream of every live
# request after each admit, commit, reset and remove.
_DEBUG_ASSERTS = os.environ.get("MSTAR_KV_DEBUG_ASSERTS", "0") == "1"

# On by default: a plan is rebuilt from what each admitted request's streams changed since
# the last one (`PlanTable`) rather than from every stream. "0" reads them all, as before.
_PLAN_CACHE = os.environ.get("MSTAR_KV_PLAN_CACHE", "1") != "0"

# admitted, but its reservation does not fit yet: the scheduler asks again. The gate's
# own waits, which it may stop asking over (ADMIT_WAIT, ADMIT_WAIT_BEHIND), are told apart
_WAIT = AdmitOutcome(ok=True, ready=False)


@dataclass
class PageArena:
    """physical storage, free list, and per-page ownership

    A page goes back to the allocator when its last owner releases it, not its
    first. A sealed page is never written again, so a second owner may read it;
    freeing clears the seal. Every caller holds the manager's `_lock`, so the
    counts and seals need none of their own.
    """
    kv_cache: KVCache
    allocator: PageAllocator
    num_owners: list[int] = field(init=False, repr=False)
    sealed: list[bool] = field(init=False, repr=False)
    # bumped as owners are added or dropped, for what the manager keeps about the pool to be
    # kept by; `acquire` hands out free pages, which the index never holds, so it does not count.
    # A release counts once its pages are back on the free list, so a reader without the lock
    # (`KVManager.admission_keys`) that takes this before the free pages never holds a pair
    # that a later state repeats
    owner_changes: int = field(default=0, init=False, repr=False)
    # told which pages gained or lost an owner, as they do: the index keeps its evictable pages by it
    owner_hook: Callable[[list[int]], None] | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.num_owners = [0] * self.allocator.max_num_pages
        self.sealed = [False] * self.allocator.max_num_pages

    def acquire(self, n: int) -> list[int] | None:
        pages = self.allocator.try_allocate(n)
        if pages is not None:
            for page in pages:
                self.num_owners[page] = 1
        return pages

    def retain(self, pages: list[int]) -> None:
        for page in pages:
            self.num_owners[page] += 1
        self.owner_changes += 1
        if self.owner_hook is not None:
            self.owner_hook(pages)

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
        self.allocator.free(freed)
        self.owner_changes += 1
        if self.owner_hook is not None:
            self.owner_hook(pages)

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
    # how many of this stream's pages the index already holds
    cursor: int = 0
    # set once this stream has been reported, so a request that is admitted
    # again (a refused admit, a second partition) is still one line
    reported: bool = False

    @classmethod
    def seed(cls, keys: list[bytes], tail: list[int], page_size: int) -> "PrefixChain":
        keyed_pages = len(keys) - (1 if tail else 0)
        return cls(
            keys=list(keys), unkeyed=tail, keyed_pages=keyed_pages,
            covered_len=keyed_pages * page_size + len(tail),
        )

    def extend(self, tokens: list[int], page_size: int) -> None:
        self.unkeyed.extend(tokens)
        self.covered_len += len(tokens)
        while len(self.unkeyed) >= page_size:
            whole = self.keyed_pages
            key = page_key(
                self.keys[whole - 1] if whole else b"",
                self.unkeyed[:page_size],
            )
            # overwrites the partial key the prompt left here, if any
            if whole < len(self.keys):
                self.keys[whole] = key
            else:
                self.keys.append(key)
            self.keyed_pages = whole + 1
            del self.unkeyed[:page_size]

    def pages_filled(self, stored_len: int, page_size: int) -> int:
        return min(stored_len // page_size, self.keyed_pages)


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
    # None while the stream is unkeyed, and once its keys stop describing it
    chain: PrefixChain | None = None
    # pages the index matched for this stream and is holding for it, from the
    # probe until `admit` converts them onto `page_indices`
    lease: list[int] | None = None
    # set at conversion, cleared at `commit`, so a refused admit's re-probe answers the same
    converted: bool = False
    # leading pages of `page_indices` the index lent this stream (a converted
    # lease, a local match): held, but never taken from the free list
    hits: int = 0
    # a lease admission took before prepare saw the request, until prepare cuts
    # the inputs to it (`apply_cached_prefix`); `admit` never converts one still set
    gate_lease: bool = False
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
            self.hits = 0

    def forget_chain(self):
        """Drop what the chain knows of this stream, for a run that starts over.

        Not part of `reset`, which an offload calls too: a reload brings the
        same tokens back, so the chain that describes them has to survive it.
        """
        self.chain = None


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


@dataclass
class Reservation:
    """What an admitted request may take from the free list over its life."""
    pages: int
    # counted by the model; one guessed from the prompt's ids can overrun
    exact: bool


@dataclass
class PlanState:
    """The admitted requests as a plan sees them, for as long as nothing moves.

    Built when a request asks to be admitted and the reserved set, the free
    pages or the owners of a page have changed since the last build; ``key`` is
    what that was taken at. Rounds left are as of the build, so they run up to a
    page's worth of tokens late until the next page is granted, which only
    delays a release in the plan.
    """
    key: tuple
    # None until asked for (`all_entries`) where the plan is kept as arrays
    entries: list[PlanEntry] | None
    held: Sequence[int]
    need: Sequence[int]
    held_total: int
    # only built for the peak fit
    planner: PeakPlanner | None = None
    # where the plan is kept as arrays (`PlanTable`): what the entries are made of, as of the
    # build; ``held`` and ``need`` again with a slot more, for a candidate; and the sum of ``need``
    rows: tuple[np.ndarray, ...] | None = None
    spare: tuple[np.ndarray, np.ndarray] | None = None
    need_total: int | None = None

    def all_entries(self) -> list[PlanEntry]:
        if self.entries is None:
            self.entries = plan_entries(*self.rows)
        return self.entries

    def owed(self) -> int:
        """Pages the admitted requests may still take."""
        return sum(self.need) if self.need_total is None else self.need_total

    def with_candidate(self, cand: PlanEntry) -> tuple[Sequence[int], Sequence[int]]:
        """``held`` and ``need`` with ``cand`` after the admitted requests."""
        if self.spare is None:
            return [*self.held, cand.held], [*self.need, cand.need]
        held, need = self.spare
        held[-1], need[-1] = cand.held, cand.need
        return held, need


@dataclass
class AdmissionDeferred(AdmitFailedReason):
    """A step carries a request that has not reserved, and its turn hasn't come.

    Not an `AllocationFailed`: nothing needs evicting. The step goes back, and
    readiness admits the request when its reservation fits.
    """
    request_id: str


LabelToStream = dict[str, CacheStream]

@dataclass
class KVSequenceInfo:
    seq_len: int
    # for tracking KV cache
    latest_kv_transfer_info: Any
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


@dataclass
class KVPlanState:
    token_to_page: torch.Tensor
    token_to_cache: torch.Tensor
    total_tokens: int | None = None

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
        nodes: set[str] | None = None,
        leads: bool = True,
    ):
        self.config = cfg
        if joint_comm_group is not None:
            # before the cache is allocated: it is sized off the head counts
            cfg.shard(joint_comm_group.world_size)
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
        if cfg.cpu_offload_pages > 0:
            self._cpu_pool = CPUPagePool(
                config=cfg, kv_cache=self.kv_cache,
                max_cpu_pages=cfg.cpu_offload_pages,
            )
        self._streams: dict[str, LabelToStream] = {}
        self._overrides: dict[str, KVReqConfig] = {}
        # its entity id is the worker id, and one worker is one copy of a node
        self._replica = (
            transfer_engine_info.my_entity_id
            if transfer_engine_info is not None else None
        )
        self._prefix_root: bytes | None = None
        self._index: PrefixIndex | None = None
        # label -> the walks its keys describe; a label not here is not gated
        self._keyed_walks: dict[str, frozenset[str]] = {}
        # nodes already warned that a declared stream reached them unkeyed
        self._warned_unkeyed: set[str] = set()
        self._rank = joint_comm_group.rank if joint_comm_group is not None else 0
        self._world_size = joint_comm_group.world_size if joint_comm_group is not None else 1
        self._comm_group = joint_comm_group
        self._device = device
        self._lock = threading.RLock()

        # (slot, label) -> KVPlanState, sized for the largest capture bucket
        self._static_plan_states: dict[tuple[int, str], KVPlanState] = {}
        self._cg_max_seq_len = 0
        self._current_plan_states: dict[str, KVPlanState] = {}
        self.reset_default_cursors()

        self._preplan_states: dict[str, KVPlanState] = {}
        self._preplanned = False
        self._preplan_key = None
        self._cached_plan_output: dict[str, KVPlanOutput] | None = None

        # Prior length and epochs for pre-forks applied by a staged step;
        # facilitates clear_preplan.
        self._preplan_fork_undo: list[tuple[str, str, int, int, int]] = []
        # (rid, to_label) for intialized reservations by staged step; cleared_preplan removes
        # recorded in admit not plan, so separate from above.
        self._preplan_new_labels: list[tuple[str, str]] = []
        # (rid, label) marked step_in_flight by a staged step, so an abandoned
        # one does not leave its streams unevictable
        self._preplan_marked: list[tuple[str, str]] = []

        # the nodes sharing this pool, whose labels a request's reservation covers
        self._nodes = nodes
        # whether this pool's count stands for the request: the only pool of its
        # spec, or the spec leader's. A guidance pool under CFG parallel only
        # reserves what the leader admitted, so no room is checked on it
        self._leads = leads
        self._decides = self._rank == 0 and leads
        self._reserved: dict[str, Reservation] = {}
        # requests that asked and have not reserved, in the order they first asked
        self._waiting: dict[str, None] = {}
        # the head's rooted prompt keys, hashed once however often it asks
        self._rooted: dict[str, list[bytes]] = {}
        # the labels each request's walks open here, read per segment per step
        self._opened: dict[str, set[str]] = {}
        # label -> the walk prepare probes it on
        self._prefill_walks: dict[str, str] = {}

        # how this pool admits: the environment, then the config. The defaults
        # (sum, fifo) take none of the paths below.
        self._fit, self._order = cfg.resolved_admission()
        self._peak = self._fit == "peak"
        self._backfill = self._order == "backfill"
        self._alog = open_log(name)
        # decisions that are not the summed test in arrival order, or that are logged
        self._planned = self._peak or self._backfill or self._alog is not None
        # whether a plan is rebuilt from what changed in the requests' streams (see `_plan_state`)
        self._plan_cache = _PLAN_CACHE and self._planned
        self._plan_table = PlanTable()
        # the admitted requests whose row in it is out of date: touched since it was set
        self._plan_dirty: set[str] = set()
        # bumped as what is reserved changes; with the arena's counters it says
        # whether a plan, or a refusal, still describes the pool
        self._reserved_epoch = 0
        # bumped as a request leaves the queue, the one way a request moves up it: one that
        # arrives goes last. What `admission_keys` says of the queue
        self._queue_epoch = 0
        self._plan: PlanState | None = None
        self._shadow: tuple[tuple, tuple[int, int] | None] | None = None
        # rid -> the state a request was last refused in: asked again in it, the answer is no
        self._refused: dict[str, tuple] = {}
        # rid -> (label, prompt tokens, decode tokens), as `_reservation` counts them
        self._shape: dict[str, list[tuple[str, int, int]]] = {}
        # under TP, the requests a step has reached on this rank: what every rank agrees on
        self._seen: set[str] = set()
        # telemetry only: when each request first asked, reserved, and what it last waited on
        self._asked_at: dict[str, float] = {}
        self._reserved_at: dict[str, float] = {}
        self._wait_state: dict[str, tuple] = {}
        # Backfill only. The requests it looks at: the first `_window_size` of `_waiting`,
        # kept as they are added and dropped (None once a member left, until it is read
        # again), so a pass over a long queue does not walk it.
        self._window_size = cfg.resolved_backfill_window()
        self._window: dict[str, None] | None = {}
        # the most a request behind the head could be admitted with, and what it was taken at
        self._room_at: tuple[tuple, int] | None = None
        # `_outstanding`, and what it was taken at
        self._owed: tuple[tuple, int] | None = None
        # off, the cheap refusals are not made and every request in the window is worked out
        # in full: what the tests compare the answers against
        self._cheap_refusals = True
        # CUDA-graph padding rows, which `_capacity` has to count the pages of
        self._padding: set[int] = set()

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
            nodes=spec.nodes if info.nodes is None else info.nodes,
            leads=(
                info.nodes is None or info.nodes >= spec.nodes
                or spec.leader in info.nodes
            ),
        )

    def build_cuda_graph_buffers(
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

    def enable_prefix_cache(
        self, root: bytes, walks: dict[str, tuple[str, str | None]] | None = None,
    ) -> bool:
        """Open the index under ``root``, the identity every key hangs from."""
        self._keyed_walks = {
            label: frozenset(walk for walk in named if walk is not None)
            for label, named in (walks or {}).items()
        }
        self._prefill_walks = {label: named[0] for label, named in (walks or {}).items()}
        if not self.config.prefix_cache:
            return False
        if self._world_size > 1:
            logger.info(
                "KV %s: prefix cache off at world size %d: the ranks index "
                "independently and would match different lengths",
                self.name, self._world_size,
            )
            return False
        self._prefix_root = root
        self._index = PrefixIndex(self._arena)
        return True

    def resolve_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str,
    ) -> int | None:
        """Match this request's prefix against the index and hold what matched.

        Answers the same length every time it is asked, and takes one reference
        however often that is: a probe is repeated whenever a step is prepared
        again.
        """
        with self._lock:
            label = self._keyed_label(rid, node_name, graph_walk)
            if label is None:
                self._warn_unkeyed(rid, node_name, graph_walk)
                return None
            stream = self._ensure_label(rid, label)
            if stream.converted:
                return stream.stored_len
            if stream.lease is not None:
                return len(stream.lease) * self.config.page_size
            if stream.stored_len or stream.offloaded or stream.read_pending:
                return None
            keys = list(stream.chain.keys)
        # outside the lock: one SHA-256 per page, and admit, commit and remove would wait on it
        rooted = [fingerprint(self._prefix_root, key) for key in keys]
        with self._lock:
            if self._streams.get(rid, {}).get(label) is not stream:
                return None
            # admission may have leased it on its own thread while this one hashed
            if stream.lease is None:
                self._lease(rid, stream, rooted)
            if stream.lease is None:
                return None
            return len(stream.lease) * self.config.page_size

    def _lease(self, rid: str, stream: CacheStream, rooted: list[bytes]) -> None:
        # one key short, so a fully cached prompt still leaves a token to run
        matched = self._index.lookup(rooted)[:len(rooted) - 1]
        if not matched:
            return
        self._arena.retain(matched)
        stream.lease = matched
        self._plan_touch(rid)
        self._lent(rid, len(matched))

    def _lent(self, rid: str, pages: int) -> None:
        # held rather than taken, so the reservation gives up what the index lends
        # (and takes back what it returns)
        reserved = self._reserved.get(rid)
        if reserved is not None:
            reserved.pages -= pages
            self._reserved_epoch += 1
            self._plan_touch(rid)

    def apply_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, inputs, matched_len: int,
    ) -> None:
        """Cut this stream's lease down to ``matched_len``, the agreed length."""
        del inputs
        with self._lock:
            label = self._keyed_label(rid, node_name, graph_walk)
            if label is None:
                return
            stream = self._streams.get(rid, {}).get(label)
            if stream is None or stream.lease is None or stream.stored_len:
                return
            stream.gate_lease = False
            keep = matched_len // self.config.page_size
            if keep < len(stream.lease):
                self._lent(rid, keep - len(stream.lease))
                self._arena.release(stream.lease[keep:])
                stream.lease = stream.lease[:keep] or None
                self._plan_touch(rid)

    def _index_filled_pages(
        self, segment: Segment, stream: CacheStream, ctx: StepContext,
    ) -> None:
        """Offer every page this commit filled to the index.

        Only whole pages, because the tail is still being written into and a
        second owner would be reading bytes that are still moving. A dummy or
        padded row is left out: its pages hold whatever the capture wrote.
        """
        chain = stream.chain
        if (
            self._index is None
            or ctx.capture
            or segment.request_id not in ctx.request_ids
            or chain is None
        ):
            return
        walks = self._keyed_walks.get(segment.label)
        if walks is not None and ctx.graph_walk not in walks:
            # another walk wrote this span (an edit's image, say), which the keys do not describe
            stream.chain = None
            return
        # what was here before this write, not after: a decode step can commit
        # before the token it writes is read back and counted
        if stream.released or stream.stored_len - segment.span > chain.covered_len:
            stream.chain = None
            return
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
            if chain.cursor else None
        )
        while chain.cursor < filled:
            key = fingerprint(self._prefix_root, chain.keys[chain.cursor])
            page = stream.page_indices[chain.cursor]
            # unparented, not chained, under a page only the index holds: a live
            # child would keep eviction from a page the pool counts as free
            if parent is not None and self._arena.num_owners[parent] == 1:
                parent = None
            if not self._index.insert(key, page, parent):
                # another request filled this page first and its copy is the
                # one the index names; ours stays private to this stream
                page = self._index.page_for(key)
            parent = page
            chain.cursor += 1

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
            "KV %s: %s/%s matched %d of its %d pages, whole prompt already "
            "cached %s, replica %s",
            self.name, segment.request_id, segment.label,
            matched, chain.keyed_pages,
            bool(matched) and matched == chain.keyed_pages,
            self._replica,
        )

    def _take_local_match(
        self, stream: CacheStream, published_len: int, rooted: list[bytes] | None,
    ) -> None:
        """Take what this cache already holds of a stream about to be read in.

        The pages a prefill rank published are often pages this rank wrote for
        an earlier request, and moving them again costs a transfer to arrive at
        bytes that are already here. Converted onto the stream rather than
        leased: the read is issued under this same lock, so nothing can come
        between the two.
        """
        if (
            self._index is None
            or not rooted
            or stream.stored_len
            or stream.offloaded
            or stream.chain is None
        ):
            return
        # never past what the other side published, and only whole pages
        matched = self._index.lookup(rooted)[
            :published_len // self.config.page_size
        ]
        if not matched:
            return
        self._arena.retain(matched)
        # `stored_len` is 0, so any pages the stream holds are empty
        self._arena.release(stream.page_indices)
        stream.page_indices = list(matched)
        stream.hits = len(matched)
        stream.stored_len = len(matched) * self.config.page_size
        stream.chain.cursor = len(matched)

    def _warn_unkeyed(self, rid: str, node_name: str, graph_walk: str) -> None:
        """Say once per node that a stream the model declared arrived with no keys.

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
        for label in overrides.get_labels(node_name, graph_walk):
            if graph_walk in self._keyed_walks.get(label, ()) and not keys.get(label):
                self._warned_unkeyed.add(node_name)
                logger.warning(
                    "KV %s: requests reach %s with no keys for %s, which the "
                    "model declared for prefix reuse, so nothing will be cached",
                    self.name, node_name, label,
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
        for label in overrides.get_labels(node_name, graph_walk):
            if keys.get(label):
                return label
        return None

    def _seed_keys(self, rid: str, label: str, stream: CacheStream) -> None:
        """Put this request's chain for ``label`` on its stream."""
        overrides = self._overrides.get(rid)
        if overrides is None or not overrides.prefix_cache:
            return
        keys = (overrides.prefix_keys or {}).get(label)
        if not keys:
            return
        tail = list((overrides.prefix_tail or {}).get(label) or ())
        stream.chain = PrefixChain.seed(keys, tail, self.config.page_size)

    def extend_prefix_chain(
        self, rid: str, node_name: str, graph_walk: str, outputs,
    ) -> None:
        """Key what this request generated, once a page of it exists.

        The ids come from the stop check's host copy, which can land after their
        step commits, so a later commit indexes the page.
        """
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

    def _release_lease(self, stream: CacheStream) -> None:
        """Give back a lease `admit` never converted; a converted one is released
        with `page_indices`."""
        if stream.lease is not None:
            self._arena.release(stream.lease)
        stream.lease = None
        stream.gate_lease = False

    def fingerprint(self) -> bytes:
        return fingerprint(
            self.config.layout.value, self.config.page_size,
            self.kv_cache.tensor.dtype, self._world_size,
            self.config.prefix_cache_salt,
        )

    def ingest_request(self, rid, overrides: KVReqConfig | None=None):
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
            if self._is_padding(rid):
                self._padding.add(rid)
            for label, stream in self._streams[rid].items():
                if stream.chain is None:
                    self._seed_keys(rid, label, stream)

    def admit_retrieve(
        self, rid: str,
        node_name: str,
        graph_walk: str,
        published: PublishedKVInfo | None
    ) -> AdmitOutcome:
        # one pool decides for each request, and the rest follow it
        if self._decides:
            gated = self._gate(rid, node_name, graph_walk)
            if gated is not None:
                return gated
        if published is None:
            return ADMIT_OK

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
        if self._index is not None and overrides.prefix_cache:
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
                    # what the index lent goes, so the request takes those pages itself
                    self._lent(rid, -(len(stream.lease or ()) + stream.hits))
                    self._release_lease(stream)
                    drop = self._arena.any_sealed(stream.page_indices)
                    if drop:
                        self._arena.release(stream.page_indices)
                    stream.reset(freed=drop)
                    stream.forget_chain()
                    self._plan_touch(rid)
                if not own and remote_reset_generation is not None:
                    stream.remote_reset_generation = remote_reset_generation
                if not own and stream.chain is not None:
                    # another rank sampled the token after this record, so the pending tail is one behind
                    stream.chain.unkeyed = None
                new_len = seq_info.seq_len
                hits = stream.hits
                self._take_local_match(stream, new_len, rooted.get(label))
                self._plan_touch(rid)
                self._lent(rid, stream.hits - hits)
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

    def admit(self, step: KVStep, ctx: StepContext) -> AdmitOutcome:
        if self._preplanned and not ctx.is_preplan:
            if self._preplan_key == self._plan_key(step, ctx):
                # pages were already reserved by the preplan pass
                return ADMIT_OK
            # a different step arrived first (see `plan`): drop the staged
            # plan and reserve for this step normally. The staged step's own
            # span pages stay with its stream, where its re-admit finds them.
            self.clear_preplan()
        # forks reserve here and copy later (plan for pre-, commit for post-),
        # so a step that never runs leaves pages resident but no page contents
        # moved — re-admitting it allocates nothing and re-copies nothing.
        # A post-fork copies the source *after* this step's spans land, so its
        # reservation covers them.
        growth = self._label_growth(step) if step.commit else {}
        forks = [(pre, 0) for pre in step.pre_forks] + [
            (post, growth) for post in step.post_forks
        ]
        # one critical section so the read-of-stored_len then alloc is atomic
        # against a concurrent reset/remove/commit on another thread
        with self._lock:
            if self._peak and self._world_size > 1:
                self._seen.update(ctx.request_ids)
            # before anything moves, so a refusal leaves nothing to unwind, and
            # before the lease converts, as a request is sized off the lease it holds
            if not ctx.capture:
                deferred = self._reserve_new(ctx)
                if deferred is not None:
                    return AdmitOutcome(ok=False, reason=deferred)
            # before the fork loop and the segment loop, both of which size
            # off `stored_len`: a pre-fork target has to cover the whole prefix
            for segment in step.segments:
                stream = self._streams.get(
                    segment.request_id, {},
                ).get(segment.label)
                if stream is None:
                    continue
                if stream.gate_lease and segment.span:
                    # prepare never cut this step's inputs to the lease the gate
                    # took, so the step writes the stream from 0
                    self._lent(segment.request_id, -len(stream.lease))
                    self._release_lease(stream)
                    self._plan_touch(segment.request_id)
                if (
                    stream.lease is not None
                    and not stream.gate_lease
                    and not stream.stored_len
                    and not stream.offloaded
                    and not stream.read_pending
                ):
                    # drop the empty pages a refused batch admit left, before taking the lease
                    self._arena.release(stream.page_indices)
                    stream.page_indices = list(stream.lease)
                    stream.hits = len(stream.lease)
                    stream.stored_len = (
                        len(stream.lease) * self.config.page_size
                    )
                    # they came out of the index, so they are already in it
                    stream.chain.cursor = len(stream.lease)
                    stream.lease = None
                    stream.converted = True
                    self._plan_touch(segment.request_id)
                self._report_admission(segment, stream)
            if ctx.is_preplan:
                # a preplan that was promoted or abandoned already cleared
                # these; reset anyway so a refused admit can't leave stale
                # entries for the next clear_preplan to act on
                self._preplan_new_labels = []
                self._preplan_marked = []
            for (from_label, to_label), extra in forks:
                for rid in ctx.padded_request_ids:
                    if self._in_sink(rid, ctx, to_label):
                        continue
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

            for segment in step.segments:
                if segment.span == 0:
                    continue
                stream = self._ensure_label(segment.request_id, segment.label)
                if self._in_sink(segment.request_id, ctx, segment.label):
                    continue
                alloc_res = self._alloc(
                    segment.request_id,
                    segment.label,
                    segment.span + stream.stored_len
                )
                if not alloc_res.success:
                    return AdmitOutcome(ok=False, reason=alloc_res.error)

            # marked here rather than in plan so the mark also covers
            # admit -> plan, where an offload would otherwise release pages
            # this step has already been given. last, and only once every
            # reservation above succeeded, so a refusal has nothing to unwind.
            # `.get` because a zero-span segment on a label nothing created
            # reserves no stream (see the loop above)
            for segment in step.segments:
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

    def _sequence_views(self, segments: list[Segment]) -> list[SequenceView]:
        views = []
        page_size = self.kv_cache.page_size
        for s in segments:
            stream = self._streams[s.request_id][s.label]
            # `page_indices` is a high-water mark, so a stream can hold more
            # pages than its tokens need (a refused admit, a reset that kept
            # its pages). slice to the length or the view addresses token 0
            # into the wrong page and reports the padding as resident context
            length = s.span + stream.stored_len
            num_pages = -(-length // page_size)
            page_idxs = stream.page_indices[:num_pages]
            if self._is_padding(s.request_id) or self._unheld(s.request_id, s.label):
                # took no page of its own (see `_in_sink`): its tokens land in the sink
                page_idxs += [SINK_PAGE] * (num_pages - len(page_idxs))
            views.append(SequenceView(
                request_id=s.request_id,
                label=s.label, page_idxs=page_idxs,
                length=length,
                to_compute=s.span,
                generation=stream.generation,
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

    def _decode_plan_state(self, views: list[SequenceView]) -> KVPlanState:
        """Write addressing for a step appending one token per request.

        The packed path needs the indptrs on device and ~a dozen kernels to
        unpack them per token. A decode step's slot is just the end of each
        stream, so build it in the same CPU pass the views came from and send
        it over as one H2D.
        """
        page_size = self.kv_cache.page_size
        pages: list[int] = []
        offsets: list[int] = []
        for view in views:
            # off the stream's page count, not its logical length: that is what
            # `build_paged_indptrs` hands attention, so a stream holding more
            # pages than its length needs stays self-consistent
            pages.append(view.page_idxs[-1])
            offsets.append((view.last_page_len(page_size) or page_size) - 1)
        locations = torch.tensor(
            [pages, offsets], dtype=torch.long
        ).to(self._device, non_blocking=True)
        return KVPlanState(
            token_to_page=locations[0],
            token_to_cache=locations[1],
            total_tokens=len(views),
        )

    def _setup_plan_states(
        self, plan_output: dict[str, KVPlanOutput],
        ctx: StepContext, lease,
    ):
        for label, indptrs in plan_output.items():
            if indptrs.is_decode:
                plan_state = self._decode_plan_state(indptrs.views)
            else:
                indptrs.cuda_indptrs = indptrs.cpu_indptrs.to_device(self._device)
                plan_state = self._compute_plan_state(
                    indptrs.cuda_indptrs,
                    total_tokens=indptrs.get_total_len()
                )
            if lease is not None:
                static_state = self._static_plan_state(lease.slot, label)
                static_state.copy_(plan_state, lease.bucket.num_tokens)
                plan_state = static_state
            if ctx.is_preplan:
                self._preplan_states[label] = plan_state
            else:
                self._current_plan_states[label] = plan_state


    def _plan_output(self, views: list[SequenceView]) -> KVPlanOutput:
        return KVPlanOutput(
            cpu_indptrs=build_paged_indptrs(views, self.kv_cache.page_size),
            views=views,
        )

    def plan(self, step: KVStep, ctx: StepContext) -> dict[str, KVPlanOutput]:
        """
        Returns list of sequence views per plan label
        """
        assert not (self._preplanned and ctx.is_preplan), (
            "KV preplan is already pending; clear_preplan before planning a "
            "different step ahead"
        )
        self.reset_default_cursors()
        if self._preplanned:
            if self._preplan_key == self._plan_key(step, ctx):
                self._current_plan_states = self._preplan_states
                res = self._cached_plan_output
                self._preplan_fork_undo = []
                self._preplan_new_labels = []
                self._preplan_marked = []
                self.clear_preplan()
                return res
            # A different step reached the GPU thread before the one planned
            # ahead (e.g. a new request's prefill while a decode step sits
            # pre-planned): it must not be served the staged plan's pages.
            # Undo the staged plan's side effects and plan inline.
            self.clear_preplan()
        undo = self._preplan_fork_undo if ctx.is_preplan else None
        for (from_label, to_label) in step.pre_forks:
            for rid in ctx.padded_request_ids:
                if not self._in_sink(rid, ctx, to_label):
                    self._apply_fork(rid, from_label, to_label, undo=undo)
        res = KVPlanOutputs(
            {
                plan_label: self._plan_output(self._sequence_views(segments))
                for plan_label, segments in group_by_plan_label(
                    step.segments, step.combined_labels
                ).items()
            },
            pre_forks=step.pre_forks,
            post_forks=step.post_forks,
        )
        self._setup_plan_states(res, ctx, ctx.slot_lease)
        if ctx.is_preplan:
            self._preplan_key = self._plan_key(step, ctx)
            self._preplanned = True
            self._cached_plan_output = res
        return res

    @property
    def supports_preplan(self):
        return True

    @staticmethod
    def _plan_key(step: KVStep, ctx: StepContext):
        """What identifies the step a pre-plan was staged for: its segments
        and the replay slot it was leased on."""
        lease = ctx.slot_lease
        return tuple(step.segments), (lease.slot if lease is not None else None)

    def clear_preplan(self):
        # the staged step is not going to run, so undo what it did to live
        # state: dropping the cached plan is not enough, the pre-forks already
        # copied pages and moved lengths
        with self._lock:
            for (
                rid,
                label,
                stored_len,
                generation,
                reset_generation,
            ) in reversed(
                self._preplan_fork_undo
            ):
                stream = self._streams.get(rid, {}).get(label)
                if stream is not None:
                    stream.stored_len = stored_len
                    stream.generation = generation
                    stream.reset_generation = reset_generation
            self._preplan_fork_undo = []
            # labels the staged step invented are removed, not rewound to 0:
            # a stream at 0 that nothing asked for is still a stream, and it
            # holds the pages the reservation took
            for rid, label in reversed(self._preplan_new_labels):
                stream = self._streams.get(rid, {}).pop(label, None)
                if stream is not None:
                    self._arena.release(stream.page_indices)
                    self._plan_touch(rid)
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
        # atomic against admit_retrieve reading stored_len on another thread
        with self._lock:
            for segment in step.segments:
                stream = self._streams[segment.request_id][segment.label]
                # cleared before the `step.commit` test: a step that keeps no
                # tokens (image_gen, action_gen) still read these pages, and
                # leaving the mark set would make the request unevictable
                stream.step_in_flight = False
                # went to the sink: a length kept here would stretch every later view of it over sink pages
                if (
                    step.commit and segment.span > 0
                    and not self._unheld(segment.request_id, segment.label)
                ):
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
                        continue
                    stream.stored_len += segment.span
                    # committed, so there is no refused admit left for a re-probe to answer
                    stream.converted = False
                    policy = step.retention.get((segment.request_id, segment.label))
                    if policy is not None:
                        self._adopt_retention(stream, policy, segment)
                    self._index_filled_pages(segment, stream, ctx)
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
                        self._apply_retention(stream, policy)
                        self._plan_touch(segment.request_id)
            # post-forks copy what this step just wrote, so they land after the
            # spans above are counted
            for (from_label, to_label) in step.post_forks:
                for rid in ctx.padded_request_ids:
                    if not self._in_sink(rid, ctx, to_label):
                        self._apply_fork(rid, from_label, to_label)
            if _DEBUG_ASSERTS:
                self.assert_pages_conserved()

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

    def _apply_retention(self, stream: CacheStream, policy: RetentionPolicy) -> int:
        """Release what the step's policy no longer keeps. Under the lock."""
        excess = stream.stored_len - policy.protected_prefix - policy.context_budget
        if excess <= 0:
            return 0
        return self._release_oldest_locked(stream, excess)

    def _release_oldest_locked(self, stream: CacheStream, num_tokens: int) -> int:
        """The page-floored front release shared by ``release_oldest`` and the
        commit-time retention. Under the lock."""
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
        self._arena.release(freed)
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
            released = self._release_oldest_locked(stream, num_tokens)
            self._plan_touch(request_id)
            return released

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

    def offload(self, rid: str) -> int:
        """Move every stream of ``rid`` to host memory. Returns pages freed.

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
            return 0
        released: set[str] = set()
        try:
            # blocking work OUTSIDE the lock: drain the in-flight reads, then
            # copy each claimed stream to host
            if read_futures:
                wait(read_futures)
            moved = [
                claim for claim in claimed
                if self._cpu_pool.offload_stream(
                    rid=rid, label=claim.label,
                    gpu_kv_cache=self.kv_cache.tensor,
                    gpu_page_indices=claim.pages,
                    stored_len=claim.stored_len, position=claim.position,
                    released=claim.released,
                    protected_prefix=claim.protected_prefix,
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
            for label, stream in streams.items():
                if stream.offloaded or not stream.page_indices:
                    continue
                stream.offloaded = True
                claimed.append(ClaimedStream(
                    label=label,
                    pages=list(stream.page_indices),
                    generation=stream.generation,
                    stored_len=stream.stored_len,
                    position=stream.position,
                    released=stream.released,
                    protected_prefix=stream.protected_prefix,
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
        with self._lock:
            streams = self._streams.get(rid, {})
            for claim in moved:
                stream = streams.get(claim.label)
                if stream is None or stream.generation != claim.generation:
                    # removed, or written to behind us. Releasing nothing here
                    # leaves every claim for `_abandon_claims` to undo.
                    return 0, set()
            freed = 0
            for claim in moved:
                stream = streams[claim.label]
                freed += len(claim.pages)
                # what the index lent comes back on reload as the request's own copies
                self._lent(rid, -stream.hits)
                self._arena.release(claim.pages)
                stream.page_indices = []
                stream.hits = 0
                # Offload changes residency, not logical stream contents.
                stream.reset(content_reset=False)
                self._plan_touch(rid)
            return freed, {claim.label for claim in moved}

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

    def reload(self, rid: str) -> bool:
        """Bring every offloaded stream of ``rid`` back on device.

        False when the device can't fit them right now; nothing moves in that
        case, so the caller can evict further and try again.
        """
        if self._cpu_pool is None or not self._cpu_pool.is_offloaded(rid):
            return False
        with self._lock:
            labels = self._cpu_pool.labels(rid)
            needed = sum(self._cpu_pool.num_pages(rid, label) for label in labels)
            if needed > self._arena.num_free and self._index is not None:
                # as `_alloc` does: once the pool is all cached pages, nothing
                # else would ever free one for this request to come back to
                self._index.evict(needed - self._arena.num_free)
            if needed > self._arena.num_free:
                return False
            for label in labels:
                stream = self._ensure_label(rid, label)
                pages = self._arena.acquire(
                    self._cpu_pool.num_pages(rid, label)
                )
                if pages is None:
                    # lost the race for pages against another consumer
                    return False
                state = self._cpu_pool.reload_stream(
                    rid=rid, label=label,
                    gpu_kv_cache=self.kv_cache.tensor,
                    gpu_page_indices=pages,
                )
                stream.page_indices = pages
                stream.stored_len = state.stored_len
                stream.position = state.position
                stream.released = state.released
                stream.protected_prefix = state.protected_prefix
                stream.offloaded = False
                self._plan_touch(rid)
        # sync outside the lock: orders the reload H2D copies before attention
        # reads them, but the pages are already assigned so it touches no
        # shared state
        self._cpu_pool.sync()
        return True

    def reclaimable(self, rid: str) -> int:
        """Device pages the request is holding; 0 once offloaded, and for one
        admitted but not yet run."""
        streams = self._streams.get(rid)
        if streams is None:
            return 0
        return sum(len(stream.page_indices) for stream in streams.values())

    def get_offload_priority(self, rid: str) -> float:
        """Device pages the request is holding — the most reclaimable first."""
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
                # a rewind would put the next write on the stream's first page,
                # over a sealed one its other owners still read. drop the pages
                # and let the next write allocate; `free` asks for the same
                self._release_lease(stream)
                # here, not in `CacheStream.reset`: an offload resets the stream
                # too, and the probe after its reload still owes the same length
                stream.converted = False
                drop = free or self._arena.any_sealed(stream.page_indices)
                if drop:
                    self._arena.release(stream.page_indices)
                stream.reset(freed=drop)
                # a stale cursor or generated key would misfile what the rerun writes
                stream.forget_chain()
                self._plan_touch(rid)
            for label, stream in self._streams.get(rid, {}).items():
                self._seed_keys(rid, label, stream)
            if _DEBUG_ASSERTS:
                self.assert_pages_conserved()

    def remove_request(self, rid: str):
        streams = self._streams.get(rid)
        if streams is not None:
            # drain in-flight reads outside the lock; see reset_request
            for stream in streams.values():
                if stream.read_future is not None:
                    wait([stream.read_future])
        with self._lock:
            if self._alog is not None and rid in self._reserved:
                self._log_release(rid)
            if rid in self._streams:
                for stream in self._streams[rid].values():
                    self._release_lease(stream)
                    self._arena.release(stream.page_indices)
            if self._cpu_pool is not None:
                self._cpu_pool.remove_request(rid)
            self._streams.pop(rid, None)
            self._overrides.pop(rid, None)
            self._transfer.remove_request(rid)
            self._reserved.pop(rid, None)
            self._wait_drop(rid)
            self._rooted.pop(rid, None)
            self._opened.pop(rid, None)
            self._padding.discard(rid)
            self._forget_admission(rid)
            if _DEBUG_ASSERTS:
                self.assert_pages_conserved()

    def assert_pages_conserved(self) -> None:
        """Check the owner counts against the streams holding the pages.

        Each assertion names the rule it checks. Host pages are not covered:
        `CPUPagePool` keeps no counts.
        """
        with self._lock:
            arena = self._arena
            free = list(arena.allocator.free_pages.queue)
            owned = [
                page for page in range(self.config.max_num_pages)
                if arena.num_owners[page] > 0
            ]
            both = sorted(set(free) & set(owned))
            assert not both, f"pages both free and owned: {both}"
            assert len(free) + len(owned) == self.config.max_num_pages, (
                f"{self.config.max_num_pages} pages in the pool, but "
                f"{len(free)} free and {len(owned)} owned"
            )

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
                    full = stream.stored_len // self.config.page_size
                    frontier.update(stream.page_indices[full:])

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

            unsealed = [
                page for page in owned
                if arena.num_owners[page] > 1 and not arena.sealed[page]
            ]
            assert not unsealed, f"pages shared before they were sealed: {unsealed}"

            crowded = sorted(
                page for page in frontier if arena.num_owners[page] != 1
            )
            assert not crowded, (
                f"pages still being written into, but not owned alone: {crowded}"
            )

            # a guessed reservation, or a request nothing counts, may take room the
            # others were promised, which the worker's hold absorbs: only pools
            # holding nothing but the model's own counts are checked
            ungated = any(
                rid not in self._reserved and self._held_fresh(rid)
                for rid, overrides in self._overrides.items()
                if overrides.max_tokens is not None
            )
            if not ungated and all(res.exact for res in self._reserved.values()):
                held = {rid: self._held_fresh(rid) for rid in self._reserved}
                over = {
                    rid: (held[rid], res.pages)
                    for rid, res in self._reserved.items() if held[rid] > res.pages
                }
                assert not over, f"requests took more pages than they reserved: {over}"
                if self._leads and self._peak:
                    assert self._admitted_are_safe(), (
                        f"admitted requests cannot all finish: {self._outstanding()} "
                        f"pages are owed, {self._supply()} are free or evictable, and "
                        "no order of finishing them is covered"
                    )
                elif self._leads:
                    outstanding, supply = self._outstanding(), self._supply()
                    assert outstanding <= supply, (
                        f"admitted requests may still take {outstanding} pages, but "
                        f"only {supply} are free or evictable"
                    )

    def post_warmup_validate(self):
        """Assert ``num_free_pages`` is identical across every TP rank

        Catches YAML drift (e.g. ``cpu_offload_pages`` set on one rank
        but not another), allocator-init bugs, and any future code path
        that adds requests asymmetrically before ``warmup`` returns. The
        ``all_gather`` itself is synchronizing, so no extra barrier is
        needed on the success path.
        """
        if self._comm_group.world_size == 1:
            return
        local_free = self._arena.num_free
        local_t = torch.tensor(
            [local_free], dtype=torch.int64, device=self._device,
        )

        for group in [
            self._comm_group.tp_group, self._comm_group.sp_group
        ]:
            gathered = group.all_gather(local_t, dim=0)
            values = gathered.cpu().tolist()
            if any(v != values[0] for v in values):
                raise RuntimeError(
                    f"KV cache {self.name!r} has asymmetric num_free_pages "
                    f"across TP ranks: {values}. v1 requires symmetric "
                    "allocator state; check the YAML for per-rank-divergent "
                    "max_num_pages / cpu_offload_pages, and any model code "
                    "that calls add_request before warmup completes."
                )

    def cleanup(self):
        self._transfer.cleanup()

    def _ensure_label(self, rid: str, label: str) -> CacheStream:
            if label not in self._streams[rid]:
                self._streams[rid][label] = CacheStream()
                self._plan_touch(rid)
                self._seed_keys(rid, label, self._streams[rid][label])
            return self._streams[rid][label]

    # Admission

    @staticmethod
    def _is_padding(rid) -> bool:
        # CUDA-graph padding rows carry negative handles; tests key requests by str
        return isinstance(rid, int) and rid < 0

    def _in_sink(self, rid, ctx: StepContext, label: str) -> bool:
        """Whether what ``rid`` writes to ``label`` goes into the sink.

        A batched replay gives its padding rows their capture span, so each
        would otherwise take a page per label on its first replay and keep it,
        in every bucket, config and slot, which no request could be admitted
        against.
        """
        return (not ctx.capture and self._is_padding(rid)) or self._unheld(rid, label)

    def _unheld(self, rid, label: str) -> bool:
        """Whether ``rid``, counted by its model, holds nothing in ``label``,
        as no walk of its own opens it.

        A model can run a label for a whole batch (Bagel's guidance branches run
        for every row once one row needs them), and a row whose request holds
        nothing there never reads back what it wrote. Pages for it would be
        pages the request never reserved, which counts only the labels it opens.
        """
        overrides = self._overrides.get(rid)
        if overrides is None or not overrides.prompt_slots:
            return False
        opened = self._opened.get(rid)
        if opened is None:
            opened = self._opened[rid] = self._labels_opened(overrides)
        return label not in opened

    def _labels_opened(self, overrides: KVReqConfig) -> set[str]:
        """The labels ``overrides`` can open on this pool's nodes, on any walk."""
        labels = set(overrides.needed_labels or ["main"])
        for (node, _), named in overrides.needed_labels_per_node_walk.items():
            if self._nodes is None or node in self._nodes:
                labels.update(named)
        for node, named in overrides.needed_labels_per_node.items():
            if self._nodes is None or node in self._nodes:
                labels.update(named)
        return labels

    def _reservation(self, rid: str) -> Reservation:
        """The pages ``rid`` may take from the free list over its life here.

        A label counts to its prompt plus what decode adds: the prompt as the
        model counted it, else as its keys describe it. ``max_seq_len`` bounds
        positions, so it caps decode, which takes one per token, and a keyed
        prompt, whose tokens are positions, but not the model's count, whose
        image tokens are not. A page the request leases, or was lent, is held
        rather than taken.
        """
        overrides = self._overrides[rid]
        page_size, cap = self.config.page_size, self.config.max_seq_len
        slots = overrides.prompt_slots or {}
        labels = self._labels_opened(overrides)
        pages = 0
        for label in labels:
            if label in slots:
                decodes = label in (overrides.decode_labels or ())
                tokens = slots[label] + (min(overrides.max_tokens, cap) if decodes else 0)
            else:
                keys = overrides.prefix_keys[label]
                tail = (overrides.prefix_tail or {}).get(label) or ()
                prompt = (len(keys) - (1 if tail else 0)) * page_size + len(tail)
                tokens = min(prompt + overrides.max_tokens, cap)
            pages += -(-tokens // page_size)
            stream = self._streams[rid].get(label)
            if stream is not None:
                pages -= len(stream.lease or ()) + stream.hits
        return Reservation(pages=max(0, pages), exact=bool(slots) and labels <= slots.keys())

    def _gated(self, rid: str, overrides: KVReqConfig | None) -> bool:
        """Whether ``rid`` still has to reserve before it runs.

        Only a request whose every label is counted, by the model or by its
        keys: a bound as loose as ``max_seq_len`` would hold back or refuse
        requests that fit, so the rest run as they always have.
        """
        if overrides is None or overrides.max_tokens is None or rid in self._reserved:
            return False
        slots = overrides.prompt_slots or {}
        keys = overrides.prefix_keys or {}
        # kept by `_gate` for a request that waits: a pass asks about each of them, and the set is fixed
        opened = self._opened.get(rid)
        if opened is None:
            opened = self._labels_opened(overrides)
        return all(label in slots or keys.get(label) for label in opened)

    def _gate(self, rid: str, node_name: str, graph_walk: str) -> AdmitOutcome | None:
        """Hold ``rid`` back until what it may take fits, in the order requests
        first asked. None once it has reserved.

        Nothing passes the head, so short requests never starve a long one
        (unless the pool backfills: then one that does not delay the head may).
        The head's hit is leased as it is admitted, not credited off a lookup
        that an eviction could undo before its prefill runs.
        """
        if self._backfill and self._behind_the_window(rid):
            return ADMIT_WAIT_BEHIND
        overrides = self._overrides.get(rid)
        if not self._gated(rid, overrides):
            return None
        with self._lock:
            if rid not in self._overrides or rid in self._reserved:
                return None
            self._wait_add(rid)
            if self._backfill and rid not in self._opened:
                self._opened[rid] = self._labels_opened(self._overrides[rid])
            head = self._head()
            if head != rid and (not self._backfill or rid not in self._eligible()):
                return ADMIT_WAIT_BEHIND
            if self._planned and self._refused.get(rid) == self._refusal_key(head):
                # nothing it was refused over has moved: skip the probe and the plan
                return ADMIT_WAIT
            label = self._probe_label(rid, node_name, graph_walk)
            keys = None
            if label is not None and rid not in self._rooted:
                keys = overrides.prefix_keys[label]
        # outside the lock, as `resolve_cached_prefix` hashes
        rooted = None if keys is None else [fingerprint(self._prefix_root, key) for key in keys]
        with self._lock:
            if rid not in self._overrides or rid in self._reserved:
                return None
            if rooted is not None:
                self._rooted[rid] = rooted
            stream = None if label is None else self._ensure_label(rid, label)
            return self._try_reserve(rid, stream, self._rooted.get(rid))

    def _try_reserve(
        self, rid: str, stream: CacheStream | None = None,
        rooted: list[bytes] | None = None,
    ) -> AdmitOutcome | None:
        """Reserve for ``rid`` if it heads the queue (or backfills) and fits,
        leasing its hit on ``stream`` as it does. None once it has reserved."""
        self._wait_add(rid)
        head = self._head()
        if head != rid and (not self._backfill or rid not in self._eligible()):
            return _WAIT
        hit = []
        if stream is not None and rooted and self._leasable(stream):
            hit = self._index.peek(rooted)[:len(rooted) - 1]
        reservation = self._reservation(rid)
        held = len(hit) + sum(len(s.lease or ()) + s.hits for s in self._streams[rid].values())
        need = reservation.pages - len(hit)
        capacity = self._capacity()
        if need + held > capacity:
            if reservation.exact:
                self._wait_drop(rid)
                self._rooted.pop(rid, None)
                if self._alog is not None:
                    self._log("refuse", rid, is_head=head == rid, need=need + held, capacity=capacity)
                return AdmitOutcome(ok=False, ready=False, reason=AdmitRuntimeError(
                    f"KV {self.name}: request {rid} needs {need + held} pages, "
                    f"and the pool has {capacity} for requests"
                ))
            # a guess proves nothing unservable: at worst it waits for an empty pool
            need = capacity - held
        if self._planned:
            # the peak test sizes the request in `_ruled_out` and again in `_admissible`: once will do
            cand = self._plan_entry(rid, need, lent=len(hit)) if self._plan_cache and self._peak else None
            if head != rid and self._ruled_out(rid, need, hit, cand):
                self._refused[rid] = self._refusal_key(head)
                return ADMIT_WAIT
            if not self._admissible(rid, head, need, hit, cand):
                return ADMIT_WAIT
        elif self._outstanding() + need > self._supply(leasing=hit):
            return ADMIT_WAIT
        if hit:
            self._lease(rid, stream, rooted)
            stream.gate_lease = True
        reservation.pages = need
        self._reserved[rid] = reservation
        self._reserved_epoch += 1
        self._plan_add(rid)
        self._wait_drop(rid)
        self._rooted.pop(rid, None)
        if self._planned:
            self._reserved_at[rid] = time.monotonic()
        return None

    def _head(self) -> str | None:
        """The request that has waited longest."""
        if self._backfill:
            return next(iter(self._eligible()), None)
        return next(iter(self._waiting), None)

    def _eligible(self) -> dict[str, None]:
        """The requests backfill looks at: the first ``backfill_window`` waiting, head first.

        Kept as requests are added to ``_waiting`` and dropped from it; read
        again from the front of it only once one of these has left.
        """
        window = self._window
        if window is None:
            window = self._window = dict.fromkeys(islice(self._waiting, self._window_size))
        if _DEBUG_ASSERTS:
            assert list(window) == list(islice(self._waiting, self._window_size)), (
                f"KV {self.name}: the backfill window is not the front of the queue"
            )
        return window

    def _behind_the_window(self, rid: str) -> bool:
        with self._lock:
            return rid in self._waiting and rid not in self._eligible()

    def _wait_add(self, rid: str) -> None:
        """Put ``rid`` last in the queue, if it is not in it."""
        waiting = self._waiting
        if rid in waiting:
            return
        waiting[rid] = None
        if not self._backfill:
            return
        if self._alog is not None:
            # from when it asked, not from when it came into the window and was first looked at
            self._asked_at.setdefault(rid, time.monotonic())
        window = self._window
        if window is not None and len(window) < self._window_size:
            # the whole queue was in the window, so this is one more of the first K
            window[rid] = None

    def _wait_drop(self, rid: str) -> None:
        """Take ``rid`` out of the queue, if it is in it."""
        if rid not in self._waiting:
            return
        del self._waiting[rid]
        self._queue_epoch += 1
        if self._backfill and self._window is not None and rid in self._window:
            # the one behind the window takes its place; read from the front when next asked
            self._window = None

    def _reserve_new(self, ctx: StepContext) -> AdmissionDeferred | None:
        """Reserve for a request that reached admit without passing this pool's readiness.

        Where this pool decides, at world size 1, it waits its turn as at
        readiness. Under TP, rank 0 has already sent the step to the other
        ranks, so refusing it would leave them in a forward rank 0 never runs;
        and a pool that does not decide reserves what the deciding one admitted.
        """
        for rid in ctx.request_ids:
            if not self._gated(rid, self._overrides.get(rid)):
                continue
            if self._decides and self._world_size == 1 and not self._in_region(ctx):
                if self._try_reserve(rid) is not None:
                    return AdmissionDeferred(
                        message=f"KV {self.name}: request {rid} has not reserved its pages",
                        request_id=rid,
                    )
                continue
            self._reserved[rid] = self._reservation(rid)
            self._reserved_epoch += 1
            self._plan_add(rid)
            self._wait_drop(rid)
            self._rooted.pop(rid, None)
            if _DEBUG_ASSERTS and self._leads:
                if self._peak:
                    assert self._admitted_are_safe(), (
                        f"KV {self.name}: {rid} was admitted into room this rank does not have"
                    )
                else:
                    assert self._outstanding() <= self._supply(), (
                        f"KV {self.name}: {rid} was admitted into room this rank does not have"
                    )
        return None

    def _probe_label(self, rid: str, node_name: str, graph_walk: str) -> str | None:
        """The label prepare probes for ``rid`` on this walk, if any.

        Only its keyed label's own walk, and only alone: a guided walk writes a
        second label from the same input, and prepare leaves it whole.
        """
        label = self._keyed_label(rid, node_name, graph_walk)
        if label is None or self._prefill_walks.get(label) != graph_walk:
            return None
        if self._overrides[rid].get_labels(node_name, graph_walk) != [label]:
            return None
        return label

    @staticmethod
    def _leasable(stream: CacheStream) -> bool:
        return stream.lease is None and not (
            stream.converted or stream.stored_len or stream.offloaded or stream.read_pending
        )

    @staticmethod
    def _in_region(ctx: StepContext) -> bool:
        # a piecewise region admits inside the forward, where a refusal fails the
        # batch; imported here, as the runner imports this package
        from mstar.engine.cuda_graph_runner import PIECEWISE_WALK

        return ctx.graph_walk == PIECEWISE_WALK

    def _capacity(self) -> int:
        """Pages one request could have with nothing else admitted: all but the
        sink and what a piecewise region's padding rows keep from capture."""
        padding = sum(
            len(stream.page_indices)
            for rid in self._padding
            for stream in self._streams.get(rid, {}).values()
        )
        if _DEBUG_ASSERTS:
            assert padding == sum(
                len(stream.page_indices)
                for rid, streams in self._streams.items() if self._is_padding(rid)
                for stream in streams.values()
            ), f"KV {self.name}: the padding rows kept are not the ones counted"
        return self.config.max_num_pages - 1 - padding

    def _held_fresh(self, rid: str) -> int:
        """Pages ``rid`` holds that it took from the free list."""
        return sum(
            len(stream.page_indices) - stream.hits
            for stream in self._streams.get(rid, {}).values()
        )

    def _supply(self, leasing: list[int] = ()) -> int:
        """Pages an allocation can still get: the free ones, and the cached ones
        eviction reaches, less any about to be leased."""
        if _DEBUG_ASSERTS and self._index is not None:
            assert self._index.evictable() == self._index._find_evictable(), (
                f"KV {self.name}: the evictable pages kept by the index are not the ones walking it finds"
            )
        cached = self._index.evictable_count(leasing) if self._index is not None else 0
        return self._arena.num_free + cached

    def _outstanding(self) -> int:
        """Pages admitted requests may still take.

        Where the pool backfills it is asked for by every request in the window,
        so it is kept until the reserved set, the pages free or a page's owners
        move, which is when the pages any request holds can have.
        """
        if not self._backfill:
            return self._outstanding_afresh()
        arena = self._arena
        key = (self._reserved_epoch, arena.num_free, arena.owner_changes)
        kept = self._owed
        if kept is not None and kept[0] == key:
            if _DEBUG_ASSERTS:
                assert kept[1] == self._outstanding_afresh(), (
                    f"KV {self.name}: the pages owed, kept at {kept[0]}, are no longer {kept[1]}"
                )
            return kept[1]
        owed = self._outstanding_afresh()
        self._owed = (key, owed)
        return owed

    def _outstanding_afresh(self) -> int:
        return sum(
            max(0, res.pages - self._held_fresh(rid))
            for rid, res in self._reserved.items()
        )

    # Admission by peak and by backfill. Only reached when the pool is set to
    # either, or is logging its decisions; the summed test in arrival order
    # (above, and in `_try_reserve`) is what runs otherwise.

    def _room(self) -> int:
        """Pages left to admit into once what the admitted may still take is
        set aside: an upper bound on the ``need`` any request behind the head can
        be admitted with, as the head's protection only takes room away.

        A page granted to an admitted request takes one from the supply and one
        from what it is owed, so the room stays put through every grant that
        stays within a reservation. It is taken again when the reserved set
        changes (a request is reserved, released or lent pages) or a page's
        owners do (pages freed, or cached pages that can now be evicted). Over
        any other change it can only have fallen since: a request that is not
        counted, or one past its reservation, took pages the room still counts.
        So what is kept is never below what there is, and the most it can do is
        have a request worked out in full that could have been refused.
        """
        key = (self._reserved_epoch, self._arena.owner_changes)
        kept = self._room_at
        if kept is None or kept[0] != key:
            kept = self._room_at = (key, self._supply() - self._outstanding())
        if _DEBUG_ASSERTS:
            assert kept[1] >= self._supply() - self._outstanding(), (
                f"KV {self.name}: the room kept at {kept[0]}, {kept[1]}, is below what there is"
            )
        return kept[1]

    def _ruled_out(
        self, rid: str, need: int, hit: list[int], cand: PlanEntry | None = None,
    ) -> bool:
        """Whether ``rid``, behind the head, surely cannot be admitted for ``need`` pages.

        Said from what is kept, without the plan, so a long queue of requests
        that do not fit costs each of them little. Never true of one that the
        full test (`_admissible`) would admit; it is left to say so for the
        rest, and to say why one that waits does.

        Summed, ``need`` is over the room. By peak, the set with ``rid`` added
        is over the pool at a round the planner checks without a pass (see
        `PeakPlanner.exceeds`).

        A pool that logs has the full test write the row a request's first
        wait is, so it is only a request that has one that is ruled out here,
        and then without another.
        """
        if not (self._backfill and self._cheap_refusals):
            return False
        if self._alog is not None:
            state = self._wait_state.get(rid)
            if state is None or state[0]:
                return False
        if not self._peak:
            return need > self._room()
        plan = self._plan_state()
        if cand is None:
            cand = self._plan_entry(rid, need, lent=len(hit))
        return plan.planner.exceeds(cand, self._supply(leasing=hit) + plan.held_total)

    def _refusal_key(self, head: str | None) -> tuple:
        """What a refusal depended on: the reserved set, the pages free, who owns
        which, and who heads the queue. Asked again with all of it unchanged,
        a request gets the same answer, so it is not worked out again."""
        arena = self._arena
        return (self._reserved_epoch, arena.num_free, arena.owner_changes, head)

    def admission_keys(self) -> tuple[int, tuple[int, int, int, int]] | None:
        """What the waits this pool's gate answers depend on, as ``(behind, front)``, for the
        scheduler to skip asking a request again while it has not moved. None where the pool
        does not decide, which answers no wait.

        ``behind``, for ``ADMIT_WAIT_BEHIND``: a request that is neither the head nor in the
        window is held by the queue alone (`_gate` answers before it looks at the pool), and
        the queue moves it only as a request ahead of it leaves (`_queue_epoch`). One that
        arrives goes last, and is ahead of no one.

        ``front``, for ``ADMIT_WAIT``: the head, or a request in the window. This is
        `_refusal_key` with the queue epoch for the head's name, which says as much: who
        heads the queue, and who is in the window, changes only as a request leaves it. The
        rest of what the answer is made of moves one of the other three, or is fixed:

        * the request: its config is fixed once it is ingested. What it holds, a lease or a
          local match, changes through `retain` and `release` of the arena (`owner_changes`).
        * the reserved set, and what each of them may still take: `_reserved_epoch` ticks as
          one is reserved, released, or lent pages. What each holds changes by a grant, which
          takes from the free list (`num_free`), or by a release (`owner_changes`).
        * the supply, the free pages and the cached ones eviction reaches, which is those two
          again: the index takes and gives up a page through `retain` and `release`.
        * the head's hit, the plan, and what the head leaves a request behind it: the same
          state, read through the same counters.

        The planning pool already answers from `_refused` under this key, so there it says
        what is relied on. The summed test in arrival order keeps nothing between asks, and
        for it these are all that ``need > supply - outstanding`` reads.

        Read without the lock, as the scheduler reads it before every scan, and in pieces:
        a key that moved while it was read is stale, which costs an ask, and what it must
        never do is match a later state other than the one the ask saw. The counters only
        rise. The free pages rise only in a `release`, which counts after it has freed, and
        an ask waits for all of it (the lock is held). So with the count read first and the
        free pages last, a key that matches again has seen no release since, and the free
        pages, which only fall otherwise, have not fallen if they match.
        """
        if not self._decides:
            return None
        arena = self._arena
        owners = arena.owner_changes
        reserved = self._reserved_epoch
        queue = self._queue_epoch
        return queue, (reserved, arena.num_free, owners, queue)

    def _life(self, rid: str) -> list[tuple[str, int, int]]:
        """``(label, prompt tokens, decode tokens)`` for each label ``rid`` opens
        here, counted as `_reservation` counts them. Fixed once the request is
        ingested, so kept."""
        life = self._shape.get(rid)
        if life is None:
            overrides = self._overrides[rid]
            page_size, cap = self.config.page_size, self.config.max_seq_len
            slots = overrides.prompt_slots or {}
            life = []
            for label in self._labels_opened(overrides):
                if label in slots:
                    decodes = label in (overrides.decode_labels or ())
                    life.append((
                        label, slots[label],
                        min(overrides.max_tokens, cap) if decodes else 0,
                    ))
                else:
                    keys = overrides.prefix_keys[label]
                    tail = (overrides.prefix_tail or {}).get(label) or ()
                    prompt = (len(keys) - (1 if tail else 0)) * page_size + len(tail)
                    decode = min(prompt + overrides.max_tokens, cap) - prompt
                    life.append((label, prompt, max(0, decode)))
            self._shape[rid] = life
        return life

    def _plan_entry(self, rid: str, claim: int, lent: int = 0) -> PlanEntry:
        """``rid`` as the plan sees it: what it holds, may take, takes at once
        (its prompt, not yet allocated, less ``lent`` pages the index will lend),
        and how many decode rounds it has left on how many labels.

        The rounds are the longest any decode label has left of its tokens, by
        the stream's committed length. A request still to prefill has all of them.
        """
        streams = self._streams.get(rid, {})
        page_size = self.config.page_size
        now = growth = rounds = 0
        for label, prompt, decode in self._life(rid):
            stream = streams.get(label)
            have = stored = 0
            if stream is not None:
                have = len(stream.page_indices) + len(stream.lease or ())
                stored = stream.stored_len
            now += max(0, -(-prompt // page_size) - have)
            if decode:
                left = decode - max(0, stored - prompt)
                if left > 0:
                    growth += 1
                    rounds = max(rounds, left)
        return PlanEntry(
            held=self._held_fresh(rid), claim=claim, now=max(0, now - lent),
            growth=growth, rounds=rounds,
        )

    def _plan_touch(self, rid: str) -> None:
        """Say that what ``rid``'s plan row is made of may have moved: its pages (held,
        leased or lent), its reservation, or the streams it has. Called wherever those are
        written; the stored length is not one of them, as it is read each time (`PlanTable`).
        """
        if self._plan_cache:
            self._plan_dirty.add(rid)

    def _plan_add(self, rid: str) -> None:
        """``rid`` has reserved: a row for it, after the others, as `_reserved` has it."""
        if self._plan_cache:
            if rid not in self._plan_table:
                self._plan_table.append(rid)
            self._plan_dirty.add(rid)

    def _plan_drop(self, rid: str) -> None:
        """``rid`` is gone, or never reserved: no row."""
        if self._plan_cache:
            self._plan_dirty.discard(rid)
            if rid in self._plan_table:
                self._plan_table.drop(rid)

    def _plan_row(self, rid: str) -> tuple[int, int, int, list[tuple[object, int, int]]]:
        """What `_plan_entry` reads of ``rid`` that only moves where `_plan_touch` is called:
        ``held``, ``claim``, ``now`` and the streams of its decode labels with their prompt
        and decode tokens."""
        streams = self._streams.get(rid, {})
        page_size = self.config.page_size
        now = 0
        decoding = []
        for label, prompt, decode in self._life(rid):
            stream = streams.get(label)
            have = 0
            if stream is not None:
                have = len(stream.page_indices) + len(stream.lease or ())
            now += max(0, -(-prompt // page_size) - have)
            if decode:
                decoding.append((ABSENT if stream is None else stream, prompt, decode))
        return self._held_fresh(rid), self._reserved[rid].pages, now, decoding

    def _plan_fields(self) -> tuple[np.ndarray, ...]:
        """The fields of every admitted request's `PlanEntry`, read again for those touched."""
        table = self._plan_table
        for rid in self._plan_dirty:
            if rid in table:
                table.set(rid, *self._plan_row(rid))
        self._plan_dirty.clear()
        if _DEBUG_ASSERTS:
            self._assert_plan_table(table)
        return table.fields()

    def _assert_plan_table(self, table: PlanTable) -> None:
        """Each row of the table is the entry counted afresh, for the same requests in the same order."""
        assert table.rids == list(self._reserved), (
            f"KV {self.name}: the plan has rows for {table.rids}, and {list(self._reserved)} are reserved"
        )
        kept = plan_entries(*table.fields())
        for rid, entry in zip(table.rids, kept, strict=True):
            fresh = self._plan_entry(rid, self._reserved[rid].pages)
            assert entry == fresh, (
                f"KV {self.name}: the plan row kept for {rid}, {entry}, is not what counting "
                f"it afresh gives, {fresh}: something that moves it did not say so"
            )

    def _plan_state(self) -> PlanState:
        """The admitted requests as a plan, kept until the reserved set, the free
        pages or a page's owners change."""
        key = self._refusal_key(None)[:3]
        state = self._plan
        if state is not None and state.key == key:
            return state
        if self._plan_cache:
            return self._plan_from_table(key)
        entries = [self._plan_entry(rid, res.pages) for rid, res in self._reserved.items()]
        state = PlanState(
            key=key, entries=entries, held=[e.held for e in entries],
            need=[e.need for e in entries], held_total=sum(e.held for e in entries),
        )
        if self._peak:
            state.planner = PeakPlanner(
                entries, self._supply() + state.held_total, self.config.page_size,
            )
        self._plan = state
        return state

    def _plan_from_table(self, key: tuple) -> PlanState:
        """`_plan_state` for a plan kept as arrays: the same plan, from the rows the
        requests touched since the last build instead of from every request.

        The arrays are the table's own: it changes only as the reserved set does, which
        moves the key, so they are what this plan was made from for as long as it is kept.
        """
        held, claim, now, growth, rounds = self._plan_fields()
        table = self._plan_table
        # one slot past the requests, for a candidate (`PlanState.with_candidate`)
        held_x, need_x = table.with_spare()
        state = PlanState(
            key=key, entries=None, held=held, need=need_x[:-1], held_total=table.held_total,
            need_total=table.need_total, spare=(held_x, need_x),
        )
        if self._peak:
            # no capacity: every ask names the one it is made against (`peak_from` takes none),
            # with the leased pages it is about to take out of the supply, which a plan has not
            state.planner = PeakPlanner.from_arrays(
                held, claim, now, growth, rounds, 0, self.config.page_size,
            )
        else:
            state.rows = (held, claim, now, growth, rounds)
        self._plan = state
        return state

    def _head_shadow(self, head: str) -> tuple[int, int] | None:
        """When the head of the queue could start, and the room it leaves then."""
        key = (*self._refusal_key(head), self._rooted.get(head) is not None)
        if self._shadow is not None and self._shadow[0] == key:
            return self._shadow[1]
        rooted = self._rooted.get(head)
        hit = []
        if rooted and self._index is not None:
            hit = self._index.peek(rooted)[:len(rooted) - 1]
        claim = min(max(0, self._reservation(head).pages - len(hit)), self._capacity())
        plan = self._plan_state()
        supply = self._supply(leasing=hit)
        if self._peak:
            shadow = plan.planner.shadow(
                self._plan_entry(head, claim, lent=len(hit)), supply + plan.held_total,
            )
        else:
            shadow = shadow_sum(plan.all_entries(), supply, claim)
        self._shadow = (key, shadow)
        return shadow

    def _admissible(
        self, rid: str, head: str, need: int, hit: list[int], cand: PlanEntry | None = None,
    ) -> bool:
        """Whether ``rid`` may reserve ``need`` pages now.

        Fits the pool by the summed test, or by the peak test and a safe state
        with it added; and, behind the head, only if the head is not delayed.
        """
        logged = self._alog is not None
        started = time.perf_counter_ns() if logged else 0
        if logged:
            self._asked_at.setdefault(rid, time.monotonic())
        supply = self._supply(leasing=hit)
        plan = self._plan_state()
        if cand is None:
            cand = self._plan_entry(rid, need, lent=len(hit))
        capacity = supply + plan.held_total
        why = None
        if self._peak:
            planned = plan.planner.peak_from(cand)
            if planned > capacity:
                why = "peak"
            elif not banker_safe(supply, *plan.with_candidate(cand)):
                why = "unsafe"
        else:
            planned = plan.held_total + plan.owed() + need
            if planned > capacity:
                why = "sum"
        if why is None and rid != head:
            rounds = max(1, cand.rounds)
            cost = footprint(cand, rounds - 1, self.config.page_size) if self._peak else need
            if not easy_allows(rounds, cost, self._head_shadow(head)):
                why = "easy"
        if why is None:
            self._refused.pop(rid, None)
            self._wait_state.pop(rid, None)
            if logged:
                self._log(
                    "reserve", rid, is_head=rid == head, planned_peak=planned,
                    plan_capacity=capacity, need=need, hit=len(hit),
                    waited_s=time.monotonic() - self._asked_at.pop(rid, time.monotonic()),
                    planner_us=(time.perf_counter_ns() - started) / 1e3,
                )
            return True
        self._refused[rid] = self._refusal_key(head)
        if logged:
            state = (rid == head, why)
            if self._wait_state.get(rid) != state:
                self._wait_state[rid] = state
                self._log(
                    "wait", rid, is_head=rid == head, why=why, planned_peak=planned,
                    plan_capacity=capacity, need=need,
                    planner_us=(time.perf_counter_ns() - started) / 1e3,
                )
        return False

    def _admitted_are_safe(self) -> bool:
        """Whether the admitted requests, as they stand, can finish one after another."""
        held = [self._held_fresh(rid) for rid in self._reserved]
        need = [max(0, res.pages - h) for res, h in zip(self._reserved.values(), held)]
        return banker_safe(self._supply(), held, need)

    def _defer_grant(self, rid: str, label: str, n: int) -> GrantDeferred | None:
        """Refuse ``n`` pages for ``rid`` if, once granted, the admitted requests
        could no longer all finish. None if the grant is safe.

        A safe state always has a request that can finish with what is free, and
        a grant to that request keeps it safe, so the request that has to
        progress is never the one refused. A grant for more pages than are free
        is refused too, if the state was safe: the others hold what it needs,
        and it waits for them, where an allocation failure would offload someone
        or hold its whole batch.

        A state that was already unsafe is not this test's to mend: something
        the admitted requests do not account for (a request nothing counts, a
        guess that ran over) took the pages, and the grant goes to the paths
        that answer a shortage of pages, as it does when the pool admits by the
        summed test.

        Under TP each rank answers off the requests a step has reached on it,
        which every rank agrees on: a request not yet reached holds nothing, so
        leaving it out only loosens the test.
        """
        started = time.perf_counter_ns() if self._alog is not None else 0
        shared = self._world_size > 1
        supply = self._supply()
        reserved = self._reserved.get(rid) if self._plan_cache else None
        if reserved is not None and max(0, reserved.pages - self._held_fresh(rid), n) <= supply:
            # the request can finish on what is free, grant and all, and gives all it holds
            # back; so a state that was safe is safe after it, and one that was not is
            # not this test's to mend. Either way, no refusal
            return None
        held, need, at = [], [], 0
        for other, res in self._reserved.items():
            if shared and other != rid and other not in self._seen:
                continue
            if other == rid:
                at = len(held)
            pages = self._held_fresh(other)
            held.append(pages)
            need.append(max(0, res.pages - pages))
        owed = need[at]
        held[at] += n
        need[at] = max(0, owed - n)
        if banker_safe(supply - n, held, need):
            return None
        held[at] -= n
        need[at] = owed
        if not banker_safe(supply, held, need):
            return None
        if self._alog is not None:
            self._log(
                "grant_deferred", rid, label=label, pages=n,
                planner_us=(time.perf_counter_ns() - started) / 1e3,
            )
        return GrantDeferred(
            message=(
                f"KV {self.name}: {n} pages for request {rid}, label {label}, would "
                "leave the admitted requests unable to all finish; not before another progresses"
            ),
            label=label, request_id=rid,
        )

    def _forget_admission(self, rid: str) -> None:
        self._reserved_epoch += 1
        self._plan_drop(rid)
        for per_rid in (
            self._refused, self._shape, self._asked_at, self._reserved_at, self._wait_state,
        ):
            per_rid.pop(rid, None)
        self._seen.discard(rid)

    def _log(self, event: str, rid: str, **fields) -> None:
        self._alog.write(
            event, self.name, rid=rid, fit=self._fit, order=self._order,
            supply=self._supply(), n_reserved=len(self._reserved),
            n_waiting=len(self._waiting), **fields,
        )

    def _log_release(self, rid: str) -> None:
        self._log(
            "release", rid, held=self._held_fresh(rid), claim=self._reserved[rid].pages,
            reserved_s=time.monotonic() - self._reserved_at.get(rid, time.monotonic()),
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
            else:
                stream.read_pending = False
        if stream.read_error is not None:
            err = stream.read_error
            return False, AdmitRuntimeError(
                f"KV retrieve for request {rid} label {label!r} failed: "
                f"{type(err).__name__}: {err}"
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
        self, rid: str, from_label: str, to_label: str, undo: list | None = None,
    ) -> None:
        """Copy a stream onto its fork target, over pages `_reserve_fork` took.

        Locked (reentrant): called from plan (pre-forks, else unguarded) and
        from the already-locked commit (post-forks).

        ``undo`` collects each target's prior length and epochs so a preplan
        that is abandoned can be reversed; see `clear_preplan`.
        """
        with self._lock:
            if from_label not in self._streams[rid]:
                return
            from_stream = self._streams[rid][from_label]
            to_stream = self._ensure_label(rid, to_label)
            if undo is not None:
                undo.append(
                    (
                        rid,
                        to_label,
                        to_stream.stored_len,
                        to_stream.generation,
                        to_stream.reset_generation,
                    )
                )
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
            if num_new_pages > 0:
                if self._peak and self._leads and request_id in self._reserved:
                    deferred = self._defer_grant(request_id, label, num_new_pages)
                    if deferred is not None:
                        return AllocResult(success=False, error=deferred)
                new_pages = self._arena.acquire(num_new_pages)
                if new_pages is None and self._index is not None:
                    # cached pages are the only ones that can be given back
                    # without failing a request that is already running
                    self._index.evict(num_new_pages - self._arena.num_free)
                    new_pages = self._arena.acquire(num_new_pages)
                if new_pages is None:
                    pages_short = num_new_pages - self._arena.num_free
                    reserved = self._reserved.get(request_id)
                    if _DEBUG_ASSERTS and self._leads and reserved is not None and reserved.exact:
                        raise AssertionError(
                            f"KV {self.name}: {request_id}/{label} was admitted on "
                            f"a reservation of {reserved.pages} pages and is "
                            f"{pages_short} short"
                        )
                    return AllocResult(
                        success=False,
                        error=AllocationFailed(
                            pages_short=pages_short,
                            request_id=request_id,
                            label=label,
                            message=(
                                f"Not enough free pages: requested {num_new_pages}, "
                                f"available {self._arena.num_free} for request {request_id}, "
                                f"label {label}."
                            ),
                        )
                    )
                stream.page_indices.extend(new_pages)
                stream.generation += 1
                self._plan_touch(request_id)
        return AllocResult()

    ### Submodule-level functionality
    # Label / layer cursors come from `AttentionResource`; the readers resolve
    # them. TODO: rename the `set_layer_idx` call sites and drop this alias.
    set_layer_idx = AttentionResource.set_default_layer_idx

    def reset_default_cursors(self) -> None:
        super().reset_default_cursors()
        # unlike attention, every read here needs a usable index
        self._default_layer_idx = 0

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
    def write_kv(
        self, k: torch.Tensor, v: torch.Tensor,
        layer_idx: int=None, label: str=None, return_tensor: bool = False,
    ) -> torch.Tensor | None:
        """Write K, V into this step's planned slots.

        Returns nothing by default: reading the slots back is a gather no
        caller wants today, and skipping it keeps the write a pure mutation.
        """
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        if label is None:
            label = self._default_label
        plan_state = self._current_plan_states[label]
        n = plan_state.total_tokens
        return self.kv_cache.write_tokens(
            layer_idx=layer_idx,
            k=k[:n], v=v[:n],
            page_idx=plan_state.token_to_page[:n],
            cache_idx=plan_state.token_to_cache[:n],
            return_tensor=return_tensor,
        )
