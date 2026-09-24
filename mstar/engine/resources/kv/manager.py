import logging
import os
import threading
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from typing import Any

import torch

from mstar.distributed.communication import JointGroups
from mstar.engine.resources.base import (
    AttentionResource,
    CGSlotSpec,
    EngineResourceInfo,
    PublishedInfo,
)
from mstar.engine.resources.kv.cache import KVCache, PageAllocator
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVSpec, KVStep
from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool
from mstar.engine.resources.kv.keys import fingerprint, page_key
from mstar.engine.resources.kv.plan import (
    SINK_PAGE,
    KVPlanOutput,
    KVPlanOutputs,
    PagedIndptrs,
    SequenceView,
    build_paged_indptrs,
    group_by_plan_label,
)
from mstar.engine.resources.kv.prefix_index import PrefixIndex
from mstar.engine.resources.kv.transfer import KVTransferManager, TransferEngineInfo
from mstar.engine.resources.step import (
    ADMIT_OK,
    AdmitFailedReason,
    AdmitOutcome,
    AdmitRuntimeError,
    AllocationFailed,
    RequestOffloading,
    Segment,
    StepContext,
)

logger = logging.getLogger(__name__)

# Off by default: `assert_pages_conserved` walks every stream of every live
# request after each admit, commit, reset and remove.
_DEBUG_ASSERTS = os.environ.get("MSTAR_KV_DEBUG_ASSERTS", "0") == "1"


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

    def copy_pages(self, src: list[int], dst: list[int]) -> None:
        self.kv_cache.copy_pages(src, dst)

    @property
    def num_free(self):
        return self.allocator.num_free


@dataclass(frozen=True)
class RetentionPolicy:
    """fifo retention of `context_budget`"""
    context_budget: int
    # front tokens the window never releases; only pages within them are indexed
    protected_prefix: int = 0


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
    released: int = 0
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
    offloaded: bool = False
    generation: int = 0

    # set from a successful admit until commit: an admitted step already holds
    # addressing into these pages, so an offload in that window must not claim
    # them. read by `_claim_for_offload`
    step_in_flight: bool = False

    def reset(self, freed: bool=False):
        self.stored_len = 0
        self.position = 0
        self.released = 0
        self.generation += 1
        self.step_in_flight = False

        if freed:
            self.page_indices.clear()

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


LabelToStream = dict[str, CacheStream]

@dataclass
class KVSequenceInfo:
    seq_len: int
    # for tracking KV cache
    latest_kv_transfer_info: Any
    page_indices: list[int] = field(default_factory=list)


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
        cfg: KVConfig,
        name: str,
        joint_comm_group: JointGroups | None,
        transfer_engine_info: TransferEngineInfo,
        device: torch.device,
        dtype=torch.bfloat16,
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
            transfer_engine_info, self.kv_cache
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
        self._cached_plan_output: dict[str, KVPlanOutput] | None = None

        # (rid, to_label, stored_len, generation) for pre-forks appliedb by
        # a staged step; facilitates clear_preplan function
        self._preplan_fork_undo: list[tuple[str, str, int, int]] = []
        # (rid, to_label) for intialized reservations by staged step; cleared_preplan removes
        # recorded in admit not plan, so separate from above.
        self._preplan_new_labels: list[tuple[str, str]] = []
        # (rid, label) marked step_in_flight by a staged step, so an abandoned
        # one does not leave its streams unevictable
        self._preplan_marked: list[tuple[str, str]] = []

    @classmethod
    def build(cls, spec: KVSpec, info: EngineResourceInfo):
        return cls(
            cfg=spec.config,
            name=spec.resource_key,
            device=info.device,
            joint_comm_group=info.joint_comm_group,
            transfer_engine_info=info.transfer_engine_info,
            dtype=info.kv_dtype,
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
            # one key short, so a fully cached prompt still leaves a token to run
            matched = self._index.lookup(rooted)[:len(rooted) - 1]
            if not matched:
                return None
            self._arena.retain(matched)
            stream.lease = matched
            return len(matched) * self.config.page_size

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
            keep = matched_len // self.config.page_size
            if keep < len(stream.lease):
                self._arena.release(stream.lease[keep:])
                stream.lease = stream.lease[:keep] or None

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
            for label, stream in self._streams[rid].items():
                if stream.chain is None:
                    self._seed_keys(rid, label, stream)

    def admit_retrieve(
        self, rid: str,
        node_name: str,
        graph_walk: str,
        published: PublishedKVInfo | None
    ) -> AdmitOutcome:
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
                own = seq_info.latest_kv_transfer_info == self._own_transfer_info()
                if not own and stream.chain is not None:
                    # another rank sampled the token after this record, so the pending tail is one behind
                    stream.chain.unkeyed = None
                new_len = seq_info.seq_len
                self._take_local_match(stream, new_len, rooted.get(label))
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
            # pages were already reserved by the preplan pass
            return ADMIT_OK
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
                    # drop the empty pages a refused batch admit left, before taking the lease
                    self._arena.release(stream.page_indices)
                    stream.page_indices = list(stream.lease)
                    stream.stored_len = (
                        len(stream.lease) * self.config.page_size
                    )
                    # they came out of the index, so they are already in it
                    stream.chain.cursor = len(stream.lease)
                    stream.lease = None
                    stream.converted = True
                self._report_admission(segment, stream)
            if ctx.is_preplan:
                # a preplan that was promoted or abandoned already cleared
                # these; reset anyway so a refused admit can't leave stale
                # entries for the next clear_preplan to act on
                self._preplan_new_labels = []
                self._preplan_marked = []
            for (from_label, to_label), extra in forks:
                for rid in ctx.padded_request_ids:
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
                if (
                    ctx.is_preplan
                    and segment.label not in self._streams.get(
                        segment.request_id, {}
                    )
                ):
                    # `_ensure_label` below makes this label. Record it
                    # here, like the fork loop above. If not, `clear_preplan`
                    # keeps the stream and its pages.
                    self._preplan_new_labels.append(
                        (segment.request_id, segment.label)
                    )
                stream = self._ensure_label(segment.request_id, segment.label)
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
        # TODO: apply retention policy

        return ADMIT_OK

    def _sequence_views(self, segments: list[Segment]) -> list[SequenceView]:
        views = []
        page_size = self.kv_cache.page_size
        for s in segments:
            stream = self._streams.get(s.request_id, {}).get(s.label)
            if stream is None:
                # `admit` reserves no stream for a zero-span segment. The
                # segment reads and writes no token, so its view is empty and
                # holds no page. A padding row then addresses SINK_PAGE.
                assert s.span == 0, (
                    f"segment {s.request_id}/{s.label!r} spans {s.span} "
                    "tokens, but admit reserved no stream for it"
                )
                views.append(SequenceView(
                    request_id=s.request_id, label=s.label,
                    page_idxs=[], length=0, to_compute=0,
                ))
                continue
            # `page_indices` is a high-water mark, so a stream can hold more
            # pages than its tokens need (a refused admit, a reset that kept
            # its pages). slice to the length or the view addresses token 0
            # into the wrong page and reports the padding as resident context
            length = s.span + stream.stored_len
            num_pages = -(-length // page_size)
            views.append(SequenceView(
                request_id=s.request_id,
                label=s.label, page_idxs=stream.page_indices[:num_pages],
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
            self._current_plan_states = self._preplan_states
            res = self._cached_plan_output
            # promotion, not abandonment: the staged forks and marks are kept,
            # so drop the undo records before clear_preplan replays them
            self._preplan_fork_undo = []
            self._preplan_new_labels = []
            self._preplan_marked = []
            # must reset here: otherwise the *next* step's admit still sees
            # `_preplanned` and skips its allocation
            self.clear_preplan()
            return res
        undo = self._preplan_fork_undo if ctx.is_preplan else None
        for (from_label, to_label) in step.pre_forks:
            for rid in ctx.padded_request_ids:
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
            self._preplanned = True
            self._cached_plan_output = res
        return res

    @property
    def supports_preplan(self):
        return True

    def clear_preplan(self):
        # the staged step is not going to run, so undo what it did to live
        # state: dropping the cached plan is not enough, the pre-forks already
        # copied pages and moved lengths
        with self._lock:
            for rid, label, stored_len, generation in reversed(
                self._preplan_fork_undo
            ):
                stream = self._streams.get(rid, {}).get(label)
                if stream is not None:
                    stream.stored_len = stored_len
                    stream.generation = generation
            self._preplan_fork_undo = []
            # labels the staged step invented are removed, not rewound to 0:
            # a stream at 0 that nothing asked for is still a stream, and it
            # holds the pages the reservation took
            for rid, label in reversed(self._preplan_new_labels):
                stream = self._streams.get(rid, {}).pop(label, None)
                if stream is not None:
                    self._arena.release(stream.page_indices)
            self._preplan_new_labels = []
            for rid, label in self._preplan_marked:
                stream = self._streams.get(rid, {}).get(label)
                if stream is not None:
                    stream.step_in_flight = False
            self._preplan_marked = []
        # rebind rather than clear: a consumed preplan dict is the live one
        self._preplanned = False
        self._preplan_states = {}
        self._cached_plan_output = None

    def commit(self, step: KVStep, ctx: StepContext):
        # atomic against admit_retrieve reading stored_len on another thread
        with self._lock:
            for segment in step.segments:
                stream = self._streams.get(
                    segment.request_id, {}
                ).get(segment.label)
                if stream is None:
                    # `admit` reserves no stream for a zero-span segment.
                    # There is no mark to clear and no span to commit.
                    continue
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
                        continue
                    stream.stored_len += segment.span
                    # committed, so there is no refused admit left for a re-probe to answer
                    stream.converted = False
                    self._index_filled_pages(segment, stream, ctx)
                    # so a claim taken in a window the mark misses still fails
                    # `_commit_offload`'s generation guard
                    stream.generation += 1
            # post-forks copy what this step just wrote, so they land after the
            # spans above are counted
            for (from_label, to_label) in step.post_forks:
                for rid in ctx.padded_request_ids:
                    self._apply_fork(rid, from_label, to_label)
            if _DEBUG_ASSERTS:
                self.assert_pages_conserved()
        # TODO: handle retention policy, free pages if not commit

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
                self._arena.release(claim.pages)
                stream.page_indices = []
                stream.reset()
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
                stream.offloaded = False
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

    def _own_transfer_info(self):
        """This cache's transfer descriptor, as `publish` stamps it."""
        return self._transfer.get_kv_transfer_info()

    def publish(self, request_id: str):
        # `remove_request` can pop the streams from another thread between the
        # forward and finalize; nothing to publish then
        streams = self._streams.get(request_id)
        if streams is None:
            return None

        transfer_info = self._own_transfer_info()
        with self._lock:
            seq_info = {
                label: KVSequenceInfo(
                    seq_len=stream.stored_len,
                    latest_kv_transfer_info=transfer_info,
                    page_indices=list(stream.page_indices),
                ) for label, stream in streams.items()
            }
        return PublishedKVInfo.build_for_rank(
            rank=self._rank, world_size=self._world_size, seq_info=seq_info,
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
            if rid in self._streams:
                for stream in self._streams[rid].values():
                    self._release_lease(stream)
                    self._arena.release(stream.page_indices)
            if self._cpu_pool is not None:
                self._cpu_pool.remove_request(rid)
            self._streams.pop(rid, None)
            self._overrides.pop(rid, None)
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
                self._seed_keys(rid, label, self._streams[rid][label])
            return self._streams[rid][label]

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

        ``undo`` collects each target's prior ``(stored_len, generation)`` so a
        preplan that is abandoned can be reversed; see `clear_preplan`.
        """
        with self._lock:
            if from_label not in self._streams[rid]:
                return
            from_stream = self._streams[rid][from_label]
            to_stream = self._ensure_label(rid, to_label)
            if undo is not None:
                undo.append(
                    (rid, to_label, to_stream.stored_len, to_stream.generation)
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
                new_pages = self._arena.acquire(num_new_pages)
                if new_pages is None and self._index is not None:
                    # cached pages are the only ones that can be given back
                    # without failing a request that is already running
                    self._index.evict(num_new_pages - self._arena.num_free)
                    new_pages = self._arena.acquire(num_new_pages)
                if new_pages is None:
                    pages_short = num_new_pages - self._arena.num_free
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
