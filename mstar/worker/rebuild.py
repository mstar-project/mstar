"""What a worker keeps so a request dropped from a node's KV can be rebuilt there, as the node's
`RebuildPolicy` declares, and how far each rebuild has run."""

from __future__ import annotations

import logging
from array import array
from collections import defaultdict, deque
from collections.abc import Collection, ItemsView, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.resources import Rebuild, StepAllowance
from mstar.engine.resources.kv.config import SpanKind, StreamLayout, rebuild_refusal
from mstar.graph.base import GraphSection
from mstar.model.rebuild import RebuildPolicy
from mstar.utils.h2d import to_device_nonblocking
from mstar.worker.host_pool import HostPool, HostSlice, aligned

if TYPE_CHECKING:
    from mstar.engine.engine import ExecutingBatch
    from mstar.model.base import KVLabel
    from mstar.model.submodule_base import HostRows

logger = logging.getLogger(__name__)

# The serving config's ``rebuild_host_mb`` unless it sets one: 8M prompt ids
# at 8 B each.
DEFAULT_REBUILD_HOST_MB = 64
# The same where a node copies what a walk reads besides the ids. A 1024-token image at hidden size 4096 is 8 MiB of
# bf16 encoder output.
DEFAULT_REBUILD_MEDIA_HOST_MB = 1024


class LayoutRow(NamedTuple):
    """One prefill step of a request on a node, as its layout says: the walk it runs and the slots it writes."""
    walk: str
    length: int


@dataclass(slots=True)
class RowRecord:
    """One prefill step of a request as it ran, kept so its rebuild can run the step again: host copies of every
    input the node read in it, as the step read it, in the request's slice, and the step's metadata. ``queued``
    counts the copies that haven't started, and ``landed`` is recorded after the last that has, None where the
    copies run in place."""
    inputs: NameToTensorList
    step_metadata: dict
    queued: int = 0
    landed: torch.Event | None = None

    def ready(self) -> bool:
        return not self.queued and (self.landed is None or self.landed.query())


@dataclass(slots=True)
class KeptIds:
    """A request's log on one node: the rows of its layout's prefill steps, a record of each step that has run, in
    order, and each id sampled once the last has. ``used`` counts the bytes of its slice the records take, and
    ``laid_out`` the samples its KV's layout already holds, which its last rebuild wrote."""
    rows: tuple[LayoutRow, ...]
    records: list[RowRecord] = field(default_factory=list)
    sampled: array = field(default_factory=lambda: array("i"))
    used: int = 0
    laid_out: int = 0


class QueuedCopy(NamedTuple):
    """A device tensor's bytes still to be copied into ``into`` once ``after`` completes, for ``record``."""
    src: torch.Tensor
    into: HostSlice
    after: torch.Event | None
    record: RowRecord


class Prefilling(NamedTuple):
    """A prompt mid-prefill on a node: the walk it prefills in, the tokens its chunks have run, and its length."""
    walk: str
    done: int
    total: int


class PlannedRow(NamedTuple):
    """One row of a rebuild: the walk it runs, the slots it writes (``total``) of an input ``length`` long, and the
    record it reads. A prompt's step in progress has no record: it reads the input the engine keeps for the prompt,
    up to where the prompt stood."""
    walk: str
    total: int
    length: int
    record: RowRecord | None


class RowRun(NamedTuple):
    """A rebuild row that has run, and the slots its chunks ran, which leave out a cached prefix."""
    row: PlannedRow
    ran: int


@dataclass(slots=True)
class RebuildCursor:
    """A rebuild under way: its rows, the one running (``row``), the end of that row's chunks that have landed, 0
    before the first, and the slots they ran (``ran``). The last row read from a record also writes every id
    ``sampled`` since but the newest, which the next decode step reads."""
    rows: tuple[PlannedRow, ...]
    sampled: array
    row: int = 0
    done: int = 0
    ran: int = 0

    @property
    def walk(self) -> str:
        return self.rows[self.row].walk

    @property
    def total(self) -> int:
        return self.rows[self.row].total

    @property
    def length(self) -> int:
        return self.rows[self.row].length

    @property
    def resumes_prefill(self) -> bool:
        """Whether the row running is its prompt's step in progress, which goes on past the rebuild."""
        return self.total < self.length


def policies_run_here(
    policies: Mapping[str, RebuildPolicy], graphs: Mapping[str, GraphSection], my_walks: Collection[tuple[str, str]],
    parallel_nodes: Collection[str],
) -> dict[str, RebuildPolicy]:
    """The policies whose rebuild this worker runs, each with the walks this worker runs its node in: it runs the
    node in every walk its loop grows the KV in, and in one of the policy's walks. A disaggregated decode node can't
    run the prefill walk its rebuild needs.

    Above one rank a policy keeps only the walks with nothing to copy, since rank 0 decides a drop from its own host
    copies, and a follower's may not have landed.
    """
    here = {}
    for node, policy in policies.items():
        copied = policy.check(node, graphs)
        walks = frozenset(
            walk for walk in policy.walks
            if (node, walk) in my_walks and not (copied[walk] and node in parallel_nodes)
        )
        if walks and all((node, walk) in my_walks for walk in policy.loop_walks(node, graphs)):
            here[node] = replace(policy, walks=walks)
    return here


class NodeWrites(NamedTuple):
    """The walks a node runs on this worker, and the loops that grow its paged KV cache's labels in them."""
    walks: frozenset[str]
    loops: frozenset[str]

    def writes(self, layout: StreamLayout) -> bool:
        return any(span.walk in self.walks for span in layout.spans) or any(
            loop in self.loops for loop, _ in layout.growth
        )


def node_writes(
    labels: Mapping[str, KVLabel], nodes: Iterable[str], my_walks: Collection[tuple[str, str]],
) -> dict[str, NodeWrites]:
    """What each of a paged KV cache's ``nodes`` that runs here writes of the cache's ``labels``."""
    writes = {}
    for node in nodes:
        walks = frozenset(walk for name, walk in my_walks if name == node)
        if walks:
            loops = frozenset(loop for label in labels.values() for walk, loop in label.growth if walk in walks)
            writes[node] = NodeWrites(walks, loops)
    return writes


class Rebuilds:
    """The log of each request on the nodes that declare a `RebuildPolicy`, checked against the graph when the
    worker starts, and the rebuilds under way. Host copies of what a step read besides ids spend the step's
    ``allowance`` (`flush`)."""

    def __init__(
        self, policies: Mapping[str, RebuildPolicy], graphs: Mapping[str, GraphSection],
        pool: HostPool | None = None, allowance: StepAllowance | None = None,
        node_dtypes: Mapping[str, torch.dtype] = MappingProxyType({}),
    ) -> None:
        # node -> walk -> what the node reads there besides the ids, which the worker copies to the host
        self._copied = {node: policy.check(node, graphs) for node, policy in policies.items()}
        # node -> the walks that read the ids, the only ones an ids span is written in
        self._ids_walks = {node: policy.ids_walks(node, graphs) for node, policy in policies.items()}
        self._policies = dict(policies)
        # node -> the dtype it runs in, which its host copies take. A node missing here copies nothing.
        self._node_dtypes = node_dtypes
        self._loop_walks = {node: policy.loop_walks(node, graphs) for node, policy in policies.items()}
        assert pool is not None or not policies, "a node that rebuilds requests keeps their ids in the host pool"
        self._pool = pool
        # node -> handle -> its log. Per node, since a colocated node of another partition samples ids of its own.
        self._kept: dict[str, dict[int, KeptIds]] = {node: {} for node in policies}
        # node -> the handles whose log hasn't recorded every prefill step yet
        self._prefilling: dict[str, set[int]] = {node: set() for node in policies}
        # node -> the handles whose pages a drop there can free, each with the slice of the pool its log takes. Only
        # these keep a log.
        self._droppable: dict[str, dict[int, HostSlice]] = {node: {} for node in policies}
        # (node, handle) -> its rebuild, from the readiness check that starts it until its last chunk lands
        self._cursors: dict[tuple[str, int], RebuildCursor] = {}
        assert allowance is not None or not any(names for walks in self._copied.values() for names in walks.values()), (
            "host copies spend the engine's StepAllowance"
        )
        self._allowance = allowance
        # the copies to the host not yet started, oldest first, and the event recorded after the last one `flush`
        # started, None before one or where copies run in place
        self._queued: deque[QueuedCopy] = deque()
        self._last_landed: torch.Event | None = None
        # nodes that have logged a request the pool couldn't hold
        self._logged_full: set[str] = set()

    @property
    def pool(self) -> HostPool | None:
        return self._pool

    @property
    def nodes(self) -> frozenset[str]:
        """The nodes whose requests can be rebuilt here."""
        return frozenset(self._policies)

    def kept(self, rid: int, node: str) -> KeptIds | None:
        return self._kept.get(node, {}).get(rid)

    def can_rebuild(self, node: str, layouts: Iterable[StreamLayout]) -> bool:
        """Whether a rebuild on ``node`` can write a request's laid-out labels again: the node declares a policy,
        and its loop grows labels its rebuild rows can lay out again, written in the policy's walks."""
        policy, layouts = self._policies.get(node), list(layouts)
        return (
            policy is not None and rebuild_refusal(layouts) is None
            and all(loop == policy.loop for layout in layouts for loop, _ in layout.growth)
            and all(
                span.walk in policy.walks
                and (span.walk in self._ids_walks[node] if span.kind == SpanKind.IDS else self._copied[node][span.walk])
                for layout in layouts for span in layout.spans
            )
        )

    def _log_room(self, node: str, layouts: Iterable[StreamLayout]) -> HostSlice | None:
        """The slice ``node``'s log takes for a request of ``layouts``, holding its steps' ids, at most one a slot, and
        the host copies for its spans other than ids. None when the pool can't hold it, or when the request has spans
        other than ids and the node's dtype is unknown.

        The ids' dtype isn't known until a step reads them, so each takes 8 B, enough for int64, the widest dtype a
        step reads ids in. A copied span takes room for ids too, since a step may read a pad id at each slot its host
        copies fill. What the steps don't use goes back to the pool once the last is recorded."""
        spans = [span for layout in layouts for span in layout.spans]
        dtype, width = self._node_dtypes.get(node), self._policies[node].copy_width
        if dtype is None and any(span.kind != SpanKind.IDS for span in spans):
            return None
        nbytes = sum(
            aligned(span.length * torch.long.itemsize)
            + (0 if span.kind == SpanKind.IDS else span.length * width[span.walk] * dtype.itemsize)
            for span in spans
        )
        piece = self._pool.take(nbytes)
        if piece is None and node not in self._logged_full:
            self._logged_full.add(node)
            logger.warning(
                "%s: the rebuild host pool has no room for the %d bytes a request's ids and host copies take, with %d "
                "of %d bytes in use, so the request can't be dropped there. Only the first on each node is logged, "
                "and the serving config's rebuild_host_mb sizes the pool.",
                node, nbytes, self._pool.nbytes_used(), self._pool.nbytes,
            )
        return piece

    def ingest_request(
        self, rid: int, writes: Mapping[str, Mapping[str, NodeWrites]],
        layouts: Mapping[str, Collection[StreamLayout]], unpausable: bool,
    ) -> dict[str, bool]:
        """For each paged KV cache in ``writes``, whether a drop here can free ``rid``'s pages there: rebuild rows
        can lay out the request's ``layouts`` in that cache again, and every node that writes those layouts can redo
        all of its writes. A node that writes nothing for the request isn't asked.

        A drop takes every cache of its node, so a node is judged by what it writes in all of them. A node keeps a
        log of ``rid`` only when a drop can free ``rid``'s pages in every cache the node writes, and ``unpausable``
        is False. A node takes the pool's room for the log now, so a request the pool can't hold is known before it is
        admitted and is never dropped on that node. The log's rows are the layout's prefill steps
        (`StreamLayout.steps`)."""
        written: dict[str, list[StreamLayout]] = defaultdict(list)
        writers: dict[str, list[str]] = defaultdict(list)
        for key, nodes in writes.items():
            for node, what in nodes.items():
                laid_out = [layout for layout in layouts.get(key, ()) if what.writes(layout)]
                if laid_out:
                    written[node] += laid_out
                    writers[key].append(node)
        refused = {node for node, laid_out in written.items() if not self.can_rebuild(node, laid_out)}
        taken = {}
        if not unpausable:
            for node in written.keys() - refused:
                piece = self._log_room(node, written[node])
                if piece is None:
                    refused.add(node)
                else:
                    taken[node] = piece
        rebuildable = {
            key: refused.isdisjoint(writers[key]) and rebuild_refusal(layouts.get(key, ())) is None for key in writes
        }
        for node, piece in taken.items():
            if all(rebuildable[key] for key, nodes in writers.items() if node in nodes):
                self._droppable[node][rid] = piece
                self._kept[node][rid] = KeptIds(tuple(
                    LayoutRow(step[0].walk, sum(span.length for span in step))
                    for layout in written[node] for step in layout.steps()
                ))
                self._prefilling[node].add(rid)
            else:
                self._pool.give(piece)
        return rebuildable

    def record(
        self, node_batch: ExecutingBatch, cached_infos: Mapping[int, CurrentForwardPassInfo] = MappingProxyType({}),
    ) -> None:
        """Record each row of the step that runs the next prefill step of a request still prefilling here, on the
        row's first chunk, which carries the step's whole input: its ids into the request's slice once the step
        completes, its other inputs queued to follow them (`flush`), and its step metadata. Once the last step is
        recorded the request keeps its samples, and the slice's unused tail goes back to the pool.

        A row the prefix cache served whole ran no forward, but a rebuild may find its pages evicted, so it is
        recorded from its inputs and its info in ``cached_infos``, which the engine takes out of the step.

        A step other than the one the request's layout says comes next, by walk or by length, or one missing an input
        the node reads, as a row the prefix cache served past its encoder, leaves the request with no log, so it is
        never dropped."""
        node = node_batch.node_name
        prefilling = self._prefilling.get(node)
        if not prefilling:
            return
        # a step of no walk that prefills, such as every decode step, has no row to record
        ctx, walks = node_batch.step_context, self._copied[node]
        if ctx.graph_walk not in walks and walks.keys().isdisjoint(ctx.request_walks.values()):
            return
        ids_input = self._policies[node].ids_input
        # The rows that ran or that the cache served: the engine takes a row it skipped or failed out of the step's
        # rows and their infos, but not out of its inputs.
        for rid in (*node_batch.request_ids, *node_batch.cached_rids):
            if rid not in prefilling or rid in ctx.rebuild_rids:
                continue
            inputs = node_batch.per_request_input_tensors.get(rid)
            chunk, walk = node_batch.chunk_ranges.get(rid), ctx.walk_of(rid)
            # A later chunk is cut from the input its first prepared, and a TP follower's unread row has no inputs,
            # which the step fails.
            if not inputs or walk not in walks or not (chunk is None or chunk.is_first):
                continue
            kept, piece = self._kept[node][rid], self._droppable[node][rid]
            row = kept.rows[len(kept.records)]
            names = sorted({*([ids_input] if walk in self._ids_walks[node] else ()), *self._copied[node][walk]})
            tensors = [
                (name, tensor.reshape(-1) if name == ids_input else tensor)
                for name in names for tensor in inputs.get(name, ())
            ]
            nbytes = sum(aligned(tensor.nbytes) for _, tensor in tensors)
            if (
                walk != row.walk or (chunk is not None and chunk.total != row.length)
                or not all(name in inputs for name in names)
            ):
                self._forget(node, rid)
                continue
            assert kept.used + nbytes <= piece.nbytes, (
                f"{node} read more in {walk!r} than its rebuild policy's copy_width says"
            )
            info = cached_infos[rid] if rid in node_batch.cached_rids else node_batch.per_request_info[rid]
            record = RowRecord({name: [] for name in names}, dict(info.step_metadata))
            for name, tensor in tensors:
                into = HostSlice(piece.offset + kept.used, tensor.nbytes)
                # The ids are few, so they go now. The copies of the other inputs share the step's host-write allowance.
                if name == ids_input:
                    copy = self._pool.copy_out(tensor, into, node_batch.completion_event)
                    record.inputs[name].append(copy.view)
                    record.landed = copy.landed
                else:
                    record.inputs[name].append(self._pool.view(into, tensor.dtype, tensor.shape))
                    if not tensor.nbytes:
                        # an empty copy spends no allowance, so `flush` would never take it off the queue
                        continue
                    self._queued.append(QueuedCopy(
                        tensor.reshape(-1).view(torch.uint8), into, node_batch.completion_event, record,
                    ))
                    record.queued += 1
                kept.used += aligned(into.nbytes)
            kept.records.append(record)
            if len(kept.records) == len(kept.rows):
                prefilling.discard(rid)
                self._trim(node, rid, kept.used)

    def flush(self) -> None:
        """Start the queued copies to the host, oldest first, up to what the step's allowance has left, once the
        step's outputs are sent or on a pass of the worker's loop that runs no step (`flush_idle`). A copy that runs
        past the allowance is split there, and the rest waits for the next allowance. The device tensor a queued copy
        reads stays held until its copy starts."""
        while self._queued:
            queued = self._queued[0]
            nbytes = self._allowance.spend(queued.src.numel())
            if not nbytes:
                return
            copy = self._pool.copy_out(queued.src[:nbytes], HostSlice(queued.into.offset, nbytes), queued.after)
            queued.record.landed = self._last_landed = copy.landed
            if nbytes < queued.src.numel():
                self._queued[0] = queued._replace(
                    src=queued.src[nbytes:], into=HostSlice(queued.into.offset + nbytes, queued.into.nbytes - nbytes),
                )
                return
            self._queued.popleft()
            queued.record.queued -= 1

    def flush_idle(self) -> None:
        """`flush` on a pass of the worker's loop that runs no step, with a step's allowance once the last copy it
        started has landed. A drop waits for the queued copies, so with every request waiting, no step would ever
        start them. Waiting for the last copy keeps a burst of such passes from queuing more than one allowance."""
        if not self._queued:
            return
        if self._last_landed is not None:
            self._last_landed.synchronize()
        self._allowance.refill()
        self.flush()

    @property
    def copies_queued(self) -> bool:
        return bool(self._queued)

    def _trim(self, node: str, rid: int, used: int) -> None:
        """Give back what ``rid``'s slice on ``node`` holds past its first ``used`` bytes."""
        piece = self._droppable[node][rid]
        if used < piece.nbytes:
            self._pool.give(HostSlice(piece.offset + used, piece.nbytes - used))
            self._droppable[node][rid] = HostSlice(piece.offset, used)

    def _forget(self, node: str, rid: int) -> None:
        kept = self._kept[node].pop(rid, None)
        self._prefilling[node].discard(rid)
        if kept is not None and any(record.queued for record in kept.records):
            # its slice goes back to the pool, so no copy may start into it
            records = {id(record) for record in kept.records}
            self._queued = deque(queued for queued in self._queued if id(queued.record) not in records)
        piece = self._droppable[node].pop(rid, None)
        if piece is not None:
            self._pool.give(piece)

    def keep_samples(
        self, node_batch: ExecutingBatch, cpu_outputs: Mapping[int, Mapping[str, list[torch.Tensor]]],
        host_rows: HostRows | None, skip_rids: Collection[int] = (),
    ) -> None:
        """Append each kept row's sample, from the stop check's host copy, to its log, once its log holds every
        prefill step, or forget the log when the sample is one of the policy's ``pad_ids``. ``skip_rids`` are rows
        whose sample is not real, as for `Engine.step_finished`."""
        node = node_batch.node_name
        kept_ids = self._kept.get(node)
        if not kept_ids:
            return
        name = self._policies[node].sample_output
        ctx = node_batch.step_context
        if host_rows is not None:
            tokens = host_rows.buffers.get(name)
            if not torch.is_tensor(tokens) or tokens.dim() == 0:
                return
            if tokens.dim() > 1 and tokens[0].numel() != 1:
                raise ValueError(self._not_one_id(node, name, tokens[0].numel()))
            # the sampler's [B] output skips the reshape, which costs more than the rest of this method
            values = tokens.tolist() if tokens.dim() == 1 else tokens.reshape(-1).tolist()
            pairs = zip(host_rows.request_ids, values, strict=False)
        else:
            pairs = []
            for rid in node_batch.request_ids:
                out = cpu_outputs.get(rid)
                if not isinstance(out, Mapping) or not out.get(name):
                    continue
                sample = out[name][0]
                if sample.numel() != 1:
                    raise ValueError(self._not_one_id(node, name, sample.numel()))
                pairs.append((rid, int(sample)))
        prefilling, pads = self._prefilling[node], self._policies[node].pad_ids
        for rid, token in pairs:
            kept = kept_ids.get(rid)
            # The host rows are the forward's, which can hold a row the worker has since taken out of the step. A
            # prefill step before the last draws a token the model throws away.
            if (
                kept is not None and ctx.is_real_row(rid) and rid not in ctx.rebuild_rids and rid not in skip_rids
                and rid not in prefilling
            ):
                if token in pads:
                    self._forget(node, rid)
                else:
                    kept.sampled.append(token)

    @staticmethod
    def _not_one_id(node: str, name: str, count: int) -> str:
        return (
            f"{node}'s rebuild policy keeps samples from {name!r}, whose rows hold {count} ids; a rebuild row "
            "prefills one id per sample"
        )

    def refusal(
        self, rid: int, node: str, walk: str, loop_iters: Mapping[str, int], prefilling: Prefilling | None,
    ) -> str | None:
        """Why ``rid``, in ``walk``, can't be dropped on ``node`` now, or None. Its rebuild reruns each prefill step
        recorded before the one in progress from its record, which has to be on the host. A request ``prefilling``
        its prompt reruns the step in progress from the input the engine keeps for the prompt. Any other reruns
        every step its layout lays out, and its samples cover its KV when the prefill and each iteration of the
        policy's loop sampled one id, since the KV holds all but the last."""
        policy = self._policies.get(node)
        if policy is None:
            return f"{node} declares no rebuild policy"
        kept = self._kept[node].get(rid)
        if kept is None:
            return f"it keeps no log on {node}"
        if prefilling is None and len(kept.records) < len(kept.rows):
            return "its prefill hasn't run every step its layout lays out"
        if not all(record.ready() for record in (kept.records[:-1] if prefilling is not None else kept.records)):
            return "its prefill steps' inputs haven't reached the host"
        if prefilling is not None:
            return None
        if walk not in self._loop_walks[node]:
            return f"it is in walk {walk!r}, where loop {policy.loop!r} doesn't run {node}"
        if len(kept.sampled) != loop_iters.get(policy.loop, 0) + 1:
            return f"its {len(kept.sampled)} kept samples don't cover its KV"
        return None

    def _plan(self, kept: KeptIds, prefilling: Prefilling | None) -> list[PlannedRow]:
        """The rows of a rebuild from ``kept``: one per prefill step recorded before the one in progress, from its
        record, then the chunks its prompt was ``prefilling`` ran. With no step in progress, one per prefill step
        its layout lays out, the last also writing every sample but the newest."""
        records = kept.records if prefilling is None else kept.records[:-1]
        rows = [
            PlannedRow(row.walk, row.length, row.length, record)
            for row, record in zip(kept.rows[:len(records)], records, strict=True)
        ]
        if prefilling is not None:
            rows.append(PlannedRow(prefilling.walk, prefilling.done, prefilling.total, None))
        else:
            grown = len(kept.sampled) - 1
            rows[-1] = rows[-1]._replace(total=rows[-1].total + grown, length=rows[-1].length + grown)
        return rows

    def tokens(self, rid: int, node: str, prefilling: Prefilling | None) -> int:
        """How many slots ``rid``'s rebuild on ``node`` writes, over every row `start` opens it with."""
        return sum(row.total for row in self._plan(self._kept[node][rid], prefilling))

    def start(self, rid: int, node: str, prefilling: Prefilling | None) -> Rebuild | None:
        """Open ``rid``'s rebuild on ``node`` over the slots `tokens` counts, which its drop checked, and return
        what it writes past its layout. A prompt ``prefilling`` its step in progress reruns that step's chunks
        last, and changes nothing the layout says, so this returns None for it."""
        assert (node, rid) not in self._cursors, f"{rid}'s rebuild on {node} starts once per drop"
        kept = self._kept[node][rid]
        self._cursors[(node, rid)] = RebuildCursor(tuple(self._plan(kept, prefilling)), kept.sampled)
        if prefilling is not None:
            return None
        rebuild, kept.laid_out = Rebuild(kept.sampled[kept.laid_out:]), len(kept.sampled) - 1
        return rebuild

    def rebuilding(self, rid: int, node: str) -> bool:
        return (node, rid) in self._cursors

    def cursor(self, rid: int, node: str) -> RebuildCursor:
        return self._cursors[(node, rid)]

    def cursors(self) -> ItemsView[tuple[str, int], RebuildCursor]:
        return self._cursors.items()

    def fill(
        self, node: str, rids: Collection[int], chunk_ranges: Mapping[int, ChunkRange],
        inputs: MutableMapping[int, NameToTensorList], infos: MutableMapping[int, CurrentForwardPassInfo],
    ) -> tuple[torch.Event, ...]:
        """Give each rebuild row of ``rids`` its row's walk and its inputs, and return the events the step waits on
        before it reads them.

        A prompt's step in progress runs every chunk from the input the engine keeps for the prompt, under the
        prompt's own step metadata. Any other row runs under its step's recorded metadata and reads its record, on
        the device when it runs its first chunk or runs whole, the last row's ids with every id sampled since but the
        newest appended. A later chunk is cut from the input the engine prepared on the first."""
        ready = []
        for rid in rids:
            cursor = self._cursors[(node, rid)]
            record = cursor.rows[cursor.row].record
            if record is None:
                infos[rid] = replace(infos[rid], graph_walk=cursor.walk)
                continue
            infos[rid] = replace(infos[rid], graph_walk=cursor.walk, step_metadata=record.step_metadata)
            chunk = chunk_ranges.get(rid)
            if chunk is None or chunk.is_first:
                inputs[rid] = self._row_inputs(node, cursor, record, ready)
        return tuple(ready)

    def _row_inputs(
        self, node: str, cursor: RebuildCursor, record: RowRecord, ready: list[torch.Event],
    ) -> NameToTensorList:
        ids_input, inputs = self._policies[node].ids_input, {}
        for name, views in record.inputs.items():
            if name == ids_input:
                ids = list(views)
                if cursor.row == len(cursor.rows) - 1 and len(cursor.sampled) > 1:
                    sampled = torch.frombuffer(cursor.sampled, dtype=torch.int32)[:-1]
                    ids[-1] = torch.cat([ids[-1], sampled.to(ids[-1].dtype)])
                inputs[name] = [to_device_nonblocking(tensor, self._pool.device) for tensor in ids]
                continue
            inputs[name] = []
            for view in views:
                tensor, landed = self._pool.copy_in(view)
                inputs[name].append(tensor)
                ready += [landed] if landed is not None else []
        return inputs

    def land_chunk(self, rid: int, node: str, chunk: ChunkRange | None) -> RowRun | None:
        """Record that ``rid``'s rebuild ran ``chunk`` of its row, or the rest of the row with no chunk, and return
        the row once it has run. The rebuild then moves to its next row, and ends after its last."""
        cursor = self._cursors[(node, rid)]
        cursor.ran += cursor.total - cursor.done if chunk is None else chunk.end - chunk.start
        cursor.done = cursor.total if chunk is None else chunk.end
        if cursor.done < cursor.total:
            return None
        run = RowRun(cursor.rows[cursor.row], cursor.ran)
        cursor.row, cursor.done, cursor.ran = cursor.row + 1, 0, 0
        if cursor.row == len(cursor.rows):
            del self._cursors[(node, rid)]
        return run

    def remove_request(self, rid: int) -> None:
        for node in self._kept:
            self._forget(node, rid)
            self._cursors.pop((node, rid), None)


