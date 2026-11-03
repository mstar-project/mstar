"""What a worker keeps so a request dropped from a node's KV can be rebuilt there, as the node's
`RebuildPolicy` declares, and how far each rebuild has run."""

from __future__ import annotations

from array import array
from collections import defaultdict, deque
from collections.abc import Collection, ItemsView, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.resources import Rebuild
from mstar.engine.resources.kv.config import SpanKind, StreamLayout, rebuild_refusal
from mstar.engine.resources.kv.host_writes import StepAllowance
from mstar.graph.base import GraphSection
from mstar.model.rebuild import RebuildPolicy
from mstar.utils.h2d import to_device_nonblocking
from mstar.worker.host_pool import HostCopy, HostPool, HostSlice, aligned

if TYPE_CHECKING:
    from mstar.engine.engine import ExecutingBatch
    from mstar.model.base import KVLabel
    from mstar.model.submodule_base import HostRows

# The serving config's ``rebuild_host_mb`` unless it sets one: 16M prompt ids
# at 4 B each.
DEFAULT_REBUILD_HOST_MB = 64
# The serving config's ``rebuild_media_host_mb`` unless it sets one, taken only where a node copies what a walk reads
# besides the ids: a Qwen3.5-9B image of 1024 tokens is 8 MiB of encoder output.
DEFAULT_REBUILD_MEDIA_HOST_MB = 1024


@dataclass(slots=True)
class HostCopies:
    """Host copies of what a request's prefill walk read besides its ids, for the request's rebuild row. Empty
    until the row's step has run. ``queued`` counts the tensors whose copy hasn't started, and ``landed`` is
    recorded after the last copy that has, None where the copies run in place."""
    inputs: NameToTensorList | None = None
    queued: int = 0
    landed: torch.Event | None = None

    def ready(self) -> bool:
        return self.inputs is not None and not self.queued and (self.landed is None or self.landed.query())


@dataclass(slots=True)
class KeptIds:
    """A request's ids on one node: its prompt copied to the host as int32, the ``walk`` it prefilled in and the
    dtype that walk read it in, each id sampled since, and the prefill row's step metadata, which its rebuild row
    runs under. ``lengths`` splits the prompt into the tensors the walk read, and is None for one tensor. A walk
    that read more than the ids has its other inputs in ``copies``."""
    prompt: HostCopy
    walk: str
    dtype: torch.dtype
    sampled: array
    step_metadata: dict
    lengths: tuple[int, ...] | None = None
    copies: HostCopies | None = None


class QueuedCopy(NamedTuple):
    """A device tensor's bytes still to be copied into ``into`` once ``after`` completes, for ``copies``."""
    src: torch.Tensor
    into: HostSlice
    after: torch.Event | None
    copies: HostCopies


class Prefilling(NamedTuple):
    """A prompt mid-prefill on a node: the walk it prefills in, the tokens its chunks have run, and its length."""
    walk: str
    done: int
    total: int


@dataclass(slots=True)
class RebuildCursor:
    """A rebuild under way: the walk its rows run, the tokens it writes (``total``) of an input ``length`` long,
    and the end of its chunks that have landed, 0 before the first.

    A rebuild from kept ids reads ``ids`` in ``dtype``, split into tensors of ``lengths``, and any host ``copies``,
    runs under its prefill's ``step_metadata``, and writes every id but the last, which the next decode step
    reads. A rebuild of a prompt mid-prefill has no ``ids``: it runs its prompt's chunks again, cut
    from the input the engine keeps for the prompt, up to where the prompt stood."""
    walk: str
    total: int
    length: int
    ids: array | None = None
    dtype: torch.dtype | None = None
    lengths: tuple[int, ...] | None = None
    copies: HostCopies | None = None
    step_metadata: dict | None = None
    done: int = 0

    @property
    def resumes_prefill(self) -> bool:
        """Whether its request's prompt goes on past it, from the input the engine keeps for the prompt."""
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
    """The kept ids of each request on the nodes that declare a `RebuildPolicy`, checked against the
    graph when the worker starts, and the rebuilds under way. Host copies of what a walk read besides the ids
    spend the step's ``allowance`` (`flush`)."""

    def __init__(
        self, policies: Mapping[str, RebuildPolicy], graphs: Mapping[str, GraphSection],
        pool: HostPool | None = None, allowance: StepAllowance | None = None,
    ) -> None:
        # node -> walk -> what the node reads there besides the ids, which the worker copies to the host
        self._copied = {node: policy.check(node, graphs) for node, policy in policies.items()}
        self._policies = dict(policies)
        self._loop_walks = {node: policy.loop_walks(node, graphs) for node, policy in policies.items()}
        # without one no request keeps ids, so none is dropped
        self._pool = pool
        # node -> handle -> its kept ids. Per node, since a colocated node of another partition samples
        # ids of its own.
        self._kept: dict[str, dict[int, KeptIds]] = {node: {} for node in policies}
        # node -> the handles whose pages a drop there can free, each with the slice of the pool its ids and host
        # copies go to. Only these keep them.
        self._droppable: dict[str, dict[int, HostSlice]] = {node: {} for node in policies}
        # (node, handle) -> its rebuild, from the readiness check that starts it until its last chunk lands
        self._cursors: dict[tuple[str, int], RebuildCursor] = {}
        self._allowance = StepAllowance() if allowance is None else allowance
        # the copies to the host not yet started, oldest first
        self._queued: deque[QueuedCopy] = deque()

    @property
    def nodes(self) -> frozenset[str]:
        """The nodes whose requests can be rebuilt here."""
        return frozenset(self._policies)

    def kept(self, rid: int, node: str) -> KeptIds | None:
        return self._kept.get(node, {}).get(rid)

    def can_rebuild(self, node: str, layouts: Iterable[StreamLayout]) -> bool:
        """Whether a rebuild on ``node`` can write a request's laid-out labels again: the node declares a policy,
        and its loop grows labels one rebuild row can lay out again, written in one of the policy's walks."""
        policy, layouts = self._policies.get(node), list(layouts)
        return (
            policy is not None and rebuild_refusal(layouts) is None
            and all(loop == policy.loop for layout in layouts for loop, _ in layout.growth)
            and all(
                span.walk in policy.walks and (span.kind == SpanKind.IDS or self._copied[node][span.walk])
                for layout in layouts for span in layout.spans
            )
        )

    def _log_bytes(self, node: str, layouts: Iterable[StreamLayout]) -> int:
        """The host bytes ``node``'s log takes for a request of ``layouts``: its prompt's ids as int32, and the
        host copies of what each other span's walk read."""
        per_slot = self._policies[node].copy_bytes_per_slot
        ids = sum(span.length for layout in layouts for span in layout.spans if span.kind == SpanKind.IDS)
        return aligned(4 * ids) + sum(
            span.length * per_slot[span.walk]
            for layout in layouts for span in layout.spans if span.kind != SpanKind.IDS
        )

    def ingest_request(
        self, rid: int, writes: Mapping[str, Mapping[str, NodeWrites]],
        layouts: Mapping[str, Collection[StreamLayout]], unpausable: bool,
    ) -> dict[str, bool]:
        """For each paged KV cache in ``writes``, whether a drop here can free ``rid``'s pages there: one
        rebuild row can lay out the request's ``layouts`` in that cache again, and every node that writes those
        layouts can redo all of its writes. A node that writes nothing for the request isn't asked.

        A drop takes every cache of its node, so a node is judged by what it writes in all of them. A node keeps
        ``rid``'s ids only when a drop can free ``rid``'s pages in every cache the node writes, and ``unpausable``
        is False. It takes the pool's room for them now, so a request the pool can't hold is known before it is
        admitted, and no drop can free it there."""
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
                piece = None if self._pool is None else self._pool.take(self._log_bytes(node, written[node]))
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
            else:
                self._pool.give(piece)
        return rebuildable

    def keep_prompts(self, node_batch: ExecutingBatch) -> None:
        """Copy a row's prompt into its request's slice once the step that read the prompt completes, when the row
        is in one of the policy's walks, a drop here can free its request's pages, and the request keeps no ids
        yet. A row of a walk that reads more than the ids gets its host copies after them (`keep_copies`)."""
        policy = self._policies.get(node_batch.node_name)
        if self._pool is None or policy is None:
            return
        walk, walks = node_batch.step_context.graph_walk, node_batch.step_context.request_walks
        if walk not in policy.walks and policy.walks.isdisjoint(walks.values()):
            return
        kept, droppable = self._kept[node_batch.node_name], self._droppable[node_batch.node_name]
        for rid, row in node_batch.per_request_input_tensors.items():
            row_walk = walks.get(rid, walk)
            # a TP follower's unread row has no inputs, and the step fails it
            if not row or rid in kept or rid not in droppable or row_walk not in policy.walks:
                continue
            tensors = [tensor.reshape(-1) for tensor in row[policy.ids_input]]
            prompt = torch.cat(tensors) if len(tensors) > 1 else tensors[0]
            copy = self._pool.copy_out(prompt, droppable[rid], node_batch.completion_event, torch.int32)
            copies = bool(self._copied[node_batch.node_name][row_walk])
            kept[rid] = KeptIds(
                copy, row_walk, prompt.dtype, array("i"), dict(node_batch.per_request_info[rid].step_metadata),
                tuple(map(len, tensors)) if len(tensors) > 1 else None, HostCopies() if copies else None,
            )
            if not copies:
                self._trim(node_batch.node_name, rid, aligned(copy.view.nbytes))

    def keep_copies(self, node_batch: ExecutingBatch) -> None:
        """Queue a copy of what each row of a copied walk read besides its ids, as the step received it, into its
        request's slice after the ids. A request whose copies don't fit its slice,
        which the policy's ``copy_bytes_per_slot`` sized, keeps nothing, so it is never dropped."""
        node = node_batch.node_name
        copied, kept_ids = self._copied.get(node, {}), self._kept.get(node)
        walks = {walk for walk, names in copied.items() if names}
        if not walks or not kept_ids:
            return
        # a step of no copied walk, such as every decode step, has no row to copy
        ctx = node_batch.step_context
        if ctx.graph_walk not in walks and walks.isdisjoint(ctx.request_walks.values()):
            return
        rows = []
        for rid in node_batch.request_ids:
            kept = kept_ids.get(rid)
            # the prefill row of a request whose copies aren't taken yet
            if (
                kept is not None and kept.copies is not None and kept.copies.inputs is None
                and node_batch.per_request_info[rid].graph_walk == kept.walk
            ):
                rows.append((rid, kept))
        for rid, kept in rows:
            piece, inputs = self._droppable[node][rid], node_batch.per_request_input_tensors[rid]
            tensors = [(name, tensor) for name in sorted(copied[kept.walk]) for tensor in inputs[name]]
            used = aligned(kept.prompt.view.nbytes)
            if used + sum(aligned(tensor.nbytes) for _, tensor in tensors) > piece.nbytes:
                self._forget(node, rid)
                continue
            kept.copies.inputs = {name: [] for name in copied[kept.walk]}
            for name, tensor in tensors:
                into = HostSlice(piece.offset + used, tensor.nbytes)
                kept.copies.inputs[name].append(self._pool.view(into, tensor.dtype, tensor.shape))
                src = tensor.reshape(-1).view(torch.uint8)
                self._queued.append(QueuedCopy(src, into, node_batch.completion_event, kept.copies))
                kept.copies.queued += 1
                used += aligned(tensor.nbytes)
            self._trim(node, rid, used)

    def flush(self) -> None:
        """Start the queued copies to the host, oldest first, up to what the step's allowance has left, once the
        step's outputs are sent. A copy that runs past the allowance is split there, and the rest waits for the next
        step's. The device tensor a queued copy reads stays held until its copy starts."""
        while self._queued:
            queued = self._queued[0]
            nbytes = self._allowance.spend(queued.src.numel())
            if not nbytes:
                return
            copy = self._pool.copy_out(queued.src[:nbytes], HostSlice(queued.into.offset, nbytes), queued.after)
            queued.copies.landed = copy.landed
            if nbytes < queued.src.numel():
                self._queued[0] = queued._replace(
                    src=queued.src[nbytes:], into=HostSlice(queued.into.offset + nbytes, queued.into.nbytes - nbytes),
                )
                return
            self._queued.popleft()
            queued.copies.queued -= 1

    def _trim(self, node: str, rid: int, used: int) -> None:
        """Give back what ``rid``'s slice on ``node`` holds past its first ``used`` bytes, once its log has all it
        keeps."""
        piece = self._droppable[node][rid]
        if used < piece.nbytes:
            self._pool.give(HostSlice(piece.offset + used, piece.nbytes - used))
            self._droppable[node][rid] = HostSlice(piece.offset, used)

    def _forget(self, node: str, rid: int) -> None:
        kept = self._kept[node].pop(rid, None)
        if kept is not None and kept.copies is not None and kept.copies.queued:
            # its slice goes back to the pool, so no copy may start into it
            self._queued = deque(queued for queued in self._queued if queued.copies is not kept.copies)
        piece = self._droppable[node].pop(rid, None)
        if piece is not None:
            self._pool.give(piece)

    def keep_samples(
        self, node_batch: ExecutingBatch, cpu_outputs: Mapping[int, Mapping[str, list[torch.Tensor]]],
        host_rows: HostRows | None, skip_rids: Collection[int] = (),
    ) -> None:
        """Append each kept row's sample, from the stop check's host copy, to its kept ids. ``skip_rids``
        are rows whose sample is not real, as for `Engine.step_finished`."""
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
        for rid, token in pairs:
            kept = kept_ids.get(rid)
            # the host rows are the forward's, which can hold a row the worker has since taken out of the step
            if kept is not None and ctx.is_real_row(rid) and rid not in ctx.rebuild_rids and rid not in skip_rids:
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
        """Why ``rid``, in ``walk``, can't be dropped on ``node`` now, or None. A request ``prefilling`` its
        prompt is rebuilt from the input the engine keeps for the prompt, which covers what its KV holds. Any
        other is rebuilt from its kept ids, which cover its KV when the prefill and each iteration of the
        policy's loop sampled one id, since the KV holds all but the last."""
        policy = self._policies.get(node)
        if policy is None:
            return f"{node} declares no rebuild policy"
        if prefilling is not None:
            return None
        kept = self._kept[node].get(rid)
        if kept is None:
            return f"it keeps no ids on {node}"
        if not kept.prompt.has_landed():
            return "its ids haven't reached the host"
        if kept.copies is not None and not kept.copies.ready():
            return "its host copies haven't landed"
        if walk not in self._loop_walks[node]:
            return f"it is in walk {walk!r}, where loop {policy.loop!r} doesn't run {node}"
        if len(kept.sampled) != loop_iters.get(policy.loop, 0) + 1:
            return f"its {len(kept.sampled)} kept samples don't cover its KV"
        return None

    def start(self, rid: int, node: str, prefilling: Prefilling | None) -> Rebuild | None:
        """Open ``rid``'s rebuild on ``node``, and return what it writes from its kept ids, or None when it runs
        the chunks its prompt was ``prefilling`` again, which the request's layout already describes."""
        assert (node, rid) not in self._cursors, f"{rid}'s rebuild on {node} starts once per drop"
        if prefilling is not None:
            self._cursors[(node, rid)] = RebuildCursor(prefilling.walk, prefilling.done, prefilling.total)
            return None
        kept = self._kept[node][rid]
        ids = array("i", kept.prompt.view.numpy().tobytes())
        ids.extend(kept.sampled)
        cursor = RebuildCursor(
            kept.walk, len(ids) - 1, len(ids) - 1, ids, kept.dtype, kept.lengths, kept.copies, kept.step_metadata,
        )
        self._cursors[(node, rid)] = cursor
        return Rebuild(cursor.walk, ids)

    def rebuilding(self, rid: int, node: str) -> bool:
        return (node, rid) in self._cursors

    def cursor(self, rid: int, node: str) -> RebuildCursor:
        return self._cursors[(node, rid)]

    def cursors(self) -> ItemsView[tuple[str, int], RebuildCursor]:
        return self._cursors.items()

    def fill(
        self, node: str, rids: Collection[int], chunk_ranges: Mapping[int, ChunkRange],
        inputs: MutableMapping[int, NameToTensorList], infos: MutableMapping[int, CurrentForwardPassInfo],
    ) -> list[torch.Event]:
        """Give each rebuild row of ``rids`` its walk, a row from kept ids its prefill's step metadata, and its kept
        ids and host copies on the device when it runs its first chunk or runs whole, and return the events the step
        waits on before it reads the copies. The ids
        are split as its prefill row read them, with the ids sampled since appended to the last tensor. A later
        chunk is cut from the input the engine prepared on the first, and every chunk of a prompt's rebuild from
        the input the engine keeps for the prompt."""
        ids_input, ready = self._policies[node].ids_input, []
        for rid in rids:
            cursor = self._cursors[(node, rid)]
            copies = cursor.copies
            infos[rid] = replace(infos[rid], graph_walk=cursor.walk)
            if cursor.ids is not None:
                infos[rid] = replace(infos[rid], step_metadata=cursor.step_metadata)
            chunk = chunk_ranges.get(rid)
            if cursor.ids is not None and (chunk is None or chunk.is_first):
                ids = to_device_nonblocking(_row_ids(cursor).to(cursor.dtype), self._pool.device)
                head = cursor.lengths[:-1] if cursor.lengths else ()
                inputs[rid] = {ids_input: list(ids.split([*head, len(ids) - sum(head)]))}
                for name, views in (copies.inputs if copies is not None else {}).items():
                    inputs[rid][name] = []
                    for view in views:
                        tensor, landed = self._pool.copy_in(view)
                        inputs[rid][name].append(tensor)
                        ready += [landed] if landed is not None else []
        return ready

    def land_chunk(self, rid: int, node: str, chunk: ChunkRange | None) -> RebuildCursor | None:
        """Record that ``rid``'s rebuild ran ``chunk``, or the rest of its ids with no chunk, and return its
        cursor once the rebuild has finished."""
        cursor = self._cursors[(node, rid)]
        cursor.done = cursor.total if chunk is None else chunk.end
        if cursor.done < cursor.total:
            return None
        return self._cursors.pop((node, rid))

    def remove_request(self, rid: int) -> None:
        for node in self._kept:
            self._forget(node, rid)
            self._cursors.pop((node, rid), None)


def _row_ids(cursor: RebuildCursor) -> torch.Tensor:
    return torch.frombuffer(cursor.ids, dtype=torch.int32)[:-1]
