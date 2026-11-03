"""What a worker keeps so a request dropped from a node's KV can be rebuilt there, as the node's
`RebuildPolicy` declares, and how far each rebuild has run."""

from __future__ import annotations

from array import array
from collections import defaultdict
from collections.abc import Callable, Collection, ItemsView, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.resources import Rebuild
from mstar.engine.resources.kv.config import StreamLayout, rebuild_refusal
from mstar.graph.base import GraphSection
from mstar.model.rebuild import RebuildPolicy
from mstar.utils.h2d import to_device_nonblocking
from mstar.worker.host_pool import HostCopy, HostPool

if TYPE_CHECKING:
    from mstar.engine.engine import ExecutingBatch
    from mstar.model.base import KVLabel
    from mstar.model.submodule_base import HostRows

# The serving config's ``rebuild_host_mb`` unless it sets one: 16M prompt ids
# at 4 B each.
DEFAULT_REBUILD_HOST_MB = 64


@dataclass(slots=True)
class HostCopies:
    """Host copies of what ``walk`` read besides a request's ids when it prefilled the request, and of that
    row's step metadata, for the request's rebuild row. Empty until the row's step has run. ``landed`` is
    recorded after the copies, and stays None on a CPU device, where the inputs are kept as they are."""
    walk: str
    inputs: NameToTensorList | None = None
    step_metadata: dict | None = None
    landed: torch.Event | None = None

    def ready(self) -> bool:
        return self.inputs is not None and (self.landed is None or self.landed.query())

    def to_device(self, device: torch.device, stream: torch.Stream | None) -> NameToTensorList:
        """Copy the inputs back to ``device`` on ``stream`` once they have landed on the host, and make the
        current stream wait for that copy. The wait is issued now, so a step still being enqueued can stall
        behind the copy."""
        if self.landed is None:
            return self.inputs
        stream.wait_event(self.landed)
        with stream:
            inputs = {name: [t.to(device, non_blocking=True) for t in ts] for name, ts in self.inputs.items()}
        ready = torch.Event(device=device)
        ready.record(stream)
        torch.accelerator.current_stream(device).wait_event(ready)
        return inputs


@dataclass(slots=True)
class KeptIds:
    """A request's ids on one node: its prompt copied to the host as int32, the dtype the walk it prefilled in
    read it in, and each id sampled since. ``lengths`` splits the prompt into the tensors that walk read, and is
    None for one tensor. A walk of the policy's ``copied_inputs`` read more inputs, which ``copies`` holds."""
    prompt: HostCopy
    dtype: torch.dtype
    sampled: array
    lengths: tuple[int, ...] | None = None
    copies: HostCopies | None = None


class Prefilling(NamedTuple):
    """A prompt mid-prefill on a node: the walk it prefills in, the tokens its chunks have run, and its length."""
    walk: str
    done: int
    total: int


@dataclass(slots=True)
class RebuildCursor:
    """A rebuild under way: the walk its rows run, the tokens it writes (``total``) of an input ``length`` long,
    and the end of its chunks that have landed, 0 before the first.

    A rebuild from kept ids reads ``ids`` in ``dtype``, split into tensors of ``lengths``, and writes every id but
    the last, which the next decode step reads. A rebuild from host copies also reads ``copied`` and runs under
    ``step_metadata``. A rebuild of a prompt mid-prefill has no ``ids``: it runs its prompt's chunks again, cut
    from the input the engine keeps for the prompt, up to where the prompt stood."""
    walk: str
    total: int
    length: int
    ids: array | None = None
    dtype: torch.dtype | None = None
    lengths: tuple[int, ...] | None = None
    copied: NameToTensorList | None = None
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
    """The policies whose rebuild this worker runs: it runs the node in the policy's walk and in every walk its
    loop grows the KV in. A disaggregated decode node can't run the prefill walk its rebuild needs.

    Each keeps the copied walks this worker runs the node in. Above one rank it keeps none, since rank 0 decides
    a drop from its own host copies, and a follower's may not have landed.
    """
    here = {}
    for node, policy in policies.items():
        if all((node, walk) in my_walks for walk in {policy.walk, *policy.loop_walks(node, graphs)}):
            here[node] = replace(policy, copied_inputs={} if node in parallel_nodes else {
                walk: names for walk, names in policy.copied_inputs.items() if (node, walk) in my_walks
            })
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
    graph when the worker starts, and the rebuilds under way. ``side_stream`` gives the stream host copies
    queue on, behind a step's own output copy, and None on a CPU device."""

    def __init__(
        self, policies: Mapping[str, RebuildPolicy], graphs: Mapping[str, GraphSection],
        pool: HostPool | None = None, side_stream: Callable[[], torch.Stream | None] = lambda: None,
    ) -> None:
        for node, policy in policies.items():
            policy.check(node, graphs)
        self._policies = dict(policies)
        self._side_stream = side_stream
        self._loop_walks = {node: policy.loop_walks(node, graphs) for node, policy in policies.items()}
        # without one no request keeps ids, so none is dropped
        self._pool = pool
        # node -> handle -> its kept ids. Per node, since a colocated node of another partition samples
        # ids of its own.
        self._kept: dict[str, dict[int, KeptIds]] = {node: {} for node in policies}
        # node -> the handles whose pages a drop there can free. Only these keep their ids.
        self._droppable: dict[str, set[int]] = {node: set() for node in policies}
        # (node, handle) -> its rebuild, from the readiness check that starts it until its last chunk lands
        self._cursors: dict[tuple[str, int], RebuildCursor] = {}

    @property
    def nodes(self) -> frozenset[str]:
        """The nodes whose requests can be rebuilt here."""
        return frozenset(self._policies)

    def kept(self, rid: int, node: str) -> KeptIds | None:
        return self._kept.get(node, {}).get(rid)

    def can_rebuild(self, node: str, layouts: Iterable[StreamLayout]) -> bool:
        """Whether a rebuild on ``node`` can write a request's laid-out labels again: the node declares a policy,
        and its loop grows labels one rebuild row can lay out again, written in a walk the policy prefills."""
        policy, layouts = self._policies.get(node), list(layouts)
        return (
            policy is not None and rebuild_refusal(layouts) is None
            and all(loop == policy.loop for layout in layouts for loop, _ in layout.growth)
            and all(span.walk in policy.prefill_walks for layout in layouts for span in layout.spans)
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
        is False."""
        written: dict[str, list[StreamLayout]] = defaultdict(list)
        writers: dict[str, list[str]] = defaultdict(list)
        for key, nodes in writes.items():
            for node, what in nodes.items():
                laid_out = [layout for layout in layouts.get(key, ()) if what.writes(layout)]
                if laid_out:
                    written[node] += laid_out
                    writers[key].append(node)
        refused = {node for node, laid_out in written.items() if not self.can_rebuild(node, laid_out)}
        rebuildable = {
            key: refused.isdisjoint(writers[key]) and rebuild_refusal(layouts.get(key, ())) is None for key in writes
        }
        if not unpausable:
            for node in written.keys() - refused:
                if all(rebuildable[key] for key, nodes in writers.items() if node in nodes):
                    self._droppable[node].add(rid)
        return rebuildable

    def keep_prompts(self, node_batch: ExecutingBatch) -> None:
        """Copy a row's prompt to the host once the step that read the prompt completes, when the row is in a walk
        the policy prefills, a drop here can free its request's pages, and the request keeps no ids yet. A row the
        pool can't hold keeps none. A row of a copied walk gets its host copies once its step has run
        (`keep_copies`)."""
        policy = self._policies.get(node_batch.node_name)
        if self._pool is None or policy is None:
            return
        walk, walks = node_batch.step_context.graph_walk, node_batch.step_context.request_walks
        if walk not in policy.prefill_walks and policy.prefill_walks.isdisjoint(walks.values()):
            return
        kept, droppable = self._kept[node_batch.node_name], self._droppable[node_batch.node_name]
        for rid, row in node_batch.per_request_input_tensors.items():
            row_walk = walks.get(rid, walk)
            # a TP follower's unread row has no inputs, and the step fails it
            if not row or rid in kept or rid not in droppable or row_walk not in policy.prefill_walks:
                continue
            tensors = [tensor.reshape(-1) for tensor in row[policy.ids_input]]
            prompt = torch.cat(tensors) if len(tensors) > 1 else tensors[0]
            copy = self._pool.copy_out(prompt, node_batch.completion_event, torch.int32)
            if copy is not None:
                kept[rid] = KeptIds(
                    copy, prompt.dtype, array("i"), tuple(map(len, tensors)) if len(tensors) > 1 else None,
                    HostCopies(row_walk) if row_walk in policy.copied_inputs else None,
                )

    def keep_copies(self, node_batch: ExecutingBatch) -> None:
        """Copy to the host what each row of a copied walk read besides its ids, as the step received it, and
        its step metadata. The host has waited on the step, so the copies need no wait."""
        node = node_batch.node_name
        policy, kept_ids = self._policies.get(node), self._kept.get(node)
        if policy is None or not policy.copied_inputs or not kept_ids:
            return
        # a step of no copied walk, such as every decode step, has no row to copy
        ctx, walks = node_batch.step_context, policy.copied_inputs.keys()
        if ctx.graph_walk not in walks and walks.isdisjoint(ctx.request_walks.values()):
            return
        rows = []
        for rid in node_batch.request_ids:
            kept = kept_ids.get(rid)
            copies = None if kept is None else kept.copies
            # the prefill row of a request whose copies aren't taken yet
            if (
                copies is not None and copies.inputs is None
                and node_batch.per_request_info[rid].graph_walk == copies.walk
            ):
                rows.append((rid, copies))
        if not rows:
            return
        stream = self._side_stream()
        for rid, copies in rows:
            inputs = node_batch.per_request_input_tensors[rid]
            names = policy.copied_inputs[copies.walk]
            copies.inputs = {name: [_to_host(tensor, stream) for tensor in inputs[name]] for name in names}
            copies.step_metadata = dict(node_batch.per_request_info[rid].step_metadata)
        if stream is not None:
            landed = torch.Event(device=stream.device)
            landed.record(stream)
            for _, copies in rows:
                copies.landed = landed

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
        cursor = RebuildCursor(self._policies[node].walk, len(ids) - 1, len(ids) - 1, ids, kept.dtype, kept.lengths)
        if kept.copies is not None:
            cursor.walk, cursor.step_metadata = kept.copies.walk, kept.copies.step_metadata
            cursor.copied = kept.copies.to_device(self._pool.device, self._side_stream())
        self._cursors[(node, rid)] = cursor
        return Rebuild(kept.sampled)

    def rebuilding(self, rid: int, node: str) -> bool:
        return (node, rid) in self._cursors

    def cursor(self, rid: int, node: str) -> RebuildCursor:
        return self._cursors[(node, rid)]

    def cursors(self) -> ItemsView[tuple[str, int], RebuildCursor]:
        return self._cursors.items()

    def fill(
        self, node: str, rids: Collection[int], chunk_ranges: Mapping[int, ChunkRange],
        inputs: MutableMapping[int, NameToTensorList], infos: MutableMapping[int, CurrentForwardPassInfo],
    ) -> None:
        """Give each rebuild row of ``rids`` its walk, and its kept ids and host copies on the device when it runs
        its first chunk or runs whole. The ids are split as its prefill row read them, with the ids sampled since
        appended to the last tensor. A later chunk is cut from the input the engine prepared on the first, and
        every chunk of a prompt's rebuild from the input the engine keeps for the prompt."""
        ids_input = self._policies[node].ids_input
        for rid in rids:
            cursor = self._cursors[(node, rid)]
            info = infos[rid]
            infos[rid] = replace(
                info, graph_walk=cursor.walk,
                step_metadata=info.step_metadata if cursor.step_metadata is None else cursor.step_metadata,
            )
            chunk = chunk_ranges.get(rid)
            if cursor.ids is not None and (chunk is None or chunk.is_first):
                ids = to_device_nonblocking(_row_ids(cursor).to(cursor.dtype), self._pool.device)
                head = cursor.lengths[:-1] if cursor.lengths else ()
                inputs[rid] = {ids_input: list(ids.split([*head, len(ids) - sum(head)])), **(cursor.copied or {})}

    def land_chunk(self, rid: int, node: str, chunk: ChunkRange | None) -> RebuildCursor | None:
        """Record that ``rid``'s rebuild ran ``chunk``, or the rest of its ids with no chunk, and return its
        cursor once the rebuild has finished."""
        cursor = self._cursors[(node, rid)]
        cursor.done = cursor.total if chunk is None else chunk.end
        if cursor.done < cursor.total:
            return None
        return self._cursors.pop((node, rid))

    def remove_request(self, rid: int) -> None:
        for node, kept in self._kept.items():
            self._droppable[node].discard(rid)
            ids = kept.pop(rid, None)
            if ids is not None:
                self._pool.give(ids.prompt.slice)
            self._cursors.pop((node, rid), None)


def _row_ids(cursor: RebuildCursor) -> torch.Tensor:
    return torch.frombuffer(cursor.ids, dtype=torch.int32)[:-1]


def _to_host(tensor: torch.Tensor, stream: torch.Stream | None) -> torch.Tensor:
    if stream is None or tensor.device.type == "cpu":
        return tensor
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    with stream:
        host.copy_(tensor, non_blocking=True)
    # Once the worker drops the tensor, the step's stream may reuse its memory before the copy has read it.
    tensor.record_stream(stream)
    return host
