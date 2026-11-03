"""What a worker keeps so a request dropped from a node's KV can be rebuilt there, as the node's
`RebuildPolicy` declares, and how far each rebuild has run."""

from __future__ import annotations

from array import array
from collections.abc import Collection, ItemsView, Mapping, MutableMapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.resources import Rebuild
from mstar.graph.base import GraphSection
from mstar.model.rebuild import RebuildPolicy
from mstar.utils.h2d import to_device_nonblocking
from mstar.worker.host_pool import HostCopy, HostPool

if TYPE_CHECKING:
    from mstar.engine.engine import ExecutingBatch
    from mstar.model.submodule_base import HostRows

# The serving config's ``rebuild_host_mb`` unless it sets one: 16M prompt ids
# at 4 B each.
DEFAULT_REBUILD_HOST_MB = 64


@dataclass(slots=True)
class KeptIds:
    """A request's ids on one node: its prompt copied to the host as int32, the dtype the policy's walk read
    it in, and each id sampled since."""
    prompt: HostCopy
    dtype: torch.dtype
    sampled: array


class Prefilling(NamedTuple):
    """A prompt mid-prefill on a node: the walk it prefills in, the tokens its chunks have run, and its length."""
    walk: str
    done: int
    total: int


@dataclass(slots=True)
class RebuildCursor:
    """A rebuild under way: the walk its rows run, the tokens it writes (``total``) of an input ``length`` long,
    and the end of its chunks that have landed, 0 before the first.

    A rebuild from kept ids reads ``ids`` in ``dtype`` and writes every id but the last, which the next decode
    step reads. A rebuild of a prompt mid-prefill has no ``ids``: it runs its prompt's chunks again, cut from the
    input the engine keeps for the prompt, up to where the prompt stood."""
    walk: str
    total: int
    length: int
    ids: array | None = None
    dtype: torch.dtype | None = None
    done: int = 0

    @property
    def resumes_prefill(self) -> bool:
        """Whether its request's prompt goes on past it, from the input the engine keeps for the prompt."""
        return self.total < self.length


def policies_run_here(
    policies: Mapping[str, RebuildPolicy], graphs: Mapping[str, GraphSection], my_walks: Collection[tuple[str, str]],
) -> dict[str, RebuildPolicy]:
    """The policies whose rebuild this worker runs: it runs the node in the policy's walk and in every walk its
    loop grows the KV in. A disaggregated decode node can't run the prefill walk its rebuild needs."""
    return {
        node: policy for node, policy in policies.items()
        if all((node, walk) in my_walks for walk in {policy.walk, *policy.loop_walks(node, graphs)})
    }


class Rebuilds:
    """The kept ids of each request on the nodes that declare a `RebuildPolicy`, checked against the
    graph when the worker starts, and the rebuilds under way."""

    def __init__(
        self, policies: Mapping[str, RebuildPolicy], graphs: Mapping[str, GraphSection],
        pool: HostPool | None = None,
    ) -> None:
        for node, policy in policies.items():
            policy.check(node, graphs)
        self._policies = dict(policies)
        self._loop_walks = {node: policy.loop_walks(node, graphs) for node, policy in policies.items()}
        # without one no request keeps ids, so none is dropped
        self._pool = pool
        # node -> handle -> its kept ids. Per node, since a colocated node of another partition samples
        # ids of its own.
        self._kept: dict[str, dict[int, KeptIds]] = {node: {} for node in policies}
        # (node, handle) -> its rebuild, from the readiness check that starts it until its last chunk lands
        self._cursors: dict[tuple[str, int], RebuildCursor] = {}

    @property
    def nodes(self) -> frozenset[str]:
        """The nodes whose requests can be rebuilt here."""
        return frozenset(self._policies)

    def kept(self, rid: int, node: str) -> KeptIds | None:
        return self._kept.get(node, {}).get(rid)

    def keep_prompts(self, node_batch: ExecutingBatch) -> None:
        """Copy the prompt of each row of the policy's walk whose request keeps no ids yet to the host, once
        the step that read it completes. A row the pool can't hold keeps none."""
        policy = self._policies.get(node_batch.node_name)
        if self._pool is None or policy is None:
            return
        walk, walks = node_batch.step_context.graph_walk, node_batch.step_context.request_walks
        if walk != policy.walk and policy.walk not in walks.values():
            return
        kept = self._kept[node_batch.node_name]
        for rid, row in node_batch.per_request_input_tensors.items():
            # a TP follower's unread row has no inputs, and the step fails it
            if not row or rid in kept or walks.get(rid, walk) != policy.walk:
                continue
            prompt = row[policy.ids_input][0]
            copy = self._pool.copy_out(prompt, node_batch.completion_event, torch.int32)
            if copy is not None:
                kept[rid] = KeptIds(copy, prompt.dtype, array("i"))

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
        kept, walk = self._kept[node][rid], self._policies[node].walk
        ids = array("i", kept.prompt.view.numpy().tobytes())
        ids.extend(kept.sampled)
        self._cursors[(node, rid)] = RebuildCursor(walk, len(ids) - 1, len(ids) - 1, ids, kept.dtype)
        return Rebuild(walk, ids)

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
        """Give each rebuild row of ``rids`` its walk, and its kept ids on the device when it runs its first
        chunk or runs whole. A later chunk is cut from the input the engine prepared on the first, and every
        chunk of a prompt's rebuild from the input the engine keeps for the prompt."""
        ids_input = self._policies[node].ids_input
        for rid in rids:
            cursor = self._cursors[(node, rid)]
            infos[rid] = replace(infos[rid], graph_walk=cursor.walk)
            chunk = chunk_ranges.get(rid)
            if cursor.ids is not None and (chunk is None or chunk.is_first):
                ids = _row_ids(cursor).to(cursor.dtype)
                inputs[rid] = {ids_input: [to_device_nonblocking(ids, self._pool.device)]}

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
            ids = kept.pop(rid, None)
            if ids is not None:
                self._pool.give(ids.prompt.slice)
            self._cursors.pop((node, rid), None)


def _row_ids(cursor: RebuildCursor) -> torch.Tensor:
    return torch.frombuffer(cursor.ids, dtype=torch.int32)[:-1]
