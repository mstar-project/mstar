"""What a worker keeps so a request dropped from a node's KV can be rebuilt there, as the node's
`RebuildPolicy` declares, and how far each rebuild has run."""

from __future__ import annotations

import logging
from array import array
from collections.abc import Collection, ItemsView, KeysView, Mapping, MutableMapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

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

logger = logging.getLogger(__name__)

# The serving config's ``rebuild_host_mb`` unless it sets one: 8M prompt ids
# at 8 B each.
DEFAULT_REBUILD_HOST_MB = 64


@dataclass(slots=True)
class KeptIds:
    """A request's ids on one node: its prompt copied to the host as the policy's walk read it, and each id
    sampled since."""
    prompt: HostCopy
    sampled: array


@dataclass(slots=True)
class RebuildCursor:
    """A rebuild under way: the policy's walk, the request's kept ids, the dtype the walk reads them in, and the
    end of its chunks that have landed, 0 before the first. It writes every id but the last (``total``), which
    the next decode step reads."""
    walk: str
    ids: array
    dtype: torch.dtype
    done: int = 0

    @property
    def total(self) -> int:
        return len(self.ids) - 1


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
        # nodes that have logged a prompt the pool couldn't hold
        self._logged_full: set[str] = set()

    @property
    def pool(self) -> HostPool | None:
        return self._pool

    @property
    def nodes(self) -> KeysView[str]:
        """The nodes that declare a `RebuildPolicy`."""
        return self._policies.keys()

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
            copy = self._pool.copy_out(prompt, node_batch.completion_event)
            if copy is not None:
                kept[rid] = KeptIds(copy, array("i"))
            elif node_batch.node_name not in self._logged_full:
                self._logged_full.add(node_batch.node_name)
                logger.warning(
                    "%s: the rebuild host pool has no room for a %d-byte prompt, with %d of %d bytes in use, so its "
                    "request keeps no ids and can't be dropped. Only the first on each node is logged, and the "
                    "serving config's rebuild_host_mb sizes the pool.",
                    node_batch.node_name, prompt.nbytes, self._pool.nbytes_used(), self._pool.nbytes,
                )

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

    def refusal(self, rid: int, node: str, walk: str, loop_iters: Mapping[str, int]) -> str | None:
        """Why ``rid``, in ``walk``, can't be dropped on ``node`` now, or None when its kept ids cover its
        KV: the prefill and each iteration of the policy's loop sample one id, and the KV holds all but
        the last."""
        policy = self._policies.get(node)
        if policy is None:
            return f"{node} declares no rebuild policy"
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

    def tokens(self, rid: int, node: str) -> int:
        """How many tokens ``rid``'s rebuild on ``node`` writes: its kept prompt and every sample but the last."""
        kept = self._kept[node][rid]
        return kept.prompt.view.numel() + len(kept.sampled) - 1

    def rebuild_of(self, rid: int, node: str) -> Rebuild:
        """What ``rid``'s rebuild on ``node`` writes, from its kept ids."""
        kept = self._kept[node][rid]
        ids = array("i", kept.prompt.view.to(torch.int32).numpy().tobytes())
        ids.extend(kept.sampled)
        return Rebuild(self._policies[node].walk, ids)

    def start(self, rid: int, node: str, rebuild: Rebuild) -> None:
        """Open ``rid``'s rebuild on ``node``, once the node's resources have taken in ``rebuild``."""
        assert (node, rid) not in self._cursors, f"{rid}'s rebuild on {node} starts once per drop"
        self._cursors[(node, rid)] = RebuildCursor(rebuild.walk, rebuild.ids, self._kept[node][rid].prompt.view.dtype)

    def rebuilding(self, rid: int, node: str) -> bool:
        return (node, rid) in self._cursors

    def cursors(self) -> ItemsView[tuple[str, int], RebuildCursor]:
        return self._cursors.items()

    def host_inputs(self, rid: int, node: str) -> NameToTensorList:
        """The input of ``rid``'s rebuild row on the host, which its length is measured from."""
        return {self._policies[node].ids_input: [_row_ids(self._cursors[(node, rid)])]}

    def fill(
        self, node: str, rids: Collection[int], chunk_ranges: Mapping[int, ChunkRange],
        inputs: MutableMapping[int, NameToTensorList], infos: MutableMapping[int, CurrentForwardPassInfo],
    ) -> None:
        """Give each rebuild row of ``rids`` the policy's walk, and its ids on the device when it runs its first
        chunk or runs whole. A later chunk is cut from the input the engine prepared on the first."""
        ids_input = self._policies[node].ids_input
        for rid in rids:
            cursor = self._cursors[(node, rid)]
            infos[rid] = replace(infos[rid], graph_walk=cursor.walk)
            chunk = chunk_ranges.get(rid)
            if chunk is None or chunk.is_first:
                ids = _row_ids(cursor).to(cursor.dtype)
                inputs[rid] = {ids_input: [to_device_nonblocking(ids, self._pool.device)]}

    def land_chunk(self, rid: int, node: str, chunk: ChunkRange | None) -> bool:
        """Record that ``rid``'s rebuild ran ``chunk``, or the rest of its ids with no chunk, and return whether
        the rebuild has finished."""
        cursor = self._cursors[(node, rid)]
        cursor.done = cursor.total if chunk is None else chunk.end
        if cursor.done < cursor.total:
            return False
        del self._cursors[(node, rid)]
        return True

    def remove_request(self, rid: int) -> None:
        for node, kept in self._kept.items():
            ids = kept.pop(rid, None)
            if ids is not None:
                self._pool.give(ids.prompt.slice)
            self._cursors.pop((node, rid), None)


def _row_ids(cursor: RebuildCursor) -> torch.Tensor:
    return torch.frombuffer(cursor.ids, dtype=torch.int32)[:-1]
