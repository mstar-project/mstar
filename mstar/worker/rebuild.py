"""What a worker keeps so a request dropped from a node's KV can be rebuilt there, as the node's
`RebuildPolicy` declares."""

from __future__ import annotations

from array import array
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.engine.resources import Rebuild, SamplerStep
from mstar.graph.base import GraphSection
from mstar.model.rebuild import RebuildPolicy

if TYPE_CHECKING:
    from mstar.engine.engine import ExecutingBatch
    from mstar.model.submodule_base import HostRows


@dataclass(slots=True)
class KeptIds:
    """A request's ids on one node: its prompt as the policy's walk read it, and each id sampled since,
    as int32."""
    prompt: torch.Tensor
    sampled: array


class RebuildRow(NamedTuple):
    """A row of the policy's walk that recomputes a dropped request's KV. Its input is the request's kept
    ids without the last one."""
    node: str
    walk: str
    inputs: NameToTensorList


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
    graph when the worker starts."""

    def __init__(self, policies: Mapping[str, RebuildPolicy], graphs: Mapping[str, GraphSection]) -> None:
        for node, policy in policies.items():
            policy.check(node, graphs)
        self._policies = dict(policies)
        self._loop_walks = {node: policy.loop_walks(node, graphs) for node, policy in policies.items()}
        # node -> handle -> its kept ids. Per node, since a colocated node of another partition samples
        # ids of its own.
        self._kept: dict[str, dict[int, KeptIds]] = {node: {} for node in policies}

    @property
    def nodes(self) -> frozenset[str]:
        """The nodes whose requests can be rebuilt here."""
        return frozenset(self._policies)

    def kept(self, rid: int, node: str) -> KeptIds | None:
        return self._kept.get(node, {}).get(rid)

    def keeps_any(self, node: str) -> bool:
        return bool(self._kept.get(node))

    def keep_prompts(
        self, node: str, graph_walk: str, request_walks: Mapping[int, str],
        inputs: Mapping[int, NameToTensorList],
    ) -> None:
        """Keep the ids of each row of the policy's walk whose request keeps none yet. The rows share
        one copy of the step's ids, each keeping a view of it."""
        policy = self._policies.get(node)
        if policy is None or (graph_walk != policy.walk and policy.walk not in request_walks.values()):
            return
        kept = self._kept[node]
        # a TP follower's unread row has no inputs, and the step fails it
        rows = [
            rid for rid, row in inputs.items()
            if row and rid not in kept and request_walks.get(rid, graph_walk) == policy.walk
        ]
        if not rows:
            return
        per_row = [[tensor.reshape(-1) for tensor in inputs[rid][policy.ids_input]] for rid in rows]
        copy = torch.cat([tensor for tensors in per_row for tensor in tensors])
        lengths = [sum(tensor.numel() for tensor in tensors) for tensors in per_row]
        for rid, prompt in zip(rows, copy.split(lengths), strict=True):
            kept[rid] = KeptIds(prompt, array("i"))

    def keep_samples(
        self, node_batch: ExecutingBatch, cpu_outputs, host_rows: HostRows | None,
    ) -> None:
        """Append each kept row's sample, from the stop check's host copy, to its kept ids."""
        node = node_batch.node_name
        kept_ids = self._kept.get(node)
        if not kept_ids:
            return
        name = self._policies[node].sample_output
        step = node_batch.step
        kept_rows = None if step is None else next(
            (s.kept_rids for s in step.steps.values() if isinstance(s, SamplerStep)), None,
        )
        ctx, rebuild = node_batch.step_context, node_batch.rebuild_rids
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
                if not isinstance(out, dict) or not out.get(name):
                    continue
                sample = out[name][0]
                if sample.numel() != 1:
                    raise ValueError(self._not_one_id(node, name, sample.numel()))
                pairs.append((rid, int(sample)))
        for rid, token in pairs:
            kept = kept_ids.get(rid)
            # the host rows are the forward's, which can hold a row the worker has since taken out of the step
            if (
                kept is not None and ctx.is_real_row(rid) and (kept_rows is None or rid in kept_rows)
                and rid not in rebuild
            ):
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
        if walk not in self._loop_walks[node]:
            return f"it is in walk {walk!r}, where loop {policy.loop!r} doesn't run {node}"
        if len(kept.sampled) != loop_iters.get(policy.loop, 0) + 1:
            return f"its {len(kept.sampled)} kept samples don't cover its KV"
        return None

    def rebuild(self, rid: int, node: str, device: torch.device) -> tuple[Rebuild, RebuildRow]:
        """What ``rid``'s rebuild row on ``node`` writes, and the row."""
        policy = self._policies[node]
        kept = self._kept[node][rid]
        # the one copy of the prompt to the host, made once per rebuild
        ids = array("i", kept.prompt.tolist())
        ids.extend(kept.sampled)
        tokens = torch.frombuffer(ids, dtype=torch.int32)[:-1].to(device, dtype=kept.prompt.dtype)
        return Rebuild(policy.walk, ids), RebuildRow(node, policy.walk, {policy.ids_input: [tokens]})

    def remove_request(self, rid: int) -> None:
        for kept in self._kept.values():
            kept.pop(rid, None)
