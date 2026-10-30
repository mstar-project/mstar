"""What a worker keeps so a request dropped from a node's KV can be rebuilt there, as the node's
`RebuildPolicy` declares."""

from __future__ import annotations

from array import array
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.engine.resources import Rebuild, SamplerStep
from mstar.engine.resources.kv.config import StreamLayout, rebuild_refusal
from mstar.graph.base import GraphSection
from mstar.model.rebuild import RebuildPolicy

if TYPE_CHECKING:
    from mstar.engine.engine import ExecutingBatch
    from mstar.model.submodule_base import HostRows


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
    """A request's ids on one node: its prompt as the walk it prefilled in read it, and each id sampled
    since, as int32. ``lengths`` splits the prompt into the tensors that walk read, and is None for one
    tensor. A walk of the policy's ``copied_inputs`` read more inputs, which ``copies`` holds."""
    prompt: torch.Tensor
    sampled: array
    lengths: tuple[int, ...] | None = None
    copies: HostCopies | None = None


class RebuildRow(NamedTuple):
    """A row of the walk a dropped request prefilled in, which recomputes the request's KV. Its inputs are
    the request's kept ids without the last one, and any host copies. With ``step_metadata`` None, the row
    runs under the request's own step metadata."""
    node: str
    walk: str
    inputs: NameToTensorList
    step_metadata: dict | None = None


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


class Rebuilds:
    """The kept ids of each request on the nodes that declare a `RebuildPolicy`, checked against the
    graph when the worker starts. ``side_stream`` gives the stream host copies queue on, behind a step's
    own output copy, and None on a CPU device."""

    def __init__(
        self, policies: Mapping[str, RebuildPolicy], graphs: Mapping[str, GraphSection],
        side_stream: Callable[[], torch.Stream | None] = lambda: None,
    ) -> None:
        for node, policy in policies.items():
            policy.check(node, graphs)
        self._policies = dict(policies)
        self._side_stream = side_stream
        self._loop_walks = {node: policy.loop_walks(node, graphs) for node, policy in policies.items()}
        # node -> the handles a rebuild there could write again (`keep`), whose ids and host copies it keeps
        self._keeps: dict[str, set[int]] = {node: set() for node in policies}
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

    def keep(self, rid: int, node: str, layouts: Iterable[StreamLayout]) -> bool:
        """Keep ``rid``'s ids and host copies on ``node`` from now on, and return True, when a rebuild there could
        write its laid-out labels again: the node declares a policy, and its loop grows labels one rebuild row
        can lay out again, written in a walk the policy prefills."""
        policy, layouts = self._policies.get(node), list(layouts)
        if (
            policy is None or rebuild_refusal(layouts) is not None
            or any(loop != policy.loop for layout in layouts for loop, _ in layout.growth)
            or any(span.walk not in policy.prefill_walks for layout in layouts for span in layout.spans)
        ):
            return False
        self._keeps[node].add(rid)
        return True

    def keep_prompts(
        self, node: str, graph_walk: str, request_walks: Mapping[int, str],
        inputs: Mapping[int, NameToTensorList],
    ) -> None:
        """Keep the ids of each row of a policy's prefill walk whose request the node keeps (`keep`) and that
        keeps none yet. The rows share one copy of the step's ids, each keeping a view of it. A row of a copied
        walk gets its host copies once its step has run (`keep_copies`)."""
        policy = self._policies.get(node)
        if policy is None or (
            graph_walk not in policy.prefill_walks and policy.prefill_walks.isdisjoint(request_walks.values())
        ):
            return
        kept, keeps = self._kept[node], self._keeps[node]
        # a TP follower's unread row has no inputs, and the step fails it
        rows = [
            rid for rid, row in inputs.items()
            if row and rid in keeps and rid not in kept and request_walks.get(rid, graph_walk) in policy.prefill_walks
        ]
        if not rows:
            return
        per_row = [[tensor.reshape(-1) for tensor in inputs[rid][policy.ids_input]] for rid in rows]
        copy = torch.cat([tensor for tensors in per_row for tensor in tensors])
        lengths = [sum(tensor.numel() for tensor in tensors) for tensors in per_row]
        for rid, tensors, prompt in zip(rows, per_row, copy.split(lengths), strict=True):
            walk = request_walks.get(rid, graph_walk)
            kept[rid] = KeptIds(
                prompt, array("i"), tuple(tensor.numel() for tensor in tensors) if len(tensors) > 1 else None,
                HostCopies(walk) if walk in policy.copied_inputs else None,
            )

    def keep_copies(self, node_batch: ExecutingBatch) -> None:
        """Copy to the host what each row of a copied walk read besides its ids, as the step received it, and
        its step metadata. The host has waited on the step, so the copies need no wait."""
        node = node_batch.node_name
        policy, kept_ids = self._policies.get(node), self._kept.get(node)
        if policy is None or not policy.copied_inputs or not kept_ids:
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
        if kept.copies is not None and not kept.copies.ready():
            return "its host copies haven't landed"
        if walk not in self._loop_walks[node]:
            return f"it is in walk {walk!r}, where loop {policy.loop!r} doesn't run {node}"
        if len(kept.sampled) != loop_iters.get(policy.loop, 0) + 1:
            return f"its {len(kept.sampled)} kept samples don't cover its KV"
        return None

    def rebuild(self, rid: int, node: str, device: torch.device) -> tuple[Rebuild, RebuildRow]:
        """What ``rid``'s rebuild row on ``node`` writes, and the row. The row's ids are split as its prefill
        row read them, with the ids sampled since appended to the last tensor."""
        policy = self._policies[node]
        kept = self._kept[node][rid]
        # the one copy of the prompt to the host, made once per rebuild
        ids = array("i", kept.prompt.tolist())
        ids.extend(kept.sampled)
        tokens = torch.frombuffer(ids, dtype=torch.int32)[:-1].to(device, dtype=kept.prompt.dtype)
        head = kept.lengths[:-1] if kept.lengths else ()
        inputs = {policy.ids_input: list(tokens.split([*head, len(tokens) - sum(head)]))}
        walk, metadata = policy.walk, None
        if kept.copies is not None:
            walk, metadata = kept.copies.walk, kept.copies.step_metadata
            inputs.update(kept.copies.to_device(device, self._side_stream()))
        return Rebuild(walk, ids), RebuildRow(node, walk, inputs, metadata)

    def remove_request(self, rid: int) -> None:
        for keeps in self._keeps.values():
            keeps.discard(rid)
        for kept in self._kept.values():
            kept.pop(rid, None)


def _to_host(tensor: torch.Tensor, stream: torch.Stream | None) -> torch.Tensor:
    if stream is None or tensor.device.type == "cpu":
        return tensor
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    with stream:
        host.copy_(tensor, non_blocking=True)
    # Once the worker drops the tensor, the step's stream may reuse its memory before the copy has read it.
    tensor.record_stream(stream)
    return host
