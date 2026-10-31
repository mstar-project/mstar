"""What a worker keeps so a request dropped from a node's KV can be rebuilt there, as the node's
`RebuildPolicy` declares."""

from __future__ import annotations

from array import array
from collections import deque
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.engine.resources import Rebuild, SamplerStep
from mstar.engine.resources.kv.config import KVReqConfig, rebuild_refusal
from mstar.graph.base import GraphSection
from mstar.model.rebuild import RebuildPolicy

if TYPE_CHECKING:
    from mstar.engine.engine import ExecutingBatch
    from mstar.model.submodule_base import HostRows


@dataclass(slots=True)
class HostCopies:
    """Host copies of what a prefill step read besides a request's ids, and of the step's metadata, for the
    request's rebuild row. ``landed`` is recorded after the copies, and is None on a CPU device, where the
    inputs are kept as they are."""
    inputs: NameToTensorList
    step_metadata: dict
    landed: torch.Event | None = None

    def ready(self) -> bool:
        return self.landed is None or self.landed.query()

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
class PrefillStep:
    """One prefill step of a request, which a rebuild row runs again: its walk, the request's ids it read, in
    the tensors it read them as, and host copies of the other inputs a copied walk read."""
    walk: str
    ids: tuple[torch.Tensor, ...] = ()
    copies: HostCopies | None = None


class RebuildRow(NamedTuple):
    """A row that runs one prefill step of a dropped request again, with the step's ids and any host copies.
    The last row's ids also carry every sampled id but the newest, which the next decode step reads. With
    ``step_metadata`` None, the row runs under the request's own step metadata."""
    node: str
    walk: str
    inputs: NameToTensorList
    step_metadata: dict | None = None


@dataclass(slots=True)
class KeptIds:
    """A request's prefill steps on one node, in the order they ran, and each id sampled since, as int32.
    ``rows`` holds the rows of its rebuild not handed out yet."""
    steps: list[PrefillStep] = field(default_factory=list)
    sampled: array = field(default_factory=lambda: array("i"))
    rows: deque[RebuildRow] = field(default_factory=deque)


def policies_run_here(
    policies: Mapping[str, RebuildPolicy], graphs: Mapping[str, GraphSection], my_walks: Collection[tuple[str, str]],
    parallel_nodes: Collection[str],
) -> dict[str, RebuildPolicy]:
    """The policies whose rebuild this worker runs: it runs the node in the policy's walk and in every walk its
    loop grows the KV in. A disaggregated decode node can't run the prefill walk its rebuild needs.

    A policy keeps only the copied walks this worker runs the node in, and none above one rank, since rank 0
    decides a drop from its own host copies, and a follower's may not have landed.
    """
    here = {}
    for node, policy in policies.items():
        if all((node, walk) in my_walks for walk in {policy.walk, *policy.loop_walks(node, graphs)}):
            here[node] = replace(policy, copied_inputs={} if node in parallel_nodes else {
                walk: names for walk, names in policy.copied_inputs.items() if (node, walk) in my_walks
            })
    return here


class Rebuilds:
    """The prefill steps and samples of each request on the nodes that declare a `RebuildPolicy`, checked
    against the graph when the worker starts. ``side_stream`` returns the worker's D2H stream, where host
    copies queue behind a step's own output copy, or None without a device."""

    def __init__(
        self, policies: Mapping[str, RebuildPolicy], graphs: Mapping[str, GraphSection],
        side_stream: Callable[[], torch.Stream | None] = lambda: None,
    ) -> None:
        for node, policy in policies.items():
            policy.check(node, graphs)
        self._policies = dict(policies)
        self._side_stream = side_stream
        self._loop_walks = {node: policy.loop_walks(node, graphs) for node, policy in policies.items()}
        # node -> the requests `keep` accepted there, whose prefill steps are recorded until their first sample
        self._accepted: dict[str, set[int]] = {node: set() for node in policies}
        # node -> handle -> what the request keeps there, from its first prefill step. Per node, since a
        # colocated node of another partition samples ids of its own.
        self._kept: dict[str, dict[int, KeptIds]] = {node: {} for node in policies}

    @property
    def nodes(self) -> frozenset[str]:
        """The nodes whose requests can be rebuilt here."""
        return frozenset(self._policies)

    def kept(self, rid: int, node: str) -> KeptIds | None:
        return self._kept.get(node, {}).get(rid)

    def keep(self, rid: int, node: str, configs: Iterable[KVReqConfig]) -> bool:
        """Keep ``rid``'s prefill steps and samples on ``node`` from now on, and return True, when a rebuild
        there could write the labels its ``configs`` lay out again: the node declares a policy, and its loop
        grows labels that rebuild rows can lay out again, written in walks the policy prefills."""
        policy, configs = self._policies.get(node), list(configs)
        layouts = [layout for config in configs for layout in (config.kv_layout or {}).values()]
        spans = [span for layout in layouts for span in layout.spans]
        if (
            policy is None or not spans or rebuild_refusal(configs) is not None
            or any(loop != policy.loop for layout in layouts for loop, _ in layout.growth)
            or any(span.walk not in policy.prefill_walks for span in spans)
        ):
            return False
        self._accepted[node].add(rid)
        return True

    def keep_prefill(self, node_batch: ExecutingBatch, step_metadata: Callable[[int], dict]) -> None:
        """Record the prefill step of each row, run or served whole by the prefix cache, whose request `keep`
        accepted and hasn't sampled yet. A step keeps the ids it read, as views of one copy shared by the step's
        rows, and host copies of the other inputs of a copied walk. The host has waited on the step, so the
        copies need no wait. ``step_metadata`` gives a request's metadata for the step."""
        node = node_batch.node_name
        policy, accepted = self._policies.get(node), self._accepted.get(node)
        if policy is None or not accepted:
            return
        ctx = node_batch.step_context
        # a step of a walk that prefills nothing, such as a decode step, has no row to record
        if not node_batch.cached_rids and not ctx.request_walks and ctx.graph_walk not in policy.prefill_walks:
            return
        kept_ids, inputs = self._kept[node], node_batch.per_request_input_tensors
        rows = []
        for rid in (*node_batch.request_ids, *node_batch.cached_rids):
            kept, walk = kept_ids.get(rid), ctx.walk_of(rid)
            # A prefill ends at its first kept sample. A rebuild row reruns a recorded step, so it isn't recorded
            # again.
            if (
                rid in accepted and walk in policy.prefill_walks and rid not in node_batch.rebuild_rids
                and not (kept is not None and kept.sampled)
            ):
                rows.append((rid, walk))
        if not rows:
            return
        per_row = [[tensor.reshape(-1) for tensor in inputs[rid].get(policy.ids_input, ())] for rid, _ in rows]
        flat = [tensor for tensors in per_row for tensor in tensors]
        views = iter(torch.cat(flat).split([tensor.numel() for tensor in flat]) if flat else ())
        stream = self._side_stream() if any(walk in policy.copied_inputs for _, walk in rows) else None
        copied = []
        for (rid, walk), tensors in zip(rows, per_row, strict=True):
            copies = None
            if walk in policy.copied_inputs:
                copies = HostCopies(
                    {name: [_to_host(tensor, stream) for tensor in inputs[rid][name]]
                     for name in policy.copied_inputs[walk]},
                    dict(step_metadata(rid)),
                )
                copied.append(copies)
            step = PrefillStep(walk, tuple(next(views) for _ in tensors), copies)
            kept_ids.setdefault(rid, KeptIds()).steps.append(step)
        if copied and stream is not None:
            landed = torch.Event(device=stream.device)
            landed.record(stream)
            for copies in copied:
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
        KV: its prefill has sampled, so every prefill step has run, its host copies have landed, the last step
        read ids for the samples to follow, and the prefill and each iteration of the policy's loop sample one
        id, of which the KV holds all but the last."""
        policy = self._policies.get(node)
        if policy is None:
            return f"{node} declares no rebuild policy"
        kept = self._kept[node].get(rid)
        if kept is None:
            return f"it keeps no ids on {node}"
        if not kept.sampled:
            return "its prefill hasn't sampled yet"
        if not kept.steps[-1].ids:
            return "its last prefill step read no ids for its samples to follow"
        if any(step.copies is not None and not step.copies.ready() for step in kept.steps):
            return "its host copies haven't landed"
        if walk not in self._loop_walks[node]:
            return f"it is in walk {walk!r}, where loop {policy.loop!r} doesn't run {node}"
        if len(kept.sampled) != loop_iters.get(policy.loop, 0) + 1:
            return f"its {len(kept.sampled)} kept samples don't cover its KV"
        return None

    def rebuild(self, rid: int, node: str, device: torch.device) -> tuple[Rebuild, RebuildRow]:
        """What ``rid``'s rebuild on ``node`` writes, and its first row. Each row runs one prefill step
        again, with any host copies back on ``device``, and `next_row` hands out the rest in order."""
        policy, kept = self._policies[node], self._kept[node][rid]
        prompt = [tensor for step in kept.steps for tensor in step.ids]
        # the one copy of the prompt to the host, made once per rebuild
        ids = array("i", torch.cat(prompt).tolist())
        ids.extend(kept.sampled)
        tokens = torch.frombuffer(ids, dtype=torch.int32)[:-1].to(device, dtype=prompt[0].dtype)
        lengths = [tensor.numel() for tensor in prompt]
        lengths[-1] += len(tokens) - sum(lengths)
        split = iter(tokens.split(lengths))
        kept.rows = deque()
        for step in kept.steps:
            inputs, metadata = {}, None
            if step.ids:
                inputs[policy.ids_input] = [next(split) for _ in step.ids]
            if step.copies is not None:
                inputs.update(step.copies.to_device(device, self._side_stream()))
                metadata = step.copies.step_metadata
            kept.rows.append(RebuildRow(node, step.walk, inputs, metadata))
        return Rebuild(kept.steps[-1].walk, ids), kept.rows.popleft()

    def next_row(self, rid: int, node: str) -> RebuildRow | None:
        """The next row of ``rid``'s rebuild on ``node``, or None once every row is handed out."""
        rows = self._kept[node][rid].rows
        return rows.popleft() if rows else None

    def drop_rows(self, rid: int, node: str) -> None:
        """Forget the rows of ``rid``'s rebuild on ``node`` not handed out yet, whose inputs are back on the
        device, since a drop between two rows starts the rebuild over."""
        self._kept[node][rid].rows.clear()

    def remove_request(self, rid: int) -> None:
        for accepted in self._accepted.values():
            accepted.discard(rid)
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
