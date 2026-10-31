"""What a node that verifies drafts must declare, checked once per walk when the engine loads
it, and what the engine keeps for the walk once it passes."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, NamedTuple

from mstar.engine.resources.base import Resource, RewindKind
from mstar.engine.resources.draft_verify.config import DraftPolicy
from mstar.engine.resources.draft_verify.resource import DraftVerifyResource
from mstar.engine.resources.sampler.resource import SamplerResource

if TYPE_CHECKING:
    from mstar.model.submodule_base import NodeSubmodule


class VerifyWalk(NamedTuple):
    """A walk that verifies drafts, as the engine runs it."""
    resource: DraftVerifyResource
    policy: DraftPolicy
    # the sampler's resource key, under which a request's config carries ignore_eos
    sampler: str


def verify_walk(
    node: str, submodule: NodeSubmodule, resources: Mapping[str, Resource], walk: str,
    combined_walks: Collection[str] = (), keyed: bool = False,
) -> VerifyWalk | None:
    """The walk's ``VerifyWalk``, or None when it doesn't verify. Raises when the node can't
    verify it: a structural problem is found once, here, rather than in a step. ``keyed``:
    the node's KV labels are keyed for the prefix cache. (A chunked verifying walk is refused
    where the model's chunking is bound, ``Engine.set_chunking_policies``.)"""
    policy = submodule.get_draft_policy(walk)
    if policy is None:
        return None
    where = f"node {node!r} verifies drafts on walk {walk!r}"
    if not isinstance(policy, DraftPolicy):
        raise TypeError(f"{where}, but get_draft_policy returned a {type(policy).__name__}")
    found = [res for res in resources.values() if isinstance(res, DraftVerifyResource)]
    if len(found) != 1:
        raise ValueError(f"{where}, so it needs one DraftVerifySpec among its resources; it has {len(found)}")
    dv = found[0]
    sampler = resources.get(dv.sampler_key)
    if not isinstance(sampler, SamplerResource):
        raise ValueError(f"{where}: its DraftVerifySpec names the sampler {dv.sampler_key!r}, which the node lacks")
    if not sampler.verifies_drafts:
        raise ValueError(
            f"{where}: declare its sampler {dv.sampler_key!r} with SamplerSpec(draft_verify=True), "
            "so a request a greedy verify would answer differently is refused"
        )
    undeclared = sorted(key for key, res in resources.items() if res.rewind_kind is None)
    if undeclared:
        raise ValueError(
            f"{where}, but {undeclared} declare no rewind_kind, so a verify step's rejected rows "
            "would stay in them; draft verify rewinds paged KV and sequential positions so far"
        )
    if walk in combined_walks:
        raise ValueError(
            f"{where}, which a combined walk batches with others; a verify row needs a step of its own walk"
        )
    if submodule.device_loopback_signals(walk):
        raise ValueError(f"{where}, whose emitted tokens are cut on the host, so its outputs can't stay on the device")
    if getattr(submodule, "reads_device_loopback", False):
        # A verify step writes none of the sampler's last-token rows such a node reads, and the
        # sampler's in-graph scatter, which such a node turns on, doesn't move the verify's offsets.
        raise ValueError(
            f"{where}, but the node reads its sampled tokens back on the device (reads_device_loopback); "
            "a verify step emits its tokens on the host"
        )
    if keyed:
        raise ValueError(
            f"{where}, whose KV labels are keyed for the prefix cache, which draft verify "
            "doesn't key yet"
        )
    if getattr(submodule, "get_stop_policies", lambda _: None)(walk) is not None:
        # the runtime would read one token a row, where a verify row emits up to k + 1
        raise ValueError(
            f"{where}, whose stops the graph runtime checks from declared policies; a verify "
            "step's stop is the engine's"
        )
    for res in resources.values():
        if res.rewind_kind is RewindKind.HOST and hasattr(res, "require_lookahead"):
            res.require_lookahead(policy.loop, policy.lookahead)
    dv.bind_sampler(sampler)
    dv.bind_settlers([res for res in resources.values() if res.rewind_kind is RewindKind.DEVICE])
    return VerifyWalk(dv, policy, dv.sampler_key)
