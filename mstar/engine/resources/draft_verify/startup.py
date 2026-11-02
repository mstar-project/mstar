"""What a node that verifies drafts must declare, checked once per walk when the engine loads
it, and what the engine keeps for the walk once it passes."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, NamedTuple

from mstar.engine.resources.base import Resource, RewindKind
from mstar.engine.resources.draft_verify.config import DRAFT_TAPS, DraftPolicy
from mstar.engine.resources.draft_verify.proposer import DraftProposer
from mstar.engine.resources.draft_verify.resource import DraftVerifyResource
from mstar.engine.resources.draft_verify.taps import TapResource
from mstar.engine.resources.sampler.resource import SamplerResource

if TYPE_CHECKING:
    from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)


class VerifyWalk(NamedTuple):
    """A walk that verifies drafts, as the engine runs it."""
    resource: DraftVerifyResource
    policy: DraftPolicy
    # the sampler's resource key, under which a request's config carries ignore_eos
    sampler: str
    # under a combined walk, the walk whose rows verify (the others are prompts); None: every row
    rows_walk: str | None = None


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
    proposer = None
    if policy.proposer is not None:
        proposer = resources.get(policy.proposer)
        if not isinstance(proposer, DraftProposer):
            raise ValueError(
                f"{where}, drafting with the proposer {policy.proposer!r}, which the node "
                f"{'declares as a ' + type(proposer).__name__ if proposer is not None else 'lacks'}"
            )
        if proposer.max_drafts != policy.max_drafts:
            raise ValueError(
                f"{where} with {policy.max_drafts} drafts a step, but its proposer "
                f"{policy.proposer!r} drafts {proposer.max_drafts}"
            )
    if policy.taps:
        taps = resources.get(DRAFT_TAPS)
        if not isinstance(taps, TapResource) or taps.layers != policy.taps:
            have = taps.layers if isinstance(taps, TapResource) else None
            raise ValueError(
                f"{where}, whose drafter reads target layers {policy.taps}: declare a TapSpec under "
                f"{DRAFT_TAPS!r} with those layers (the node's taps {have})"
            )
    if walk in combined_walks:
        problem = _mixing_problem(policy, proposer, sampler, resources)
        if problem is not None:
            raise ValueError(f"{where}, which a combined walk batches with prompt rows, but {problem}")
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
    if policy.device_rewind:
        unable = sorted(
            key for key, res in resources.items() if res.plan_reads_lengths and not res.corrects_on_device
        )
        if proposer is None and not unable:
            dv.enable_device_rewind([res for res in resources.values() if res.plan_reads_lengths])
        elif walk in combined_walks:
            raise ValueError(
                f"{where}, which a combined walk batches with prompt rows, but {unable} can't "
                "correct a plan on the device"
            )
        else:
            # waits for each verdict before the step after it, which is correct, just slower
            logger.info(
                "%s, taking verdicts back on the host: %s", where,
                "an engine-side proposer drafts from them" if proposer is not None
                else f"{unable} can't correct a plan on the device",
            )
    if proposer is not None:
        dv.bind_proposer(proposer)
        proposer.bind_verifier(dv)
    return VerifyWalk(dv, policy, dv.sampler_key)


def _mixing_problem(
    policy: DraftPolicy, proposer: DraftProposer | None, sampler: SamplerResource,
    resources: Mapping[str, Resource],
) -> str | None:
    """Why a walk's verify rows can't share a step with prompt rows (`with_mixed_verify`), or
    None. Such a step runs its prompts through the sampler and plans them with the verify rows,
    so nothing may wait for a verdict on the host or touch state a prompt row holds."""
    if proposer is not None:
        return "an engine-side proposer drafts it"
    if sampler.sampled_verify:
        return "its sampler verifies by rejection sampling, whose draws a mixed step commits for its prompt rows only"
    settle = sorted(key for key, res in resources.items() if res.rewind_kind is RewindKind.DEVICE)
    if settle:
        return f"{settle} settle in place"
    if not policy.device_rewind:
        return "its verdicts come back on the host"
    return None


def mixed_verify_walk(
    node: str, combined: str, members: Collection[str], walks: Mapping[str, VerifyWalk | None],
    resources: Mapping[str, Resource],
) -> VerifyWalk | None:
    """A combined walk's ``VerifyWalk`` from its members' (``walks``): a step of it verifies the
    rows of its one member that verifies, run first, beside the others' prompt rows; None when no
    member verifies. Binds the attention resources that correct their plans on the device to
    plan its steps for that (FA2 at any span)."""
    verifying = sorted(walk for walk in members if walks.get(walk) is not None)
    if not verifying:
        return None
    if len(verifying) > 1:
        raise ValueError(f"node {node!r}: combined walk {combined!r} batches walks {verifying} that each verify drafts")
    found = walks[verifying[0]]
    for res in resources.values():
        bind = getattr(res, "correct_mixed_walk", None)
        if bind is not None and res.device_correct:
            bind(combined)
    return found._replace(rows_walk=verifying[0])

