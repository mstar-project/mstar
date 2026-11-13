"""A guaranteed request on a verifying node reserves its whole footprint, the verify slack included.

With no host pool every request on a verifying node is guaranteed (no rebuild policy, no host to move to), so its
reservation has to cover the most a verify step's rejected drafts can hold before their take-back: the request
layout's scratch, which the conductor floors at the slack of 2(k + 1) (`KVSpec.request_layout`). Such a pool moves
nothing, so it lends nothing. With a host pool a guaranteed request lends what it hasn't written of that
reservation, the slack included, and takes those pages back by moving a borrower when its grant runs short. That
take-back of pages is apart from the verify take-back of rejected tokens, which lowers lengths only.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np
import pytest
import test_kv_host_index as host
from test_kv_admission import _runner
from test_kv_preemption import _offload, _pool, _preempt, _ready, _running, _step
from test_kv_prefix_lease import PAGE_SIZE, WALK, _ctx, _StubTransfer

from mstar.engine.resources.draft.ledger import VerifyLedger
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PrefixSpan, StreamLayout
from mstar.engine.resources.step import Segment, SubmoduleStep

K = 3


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


@pytest.mark.parametrize("scratch", [2 * (K + 1), 0], ids=["slack", "no_slack"])
def test_a_guaranteed_verifying_request_grows_to_its_cap_inside_its_reservation(scratch):
    """Every verify step writes k + 1 tokens and keeps one, the worst case. Its k rejected tokens are taken back
    two steps later, the asynchronous rule, so the stored length runs up to 2k + 1 past the verified one. The
    other request takes the pool's room past this one's reservation, so a grant past it has nothing to take.
    Negative control: with no scratch the last steps run past the reservation and are refused."""
    kv = _pool()
    kv.moves = False  # no host pool and no rebuild policy: the worker moves nothing, so nothing is lent
    prompt, decodes = 2 * PAGE_SIZE, 6 * PAGE_SIZE
    layout = StreamLayout(spans=(PrefixSpan(prompt, prompt, WALK),), growth=(("decode", 1),), scratch=scratch)
    _runner(kv).ingest_request("r", {"kv": KVReqConfig(
        kv_layout={"main": layout}, loop_caps={"decode": decodes}, unpausable=True,
    )}, 0)
    assert _ready(kv, "r") and _step(kv, "r", prompt).ok
    reserved = kv._allotments["r"].pages
    assert reserved == -(-(prompt + decodes + scratch) // PAGE_SIZE)
    # the reserved pages stay free in the arena but out of the room; another request takes the rest
    _running(kv, "x", kv._room(), arrival=1)
    assert kv._room() == 0, "room left, which a grant past the reservation could take"

    refused, pending = [], []
    for step in range(decodes):
        if len(pending) == 2:
            kv.take_back(["r"], [pending.pop(0)], "main")
        if not _step(kv, "r", K + 1).ok:
            refused.append(step)
            break
        pending.append(K)
    if scratch:
        assert not refused, f"a grant inside the reservation was refused at step {refused}"
    else:
        assert refused and refused[0] >= decodes - 2 * K - 1, refused


def _guaranteed(kv, rid: str, prompt: int, decodes: int) -> None:
    layout = StreamLayout(spans=(PrefixSpan(prompt, prompt, WALK),), growth=(("decode", 1),), scratch=2 * (K + 1))
    _runner(kv).ingest_request(rid, {"kv": KVReqConfig(
        kv_layout={"main": layout}, loop_caps={"decode": decodes}, unpausable=True,
    )}, 0)
    assert _ready(kv, rid) and _step(kv, rid, prompt).ok


def _verify_step(rid: str) -> SubmoduleStep:
    step = SubmoduleStep(steps={"kv": KVStep(segments=(Segment(rid, "main", K + 1),))})
    step.set_ctx(_ctx(rid))
    return step


def _stored(kv, rid: str) -> int:
    return kv._streams[rid]["main"].stored_len


def test_a_guaranteed_verifying_request_that_lent_its_slack_takes_back_its_tokens_once_and_its_pages_by_moving():
    """With a host pool, g lends what it hasn't written of its reservation, slack included, and o borrows it. Each
    verify step keeps one token of k + 1 (synchronous: the result is due at the next admit). When g's grant runs
    short, that admit has already lowered g's length by the rejected tokens (`StepRunner._take_back`, ahead of every
    resource). The refused step leaves the length lowered, the grant moves o to the host to take its pages back, and
    the retry finds nothing due, so each result lowered g's length exactly once."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(16)
    assert kv.moves
    prompt, decodes = 2 * PAGE_SIZE, 6 * PAGE_SIZE
    _guaranteed(kv, "g", prompt, decodes)
    _running(kv, "o", 1, arrival=1, decodes=8 * PAGE_SIZE)
    while kv._room() >= 0 and _step(kv, "o", PAGE_SIZE).ok:
        pass
    assert kv._room() < 0, "o borrowed nothing of what g lends"

    runner, ledger, taken = _runner(kv), VerifyLedger(WALK, synchronous=True), []
    take_back = kv.take_back

    def recording(rids, counts, label):
        taken.append((tuple(rids), tuple(counts)))
        take_back(rids, counts, label)

    kv.take_back = recording
    verified = prompt
    for _ in range(decodes):
        step = _verify_step("g")
        outcome = runner.admit(step, ledger)
        if not outcome.ok:
            break
        assert _stored(kv, "g") == verified
        runner.plan(step)
        runner.commit(step)
        ledger.record(ledger.committed(["g"], {"kv": "main"}, [K + 1]), ["g"], np.array([0]))
        verified += 1
    else:
        pytest.fail("g's grant never ran short of the pages it lent")

    assert _stored(kv, "g") == verified, "the refused admit didn't apply the result due at it"
    applied = len(taken)
    victims = _preempt(kv, "g", outcome.outcome.reason.pages_short, droppable=lambda rid: False)
    assert victims.to_host == ["o"] and victims.takes_back, victims
    _offload(kv, "o")

    assert runner.admit(step, ledger).ok, "g's grant failed after its borrower moved"
    assert len(taken) == applied and _stored(kv, "g") == verified, "the retry applied a result again"
    assert taken == [(("g",), (K,))] * applied
