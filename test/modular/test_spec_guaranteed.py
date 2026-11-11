"""A guaranteed request on a verifying node reserves its whole footprint, the verify slack included.

With no host pool every request on a verifying node is guaranteed (no rebuild policy, no host to move to), so its
reservation has to cover the most a verify step's rejected drafts can hold before their take-back: the request
layout's scratch, which the conductor floors at the slack of 2(k + 1) (`KVSpec.request_layout`).
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
from test_kv_admission import _runner
from test_kv_preemption import _pool, _ready, _running, _step
from test_kv_prefix_lease import PAGE_SIZE, WALK, _StubTransfer

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, PrefixSpan, StreamLayout

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
