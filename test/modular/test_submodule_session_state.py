"""Per-session submodule state, alongside the per-request store.

Same shape as ``PerRequestState``, different lifecycle: the engine drops a
request's state at request teardown and a session's only when the session ends,
and the injected view resolves a rid to its session's state (or None when the
request belongs to no session).
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.model.submodule_base import (
    LazySessionStates,
    ModelInputsFromEngine,
    NodeSubmodule,
)


class _Sub(NodeSubmodule):
    def forward(self, *args, **kwargs):
        return {}

    def prepare_inputs(self, *args, **kwargs):
        return None


def test_session_state_is_created_on_first_access_and_kept():
    sub = _Sub()

    state = sub.session_state("s")
    state.add("latents", torch.ones(2))

    assert sub.session_state("s") is state
    assert torch.equal(sub.session_state("s")["latents"], torch.ones(2))


def test_request_and_session_stores_are_separate():
    sub = _Sub()
    sub.request_state("r0").add("step", 1)
    sub.session_state("s").add("step", 99)

    assert sub.request_state("r0")["step"] == 1
    assert sub.session_state("s")["step"] == 99


def test_request_cleanup_leaves_the_session_state_standing():
    sub = _Sub()
    sub.request_state("r0").add("step", 1)
    sub.session_state("s").add("carried", 7)

    sub.cleanup_request("r0")

    assert "r0" not in sub.request_states
    assert sub.session_states["s"]["carried"] == 7


def test_session_cleanup_drops_it():
    sub = _Sub()
    sub.session_state("s").add("carried", 7)

    sub.cleanup_session("s")

    assert sub.session_states == {}


def test_cleanup_of_an_unknown_session_is_a_no_op():
    _Sub().cleanup_session("never-existed")


# ── the injected view ───────────────────────────────────────────────────────

def test_the_view_resolves_a_rid_to_its_session_s_state():
    sub = _Sub()
    sub.session_state("s").add("carried", 3)
    view = LazySessionStates(sub, ["r0", "r1"], {"r0": "s"})

    assert view["r0"]["carried"] == 3
    assert view["r1"] is None  # in the batch, but in no session
    assert len(view) == 2
    assert list(view) == ["r0", "r1"]


def test_the_view_refuses_a_rid_outside_the_batch():
    view = LazySessionStates(_Sub(), ["r0"], {"r9": "s"})

    with pytest.raises(KeyError):
        _ = view["r9"]


def test_the_view_creates_the_session_state_on_first_read():
    sub = _Sub()
    view = LazySessionStates(sub, ["r0"], {"r0": "s"})

    view["r0"].add("first", True)

    assert sub.session_states["s"]["first"] is True


def test_engine_inputs_carry_no_session_states_by_default():
    inputs = ModelInputsFromEngine(request_ids=["r0"], per_request_info={})

    assert inputs.per_session_states is None
