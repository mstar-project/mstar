"""The graph runtime's stop check (``check_stops_inline``) decides what the
engine's declared-policy check does, through the worker's async decode loop.

The runtime reads each loop's count from its walk state at postprocess time,
where the engine reads the count the step was built with; this drives a
speculated decode loop in the worker's stage order and compares the two at
every step's stop check.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_spec_rows import (
    LOOP,
    NODE,
    PART,
    _admit,
    _advance,
    _first_step,
    _have_rust,
    _rows_only,
    _speculate,
    _worker,
)

from mstar.engine.engine import Engine
from mstar.model.submodule_base import HostRows, StopKind, StopPolicy

pytestmark = pytest.mark.skipif(not _have_rust(), reason="mstar_rust not built")

EOS = 7
SAMPLER = "sampler"


def _engine():
    policy = StopPolicy(StopKind.AR, LOOP, token_signal="new_token", eos_tokens=(EOS,), sampler=SAMPLER)
    eng = Engine.__new__(Engine)
    eng._submodules = {NODE: SimpleNamespace(submodule=SimpleNamespace(get_stop_policies=lambda walk: (policy,)))}
    eng._stop_policies = {}
    eng._stop_limits = {}
    return eng


def _worker_with_limits(n, rng):
    w = _worker("rust")
    w._stop_rules_declared = set()
    rids = _admit(w, n)
    for i, rid in enumerate(rids):
        info = w.request_state.get_fwd_info(rid, PART)
        info.max_tokens = 2 + i
        info.resource_configs[SAMPLER] = SimpleNamespace(ignore_eos=bool(rng.integers(2)))
    return w, rids


@pytest.mark.parametrize("seed", range(4))
def test_runtime_stop_check_matches_the_engine_through_a_speculated_loop(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    w, rids = _worker_with_limits(6, rng)
    eng = _engine()
    pending = _first_step(w, rids)
    stopped = 0
    for _ in range(8):
        spec = _speculate(w, pending, rows=True, monkeypatch=monkeypatch)
        assert spec is not None
        w._thread_outputs_to_speculative(spec, _rows_only(pending))
        # postprocess N: N+1 is speculated, N is not routed yet
        nb = pending.node_batch
        tokens = torch.tensor(rng.choice([EOS, 1, 2], size=len(nb.request_ids)))[:, None]
        host_rows = HostRows(tuple(nb.request_ids), {"new_token": tokens})
        expected = Engine.check_stop_for_batch(eng, nb, {}, host_rows=host_rows)
        got = w._check_stops_in_runtime(pending, eng, host_rows)
        assert got == expected
        stopped += len(got)
        pending = _advance(w, pending, spec)
    assert stopped, "the limits are small enough that rows reach them"


def test_rows_out_of_forward_order_and_without_a_row():
    rng = np.random.default_rng(0)
    w, rids = _worker_with_limits(4, rng)
    eng = _engine()
    pending = _first_step(w, rids)
    nb = pending.node_batch
    # forward order reversed, and the last rid has no row (no output)
    row_rids = tuple(reversed(nb.request_ids[:-1]))
    tokens = torch.tensor([EOS, EOS, EOS])
    host_rows = HostRows(row_rids, {"new_token": tokens})
    expected = Engine.check_stop_for_batch(eng, nb, {}, host_rows=host_rows)
    assert w._check_stops_in_runtime(pending, eng, host_rows) == expected
    assert nb.request_ids[-1] not in expected


def test_limits_go_with_the_request():
    rng = np.random.default_rng(1)
    w, rids = _worker_with_limits(2, rng)
    eng = _engine()
    pending = _first_step(w, rids)
    host_rows = HostRows(tuple(pending.node_batch.request_ids), {"new_token": torch.tensor([1, 1])})
    w._check_stops_in_runtime(pending, eng, host_rows)
    rt = w._graph_runtime
    _, _, missing = rt.check_stops_inline(
        PART, NODE, pending.node_batch.step_context.graph_walk, rids,
        torch.tensor([1, 1]), list(rids),
    )
    assert missing == []
    rt.remove_request(rids[0])
    rt.remove_request(rids[1])
    again = _admit(w, 2, start=10)
    _, _, missing = rt.check_stops_inline(
        PART, NODE, pending.node_batch.step_context.graph_walk, again,
        torch.tensor([1, 1]), list(again),
    )
    assert sorted(missing) == sorted(again), "a recycled handle has no limits"
