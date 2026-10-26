"""Each graph runtime's stop check (``GraphRuntime.check_stops``) stops what
the declared policy does (``StopPolicy.stops``) on the loop count the step was
built with, through the worker's async decode loop.

A runtime reads each loop's count from its walk state at postprocess time, by
when step N+1 has been speculated; this drives a speculated decode loop in the
worker's stage order and compares at every step's stop check.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_spec_rows import (
    KINDS,
    LOOP,
    NODE,
    PART,
    WALK,
    _admit,
    _advance,
    _first_step,
    _rows_only,
    _speculate,
    _worker,
)

from mstar.model.submodule_base import HostRows, StopKind, StopPolicy

EOS = 7
SAMPLER = "sampler"
POLICY = StopPolicy(StopKind.AR, token_signal="new_token", eos_tokens=(EOS,), sampler=SAMPLER)


def _declared_worker(kind, policy=POLICY):
    w = _worker(kind)
    w._graph_runtime.set_stop_policies(NODE, WALK, {LOOP: policy})
    w._stop_token_signal = {(NODE, WALK): policy.token_signal}
    return w


def _expected(nb, host_rows) -> dict[int, set[str]]:
    tokens = dict(zip(host_rows.request_ids, host_rows.buffers["new_token"][:, 0].tolist(), strict=True))
    out = {}
    for rid in nb.request_ids:
        info = nb.per_request_info_wrapped[rid]
        if rid in tokens and POLICY.stops(
            info.dynamic_loop_iter_counts.get(LOOP, 0), tokens[rid], POLICY.limits(info),
        ):
            out[rid] = {LOOP}
    return out


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(4))
def test_the_runtime_stops_what_the_policy_does_through_a_speculated_loop(kind, seed, monkeypatch):
    rng = np.random.default_rng(seed)
    w = _declared_worker(kind)
    rids = _admit(w, 6)
    for i, rid in enumerate(rids):
        info = w.request_state.get_fwd_info(rid, PART)
        info.max_tokens = 2 + i
        info.resource_configs[SAMPLER] = SimpleNamespace(ignore_eos=bool(rng.integers(2)))
    pending = _first_step(w, rids)
    stopped = 0
    for _ in range(8):
        spec = _speculate(w, pending, rows=True, monkeypatch=monkeypatch)
        assert spec is not None
        w._thread_outputs_to_speculative(spec, _rows_only(pending))
        # postprocess N: N+1 is speculated, N is not routed yet
        nb = pending.node_batch
        # host rows in another order than the step's, and sometimes one row
        # with no output at all
        ids = list(nb.request_ids)
        order = rng.permutation(len(ids))[: len(ids) - int(rng.random() < 0.5)]
        tokens = torch.tensor(rng.choice([EOS, 1, 2], size=len(order)))[:, None]
        host_rows = HostRows(tuple(ids[i] for i in order), {"new_token": tokens})
        got = w._check_stops_in_runtime(pending, host_rows)
        assert got == _expected(nb, host_rows)
        stopped += len(got)
        pending = _advance(w, pending, spec)
    assert stopped, "the limits are small enough that rows reach them"


@pytest.mark.parametrize("kind", KINDS)
def test_limits_are_taken_once_per_request_and_walk_and_dropped_with_it(kind):
    calls = []

    def max_iters(info):
        calls.append(info.rid_handle)
        return 3

    w = _declared_worker(kind, StopPolicy(StopKind.ITER, max_iters=max_iters))
    rt = w._graph_runtime

    def check(rid):
        infos = {rid: w.request_state.get_fwd_info(rid, PART)}
        return rt.check_stops(PART, NODE, WALK, [rid], None, [], infos)

    (rid,) = _admit(w, 1)
    check(rid)
    check(rid)
    assert calls == [rid]
    rt.remove_request(rid)
    w.request_state.remove_request(rid)
    (again,) = _admit(w, 1, start=1)
    assert again == rid, "the handle is recycled"
    check(again)
    assert calls == [rid, rid], "a recycled handle takes its own limits"
