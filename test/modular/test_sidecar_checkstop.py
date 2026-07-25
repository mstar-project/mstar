"""MSTAR_SIDECAR_CHECKSTOP: unit tests for the deferred-consume check_stop offload.

Scope of this build is the deferred-consume + SAME-STEP decision only (the
fuller sidecar-EOS/StopFeedback path is not built — see the worker flag
comment). These tests cover the two pieces that changed, CPU-only:

- ``Worker._compute_new_stops``: the stop-state COMPUTATION extracted verbatim
  from ``_postprocess_batch`` so shadow mode can recompute it. Must reproduce
  the legacy fast-path decisions (thinker EOS / ignore_eos / max_tokens; talker
  codec_eos / talker_max_tokens) and fall back to ``engine.check_stop_for_batch``
  when there is no uniform flat buffer.
- ``Worker._await_checkstop`` / ``Worker._checkstop_barrier``: the poll-or-wait
  barrier and its counters (checkstop_deferred_consume / checkstop_sync_fallback)
  plus the no-op path when there is no deferred copy.

The methods only touch a handful of ``self`` attributes, so a SimpleNamespace
stub with the real ``_ws_inc`` bound to it drives the exact production code.
"""

from __future__ import annotations

import types

import torch

from mstar.worker.worker import Worker


def _stub(**attrs):
    """A bare object exposing the attributes the method-under-test reads, with
    the REAL ``_ws_inc`` bound so counter behavior is exercised for real."""
    s = types.SimpleNamespace(_walk_stats={}, **attrs)
    s._ws_inc = types.MethodType(Worker._ws_inc, s)
    return s


def _info(*, ignore_eos=False, iters=0, max_tokens=100, talker_max=None):
    step_metadata = {} if talker_max is None else {"talker_max_tokens": talker_max}
    return types.SimpleNamespace(
        sampling_config={"Thinker": types.SimpleNamespace(ignore_eos=ignore_eos)},
        dynamic_loop_iter_counts={
            "thinker_decode_loop": iters,
            "talker_decode_loop": iters,
        },
        max_tokens=max_tokens,
        step_metadata=step_metadata,
    )


def _batch(walk, per_request_info):
    node_batch = types.SimpleNamespace(
        per_request_info=per_request_info, node_name="Thinker",
    )
    return types.SimpleNamespace(
        graph_walk=walk, node_name="Thinker", node_batch=node_batch,
    )


def _cpu_output(tokens, rids):
    out = types.SimpleNamespace()
    out._checkstop_flat = torch.tensor(tokens, dtype=torch.long)
    out._checkstop_rids = rids
    return out


# --------------------------------------------------------------------------
# _compute_new_stops — thinker fast path
# --------------------------------------------------------------------------

def test_thinker_eos_stops():
    self = _stub(_fast_checkstop=True, _fast_checkstop_talker=False,
                 _thinker_eos_id=7, _talker_codec_eos_id=None)
    batch_N = _batch("thinker_decode", {"r0": _info(iters=3), "r1": _info(iters=3)})
    cpu_output = _cpu_output([7, 4], ["r0", "r1"])
    stops = Worker._compute_new_stops(self, batch_N, engine=None, cpu_output=cpu_output)
    assert stops == {"r0": {"thinker_decode_loop"}}


def test_thinker_ignore_eos_suppresses_stop():
    self = _stub(_fast_checkstop=True, _fast_checkstop_talker=False,
                 _thinker_eos_id=7, _talker_codec_eos_id=None)
    batch_N = _batch("thinker_decode", {"r0": _info(ignore_eos=True, iters=3)})
    cpu_output = _cpu_output([7], ["r0"])
    stops = Worker._compute_new_stops(self, batch_N, engine=None, cpu_output=cpu_output)
    assert stops == {}


def test_thinker_max_tokens_stops():
    self = _stub(_fast_checkstop=True, _fast_checkstop_talker=False,
                 _thinker_eos_id=7, _talker_codec_eos_id=None)
    # iter+1 >= max_tokens with a non-EOS token.
    batch_N = _batch("thinker_decode", {"r0": _info(iters=99, max_tokens=100)})
    cpu_output = _cpu_output([4], ["r0"])
    stops = Worker._compute_new_stops(self, batch_N, engine=None, cpu_output=cpu_output)
    assert stops == {"r0": {"thinker_decode_loop"}}


def test_thinker_eos_id_lazy_from_engine():
    self = _stub(_fast_checkstop=True, _fast_checkstop_talker=False,
                 _thinker_eos_id=None, _talker_codec_eos_id=None)
    submod = types.SimpleNamespace(
        config=types.SimpleNamespace(im_end_token_id=9))
    engine = types.SimpleNamespace(
        submodule_management={"Thinker": types.SimpleNamespace(submodule=submod)})
    batch_N = _batch("thinker_decode", {"r0": _info(iters=0)})
    cpu_output = _cpu_output([9], ["r0"])
    stops = Worker._compute_new_stops(self, batch_N, engine=engine, cpu_output=cpu_output)
    assert stops == {"r0": {"thinker_decode_loop"}}
    assert self._thinker_eos_id == 9  # cached


# --------------------------------------------------------------------------
# _compute_new_stops — talker fast path
# --------------------------------------------------------------------------

def test_talker_codec_eos_stops_and_counts():
    self = _stub(_fast_checkstop=False, _fast_checkstop_talker=True,
                 _thinker_eos_id=None, _talker_codec_eos_id=5)
    batch_N = _batch("talker_decode", {"r0": _info(iters=2, talker_max=50)})
    cpu_output = _cpu_output([5], ["r0"])
    stops = Worker._compute_new_stops(self, batch_N, engine=None, cpu_output=cpu_output)
    assert stops == {"r0": {"talker_decode_loop"}}
    # The talker fast path bumps its mechanism counter.
    assert self._walk_stats.get("talker_fast_checkstop_steps") == 1


# --------------------------------------------------------------------------
# _compute_new_stops — generic fallback
# --------------------------------------------------------------------------

def test_generic_fallback_to_engine():
    self = _stub(_fast_checkstop=True, _fast_checkstop_talker=False,
                 _thinker_eos_id=7, _talker_codec_eos_id=None)
    sentinel = {"rX": {"loopY"}}
    calls = {}

    def fake_check(node_batch, cpu_output):
        calls["hit"] = (node_batch, cpu_output)
        return sentinel

    engine = types.SimpleNamespace(check_stop_for_batch=fake_check)
    # No _checkstop_flat => not a uniform batch => generic path.
    cpu_output = types.SimpleNamespace()
    nb = types.SimpleNamespace(per_request_info={}, node_name="Thinker")
    batch_N = types.SimpleNamespace(graph_walk="prefill_text", node_name="Thinker",
                                    node_batch=nb)
    stops = Worker._compute_new_stops(self, batch_N, engine=engine, cpu_output=cpu_output)
    assert stops is sentinel
    assert calls["hit"] == (nb, cpu_output)


# --------------------------------------------------------------------------
# _await_checkstop — poll vs blocking fallback
# --------------------------------------------------------------------------

class _FakeEvent:
    def __init__(self, ready):
        self._ready = ready
        self.synced = False

    def query(self):
        return self._ready

    def synchronize(self):
        self.synced = True


def test_await_ready_consumes_without_wait():
    self = _stub()
    ev = _FakeEvent(ready=True)
    cpu_output = types.SimpleNamespace(_checkstop_event=ev)
    Worker._await_checkstop(self, cpu_output)
    assert self._walk_stats.get("checkstop_deferred_consume") == 1
    assert "checkstop_sync_fallback" not in self._walk_stats
    assert ev.synced is False


def test_await_not_ready_falls_back_to_wait():
    self = _stub()
    ev = _FakeEvent(ready=False)
    cpu_output = types.SimpleNamespace(_checkstop_event=ev)
    Worker._await_checkstop(self, cpu_output)
    assert self._walk_stats.get("checkstop_sync_fallback") == 1
    assert "checkstop_deferred_consume" not in self._walk_stats
    assert ev.synced is True


def test_await_no_event_is_noop():
    self = _stub()
    cpu_output = types.SimpleNamespace()  # no _checkstop_event (non-CUDA path)
    Worker._await_checkstop(self, cpu_output)
    assert self._walk_stats == {}


# --------------------------------------------------------------------------
# _checkstop_barrier — defer vs legacy termination
# --------------------------------------------------------------------------

class _FakeSide:
    def __init__(self):
        self.synced = False

    def synchronize(self):
        self.synced = True


def test_barrier_legacy_blocks_and_returns_none():
    self = _stub(_checkstop_event=None)
    side = _FakeSide()
    ev = Worker._checkstop_barrier(self, side, defer=False)
    assert ev is None
    assert side.synced is True


def test_barrier_defer_records_without_blocking():
    recorded = {}

    class _RecEvent:
        def record(self, stream):
            recorded["stream"] = stream

    self = _stub(_checkstop_event=_RecEvent())
    side = _FakeSide()
    ev = Worker._checkstop_barrier(self, side, defer=True)
    assert ev is self._checkstop_event
    assert recorded["stream"] is side
    assert side.synced is False  # deferred: no blocking wait
