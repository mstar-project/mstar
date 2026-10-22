"""A piecewise region's `can_lease` gates the engine's early lease.

The bucket key covers batch size and token count only; a limit it does not
express (a replay's attention segments, say) is the config's `can_lease`. A
batch it rejects gets no lease for that region, and the other regions are
leased as before.
"""

from types import SimpleNamespace

from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner
from mstar.engine.engine import Engine


class _Runner:
    def __init__(self, lease_before_step=True, accepts=True):
        self.lease_before_step = lease_before_step
        self._accepts = accepts

    def can_lease(self, inputs):
        return self._accepts

    def lease_slot(self, bs, total_tokens):
        return ("lease", bs, total_tokens)


class _Ctx:
    slot_lease = None
    piecewise_leases = None

    def set_piecewise_leases(self, leases):
        self.piecewise_leases = leases


def _lease(runners, inputs):
    engine = SimpleNamespace(_submodules={"node": SimpleNamespace(piecewise_runners=runners)})
    ctx = _Ctx()
    Engine._maybe_lease_piecewise_regions(engine, "node", ctx, inputs)
    return ctx.piecewise_leases


def test_a_refused_region_gets_no_lease_and_the_others_still_do():
    inputs = [SimpleNamespace(input_seq_len=100)]
    leases = _lease({"ok": _Runner(), "refused": _Runner(accepts=False)}, inputs)
    assert leases == {"ok": ("lease", 1, 100)}


def test_nothing_leased_leaves_the_context_alone():
    assert _lease({"refused": _Runner(accepts=False)}, [SimpleNamespace(input_seq_len=8)]) is None


def test_runner_defers_to_the_configs_predicate():
    def runner(predicate):
        return SimpleNamespace(_config=SimpleNamespace(can_lease=predicate))

    assert PiecewiseAcceleratorGraphRunner.can_lease(runner(None), [])
    assert not PiecewiseAcceleratorGraphRunner.can_lease(runner(lambda inputs: len(inputs) < 2), [1, 2])
