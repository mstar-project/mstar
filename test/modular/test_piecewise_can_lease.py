"""A piecewise region's `can_lease` gates the engine's early lease.

The bucket key covers batch size and token count only; a limit it does not
express (a replay's attention segments, say) is the config's `can_lease`. A
batch it rejects gets no lease for that region, and the other regions are
leased as before.
"""

from types import SimpleNamespace

import pytest

from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner
from mstar.engine.engine import Engine


class _Runner:
    one_row_per_step = False

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


def test_a_one_row_per_step_region_leases_the_whole_step_as_one_row():
    """A region that replays the whole step as one row (a packed encoder) is
    leased at batch size 1 for every token of the step, through the real
    runner; a per-row region beside it is leased at the step's batch size."""
    def real_runner(one_row):
        runner = PiecewiseAcceleratorGraphRunner.__new__(PiecewiseAcceleratorGraphRunner)
        runner._config = SimpleNamespace(lease_before_step=True, can_lease=None, one_row_per_step=one_row)
        runner._current_slot = 0
        runner.resolved = []
        runner._resolve = lambda bs, total: runner.resolved.append((bs, total)) or SimpleNamespace(bucket=(bs, total))
        return runner

    packed, per_row = real_runner(True), real_runner(False)
    inputs = [SimpleNamespace(input_seq_len=n) for n in (100, 30, 7)]
    leases = _lease({"packed": packed, "per_row": per_row}, inputs)
    assert packed.resolved == [(1, 137)] and per_row.resolved == [(3, 137)]
    assert leases["packed"].bucket == (1, 137) and leases["packed"].slot == 0
    assert leases["per_row"].bucket == (3, 137)


@pytest.mark.parametrize("batched, overrides", [
    (False, {"capture_batch_sizes": [1, 2]}),
    (False, {"lease_before_step": False}),
    (True, {}),
])
def test_one_row_per_step_is_refused_where_it_means_nothing(batched, overrides):
    from mstar.engine.accelerator_graph_config import PiecewiseBatchedConfig, PiecewisePackedConfig

    kwargs = {
        "capture_fn": lambda inp: {}, "make_static_inputs": lambda shape: {},
        "lease_before_step": True, "one_row_per_step": True, "capture_batch_sizes": [1],
    }
    PiecewisePackedConfig(**kwargs, total_tokens=[8])  # the shape it is for
    PiecewisePackedConfig(**{**kwargs, "capture_batch_sizes": (1,)}, total_tokens=[8])
    with pytest.raises(ValueError, match="one_row_per_step"):
        if batched:
            PiecewiseBatchedConfig(**kwargs, seq_len=8)
        else:
            PiecewisePackedConfig(**{**kwargs, **overrides}, total_tokens=[8])
