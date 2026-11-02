"""A piecewise region's `can_lease` gates the engine's early lease.

The bucket key covers batch size and token count only; a limit it does not
express (a replay's attention segments, say) is the config's `can_lease`. A
batch it rejects gets no lease for that region, and the other regions are
leased as before.
"""

from types import SimpleNamespace

import pytest

from mstar.engine.accelerator_graph_config import PiecewiseConfigType
from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner, PiecewiseGraphKey
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


def _packed_runner(one_row):
    """A real runner over captured packed buckets of 256 tokens, with the capture
    and the step lifecycle stubbed out. A one-row region captures batch size 1
    only (its config refuses more); a per-row one captures 1, 2 and 4."""
    sizes = [1] if one_row else [1, 2, 4]
    runner = PiecewiseAcceleratorGraphRunner.__new__(PiecewiseAcceleratorGraphRunner)
    runner._config = SimpleNamespace(
        lease_before_step=True, can_lease=None, one_row_per_step=one_row,
        get_config_type=lambda: PiecewiseConfigType.PACKED,
        replay_seq_lens=lambda shape, seq_lens, real_bs: list(seq_lens),
    )
    runner._label, runner._current_slot, runner._capture_batch_sizes = "region", 0, sizes
    runner._graphs = {
        PiecewiseGraphKey(bs=bs, seq_len=256, slot=0): SimpleNamespace(
            bucket=(bs, 256), shape=SimpleNamespace(bs=bs), dummy_rids=list(range(bs)),
            static_inputs={}, static_outputs={}, graph=SimpleNamespace(replay=lambda: None),
        )
        for bs in sizes
    }
    runner.declared = []
    runner._declare = lambda ids, seq_lens, bucket, **kw: runner.declared.append((ids, bucket, kw["real_bs"]))
    runner._plan = lambda step, shape: None
    runner._step_runner = SimpleNamespace(commit=lambda step: None)
    runner._dummy_rows = SimpleNamespace(reset=lambda rids: None)
    return runner


def test_a_one_row_per_step_region_leases_and_replays_the_whole_step_as_one_row():
    """A region that replays the whole step as one row (a packed encoder) is
    leased and replayed at batch size 1 for every token of the step, without
    the caller saying so; a per-row region beside it is leased at the step's
    batch size."""
    packed, per_row = _packed_runner(True), _packed_runner(False)
    inputs = [SimpleNamespace(input_seq_len=n) for n in (100, 30, 7)]
    leases = _lease({"packed": packed, "per_row": per_row}, inputs)
    assert leases["packed"].bucket == (1, 256) and leases["packed"].slot == 0
    assert leases["per_row"].bucket == (4, 256)
    assert packed.can_run(3, 137)
    # the replay a lease promised: every segment, one row named by the first request
    packed.run({}, request_ids=[7, 8, 9], seq_lens=[100, 30, 7])
    assert packed.declared == [([7], (1, 256), 1)]


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


@pytest.mark.parametrize("one_row, request_ids, seq_lens, rows", [
    (False, ["a", "b", "pad"], [5, 3, 0], ["a", "b", "pad"]),
    (True, ["a"], [5, 3, 2], ["a", "a", "a"]),
])
def test_attn_key_declares_one_segment_per_row(one_row, request_ids, seq_lens, rows):
    """A packed region naming its attention resource declares a non-causal
    segment per row, all on the one row under `one_row_per_step`, and so does
    a copy of its config."""
    from dataclasses import replace

    from mstar.engine.accelerator_graph_config import PiecewisePackedConfig

    kwargs = {"capture_fn": lambda inp: {}, "make_static_inputs": lambda shape: {}, "attn_key": "attn"}
    if one_row:
        kwargs.update(lease_before_step=True, one_row_per_step=True, capture_batch_sizes=[1])
    config = PiecewisePackedConfig(**kwargs, total_tokens=[8])
    for step in (config.region_step(request_ids, seq_lens),
                 replace(config, total_tokens=[16]).region_step(request_ids, seq_lens)):
        assert [(s.request_id, s.label, s.span) for s in step.segments] == [
            (rid, "main", n) for rid, n in zip(rows, seq_lens, strict=True)
        ]
        assert list(step.steps) == ["attn"] and step.steps["attn"].causal is False
    with pytest.raises(ValueError, match="attn_key"):
        PiecewisePackedConfig(**kwargs, declare_step=lambda ids, lens: None, total_tokens=[8])
