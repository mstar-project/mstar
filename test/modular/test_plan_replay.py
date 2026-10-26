"""``PlanReplay`` against a stand-in for a FlashInfer graph wrapper: it learns
the plan from the wrapper, replays it identically, keeps the wrapper's own
plan for what it cannot learn, and stops replaying when the plan changes."""
from types import SimpleNamespace

import numpy as np
import torch

from mstar.engine.resources.attn import plan_replay
from mstar.engine.resources.attn.plan_replay import PlanReplay

ROWS = 4


class _Wrapper:
    """Shaped like FlashInfer's: copies the step's arrays into fixed buffers,
    sets per-step attributes, and calls a shared planner module with the
    arrays and fixed arguments."""

    def __init__(self, extra=None):
        self.device = torch.device("cpu")
        self._workspace = torch.zeros(8, dtype=torch.int64)
        self._indptr_buf = torch.zeros(ROWS + 1, dtype=torch.int32)
        self._lens_buf = torch.zeros(64, dtype=torch.int32)
        self._range = torch.arange(ROWS + 1, dtype=torch.int32)
        self._cached_module = None
        self._plan_info = None
        # what a later FlashInfer might also set per step
        self.extra = extra
        self.plans = [0]  # in place: an attribute that changed would be learned from

    def plan(self, kv_indptr, kv_lens, causal):
        self.plans[0] += 1
        self._cached_module = _MODULE
        self._indptr_buf.copy_(kv_indptr)
        self._lens_buf[: kv_lens.shape[0]].copy_(kv_lens)
        self._max_kv_len = int(kv_lens.max())
        self._seq_lens = kv_lens
        self._causal = causal
        if self.extra is not None:
            self._extra = self.extra(kv_lens)
        self._plan_info = _MODULE.plan(
            self._workspace, self._range, kv_indptr, kv_lens, ROWS, causal,
        )


def _planner(workspace, qo, kv_indptr, kv_lens, rows, causal):
    workspace[0] = int(kv_lens.sum())
    return [int(kv_indptr[-1]), int(kv_lens.sum()), rows, int(causal)]


_MODULE = SimpleNamespace(plan=_planner)


def _step(rng, i=0):
    lens = rng.integers(1, 40, ROWS)
    lens[0] = 50 + i
    lens[-1] = i + 1
    npages = (lens + 15) // 16
    t = lambda x: torch.from_numpy(np.asarray(x, dtype=np.int32))  # noqa: E731
    return dict(kv_indptr=t(np.concatenate(([0], np.cumsum(npages)))), kv_lens=t(lens))


def _synthetic(n):
    return [_step(np.random.default_rng(100 + i), i) for i in range(n)]


def _state(w):
    return (
        w._indptr_buf.tolist(), w._lens_buf.tolist(), w._workspace.tolist(), w._plan_info,
        w._max_kv_len, w._seq_lens.tolist(), w._causal, getattr(w, "_extra", None),
    )


def _drive(replayed: _Wrapper, steps: int, change_at: int | None = None):
    """Plan the same random steps on ``replayed`` through ``PlanReplay`` and
    on a twin through its own plan; their states must agree every step."""
    twin = _Wrapper(replayed.extra)
    replay = PlanReplay(replayed, _synthetic)
    rng = np.random.default_rng(0)
    for i in range(steps):
        if i == change_at:
            replayed.extra = twin.extra = lambda lens: int(lens[0]) * 3
        arrays = _step(rng)

        def own_plan(a):
            replayed.plan(a["kv_indptr"], a["kv_lens"], causal=True)

        replay.plan((torch.bfloat16, True), arrays, own_plan)
        twin.plan(arrays["kv_indptr"], arrays["kv_lens"], causal=True)
        assert _state(replayed) == _state(twin), f"step {i}"
    return replay


def test_a_replay_plans_as_the_wrapper_does():
    w = _Wrapper()
    steps = plan_replay._CHECK_FIRST + 40
    _drive(w, steps)
    # learning and the checked steps used the wrapper's plan; the rest replayed
    assert w.plans[0] == plan_replay._N_SYNTHETIC + plan_replay._CHECK_FIRST


def test_what_cannot_be_learned_keeps_the_wrappers_plan():
    w = _Wrapper(extra=lambda lens: float(lens.sum()) / 7)  # no derivation gives this
    replay = _drive(w, 30)
    assert all(r is None for r in replay._replays.values())


def test_a_plan_that_changes_turns_the_replay_off():
    w = _Wrapper()
    replay = _drive(w, plan_replay._CHECK_FIRST + 10, change_at=3)
    assert all(r is None for r in replay._replays.values())
