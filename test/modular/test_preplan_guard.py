"""A staged pre-plan promotes only into the step it was planned for.

The plan thread pre-plans the speculated next step of a node while the current
one runs; if the scheduler then dispatches a *different* batch first (a new
request's prefill while a decode step sits pre-planned), that batch must plan
inline and the staged plan must be dropped — not served the other step's
pages (KV) or dereferenced against a lease it does not hold (sampler).
"""

from __future__ import annotations

import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources.kv.config import KVConfig, KVStep
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import Segment, StepContext

PS = 8


def _manager() -> KVManager:
    cfg = KVConfig(num_layers=1, num_kv_heads=1, head_dim=4, max_seq_len=64 * PS, max_num_pages=64, page_size=PS)
    return KVManager(
        cfg=cfg, name="kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _ctx(*rids: str, preplan: bool = False) -> StepContext:
    return StepContext(request_ids=tuple(rids), graph_walk="walk", slot=0, capture=False, is_preplan=preplan)


def _grow(m: KVManager, rid: str, span: int) -> None:
    step = KVStep(segments=(Segment(rid, "main", span),), commit=True)
    ctx = _ctx(rid)
    assert m.admit(step, ctx).ok
    m.plan(step, ctx)
    m.commit(step, ctx)


def test_kv_preplan_promotes_only_into_its_own_step() -> None:
    m = _manager()
    for rid in ("a", "b"):
        m.ingest_request(rid)
    _grow(m, "a", 2 * PS)

    # Stage a's next (decode) step ahead.
    step_a = KVStep(segments=(Segment("a", "main", 1),), commit=True)
    ctx_a = _ctx("a", preplan=True)
    assert m.admit(step_a, ctx_a).ok
    staged = m.plan(step_a, ctx_a)
    assert m._preplanned and staged["main"].views[0].request_id == "a"

    # A different batch (b's prefill) reaches the GPU thread first: it is
    # planned inline, on its own pages, and the staged plan is dropped.
    step_b = KVStep(segments=(Segment("b", "main", 3 * PS),), commit=True)
    ctx_b = _ctx("b")
    assert m.admit(step_b, ctx_b).ok
    out_b = m.plan(step_b, ctx_b)
    assert not m._preplanned
    (view,) = out_b["main"].views
    assert view.request_id == "b" and view.to_compute == 3 * PS
    m.commit(step_b, ctx_b)
    assert m._streams["b"]["main"].stored_len == 3 * PS

    # a's step then plans inline like any un-staged step and commits its token.
    assert m.admit(step_a, _ctx("a")).ok
    out_a = m.plan(step_a, _ctx("a"))
    assert out_a["main"].views[0].request_id == "a"
    m.commit(step_a, _ctx("a"))
    assert m._streams["a"]["main"].stored_len == 2 * PS + 1


def test_kv_preplan_still_promotes_for_its_own_step() -> None:
    m = _manager()
    m.ingest_request("a")
    _grow(m, "a", PS)
    step = KVStep(segments=(Segment("a", "main", 1),), commit=True)
    assert m.admit(step, _ctx("a", preplan=True)).ok
    staged = m.plan(step, _ctx("a", preplan=True))
    promoted = m.plan(step, _ctx("a"))
    assert promoted is staged and not m._preplanned
    m.commit(step, _ctx("a"))
    assert m._streams["a"]["main"].stored_len == PS + 1


# ── runner-level: the stage is all-or-nothing across resources ──────────────
#
# The attention wrappers and the position manager promote whatever is staged
# when their `plan` is next called; they plan against the KV plan output, so
# a KV manager that drops its stage for a foreign step while they promote
# theirs would attend with wrappers laid out for another step's rows
# (observed on H100 as FlashInfer's "q implies q_len_per_req=5 but plan()
# used 1" once five decode rows followed a one-row pre-plan). The runner sees
# the whole step, so it drops the stage on every resource before a foreign
# step admits or plans.

from mstar.engine.resources.base import Resource  # noqa: E402
from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.engine.resources.step import ResourceStep, SlotLease, SubmoduleStep  # noqa: E402


class _Blind(Resource):
    """Promotes a staged pre-plan into whichever step calls `plan` next."""

    def __init__(self, deps: tuple[str, ...] = ()):
        self._deps = set(deps)
        self._preplanned = False
        self.events: list[str] = []

    @classmethod
    def build(cls, spec, info):  # pragma: no cover - not built from a spec here
        raise NotImplementedError

    def depends_on(self):
        return set(self._deps)

    @property
    def supports_preplan(self):
        return True

    def plan(self, step, ctx):
        if self._preplanned:
            self._preplanned = False
            self.events.append("promote")
            return "staged"
        self._preplanned = ctx.is_preplan
        self.events.append("pre_plan" if ctx.is_preplan else "plan")
        return "fresh"

    def clear_preplan(self):
        self._preplanned = False
        self.events.append("clear")


def _step(*rids: str, span: int = 1, slot: int | None = 1, preplan: bool = False) -> SubmoduleStep:
    step = SubmoduleStep(
        steps={"kv": ResourceStep(), "attn": ResourceStep()},
        segments=[Segment(rid, "main", span) for rid in rids],
    )
    lease = None if slot is None else SlotLease(slot=slot, bucket=None)
    step.set_ctx(StepContext(
        request_ids=tuple(rids), graph_walk="decode", slot=slot or 0, capture=False,
        is_preplan=preplan, slot_lease=lease,
    ))
    return step


def _runner() -> tuple[StepRunner, _Blind, _Blind]:
    kv, attn = _Blind(), _Blind(deps=("kv",))
    return StepRunner({"kv": kv, "attn": attn}), kv, attn


def test_runner_drops_the_stage_on_every_resource_for_a_foreign_step() -> None:
    runner, kv, attn = _runner()
    staged = _step("a", preplan=True)
    assert runner.pre_admit(staged).ok
    runner.pre_plan(staged)
    assert kv.events == ["pre_plan"] and attn.events == ["pre_plan"]

    # b's prefill (more rows, no lease) reaches the GPU thread first: both
    # resources drop the stage and plan b afresh — the blind promoter included.
    foreign = _step("b", span=17, slot=None)
    assert runner.admit(foreign).ok
    out = runner.plan(foreign)
    assert out == {"kv": "fresh", "attn": "fresh"}
    assert kv.events == ["pre_plan", "clear", "plan"]
    assert attn.events == ["pre_plan", "clear", "plan"]

    # a's own step then plans inline like any un-staged step.
    assert runner.plan(_step("a")) == {"kv": "fresh", "attn": "fresh"}
    assert attn.events[-1] == "plan"


def test_runner_promotes_the_stage_into_its_own_step_only() -> None:
    runner, kv, attn = _runner()
    runner.pre_plan(_step("a", "b", preplan=True))
    assert runner.plan(_step("a", "b")) == {"kv": "staged", "attn": "staged"}
    assert kv.events == ["pre_plan", "promote"] and attn.events == ["pre_plan", "promote"]
    # consumed: nothing stays staged for the next step to pick up
    assert runner.plan(_step("a", "b")) == {"kv": "fresh", "attn": "fresh"}

    # the same rows re-declared without their lease are a different step
    runner.pre_plan(_step("a", "b", preplan=True))
    assert runner.plan(_step("a", "b", slot=None)) == {"kv": "fresh", "attn": "fresh"}
    assert attn.events[-2:] == ["clear", "plan"]

    # and so are the same rows on another slot, or with another span
    runner.pre_plan(_step("a", "b", preplan=True))
    assert runner.plan(_step("a", "b", slot=2)) == {"kv": "fresh", "attn": "fresh"}
    runner.pre_plan(_step("a", "b", preplan=True))
    assert runner.plan(_step("a", "b", span=4)) == {"kv": "fresh", "attn": "fresh"}


def test_runner_clear_preplan_reaches_every_resource() -> None:
    runner, kv, attn = _runner()
    runner.pre_plan(_step("a", preplan=True))
    runner.clear_preplan()
    assert kv.events[-1] == "clear" and attn.events[-1] == "clear"
    assert runner.plan(_step("a")) == {"kv": "fresh", "attn": "fresh"}
