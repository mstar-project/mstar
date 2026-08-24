"""Unit tests for the resource lifecycle runner and the step envelope.

``StepRunner`` is meant to be kind-blind: every method is a sweep over
topo-sorted resource keys calling one lifecycle method, and the only value
it moves between resources is each ``plan``'s return, filed under that
resource's key. These tests pin that behaviour with recording stubs, so a
future kind (one with no segments, one that publishes nothing, one that
narrows) cannot quietly acquire a special case in the runner.
"""

from __future__ import annotations

import subprocess
import sys

sys.path.insert(0, ".")

import pytest

from mstar.engine.resources import (
    AdmitOutcome,
    AllocationFailed,
    KVStep,
    Resource,
    Segment,
    StepContext,
    StepRunner,
    SubmoduleStep,
    topo_sort,
)


class _Stub(Resource):
    """Records every lifecycle call it receives, in order."""

    def __init__(
        self,
        name: str,
        deps: tuple[str, ...] = (),
        plan_value=None,
        admit_outcome: AdmitOutcome | None = None,
        published=None,
        narrowed=None,
    ):
        self.name = name
        self._deps = set(deps)
        self._plan_value = plan_value if plan_value is not None else f"{name}-plan"
        self._admit_outcome = admit_outcome
        self._published = published
        self._narrowed = narrowed
        self.calls: list[str] = []
        # plan_results as this resource saw it when its own plan ran
        self.deps_seen: dict | None = None

    @classmethod
    def build(cls, spec, device, comm_group, **engine_kwargs):
        raise NotImplementedError("stub is constructed directly")

    def depends_on(self) -> set[str]:
        return set(self._deps)

    def ingest_request(self, rid, overrides):
        self.calls.append(f"ingest:{rid}:{overrides}")

    def remove_request(self, rid):
        self.calls.append(f"remove:{rid}")

    def admit(self, step, ctx):
        self.calls.append("admit")
        return self._admit_outcome or AdmitOutcome(ok=True)

    def narrow(self, step, ctx):
        self.calls.append("narrow")
        return self._narrowed

    def plan(self, step, ctx):
        self.calls.append("plan")
        self.deps_seen = dict(ctx.plan_results)
        return self._plan_value

    def commit(self, step, ctx):
        self.calls.append("commit")

    def publish(self, request_id):
        self.calls.append(f"publish:{request_id}")
        return self._published

    def build_cuda_graph_buffers(self, slots, max_bs, max_seq_len):
        self.calls.append(f"cg_buffers:{len(slots)}:{max_bs}:{max_seq_len}")


def _ctx(request_ids=("r1",), slot=0):
    return StepContext(
        request_ids=tuple(request_ids),
        graph_walk="decode",
        slot=slot,
        capture=False,
    )


def _step(keys, segments=(("r1", "main", 1),), ctx=None):
    segs = tuple(Segment(*s) for s in segments)
    return SubmoduleStep(
        ctx=ctx or _ctx(),
        segments=segs,
        steps={key: KVStep(segments=segs) for key in keys},
    )


# --- topo_sort -------------------------------------------------------------


def test_topo_sort_orders_dependencies_first():
    resources = {
        "attn": _Stub("attn", deps=("kv",)),
        "kv": _Stub("kv"),
        "rope": _Stub("rope", deps=("kv",)),
    }
    order = topo_sort(resources)
    assert order[0] == "kv"
    assert set(order[1:]) == {"attn", "rope"}


def test_topo_sort_breaks_ties_alphabetically():
    """Independent resources have unspecified relative order by contract;
    pinning it keeps plan sweeps reproducible run to run."""
    resources = {"zulu": _Stub("zulu"), "alpha": _Stub("alpha"), "mike": _Stub("mike")}
    assert topo_sort(resources) == ("alpha", "mike", "zulu")


def test_topo_sort_is_transitive():
    resources = {
        "c": _Stub("c", deps=("b",)),
        "b": _Stub("b", deps=("a",)),
        "a": _Stub("a"),
    }
    assert topo_sort(resources) == ("a", "b", "c")


def test_topo_sort_rejects_unknown_dependency():
    with pytest.raises(ValueError, match="unknown key"):
        topo_sort({"attn": _Stub("attn", deps=("kv",))})


def test_topo_sort_rejects_self_dependency():
    with pytest.raises(ValueError, match="depends on itself"):
        topo_sort({"kv": _Stub("kv", deps=("kv",))})


def test_topo_sort_rejects_cycle():
    resources = {
        "a": _Stub("a", deps=("b",)),
        "b": _Stub("b", deps=("a",)),
    }
    with pytest.raises(ValueError, match="cycle"):
        topo_sort(resources)


# --- plan ------------------------------------------------------------------


def test_plan_runs_in_dependency_order_and_threads_plan_results():
    """The plan value is the whole protocol between resources: KV's return
    must be visible to attention, and neither goes through the runner."""
    kv = _Stub("kv", plan_value={"views": "kv-views"})
    attn = _Stub("attn", deps=("kv",))
    runner = StepRunner({"attn": attn, "kv": kv})

    step = _step(["kv", "attn"])
    results = runner.plan(step)

    assert kv.deps_seen == {}, "kv plans first, so it sees no dependency values"
    assert attn.deps_seen == {"kv": {"views": "kv-views"}}
    assert results == {"kv": {"views": "kv-views"}, "attn": "attn-plan"}
    assert step.ctx.plan_results is results, "results are filed on the context"


def test_plan_clears_stale_results_from_a_reused_context():
    kv = _Stub("kv")
    runner = StepRunner({"kv": kv})
    ctx = _ctx()
    ctx.plan_results["ghost"] = "from a previous step"

    runner.plan(_step(["kv"], ctx=ctx))

    assert "ghost" not in ctx.plan_results


def test_plan_skips_resources_the_step_does_not_declare():
    kv, sampler = _Stub("kv"), _Stub("sampler")
    runner = StepRunner({"kv": kv, "sampler": sampler})

    runner.plan(_step(["kv"]))

    assert kv.calls == ["plan"]
    assert sampler.calls == [], "an undeclared resource does not plan"


# --- admit -----------------------------------------------------------------


def test_admit_short_circuits_and_preserves_the_failure_reason():
    """Admission failure is a scheduling signal carrying a diagnostic
    payload, not an exception — the caller needs which request, which
    label, and how many pages short."""
    reason = AllocationFailed(
        message="Not enough free pages", pages_short=3, label="main", request_id="r1",
    )
    kv = _Stub("kv", admit_outcome=AdmitOutcome(ok=False, reason=reason))
    attn = _Stub("attn", deps=("kv",))
    runner = StepRunner({"attn": attn, "kv": kv})

    outcome = runner.admit(_step(["kv", "attn"]))

    assert outcome.ok is False
    assert outcome.reason is reason
    assert attn.calls == [], "nothing downstream of the failure admits"


def test_admit_reports_not_ready_when_any_resource_is_pending():
    kv = _Stub("kv", admit_outcome=AdmitOutcome(ok=True, ready=False))
    runner = StepRunner({"kv": kv, "sampler": _Stub("sampler")})

    outcome = runner.admit(_step(["kv", "sampler"]))

    assert outcome.ok is True
    assert outcome.ready is False


def test_admit_is_ready_when_every_resource_is():
    runner = StepRunner({"kv": _Stub("kv"), "sampler": _Stub("sampler")})
    outcome = runner.admit(_step(["kv", "sampler"]))
    assert (outcome.ok, outcome.ready) == (True, True)


# --- narrow ----------------------------------------------------------------


def test_narrow_returns_the_same_step_when_nothing_narrows():
    runner = StepRunner({"kv": _Stub("kv")})
    step = _step(["kv"])
    assert runner.narrow(step) is step


def test_narrow_replaces_only_the_steps_that_narrowed():
    narrowed = KVStep(segments=(Segment("r1", "main", 0),))
    kv = _Stub("kv", narrowed=narrowed)
    attn = _Stub("attn", deps=("kv",))
    runner = StepRunner({"attn": attn, "kv": kv})

    step = _step(["kv", "attn"])
    out = runner.narrow(step)

    assert out is not step
    assert out.get("kv") is narrowed
    assert out.get("attn") is step.get("attn")
    assert out.segments == step.segments, (
        "narrow rewrites per-resource steps only; whether the authoritative "
        "batch layout follows is the open bucket-ordering question"
    )


# --- commit / publish ------------------------------------------------------


def test_commit_sweeps_declared_resources_in_plan_order():
    kv, attn = _Stub("kv"), _Stub("attn", deps=("kv",))
    runner = StepRunner({"attn": attn, "kv": kv})

    runner.commit(_step(["kv", "attn"]))

    assert kv.calls == ["commit"] and attn.calls == ["commit"]


def test_publish_collects_per_request_per_key_and_omits_the_silent():
    kv = _Stub("kv", published="kv-info")
    sampler = _Stub("sampler", published=None)
    runner = StepRunner({"kv": kv, "sampler": sampler})

    out = runner.publish(["r1", "r2"])

    assert out == {"r1": {"kv": "kv-info"}, "r2": {"kv": "kv-info"}}
    assert "publish:r2" in sampler.calls, "asked, but returned nothing durable"


def test_publish_sweeps_every_resource_not_just_a_step_s_keys():
    """Publish is not step-scoped: it runs after execution over whatever the
    node holds, so a resource that sat out the step still describes itself."""
    kv = _Stub("kv", published="kv-info")
    runner = StepRunner({"kv": kv})
    assert runner.publish(["r1"]) == {"r1": {"kv": "kv-info"}}


# --- request lifetime ------------------------------------------------------


def test_ingest_request_routes_overrides_by_resource_key():
    kv, sampler = _Stub("kv"), _Stub("sampler")
    runner = StepRunner({"kv": kv, "sampler": sampler})

    runner.ingest_request("r1", {"kv": "kv-cfg"})

    assert kv.calls == ["ingest:r1:kv-cfg"]
    assert sampler.calls == ["ingest:r1:None"], "no override for this key"


def test_admit_retrieve_short_circuits_and_aggregates_ready():
    kv = _Stub("kv")
    kv.admit_retrieve = lambda rid, node, walk, pub: AdmitOutcome(ok=True, ready=False)
    runner = StepRunner({"kv": kv, "sampler": _Stub("sampler")})

    outcome = runner.admit_retrieve("r1", "llm", "decode", {"kv": "published"})

    assert (outcome.ok, outcome.ready) == (True, False)


def test_build_cuda_graph_buffers_reaches_every_resource():
    kv, sampler = _Stub("kv"), _Stub("sampler")
    runner = StepRunner({"kv": kv, "sampler": sampler})

    runner.build_cuda_graph_buffers(slots=[object(), object()], max_bs=8, max_seq_len=64)

    assert kv.calls == ["cg_buffers:2:8:64"]
    assert sampler.calls == ["cg_buffers:2:8:64"]


# --- the step envelope -----------------------------------------------------


def test_unknown_resource_key_in_a_step_is_an_error():
    """A model declaring a key the node has no resource for is a config
    mismatch; skipping it silently would surface later as a KeyError on an
    artifact lookup deep inside a forward."""
    runner = StepRunner({"kv": _Stub("kv")})
    with pytest.raises(KeyError, match="does not have"):
        runner.plan(_step(["kv", "ghost"]))


def test_resource_step_without_segments_inherits_the_batch_layout():
    segs = (Segment("r1", "main", 1), Segment("r2", "main", 1))
    step = SubmoduleStep(ctx=_ctx(), segments=segs, steps={"sampler": KVStep()})
    assert step.segments_for("sampler") == segs


def test_validate_accepts_an_order_preserving_subsequence():
    a, b, c = Segment("r1", "main", 1), Segment("r2", "main", 1), Segment("r3", "main", 1)
    SubmoduleStep(
        ctx=_ctx(), segments=(a, b, c), steps={"cross": KVStep(segments=(a, c))},
    ).validate()


def test_validate_rejects_a_reordered_subsequence():
    a, b, c = Segment("r1", "main", 1), Segment("r2", "main", 1), Segment("r3", "main", 1)
    step = SubmoduleStep(
        ctx=_ctx(), segments=(a, b, c), steps={"cross": KVStep(segments=(c, a))},
    )
    with pytest.raises(ValueError, match="order-preserving"):
        step.validate()


def test_validate_rejects_a_segment_outside_the_batch_layout():
    a = Segment("r1", "main", 1)
    step = SubmoduleStep(
        ctx=_ctx(), segments=(a,), steps={"cross": KVStep(segments=(Segment("r9", "x", 1),))},
    )
    with pytest.raises(ValueError, match="not an order-preserving"):
        step.validate()


def test_admit_validates_the_step():
    runner = StepRunner({"kv": _Stub("kv")})
    step = SubmoduleStep(
        ctx=_ctx(),
        segments=(Segment("r1", "main", 1),),
        steps={"kv": KVStep(segments=(Segment("r9", "ghost", 1),))},
    )
    with pytest.raises(ValueError, match="order-preserving"):
        runner.admit(step)


# --- the kv_store severance ------------------------------------------------


def test_v1_kv_layer_imports_without_kv_store_or_the_conductor():
    """``v1`` replaced ``kv_store.py``, which is now deleted; importing v1 must
    still not drag in the conductor and sampling kernels that sat behind it.
    Run in a subprocess because sys.modules is shared across the pytest
    session (and conftest stubs triton in-process)."""
    probe = (
        "import sys, mstar.engine.v1.kv_manager, mstar.engine.resources;"
        "leaked = [m for m in ("
        "  'mstar.conductor.request_info',"
        "  'mstar.communication.tensors',"
        "  'triton',"
        ") if m in sys.modules];"
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"v1 pulled in {result.stdout.strip()}"
