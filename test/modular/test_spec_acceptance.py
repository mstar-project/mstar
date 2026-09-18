"""The acceptance resource of a speculating node (``SpecAcceptance``) and the KV manager's use of
its verdicts: a verify step commits ``k + 1`` tokens per request and the next plan takes the
rejected tail back."""
import torch

from mstar.engine.resources import (
    SPEC_ACCEPTANCE,
    KVConfig,
    KVSpec,
    KVStep,
    Segment,
    SpecAccepted,
    SpecAcceptanceSpec,
    SpecStep,
    StepContext,
    SubmoduleStep,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv import manager as kv_manager_module

K = 4
CPU = torch.device("cpu")


def spec_resource():
    return build_resource(SpecAcceptanceSpec("spec_acceptance", {"LLM"}, num_speculative=K),
                          EngineResourceInfo(device=CPU))


def step(rids, padded=None, span=K + 1):
    s = SubmoduleStep(segments=[Segment(r, "main", span) for r in (padded or rids)], steps={"spec_acceptance": SpecStep()})
    ctx = StepContext(request_ids=list(rids), graph_walk="decode", slot=0, capture=False)
    if padded is not None:
        ctx.set_padded_rids(list(padded))
    s.set_ctx(ctx)
    return s, ctx


def test_verdicts_flow_from_stage_to_the_next_plan():
    res = spec_resource()
    for rid in ("a", "b", "c"):
        res.ingest_request(rid)
    s, ctx = step(["a", "b"])
    assert res.plan(s.get("spec_acceptance"), ctx) == {}  # nothing verified yet
    # the forward verified a and b: a kept 3 drafts, b kept 1; then the host noted the step
    toks = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=torch.int32)
    res.stage(torch.tensor([3, 1], dtype=torch.int32), toks)
    res.note_step(["a", "b"])
    assert res.accepted_for(["a", "b"]) == [3, 1]
    va, vb = res.verdicts_for(["a", "b"])
    assert va.tokens == [10, 11, 12, 13, 14] and vb.accepted == 1 and vb.tokens[: vb.accepted + 1] == [20, 21]
    # next step: a and c (b is held), plus a padding row that must publish nothing
    s2, ctx2 = step(["a", "c"], padded=["a", "c", "__cg_LLM_0_2__"])
    out = res.plan(s2.get("spec_acceptance"), ctx2)
    assert out == {"a": SpecAccepted(accepted=3, rejected=1, label="main")}
    # a's verdict is consumed; b's waits for the step it next appears in
    s3, ctx3 = step(["b"])
    assert res.plan(s3.get("spec_acceptance"), ctx3) == {"b": SpecAccepted(accepted=1, rejected=3, label="main")}
    assert res.plan(s3.get("spec_acceptance"), ctx3) == {}
    # removal drops a pending verdict
    res.stage(torch.tensor([2], dtype=torch.int32)); res.note_step(["c"]); res.remove_request("c")
    s4, ctx4 = step(["c"])
    assert res.plan(s4.get("spec_acceptance"), ctx4) == {}


def test_slot_buffers_are_copied_out_before_reuse():
    """Two capture slots alternate; a verdict read late must still be the one of its own step."""
    res = spec_resource()
    res.ingest_request("a"); res.ingest_request("b")
    res.build_cuda_graph_buffers([], max_bs=4, max_seq_len=16)
    res._current_slot = 0
    res.stage(torch.tensor([4, 0], dtype=torch.int32)); res.note_step(["a", "b"])
    # b is held for two steps; slot 0 is written again by a later step with other rows
    res._current_slot = 1
    res.stage(torch.tensor([1], dtype=torch.int32)); res.note_step(["a"])
    s, ctx = step(["a"])
    assert res.plan(s.get("spec_acceptance"), ctx)["a"].accepted == 1  # settles everything pending, b included
    res._current_slot = 0
    res.stage(torch.tensor([2, 2], dtype=torch.int32)); res.note_step(["a", "x"])
    s2, ctx2 = step(["b"])
    assert res.plan(s2.get("spec_acceptance"), ctx2)["b"].accepted == 0  # not the 2 now sitting in slot 0 row 1


def kv_manager(monkeypatch):
    monkeypatch.setattr(kv_manager_module, "KVTransferManager", lambda info, kv_cache: None)
    return kv_manager_module.KVManager(
        cfg=KVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, max_num_pages=8, page_size=4),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )


def test_kv_plan_takes_the_rejected_tail_back(monkeypatch):
    kv = kv_manager(monkeypatch)
    kv.ingest_request("a")
    # a verify step of k + 1 = 5 tokens: admit, plan, commit grow the stream by 5
    s, ctx = step(["a"])
    s = SubmoduleStep(segments=s.segments, steps={"kv": KVStep()}); s.set_ctx(ctx)
    assert kv.admit(s.get("kv"), ctx).ok
    kv.plan(s.get("kv"), ctx)
    kv.commit(s.get("kv"), ctx)
    assert kv._streams["a"]["main"].stored_len == K + 1
    # the next step's plan runs after the spec resource published a's verdict: 1 of 4 drafts kept
    s2, ctx2 = step(["a"])
    s2 = SubmoduleStep(segments=s2.segments, steps={"kv": KVStep()}); s2.set_ctx(ctx2)
    assert kv.admit(s2.get("kv"), ctx2).ok
    ctx2.plan_results[SPEC_ACCEPTANCE] = {"a": SpecAccepted(accepted=1, rejected=3)}
    out = kv.plan(s2.get("kv"), ctx2)
    view = next(iter(out.values())).views[0]
    assert kv._streams["a"]["main"].stored_len == 2  # 5 committed, 3 taken back: bonus + 1 accepted draft
    assert view.length == 2 + (K + 1) and view.to_compute == K + 1
    kv.commit(s2.get("kv"), ctx2)
    assert kv._streams["a"]["main"].stored_len == 2 + K + 1
    # no verdict, no change
    s3, ctx3 = step(["a"])
    s3 = SubmoduleStep(segments=s3.segments, steps={"kv": KVStep()}); s3.set_ctx(ctx3)
    assert kv.admit(s3.get("kv"), ctx3).ok
    before = kv._streams["a"]["main"].stored_len
    kv.plan(s3.get("kv"), ctx3)
    assert kv._streams["a"]["main"].stored_len == before
