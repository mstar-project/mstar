"""``AttentionStep.context_only``: a step's queries attend to the stream's stored context only (the
entries the same step appends are excluded), with their own query counts; and the MLA attention
returns the natural log-sum-exp for merging with another key segment. CPU, on the torch fallback
of the FlashInfer MLA wrapper, against a dense reference."""
import torch

from mstar.engine.resources import AttentionStep, KVStep, Segment, StepContext, SubmoduleStep
from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVLayout
from mstar.engine.resources.kv.plan import SequenceView, context_only_views

CPU = torch.device("cpu")
LAT, ROPE, H, PAGE = 8, 4, 2, 4


class _StubTransfer:
    def __init__(self, *a, **k):
        pass

    def cleanup(self):
        pass


def _cfg():
    return KVConfig(num_layers=1, num_kv_heads=1, head_dim=LAT + ROPE, max_seq_len=64, max_num_pages=8, page_size=PAGE,
                    layout=KVLayout.MLA, kv_lora_rank=LAT, qk_rope_head_dim=ROPE, num_qo_heads=H)


def test_context_only_views_drop_the_appended_tail():
    views = [SequenceView("a", "main", [3, 7, 1], length=8, to_compute=3),
             SequenceView("b", "main", [4, 2], length=6, to_compute=3)]
    out = context_only_views(views, [2, 2], PAGE)
    assert out[0].length == 5 and out[0].to_compute == 2 and out[0].page_idxs == [3, 7]
    assert out[1].length == 3 and out[1].to_compute == 2 and out[1].page_idxs == [4]
    empty = context_only_views([SequenceView("c", "main", [9], length=3, to_compute=3)], [2], PAGE)[0]
    assert empty.length == 0 and empty.page_idxs == []


def test_queries_attend_to_the_stored_context_only(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    cfg = _cfg()
    kv = manager_mod.KVManager(cfg=cfg, name="mla_kv", joint_comm_group=None, transfer_engine_info=None, device=CPU,
                               dtype=torch.float32)
    attn = FlashInferMLAManager(kv_cache="mla_kv", device=CPU, dtype=torch.float32, kv_config=cfg, sm_scale=0.3)
    for rid in ("a", "b"):
        kv.ingest_request(rid)
    # prefill: 5 and 3 context entries
    s = SubmoduleStep(segments=[Segment("a", "main", 5), Segment("b", "main", 3)], steps={"mla_kv": KVStep()})
    ctx = StepContext(request_ids=["a", "b"], graph_walk="prefill", slot=0, capture=False)
    s.set_ctx(ctx)
    assert kv.admit(s.get("mla_kv"), ctx).ok
    kv.plan(s.get("mla_kv"), ctx)
    torch.manual_seed(0)
    context = torch.randn(8, LAT + ROPE)
    kv.set_default_layer_idx(0)
    kv.write_kv(context)
    kv.commit(s.get("mla_kv"), ctx)
    # a draft step: the cache appends 3 entries per request, 2 queries per request look at the context alone
    s2 = SubmoduleStep(
        segments=[Segment("a", "main", 3), Segment("b", "main", 3)],
        steps={"mla_kv": KVStep(),
               "attn": AttentionStep(causal=False, context_only=True,
                                     segments=(Segment("a", "main", 2), Segment("b", "main", 2)))},
    )
    ctx2 = StepContext(request_ids=["a", "b"], graph_walk="decode", slot=0, capture=False)
    s2.set_ctx(ctx2)
    assert kv.admit(s2.get("mla_kv"), ctx2).ok
    ctx2.plan_results["mla_kv"] = kv.plan(s2.get("mla_kv"), ctx2)
    attn.plan(s2.get("attn"), ctx2)
    qo, kv_indptr, kv_indices, kv_len, causal = attn._current_plan_states["main"]._fb
    assert qo == [0, 2, 4] and kv_len == [5, 3] and causal is False
    q_nope, q_pe = torch.randn(4, H, LAT), torch.randn(4, H, ROPE)
    out, lse = attn.run(q_nope, kv_cache_layer=kv.layer_view(0), q_pe=q_pe, return_lse=True)
    assert out.shape == (4, H, LAT) and lse.shape == (4, H)
    for i, (start, n) in enumerate(((0, 5), (5, 3))):  # a's 5 entries, then b's 3, as written
        keys = context[start:start + n]
        c, kpe = keys.split([LAT, ROPE], dim=-1)
        qn, qp = q_nope[2 * i:2 * i + 2], q_pe[2 * i:2 * i + 2]
        scores = (torch.einsum("qhl,kl->hqk", qn, c) + torch.einsum("qhr,kr->hqk", qp, kpe)) * 0.3
        want = torch.einsum("hqk,kl->qhl", torch.softmax(scores, -1), c)
        assert torch.allclose(out[2 * i:2 * i + 2], want, atol=1e-5)
        assert torch.allclose(lse[2 * i:2 * i + 2], torch.logsumexp(scores, -1).transpose(0, 1), atol=1e-5)
    # the plain (non context-only) plan of the same step sees the appended tail: 8 and 6
    s3 = SubmoduleStep(segments=[Segment("a", "main", 3), Segment("b", "main", 3)],
                       steps={"mla_kv": KVStep(), "attn": AttentionStep(causal=True)})
    ctx3 = StepContext(request_ids=["a", "b"], graph_walk="decode", slot=0, capture=False)
    s3.set_ctx(ctx3)
    ctx3.plan_results["mla_kv"] = ctx2.plan_results["mla_kv"]
    attn.plan(s3.get("attn"), ctx3)
    assert attn._current_plan_states["main"]._fb[3] == [8, 6] and attn._current_plan_states["main"]._fb[0] == [0, 3, 6]
