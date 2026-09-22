"""GPU: the draft's served path at real latent dims (context KV in a paged MLA cache read by FlashInfer with
``context_only`` plans, the fused block attention and merge, the fused rope, Markov sampling) against the dense
reference ``draft_dense`` over the same context states, in bf16. The CPU test covers the torch fallback only."""
import pytest
import torch
from torch import nn

from mstar.engine.resources import AttentionStep, KVStep, Segment, StepContext, SubmoduleStep
from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVLayout
from mstar.model.kimi_k3.dspark.config import DSparkConfig, YarnParams
from mstar.model.kimi_k3.dspark.model import DSparkDraft

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")
K = 7
CFG = DSparkConfig(hidden_size=1024, intermediate_size=2048, num_hidden_layers=2, num_attention_heads=8,
                   q_lora_rank=256,
                   kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, vocab_size=4096,
                   target_hidden_size=1024, target_layer_ids=(1, 3), mask_token_id=4095, markov_rank=16,
                   rope=YarnParams(original_max_position_embeddings=2048, factor=2.0))


class _StubTransfer:
    def __init__(self, *a, **k):
        pass

    def cleanup(self):
        pass


def build(monkeypatch):
    torch.manual_seed(0)
    # the target's embedding and head are held by reference (not submodules), so make them on the device
    embed = nn.Embedding(CFG.vocab_size, CFG.hidden_size).to(DEV, torch.bfloat16)
    head = nn.Linear(CFG.hidden_size, CFG.vocab_size, bias=False).to(DEV, torch.bfloat16)
    draft = DSparkDraft(CFG, embed, head, max_positions=4096)
    for name, p in draft.named_parameters():
        p.data = torch.ones_like(p) if name.endswith("norm.weight") else torch.randn_like(p) * 0.05
    draft = draft.to(DEV, torch.bfloat16)
    draft.rope.__init__(CFG.qk_rope_head_dim, CFG.rope, 4096)
    draft.rope.to(DEV)
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    kv_cfg = KVConfig(num_layers=CFG.num_hidden_layers, num_kv_heads=1,
                      head_dim=CFG.kv_lora_rank + CFG.qk_rope_head_dim,
                      max_seq_len=1024, max_num_pages=32, page_size=64, layout=KVLayout.MLA,
                      kv_lora_rank=CFG.kv_lora_rank,
                      qk_rope_head_dim=CFG.qk_rope_head_dim, num_qo_heads=CFG.num_attention_heads)
    kv = manager_mod.KVManager(cfg=kv_cfg, name="dspark_kv", joint_comm_group=None, transfer_engine_info=None,
                               device=DEV,
                               dtype=torch.bfloat16)
    attn = FlashInferMLAManager(kv_cache="dspark_kv", device=DEV, dtype=torch.bfloat16, kv_config=kv_cfg,
                                sm_scale=draft.layers[0].self_attn.scale)
    draft.bind_resources({"dspark_kv": kv, "dspark_attn": attn})
    return draft, kv, attn


def kv_step(kv, rid, span, walk):
    s = SubmoduleStep(segments=[Segment(rid, "main", span)], steps={"dspark_kv": KVStep()})
    ctx = StepContext(request_ids=[rid], graph_walk=walk, slot=0, capture=False)
    s.set_ctx(ctx)
    assert kv.admit(s.get("dspark_kv"), ctx).ok
    ctx.plan_results["dspark_kv"] = kv.plan(s.get("dspark_kv"), ctx)
    return s, ctx


@pytest.mark.parametrize("tc", [100, 300])
def test_paged_draft_matches_the_dense_reference(monkeypatch, tc):
    draft, kv, attn = build(monkeypatch)
    kv.ingest_request("a")
    torch.manual_seed(1)
    # the prompt's context: tc combined states, written by the prefill
    states = torch.randn(tc, CFG.hidden_size, device=DEV).to(torch.bfloat16)
    s, ctx = kv_step(kv, "a", tc, "prefill")
    draft.write_context(states, torch.arange(tc, device=DEV))
    kv.commit(s.get("dspark_kv"), ctx)
    # a decode step: K + 1 entries appended, the K queries attend to the stored context alone
    s, ctx = kv_step(kv, "a", K + 1, "decode")
    astep = AttentionStep(causal=False, context_only=True, segments=(Segment("a", "main", K),))
    attn.plan(astep, ctx)
    ctx_len = attn.kv_len_buf()
    assert ctx_len.tolist() == [tc]
    positions = (ctx_len[:, None] + torch.arange(K, device=DEV)[None, :]).reshape(-1)
    bonus = 7
    with torch.no_grad():
        hidden = draft.block_hidden(draft.block_ids(torch.tensor([bonus], device=DEV), K).reshape(-1), positions, 1)
        drafts = draft.draft(torch.tensor([bonus], device=DEV), positions, K)
        want_drafts, want_hidden = draft.draft_dense(states, bonus, K)
        logits = draft.lm_head(hidden).float()
        want_logits = draft.lm_head(want_hidden).float()
    torch.cuda.synchronize()
    diff = (hidden.float() - want_hidden.float()).abs().max().item()
    scale = want_hidden.float().abs().mean().item()
    print(f"tc={tc}: max |hidden diff| {diff:.4f} (mean |hidden| {scale:.4f}); "
          f"drafts {drafts[0].tolist()} vs {want_drafts.tolist()}")
    assert torch.allclose(hidden.float(), want_hidden.float(), atol=0.15 * scale + 0.05, rtol=0.05), diff
    assert torch.allclose(logits, want_logits, atol=0.5, rtol=0.05)
    assert (drafts[0] == want_drafts).sum().item() >= K - 1, (drafts[0].tolist(), want_drafts.tolist())
