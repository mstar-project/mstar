"""The DSpark draft module at tiny dimensions on CPU: the paged path (context KV in a latent cache,
``context_only`` attention on the torch fallback, dense block attention merged by log-sum-exp,
Markov sampling) against the dense reference ``draft_dense``, across a step whose rejected
context tail is trimmed."""
import torch
from torch import nn

from mstar.engine.resources import AttentionStep, KVStep, Segment, StepContext, SubmoduleStep
from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVLayout
from mstar.model.kimi_k3.dspark.config import DSparkConfig, YarnParams
from mstar.model.kimi_k3.dspark.model import DSparkDraft

CPU = torch.device("cpu")
K = 3
CFG = DSparkConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2, q_lora_rank=16,
                   kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8, vocab_size=50,
                   target_hidden_size=12, target_layer_ids=(0, 1, 2), mask_token_id=49, markov_rank=4,
                   rope=YarnParams(original_max_position_embeddings=64, factor=2.0))


class _StubTransfer:
    def __init__(self, *a, **k):
        pass

    def cleanup(self):
        pass


def build(monkeypatch):
    torch.manual_seed(0)
    embed, head = nn.Embedding(CFG.vocab_size, CFG.hidden_size), nn.Linear(CFG.hidden_size, CFG.vocab_size, bias=False)
    draft = DSparkDraft(CFG, embed, head, max_positions=128)
    for name, p in draft.named_parameters():
        p.data = torch.ones_like(p) if name.endswith("norm.weight") else torch.randn_like(p) * 0.2
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    kv_cfg = KVConfig(num_layers=CFG.num_hidden_layers, num_kv_heads=1, head_dim=CFG.kv_lora_rank + CFG.qk_rope_head_dim,
                      max_seq_len=64, max_num_pages=16, page_size=4, layout=KVLayout.MLA, kv_lora_rank=CFG.kv_lora_rank,
                      qk_rope_head_dim=CFG.qk_rope_head_dim, num_qo_heads=CFG.num_attention_heads)
    kv = manager_mod.KVManager(cfg=kv_cfg, name="dspark_kv", joint_comm_group=None, transfer_engine_info=None, device=CPU,
                               dtype=torch.float32)
    attn = FlashInferMLAManager(kv_cache="dspark_kv", device=CPU, dtype=torch.float32, kv_config=kv_cfg,
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


def test_paged_draft_matches_the_dense_reference_across_a_trim(monkeypatch):
    draft, kv, attn = build(monkeypatch)
    kv.ingest_request("a")
    # prefill: 6 context tokens from (fake) target aux states
    aux = torch.randn(6, CFG.context_width)
    states = draft.combine(aux)
    s, ctx = kv_step(kv, "a", 6, "prefill")
    draft.write_context(states, torch.arange(6))
    kv.commit(s.get("dspark_kv"), ctx)

    def draft_step(context_states, bonus):
        # the step appends K + 1 context entries; the K queries attend to the stored context alone
        s, ctx = kv_step(kv, "a", K + 1, "decode")
        astep = AttentionStep(causal=False, context_only=True, segments=(Segment("a", "main", K),))
        attn.plan(astep, ctx)
        ctx_len = attn.kv_len_buf()
        assert ctx_len.tolist() == [context_states.shape[0]]
        positions = (ctx_len[:, None] + torch.arange(K)[None, :]).reshape(-1)
        drafts = draft.draft(torch.tensor([bonus]), positions, K)
        hidden = draft.block_hidden(draft.block_ids(torch.tensor([bonus]), K).reshape(-1), positions, 1)
        want_drafts, want_hidden = draft.draft_dense(context_states, bonus, K)
        assert torch.allclose(hidden, want_hidden, atol=1e-4, rtol=1e-4), (hidden - want_hidden).abs().max()
        assert drafts[0].tolist() == want_drafts.tolist()
        return s, ctx, positions

    s, ctx, positions = draft_step(states, bonus=7)
    # end of the step: the context KV of this block's K + 1 positions (from the step's aux states)
    aux2 = torch.randn(K + 1, CFG.context_width)
    states2 = draft.combine(aux2)
    draft.write_context(states2, torch.arange(6, 6 + K + 1))
    kv.commit(s.get("dspark_kv"), ctx)
    # the target accepted 1 draft: bonus + 1 kept, 2 rejected; the next plan trims the cache
    kv.correct_len("a", "main", -2)
    assert kv._streams["a"]["main"].stored_len == 8
    all_states = torch.cat([states, states2[:2]])
    draft_step(all_states, bonus=11)


def test_block_ids_and_markov_sampling():
    torch.manual_seed(1)
    draft = DSparkDraft(CFG, nn.Embedding(50, 32), nn.Linear(32, 50, bias=False), max_positions=16)
    ids = draft.block_ids(torch.tensor([3, 5]), K)
    assert ids.tolist() == [[3, 49, 49], [5, 49, 49]]
    logits = torch.zeros(2, K, 50)
    logits[0, 0, 10] = logits[0, 1, 20] = logits[0, 2, 30] = 5.0
    logits[1, :, 7] = 5.0
    for p in draft.markov_head.parameters():
        p.data.zero_()
    assert draft.markov_sample(logits, torch.tensor([3, 5])).tolist() == [[10, 20, 30], [7, 7, 7]]
    # a Markov bias that makes "previous token + 1" win chains left to right through the block
    w1 = torch.zeros(50, 4)
    w1[:, 0] = 1.0
    w1[:, 1] = torch.arange(50).float()
    draft.markov_head.markov_w1.weight.data = w1
    w2 = torch.zeros(50, 4)
    w2[:, 0] = -1e3 * torch.arange(50).float()  # bias(prev)[t] = -1e3 * t + 1e3 * t * prev: peaks at t = prev + 1 ...
    w2[:, 1] = 1e3 * torch.ones(50)
    # ... only for the row t = prev + 1 when the logits add a nudge; simpler: check the chain explicitly
    draft.markov_head.markov_w2.weight.data.zero_()
    draft.markov_head.markov_w2.weight.data[:, 1] = 0.0
    bias_of = lambda prev: torch.zeros(50).index_fill_(0, torch.tensor([(prev + 1) % 50]), 10.0)  # noqa: E731
    draft.markov_head.bias = lambda prev: torch.stack([bias_of(int(p)) for p in prev])  # type: ignore[method-assign]
    assert draft.markov_sample(torch.zeros(1, K, 50), torch.tensor([3])).tolist() == [[4, 5, 6]]
