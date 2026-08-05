"""Phase C DSA indexer: skip formula, selection semantics, sparse==dense, load.

The load tests reuse ``test_glm52_moe._fabricate_checkpoint`` (same reduced
fp8 stream, now carrying indexer keys on FULL layers). The bitwise test
stubs ``flashinfer.norm.rmsnorm`` via monkeypatch — ``run_rms_norm`` does
``import flashinfer`` per call, so the stub is scoped to that test and
never leaks into sessions that have the real library.
"""
import sys
import types

import torch
from test_glm52_moe import BLOCK, _fabricate_checkpoint

from mstar.model.glm52.components.attention import (
    Glm52MLAAttention,
    masked_reference_attention,
)
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
from mstar.model.glm52.components.indexer import (
    Glm52Indexer,
    is_full_indexer_layer,
)
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.weight_loader import load_glm52_hf_weights

# Observed in the GLM-5.2 checkpoint: 21 FULL main layers (indexer_types),
# hardcoded so a broken formula cannot regenerate its own expectation.
CHECKPOINT_FULL_LAYERS = [
    0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66,
    70, 74,
]


def test_skip_formula_full_model_golden():
    cfg = Glm52ModelConfig()
    got = [
        i for i in range(cfg.num_hidden_layers) if is_full_indexer_layer(cfg, i)
    ]
    # Spec formula with GLM literals: skip = max(l - 3 + 1, 0) % 4 != 0 —
    # offset=3 gives leading FULLs 0..2 and anchors the series at layer 2.
    expected = [l for l in range(78) if max(l - 3 + 1, 0) % 4 == 0]
    assert got == expected
    assert got == CHECKPOINT_FULL_LAYERS
    assert len(got) == 21


def test_skip_formula_reduced_is_full_then_shared():
    cfg = Glm52ModelConfig.reduced()
    flags = [
        is_full_indexer_layer(cfg, i) for i in range(cfg.num_hidden_layers)
    ]
    assert flags == [True, False]


def test_attention_builds_indexer_only_on_full_layers():
    cfg = Glm52ModelConfig.reduced()
    assert Glm52MLAAttention(cfg, layer_idx=0).indexer is not None
    assert Glm52MLAAttention(cfg, layer_idx=1).indexer is None
    # Layer-agnostic construction (component tests) stays indexer-free.
    assert Glm52MLAAttention(cfg).indexer is None


def _build_indexer(seed):
    torch.manual_seed(seed)
    cfg = Glm52ModelConfig.reduced()
    # Tiny topk so short test sequences exercise the truncation regime;
    # reduced() itself keeps a serve-safe 64 (the preprocess guard refuses
    # ctx > index_topk, and the GPU e2e serve tests run real contexts).
    cfg.index_topk = 4
    idx = Glm52Indexer(cfg)
    _randomize_indexer(idx)
    return idx, cfg


def _randomize_indexer(idx):
    idx.wq_b.weight.data.normal_(0, 0.05)
    idx.wk.weight.data.normal_(0, 0.05)
    idx.weights_proj.weight.data.normal_(0, 0.05)
    idx.k_norm.weight.data.normal_(1.0, 0.02)
    idx.k_norm.bias.data.normal_(0, 0.02)


def test_selection_window_includes_self_and_pads_with_minus_one():
    idx, cfg = _build_indexer(seed=0)
    t = 7  # rows 0..3 have prefix <= topk=4; rows 4..6 exceed it
    h = torch.randn(t, cfg.hidden_size) * 0.1
    q_c = torch.randn(t, cfg.q_lora_rank) * 0.1
    pos = torch.arange(t)

    sel = idx.compute_selection(q_c, h, pos, idx.compute_k(h, pos))

    assert sel.shape == (t, cfg.index_topk) and sel.dtype == torch.int32
    for i in range(t):
        window = i + 1  # positions 0..i, INCLUDING self
        n = min(cfg.index_topk, window)
        row = sel[i].tolist()
        picked = [x for x in row if x >= 0]
        assert len(picked) == n
        assert row[n:] == [-1] * (cfg.index_topk - n)  # early rows padded
        if window <= cfg.index_topk:
            # Prefix < topk selects ALL of it — set equality, order-free.
            assert set(picked) == set(range(window))
        else:
            # Prefix > topk: exactly topk distinct causal picks, no -1.
            assert len(set(picked)) == cfg.index_topk
            assert all(0 <= x <= i for x in picked)


def test_indexer_rope_hits_first_dims_only():
    idx, cfg = _build_indexer(seed=3)
    r = cfg.qk_rope_head_dim
    h = torch.randn(2, cfg.hidden_size)
    pos = torch.tensor([0, 5])

    k = idx.compute_k(h, pos)
    raw = idx.k_norm(idx.wk(h))  # un-roped reference

    # Position 0: the rotation is the identity.
    torch.testing.assert_close(k[0], raw[0], rtol=1e-6, atol=1e-6)
    # Position > 0: tail dims pass through untouched, FIRST r dims rotate —
    # the reversed-slice trap (main MLA ropes the LAST dims instead).
    assert torch.equal(k[1, r:], raw[1, r:])
    assert not torch.allclose(k[1, :r], raw[1, :r])

    # Same guard through the per-head q path.
    q = torch.randn(2, cfg.index_n_heads, cfg.index_head_dim)
    q_rot = idx._rope_first_dims(q, pos)
    assert torch.equal(q_rot[1, :, r:], q[1, :, r:])
    assert not torch.allclose(q_rot[1, :, :r], q[1, :, :r])


def _stub_flashinfer(monkeypatch):
    """CPU ``flashinfer.norm.rmsnorm`` so RMSNorm-backed forwards run here."""

    def rmsnorm(x, weight, eps=1e-6):
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        return (normed * weight.float()).to(x.dtype)

    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=rmsnorm)
    monkeypatch.setitem(sys.modules, "flashinfer", fi)


class _CausalDenseHandle:
    """Stands in for ``BatchedCacheManager.run_attention`` on CPU.

    Uses the SAME masked-attention helper as the sparse path, with a plain
    causal mask — so bitwise dense-vs-sparse equality reduces exactly to
    mask equality, i.e. to the selection covering the full causal prefix.
    """

    def run_attention(self, q, k, v):
        t = q.shape[0]
        mask = torch.triu(q.new_full((t, t), float("-inf")), diagonal=1)
        return masked_reference_attention(q, k, v, mask)


def _build_attention(seed):
    torch.manual_seed(seed)
    cfg = Glm52ModelConfig.reduced()  # mla_absorb=False
    cfg.index_topk = 4  # ctx <= topk regime at test-sized sequences
    attn = Glm52MLAAttention(cfg, layer_idx=0)  # FULL layer -> has indexer
    for lin in (attn.q_a_proj, attn.q_b_proj, attn.kv_a_proj_with_mqa,
                attn.kv_b_proj, attn.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (attn.q_a_layernorm, attn.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    _randomize_indexer(attn.indexer)
    return attn, cfg


def test_sparse_forward_matches_dense_bitwise_within_topk(monkeypatch):
    """THE identity Phase C rests on: at ctx <= topk the selection is the
    full prefix, so DSA attention IS dense causal attention — bitwise."""
    _stub_flashinfer(monkeypatch)
    attn, cfg = _build_attention(seed=2)
    t = cfg.index_topk  # ctx <= topk regime
    h = torch.randn(t, cfg.hidden_size) * 0.1
    pos = torch.arange(t)

    q_c = attn.q_a_layernorm(attn.q_a_proj(h))  # the shared q latent
    k_hist = attn.indexer.compute_k(h, pos)
    sel = attn.indexer.compute_selection(q_c, h, pos, k_hist)
    assert (sel == -1).any()  # early rows really are padded

    dense = attn(h, _CausalDenseHandle(), pos)
    sparse = attn(h, None, pos, dsa_selection=sel)
    assert torch.equal(dense, sparse)


def test_load_indexer_keys_land_dequantized_on_full_layer():
    torch.manual_seed(4)
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    model = Glm52ForCausalLM(cfg)
    state, refs = _fabricate_checkpoint(cfg)

    # The stream also carries a poison MTP indexer fp8 key with NO scale
    # sibling — if the layer-index skip stopped running before the dequant
    # stream, this load would raise "unpaired".
    loaded = load_glm52_hf_weights(
        model, iter(state), cfg.n_routed_experts,
        quant_config=cfg.quantization_config, fp8_experts=True,
        num_hidden_layers=cfg.num_hidden_layers,
    )

    # Completeness both ways still holds with the indexer params included.
    assert loaded == set(dict(model.named_parameters()))

    idxr = model.model.layers[0].self_attn.indexer
    assert idxr is not None
    assert model.model.layers[1].self_attn.indexer is None  # SHARED layer

    # fp8 pairs dequantized bit-exactly despite plain ``.weight`` names.
    _, _, wq_deq = refs["model.layers.0.self_attn.indexer.wq_b"]
    assert torch.equal(idxr.wq_b.weight.data, wq_deq.to(idxr.wq_b.weight.dtype))
    _, _, wk_deq = refs["model.layers.0.self_attn.indexer.wk"]
    assert torch.equal(idxr.wk.weight.data, wk_deq.to(idxr.wk.weight.dtype))

    # bf16 passthrough, including the k_norm BIAS (full LayerNorm).
    ckpt = dict(state)
    for param, key in (
        (idxr.weights_proj.weight, "weights_proj.weight"),
        (idxr.k_norm.weight, "k_norm.weight"),
        (idxr.k_norm.bias, "k_norm.bias"),
    ):
        ref = ckpt[f"model.layers.0.self_attn.indexer.{key}"]
        assert torch.equal(param.data, ref.to(param.dtype))


def test_load_indexer_flag_off_skips_indexer_keys():
    torch.manual_seed(5)
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    model = Glm52ForCausalLM(cfg)
    state, _ = _fabricate_checkpoint(cfg)

    loaded = load_glm52_hf_weights(
        model, iter(state), cfg.n_routed_experts,
        quant_config=cfg.quantization_config, fp8_experts=True,
        num_hidden_layers=cfg.num_hidden_layers, load_indexer=False,
    )

    params = set(dict(model.named_parameters()))
    indexer_params = {n for n in params if ".indexer." in n}
    assert indexer_params  # layer 0 does carry an indexer
    assert loaded == params - indexer_params
