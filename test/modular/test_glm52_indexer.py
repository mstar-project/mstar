"""DSA indexer: skip formula, selection semantics, sparse == dense, load."""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


def _cpu_flashinfer() -> types.ModuleType:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    return fi


if "flashinfer" not in sys.modules:
    sys.modules["flashinfer"] = _cpu_flashinfer()


@pytest.fixture(autouse=True)
def _force_cpu_flashinfer(monkeypatch):
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


from test_glm52_moe import BLOCK, _fabricate_checkpoint  # noqa: E402

from mstar.model.glm52.components.attention import (  # noqa: E402
    Glm52MLAAttention,
    masked_reference_attention,
)
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.components.indexer import (  # noqa: E402
    Glm52Indexer,
    is_full_indexer_layer,
    select_topk_causal,
)
from mstar.model.glm52.config import (  # noqa: E402
    ATTN_RESOURCE,
    KV_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.glm52.weight_loader import load_glm52_hf_weights  # noqa: E402

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
    cfg.dsa_long_context = True  # the indexer is built only on the DSA path
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


class _CausalDenseResources:
    """The naive path's two resources, on CPU, for ONE dense pass: the layer
    writes K/V through ``kv.write_kv`` and attends through ``attn.run``.
    """

    def __init__(self):
        self.k = self.v = None

    # kv side
    def write_kv(self, k, v, layer_idx=None):
        self.k, self.v = k, v

    def layer_view(self, layer_idx=None):
        return None

    # attention side
    def run(self, q, kv_cache_layer=None, **kwargs):
        t = q.shape[0]
        mask = torch.triu(q.new_full((t, t), float("-inf")), diagonal=1)
        return masked_reference_attention(q, self.k, self.v, mask)


def _build_attention(seed):
    torch.manual_seed(seed)
    cfg = Glm52ModelConfig.reduced()  # mla_absorb=False
    cfg.dsa_long_context = True  # the indexer is built only on the DSA path
    cfg.index_topk = 4  # ctx <= topk regime at test-sized sequences
    attn = Glm52MLAAttention(cfg, layer_idx=0)  # FULL layer -> has indexer
    for lin in (attn.q_a_proj, attn.q_b_proj, attn.kv_a_proj_with_mqa,
                attn.kv_b_proj, attn.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (attn.q_a_layernorm, attn.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    _randomize_indexer(attn.indexer)
    return attn, cfg


def test_sparse_forward_matches_dense_bitwise_within_topk():
    """At ctx <= topk the selection IS the full prefix, so DSA attention is
    dense causal attention — bitwise, not merely close."""
    attn, cfg = _build_attention(seed=2)
    dense_resources = _CausalDenseResources()
    attn.bind_resources({KV_RESOURCE: dense_resources, ATTN_RESOURCE: dense_resources})
    t = cfg.index_topk  # ctx <= topk regime
    h = torch.randn(t, cfg.hidden_size) * 0.1
    pos = torch.arange(t)

    q_c = attn.q_a_layernorm(attn.q_a_proj(h))  # the shared q latent
    k_hist = attn.indexer.compute_k(h, pos)
    sel = attn.indexer.compute_selection(q_c, h, pos, k_hist)
    assert (sel == -1).any()  # early rows really are padded

    dense = attn(h, pos)
    assert dense_resources.k is not None  # the dense pass went through the resources
    sparse = attn(h, pos, dsa_selection=sel)
    assert torch.equal(dense, sparse)


def test_load_indexer_keys_land_dequantized_on_full_layer():
    torch.manual_seed(4)
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    cfg.dsa_long_context = True  # indexer keys load only on the DSA path
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

    # fp8 pairs dequantize exactly despite their plain ``.weight`` names.
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
    cfg.dsa_long_context = True  # the explicit load_indexer=False must still win
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


# -- batched selection == the original per-token loop ------------------------
# Verbatim copy of the loop compute_selection ran before it went batched
# (one topk launch + one host sync per token); kept here as the reference.
def _reference_compute_selection(idx, q_c, hidden_states, positions, k_history):
    num_tokens = q_c.shape[0]
    num_keys = k_history.shape[0]
    if num_keys <= int(positions.max()):
        raise ValueError(
            f"k_history has {num_keys} rows but positions reach "
            f"{int(positions.max())}; the causal window includes self"
        )

    q = idx.wq_b(q_c).view(num_tokens, idx.n_heads, idx.head_dim)
    q = idx._rope_first_dims(q, positions)
    w = idx.weights_proj(hidden_states) * idx.weight_scale  # (T, H)

    # score[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s]): per-head ReLU
    # BEFORE the weighted sum; the raw weights get no softmax/sigmoid.
    dots = torch.einsum("thd,sd->ths", q, k_history).relu()
    scores = torch.einsum("th,ths->ts", w, dots)

    # Causal window INCLUDING self: candidates are positions 0..p_t.
    key_pos = torch.arange(num_keys, device=scores.device)
    scores = scores.masked_fill(
        key_pos.unsqueeze(0) > positions.unsqueeze(1), float("-inf"))

    return _reference_select_topk_causal(scores, positions, idx.topk), scores


def _reference_select_topk_causal(scores, positions, topk):
    num_tokens = scores.shape[0]
    selection = torch.full(
        (num_tokens, topk), -1, dtype=torch.int32, device=scores.device)
    for t in range(num_tokens):
        n = min(topk, int(positions[t]) + 1)
        selection[t, :n] = scores[t].topk(n).indices.to(torch.int32)
    return selection


def _selected_scores(scores, selection):
    """(T, topk) picked scores, -inf at -1 padding, sorted descending: the
    top-k VALUES a row selected, independent of which tied key won."""
    picked = scores.gather(1, selection.clamp(min=0).long())
    picked = picked.masked_fill(selection < 0, float("-inf"))
    return picked.sort(dim=1, descending=True).values


def _assert_same_topk(new, ref, scores):
    assert new.shape == ref.shape and new.dtype == ref.dtype == torch.int32
    assert torch.equal(new < 0, ref < 0)  # identical -1 padding
    assert torch.equal(_selected_scores(scores, new), _selected_scores(scores, ref))
    # any disagreement with torch.topk's pick is a genuine tie: the two keys
    # the rows differ on carry the same score
    for t in torch.nonzero((new != ref).any(dim=1)).flatten().tolist():
        diff = (new[t] != ref[t]) & (new[t] >= 0)
        assert torch.equal(scores[t, new[t][diff].long()], scores[t, ref[t][diff].long()])


@pytest.mark.parametrize("num_tokens,topk,start", [
    (1, 4, 0),       # single token, window 1 < topk
    (5, 4, 0),       # rows below, at and above topk
    (64, 4, 0),      # deep truncation regime
    (300, 64, 0),    # prefill-sized batch
    (5, 64, 0),      # num_keys < topk: k_eff clamps to the history
    (5, 4, 37),      # decode-style continuation: positions 37..41 over 42 keys
    (1, 64, 100),    # one decode token deep into a long history
])
def test_batched_selection_matches_the_original_loop(num_tokens, topk, start):
    torch.manual_seed(num_tokens * 7 + topk + start)
    cfg = Glm52ModelConfig.reduced()
    cfg.index_topk = topk
    idx = Glm52Indexer(cfg)
    _randomize_indexer(idx)
    total = start + num_tokens
    h_all = torch.randn(total, cfg.hidden_size) * 0.1
    pos_all = torch.arange(total)
    k_hist = idx.compute_k(h_all, pos_all)
    h = h_all[start:]
    q_c = torch.randn(num_tokens, cfg.q_lora_rank) * 0.1
    pos = pos_all[start:]

    ref, scores = _reference_compute_selection(idx, q_c, h, pos, k_hist)
    new = idx.compute_selection(q_c, h, pos, k_hist)

    _assert_same_topk(new, ref, scores)
    # the per-head relu yields exact-zero scores, so real inputs DO tie; on
    # rows without a tie the two agree bitwise, order included
    tie_free = torch.tensor([
        scores[t][scores[t] > float("-inf")].unique().numel()
        == int((scores[t] > float("-inf")).sum()) for t in range(num_tokens)])
    assert torch.equal(new[tie_free], ref[tie_free])


@pytest.mark.parametrize("num_tokens,num_keys,topk", [
    (1, 1, 4), (5, 5, 4), (64, 64, 8), (300, 300, 64), (16, 200, 2048), (7, 300, 32),
])
def test_select_topk_causal_is_bitwise_the_loop_without_ties(num_tokens, num_keys, topk):
    torch.manual_seed(num_tokens + num_keys + topk)
    scores = torch.randn(num_tokens, num_keys, dtype=torch.float64)  # tie-free
    positions = torch.randint(0, num_keys, (num_tokens,))  # unsorted, decode-style
    scores = scores.masked_fill(
        torch.arange(num_keys).unsqueeze(0) > positions.unsqueeze(1), float("-inf"))
    assert torch.equal(
        select_topk_causal(scores, positions, topk),
        _reference_select_topk_causal(scores, positions, topk))


@pytest.mark.parametrize("num_tokens,num_keys,topk", [(5, 20, 4), (64, 300, 8), (300, 300, 64)])
def test_select_topk_causal_breaks_ties_toward_the_earlier_key(num_tokens, num_keys, topk):
    torch.manual_seed(num_tokens)
    scores = torch.randint(0, 3, (num_tokens, num_keys)).float()  # heavy ties
    positions = torch.randint(0, num_keys, (num_tokens,))
    scores = scores.masked_fill(
        torch.arange(num_keys).unsqueeze(0) > positions.unsqueeze(1), float("-inf"))

    new = select_topk_causal(scores, positions, topk)
    ref = _reference_select_topk_causal(scores, positions, topk)
    _assert_same_topk(new, ref, scores)  # same top-k values, same padding
    for t in range(num_tokens):
        row = new[t][new[t] >= 0].long()
        assert row.unique().numel() == row.numel() and bool((row <= positions[t]).all())
        # descending score; among equals, ascending key position
        s = scores[t, row]
        assert bool((s[:-1] >= s[1:]).all())
        same = s[:-1] == s[1:]
        assert bool((row[:-1][same] < row[1:][same]).all())

