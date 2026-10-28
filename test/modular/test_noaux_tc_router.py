"""NoAuxTcRouter against the routers of the DeepSeek-family checkpoints it serves, transcribed from
their reference code: Kimi K2.7-Code's ``MoEGate`` (``modeling_deepseek.py``, HF 74797c9c623),
Kimi K3's ``KimiMoEGate`` (``modeling_kimi_linear.py``, f831ab66814) and DeepSeek-V4.1's ``Gate``
(``inference/model.py``, 314113f8a7). GLM-5.2's is in ``test_glm52_moe.py``."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.model.components.moe import NoAuxTcRouter  # noqa: E402

HIDDEN, EXPERTS, TOKENS = 64, 32, 40


def _k27_moe_gate(h, weight, bias, top_k, n_group, topk_group, scale):
    """K2.7's MoEGate.forward: sigmoid, the group-limited noaux_tc pick, + 1e-20 renorm."""
    logits = F.linear(h.float(), weight.float())
    scores = logits.sigmoid()
    n = scores.shape[0]
    scores_for_choice = scores + bias.unsqueeze(0)
    group_scores = scores_for_choice.view(n, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(n, n_group, EXPERTS // n_group).reshape(n, -1)
    tmp = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    _, idx = torch.topk(tmp, k=top_k, dim=-1, sorted=False)
    w = scores.gather(1, idx)
    if top_k > 1:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w * scale, idx


def _k3_moe_gate(h, weight, bias, top_k, activation, scale):
    """K3's KimiMoEGate.forward with num_expert_group 1."""
    logits = F.linear(h.float(), weight.float())
    scores = logits.sigmoid() if activation == "sigmoid" else logits.softmax(dim=1)
    _, idx = torch.topk(scores + bias.unsqueeze(0), k=top_k, dim=-1, sorted=False)
    w = scores.gather(1, idx)
    if top_k > 1:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w * scale, idx


def _v41_gate(h, weight, bias, bias_vl, image_mask, top_k, scale):
    """DeepSeek-V4.1's Gate.forward (score_func sqrtsoftplus, gate_temp 1)."""
    scores = F.softplus(F.linear(h.float(), weight.float())).sqrt()
    if image_mask is not None:
        bias = torch.where(image_mask.unsqueeze(-1), bias_vl, bias)
    idx = (scores + bias).topk(top_k, dim=-1)[1]
    w = scores.gather(1, idx)
    if top_k > 1:
        w /= w.sum(dim=-1, keepdim=True) + 1e-20
    return w * scale, idx


def _by_expert(weights, ids):
    """Per row, the (expert, weight) pairs in expert order: top-k order is not part of the result."""
    order = ids.argsort(dim=-1)
    return ids.gather(1, order), weights.gather(1, order)


def _router(**kw):
    r = NoAuxTcRouter(HIDDEN, EXPERTS, kw.pop("top_k"), kw.pop("scale"), **kw)
    r.weight.data.normal_(0, 0.5)
    r.e_score_correction_bias.data.normal_(0, 0.3)
    if r.vl_e_score_correction_bias is not None:
        r.vl_e_score_correction_bias.data.normal_(0, 0.3)
    r.finalize_weights()
    return r


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_kimi_k27():
    r = _router(top_k=8, scale=2.827, topk_norm_eps=1e-20)
    h = torch.randn(TOKENS, HIDDEN)
    w, ids, state = r(h)
    assert state is None and w.dtype == torch.float32
    ref = _k27_moe_gate(h, r.weight, r.e_score_correction_bias, 8, 1, 1, 2.827)
    for got, want in zip(_by_expert(w, ids), _by_expert(*ref), strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize("activation", ["sigmoid", "softmax"])
def test_kimi_k3(activation):
    r = _router(top_k=16, scale=1.0, topk_norm_eps=1e-20, scoring_func=activation)
    h = torch.randn(TOKENS, HIDDEN)
    w, ids, _ = r(h)
    ref = _k3_moe_gate(h, r.weight, r.e_score_correction_bias, 16, activation, 1.0)
    for got, want in zip(_by_expert(w, ids), _by_expert(*ref), strict=True):
        assert torch.equal(got, want)


def test_deepseek_v41_with_image_rows():
    r = _router(top_k=6, scale=1.5, topk_norm_eps=1e-20, scoring_func="sqrtsoftplus", vl_bias=True)
    h = torch.randn(TOKENS, HIDDEN)
    image = torch.rand(TOKENS) < 0.4
    for mask in (None, image):
        w, ids, _ = r(h, vl_rows=mask)
        with torch.no_grad():
            ref = _v41_gate(h, r.weight, r.e_score_correction_bias, r.vl_e_score_correction_bias,
                            mask, 6, 1.5)
        (got_ids, got_w), (want_ids, want_w) = _by_expert(w, ids), _by_expert(*ref)
        assert torch.equal(got_ids, want_ids)
        # V4.1 sorts its top-k, so its renormalizing sum adds the same six scores in another
        # order: the weights can differ in the last fp32 bit
        torch.testing.assert_close(got_w, want_w, rtol=4e-7, atol=0)
    # the image bias moves only the image rows' picks
    _, plain, _ = r(h)
    _, mixed, _ = r(h, vl_rows=image)
    assert torch.equal(_by_expert(plain, plain)[0][~image], _by_expert(mixed, mixed)[0][~image])
    assert not torch.equal(_by_expert(plain, plain)[0][image], _by_expert(mixed, mixed)[0][image])


def test_eps_keeps_an_underflowed_row_finite():
    h = torch.full((1, HIDDEN), 16.0)
    for eps, finite in ((1e-20, True), (0.0, False)):
        r = _router(top_k=2, scale=1.0, topk_norm_eps=eps)
        r.weight.data.fill_(-1.0)  # logits -1024: every sigmoid underflows to 0
        r.finalize_weights()
        assert bool(torch.isfinite(r(h)[0]).all()) is finite


def test_refusals():
    with pytest.raises(ValueError, match="scoring_func"):
        NoAuxTcRouter(HIDDEN, EXPERTS, 2, 1.0, scoring_func="tanh")
    r = _router(top_k=2, scale=1.0)
    with pytest.raises(ValueError, match="vl_bias"):
        r(torch.randn(3, HIDDEN), vl_rows=torch.ones(3, dtype=torch.bool))
    assert "vl_e_score_correction_bias" not in dict(r.named_parameters())
