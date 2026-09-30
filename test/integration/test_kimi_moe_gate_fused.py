"""CPU check for the fused n_group == 1 Kimi router path via TRITON_INTERPRET.

TRITON_INTERPRET must be set before triton is first imported (it runs the
kernel through Triton's interpreter instead of compiling for a GPU), so it is
set here before any mstar import pulls triton in transitively.
"""
import os

os.environ["TRITON_INTERPRET"] = "1"

import torch

from mstar.model.kimi_k2_7.components.moe import _fused_group1_topk

TOP_K = 8


def _old_gate_forward(scores, bias, n_group, topk_group, top_k, norm_topk_prob, routed_scaling_factor):
    """Pre-fusion KimiMoEGate router math (copied verbatim from before the
    n_group == 1 short-circuit). n_group == 1 collapses the group stage to a
    no-op, which is exactly the identity the fused kernel exploits."""
    num_token = scores.shape[0]
    if bias is not None:
        original_scores = scores
        scores = scores + bias.unsqueeze(0)
        group_scores = scores.view(num_token, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    else:
        original_scores = scores
        group_scores = scores.view(num_token, n_group, -1).max(dim=-1).values

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, n_group, scores.shape[-1] // n_group)
        .reshape(num_token, -1)
    )
    masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

    if bias is not None:
        topk_ids = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)

    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids


def _check(E, T, has_bias, norm_topk_prob, scale, seed):
    torch.manual_seed(seed)
    scores = torch.rand(T, E, dtype=torch.float32)
    bias = torch.randn(E, dtype=torch.float32) * 0.1 if has_bias else None

    ref_weights, ref_ids = _old_gate_forward(
        scores, bias, n_group=1, topk_group=1, top_k=TOP_K,
        norm_topk_prob=norm_topk_prob, routed_scaling_factor=scale,
    )
    got_weights, got_ids = _fused_group1_topk(scores, bias, TOP_K, norm_topk_prob, scale)

    assert got_ids.dtype == torch.int64
    assert got_weights.dtype == torch.float32
    for t in range(T):
        ref_set = set(ref_ids[t].tolist())
        got_set = set(got_ids[t].tolist())
        assert ref_set == got_set, (E, T, has_bias, norm_topk_prob, scale, t, ref_set, got_set)
        ref_map = dict(zip(ref_ids[t].tolist(), ref_weights[t].tolist(), strict=True))
        got_map = dict(zip(got_ids[t].tolist(), got_weights[t].tolist(), strict=True))
        for eid, w in ref_map.items():
            assert abs(w - got_map[eid]) < 1e-5, (E, T, eid, w, got_map[eid])


def test_fused_gate_matches_reference():
    for E in (32, 96, 192, 384):
        for T in (1, 8, 32, 300):
            for has_bias in (True, False):
                for norm_topk_prob in (True, False):
                    for scale in (1.0, 2.827):
                        _check(E, T, has_bias, norm_topk_prob, scale, seed=E + T)


def test_fused_gate_tie_break_lowest_index():
    E, T = 32, 4
    scores = torch.zeros(T, E, dtype=torch.float32)
    # TOP_K + 2 experts exactly tied at the max score: the lowest TOP_K
    # indices among the tie must win, deterministically, for every token.
    scores[:, : TOP_K + 2] = 1.0
    weights, ids = _fused_group1_topk(scores, None, TOP_K, True, 1.0)
    expected_ids = list(range(TOP_K))
    for t in range(T):
        assert sorted(ids[t].tolist()) == expected_ids

    weights2, ids2 = _fused_group1_topk(scores, None, TOP_K, True, 1.0)
    torch.testing.assert_close(ids, ids2)
    torch.testing.assert_close(weights, weights2)
