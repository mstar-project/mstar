"""Pinned DeepSeek-V4.1 Gate outputs from the published inference/model.py.

Reference revision: 314113f8a7dadb388215396ba5fbee2c610a04d6.
"""

import pytest
import torch

from mstar.model.components.moe import NoAuxTcRouter


def test_sqrtsoftplus_image_routing_matches_published_gate() -> None:
    router = NoAuxTcRouter(
        5, 6, 2, routed_scaling_factor=1.5, score_func="sqrtsoftplus",
        gate_temp=0.7, image_bias=True,
    )
    gen = torch.Generator().manual_seed(44)
    tokens = torch.randn(4, 5, generator=gen)
    with torch.no_grad():
        router.weight.copy_(torch.randn(6, 5, generator=gen))
        router.e_score_correction_bias.copy_(torch.tensor([0.1, 0, 0.2, 0, 0, 0]))
        router.bias_vl.copy_(torch.tensor([-1, 2, 0, 0, 0, 0]))
    router.finalize_weights()

    image_mask = torch.tensor([False, True, False, True])
    weights, indices, state = router(tokens, image_mask=image_mask)
    assert state is None
    assert indices.tolist() == [[3, 1], [1, 2], [0, 1], [1, 2]]
    torch.testing.assert_close(
        weights,
        torch.tensor([
            [0.8530178070068359, 0.6469821333885193],
            [0.399028480052948, 1.1009714603424072],
            [0.8786764144897461, 0.6213235855102539],
            [0.6182732582092285, 0.8817267417907715],
        ]),
        rtol=0, atol=1e-6,
    )
    text_only, text_ids, _ = router(tokens)
    assert text_ids[1].tolist() != indices[1].tolist()
    assert not torch.equal(text_only, weights)


def test_v41_router_refuses_incompatible_fused_path() -> None:
    router = NoAuxTcRouter(
        5, 6, 2, routed_scaling_factor=1.5, score_func="sqrtsoftplus", image_bias=True,
    )
    with pytest.raises(ValueError, match="unit-temperature sigmoid"):
        router.finalize_weights(topk_kernel=True)
    with pytest.raises(ValueError, match="one boolean per token"):
        router(torch.ones(2, 5), image_mask=torch.tensor([True]))
    with pytest.raises(ValueError, match="one boolean per token"):
        router(torch.ones(2, 5), image_mask=torch.tensor([1, 0]))


def test_v41_router_requires_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        NoAuxTcRouter(5, 6, 2, routed_scaling_factor=1.5, gate_temp=0)
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        NoAuxTcRouter(5, 6, 2, routed_scaling_factor=1.5, gate_temp=float("nan"))


def test_v41_single_expert_keeps_its_unbiased_score() -> None:
    router = NoAuxTcRouter(5, 6, 1, routed_scaling_factor=1.5, score_func="sqrtsoftplus")
    weights, indices, _ = router(torch.zeros(2, 5))
    assert indices.shape == (2, 1)
    expected = torch.nn.functional.softplus(torch.tensor(0.0)).sqrt() * 1.5
    torch.testing.assert_close(weights, expected.expand(2, 1))
    assert not torch.equal(weights, torch.full_like(weights, 1.5))


def test_default_sigmoid_top_one_keeps_existing_normalization() -> None:
    router = NoAuxTcRouter(5, 6, 1, routed_scaling_factor=1.5)
    weights, _, _ = router(torch.zeros(2, 5))
    torch.testing.assert_close(weights, torch.full((2, 1), 1.5))


def test_v41_nonvision_mask_uses_text_bias() -> None:
    router = NoAuxTcRouter(5, 6, 2, routed_scaling_factor=1.5, score_func="sqrtsoftplus")
    tokens = torch.ones(2, 5)
    weights, ids, _ = router(tokens, image_mask=torch.tensor([False, True]))
    text_weights, text_ids, _ = router(tokens)
    torch.testing.assert_close(weights, text_weights)
    assert torch.equal(ids, text_ids)
