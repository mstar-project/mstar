"""The ``noaux_tc`` router's scoring functions."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.model.components.moe import NoAuxTcRouter


class TestRouterScoring:
    def test_sqrtsoftplus_scores_route_and_weigh(self):
        torch.manual_seed(0)
        router = NoAuxTcRouter(hidden_size=16, n_routed_experts=8, num_experts_per_tok=2,
                               routed_scaling_factor=1.5, scoring_func="sqrtsoftplus")
        torch.nn.init.normal_(router.weight)
        torch.nn.init.normal_(router.e_score_correction_bias)
        x = torch.randn(5, 16)
        weights, ids, _ = router(x)
        scores = torch.nn.functional.softplus(x @ router.weight.T).sqrt()
        want_ids = (scores + router.e_score_correction_bias).topk(2, dim=-1).indices
        assert torch.equal(ids.sort(-1).values, want_ids.sort(-1).values)
        # the bias picks the experts; the weights are the unbiased scores, renormalized and scaled
        picked = scores.gather(1, ids)
        assert torch.allclose(weights, 1.5 * picked / picked.sum(-1, keepdim=True))

    def test_an_unknown_scoring_function_is_refused(self):
        with pytest.raises(ValueError, match="scoring_func"):
            NoAuxTcRouter(16, 8, 2, 1.0, scoring_func="softmax")

    def test_the_router_kernels_are_refused_for_another_scoring(self):
        router = NoAuxTcRouter(16, 8, 2, 1.0, scoring_func="sqrtsoftplus")
        with pytest.raises(ValueError, match="sigmoid"):
            router.finalize_weights(topk_kernel=True)
