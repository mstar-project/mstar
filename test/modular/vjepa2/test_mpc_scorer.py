"""P3.B unit tests for the MPC cost scorer submodule.

Parity anchor: ``vjepa2/notebooks/utils/mpc_utils.py::l1`` — the CEM
objective upstream uses to rank K action candidates against a goal
latent.  Implemented in :class:`VJepa2MPCScorerSubmodule._cost`.

Three things this file exercises, all CPU with synthetic tensors so the
test is cheap and environment-independent:

1. L1 scorer math: costs produced by ``forward`` match a hand-computed
   reference over the last two dims.
2. Argmin selection: ``best_index`` picks the candidate with the lowest
   cost (ties resolved by ``torch.argmin`` semantics — first occurrence).
3. Cost-function dispatch: ``l1`` / ``l2`` / ``cosine`` produce different
   rankings for a hand-crafted case where each objective ranks candidates
   differently.
4. Shape-mismatch guards raise (goal dim != predicted dim) — the scorer
   must not silently fall through to NaN.
"""

from __future__ import annotations

import pytest
import torch

from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.model.vjepa2.config import VJepa2Config

try:
    from mminf.model.vjepa2.submodules import VJepa2MPCScorerSubmodule
except (ImportError, AttributeError) as e:  # pragma: no cover - env-specific
    pytest.skip(
        f"Cannot import VJepa2MPCScorerSubmodule in this env: {e}",
        allow_module_level=True,
    )


def _make_info() -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        graph_walk="prefill_video_mpc",
        requires_cfg=False,
        fwd_index=0,
        random_seed=0,
        max_tokens=0,
        sampling_config={},
    )


def _make_config(cost_fn: str = "l1") -> VJepa2Config:
    return VJepa2Config(mpc_cost_fn=cost_fn)


class TestL1Scorer:
    def test_l1_matches_hand_computed_reference(self):
        """For a tiny synthetic batch, assert the submodule's cost values
        exactly match ``(pred - goal).abs().mean(dim=last-dim-flatten)``."""
        torch.manual_seed(0)
        K, N, D = 4, 16, 8
        pred = torch.randn(K, N, D)
        goal = torch.randn(1, N, D)

        submodule = VJepa2MPCScorerSubmodule(_make_config("l1"))
        with torch.no_grad():
            out = submodule(
                request_info=_make_info(),
                predicted_hidden=pred,
                goal_hidden=goal,
            )

        costs = out["costs"][0]
        assert costs.shape == (K,)
        # Reference: flatten feature dims, L1-mean.
        ref = (pred.flatten(1) - goal.flatten(1)).abs().mean(dim=-1)
        torch.testing.assert_close(costs, ref)

    def test_best_index_picks_lowest_cost(self):
        torch.manual_seed(1)
        K, N, D = 5, 8, 4
        # Craft candidates so rid=2 is closest to goal.
        goal = torch.randn(1, N, D)
        pred = torch.randn(K, N, D) * 10.0
        pred[2] = goal.squeeze(0) + 1e-3 * torch.randn(N, D)

        submodule = VJepa2MPCScorerSubmodule(_make_config("l1"))
        with torch.no_grad():
            out = submodule(
                request_info=_make_info(),
                predicted_hidden=pred,
                goal_hidden=goal,
            )
        assert int(out["best_index"][0].item()) == 2

    def test_goal_hidden_without_batch_dim_is_accepted(self):
        """Client may send goal as [N, D] rather than [1, N, D]; the
        submodule must unsqueeze rather than error out."""
        K, N, D = 3, 4, 6
        pred = torch.randn(K, N, D)
        goal = torch.randn(N, D)  # no batch dim

        submodule = VJepa2MPCScorerSubmodule(_make_config("l1"))
        with torch.no_grad():
            out = submodule(
                request_info=_make_info(),
                predicted_hidden=pred,
                goal_hidden=goal,
            )
        assert out["costs"][0].shape == (K,)
        assert int(out["best_index"][0].item()) in range(K)


class TestCostFnDispatch:
    def _scenario(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct a case where L1 and cosine disagree.

        - Candidate 0: a scaled copy of goal.  Cosine dist = 0 (perfect
          direction) but L1 is large (scale mismatch).
        - Candidate 1: goal + small random noise.  Cosine dist tiny;
          L1 tiny.
        - Candidate 2: an orthogonal vector.  Cosine dist ≈ 1; L1 large.
        """
        torch.manual_seed(42)
        N, D = 8, 16
        goal = torch.randn(1, N, D)
        g_flat = goal.flatten(1)[0]  # [N*D]
        # Orthogonal basis: pick a random vector and subtract its goal projection.
        v = torch.randn_like(g_flat)
        orthogonal = v - (v @ g_flat / (g_flat @ g_flat)) * g_flat
        pred = torch.stack(
            [
                (goal.squeeze(0) * 10.0),                                # scaled goal
                (goal.squeeze(0) + 1e-2 * torch.randn(N, D)),            # near goal
                orthogonal.reshape(N, D),                                # orthogonal
            ],
            dim=0,
        )
        return pred, goal

    def test_l1_best_is_near_goal(self):
        pred, goal = self._scenario()
        submodule = VJepa2MPCScorerSubmodule(_make_config("l1"))
        with torch.no_grad():
            out = submodule(
                request_info=_make_info(),
                predicted_hidden=pred,
                goal_hidden=goal,
            )
        assert int(out["best_index"][0].item()) == 1  # "near goal" wins under L1.

    def test_cosine_accepts_scaled_copy(self):
        """Under cosine, the scaled copy (direction = goal) is a top
        candidate — we just confirm it beats the orthogonal one."""
        pred, goal = self._scenario()
        submodule = VJepa2MPCScorerSubmodule(_make_config("cosine"))
        with torch.no_grad():
            out = submodule(
                request_info=_make_info(),
                predicted_hidden=pred,
                goal_hidden=goal,
            )
        costs = out["costs"][0]
        # Scaled copy (cand 0) and near-goal (cand 1) should both beat
        # orthogonal (cand 2).  Don't assert exact ordering between 0/1
        # — near-goal also scores well on cosine because noise is small.
        assert costs[0].item() < costs[2].item()
        assert costs[1].item() < costs[2].item()

    def test_l2_costs_are_non_negative(self):
        pred, goal = self._scenario()
        submodule = VJepa2MPCScorerSubmodule(_make_config("l2"))
        with torch.no_grad():
            out = submodule(
                request_info=_make_info(),
                predicted_hidden=pred,
                goal_hidden=goal,
            )
        costs = out["costs"][0]
        assert (costs >= 0).all()

    def test_unknown_cost_fn_raises(self):
        pred, goal = self._scenario()
        submodule = VJepa2MPCScorerSubmodule(_make_config("nonexistent"))
        with pytest.raises(ValueError, match="Unknown mpc_cost_fn"):
            submodule(
                request_info=_make_info(),
                predicted_hidden=pred,
                goal_hidden=goal,
            )


class TestShapeGuards:
    def test_feature_shape_mismatch_raises(self):
        """predicted ``[K, N, D1]`` vs goal ``[1, N, D2]`` (D1 != D2)
        must raise rather than silently broadcasting / flattening."""
        K, N, D1, D2 = 3, 4, 6, 8
        pred = torch.randn(K, N, D1)
        goal = torch.randn(1, N, D2)

        submodule = VJepa2MPCScorerSubmodule(_make_config("l1"))
        with pytest.raises(ValueError, match="feature shape"):
            submodule(
                request_info=_make_info(),
                predicted_hidden=pred,
                goal_hidden=goal,
            )

    def test_goal_batch_gt_one_raises(self):
        K, N, D = 3, 4, 6
        pred = torch.randn(K, N, D)
        goal = torch.randn(2, N, D)  # batch dim > 1 — ambiguous.

        submodule = VJepa2MPCScorerSubmodule(_make_config("l1"))
        with pytest.raises(ValueError, match="batch dim 1"):
            submodule(
                request_info=_make_info(),
                predicted_hidden=pred,
                goal_hidden=goal,
            )


class TestMPCPredictor:
    """Exercises :class:`VJepa2MPCPredictorSubmodule` — verifies the K-way
    broadcast math actually runs through the AC predictor and emits the
    right output shape.  Bit-parity against a reference loop is covered
    in the GPU integration test; here we just check shape + dtype.
    """

    def _tiny_ac_config(self) -> VJepa2Config:
        """Small AC config so CPU forward is fast."""
        from mminf.model.vjepa2.config import VJepa2ACPredictorConfig

        cfg = VJepa2Config(
            predictor_kind="ac",
            crop_size=16,
            patch_size=4,
            frames_per_clip=4,
            tubelet_size=2,
            hidden_size=32,
        )
        cfg.ac_predictor = VJepa2ACPredictorConfig(
            img_size=(16, 16),
            patch_size=4,
            num_frames=4,
            tubelet_size=2,
            embed_dim=32,
            predictor_embed_dim=16,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            action_embed_dim=7,
            is_frame_causal=True,
            use_rope=True,
        )
        return cfg

    def test_k_way_broadcast_produces_k_outputs(self):
        try:
            from mminf.model.vjepa2.components.ac_predictor import (
                VisionTransformerPredictorAC,
            )
            from mminf.model.vjepa2.submodules import VJepa2MPCPredictorSubmodule
        except (ImportError, AttributeError) as e:
            pytest.skip(f"AC predictor unavailable in this env: {e}")

        torch.manual_seed(0)
        cfg = self._tiny_ac_config()
        predictor = VisionTransformerPredictorAC(cfg.ac_predictor).eval()
        submodule = VJepa2MPCPredictorSubmodule(predictor, cfg)

        n_tokens = cfg.grid_depth * cfg.grid_size * cfg.grid_size
        enc = torch.randn(1, n_tokens, cfg.hidden_size)
        # T_action = frames_per_clip / tubelet_size = 2 for this tiny cfg.
        t_action = cfg.frames_per_clip // cfg.tubelet_size
        K = 4
        actions = torch.randn(K, t_action, 7)
        states = torch.randn(K, t_action, 7)

        packed = submodule.preprocess(
            graph_walk="prefill_video_mpc",
            per_request_inputs=[
                {
                    "encoder_hidden": [enc],
                    "actions": [actions],
                    "states": [states],
                }
            ],
            request_ids=["rid_0"],
            per_request_info={"rid_0": _make_info()},
        )
        assert packed["actions"].shape == (K, t_action, 7)
        assert packed["encoder_hidden"].shape == (1, n_tokens, cfg.hidden_size)

        with torch.no_grad():
            out = submodule(request_info=_make_info(), **packed)
        pred = out["predicted_hidden"][0]
        # Predictor's output has the same N as encoder input (matches AC
        # predictor's per-frame interleave + final projection).
        assert pred.shape[0] == K
        assert pred.dim() == 3
