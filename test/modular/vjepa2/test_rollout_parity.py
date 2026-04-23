"""Parity test for V-JEPA 2 Phase-2 rollout submodule.

Mirrors upstream
``vjepa2/evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.py::AnticipativeWrapper.forward``
as an in-test reference (reproduces its sliding-window logic directly in
Python against our ported ``VJEPA2Predictor``), and asserts that calling
``VJepa2RolloutPredictorSubmodule.forward`` H times with the loop-back
wiring yields the same per-iteration ``predicted_hidden`` tensors
bit-exactly.

Pure CPU, tiny config — no GPU or HF cache required.  For full-weight
GPU parity against ``AnticipativeWrapper`` on a real checkpoint, see the
downstream integration test the user can enable when running on the cluster.
"""

from __future__ import annotations

import pytest
import torch

from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.model.vjepa2.components.predictor import VJEPA2Predictor
from mminf.model.vjepa2.config import VJepa2Config

# ``submodules`` pulls in ``mminf.engine`` at import, which on some local
# torch builds fails to set dynamo-config attributes that only exist in
# newer torch.  Skip rather than crash the collection so the pure-component
# parity tests in this directory still run in either environment.
try:
    from mminf.model.vjepa2.submodules import VJepa2RolloutPredictorSubmodule
except (ImportError, AttributeError) as e:  # pragma: no cover - env-specific
    pytest.skip(
        f"Cannot import VJepa2RolloutPredictorSubmodule in this env: {e}",
        allow_module_level=True,
    )


def _tiny_config() -> VJepa2Config:
    """Small-but-realistic V-JEPA 2 shape so the rollout math exercises
    both the sliding window (N_pred < N_ctxt) and the predictor's RoPE
    positions beyond the encoder's native range (skip_positions > N).
    """
    return VJepa2Config(
        patch_size=4,
        crop_size=16,
        frames_per_clip=4,
        tubelet_size=2,
        hidden_size=48,
        num_attention_heads=4,
        num_hidden_layers=2,
        mlp_ratio=2.0,
        pred_hidden_size=24,
        pred_num_attention_heads=4,
        pred_num_hidden_layers=2,
        pred_num_mask_tokens=4,
        pred_mlp_ratio=2.0,
        hidden_act="gelu",
    )


def _make_request_info(iter_idx: int, rollout_horizon: int) -> CurrentForwardPassInfo:
    """Minimal ``CurrentForwardPassInfo`` that exposes the loop iter count
    the submodule expects (populated by ``worker.py:941-943`` in production).
    """
    info = CurrentForwardPassInfo(
        graph_walk="prefill_video_rollout",
        requires_cfg=False,
        fwd_index=0,
        random_seed=0,
        max_tokens=0,
        sampling_config={},
    )
    info.dynamic_loop_iter_counts["rollout_loop"] = iter_idx
    info.step_metadata["rollout_horizon"] = rollout_horizon
    return info


def _anticipative_reference(
    predictor: VJEPA2Predictor,
    x_full: torch.Tensor,
    num_steps: int,
    num_output_frames: int,
    frames_per_second: int,
    anticipation_seconds: float,
    config: VJepa2Config,
) -> list[torch.Tensor]:
    """In-test re-implementation of ``AnticipativeWrapper.forward`` lines
    198-223, stripped to the masked-predictor (non-hierarchical, non-AC)
    path.  Returns the per-iteration predicted tensors — the same quantity
    our submodule emits on its ``predicted_hidden`` edge each iter.
    """
    b, n, _ = x_full.shape
    grid = config.grid_size

    ctxt_positions = torch.arange(n).unsqueeze(0).repeat(b, 1).to(x_full.device)
    anticipation_steps = int(anticipation_seconds * frames_per_second / config.tubelet_size)
    skip_positions = n + (grid * grid) * anticipation_steps
    n_pred = grid * grid * (num_output_frames // config.tubelet_size)
    tgt_positions = (torch.arange(n_pred).unsqueeze(0).repeat(b, 1) + skip_positions).to(x_full.device)

    x_pred_input = x_full
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(num_steps):
            x_pred = predictor(x_pred_input, [ctxt_positions], [tgt_positions])
            predictions.append(x_pred)
            x_pred_input = torch.cat([x_pred_input[:, n_pred:, :], x_pred], dim=1)
    return predictions


def _submodule_loop(
    submodule: VJepa2RolloutPredictorSubmodule,
    x_full: torch.Tensor,
    num_steps: int,
) -> list[torch.Tensor]:
    """Drive our submodule's ``forward`` H times, threading ``encoder_hidden``
    from iter k to iter k+1 the way the mminf DynamicLoop does in production.
    """
    encoder_hidden = x_full
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for k in range(num_steps):
            info = _make_request_info(iter_idx=k, rollout_horizon=num_steps)
            out = submodule.forward(info, encoder_hidden=encoder_hidden)
            predictions.append(out["predicted_hidden"][0])
            encoder_hidden = out["encoder_hidden"][0]
    return predictions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRolloutParity:
    def test_bit_exact_parity_h4(self):
        """Running H=4 iterations through our submodule produces the same
        per-iter ``predicted_hidden`` as the inline AnticipativeWrapper
        reference when both use the same predictor weights + input."""
        torch.manual_seed(0)
        config = _tiny_config()
        predictor = VJEPA2Predictor(config).eval()

        b = 2
        x_full = torch.randn(b, config.num_patches, config.hidden_size)

        submodule = VJepa2RolloutPredictorSubmodule(
            predictor=predictor,
            config=config,
            num_output_frames=2,
            frames_per_second=4,
            anticipation_seconds=1.0,
        )

        num_steps = 4
        ref = _anticipative_reference(
            predictor,
            x_full,
            num_steps=num_steps,
            num_output_frames=2,
            frames_per_second=4,
            anticipation_seconds=1.0,
            config=config,
        )
        ours = _submodule_loop(submodule, x_full, num_steps=num_steps)

        assert len(ref) == len(ours) == num_steps
        for k, (r, o) in enumerate(zip(ref, ours, strict=True)):
            assert r.shape == o.shape, f"iter {k}: shape {r.shape} vs {o.shape}"
            diff = (r - o).abs().max().item()
            assert diff < 1e-6, f"iter {k}: max abs diff = {diff}"

    def test_shapes_and_sliding_window_invariant(self):
        """After each iteration the ``encoder_hidden`` loop-back is the
        same length as the input; the oldest N_pred tokens are dropped and
        N_pred predicted tokens are appended."""
        torch.manual_seed(1)
        config = _tiny_config()
        predictor = VJEPA2Predictor(config).eval()

        b = 1
        x_full = torch.randn(b, config.num_patches, config.hidden_size)

        submodule = VJepa2RolloutPredictorSubmodule(
            predictor=predictor,
            config=config,
            num_output_frames=2,
            frames_per_second=4,
            anticipation_seconds=1.0,
        )

        n = x_full.size(1)
        grid = config.grid_size
        n_pred = grid * grid * (2 // config.tubelet_size)

        encoder_hidden = x_full
        with torch.no_grad():
            for k in range(3):
                info = _make_request_info(iter_idx=k, rollout_horizon=3)
                out = submodule.forward(info, encoder_hidden=encoder_hidden)
                predicted = out["predicted_hidden"][0]
                next_hidden = out["encoder_hidden"][0]
                assert predicted.shape == (b, n_pred, config.hidden_size)
                assert next_hidden.shape == (b, n, config.hidden_size)
                # The tail of next_hidden is the new prediction.
                torch.testing.assert_close(next_hidden[:, -n_pred:, :], predicted)
                # The head of next_hidden is the tail of the prior hidden.
                torch.testing.assert_close(next_hidden[:, : n - n_pred, :], encoder_hidden[:, n_pred:, :])
                encoder_hidden = next_hidden


class TestEarlyExit:
    def test_register_loop_stop_at_requested_horizon(self):
        """When ``step_metadata["rollout_horizon"]`` is reached, the
        submodule requests that the DynamicLoop terminate — the signal
        lands in ``request_info.dynamic_loop_stop_signals``."""
        torch.manual_seed(2)
        config = _tiny_config()
        predictor = VJEPA2Predictor(config).eval()

        submodule = VJepa2RolloutPredictorSubmodule(
            predictor=predictor,
            config=config,
            num_output_frames=2,
            frames_per_second=4,
            anticipation_seconds=1.0,
        )

        x_full = torch.randn(1, config.num_patches, config.hidden_size)
        horizon = 3
        stop_seen_at: list[int] = []
        encoder_hidden = x_full
        with torch.no_grad():
            # Drive the submodule as if max_iters were larger than horizon;
            # after iter horizon-1 the submodule should register the stop.
            for k in range(horizon + 2):
                info = _make_request_info(iter_idx=k, rollout_horizon=horizon)
                out = submodule.forward(info, encoder_hidden=encoder_hidden)
                encoder_hidden = out["encoder_hidden"][0]
                if "rollout_loop" in info.dynamic_loop_stop_signals:
                    stop_seen_at.append(k)

        # The very first iter where the stop fires should be iter == horizon - 1,
        # because the submodule checks ``iter_idx + 1 >= horizon``.
        assert stop_seen_at, "submodule never registered a loop stop"
        assert stop_seen_at[0] == horizon - 1
