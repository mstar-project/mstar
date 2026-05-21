"""Parity test for V-JEPA 2 Phase-3.D action-conditioned rollout submodule.

Builds a hand-rolled Python reference that does sliding-window AC rollout
directly against the ported ``VisionTransformerPredictorAC``, then calls
:class:`VJepa2ACRolloutPredictorSubmodule` H times with the loop-back wiring
and asserts the per-iter ``predicted_hidden`` tensors match bit-exactly.

Explicit divergence from upstream: upstream
``vjepa2/notebooks/utils/mpc_utils.py::cem`` uses growing-context (T: 1 →
rollout+1) from a single-tubelet initial encoding.  Our encoder output is
``T=grid_depth``, so we slide the window instead.  See the plan's P3.D
"Sliding-window vs upstream growing-context" note for the full rationale.

Pure CPU, tiny config — no GPU or HF cache required.
"""

from __future__ import annotations

import pytest
import torch

from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC
from mminf.model.vjepa2.config import VJepa2ACPredictorConfig, VJepa2Config

# submodules module pulls in mminf.engine at import, which on some local
# torch builds fails to set dynamo-config attributes that only exist in
# newer torch.  Skip rather than crash collection so the pure-component
# parity tests in this directory still run in either environment.
try:
    from mminf.model.vjepa2.submodules import VJepa2ACRolloutPredictorSubmodule
except (ImportError, AttributeError) as e:  # pragma: no cover - env-specific
    pytest.skip(
        f"Cannot import VJepa2ACRolloutPredictorSubmodule in this env: {e}",
        allow_module_level=True,
    )


def _tiny_config() -> tuple[VJepa2Config, VJepa2ACPredictorConfig]:
    """Small-but-realistic AC config for fast CPU parity.

    Chose ``grid_depth = num_frames // tubelet_size = 4 // 2 = 2`` so the
    sliding-window test exercises iter_idx slicing against ``T_ctx=2``
    timesteps with H=3 (non-degenerate rollout that actually slides).
    """
    ac_cfg = VJepa2ACPredictorConfig(
        img_size=(16, 16),
        patch_size=4,
        num_frames=4,
        tubelet_size=2,
        embed_dim=24,
        predictor_embed_dim=24,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layer_norm_eps=1e-6,
        is_frame_causal=True,
        use_rope=True,
        action_embed_dim=7,
        use_extrinsics=False,
    )
    cfg = VJepa2Config(
        patch_size=4,
        crop_size=16,
        frames_per_clip=4,
        tubelet_size=2,
        hidden_size=24,
        predictor_kind="ac",
        ac_predictor=ac_cfg,
    )
    return cfg, ac_cfg


def _make_request_info(iter_idx: int, rollout_horizon: int) -> CurrentForwardPassInfo:
    """Minimal ``CurrentForwardPassInfo`` that exposes the loop iter count
    the submodule expects (populated by ``worker.py`` in production).
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


def _reference_rollout(
    predictor: VisionTransformerPredictorAC,
    encoder_hidden: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    extrinsics: torch.Tensor | None,
    num_steps: int,
    t_ctx: int,
    window: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Hand-rolled sliding-window AC rollout.

    Per iter k:
      * Slice actions/states on the time dim: ``[..., k : k + T_ctx, :]``.
      * Run the AC predictor (no compile, no caching — pure eager).
      * Take ``predicted[:, -window:, :]`` as the new tubelet group.
      * Slide: ``cat([encoder_hidden[:, window:, :], new_tg], dim=1)``.

    Returns ``(per_iter_new_tg, per_iter_next_encoder_hidden)`` as Python
    lists.  Bit-exact with the submodule's ``_rollout_step`` because the
    submodule's math is identical — the only differences are logging and
    the register_loop_stop side effect.
    """
    eh = encoder_hidden
    new_tgs: list[torch.Tensor] = []
    next_ehs: list[torch.Tensor] = []
    with torch.no_grad():
        for k in range(num_steps):
            end = k + t_ctx
            acts_k = actions[:, k:end].contiguous()
            sts_k = states[:, k:end].contiguous()
            ext_k = extrinsics[:, k:end].contiguous() if extrinsics is not None else None
            predicted = predictor(eh, acts_k, sts_k, extrinsics=ext_k)
            new_tg = predicted[:, -window:, :]
            eh = torch.cat([eh[:, window:, :], new_tg], dim=1)
            new_tgs.append(new_tg)
            next_ehs.append(eh)
    return new_tgs, next_ehs


def _submodule_loop(
    submodule: VJepa2ACRolloutPredictorSubmodule,
    encoder_hidden: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    num_steps: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Drive the submodule H times, threading ``encoder_hidden`` via
    loop-back just like the mminf DynamicLoop does.
    """
    eh = encoder_hidden
    new_tgs: list[torch.Tensor] = []
    next_ehs: list[torch.Tensor] = []
    with torch.no_grad():
        for k in range(num_steps):
            info = _make_request_info(iter_idx=k, rollout_horizon=num_steps)
            out = submodule.forward(
                info,
                encoder_hidden=eh,
                actions=actions,
                states=states,
            )
            new_tgs.append(out["predicted_hidden"][0])
            eh = out["encoder_hidden"][0]
            next_ehs.append(eh)
    return new_tgs, next_ehs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestACRolloutParity:
    def test_bit_exact_parity_h3(self):
        """Running H=3 iterations through the submodule produces the same
        per-iter ``predicted_hidden`` AND ``encoder_hidden`` loop-back as
        the hand-rolled sliding-window reference.
        """
        torch.manual_seed(0)
        cfg, _ac = _tiny_config()
        predictor = VisionTransformerPredictorAC(cfg.ac_predictor).eval()

        b = 1
        t_ctx = cfg.grid_depth  # 2
        window = cfg.grid_size * cfg.grid_size  # 16
        n = t_ctx * window  # 32

        num_steps = 3
        t_total = t_ctx + num_steps - 1  # 4

        encoder_hidden = torch.randn(b, n, cfg.hidden_size)
        actions = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)
        states = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)

        submodule = VJepa2ACRolloutPredictorSubmodule(predictor, cfg)

        ref_new, ref_eh = _reference_rollout(
            predictor,
            encoder_hidden,
            actions,
            states,
            extrinsics=None,
            num_steps=num_steps,
            t_ctx=t_ctx,
            window=window,
        )
        ours_new, ours_eh = _submodule_loop(
            submodule, encoder_hidden, actions, states, num_steps=num_steps,
        )

        assert len(ref_new) == len(ours_new) == num_steps
        for k, (r, o) in enumerate(zip(ref_new, ours_new, strict=True)):
            assert r.shape == o.shape == (b, window, cfg.hidden_size), f"iter {k}: shape"
            diff = (r - o).abs().max().item()
            assert diff == 0.0, f"iter {k}: predicted_hidden max abs diff = {diff}"
        for k, (r, o) in enumerate(zip(ref_eh, ours_eh, strict=True)):
            assert r.shape == o.shape == (b, n, cfg.hidden_size), f"iter {k}: next_encoder_hidden shape"
            diff = (r - o).abs().max().item()
            assert diff == 0.0, f"iter {k}: next_encoder_hidden max abs diff = {diff}"

    def test_sliding_window_invariant(self):
        """After each iter the encoder_hidden head is the prior tail; the
        encoder_hidden tail is the newly-predicted tubelet group.  This is
        the core of the sliding-window contract.
        """
        torch.manual_seed(1)
        cfg, _ac = _tiny_config()
        predictor = VisionTransformerPredictorAC(cfg.ac_predictor).eval()
        submodule = VJepa2ACRolloutPredictorSubmodule(predictor, cfg)

        b = 1
        t_ctx = cfg.grid_depth
        window = cfg.grid_size * cfg.grid_size
        n = t_ctx * window
        num_steps = 3
        t_total = t_ctx + num_steps - 1

        eh = torch.randn(b, n, cfg.hidden_size)
        actions = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)
        states = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)

        with torch.no_grad():
            for k in range(num_steps):
                info = _make_request_info(iter_idx=k, rollout_horizon=num_steps)
                out = submodule.forward(
                    info,
                    encoder_hidden=eh,
                    actions=actions,
                    states=states,
                )
                predicted = out["predicted_hidden"][0]
                next_eh = out["encoder_hidden"][0]
                assert predicted.shape == (b, window, cfg.hidden_size)
                assert next_eh.shape == (b, n, cfg.hidden_size)
                # Tail of next_eh is the new prediction.
                torch.testing.assert_close(next_eh[:, -window:, :], predicted)
                # Head of next_eh is the tail of the prior eh.
                torch.testing.assert_close(next_eh[:, : n - window, :], eh[:, window:, :])
                eh = next_eh

    def test_identity_loopback_actions_states(self):
        """The submodule passes actions/states through unchanged on every
        iter — identity loop-back is what lets the graph dispatcher keep
        routing them without the client resending them per iter.
        """
        torch.manual_seed(2)
        cfg, _ac = _tiny_config()
        predictor = VisionTransformerPredictorAC(cfg.ac_predictor).eval()
        submodule = VJepa2ACRolloutPredictorSubmodule(predictor, cfg)

        b = 1
        t_ctx = cfg.grid_depth
        window = cfg.grid_size * cfg.grid_size
        n = t_ctx * window
        t_total = t_ctx + 2

        eh = torch.randn(b, n, cfg.hidden_size)
        actions = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)
        states = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)

        with torch.no_grad():
            for k in range(3):
                info = _make_request_info(iter_idx=k, rollout_horizon=3)
                out = submodule.forward(
                    info,
                    encoder_hidden=eh,
                    actions=actions,
                    states=states,
                )
                # Identity loop-back: the returned tensors are the same
                # object (or at least bit-exactly equal) as what we passed in.
                torch.testing.assert_close(out["actions"][0], actions)
                torch.testing.assert_close(out["states"][0], states)
                eh = out["encoder_hidden"][0]


class TestACRolloutEarlyExit:
    def test_register_loop_stop_at_requested_horizon(self):
        """After iter == horizon - 1 the submodule registers a stop signal
        on the ``rollout_loop`` — same contract as masked rollout.
        """
        torch.manual_seed(3)
        cfg, _ac = _tiny_config()
        predictor = VisionTransformerPredictorAC(cfg.ac_predictor).eval()
        submodule = VJepa2ACRolloutPredictorSubmodule(predictor, cfg)

        b = 1
        t_ctx = cfg.grid_depth
        window = cfg.grid_size * cfg.grid_size
        n = t_ctx * window
        horizon = 3
        # Provide enough trajectory for horizon + 2 iters so the loop can
        # over-shoot and we observe the stop signal actually firing at
        # horizon - 1.
        t_total = t_ctx + horizon + 2

        eh = torch.randn(b, n, cfg.hidden_size)
        actions = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)
        states = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)

        stop_seen_at: list[int] = []
        with torch.no_grad():
            for k in range(horizon + 2):
                info = _make_request_info(iter_idx=k, rollout_horizon=horizon)
                out = submodule.forward(
                    info,
                    encoder_hidden=eh,
                    actions=actions,
                    states=states,
                )
                eh = out["encoder_hidden"][0]
                if "rollout_loop" in submodule.check_stop(info.request_id, info, out):
                    stop_seen_at.append(k)

        assert stop_seen_at, "submodule never registered a loop stop"
        assert stop_seen_at[0] == horizon - 1


class TestACRolloutTrajectoryTooShort:
    def test_raises_on_short_trajectory(self):
        """When ``actions/states`` length < iter_idx + T_ctx at some iter,
        the submodule raises a clear error.  This is a backstop: the model
        class validates trajectory length up-front in ``process_prompt``
        when ``rollout_horizon > 1``, but the submodule also guards so
        programmatic callers (e.g. unit tests) don't get a silent out-of-
        bounds slice.
        """
        torch.manual_seed(4)
        cfg, _ac = _tiny_config()
        predictor = VisionTransformerPredictorAC(cfg.ac_predictor).eval()
        submodule = VJepa2ACRolloutPredictorSubmodule(predictor, cfg)

        b = 1
        t_ctx = cfg.grid_depth
        window = cfg.grid_size * cfg.grid_size
        n = t_ctx * window
        # Only T_ctx entries — enough for iter 0, but iter 1 slices
        # [1 : 1 + T_ctx] which runs off the end.
        t_total = t_ctx

        eh = torch.randn(b, n, cfg.hidden_size)
        actions = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)
        states = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)

        # Iter 0 works (uses actions[0:T_ctx] = full trajectory).
        with torch.no_grad():
            info0 = _make_request_info(iter_idx=0, rollout_horizon=3)
            out0 = submodule.forward(
                info0, encoder_hidden=eh, actions=actions, states=states,
            )
            eh = out0["encoder_hidden"][0]

            info1 = _make_request_info(iter_idx=1, rollout_horizon=3)
            with pytest.raises(ValueError, match="trajectory too short"):
                submodule.forward(
                    info1, encoder_hidden=eh, actions=actions, states=states,
                )
