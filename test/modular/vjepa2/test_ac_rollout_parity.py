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
import torch.nn.functional as F

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.submodule_base import ModelInputsFromEngine, NameToTensorList
from mstar.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC
from mstar.model.vjepa2.config import VJepa2ACPredictorConfig, VJepa2Config

from .fake_resources import bind_fake_resources

# submodules module pulls in mstar.engine at import, which on some local
# torch builds fails to set dynamo-config attributes that only exist in
# newer torch.  Skip rather than crash collection so the pure-component
# parity tests in this directory still run in either environment.
try:
    from mstar.model.vjepa2.submodules import VJepa2ACRolloutPredictorSubmodule
except (ImportError, AttributeError) as e:  # pragma: no cover - env-specific
    pytest.skip(
        f"Cannot import VJepa2ACRolloutPredictorSubmodule in this env: {e}",
        allow_module_level=True,
    )

_GRAPH_WALK = "prefill_video_rollout"


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
        request_id="r0",
        graph_walk="prefill_video_rollout",
        fwd_index=0,
        random_seed=0,
        max_tokens=0,
    )
    info.dynamic_loop_iter_counts["rollout_loop"] = iter_idx
    info.step_metadata["rollout_horizon"] = rollout_horizon
    return info


def _engine_inputs(info: CurrentForwardPassInfo) -> ModelInputsFromEngine:
    return ModelInputsFromEngine(
        request_ids=[info.request_id],
        per_request_info={info.request_id: info},
    )


def _reference_rollout(
    predictor: VisionTransformerPredictorAC,
    encoder_hidden: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    extrinsics: torch.Tensor | None,
    num_steps: int,
    tokens_per_step: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Hand-rolled growing-sequence AC rollout.

    Per iter k:
      * Take one timestep of actions/states: ``[..., k : k + 1, :]``.
      * Run the AC predictor under the ``"main"`` label with ``t_0=k``, so
        the step's K/V lands in the cache and the sequence grows.
      * Per-step LayerNorm, matching upstream's ``step_predictor`` body.
      * Feed the result straight back in — the context is the last
        prediction, nothing is concatenated or slid.

    Returns ``(per_iter_new_tg, per_iter_next_encoder_hidden)``; the two are
    the same tensor each iter, kept as separate lists so the caller can check
    both loop-back names.

    Bit-exact with the submodule's ``_rollout_step``: same math, and each
    side runs against its own freshly-bound cache. What this pins is the
    submodule's per-iter orchestration — the slicing, the norm, the
    loop-back wiring. The cache's own correctness is covered by
    test_ac_kv_cache_parity.py.
    """
    resources = bind_fake_resources(predictor)
    eh = encoder_hidden
    new_tgs: list[torch.Tensor] = []
    next_ehs: list[torch.Tensor] = []
    with torch.no_grad():
        for k in range(num_steps):
            acts_k = actions[:, k:k + 1].contiguous()
            sts_k = states[:, k:k + 1].contiguous()
            ext_k = extrinsics[:, k:k + 1].contiguous() if extrinsics is not None else None
            resources.plan([tokens_per_step])
            predicted = predictor(
                eh, acts_k, sts_k, extrinsics=ext_k, t_0=k, label="main",
            )
            new_tg = F.layer_norm(predicted, (predicted.size(-1),))
            eh = new_tg
            new_tgs.append(new_tg)
            next_ehs.append(eh)
    return new_tgs, next_ehs


def _step(
    submodule: VJepa2ACRolloutPredictorSubmodule,
    info: CurrentForwardPassInfo,
    encoder_hidden: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    resources=None,
) -> NameToTensorList:
    """One engine-shaped rollout step.

    Goes through ``prepare_inputs`` → ``preprocess`` → ``forward`` rather than
    calling ``forward`` directly: per-iter slicing of the constant
    actions/states buffers lives in ``prepare_inputs``, so a direct forward
    would hand the predictor the whole trajectory.

    ``resources`` carries the KV/attention history across steps. Pass one to
    keep the cache growing over a rollout; leave it None for a fresh cache.
    """
    node_inputs = submodule.prepare_inputs(
        graph_walk=_GRAPH_WALK,
        fwd_info=info,
        inputs={
            "encoder_hidden": [encoder_hidden],
            "actions": [actions],
            "states": [states],
        },
    )
    packed = submodule.preprocess(
        graph_walk=_GRAPH_WALK,
        engine_inputs=_engine_inputs(info),
        inputs=[node_inputs],
    )
    # The rollout step attends under a plan label, so the blocks need KV +
    # attention resources bound before the forward.
    cfg = submodule.config
    cond_tokens = 3 if cfg.ac_predictor.use_extrinsics else 2
    tokens_per_req = packed["encoder_hidden"].size(1) + cond_tokens
    if resources is None:
        resources = bind_fake_resources(submodule.predictor)
    resources.plan([tokens_per_req])
    return submodule.forward(
        _GRAPH_WALK, _engine_inputs(info), **packed
    )


def _submodule_loop(
    submodule: VJepa2ACRolloutPredictorSubmodule,
    encoder_hidden: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    num_steps: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Drive the submodule H times, threading ``encoder_hidden`` via
    loop-back just like the mstar DynamicLoop does.

    Resources are bound once for the whole rollout so the KV cache grows
    across iters, as it does in production; rebinding per step would reset
    the history and make every iter look like a fresh step-0.
    """
    resources = bind_fake_resources(submodule.predictor)
    eh = encoder_hidden
    new_tgs: list[torch.Tensor] = []
    next_ehs: list[torch.Tensor] = []
    with torch.no_grad():
        for k in range(num_steps):
            info = _make_request_info(iter_idx=k, rollout_horizon=num_steps)
            out = _step(submodule, info, eh, actions, states, resources=resources)
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
        the hand-rolled growing-sequence reference.
        """
        torch.manual_seed(0)
        cfg, _ac = _tiny_config()
        predictor = VisionTransformerPredictorAC(cfg.ac_predictor).eval()

        b = 1
        window = cfg.grid_size * cfg.grid_size  # 16
        n = window  # context is one frame group

        num_steps = 3
        t_total = num_steps
        cond_tokens = 3 if cfg.ac_predictor.use_extrinsics else 2
        tokens_per_step = n + cond_tokens

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
            tokens_per_step=tokens_per_step,
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

    def test_growing_cache_invariant(self):
        """History lives in the KV cache, not in the loop-back tensor.

        The rollout does not slide a multi-frame window any more: each iter's
        context is exactly the previous iter's prediction — one frame group,
        fixed shape — and everything older is reachable only because the KV
        cache keeps growing by one step's tokens per iter.

        So the loop-back tensor carries no history of its own (there is no
        "head is the prior tail" relationship to assert), and the thing that
        has to hold instead is that the cache accumulates every step.
        """
        torch.manual_seed(1)
        cfg, _ac = _tiny_config()
        predictor = VisionTransformerPredictorAC(cfg.ac_predictor).eval()
        submodule = VJepa2ACRolloutPredictorSubmodule(predictor, cfg)

        b = 1
        window = cfg.grid_size * cfg.grid_size
        n = window  # context is one frame group
        num_steps = 3
        t_total = num_steps

        eh = torch.randn(b, n, cfg.hidden_size)
        actions = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)
        states = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)

        cond_tokens = 3 if cfg.ac_predictor.use_extrinsics else 2
        tokens_per_step = n + cond_tokens
        # Bind once so the cache accumulates across iters, as in a real rollout.
        resources = bind_fake_resources(submodule.predictor)

        with torch.no_grad():
            for k in range(num_steps):
                info = _make_request_info(iter_idx=k, rollout_horizon=num_steps)
                out = _step(submodule, info, eh, actions, states, resources=resources)
                predicted = out["predicted_hidden"][0]
                next_eh = out["encoder_hidden"][0]

                # Fixed-size context every iter — no growth in the tensor.
                assert predicted.shape == (b, window, cfg.hidden_size)
                assert next_eh.shape == (b, n, cfg.hidden_size)
                # The loop-back IS the new prediction, in full.
                torch.testing.assert_close(next_eh, predicted)

                # The cache is what grew: k+1 steps' worth of tokens per layer.
                for layer_kvs in resources._kv.values():
                    cached = torch.cat(layer_kvs[0][0], dim=0)
                    assert cached.shape[0] == (k + 1) * tokens_per_step

                eh = next_eh

    def test_prepare_inputs_slices_the_iters_timestep(self):
        """Per-iter slicing of the constant actions/states buffers.

        The graph deliberately has no actions/states loop-back edges (see the
        NOTE in ``VJepa2ACModel.get_worker_graphs``): the loop primitives
        re-pass the client's original buffers every iter, and
        ``prepare_inputs`` picks out timestep ``iter_idx``. That indexing is
        what makes the constant buffers behave like a per-iter stream.
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

        for k in range(3):
            info = _make_request_info(iter_idx=k, rollout_horizon=3)
            node_inputs = submodule.prepare_inputs(
                graph_walk=_GRAPH_WALK,
                fwd_info=info,
                inputs={
                    "encoder_hidden": [eh],
                    "actions": [actions],
                    "states": [states],
                },
            )
            # Exactly one timestep, and it is iter k's.
            torch.testing.assert_close(
                node_inputs.tensor_inputs["actions"], actions[:, k : k + 1]
            )
            torch.testing.assert_close(
                node_inputs.tensor_inputs["states"], states[:, k : k + 1]
            )


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
        # One frame group of context per iter — the rollout's window is
        # [B, H*W, D], one timestep of actions/states per step.
        n = cfg.grid_size * cfg.grid_size
        horizon = 3
        # Provide enough trajectory for horizon + 2 iters so the loop can
        # over-shoot and we observe the stop signal actually firing at
        # horizon - 1.
        t_total = horizon + 2

        eh = torch.randn(b, n, cfg.hidden_size)
        actions = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)
        states = torch.randn(b, t_total, cfg.ac_predictor.action_embed_dim)

        stop_seen_at: list[int] = []
        with torch.no_grad():
            for k in range(horizon + 2):
                info = _make_request_info(iter_idx=k, rollout_horizon=horizon)
                out = _step(submodule, info, eh, actions, states)
                eh = out["encoder_hidden"][0]
                if "rollout_loop" in submodule.check_stop(info.request_id, info, out):
                    stop_seen_at.append(k)

        assert stop_seen_at, "submodule never registered a loop stop"
        assert stop_seen_at[0] == horizon - 1

