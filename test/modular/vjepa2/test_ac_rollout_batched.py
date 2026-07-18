"""Batched-execution contract tests for the AC rollout submodule.

``forward_batched`` runs one ``[B, N, D]`` forward for a batch of rollout
requests and must hand each request its own ``[1, N, D]`` row of the
result — the loop-back ``encoder_hidden`` it returns becomes that
request's next-iter input, so handing out the whole batched tensor (or an
empty list) corrupts every rollout after the first iteration.

Pure CPU, tiny config — no GPU or HF cache required.
"""

from __future__ import annotations

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC
from mstar.model.vjepa2.config import VJepa2ACPredictorConfig, VJepa2Config

try:
    from mstar.model.vjepa2.submodules import VJepa2ACRolloutPredictorSubmodule
except (ImportError, AttributeError) as e:  # pragma: no cover - env-specific
    pytest.skip(
        f"Cannot import VJepa2ACRolloutPredictorSubmodule in this env: {e}",
        allow_module_level=True,
    )


def _tiny_config() -> VJepa2Config:
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
    return VJepa2Config(
        patch_size=4,
        crop_size=16,
        frames_per_clip=4,
        tubelet_size=2,
        hidden_size=24,
        predictor_kind="ac",
        ac_predictor=ac_cfg,
    )


def _request_info(rid: str, iter_idx: int = 0) -> CurrentForwardPassInfo:
    info = CurrentForwardPassInfo(
        request_id=rid,
        graph_walk="prefill_video_rollout",
        requires_cfg=False,
        fwd_index=0,
        random_seed=0,
        max_tokens=0,
        sampling_config={},
    )
    info.dynamic_loop_iter_counts["rollout_loop"] = iter_idx
    info.step_metadata["rollout_horizon"] = 4
    return info


def _engine_inputs(rids: list[str]) -> ModelInputsFromEngine:
    return ModelInputsFromEngine(
        request_ids=rids,
        per_request_info={rid: _request_info(rid) for rid in rids},
        cache_manager=None,
        piecewise_runners={},
    )


def _make_submodule() -> tuple[VJepa2ACRolloutPredictorSubmodule, VJepa2Config]:
    torch.manual_seed(0)
    cfg = _tiny_config()
    predictor = VisionTransformerPredictorAC(cfg.ac_predictor).eval()
    return VJepa2ACRolloutPredictorSubmodule(predictor, cfg), cfg


class TestACRolloutForwardBatched:
    def test_splits_rows_per_request(self):
        """Each rid gets exactly its own [1, N, D] row of the batched
        rollout-step output, for both loop-back names."""
        submodule, cfg = _make_submodule()
        rids = ["req-a", "req-b", "req-c"]
        b = len(rids)
        window = cfg.grid_size * cfg.grid_size
        d = cfg.hidden_size

        step_out = torch.arange(b * window * d, dtype=torch.float32).reshape(
            b, window, d
        )
        submodule._rollout_step = lambda *args, **kwargs: step_out

        # One frame group per iter: T=1, so actions/states carry one timestep.
        n = window
        out = submodule.forward_batched(
            graph_walk="prefill_video_rollout",
            engine_inputs=_engine_inputs(rids),
            encoder_hidden=torch.randn(b, n, d),
            actions=torch.randn(b, 1, 7),
            states=torch.randn(b, 1, 7),
        )

        assert set(out.keys()) == set(rids)
        for i, rid in enumerate(rids):
            for name in ("encoder_hidden", "predicted_hidden"):
                tensors = out[rid][name]
                assert isinstance(tensors, list) and len(tensors) == 1, (
                    f"{rid}/{name}: expected a single-tensor list, got {tensors!r}"
                )
                assert tensors[0].shape == (1, window, d)
                assert torch.equal(tensors[0], step_out[i : i + 1])

    def test_pair_matches_singles(self):
        """A 2-request batch produces the same per-request outputs as two
        single-request batches over the same inputs."""
        submodule, cfg = _make_submodule()
        window = cfg.grid_size * cfg.grid_size
        n = window
        d = cfg.hidden_size

        torch.manual_seed(1)
        encoder_hidden = torch.randn(2, n, d)
        actions = torch.randn(2, 1, 7)
        states = torch.randn(2, 1, 7)

        with torch.no_grad():
            pair = submodule.forward_batched(
                graph_walk="prefill_video_rollout",
                engine_inputs=_engine_inputs(["r0", "r1"]),
                encoder_hidden=encoder_hidden,
                actions=actions,
                states=states,
            )
            singles = {
                rid: submodule.forward_batched(
                    graph_walk="prefill_video_rollout",
                    engine_inputs=_engine_inputs([rid]),
                    encoder_hidden=encoder_hidden[i : i + 1],
                    actions=actions[i : i + 1],
                    states=states[i : i + 1],
                )[rid]
                for i, rid in enumerate(["r0", "r1"])
            }

        for rid in ("r0", "r1"):
            for name in ("encoder_hidden", "predicted_hidden"):
                got = pair[rid][name][0]
                want = singles[rid][name][0]
                assert got.shape == want.shape == (1, window, d)
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
