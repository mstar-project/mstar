"""P3.A unit tests for cross-request batching on V-JEPA 2 submodules.

Three things this file exercises, all on CPU with a tiny config so the test
is cheap and environment-independent:

1. ``can_batch(batch)`` returns True iff all requests' inputs have matching
   shapes; False on any shape mismatch or missing field.
2. ``preprocess`` + ``forward_batched`` together produce per-rid outputs
   whose values bit-match the sequential ``forward`` path request-by-request.
   This is the critical correctness invariant — a serving system that
   quietly diverges under batching would be terrifying.
3. ``EncoderDecoderEngine._execute_batched`` routes to ``forward_batched``
   when present and returns a ``NodeOutput`` whose per-rid slots look
   identical to what ``_execute_sequential`` would have produced.

Covers both ``VJepa2EncoderSubmodule`` (shape-only batch) and
``VJepa2PredictorSubmodule`` (shape + mask agreement).
"""

from __future__ import annotations

import pytest
import torch

from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.model.vjepa2.components.predictor import VJEPA2Predictor
from mminf.model.vjepa2.components.vit_encoder import VJEPA2Encoder
from mminf.model.vjepa2.config import VJepa2Config

# ``mminf.engine`` / ``mminf.model.vjepa2.submodules`` hit
# ``torch._dynamo.config.recompile_limit`` at import, which doesn't exist on
# older torch builds.  Match the pattern in ``test_rollout_parity.py`` and
# skip the module if any of these imports fail in this env.
try:
    from mminf.engine.base import NodeBatch
    from mminf.engine.enc_dec_engine import EncoderDecoderEngine
    from mminf.model.vjepa2.submodules import (
        VJepa2EncoderSubmodule,
        VJepa2PredictorSubmodule,
    )
except (ImportError, AttributeError) as e:  # pragma: no cover - env-specific
    pytest.skip(
        f"Cannot import V-JEPA 2 submodules / ENC_DEC engine in this env: {e}",
        allow_module_level=True,
    )


def _tiny_config() -> VJepa2Config:
    """Tiny shape so CPU-only forward is fast.  ``num_patches`` comes out
    to ``4*4*2 = 32`` under these settings — enough for RoPE + attention
    to exercise but small enough to run in milliseconds.
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


def _make_info(graph_walk: str = "prefill_video") -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        graph_walk=graph_walk,
        requires_cfg=False,
        fwd_index=0,
        random_seed=0,
        max_tokens=0,
    )


def _make_batch(
    node_name: str,
    graph_walk: str,
    per_request_inputs: dict[str, dict],
) -> NodeBatch:
    rids = list(per_request_inputs.keys())
    return NodeBatch(
        node_name=node_name,
        graph_walk=graph_walk,
        request_ids=rids,
        per_request_input_tensors=per_request_inputs,
        per_request_info={rid: _make_info(graph_walk) for rid in rids},
    )


class TestEncoderCanBatch:
    def test_homogeneous_shapes_can_batch(self):
        config = _tiny_config()
        encoder = VJEPA2Encoder(config).eval()
        submodule = VJepa2EncoderSubmodule(encoder, config)

        frames = torch.randn(config.frames_per_clip, 3, config.crop_size, config.crop_size)
        batch = _make_batch(
            node_name="video_encoder",
            graph_walk="prefill_video",
            per_request_inputs={
                "rid_0": {"video_frames": [frames]},
                "rid_1": {"video_frames": [frames.clone()]},
            },
        )
        assert submodule.can_batch(batch) is True

    def test_heterogeneous_shapes_cannot_batch(self):
        config = _tiny_config()
        encoder = VJEPA2Encoder(config).eval()
        submodule = VJepa2EncoderSubmodule(encoder, config)

        frames_a = torch.randn(config.frames_per_clip, 3, config.crop_size, config.crop_size)
        # Different crop_size → can_batch must return False so the engine
        # falls through to sequential per-request execution.
        frames_b = torch.randn(config.frames_per_clip, 3, config.crop_size * 2, config.crop_size)
        batch = _make_batch(
            node_name="video_encoder",
            graph_walk="prefill_video",
            per_request_inputs={
                "rid_0": {"video_frames": [frames_a]},
                "rid_1": {"video_frames": [frames_b]},
            },
        )
        assert submodule.can_batch(batch) is False

    def test_missing_frames_cannot_batch(self):
        config = _tiny_config()
        encoder = VJEPA2Encoder(config).eval()
        submodule = VJepa2EncoderSubmodule(encoder, config)

        batch = _make_batch(
            node_name="video_encoder",
            graph_walk="prefill_video",
            per_request_inputs={"rid_0": {}, "rid_1": {}},
        )
        assert submodule.can_batch(batch) is False

    def test_b1_rejected_routes_to_sequential_path(self):
        """``can_batch`` returns False for B=1 regardless of shape
        homogeneity, so single-request traffic keeps using the proven
        sequential ``forward`` path.

        Rationale: on vjepa2-ac-vitg (40 layers × 1408 hidden dim), a
        torch.compile trace through ``forward_batched`` measured ~20×
        slower at warm than the same input through ``forward``.  Routing
        B=1 through forward_batched would strictly regress single-request
        latency from the Phase 2 baseline.  forward_batched only pays off
        when the scheduler genuinely co-batches concurrent requests, so
        we gate can_batch on ``len(request_ids) >= 2``.
        """
        config = _tiny_config()
        encoder = VJEPA2Encoder(config).eval()
        submodule = VJepa2EncoderSubmodule(encoder, config)

        frames = torch.randn(config.frames_per_clip, 3, config.crop_size, config.crop_size)
        batch = _make_batch(
            node_name="video_encoder",
            graph_walk="prefill_video",
            per_request_inputs={"rid_0": {"video_frames": [frames]}},
        )
        assert submodule.can_batch(batch) is False


class TestEncoderBatchedParity:
    """Bit-parity: forward_batched on B=2 must match two separate B=1
    forward calls. If this ever regresses, cross-request batching is
    producing numerically different outputs from sequential — a silent
    correctness failure.
    """

    def _run_sequential(self, submodule, frames_list):
        """Run the submodule one-rid-at-a-time via the sequential path."""
        outputs = []
        for frames in frames_list:
            packed = submodule.preprocess(
                graph_walk="prefill_video",
                per_request_inputs=[{"video_frames": [frames]}],
                request_ids=["rid"],
                per_request_info={"rid": _make_info()},
            )
            out = submodule(request_info=_make_info(), **packed)
            outputs.append(out["encoder_hidden"][0])
        return outputs

    def _run_batched(self, submodule, frames_list):
        rids = [f"rid_{i}" for i in range(len(frames_list))]
        per_request_inputs = [
            {"video_frames": [f]} for f in frames_list
        ]
        packed = submodule.preprocess(
            graph_walk="prefill_video",
            per_request_inputs=per_request_inputs,
            request_ids=rids,
            per_request_info={rid: _make_info() for rid in rids},
        )
        per_rid = submodule.forward_batched(
            graph_walk="prefill_video",
            request_ids=rids,
            packed_inputs=packed,
            per_request_info={rid: _make_info() for rid in rids},
        )
        return [per_rid[rid]["encoder_hidden"][0] for rid in rids]

    def test_encoder_batched_matches_sequential(self):
        torch.manual_seed(0)
        config = _tiny_config()
        encoder = VJEPA2Encoder(config).eval()
        submodule = VJepa2EncoderSubmodule(encoder, config)

        frames = [
            torch.randn(config.frames_per_clip, 3, config.crop_size, config.crop_size),
            torch.randn(config.frames_per_clip, 3, config.crop_size, config.crop_size),
        ]
        with torch.no_grad():
            seq_outs = self._run_sequential(submodule, frames)
            bat_outs = self._run_batched(submodule, frames)

        assert len(seq_outs) == len(bat_outs) == 2
        for i, (s, b) in enumerate(zip(seq_outs, bat_outs, strict=True)):
            assert s.shape == b.shape, f"rid {i}: shape {s.shape} vs {b.shape}"
            # Eager CPU path → exact match expected.
            diff = (s - b).abs().max().item()
            assert diff < 1e-6, f"rid {i}: max abs diff {diff}"

    def test_preprocess_stacks_dim0(self):
        config = _tiny_config()
        encoder = VJEPA2Encoder(config).eval()
        submodule = VJepa2EncoderSubmodule(encoder, config)

        frames = torch.randn(config.frames_per_clip, 3, config.crop_size, config.crop_size)
        packed = submodule.preprocess(
            graph_walk="prefill_video",
            per_request_inputs=[
                {"video_frames": [frames]},
                {"video_frames": [frames.clone()]},
            ],
            request_ids=["rid_0", "rid_1"],
            per_request_info={"rid_0": _make_info(), "rid_1": _make_info()},
        )
        assert packed["pixel_values_videos"].shape == (
            2, config.frames_per_clip, 3, config.crop_size, config.crop_size,
        )


class TestPredictorBatchedParity:
    def test_default_masks_batched_matches_sequential(self):
        torch.manual_seed(1)
        config = _tiny_config()
        predictor = VJEPA2Predictor(config).eval()
        submodule = VJepa2PredictorSubmodule(predictor, config)

        # Encoder-output-shaped hiddens.
        enc_a = torch.randn(1, config.num_patches, config.hidden_size)
        enc_b = torch.randn(1, config.num_patches, config.hidden_size)

        with torch.no_grad():
            # Sequential path, one rid at a time.
            seq_outs = []
            for enc in (enc_a, enc_b):
                packed = submodule.preprocess(
                    graph_walk="prefill_video",
                    per_request_inputs=[{"encoder_hidden": [enc]}],
                    request_ids=["rid"],
                    per_request_info={"rid": _make_info()},
                )
                out = submodule(request_info=_make_info(), **packed)
                seq_outs.append(out["predicted_hidden"][0])

            # Batched path, both rids at once.
            packed = submodule.preprocess(
                graph_walk="prefill_video",
                per_request_inputs=[
                    {"encoder_hidden": [enc_a]},
                    {"encoder_hidden": [enc_b]},
                ],
                request_ids=["rid_0", "rid_1"],
                per_request_info={"rid_0": _make_info(), "rid_1": _make_info()},
            )
            per_rid = submodule.forward_batched(
                graph_walk="prefill_video",
                request_ids=["rid_0", "rid_1"],
                packed_inputs=packed,
                per_request_info={"rid_0": _make_info(), "rid_1": _make_info()},
            )
            bat_outs = [per_rid["rid_0"]["predicted_hidden"][0], per_rid["rid_1"]["predicted_hidden"][0]]

        for i, (s, b) in enumerate(zip(seq_outs, bat_outs, strict=True)):
            assert s.shape == b.shape, f"rid {i}: shape {s.shape} vs {b.shape}"
            diff = (s - b).abs().max().item()
            assert diff < 1e-5, f"rid {i}: max abs diff {diff}"

    def test_can_batch_rejects_shape_mismatch(self):
        config = _tiny_config()
        predictor = VJEPA2Predictor(config).eval()
        submodule = VJepa2PredictorSubmodule(predictor, config)

        enc_a = torch.randn(1, config.num_patches, config.hidden_size)
        # Different N → different shape → cannot batch.
        enc_b = torch.randn(1, config.num_patches // 2, config.hidden_size)

        batch = _make_batch(
            node_name="predictor",
            graph_walk="prefill_video",
            per_request_inputs={
                "rid_0": {"encoder_hidden": [enc_a]},
                "rid_1": {"encoder_hidden": [enc_b]},
            },
        )
        assert submodule.can_batch(batch) is False


class TestEngineRouting:
    """Covers the engine-side plumbing at
    ``EncoderDecoderEngine._execute_batched`` — checks the new signatures
    reach the submodule correctly and that per-rid outputs flow back
    without the pre-P3.A dict-as-list bug re-appearing.
    """

    def test_execute_batched_routes_to_forward_batched(self):
        torch.manual_seed(0)
        config = _tiny_config()
        encoder = VJEPA2Encoder(config).eval()
        submodule = VJepa2EncoderSubmodule(encoder, config)

        frames = torch.randn(config.frames_per_clip, 3, config.crop_size, config.crop_size)
        batch = _make_batch(
            node_name="video_encoder",
            graph_walk="prefill_video",
            per_request_inputs={
                "rid_0": {"video_frames": [frames]},
                "rid_1": {"video_frames": [frames.clone()]},
            },
        )

        engine = EncoderDecoderEngine()
        engine.load_model({"video_encoder": submodule}, device=torch.device("cpu"))

        assert submodule.can_batch(batch) is True
        with torch.no_grad():
            output = engine._execute_batched(batch, submodule)

        assert set(output.per_request_output_tensors.keys()) == {"rid_0", "rid_1"}
        for rid in ("rid_0", "rid_1"):
            slot = output.per_request_output_tensors[rid]
            assert "encoder_hidden" in slot
            enc = slot["encoder_hidden"][0]
            assert enc.shape == (1, config.num_patches, config.hidden_size)

    def test_execute_batched_raises_when_forward_batched_missing(self):
        """Engine must reject submodules that opt into batching without
        implementing ``forward_batched`` — else the caller would silently
        fall back to the pre-P3.A buggy split path."""

        class BrokenSubmodule:
            # No forward_batched; can_batch says yes regardless.
            def can_batch(self, _batch):
                return True

            def preprocess(self, **_kw):
                return {}

        engine = EncoderDecoderEngine()
        engine.load_model({"x": BrokenSubmodule()}, device=torch.device("cpu"))

        batch = _make_batch(
            node_name="x",
            graph_walk="any",
            per_request_inputs={"rid_0": {}, "rid_1": {}},
        )
        with pytest.raises(RuntimeError, match="forward_batched"):
            engine._execute_batched(batch, engine.submodules["x"])
