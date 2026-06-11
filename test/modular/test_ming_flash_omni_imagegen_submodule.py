"""Tests for ImageGenSubmodule (step 9b).

Pure-Python: a stub MingImagePipeline wrapped in ImageGenSubmodule — verifies
input marshalling (prepare_inputs slicing the thinker hidden states), the
stateless flavor, default sampling-param derivation from the ImageGenConfig,
and that forward routes through pipeline.generate and emits an ``image`` edge.
No diffusers, no checkpoint.
"""

from __future__ import annotations

import pytest
import torch

from mstar.model.ming_omni_flash.config import (
    AudioEncoderConfig,
    ImageGenConfig,
    MingFlashOmniModelConfig,
    ThinkerLLMConfig,
    VisionEncoderConfig,
)
from mstar.model.ming_omni_flash.submodules import ImageGenSubmodule


class _StubPipeline:
    """Records the args generate() was called with; returns a fixed image."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, thinker_hidden_states, params, *, negative_hidden=None, byte5_texts=None):
        self.calls.append(
            {
                "hidden": thinker_hidden_states,
                "params": params,
                "negative": negative_hidden,
            }
        )
        b = thinker_hidden_states.shape[0] if thinker_hidden_states.dim() == 3 else 1
        return torch.zeros(b, 3, 64, 64)


def _config(default_height=512, default_width=768, steps=7, cfg=3.5) -> MingFlashOmniModelConfig:
    ig = ImageGenConfig()
    ig.default_height = default_height
    ig.default_width = default_width
    ig.num_inference_steps = steps
    ig.guidance_scale = cfg
    return MingFlashOmniModelConfig(
        local_dir="",
        mlp_depth=2,
        thinker_llm=ThinkerLLMConfig(),
        vision=VisionEncoderConfig(),
        audio_encoder=AudioEncoderConfig(),
        image_gen=ig,
    )


def _submodule() -> tuple[ImageGenSubmodule, _StubPipeline]:
    pipe = _StubPipeline()
    sub = ImageGenSubmodule(pipeline=pipe, config=_config())
    return sub, pipe


def test_stateless_flavor_is_audio_codec() -> None:
    sub, _ = _submodule()
    assert sub.get_stateless_flavor() == "audio_codec"


def test_default_params_from_image_gen_config() -> None:
    sub, _ = _submodule()
    p = sub.default_params
    assert p.height == 512 and p.width == 768
    assert p.num_inference_steps == 7
    assert p.guidance_scale == 3.5


def test_prepare_inputs_pulls_hidden_states() -> None:
    sub, _ = _submodule()
    hidden = torch.randn(1, 256, 4096)
    out = sub.prepare_inputs(graph_walk="imagegen", fwd_info=None, inputs={"thinker_hidden_states": [hidden]})
    assert torch.equal(out.tensor_inputs["thinker_hidden_states"], hidden)
    assert out.tensor_inputs["negative_thinker_hidden_states"] is None


def test_prepare_inputs_passes_negative_when_present() -> None:
    sub, _ = _submodule()
    hidden = torch.randn(1, 16, 4096)
    neg = torch.randn(1, 16, 4096)
    out = sub.prepare_inputs(
        graph_walk="imagegen",
        fwd_info=None,
        inputs={"thinker_hidden_states": [hidden], "negative_thinker_hidden_states": [neg]},
    )
    assert torch.equal(out.tensor_inputs["negative_thinker_hidden_states"], neg)


def test_prepare_inputs_raises_on_missing_hidden() -> None:
    sub, _ = _submodule()
    with pytest.raises(ValueError, match="missing 'thinker_hidden_states'"):
        sub.prepare_inputs(graph_walk="imagegen", fwd_info=None, inputs={})


def test_forward_emits_image_via_pipeline() -> None:
    sub, pipe = _submodule()
    hidden = torch.randn(1, 16, 4096)
    out = sub.forward(graph_walk="imagegen", engine_inputs=None, thinker_hidden_states=hidden)
    assert "image" in out
    img = out["image"][0]
    assert img.shape == (1, 3, 64, 64)
    # The pipeline was driven with the default params and no negative.
    assert len(pipe.calls) == 1
    assert pipe.calls[0]["params"] is sub.default_params
    assert pipe.calls[0]["negative"] is None


def test_forward_forwards_negative_hidden() -> None:
    sub, pipe = _submodule()
    hidden = torch.randn(1, 16, 4096)
    neg = torch.randn(1, 16, 4096)
    sub.forward(
        graph_walk="imagegen",
        engine_inputs=None,
        thinker_hidden_states=hidden,
        negative_thinker_hidden_states=neg,
    )
    assert torch.equal(pipe.calls[0]["negative"], neg)
