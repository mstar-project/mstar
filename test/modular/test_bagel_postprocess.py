"""BAGEL's image response: the shared PNG writer must decode to the pixels the old PIL path
sent, and a format it cannot write fails at intake."""
import io

import numpy as np
import pytest
import torch

from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.bagel.config import BagelAutoEncoderConfig, BagelModelConfig, BagelViTConfig
from mstar.model.bagel.submodules import IMAGE_SENTINELS

PIL = pytest.importorskip("PIL.Image")


def _model() -> BagelModel:
    model = BagelModel.__new__(BagelModel)
    model.config = BagelModelConfig(vae_config=BagelAutoEncoderConfig(), vit_config=BagelViTConfig())
    return model


def test_image_png_decodes_to_the_truncated_pixels():
    g = torch.Generator().manual_seed(0)
    output = torch.rand(1, 3, 48, 64, generator=g)
    model = BagelModel.__new__(BagelModel)
    png = model.postprocess(output, "image", {})
    decoded = np.asarray(PIL.open(io.BytesIO(png)).convert("RGB"))
    # the old path: permute to HWC, scale, truncate to uint8
    expected = (output[0].permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    assert np.array_equal(decoded, expected)


def test_unsupported_output_format_fails_at_intake():
    model = _model()
    with pytest.raises(ValueError, match="output_format"):
        model.process_prompt(None, [], ["image"], output_format="gif")


@pytest.mark.parametrize("size", [{"width": 8}, {"height": 15}, {"width": 0}, {"height": True}, {"width": "512"}])
def test_an_image_under_one_latent_patch_fails_at_intake(size):
    """Under 16 pixels a side, the image is no latent patch at all, and the
    request would otherwise fail deep in the denoise loop."""
    model = _model()
    with pytest.raises(ValueError, match="width|height|16 pixels"):
        model.process_prompt(None, [], ["image"], **size)
    model.process_prompt(None, [], ["image"], width=16, height=512)


@pytest.mark.parametrize("hw, size", [((50, 5000), {}), ((512, 512), {"width": 8})])
def test_an_edit_whose_output_is_under_one_latent_patch_fails_at_intake(hw, size):
    """An edit's output takes its input's size after the fit to 1024 (and to any
    width/height), so an aspect the cap squeezes under 16 pixels (5000x50 ->
    1024x10) is refused like a tiny requested size."""
    model = _model()
    with pytest.raises(ValueError, match="16 pixels"):
        model.process_prompt(None, ["image"], ["image"], tensors={"image_inputs": [torch.zeros(3, *hw)]}, **size)


def test_a_null_size_is_the_default_size():
    """An explicit null width/height, as a client's extra_body can send, is the
    default 1024x1024 a generation gets when it names no size."""
    output = _model().process_prompt(None, [], ["image"], tensors={}, width=None, height=None)
    assert output.metadata["kv_layout"]["kv"]["main"].scratch == (1024 // 16) ** 2 + IMAGE_SENTINELS


def test_a_null_sampling_knob_is_the_default():
    """An explicit null top_k, as a client's extra_body can send, samples with the config's top_k."""
    model = _model()
    assert model.get_sampling_config("LLM", {"top_k": None}).top_k == model.config.top_k
