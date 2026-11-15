"""A BAGEL request with neither text nor an image is refused in ``process_prompt``,
a 400 before admission, as Orpheus's is. Without either, the prefill schedule is
empty and the request would open on a decode with no token; blank text would
generate from the template alone.
"""

from __future__ import annotations

import pytest
import torch
from test_model_kv_layouts import _bagel


def _model():
    model = _bagel()
    model.config.num_timesteps, model.config.timestep_shift = 3, 3.0  # a generation's guidance window
    return model


NO_TEXT = {"none": (None, []), "empty": ("", ["text"]), "blank": (" \n\t", ["text"])}


@pytest.mark.parametrize("output", ["text", "image"])
@pytest.mark.parametrize(("prompt", "inputs"), NO_TEXT.values(), ids=NO_TEXT.keys())
def test_no_text_and_no_image_is_refused(prompt, inputs, output):
    with pytest.raises(ValueError, match="BAGEL requires a text prompt or an image"):
        _model().process_prompt(prompt, inputs, [output], tensors={})


@pytest.mark.parametrize("output", ["text", "image"])
def test_an_image_alone_is_still_prefilled(output):
    """Understanding or editing an image with no text: the image is the prefill."""
    out = _model().process_prompt("", ["image"], [output], tensors={"image_inputs": [torch.rand(3, 64, 64)]})

    assert out.metadata["state_layout"]["kv"]["main"].spans


@pytest.mark.parametrize("output", ["text", "image"])
@pytest.mark.parametrize("inputs", [["image"], ["image", "text"]])
def test_a_declared_image_that_never_came_is_refused(inputs, output):
    with pytest.raises(ValueError, match="multimodal input mismatch"):
        _model().process_prompt(None, inputs, [output], tensors={"image_inputs": []})
