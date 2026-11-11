"""Image preprocessing and prompt token parity, without model weights or a GPU."""

import json
import os
from pathlib import Path

import pytest
import torch

from mstar.model.ministral3.components.vision import PixtralConfig
from mstar.model.ministral3.image import IMAGE_GRIDS, Ministral3ImageProcessor
from mstar.model.ministral3.ministral3_model import Ministral3Model
from mstar.model.multimodal import MEDIA_LAYOUT, PromptPart

FIXTURES = Path(__file__).parent / "fixtures" / "ministral3"


class ImageTokens:
    """Only the metadata constructor's token-validation interface."""

    def convert_tokens_to_ids(self, token):
        return {"[IMG]": 10, "[IMG_BREAK]": 12, "[IMG_END]": 13}[token]

    def encode(self, token, **kwargs):
        return [self.convert_tokens_to_ids(token)]


@pytest.fixture
def processor():
    pytest.importorskip("torchvision")
    config = PixtralConfig.from_hf(json.loads((FIXTURES / "config.json").read_text()))
    return Ministral3ImageProcessor(FIXTURES, config, ImageTokens())


def reference_image_processor():
    pytest.importorskip("torchvision")
    module = pytest.importorskip("transformers.models.pixtral.image_processing_pixtral")
    metadata = json.loads((FIXTURES / "processor_config.json").read_text())
    params = {key: value for key, value in metadata["image_processor"].items() if value is not None}
    return module.PixtralImageProcessor(**params)


def image(height, width):
    generator = torch.Generator().manual_seed(height * 10 + width)
    return torch.randint(0, 256, (3, height, width), generator=generator, dtype=torch.uint8)


@pytest.mark.parametrize("shape", [(28, 56), (31, 63), (77, 29), (1, 1), (1601, 41), (21, 1601)])
def test_resize_normalize_and_patch_order_match_reference(processor, shape):
    raw = image(*shape)
    reference = reference_image_processor()(raw, patch_size=28, return_tensors="pt",
                                            input_data_format="channels_first")
    expected = reference["pixel_values"][0]
    actual = processor.normalize(raw)
    assert list(actual.shape[-2:]) == list(reference["image_sizes"][0])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Data worker float [0,1] pixels reconstruct the same byte image.
    torch.testing.assert_close(processor.normalize(raw.float() / 255), expected, atol=0, rtol=0)
    prepared = processor.prepare(raw, {})
    _, h, w = prepared.grids[0]
    assert prepared.tokens == [h * w // 4]
    patches = prepared.inputs["pixel_values"]
    # Undo block ordering independently and reconstruct the full normalized image.
    raster = patches.reshape(h // 2, w // 2, 2, 2, 3, 14, 14)
    restored = raster.permute(4, 0, 2, 5, 1, 3, 6).reshape(3, h * 14, w * 14)
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)


@pytest.mark.parametrize("channels", [1, 4])
def test_gray_and_alpha_follow_loader_rgb_convention(processor, channels):
    raw = image(31, 63)
    altered = raw[:1] if channels == 1 else torch.cat((raw, raw[:1]))
    rgb = raw[:1].expand(3, -1, -1) if channels == 1 else raw
    torch.testing.assert_close(processor.normalize(altered), processor.normalize(rgb), atol=0, rtol=0)


@pytest.mark.parametrize("raw", [torch.empty(3, 0, 2), torch.empty(2, 10, 10), torch.ones(3, 4, 4) * 255,
                                torch.full((3, 4, 4), float("nan")), torch.zeros(3, 4, 4, dtype=torch.int64),
                                torch.zeros(3, 1, 100000, dtype=torch.uint8)])
def test_malformed_images_fail(processor, raw):
    with pytest.raises(ValueError):
        processor.prepare(raw, {})


@pytest.fixture(scope="module")
def checkpoint_model():
    directory = os.environ.get("MINISTRAL3_METADATA_DIR")
    if directory is None:
        pytest.skip("set MINISTRAL3_METADATA_DIR to cached checkpoint metadata; no test downloads")
    pytest.importorskip("torchvision")
    pytest.importorskip("transformers")
    return Ministral3Model(directory)


@pytest.mark.parametrize("parts,shapes", [
    ([PromptPart("image")], [(31, 63)]),
    ([PromptPart("text", "What is here?"), PromptPart("image")], [(31, 63)]),
    ([PromptPart("image"), PromptPart("text", "Compare: "), PromptPart("image", index=1),
      PromptPart("text", " café 中文")], [(77, 29), (31, 63)]),
    ([PromptPart("image", role="user"), PromptPart("text", "Describe", role="user"),
      PromptPart("text", "An image", role="assistant"), PromptPart("image", index=1, role="user"),
      PromptPart("text", "And this?", role="user")], [(31, 63), (77, 29)]),
])
def test_image_chat_ids_match_reference_processor(checkpoint_model, parts, shapes):
    from transformers.models.pixtral.processing_pixtral import PixtralProcessor

    model = checkpoint_model
    raw = [image(*shape) for shape in shapes]
    result = model.process_prompt("", [part.modality for part in parts], ["text"],
                                  tensors={"image_inputs": raw}, prompt_parts=parts)
    # Construct the reference conversation independently of M*'s message adapter.
    messages = []
    for part in parts:
        role = part.role or "user"
        if not messages or messages[-1]["role"] != role:
            messages.append({"role": role, "content": []})
        messages[-1]["content"].append({"type": "text", "text": part.text} if part.modality == "text"
                                      else {"type": "image"})
    rendered = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    reference = PixtralProcessor(reference_image_processor(), model.tokenizer,
                                 patch_size=14, spatial_merge_size=2)
    expected = reference(text=rendered, images=raw, return_tensors="pt", add_special_tokens=False,
                         input_data_format="channels_first")
    ids = result.new_input_tensors["text_inputs"][0]
    torch.testing.assert_close(ids, expected["input_ids"][0], atol=0, rtol=0)
    layout = result.metadata[MEDIA_LAYOUT]
    assert len(layout[IMAGE_GRIDS]) == len(raw)
    assert sum(layout["tokens"]) == int((ids == model.image_processor.image_id).sum())
    for offset, count in zip(layout["offset"], layout["tokens"], strict=True):
        assert (ids[offset:offset + count] == model.image_processor.image_id).all()
        assert ids[offset + count] in (model.image_processor.break_id, model.image_processor.end_id)
    # Dropping row markers must not pass token parity.
    without_delimiters = ids[(ids != model.image_processor.break_id) & (ids != model.image_processor.end_id)]
    assert not torch.equal(without_delimiters, expected["input_ids"][0])
