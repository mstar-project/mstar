"""Bagel lays an understood image out the way its walks will write it.

The layout has to match the walks exactly: the order the prefill schedule runs
them in, and each image's own file over the slots its ViT walk writes, patches
after both resizes plus two sentinels. An image keyed by another's file is
served the other's KV.
"""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.bagel.submodules import ViTEncoderSubmodule
from mstar.model.base import ProcessPromptOutput
from mstar.model.multimodal import PromptPart
from mstar.worker.engine_manager import _refuse_unservable_walks

PATCH_SIZE = 14
MAX_PATCHES_PER_SIDE = 70
# two images of different sizes, so a span sized or keyed by the wrong one shows
SIZES = [(480, 640), (1200, 1600)]


class _StubTokenizer:
    """Enough of a tokenizer to render and scan: specials are single ids."""

    SPECIALS = {
        "<|im_start|>": 101, "<|im_end|>": 102, "<|vision_start|>": 103,
        "<|image_pad|>": 104, "<|vision_end|>": 105,
    }
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return self.SPECIALS.get(token, self.unk_token_id)

    def encode(self, text):
        pattern = "|".join(re.escape(t) for t in self.SPECIALS)
        ids = []
        for piece in re.split(f"({pattern})", text):
            ids.extend([self.SPECIALS[piece]] if piece in self.SPECIALS else [1000 + ord(c) for c in piece])
        return ids


class _StubBagel(BagelModel):
    """A BagelModel with only what process_prompt needs."""

    def __init__(self):
        self.config = SimpleNamespace(
            think_mode=False, vit_config=SimpleNamespace(patch_size=PATCH_SIZE),
            vit_max_num_patch_per_side=MAX_PATCHES_PER_SIDE,
        )
        self.tokenizer = _StubTokenizer()
        self.boi_token_id = _StubTokenizer.SPECIALS["<|vision_start|>"]
        self.eoi_token_id = _StubTokenizer.SPECIALS["<|vision_end|>"]


def _prompt(modalities: list[str], output_modalities=("text",), tensors=True, **kwargs):
    """Bagel's output for text and images in ``modalities``, and the images."""
    texts = iter(["what is", "in this picture", "and in this one"])
    parts, images = [], []
    for modality in modalities:
        if modality == "text":
            parts.append(PromptPart(modality="text", text=next(texts)))
        else:
            parts.append(PromptPart(modality="image", index=len(images)))
            images.append(torch.rand(3, *SIZES[len(images)]))
    output = _StubBagel().process_prompt(
        "\n".join(part.text for part in parts if part.text), modalities, list(output_modalities),
        tensors={"image_inputs": images} if tensors else None, prompt_parts=parts, **kwargs,
    )
    return output, images


def _vit_slots(image: torch.Tensor, image_preprocess: str) -> int:
    """What the ViT walk writes for ``image``: its patches, and the two sentinels."""
    vit = ViTEncoderSubmodule(None, None, None, PATCH_SIZE, MAX_PATCHES_PER_SIDE)
    return vit.prepare_inputs(
        "prefill_vit", SimpleNamespace(step_metadata={"image_preprocess": image_preprocess}),
        {"image_inputs": [image]},
    ).kwargs["max_seqlen"] + 2


PROMPTS = {
    "one image": (["text", "image", "text"], "default"),
    "two images": (["text", "image", "text", "image", "text"], "default"),
    "image first": (["image", "text", "image"], "default"),
    "vllm's square": (["text", "image", "text"], "vllm"),
}


@pytest.mark.parametrize(("modalities", "image_preprocess"), PROMPTS.values(), ids=PROMPTS.keys())
def test_each_image_is_laid_out_in_schedule_order_by_its_own_file_over_its_vits_slots(modalities, image_preprocess):
    output, images = _prompt(modalities, image_preprocess=image_preprocess)
    layout = output.metadata["prefix_layout"]["kv"]["main"]
    schedule = BagelModel._build_prefill_schedule(
        BagelModel.__new__(BagelModel), input_modalities=modalities,
        input_signals={"text_inputs": output.new_input_tensors["text_inputs"], "image_inputs": images},
        is_understanding=True,
    )

    spans = [span for span in layout if span.kind == "digest"]
    assert [span.walk for span in layout] == [walk for walk, _ in schedule], (
        "the layout names its writes in another order than the walks make them"
    )
    assert [span.source for span in spans] == [("image", index) for index in range(len(images))], (
        "an image is keyed by another's file, so its pages are served for a "
        "prompt that shows a different picture"
    )
    assert [span.length for span in spans] == [_vit_slots(image, image_preprocess) for image in images], (
        "an image is sized by other slots than its ViT walk writes, so the walks after it lose their reuse"
    )


def test_bagel_lays_out_only_the_understood_prompts_it_has_images_for():
    interleaved = ["text", "image", "text"]
    cases = {
        "understood with its image": (interleaved, ("text",), True),
        "text only": (["text"], ("text",), True),
        "generation": (interleaved, ("image",), True),
        "understood without tensors": (interleaved, ("text",), False),
        # text first, so understood, but an image out too, so its input is resized first
        "understood and edited": (interleaved, ("text", "image"), True),
    }

    laid_out = {name: isinstance(_prompt(*case)[0], ProcessPromptOutput) for name, case in cases.items()}

    assert laid_out == {name: name == "understood with its image" for name in cases}, (
        "a layout went to a prompt with no understood image or none of its "
        "tensors, or was missing from the one that has both"
    )


def test_bagels_walks_pass_the_load_check_under_cfg_parallelism():
    model = BagelModel.__new__(BagelModel)
    model.config = SimpleNamespace(
        num_hidden_layers=2, num_key_value_heads=2, hidden_size=64, num_attention_heads=4,
        max_position_embeddings=128, num_timesteps=4, rope_theta=10000.0, vocab_size=256,
        vit_config=SimpleNamespace(hidden_size=64, num_attention_heads=4, patch_size=PATCH_SIZE),
    )
    # the KV then spans LLM_cfg_text and LLM_cfg_img, which no prefill walk runs
    model._has_cfg_parallel = True
    model._image_gen_remote_handoff = False
    assert model.prefix_key_streams()["kv"]["main"].layout_walks == ("prefill_vit",), (
        "bagel names no layout walk, so the check below never looks at its walks"
    )

    _refuse_unservable_walks(model.get_node_resources(), model)
