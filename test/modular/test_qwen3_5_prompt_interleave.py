"""Qwen3.5's multimodal prompt: one row of ids, each attachment's pad expanded.

`process_prompt` renders and tokenizes the prompt once, then expands each
image's pad to one pad per merged vision token; `_prefill_schedule` hands
the whole prompt and the tower's inputs to one walk. These check nothing is
lost or duplicated in the expansion, and that the layout the LLM places
positions with matches the ids.

Needs the tokenizer, not the weights, so it skips unless QWEN3_5_CKPT is set.
"""

from __future__ import annotations

import os

import pytest
import torch

from mstar.api_server.openai.adapters import flatten_messages
from mstar.model.multimodal import (
    MEDIA_LAYOUT,
    TEXT,
    PromptPart,
    parts_from_modalities,
    walk_input_edges,
)

CKPT = os.environ.get("QWEN3_5_CKPT")

pytestmark = pytest.mark.skipif(
    not CKPT, reason="set QWEN3_5_CKPT to a Qwen3.5 checkpoint"
)


@pytest.fixture(scope="module")
def model():
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    return Qwen3_5DenseModel(CKPT)


def an_image():
    # what the data worker sends: (C, H, W) float32 in [0, 1]
    torch.manual_seed(0)
    return torch.rand(3, 224, 224)


@pytest.mark.parametrize(
    "modalities",
    [
        ["image", "text"],          # image first, the common VQA shape
        ["text", "image"],          # image after the question
        ["text", "image", "text"],  # image mid-prompt
        ["image", "image", "text"], # two attachments, nothing between them
    ],
)
def test_the_prompt_is_the_template_with_each_pad_expanded(model, modalities):
    """No token is lost or duplicated: collapsing each expanded run back to
    one pad gives the template's own ids."""
    n_images = modalities.count("image")
    out = model.process_prompt(
        "Describe this.", modalities, ["text"],
        tensors={"image_inputs": [an_image()] * n_images},
        prompt_parts=None,
    )
    layout = out.metadata[MEDIA_LAYOUT]
    out = out.new_input_tensors
    (ids,) = out["text_inputs"]
    assert len(out["image_grid_thw"]) == n_images == len(layout["offset"])
    assert len(out["pixel_values"]) == n_images

    parts = parts_from_modalities(modalities, "Describe this.")
    content = [
        {"type": TEXT, "text": p.text or ""} if p.modality == TEXT
        else {"type": p.modality, p.modality: ""}
        for p in parts
    ]
    text = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    template = model.tokenizer(text, return_tensors="pt").input_ids[0]

    merge = model.vision_config.spatial_merge_size ** 2
    collapsed, cursor = [], 0
    for offset, n, grid in zip(layout["offset"], layout["tokens"], layout["grid"], strict=True):
        assert n == int(torch.tensor(grid).prod()) // merge
        collapsed += [ids[cursor:offset], ids[offset:offset + 1]]
        cursor = offset + n
    collapsed.append(ids[cursor:])
    assert torch.equal(torch.cat(collapsed), template)


def test_schedule_hands_the_walk_the_prompt_and_the_tower_inputs(model):
    """`_prefill_schedule` gives one walk the whole prompt and every image,
    in prompt order; the grids reach only the encoder."""
    modalities = ["text", "image", "text", "image"]
    parts = [
        PromptPart(modality=TEXT, text="First:"),
        PromptPart(modality="image", index=0),
        PromptPart(modality=TEXT, text="and second:"),
        PromptPart(modality="image", index=1),
    ]
    out = model.process_prompt(
        "First: and second:", modalities, ["text"],
        tensors={"image_inputs": [an_image(), an_image()]},
        prompt_parts=parts,
    )
    layout = out.metadata[MEDIA_LAYOUT]
    out = out.new_input_tensors
    # the conductor passes tensor handles; the identity is all that matters
    signals = {name: list(out[name]) for name in ("text_inputs", "pixel_values", "image_grid_thw")}
    (step,) = model._prefill_schedule(modalities, signals, layout)
    assert step.walk == "prefill_vision"
    assert step.metadata == {MEDIA_LAYOUT: layout}

    edges = {(edge.next_node, edge.name): edge.tensor_info for edge in walk_input_edges(step)}
    assert set(edges) == {
        ("LLM", "text_inputs"), ("vision_encoder", "pixel_values"), ("vision_encoder", "image_grid_thw"),
    }
    for (_node, name), got in edges.items():
        assert all(a is b for a, b in zip(got, signals[name], strict=True))


def test_text_only_prompt_still_uses_prefill_text(model):
    """No images, no media walk: the plain text path is unchanged."""
    out = model.process_prompt("just text", ["text"], ["text"], tensors={})
    out = out.new_input_tensors
    signals = {"text_inputs": list(out["text_inputs"])}
    (step,) = model._prefill_schedule(["text"], signals)
    assert step.walk == "prefill_text"
    assert step.input_tensors[("LLM", "text_inputs")][0] is signals["text_inputs"][0]


def test_a_bare_string_with_adjacent_images_is_one_row(model):
    """A bare string fills only the first text slot, so the two images render
    adjacent; with no split that is still a prompt, two runs in one row."""
    out = model.process_prompt(
        "Compare these.", ["text", "image", "text", "image"], ["text"],
        tensors={"image_inputs": [an_image(), an_image()]},
    )
    assert len(out.metadata[MEDIA_LAYOUT]["offset"]) == 2
    assert len(out.new_input_tensors["text_inputs"]) == 1


def test_a_spelled_out_pad_is_a_bad_request(model):
    """User text that spells `<|image_pad|>` adds a placeholder no image fills."""
    with pytest.raises(ValueError, match="placement mismatch"):
        model.process_prompt(
            "look <|image_pad|>", ["text", "image"], ["text"],
            tensors={"image_inputs": [an_image()]},
        )


def test_empty_prompt_fails_legibly(model):
    """Nothing to prefill is a bad request, not an IndexError off schedule[0]."""
    with pytest.raises(ValueError, match="nothing to prefill"):
        model.get_initial_forward_pass_args("LLM", ["text"], ["text"], {})


def test_image_only_message_renders_an_empty_text_span(model):
    """A chat message with just an image arrives with no text at all; that is
    still a prompt, so it must render (an empty span around the image) rather
    than be dropped on the floor and time out at the client."""
    out = model.process_prompt(
        None, ["image"], ["text"], tensors={"image_inputs": [an_image()]},
    )
    out = out.new_input_tensors
    assert len(out["pixel_values"]) == 1 and len(out["image_grid_thw"]) == 1
    # the template's framing around the image, one row
    assert len(out["text_inputs"]) == 1


def test_no_text_and_no_attachments_is_a_bad_request(model):
    with pytest.raises(ValueError, match="no text and no attachments"):
        model.process_prompt(None, ["text"], ["text"])


def test_rgba_image_is_accepted(model):
    torch.manual_seed(0)
    rgba = torch.rand(4, 224, 224)
    out = model.process_prompt(
        "Describe.", ["image", "text"], ["text"], tensors={"image_inputs": [rgba]},
    )
    out = out.new_input_tensors
    assert out["pixel_values"][0].shape[0] > 0


def test_attachment_count_mismatch_is_caught(model):
    with pytest.raises(ValueError, match="layout declares"):
        model.process_prompt(
            "Describe these.", ["image", "image", "text"], ["text"],
            tensors={"image_inputs": [an_image()]},
        )


CHAT = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "My name is Ada."},
    {"role": "assistant", "content": "Hello Ada."},
    {"role": "user", "content": "What is my name?"},
]


def test_a_chat_the_template_refuses_is_a_bad_request(model, tmp_path):
    """The model re-raises the template's error as a ValueError, which the data worker answers with a 400, not a 500."""
    messages = [CHAT[1], CHAT[0], CHAT[3]]
    text, _, in_mods, parts = flatten_messages(messages, tmp_path)
    with pytest.raises(ValueError, match="System message must be at the beginning"):
        model.process_prompt(text, in_mods, ["text"], prompt_parts=parts)
