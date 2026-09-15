"""Qwen3.5's multimodal prompt: split at the attachments, scheduled in order.

The contract `process_prompt` and `_prefill_schedule` share is that the nth
text segment belongs to the nth text step of the prefill plan. These check the
two halves agree, and that nothing is lost or duplicated at a seam — the
segments plus the spans they were split around must rebuild the prompt
verbatim.

Needs the tokenizer, not the weights, so it skips unless QWEN3_5_CKPT is set.
"""

from __future__ import annotations

import os

import pytest
import torch

from mstar.model.qwen3_5.qwen3_5_model import IMAGE_PART, TEXT_PART
from mstar.model.multimodal import (
    TEXT,
    PromptPart,
    parts_from_modalities,
    prefill_plan,
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


def test_text_only_is_one_segment(model):
    out = model.process_prompt("What is 2 + 2?", ["text"], ["text"])
    assert list(out) == ["text_inputs"]
    assert len(out["text_inputs"]) == 1
    assert out["text_inputs"][0].dim() == 1


@pytest.mark.parametrize(
    "modalities",
    [
        ["image", "text"],          # image first, the common VQA shape
        ["text", "image"],          # image after the question
        ["text", "image", "text"],  # image mid-prompt
        ["image", "image", "text"], # two attachments, nothing between them
    ],
)
def test_segments_rebuild_the_prompt(model, modalities):
    """No token is lost or duplicated at a split seam."""
    n_images = modalities.count("image")
    out = model.process_prompt(
        "Describe this.", modalities, ["text"],
        tensors={"image_inputs": [an_image()] * n_images},
        prompt_parts=None,
    )
    segments = out["text_inputs"]
    assert len(out["image_grid_thw"]) == n_images
    assert len(out["pixel_values"]) == n_images

    # Re-render the same prompt to get the ids the split came from.
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
    ids = model.tokenizer(text, return_tensors="pt").input_ids[0]

    start, pad, end = model._placeholder_specs()["image"]
    # segments interleave with the spans they were split around, and each span
    # is exactly <|vision_start|><|image_pad|><|vision_end|> as the template
    # writes it — one pad, not one per token
    span = torch.tensor([start, pad, end])
    plan = prefill_plan(parts)
    rebuilt, texts = [], iter(segments)
    for step in plan:
        rebuilt.append(next(texts) if step.modality == TEXT else span)
    assert torch.equal(torch.cat(rebuilt), ids)

    # the sentinels belong to the vision walk, so no segment may carry one
    for seg in segments:
        assert not torch.isin(seg, span).any()


def test_schedule_matches_the_split(model):
    """`_prefill_schedule` consumes the segments `process_prompt` produced, in
    the order the prompt wrote them."""
    modalities = ["text", "image", "text", "image"]
    # A layout with more than one text slot needs `prompt_parts` to fill them;
    # see `test_multi_text_layout_needs_prompt_parts`.
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
    # the conductor passes tensor handles; the identity is all that matters
    signals = {
        "text_inputs": list(out["text_inputs"]),
        "pixel_values": list(out["pixel_values"]),
        "image_grid_thw": list(out["image_grid_thw"]),
    }
    schedule = model._prefill_schedule(modalities, signals)

    # An image prompt is one walk, whatever its layout: the interleaving
    # travels as `order` rather than as a walk per span.
    assert [step.walk for step in schedule] == ["prefill_vision"]
    step = schedule[0]

    planned = [
        TEXT_PART if s.modality == TEXT else IMAGE_PART
        for s in prefill_plan(parts_from_modalities(modalities))
    ]
    assert list(step.order) == planned

    # the nth tag of a kind takes the nth tensor of that kind, in order
    for name in ("text_inputs", "pixel_values", "image_grid_thw"):
        got = step.input_tensors[name]
        assert len(got) == len(signals[name])
        for a, b in zip(got, signals[name], strict=True):
            assert a is b

    # the grid has to reach both halves of the walk: the encoder lays out its
    # patches with it, the LLM places the 3D positions with it. The LLM also
    # reads the text spans now.
    edges = type(model)._walk_inputs(step)
    nodes = {(edge.next_node, edge.name) for edge in edges}
    assert ("vision_encoder", "pixel_values") in nodes
    assert ("vision_encoder", "image_grid_thw") in nodes
    assert ("LLM", "image_grid_thw") in nodes
    assert ("LLM", "text_inputs") in nodes
    # and each edge hands over every tensor of its name, not just the first
    by_name = {edge.name: edge for edge in edges if edge.next_node == "LLM"}
    assert len(by_name["text_inputs"].tensor_info) == len(signals["text_inputs"])


def test_text_only_prompt_still_uses_prefill_text(model):
    """No images, no merged walk: the plain text path is unchanged."""
    out = model.process_prompt("just text", ["text"], ["text"], tensors={})
    signals = {"text_inputs": list(out["text_inputs"])}
    schedule = model._prefill_schedule(["text"], signals)

    assert [s.walk for s in schedule] == ["prefill_text"] * len(schedule)
    got = [s.input_tensors["text_inputs"][0] for s in schedule]
    for a, b in zip(got, signals["text_inputs"], strict=True):
        assert a is b


def test_multi_text_layout_needs_prompt_parts(model):
    """A bare string fills only the first text slot.

    The rest render empty, so the two images end up adjacent and the split
    drops a piece the plan still counts. `check_plan` is what catches it —
    the alternative is a silent off-by-one between segments and walks.
    """
    with pytest.raises(ValueError, match="placement mismatch"):
        model.process_prompt(
            "Compare these.", ["text", "image", "text", "image"], ["text"],
            tensors={"image_inputs": [an_image(), an_image()]},
        )


def test_empty_prompt_fails_legibly(model):
    """Nothing to prefill is a bad request, not an IndexError off schedule[0]."""
    with pytest.raises(ValueError, match="nothing to prefill"):
        model.get_initial_forward_pass_args("LLM", ["text"], ["text"], {})


def test_video_is_refused_rather_than_run_as_an_image(model):
    with pytest.raises(NotImplementedError, match="video"):
        model.process_prompt(
            "What happens?", ["video", "text"], ["text"],
            tensors={"video_inputs": [an_image()]},
        )


def test_attachment_count_mismatch_is_caught(model):
    with pytest.raises(ValueError, match="layout declares"):
        model.process_prompt(
            "Describe these.", ["image", "image", "text"], ["text"],
            tensors={"image_inputs": [an_image()]},
        )
