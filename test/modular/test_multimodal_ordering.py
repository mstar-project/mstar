"""Ordering guarantees of the multimodal prompt adapter."""

import logging
import re
from types import SimpleNamespace

import pytest
import torch

from mstar.api_server.openai.adapters import flatten_messages
from mstar.model.multimodal import (
    PromptPart,
    check_attachments,
    parts_from_modalities,
    prefill_plan,
)


def _mods(plan):
    return [(p.modality, p.index) for p in plan]


def test_layout_survives_intake(tmp_path):
    """Text written between two attachments keeps its position."""
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
        {"type": "text", "text": "and this one"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
    ]}]
    text, file_paths, in_mods, parts = flatten_messages(messages, tmp_path)
    assert in_mods == ["image", "text", "image"]
    assert [p.modality for p in parts] == ["image", "text", "image"]
    assert [p.index for p in parts if p.modality == "image"] == [0, 1]
    assert len(file_paths["image"]) == 2
    assert text == "and this one"


def test_repeated_modality_is_indexed(tmp_path):
    """N attachments of one modality address N distinct inputs, in order."""
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}
        for _ in range(4)
    ]}]
    _, file_paths, in_mods, parts = flatten_messages(messages, tmp_path)
    assert in_mods == ["image"] * 4
    assert [p.index for p in parts] == [0, 1, 2, 3]
    assert len(set(file_paths["image"])) == 4


def test_single_attachment_layout_is_unchanged(tmp_path):
    """The common single-image request plans exactly as it did before."""
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
        {"type": "text", "text": "describe it"},
    ]}]
    _, _, in_mods, _ = flatten_messages(messages, tmp_path)
    plan = prefill_plan(parts_from_modalities(in_mods))
    assert _mods(plan) == [("text", 0), ("image", 0), ("text", 1)]


def test_plan_orders_attachments_as_written():
    parts = parts_from_modalities(["audio", "image", "audio"])
    plan = prefill_plan(parts)
    assert _mods(plan) == [
        ("text", 0), ("audio", 0), ("image", 0), ("audio", 1), ("text", 1),
    ]


def test_adjacent_text_collapses_into_one_span():
    """Two text parts in a row are one contiguous run in the rendered prompt."""
    parts = [
        PromptPart(modality="text", text="a"),
        PromptPart(modality="text", text="b"),
        PromptPart(modality="image", index=0),
    ]
    assert _mods(prefill_plan(parts)) == [("text", 0), ("image", 0), ("text", 1)]


def test_leading_text_is_optional():
    """BAGEL's generation prompt opens straight into the attachment."""
    parts = parts_from_modalities(["image", "text"])
    assert _mods(prefill_plan(parts, leading_text=False)) == [
        ("image", 0), ("text", 0),
    ]


def test_plan_carries_each_segment_text():
    parts = [
        PromptPart(modality="text", text="A"),
        PromptPart(modality="image", index=0),
        PromptPart(modality="text", text="B"),
        PromptPart(modality="image", index=1),
    ]
    plan = prefill_plan(parts)
    assert [(p.modality, p.text) for p in plan] == [
        ("text", "A"), ("image", None), ("text", "B"), ("image", None),
        ("text", None),
    ]


@pytest.mark.parametrize(("layout", "counts", "wanted"), [
    (["image", "image", "text"], {"image": 1}, "declares 2 image"),
    (["image", "text", "image"], {"image": 1}, "declares 2 image"),
    (["image", "text"], {"image": 2}, "declares 1 image"),
    (["text"], {"image": 1}, "declares 0 image"),
    (["audio", "text"], {"image": 1}, "declares 1 audio"),
])
def test_a_layout_that_does_not_match_its_attachments_is_refused(
    layout, counts, wanted,
):
    """The count the caller declared has to be the count that arrived.

    ``check_plan`` cannot catch these: it compares the plan against a prompt
    rendered from the same parts, so the placeholders always agree with the
    layout however many attachments were uploaded.
    """
    with pytest.raises(ValueError, match=wanted):
        check_attachments(parts_from_modalities(layout), counts)


@pytest.mark.parametrize(("layout", "counts"), [
    (["text"], {}),
    (["image", "text"], {"image": 1}),
    (["image", "image", "text"], {"image": 2}),
    (["image", "audio", "text"], {"image": 1, "audio": 1}),
    (["text"], {"image": 0, "audio": 0, "video": 0}),
])
def test_a_matching_layout_passes(layout, counts):
    check_attachments(parts_from_modalities(layout), counts)


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
            if piece in self.SPECIALS:
                ids.append(self.SPECIALS[piece])
            else:
                ids.extend(1000 + ord(c) for c in piece)
        return ids

    def decode(self, ids):
        inverse = {v: k for k, v in self.SPECIALS.items()}
        return "".join(
            inverse[i] if i in inverse else chr(i - 1000) for i in ids
        )


@pytest.fixture
def bagel():
    """A BagelModel with only what process_prompt needs."""
    from mstar.model.bagel.bagel_model import BagelModel

    class _Bagel(BagelModel):
        def __init__(self):
            self.config = SimpleNamespace(think_mode=False)
            self.tokenizer = _StubTokenizer()
            self.boi_token_id = _StubTokenizer.SPECIALS["<|vision_start|>"]
            self.eoi_token_id = _StubTokenizer.SPECIALS["<|vision_end|>"]

    return _Bagel()


def _decoded(bagel, spans):
    return [bagel.tokenizer.decode(s.tolist()) for s in spans]


@pytest.mark.parametrize(
    ("prompt", "in_mods", "out_mods"),
    [
        ("describe it", ["image", "text"], ["text"]),
        ("hello", ["text"], ["text"]),
        ("a cat", ["text"], ["image"]),
        ("make it night", ["image", "text"], ["image"]),
        ("compare", ["image", "image", "text"], ["text"]),
    ],
)
def test_legacy_layouts_tokenize_exactly_as_before(bagel, prompt, in_mods, out_mods):
    """Requests with no ordering to preserve keep their existing prompt."""
    spans = _decoded(bagel, bagel.process_prompt(prompt, in_mods, out_mods)["text_inputs"])
    expected = {
        ("image", "text", "text"): [
            bagel.VLM_UNDERSTANDING_PREFIX.format(
                system_prompt=bagel.BAGEL_DEFAULT_SYSTEM_PROMPT
            ),
            bagel.VLM_UNDERSTANDING_SUFFIX.format(prompt=prompt),
        ],
    }.get((in_mods[0], in_mods[-1], out_mods[0]))
    if expected is not None:
        assert spans == expected
    assert spans  # every shape still produces at least one span


def test_text_before_an_attachment_stays_before_it(bagel):
    """The ordering fix: text written first is prefilled first."""
    parts = [
        PromptPart(modality="text", text="look at this"),
        PromptPart(modality="image", index=0),
    ]
    spans = _decoded(bagel, bagel.process_prompt(
        "look at this", ["text", "image"], ["text"], prompt_parts=parts
    )["text_inputs"])
    assert spans[0].endswith("look at this")
    assert "look at this" not in spans[-1]


def test_interleaved_layout_prefills_end_to_end(bagel):
    """text -> image -> text -> image -> text, spans and schedule agreeing."""
    parts = [
        PromptPart(modality="text", text="A"),
        PromptPart(modality="image", index=0),
        PromptPart(modality="text", text="B"),
        PromptPart(modality="image", index=1),
        PromptPart(modality="text", text="C"),
    ]
    in_mods = [p.modality for p in parts]
    tensors = bagel.process_prompt(
        "A\nB\nC", in_mods, ["text"], prompt_parts=parts
    )["text_inputs"]
    spans = _decoded(bagel, tensors)
    assert len(spans) == 3
    assert spans[0].endswith("A")
    # The newline is the one the understanding template used to supply after
    # the attachments, now written at the span that follows one.
    assert spans[1] == "\nB"
    assert spans[2].startswith("\nC")

    schedule = bagel._build_prefill_schedule(
        input_modalities=in_mods,
        input_signals={"text_inputs": tensors, "image_inputs": ["i0", "i1"]},
        is_understanding=True,
    )
    assert [walk for walk, _ in schedule] == [
        "prefill_text", "prefill_vit", "prefill_text", "prefill_vit", "prefill_text",
    ]


def test_bagel_refuses_an_attachment_it_has_no_encoder_for(bagel):
    """An audio part would otherwise be encoded as an image and answered."""
    with pytest.raises(ValueError, match="no encoder for audio"):
        bagel.process_prompt("describe it", ["audio", "text"], ["text"])


def test_bagel_skips_a_part_it_cannot_prefill(caplog):
    """A part with no matching input is skipped and logged, not raised.

    This runs in the conductor, whose loop only logs what escapes it, so a
    raise here strands the request instead of answering it. check_plan
    already rejected the mismatch at intake, where a 400 can still reach
    the client.
    """
    from mstar.model.bagel.bagel_model import BagelModel

    model = BagelModel.__new__(BagelModel)
    with caplog.at_level(logging.WARNING):
        schedule = BagelModel._build_prefill_schedule(
            model,
            input_modalities=["image", "image", "text"],
            input_signals={"text_inputs": ["t0", "t1"], "image_inputs": ["i0"]},
            is_understanding=True,
        )
    # The image it has still prefills; only the one with no input drops.
    assert [walk for walk, _ in schedule] == [
        "prefill_text", "prefill_vit", "prefill_text",
    ]
    assert "skipping it" in caplog.text


def test_bagel_spans_come_from_one_tokenization(bagel):
    """The scan splits an already-tokenized prompt, never re-tokenizes a seam."""
    parts = [
        PromptPart(modality="text", text="A"),
        PromptPart(modality="image", index=0),
        PromptPart(modality="text", text="B"),
    ]
    rendered = bagel._render_prompt(
        parts, is_understanding=True, system_prompt=bagel.BAGEL_DEFAULT_SYSTEM_PROMPT
    )
    assert bagel.IMAGE_PLACEHOLDER in rendered
    spans = bagel.process_prompt(
        "A\nB", [p.modality for p in parts], ["text"], prompt_parts=parts
    )["text_inputs"]
    # Concatenating the spans back, with the placeholder's interior restored,
    # reproduces the single tokenization they were sliced out of.
    whole = bagel.tokenizer.encode(rendered)
    rejoined = spans[0].tolist() + [
        bagel.boi_token_id,
        bagel.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        bagel.eoi_token_id,
    ] + spans[1].tolist()
    assert rejoined == whole


def test_qwen_schedule_walks_the_plan():
    """Qwen3-Omni prefills each attachment where it was written."""
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    model = Qwen3OmniModel.__new__(Qwen3OmniModel)
    mods = ["text", "image", "text", "image", "text", "audio"]
    signals = {
        "text_inputs": ["t0", "t1", "t2", "t3"],
        "pixel_values": ["px0", "px1"],
        "image_grid_thw": ["g0", "g1"],
        "audio_features": ["af0"],
        "audio_seqlens": ["as0"],
    }
    schedule = Qwen3OmniModel._build_thinker_prefill_schedule(model, mods, signals)
    assert [walk for walk, _ in schedule] == [
        "prefill_text", "prefill_vision", "prefill_text", "prefill_vision",
        "prefill_text", "prefill_audio", "prefill_text",
    ]
    # Each vision walk carries its own image, in order.
    vision = [entry for walk, entry in schedule if walk == "prefill_vision"]
    assert [e["pixel_values"] for e in vision] == ["px0", "px1"]
    assert [e["image_grid_thw"] for e in vision] == ["g0", "g1"]


def test_qwen_placeholder_ids_come_from_the_tokenizer():
    """Not from thinker_config, which disagrees with it on the checkpoint."""
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    vocab = {
        "<|audio_start|>": 151645, "<|audio_end|>": 151646,
        "<|audio_pad|>": 151647, "<|image_pad|>": 151648,
        "<|video_pad|>": 151649, "<|vision_start|>": 151650,
        "<|vision_end|>": 151651,
    }
    model = Qwen3OmniModel.__new__(Qwen3OmniModel)
    model.tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda t: vocab.get(t, 0), unk_token_id=0
    )
    specs = Qwen3OmniModel._placeholder_specs(model)
    assert specs["image"] == (151650, 151648, 151651)
    assert specs["audio"] == (151645, 151647, 151646)


def test_a_modality_the_tokenizer_cannot_place_is_left_out():
    """A missing placeholder drops that modality rather than scanning for 0."""
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    vocab = {"<|image_pad|>": 5, "<|vision_start|>": 6, "<|vision_end|>": 7}
    model = Qwen3OmniModel.__new__(Qwen3OmniModel)
    model.tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda t: vocab.get(t, 0), unk_token_id=0
    )
    assert set(Qwen3OmniModel._placeholder_specs(model)) == {"image"}


# ── the layout Bagel keys an understanding prompt's KV by ───────────────
# The layout has to match the walks exactly: the order the prefill schedule
# runs them in, and each image's own file over the slots its ViT walk writes.
# An image keyed by another image's file would be served that image's KV.

PATCH_SIZE, MAX_PATCHES_PER_SIDE = 14, 70
# Two images of different sizes, so a span sized or keyed by the wrong image fails the test.
SIZES = [(480, 640), (1200, 1600)]


def _laid_out(bagel, modalities: list[str], output_modalities=("text",), tensors=True, **kwargs):
    """Return what Bagel's `process_prompt` gives for the text and images in ``modalities``, with the images."""
    bagel.config.vit_config = SimpleNamespace(patch_size=PATCH_SIZE)
    bagel.config.vit_max_num_patch_per_side = MAX_PATCHES_PER_SIDE
    texts = iter(["what is", "in this picture", "and in this one"])
    parts, images = [], []
    for modality in modalities:
        if modality == "text":
            parts.append(PromptPart(modality="text", text=next(texts)))
        else:
            parts.append(PromptPart(modality="image", index=len(images)))
            images.append(torch.rand(3, *SIZES[len(images)]))
    output = bagel.process_prompt(
        "\n".join(part.text for part in parts if part.text), modalities, list(output_modalities),
        tensors={"image_inputs": images} if tensors else None, prompt_parts=parts, **kwargs,
    )
    return output, images


def _vit_slots(image: torch.Tensor, image_preprocess: str) -> int:
    """How many slots the ViT walk writes for ``image``: its patches and the two sentinels."""
    from mstar.model.bagel.submodules import IMAGE_SENTINELS, ViTEncoderSubmodule

    vit = ViTEncoderSubmodule(None, None, None, PATCH_SIZE, MAX_PATCHES_PER_SIDE)
    return vit.prepare_inputs(
        "prefill_vit", SimpleNamespace(step_metadata={"image_preprocess": image_preprocess}),
        {"image_inputs": [image]},
    ).kwargs["max_seqlen"] + IMAGE_SENTINELS


_LAID_OUT = {
    "one image": (["text", "image", "text"], "default"),
    "two images": (["text", "image", "text", "image", "text"], "default"),
    "image first": (["image", "text", "image"], "default"),
    "vllm's square": (["text", "image", "text"], "vllm"),
}


@pytest.mark.parametrize(("modalities", "image_preprocess"), _LAID_OUT.values(), ids=_LAID_OUT.keys())
def test_bagel_lays_each_image_out_in_schedule_order_by_its_own_file_over_its_vits_slots(
    bagel, modalities, image_preprocess,
):
    output, images = _laid_out(bagel, modalities, image_preprocess=image_preprocess)
    layout = output.metadata["prefix_layout"]["kv"]["main"]
    schedule = bagel._build_prefill_schedule(
        input_modalities=modalities, is_understanding=True,
        input_signals={"text_inputs": output.new_input_tensors["text_inputs"], "image_inputs": images},
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


def test_bagel_lays_out_only_the_understood_prompts_it_has_images_for(bagel):
    interleaved = ["text", "image", "text"]
    cases = {
        "understood with its image": (interleaved, ("text",), True),
        "text only": (["text"], ("text",), True),
        "generation": (interleaved, ("image",), True),
        "understood without tensors": (interleaved, ("text",), False),
        # Text comes first, so the prompt is understood, but it also outputs an image, so its input is resized first.
        "understood and edited": (interleaved, ("text", "image"), True),
    }

    # A prompt with no layout comes back as a plain dict of its tensors.
    laid_out = {name: not isinstance(_laid_out(bagel, *case)[0], dict) for name, case in cases.items()}

    assert laid_out == {name: name == "understood with its image" for name in cases}, (
        "a layout went to a prompt with no understood image or none of its "
        "tensors, or was missing from the one that has both"
    )
