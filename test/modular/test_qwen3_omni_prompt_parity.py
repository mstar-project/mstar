"""Qwen3-Omni prompt parity: the layouts that predate ordering must not move.

The ordering fix rewrote how ``process_prompt`` builds its prompt — from
"split a sentinel out of a string-content template" to "render a content list,
tokenize once, slice around the placeholders". Every request shape that the
old template *could* express has to come out of the new path with byte-identical
token ids, or a shipped benchmark silently changes meaning.

So this file keeps the pre-PR prompt builder as an oracle and asserts the real
``Qwen3OmniModel.process_prompt`` reproduces it exactly, layout by layout. The
two layouts the old code could not express (text before an attachment, text
between two) are asserted to differ, since that is the fix.

The stub processor below reproduces the released checkpoint's chat template:
the rendering rules were read off the real ``AutoProcessor`` and are pinned in
``test_stub_matches_released_checkpoint``, which runs only when the weights are
present (``MSTAR_QWEN3_OMNI_PATH``). Everything else runs weight-free in CI.
"""

from __future__ import annotations

import os

import pytest
import torch

from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

# Pre-PR marker: split out of the templated prompt before tokenization.
_MM_SPLIT_SENTINEL = "<<<mstar_modality_split>>>"

# The placeholder triple the template writes per attachment.
PLACEHOLDERS = {
    "image": "<|vision_start|><|image_pad|><|vision_end|>",
    "video": "<|vision_start|><|video_pad|><|vision_end|>",
    "audio": "<|audio_start|><|audio_pad|><|audio_end|>",
}

# Real ids from the released checkpoint, so a drift in the vocab shows up here.
SPECIAL_IDS = {
    "<|im_start|>": 151644, "<|im_end|>": 151645,
    "<|vision_start|>": 151652, "<|vision_end|>": 151653,
    "<|image_pad|>": 151655, "<|video_pad|>": 151656,
    "<|audio_start|>": 151669, "<|audio_end|>": 151670,
    "<|audio_pad|>": 151675,
}


class _StubTokenizer:
    """Specials are single ids; ordinary characters are one id each.

    Character-level is deliberate: it makes a re-tokenized seam impossible to
    hide, so a boundary that shifted shows up as a different id sequence.
    """

    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return SPECIAL_IDS.get(token, self.unk_token_id)

    def _encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for tok, tid in SPECIAL_IDS.items():
                if text.startswith(tok, i):
                    ids.append(tid)
                    i += len(tok)
                    break
            else:
                ids.append(1000 + ord(text[i]))
                i += 1
        return ids

    def __call__(self, text: str, return_tensors: str | None = None):
        ids = self._encode(text)
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids) -> str:
        inverse = {v: k for k, v in SPECIAL_IDS.items()}
        return "".join(
            inverse[int(i)] if int(i) in inverse else chr(int(i) - 1000) for i in ids
        )


class _StubProcessor:
    """The checkpoint's ChatML template, for string and list content alike.

    List items are concatenated with no separator between them — that is what
    the released template does, and what lets two attachments sit adjacent.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        out = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                rendered = "".join(
                    part["text"] if part["type"] == "text"
                    else PLACEHOLDERS[part["type"]]
                    for part in content
                )
            else:
                rendered = content
            out.append(f"<|im_start|>{msg['role']}\n{rendered}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
        return "".join(out)


@pytest.fixture
def model():
    """A Qwen3OmniModel with only what ``process_prompt`` reads."""
    m = Qwen3OmniModel.__new__(Qwen3OmniModel)
    m.tokenizer = _StubTokenizer()
    m._processor = _StubProcessor()
    return m


def _old_process_prompt(model, prompt, input_modalities):
    """The pre-PR prompt builder, verbatim in behaviour.

    Modality content goes inside the user turn ahead of the prompt text; the
    templated string is split at the sentinel into the text before it and the
    text after, each tokenized on its own.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    num_mm = sum(1 for m in input_modalities if m != "text")
    if prompt is not None or num_mm:
        messages.append({
            "role": "user",
            "content": (_MM_SPLIT_SENTINEL if num_mm else "") + (prompt or ""),
        })
    text = model._processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if num_mm:
        head, sep, tail = text.partition(_MM_SPLIT_SENTINEL)
        assert sep and _MM_SPLIT_SENTINEL not in tail, text
        segments = [head, tail]
    else:
        segments = [text]
    return [
        model.tokenizer(seg, return_tensors="pt")["input_ids"][0]
        for seg in segments if seg
    ]


def _new_text_inputs(model, prompt, input_modalities, prompt_parts=None):
    return model.process_prompt(
        prompt, input_modalities, ["text"], prompt_parts=prompt_parts,
    )["text_inputs"]


def _ids(segments):
    return [s.tolist() for s in segments]


# Every shape the old template could express. Media first, text last: that is
# the only layout it had, whatever order the caller asked for.
LEGACY_LAYOUTS = [
    pytest.param("Name three primary colors.", ["text"], id="text-only"),
    pytest.param("Describe the image.", ["image", "text"], id="i2t"),
    pytest.param("Transcribe the audio.", ["audio", "text"], id="s2t"),
    pytest.param("Describe.", ["video", "text"], id="v2t"),
    pytest.param(None, ["image"], id="image-no-text"),
    pytest.param("Compare.", ["image", "image", "text"], id="two-images"),
    pytest.param("Describe.", ["image", "audio", "text"], id="image+audio"),
    pytest.param("Go.", ["image", "image", "image", "text"], id="three-images"),
]


@pytest.mark.parametrize(("prompt", "input_modalities"), LEGACY_LAYOUTS)
def test_legacy_layout_tokenizes_exactly_as_before(model, prompt, input_modalities):
    """The bytes a shipped benchmark sends must produce the bytes it did."""
    old = _old_process_prompt(model, prompt, input_modalities)
    new = _new_text_inputs(model, prompt, input_modalities)
    assert _ids(new) == _ids(old)


@pytest.mark.parametrize(("prompt", "input_modalities"), LEGACY_LAYOUTS)
def test_legacy_layout_span_count_is_stable(model, prompt, input_modalities):
    """One prefill walk per span, unchanged — the schedule length follows it."""
    old = _old_process_prompt(model, prompt, input_modalities)
    new = _new_text_inputs(model, prompt, input_modalities)
    assert len(new) == len(old)


def test_placeholders_are_not_in_any_text_span(model):
    """The walks re-emit the sentinels, so no span may carry one."""
    spans = _new_text_inputs(model, "Describe.", ["image", "audio", "text"])
    carried = {int(i) for s in spans for i in s}
    for token in ("<|image_pad|>", "<|audio_pad|>", "<|vision_start|>",
                  "<|vision_end|>", "<|audio_start|>", "<|audio_end|>"):
        assert SPECIAL_IDS[token] not in carried, f"{token} leaked into a text span"


def test_spans_come_from_one_tokenization(model):
    """Rejoining the spans around the placeholders rebuilds the whole prompt.

    This is the property the rewrite bought: no seam is tokenized twice, so no
    BPE merge can be cut at a boundary.
    """
    mods = ["image", "text"]
    spans = _new_text_inputs(model, "Describe the image.", mods)
    rejoined = (
        spans[0].tolist()
        + [SPECIAL_IDS["<|vision_start|>"], SPECIAL_IDS["<|image_pad|>"],
           SPECIAL_IDS["<|vision_end|>"]]
        + spans[1].tolist()
    )
    whole = model.tokenizer(
        model._processor.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": ""},
                {"type": "text", "text": "Describe the image."},
            ]},
        ]), return_tensors="pt")["input_ids"][0].tolist()
    assert rejoined == whole


# The layouts the old template could not express. These SHOULD differ — the
# old path moved the text behind the attachment; that was the bug.
@pytest.mark.parametrize(("prompt", "input_modalities"), [
    pytest.param("Look at this.", ["text", "image"], id="text-first"),
    pytest.param("Between.", ["image", "text", "image"], id="interleaved"),
])
def test_new_layouts_differ_from_the_old_path(model, prompt, input_modalities):
    old = _old_process_prompt(model, prompt, input_modalities)
    new = _new_text_inputs(model, prompt, input_modalities)
    assert _ids(new) != _ids(old)
    # And the text lands where it was written, not swept to the end.
    if input_modalities[0] == "text":
        assert model.tokenizer.decode(new[0]).endswith(prompt)


def test_schedule_matches_the_spans(model):
    """As many text walks as spans, one media walk per attachment, in order."""
    mods = ["image", "text", "image"]
    spans = _new_text_inputs(model, "Between.", mods)
    signals = {
        "text_inputs": [f"t{i}" for i in range(len(spans))],
        "pixel_values": ["px0", "px1"],
        "image_grid_thw": ["g0", "g1"],
    }
    schedule = model._build_thinker_prefill_schedule(mods, signals)
    assert [walk for walk, _ in schedule] == [
        "prefill_text", "prefill_vision", "prefill_text", "prefill_vision",
        "prefill_text",
    ]
    assert sum(1 for w, _ in schedule if w == "prefill_text") == len(spans)


@pytest.mark.skipif(
    not os.environ.get("MSTAR_QWEN3_OMNI_PATH"),
    reason="set MSTAR_QWEN3_OMNI_PATH to check the stub against real weights",
)
def test_stub_matches_released_checkpoint():
    """Pin the stub template to the checkpoint's, so the parity above is real."""
    from transformers import AutoProcessor

    real = AutoProcessor.from_pretrained(
        os.environ["MSTAR_QWEN3_OMNI_PATH"], trust_remote_code=True,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": ""},
            {"type": "text", "text": "Describe the image."},
        ]},
    ]
    assert real.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    ) == _StubProcessor().apply_chat_template(messages)
