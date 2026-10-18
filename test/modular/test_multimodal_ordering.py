"""Ordering guarantees of the multimodal prompt adapter."""

import logging
import re
from types import SimpleNamespace

import pytest
import torch

from mstar.api_server.openai.adapters import flatten_messages
from mstar.api_server.openai.serving_chat import _build_response
from mstar.model.multimodal import (
    PromptPart,
    check_attachments,
    messages_from_parts,
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


def _roles(parts):
    return [(p.modality, p.role, p.text) for p in parts]


def test_each_message_keeps_its_role(tmp_path):
    """A reply stays its own part, so a model can render it as its own turn."""
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            {"type": "text", "text": "what is it"},
        ]},
        {"role": "assistant", "content": "a cat"},
        {"role": "user", "content": "and its color?"},
    ]
    text, _, _, parts = flatten_messages(messages, tmp_path)
    assert text == "Be brief.\nwhat is it\na cat\nand its color?"
    assert _roles(parts) == [
        ("text", "system", "Be brief."),
        ("image", "user", None),
        ("text", "user", "what is it"),
        ("text", "assistant", "a cat"),
        ("text", "user", "and its color?"),
    ], "text merged across a role change, so the reply lands inside a user turn"


def test_text_within_one_message_still_merges(tmp_path):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]}]
    text, _, in_mods, parts = flatten_messages(messages, tmp_path)
    assert text == "first\nsecond"
    assert in_mods == ["text"]
    assert _roles(parts) == [("text", "user", "first\nsecond")], (
        "one message split into two parts would render as two turns"
    )


def test_messages_of_one_role_stay_one_turn(tmp_path):
    """The layout has no slot for a boundary between them, so they merge."""
    messages = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]
    text, _, _, parts = flatten_messages(messages, tmp_path)
    assert text == "first\nsecond"
    assert _roles(parts) == [("text", "user", "first\nsecond")], (
        "two user messages no longer render as the one turn they did before"
    )


def test_a_developer_message_is_a_system_message(tmp_path):
    """OpenAI's newer name for it, which Qwen3-Omni's template drops, text and all."""
    messages = [
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ]
    _, _, _, parts = flatten_messages(messages, tmp_path)
    assert _roles(parts) == [("text", "system", "Be brief."), ("text", "user", "hi")], (
        "a developer message reached the templates under a role they do not render"
    )


@pytest.mark.parametrize("role", ["tool", "function", "usr"])
def test_a_role_a_chat_cannot_render_is_refused(tmp_path, role):
    refused = f"^a message's role must be system, developer, user or assistant, not '{role}'$"
    with pytest.raises(ValueError, match=refused):
        flatten_messages([{"role": role, "content": "hi"}], tmp_path)


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


def test_messages_from_parts_gives_one_message_per_turn():
    """An attachment stays in its own turn, as the item the template writes a placeholder for."""
    parts = [
        PromptPart(modality="text", text="Be brief.", role="system"),
        PromptPart(modality="image", index=0, role="user"),
        PromptPart(modality="text", text="What is it?", role="user"),
        PromptPart(modality="text", text="A cat.", role="assistant"),
    ]
    assert messages_from_parts(parts) == [
        {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
        {"role": "user", "content": [
            {"type": "image", "image": ""}, {"type": "text", "text": "What is it?"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]},
    ], "the chat reaches the template as other than one message per turn"


@pytest.mark.parametrize(("roles", "leads"), [
    (["user"], True),
    (["user", "system", "user"], False),
])
def test_the_default_system_message_leads_only_when_the_client_sent_none(roles, leads):
    parts = [PromptPart(modality="text", text=f"m{i}", role=r) for i, r in enumerate(roles)]
    messages = messages_from_parts(parts, default_system="Default.")
    assert (messages[0] == {"role": "system", "content": "Default."}) == leads, (
        "a template would write the default and the client's system message both"
        if not leads else "a chat with no system message lost the default one"
    )


def test_a_part_with_no_role_is_the_users():
    """An entrypoint with no messages sends role-less parts."""
    assert messages_from_parts([PromptPart(modality="text", text="hi")]) == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ], "a role-less part did not render as the user turn it always did"


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
            inverse[int(i)] if int(i) in inverse else chr(int(i) - 1000) for i in ids
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
        ("describe it", ["text", "image"], ["text"]),
        ("hello", ["text"], ["text"]),
        ("a cat", ["text"], ["image"]),
        ("make it night", ["image", "text"], ["image"]),
        ("compare", ["image", "image", "text"], ["text"]),
    ],
)
@pytest.mark.parametrize("as_chat", [False, True], ids=["legacy", "one-message-chat"])
def test_legacy_layouts_tokenize_exactly_as_before(
    bagel, tmp_path, prompt, in_mods, out_mods, as_chat,
):
    """Requests with no ordering to preserve keep their existing prompt, and so
    does a chat of one user message in the same layout. Text alone renders in
    role blocks, as text with an image does."""
    parts = None
    if as_chat:
        content = [
            {"type": "text", "text": prompt} if m == "text"
            else {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}
            for m in in_mods
        ]
        _, _, _, parts = flatten_messages([{"role": "user", "content": content}], tmp_path)
    spans = _decoded(bagel, bagel.process_prompt(
        prompt, in_mods, out_mods, prompt_parts=parts,
    )["text_inputs"])
    expected = {
        ("image", "text", "text"): [
            bagel.VLM_UNDERSTANDING_PREFIX.format(
                system_prompt=bagel.BAGEL_DEFAULT_SYSTEM_PROMPT
            ),
            bagel.VLM_UNDERSTANDING_SUFFIX.format(prompt=prompt),
        ],
        ("text", "image", "text"): [
            bagel.VLM_UNDERSTANDING_PREFIX.format(
                system_prompt=bagel.BAGEL_DEFAULT_SYSTEM_PROMPT
            ) + prompt,
            bagel.VLM_UNDERSTANDING_SUFFIX.format(prompt=""),
        ],
        ("text", "text", "text"): [
            f"<|im_start|>system\n{bagel.BAGEL_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        ],
    }.get((in_mods[0], in_mods[-1], out_mods[0]))
    if expected is not None:
        assert spans == expected
    assert spans  # every shape still produces at least one span


def _chat_ids(bagel, messages, tmp_path):
    text, _, in_mods, parts = flatten_messages(messages, tmp_path)
    [ids] = bagel.process_prompt(text, in_mods, ["text"], prompt_parts=parts)["text_inputs"]
    return ids.tolist()


FIRST_TURN = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Name a color."},
]
SECOND_TURN = FIRST_TURN + [
    {"role": "assistant", "content": "Blue."},
    {"role": "user", "content": "Another."},
]


def test_a_legacy_prompt_with_no_text_slot_still_renders(bagel):
    """A caller that sends a prompt but no text slot in its layout got it rendered before."""
    [span] = _decoded(bagel, bagel.process_prompt("hello", [], ["text"])["text_inputs"])
    assert span == (
        f"<|im_start|>system\n{bagel.BAGEL_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    ), "a prompt with no text slot in its layout was dropped"


@pytest.mark.parametrize(("messages", "blocks"), [
    (SECOND_TURN[1:], [("system", None), ("user", "Name a color."), ("assistant", "Blue."), ("user", "Another.")]),
    (SECOND_TURN, [("system", "Be brief."), ("user", "Name a color."), ("assistant", "Blue."), ("user", "Another.")]),
    (FIRST_TURN[:1], [("system", None), ("system", "Be brief.")]),
    (
        [FIRST_TURN[1], FIRST_TURN[0], SECOND_TURN[3]],
        [("system", None), ("user", "Name a color."), ("system", "Be brief."), ("user", "Another.")],
    ),
], ids=["default-system", "client-system", "lone-system", "later-system"])
def test_a_text_chat_renders_one_role_block_per_turn(bagel, tmp_path, messages, blocks):
    """Without a role label the model cannot tell its own turn from the client's."""
    rendered = bagel.tokenizer.decode(_chat_ids(bagel, messages, tmp_path))
    assert rendered == "".join(
        f"<|im_start|>{role}\n{text or bagel.BAGEL_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n" for role, text in blocks
    ) + "<|im_start|>assistant\n", "a text turn rendered without its role label"


def test_think_mode_still_instructs_a_client_system_prompt(bagel, tmp_path):
    from mstar.model.bagel.bagel_model import VLM_THINK_SYSTEM_PROMPT

    bagel.config.think_mode = True
    rendered = bagel.tokenizer.decode(_chat_ids(bagel, FIRST_TURN, tmp_path))
    assert rendered.startswith(f"<|im_start|>system\nBe brief. {VLM_THINK_SYSTEM_PROMPT}<|im_end|>\n"), (
        "a client's system prompt dropped the think-mode instruction"
    )


def test_the_next_turn_extends_the_last_prompt_and_its_reply(bagel, tmp_path):
    """Turn 2 re-sends turn 1 and its reply, which the cache keyed as generated.

    Flattened, the reply followed a newline where turn 1's prompt had closed
    its turn, so no page keyed from the reply ever matched.
    """
    first = _chat_ids(bagel, FIRST_TURN, tmp_path)
    second = _chat_ids(bagel, SECOND_TURN, tmp_path)
    reply = bagel.tokenizer.encode("Blue.") + [_StubTokenizer.SPECIALS["<|im_end|>"]]
    assert second[:len(first) + len(reply)] == first + reply, (
        "turn 2 does not start with turn 1's prompt and reply, so the reply's pages never match"
    )


def test_a_reply_comes_back_without_its_end_token(bagel):
    """Sent back as an assistant message, the reply gets its end token from the template."""
    bagel.eos_token_id = _StubTokenizer.SPECIALS["<|im_end|>"]
    ids = bagel.tokenizer.encode("Blue.<|im_end|>")
    reply = b"".join(bagel.postprocess(torch.tensor([i]), "text") for i in ids)
    assert reply == b"Blue.", "the reply ends with a literal end token, which the next turn writes twice"


def test_ignore_eos_keeps_every_end_token(bagel):
    """Generation runs past them, and the benchmark counts one token per non-empty chunk."""
    bagel.eos_token_id = _StubTokenizer.SPECIALS["<|im_end|>"]
    ids = bagel.tokenizer.encode("Blue.<|im_end|><|im_end|>")
    chunks = [bagel.postprocess(torch.tensor([i]), "text", request_kwargs={"ignore_eos": True}) for i in ids]
    assert all(chunks) and b"".join(chunks) == b"Blue.<|im_end|><|im_end|>", (
        "an end token decoded to an empty chunk, which the benchmark does not count"
    )


def test_an_image_chat_puts_each_turn_in_its_own_role_block(bagel, tmp_path):
    """A message's attachment sits inside its own turn."""
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            {"type": "text", "text": "What is it?"},
        ]},
        {"role": "assistant", "content": "A cat."},
        {"role": "user", "content": "Its color?"},
    ]
    text, _, in_mods, parts = flatten_messages(messages, tmp_path)
    spans = bagel.process_prompt(text, in_mods, ["text"], prompt_parts=parts)["text_inputs"]
    rendered = bagel.IMAGE_PLACEHOLDER.join(_decoded(bagel, spans))
    assert rendered == (
        f"<|im_start|>system\n{bagel.BAGEL_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{bagel.IMAGE_PLACEHOLDER}\nWhat is it?<|im_end|>\n"
        "<|im_start|>assistant\nA cat.<|im_end|>\n"
        "<|im_start|>user\nIts color?<|im_end|>\n"
        "<|im_start|>assistant\n"
    ), "the chat was flattened into one user turn"


def test_a_replys_image_comes_back_in_the_next_user_turn(bagel, tmp_path):
    """A client appends the assistant message a chat response carries and goes on."""
    reply = _build_response("bagel", "chatcmpl-0", [
        SimpleNamespace(modality="text", data=b"Here it is."),
        SimpleNamespace(modality="image", data=b"\x89PNG"),
    ], 24000)["choices"][0]["message"]
    messages = [
        {"role": "user", "content": "Draw a red cube."},
        reply,
        {"role": "user", "content": "Make it blue."},
    ]
    text, _, in_mods, parts = flatten_messages(messages, tmp_path)
    spans = bagel.process_prompt(text, in_mods, ["text"], prompt_parts=parts)["text_inputs"]
    assert bagel.IMAGE_PLACEHOLDER.join(_decoded(bagel, spans)) == (
        f"<|im_start|>system\n{bagel.BAGEL_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\nDraw a red cube.<|im_end|>\n"
        "<|im_start|>assistant\nHere it is.<|im_end|>\n"
        f"<|im_start|>user\n{bagel.IMAGE_PLACEHOLDER}\nMake it blue.<|im_end|>\n"
        "<|im_start|>assistant\n"
    ), "the reply's image did not land in the next user turn"


def test_an_image_chat_puts_a_leading_system_message_in_the_system_block(bagel, tmp_path):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            {"type": "text", "text": "What is it?"},
        ]},
    ]
    text, _, in_mods, parts = flatten_messages(messages, tmp_path)
    spans = bagel.process_prompt(text, in_mods, ["text"], prompt_parts=parts)["text_inputs"]
    assert bagel.IMAGE_PLACEHOLDER.join(_decoded(bagel, spans)) == (
        "<|im_start|>system\nBe brief.<|im_end|>\n"
        f"<|im_start|>user\n{bagel.IMAGE_PLACEHOLDER}\nWhat is it?<|im_end|>\n"
        "<|im_start|>assistant\n"
    ), "the client's system prompt was written twice, or the default kept"


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
