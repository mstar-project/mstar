"""Tests for Ming-flash-omni-2.0 prompt utilities (step 8)."""

from __future__ import annotations

import json

from mminf.model.ming_omni_flash.components.prompt_utils import (
    BASE_CAPTION_TEMPLATE,
    DEFAULT_NUM_QUERY_TOKENS,
    IMAGE_PATCH_TOKEN,
    create_instruction,
    maybe_expand_image_gen_prompt,
)

# ---------------------------------------------------------------------------
# Image-gen query-token expansion
# ---------------------------------------------------------------------------


def test_expand_appends_default_256_patch_block() -> None:
    out = maybe_expand_image_gen_prompt("draw a cat")
    assert out.startswith("draw a cat<image>")
    assert out.endswith("</image>")
    assert out.count(IMAGE_PATCH_TOKEN) == DEFAULT_NUM_QUERY_TOKENS  # 256


def test_expand_respects_custom_token_count() -> None:
    out = maybe_expand_image_gen_prompt("x", num_query_tokens=4)
    assert out == "x<image>" + IMAGE_PATCH_TOKEN * 4 + "</image>"


def test_expand_is_noop_when_already_has_patch_block() -> None:
    pre = "y<image>" + IMAGE_PATCH_TOKEN * 16 + "</image>"
    assert maybe_expand_image_gen_prompt(pre) == pre


def test_expand_is_noop_on_empty_or_non_string() -> None:
    assert maybe_expand_image_gen_prompt("") == ""
    assert maybe_expand_image_gen_prompt(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TTS caption builder
# ---------------------------------------------------------------------------


def test_create_instruction_returns_valid_json_with_defaults() -> None:
    s = create_instruction({})
    parsed = json.loads(s)
    assert "audio_sequence" in parsed
    item = parsed["audio_sequence"][0]
    assert item["序号"] == 1
    assert item["说话人"] == "speaker_1"
    assert item["BGM"]["Genre"] is None


def test_create_instruction_merges_known_keys() -> None:
    s = create_instruction({"说话人": "speaker_2", "情感": "happy"})
    item = json.loads(s)["audio_sequence"][0]
    assert item["说话人"] == "speaker_2"
    assert item["情感"] == "happy"


def test_create_instruction_ignores_unknown_keys() -> None:
    s = create_instruction({"unknown_field": "x", "情感": "sad"})
    item = json.loads(s)["audio_sequence"][0]
    assert "unknown_field" not in item
    assert item["情感"] == "sad"


def test_create_instruction_does_not_mutate_base_template() -> None:
    """create_instruction must deep-copy — calls must not leak into each other."""
    create_instruction({"说话人": "speaker_9"})
    # The module-level template is untouched.
    assert BASE_CAPTION_TEMPLATE["audio_sequence"][0]["说话人"] == "speaker_1"
    # A fresh call still sees the default.
    item = json.loads(create_instruction({}))["audio_sequence"][0]
    assert item["说话人"] == "speaker_1"


def test_create_instruction_emits_unescaped_unicode() -> None:
    """ensure_ascii=False keeps the Chinese field names readable."""
    s = create_instruction({})
    assert "说话人" in s   # not \uXXXX-escaped


def test_create_instruction_nested_bgm_key_not_merged_at_top_level() -> None:
    """BGM is a nested dict on the template; a top-level 'BGM' string replaces it
    only because the key exists — verify the merge is shallow (matches upstream)."""
    s = create_instruction({"BGM": {"Genre": "jazz"}})
    item = json.loads(s)["audio_sequence"][0]
    assert item["BGM"] == {"Genre": "jazz"}
