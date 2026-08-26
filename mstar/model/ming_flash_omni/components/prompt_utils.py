"""Ming-flash-omni-2.0 prompt utilities (step 8).

Port of vllm-omni's ``ming_flash_omni/prompt_utils.py``. Two unrelated
helper families share the file because both are tightly coupled to
Ming-specific prompt conventions:

1. **Image-gen query-token expansion** — string-level helpers that mark
   the ``<image><imagePatch>*N</image>`` block the thinker substitutes
   with learnable image-gen query embeddings during forward. Used by the
   ImageGen path (step 9); included here so the constants live in one
   place.

2. **TTS / talker caption builder** — the JSON caption template + merge
   helper for the standalone ``ming_flash_omni_tts`` talker-only deploy.
   Lets the talker accept the same JSON caption shape vllm-omni speaks
   (speaker / dialect / style / emotion / BGM controls).
"""

from __future__ import annotations

import copy
import json
from typing import Any

# ============================================================
# Image-gen query-token block (thinker stage — used by step 9)
# ============================================================

_IMAGE_OPEN_TOKEN = "<image>"
_IMAGE_CLOSE_TOKEN = "</image>"
IMAGE_PATCH_TOKEN = "<imagePatch>"

# Matches ``ImageGenConfig(img_gen_scales=[16])`` → 16*16 = 256 on the
# released inclusionAI/Ming-flash-omni-2.0 checkpoint.
DEFAULT_NUM_QUERY_TOKENS = 256


def maybe_expand_image_gen_prompt(
    prompt: str,
    num_query_tokens: int = DEFAULT_NUM_QUERY_TOKENS,
) -> str:
    """Append the ``<image><imagePatch>*N</image>`` suffix for text-to-image.

    The thinker expects image-generation requests to end with an N-wide
    block of ``<imagePatch>`` tokens (wrapped in ``<image>`` / ``</image>``)
    whose positions get substituted with learnable query embeddings during
    forward.

    No-op (returns the input unchanged) when ``prompt`` is not a non-empty
    string, or already contains an ``<imagePatch>`` block (avoids double
    expansion).

    Args:
        prompt: raw user prompt text.
        num_query_tokens: total query tokens to emit (default 256).
    """
    if not isinstance(prompt, str) or not prompt:
        return prompt
    if IMAGE_PATCH_TOKEN in prompt:
        return prompt
    suffix = _IMAGE_OPEN_TOKEN + (IMAGE_PATCH_TOKEN * num_query_tokens) + _IMAGE_CLOSE_TOKEN
    return prompt + suffix


# ============================================================
# TTS / talker caption builder (talker-only deploy)
# ============================================================

DEFAULT_PROMPT = "Please generate speech based on the following description.\n"

# Base caption schema the standalone talker understands. Keys are the
# Ming-native Chinese field names (序号 = index, 说话人 = speaker,
# 方言 = dialect, 风格 = style, 语速 = speed, 基频 = pitch, 音量 = volume,
# 情感 = emotion, BGM = background music block, IP = persona).
BASE_CAPTION_TEMPLATE: dict[str, Any] = {
    "audio_sequence": [
        {
            "序号": 1,
            "说话人": "speaker_1",
            "方言": None,
            "风格": None,
            "语速": None,
            "基频": None,
            "音量": None,
            "情感": None,
            "BGM": {
                "Genre": None,
                "Mood": None,
                "Instrument": None,
                "Theme": None,
                "ENV": None,
                "SNR": None,
            },
            "IP": None,
        }
    ]
}


def create_instruction(user_input: dict[str, Any]) -> str:
    """Return a JSON caption string for ``audio_sequence[0]``.

    Only keys already present on the base template are merged in; unknown
    keys are silently ignored so the output schema stays stable (the
    talker's prompt parser keys off the exact field set).

    Args:
        user_input: partial caption controls, e.g.
            ``{"说话人": "speaker_2", "情感": "happy"}``.

    Returns:
        A UTF-8 JSON string (``ensure_ascii=False`` to keep the Chinese
        field names readable, matching upstream).
    """
    caption = copy.deepcopy(BASE_CAPTION_TEMPLATE)
    item = caption["audio_sequence"][0]
    for key, value in user_input.items():
        if key in item:
            item[key] = value
    return json.dumps(caption, ensure_ascii=False)


__all__ = [
    "IMAGE_PATCH_TOKEN",
    "DEFAULT_NUM_QUERY_TOKENS",
    "maybe_expand_image_gen_prompt",
    "DEFAULT_PROMPT",
    "BASE_CAPTION_TEMPLATE",
    "create_instruction",
]
