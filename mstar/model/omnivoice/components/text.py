"""Prompt construction: style tokens, text tokens, and the canvas prefix.

Ported from ``OmniVoice._prepare_inference_inputs`` and its helpers rather than
imported, because these are private to the reference package and decide parity:
a drift in whitespace handling or in how a non-verbal tag is split changes the
token ids the backbone sees.  Pinned to the behaviour of OmniVoice as of the
``k2-fsa/OmniVoice`` checkpoint; the parity test is what catches upstream drift.

The prefix these build is fixed for a request's whole diffusion loop, so it is
computed once on the data worker and persisted, not rebuilt per step.
"""

import re

import torch

# Tags tokenized standalone so their ids do not depend on the surrounding
# language's context.
_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


def combine_text(text: str, ref_text: str | None = None) -> str:
    """Join the reference transcript to the target text and normalise spacing.

    Voice cloning conditions on transcript *and* audio, so the reference
    transcript is prepended to the target text and the whole thing is read as
    one utterance.
    """
    full_text = (ref_text.strip() + " " + text.strip()) if ref_text else text.strip()
    full_text = re.sub(r"[\r\n]+", "", full_text)
    full_text = full_text.replace("（", "(").replace("）", ")")
    full_text = re.sub(r"[ \t]+", " ", full_text)
    chinese_range = r"[一-鿿]"
    pattern = rf"(?<={chinese_range})\s+|\s+(?={chinese_range})"
    return re.sub(pattern, "", full_text)


def tokenize_with_nonverbal_tags(text: str, tokenizer) -> torch.Tensor:
    """Tokenize to ``[1, N]``, splitting non-verbal tags into their own calls."""
    parts: list[list[int]] = []
    last_end = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        if m.start() > last_end:
            ids = tokenizer(text[last_end : m.start()], add_special_tokens=False).input_ids
            if ids:
                parts.append(ids)
        tag_ids = tokenizer(m.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.append(tag_ids)
        last_end = m.end()
    if last_end < len(text):
        ids = tokenizer(text[last_end:], add_special_tokens=False).input_ids
        if ids:
            parts.append(ids)

    if not parts:
        return tokenizer(text, return_tensors="pt").input_ids
    combined: list[int] = []
    for p in parts:
        combined.extend(p)
    return torch.tensor([combined], dtype=torch.long)


def build_style_text(
    language: str | None,
    instruct: str | None,
    denoise: bool,
    has_reference: bool,
) -> str:
    """The control preamble.

    ``<|denoise|>`` only applies when there is reference audio to denoise, so
    the flag alone does not add it.  Absent language or instruction are the
    literal string ``None`` rather than an omitted span — the model was trained
    with the span always present.
    """
    style_text = ""
    if denoise and has_reference:
        style_text += "<|denoise|>"
    style_text += f"<|lang_start|>{language or 'None'}<|lang_end|>"
    style_text += f"<|instruct_start|>{instruct or 'None'}<|instruct_end|>"
    return style_text


def build_prefix(
    tokenizer,
    text: str,
    num_audio_codebook: int,
    language: str | None = None,
    instruct: str | None = None,
    ref_text: str | None = None,
    ref_audio_tokens: torch.Tensor | None = None,
    denoise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Everything before the target canvas.

    Returns ``(prefix_ids [C, N], prefix_audio_mask [N])``.  Text spans repeat
    one row of ids across all ``C`` codebook rows; the reference audio span is
    genuinely per-codebook and is the only part of the prefix flagged as audio.
    """
    style_text = build_style_text(
        language=language,
        instruct=instruct,
        denoise=denoise,
        has_reference=ref_audio_tokens is not None,
    )
    style_tokens = (
        tokenizer(style_text, return_tensors="pt")
        .input_ids.repeat(num_audio_codebook, 1)
        .unsqueeze(0)
    )

    wrapped_text = f"<|text_start|>{combine_text(text, ref_text)}<|text_end|>"
    text_tokens = (
        tokenize_with_nonverbal_tags(wrapped_text, tokenizer)
        .repeat(num_audio_codebook, 1)
        .unsqueeze(0)
    )

    parts = [style_tokens, text_tokens]
    text_len = style_tokens.shape[-1] + text_tokens.shape[-1]
    ref_len = 0
    if ref_audio_tokens is not None:
        ref_len = ref_audio_tokens.shape[-1]
        parts.append(ref_audio_tokens.unsqueeze(0).to(dtype=torch.long))

    prefix_ids = torch.cat(parts, dim=2)[0]
    prefix_audio_mask = torch.zeros(text_len + ref_len, dtype=torch.bool)
    if ref_len:
        prefix_audio_mask[text_len:] = True
    return prefix_ids, prefix_audio_mask
