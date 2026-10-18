"""CosyVoice3's text frontend against the reference's.

``REFERENCE`` holds what ``CosyVoiceFrontEnd.text_normalize`` (FunAudioLLM/
CosyVoice at 074ca6d, ``split=True``) returns with neither ttsfrd nor wetext
installed, its fallback, for the CosyVoice3 texts in the reference's
``example.py`` and for sentences with digits, dates and units. Each returned
one piece. The cases that spell out digits need ``inflect`` (the cosyvoice3
extra) and skip without it.
"""

from __future__ import annotations

import builtins
import logging
import math

import pytest
import torch

from mstar.model.cosyvoice3.components.text_frontend import CosyVoice3TextFrontend, close_sentences
from mstar.model.cosyvoice3.config import END_OF_PROMPT, END_OF_PROMPT_ID, SYSTEM_PROMPT
from mstar.model.cosyvoice3.cosyvoice3_model import CosyVoice3Model
from mstar.model.cosyvoice3.submodules import TEXT_INPUTS

REFERENCE = [
    # example.py, cosyvoice3_example()
    (
        "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
        "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
    ),
    (
        "You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，"
        "[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]",
        "You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，"
        "[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]",
    ),
    ("好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。", "好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。"),
    (
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
    ),
    ("高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。", "高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。"),
    (
        "You are a helpful assistant.<|endofprompt|>レキシ テキ セカイ ニ オイ テ ワ、カコ ワ タンニ スギサッ タ "
        "モノ デ ワ ナイ、プラトン ノ イウ ゴトク ヒ ユー ガ ユー デ アル。",
        "You are a helpful assistant.<|endofprompt|>レキシ テキ セカイ ニ オイ テ ワ、カコ ワ タンニ スギサッ タ "
        "モノ デ ワ ナイ、プラトン ノ イウ ゴトク ヒ ユー ガ ユー デ アル。",
    ),
    # example.py, the CosyVoice2 cross-lingual and fine-grained texts
    (
        "And then later on, fully acquiring that company. So keeping management in line, interest in line with "
        "the asset that's coming into the family is a reason why sometimes we don't buy the whole thing.",
        "And then later on, fully acquiring that company. So keeping management in line, interest in line with "
        "the asset that's coming into the family is a reason why sometimes we don't buy the whole thing.",
    ),
    (
        "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。",
        "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。",
    ),
    # digits, dates and units
    ("Flight 247 leaves from gate 12.", "Flight two hundred and forty-seven leaves from gate twelve."),
    (
        "The company was founded in 2021 and hired 350 people by 2023",
        "The company was founded in two thousand and twenty-one and hired three hundred and fifty people by "
        "two thousand and twenty-three.",
    ),
    ("It costs $12.50, or 15% off the list price.", "It costs $twelve.fifty, or fifteen% off the list price."),
    (
        "Call 555-0123 between 9 and 5.",
        "Call five hundred and fifty-five-one hundred and twenty-three between nine and five.",
    ),
    ("The meeting starts at 3:45 PM on March 3.", "The meeting starts at three:forty-five PM on March three."),
    ("Version 2.0.5 is 3x faster over 10 km.", "Version two.zero.five is threex faster over ten km."),
    ("Wait... 3 more minutes!", "Wait. three more minutes!"),
    ('He said "I have 2 dogs." Then he left.', 'He said "I have two dogs." Then he left.'),
    (
        "Room 101 is on floor 1; room 1001 is on floor 10.",
        "Room one hundred and one is on floor one; room one thousand and one is on floor ten.",
    ),
    ("面积是10m²，长度是3米、", "面积是10m平方，长度是3米。"),
    ("这是（一个）【测试】 - 好的.", "这是一个测试-好的。"),
    ("今天是 hello world 的 第 2 天，", "今天是hello world的第2天。"),
]


def _needs_inflect(text: str) -> bool:
    return (
        any(c.isdecimal() for c in text)
        and not ("<|" in text and "|>" in text)
        and not any("\u4e00" <= c <= "\u9fff" for c in text)
    )


@pytest.mark.parametrize("text, expected", REFERENCE, ids=[t[:24] for t, _ in REFERENCE])
def test_normalize_matches_the_reference_frontend(text, expected):
    if _needs_inflect(text):
        pytest.importorskip("inflect")
    assert CosyVoice3TextFrontend().normalize(text) == expected


def test_a_decimal_point_in_chinese_text_stays_a_point():
    # the reference's fallback gives "我有3个苹果，价格是12。5元。"
    assert CosyVoice3TextFrontend().normalize("我有3个苹果，价格是12.5元") == "我有3个苹果，价格是12.5元。"
    assert CosyVoice3TextFrontend().normalize("第1.5节.第2节") == "第1.5节。第2节。"


def test_superscript_digits_are_not_spelled_out():
    pytest.importorskip("inflect")
    # inflect reads "²" as "zero"; the reference's fallback gives "ten mzero room."
    assert CosyVoice3TextFrontend().normalize("A 10 m² room") == "A ten m² room."


def test_a_digit_run_too_long_for_inflect_stays_as_written():
    pytest.importorskip("inflect")
    digits = "9" * 40
    assert CosyVoice3TextFrontend().normalize(f"Code {digits} and 7") == f"Code {digits} and seven."


@pytest.mark.parametrize(
    "text, lang, expected",
    [
        ("no stop", "en", "no stop."),
        ("Really?! Yes", "en", "Really? Yes."),
        ('"Go." she said', "en", '"Go." she said.'),
        ("没有句号", "zh", "没有句号。"),
        ("...", "en", ""),
        ("!?", "zh", ""),
    ],
)
def test_close_sentences_follows_the_reference_splitter(text, lang, expected):
    assert close_sentences(text, lang) == expected


def test_text_left_alone():
    frontend = CosyVoice3TextFrontend()
    # a <|...|> tag turns the reference's frontend off; so does nothing to speak
    tagged = f"{SYSTEM_PROMPT}{END_OF_PROMPT}Flight 247"
    assert frontend.normalize(tagged) == tagged
    assert frontend.normalize("...") == "..."
    assert frontend.normalize("  ") == "  "


def test_without_inflect_digits_pass_through_with_one_warning(monkeypatch, caplog):
    real_import = builtins.__import__

    def no_inflect(name, *args, **kwargs):
        if name == "inflect":
            raise ImportError("No module named 'inflect'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_inflect)
    frontend = CosyVoice3TextFrontend()
    with caplog.at_level(logging.WARNING):
        frontend.warmup()
        assert frontend.normalize("Flight 247") == "Flight 247."
        assert frontend.normalize("Gate 12") == "Gate 12."
    assert sum("inflect" in r.getMessage() for r in caplog.records) == 1


# -- through process_prompt ----------------------------------------------------------


class _Tokenizer:
    """One id per whitespace word; ``<|endofprompt|>`` is its trained id."""

    def __call__(self, text: str) -> torch.Tensor:
        ids: list[int] = []
        for i, part in enumerate(text.split(END_OF_PROMPT)):
            if i:
                ids.append(END_OF_PROMPT_ID)
            ids += [1000 + sum(map(ord, word)) % 1000 for word in part.split()]
        return torch.tensor(ids, dtype=torch.long)


@pytest.fixture
def model(tmp_path, monkeypatch) -> CosyVoice3Model:
    monkeypatch.setattr(CosyVoice3Model, "_build_text_tokenizer", lambda self: _Tokenizer())
    return CosyVoice3Model(model_path_hf=str(tmp_path))


def _clip(seconds: float = 1.0) -> torch.Tensor:
    t = torch.arange(int(seconds * 24_000)) / 24_000
    return 0.3 * torch.sin(2 * math.pi * 220 * t)


def _text_ids(out, prompt: str) -> torch.Tensor:
    ids = out.new_input_tensors[TEXT_INPUTS][0]
    return ids[_Tokenizer()(prompt).numel() :]


def test_process_prompt_tokenizes_the_normalized_text(model):
    pytest.importorskip("inflect")
    model.warmup_preprocess()
    tok = _Tokenizer()
    clip = {"audio_inputs": [_clip()]}
    zero = model.process_prompt("Flight 247", ["text", "audio"], ["audio"], tensors=clip, ref_text="Gate 12")
    # the transcript stays as written: the reference's callers write <|endofprompt|> into it
    prompt = f"{SYSTEM_PROMPT}{END_OF_PROMPT}Gate 12"
    assert torch.equal(zero.new_input_tensors[TEXT_INPUTS][0][: tok(prompt).numel()], tok(prompt))
    assert torch.equal(_text_ids(zero, prompt), tok("Flight two hundred and forty-seven."))

    cross = model.process_prompt("Flight 247", ["text", "audio"], ["audio"], tensors=clip)
    assert torch.equal(_text_ids(cross, f"{SYSTEM_PROMPT}{END_OF_PROMPT}"), tok("Flight two hundred and forty-seven."))

    instruct = model.process_prompt("Flight 247", ["text", "audio"], ["audio"], tensors=clip, instruct="Speak 2x")
    prompt = f"{SYSTEM_PROMPT} Speak 2x{END_OF_PROMPT}"
    assert torch.equal(_text_ids(instruct, prompt), tok("Flight two hundred and forty-seven."))


def test_text_frontend_false_sends_the_text_as_written(model):
    tok = _Tokenizer()
    out = model.process_prompt(
        "Flight 247", ["text", "audio"], ["audio"], tensors={"audio_inputs": [_clip()]}, text_frontend=False
    )
    assert torch.equal(_text_ids(out, f"{SYSTEM_PROMPT}{END_OF_PROMPT}"), tok("Flight 247"))
    with pytest.raises(ValueError, match="text_frontend must be true or false"):
        model.process_prompt(
            "Flight 247", ["text", "audio"], ["audio"], tensors={"audio_inputs": [_clip()]}, text_frontend="no"
        )
