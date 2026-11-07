"""CosyVoice3's text frontend against the reference's.

``REFERENCE`` holds what ``CosyVoiceFrontEnd.text_normalize`` (FunAudioLLM/
CosyVoice at 074ca6d, ``split=True``) returns with neither ttsfrd nor wetext
installed, its fallback, for the CosyVoice3 texts in the reference's
``example.py`` and for text whose digits read the same either way. Each
returned one piece.

``SPOKEN`` holds what the reference's default install (``wetext`` 0.0.4)
makes of English numbers, by category; the built-in rules read them the same.
``MONTH_CASE`` and ``DIFFERENT`` are where they deliberately do not, with
``wetext``'s reading kept for comparison. With ``wetext`` installed and its
grammars cached, ``test_wetext_reads_as_recorded`` checks those readings.
"""

from __future__ import annotations

import logging
import math
import re
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from mstar.api_server.openai import adapters
from mstar.api_server.openai.speech_chunking import split_sentences
from mstar.model.cosyvoice3.components import text_frontend
from mstar.model.cosyvoice3.components.english_normalizer import normalize_english
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
    # digits that read the same with and without wetext
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


# What the same reference, same install, returns for text mixing English and
# Chinese: it decides Chinese for the whole text, so no digit is spelled out and
# the English pieces take the Chinese rules (a "." becomes "。", blanks go).
# Each text with its pieces, as the reference's own splitter cuts them.
MIXED = [
    ("我今天买了3本书，一共花了$12。", ["我今天买了3本书，一共花了$12。"]),
    (
        "The meeting starts at 4:05 pm in room 101.我们在会议室见面，大概需要20分钟。",
        ["The meeting starts at 4:05 pm in room 101。我们在会议室见面，大概需要20分钟。"],
    ),
    (
        "2024年的销量是 1,000,000 件，Flight 247 leaves from gate 12 at 7:30. 请提前到达。",
        ["2024年的销量是1,000,000件，Flight 247 leaves from gate 12 at 7:30。请提前到达。"],
    ),
    (
        "Call 555-867-5309 now. 然后我们去吃饭，He was born in 1987 and she finished 21st.",
        ["Call 555-867-5309 now。然后我们去吃饭，He was born in 1987 and she finished 21st。"],
    ),
    ("Hello world 123 你好", ["Hello world 123你好。"]),
    (
        "Tickets cost $85 each. Growth hit 12% last year. 今天天气很好，我们一起去公园玩，然后回家做饭，"
        "晚上看电影，明天还要上班，所以要早点睡觉。 The 19th century saw 3 wars and 1,200 treaties; "
        "the year 2008 saw 5 more. 谢谢大家。",
        [
            "Tickets cost $85 each。 Growth hit 12% last year。今天天气很好，我们一起去公园玩，然后回家做饭，"
            "晚上看电影，明天还要上班，所以要早点睡觉。",
            "The 19th century saw 3 wars and 1,200 treaties; the year 2008 saw 5 more。谢谢大家。",
        ],
    ),
]

SPOKEN = [
    ("thousands", "The prize was 1,000,000 dollars.", "The prize was one million dollars."),
    (
        "thousands",
        "The population reached 8,045,311 in the census.",
        "The population reached eight million forty five thousand three hundred and eleven in the census.",
    ),
    ("thousands", "It is about 29,032 feet.", "It is about twenty nine thousand and thirty two feet."),
    ("cardinal", "Flight 247 leaves from gate 12.", "Flight two hundred and forty seven leaves from gate twelve."),
    ("cardinal", "The building has 102 floors.", "The building has one hundred and two floors."),
    ("cardinal", "We have 5000 units.", "We have five thousand units."),
    ("decimal", "Pi is about 3.14159.", "Pi is about three point one four one five nine."),
    ("decimal", "It is 0.5 now.", "It is zero point five now."),
    ("decimal", "About 1.5 million people.", "About one point five million people."),
    (
        "ordinal",
        "She finished 21st, he came 3rd and their cousin was 22nd.",
        "She finished twenty first, he came third and their cousin was twenty second.",
    ),
    ("ordinal", "1st 2nd 3rd 4th 101st 1000th", "first second third fourth one hundred first one thousandth"),
    ("ordinal", "The 19th century.", "The nineteenth century."),
    ("currency", "The house sold for $250,000.", "The house sold for two hundred fifty thousand dollars."),
    ("currency", "Pay $1 now.", "Pay one dollar now."),
    ("currency", "It costs €20 and £5.", "It costs twenty euros and five pounds."),
    ("currency", "Revenue was $3.2 billion.", "Revenue was three point two billion dollars."),
    ("currency", "Tickets cost $85 each.", "Tickets cost eighty five dollars each."),
    ("percent", "Growth hit 12.5% last year.", "Growth hit twelve point five percent last year."),
    (
        "percent",
        "Rates rose from 6.25% to 7.1%.",
        "Rates rose from six point two five percent to seven point one percent.",
    ),
    ("percent", "It is 100% done.", "It is one hundred percent done."),
    ("year", "He was born in 1987.", "He was born in nineteen eighty seven."),
    ("year", "In 2008, prices fell.", "In two thousand eight, prices fell."),
    ("year", "It opened in 2010.", "It opened in twenty ten."),
    ("year", "In 1900 the city was small.", "In nineteen hundred the city was small."),
    ("year", "The 2020 census.", "The twenty twenty census."),
    ("year", "Run 1500 meters.", "Run fifteen hundred meters."),
    ("time", "Meet me at 4:05 pm.", "Meet me at four o five PM."),
    ("time", "It is 5:00.", "It is five o'clock."),
    (
        "time",
        "The train leaves at 06:40 and arrives at 13:05.",
        "The train leaves at six forty and arrives at thirteen o five.",
    ),
    ("time", "The meeting is at 10:15 AM.", "The meeting is at ten fifteen AM."),
    ("time", "Flight 247 leaves at 7:30.", "Flight two hundred and forty seven leaves at seven thirty."),
    ("phone", "Call 555-867-5309 now.", "Call five five five, eight six seven, five three zero nine now."),
    ("phone", "Call (415) 555-0199 today.", "Call four one five, five five five, zero one nine nine today."),
    ("range", "Read pages 10-20.", "Read pages ten to twenty."),
    ("range", "Rooms 3-5 are open.", "Rooms three to five are open."),
    ("range", "The score was 3:2.", "The score was three to two."),
    ("negative", "The temperature fell to -12 degrees.", "The temperature fell to negative twelve degrees."),
    ("negative", "-17", "negative seventeen"),
    (
        "fraction",
        "Add 1/2 cup of milk and 3/4 cup of sugar.",
        "Add one half cup of milk and three quarters cup of sugar.",
    ),
    ("fraction", "Use 2 1/2 cups of flour.", "Use two and a half cups of flour."),
    ("fraction", "1/3 and 5/8 and 2/3.", "one third and five eighths and two thirds."),
    ("unit", "It weighs 2.5 kg.", "It weighs two point five kilograms."),
    ("unit", "The limit is 65 mph.", "The limit is sixty five miles per hour."),
    ("unit", "It is 30°C outside.", "It is thirty degrees Celsius outside."),
    ("unit", "It weighs 5 lbs.", "It weighs five pounds."),
    ("alphanumeric", "Model GPT4 in Q4 at gate B12.", "Model GPT four in Q four at gate B twelve."),
    ("alphanumeric", "Use 3 x 4.", "Use three times four."),
    ("digits", "The code is 4 8 1 5.", "The code is four eight one five."),
    ("other", "Top 10.", "Top ten."),
]

# numeric dates: the same words, the month capitalized
MONTH_CASE = [
    (
        "The concert on 12/25/2023 starts soon.",
        "The concert on the twenty fifth of December twenty twenty three starts soon.",
        "The concert on the twenty fifth of december twenty twenty three starts soon.",
    ),
    (
        "It was 2023-10-19.",
        "It was the nineteenth of October twenty twenty three.",
        "It was the nineteenth of october twenty twenty three.",
    ),
]

# (text, the rules' reading, wetext's reading): where wetext is wrong or raises, or reorders
# a date written with a month name
DIFFERENT = [
    (
        "Your appointment is on Monday, March 3rd, at noon.",
        "Your appointment is on Monday, March third, at noon.",
        "Your appointment is on Monday, the third of march, at noon.",
    ),
    (
        "The meeting is on March 3, 2024.",
        "The meeting is on March third, twenty twenty four.",
        "The meeting is on the third of march, twenty twenty four.",
    ),
    (
        "His score went from 1,180 to 1,420.",
        "His score went from one thousand one hundred and eighty to one thousand four hundred and twenty.",
        "His score went from thousand one hundred and eighty to thousand four hundred and twenty.",
    ),
    (
        "It costs $4.99 today.",
        "It costs four dollars and ninety nine cents today.",
        "It costs four point nine nine dollars today.",
    ),
    (
        "She paid $12.50 for parking.",
        "She paid twelve dollars and fifty cents for parking.",
        "She paid twelve point five dollars for parking.",
    ),
    (
        "The fare is $1,299.99.",
        "The fare is one thousand two hundred and ninety nine dollars and ninety nine cents.",
        "The fare is thousand two hundred and ninety nine point nine nine dollars.",
    ),
    ("It costs $0.01.", "It costs one cent.", "It costs zero point oh one dollars."),
    ("It was built in 1905.", "It was built in nineteen oh five.", "It was built in nineteen zero five."),
    (
        "The 1990s were fun.",
        "The nineteen nineties were fun.",
        "The one thousand nine hundred and ninety seconds were fun.",
    ),
    (
        "He was born on July 4, 1976, and moved.",
        "He was born on July fourth, nineteen seventy six, and moved.",
        "He was born on the fourth of july, nineteen seventy six and moved.",
    ),
    ("It ships on 11/02.", "It ships on the second of November.", "It ships on eleven slash oh two."),
    (
        "Call 555-0199 today.",
        "Call five five five, zero one nine nine today.",
        "Call five hundred and fifty five-oh one hundred and ninety nine today.",
    ),
    (
        "The years 2019-2021 were hard.",
        "The years twenty nineteen to twenty twenty one were hard.",
        "The years two thousand and nineteen to two thousand and twenty one were hard.",
    ),
    ("1,2,3,4,5", "one, two, three, four, five", "one,two,three,four,five"),
    ("Version 2.10.3 is out.", "Version two point ten point three is out.", "Version two.ten point three is out."),
    ("Item #5.", "Item number five.", "<raises AssertionError>"),
]


@pytest.mark.parametrize("text, expected", REFERENCE, ids=[t[:24] for t, _ in REFERENCE])
def test_normalize_matches_the_reference_frontend(text, expected):
    assert CosyVoice3TextFrontend().normalize(text) == expected


@pytest.mark.parametrize("category, text, expected", SPOKEN, ids=[f"{c}-{t[:20]}" for c, t, _ in SPOKEN])
def test_numbers_read_as_the_reference_reads_them(category, text, expected):  # noqa: ARG001
    assert normalize_english(text) == expected
    assert CosyVoice3TextFrontend().normalize(text) == close_sentences(expected, "en")


@pytest.mark.parametrize("text, expected, wetext", MONTH_CASE, ids=[t[:24] for t, _, _ in MONTH_CASE])
def test_a_month_keeps_its_case(text, expected, wetext):
    assert normalize_english(text) == expected
    assert expected.lower() == wetext.lower()


@pytest.mark.parametrize("text, expected, wetext", DIFFERENT, ids=[t[:24] for t, _, _ in DIFFERENT])
def test_where_the_rules_read_numbers_differently_from_wetext(text, expected, wetext):
    assert normalize_english(text) == expected
    assert expected != wetext


def test_wetext_reads_as_recorded():
    frontend = CosyVoice3TextFrontend()
    pytest.importorskip("wetext")
    frontend.warmup()
    if not frontend._wetext:
        pytest.skip("wetext could not load its grammars")
    for text, wetext in [(t, w) for _, t, w in SPOKEN] + [(t, w) for t, _, w in MONTH_CASE + DIFFERENT]:
        if not wetext.startswith("<raises"):
            assert frontend.normalize(text) == close_sentences(wetext, "en"), text
    # wetext raises on this one; the rules read it
    assert frontend.normalize("Item #5.") == "Item number five."


def test_the_rules_take_linear_time():
    for text in ("7" * 100_000, "1," * 50_000, "1-" * 50_000, "1:" * 50_000, "$1 " * 30_000, "a1" * 50_000):
        start = time.perf_counter()
        normalize_english(text)
        assert time.perf_counter() - start < 5, text[:8]


def test_a_run_past_the_largest_scale_is_read_digit_by_digit():
    digits = "9" * 40
    assert normalize_english(f"Code {digits} and 7") == "Code " + " ".join(["nine"] * 40) + " and seven"
    assert normalize_english("The national debt is 34,567,891,234,567 dollars.").startswith(
        "The national debt is thirty four trillion five hundred sixty seven billion"
    )


_19_DIGITS = "1234567890123456789"
_19_SPOKEN = "one two three four five six seven eight nine zero one two three four five six seven eight nine"


@pytest.mark.parametrize(
    "text, spoken",
    [
        (_19_DIGITS, _19_SPOKEN),
        (f"${_19_DIGITS}", f"{_19_SPOKEN} dollars"),
        (f"${_19_DIGITS}.50", f"{_19_SPOKEN} dollars and fifty cents"),
        (f"{_19_DIGITS}.5", f"{_19_SPOKEN} point five"),
        (f"{_19_DIGITS}th", _19_SPOKEN.removesuffix("nine") + "ninth"),
        (f"{_19_DIGITS} 1/2 cups", f"{_19_SPOKEN} and a half cups"),
        (f"{_19_DIGITS}%", f"{_19_SPOKEN} percent"),
        # the largest number the scales name is still read in words
        (
            "999,999,999,999,999,999th",
            "nine hundred ninety nine quadrillion nine hundred ninety nine trillion nine hundred ninety nine "
            "billion nine hundred ninety nine million nine hundred ninety nine thousand nine hundred ninety ninth",
        ),
    ],
    ids=["integer", "money", "money-cents", "decimal", "ordinal", "mixed-fraction", "percent", "largest"],
)
def test_a_number_past_the_largest_scale_is_read_digit_by_digit_everywhere(text, spoken):
    assert normalize_english(text) == spoken


@pytest.mark.parametrize(
    "text, end",
    [
        ("{}.5", "one one point five"),
        ("${}", "one one dollars"),
        ("${}.25", "one one dollars and twenty five cents"),
        ("{}th", "one one first"),
        ("{} 1/2", "one one and a half"),
        ("{}/2", "one one slash two"),
        ("$1-{}", "one one dollars"),
    ],
)
def test_a_run_too_long_for_int_is_read_digit_by_digit(text, end):
    """Python's int() refuses more than 4,300 digits."""
    spoken = normalize_english(text.format("1" * 5000))
    assert spoken.endswith(end)
    assert spoken.count("one ") >= 4999


@pytest.mark.parametrize("text", ["0" * 5000 + "1/2", "1/" + "0" * 5000 + "2", "3 " + "0" * 5000 + "1/2"])
def test_a_fraction_with_thousands_of_leading_zeros_does_not_raise(text):
    """Leading zeros were counted off before the length check but not before int()."""
    assert normalize_english(text)
    assert CosyVoice3TextFrontend().normalize(text)



def test_a_long_run_of_thousands_groups_or_commas_takes_linear_time():
    """Each took over 5 s at this size: a rule could start after every comma and rescan the rest."""
    frontend = CosyVoice3TextFrontend()
    for normalize, text in [
        (normalize_english, "1" + ",234" * 10_000),
        (normalize_english, "1,23" * 10_000),
        (lambda t: frontend.normalize(t, chinese=True), "好" + "，" * 40_000),
        # each closing quote used to copy the whole sentence before it
        (frontend.normalize, "a" * 200_000 + '."' * 200_000),
    ]:
        start = time.perf_counter()
        normalize(text)
        assert time.perf_counter() - start < 1, text[:8]


@pytest.mark.parametrize(
    "text, spoken",
    [
        ("100-1000 users", "one hundred to one thousand users"),
        ("500-1000 users", "five hundred to one thousand users"),
        ("pages 120-1500", "pages one hundred and twenty to fifteen hundred"),
        ("$100-1000", "one hundred to one thousand dollars"),
        ("$10-$20 each", "ten to twenty dollars each"),
    ],
)
def test_a_range_of_three_and_four_digits_is_not_a_phone_number(text, spoken):
    assert normalize_english(text) == spoken


@pytest.mark.parametrize(
    "text, spoken",
    [
        ("555-1234", "five five five, one two three four"),
        ("call 555-1234", "call five five five, one two three four"),
        ("Call 555-0199 today.", "Call five five five, zero one nine nine today."),
        ("(415) 555-1234", "four one five, five five five, one two three four"),
        ("415-555-1234", "four one five, five five five, one two three four"),
    ],
)
def test_phone_numbers_are_still_read_digit_by_digit(text, spoken):
    assert normalize_english(text) == spoken


def test_text_without_digits_is_unchanged():
    assert normalize_english("Prices: $, %, # and ½.") == "Prices: $, %, # and ½."
    assert normalize_english("about 50 percent more") == "about fifty percent more"


def test_a_decimal_point_in_chinese_text_stays_a_point():
    # the reference's fallback gives "我有3个苹果，价格是12。5元。"
    assert CosyVoice3TextFrontend().normalize("我有3个苹果，价格是12.5元") == "我有3个苹果，价格是12.5元。"
    assert CosyVoice3TextFrontend().normalize("第1.5节.第2节") == "第1.5节。第2节。"


def test_superscript_digits_are_not_spelled_out():
    # inflect, the reference's fallback, reads "²" as "zero": "ten mzero room."
    assert CosyVoice3TextFrontend().normalize("A 10 m² room") == "A ten m² room."


@pytest.mark.parametrize(
    "text, lang, expected",
    [
        ("no stop", "en", "no stop."),
        ("Really?! Yes", "en", "Really? Yes."),
        ('"Go." she said', "en", '"Go." she said.'),
        ("没有句号", "zh", "没有句号。"),
        ("...", "en", ""),
        ("!?", "zh", ""),
        ("", "zh", ""),
        (" 。", "zh", ""),
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


@pytest.mark.parametrize("text", ["（）", "()", "【】", "（【】）", "——", "``"])
def test_a_piece_of_only_brackets_is_left_as_written(text):
    """The Chinese rules drop brackets, so nothing is left to speak; like "...", it is kept."""
    frontend = CosyVoice3TextFrontend()
    assert frontend.normalize(text, chinese=True) == text
    assert frontend.normalize(text) == text


def test_chinese_text_that_ends_in_a_piece_of_only_brackets(model):
    sentence = "今天天气很好，我们一起去公园玩，然后回家做饭，晚上看电影，明天还要上班，所以要早点睡觉，"
    text = (sentence + "你说好吗，我觉得这样安排挺好的呀？") * 3
    pieces, kwargs = _served_pieces(text + "（）" * 12)
    assert pieces[-1] == "（）" * 12
    frontend = CosyVoice3TextFrontend()
    spoken = [frontend.normalize(p, chinese=kwargs["text_has_chinese"]) for p in pieces]
    assert spoken[:-1] == [frontend.normalize(p) for p in _served_pieces(text)[0]]
    assert spoken[-1] == pieces[-1]
    out = model.process_prompt(pieces[-1], ["text", "audio"], ["audio"], tensors={"audio_inputs": [_clip()]}, **kwargs)
    assert torch.equal(_text_ids(out, f"{SYSTEM_PROMPT}{END_OF_PROMPT}"), _Tokenizer()(pieces[-1]))


def test_text_that_normalizes_to_nothing_is_a_value_error(model, monkeypatch):
    monkeypatch.setattr(model.text_frontend, "normalize", lambda text, chinese=False: "")
    with pytest.raises(ValueError, match="no text to speak"):
        model.process_prompt("（）", ["text", "audio"], ["audio"], tensors={"audio_inputs": [_clip()]})


# -- wetext -----------------------------------------------------------------------------


class _FakeWetext:
    """``wetext.Normalizer``: reads "Gate 12" and "12.5元", raises on "#"."""

    made: list[tuple[str, str, str]] = []

    def __init__(self, tagger, verbalizer, lang):
        self.lang = lang
        _FakeWetext.made.append((tagger, verbalizer, lang))

    def normalize(self, text):
        assert "#" not in text
        return text.replace("Gate 12", "Gate twelve").replace("12.5元", "十二点五元")


def _fake_wetext(monkeypatch, snapshot):
    _FakeWetext.made = []
    monkeypatch.setitem(sys.modules, "wetext", SimpleNamespace(Normalizer=_FakeWetext))
    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=snapshot))


def test_wetext_runs_before_the_rules_when_installed(monkeypatch):
    calls = []

    def snapshot(repo, local_files_only=False):
        calls.append((repo, local_files_only))
        return "/cache/wetext"

    _fake_wetext(monkeypatch, snapshot)
    frontend = CosyVoice3TextFrontend()
    frontend.warmup()
    assert calls == [("pengzhendong/wetext", True)]
    assert _FakeWetext.made == [
        ("/cache/wetext/en/tn/tagger.fst", "/cache/wetext/en/tn/verbalizer.fst", "en"),
        ("/cache/wetext/zh/tn/tagger.fst", "/cache/wetext/zh/tn/verbalizer.fst", "zh"),
    ]
    # what wetext leaves (here "4:05") the rules read
    assert frontend.normalize("Gate 12 at 4:05") == "Gate twelve at four o five."
    assert frontend.normalize("价格是12.5元") == "价格是十二点五元。"
    # a text wetext raises on, or one too long for it, goes to the rules alone
    assert frontend.normalize("Gate 12, item #5") == "Gate twelve, item number five."
    long = "Gate 12 " + "word " * text_frontend.WETEXT_MAX_CHARS
    assert frontend.normalize(long).startswith("Gate twelve word")
    assert len(_FakeWetext.made) == 2


def test_wetext_grammars_come_from_the_cache_before_the_network(monkeypatch):
    calls = []

    def snapshot(repo, local_files_only=False):
        calls.append(local_files_only)
        if local_files_only:
            raise FileNotFoundError("not cached")
        return "/downloaded"

    _fake_wetext(monkeypatch, snapshot)
    frontend = CosyVoice3TextFrontend()
    frontend.warmup()
    assert calls == [True, False]
    assert [t for t, _, _ in _FakeWetext.made] == ["/downloaded/en/tn/tagger.fst", "/downloaded/zh/tn/tagger.fst"]


def test_wetext_that_cannot_load_leaves_the_rules_with_one_warning(monkeypatch, caplog):
    def snapshot(repo, local_files_only=False):
        raise OSError("403: rate limited")

    _fake_wetext(monkeypatch, snapshot)
    frontend = CosyVoice3TextFrontend()
    with caplog.at_level(logging.WARNING):
        frontend.warmup()
        frontend.warmup()
        assert frontend.normalize("Gate 12") == "Gate twelve."
    assert sum("wetext could not load" in r.getMessage() for r in caplog.records) == 1


def test_without_wetext_the_rules_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "wetext", None)
    frontend = CosyVoice3TextFrontend()
    frontend.warmup()
    assert frontend.normalize("Gate 12") == "Gate twelve."
    # text with Chinese keeps its digits, as the reference's fallback does
    assert frontend.normalize("价格是12.5元") == "价格是12.5元。"


def test_normalize_never_loads_wetext(monkeypatch):
    def snapshot(repo, local_files_only=False):
        raise AssertionError("normalize reached ModelScope")

    _fake_wetext(monkeypatch, snapshot)
    assert CosyVoice3TextFrontend().normalize("Gate 12") == "Gate twelve."
    assert _FakeWetext.made == []


# -- piece sizing in the API server ----------------------------------------------------

_IDS = ", ".join(str(48213907 + 7919 * i) for i in range(35))[:319]  # about 31 8-digit IDs
_tokens = adapters.CosyVoice3Adapter._estimated_tokens


class _TenfoldDigits:
    """A frontend that spells each digit as ten characters."""

    def normalize(self, text: str) -> str:
        return re.sub(r"\d", "x" * 10, text)


def test_pieces_are_sized_on_the_spelled_out_text(monkeypatch):
    """The reference splits after normalizing, so its pieces never grow; a piece
    of digits that fits 80 tokens as written must still fit once spelled out."""
    monkeypatch.setattr(adapters.CosyVoice3Adapter, "_frontend", _TenfoldDigits())
    adapter = adapters.CosyVoice3Adapter()
    pieces = adapter.split_speech_text(_IDS)
    assert len(pieces) > 1
    assert all(_tokens(_TenfoldDigits().normalize(p)) <= 80 for p in pieces)
    assert " ".join(pieces).split() == _IDS.split()
    words = "Several members raised concerns about the budget. " * 6
    assert adapter.split_speech_text(words) == split_sentences(words, max_chars=320, min_chars=80)


def test_pieces_are_sized_as_written_with_the_frontend_off(monkeypatch):
    from types import SimpleNamespace

    from mstar.api_server.openai.serving_speech import _plan_chunks

    monkeypatch.setattr(adapters.CosyVoice3Adapter, "_frontend", _TenfoldDigits())
    adapter = adapters.CosyVoice3Adapter()
    text = ", ".join(["48213907"] * 200)
    as_written = split_sentences(text, max_chars=320, min_chars=80)
    off = SimpleNamespace(model_extra={"text_frontend": False})
    assert _plan_chunks(off, adapter, text) == as_written
    assert len(_plan_chunks(SimpleNamespace(model_extra={}), adapter, text[:600])) > len(as_written[:2])


def test_an_input_over_the_piece_cap_is_not_normalized_piece_by_piece(monkeypatch):
    calls = []

    class Counting(_TenfoldDigits):
        def normalize(self, text):
            calls.append(text)
            return super().normalize(text)

    monkeypatch.setattr(adapters.CosyVoice3Adapter, "_frontend", Counting())
    adapter = adapters.CosyVoice3Adapter()
    pieces = adapter.split_speech_text("7 " * 50000)
    assert len(pieces) > adapter.speech_chunk_max_pieces
    assert calls == []
    # past the cap partway: the rest is left as split, it is refused either way
    calls.clear()
    pieces = adapter.split_speech_text(" ".join(["48213907"] * 600))
    assert len(pieces) > adapter.speech_chunk_max_pieces
    assert len(calls) < 3 * adapter.speech_chunk_max_pieces


def test_digit_heavy_pieces_fit_the_model(monkeypatch):
    monkeypatch.setattr(adapters.CosyVoice3Adapter, "_frontend", None)
    frontend = CosyVoice3TextFrontend()
    pieces = adapters.CosyVoice3Adapter().split_speech_text(_IDS)
    assert len(frontend.normalize(_IDS)) > 2000, "the example no longer grows when spelled out"
    assert all(_tokens(frontend.normalize(p)) <= 80 for p in pieces)
    assert " ".join(pieces).split() == _IDS.split()


_NON_LATIN = [
    "회의는 2024년 3월 15일 오후 7시 30분에 시작합니다. 참석자는 125명이고 예산은 4,500,000원입니다. ",
    "かいぎは 2024ねん 3がつ 15にち ごご 7じ 30ぷんに はじまります。さんかしゃは 125にんです。",
    "Встреча начнётся 15 марта 2024 года в 19:30. Участников 125, бюджет 4 500 000 рублей. ",
]


@pytest.mark.parametrize("sentences", _NON_LATIN, ids=["hangul", "kana", "cyrillic"])
def test_text_in_other_scripts_keeps_its_digits_and_is_sized_as_written(monkeypatch, sentences):
    """Read as English words, digits made served kana and Russian number text
    worse than leaving them to the LM, so text whose letters are mostly not
    Latin is left as written, unlike the reference."""
    monkeypatch.setattr(adapters.CosyVoice3Adapter, "_frontend", None)
    frontend = CosyVoice3TextFrontend()
    text = sentences * 6
    assert frontend.normalize(text) == text
    pieces = adapters.CosyVoice3Adapter().split_speech_text(text)
    assert all(frontend.normalize(p) == p and _tokens(p) <= 80 for p in pieces)
    assert re.sub(r"\s", "", "".join(pieces)) == re.sub(r"\s", "", text)


@pytest.mark.parametrize(
    "text, spoken",
    [
        ("Café 12 opens at 9:45.", "Café twelve opens at nine forty five."),
        ("Größe 12 kostet 4 Euro.", "Größe twelve kostet four Euro."),
        ("Room 2024.", "Room twenty twenty four."),
        ("2024.", "twenty twenty four."),
    ],
)
def test_latin_script_text_reads_its_digits(text, spoken):
    assert CosyVoice3TextFrontend().normalize(text) == spoken


@pytest.mark.parametrize(
    "text, spoken",
    [
        # most letters decide: an English word in kana or Cyrillic text keeps the digits
        ("かいぎは Zoomで 3じに はじまります。", "かいぎは Zoomで 3じに はじまります。"),
        ("Встреча с Microsoft в 19:30.", "Встреча с Microsoft в 19:30."),
        # and a kana or Hangul word in English text does not
        ("Say ありがとう 3 times.", "Say ありがとう three times."),
        ("The word 안녕 has 2 syllables.", "The word 안녕 has two syllables."),
    ],
)
def test_mixed_scripts_follow_most_of_their_letters(text, spoken):
    assert CosyVoice3TextFrontend().normalize(text) == spoken


@pytest.mark.parametrize(
    "text",
    [
        "The meeting starts on March 15, 2024 at 7:30 PM. There are 125 attendees and the budget is "
        "$4,500,000. Please arrive 10 minutes early and bring your ID. Phase 1 is 80% complete, and the "
        "rest should be done by June 30. Phase 2 will add 36 engineers at a cost of $1,299 each per week. "
        "Flight 247 leaves Gate 12 at 9:45, and order numbers 48213907, 55120934 and 61028711 ship on "
        "the 3rd. For questions, call 555-0199.",
        _IDS,
    ],
    ids=["english", "ids"],
)
def test_spelled_out_pieces_are_even_and_keep_numbers_whole(monkeypatch, text):
    """A re-split piece is neither a short tail ("3rd.") nor a cut number ("9:" and "45")."""
    monkeypatch.setattr(adapters.CosyVoice3Adapter, "_frontend", None)
    frontend = CosyVoice3TextFrontend()
    pieces = adapters.CosyVoice3Adapter().split_speech_text(text)
    assert len(pieces) > 1
    assert " ".join(pieces) == text
    assert all(20 <= _tokens(frontend.normalize(p)) <= 80 for p in pieces)


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


def test_process_prompt_tokenizes_the_normalized_text(model, monkeypatch):
    monkeypatch.setitem(sys.modules, "wetext", None)
    model.warmup_preprocess()
    tok = _Tokenizer()
    clip = {"audio_inputs": [_clip()]}
    zero = model.process_prompt("Flight 247", ["text", "audio"], ["audio"], tensors=clip, ref_text="Gate 12")
    # the transcript stays as written: the reference's callers write <|endofprompt|> into it
    prompt = f"{SYSTEM_PROMPT}{END_OF_PROMPT}Gate 12"
    assert torch.equal(zero.new_input_tensors[TEXT_INPUTS][0][: tok(prompt).numel()], tok(prompt))
    assert torch.equal(_text_ids(zero, prompt), tok("Flight two hundred and forty seven."))

    cross = model.process_prompt("Flight 247", ["text", "audio"], ["audio"], tensors=clip)
    assert torch.equal(_text_ids(cross, f"{SYSTEM_PROMPT}{END_OF_PROMPT}"), tok("Flight two hundred and forty seven."))

    instruct = model.process_prompt("Flight 247", ["text", "audio"], ["audio"], tensors=clip, instruct="Speak 2x")
    prompt = f"{SYSTEM_PROMPT} Speak 2x{END_OF_PROMPT}"
    assert torch.equal(_text_ids(instruct, prompt), tok("Flight two hundred and forty seven."))


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


# -- text with English and Chinese ----------------------------------------------------


def _speech_request(text: str, **extra) -> SimpleNamespace:
    return SimpleNamespace(input=text, voice="zero_shot", model_extra=dict(extra))


def _served_pieces(text: str, **extra) -> tuple[list[str], dict]:
    """The pieces the API server submits for ``text`` and the kwargs each carries."""
    adapter = adapters.CosyVoice3Adapter()
    req = _speech_request(text, **extra)
    return adapter.split_speech_text(text, req), adapter.speech_to_request(req, None).model_kwargs


def _squeeze(text: str) -> str:
    return "".join(text.split())


@pytest.mark.parametrize("text, reference", MIXED, ids=[t[:24] for t, _ in MIXED])
def test_mixed_text_reads_as_the_reference_reads_it(text, reference):
    pieces, kwargs = _served_pieces(text)
    assert kwargs["text_has_chinese"] is True
    frontend = CosyVoice3TextFrontend()
    spoken = [frontend.normalize(p, chinese=kwargs["text_has_chinese"]) for p in pieces]
    # the pieces are cut apart the same way only when the text is short; what is said is the same
    assert _squeeze("".join(spoken)) == _squeeze("".join(reference))
    assert not re.search(r"\b(one|two|three|four|five|hundred|thousand|million|eleven|seven)\b", " ".join(spoken))


@pytest.mark.parametrize("text, reference", MIXED[:5], ids=[t[:24] for t, _ in MIXED[:5]])
def test_mixed_text_through_process_prompt_tokenizes_the_reference_text(model, text, reference):
    (piece,), kwargs = _served_pieces(text)
    out = model.process_prompt(
        piece, ["text", "audio"], ["audio"], tensors={"audio_inputs": [_clip()]}, **kwargs
    )
    assert torch.equal(_text_ids(out, f"{SYSTEM_PROMPT}{END_OF_PROMPT}"), _Tokenizer()(reference[0]))


def test_an_english_piece_of_mixed_text_keeps_its_digits_where_alone_it_is_spelled_out():
    frontend = CosyVoice3TextFrontend()
    piece = "Flight 247 leaves at 7:30."
    assert frontend.normalize(piece) == "Flight two hundred and forty seven leaves at seven thirty."
    assert frontend.normalize(piece, chinese=True) == "Flight 247 leaves at 7:30。"


def test_pure_english_text_carries_no_flag_and_pure_chinese_text_reads_as_before():
    frontend = CosyVoice3TextFrontend()
    pieces, kwargs = _served_pieces("Flight 247 leaves at 7:30. Call 555-867-5309 now.")
    assert "text_has_chinese" not in kwargs
    assert frontend.normalize(pieces[0]) == (
        "Flight two hundred and forty seven leaves at seven thirty. "
        "Call five five five, eight six seven, five three zero nine now."
    )
    for text in ("航班247次在7:30起飞。", "面积是10m²，长度是3米、", "今天是 hello world 的 第 2 天，"):
        pieces, kwargs = _served_pieces(text)
        assert kwargs["text_has_chinese"] is True  # every piece has Chinese: the flag changes nothing
        assert [frontend.normalize(p, chinese=True) for p in pieces] == [frontend.normalize(p) for p in pieces]


def test_the_client_cannot_set_text_has_chinese():
    _, kwargs = _served_pieces("Flight 247 leaves at 7:30.", text_has_chinese=True)
    assert "text_has_chinese" not in kwargs
    _, kwargs = _served_pieces("Flight 247 在7:30起飞。", text_has_chinese=False)
    assert kwargs["text_has_chinese"] is True


def test_text_has_chinese_must_be_a_flag(model):
    with pytest.raises(ValueError, match="text_has_chinese must be true or false"):
        model.process_prompt(
            "Flight 247", ["text", "audio"], ["audio"], tensors={"audio_inputs": [_clip()]}, text_has_chinese="yes"
        )


def test_text_has_chinese_does_not_touch_text_sent_as_written(model):
    out = model.process_prompt(
        "Flight 247", ["text", "audio"], ["audio"], tensors={"audio_inputs": [_clip()]},
        text_frontend=False, text_has_chinese=True,
    )
    assert torch.equal(_text_ids(out, f"{SYSTEM_PROMPT}{END_OF_PROMPT}"), _Tokenizer()("Flight 247"))


def test_a_chinese_piece_heavy_in_digits_is_cut_before_wetext_can_spell_it_past_the_limit():
    """wetext spells 98765432 in 15 characters, so 300 characters of such text
    would reach the model as 531; a digit counts twice against the piece cap,
    so the piece is cut at its commas instead."""
    from mstar.api_server.openai import adapters

    text = "订单" + ("98765432，" * 34)[:297] + "。"
    assert len(text) == 300
    pieces = adapters.CosyVoice3Adapter().split_speech_text(text)
    assert len(pieces) > 1 and "".join(pieces) == text
    assert all(len(p) + sum(c.isdigit() for c in p) <= 300 for p in pieces)
