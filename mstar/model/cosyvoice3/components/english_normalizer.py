"""English text normalization for CosyVoice3: numbers, money, dates and times as words.

The reference frontend runs ``wetext``'s English grammar (a WeTextProcessing
runtime) before spelling out what digits remain. This is the rule set that
stands in for it when ``wetext`` is not installed or cannot load its grammars,
and that reads the digits ``wetext`` leaves behind. It follows ``wetext``'s
spoken forms ("1,000,000" -> "one million", "4:05" -> "four o five", "1987"
-> "nineteen eighty seven", "21st" -> "twenty first", "3/14/2021" -> "the
fourteenth of March twenty twenty one", "555-867-5309" -> "five five five,
eight six seven, five three zero nine") with these differences, where its
reading is wrong or crashes (``wetext`` 0.0.4's reading in brackets):

- "1,299" is "one thousand two hundred and ninety nine" ("thousand two
  hundred...": the "one" is lost from 1,000 to 1,999 with a separator);
- "$4.99" is "four dollars and ninety nine cents" ("four point nine nine
  dollars"), "$0.01" "one cent";
- "1905" is "nineteen oh five" ("nineteen zero five"), and "2019-2021" a range
  of years ("two thousand and nineteen to...");
- "11/02" is "the second of November" ("eleven slash oh two"), "555-0199" a
  phone number ("five hundred and fifty five-oh..."), "1990s" "nineteen
  nineties" ("...ninety seconds"), "#5" "number five" (raises), "2.10.3" "two
  point ten point three" ("two.ten point three"), "1,2,3" "one, two, three";
- a date written with a month name keeps its word order ("March 3, 2024"
  -> "March third, twenty twenty four", not "the third of march, ..."), and a
  month keeps its case.

Every rule needs a digit, so text without digits comes back unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen"
).split()
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = ["", "thousand", "million", "billion", "trillion", "quadrillion"]
_MONTHS = "January February March April May June July August September October November December".split()
_ORDINAL_WORDS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth"}
_ORDINAL_WORDS |= {"nine": "ninth", "twelve": "twelfth"}
# (singular, plural) for the main unit, then for the hundredth
_CURRENCIES = {
    "$": (("dollar", "dollars"), ("cent", "cents")),
    "€": (("euro", "euros"), ("cent", "cents")),
    "£": (("pound", "pounds"), ("penny", "pence")),
}
_UNITS = {
    "kg": ("kilogram", "kilograms"),
    "km": ("kilometer", "kilometers"),
    "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"),
    "mph": ("mile per hour", "miles per hour"),
    "km/h": ("kilometer per hour", "kilometers per hour"),
    "kph": ("kilometer per hour", "kilometers per hour"),
    "lb": ("pound", "pounds"),
    "lbs": ("pound", "pounds"),
    "oz": ("ounce", "ounces"),
    "°C": ("degree Celsius", "degrees Celsius"),
    "°F": ("degree Fahrenheit", "degrees Fahrenheit"),
    "°": ("degree", "degrees"),
}
# more digits than the largest scale names are read one digit at a time
_MAX_CARDINAL_DIGITS = 3 * len(_SCALES)

# a number as written: thousands groups or a plain run, then an optional decimal part
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
# not inside a thousands group: a rule that may start after a comma would
# otherwise scan the rest of "1,234,567,..." again from every comma
_NOT_IN_GROUP = r"(?!(?<=\d,)\d{3})"
_MONTH_WORD = r"(?:" + "|".join(_MONTHS) + r")"
_DAY_SUFFIX = r"(?:st|nd|rd|th)?"


def _below_hundred(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (" " + _ONES[ones] if ones else "")


def _below_thousand(n: int, with_and: bool) -> str:
    hundreds, rest = divmod(n, 100)
    words = []
    if hundreds:
        words.append(_ONES[hundreds] + " hundred")
    if rest:
        if hundreds and with_and:
            words.append("and")
        words.append(_below_hundred(rest))
    return " ".join(words)


def cardinal(n: int | str) -> str:
    """``wetext``'s cardinal: "and" only inside the last group of three
    (247 -> "two hundred and forty seven", 29,032 -> "twenty nine thousand and
    thirty two", 250,000 -> "two hundred fifty thousand"). A number past
    the largest scale is read digit by digit. ``n`` can be the digits as
    written, separators included, so a run too long for ``int()`` is never
    converted."""
    if isinstance(n, str):
        run = n.replace(",", "").lstrip("0")
        if len(run) > _MAX_CARDINAL_DIGITS:
            return digits(run)
        n = int(run or "0")
    if n == 0:
        return "zero"
    if n >= 1000 ** len(_SCALES):
        return digits(str(n))
    groups = []
    while n:
        n, group = divmod(n, 1000)
        groups.append(group)
    words = []
    for scale in range(len(groups) - 1, -1, -1):
        group = groups[scale]
        if not group:
            continue
        if scale:
            words.append(f"{_below_thousand(group, with_and=False)} {_SCALES[scale]}")
            continue
        if len(groups) > 1 and group < 100:
            words.append("and")
        words.append(_below_thousand(group, with_and=True))
    return " ".join(words)


def ordinal(n: int | str) -> str:
    """``wetext``'s ordinal, without the cardinal's "and" (101 -> "one hundred first")."""
    head, _, last = cardinal(n).replace(" and", "").rpartition(" ")
    if last in _ORDINAL_WORDS:
        last = _ORDINAL_WORDS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return f"{head} {last}" if head else last


def digits(run: str, zero: str = "zero") -> str:
    return " ".join(zero if c == "0" else _ONES[int(c)] for c in run)


def year(n: int) -> str:
    """1987 -> "nineteen eighty seven", 2008 -> "two thousand eight", 1900 ->
    "nineteen hundred", 1905 -> "nineteen oh five", 2010 -> "twenty ten"."""
    high, low = divmod(n, 100)
    if 2000 <= n <= 2009:
        return "two thousand" + (" " + _ONES[low] if low else "")
    if low == 0:
        return _below_hundred(high) + " hundred"
    if low < 10:
        return f"{_below_hundred(high)} oh {_ONES[low]}"
    return f"{_below_hundred(high)} {_below_hundred(low)}"


def _plural(words: str) -> str:
    head, _, last = words.rpartition(" ")
    last = last[:-1] + "ies" if last.endswith("y") else last + "s"
    return f"{head} {last}" if head else last


def integer(written: str) -> str:
    """An integer as written: separators dropped, four digits from 1100 to 2099
    read as a year, a leading zero or a run past the scales digit by digit."""
    run = written.replace(",", "")
    if len(run) > _MAX_CARDINAL_DIGITS or (len(run) > 1 and run[0] == "0"):
        return digits(run)
    n = int(run)
    if len(run) == 4 and "," not in written and 1100 <= n <= 2099:
        return year(n)
    return cardinal(n)


def number(written: str) -> str:
    """An integer or a decimal: "3.14159" -> "three point one four one five nine"."""
    whole, dot, fraction = written.partition(".")
    if not dot:
        return integer(whole)
    whole_words = cardinal(whole) if whole else "zero"
    return f"{whole_words} point {digits(fraction, zero='oh')}"


def _money(match: re.Match) -> str:
    (major, majors), (minor, minors) = _CURRENCIES[match["cur"]]
    whole, cents, scale = match["whole"], match["cents"], match["scale"]
    if scale:
        value = number(whole + (f".{cents}" if cents else ""))
        return f"{value} {scale} {majors}"
    run = whole.replace(",", "").lstrip("0")
    major_words = f"{cardinal(whole)} {major if run == '1' else majors}"
    if cents is None or len(cents) != 2:
        if cents is not None:
            return f"{number(whole + '.' + cents)} {majors}"
        return major_words
    c = int(cents)
    if not c:
        return major_words
    minor_words = f"{cardinal(c)} {minor if c == 1 else minors}"
    return minor_words if not run else f"{major_words} and {minor_words}"


def _date(month: int, day: int, year_written: str | None, sep: str = " ") -> str:
    out = f"the {ordinal(day)} of {_MONTHS[month - 1]}"
    if year_written:
        out += sep + (integer(year_written) if len(year_written) == 4 else _below_hundred(int(year_written)))
    return out


def _valid_date(month: int, day: int) -> bool:
    return 1 <= month <= 12 and 1 <= day <= 31


def _slash_date(match: re.Match) -> str:
    month, day = int(match["m"]), int(match["d"])
    if not _valid_date(month, day):
        return match[0]
    return _date(month, day, match["y"])


def _short_slash_date(match: re.Match) -> str:
    month, day = int(match["m"]), int(match["d"])
    # "11/02", "12/25": a date; "1/2", "3/4", "1/10": a fraction, read later
    if _valid_date(month, day) and (match["d"][0] == "0" or day > 12):
        return _date(month, day, None)
    return match[0]


def _iso_date(match: re.Match) -> str:
    month, day = int(match["m"]), int(match["d"])
    if not _valid_date(month, day):
        return match[0]
    return _date(month, day, match["y"])


def _month_day(match: re.Match) -> str:
    day = int(match["d"])
    if not 1 <= day <= 31:
        return match[0]
    out = f"{match['month']} {ordinal(day)}"
    if match["y"]:
        out += match["ysep"] + integer(match["y"])
    return out


def _day_month(match: re.Match) -> str:
    day = int(match["d"])
    if not 1 <= day <= 31:
        return match[0]
    out = f"the {ordinal(day)} of {match['month']}"
    return out + (" " + integer(match["y"]) if match["y"] else "")


def _time(match: re.Match) -> str:
    hour, minute, second = int(match["h"]), int(match["m"]), match["s"]
    if hour > 24 or minute > 59 or (second is not None and int(second) > 59):
        return match[0]
    if minute == 0 and second is None:
        out = f"{cardinal(hour)} o'clock"
    elif minute < 10:
        out = f"{cardinal(hour)} o {_ONES[minute]}"
    else:
        out = f"{cardinal(hour)} {cardinal(minute)}"
    if second is not None and int(second):
        out += f" and {cardinal(int(second))} seconds"
    if match["ap"]:
        out += f" {match['ap'].upper()}M"
    return out


def _fraction(written_numerator: str, written_denominator: str) -> str | None:
    # leading zeros stripped before int(), which refuses over 4,300 digits
    written_numerator = written_numerator.lstrip("0") or "0"
    written_denominator = written_denominator.lstrip("0") or "0"
    if len(written_denominator) > 3 or len(written_numerator) > 3:
        return None
    numerator, denominator = int(written_numerator), int(written_denominator)
    if not 0 < numerator < denominator <= 100:
        return None
    if denominator == 2:
        return "one half" if numerator == 1 else f"{cardinal(numerator)} halves"
    name = "quarter" if denominator == 4 else ordinal(denominator)
    return f"{cardinal(numerator)} {name if numerator == 1 else _plural(name)}"


def _mixed_fraction(match: re.Match) -> str:
    words = _fraction(match["n"], match["dd"])
    if words is None:
        return match[0]
    if match["n"].lstrip("0") == "1":
        words = "a " + words.split(" ", 1)[1]
    return f"{cardinal(match['w'])} and {words}"


def _simple_fraction(match: re.Match) -> str:
    words = _fraction(match["n"], match["dd"])
    return words if words is not None else f"{integer(match['n'])} slash {integer(match['dd'])}"


def _unit(match: re.Match) -> str:
    singular, plural = _UNITS[match["unit"]]
    one = match["num"] == "1"
    return f"{number(match['num'])} {singular if one else plural}"


def _phone(match: re.Match) -> str:
    return ", ".join(digits(group) for group in re.findall(r"\d+", match[0]))


def _local_phone(match: re.Match) -> str:
    """555-0199 as a phone number, but 100-1000 and 500-1000 as ranges, left
    for the range rule: no exchange starts with 0 or 1, and a round hundred
    above a round first number is a range far more often than a number."""
    a, b = match[0].split("-")
    if a[0] in "01" or (a[-1] == "0" and b.endswith("00") and int(a) < int(b)):
        return match[0]
    return _phone(match)


def _money_range(match: re.Match) -> str:
    (major, majors), _ = _CURRENCIES[match["cur"]]
    one = match["b"].lstrip("0") == "1"
    return f"{cardinal(match['a'])} to {cardinal(match['b'])} {major if one else majors}"


def _percent(match: re.Match) -> str:
    return ("negative " if match["neg"] else "") + number(match["num"]) + " percent"


def _decade(match: re.Match) -> str:
    n = int(match[1])
    return _plural(year(n) if 1100 <= n <= 2099 else cardinal(n))


def _version(match: re.Match) -> str:
    return " point ".join(integer(part) for part in match[0].split("."))


_RULES: list[tuple[re.Pattern, Callable[[re.Match], str]]] = [
    # 2024-10-19T04:50
    (re.compile(r"(?<=\d)T(?=\d{2}:\d{2})"), lambda m: " at "),
    # (415) 555-0199, 555-867-5309, 555.867.5309, 1-800-555-0199, 555-0199
    (
        re.compile(r"(?<![\w.,-])(?:\+?1[-.\s])?(?:\(\d{3}\)\s?|\d{3}([-.]))\d{3}(?(1)\1|[-.])\d{4}(?![\w-])"),
        _phone,
    ),
    (re.compile(r"(?<![\w.,$€£-])\d{3}-\d{4}(?![\w-])"), _local_phone),
    # 2023-10-19, 3/14/2021, 12/25/23, 11/02
    (re.compile(r"(?<![\w.-])(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?![\d-])"), _iso_date),
    (re.compile(r"(?<![\w./])(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{4}|\d{2})(?![\w/])"), _slash_date),
    (re.compile(r"(?<![\w./])(?P<m>\d{1,2})/(?P<d>\d{2})(?![\w/])"), _short_slash_date),
    # March 3, 2024; March 3rd; 19 October 2026: the written order
    (
        re.compile(
            rf"\b(?P<month>{_MONTH_WORD})\.?\s+(?P<d>\d{{1,2}}){_DAY_SUFFIX}\b(?:(?P<ysep>,?\s+)(?P<y>\d{{4}})\b)?"
        ),
        _month_day,
    ),
    (re.compile(rf"\b(?P<d>\d{{1,2}}){_DAY_SUFFIX}\s+(?P<month>{_MONTH_WORD})\b(?:\s+(?P<y>\d{{4}})\b)?"), _day_month),
    # $100-1000, $10-$20
    (
        re.compile(
            r"(?P<cur>[$€£])\s?(?P<a>\d{1,3}(?:,\d{3})+|\d+)\s?[-\u2013]\s?(?P=cur)?(?P<b>\d{1,3}(?:,\d{3})+|\d+)"
            r"(?![.,]?\d|\w|-|\s+(?:thousand|million|billion|trillion)\b)"
        ),
        _money_range,
    ),
    # $1,299.99, $3.2 billion, €20, £5
    (
        re.compile(
            r"(?P<cur>[$€£])\s?(?P<whole>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<cents>\d+))?"
            r"(?:\s+(?P<scale>thousand|million|billion|trillion)\b)?"
        ),
        _money,
    ),
    # 12.5%, -3%
    (re.compile(rf"(?<![\w.]){_NOT_IN_GROUP}(?P<neg>-)?(?P<num>{_NUM})\s?%"), _percent),
    # 1st, 22nd, 101st, 1,000th
    (
        re.compile(rf"\b{_NOT_IN_GROUP}(\d{{1,3}}(?:,\d{{3}})+|\d+)(?:st|nd|rd|th)\b", re.IGNORECASE),
        lambda m: ordinal(m[1]),
    ),
    # 1990s, '80s
    (re.compile(r"\b(\d{3}0)s\b"), _decade),
    (re.compile(r"(?<![\w'])'?([1-9]0)s\b"), lambda m: _plural(cardinal(int(m[1])))),
    # 2.5 kg, 65 mph, 30°C
    (
        re.compile(rf"(?<![\w.]){_NOT_IN_GROUP}(?P<num>{_NUM})\s?(?P<unit>km/h|°C|°F|°|kg|km|cm|mm|mph|kph|lbs|lb|oz)(?![\w/])"),
        _unit,
    ),
    # 3x faster, 3 x 4
    (re.compile(r"(?<=\d)x\b"), lambda m: " times"),
    (re.compile(r"(?<=\d)\s?[x×]\s?(?=\d)"), lambda m: " times "),
    # B12, 4B, Q4
    (re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])"), lambda m: " "),
    # version 2.10.3
    (re.compile(r"(?<![\w.])\d+(?:\.\d+){2,}(?!\w|\.\d)"), _version),
    # 4:05, 13:00, 10:30:15; then 3:2 as a score
    (
        re.compile(
            r"(?<![\w:.])(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?(?![\d:])(?:\s?(?P<ap>[AaPp])(?:\.[Mm]\.|[Mm]\b))?"
        ),
        _time,
    ),
    (re.compile(r"(?<![\w:.])(\d+):(\d+)(?![\d:])"), lambda m: f"{integer(m[1])} to {integer(m[2])}"),
    # 2 1/2, 3/4
    (re.compile(r"(?<![\w./])(?P<w>\d+)\s+(?P<n>\d+)/(?P<dd>\d+)(?![\w/])"), _mixed_fraction),
    (re.compile(r"(?<![\w./])(?P<n>\d+)/(?P<dd>\d+)(?![\w/])"), _simple_fraction),
    # pages 10-20, 2019-2021
    (
        re.compile(rf"(?<![\w.,-])(?P<a>{_NUM})\s?[-\u2013]\s?(?P<b>{_NUM})(?!\w|[.,]\d|-)"),
        lambda m: f"{number(m['a'])} to {number(m['b'])}",
    ),
    # -12 degrees
    (re.compile(rf"(?<![^\s(])[-\u2212](?P<num>{_NUM})"), lambda m: "negative " + number(m["num"])),
    # #5
    (re.compile(r"#(?=\d)"), lambda m: "number "),
    # 1,2,3 as a list, not a thousands group
    (re.compile(r"(?<=\d),(?=\d)(?!\d{3}(?!\d))"), lambda m: ", "),
    # every other number
    (re.compile(_NUM), lambda m: number(m[0])),
]


def normalize_english(text: str) -> str:
    if not any(c.isdecimal() for c in text):
        return text
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    return text
