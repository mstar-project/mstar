"""Qwen3-ASR's repetition guard matches the reference SDK's ``detect_and_fix_repetitions``."""

import random

import pytest

from mstar.model.qwen3_asr.repetition import fix_repetitions


# Verbatim from qwen-asr 0.0.6, qwen_asr/inference/utils.py (Apache-2.0): the oracle.
def detect_and_fix_repetitions(text, threshold=20):
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1

            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i+count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s

        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break

                pattern = s[i:i+k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx+k] != pattern:
                        valid = False
                        break

                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index+k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break

            if found:
                break
            else:
                result.append(s[i])
                i += 1

        if not found:
            result.append(s[i:])
        return ''.join(result)

    text_raw = text
    text = fix_char_repeats(text_raw, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


@pytest.mark.parametrize("text", [
    "",
    "language English<asr_text>hello world",
    "a" * 20, "a" * 21, "a" * 45, "x" + "a" * 39, "ab" * 19 + "c", "ab" * 20, "ab" * 25 + "tail",
    "the cat " * 30 + "sat down.",
    "language English<asr_text>yes, " * 3 + "no no " * 40,
    "start " + "ha" * 30 + " middle " + "the end " * 22,
    "\n" * 30 + "x",
    "很好" * 25 + "。",
])
def test_known_cases_match_the_reference(text):
    assert fix_repetitions(text) == detect_and_fix_repetitions(text)


def test_random_text_matches_the_reference():
    rng = random.Random(0)
    for _ in range(3000):
        pieces = []
        for _ in range(rng.randint(1, 6)):
            unit = "".join(rng.choice("ab c\n") for _ in range(rng.randint(1, 22)))
            pieces.append(unit * rng.choice([1, 2, 5, 19, 20, 21, 30]))
        text = "".join(pieces)
        assert fix_repetitions(text) == detect_and_fix_repetitions(text), repr(text)


def test_a_long_transcript_without_loops_is_unchanged():
    words = ["speech", "is", "long", "and", "varied", "today", "we", "test", "it"]
    rng = random.Random(1)
    text = " ".join(rng.choice(words) + str(i % 97) for i in range(4000))
    assert fix_repetitions(text) == text
