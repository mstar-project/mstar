"""Collapse the repetition loops Qwen3-ASR can fall into, as the reference SDK does.

A port of ``detect_and_fix_repetitions`` from qwen-asr 0.0.6
(``qwen_asr/inference/utils.py``, Apache-2.0), which ``parse_asr_output`` runs
on every raw transcript. Same output, as two regular expressions instead of a
Python loop per character, which took 126 ms (vs 10 ms) on a 19k-character
transcript, on the API server's event loop.
"""

import re

THRESHOLD = 20
MAX_PATTERN = 20

# a character repeated more than THRESHOLD times in a row
_CHAR_RUN = re.compile(r"(.)\1{%d,}" % THRESHOLD, re.S)
# the shortest pattern of up to MAX_PATTERN characters repeated THRESHOLD or
# more times in a row, at the leftmost place one starts
_PATTERN_RUN = re.compile(r"(.{1,%d}?)\1{%d,}" % (MAX_PATTERN, THRESHOLD - 1), re.S)


def fix_repetitions(text: str) -> str:
    """``text`` with each run collapsed to one copy of what repeats."""
    text = _CHAR_RUN.sub(r"\1", text)
    out = []
    while True:
        # the reference scans for a start only up to 2 * THRESHOLD characters
        # from the end of what is left
        last_start = len(text) - 2 * THRESHOLD
        m = _PATTERN_RUN.search(text) if last_start >= 0 else None
        if m is None or m.start() > last_start:
            out.append(text)
            return "".join(out)
        out.append(text[:m.start()])
        out.append(m.group(1))
        text = text[m.end():]
