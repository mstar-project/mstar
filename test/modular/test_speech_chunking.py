"""Sentence chunking on ``/v1/audio/speech``: the splitter and the adapters'
chunk settings. mstar-server plans and orders the sub-requests
(``rust/server/src/speech_chunking.rs``; its tests and test/rust/test_rust_speech.py).
"""

import re
import runpy
import sys
import types
import unicodedata
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from mstar.api_server.entrypoint import APIServer  # noqa: E402
from mstar.api_server.openai import adapters  # noqa: E402
from mstar.api_server.openai.speech_chunking import split_sentences  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MARK_TABLE = REPO_ROOT / "rust/server/src/unicode_marks.rs"
GENERATOR = REPO_ROOT / "scripts/gen_unicode_marks.py"

LONG_TEXT = (
    "The train to the coast leaves at seven tomorrow morning, so please pack your bag tonight. "
    "She opened the window and let the cool evening air drift into the kitchen. "
    "Our meeting has been moved to Thursday afternoon because the room is being repainted! "
    "A gentle rain fell over the harbor while the fishing boats returned one by one. "
    "Remember to water the tomatoes twice a week during the hottest part of the summer? "
    "The museum's new exhibit traces the history of printing from wooden blocks to modern presses."
)


# ---------------------------------------------------------------------------
# splitter
# ---------------------------------------------------------------------------


def test_short_text_is_one_chunk():
    assert split_sentences("Hello there. How are you?", max_chars=400) == ["Hello there. How are you?"]
    assert split_sentences("   ", max_chars=400) == []


def test_sentences_are_grouped_up_to_max_chars_and_never_cut():
    chunks = split_sentences(LONG_TEXT, max_chars=200)
    assert len(chunks) == 3
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks) == " ".join(LONG_TEXT.split())
    # every chunk ends where a sentence ends
    assert all(c[-1] in ".!?" for c in chunks)


def test_cjk_terminators_and_paragraph_breaks_split():
    text = "今天天气很好。我们去公园散步吧！你觉得怎么样？\n\nSecond paragraph here."
    chunks = split_sentences(text, max_chars=9, min_chars=1)
    assert chunks[:3] == ["今天天气很好。", "我们去公园散步吧！", "你觉得怎么样？"]
    assert " ".join(chunks[3:]) == "Second paragraph here."

    paragraphs = split_sentences("First paragraph no period\n\nsecond paragraph no period", max_chars=30, min_chars=1)
    assert paragraphs == ["First paragraph no period", "second paragraph no period"]


def test_cjk_sentences_in_one_chunk_are_joined_without_a_space():
    text = "今天天气很好。我们去公园散步吧！你觉得怎么样？" * 2
    chunks = split_sentences(text, max_chars=30, min_chars=1)
    assert "".join(chunks) == text and all(" " not in c for c in chunks)
    assert split_sentences("Hi there. 你好。Bye now.", max_chars=12, min_chars=1) == ["Hi there.", "你好。Bye now."]


def test_quotes_after_terminators_stay_with_their_sentence():
    text = 'He said "Wait!" Then she left. "Really?" she asked. Yes.'
    chunks = split_sentences(text, max_chars=20, min_chars=1)
    assert chunks[0] == 'He said "Wait!"'
    assert chunks[1] == "Then she left."
    assert chunks[2] == '"Really?" she asked.'


def test_overlong_sentence_is_wrapped_at_clauses_then_spaces():
    sentence = "alpha beta gamma, delta epsilon zeta, eta theta iota, kappa lambda mu nu xi omicron"
    chunks = split_sentences(sentence, max_chars=30, min_chars=1)
    assert all(len(c) <= 30 for c in chunks)
    assert " ".join(chunks).replace(" ,", ",") == sentence
    assert chunks[0] == "alpha beta gamma,"


def test_tiny_trailing_fragment_merges_into_previous_chunk():
    text = ("A sentence that is fairly long and goes on for a while to fill the chunk nicely. "
            "Yes.")
    chunks = split_sentences(text, max_chars=90, min_chars=24)
    assert chunks == [" ".join(text.split())]


def test_max_chars_must_be_positive():
    with pytest.raises(ValueError):
        split_sentences("a. b.", max_chars=0)


def test_an_even_cut_lands_where_this_interpreters_unicodedata_puts_it():
    """A cut that falls on a combining mark moves back off it, so where this
    input is cut depends on the table the splitter reads: U+0897 is unassigned
    up to Unicode 15.1.0 and a mark from 16.0.0. The Rust frontend reads a
    table generated from ``unicodedata`` so that it answers the same way
    (``rust/server/src/unicode_marks.rs``)."""
    text = "a" * 267 + "\u0897" + "b" * 533
    chunks = split_sentences(text, max_chars=400, min_chars=24)
    moves = unicodedata.category("\u0897").startswith("M")
    assert [len(c) for c in chunks] == ([266, 268, 267] if moves else [267, 267, 267])
    assert "".join(chunks) == text


def test_the_rust_mark_table_is_this_interpreters_unicodedata():
    """The Rust frontend cannot ask ``unicodedata``, so it reads a table
    generated from it. Both frontends must call the same code points marks, or
    they cut long text in different places; this fails when a CPython upgrade
    moves the categories and the table has not been regenerated."""
    source = MARK_TABLE.read_text()
    ranges = [(int(low, 16), int(high, 16))
              for low, high in re.findall(r"\(0x([0-9a-f]+), 0x([0-9a-f]+)\)", source)]
    assert len(ranges) > 100, f"no table parsed out of {MARK_TABLE}"
    tabled = {code for low, high in ranges for code in range(low, high + 1)}
    marks = {code for code in range(sys.maxunicode + 1)
             if unicodedata.category(chr(code)).startswith("M")}
    disagree = sorted(tabled ^ marks)
    assert not disagree, (
        f"{len(disagree)} code points disagree with unicodedata "
        f"{unicodedata.unidata_version} (first {[hex(c) for c in disagree[:8]]}): regenerate "
        f"with `python {GENERATOR.relative_to(REPO_ROOT)}`, which moves Rust cut points"
    )
    # and the checked-in file is what the generator writes, so regenerating it
    # is the whole fix when this does fail
    generator = runpy.run_path(str(GENERATOR))
    assert source == generator["render"](generator["mark_ranges"]())


@pytest.mark.parametrize(
    "text",
    [
        "你好" * 300,
        "สวัสดี" * 200,
        "x" * 1000,
        "x" * 410,
        "x" * 401 + ". " + "y" * 399 + ".",
    ],
    ids=["cjk-600", "thai-1200", "latin-1000", "latin-410-short-leftover", "latin-401-then-sentence"],
)
def test_text_with_no_break_points_is_still_cut_to_max_chars(text):
    chunks = split_sentences(text, max_chars=400)
    assert len(chunks) > 1
    assert max(len(c) for c in chunks) <= 400
    assert min(len(c) for c in chunks) >= 24
    assert not any(unicodedata.category(c[0]).startswith("M") for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_orpheus_splits_an_input_longer_than_its_output_cap():
    """2,048 output tokens are about 25 s of speech; a longer input came back cut short with a 200."""
    adapter = adapters.OrpheusAdapter()
    text = "The quick brown fox jumps over the lazy dog. " * 22
    assert len(text) >= adapter.speech_chunk_min_chars
    assert len(adapter.split_speech_text(text)) > 1


def test_release_request_aborts_only_requests_still_running():
    import threading

    aborted = []
    done, running = threading.Event(), threading.Event()
    done.set()
    server = types.SimpleNamespace(
        request_lock=threading.Lock(),
        _live={"done": "done#1", "running": "running#2"},
        pending_requests={"done#1": types.SimpleNamespace(event=done),
                          "running#2": types.SimpleNamespace(event=running)},
        _retire=lambda key: None,
        _record_end=lambda req, outcome: None,
        abort_submission=aborted.append,
    )
    APIServer.release_request(server, "done")
    APIServer.release_request(server, "running")
    assert aborted == ["running#2"] and "done#1" not in server.pending_requests
