"""Chunked tokenization: cuts only at the start of a whitespace run, chunks
concatenate to the text, and the ids are identical to one encode (checked
against the real Qwen tokenizer when it is in the local cache)."""
import json
import os

import pytest

from mstar.utils.fast_tokenize import cut_points, encode_ids, split_at


def test_cuts_start_whitespace_runs_and_chunks_rebuild_the_text():
    text = ("word " * 300 + "a   b  \n\n  c. \n d" + " tail " * 300) * 4
    cuts = cut_points(text, chunk_chars=512, max_chunks=8)
    assert 1 <= len(cuts) <= 7
    for c in cuts:
        assert text[c] == " " and not text[c - 1].isspace()
    assert "".join(split_at(text, cuts)) == text
    assert cut_points("short text", chunk_chars=4096) == []
    assert cut_points("x" * 10000, chunk_chars=4096) == []  # no space: no safe cut


def test_a_short_text_or_a_slow_tokenizer_takes_the_plain_path():
    class _Slow:
        def __call__(self, text):
            class _R:
                input_ids = [len(text)]
            return _R()

    assert encode_ids(_Slow(), "x" * 10000) == [10000]


def _tokenizer():
    cache = os.environ.get("MSTAR_TEST_HF_HOME")
    if not cache:
        pytest.skip("set MSTAR_TEST_HF_HOME to a cache holding Qwen/Qwen3.5-0.8B for the identity check")
    os.environ.setdefault("HF_HOME", cache)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")


def test_chunked_ids_are_identical_to_one_encode_on_the_benchmark_prompts():
    tok = _tokenizer()
    prompts = os.environ.get("MSTAR_TEST_PROMPTS")
    texts = []
    if prompts and os.path.exists(prompts):
        d = json.load(open(prompts))
        for key in ("long8k", "long", "short"):
            texts += [x if isinstance(x, str) else x.get("prompt", "") for x in d.get(key, [])]
    texts += [
        "a   b  \n\n  c. \n d" * 2000, ("<|im_start|>user\nhello   world<|im_end|>\n" * 400),
        "x,y;z!! \n\nq" * 1500, "tab\tsep  arated\r\nlines " * 1200, "数字 123 456  文本 " * 1000,
    ]
    for t in texts:
        whole = tok(t).input_ids
        assert encode_ids(tok, t, chunk_chars=2048, max_chunks=8) == whole
        assert encode_ids(tok, t, chunk_chars=512, max_chunks=16) == whole


def test_chunked_encode_turns_the_crate_parallelism_on(monkeypatch):
    """The Rust backend only runs a batch in parallel while
    TOKENIZERS_PARALLELISM reads true; a serving environment that exported
    it false gets it flipped before the batch, unless the knob says no."""
    from mstar.utils.fast_tokenize import encode_ids

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    class _Backend:
        def encode_batch(self, parts, add_special_tokens=False):
            return [_Enc([len(p)]) for p in parts]

    class _Tok:
        backend_tokenizer = _Backend()

        def __call__(self, text):
            raise AssertionError("the chunked path was expected")

    text = "word " * 4000
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.delenv("MSTAR_TOKENIZE_PARALLEL", raising=False)
    ids = encode_ids(_Tok(), text, chunk_chars=4096, max_chunks=8)
    assert sum(ids) == len(text) and os.environ["TOKENIZERS_PARALLELISM"] == "true"

    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("MSTAR_TOKENIZE_PARALLEL", "0")
    encode_ids(_Tok(), text, chunk_chars=4096, max_chunks=8)
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
