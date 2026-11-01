"""BAGEL, Qwen3.5 and Qwen3-Omni stream a token's raw bytes per step (#517).

``postprocess`` used to ``tokenizer.decode`` each step's ids, so a CJK
character or emoji split across two tokens became U+FFFD in the worker, before
the API server's incremental decoder (``test_utf8_split_tokens.py``) saw it.
It now sends the tokens' raw bytes, and joined they must decode to exactly
what ``tokenizer.decode`` gives on the whole reply: special and added tokens
(``<think>``, ``<|im_end|>`` under ``ignore_eos``) included.
"""

import asyncio
import json
import random
from types import SimpleNamespace

import pytest
import torch

from mstar.model.utils import TextChunkDetokenizer

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
# not special: decode keeps them, so the chunks must too
ADDED = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<中文>"]
REPLY = "<think>\n好的。</think>\n\n猫是人类的宠物 🐱😀，café ✓ <tool_call>{}</tool_call>"


def _corpus():
    return ["hello world, the cat sat on the mat. " * 20, "猫 宠物 😀 café", REPLY]


@pytest.fixture(scope="module")
def fast_tokenizer():
    """A tiny Qwen2-style byte-level BPE, small enough that CJK and emoji span tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(_corpus(), trainers.BpeTrainer(
        vocab_size=320, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    ))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
    )
    fast.add_tokens(ADDED)
    return fast


@pytest.fixture(scope="module")
def slow_tokenizer(fast_tokenizer, tmp_path_factory):
    """BAGEL's own (slow, Python) tokenizer over the same vocabulary."""
    pytest.importorskip("regex")
    from mstar.model.bagel.components.tokenization import BagelTokenizer, add_special_tokens

    path = tmp_path_factory.mktemp("bagel_tok")
    fast_tokenizer.backend_tokenizer.model.save(str(path))
    tok = BagelTokenizer(str(path / "vocab.json"), str(path / "merges.txt"))
    tok.add_tokens(ADDED)
    return add_special_tokens(tok)[0]


@pytest.fixture(params=["fast", "slow"])
def tokenizer(request):
    return request.getfixturevalue(f"{request.param}_tokenizer")


def _ids(tok, text):
    return tok.encode(text, add_special_tokens=False)


def _per_step(detok, ids):
    return [detok.to_bytes([i]) for i in ids]


def test_the_reply_splits_characters_across_tokens(tokenizer):
    """Otherwise the tests below would pass on main too."""
    chunks = _per_step(TextChunkDetokenizer(tokenizer), _ids(tokenizer, REPLY))
    split = [c for c in chunks if c.decode("utf-8", "replace") != c.decode("utf-8", "ignore")]
    assert split, "no token ends mid-character"
    assert "\ufffd" in "".join(tokenizer.decode([i]) for i in _ids(tokenizer, REPLY))


@pytest.mark.parametrize("text", [
    REPLY,
    "<|im_start|>assistant\n中文<|im_end|><|endoftext|>",
    "  spaces\t\ttabs\r\n<中文>end ",
])
def test_joined_bytes_equal_one_decode(tokenizer, text):
    detok = TextChunkDetokenizer(tokenizer)
    ids = _ids(tokenizer, text)
    assert detok._bytes is not None, "a byte-level tokenizer fell back to decode"
    try:
        expected = tokenizer.decode(ids)
    except KeyError:
        pytest.skip("the slow tokenizer cannot decode an added token outside its byte map")
    assert b"".join(_per_step(detok, ids)).decode() == expected


def test_joined_bytes_equal_one_decode_for_any_ids(tokenizer):
    """Random ids: specials, added tokens, unfinished characters, ids past the vocabulary."""
    detok = TextChunkDetokenizer(tokenizer)
    rng = random.Random(0)
    n = len(tokenizer)
    for _ in range(300):
        ids = [rng.randrange(n + 3) for _ in range(rng.randint(1, 30))]
        try:
            expected = tokenizer.decode(ids)
        except (TypeError, KeyError):  # the slow tokenizer's own decode fails on these
            continue
        assert b"".join(_per_step(detok, ids)).decode("utf-8", "replace") == expected, ids


def test_an_added_token_that_decode_writes_as_is_stays_as_is(fast_tokenizer, tmp_path):
    """transformers 4.x slow tokenizers (BAGEL's) decode a non-special added
    token as written. "«é»" is all byte-alphabet characters, so the byte map
    would turn it into U+FFFD. 5.x maps it like the fast tokenizer does, so
    this emulates 4.x's ``_decode`` on top of BAGEL's tokenizer."""
    pytest.importorskip("regex")
    from mstar.model.bagel.components.tokenization import BagelTokenizer, add_special_tokens

    fast_tokenizer.backend_tokenizer.model.save(str(tmp_path))
    tok = BagelTokenizer(str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"))
    tok.add_tokens(["«é»", "<think>"])
    tok = add_special_tokens(tok)[0]
    added = set(tok.added_tokens_encoder) - set(tok.all_special_tokens)
    byte_map_decode = tok.decode

    def legacy_decode(ids, **kwargs):
        text, run = "", []
        for i in ids:
            token = tok.convert_ids_to_tokens(i)
            if token in added:
                text += (byte_map_decode(run) if run else "") + token
                run = []
            else:
                run.append(i)
        return text + (byte_map_decode(run) if run else "")

    tok.decode = legacy_decode
    ids = _ids(tok, "猫 «é» café<think>")
    assert tok.convert_tokens_to_ids("«é»") in ids
    assert legacy_decode(ids) == "猫 «é» café<think>"
    detok = TextChunkDetokenizer(tok)
    assert b"".join(_per_step(detok, ids)).decode() == legacy_decode(ids)
    # the tokenizer's own decode: 5.x maps "«é»" through the bytes, 4.x does not
    tok.decode = byte_map_decode
    i = tok.convert_tokens_to_ids("«é»")
    as_is = {i: "«é»".encode()} if byte_map_decode([i]) == "«é»" else {}
    assert TextChunkDetokenizer(tok)._bytes.verbatim == as_is


def test_a_tokenizer_that_is_not_byte_level_keeps_decode():
    class SentencePieceLike:
        def decode(self, ids):
            return "".join(chr(0x4E00 + i) for i in ids)

    detok = TextChunkDetokenizer(SentencePieceLike())
    assert detok._bytes is None
    assert detok.to_bytes([1, 2]) == "丁丂".encode()


def test_cleanup_of_tokenization_spaces_keeps_decode(fast_tokenizer):
    tok = SimpleNamespace(
        clean_up_tokenization_spaces=True, backend_tokenizer=fast_tokenizer.backend_tokenizer,
        decode=lambda ids: "x",
    )
    assert TextChunkDetokenizer(tok).to_bytes([1]) == b"x"


# ----------------------------------------------------------------------
# each model's postprocess, end to end through chat completions
# ----------------------------------------------------------------------

def _model(name, tokenizer):
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if name == "bagel":
        from mstar.model.bagel.bagel_model import BagelModel

        m = BagelModel.__new__(BagelModel)
        m.eos_token_id = im_end
    elif name == "qwen3_5":
        from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

        m = Qwen3_5DenseModel.__new__(Qwen3_5DenseModel)
        m.config = SimpleNamespace(stop_token_ids=(im_end, tokenizer.convert_tokens_to_ids("<|endoftext|>")))
    else:
        Qwen3OmniModel = pytest.importorskip(  # its module needs flashinfer
            "mstar.model.qwen3_omni.qwen3_omni_model", exc_type=ImportError,
        ).Qwen3OmniModel
        m = Qwen3OmniModel.__new__(Qwen3OmniModel)
        m.config = SimpleNamespace(im_end_token_id=im_end)
    m.tokenizer = tokenizer
    return m


MODELS = [("bagel", "slow"), ("qwen3_5", "fast"), ("qwen3_omni", "fast")]


def _chunks(model, ids, request_kwargs=None):
    return [
        SimpleNamespace(
            modality="text", metadata={},
            data=model.postprocess(torch.tensor([i]), "text", request_kwargs=request_kwargs),
        )
        for i in ids
    ]


class _BatchAPI:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_result_chunk_batches(self, request_id):
        for c in self._chunks:
            yield [c]


def _sse_content(chunks):
    from mstar.api_server.openai.serving_chat import _stream_events

    async def run():
        return [f async for f in _stream_events(_BatchAPI(chunks).iter_result_chunk_batches("r"), "m", "r", 24000)]

    content = ""
    for frame in asyncio.run(run()):
        for line in frame.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                content += json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
    return content


@pytest.mark.parametrize(("name", "kind"), MODELS)
def test_a_split_character_comes_through_chat_whole(name, kind, request):
    pytest.importorskip("fastapi")
    from mstar.api_server.openai.serving_chat import _build_response

    tok = request.getfixturevalue(f"{kind}_tokenizer")
    model = _model(name, tok)
    ids = _ids(tok, REPLY + "<|im_end|>")
    chunks = _chunks(model, ids)
    expected = tok.decode(ids[:-1])
    assert "\ufffd" not in expected
    assert _sse_content(chunks) == expected
    assert _build_response("m", "r", chunks, 24000)["choices"][0]["message"]["content"] == expected


@pytest.mark.parametrize(("name", "kind"), MODELS)
def test_ignore_eos_keeps_each_end_token_as_its_own_chunk(name, kind, request):
    tok = request.getfixturevalue(f"{kind}_tokenizer")
    ids = _ids(tok, "猫<|im_end|><|im_end|>")
    chunks = [c.data for c in _chunks(_model(name, tok), ids, {"ignore_eos": True})]
    assert all(chunks)
    assert b"".join(chunks).decode() == tok.decode(ids) == "猫<|im_end|><|im_end|>"


# ----------------------------------------------------------------------
# the checkpoints' own tokenizers, when cached (GPU hosts)
# ----------------------------------------------------------------------

def _cached(repo):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    try:
        # a local directory, as the models load it: a repo id makes some
        # transformers versions query the Hub even when offline
        path = snapshot_download(repo, local_files_only=True)
        if repo.startswith("ByteDance-Seed/BAGEL"):
            from mstar.model.bagel.components.tokenization import BagelTokenizer, add_special_tokens

            return add_special_tokens(BagelTokenizer.from_pretrained(path))[0]
        return AutoTokenizer.from_pretrained(path)
    except (OSError, ValueError, TypeError) as exc:
        pytest.skip(f"{repo} tokenizer not cached: {exc}")


@pytest.mark.parametrize("repo", [
    "ByteDance-Seed/BAGEL-7B-MoT", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
])
def test_checkpoint_tokenizers_match_decode(repo):
    tok = _cached(repo)
    detok = TextChunkDetokenizer(tok)
    assert detok._bytes is not None
    text = REPLY + "<|im_end|><|endoftext|>"
    ids = _ids(tok, text)
    assert b"".join(_per_step(detok, ids)).decode() == tok.decode(ids) == text
    rng = random.Random(0)
    for _ in range(200):
        ids = [rng.randrange(len(tok)) for _ in range(rng.randint(1, 30))]
        assert b"".join(_per_step(detok, ids)).decode("utf-8", "replace") == tok.decode(ids), ids
