"""Per-token text output under a byte-level BPE tokenizer: each token's raw
bytes, so a character split across tokens survives streaming, while whole
characters, added tokens and stop tokens come out as ``tokenizer.decode``
gives them."""
import codecs
import os

import pytest
import torch

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from mstar.model.utils import TokenBytes, drop_stop_tokens  # noqa: E402

TEXTS = ["hello world", "你好 🙂 café", "<|im_end|> after", "𠀋 ꙮ ༀ mixed ascii"]


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    # a small vocab, so multi-byte characters stay split across byte tokens
    trainer = trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|im_end|>"],
    )
    tok.train_from_iterator(["hello world café"] * 20, trainer)
    return transformers.PreTrainedTokenizerFast(tokenizer_object=tok)


def _stream(chunks: list[bytes]) -> str:
    """What the chat SSE stream sends: an incremental decode, flushed at the end."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    return "".join(decoder.decode(c) for c in chunks) + decoder.decode(b"", final=True)


def test_joined_token_bytes_are_the_text(tokenizer):
    token_bytes = TokenBytes(tokenizer)
    for text in TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert token_bytes.join(ids).decode("utf-8") == tokenizer.decode(ids) == text


def test_a_split_character_survives_per_token_emission(tokenizer):
    token_bytes = TokenBytes(tokenizer)
    ids = tokenizer.encode("🙂", add_special_tokens=False)
    assert len(ids) > 1  # split across byte tokens
    # what decoding each token on its own sends: replacement characters
    assert "�" in "".join(tokenizer.decode([i]) for i in ids)
    assert _stream([token_bytes(i) for i in ids]) == "🙂"


def test_an_id_past_the_tokenizer_vocab_has_no_bytes(tokenizer):
    # a model's padded output rows can be sampled; `decode` gives them no text
    tb = TokenBytes(tokenizer)
    past = len(tokenizer) + 3
    assert tokenizer.decode([past]) == ""
    assert tb(past) == b""
    assert tb.join([*tokenizer.encode("hello"), past]) == b"hello"


def test_a_tokenizer_without_a_byte_level_decoder_is_refused():
    from tokenizers import Tokenizer, decoders, models

    tok = Tokenizer(models.WordLevel({"a": 0, "[UNK]": 1}, unk_token="[UNK]"))
    tok.decoder = decoders.WordPiece()
    with pytest.raises(ValueError, match="ByteLevel"):
        TokenBytes(transformers.PreTrainedTokenizerFast(tokenizer_object=tok))


# -- Qwen3.5, against its own tokenizer ---------------------------------------


def _qwen3_5_path() -> str:
    """A Qwen3.5 checkpoint directory (its tokenizer and config; no weights are loaded)."""
    if path := os.environ.get("QWEN3_5_CKPT"):
        return path
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download("Qwen/Qwen3.5-0.8B", local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached
        pytest.skip("Qwen/Qwen3.5-0.8B is not cached; set QWEN3_5_CKPT")


@pytest.fixture(scope="module")
def qwen():
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    return Qwen3_5DenseModel(_qwen3_5_path())


QWEN_TEXTS = [
    "𠀋𪚥龘鱻 ꙮ ༀ",  # rare CJK and others, split across tokens
    "👩‍👩‍👧‍👦 and 🧑‍🔬🫠",  # emoji ZWJ sequences
    "<think>\nhmm</think>\n\nok",  # added tokens the reply keeps
    '<tool_call>{"a": 1}</tool_call>',
    "plain ascii, 你好, café, 🙂",
]


def _main_postprocess(qwen, ids, request_kwargs=None) -> bytes:
    """Per-token text before TokenBytes: stop tokens dropped, then ``tokenizer.decode``."""
    kept = drop_stop_tokens(torch.tensor(ids), qwen.config.stop_token_ids, request_kwargs)
    return qwen.tokenizer.decode(kept).encode("utf-8")


@pytest.mark.parametrize("text", QWEN_TEXTS)
def test_qwen_streams_and_joins_to_the_decoded_text(qwen, text):
    ids = qwen.tokenizer.encode(text, add_special_tokens=False)
    chunks = qwen.postprocess_inline(ids, "text", [None] * len(ids))
    assert _stream(chunks) == qwen.tokenizer.decode(ids) == text
    assert b"".join(chunks).decode("utf-8") == text
    # the tensor path gives the same bytes
    assert [qwen.postprocess(torch.tensor([i]), "text") for i in ids] == chunks


def test_qwen_split_characters_were_garbled_token_by_token(qwen):
    ids = qwen.tokenizer.encode("👩‍👩‍👧‍👦", add_special_tokens=False)
    assert "�" in _stream([_main_postprocess(qwen, [i]) for i in ids])


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_qwen_stop_tokens_follow_ignore_eos(qwen, ignore_eos):
    stops = sorted(qwen.config.stop_token_ids)
    assert qwen.tokenizer.convert_ids_to_tokens(stops) == ["<|endoftext|>", "<|im_end|>"]
    ids = [*qwen.tokenizer.encode("ok", add_special_tokens=False), *stops]
    kwargs = {"ignore_eos": True} if ignore_eos else None
    chunks = qwen.postprocess_inline(ids, "text", [kwargs] * len(ids))
    assert chunks == [_main_postprocess(qwen, [i], kwargs) for i in ids]
    assert b"".join(chunks) == (b"ok<|endoftext|><|im_end|>" if ignore_eos else b"ok")


def test_qwen_matches_decode_for_every_token_that_is_whole_text(qwen):
    """An id whose bytes are complete UTF-8 on their own gives what
    ``decode([id])`` gives; for the rest, ``decode`` gave U+FFFD."""
    tok = qwen.tokenizer
    ids = list(range(len(tok)))
    chunks = qwen.postprocess_inline(ids, "text", [{"ignore_eos": True}] * len(ids))
    for i, chunk in zip(ids, chunks, strict=True):
        decoded = tok.decode([i])
        try:
            assert chunk.decode("utf-8") == decoded, i
        except UnicodeDecodeError:
            assert "�" in decoded, i


def test_qwen_padded_vocab_ids_match_main(qwen):
    # the checkpoint's output vocab is padded past the tokenizer's
    ids = list(range(len(qwen.tokenizer), len(qwen.tokenizer) + 5))
    chunks = qwen.postprocess_inline(ids, "text", [None] * len(ids))
    assert b"".join(chunks) == _main_postprocess(qwen, ids) == b""
