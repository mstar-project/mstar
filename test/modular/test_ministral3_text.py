"""Metadata and text boundary checks; no weights or GPU needed.

The small fixtures pin the source revision and tokenizer hashes. Optional real
tokenizer tests read MINISTRAL3_METADATA_DIR (test-only, a local directory with
the pinned config/tokenizer files) and never download anything. Their golden IDs
were produced with mistral-common, not with the production HF tokenizer.
"""

import codecs
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from mstar.model.ministral3.config import Ministral3TextConfig
from mstar.model.ministral3.ministral3_model import Ministral3Model
from mstar.model.ministral3.text import Ministral3TextProcessor
from mstar.model.multimodal import PromptPart
from mstar.model.utils import drop_stop_tokens

FIXTURES = Path(__file__).parent / "fixtures" / "ministral3"
REFERENCE = json.loads((FIXTURES / "tokenizer_reference.json").read_text())


@pytest.fixture
def metadata(tmp_path):
    for name in ("config.json", "generation_config.json"):
        (tmp_path / name).write_bytes((FIXTURES / name).read_bytes())
    return tmp_path


def test_released_config_geometry_and_yarn():
    config = Ministral3TextConfig.from_hf(FIXTURES, tokenizer_eos_token_id=2)
    assert config == Ministral3TextConfig(stop_token_ids=(2,))
    assert config.num_attention_heads * config.head_dim == 4096
    assert config.hidden_size == 3072


def test_geometry_is_read_from_file_not_defaulted(metadata):
    path = metadata / "config.json"
    raw = json.loads(path.read_text())
    raw["text_config"].update(hidden_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=32)
    path.write_text(json.dumps(raw))
    config = Ministral3TextConfig.from_hf(metadata, tokenizer_eos_token_id=2)
    assert (config.hidden_size, config.num_attention_heads * config.head_dim) == (96, 128)
    del raw["text_config"]["head_dim"]
    path.write_text(json.dumps(raw))
    with pytest.raises(KeyError, match="head_dim"):
        Ministral3TextConfig.from_hf(metadata, tokenizer_eos_token_id=2)


@pytest.mark.parametrize("field,value", [
    ("hidden_act", "gelu"), ("sliding_window", 4096), ("attention_bias", True),
    ("qk_norm", True), ("partial_rotary_factor", 0.5),
    ("rope_parameters", {"rope_type": "default"}),
])
def test_unsupported_architecture_fails_explicitly(metadata, field, value):
    path = metadata / "config.json"
    raw = json.loads(path.read_text())
    raw["text_config"][field] = value
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        Ministral3TextConfig.from_hf(metadata, tokenizer_eos_token_id=2)


def test_stop_ids_combine_tokenizer_model_and_generation(metadata):
    (metadata / "generation_config.json").write_text('{"eos_token_id": [2, 7]}')
    path = metadata / "config.json"
    raw = json.loads(path.read_text())
    raw["text_config"]["eos_token_id"] = 8
    path.write_text(json.dumps(raw))
    assert Ministral3TextConfig.from_hf(metadata, tokenizer_eos_token_id=9).stop_token_ids == (2, 7, 8, 9)
    (metadata / "generation_config.json").unlink()
    assert Ministral3TextConfig.from_hf(metadata, tokenizer_eos_token_id=9).stop_token_ids == (8, 9)


@pytest.mark.parametrize("eos", [None, True, -1, 131072])
def test_invalid_tokenizer_eos_is_rejected(eos):
    with pytest.raises(ValueError):
        Ministral3TextConfig.from_hf(FIXTURES, tokenizer_eos_token_id=eos)


@pytest.mark.parametrize("parts", [[], [PromptPart("image")], [PromptPart("audio")],
                                   [PromptPart("video")], [PromptPart("text", "result", role="tool")]])
def test_unsupported_input_is_rejected_before_tokenization(parts):
    processor = object.__new__(Ministral3TextProcessor)
    with pytest.raises(ValueError):
        processor.prompt_ids(parts)


@pytest.fixture(scope="module")
def processor():
    directory = os.environ.get("MINISTRAL3_METADATA_DIR")
    if directory is None:
        pytest.skip("set MINISTRAL3_METADATA_DIR to cached pinned tokenizer files; no downloads in tests")
    pytest.importorskip("transformers")
    for name, digest in REFERENCE["sha256"].items():
        if name == "tekken.json":  # only needed to regenerate the recorded oracle
            continue
        assert hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() == digest, name
    return Ministral3TextProcessor(directory)


@pytest.mark.parametrize("case", REFERENCE["text_cases"])
def test_token_ids_match_mistral_common(processor, case):
    assert processor.tokenizer.encode(case["text"], add_special_tokens=False) == case["ids"]


@pytest.mark.parametrize("case", REFERENCE["chat_cases"])
def test_chat_ids_match_mistral_common(processor, case):
    parts = [PromptPart("text", text=m["content"], role=m["role"]) for m in case["messages"]]
    actual = processor.prompt_ids(parts)
    assert actual.dtype == torch.long and actual.device.type == "cpu"
    assert actual.tolist() == case["ids"]
    assert actual.tolist().count(processor.tokenizer.bos_token_id) == 1
    # Omitting the instruction format is a meaningful negative control.
    assert actual.tolist() != processor.tokenizer.encode(case["messages"][-1]["content"], add_special_tokens=False)


def test_default_system_and_bos_follow_published_template(processor):
    messages = [{"role": "user", "content": "Hello"}]
    rendered = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    actual = processor.prompt_ids([PromptPart("text", "Hello")]).tolist()
    assert actual == processor.tokenizer.encode(rendered, add_special_tokens=False)
    assert actual[0] == processor.tokenizer.bos_token_id == 1
    assert actual.count(1) == 1
    assert processor.config.stop_token_ids == (processor.tokenizer.eos_token_id,) == (2,)
    assert processor.tokenizer.pad_token_id == 11


def test_prompt_length_and_bad_chat_fail(processor, monkeypatch):
    monkeypatch.setattr(processor, "config", replace(processor.config, max_position_embeddings=1))
    with pytest.raises(ValueError, match="exceeds"):
        processor.prompt_ids([PromptPart("text", "Hello")])
    with pytest.raises(ValueError, match="chat template refused"):
        processor.prompt_ids([PromptPart("text", "Hello", role="assistant")])


def test_streamed_bytes_preserve_split_unicode_and_stop_handling(processor):
    text = "café 中文 Ελληνικά 👋🏽\n"
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    chunks = [processor.output.detokenizer.token_bytes(i) for i in ids]
    # Ensure this test actually exercises incomplete UTF-8 chunks.
    assert any(chunk.decode("utf-8", errors="replace").encode("utf-8") != chunk for chunk in chunks)
    decoder = codecs.getincrementaldecoder("utf-8")()
    streamed = "".join(decoder.decode(chunk) for chunk in chunks) + decoder.decode(b"", final=True)
    assert streamed == processor.tokenizer.decode(ids) == text
    eos = processor.tokenizer.eos_token_id
    output = torch.tensor([*ids, eos])
    filtered = drop_stop_tokens(output, processor.output.stop_token_ids, {})
    assert processor.output.detokenizer.to_bytes(filtered).decode("utf-8") == text
    kept = drop_stop_tokens(output, processor.output.stop_token_ids, {"ignore_eos": True})
    assert processor.output.detokenizer.to_bytes(kept).decode("utf-8") == processor.tokenizer.decode([*ids, eos])


def test_outer_model_preserves_real_tokenizer_intake_and_output(processor):
    model = Ministral3Model(os.environ["MINISTRAL3_METADATA_DIR"])
    for case in REFERENCE["chat_cases"]:
        parts = [PromptPart("text", text=m["content"], role=m["role"]) for m in case["messages"]]
        result = model.process_prompt(
            "\n".join(p.text for p in parts), ["text"] * len(parts), ["text"], prompt_parts=parts,
        )
        assert result.new_input_tensors["text_inputs"][0].tolist() == case["ids"]
    text = "café 中文 👋🏽\n"
    ids = processor.tokenizer.encode(text, add_special_tokens=False) + [processor.tokenizer.eos_token_id]
    assert model.postprocess(torch.tensor(ids), "text") == text.encode("utf-8")
    assert b"".join(model.postprocess_inline(ids, "text", [{} for _ in ids])) == text.encode("utf-8")
