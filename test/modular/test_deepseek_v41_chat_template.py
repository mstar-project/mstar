"""DeepSeek-V4.1's prompt: the chat template its checkpoint ships writes what its reference
encoder writes, through ``render_chat_template``, the call the model's ``process_prompt`` makes.

The checkpoint's ``tokenizer_config.json`` holds no template: transformers loads
``chat_template.jinja`` from beside it. ``chat_templates/deepseek_v41.jinja`` is
deepseek-ai/DeepSeek-V4.1-Flash-1017 @08c8657f02's. ``chat_templates/deepseek_v41_encoding/``
holds deepseek-ai/DeepSeek-V4.1-Flash @314113f8a7's ``encoding/tests``, the prompts its encoder
(``encoding/encoding.py``, the same file in both repositories) writes for five conversations,
and ``cases.json``, that encoder's prompts for the chats its goldens leave out: both modes,
``drop_thinking``, ``reasoning_effort``, tool results and the turn a forced call opens. All are
DeepSeek's, under the MIT license in ``chat_templates/LICENSE.deepseek``.
"""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from mstar.model.multimodal import DEEPSEEK_V41_TEMPLATE_KWARGS, render_chat_template  # noqa: E402

HERE = Path(__file__).parent / "chat_templates"
ENCODING = HERE / "deepseek_v41_encoding"
CASES = json.loads((ENCODING / "cases.json").read_text())


def byte_tokenizer():
    """A tokenizer with DeepSeek-V4.1's template whose ids are the prompt's bytes, so they
    decode back to the text the template wrote."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok)
    tokenizer.chat_template = (HERE / "deepseek_v41.jinja").read_text()
    return tokenizer


def rendered(tokenizer, request_kwargs: dict, prompt: str = "", parts=None) -> str:
    """The text of the ids ``process_prompt`` gets for a request's kwargs."""
    ids = render_chat_template(tokenizer, prompt, parts, request_kwargs, DEEPSEEK_V41_TEMPLATE_KWARGS)
    return tokenizer.decode(ids)


@pytest.fixture(scope="module")
def tokenizer():
    return byte_tokenizer()


def _objects(messages):
    """Each call's arguments as an object, as the chat adapter sends them."""
    messages = json.loads(json.dumps(messages))
    for m in messages:
        for call in m.get("tool_calls") or []:
            args = call["function"].get("arguments")
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args) if args.strip() else {}
    return messages


@pytest.mark.parametrize("i", range(1, 6))
def test_the_template_writes_the_encoders_own_goldens(tokenizer, i):
    raw = json.loads((ENCODING / f"test_input_{i}.json").read_text())
    raw = {"messages": raw} if isinstance(raw, list) else raw
    # the encoder's test runs a case in its chat mode unless the case names one
    thinking = raw.get("thinking_mode") == "thinking"
    switches = {"enable_thinking": thinking, "reasoning_effort": raw.get("reasoning_effort")}
    kwargs = {"messages": _objects(raw["messages"]), "tools": raw.get("tools"), "chat_template_kwargs": switches}
    assert rendered(tokenizer, kwargs) == (ENCODING / f"test_output_{i}.txt").read_text()


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda c: c["id"])
def test_the_template_writes_what_the_encoder_writes(tokenizer, case):
    kwargs = {"messages": case["messages"], "tools": CASES["toolsets"].get(case["tools"]),
              "chat_template_kwargs": {"enable_thinking": case["thinking"], **case["switches"]}}
    if case["prompt"] is None:
        # what the encoder refuses is a 400, not a prompt of the template's own
        with pytest.raises(ValueError, match="refused"):
            rendered(tokenizer, kwargs)
    else:
        assert rendered(tokenizer, kwargs) == case["prompt"]


def test_a_bare_prompt_is_one_user_turn_that_thinks(tokenizer):
    """``/generate``: no messages, so the prompt is the user's, in the template's default mode."""
    assert rendered(tokenizer, {}, "Hello") == (
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more "
        "thorough the reasoning)\n\n<｜User｜>Hello<｜Assistant｜><think>")


def _checkpoint_tokenizer():
    """DeepSeek-V4.1's own tokenizer: ``DSV41_REFERENCE_DIR`` (a directory with its
    ``tokenizer.json``), else the Hugging Face cache; None when neither has it."""
    from transformers import AutoTokenizer

    path = os.environ.get("DSV41_REFERENCE_DIR")
    if not path:
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download("deepseek-ai/DeepSeek-V4.1-Flash", allow_patterns=["tokenizer*.json", "*.jinja"],
                                     local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached: skip
            return None
    return AutoTokenizer.from_pretrained(path) if (Path(path) / "tokenizer.json").exists() else None


def test_the_checkpoints_tokenizer_carries_its_template():
    """The serving path's assumption: ``chat_template.jinja`` is the tokenizer's template, and
    its marks are single tokens that a detokenizer skipping special tokens keeps."""
    tok = _checkpoint_tokenizer()
    if tok is None:
        pytest.skip("DeepSeek-V4.1's tokenizer is not here (DSV41_REFERENCE_DIR or the Hugging Face cache)")
    assert "｜DSML｜" in (tok.chat_template or "")
    ids = tok.encode("</think>｜DSML｜<｜end▁of▁sentence｜>", add_special_tokens=False)
    assert len(ids) == 3
    assert tok.decode(ids, skip_special_tokens=True) == "</think>｜DSML｜"
