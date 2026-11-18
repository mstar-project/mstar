"""On an empty cache the API-side Qwen3.5 model has only the metadata files
(workers download the weights later), and must render chats as the full
checkpoint does. LightOnOCR-3 and Surya keep their template only in
``chat_template.jinja``; Chandra's ``tokenizer_config.json`` holds another.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from test_qwen3_5_video import CHAT_TEMPLATE, write_synthetic_checkpoint  # noqa: E402  (sibling helper)

# Chandra's tokenizer_config.json template opens a <think> block; its .jinja closes it
OTHER_TEMPLATE = CHAT_TEMPLATE.replace("<think>\n'", "<think>\n\n</think>\n\n'")


def _repo(root: Path, tokenizer_config_template: str | None) -> Path:
    """A checkpoint whose template is in ``chat_template.jinja``, and
    ``tokenizer_config_template`` (if any) in ``tokenizer_config.json``."""
    write_synthetic_checkpoint(root)
    (root / "chat_template.jinja").write_text(CHAT_TEMPLATE)
    config_path = root / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config.pop("chat_template", None)
    if tokenizer_config_template is not None:
        config["chat_template"] = tokenizer_config_template
    config_path.write_text(json.dumps(config))
    return root


def _hub(repo: Path):
    """``snapshot_download`` on an empty cache: only the files of ``repo`` that
    ``allow_patterns`` selects, filtered as the hub filters them."""
    from huggingface_hub.utils import filter_repo_objects

    def snapshot_download(repo_id, cache_dir=None, allow_patterns=None, ignore_patterns=None, **_):
        snapshot = Path(cache_dir) / repo_id.replace("/", "--")
        snapshot.mkdir(parents=True, exist_ok=True)
        names = [p.name for p in repo.iterdir() if p.is_file()]
        for name in filter_repo_objects(names, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns):
            shutil.copy(repo / name, snapshot / name)
        return str(snapshot)

    return snapshot_download


@pytest.mark.parametrize("tokenizer_config_template", [None, OTHER_TEMPLATE], ids=["jinja-only", "jinja-differs"])
def test_a_fresh_cache_renders_chats_with_the_checkpoints_template(tmp_path, monkeypatch, tokenizer_config_template):
    import huggingface_hub

    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    repo = _repo(tmp_path / "repo", tokenizer_config_template)
    full = Qwen3_5DenseModel(str(repo))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _hub(repo))
    fresh = Qwen3_5DenseModel("org/ocr-fine-tune", cache_dir=str(tmp_path / "cache"))

    def ids(model):
        [text] = model.process_prompt("OCR this.", ["text"], ["text"]).new_input_tensors["text_inputs"]
        return text.tolist()

    assert ids(fresh) == ids(full)


def _prepared_kwargs(model, structured_outputs, **client) -> dict:
    """The request kwargs the data worker hands the conductor, without its threads or sockets."""
    from mstar.api_server.data_worker import PreprocessWorkerThread
    from mstar.api_server.request_types import PreprocessInput

    worker = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    worker.model, worker.device, worker._keyed_labels = model, "cpu", None
    return worker._prepare_input(PreprocessInput(
        request_id="r", text="Reply in JSON.", file_paths=None, input_modalities=["text"],
        output_modalities=["text"],
        model_kwargs={"prefix_cache": False, "structured_outputs": structured_outputs, **client},
    )).model_kwargs


@pytest.mark.parametrize("template, after", [(CHAT_TEMPLATE, "</think>"), (OTHER_TEMPLATE, None)],
                         ids=["thinking", "reasoning closed"])
def test_a_reply_grammar_starts_where_the_prompt_leaves_the_reasoning(tmp_path, template, after):
    """A thinking turn reasons freely and then answers in its grammar; a turn
    whose prompt already closed the reasoning is held to it from the start."""
    from mstar.engine.resources.sampler.grammar import GRAMMAR_AFTER_KWARG
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    repo = _repo(tmp_path / "repo", None)
    (repo / "chat_template.jinja").write_text(template)
    model = Qwen3_5DenseModel(str(repo))
    kwargs = _prepared_kwargs(model, {"json_object": True})
    assert kwargs.get(GRAMMAR_AFTER_KWARG) == after and kwargs["structured_outputs"] == {"json_object": True}
    # only the data worker says where the grammar starts: a client's value is dropped
    kwargs = _prepared_kwargs(model, {"json_object": True}, **{GRAMMAR_AFTER_KWARG: "x"})
    assert kwargs.get(GRAMMAR_AFTER_KWARG) == after


def test_qwen3_5_attaches_the_requests_grammar_to_its_sampler_settings(tmp_path):
    """Qwen3.5 declares STRUCTURED_OUTPUTS, so its own sampler settings carry the request's
    grammar (the conductor then checks a sampler serves it); a bad one is the client's 400."""
    pytest.importorskip("xgrammar")
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    model = Qwen3_5DenseModel(str(_repo(tmp_path / "repo", None)))
    settings = model.get_request_resource_configs(
        ["text"], ["text"], {"temperature": 0.2, "structured_outputs": {"regex": "a+"}})["sampler"]
    assert settings.temperature == 0.2 and settings.grammar.kind == "regex"
    assert model.get_request_resource_configs(["text"], ["text"], {})["sampler"].grammar is None
    with pytest.raises(ValueError, match="structured_outputs"):
        model.get_request_resource_configs(["text"], ["text"], {"structured_outputs": {"regex": "a", "json": {}}})


def test_a_forced_tool_call_gets_its_grammar_from_the_data_worker(tmp_path):
    """``tool_choice`` required: the data worker builds Qwen3.5's call format for the sampler, past
    the reasoning a thinking prompt left open; ``auto`` gets none, and a client's own is dropped."""
    pytest.importorskip("xgrammar")
    from mstar.engine.resources.sampler.grammar import GRAMMAR_TOOLS_KWARG
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    repo = _repo(tmp_path / "repo", None)
    (repo / "chat_template.jinja").write_text(CHAT_TEMPLATE)
    model = Qwen3_5DenseModel(str(repo))
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    forced = _prepared_kwargs(model, None, tools=tools, tool_choice="required")
    assert "</think>" in forced[GRAMMAR_TOOLS_KWARG], "the call format lets the reasoning finish first"
    free = _prepared_kwargs(model, None, tools=tools, tool_choice="auto", **{GRAMMAR_TOOLS_KWARG: "x"})
    assert GRAMMAR_TOOLS_KWARG not in free


# the released checkpoints' generation prompt: a thinking block opened only when enable_thinking is true
FLAG_TEMPLATE = CHAT_TEMPLATE.replace(
    "{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\n<think>\n' }}{%- endif -%}",
    "{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is true -%}{{ '<think>\n' }}"
    "{%- else -%}{{ '<think>\n\n</think>\n\n' }}{%- endif -%}{%- endif -%}")
assert FLAG_TEMPLATE != CHAT_TEMPLATE


def _worker(model):
    from mstar.api_server.data_worker import PreprocessWorkerThread

    worker = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    worker.model, worker.device, worker._keyed_labels = model, "cpu", None
    return worker


def _starts_in_reasoning(worker, **client) -> bool:
    """What the data worker tells a chat's reply parser: does the reply start inside its reasoning."""
    from mstar.api_server.request_types import PreprocessInput

    return worker._prepare_input(PreprocessInput(
        request_id="r", text="Hello.", file_paths=None, input_modalities=["text"], output_modalities=["text"],
        model_kwargs={"prefix_cache": False, **client}, parse_reply=True,
    )).reply_in_reasoning


def _counting_scans(monkeypatch) -> list:
    from mstar.api_server import data_worker

    calls = []
    real = data_worker.after_reasoning

    def counted(text_output, prompt_ids):
        calls.append(1)
        return real(text_output, prompt_ids)
    monkeypatch.setattr(data_worker, "after_reasoning", counted)
    return calls


def test_qwen3_5_says_whether_a_reply_starts_in_its_reasoning_and_the_prompt_agrees(tmp_path, monkeypatch):
    """Qwen3.5 answers from enable_thinking, for a dict lookup; the data worker reads the prompt for
    the first answer of each kind, and after both agree reads it no more."""
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    repo = _repo(tmp_path / "repo", None)
    (repo / "chat_template.jinja").write_text(FLAG_TEMPLATE)
    worker = _worker(Qwen3_5DenseModel(str(repo)))
    scans = _counting_scans(monkeypatch)
    for _ in range(3):
        assert _starts_in_reasoning(worker) is True  # thinking on, the default
        assert _starts_in_reasoning(worker, enable_thinking=True) is True
        assert _starts_in_reasoning(worker, enable_thinking=False) is False
    assert len(scans) == 2, "one read of the prompt per kind of answer, then the model's answer alone"


@pytest.mark.parametrize("template, client, prompt_says", [
    (CHAT_TEMPLATE, {"enable_thinking": False}, True),  # a template that always opens the block
    (OTHER_TEMPLATE, {}, False),  # one that always closes it
], ids=["always-opens", "always-closes"])
def test_a_template_that_ignores_enable_thinking_is_read_from_the_prompt(tmp_path, monkeypatch, caplog,
                                                                       template, client, prompt_says):
    """The model's answer disagrees with its prompt: the prompt wins, a warning says so, and the data
    worker reads the prompt on every request from then on."""
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    repo = _repo(tmp_path / "repo", None)
    (repo / "chat_template.jinja").write_text(template)
    worker = _worker(Qwen3_5DenseModel(str(repo)))
    scans = _counting_scans(monkeypatch)
    with caplog.at_level("WARNING", logger="mstar.api_server.data_worker"):
        assert _starts_in_reasoning(worker, **client) is prompt_says
    assert "reading the prompt from now on" in caplog.text
    for flag in (True, False, None):
        kwargs = {} if flag is None else {"enable_thinking": flag}
        assert _starts_in_reasoning(worker, **kwargs) is prompt_says
    assert len(scans) == 4, "after a disagreement, every request reads its prompt"


CHATS = [
    [{"role": "user", "content": "What's 2 + 2?"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi."}],
    [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello!", "reasoning_content": "Greet."},
     {"role": "user", "content": "How are you?"}],
]


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("chat", range(len(CHATS)))
def test_on_the_released_template_qwen3_5s_answer_is_the_prompts(chat, thinking):
    """On the checkpoint's own template (tools included), the answer Qwen3.5 gives from its settings
    is the one reading the rendered prompt gives."""
    import os

    import torch

    from mstar.engine.resources.sampler.grammar import after_reasoning
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    path = os.environ.get("QWEN3_5_CKPT")
    if not path:
        pytest.skip("set QWEN3_5_CKPT to the checkpoint's directory")
    model = Qwen3_5DenseModel(path)
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    for with_tools in (None, tools):
        text = model.tokenizer.apply_chat_template(CHATS[chat], tools=with_tools, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=thinking)
        ids = torch.tensor(model.tokenizer.encode(text, add_special_tokens=False))
        from_prompt = after_reasoning(model.text_output(), ids) is not None
        assert model.reply_starts_in_reasoning({"enable_thinking": thinking}) is from_prompt is thinking
