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


@pytest.mark.parametrize("template, after", [(CHAT_TEMPLATE, "</think>"), (OTHER_TEMPLATE, None)],
                         ids=["thinking", "reasoning closed"])
def test_a_reply_grammar_starts_where_the_prompt_leaves_the_reasoning(tmp_path, template, after):
    """A thinking turn reasons freely and then answers in its grammar; a turn
    whose prompt already closed the reasoning is held to it from the start."""
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    repo = _repo(tmp_path / "repo", None)
    (repo / "chat_template.jinja").write_text(template)
    model = Qwen3_5DenseModel(str(repo))
    out = model.process_prompt("Reply in JSON.", ["text"], ["text"], structured_outputs={"json_object": True})
    assert out.metadata.get("structured_outputs", {}).get("after") == after
