"""Models resolve a checkpoint through ``mstar.utils.hf_snapshot``: a download
that fails raises, and each model fetches the files its loaders read."""
import importlib

import huggingface_hub
import pytest


@pytest.mark.parametrize("model", [
    "orpheus.orpheus_model.OrpheusModel", "whisper.whisper_model.WhisperModel",
    "qwen3_omni.qwen3_omni_model.Qwen3OmniModel",
])
def test_a_failed_checkpoint_download_raises_instead_of_returning_the_repo_id(monkeypatch, model):
    """The repo id handed back in place of a snapshot hid the download's own
    error (#681): loading failed later, on a path that does not exist."""
    def not_cached(*args, **kwargs):
        raise huggingface_hub.errors.LocalEntryNotFoundError("not in the cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)
    module, cls = model.rsplit(".", 1)
    with pytest.raises(huggingface_hub.errors.LocalEntryNotFoundError):
        getattr(importlib.import_module(f"mstar.model.{module}"), cls)("org/repo")


def test_the_shared_patterns_keep_a_chat_template_saved_as_jinja():
    """transformers saves a chat template as ``chat_template.jinja`` by default;
    filtered out, a cold download serves a processor that can't format a chat."""
    from huggingface_hub.utils import filter_repo_objects

    from mstar.utils.hf_snapshot import SAFETENSORS_SNAPSHOT

    files = ["config.json", "model.safetensors", "chat_template.jinja", "nano/chat_template.jinja"]
    assert list(filter_repo_objects(files, allow_patterns=SAFETENSORS_SNAPSHOT)) == files
