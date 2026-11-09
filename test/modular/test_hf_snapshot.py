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
