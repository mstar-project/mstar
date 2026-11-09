import sys

import pytest

sys.path.insert(0, ".")


from mstar.model.orpheus.config import OrpheusModelConfig
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.model.orpheus.submodules import OrpheusLLMSubmodule


def _make_model() -> OrpheusModel:
    model = object.__new__(OrpheusModel)
    model.config = OrpheusModelConfig()
    return model


def test_the_llm_captures_every_walk_it_runs():
    """A walk no capture replays runs eagerly, and nothing reports it."""
    model = _make_model()
    llm = OrpheusLLMSubmodule(None, model.config)

    captured = {walk for config in llm.get_accelerator_graph_configs("cpu") for walk in config.replay_graph_walks}
    runs = {walk for walk, graph in model.get_graph_walk_graphs().items() if "LLM" in graph.get_nodes()}

    assert runs | set(model.get_combined_graph_walks()["LLM"]) <= captured


def test_no_checkpoint_download_fetches_the_training_state(monkeypatch):
    """The Orpheus repos ship ~72 GB of optimizer and FSDP state next to the
    weights (#681): the tokenizer, checkpoint_path and the LLM load each fetch
    only what they read."""
    from fnmatch import fnmatch

    import huggingface_hub

    from mstar.distributed.communication import CommGroup
    from mstar.model import loader
    from mstar.model.orpheus import orpheus_model

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: calls.append(kw) or "/snap")
    monkeypatch.setattr(orpheus_model.AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(loader, "load_weights", lambda *a, **k: None)
    model = OrpheusModel("canopylabs/orpheus-3b-0.1-ft")
    model.checkpoint_path()
    model._create_llm_submodule("meta", tp_group=CommGroup.trivial())

    def fetched(name, call):
        return any(fnmatch(name, p) for p in call["allow_patterns"] or ["*"])

    assert len(calls) == 3
    for call in calls:
        skipped = ["optimizer.bin", "pytorch_model_fsdp.bin", "rng_state_0.pth", "scheduler.pt", "training_args.bin"]
        assert not any(fetched(f, call) for f in skipped), call
        assert fetched("tokenizer.json", call), call
    for call in calls[1:]:
        assert all(fetched(f, call) for f in ["config.json", "model.safetensors.index.json",
                                               "model-00001-of-00004.safetensors"]), call


@pytest.mark.parametrize("model", [
    "orpheus.orpheus_model.OrpheusModel", "whisper.whisper_model.WhisperModel",
    "qwen3_omni.qwen3_omni_model.Qwen3OmniModel",
])
def test_a_failed_checkpoint_download_raises_instead_of_returning_the_repo_id(monkeypatch, model):
    """The repo id handed back in place of a snapshot hid the download's own
    error (#681): loading failed later, on a path that does not exist."""
    import importlib

    import huggingface_hub

    def not_cached(*args, **kwargs):
        raise huggingface_hub.errors.LocalEntryNotFoundError("not in the cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)
    module, cls = model.rsplit(".", 1)
    with pytest.raises(huggingface_hub.errors.LocalEntryNotFoundError):
        getattr(importlib.import_module(f"mstar.model.{module}"), cls)("org/repo")
