import dataclasses
import json
import sys
import types

import pytest
import torch

from mstar.conductor.conductor import Conductor
from mstar.engine.state.adapters import node_resources, sampling_configs
from mstar.model.orpheus.config import SAMPLER, OrpheusModelConfig
from mstar.model.orpheus.orpheus_model import OrpheusModel, _checkpoint_sampling_defaults

sys.path.insert(0, ".")


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



# ── generation-input contract ───────────────────────────────────────────────

class _Tokenizer:
    def __init__(self):
        self.seen = []

    def __call__(self, text, return_tensors=None):
        self.seen.append(text)
        return types.SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))


def _model(config=None) -> OrpheusModel:
    model = object.__new__(OrpheusModel)
    model.config = config or OrpheusModelConfig()
    model.tokenizer = _Tokenizer()
    return model


def test_request_kwargs_are_the_keys_it_reads():
    keys = _model().request_kwargs()
    for read in ("voice", "temperature", "top_p", "top_k", "min_p",
                 "repetition_penalty", "penalize_prompt", "ignore_eos", "max_output_tokens"):
        assert read in keys


def test_default_output_cap():
    model = _model()
    assert model.get_max_output_tokens() == 2048
    c = Conductor.__new__(Conductor)
    c.model = model
    assert c._max_output_tokens({"max_output_tokens": 4096}) == 4096


def test_config_defaults_and_forwarded_knobs():
    model = _model()
    cfg = model.get_request_resource_configs([], [], {})[SAMPLER]
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.repetition_penalty) == (0.6, 0.8, 0, 1.3)
    cfg = model.get_request_resource_configs([], [], {"top_k": 50, "penalize_prompt": False})[SAMPLER]
    assert (cfg.top_k, cfg.penalize_prompt) == (50, False)


def _sampling(model, model_kwargs: dict):
    """The model's sampler spec and a request's config for it, as the engine builds and the conductor resolves them."""
    spec = next(s for s in node_resources(model) if s.resource_key == SAMPLER)
    config = sampling_configs(model.get_request_resource_configs([], [], model_kwargs), model.declare_state())
    return spec, config[SAMPLER]


def test_min_p_is_refused_unless_the_checkpoint_defaults_it():
    spec, config = _sampling(_model(), {"min_p": 0.1})
    with pytest.raises(ValueError, match="min_p"):
        config.validate(spec)
    spec, config = _sampling(_model(dataclasses.replace(OrpheusModelConfig(), min_p=0.05)), {})
    config.validate(spec)


def test_absent_voice_is_the_default():
    model = _model()
    model.process_prompt("hi", ["text"], ["audio"])
    assert model.tokenizer.seen == ["tara: hi"]


@pytest.mark.parametrize("voice", ["", 3, ["tara"], "nobody"])
def test_bad_voice_is_a_value_error(voice):
    with pytest.raises(ValueError, match="voice"):
        _model().process_prompt("hi", ["text"], ["audio"], voice=voice)


def test_checkpoint_knob_values_are_ignored_but_do_sample_false_is_greedy(tmp_path):
    # the file Orpheus ships is Llama-3.2-3B's chat preset, not Orpheus's recipe
    llama = {"do_sample": True, "temperature": 0.6, "top_p": 0.9, "max_new_tokens": 9000}
    (tmp_path / "generation_config.json").write_text(json.dumps(llama))
    assert _checkpoint_sampling_defaults(str(tmp_path), None) == {}
    (tmp_path / "generation_config.json").write_text(json.dumps({**llama, "do_sample": False}))
    defaults = _checkpoint_sampling_defaults(str(tmp_path), None)
    assert defaults == {"temperature": 0.0}
    config = dataclasses.replace(OrpheusModelConfig(), **defaults)
    assert (config.temperature, config.top_p, config.repetition_penalty) == (0.0, 0.8, 1.3)


def test_unreachable_checkpoint_keeps_config_defaults(monkeypatch):
    import huggingface_hub

    def gated(*a, **kw):
        raise OSError("401 gated")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", gated)
    assert _checkpoint_sampling_defaults("canopylabs/orpheus-3b-0.1-ft", None) == {}
    assert _checkpoint_sampling_defaults("", None) == {}
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
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw) or "/snap")
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


def test_orpheus_tp2_config_validates_and_covers_all_walks():
    """Shipped configs/orpheus_tp2.yaml: TP2 LLM feeding TP1 snac_decoder
    validates cleanly, places worker_0 and worker_1 with matching subgroup
    tuples, and captures every declared walk (#706)."""
    from pathlib import Path

    import yaml

    from mstar.model.orpheus.submodules import SNACDecoderSubmodule
    from mstar.validation import validate_deployment

    cfg_path = Path("configs/orpheus_tp2.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    model = _make_model()

    validate_deployment(model, cfg, str(cfg_path))

    groups = {tuple(g["node_names"]): g for g in cfg["node_groups"]}
    assert groups[("LLM",)]["ranks"] == [0, 1]
    assert groups[("LLM",)]["tp_size"] == 2
    assert groups[("snac_decoder",)]["ranks"] == [0]

    llm = OrpheusLLMSubmodule(None, model.config)
    snac = SNACDecoderSubmodule(torch.nn.Linear(1, 1), model.config)
    llm_captured = {
        walk for c in llm.get_accelerator_graph_configs("cpu") for walk in c.replay_graph_walks
    }
    snac_captured = {
        walk for c in snac.get_accelerator_graph_configs("cpu") for walk in c.replay_graph_walks
    }
    assert set(groups[("LLM",)]["graph_walks"]) <= llm_captured
    assert set(groups[("snac_decoder",)]["graph_walks"]) <= snac_captured
