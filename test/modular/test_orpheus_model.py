import dataclasses
import json
import sys
import types

import pytest
import torch

from mstar.conductor.conductor import Conductor
from mstar.model.orpheus.config import SAMPLER, OrpheusModelConfig
from mstar.model.orpheus.orpheus_model import OrpheusModel, _checkpoint_sampling_defaults

sys.path.insert(0, ".")


from mstar.conductor.request_info import CurrentForwardConductorMetadata


def _make_model() -> OrpheusModel:
    model = object.__new__(OrpheusModel)
    model.config = OrpheusModelConfig()
    return model


def test_orpheus_prefill_transitions_to_decode():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["audio"],
        graph_walk="prefill",
        is_prefill=True,
    )

    result = model.get_partition_forward_pass_args(
        partition_name="LLM",
        partition_metadata=metadata,
        persist_signals={"new_token": []},
    )

    assert result.full_metadata.graph_walk == "decode"
    assert result.step_metadata["is_prefill"] is False
    assert result.request_done is False
    assert "decode_finished" not in result.full_metadata.kwargs


def test_orpheus_decode_eos_marks_done():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["audio"],
        graph_walk="decode",
        is_prefill=False,
        kwargs={
            "decode_finished": False,
        },
    )

    result = model.get_partition_forward_pass_args(
        partition_name="LLM",
        partition_metadata=metadata,
        persist_signals={},
    )

    assert result.request_done is True
    assert result.full_metadata.kwargs["decode_finished"] is True



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


def test_limit_is_the_decode_loop_bound():
    model = _model()
    assert model.get_max_output_tokens_limit() == 2048
    assert model.get_graph_walk_graphs()["decode"].max_iters == 2048
    c = Conductor.__new__(Conductor)
    c.model = model
    assert c._max_output_tokens({"max_output_tokens": 2048}) == 2048
    with pytest.raises(ValueError, match="at most 2048"):
        c._max_output_tokens({"max_output_tokens": 2049})


def test_config_defaults_and_forwarded_knobs():
    model = _model()
    cfg = model.get_request_resource_configs({}, {})[SAMPLER]
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.repetition_penalty) == (0.6, 0.8, 0, 1.3)
    cfg = model.get_request_resource_configs({}, {"top_k": 50, "penalize_prompt": False})[SAMPLER]
    assert (cfg.top_k, cfg.penalize_prompt) == (50, False)


def test_min_p_is_refused_unless_the_checkpoint_defaults_it():
    model = _model()
    spec = next(s for s in model.get_node_resources() if s.resource_key == SAMPLER)
    with pytest.raises(ValueError, match="min_p"):
        model.get_request_resource_configs({}, {"min_p": 0.1})[SAMPLER].validate(spec)
    model = _model(dataclasses.replace(OrpheusModelConfig(), min_p=0.05))
    spec = next(s for s in model.get_node_resources() if s.resource_key == SAMPLER)
    model.get_request_resource_configs({}, {})[SAMPLER].validate(spec)


def test_absent_voice_is_the_default():
    model = _model()
    model.process_prompt("hi", ["text"], ["audio"])
    assert model.tokenizer.seen == ["tara: hi"]


@pytest.mark.parametrize("voice", ["", 3, ["tara"], "nobody"])
def test_bad_voice_is_a_value_error(voice):
    with pytest.raises(ValueError, match="voice"):
        _model().process_prompt("hi", ["text"], ["audio"], voice=voice)


def test_checkpoint_generation_config_overrides_config_defaults(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 9000}
    ))
    defaults = _checkpoint_sampling_defaults(str(tmp_path), None)
    assert defaults == {"temperature": 0.7, "top_p": 0.9}
    config = dataclasses.replace(OrpheusModelConfig(), **defaults)
    assert (config.temperature, config.top_p, config.repetition_penalty) == (0.7, 0.9, 1.3)


def test_unreachable_checkpoint_keeps_config_defaults(monkeypatch):
    import huggingface_hub

    def gated(*a, **kw):
        raise OSError("401 gated")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", gated)
    assert _checkpoint_sampling_defaults("canopylabs/orpheus-3b-0.1-ft", None) == {}
    assert _checkpoint_sampling_defaults("", None) == {}
