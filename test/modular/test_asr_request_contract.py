"""The generation-input contract for Canary, Canary-Qwen and Qwen3-ASR: the keys
each reads, its greedy and checkpoint sampling defaults, the sampler
capabilities those defaults turn on, the typed knobs ``process_prompt`` refuses,
and an output length past the decode loop refused by the conductor, not clamped."""

import json
from types import SimpleNamespace

import pytest
import torch
from test_canary_model import snapshot  # noqa: F401  (fixture)
from test_canary_qwen_model import _model_with_word_tokenizer
from test_qwen3_asr_model import _make_model as _qwen3_asr

from mstar.conductor.conductor import Conductor
from mstar.engine.state.adapters import node_resources, sampling_configs
from mstar.model.canary.canary_model import CanaryModel
from mstar.model.canary.config import CanaryModelConfig
from mstar.model.canary_qwen.canary_qwen_model import CanaryQwenModel
from mstar.model.components.fastconformer import FastConformerConfig
from mstar.model.components.request_kwargs import SAMPLING_KEYS, sampling_defaults

SAMPLER = "sampler"
AUDIO = {"audio_inputs": [torch.zeros(16_000)]}
# Qwen/Qwen3-ASR-{0.6B,1.7B}'s generation_config.json, less its token ids
QWEN3_ASR_GENERATION_CONFIG = {"do_sample": False, "temperature": 0.000001}


def _canary() -> CanaryModel:
    model = object.__new__(CanaryModel)
    model.config = CanaryModelConfig(encoder=FastConformerConfig())
    return model


def _canary_qwen() -> CanaryQwenModel:
    model = _model_with_word_tokenizer()
    model.config.llm = SimpleNamespace(
        num_hidden_layers=1, num_key_value_heads=1, head_dim=8, num_attention_heads=1, rope_theta=1e6, vocab_size=16,
    )
    model.config.encoder = SimpleNamespace(output_lengths=lambda n: 3)
    return model


MODELS = {"canary": _canary, "canary_qwen": _canary_qwen, "qwen3_asr": _qwen3_asr}
READS = {
    "canary": {"language", "target_language", "punctuation"},
    "canary_qwen": {"instruction", "language"},
    "qwen3_asr": {"language", "initial_prompt", "assistant_prefix"},
}


def _sampler(model, **kw):
    # Canary's settings are its declared sampler's, which convert to the resource's config
    return sampling_configs(model.get_request_resource_configs([], [], kw), model.declare_state())[SAMPLER]


def _spec(model):
    return next(s for s in node_resources(model) if s.resource_key == SAMPLER)


def _loop_conductor(model) -> Conductor:
    """A conductor with only what ``_loop_caps`` reads: the model and its loops' ``max_iters``."""
    c = Conductor.__new__(Conductor)
    c.model = model
    c._loop_max_iters = {}
    for walk in model.get_graph_walk_graphs().values():
        for name, loop in walk.get_loops().items():
            c._loop_max_iters[name] = max(loop.max_iters, c._loop_max_iters.get(name, 0))
    return c


@pytest.mark.parametrize("name", MODELS)
def test_request_kwargs_are_the_keys_it_reads(name):
    """Not the adapter's ``initial_prompt`` / ``timestamps`` / ``task`` unless read: those are logged as ignored."""
    assert MODELS[name]().request_kwargs() == {*READS[name], "max_output_tokens", *SAMPLING_KEYS}


@pytest.mark.parametrize("name", MODELS)
def test_default_is_greedy_and_every_knob_is_forwarded(name):
    model = MODELS[name]()
    cfg = _sampler(model)
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.min_p, cfg.repetition_penalty) == (0.0, 1.0, 0, 0.0, 1)
    knobs = dict(temperature=0.4, top_p=0.9, top_k=5, min_p=0.1, repetition_penalty=1.2,
                 penalize_prompt=False, ignore_eos=True)
    cfg = _sampler(model, **knobs)
    assert {k: getattr(cfg, k) for k in knobs} == knobs


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("kw", [{"repetition_penalty": 1.2}, {"min_p": 0.1}])
def test_penalty_and_min_p_are_refused_without_a_checkpoint_default(name, kw):
    model = MODELS[name]()
    with pytest.raises(ValueError, match="does not support"):
        _sampler(model, **kw).validate(_spec(model))
    _sampler(model).validate(_spec(model))
    # a checkpoint that defaults them turns the capability on
    model.sampling_defaults = {"temperature": 0.0, "top_p": 1.0, "min_p": 0.05, "repetition_penalty": 1.1}
    _sampler(model).validate(_spec(model))
    _sampler(model, **kw).validate(_spec(model))


def test_checkpoint_generation_config_sets_the_defaults(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"temperature": 0.7, "top_k": 3, "repetition_penalty": 1.1, "max_new_tokens": 10}
    ))
    assert sampling_defaults(tmp_path) == {"temperature": 0.7, "top_p": 1.0, "top_k": 3, "repetition_penalty": 1.1}
    # Qwen3-ASR's: do_sample false is greedy
    (tmp_path / "generation_config.json").write_text(json.dumps(QWEN3_ASR_GENERATION_CONFIG))
    assert sampling_defaults(tmp_path) == {"temperature": 0.0, "top_p": 1.0}
    # canary-qwen-2.5b has no file
    (tmp_path / "generation_config.json").unlink()
    assert sampling_defaults(tmp_path) == {"temperature": 0.0, "top_p": 1.0}


def test_canary_reads_its_checkpoints_generation_config(snapshot):  # noqa: F811
    (snapshot / "generation_config.json").write_text(json.dumps({"bos_token_id": 4, "eos_token_id": 3}))
    model = CanaryModel(str(snapshot))
    assert model.sampling_defaults == {"temperature": 0.0, "top_p": 1.0}
    (snapshot / "generation_config.json").write_text(json.dumps({"do_sample": True, "temperature": 0.5, "min_p": 0.1}))
    model = CanaryModel(str(snapshot))
    assert _sampler(model).temperature == 0.5 and _spec(model).enable_min_p


@pytest.mark.parametrize("kw", [
    {"language": 5}, {"language": "xx"}, {"target_language": ["en"]}, {"target_language": "zh"},
    {"punctuation": "yes"}, {"punctuation": 1}, {"temperature": "0.2"},
])
def test_canary_bad_knob_is_a_value_error_in_process_prompt(snapshot, kw):  # noqa: F811
    model = CanaryModel(str(snapshot))
    with pytest.raises(ValueError):
        model.process_prompt(None, ["audio"], ["text"], tensors=dict(AUDIO), **kw)


@pytest.mark.parametrize("kw", [
    {"instruction": 5}, {"instruction": ["x"]}, {"language": "de"}, {"language": 5}, {"top_p": "1"},
])
def test_canary_qwen_bad_knob_is_a_value_error_in_process_prompt(kw):
    model = _canary_qwen()
    with pytest.raises(ValueError):
        model.process_prompt(None, ["audio"], ["text"], tensors=dict(AUDIO), **kw)
    # English in any spelling is the default
    model.process_prompt(None, ["audio"], ["text"], tensors=dict(AUDIO), language="en-US")


@pytest.mark.parametrize("kw", [
    {"language": 5}, {"language": "Klingon"}, {"initial_prompt": 5}, {"assistant_prefix": ["x"]},
    {"max_output_tokens": "32"},
])
def test_qwen3_asr_bad_knob_is_a_value_error_in_process_prompt(kw):
    model = _qwen3_asr()
    with pytest.raises(ValueError):
        model.process_prompt(None, ["audio"], ["text"], dict(AUDIO), **kw)


@pytest.mark.parametrize("name,default", [("canary", 1024 - 10), ("canary_qwen", CanaryQwenModel.MAX_OUTPUT_TOKENS)])
def test_canary_over_limit_is_refused_not_clamped(name, default):
    """The prefill samples the first token, so the decode loop runs ``default - 1``;
    one past the default needs one more iteration than the loop has."""
    model = MODELS[name]()
    assert model.get_max_output_tokens() == default
    assert model.get_max_output_tokens(max_output_tokens=default + 1) == default + 1
    c = _loop_conductor(model)
    assert c._loop_caps({"max_output_tokens": 10}) == {"decode_loop": 9}
    c._loop_caps({"max_output_tokens": default})
    with pytest.raises(ValueError, match=f"at most {default - 1}"):
        c._loop_caps({"max_output_tokens": default + 1})


def test_qwen3_asr_length_past_the_default_is_served_to_the_loop_bound():
    """4096 is a default, not a bound: the KV pool's room decides the rest."""
    model = _qwen3_asr()
    c = _loop_conductor(model)
    (max_iters,) = c._loop_max_iters.values()
    assert model.get_max_output_tokens() == 4096
    c._loop_caps({"max_output_tokens": 8192})
    c._loop_caps({"max_output_tokens": max_iters + 1})
    with pytest.raises(ValueError, match="at most"):
        c._loop_caps({"max_output_tokens": max_iters + 2})
