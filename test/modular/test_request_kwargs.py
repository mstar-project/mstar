"""The ASR models check request kwargs on the data worker and again where the
conductor reads them, so a stray form field cannot reach a step. The API has
already normalized the generation knobs, so the check coerces nothing. A
declared sampler gets every field of the request's sampling config, so no knob
falls back to its default on the way, and the same flags the model's defaults
give a sampler on resources."""
from dataclasses import fields, replace

import pytest

from mstar.engine.resources.sampler.config import SamplerSpec, SamplingReqConfig
from mstar.engine.state import Sampler, SampleSettings
from mstar.engine.state.adapters import build_specs
from mstar.model.components.request_kwargs import (
    checked_request_kwargs,
    sample_settings,
    sampler_capabilities,
    sampler_flags,
)
from mstar.utils.generation_kwargs import normalize_generation_kwargs


def test_known_fields_pass_unchanged():
    kwargs = {
        "max_output_tokens": 32, "seed": 7, "temperature": 0.2, "top_p": 1,
        "ignore_eos": False, "language": "en", "timestamps": "word", "extra": object(),
    }
    assert checked_request_kwargs(kwargs) == kwargs  # unknown fields pass through untouched


@pytest.mark.parametrize("bad", [
    {"max_output_tokens": "abc"}, {"max_output_tokens": 0}, {"max_output_tokens": True},
    {"temperature": "hot"}, {"temperature": -1}, {"temperature": float("nan")},
    {"top_p": 0}, {"top_p": 1.5}, {"ignore_eos": "maybe"}, {"language": 5},
    {"timestamps": "letters"}, {"seed": 2**63}, {"seed": -2**64},
    # strings the API would have normalized are not coerced here
    {"max_output_tokens": "32"}, {"temperature": "0.2"}, {"ignore_eos": "false"}, {"top_k": 5.0},
])
def test_bad_values_name_the_field(bad):
    (key,) = bad
    with pytest.raises(ValueError, match=key):
        checked_request_kwargs(bad)


def test_what_the_api_normalized_passes():
    raw = {"max_output_tokens": "32", "temperature": "0.2", "top_k": 5.0, "seed": "7", "language": "en"}
    normalized = normalize_generation_kwargs(raw)
    assert checked_request_kwargs(normalized) == normalized == {
        "max_output_tokens": 32, "temperature": 0.2, "top_k": 5, "seed": 7, "language": "en",
    }


def test_null_is_unset():
    assert checked_request_kwargs(None) == {}
    assert checked_request_kwargs({"temperature": None, "language": None}) == {}


def test_sample_settings_carry_every_field_a_sampling_config_sets():
    knobs = {
        "temperature": 0.3, "top_k": 7, "top_p": 0.8, "min_p": 0.05, "repetition_penalty": 1.2, "repetition_window": 16,
        "penalize_prompt": False, "min_tokens": 3, "ignore_eos": True, "top_p_first": True, "top_p_min_keep": 2,
        "return_logprobs": True, "logprobs_mode": "processed",
    }
    assert knobs.keys() == {field.name for field in fields(SampleSettings)}, "a field with no knob here goes unchecked"
    settings = sample_settings(SamplingReqConfig(**knobs))
    for name, value in knobs.items():
        assert value != getattr(SampleSettings(), name), f"{name} is set to its default, so it proves nothing"
        assert getattr(settings, name) == value, f"the request's {name} would reach the sampler as its default"


def test_sample_settings_and_sampling_configs_default_alike():
    names = {field.name for field in fields(SampleSettings)}
    assert {field.name for field in fields(SamplingReqConfig)} == names | {"_seed"}, (
        "a field on one class only would never reach a declared sampler, or be set from a request"
    )
    for name in names:
        assert getattr(SampleSettings(), name) == getattr(SamplingReqConfig(), name), (
            f"an unset {name} would sample differently on the state contract"
        )


@pytest.mark.parametrize("defaults, penalty, min_p", [
    ({"temperature": 0.0, "top_p": 1.0}, False, False),
    ({"temperature": 0.7, "min_p": 0.0}, False, False),
    ({"temperature": 0.7, "min_p": 0.05}, False, True),
    ({"temperature": 0.9, "top_p": 0.95, "repetition_penalty": 1.04}, True, False),
], ids=["greedy", "min-p-zero", "min-p", "penalty"])
def test_a_declared_samplers_flags_are_its_defaults_capabilities(defaults, penalty, min_p):
    (spec,) = build_specs([Sampler("sampler", nodes={"LLM"}, vocab_size=8, **sampler_flags(defaults))])
    on_resources = SamplerSpec(resource_key="sampler", nodes={"LLM"}, vocab_size=8)
    on_resources = replace(on_resources, **sampler_capabilities(defaults))
    assert (spec.enable_repetition_penalty, spec.enable_min_p) == (penalty, min_p), "a request's knob would be refused"
    assert spec == on_resources, "the declared sampler would take other flags than the same defaults give on resources"
