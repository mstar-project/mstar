"""What GLM-5.2 takes of a request's kwargs: an unset one is the default, and
one it cannot read is refused in process_prompt, the data worker's 400, before
the conductor reads it and any rank runs the request."""

import pytest

from mstar.engine.resources import SamplerSpec
from mstar.model.glm52.config import SAMPLER_RESOURCE
from mstar.model.glm52.glm52_model import Glm52Model


def _model(**kwargs) -> Glm52Model:
    # byte tokenizer: no HF IO
    return Glm52Model(model_path_hf="", tokenizer_mode="byte", **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"top_k": None}, {"temperature": None, "top_p": None}, {"repetition_penalty": None},
    {"max_output_tokens": None}, {"seed": None},
])
def test_a_null_kwarg_is_unset(kwargs):
    m = _model()
    sampling = m.get_request_resource_configs({}, kwargs)[SAMPLER_RESOURCE]
    default = m.get_request_resource_configs({}, {})[SAMPLER_RESOURCE]
    assert sampling == default
    assert m.get_max_output_tokens(**kwargs) == m.get_max_output_tokens()
    assert m.process_prompt("Hi", ["text"], ["text"], **kwargs)["text_inputs"][0].tolist() == [72, 105]


@pytest.mark.parametrize("kwargs", [
    {"top_k": "many"}, {"temperature": "hot"}, {"top_p": [0.9]}, {"repetition_penalty": float("nan")},
    {"max_output_tokens": "100x"}, {"max_output_tokens": 1.5}, {"seed": "abc"},
    {"top_k": "1e30"}, {"seed": 2**63}, {"seed": -(2**63) - 1},
    {"temperature": -1}, {"top_p": 0}, {"top_p": -0.5}, {"top_p": 1.5},
    {"repetition_penalty": 0},
])
def test_a_kwarg_it_cannot_read_is_refused_in_process_prompt(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        _model().process_prompt("Hi", ["text"], ["text"], **kwargs)


def test_the_edges_of_each_range_are_served():
    kwargs = {"temperature": 0, "top_p": 1, "repetition_penalty": 1}
    assert _model().process_prompt("Hi", ["text"], ["text"], **kwargs)["text_inputs"][0].tolist() == [72, 105]


@pytest.mark.parametrize("kwargs", [{"temperature": "0.7"}, {"top_k": "20"}, {"max_output_tokens": "64"}])
def test_numbers_sent_as_strings_are_refused(kwargs):
    # one meaning per knob on every model (mstar/utils/generation_kwargs.py): types are strict
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        _model().process_prompt("Hi", ["text"], ["text"], **kwargs)


def test_min_p_is_refused_in_process_prompt():
    m = _model()
    # the sampler that would read it is built without it
    [spec] = [s for s in m.get_node_resources() if isinstance(s, SamplerSpec)]
    assert not spec.enable_min_p
    with pytest.raises(ValueError, match="min_p"):
        m.process_prompt("Hi", ["text"], ["text"], min_p=0.05)
    assert m.process_prompt("Hi", ["text"], ["text"], min_p=0.0)["text_inputs"][0].numel() == 2


def test_a_request_without_max_tokens_runs_to_the_window():
    # a thinking reply was cut at a fixed 1024 tokens with "length", on any window
    m = _model(dsa_long_context=True, max_seq_len=8192)
    assert m.get_max_output_tokens() == m.get_max_output_tokens(max_output_tokens=None) == 8192
    assert m.get_max_output_tokens(max_output_tokens=100) == 100


@pytest.mark.parametrize("kwargs", [{"repetition_penalty": 1.1}])
def test_mtp_refuses_sampling_it_cannot_serve_in_process_prompt(kwargs):
    with pytest.raises(ValueError, match="repetition_penalty"):
        _model(mtp_num_draft_tokens=2).process_prompt("Hi", ["text"], ["text"], **kwargs)
    # without MTP it samples
    assert _model().process_prompt("Hi", ["text"], ["text"], **kwargs)["text_inputs"][0].numel() == 2


def test_mtp_samples_with_temperature():
    m = _model(mtp_num_draft_tokens=2)
    assert m.process_prompt("Hi", ["text"], ["text"], temperature=0.7, top_p=0.9)["text_inputs"][0].numel() == 2


@pytest.mark.parametrize("kwargs", [{}, {"temperature": 0}, {"temperature": None}])
def test_mtp_serves_greedy(kwargs):
    m = _model(mtp_num_draft_tokens=2)
    assert m.process_prompt("Hi", ["text"], ["text"], **kwargs)["text_inputs"][0].numel() == 2
    assert m.get_request_resource_configs({}, kwargs)[SAMPLER_RESOURCE].temperature == 0
