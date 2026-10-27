"""What GLM-5.2 takes of a request's kwargs: an unset one is the default, and one it cannot
read is refused at the conductor's intake, the client's 400, before any rank runs the request."""

import pytest

from mstar.conductor.conductor import Conductor
from mstar.engine.resources import SamplerSpec
from mstar.model.glm52.config import SAMPLER_RESOURCE
from mstar.model.glm52.glm52_model import Glm52Model


def _model(**kwargs) -> Glm52Model:
    # byte tokenizer: no HF IO
    return Glm52Model(model_path_hf="", tokenizer_mode="byte", **kwargs)


def _intake(model: Glm52Model, kwargs: dict) -> int:
    """The conductor's checks of a new request's kwargs, before it is dispatched; there a
    ValueError is the client's 400 (``Conductor._admit_request``)."""
    conductor = Conductor.__new__(Conductor)
    conductor.model, conductor.model_config = model, {}
    conductor._kv_labels = {key: tuple(labels) for key, labels in model.kv_labels().items()}
    conductor._loop_max_iters = {
        name: loop.max_iters for walk in model.get_graph_walk_graphs().values()
        for name, loop in walk.get_loops().items()
    }
    # in the order _admit_request runs them
    max_output_tokens = conductor._max_output_tokens(kwargs)
    conductor._loop_caps(kwargs)
    specs = conductor._resource_specs()
    for key, cfg in conductor._get_resource_configs(kwargs, {}).items():
        cfg.validate(specs.get(key))
    return max_output_tokens


@pytest.mark.parametrize("kwargs", [
    {"top_k": None}, {"temperature": None, "top_p": None}, {"repetition_penalty": None},
    {"max_output_tokens": None}, {"seed": None},
])
def test_a_null_kwarg_is_unset(kwargs):
    m = _model()
    sampling = m.get_request_resource_configs({}, kwargs)[SAMPLER_RESOURCE]
    default = m.get_request_resource_configs({}, {})[SAMPLER_RESOURCE]
    assert sampling == default
    assert _intake(m, kwargs) == _intake(m, {}) == m.get_max_output_tokens()
    out = m.process_prompt("Hi", ["text"], ["text"], **kwargs)
    assert out.new_input_tensors["text_inputs"][0].tolist() == [72, 105]


@pytest.mark.parametrize("kwargs", [
    {"top_k": "many"}, {"temperature": "hot"}, {"top_p": [0.9]}, {"repetition_penalty": float("nan")},
    {"max_output_tokens": "100x"}, {"max_output_tokens": 1.5}, {"seed": "abc"},
    {"top_k": "1e30"}, {"seed": 2**63}, {"seed": -(2**63) - 1},
    {"temperature": -1}, {"top_p": 0}, {"top_p": -0.5}, {"top_p": 1.5},
    {"repetition_penalty": 0},
])
def test_a_kwarg_it_cannot_read_is_refused_at_intake(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        _intake(_model(), kwargs)


def test_the_edges_of_each_range_are_served():
    assert _intake(_model(), {"temperature": 0, "top_p": 1, "repetition_penalty": 1}) == 2048


@pytest.mark.parametrize("kwargs", [{"temperature": "0.7"}, {"top_k": "20"}, {"max_output_tokens": "64"}])
def test_numbers_sent_as_strings_are_refused(kwargs):
    # one meaning per knob on every model (mstar/utils/generation_kwargs.py): types are strict
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        _intake(_model(), kwargs)


def test_min_p_is_refused_at_intake():
    m = _model()
    # the sampler that would read it is built without it
    [spec] = [s for s in m.get_node_resources() if isinstance(s, SamplerSpec)]
    assert not spec.enable_min_p
    with pytest.raises(ValueError, match="min_p"):
        _intake(m, {"min_p": 0.05})
    _intake(m, {"min_p": 0.0})


def test_max_tokens_past_the_window_is_refused_at_intake():
    m = _model()
    assert _intake(m, {"max_output_tokens": 2048}) == 2048
    with pytest.raises(ValueError, match="runs at most 2047"):
        _intake(m, {"max_output_tokens": 2049})


def test_a_request_without_max_tokens_runs_to_the_window():
    # a thinking reply was cut at a fixed 1024 tokens with "length", on any window
    m = _model(dsa_long_context=True, max_seq_len=8192)
    assert m.get_max_output_tokens() == m.get_max_output_tokens(max_output_tokens=None) == 8192
    assert m.get_max_output_tokens(max_output_tokens=100) == 100
    assert _intake(m, {}) == 8192


@pytest.mark.parametrize("kwargs", [{"temperature": 0.7}, {"repetition_penalty": 1.1}])
def test_mtp_refuses_sampling_it_cannot_serve_in_process_prompt(kwargs):
    with pytest.raises(ValueError, match="greedy only"):
        _model(mtp_num_draft_tokens=2).process_prompt("Hi", ["text"], ["text"], **kwargs)
    # without MTP it samples
    out = _model().process_prompt("Hi", ["text"], ["text"], **kwargs)
    assert out.new_input_tensors["text_inputs"][0].numel() == 2


@pytest.mark.parametrize("kwargs", [{}, {"temperature": 0}, {"temperature": None}])
def test_mtp_serves_greedy(kwargs):
    m = _model(mtp_num_draft_tokens=2)
    assert m.process_prompt("Hi", ["text"], ["text"], **kwargs).new_input_tensors["text_inputs"][0].numel() == 2
    assert m.get_request_resource_configs({}, kwargs)[SAMPLER_RESOURCE].temperature == 0
