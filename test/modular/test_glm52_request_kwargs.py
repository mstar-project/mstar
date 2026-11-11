"""What GLM-5.2 takes of a request's kwargs: an unset one is the default, and one it cannot
read is refused at the conductor's intake, the client's 400, before any rank runs the request."""

import json
from types import SimpleNamespace

import pytest

from mstar.conductor.conductor import Conductor
from mstar.engine.resources import SamplerSpec, SamplingReqConfig
from mstar.engine.resources.sampler.config import BOOL_KNOBS, INT_KNOB_RULES, REAL_KNOB_RULES
from mstar.model.glm52.config import SAMPLER_RESOURCE
from mstar.model.glm52.glm52_model import SAMPLING_KEYS, Glm52Model, generation_config_dir

# zai-org/GLM-5.2(-FP8)'s generation_config.json, as published
PUBLISHED = {
    "_from_model_config": True, "eos_token_id": [154820, 154827, 154829],
    "pad_token_id": 154820, "temperature": 1.0, "top_p": 0.95,
}


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
    # one worker that runs every node
    nodes = {name for walk in model.get_graph_walk_graphs().values() for name in walk.get_nodes()}
    conductor._per_worker_graphs = {"worker": [SimpleNamespace(section=SimpleNamespace(get_nodes=lambda: nodes))]}
    # in the order _admit_request runs them
    max_output_tokens = conductor._max_output_tokens(kwargs)
    conductor._loop_caps(kwargs)
    configs = conductor._get_resource_configs(["text"], ["text"], kwargs)
    for specs in conductor._resource_specs():
        for key, cfg in configs.items():
            cfg.validate(specs.get(key))
    return max_output_tokens


@pytest.mark.parametrize("kwargs", [
    {"top_k": None}, {"temperature": None, "top_p": None}, {"repetition_penalty": None},
    {"max_output_tokens": None}, {"seed": None},
])
def test_a_null_kwarg_is_unset(kwargs):
    m = _model()
    sampling = m.get_request_resource_configs([], [], kwargs)[SAMPLER_RESOURCE]
    default = m.get_request_resource_configs([], [], {})[SAMPLER_RESOURCE]
    assert sampling == default
    assert _intake(m, kwargs) == _intake(m, {}) == m.get_max_output_tokens()
    out = m.process_prompt("Hi", ["text"], ["text"], **kwargs)
    assert out.new_input_tensors["text_inputs"][0].tolist() == [72, 105]


@pytest.mark.parametrize("knob", [*REAL_KNOB_RULES, *INT_KNOB_RULES, *BOOL_KNOBS])
def test_a_sampler_knob_is_honored_or_refused(knob):
    """A knob the API accepts reaches the sampler, or intake refuses it: never dropped."""
    value = (not getattr(SamplingReqConfig(), knob) if knob in BOOL_KNOBS
             else 5 if knob in INT_KNOB_RULES else 0.5)
    m = _model()
    try:
        _intake(m, {knob: value})
    except ValueError:
        return
    assert getattr(m.get_request_resource_configs([], [], {knob: value})[SAMPLER_RESOURCE], knob) == value


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


@pytest.mark.parametrize("kwargs", [{"temperature": "warm"}, {"top_k": "20.5"}, {"max_output_tokens": "lots"}])
def test_strings_that_are_not_the_knobs_type_are_refused(kwargs):
    # one meaning per knob on every model (mstar/utils/generation_kwargs.py)
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        _intake(_model(), kwargs)


def test_return_logprobs_is_refused_where_the_sampler_never_sees_it():
    # GLM-5.2 builds its sampler config from its own keys, which leave it out
    with pytest.raises(ValueError, match="does not report logprobs"):
        _intake(_model(), {"return_logprobs": True})
    _intake(_model(), {"return_logprobs": False})


def _spec(m: Glm52Model) -> SamplerSpec:
    [spec] = [s for s in m.get_node_resources() if isinstance(s, SamplerSpec)]
    return spec


def test_min_p_is_refused_at_intake():
    m = _model()
    # the checkpoint defaults no min_p, so the sampler is built without the filter
    assert not _spec(m).enable_min_p
    with pytest.raises(ValueError, match="min_p"):
        _intake(m, {"min_p": 0.05})
    _intake(m, {"min_p": 0.0})


def test_request_kwargs_are_the_keys_it_reads():
    # TextChatAdapter maps temperature, top_p and max_tokens onto these; seed is the conductor's
    assert _model().request_kwargs() == {
        "max_output_tokens", "temperature", "top_p", "top_k", "min_p",
        "repetition_penalty", "penalize_prompt", "ignore_eos",
    }


def test_without_a_generation_config_the_published_recipe_applies():
    cfg = _model().get_request_resource_configs([], [], {})[SAMPLER_RESOURCE]
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.min_p, cfg.repetition_penalty) == (1.0, 0.95, 0, 0, 1)
    # built without __init__, as other tests do
    bare = object.__new__(Glm52Model)
    bare.config = _model().config
    assert bare.get_request_resource_configs([], [], {})[SAMPLER_RESOURCE] == cfg


def test_the_checkpoint_generation_config_sets_the_defaults(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(PUBLISHED))
    assert _model(checkpoint_path=str(tmp_path)).sampling_defaults == {"temperature": 1.0, "top_p": 0.95}
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {**PUBLISHED, "top_k": 40, "min_p": 0.05, "repetition_penalty": 1.1, "max_new_tokens": 9}))
    m = _model(checkpoint_path=str(tmp_path))
    cfg = m.get_request_resource_configs([], [], {})[SAMPLER_RESOURCE]
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.min_p, cfg.repetition_penalty) == (1.0, 0.95, 40, 0.05, 1.1)
    # a checkpoint min_p turns the filter on, so its own default is not refused
    assert _spec(m).enable_min_p
    cfg.validate(_spec(m))
    # the length is the window's, not the file's
    assert m.get_max_output_tokens() == m.context_limit()
    (tmp_path / "generation_config.json").write_text(json.dumps({**PUBLISHED, "do_sample": False}))
    assert _model(checkpoint_path=str(tmp_path)).sampling_defaults["temperature"] == 0.0


def test_a_hub_id_reads_the_cached_file_and_never_fetches(tmp_path):
    snapshot = tmp_path / "models--zai-org--GLM-5.2-FP8" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "generation_config.json").write_text(json.dumps({**PUBLISHED, "top_p": 0.8}))
    (tmp_path / "models--zai-org--GLM-5.2-FP8" / "refs").mkdir()
    (tmp_path / "models--zai-org--GLM-5.2-FP8" / "refs" / "main").write_text("abc")
    m = Glm52Model("zai-org/GLM-5.2-FP8", cache_dir=str(tmp_path), tokenizer_mode="byte")
    assert m.sampling_defaults == {"temperature": 1.0, "top_p": 0.8}
    # not cached: the published recipe, offline
    assert generation_config_dir("zai-org/not-cached", str(tmp_path)) is None


def test_every_knob_reaches_the_sampler():
    sent = {"temperature": 0.3, "top_p": 0.5, "top_k": 7, "min_p": 0.1,
            "repetition_penalty": 1.2, "penalize_prompt": False, "ignore_eos": True}
    assert set(sent) == set(SAMPLING_KEYS)
    cfg = _model().get_request_resource_configs([], [], sent)[SAMPLER_RESOURCE]
    assert {k: getattr(cfg, k) for k in sent} == sent


@pytest.mark.parametrize("kwargs", [
    {"penalize_prompt": "no"}, {"ignore_eos": 1}, {"top_k": 2.5}, {"min_p": "lots"}, {"max_output_tokens": True},
])
def test_a_mistyped_knob_is_refused_in_process_prompt(kwargs):
    # in the data worker, before the request reaches the conductor
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        _model().process_prompt("Hi", ["text"], ["text"], **kwargs)


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
    assert m.get_request_resource_configs([], [], kwargs)[SAMPLER_RESOURCE].temperature == 0
