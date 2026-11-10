"""GLM-5.3's request kwargs: a null one is unset, a mistyped one is refused in
process_prompt (a 400) before any rank sees the request, and min_p by the sampler at admission."""

import json

import pytest

from mstar.engine.resources import SamplerSpec, SamplingReqConfig
from mstar.engine.resources.sampler.config import BOOL_KNOBS, INT_KNOB_RULES, REAL_KNOB_RULES
from mstar.model.glm5_next.config import SAMPLER
from mstar.model.glm5_next.glm5_next_model import SAMPLING_KEYS, Glm5NextModel, generation_config_dir

# zai-org/GLM-5.3-Flash's generation_config.json, as published
PUBLISHED = {
    "_from_model_config": True, "eos_token_id": [154820, 154827, 154829],
    "pad_token_id": 154820, "temperature": 1.0, "top_p": 0.95,
}


def _model(**kwargs) -> Glm5NextModel:
    # byte tokenizer: no HF IO
    return Glm5NextModel("x", tokenizer_mode="byte", **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"top_k": None}, {"temperature": None, "top_p": None}, {"repetition_penalty": None},
    {"max_output_tokens": None}, {"seed": None},
])
def test_a_null_kwarg_is_unset(kwargs):
    m = _model()
    sampling = m.get_request_resource_configs([], [], kwargs)[SAMPLER]
    assert sampling == m.get_request_resource_configs([], [], {})[SAMPLER]
    assert m.get_max_output_tokens(**kwargs) == m.get_max_output_tokens()
    out = m.process_prompt("Hi", ["text"], ["text"], **kwargs)
    assert out.new_input_tensors["text_inputs"][0].tolist() == [72, 105]


def test_a_null_temperature_under_mtp_is_greedy():
    m = _model(mtp_num_draft_tokens=3)
    assert m.process_prompt("Hi", ["text"], ["text"], temperature=None).new_input_tensors["text_inputs"][0].numel() == 2
    assert m.get_request_resource_configs([], [], {"temperature": None})[SAMPLER].temperature == 0


@pytest.mark.parametrize("knob", [*REAL_KNOB_RULES, *INT_KNOB_RULES, *BOOL_KNOBS])
def test_a_sampler_knob_is_honored_or_refused(knob):
    """A knob the API accepts reaches the sampler, or process_prompt refuses it: never dropped."""
    value = (not getattr(SamplingReqConfig(), knob) if knob in BOOL_KNOBS
             else 5 if knob in INT_KNOB_RULES else 0.5)
    m = _model()
    try:
        m.process_prompt("Hi", ["text"], ["text"], **{knob: value})
    except ValueError:
        return
    assert getattr(m.get_request_resource_configs([], [], {knob: value})[SAMPLER], knob) == value


@pytest.mark.parametrize("kwargs", [
    {"top_k": "many"}, {"temperature": "hot"}, {"top_p": [0.9]}, {"repetition_penalty": float("nan")},
    {"max_output_tokens": "100x"}, {"max_output_tokens": 1.5}, {"seed": "abc"},
])
def test_a_kwarg_it_cannot_read_is_refused_in_process_prompt(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        _model().process_prompt("Hi", ["text"], ["text"], **kwargs)


def _spec(m: Glm5NextModel) -> SamplerSpec:
    [spec] = [s for s in m.get_node_resources() if isinstance(s, SamplerSpec)]
    return spec


@pytest.mark.parametrize("k", [0, 3])
def test_min_p_is_forwarded_and_refused_at_admission(k):
    m = _model(mtp_num_draft_tokens=k)
    # the checkpoint defaults no min_p, so the sampler is built without the filter
    assert not _spec(m).enable_min_p
    cfg = m.get_request_resource_configs([], [], {"min_p": 0.2})[SAMPLER]
    assert cfg.min_p == 0.2
    with pytest.raises(ValueError, match="min_p"):
        cfg.validate(_spec(m))
    m.get_request_resource_configs([], [], {"min_p": 0})[SAMPLER].validate(_spec(m))


def test_request_kwargs_are_the_keys_it_reads():
    # TextChatAdapter maps temperature, top_p and max_tokens onto these; seed is the conductor's
    assert _model().request_kwargs() == {
        "max_output_tokens", "temperature", "top_p", "top_k", "min_p",
        "repetition_penalty", "penalize_prompt", "ignore_eos",
    }


def test_without_a_generation_config_the_published_recipe_applies():
    cfg = _model().get_request_resource_configs([], [], {})[SAMPLER]
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.min_p, cfg.repetition_penalty) == (1.0, 0.95, 0, 0, 1)
    # built without __init__, as other tests do
    bare = object.__new__(Glm5NextModel)
    bare.config = _model().config
    assert bare.get_request_resource_configs([], [], {})[SAMPLER] == cfg


def test_the_checkpoint_generation_config_sets_the_defaults(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(PUBLISHED))
    assert _model(checkpoint_path=str(tmp_path)).sampling_defaults == {"temperature": 1.0, "top_p": 0.95}
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {**PUBLISHED, "top_k": 40, "min_p": 0.05, "repetition_penalty": 1.1}))
    m = _model(checkpoint_path=str(tmp_path))
    cfg = m.get_request_resource_configs([], [], {})[SAMPLER]
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.min_p, cfg.repetition_penalty) == (1.0, 0.95, 40, 0.05, 1.1)
    # a checkpoint min_p turns the filter on, so its own default is not refused
    assert _spec(m).enable_min_p
    cfg.validate(_spec(m))
    (tmp_path / "generation_config.json").write_text(json.dumps({**PUBLISHED, "do_sample": False}))
    assert _model(checkpoint_path=str(tmp_path)).sampling_defaults["temperature"] == 0.0


def test_a_hub_id_reads_the_cached_file_and_never_fetches(tmp_path):
    repo = tmp_path / "models--zai-org--GLM-5.3-Flash"
    (repo / "snapshots" / "abc").mkdir(parents=True)
    (repo / "snapshots" / "abc" / "generation_config.json").write_text(json.dumps({**PUBLISHED, "top_p": 0.8}))
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc")
    m = Glm5NextModel("zai-org/GLM-5.3-Flash", cache_dir=str(tmp_path), tokenizer_mode="byte")
    assert m.sampling_defaults == {"temperature": 1.0, "top_p": 0.8}
    # not cached: the published recipe, offline
    assert generation_config_dir("zai-org/not-cached", str(tmp_path)) is None


def test_every_knob_reaches_the_sampler():
    sent = {"temperature": 0.3, "top_p": 0.5, "top_k": 7, "min_p": 0.1,
            "repetition_penalty": 1.2, "penalize_prompt": False, "ignore_eos": True}
    assert set(sent) == set(SAMPLING_KEYS)
    cfg = _model().get_request_resource_configs([], [], sent)[SAMPLER]
    assert {k: getattr(cfg, k) for k in sent} == sent


@pytest.mark.parametrize("kwargs", [
    {"penalize_prompt": "no"}, {"ignore_eos": 1}, {"top_k": 2.5}, {"min_p": "lots"}, {"max_output_tokens": True},
])
def test_a_mistyped_knob_is_refused_in_process_prompt(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        _model().process_prompt("Hi", ["text"], ["text"], **kwargs)


@pytest.mark.parametrize("kwargs", [{"temperature": 0.7}, {"repetition_penalty": 1.1}])
def test_mtp_refuses_sampling_it_cannot_serve(kwargs):
    with pytest.raises(ValueError, match="greedy only"):
        _model(mtp_num_draft_tokens=2).process_prompt("Hi", ["text"], ["text"], **kwargs)
    assert _model().process_prompt("Hi", ["text"], ["text"], **kwargs).new_input_tensors["text_inputs"][0].numel() == 2


def test_a_request_without_max_tokens_runs_to_the_window():
    m = _model()
    assert m.get_max_output_tokens() == m.get_max_output_tokens(max_output_tokens=None) == m.config.index_topk
    assert m.get_max_output_tokens(max_output_tokens=100) == 100
    # past the window: returned as asked, for the conductor to refuse against the loop
    assert m.get_max_output_tokens(max_output_tokens=10_000) == 10_000
