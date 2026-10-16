"""What GLM-5.3 takes of a request's kwargs: an unset one is the default, and
one it cannot read is refused in process_prompt, the data worker's 400, before
the conductor reads it and any rank runs the request."""

import pytest

from mstar.engine.resources import SamplerSpec
from mstar.model.glm5_next.config import SAMPLER
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel


def _model(**kwargs) -> Glm5NextModel:
    # byte tokenizer: no HF IO
    return Glm5NextModel("x", tokenizer_mode="byte", **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"top_k": None}, {"temperature": None, "top_p": None}, {"repetition_penalty": None},
    {"max_output_tokens": None}, {"seed": None},
])
def test_a_null_kwarg_is_unset(kwargs):
    m = _model()
    sampling = m.get_request_resource_configs({}, kwargs)[SAMPLER]
    assert sampling == m.get_request_resource_configs({}, {})[SAMPLER]
    assert m.get_max_output_tokens(**kwargs) == m.get_max_output_tokens()
    assert m.process_prompt("Hi", ["text"], ["text"], **kwargs)["text_inputs"][0].tolist() == [72, 105]


def test_a_null_temperature_under_mtp_is_greedy():
    m = _model(mtp_num_draft_tokens=3)
    assert m.process_prompt("Hi", ["text"], ["text"], temperature=None)["text_inputs"][0].numel() == 2
    assert m.get_request_resource_configs({}, {"temperature": None})[SAMPLER].temperature == 0


@pytest.mark.parametrize("kwargs", [
    {"top_k": "many"}, {"temperature": "hot"}, {"top_p": [0.9]}, {"repetition_penalty": float("nan")},
    {"max_output_tokens": "100x"}, {"max_output_tokens": 1.5}, {"seed": "abc"},
])
def test_a_kwarg_it_cannot_read_is_refused_in_process_prompt(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        _model().process_prompt("Hi", ["text"], ["text"], **kwargs)


@pytest.mark.parametrize("k", [0, 3])
def test_min_p_is_refused_in_process_prompt(k):
    m = _model(mtp_num_draft_tokens=k)
    # the sampler that would read it is built without it; it was dropped (0.2 ran as 0)
    [spec] = [s for s in m.get_node_resources() if isinstance(s, SamplerSpec)]
    assert not spec.enable_min_p
    with pytest.raises(ValueError, match="min_p"):
        m.process_prompt("Hi", ["text"], ["text"], min_p=0.2)
    assert m.process_prompt("Hi", ["text"], ["text"], min_p=0)["text_inputs"][0].numel() == 2


def test_a_request_without_max_tokens_runs_to_the_window():
    # a thinking reply was cut at a fixed 1024 tokens with "length", on any window
    m = _model()
    assert m.get_max_output_tokens() == m.get_max_output_tokens(max_output_tokens=None) == m.config.index_topk
    assert m.get_max_output_tokens(max_output_tokens=100) == 100


def test_a_long_context_request_without_max_tokens_runs_to_its_window():
    m = _model(dsa_long_context=True, max_seq_len=1 << 20)
    assert m.get_max_output_tokens() == m.get_max_output_tokens(max_output_tokens=None) == 1 << 20
    with pytest.raises(ValueError, match="max_output_tokens"):
        m.process_prompt("Hi", ["text"], ["text"], max_output_tokens="lots")
