"""What GLM-5.3 takes of a request's kwargs: an unset one is the default, and
one it cannot read is refused in process_prompt, the data worker's 400, before
the conductor reads it and any rank runs the request."""

import pytest

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
