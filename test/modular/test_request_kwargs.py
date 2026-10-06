"""The ASR models check request kwargs once in the API process and again
where the conductor reads them, so a stray form field cannot reach a step."""
import pytest

from mstar.model.components.request_kwargs import checked_request_kwargs


def test_known_fields_are_coerced():
    out = checked_request_kwargs({
        "max_output_tokens": "32", "seed": 7, "temperature": "0.2", "top_p": 1,
        "ignore_eos": "false", "language": "en", "timestamps": "word", "extra": object(),
    })
    assert out["max_output_tokens"] == 32 and out["seed"] == 7
    assert out["temperature"] == 0.2 and out["top_p"] == 1.0
    assert out["ignore_eos"] is False and out["language"] == "en" and out["timestamps"] == "word"
    assert "extra" in out  # unknown fields pass through untouched


@pytest.mark.parametrize("bad", [
    {"max_output_tokens": "abc"}, {"max_output_tokens": 0}, {"max_output_tokens": True},
    {"temperature": "hot"}, {"temperature": -1}, {"temperature": float("nan")},
    {"top_p": 0}, {"top_p": 1.5}, {"ignore_eos": "maybe"}, {"language": 5},
    {"timestamps": "letters"},
])
def test_bad_values_name_the_field(bad):
    (key,) = bad
    with pytest.raises(ValueError, match=key):
        checked_request_kwargs(bad)


def test_none_and_empty_pass():
    assert checked_request_kwargs(None) == {}
    assert checked_request_kwargs({"temperature": None, "language": None}) == {"temperature": None, "language": None}
