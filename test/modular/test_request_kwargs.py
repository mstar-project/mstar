"""The ASR models check request kwargs on the data worker and again where the
conductor reads them, so a stray form field cannot reach a step. The API has
already normalized the generation knobs, so the check coerces nothing."""
import pytest

from mstar.model.components.request_kwargs import checked_request_kwargs
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
