import pytest

from mstar.api_server.positronic import _reset_requested


@pytest.mark.parametrize(
    ("obs", "expected"),
    [
        ({}, False),
        ({"reset_context": None}, False),
        ({"reset_context": False}, False),
        ({"reset_context": True}, True),
        ({"reset_context": 0}, False),
        ({"reset_context": 1}, True),
        ({"reset_context": -1}, True),
        ({"reset_context": ""}, False),
        ({"reset_context": "false"}, True),
        ({"reset_context": []}, False),
        ({"reset_context": [None]}, True),
        ({"reset_context": 0, "__reset_context__": 1}, True),
    ],
)
def test_positronic_reset_matches_reference_truthiness(obs, expected):
    assert _reset_requested(obs) is expected
