"""A client-pinned ``request_id`` is at most 128 characters from
``[A-Za-z0-9._:-]``: ``APIServer.submit_request`` refuses anything else,
while every id the server mints still fits. mstar-server's own checks on
``/generate`` and ``/generate/ws`` are tested in test/rust/."""

from __future__ import annotations

import uuid

import pytest

from mstar.api_server.entrypoint import REQUEST_ID_ERROR, APIServer, check_request_id

LONGEST = "a" * 128
BAD_IDS = ["a" * 129, "caf\u00e9", "a b", "a/b", "a#1", "a\n"]


def rid(prefix: str) -> str:
    """An id as mstar-server mints one (``serving::rid``)."""
    return f"{prefix}-{uuid.uuid4().hex}"


@pytest.mark.parametrize("request_id", [
    "", "warmup-0", "obs.1:a_b-C", LONGEST, str(uuid.uuid4()), rid("img"),
    f"{rid('speech')}-3", f"tensor-{uuid.uuid4()}", rid(f"rt-{uuid.uuid4().hex[:16]}"),
], ids=[
    "empty", "warmup", "punctuation", "longest", "uuid", "minted", "minted-suffix",
    "tensor-uuid", "minted-realtime",
])
def test_check_accepts_client_and_minted_ids(request_id):
    check_request_id(request_id)


@pytest.mark.parametrize("request_id", BAD_IDS + [7])
def test_check_refuses(request_id):
    with pytest.raises(ValueError, match=r"^request_id must be at most 128 characters"):
        check_request_id(request_id)


def test_submit_request_refuses_directly():
    with pytest.raises(ValueError, match=r"^request_id must be at most 128 characters"):
        APIServer.__new__(APIServer).submit_request(
            text="hi", input_modalities=["text"], output_modalities=["text"], request_id="a b",
        )


def test_error_text_matches_the_rust_frontend_test():
    assert REQUEST_ID_ERROR == "request_id must be at most 128 characters from [A-Za-z0-9._:-]"
