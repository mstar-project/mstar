"""make_test_api_server must build the same fields as APIServer.__init__.

Tests build their APIServer with it instead of running __init__ (which starts
the preprocess worker and opens sockets). A field __init__ sets but the helper
misses would surface as an AttributeError in whichever test reaches it first,
so this compares the two directly.
"""
from types import SimpleNamespace

import pytest
from api_server_helpers import make_test_api_server

from mstar.api_server import entrypoint


@pytest.fixture
def real_server(monkeypatch, tmp_path):
    monkeypatch.setattr(entrypoint, "PreprocessWorker", lambda **kwargs: SimpleNamespace(output_ready=None))
    monkeypatch.setattr(
        entrypoint, "make_communicator", lambda **kwargs: SimpleNamespace(register_event_for_poll=lambda event: None),
    )
    return entrypoint.APIServer(upload_dir=str(tmp_path / "uploads"))


def test_helper_sets_every_field_init_sets(real_server):
    helper = make_test_api_server()
    real_fields, helper_fields = set(vars(real_server)), set(vars(helper))
    assert real_fields - helper_fields == set(), "make_test_api_server misses these"
    assert helper_fields - real_fields == set(), "APIServer.__init__ no longer sets these"


def test_field_types_match_init(real_server):
    helper = make_test_api_server()
    for name in vars(real_server):
        if name in {"preprocess_worker", "communicator", "_msg_thread", "upload_dir"}:
            continue
        real, made = getattr(real_server, name), getattr(helper, name)
        assert type(made) is type(real), name


def test_each_server_gets_fresh_request_state():
    a, b = make_test_api_server(), make_test_api_server()
    assert a.pending_requests is not b.pending_requests
    assert a.request_lock is not b.request_lock
    assert next(a._generation) == next(b._generation) == 1


def test_overrides_replace_fields_and_reject_unknown_names():
    server = make_test_api_server(timeout_seconds=1.0, _liveness_interval_s=0.0)
    assert server.timeout_seconds == 1.0
    assert server._liveness_interval_s == 0.0
    with pytest.raises(TypeError, match="no field 'timeout'"):
        make_test_api_server(timeout=1.0)
