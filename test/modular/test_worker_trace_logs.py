"""Join worker request intake logs to the API request trace."""

import logging
from types import SimpleNamespace

import pytest

from mstar.utils.logging_config import REQUEST_LOGGER, configure_request_log
from mstar.worker.input_ingest import InputIngest

PARENT = "00-0123456789abcdef0123456789abcdef-aaaaaaaaaaaaaaaa-01"


class _ReachedRuntimeError(Exception):
    pass


def _ingest_until_runtime(parent, *, include_field=True):
    ingest = object.__new__(InputIngest)
    ingest._worker_id = "w0"
    ingest._lifecycle = SimpleNamespace(takes_input=lambda wire_rid: True)

    def reached(**kwargs):
        raise _ReachedRuntimeError

    ingest._runtime = SimpleNamespace(add_request=reached)
    body = SimpleNamespace(
        request_id="client#1", partition_worker_graph_ids=[],
        worker_graph_to_workers={}, loop_caps={},
        request_info=SimpleNamespace(partition_name="default", graph_walk="prefill"),
    )
    if include_field:
        body.traceparent = parent
    with pytest.raises(_ReachedRuntimeError):
        ingest._add_new_request(body)


@pytest.mark.parametrize("parent,include_field,has_trace", [
    (PARENT, True, True),
    (None, True, False),
    ("malformed", True, False),
    (None, False, False),
])
def test_worker_intake_log_correlates_only_valid_context(
    caplog, parent, include_field, has_trace,
):
    with caplog.at_level(logging.DEBUG, logger=REQUEST_LOGGER):
        _ingest_until_runtime(parent, include_field=include_field)
    assert "Worker w0 received request client#1" in caplog.text
    assert ("trace_id=0123456789abcdef0123456789abcdef" in caplog.text) is has_trace


def test_worker_intake_skips_trace_parse_when_debug_is_off(monkeypatch, caplog):
    def must_not_parse(parent):
        raise AssertionError("trace parsing ran with debug logging off")

    monkeypatch.setattr("mstar.worker.input_ingest.trace_id_from_parent", must_not_parse)
    with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
        _ingest_until_runtime(PARENT)
    assert "Worker w0 received request" not in caplog.text


def test_request_log_env_enables_worker_trace_line(monkeypatch, caplog):
    monkeypatch.setenv("MSTAR_LOG_REQUESTS", "1")
    request_logger = logging.getLogger(REQUEST_LOGGER)
    old_level = request_logger.level
    try:
        with caplog.at_level(logging.DEBUG):
            request_logger.setLevel(logging.INFO)
            configure_request_log()
            assert request_logger.isEnabledFor(logging.DEBUG)
            _ingest_until_runtime(PARENT)
    finally:
        request_logger.setLevel(old_level)
    assert "Worker w0 received request client#1 trace_id=0123456789abcdef0123456789abcdef" in caplog.text
