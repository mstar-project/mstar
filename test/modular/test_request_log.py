"""The request lifecycle lines log at DEBUG to their own logger, which
``MSTAR_LOG_REQUESTS=1`` turns on without the rest of DEBUG."""

from __future__ import annotations

import logging

import pytest

from mstar.utils.logging_config import REQUEST_LOGGER, configure_request_log


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def info_root():
    """The root logger at INFO with one handler, as ``basicConfig`` leaves it."""
    root, requests = logging.getLogger(), logging.getLogger(REQUEST_LOGGER)
    saved = root.handlers[:], root.level, requests.level
    lines = _Lines()
    root.handlers[:] = [lines]
    root.setLevel(logging.INFO)
    yield lines
    root.handlers[:], root.level = saved[0], saved[1]
    requests.setLevel(saved[2])


def _log_both():
    logging.getLogger(REQUEST_LOGGER).debug("Request r1 done")
    logging.getLogger("mstar.communication").debug("a transport message")


@pytest.mark.parametrize("value", [None, "0"])
def test_the_request_lines_stay_off_at_info(info_root, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("MSTAR_LOG_REQUESTS", raising=False)
    else:
        monkeypatch.setenv("MSTAR_LOG_REQUESTS", value)
    configure_request_log()
    _log_both()
    assert info_root.lines == []


def test_the_switch_turns_on_the_request_lines_alone(info_root, monkeypatch):
    monkeypatch.setenv("MSTAR_LOG_REQUESTS", "1")
    configure_request_log()
    _log_both()
    assert info_root.lines == ["Request r1 done"]
