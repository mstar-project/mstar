"""Every metric family at ``/metrics`` has an entry in docs/metrics.rst.

The families are read from the API server's real registry, built the way
``main()`` builds it (``APIServer.enable_metrics``), so a family added to any
component shows up here without a list to keep in step.
"""

import re
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("prometheus_client")

from api_server_helpers import make_test_api_server  # noqa: E402

from mstar.api_server.data_worker import PreprocessWorker  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# ``mstar_name{model_name, ...}`` on its own line, then the type as the first
# word of the indented description.
HEADING = re.compile(r"^``mstar_\w+\{.*$", re.M)
ENTRY = re.compile(r"^``(mstar_\w+)\{([^}]*)\}``\n(?:[ \t]*\n)*[ \t]+(Counter|Gauge|Histogram)\b", re.M)


def _documented() -> dict[str, tuple[str, list[str]]]:
    """``{name: (type, labels)}`` for each family entry in docs/metrics.rst."""
    text = (ROOT / "docs/metrics.rst").read_text()
    entries = list(ENTRY.finditer(text))
    unparsed = sorted(set(HEADING.findall(text)) - {m[0].split("\n")[0] for m in entries})
    assert not unparsed, f"family entries without a type on their first description line: {unparsed}"
    return {
        m[1]: (m[3].lower(), [label.strip() for label in m[2].split(",")])
        for m in entries
    }


def _registered() -> dict[str, str]:
    """``{name: type}`` for every family the API server exports, counters
    under their exposed ``_total`` name."""
    # The data worker thread is not started; enable_metrics only wires its
    # accessors into the registry.
    server = make_test_api_server(
        preprocess_worker=mock.create_autospec(PreprocessWorker, instance=True))
    server.enable_metrics()
    return {
        f.name + "_total" if f.type == "counter" else f.name: f.type
        for f in server.metrics.registry.collect()
    }


def test_every_registered_family_is_documented():
    documented = _documented()
    registered = _registered()
    missing = sorted(set(registered) - set(documented))
    assert not missing, f"add an entry to docs/metrics.rst: {missing}"
    wrong_type = {n: (documented[n][0], t) for n, t in registered.items() if documented[n][0] != t}
    assert not wrong_type, f"documented vs registered type: {wrong_type}"


def test_every_documented_family_has_model_name_first():
    for name, (_, labels) in _documented().items():
        assert labels[0] == "model_name", name


def test_every_documented_family_is_registered():
    stale = sorted(set(_documented()) - set(_registered()))
    assert not stale, f"docs/metrics.rst documents families /metrics does not export: {stale}"
