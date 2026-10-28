"""``Engine.add_request`` hands each submodule the ingested request's info, after
its resources took it, and leaves the request on none of them when one raises."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import Engine


class _Runner:
    def __init__(self, log):
        self.log = log

    def ingest_request(self, rid, overrides=None):
        self.log.append(("ingest", rid))

    def remove_request(self, rid):
        self.log.append(("remove", rid))


class _Submodule:
    def __init__(self, log, name, raises=False):
        self.log, self.name, self.raises = log, name, raises

    def on_request_added(self, request_id, request_info):
        self.log.append((self.name, request_id, request_info.random_seed))
        if self.raises:
            raise ValueError("no size in the request")

    def cleanup_request(self, request_id):
        self.log.append((f"{self.name}.cleanup", request_id))


def _engine(*submodules):
    engine = Engine.__new__(Engine)
    log = []
    engine._runner = _Runner(log)
    engine._resources = {}
    engine._loopback_steps, engine._resident_delta = {}, {}
    engine._submodules = {
        name: SimpleNamespace(submodule=_Submodule(log, name, raises)) for name, raises in submodules
    }
    return engine, log


def _info(seed=5):
    return CurrentForwardPassInfo(request_id="r", graph_walk="w", fwd_index=0, random_seed=seed, max_tokens=0)


def test_each_submodule_hears_of_the_request_after_its_resources():
    engine, log = _engine(("dit", False), ("vae", False))
    engine.add_request(3, None, _info())
    assert log == [("ingest", 3), ("dit", 3, 5), ("vae", 3, 5)]


def test_without_request_info_only_the_resources_take_it():
    engine, log = _engine(("dit", False))
    engine.add_request(3)
    assert log == [("ingest", 3)]


def test_a_raising_submodule_leaves_the_request_nowhere():
    engine, log = _engine(("dit", True), ("vae", False))
    with pytest.raises(ValueError):
        engine.add_request(3, None, _info())
    assert log == [("ingest", 3), ("dit", 3, 5), ("remove", 3), ("dit.cleanup", 3), ("vae.cleanup", 3)]
