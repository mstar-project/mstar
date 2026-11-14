"""A piecewise region's eager run answers its calls for its own step, nested in its node's step.

A node's step opens its handles around its forward, and a region inside that forward declares, plans and commits a
step of its own over its own rows. Its calls have to be checked against that step, as they were at capture, not against
the node's: two regions of one forward each add to and write the same layers, which the node's step would refuse as a
second add or write. The node's step answers again once the region ends, whatever the region raised.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mstar.engine.resources.step import StepHandle, SubmoduleStep


class _Recording(StepHandle):
    def __init__(self, name, log):
        self.name, self.log = name, log

    def begin_step(self, step):
        assert self.open_step is None, "a step began while another is open"
        self.open_step = step
        self.log.append(("begin", self.name, step.tag))

    def check_step(self, step):
        self.log.append(("check", self.name, step.tag))

    def end_step(self, step):
        self.open_step = None
        self.log.append(("end", self.name, step.tag))

    def commit_step(self, step):
        self.log.append(("commit", self.name, step.tag))


def _step(tag, *handles) -> SubmoduleStep:
    step = SubmoduleStep()
    object.__setattr__(step, "tag", tag)
    step.set_state_handles(handles)
    return step


def test_a_region_answers_for_its_own_step_then_hands_back_to_the_nodes():
    log = []
    shared, own = _Recording("shared", log), _Recording("own", log)
    node = _step("node", shared)
    node.begin_calls()
    region = _step("region", shared, own)
    with region.nested_calls():
        assert shared.open_step is region and own.open_step is region
    assert shared.open_step is node and own.open_step is None
    assert log == [
        ("begin", "shared", "node"),
        ("end", "shared", "node"), ("begin", "shared", "region"), ("begin", "own", "region"),
        ("check", "shared", "region"), ("check", "own", "region"),
        ("end", "shared", "region"), ("end", "own", "region"),
        ("begin", "shared", "node"),
    ]


def test_a_region_that_raises_skips_its_check_and_still_hands_back():
    log = []
    shared = _Recording("shared", log)
    node = _step("node", shared)
    node.begin_calls()
    with pytest.raises(RuntimeError, match="boom"), _step("region", shared).nested_calls():
        raise RuntimeError("boom")
    assert shared.open_step is node
    assert ("check", "shared", "region") not in log


def test_two_regions_of_one_forward_each_open_afresh():
    log = []
    shared = _Recording("shared", log)
    _step("node", shared).begin_calls()
    for tag in ("first", "second"):
        with _step(tag, shared).nested_calls():
            pass
    checks = [entry for entry in log if entry[0] == "check"]
    assert checks == [("check", "shared", "first"), ("check", "shared", "second")]


def test_a_region_outside_any_node_step_opens_as_a_capture_does():
    log = []
    handle = _Recording("h", log)
    with _step("region", handle).nested_calls():
        assert handle.open_step is not None
    assert handle.open_step is None
    assert log == [("begin", "h", "region"), ("check", "h", "region"), ("end", "h", "region")]
