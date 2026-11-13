"""``StubResource`` and ``make_resource`` stand in for a real ``Resource``."""
import pytest

from mstar.engine.engine import SubmoduleManagement
from mstar.engine.resources.base import ADMIT_OK, Resource
from mstar.engine.resources.runner import StepRunner
from mstar.testing_utils.engine import make_mgmt
from mstar.testing_utils.resource import StubResource, make_resource


def test_a_stub_inherits_the_base_defaults():
    resource = make_resource()
    assert isinstance(resource, Resource)
    assert resource.admit(None, None) is ADMIT_OK
    assert not resource.supports_preplan and not resource.force_double_buffer and not resource.supports_eviction


def test_it_builds_itself_like_a_real_resource():
    assert isinstance(StubResource.build(None, None), StubResource)


def test_an_override_replaces_a_method_and_takes_the_call_arguments():
    assert make_resource(reclaimable=lambda rid: rid + 1).reclaimable(1) == 2


def test_an_override_replaces_a_property():
    assert make_resource(force_double_buffer=True).force_double_buffer is True
    assert make_mgmt(resources={"kv": make_resource(force_double_buffer=True)}).num_slots == 2


def test_a_name_resource_does_not_have_is_refused():
    with pytest.raises(TypeError, match="force_double_bufer"):
        make_resource(force_double_bufer=True)


def test_an_override_counts_as_the_class_overriding_the_method():
    """``StepRunner`` chooses who publishes by comparing the method on the class."""
    runner = StepRunner({"quiet": make_resource(), "loud": make_resource(publish=lambda rid: None)})
    assert runner._publish_order == ["loud"]


def test_a_stub_fits_a_mgmt_with_no_edits():
    assert isinstance(make_mgmt(resources={"kv": make_resource()}), SubmoduleManagement)
