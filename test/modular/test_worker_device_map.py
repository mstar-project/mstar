"""Deployment ``rank_devices``: which device string each worker rank gets."""

from mstar.conductor.conductor import worker_device


def test_rank_is_the_device_by_default():
    assert worker_device("cuda", 0) == "cuda:0"
    assert worker_device("cuda", 3, {}) == "cuda:3"
    assert worker_device("xpu", 1) == "xpu:1"


def test_rank_devices_places_ranks_on_a_shared_device():
    mapping = {1: 0, 2: 0}
    assert [worker_device("cuda", rank, mapping) for rank in (0, 1, 2, 3)] == [
        "cuda:0", "cuda:0", "cuda:0", "cuda:3",
    ]


def test_cpu_deployments_ignore_the_mapping():
    assert worker_device("cpu", 1, {1: 0}) == "cpu"
