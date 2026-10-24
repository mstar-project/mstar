"""Publication is skipped for a request whose every node declaring a
resource runs on this worker's lockstep instance: only another instance's
``admit_retrieve`` reads published state, so nothing would ever read it."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mstar.engine.resources import StepRunner
from mstar.worker.worker import Worker
from test.modular.test_resource_runner import _Stub


def _runner():
    kv = _Stub("kv", published="kv-info", final_published="kv-final")
    pos = _Stub("pos", published="pos-info")
    runner = StepRunner(
        {"kv": kv, "pos": pos},
        node_resources={"LLM": ["kv", "pos"], "enc": ["pos"]},
    )
    return runner, kv, pos


def test_a_skipped_key_is_neither_published_nor_snapshotted_for_that_rid():
    runner, kv, pos = _runner()
    runner.skip_publish(1, ["kv"])
    assert runner.publish([1, 2], node_name="LLM") == {
        1: {"pos": "pos-info"}, 2: {"kv": "kv-info", "pos": "pos-info"},
    }
    assert "publish:1" not in kv.calls and "publish:2" in kv.calls
    snaps = runner.publish_snapshot([1, 2], node_name="LLM", graph_walk="decode")
    assert {rid: set(s) for rid, s in snaps.items()} == {1: {"pos"}, 2: {"kv", "pos"}}
    assert runner.publish_after_stop([1, 2], node_name="LLM", graph_walk="decode") == {
        1: {}, 2: {"kv": "kv-final"},
    }
    # a batch skipping every publisher never reaches the resources
    runner.skip_publish(1, ["pos"])
    runner.skip_publish(2, ["kv", "pos"])
    kv.calls.clear()
    pos.calls.clear()
    assert runner.publish_snapshot([1, 2], node_name="LLM", graph_walk="decode") == {}
    assert runner.publish([1, 2], node_name="LLM") == {1: {}, 2: {}}
    assert kv.calls == [] and pos.calls == []


def test_a_key_never_published_leaves_every_step():
    runner, kv, _ = _runner()
    runner.never_publish(["kv"])
    assert runner.publisher_nodes == {"pos": frozenset({"LLM", "enc"})}
    assert runner.publish([1, 2], node_name="LLM") == {1: {"pos": "pos-info"}, 2: {"pos": "pos-info"}}
    assert runner.publish_after_stop([1], node_name="LLM", graph_walk="decode") == {1: {}}
    assert set(runner.publish_snapshot([1], node_name="LLM", graph_walk="decode")[1]) == {"pos"}
    assert not any(c.startswith("publish") for c in kv.calls)


def test_removal_purges_the_handle_so_a_recycled_one_publishes_again():
    runner, _, _ = _runner()
    runner.skip_publish(1, ["kv"])
    runner.remove_request(1)
    assert runner.publish([1], node_name="LLM") == {1: {"kv": "kv-info", "pos": "pos-info"}}


class _Engine:
    def __init__(self, publisher_nodes):
        self.publisher_nodes = publisher_nodes
        self.skipped = {}
        self.never = []

    def skip_publish(self, rid, keys):
        self.skipped[rid] = list(keys)

    def never_publish(self, keys):
        self.never = list(keys)


def _worker(worker_id, wg_nodes, publisher_nodes):
    worker = Worker.__new__(Worker)
    worker.worker_id = worker_id
    worker._all_wg_nodes = wg_nodes
    engine = _Engine(publisher_nodes)
    worker.engine_manager = type("EM", (), {"engine": engine})()
    return worker, engine


PUBLISHERS = {"kv": frozenset({"LLM"}), "vision_kv": frozenset({"vision_encoder"})}
# prefill and decode walks each have a worker graph holding the LLM
WG_NODES = {0: {"LLM", "vision_encoder"}, 1: {"LLM"}, 2: {"vision_encoder"}}


ME, PEER = ["worker_0"], ["worker_1"]
TP = ["worker_0", "worker_1"]


@pytest.mark.parametrize("worker_id, wg_to_workers, skipped", [
    ("worker_0", {0: ME, 1: ME, 2: ME}, ["kv", "vision_kv"]),
    # the decode walk's LLM runs elsewhere (e.g. disaggregated decode)
    ("worker_0", {0: ME, 1: PEER, 2: ME}, ["vision_kv"]),
    # one TP instance, whatever order its ranks are listed in
    ("worker_1", {0: TP, 1: list(reversed(TP)), 2: TP}, ["kv", "vision_kv"]),
])
def test_a_request_skips_what_only_its_own_instance_declares(worker_id, wg_to_workers, skipped):
    worker, engine = _worker(worker_id, WG_NODES, PUBLISHERS)
    worker._skip_unshared_publish(7, wg_to_workers)
    assert engine.skipped == {7: skipped}


@pytest.mark.parametrize("worker_id, instances, never", [
    ("worker_0", {0: [ME], 1: [ME], 2: [ME]}, ["kv", "vision_kv"]),
    # the vision encoder has a replica per rank: a request may run it on either
    ("worker_1", {0: [TP], 1: [TP], 2: [ME, PEER]}, ["kv"]),
])
def test_a_resource_with_one_instance_in_the_deployment_is_never_published(
    worker_id, instances, never,
):
    worker, engine = _worker(worker_id, WG_NODES, PUBLISHERS)
    worker._never_publish_unshared(instances)
    assert engine.never == never
