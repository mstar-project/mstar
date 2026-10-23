"""Publication is skipped for a request whose every node declaring a
resource runs on this worker's lockstep instance: only another instance's
``admit_retrieve`` reads published state, so nothing would ever read it."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

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


def test_publisher_nodes_are_the_declaring_nodes():
    runner, _, _ = _runner()
    assert runner.publisher_nodes == {
        "kv": frozenset({"LLM"}), "pos": frozenset({"LLM", "enc"}),
    }


def test_a_skipped_key_is_neither_published_nor_snapshotted_for_that_rid():
    runner, kv, _ = _runner()
    runner.skip_publish(1, ["kv"])
    assert runner.publish([1, 2], node_name="LLM") == {
        1: {"pos": "pos-info"}, 2: {"kv": "kv-info", "pos": "pos-info"},
    }
    assert "publish:1" not in kv.calls and "publish:2" in kv.calls
    assert runner.publish_after_stop([1, 2], node_name="LLM", graph_walk="decode") == {
        1: {}, 2: {"kv": "kv-final"},
    }


def test_a_batch_skipping_every_publisher_never_reaches_the_resources():
    runner, kv, pos = _runner()
    for rid in (1, 2):
        runner.skip_publish(rid, ["kv", "pos"])
    kv.calls.clear()
    pos.calls.clear()
    assert runner.publish_snapshot([1, 2], node_name="LLM", graph_walk="decode") == {}
    assert runner.publish([1, 2], node_name="LLM") == {1: {}, 2: {}}
    assert kv.calls == [] and pos.calls == []


def test_removal_purges_the_handle_so_a_recycled_one_publishes_again():
    runner, _, _ = _runner()
    runner.skip_publish(1, ["kv"])
    runner.remove_request(1)
    assert runner.publish([1], node_name="LLM") == {1: {"kv": "kv-info", "pos": "pos-info"}}


class _Engine:
    def __init__(self, publisher_nodes):
        self.publisher_nodes = publisher_nodes
        self.skipped = {}

    def skip_publish(self, rid, keys):
        self.skipped[rid] = list(keys)


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


def test_every_declaring_node_on_this_instance_skips():
    worker, engine = _worker("worker_0", WG_NODES, PUBLISHERS)
    worker._skip_unshared_publish(7, {0: ["worker_0"], 1: ["worker_0"], 2: ["worker_0"]})
    assert engine.skipped == {7: ["kv", "vision_kv"]}


def test_a_declaring_node_on_another_instance_keeps_publishing():
    # the decode walk's LLM runs elsewhere (e.g. disaggregated decode)
    worker, engine = _worker("worker_0", WG_NODES, PUBLISHERS)
    worker._skip_unshared_publish(7, {0: ["worker_0"], 1: ["worker_1"], 2: ["worker_0"]})
    assert engine.skipped == {7: ["vision_kv"]}


def test_a_tp_instance_counts_as_one_owner():
    worker, engine = _worker("worker_1", WG_NODES, PUBLISHERS)
    both = ["worker_0", "worker_1"]
    worker._skip_unshared_publish(7, {0: both, 1: list(reversed(both)), 2: both})
    assert engine.skipped == {7: ["kv", "vision_kv"]}


def test_a_resource_no_worker_graph_declares_is_left_alone():
    worker, engine = _worker("worker_0", {0: {"other"}}, PUBLISHERS)
    worker._skip_unshared_publish(7, {0: ["worker_0"]})
    assert engine.skipped == {}
