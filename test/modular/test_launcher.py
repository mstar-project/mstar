"""Tests for the worker launchers and the node-agent handshake."""

import os
import pickle
import time

import pytest

from mstar.cluster.endpoints import ControlPlaneEndpoints
from mstar.cluster.launcher import LocalLauncher, NodeAgentLauncher
from mstar.cluster.spec import ClusterSpec
from mstar.communication.communicator import ZMQCommunicator
from mstar.utils.ipc_format import (
    AgentJoin,
    AgentMessage,
    AgentMessageType,
    AgentWorkerDied,
    LaunchSpec,
)


def _endpoints(n_hosts=3):
    base = 23000 + (os.getpid() % 400) * 8
    hosts = [
        {"addr": "127.0.0.1", "gpus": [i], "zmq_port_base": base + i * 250}
        for i in range(n_hosts)
    ]
    return ControlPlaneEndpoints(ClusterSpec.from_config({"cluster": {"hosts": hosts}}))


def _drain_until(comm, want, timeout_s=5.0):
    got = []
    deadline = time.monotonic() + timeout_s
    while len(got) < want and time.monotonic() < deadline:
        got.extend(comm.get_all_new_messages())
        time.sleep(0.01)
    return got


def _spec_for(node_rank, worker_ids):
    return LaunchSpec(
        node_rank=node_rank,
        model_name="dummy",
        model_kwargs={},
        cache_dir=None,
        log_level="INFO",
        host_env={"X": "1"},
        workers=[{"worker_id": wid, "device": "cuda:0"} for wid in worker_ids],
    )


class TestNodeAgentHandshake:
    def test_join_receives_launch_spec(self):
        eps = _endpoints()
        conductor = ZMQCommunicator("conductor", [], endpoints=eps)
        agent = ZMQCommunicator("node_agent_1", ["conductor"], endpoints=eps)
        try:
            launcher = NodeAgentLauncher(
                communicator=conductor,
                launch_specs={1: _spec_for(1, ["worker_1"])},
                join_timeout_s=30,
            )
            launcher.ensure_workers()
            agent.send("conductor", AgentMessage(
                AgentMessageType.AGENT_JOIN,
                AgentJoin(node_rank=1, addr="127.0.0.1", visible_gpus=1, pid=1),
            ))
            for msg in _drain_until(conductor, want=1):
                launcher.handle_agent_message(msg)
            (reply,) = _drain_until(agent, want=1)
            assert reply.message_type == AgentMessageType.LAUNCH_SPEC
            assert [w["worker_id"] for w in reply.body.workers] == ["worker_1"]
            assert reply.body.host_env == {"X": "1"}
            assert launcher.poll() == []
        finally:
            conductor.close()
            agent.close()

    def test_unexpected_rank_rejected(self):
        eps = _endpoints(n_hosts=3)
        conductor = ZMQCommunicator("conductor", [], endpoints=eps)
        stray = ZMQCommunicator("node_agent_2", ["conductor"], endpoints=eps)
        try:
            launcher = NodeAgentLauncher(
                communicator=conductor,
                launch_specs={1: _spec_for(1, ["worker_1"])},  # host 2 not expected
                join_timeout_s=30,
            )
            launcher.ensure_workers()
            stray.send("conductor", AgentMessage(
                AgentMessageType.AGENT_JOIN,
                AgentJoin(node_rank=2, addr="127.0.0.1", visible_gpus=1, pid=2),
            ))
            for msg in _drain_until(conductor, want=1):
                launcher.handle_agent_message(msg)
            (reply,) = _drain_until(stray, want=1)
            assert reply.message_type == AgentMessageType.AGENT_JOIN_REJECTED
            assert "not part of this deployment" in reply.body.reason
        finally:
            conductor.close()
            stray.close()

    def test_worker_death_relay_and_join_timeout(self):
        eps = _endpoints()
        conductor = ZMQCommunicator("conductor", [], endpoints=eps)
        try:
            launcher = NodeAgentLauncher(
                communicator=conductor,
                launch_specs={1: _spec_for(1, ["worker_1"])},
                join_timeout_s=0.05,
            )
            launcher.ensure_workers()
            launcher.handle_agent_message(AgentMessage(
                AgentMessageType.AGENT_WORKER_DIED,
                AgentWorkerDied(node_rank=1, worker_id="worker_1", exitcode=-9),
            ))
            time.sleep(0.1)
            with pytest.raises(RuntimeError, match="did not join"):
                launcher.poll()

            # After the expected agent joins, poll returns queued events.
            launcher._joined.add(1)
            events = launcher.poll()
            assert [(e.worker_id, e.exitcode) for e in events] == [("worker_1", -9)]
        finally:
            conductor.close()

    def test_launch_spec_pickles(self):
        spec = _spec_for(1, ["worker_1", "worker_2"])
        assert pickle.loads(pickle.dumps(spec)).workers[1]["worker_id"] == "worker_2"


class TestLocalLauncher:
    def test_lifecycle(self):
        durations = {"worker_0": 0.3, "worker_1": 60.0}
        launcher = LocalLauncher(
            worker_ids=["worker_0", "worker_1"],
            model=None,
            build_spawn_kwargs=lambda wid: {
                "worker_id": wid, "seconds": durations[wid]
            },
            target=_sleep_target_no_model,
        )
        launcher.ensure_workers()
        assert launcher.poll() == []

        deadline = time.monotonic() + 10
        events = []
        while not events and time.monotonic() < deadline:
            events = launcher.poll()
            time.sleep(0.05)
        assert [e.worker_id for e in events] == ["worker_0"]
        assert events[0].exitcode == 0

        launcher.shutdown()
        assert launcher.poll() == []  # already-reported deaths are not repeated


def _sleep_target_no_model(worker_id: str, seconds: float, model=None):
    time.sleep(seconds)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
