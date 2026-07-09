"""Per-host worker supervisor for multi-host deployments.

One ``mstar-node`` process runs on every non-head host:

    mstar-node --config <same config as the head> --node-rank <k>

It joins the conductor (whose address it derives from the config's
``cluster:`` section), receives a launch spec, spawns this host's worker
processes, and reports worker deaths back. The model object is constructed
locally from the registry so weights load from this host's own cache rather
than crossing the network.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import signal
import time

import yaml

from mstar.cluster.endpoints import ControlPlaneEndpoints
from mstar.cluster.spec import ClusterSpec
from mstar.communication.communicator import ZMQCommunicator
from mstar.utils.ipc_format import (
    AgentJoin,
    AgentMessage,
    AgentMessageType,
    AgentWorkerDied,
    LaunchSpec,
)
from mstar.utils.logging_config import quiet_noisy_loggers

logger = logging.getLogger(__name__)


class NodeAgent:
    def __init__(self, config_path: str, node_rank: int):
        self.config_path = config_path
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.cluster_spec = ClusterSpec.from_config(config)
        if not self.cluster_spec.is_multi_host():
            raise ValueError(
                "mstar-node only applies to multi-host deployments; this config "
                "has no (multi-host) cluster: section"
            )
        if not 0 < node_rank < len(self.cluster_spec.hosts):
            raise ValueError(
                f"--node-rank must name a non-head host (1..{len(self.cluster_spec.hosts) - 1}); "
                f"host 0 is launched by mstar-serve itself"
            )
        self.node_rank = node_rank
        self.my_id = f"node_agent_{node_rank}"
        self.host = self.cluster_spec.hosts[node_rank]
        self.communicator = ZMQCommunicator(
            my_id=self.my_id,
            push_ids=["conductor"],
            endpoints=ControlPlaneEndpoints(self.cluster_spec),
        )
        self._procs: dict[str, mp.Process] = {}
        self._reported_dead: set[str] = set()
        self._launched = False
        self._running = True
        self._exit_code = 0

    # ------------------------------------------------------------------

    def run(self) -> int:
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        import torch

        visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        expected = len(self.host.gpus)
        if visible_gpus < expected:
            logger.warning(
                "Host declares %d gpu(s) in the cluster spec but only %d are "
                "visible here", expected, visible_gpus,
            )
        self.communicator.send(
            "conductor",
            AgentMessage(
                AgentMessageType.AGENT_JOIN,
                AgentJoin(
                    node_rank=self.node_rank,
                    addr=self.host.addr,
                    visible_gpus=visible_gpus,
                    pid=os.getpid(),
                ),
            ),
        )
        logger.info(
            "Joined as %s (%s); waiting for launch spec", self.my_id, self.host.addr
        )

        while self._running:
            for message in self.communicator.get_all_new_messages():
                if not isinstance(message, AgentMessage):
                    logger.warning("Unexpected message type: %s", type(message))
                    continue
                if message.message_type == AgentMessageType.LAUNCH_SPEC:
                    self._launch(message.body)
                elif message.message_type == AgentMessageType.AGENT_SHUTDOWN:
                    logger.info("Conductor requested shutdown")
                    self._running = False
                elif message.message_type == AgentMessageType.AGENT_JOIN_REJECTED:
                    logger.error("Join rejected: %s", message.body.reason)
                    self._running = False
                    self._exit_code = 1
                else:
                    logger.warning("Unhandled agent message: %s", message.message_type)
            self._check_children()
            time.sleep(0.05)

        self._terminate_children()
        return self._exit_code

    def _on_signal(self, signum, frame):
        logger.info("Received signal %d; shutting down", signum)
        self._running = False

    # ------------------------------------------------------------------

    def _launch(self, spec: LaunchSpec) -> None:
        if self._launched:
            logger.warning("Launch spec already applied; ignoring duplicate")
            return
        self._launched = True

        for key, value in spec.host_env.items():
            os.environ[key] = value

        from mstar.model.registry import HF_MODELS, get_model_class

        model = get_model_class(spec.model_name)(
            model_path_hf=HF_MODELS.get(spec.model_name, {}).get("model_path_hf", ""),
            cache_dir=spec.cache_dir,
            **spec.model_kwargs,
        )
        # Models may specialize themselves from the deployment config (e.g.
        # BAGEL only registers its CFG-parallel walks when the placement names
        # those nodes). The conductor's model instance goes through these
        # calls before its local workers spawn; make the same calls here so
        # agent-spawned workers build identical graphs.
        model.get_worker_graphs(self.config_path)
        model.get_sharding_config(self.config_path)

        from mstar.conductor.conductor import _worker_process_target

        ctx = mp.get_context("spawn")
        for kwargs in spec.workers:
            kwargs = dict(kwargs)
            kwargs["model"] = model
            p = ctx.Process(target=_worker_process_target, kwargs=kwargs, daemon=False)
            p.start()
            self._procs[kwargs["worker_id"]] = p
            logger.info(
                "Spawned %s on device %s (pid %d)",
                kwargs["worker_id"], kwargs.get("device"), p.pid,
            )

    def _check_children(self) -> None:
        for worker_id, p in self._procs.items():
            if worker_id not in self._reported_dead and not p.is_alive():
                self._reported_dead.add(worker_id)
                logger.error("%s exited with code %s", worker_id, p.exitcode)
                self.communicator.send(
                    "conductor",
                    AgentMessage(
                        AgentMessageType.AGENT_WORKER_DIED,
                        AgentWorkerDied(
                            node_rank=self.node_rank,
                            worker_id=worker_id,
                            exitcode=p.exitcode,
                        ),
                    ),
                )

    def _terminate_children(self) -> None:
        for p in self._procs.values():
            if p.is_alive():
                p.terminate()
        for p in self._procs.values():
            p.join(timeout=5)
        for p in self._procs.values():
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        self._procs.clear()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mstar-node",
        description="mstar per-host worker agent for multi-host deployments",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the deployment's YAML config (same file the head uses)",
    )
    parser.add_argument("--node-rank", required=True, type=int,
                        help="Index of this host in the config's cluster.hosts list")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f"%(asctime)s %(levelname)s [node_agent_{args.node_rank}] %(name)s: %(message)s",
    )
    quiet_noisy_loggers()

    agent = NodeAgent(config_path=args.config, node_rank=args.node_rank)
    raise SystemExit(agent.run())


if __name__ == "__main__":
    main()
