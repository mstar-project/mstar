"""Worker process launchers.

The conductor does not care where worker processes come from; it hands each
launcher the specs it is responsible for and consumes a stream of lifecycle
events. Two implementations exist: ``LocalLauncher`` spawns processes on the
conductor's own host (the historical single-host path), and
``NodeAgentLauncher`` hands launch specs to ``mstar-node`` agents that joined
from other hosts and relays the worker-death reports they send back.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from mstar.utils.ipc_format import (
    AgentJoin,
    AgentJoinRejected,
    AgentMessage,
    AgentMessageType,
    AgentShutdown,
    AgentWorkerDied,
    LaunchSpec,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerEvent:
    worker_id: str
    exitcode: int | None


class Launcher(ABC):
    @abstractmethod
    def ensure_workers(self) -> None:
        """Start (or arrange the starting of) this launcher's workers."""

    @abstractmethod
    def poll(self) -> list[WorkerEvent]:
        """Return newly observed worker deaths. May raise on startup failure."""

    @abstractmethod
    def shutdown(self) -> None:
        """Stop this launcher's workers. Safe to call more than once."""


class LocalLauncher(Launcher):
    """Spawns one process per worker on this host via the spawn context."""

    def __init__(
        self,
        worker_ids: list[str],
        model,
        build_spawn_kwargs: Callable[[str], dict],
        target: Callable | None = None,
    ):
        if target is None:
            from mstar.conductor.conductor import _worker_process_target
            target = _worker_process_target
        self._worker_ids = worker_ids
        self._model = model
        self._build_spawn_kwargs = build_spawn_kwargs
        self._target = target
        self._procs: dict[str, mp.Process] = {}
        self._reported: set[str] = set()

    def ensure_workers(self) -> None:
        ctx = mp.get_context("spawn")
        for worker_id in self._worker_ids:
            kwargs = dict(self._build_spawn_kwargs(worker_id))
            kwargs["model"] = self._model
            p = ctx.Process(target=self._target, kwargs=kwargs, daemon=False)
            p.start()
            self._procs[worker_id] = p

    def poll(self) -> list[WorkerEvent]:
        events = []
        for worker_id, p in self._procs.items():
            if worker_id not in self._reported and not p.is_alive():
                self._reported.add(worker_id)
                events.append(WorkerEvent(worker_id=worker_id, exitcode=p.exitcode))
        return events

    def shutdown(self) -> None:
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


class NodeAgentLauncher(Launcher):
    """Ships launch specs to remote ``mstar-node`` agents as they join.

    The conductor owns the message pump; it forwards ``AgentMessage`` traffic
    to :meth:`handle_agent_message`. ``poll`` surfaces relayed worker deaths
    and enforces the join deadline while any expected agent is missing.
    """

    def __init__(
        self,
        communicator,
        launch_specs: dict[int, LaunchSpec],
        join_timeout_s: float,
    ):
        self._communicator = communicator
        self._launch_specs = launch_specs
        self._expected = set(launch_specs)
        self._joined: set[int] = set()
        self._join_timeout_s = join_timeout_s
        self._deadline: float | None = None
        self._events: list[WorkerEvent] = []

    def ensure_workers(self) -> None:
        self._deadline = time.monotonic() + self._join_timeout_s
        logger.info(
            "Waiting for node agent(s) %s to join (timeout %.0fs)",
            sorted(self._expected), self._join_timeout_s,
        )

    def handle_agent_message(self, message: AgentMessage) -> None:
        body = message.body
        if message.message_type == AgentMessageType.AGENT_JOIN:
            self._handle_join(body)
        elif message.message_type == AgentMessageType.AGENT_WORKER_DIED:
            logger.error(
                "Node agent %d reports worker %s died (exit %s)",
                body.node_rank, body.worker_id, body.exitcode,
            )
            self._events.append(
                WorkerEvent(worker_id=body.worker_id, exitcode=body.exitcode)
            )
        else:
            logger.warning("Unexpected agent message: %s", message.message_type)

    def _handle_join(self, body: AgentJoin) -> None:
        k = body.node_rank
        if k not in self._expected:
            logger.error("Rejecting join from unexpected node rank %d", k)
            try:
                self._communicator.send(
                    f"node_agent_{k}",
                    AgentMessage(
                        AgentMessageType.AGENT_JOIN_REJECTED,
                        AgentJoinRejected(
                            f"node rank {k} is not part of this deployment "
                            f"(expected {sorted(self._expected)})"
                        ),
                    ),
                )
            except Exception:
                # A rank outside the cluster spec has no addressable endpoint;
                # dropping the join is all we can do.
                logger.exception("Could not notify rejected node agent %d", k)
            return
        if k in self._joined:
            logger.warning("Duplicate join from node agent %d; resending spec", k)
        else:
            logger.info(
                "Node agent %d joined from %s (%d visible gpu(s), pid %d)",
                k, body.addr, body.visible_gpus, body.pid,
            )
        self._joined.add(k)
        self._communicator.send(
            f"node_agent_{k}",
            AgentMessage(AgentMessageType.LAUNCH_SPEC, self._launch_specs[k]),
        )

    def poll(self) -> list[WorkerEvent]:
        missing = self._expected - self._joined
        if missing and self._deadline is not None and time.monotonic() > self._deadline:
            raise RuntimeError(
                f"node agent(s) {sorted(missing)} did not join within "
                f"{self._join_timeout_s:.0f}s — start `mstar-node --config <cfg> "
                f"--node-rank <k>` on those hosts"
            )
        events, self._events = self._events, []
        return events

    def shutdown(self) -> None:
        for k in sorted(self._joined):
            try:
                self._communicator.send(
                    f"node_agent_{k}",
                    AgentMessage(AgentMessageType.AGENT_SHUTDOWN, AgentShutdown()),
                )
            except Exception:
                logger.exception("Failed to send shutdown to node agent %d", k)
        self._joined.clear()


__all__ = [
    "Launcher",
    "LocalLauncher",
    "NodeAgentLauncher",
    "WorkerEvent",
    "AgentWorkerDied",
]
