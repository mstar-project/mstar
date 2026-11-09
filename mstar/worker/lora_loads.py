"""Runtime LoRA adapter loads and unloads on a worker: a load reads in the background
and is acknowledged from the main loop.

A load reads, converts, shards, pins and digests a whole adapter file. On the main loop it
would stall scheduling for as long as that takes, so each pool reads on its own reader
thread and the main loop acknowledges the load once every pool has it.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mstar.engine.resources.lora.config import describe_load_error
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    LoadLoraAdapter,
    LoraAdapterAck,
    UnloadLoraAdapter,
)

if TYPE_CHECKING:
    from mstar.engine.resources.lora.pool import LoraAdapterPool
    from mstar.worker.services import WorkerServices

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoraLoad:
    """One load in flight: each pool's read, and what to answer once they are all done."""

    body: LoadLoraAdapter
    reads: list[tuple["LoraAdapterPool", Future]] = field(default_factory=list)
    error: str | None = None
    # later loads and unloads of the same name, applied in order after its ack: an
    # unload before the load lands would be a no-op, and the load would then serve a name
    # nothing else does; a load (a retry after the conductor gave up on this one) would
    # race this one's read and the rollback behind it
    behind: list[LoadLoraAdapter | UnloadLoraAdapter] = field(default_factory=list)


class LoraLoads:
    """The worker's LOAD_LORA_ADAPTER and UNLOAD_LORA_ADAPTER messages, and the loads still
    being read.

    Each is applied as the message arrives: a load or an unload touches no slot state an
    admit decides from (``LoraAdapterPool``'s docstring), so the ranks of a lockstep
    instance need not apply it at the same step. A load or an unload of a name still being
    read waits for that load's ack."""

    def __init__(self, services: "WorkerServices"):
        self._worker_id = services.worker_id
        self._engines = services.engine_manager
        self._communicator = services.communicator
        self._loads: list[LoraLoad] = []

    def __bool__(self) -> bool:
        return bool(self._loads)

    def load(self, body: LoadLoraAdapter) -> None:
        """Start reading a runtime-loaded adapter into this rank's host tier; it is
        acknowledged once every pool has it (``finish``). A pool that fails rolls back the
        ones that loaded, and the conductor rolls back the other workers."""
        if self._defer(body):
            return
        load = LoraLoad(body)
        digest = None if body.lora_digest is None else bytes.fromhex(body.lora_digest)
        # a pool that refuses outright (the name already served) fails the load without
        # the rest being asked
        for pool in self._engines.engine.lora_pools():
            try:
                load.reads.append((pool, pool.load_adapter_async(body.lora_name, body.lora_path, digest)))
            except Exception as exc:  # noqa: BLE001 - a bad adapter fails the load, not the worker
                logger.error("LoRA adapter %s failed to load: %s", body.lora_name, exc)
                load.error = describe_load_error(exc)
                break
        self._loads.append(load)
        self.finish()

    def unload(self, body: UnloadLoraAdapter) -> None:
        """Stop serving an adapter to new requests; running ones keep it until they end."""
        if self._defer(body):
            return
        error = None
        try:
            for pool in self._engines.engine.lora_pools():
                pool.unload_adapter(body.lora_name)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Worker %s: LoRA adapter %s failed to unload", self._worker_id, body.lora_name)
            error = describe_load_error(exc)
        self._ack(body.op_id, error)

    def finish(self) -> None:
        """Acknowledge the loads whose reads completed, then apply the loads and unloads
        that arrived behind them, in order."""
        for load in self._completed():
            self._ack(load.body.op_id, load.error)
            for body in load.behind:
                if isinstance(body, LoadLoraAdapter):
                    self.load(body)
                else:
                    self.unload(body)

    def _defer(self, body: LoadLoraAdapter | UnloadLoraAdapter) -> bool:
        """Hold ``body`` behind a load of the same name still being read; False if none is."""
        for load in self._loads:
            if load.body.lora_name == body.lora_name:
                load.behind.append(body)
                return True
        return False

    def _completed(self) -> list[LoraLoad]:
        """The loads whose reads have all completed, each with its error set if any pool
        failed; the pools that loaded it then are told to forget it, as the conductor
        tells the other workers."""
        done, waiting = [], []
        for load in self._loads:
            if all(future.done() for _, future in load.reads):
                done.append(load)
            else:
                waiting.append(load)
        self._loads = waiting
        for load in done:
            loaded = []
            for pool, future in load.reads:
                try:
                    future.result()
                    loaded.append(pool)
                except Exception as exc:  # noqa: BLE001
                    logger.error("LoRA adapter %s failed to load: %s", load.body.lora_name, exc)
                    load.error = load.error or describe_load_error(exc)
            if load.error is not None:
                for pool in loaded:
                    pool.unload_adapter(load.body.lora_name)
        return done

    def _ack(self, op_id: str, error: str | None) -> None:
        self._communicator.send("conductor", ConductorMessage(
            message_type=ConductorMessageType.LORA_ADAPTER_ACK,
            body=LoraAdapterAck(op_id=op_id, worker_id=self._worker_id, error=error),
        ))
