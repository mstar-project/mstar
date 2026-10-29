"""Runtime LoRA adapter loads on a worker: read in the background, acknowledged from the
main loop.

A load reads, converts, shards, pins and digests a whole adapter file. On the main loop it
would stall scheduling for as long as that takes, so each pool reads on its own reader
thread and the main loop acknowledges the load once every pool has it.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from mstar.engine.resources.lora.config import describe_load_error
from mstar.utils.ipc_format import LoadLoraAdapter, UnloadLoraAdapter

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource
    from mstar.engine.resources.lora.pool import LoraAdapterPool

logger = logging.getLogger(__name__)


def lora_pools(resources: Mapping[str, "Resource"]) -> list["LoraAdapterPool"]:
    """The LoRA adapter pools among an engine's resources."""
    from mstar.engine.resources.lora.pool import LoraAdapterPool

    return [resource for resource in resources.values() if isinstance(resource, LoraAdapterPool)]


@dataclass(slots=True)
class LoraLoad:
    """One load in flight: each pool's read, and what to answer once they are all done."""

    body: LoadLoraAdapter
    reads: list[tuple["LoraAdapterPool", Future]] = field(default_factory=list)
    error: str | None = None
    # unloads of the same name that arrived while it was read, applied after its ack: an
    # unload before the load lands would be a no-op, and the load would then serve a
    # name nothing else does
    unloads: list[UnloadLoraAdapter] = field(default_factory=list)


class LoraLoads:
    """The worker's runtime adapter loads in flight."""

    def __init__(self):
        self._loads: list[LoraLoad] = []

    def __bool__(self) -> bool:
        return bool(self._loads)

    def start(self, body: LoadLoraAdapter, pools: list["LoraAdapterPool"]) -> None:
        """Hand every pool the read and return at once. A pool that refuses outright (the
        name already served) fails the load without the rest being asked."""
        load = LoraLoad(body)
        digest = bytes.fromhex(body.lora_digest)
        for pool in pools:
            try:
                load.reads.append((pool, pool.load_adapter_async(body.lora_name, body.lora_path, digest)))
            except Exception as exc:  # noqa: BLE001 - a bad adapter fails the load, not the worker
                logger.error("LoRA adapter %s failed to load: %s", body.lora_name, exc)
                load.error = describe_load_error(exc)
                break
        self._loads.append(load)

    def defer_unload(self, body: UnloadLoraAdapter) -> bool:
        """Hold ``body`` behind a load of the same name still being read; False if none is."""
        for load in self._loads:
            if load.body.lora_name == body.lora_name:
                load.unloads.append(body)
                return True
        return False

    def finish(self) -> list[LoraLoad]:
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
