import logging
from dataclasses import dataclass, field
from typing import Callable

import torch

from mminf.engine.base import BaseEngine
from mminf.engine.kv_cache_engine import KVCacheEngine
from mminf.engine.kv_store import KVCacheConfig, TransferEngineInfo
from mminf.engine.stateless_engine import (
    StatelessEngine,
    StatelessEngineConfig,
    make_audio_codec_config,
    make_enc_dec_config,
    make_flow_config,
)
from mminf.model.base import Model


def _make_kv_cache(autocast_dtype: torch.dtype, enable_nvtx: bool) -> BaseEngine:
    return KVCacheEngine(autocast_dtype=autocast_dtype, enable_nvtx=enable_nvtx)


def _make_stateless_factory(
    config_factory: Callable[[torch.dtype | None], StatelessEngineConfig],
) -> Callable[[torch.dtype | None, bool], BaseEngine]:
    def factory(autocast_dtype: torch.dtype | None, enable_nvtx: bool) -> BaseEngine:
        return StatelessEngine(
            config=config_factory(autocast_dtype),
            enable_nvtx=enable_nvtx,
        )

    return factory


# Each engine type string maps to a factory that takes (autocast_dtype,
# enable_nvtx) and returns a freshly constructed engine. The split between
# AR (stateful, paged KV cache) and the stateless flavors lives here.
ENGINE_TYPE_FACTORIES: dict[str, Callable[[torch.dtype | None, bool], BaseEngine]] = {
    "kv_cache": _make_kv_cache,
    "flow": _make_stateless_factory(make_flow_config),
    "enc_dec": _make_stateless_factory(make_enc_dec_config),
    "audio_codec": _make_stateless_factory(make_audio_codec_config),
}

logger = logging.getLogger(__name__)


@dataclass
class EngineManager:
    """Maps node names to engine instances."""
    node_to_engine: dict[str, BaseEngine] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        node_names: set[str],
        device: torch.device,
        kv_config: list[KVCacheConfig],
        model_config: dict,
        transfer_engine_info: TransferEngineInfo,
        model: Model,
        enable_nvtx: bool = False,
    ) -> "EngineManager":
        """
        Build an EngineManager from a list of engine configs.

        The Model object (if provided) supplies nn.Module submodules for
        each node via model.get_submodule(node_name). In dummy mode
        (model=None or get_submodule returns None), engines run without
        real computation.
        """
        type_to_nodes = {}
        node_to_engine_type = model.get_node_engine_types()
        for node_name in node_names:
            type_to_nodes.setdefault(node_to_engine_type[node_name].value, []).append(node_name)

        node_to_engine: dict[str, BaseEngine] = {}

        # Resolve autocast dtype: explicit YAML config wins; otherwise we
        # fall back to the Model's own preference (so models that need to
        # match a reference numerically can override get_autocast_dtype
        # without forcing every config file to set the same value).
        autocast_dtype = model.get_autocast_dtype()
        if "autocast_dtype" in model_config:
            autocast_dtype = model_config["autocast_dtype"]

        for engine_type_str, engine_node_names in type_to_nodes.items():
            factory = ENGINE_TYPE_FACTORIES[engine_type_str]
            engine = factory(autocast_dtype, enable_nvtx)

            # Extract submodules from the Model for this engine's nodes
            submodules: dict[str, torch.nn.Module] = {}
            if model is not None:
                for name in engine_node_names:
                    submodule = model.get_submodule(name, device)
                    if submodule is not None:
                        if engine.has_autocast():
                            submodules[name] = submodule.to(
                                device=device,
                                dtype=autocast_dtype
                            )
                        else:
                            submodules[name] = submodule.to(
                                device=device,
                            )

            engine.load_model(
                submodules,
                kv_cache_config=kv_config,
                device=device,
                transfer_engine_info=transfer_engine_info,
                kv_cache_type=autocast_dtype
            )
            logger.info("Engine %s loaded in on device %s", engine_type_str, str(device))

            for name in engine_node_names:
                node_to_engine[name] = engine
        logger.info("All engines loaded on device %s", str(device))

        return cls(node_to_engine=node_to_engine)

    def warmup_all(self) -> None:
        """Call warmup() on all unique engines for CUDA graph capture."""
        seen = set()
        with torch.no_grad():
            for engine in self.node_to_engine.values():
                eid = id(engine)
                if eid not in seen:
                    seen.add(eid)
                    engine.warmup()

    def get_engine(self, node_name: str) -> BaseEngine:
        return self.node_to_engine[node_name]

    def add_request(self, request_id: str) -> None:
        """Propagate add_request to all unique engines."""
        seen = set()
        for engine in self.node_to_engine.values():
            eid = id(engine)
            if eid not in seen:
                seen.add(eid)
                engine.add_request(request_id)

    def remove_request(self, request_id: str) -> None:
        """Propagate remove_request to all unique engines."""
        seen = set()
        for engine in self.node_to_engine.values():
            eid = id(engine)
            if eid not in seen:
                seen.add(eid)
                engine.remove_request(request_id)

    def set_alloc_write_policies(self, policy):
        for engine in self._unique_engines():
            engine.set_alloc_write_policy(policy)

    def lru_tracked_nodes(self) -> list[str]:
        """Aggregate ``engine.lru_tracked_nodes()`` across unique engines.
        The worker uses this to seed / clean up the per-request LRU
        timestamps it needs for offload-victim selection.
        """
        out: list[str] = []
        for engine in self._unique_engines():
            out.extend(engine.lru_tracked_nodes())
        return out

    def _unique_engines(self) -> list[BaseEngine]:
        seen = set()
        result = []
        for engine in self.node_to_engine.values():
            eid = id(engine)
            if eid not in seen:
                seen.add(eid)
                result.append(engine)
        return result

    def shutdown(self) -> None:
        for engine in self._unique_engines():
            engine.shutdown()
