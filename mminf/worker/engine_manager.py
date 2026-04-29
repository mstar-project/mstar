import logging
from dataclasses import dataclass, field

import torch

from mminf.engine.ar_engine import AREngine
from mminf.engine.audio_codec_engine import AudioCodecEngine
from mminf.engine.base import BaseEngine
from mminf.engine.enc_dec_engine import EncoderDecoderEngine
from mminf.engine.flow_engine import FlowEngine
from mminf.engine.kv_store import KVCacheConfig, TransferEngineInfo
from mminf.model.base import Model

ENGINE_TYPE_TO_CLASS: dict[str, type[BaseEngine]] = {
    "ar": AREngine,
    "flow": FlowEngine,
    "enc_dec": EncoderDecoderEngine,
    "audio_codec": AudioCodecEngine,
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
            engine_cls = ENGINE_TYPE_TO_CLASS[engine_type_str]

            engine = engine_cls(
                autocast_dtype=autocast_dtype,
                enable_nvtx=enable_nvtx,
            )

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
        for engine in self.node_to_engine.values():
            if isinstance(engine, AREngine):
                for submod_mgmt in engine.submodule_management.values():
                    submod_mgmt.kv_management.alloc_manager.write_policy = policy

    def get_ar_engine(self) -> "AREngine | None":
        """Return the first AR engine instance, if any."""
        for engine in self.node_to_engine.values():
            if isinstance(engine, AREngine):
                return engine
        return None

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
