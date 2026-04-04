import logging
from dataclasses import dataclass, field

import torch

from mminf.engine.ar_engine import AREngine
from mminf.engine.audio_codec_engine import AudioCodecEngine
from mminf.engine.base import BaseEngine
from mminf.engine.enc_dec_engine import EncoderDecoderEngine
from mminf.engine.flow_engine import FlowEngine
from mminf.engine.kv_store import PagedAllocationManager, TransferEngineInfo
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
    def from_config(
        cls,
        engine_configs: list[dict],
        device: torch.device,
        transfer_engine_info: TransferEngineInfo,
        enable_nvtx: bool = False,
        model: Model | None = None,
    ) -> "EngineManager":
        """
        Build an EngineManager from a list of engine configs.

        The Model object (if provided) supplies nn.Module submodules for
        each node via model.get_submodule(node_name). In dummy mode
        (model=None or get_submodule returns None), engines run without
        real computation.

        engine_configs example:
        [
            {"engine_type": "ar", "node_names": ["LLM"], "model_config": {...}},
            {"engine_type": "flow", "node_names": ["flow"], "model_config": {...}},
            {"engine_type": "enc_dec", "node_names": ["text_emb", "image_emb", "VAE_dec"], ...}
        ]
        """
        node_to_engine: dict[str, BaseEngine] = {}

        for cfg in engine_configs:
            engine_type_str = cfg["engine_type"]
            if "node_names" not in cfg:
                raise KeyError(
                    f"Engine config missing `node_names`: {cfg}"
                )
            node_names = cfg["node_names"]
            model_config = cfg.get("model_config", {})

            engine_cls = ENGINE_TYPE_TO_CLASS[engine_type_str]

            engine = engine_cls(
                kv_cache_config=model_config.get("kv_cache", {}),
                autocast_dtype=model_config.get("autocast_dtype", torch.bfloat16),
                enable_nvtx=enable_nvtx,
            )

            # Extract submodules from the Model for this engine's nodes
            submodules: dict[str, torch.nn.Module] = {}
            if model is not None:
                for name in node_names:
                    submodule = model.get_submodule(name, device)
                    if submodule is not None:
                        submodules[name] = submodule.to(
                            device=device,
                            dtype=model.get_autocast_dtype()
                        )

            engine.load_model(
                submodules, model_config, device,
                transfer_engine_info=transfer_engine_info
            )
            logger.info("Engine %s loaded in on device %s", cfg["engine_type"], str(device))

            for name in node_names:
                node_to_engine[name] = engine
        logger.info("All engines loaded on device %s", str(device))

        return cls(node_to_engine=node_to_engine)

    def warmup_all(self) -> None:
        """Call warmup() on all unique engines for CUDA graph capture."""
        seen = set()
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

    def get_ar_alloc_manager(self) -> PagedAllocationManager | None:
        """Return the PagedAllocationManager from the first AR engine, if any."""
        for engine in self.node_to_engine.values():
            if isinstance(engine, AREngine) and engine.alloc_manager is not None:
                return engine.alloc_manager
        return None

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
