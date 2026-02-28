from dataclasses import dataclass, field

import torch

from mminf.engine.base import BaseEngine, EngineType
from mminf.engine.ar_engine import AREngine
from mminf.engine.flow_engine import FlowEngine
from mminf.engine.enc_dec_engine import EncoderDecoderEngine
from mminf.engine.audio_codec_engine import AudioCodecEngine


ENGINE_TYPE_TO_CLASS: dict[str, type[BaseEngine]] = {
    "ar": AREngine,
    "flow": FlowEngine,
    "enc_dec": EncoderDecoderEngine,
    "audio_codec": AudioCodecEngine,
}


@dataclass
class EngineManager:
    """Maps stage names to engine instances."""
    stage_to_engine: dict[str, BaseEngine] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls, engine_configs: list[dict], device: torch.device
    ) -> "EngineManager":
        """
        Build an EngineManager from a list of engine configs.

        engine_configs example:
        [
            {"engine_type": "ar", "stage_names": ["LLM"], "model_config": {...}},
            {"engine_type": "flow", "stage_names": ["flow"], "model_config": {...}},
            {"engine_type": "enc_dec", "stage_names": ["text_emb", "image_emb", "VAE_dec"], ...}
        ]
        """
        stage_to_engine: dict[str, BaseEngine] = {}

        for cfg in engine_configs:
            engine_type_str = cfg["engine_type"]
            stage_names = cfg["stage_names"]
            model_config = cfg.get("model_config", {})
            model = cfg.get("model", None)

            engine_cls = ENGINE_TYPE_TO_CLASS[engine_type_str]

            if engine_type_str == "ar":
                engine = engine_cls(
                    model=model,
                    kv_cache_config=model_config.get("kv_cache", {}),
                )
            else:
                engine = engine_cls(model=model)

            engine.load_model(model_config, device)

            for name in stage_names:
                stage_to_engine[name] = engine

        return cls(stage_to_engine=stage_to_engine)

    def get_engine(self, stage_name: str) -> BaseEngine:
        return self.stage_to_engine[stage_name]

    def add_request(self, request_id: str) -> None:
        """Propagate add_request to all unique engines."""
        seen = set()
        for engine in self.stage_to_engine.values():
            eid = id(engine)
            if eid not in seen:
                seen.add(eid)
                engine.add_request(request_id)

    def remove_request(self, request_id: str) -> None:
        """Propagate remove_request to all unique engines."""
        seen = set()
        for engine in self.stage_to_engine.values():
            eid = id(engine)
            if eid not in seen:
                seen.add(eid)
                engine.remove_request(request_id)

    def shutdown(self) -> None:
        seen = set()
        for engine in self.stage_to_engine.values():
            eid = id(engine)
            if eid not in seen:
                seen.add(eid)
                engine.shutdown()
