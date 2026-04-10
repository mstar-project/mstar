from mminf.model.bagel.bagel_model import BagelModel
from mminf.model.base import Model
from mminf.model.dummy_model import DummyModel
from mminf.model.orpheus.orpheus_model import OrpheusModel
from mminf.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
from mminf.model.pi05.pi05_model import Pi05Model

MODEL_REGISTRY: dict[str, type[Model]] = {
    "dummy": DummyModel,
    "bagel": BagelModel,
    "orpheus": OrpheusModel,
    "qwen3_omni": Qwen3OmniModel,
    "pi05": Pi05Model,
}

HF_MODELS: dict[str, dict] = {
    "bagel": {"model_path_hf": "ByteDance-Seed/BAGEL-7B-MoT"},
    "orpheus": {"model_path_hf": "canopylabs/orpheus-3b-0.1-ft"},
    "qwen3_omni": {"model_path_hf": "Qwen/Qwen3-Omni-30B-A3B-Instruct"},
    "pi05": {"model_path_hf": "physical-intelligence/pi05_base"},
}


def get_model_class(name: str) -> type[Model]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model name: {name!r}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
