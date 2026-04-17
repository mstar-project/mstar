from mminf.model.bagel.bagel_model import BagelModel
from mminf.model.base import Model
from mminf.model.dummy_model import DummyModel
from mminf.model.orpheus.orpheus_model import OrpheusModel
from mminf.model.pi05.pi05_model import Pi05Model
from mminf.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

MODEL_REGISTRY: dict[str, type[Model]] = {
    "dummy": DummyModel,
    "bagel": BagelModel,
    "orpheus": OrpheusModel,
    "pi05": Pi05Model,
    "qwen3_omni": Qwen3OmniModel,
}

HF_MODELS: dict[str, dict] = {
    "bagel": {"model_path_hf": "ByteDance-Seed/BAGEL-7B-MoT"},
    "orpheus": {"model_path_hf": "canopylabs/orpheus-3b-0.1-ft"},
    # Pi0.5 PyTorch port published by lerobot — single safetensors blob
    # (~14 GB). mminf/model/pi05/weight_loader.py handles the lerobot->mminf
    # state-dict remap inside Pi05Model.get_submodule().
    "pi05": {"model_path_hf": "lerobot/pi05_base"},
    "qwen3_omni": {"model_path_hf": "Qwen/Qwen3-Omni-30B-A3B-Instruct"},
}


def get_model_class(name: str) -> type[Model]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model name: {name!r}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
