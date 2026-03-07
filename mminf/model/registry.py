from mminf.model.bagel.bagel_model import BagelModel
from mminf.model.base import Model
from mminf.model.dummy_model import DummyModel

MODEL_REGISTRY: dict[str, type[Model]] = {
    "dummy": DummyModel,
    "bagel": BagelModel,
}

HF_MODELS: dict[str, dict] = {
    "bagel": {
        "model_path_hf": "ByteDance-Seed/BAGEL-7B-MoT"
    }
}


def get_model_class(name: str) -> type[Model]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model name: {name!r}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
