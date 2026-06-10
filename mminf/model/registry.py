from mminf.model.bagel.bagel_model import BagelModel
from mminf.model.base import Model
from mminf.model.orpheus.orpheus_model import OrpheusModel
from mminf.model.pi05.pi05_model import Pi05Model
from mminf.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
from mminf.model.vjepa2.vjepa2_model import VJepa2ACModel, VJepa2Model

MODEL_REGISTRY: dict[str, type[Model]] = {
    "bagel": BagelModel,
    "orpheus": OrpheusModel,
    "pi05": Pi05Model,
    "qwen3_omni": Qwen3OmniModel,
    "vjepa2": VJepa2Model,
    "vjepa2_ac": VJepa2ACModel,
}

HF_MODELS: dict[str, dict] = {
    "bagel": {"model_path_hf": "ByteDance-Seed/BAGEL-7B-MoT"},
    "orpheus": {"model_path_hf": "canopylabs/orpheus-3b-0.1-ft"},
    # Pi0.5 PyTorch port published by lerobot — single safetensors blob
    # (~14 GB). mminf/model/pi05/weight_loader.py handles the lerobot->mminf
    # state-dict remap inside Pi05Model.get_submodule().
    "pi05": {"model_path_hf": "lerobot/pi05_base"},
    "qwen3_omni": {"model_path_hf": "Qwen/Qwen3-Omni-30B-A3B-Instruct"},
    # V-JEPA 2 standard (encoder + masked predictor).  Default is ViT-L @ 256
    # (~300M); the same class loads vitl/h/g at 256 or 384 by reading
    # config.json.
    "vjepa2": {"model_path_hf": "facebook/vjepa2-vitl-fpc64-256"},
    # V-JEPA 2-AC (encoder + action-conditioned predictor).  HF doesn't host
    # an AC checkpoint; weights come from the public S3 mirror
    # ``https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`` via
    # ``download_vjepa2_ac_upstream_pt`` — the ``model_path_hf`` string is
    # kept as a logical identifier but isn't resolved against HuggingFace.
    "vjepa2_ac": {"model_path_hf": "vjepa2-ac-vitg"},
}


def get_model_class(name: str) -> type[Model]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model name: {name!r}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
