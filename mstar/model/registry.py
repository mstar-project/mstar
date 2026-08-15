import importlib
from collections.abc import Mapping

from mstar.model.base import Model

# name -> "module.path:ClassName". Kept as strings so importing the registry does
# NOT import every model module. Eager imports pulled in every model's optional
# heavy deps (e.g. qwen3_omni -> flashinfer), so serving ONE model required all of
# them installed — and a missing/broken dep for any model broke the whole server.
_MODEL_CLASS_PATHS: dict[str, str] = {
    "bagel": "mstar.model.bagel.bagel_model:BagelModel",
    "cosmos3": "mstar.model.cosmos3.cosmos3_model:Cosmos3Model",
    "cosmos3_droid": "mstar.model.cosmos3.cosmos3_model:Cosmos3Model",
    "cosmos3_super": "mstar.model.cosmos3.cosmos3_model:Cosmos3Model",
    "higgs_audio": "mstar.model.higgs_audio.higgs_audio_model:HiggsAudioModel",
    "lingbot": "mstar.model.lingbot.lingbot_model:LingBotModel",
    "orpheus": "mstar.model.orpheus.orpheus_model:OrpheusModel",
    "pi05": "mstar.model.pi05.pi05_model:Pi05Model",
    "qwen3_omni": "mstar.model.qwen3_omni.qwen3_omni_model:Qwen3OmniModel",
    "qwen3_tts": "mstar.model.qwen3_tts.qwen3_tts_model:Qwen3TTSModel",
    "vjepa2": "mstar.model.vjepa2.vjepa2_model:VJepa2Model",
    "vjepa2_ac": "mstar.model.vjepa2.vjepa2_model:VJepa2ACModel",
    "wan22": "mstar.model.wan22.wan22_model:Wan22Model",
    "whisper_large": "mstar.model.whisper.whisper_model:WhisperModel",
}


class _LazyModelRegistry(Mapping):
    """Read-only name -> Model-subclass mapping that imports each model module
    only on first item access. Membership (``in``) and key iteration never import,
    so listing available models stays dependency-free; ``MODEL_REGISTRY[name]``
    imports just that model (raising the real ImportError if its deps are absent).
    """

    def __init__(self, class_paths: dict[str, str]):
        self._class_paths = class_paths
        self._cache: dict[str, type[Model]] = {}

    def __getitem__(self, name: str) -> type[Model]:
        if name not in self._class_paths:
            raise KeyError(name)
        if name not in self._cache:
            module_path, class_name = self._class_paths[name].split(":")
            self._cache[name] = getattr(importlib.import_module(module_path), class_name)
        return self._cache[name]

    def __contains__(self, name: object) -> bool:
        return name in self._class_paths

    def __iter__(self):
        return iter(self._class_paths)

    def __len__(self) -> int:
        return len(self._class_paths)


MODEL_REGISTRY: Mapping[str, type[Model]] = _LazyModelRegistry(_MODEL_CLASS_PATHS)

HF_MODELS: dict[str, dict] = {
    "bagel": {"model_path_hf": "ByteDance-Seed/BAGEL-7B-MoT"},
    # NVIDIA Cosmos3-Nano generator (diffusers transformer/ + Wan VAE + UniPC).
    "cosmos3": {"model_path_hf": "nvidia/Cosmos3-Nano"},
    # Cosmos3-Nano-Policy-DROID — Nano-sized action-policy fine-tune for the
    # DROID robot platform (domain droid_lerobot, 10-dim raw actions). Same
    # class; the checkpoint's config disables the sound pathway (sound_gen
    # false, no sound_tokenizer/), so the model self-serves without audio.
    "cosmos3_droid": {"model_path_hf": "nvidia/Cosmos3-Nano-Policy-DROID"},
    # Cosmos3-Super (64B) — same architecture + class; dims (64 layers / 5120
    # hidden / 25600 intermediate) load from the checkpoint's config.json, so it
    # needs tensor parallelism (it does not fit on one GPU).
    "cosmos3_super": {"model_path_hf": "nvidia/Cosmos3-Super"},
    # Higgs-Audio v3 STT: Whisper-style audio tower + Qwen3-1.7B LLM.
    # (The v2 checkpoints are TTS/generation models, not ASR.)
    "higgs_audio": {"model_path_hf": "bosonai/higgs-audio-v3-stt"},
    # LingBot-Video dense 1.3B text-to-video (Qwen3-VL text encoder + flow DiT
    # with sparse MoE + Wan VAE decoder, FlowUniPC scheduler).
    "lingbot": {"model_path_hf": "robbyant/lingbot-video-dense-1.3b"},
    "orpheus": {"model_path_hf": "canopylabs/orpheus-3b-0.1-ft"},
    # Pi0.5 PyTorch port published by lerobot — single safetensors blob
    # (~14 GB). mstar/model/pi05/weight_loader.py handles the lerobot->mstar
    # state-dict remap inside Pi05Model.get_submodule().
    "pi05": {"model_path_hf": "lerobot/pi05_base"},
    "qwen3_omni": {"model_path_hf": "Qwen/Qwen3-Omni-30B-A3B-Instruct"},
    "qwen3_tts": {"model_path_hf": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"},
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
    # Wan2.2-TI2V-5B (dense video DiT + UMT5-XXL + Wan2.2-VAE).  TI2V-5B
    # only; the A14B MoE variants are a separate follow-up.
    "wan22": {"model_path_hf": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"},
    # Whisper works for any size; the registry key pins large-v3, the
    # standard ASR-benchmark checkpoint.
    "whisper_large": {"model_path_hf": "openai/whisper-large-v3"},
}


def get_model_class(name: str) -> type[Model]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model name: {name!r}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
