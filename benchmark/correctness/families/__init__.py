"""Maps each config's model key to a family. Every model in configs/ needs an entry here or in
UNSUPPORTED (a test checks this)."""

from __future__ import annotations

from benchmark.correctness.families.asr_models import HiggsAsr, WhisperAsr
from benchmark.correctness.families.base import Family
from benchmark.correctness.families.duplex import NemotronDuplex
from benchmark.correctness.families.image import ImageFamily, NoReferenceImage
from benchmark.correctness.families.omni import OmniFamily, OmniText
from benchmark.correctness.families.robotics import Cosmos3DroidActions, Pi05Actions, Vjepa2AcLatents, Vjepa2Embedding
from benchmark.correctness.families.text import TextFamily
from benchmark.correctness.families.tts_models import ChatterboxTts, KokoroTts, OmniVoiceTts, OrpheusTts, Qwen3Tts
from benchmark.correctness.families.video import NoReferenceVideo, VideoFamily
from benchmark.correctness.families.world import WaypointWorld


def _text(hf_id: str) -> type[TextFamily]:
    return type(f"Text_{hf_id.rsplit('/', 1)[-1]}", (TextFamily,), {"hf_id": hf_id})


class Flux2KleinImage(ImageFamily):
    hf_id = "black-forest-labs/FLUX.2-klein-4B"
    steps = 4
    ref_extra = {"guidance_scale": 1.0}  # the server ignores guidance, CFG is off


class Flux2Klein9bImage(Flux2KleinImage):
    hf_id = "black-forest-labs/FLUX.2-klein-9B"


class ZImageTurboImage(ImageFamily):
    hf_id = "Tongyi-MAI/Z-Image-Turbo"
    steps = 8
    ref_extra = {"guidance_scale": 0.0}


class Wan22Video(VideoFamily):
    # Settings from test/wan22/t2v_request.sh, with an empty negative prompt on both sides.
    hf_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    height, width, num_frames, steps, fps = 480, 832, 33, 20, 24
    guidance = 5.0
    negative_prompt = ""


class Cosmos3Video(VideoFamily):
    # Cosmos3 draws its noise on the GPU, not the CPU.
    hf_id = "nvidia/Cosmos3-Nano"
    generator_device = "cuda"
    height, width, num_frames, steps, fps = 480, 832, 17, 35, 24
    guidance = 6.0


class Cosmos3EdgeVideo(Cosmos3Video):
    hf_id = "nvidia/Cosmos3-Edge"
    num_frames, steps = 33, 20


class Cosmos3SuperVideo(Cosmos3Video):
    hf_id = "nvidia/Cosmos3-Super"


class Cosmos3SuperT2I(NoReferenceImage):
    # Distilled 4-step, no reference in the repo or in diffusers.
    hf_id = "nvidia/Cosmos3-Super-Text2Image-4Step"
    steps = 4
    guidance = 1.0

    def reference_description(self) -> str:
        return "nothing in the repo or diffusers runs the distilled 4-step sampler"


class Cosmos3SuperI2V(NoReferenceVideo):
    hf_id = "nvidia/Cosmos3-Super-Image2Video-4Step"
    image_to_video = True
    height, width, num_frames, steps, fps = 480, 832, 17, 4, 24
    guidance = 1.0
    no_reference_reason = "nothing in the repo or diffusers runs the distilled 4-step sampler"


FAMILIES: dict[str, type[Family]] = {
    "qwen3_5_0.8b": _text("Qwen/Qwen3.5-0.8B"),
    "qwen3_5_2b": _text("Qwen/Qwen3.5-2B"),
    "qwen3_5_4b": _text("Qwen/Qwen3.5-4B"),
    "qwen3_5_9b": _text("Qwen/Qwen3.5-9B"),
    "qwen3_5_27b": _text("Qwen/Qwen3.5-27B"),
    "qwen3_8_27b": _text("Qwen/Qwen3.8-27B"),
    "whisper_large": WhisperAsr,
    "higgs_audio": HiggsAsr,
    "kokoro": KokoroTts,
    "orpheus": OrpheusTts,
    "chatterbox": ChatterboxTts,
    "chatterbox_turbo": ChatterboxTts,
    "chatterbox_multilingual": ChatterboxTts,
    "qwen3_tts": Qwen3Tts,
    "qwen3_tts_1p7b": Qwen3Tts,
    "qwen3_tts_voicedesign": Qwen3Tts,
    "qwen3_tts_base": Qwen3Tts,
    "omnivoice": OmniVoiceTts,
    "flux2_klein": Flux2KleinImage,
    "flux2_klein_9b": Flux2Klein9bImage,
    "z_image_turbo": ZImageTurboImage,
    "wan22": Wan22Video,
    "cosmos3": Cosmos3Video,
    "cosmos3_edge": Cosmos3EdgeVideo,
    "cosmos3_super": Cosmos3SuperVideo,
    "cosmos3_super_t2i_4step": Cosmos3SuperT2I,
    "cosmos3_super_i2v_4step": Cosmos3SuperI2V,
    "cosmos3_droid": Cosmos3DroidActions,
    "cosmos3_edge_droid": Cosmos3DroidActions,
    "qwen3_omni": OmniFamily,
    "bagel": NoReferenceImage,
    "pi05": Pi05Actions,
    "vjepa2": Vjepa2Embedding,
    "vjepa2_ac": Vjepa2AcLatents,
    "waypoint": WaypointWorld,
    "nemotron_duplex": NemotronDuplex,
}

UNSUPPORTED: dict[str, str] = {
    "test_text_session": "test-only config (session plumbing), no model output to verify",
}


def family_for_config(config_name: str, model_key: str, cfg: dict, opts: dict[str, str]) -> Family:
    if model_key in UNSUPPORTED:
        raise NotImplementedError(f"{config_name} ({model_key}): {UNSUPPORTED[model_key]}")
    if model_key not in FAMILIES:
        raise NotImplementedError(f"{config_name}: no correctness family for model {model_key!r}")
    if model_key == "qwen3_omni" and opts.get("set") == "text":
        return OmniText(model_key, cfg, opts)
    return FAMILIES[model_key](model_key, cfg, opts)
