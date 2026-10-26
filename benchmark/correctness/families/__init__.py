"""Maps each config's model key to a family. Every model in configs/ needs an entry here or in
UNSUPPORTED (a test checks this)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from benchmark.correctness.families.asr_models import HiggsAsr, WhisperAsr
from benchmark.correctness.families.av import AudioVideoFamily
from benchmark.correctness.families.base import Family
from benchmark.correctness.families.duplex import NemotronDuplex
from benchmark.correctness.families.image import ImageFamily, NoReferenceImage
from benchmark.correctness.families.omni import OmniFamily, OmniText
from benchmark.correctness.families.robotics import (
    Cosmos3DroidActions,
    Pi05Actions,
    Pi05DroidActions,
    Vjepa2AcLatents,
    Vjepa2Embedding,
)
from benchmark.correctness.families.text import TextFamily, TextImageFamily
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


class Wan22I2V(Wan22Video):
    """Wan2.2-TI2V-5B image-to-video with diffusers' WanImageToVideoPipeline (expand_timesteps).

    The set's first frames are 320x180 and mstar's wan22 refuses an image that isn't the request's size, so the
    reference conditions on frames resized to 480x832 by diffusers' own VideoProcessor (PIL lanczos, what the
    pipeline would do itself) and keeps them in <ref_dir>/conditioning/ for the served runs to send.
    """

    image_to_video = True

    def load_pipeline(self, device: str):
        import torch
        from diffusers import WanImageToVideoPipeline

        return WanImageToVideoPipeline.from_pretrained(self.hf_id, torch_dtype=torch.bfloat16).to(device)

    def make_reference(self, sset, ref_dir: Path, device: str) -> None:
        from diffusers.video_processor import VideoProcessor
        from PIL import Image

        cond = ref_dir / "conditioning"
        cond.mkdir(parents=True, exist_ok=True)
        vp = VideoProcessor(vae_scale_factor=16)
        items = []
        for it in sset.items:
            img = vp.resize(Image.open(sset.path(it.files["image"])).convert("RGB"), self.height, self.width)
            path = cond / f"{it.id}.png"
            img.save(path)
            items.append(dataclasses.replace(it, files={**it.files, "image": str(path.resolve())}))
        super().make_reference(dataclasses.replace(sset, items=items), ref_dir, device)


class Cosmos3Reference:
    """diffusers' Cosmos3OmniPipeline built from the checkpoint's parts, prompted like mstar.

    mstar leaves out the system prompt and the resolution and duration sentences unless a request
    asks for them. diffusers 0.41 adds all three by default, which changes the conditioning.
    """

    generator_device = "cuda"  # Cosmos3 draws its noise on the GPU, not the CPU
    ref_call_kwargs = {"use_system_prompt": False, "add_resolution_template": False,
                       "add_duration_template": False, "enable_safety_check": False}
    ref_scheduler_kwargs: dict = {}  # where the serving recipe differs from the checkpoint's scheduler config

    def load_pipeline(self, device: str):
        import torch
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
        from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        snap = Path(snapshot_download(self.hf_id))
        index = json.loads((snap / "model_index.json").read_text())
        dtype = torch.bfloat16
        return Cosmos3OmniPipeline(
            transformer=Cosmos3OmniTransformer.from_pretrained(snap, subfolder="transformer", torch_dtype=dtype),
            text_tokenizer=AutoTokenizer.from_pretrained(snap / "text_tokenizer"),
            vae=AutoencoderKLWan.from_pretrained(snap, subfolder="vae", torch_dtype=dtype),
            scheduler=UniPCMultistepScheduler.from_pretrained(snap, subfolder="scheduler",
                                                              **self.ref_scheduler_kwargs),
            sound_tokenizer=None, enable_safety_checker=False,
            use_native_flow_schedule=bool(index.get("use_native_flow_schedule", False)),
        ).to(device)

    def reference_description(self) -> str:
        extra = f", scheduler {self.ref_scheduler_kwargs}" if self.ref_scheduler_kwargs else ""
        return super().reference_description() + ", no system prompt or metadata sentences" + extra

    @staticmethod
    def reference_frames(out):
        return out.video

    def fp32_anchor(self, pipe, moves: list[float]) -> None:
        """Cosmos3OmniPipeline encodes the conditioning clip with the VAE in bf16. Run that encode in fp32 and
        record how far it moves the anchor (latent frame 0) from the bf16 encode."""
        import torch

        encode = pipe._encode_video

        def fp32_encode(x):
            bf16 = encode(x)[:, :, :1].float()
            dtype = pipe.vae.dtype
            pipe.vae.to(torch.float32)
            try:
                out = encode(x)
            finally:
                pipe.vae.to(dtype)
            moves.append(float((out[:, :, :1].float() - bf16).norm() / bf16.norm()))
            return out

        pipe._encode_video = fp32_encode


class Cosmos3Video(Cosmos3Reference, VideoFamily):
    hf_id = "nvidia/Cosmos3-Nano"
    height, width, num_frames, steps, fps = 480, 832, 17, 35, 24
    guidance = 6.0


class Cosmos3WindowedVideo(Cosmos3Video):
    """Windowed autoregressive t2v. Its requests set window_mode, which the plain video family never
    sends, so the walk the config serves runs. diffusers has no windowed sampler, so the verdict is two
    server instances agreeing and PSNR against the t2v reference is only a note.
    --family-opt window_mode=kv|chained (default kv), window_frames=N (default 13, two windows of four
    latent frames for the 17-frame clip; three-latent-frame clips come out smeared even from plain t2v
    and diffusers)."""

    default_restarts = 2
    self_consistency_only = True
    reference_is_gate = False

    @property
    def request_extra(self) -> dict:
        return {"window_mode": self.opt("window_mode", "kv"), "window_frames": self.opt("window_frames", 13, int)}

    def reference_description(self) -> str:
        return super().reference_description() + " (whole clip at once, not windowed)"


class Cosmos3I2V(Cosmos3Video):
    """Image-to-video. diffusers 0.41 resizes the frame by cover-scale and center crop, as mstar's default
    conditioning_resize (aspect_crop) does."""

    image_to_video = True


class Cosmos3T2I(Cosmos3Reference, ImageFamily):
    """Text-to-image. mstar's t2i recipe (flow shift 3, guidance only for t in [400, 1000]) has no
    diffusers switch, so both sides run the checkpoint scheduler's shift with guidance on every step."""

    hf_id = "nvidia/Cosmos3-Nano"
    steps = 35
    guidance = 6.0
    request_extra = {"flow_shift": 1.0, "guidance_interval": [0.0, 1000.0]}

    def pipeline_kwargs(self, it) -> dict:
        return {**super().pipeline_kwargs(it), "num_frames": 1, "output_type": "pil", **self.ref_call_kwargs}

    def reference_image(self, pipe, prompt: str, generator, kwargs: dict):
        return pipe(prompt=prompt, generator=generator, **kwargs).video[0]


class Cosmos3EdgeVideo(Cosmos3Video):
    # The Edge yaml serves video at flow shift 12 on the native flow schedule, without the karras transform.
    hf_id = "nvidia/Cosmos3-Edge"
    num_frames, steps = 33, 20
    ref_scheduler_kwargs = {"flow_shift": 12.0, "use_karras_sigmas": False}


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
    "ltx2_5": AudioVideoFamily,
}

# Configs whose row sends a request mode their model's family doesn't.
CONFIG_FAMILIES: dict[str, type[Family]] = {
    "cosmos3_nano_ar": Cosmos3WindowedVideo,
    "pi05_droid": Pi05DroidActions,
}

_OMNILINGUAL = "no family yet, its reference (omnilingual-asr on fairseq2) needs its own venv"

UNSUPPORTED: dict[str, str] = {
    "fish_s2_pro": "no family yet, it needs a reference build of Fish Audio's S2-Pro",
    "lingbot_world_1_3b": "no family yet, its camera-controlled world video needs its own reference",
    "omnilingual_asr_ctc_300m": _OMNILINGUAL,
    "omnilingual_asr_ctc_1b": _OMNILINGUAL,
    "omnilingual_asr_ctc_3b": _OMNILINGUAL,
    "omnilingual_asr_ctc_7b": _OMNILINGUAL,
    "test_text_session": "test-only config (session plumbing), no model output to verify",
}


def family_for_config(config_name: str, model_key: str, cfg: dict, opts: dict[str, str]) -> Family:
    if model_key in UNSUPPORTED:
        raise NotImplementedError(f"{config_name} ({model_key}): {UNSUPPORTED[model_key]}")
    if model_key not in FAMILIES:
        raise NotImplementedError(f"{config_name}: no correctness family for model {model_key!r}")
    if model_key == "qwen3_omni" and opts.get("set") == "text":
        return OmniText(model_key, cfg, opts)
    if model_key == "cosmos3" and opts.get("mode") in ("t2i", "i2v"):
        return {"t2i": Cosmos3T2I, "i2v": Cosmos3I2V}[opts["mode"]](model_key, cfg, opts)
    if config_name in CONFIG_FAMILIES:
        return CONFIG_FAMILIES[config_name](model_key, cfg, opts)
    if model_key == "wan22" and opts.get("mode") == "i2v":
        return Wan22I2V(model_key, cfg, opts)
    if opts.get("set") == "image" and issubclass(FAMILIES[model_key], TextFamily):
        hf_id = FAMILIES[model_key].hf_id
        cls = type(f"TextImage_{hf_id.rsplit('/', 1)[-1]}", (TextImageFamily,), {"hf_id": hf_id})
        return cls(model_key, cfg, opts)
    return FAMILIES[model_key](model_key, cfg, opts)
