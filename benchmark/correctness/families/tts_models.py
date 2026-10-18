"""Per-model TTS request kwargs and reference implementations.

References that aren't in the serving venv (kokoro, snac) are pip-installed with --no-deps into
<data-root>/refenv and only appended to sys.path here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from benchmark.correctness.families.tts import TtsFamily
from benchmark.correctness.sets import DEFAULT_DATA_ROOT, SampleSet

REFENV = DEFAULT_DATA_ROOT / "refenv"


def use_refenv() -> None:
    if REFENV.is_dir() and str(REFENV) not in sys.path:
        sys.path.append(str(REFENV))


def _stub_missing_espeak() -> None:
    """kokoro imports misaki.espeak, which needs phonemizer. mstar runs without the espeak
    fallback when it's missing (kokoro/g2p.py), so stub it out the same way.
    """
    import types

    try:
        import misaki.espeak  # noqa: F401
        return
    except ImportError:
        pass
    import misaki

    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("espeak-ng / phonemizer not installed")

    stub = types.ModuleType("misaki.espeak")
    stub.EspeakFallback = _Unavailable
    stub.EspeakG2P = _Unavailable
    sys.modules["misaki.espeak"] = stub
    misaki.espeak = stub


def _stub_unused_qwen_tts_deps() -> None:
    """qwen_tts imports its 25 Hz tokenizer eagerly, which needs sox and onnxruntime. The 12 Hz
    checkpoints never call into it, so empty modules are enough when those aren't installed."""
    import importlib.util
    import types

    for name in ("sox", "onnxruntime"):
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            continue
        stub = types.ModuleType(name)

        def _missing(attr, name=name):
            raise RuntimeError(f"{name} is not installed (only qwen_tts' 25 Hz tokenizer uses it)")

        stub.__getattr__ = _missing
        sys.modules[name] = stub


class KokoroTts(TtsFamily):
    """Kokoro-82M. Nothing is sampled, so durations should match the kokoro package."""

    deterministic = True

    @property
    def voice(self) -> str:
        return self.opt("voice", "af_heart")

    def tts_kwargs(self, i: int, item) -> dict:
        return {"voice": self.voice, "speed": 1.0, "seed": i}

    def reference_id(self) -> str:
        return f"kokoro-pkg-0.9.4-{self.voice}"

    def reference_description(self) -> str:
        return f"`kokoro` 0.9.4 package KPipeline (eager fp32), voice {self.voice}, speed 1.0"

    def reference_synthesize(self, sset: SampleSet, device: str):
        use_refenv()
        _stub_missing_espeak()
        import torch
        from kokoro import KModel, KPipeline

        model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
        pipe = KPipeline(lang_code=self.voice[0], repo_id="hexgrad/Kokoro-82M", model=model)
        for i, it in enumerate(sset.items):
            torch.manual_seed(i)
            chunks = [r.audio.detach().float().cpu().numpy() for r in pipe(it.prompt, voice=self.voice, speed=1.0)
                      if r.audio is not None]
            yield it, (np.concatenate(chunks) if chunks else np.zeros(0, np.float32)), 24000


class OrpheusTts(TtsFamily):
    """Orpheus 3B, sampled with the model defaults on both sides, so WER and duration are the gate."""

    LM_ID = "canopylabs/orpheus-3b-0.1-ft"
    TOKENIZER_ID = "canopylabs/orpheus-3b-0.1-pretrained"
    SAMPLING = {"temperature": 0.6, "top_p": 0.8, "repetition_penalty": 1.3}

    @property
    def voice(self) -> str:
        return self.opt("voice", "tara")

    def tts_kwargs(self, i: int, item) -> dict:
        return {"voice": self.voice, "seed": i, "max_output_tokens": 2048, **self.SAMPLING}

    def reference_id(self) -> str:
        return f"hf-orpheus-3b-ft-snac-{self.voice}-sampled"

    def reference_description(self) -> str:
        return (f"HF transformers LlamaForCausalLM {self.LM_ID} (bf16, sdpa) sampled {self.SAMPLING}, seed = item "
                f"index, + `snac` 1.2.1 SNAC 24 kHz full-sequence decode, voice {self.voice}")

    def reference_synthesize(self, sset: SampleSet, device: str):
        use_refenv()
        import torch
        from snac import SNAC
        from transformers import AutoTokenizer, LlamaForCausalLM

        tok = AutoTokenizer.from_pretrained(self.TOKENIZER_ID)
        lm = LlamaForCausalLM.from_pretrained(self.LM_ID, dtype=torch.bfloat16, attn_implementation="sdpa")
        lm = lm.to(device).eval()
        snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        for i, it in enumerate(sset.items):
            # Same prompt layout as mstar/model/orpheus/orpheus_model.py.
            ids = [128259] + tok(f"{self.voice}: {it.prompt}").input_ids + [128009, 128260, 128261, 128257]
            torch.manual_seed(i)
            with torch.inference_mode():
                out = lm.generate(torch.tensor([ids], device=device), max_new_tokens=2048, do_sample=True,
                                  eos_token_id=128258, pad_token_id=128258, **self.SAMPLING)
            gen = [t for t in out[0, len(ids):].tolist() if t != 128258]
            yield it, self.snac_decode(snac, gen, device), 24000

    @staticmethod
    def snac_decode(snac, tokens: list[int], device: str) -> np.ndarray:
        import torch

        n = len(tokens) // 7
        if n == 0:
            return np.zeros(0, np.float32)
        codes = (np.asarray(tokens[: 7 * n]) - 128266) % 4096
        f = codes.reshape(n, 7)
        levels = [f[:, [0]].reshape(1, -1), f[:, [1, 4]].reshape(1, -1), f[:, [2, 3, 5, 6]].reshape(1, -1)]
        with torch.inference_mode():
            audio = snac.decode([torch.as_tensor(lv, device=device, dtype=torch.long) for lv in levels])
        return audio[0, 0].float().cpu().numpy()


class ChatterboxTts(TtsFamily):
    """Chatterbox base, turbo and multilingual, with checkpoint defaults and the built-in voice."""

    def tts_kwargs(self, i: int, item) -> dict:
        kw = {"voice": "default", "seed": i}
        if self.model_key == "chatterbox_multilingual":
            kw["language_id"] = "en"
        return kw

    def reference_id(self) -> str:
        return f"chatterbox-tts-pkg-{self.model_key}-sampled"

    def reference_description(self) -> str:
        return "`chatterbox-tts` package ChatterboxTTS / ChatterboxTurboTTS / multilingual, built-in voice, defaults"

    def reference_synthesize(self, sset: SampleSet, device: str):
        use_refenv()
        import torch

        if self.model_key == "chatterbox_turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS as Cls
        elif self.model_key == "chatterbox_multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Cls
        else:
            from chatterbox.tts import ChatterboxTTS as Cls
        model = Cls.from_pretrained(device=device)
        for i, it in enumerate(sset.items):
            torch.manual_seed(i)
            kw = {"language_id": "en"} if self.model_key == "chatterbox_multilingual" else {}
            wav = model.generate(it.prompt, **kw)
            yield it, wav.squeeze().float().cpu().numpy(), model.sr


class Qwen3Tts(TtsFamily):
    """Qwen3-TTS CustomVoice, VoiceDesign and Base, sampled with the checkpoint's settings."""

    HF = {"qwen3_tts": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "qwen3_tts_1p7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
          "qwen3_tts_voicedesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
          "qwen3_tts_base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}
    INSTRUCT = "A clear, friendly adult female voice with a neutral accent."

    def tts_kwargs(self, i: int, item) -> dict:
        kw = {"language": "English", "seed": i}
        if self.model_key == "qwen3_tts_voicedesign":
            kw["instruct"] = self.INSTRUCT
        elif self.model_key == "qwen3_tts_base":
            kw["ref_text"] = item.params["prompt_text"]
        else:
            kw["voice"] = self.opt("voice", "vivian")
        return kw

    def request_files(self, sset: SampleSet, item) -> list[Path]:
        return [sset.path(item.files["prompt_audio"])] if self.model_key == "qwen3_tts_base" else []

    def reference_id(self) -> str:
        return f"qwen-tts-0.1.1-{self.HF[self.model_key].split('/')[-1]}-sampled"

    def reference_description(self) -> str:
        return f"`qwen-tts` 0.1.1 Qwen3TTSModel {self.HF[self.model_key]} (bf16, sdpa), checkpoint sampling defaults"

    def reference_synthesize(self, sset: SampleSet, device: str):
        _stub_unused_qwen_tts_deps()
        import torch
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(self.HF[self.model_key], dtype=torch.bfloat16,
                                              attn_implementation="sdpa", device_map=device)
        for i, it in enumerate(sset.items):
            torch.manual_seed(i)
            if self.model_key == "qwen3_tts_voicedesign":
                wavs, sr = model.generate_voice_design(text=it.prompt, language="English", instruct=self.INSTRUCT)
            elif self.model_key == "qwen3_tts_base":
                wavs, sr = model.generate_voice_clone(text=it.prompt, language="English",
                                                      ref_audio=str(sset.path(it.files["prompt_audio"])),
                                                      ref_text=it.params["prompt_text"])
            else:
                wavs, sr = model.generate_custom_voice(text=it.prompt, language="English",
                                                       speaker=self.opt("voice", "vivian"))
            yield it, np.asarray(wavs[0], dtype=np.float32), sr


class OmniVoiceTts(TtsFamily):
    """OmniVoice with the auto voice and default decoding settings."""

    def tts_kwargs(self, i: int, item) -> dict:
        return {"language": "English", "seed": i}

    def reference_id(self) -> str:
        return "omnivoice-pkg-08be0b4-auto-voice"

    def reference_description(self) -> str:
        return "`omnivoice` (k2-fsa/OmniVoice @08be0b4) OmniVoice.generate, fp16, auto voice, default settings"

    def reference_synthesize(self, sset: SampleSet, device: str):
        import torch
        from omnivoice.models.omnivoice import OmniVoice

        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=torch.float16)
        for i, it in enumerate(sset.items):
            torch.manual_seed(i)
            audio = model.generate(text=it.prompt, language="English")
            x = audio[0] if isinstance(audio, (list, tuple)) else audio
            x = x.float().cpu().numpy() if hasattr(x, "cpu") else np.asarray(x, np.float32)
            yield it, x.reshape(-1), 24000
