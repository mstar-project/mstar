"""Model-based scorers ("judges"): ASR for TTS round-trip WER, speaker
similarity, and predicted MOS. All heavy imports are lazy, and each judge is
loaded once per process and reused across repeats and sweep points.

Every judge has an ``id`` that goes into the results file, so a WER from one
ASR judge is never compared with a WER from another by accident.

* ``whisper:<hf id>``: Whisper through transformers (default
  ``openai/whisper-large-v3``, the Seed-TTS-eval English judge); long clips use
  Whisper's sequential long-form decoding instead of being cut at 30 s.
* ``wavlm-sv:<hf id>``: cosine similarity of WavLM x-vectors (default
  ``microsoft/wavlm-base-plus-sv``). Seed-TTS-eval's SIM uses a WavLM-large
  ECAPA checkpoint instead, so the absolute values differ from published
  tables; compare runs scored with the same judge id.
* ``utmos22_strong``: UTMOS22 strong learner via ``tarepan/SpeechMOS`` on
  torch.hub (predicted MOS on a 1 to 5 scale, 16 kHz).
* ``clip:<hf id>``: CLIP score of an image against its prompt, 100 times the
  cosine similarity of the two embeddings floored at 0 (default
  ``openai/clip-vit-base-patch32``).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import numpy as np

from benchmark.eval.audio import resample

RATE = 16000


def _device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class WhisperJudge:
    def __init__(self, model_id: str = "openai/whisper-large-v3", device: str | None = None, batch_size: int = 16):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.id = f"whisper:{model_id}"
        self.device = _device(device)
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self.torch = torch

    def transcribe(self, clips: Sequence[tuple[np.ndarray, int]], langs: Sequence[str | None]) -> list[str]:
        out: list[str | None] = [None] * len(clips)
        by_lang: dict[str | None, list[int]] = {}
        for i, lang in enumerate(langs):
            by_lang.setdefault(lang, []).append(i)
        for lang, idx in by_lang.items():
            for s in range(0, len(idx), self.batch_size):
                part = idx[s : s + self.batch_size]
                audio = [resample(clips[i][0], clips[i][1], RATE) for i in part]
                for i, text in zip(part, self._generate(audio, lang), strict=True):
                    out[i] = text.strip()
        return [t or "" for t in out]

    def _generate(self, audio: list[np.ndarray], lang: str | None) -> list[str]:
        long_form = any(len(a) > 30 * RATE for a in audio)
        kwargs = {"truncation": False, "padding": "longest"} if long_form else {}
        feats = self.processor(audio, sampling_rate=RATE, return_tensors="pt", return_attention_mask=True, **kwargs)
        gen_kwargs = {"task": "transcribe", "return_timestamps": long_form}
        if lang:
            gen_kwargs["language"] = lang.split("_")[0].split("-")[0]
        with self.torch.inference_mode():
            ids = self.model.generate(
                feats.input_features.to(self.device, self.dtype),
                attention_mask=feats.attention_mask.to(self.device),
                **gen_kwargs,
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True)


class SpeakerSimJudge:
    def __init__(self, model_id: str = "microsoft/wavlm-base-plus-sv", device: str | None = None):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.id = f"wavlm-sv:{model_id}"
        self.device = _device(device)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id).to(self.device).eval()
        self.torch = torch

    def embed(self, clip: tuple[np.ndarray, int]) -> np.ndarray:
        x = resample(clip[0], clip[1], RATE)
        inputs = self.extractor(x, sampling_rate=RATE, return_tensors="pt")
        with self.torch.inference_mode():
            emb = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).embeddings[0]
        emb = self.torch.nn.functional.normalize(emb.float(), dim=-1)
        return emb.cpu().numpy()

    def similarity(self, generated: tuple[np.ndarray, int], reference: tuple[np.ndarray, int]) -> float:
        return float(np.dot(self.embed(generated), self.embed(reference)))


class MosJudge:
    def __init__(self, device: str | None = None, hub_ref: str = "tarepan/SpeechMOS:v1.2.0"):
        import torch

        self.id = f"utmos22_strong:{hub_ref}"
        self.device = _device(device)
        self.model = torch.hub.load(hub_ref, "utmos22_strong", trust_repo=True).to(self.device).eval()
        self.torch = torch

    def score(self, clip: tuple[np.ndarray, int]) -> float | None:
        x = resample(clip[0], clip[1], RATE)
        if len(x) < RATE // 10:
            return None  # too short to score
        wav = self.torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            return float(self.model(wav, RATE).item())


class ClipJudge:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.id = f"clip:{model_id}"
        self.device = _device(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.torch = torch

    def score(self, images: Sequence, text: str) -> float:
        """Mean CLIP score of ``images`` (PIL images, e.g. a video's sampled frames) against ``text``."""
        inputs = self.processor(text=[text], images=list(images), return_tensors="pt", padding=True, truncation=True)
        with self.torch.inference_mode():
            out = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        img = self.torch.nn.functional.normalize(out.image_embeds.float(), dim=-1)
        txt = self.torch.nn.functional.normalize(out.text_embeds.float(), dim=-1)
        return float((100 * (img @ txt.T)).clamp(min=0).mean().item())


@cache
def asr_judge(model_id: str, device: str | None = None) -> WhisperJudge:
    return WhisperJudge(model_id, device)


@cache
def sim_judge(model_id: str, device: str | None = None) -> SpeakerSimJudge:
    return SpeakerSimJudge(model_id, device)


@cache
def mos_judge(device: str | None = None) -> MosJudge:
    return MosJudge(device)


@cache
def clip_judge(model_id: str, device: str | None = None) -> ClipJudge:
    return ClipJudge(model_id, device)
