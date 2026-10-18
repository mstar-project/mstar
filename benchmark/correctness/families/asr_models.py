"""Per-model ASR adapters and their HF transformers references."""

from __future__ import annotations

from benchmark.correctness import metrics as M
from benchmark.correctness.families.asr import AsrFamily
from benchmark.correctness.sets import SampleSet


class WhisperAsr(AsrFamily):
    """Whisper large-v3. The server forces <|en|><|transcribe|><|notimestamps|> and ignores the
    text field. HF generate(language="en", task="transcribe") builds the same prompt.
    """

    hf_id = "openai/whisper-large-v3"

    def model_kwargs(self, i: int) -> dict:
        return {"temperature": 0.0, "max_output_tokens": 444, "language": "en", "task": "transcribe", "seed": i}

    def reference_id(self) -> str:
        return "hf-whisper-large-v3-greedy-bf16"

    def reference_description(self) -> str:
        return "HF transformers WhisperForConditionalGeneration openai/whisper-large-v3 (bf16, sdpa), greedy, en"

    def reference_transcribe(self, sset: SampleSet, device: str) -> list[str]:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        proc = WhisperProcessor.from_pretrained(self.hf_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.hf_id, dtype=torch.bfloat16,
                                                                attn_implementation="sdpa").to(device).eval()
        out = []
        for it in sset.items:
            x, sr = M.read_wav(sset.path(it.files["audio"]))
            feats = proc(M.resample(x, sr, 16000), sampling_rate=16000, return_tensors="pt").input_features
            with torch.inference_mode():
                ids = model.generate(feats.to(device, torch.bfloat16), language="en", task="transcribe",
                                     max_new_tokens=444, do_sample=False)
            out.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
        return out


class HiggsAsr(AsrFamily):
    """Higgs-audio v3 STT. No reference wired up yet (HF remote code), so it's scored against
    ground truth only.
    """

    hf_id = "bosonai/higgs-audio-v3-stt"
    abs_wer_pass = 6.0

    def model_kwargs(self, i: int) -> dict:
        return {"temperature": 0.0, "max_output_tokens": 448, "seed": i}

    def reference_id(self) -> str:
        return "hf-higgs-audio-v3-stt-remote-code"

    def reference_description(self) -> str:
        return "HF remote-code HiggsAudio3Model (not implemented)"
