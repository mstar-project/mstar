"""Workloads (what one cell sends) and tables (rows x arms x workloads)."""

from __future__ import annotations

from dataclasses import dataclass

TEXT_PROMPTS = "benchmark/assets/simple_text_queries.txt"
LONG_PROMPTS = "benchmark/assets/long_text_queries_1k.txt"


@dataclass(frozen=True)
class Workload:
    name: str
    family: str
    request_type: str  # benchmark.runner --request-type
    concurrency: tuple[int, ...]
    # Prompt source: a text file (one prompt per line, repo-relative) or "set:<name>", a correctness runner set.
    prompts: str
    output_len: int | None = None  # fixed output length in tokens (sent with ignore_eos)
    unique_prompts: bool = False  # runner --unique-prompts, so no request reuses a cached prefix
    min_requests: int = 16
    waves: int = 5  # requests per cell = max(min_requests, waves * concurrency)
    warmup_waves_first: int = 3  # before the first repeat of a cell
    warmup_waves_next: int = 1
    # Metric whose repeat-to-repeat spread decides a re-run: (key in cell stats, higher is better).
    primary: tuple[str, bool] = ("tok_s", True)

    def num_requests(self, c: int) -> int:
        return max(self.min_requests, self.waves * c)


WORKLOADS: dict[str, Workload] = {w.name: w for w in [
    # Short chat prompts, 256 output tokens each: decode-bound, the usual serving cell.
    Workload("text-chat-256", "text", "text_to_text", (1, 4, 8, 16, 32, 64), TEXT_PROMPTS, output_len=256),
    # ~1k-token prompts, 64 output tokens: prefill-heavy. Unique prompt tags keep prefix caches out of it.
    Workload("text-long-1k", "text", "text_to_text", (1, 4, 16, 64), LONG_PROMPTS, output_len=64,
             unique_prompts=True, primary=("tok_s", True)),
    # Seed-TTS read-aloud sentences (tts_seedtts_v1), model defaults, natural stop.
    Workload("tts-seed", "tts", "text_to_speech", (1, 8, 32, 64), "set:tts_seedtts_v1", min_requests=40,
             primary=("audio_s_per_s", True)),
    # LibriSpeech test-clean utterances (asr_librispeech_v1).
    # min_requests = the set size, so the first run at concurrency 1 covers every item for the WER check.
    Workload("asr-libri", "asr", "audio_to_text", (1, 8, 32), "set:asr_librispeech_v1", min_requests=100,
             primary=("rtfx", True)),
    # Qwen3-Omni answering out loud: a text question, a food photo, a LibriSpeech clip (omni_v1 for the
    # media). No ignore_eos: the talker is undefined past EOS (.claude/skills/benchmarking).
    Workload("omni-text-speech", "omni", "text_to_speech", (1, 4, 16, 32), TEXT_PROMPTS,
             primary=("audio_s_per_s", True)),
    Workload("omni-image-speech", "omni", "image_to_speech", (1, 4, 16, 32), "set:omni_v1",
             primary=("audio_s_per_s", True)),
    Workload("omni-audio-speech", "omni", "audio_to_speech", (1, 4, 16, 32), "set:omni_v1",
             primary=("audio_s_per_s", True)),
    # 20 prompts x 2 sizes at fixed seeds (image_v1), steps and guidance from the correctness family. The
    # first run at concurrency 1 covers the set, for PSNR against the diffusers reference.
    Workload("image-gen", "image", "text_to_image", (1, 4, 8, 16), "set:image_v1", min_requests=40,
             warmup_waves_first=1, primary=("images_per_s", True)),
    # 5 prompts (video_v1) at the correctness family's resolution, frames, steps and guidance.
    Workload("video-t2v", "video", "text_to_video", (1, 2), "set:video_v1", min_requests=5, waves=1,
             warmup_waves_first=1, warmup_waves_next=0, primary=("clips_per_min", True)),
    # The same prompts conditioned on video_v1's first frames (DROID episodes), for the rows that take an image.
    Workload("video-i2v", "video", "image_to_video", (1, 2), "set:video_v1", min_requests=5, waves=1,
             warmup_waves_first=1, warmup_waves_next=0, primary=("clips_per_min", True)),
]}


@dataclass(frozen=True)
class Table:
    name: str
    rows: tuple[str, ...]
    arms: tuple[str, ...]
    workloads: tuple[str, ...]
    repeats: int = 3
    spread_limit: float = 0.05
    note: str = ""


TABLES: dict[str, Table] = {t.name: t for t in [
    Table("qwen35-dense", ("qwen3_5_0.8b", "qwen3_5_2b", "qwen3_5_4b", "qwen3_5_9b", "qwen3_5_9b_tp2"),
          ("mstar", "mstar-tpasync", "mstar-base", "mstar-base-tpasync", "vllm", "vllm-aiter", "sglang-aiter"),
          ("text-chat-256", "text-long-1k"),
          note="Qwen3.5 dense, text in and out, greedy, fixed output length"),
    Table("tts", ("kokoro", "orpheus", "qwen3tts_1p7b"),
          ("mstar", "mstar-colocated", "mstar-base", "ref-kokoro", "ref-orpheus-vllm", "vllm-omni", "vllm-omni-tuned",
           "sglang-omni"), ("tts-seed",),
          note="TTS: Kokoro and Orpheus against their reference implementations, Qwen3-TTS 1.7B (two mstar "
               "configs) against vllm-omni and sglang-omni"),
    Table("omni", ("qwen3omni_colocated", "qwen3omni_2gpu"),
          ("mstar", "mstar-aitermoe", "mstar-base", "vllm-omni", "vllm-omni-tuned"),
          ("omni-text-speech", "omni-image-speech", "omni-audio-speech"),
          note="Qwen3-Omni, spoken answers to text, image and audio questions"),
    Table("image", ("flux2_klein", "flux2_klein_9b", "z_image_turbo", "cosmos3_nano_t2i"),
          ("mstar", "vllm-omni", "vllm-omni-rc", "sglang-diffusion", "sglang-diffusion-bcg", "ref-diffusers"),
          ("image-gen",),
          note="Image generation, latency at concurrency 1 and images/s above it"),
    Table("video", ("wan22", "cosmos3_nano", "cosmos3_edge"),
          ("mstar", "vllm-omni", "vllm-omni-rc", "sglang-diffusion", "sglang-diffusion-compile", "ref-diffusers"),
          ("video-t2v", "video-i2v"),
          note="Text- and image-to-video, time per clip at concurrency 1 and makespan above it"),
    Table("asr", ("whisper_large",), ("mstar", "mstar-split", "mstar-base", "vllm-whisper", "sglang-whisper"),
          ("asr-libri",),
          note="Whisper large-v3 on the OpenAI transcription route"),
    # Functional runs on one dev slice: every cell type once, no timing value.
    Table("dry-text", ("qwen3_5_0.8b",), ("mstar", "vllm-aiter", "sglang-aiter"), ("text-chat-256",), repeats=1),
    Table("dry-tts", ("kokoro", "orpheus_colocated"), ("mstar", "ref-kokoro", "ref-orpheus-vllm"), ("tts-seed",),
          repeats=1),
    Table("dry-omni", ("qwen3tts",), ("vllm-omni",), ("tts-seed",), repeats=1),
    Table("dry-asr", ("whisper_large",), ("mstar",), ("asr-libri",), repeats=1),
]}
