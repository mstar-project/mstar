from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional


class Status(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PROGRESS = "progress"


class RequestType(Enum):
    # Text input
    T2T = "text_to_text"
    T2I = "text_to_image"
    T2S = "text_to_speech"

    # Image inputs
    I2T = "image_to_text"
    I2I = "image_to_image"
    I2S = "image_to_speech"

    # Audio input
    A2T = "audio_to_text"
    A2S = "audio_to_speech"

    # Video input
    V2T = "video_to_text"
    V2S = "video_to_speech"

    def get_output_modalities(self):
        if self in [RequestType.I2I, RequestType.T2I]:
            return "image"
        if self in [RequestType.T2S, RequestType.I2S, RequestType.V2S, RequestType.A2S]:
            return "audio"
        return "text"

    def get_input_modalities(self):
        if self in [RequestType.I2I, RequestType.I2T, RequestType.I2S]:
            return "image"
        if self in [RequestType.V2T, RequestType.V2S]:
            return "video"
        if self in [RequestType.A2T, RequestType.A2S]:
            return "audio"
        return "text"


class Model(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs
        self._tokenizer = None

    def get_model_kwargs(self, request_type: RequestType):
        return {}

    def get_openai_system_message(self) -> Optional[dict]:
        """Return the OpenAI-format system message to prepend on /v1/chat/completions
        requests, or None to send no system role.

        Per-model because the right answer is model-specific:
          - Qwen3-Omni (audio-output models in general): the official examples
            (gradio_demo, openai_chat_completion_client_for_multimodal_generation,
            seed_tts_dataset.SEED_TTS_DEFAULT_OMNI_SYSTEM_PROMPT) all hardcode the
            "You are Qwen…" preamble and the talker's behavior degrades without it.
          - BAGEL: the model has its own `<|fim_middle|><|im_start|>…<|im_end|>`
            template; an extra system role corrupts prompt handling and produces
            off-prompt images. vllm-omni's own diffusion bench
            (`benchmarks/diffusion/backends.py`) sends user-only messages for BAGEL.

        Default: None (no system message). Override in subclasses that need one.
        """
        return None

    @abstractmethod
    def get_hf_url(self):
        pass

    @abstractmethod
    def get_supported_modalities(self):
        pass

    def get_tokenizer(self):
        """Lazy-load the model's HF tokenizer for per-chunk re-tokenization in
        ITL aggregation (matches sglang.bench_serving --accept-length path).
        Cached on the instance to avoid repeated downloads."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.get_hf_url(), trust_remote_code=True)
        return self._tokenizer


class Bagel(Model):
    def __init__(self, disable_cfg: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.disable_cfg = disable_cfg

    def get_model_kwargs(self, request_type: RequestType):
        # Force greedy on the thinker for cross-system parity. Without this,
        # mminf falls back to mminf/model/bagel/config.py's temperature=0.6
        # default while vllm-omni gets temperature=0 from request.py:952 — the
        # two systems would generate different token sequences, mostly
        # affecting I2T latency (variable EOS timing) but also leaking into
        # T2I/I2I via the tokens emitted before the image-gen handoff.
        kwargs = {"temperature": 0.0}
        if self.disable_cfg:
            kwargs.update({
                "cfg_img_scale": 1.0,
                "cfg_text_scale": 1.0,
            })
        elif request_type == RequestType.I2I:
            kwargs.update({
                "cfg_img_scale": 2.0,
                "cfg_interval": [0.0, 1.0],
                "cfg_renorm_type": "text_channel",
            })
        return kwargs

    def get_hf_url(self):
        return "ByteDance-Seed/BAGEL-7B-MoT"

    def get_supported_modalities(self):
        return {RequestType.T2T, RequestType.T2I, RequestType.I2I, RequestType.I2T}


class Orpheus(Model):
    def get_hf_url(self):
        return "canopylabs/orpheus-3b-0.1-ft"

    def get_supported_modalities(self):
        return {RequestType.T2S}


class Qwen3Omni(Model):
    def get_hf_url(self):
        return "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    def get_openai_system_message(self) -> Optional[dict]:
        # Verbatim text from vllm-omni's official Qwen3-Omni examples
        # (gradio_demo, openai_chat_completion_client_for_multimodal_generation,
        # benchmarks/data_modules/seed_tts_dataset.SEED_TTS_DEFAULT_OMNI_SYSTEM_PROMPT).
        # Required for correct talker behavior on /v1/chat/completions.
        return {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                    ),
                }
            ],
        }

    def get_model_kwargs(self, request_type: RequestType):
        # Cap thinker output at 256 tokens for cross-system fairness. Matches
        # sglang-omni's H200 conventions (THINKER_MAX_NEW_TOKENS=256 in
        # benchmarks/tasks/tts.py:911, max_tokens=256 default in
        # video_understanding.py and benchmark_omni_videomme.py) and
        # vllm-omni's bench convention (always sets max_tokens via
        # per-dataset --output-len flags → patch.py:336). Without a cap the
        # comparison becomes "whose EOS detection terminates earlier?"
        # rather than "whose decode is faster per token?".
        #
        # Force greedy on every sub-model that participates so cross-system
        # runs see deterministic tokens for the same prompt. mminf's
        # qwen3_omni_model.py:521-540 defaults are thinker=0.7, talker=0.9,
        # cp=1.0, which would otherwise make output length (and therefore
        # RTF / audio duration / text token count) vary across runs.
        # Send both `max_tokens` (OpenAI convention — vllm-omni / sglang-omni)
        # and `max_output_tokens` (mminf's own kwarg, read in
        # mminf/model/base.py:372-373; default MAX_OUTPUT_TOKENS=2048). Without
        # the second key, mminf silently ignores the cap and runs to natural
        # EOS, which on T2T was observed at ~361 tokens/req vs vllm-omni's 256.
        kwargs = {
            "max_tokens": 256,
            "max_output_tokens": 256,
            "thinker_temperature": 0.0,
        }
        if request_type in (
            RequestType.T2S,
            RequestType.I2S,
            RequestType.A2S,
            RequestType.V2S,
        ):
            kwargs["talker_temperature"] = 0.0
            kwargs["cp_temperature"] = 0.0
        return kwargs

    def get_supported_modalities(self):
        return {
            RequestType.T2T,
            RequestType.T2S,
            RequestType.I2T,
            RequestType.I2S,
            RequestType.A2T,
            RequestType.A2S,
            RequestType.V2T,
            RequestType.V2S,
        }


class ModelType(Enum):
    BAGEL = "bagel"
    ORPHEUS = "orpheus"
    QWEN3OMNI = "qwen3omni"

    def inst(self, **kwargs) -> Model:
        if self == ModelType.BAGEL:
            return Bagel(**kwargs)
        if self == ModelType.ORPHEUS:
            return Orpheus(**kwargs)
        if self == ModelType.QWEN3OMNI:
            return Qwen3Omni(**kwargs)
        raise NotImplementedError(f"Unknown model type {self}")
