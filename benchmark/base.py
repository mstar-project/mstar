from abc import ABC, abstractmethod
from enum import Enum


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
        if self.disable_cfg:
            return {
                "cfg_img_scale": 1.0,
                "cfg_text_scale": 1.0,
            }
        if request_type == RequestType.I2I:
            return {
                "cfg_img_scale": 2.0,
                "cfg_interval": [0.0, 1.0],
                "cfg_renorm_type": "text_channel",
            }
        return {}

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
