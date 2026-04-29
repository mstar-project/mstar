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

    # Robotics
    VLA = "vision_language_action"   # images + text → action  (pi0.5)
    V2V = "video_to_video"           # video + metadata → video (world model)

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
        if self == RequestType.VLA:
            return "action"
        if self == RequestType.V2V:
            return "video"
        return "text"

    def get_input_modalities(self):
        if self in [RequestType.I2I, RequestType.I2T, RequestType.I2S]:
            return "image"
        if self in [RequestType.V2T, RequestType.V2S, RequestType.V2V]:
            return "video"
        if self in [RequestType.A2T, RequestType.A2S]:
            return "audio"
        if self == RequestType.VLA:
            return "image,text"
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


class Pi05(Model):
    """Physical Intelligence Pi0.5 VLA model.

    Input:  3 RGB images (base, left-wrist, right-wrist) + text task + robot state
    Output: action trajectory [50, 32] as raw float32 bytes
    """

    def __init__(self, action_dim: int = 32, action_horizon: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.action_dim = action_dim
        self.action_horizon = action_horizon

    def get_hf_url(self):
        return "physical-intelligence/pi0.5"

    def get_supported_modalities(self):
        return {RequestType.VLA}

    def get_model_kwargs(self, request_type: RequestType):
        return {}  # robot_state is per-request and lives on RequestInput.model_kwargs


class VJepa2AC(Model):
    """V-JEPA 2 action-conditioned world model.

    Input:  video clip + per-step actions + states (in model_kwargs)
    Output: predicted latent hidden states as raw float32 bytes
    """

    def __init__(
        self,
        rollout_horizon: int = 4,
        action_dim: int = 7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rollout_horizon = rollout_horizon
        self.action_dim = action_dim

    def get_hf_url(self):
        return "facebook/vjepa2-ac-vitg-256"

    def get_supported_modalities(self):
        return {RequestType.V2V}

    def get_model_kwargs(self, request_type: RequestType):
        # actions/states/rollout_horizon are per-request and live on RequestInput.model_kwargs
        return {}


class ModelType(Enum):
    BAGEL = "bagel"
    ORPHEUS = "orpheus"
    QWEN3OMNI = "qwen3omni"
    PI05 = "pi05"
    VJEPA2AC = "vjepa2ac"

    def inst(self, **kwargs) -> Model:
        if self == ModelType.BAGEL:
            return Bagel(**kwargs)
        if self == ModelType.ORPHEUS:
            return Orpheus(**kwargs)
        if self == ModelType.QWEN3OMNI:
            return Qwen3Omni(**kwargs)
        if self == ModelType.PI05:
            return Pi05(**kwargs)
        if self == ModelType.VJEPA2AC:
            return VJepa2AC(**kwargs)
        raise NotImplementedError(f"Unknown model type {self}")
