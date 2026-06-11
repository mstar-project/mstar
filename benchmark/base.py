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
    def __init__(self, disable_cfg: bool = False, image_preprocess: str = "vllm", **kwargs):
        super().__init__(**kwargs)
        self.disable_cfg = disable_cfg
        self.image_preprocess = image_preprocess
        self.disable_cfg = disable_cfg

    def get_model_kwargs(self, request_type: RequestType):
        # Force greedy on the thinker for cross-system parity. Without this,
        # mstar falls back to mstar/model/bagel/config.py's temperature=0.6
        # default while vllm-omni gets temperature=0 from request.py:952 — the
        # two systems would generate different token sequences, mostly
        # affecting I2T latency (variable EOS timing) but also leaking into
        # T2I/I2I via the tokens emitted before the image-gen handoff.
        kwargs = {
            "temperature": 0.0,
            "image_preprocess": self.image_preprocess,
            "max_tokens": 128,
            "max_output_tokens": 128,
        }
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

    def get_openai_system_message(self) -> Optional[dict]:
        return {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are BAGEL, a helpful assistant created by ByteDance.",
                }
            ],
        }


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
        # Force greedy on the Thinker (text decoding) so cross-system runs
        # see deterministic text for the same prompt. Send both `max_tokens`
        # (OpenAI convention — vllm-omni / sglang-omni) and `max_output_tokens`
        # (mstar's own kwarg, read in mstar/model/base.py:372-373; default
        # MAX_OUTPUT_TOKENS=2048). Without the second key, mstar silently
        # ignores the cap and runs to natural EOS, which on T2T was observed
        # at ~361 tokens/req vs vllm-omni's 256.
        #
        # talker_temperature / cp_temperature intentionally NOT set: vllm-omni's
        # ``serving_chat._build_sampling_params_list_from_request`` only applies
        # the request's ``temperature`` to Stage 0 (Thinker) and clones YAML
        # defaults for Stage 1 (Talker) — ``temperature=0.9, top_k=50,
        # repetition_penalty=1.05`` per ``vllm-omni/vllm_omni/deploy/qwen3_omni_moe.yaml``.
        # vllm-omni's own benchmark (``vllm_omni/benchmarks/patch/patch.py:335``)
        # sends ``temperature: 0.0`` over the wire just like we do, but the
        # vllm-omni server quietly drops it for the Talker stage. vllm-omni even
        # warns explicitly that ``temperature=0`` on Talker may cause repetitive
        # output. Forcing greedy on mstar's Talker here was a false equivalence —
        # mstar has no stage isolation, so it actually applied talker_temperature=0,
        # collapsing the Talker to a degenerate "mee mee mee" repetition attractor.
        # Letting mstar use its own model default (talker_temperature=0.9) restores
        # apples-to-apples behavior with vllm-omni's effective Talker sampling.
        return {
            "max_tokens": 256,
            "max_output_tokens": 256,
            "thinker_temperature": 0.0,
        }

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
