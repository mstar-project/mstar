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
    
    def get_model_kwargs(self, request_type: RequestType):
        return {}

    @abstractmethod
    def get_hf_url(self):
        pass

    @abstractmethod
    def get_supported_modalities(self):
        pass


class Bagel(Model):
    def __init__(self, disable_cfg: bool=False, **kwargs):
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
        return {
            RequestType.T2T,
            RequestType.T2I,
            RequestType.I2I,
            RequestType.I2T
        }


class Orpheus(Model):
    def get_hf_url(self):
        return "canopylabs/orpheus-3b-0.1-ft"
    
    def get_supported_modalities(self):
        return {
            RequestType.T2S
        }

class Qwen3Omni(Model):
    def get_hf_url(self):
        return "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    
    def get_supported_modalities(self):
        return {
            RequestType.T2T,
            RequestType.T2S,
            RequestType.I2T,
            RequestType.I2S,
            RequestType.A2T,
            RequestType.A2S,
            RequestType.V2T,
            RequestType.V2S
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