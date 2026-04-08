from enum import Enum


class Status(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PROGRESS = "progress"


class RequestType(Enum):
    T2T = "text_to_text"
    T2I = "text_to_image"
    I2T = "image_to_text"
    I2I = "image_to_image"

    def get_output_modalities(self):
        if self in [RequestType.I2I, RequestType.T2I]:
            return "image"
        return "text"


class Model(Enum):
    BAGEL = "bagel"

    def get_model_kwargs(self, request_type: RequestType):
        if self == Model.BAGEL:
            if request_type == RequestType.I2I:
                return {
                "cfg_img_scale": 2.0,
                "cfg_interval": [0.0, 1.0],
                "cfg_renorm_type": "text_channel",
            }
        return {}

    def get_hf_url(self):
        if self == Model.BAGEL:
            return "ByteDance-Seed/BAGEL-7B-MoT"
        return ""
