"""DeepSeek-OCR's checkpoint config and image modes.

The language model is DeepSeek-V2's MoE decoder with plain multi-head attention
(``use_mla: false``): Llama attention, a dense first layer, then 64 routed
SwiGLU experts (softmax top-6, not renormalized) plus two shared ones.
"""

import json
from dataclasses import dataclass
from pathlib import Path

# Resource names shared by the decoder components and the LLM node.
KV_CACHE = "kv_cache"
ATTN = "attn"
ROPE = "rope"
SAMPLER = "sampler"

PREFILL_WALKS = ("prefill", "prefill_image")

# The tokenizer's ``<image>``; every image position carries it.
IMAGE_TOKEN = "<image>"

# model_kwargs key: every image's [cols, rows] tile grid, from process_prompt;
# STEP_GRIDS_KEY is the step_metadata key the vision node reads them from
IMAGE_GRIDS_KEY, STEP_GRIDS_KEY = "image_grids", "step_image_grids"


@dataclass(frozen=True)
class DeepseekOCRTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    moe_layer_freq: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    bos_token_id: int
    eos_token_id: int
    # Unlimited-OCR's R-SWA: generated tokens see the whole prompt plus this
    # many recent generated tokens (see ``DeepseekOCRModel``)
    sliding_window: int | None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def shared_intermediate_size(self) -> int:
        return self.moe_intermediate_size * self.n_shared_experts

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_k_dense_replace and layer_idx % self.moe_layer_freq == 0

    @classmethod
    def from_dict(cls, data: dict) -> "DeepseekOCRTextConfig":
        # Only the MHA, softmax-greedy variant is ported; anything else would load
        # and run with the wrong math.
        supported = {
            "use_mla": False, "topk_method": "greedy", "scoring_func": "softmax",
            "hidden_act": "silu", "n_group": 1, "topk_group": 1, "attention_bias": False,
            "rope_scaling": None, "tie_word_embeddings": False,
        }
        for key, value in supported.items():
            if data.get(key, value) != value:
                raise ValueError(f"DeepSeek-OCR: unsupported {key}={data[key]!r} (expected {value!r})")
        if data.get("num_key_value_heads", data["num_attention_heads"]) != data["num_attention_heads"]:
            raise ValueError("DeepSeek-OCR: the MHA path has as many KV heads as query heads")
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            moe_intermediate_size=data["moe_intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_attention_heads"],
            n_routed_experts=data["n_routed_experts"],
            n_shared_experts=data["n_shared_experts"],
            num_experts_per_tok=data["num_experts_per_tok"],
            first_k_dense_replace=data.get("first_k_dense_replace", 0),
            moe_layer_freq=data.get("moe_layer_freq", 1),
            norm_topk_prob=data.get("norm_topk_prob", False),
            routed_scaling_factor=data.get("routed_scaling_factor", 1.0),
            max_position_embeddings=data["max_position_embeddings"],
            rope_theta=data.get("rope_theta", 10000.0),
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            bos_token_id=data.get("bos_token_id", 0),
            eos_token_id=data.get("eos_token_id", 1),
            sliding_window=data.get("sliding_window_size") or data.get("sliding_window"),
        )


@dataclass(frozen=True)
class ImageMode:
    """How an image becomes views (the model card's modes).

    The global view is the image padded to ``base_size`` square. With
    ``crop_mode``, an image larger than ``image_size`` on either side also gets
    ``image_size`` tiles in the best (cols, rows) grid of 2..``max_crops``.
    Without it the single view is ``image_size`` square.
    """

    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True
    min_crops: int = 2
    # The model card's ``infer`` tiles up to 9; vLLM and DeepSeek's own vLLM
    # script cap it at 6; Unlimited-OCR at 32.
    max_crops: int = 9
    patch_size: int = 16
    downsample_ratio: int = 4

    def __post_init__(self):
        for size in (self.base_size, self.image_size):
            if size % (self.patch_size * self.downsample_ratio):
                raise ValueError(f"view size {size} is not a multiple of {self.patch_size * self.downsample_ratio}")
        if not 1 <= self.min_crops <= self.max_crops:
            raise ValueError(f"crops {self.min_crops}..{self.max_crops} are not a range")

    @property
    def global_size(self) -> int:
        """Side of the one view every image gets."""
        return self.base_size if self.crop_mode else self.image_size

    def side_tokens(self, size: int) -> int:
        return size // self.patch_size // self.downsample_ratio


MODES = {
    "tiny": ImageMode(base_size=512, image_size=512, crop_mode=False),
    "small": ImageMode(base_size=640, image_size=640, crop_mode=False),
    "base": ImageMode(base_size=1024, image_size=1024, crop_mode=False),
    "large": ImageMode(base_size=1280, image_size=1280, crop_mode=False),
    "gundam": ImageMode(base_size=1024, image_size=640, crop_mode=True),
}


@dataclass(frozen=True)
class DeepseekOCRConfig:
    text: DeepseekOCRTextConfig
    projector_input_dim: int
    projector_dim: int

    @classmethod
    def from_json(cls, path: Path) -> "DeepseekOCRConfig":
        data = json.loads(Path(path).read_text())
        projector = data["projector_config"]
        if projector.get("projector_type") != "linear" or data.get("tile_tag", "2D") != "2D":
            raise ValueError("DeepSeek-OCR: only the linear projector with 2D tile tags is ported")
        vision = data["vision_config"]["width"]
        sam, clip = vision["sam_vit_b"], vision["clip-l-14-224"]
        if (sam["width"], sam["layers"], clip["width"], clip["layers"]) != (768, 12, 1024, 24):
            raise ValueError("DeepSeek-OCR: only DeepEncoder's SAM-B + CLIP-L is ported")
        return cls(
            text=DeepseekOCRTextConfig.from_dict(data["language_config"]),
            projector_input_dim=projector["input_dim"],
            projector_dim=projector["n_embed"],
        )
