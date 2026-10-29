"""GR00T N1.7's configs, read off two ``config.json`` files: GR00T's own
(``nvidia/GR00T-N1.7-3B``: the action head, ``select_layer``) and its
backbone's (``nvidia/Cosmos-Reason2-2B``, a Qwen3-VL: text and vision dims).

The checkpoint wins over the reference's class defaults, which disagree with it
(``select_layer`` 12 vs 16, DT ``num_layers`` 16 vs 32). Fields the checkpoint
leaves out keep the class default (``gr00t/configs/model/gr00t_n1d7.py``).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, fields
from pathlib import Path

from mstar.model.qwen3_5.config import Qwen3_5VisionConfig

# resource keys. The ViT keeps `qwen3_5`'s key, because the tower we reuse
# looks itself up by that name; the LLM's attention is ours, so is this.
LLM_ATTN = "llm_attn"
# the bridge is 32 heads x 64 where the ViT is 16 x 64, and a ragged
# resource is per head geometry, so it cannot share the tower's
BRIDGE_ATTN = "bridge_attn"

# The DT's two attentions. Same 32 x 48 geometry, but a resource is one *kind*
# of attention, so they cannot share one: the even blocks cross-attend the
# bridge's text or image rows, the odd blocks attend the 41 `sa` rows.
DT_ATTN = "dt_attn"
DT_XATTN = "dt_xattn"

# The spans a `denoise_step` declares. The bridge's rows are packed into two
# per-request spans so the loop carries no attention masks at all; the cross
# pairs say which queries read which of them.
SA_SPAN = "sa"
VL_TEXT_SPAN = "vl_text"
VL_IMAGE_SPAN = "vl_image"
DT_XATTN_PAIRS = ((SA_SPAN, VL_TEXT_SPAN), (SA_SPAN, VL_IMAGE_SPAN))


def _read(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    return json.loads(path.read_text())


def _known(cls, raw: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


@dataclass
class Gr00tN17TextConfig:
    """The Qwen3 LLM, cut to its first ``num_hidden_layers`` (= ``select_layer``)."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    mrope_section: tuple[int, ...]
    mrope_interleaved: bool
    image_token_id: int
    vision_start_token_id: int


@dataclass
class Gr00tN17StackConfig:
    """One of the action head's transformer stacks: the DT
    (``diffusion_model_cfg``) or the VL bridge (``vl_self_attention_cfg``)."""

    num_layers: int
    num_attention_heads: int
    attention_head_dim: int
    output_dim: int | None = None  # DT only

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim


@dataclass
class Gr00tN17Config:
    text: Gr00tN17TextConfig
    vision: Qwen3_5VisionConfig  # DeepStack off; GR00T taps the blocks itself
    dit: Gr00tN17StackConfig
    bridge: Gr00tN17StackConfig

    deepstack_visual_indexes: tuple[int, ...]
    backbone_embedding_dim: int
    input_embedding_dim: int = 1536
    hidden_size: int = 1024
    action_horizon: int = 40
    max_action_dim: int = 132
    max_state_dim: int = 132
    max_num_embodiments: int = 32
    state_history_length: int = 1
    max_seq_len: int = 1024
    num_inference_timesteps: int = 4
    num_timestep_buckets: int = 1000
    attend_text_every_n_blocks: int = 2

    @classmethod
    def from_hf(cls, path: str | Path, backbone_path: str | Path) -> "Gr00tN17Config":
        raw = _read(path)
        backbone = _read(backbone_path)
        tc = backbone["text_config"]
        _check_supported(raw, tc)
        text = Gr00tN17TextConfig(**_known(Gr00tN17TextConfig, {
            **tc,
            **tc["rope_scaling"],
            "mrope_section": tuple(tc["rope_scaling"]["mrope_section"]),
            "num_hidden_layers": raw["select_layer"],
            "image_token_id": backbone["image_token_id"],
            "vision_start_token_id": backbone["vision_start_token_id"],
        }))
        vision = Qwen3_5VisionConfig.from_hf(backbone_path)
        deepstack = tuple(vision.deepstack_visual_indexes)
        vision = dataclasses.replace(vision, deepstack_visual_indexes=())
        return cls(
            text=text,
            vision=vision,
            dit=Gr00tN17StackConfig(**_known(Gr00tN17StackConfig, raw["diffusion_model_cfg"])),
            bridge=Gr00tN17StackConfig(**_known(Gr00tN17StackConfig, raw["vl_self_attention_cfg"])),
            deepstack_visual_indexes=deepstack,
            **_known(cls, raw),
        )


# The one reference variant this port implements; any other value is a
# different forward, so it fails here rather than running the wrong one.
_REQUIRED = {
    "use_alternate_vl_dit": True,
    "use_vlln": True,
    "use_vl_self_attention": True,
    "add_pos_embed": True,
    "apply_sincos_state_encoding": False,
    "use_future_tokens": False,
    "reproject_vision": False,
}
_REQUIRED_DIT = {"norm_type": "ada_norm", "interleave_self_attention": True, "positional_embeddings": None}


def _check_supported(raw: dict, text_config: dict) -> None:
    bad = {k: raw.get(k) for k, v in _REQUIRED.items() if raw.get(k, v) != v}
    if not 0 < raw["select_layer"] <= text_config["num_hidden_layers"]:
        bad["select_layer"] = raw["select_layer"]
    # the reference swaps a 0-layer bridge for nn.Identity; this port has no such path
    if raw["vl_self_attention_cfg"]["num_layers"] < 1:
        bad["vl_self_attention_cfg.num_layers"] = raw["vl_self_attention_cfg"]["num_layers"]
    dit = raw["diffusion_model_cfg"]
    bad |= {f"diffusion_model_cfg.{k}": dit.get(k) for k, v in _REQUIRED_DIT.items() if dit.get(k, v) != v}
    if raw["vl_self_attention_cfg"].get("positional_embeddings") is not None:
        bad["vl_self_attention_cfg.positional_embeddings"] = raw["vl_self_attention_cfg"]["positional_embeddings"]
    if bad:
        raise ValueError(f"unsupported GR00T N1.7 variant: {bad}")
