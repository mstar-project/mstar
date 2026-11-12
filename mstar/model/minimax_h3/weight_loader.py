"""The original-format checkpoint (``FL2VA/``) into the native modules: each ``build_*``
materializes its module and streams the shards in through ``load_native``, which raises
unless every kept key reaches a parameter and every parameter is reached. The
transformer's q/k/v rows are interleaved per head and its ``fc1`` is ``[gate; value]``
(diffusers' ``qkv_layout="interleaved"``); both are unfused into the parallel linears'
shards. The DiT node loads the transformer minus the token refiner, which the text node
loads with ``condition_proj``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.bigvgan import fold_weight_norm
from mstar.model.components.diffusion.qwen.encoder import QwenHiddenStateEncoder
from mstar.model.components.diffusion.qwen.weight_loading import (
    qkv_rules,
    remap_text_encoder_key,
    text_encoder_skip,
)
from mstar.model.components.diffusion.weight_loading import (
    iter_transformers_component,
    load_native,
    materialize,
)
from mstar.model.components.qwen3_vit import VisionRotaryEmbedding
from mstar.model.components.qwen_vision import QwenVisionConfig, QwenVisionModel
from mstar.model.loader.base import StackedParamRule, load_weights_into
from mstar.model.minimax_h3.components.audio_vae import H3AudioDecoder
from mstar.model.minimax_h3.components.transformer import H3DiT, H3TextConditioner, rope_inv_freq
from mstar.model.minimax_h3.components.video_encoder import H3KeyframeEncoder
from mstar.model.minimax_h3.components.video_vae import H3VideoDecoder
from mstar.model.minimax_h3.config import PARTITION, H3Config

logger = logging.getLogger(__name__)

_FUSED_RULES = [
    StackedParamRule(".attn.qkv_proj", ".attn.q_proj", "q"),
    StackedParamRule(".attn.qkv_proj", ".attn.k_proj", "k"),
    StackedParamRule(".attn.qkv_proj", ".attn.v_proj", "v"),
    StackedParamRule(".mlp.fc1", ".mlp.gate_proj", 0),
    StackedParamRule(".mlp.fc1", ".mlp.value_proj", 1),
]
_TEXT_PREFIXES = ("token_refiner.", "condition_proj.")
_VISION_PREFIX = "model.visual."


def split_fused(weights: Iterator[tuple[str, torch.Tensor]], heads: int, head_dim: int):
    """Unfuse the checkpoint's per-head-interleaved q/k/v rows and its ``[gate; value]``
    feed-forward rows into the shards ``_FUSED_RULES`` route (TP-sliceable)."""
    for name, tensor in weights:
        if name.endswith(".attn.qkv_proj.weight"):
            base = name[: -len("qkv_proj.weight")]
            q, k, v = tensor.unflatten(0, (heads, 3, head_dim)).unbind(1)
            yield base + "q_proj.weight", q.flatten(0, 1)
            yield base + "k_proj.weight", k.flatten(0, 1)
            yield base + "v_proj.weight", v.flatten(0, 1)
        elif name.endswith(".mlp.fc1.weight"):
            base = name[: -len("fc1.weight")]
            gate, value = tensor.chunk(2, dim=0)
            yield base + "gate_proj.weight", gate
            yield base + "value_proj.weight", value
        else:
            yield name, tensor


def _check_inv_freq(stored: torch.Tensor, config: H3Config) -> None:
    computed = rope_inv_freq(config.transformer)
    if not torch.equal(stored.float().cpu(), computed):
        raise RuntimeError("the checkpoint's rope.inv_freq differs from the computed table; refusing to guess")


def build_transformer(
    config: H3Config, snapshot: Path, device, comm_group: CommGroup | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> H3DiT:
    cfg = config.transformer
    with torch.device("meta"):
        dit = H3DiT(cfg, comm_group)
    dit.to(dtype)
    dit.keep_fp32()  # on meta: the fp32 modules are allocated in fp32, never rounded
    dit.to_empty(device=device)

    def skip(name: str) -> bool:
        return name.startswith(_TEXT_PREFIXES) or name == "rope.inv_freq"

    def weights():
        # the refiner's keys stay on disk; the stored rope table is checked, then dropped
        unread = lambda name: name.startswith(_TEXT_PREFIXES)  # noqa: E731
        for name, tensor in iter_transformers_component(
            Path(snapshot) / PARTITION / "transformer", device, skip=unread,
        ):
            if name == "rope.inv_freq":
                _check_inv_freq(tensor, config)
            yield name, tensor

    return load_native(
        dit, split_fused(weights(), cfg.num_attention_heads, cfg.attention_head_dim), lambda n: n,
        "MiniMax-H3 transformer", stacked_params=_FUSED_RULES, skip=skip,
    )


def build_text_conditioner(config: H3Config, snapshot: Path, device, dtype: torch.dtype = torch.bfloat16):
    cfg = config.transformer
    with torch.device("meta"):
        module = H3TextConditioner(cfg)
    materialize(module, dtype, device)

    def skip(name: str) -> bool:
        return not name.startswith(_TEXT_PREFIXES)

    weights = iter_transformers_component(Path(snapshot) / PARTITION / "transformer", device, skip=skip)
    return load_native(
        module, split_fused(weights, cfg.num_attention_heads, cfg.attention_head_dim), lambda n: n,
        "MiniMax-H3 text conditioner", stacked_params=_FUSED_RULES, skip=skip,
    )


def build_text_encoder(config: H3Config, snapshot: Path, device, dtype: torch.dtype = torch.bfloat16
                       ) -> QwenHiddenStateEncoder:
    """Qwen3-VL-32B's language model through layer 50; the vision tower, the 14 layers
    after the tap, the final norm and the LM head stay on disk."""
    with torch.device("meta"):
        encoder = QwenHiddenStateEncoder(config.text)
    materialize(encoder, dtype, device)
    skip = text_encoder_skip(config.text)
    return load_native(
        encoder, iter_transformers_component(Path(snapshot) / PARTITION / "text_encoder", device, skip=skip),
        remap_text_encoder_key, "MiniMax-H3 Qwen3-VL text encoder", stacked_params=qkv_rules(config.text),
        skip=skip,
    )


def build_vision_tower(snapshot: Path, device, dtype: torch.dtype = torch.bfloat16) -> QwenVisionModel:
    """Qwen3-VL-32B's vision tower with its deepstack mergers (FL2VA keyframes), from the
    text encoder's shards (``model.visual.``). Its rotary ``inv_freq`` is derived, fp32 as
    the reference keeps it, so completeness is over the persistent state."""
    source = Path(snapshot) / PARTITION / "text_encoder"
    config = QwenVisionConfig.from_hf(source)
    with torch.device("meta"):
        tower = QwenVisionModel(config)
    materialize(tower, dtype, device)
    tower.rotary_pos_emb.register_buffer(
        "inv_freq", VisionRotaryEmbedding(config).inv_freq.to(device), persistent=False,
    )
    weights = iter_transformers_component(source, device, skip=lambda n: not n.startswith(_VISION_PREFIX))
    loaded = load_weights_into(tower, weights, name_remapper=lambda n: n[len(_VISION_PREFIX):])
    missing = sorted(set(tower.state_dict()) - loaded)
    if missing:
        raise RuntimeError(f"MiniMax-H3 vision tower: {len(missing)} unloaded parameters {missing[:5]}")
    return tower.eval()


def build_video_decoder(config: H3Config, snapshot: Path, device, dtype: torch.dtype = torch.bfloat16
                        ) -> H3VideoDecoder:
    with torch.device("meta"):
        vae = H3VideoDecoder(config.video_vae)
    vae.to(dtype)
    vae.keep_fp32()
    vae.to_empty(device=device)

    def skip(name: str) -> bool:
        return name.startswith(("encoder.", "quant_conv.")) or name == "decoder.mask_token"

    source = Path(snapshot) / PARTITION / "video_vae" / "source"
    return load_native(
        vae, iter_transformers_component(source, device, skip=skip), lambda n: n, "MiniMax-H3 video VAE decoder",
        skip=skip,
    )


def build_keyframe_encoder(config: H3Config, snapshot: Path, device) -> H3KeyframeEncoder:
    """The video VAE's encoder half (FL2VA keyframes), fp32 as the reference pins it."""
    with torch.device("meta"):
        encoder = H3KeyframeEncoder(config.video_encoder, config.video_vae)
    encoder.to_empty(device=device)

    def skip(name: str) -> bool:
        return not name.startswith(("encoder.", "quant_conv."))

    source = Path(snapshot) / PARTITION / "video_vae" / "source"
    return load_native(
        encoder, iter_transformers_component(source, device, skip=skip), lambda n: n, "MiniMax-H3 keyframe encoder",
        skip=skip,
    )


def build_audio_vae(config: H3Config, snapshot: Path, device) -> H3AudioDecoder:
    """The audio VAE's decoder half, fp32 as the reference keeps it. The encoder and the
    posterior heads (``encoder.``, ``pre_block.``, ``mean_proj.``, ``logs_proj.``) stay on
    disk; weight-norm pairs are folded on ``device``, as the reference evaluates them."""
    with torch.device("meta"):
        vae = H3AudioDecoder(config.audio_vae)
    vae.to_empty(device=device)

    def skip(name: str) -> bool:
        return not name.startswith(("dec_in_proj.", "decoder."))

    source = Path(snapshot) / PARTITION / "audio_vae"
    weights = fold_weight_norm(iter_transformers_component(source, device, skip=skip))
    return load_native(vae, weights, lambda n: n, "MiniMax-H3 audio VAE decoder", skip=skip)
