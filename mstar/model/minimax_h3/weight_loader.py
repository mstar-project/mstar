"""The original-format checkpoint (``FL2VA/``) into the native modules: each ``build_*``
materializes its module and streams the shards in through ``load_native``, which raises
unless every kept key reaches a parameter and every parameter is reached. The
transformer's q/k/v rows are interleaved per head and its ``fc1`` is ``[gate; value]``
(diffusers' ``qkv_layout="interleaved"``); both are unfused into the parallel linears'
shards. The DiT node loads the transformer minus the token refiner, which the text node
loads with ``condition_proj``.

The transformer can instead come from a diffusers-layout directory (``dit_source``: a
distilled release such as FastVideo's FastH3), whose names ``from_diffusers`` maps back.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.bigvgan import fold_weight_norm
from mstar.model.components.diffusion.qwen3.encoder import Qwen3HiddenStateEncoder
from mstar.model.components.diffusion.qwen3.weight_loading import (
    make_text_encoder,
    qkv_rules,
    qwen3_vl_text_skip,
    remap_qwen3_vl_text_key,
)
from mstar.model.components.diffusion.weight_loading import (
    iter_diffusers_component,
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

# diffusers' MiniMaxH3Transformer3DModel names -> FL2VA's, the inverse of diffusers'
# convert_minimax_h3_transformer_checkpoint_to_diffusers: whole prefixes (the first match),
# then the block-level parts. q/k/v map straight to the unfused shards.
_DIFFUSERS_PREFIXES = (
    ("transformer_blocks.", "blocks."),
    ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
    ("time_embedder.linear_1.", "time_embedder.proj_in."),
    ("time_embedder.linear_2.", "time_embedder.proj_out."),
    ("proj_in.", "video_patch_proj."),
    ("audio_proj_in.", "audio_patch_proj."),
    ("context_embedder.", "condition_proj."),
    ("norm_out.norm.", "final_layer.norm."),
    ("norm_out.linear.", "final_layer.adaln_proj.linear."),
    ("proj_out.", "final_layer.video_out."),
    ("audio_proj_out.", "final_layer.audio_out."),
)
# VSA's learned gate on its compression branch (FastH3 was distilled with VSA), read by the ``vsa`` ragged backend
_VSA_GATE = ".attn.to_gate_compress."
_DIFFUSERS_BLOCK_PARTS = (
    (".attn.to_q.", ".attn.q_proj."), (".attn.to_k.", ".attn.k_proj."), (".attn.to_v.", ".attn.v_proj."),
    (".attn.norm_q.", ".attn.q_norm."), (".attn.norm_k.", ".attn.k_norm."), (".attn.to_out.0.", ".attn.out_proj."),
    (".ff.net.0.proj.", ".mlp.fc1."), (".ff.net.2.", ".mlp.fc2."), (_VSA_GATE, ".attn.gate_compress."),
)
_DIFFUSERS_TEXT_PREFIXES = ("token_refiner.", "context_embedder.")


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


def native_name(name: str) -> str:
    """A diffusers ``MiniMaxH3Transformer3DModel`` key under FL2VA's name (``fc1`` still fused)."""
    for diffusers, native in _DIFFUSERS_PREFIXES:
        if name.startswith(diffusers):
            name = native + name[len(diffusers):]
            break
    for diffusers, native in _DIFFUSERS_BLOCK_PARTS:
        name = name.replace(diffusers, native)
    return name


def from_diffusers(weights: Iterator[tuple[str, torch.Tensor]]):
    """A diffusers-layout transformer under the names ``split_fused`` yields: diffusers keeps q/k/v
    apart already, and its SwiGLU input is ``[value; gate]`` where FL2VA's is ``[gate; value]``."""
    for name, tensor in weights:
        name = native_name(name)
        if name.endswith(".mlp.fc1.weight"):
            base = name[: -len("fc1.weight")]
            value, gate = tensor.chunk(2, dim=0)
            yield base + "gate_proj.weight", gate
            yield base + "value_proj.weight", value
        else:
            yield name, tensor


def _check_inv_freq(stored: torch.Tensor, config: H3Config) -> None:
    computed = rope_inv_freq(config.transformer)
    if not torch.equal(stored.float().cpu(), computed):
        raise RuntimeError("the checkpoint's rope.inv_freq differs from the computed table; refusing to guess")


def _transformer_weights(
    config: H3Config, snapshot: Path, device, text: bool, dit_source: Path | None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The transformer's tensors under ``split_fused``'s names: the token refiner and
    ``condition_proj`` (``text``) or the rest, the other part left on disk. FL2VA's, or the
    diffusers-layout directory ``dit_source``'s."""
    if dit_source is not None:
        gate = config.transformer.vsa_gate

        def unread(name: str) -> bool:
            return (_VSA_GATE in name and not gate) or name.startswith(_DIFFUSERS_TEXT_PREFIXES) != text

        return from_diffusers(iter_diffusers_component(Path(dit_source), device, skip=unread))

    def weights():
        # the stored rope table is checked, then dropped
        unread = lambda name: name.startswith(_TEXT_PREFIXES) != text  # noqa: E731
        for name, tensor in iter_transformers_component(
            Path(snapshot) / PARTITION / "transformer", device, skip=unread,
        ):
            if name == "rope.inv_freq":
                _check_inv_freq(tensor, config)
            yield name, tensor

    cfg = config.transformer
    return split_fused(weights(), cfg.num_attention_heads, cfg.attention_head_dim)


def build_transformer(
    config: H3Config, snapshot: Path, device, comm_group: CommGroup | None = None,
    dtype: torch.dtype = torch.bfloat16, dit_source: Path | None = None,
) -> H3DiT:
    with torch.device("meta"):
        dit = H3DiT(config.transformer, comm_group)
    dit.to(dtype)
    dit.keep_fp32()  # on meta: the fp32 modules are allocated in fp32, never rounded
    dit.to_empty(device=device)

    def skip(name: str) -> bool:
        return name.startswith(_TEXT_PREFIXES) or name == "rope.inv_freq"

    return load_native(
        dit, _transformer_weights(config, snapshot, device, False, dit_source), lambda n: n,
        "MiniMax-H3 transformer", stacked_params=_FUSED_RULES, skip=skip,
    )


def build_text_conditioner(config: H3Config, snapshot: Path, device, dtype: torch.dtype = torch.bfloat16,
                           dit_source: Path | None = None):
    with torch.device("meta"):
        module = H3TextConditioner(config.transformer)
    materialize(module, dtype, device)

    def skip(name: str) -> bool:
        return not name.startswith(_TEXT_PREFIXES)

    return load_native(
        module, _transformer_weights(config, snapshot, device, True, dit_source), lambda n: n,
        "MiniMax-H3 text conditioner", stacked_params=_FUSED_RULES, skip=skip,
    )


def build_text_encoder(config: H3Config, snapshot: Path, device, dtype: torch.dtype = torch.bfloat16
                       ) -> Qwen3HiddenStateEncoder:
    """Qwen3-VL-32B's language model through layer 50; the vision tower, the 14 layers
    after the tap, the final norm and the LM head stay on disk."""
    with torch.device("meta"):
        encoder = make_text_encoder(config.text)
    materialize(encoder, dtype, device)
    skip = qwen3_vl_text_skip(config.text)
    return load_native(
        encoder, iter_transformers_component(Path(snapshot) / PARTITION / "text_encoder", device, skip=skip),
        remap_qwen3_vl_text_key, "MiniMax-H3 Qwen3-VL text encoder", stacked_params=qkv_rules(config.text),
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
