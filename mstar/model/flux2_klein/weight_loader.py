"""Streaming checkpoint -> native module loading for FLUX.2 [klein] (mstar loader pattern).

Each ``build_*`` constructs the native module on the meta device, casts it to the
serving dtype while still on meta (so ``to_empty`` allocates storage in the final
dtype once), moves it to the device, then streams the diffusers / transformers
safetensors shards through ``load_weights_into`` with a name remapper.

Completeness is a hard contract: a checkpoint key that reaches no parameter, or a
parameter no key reached, raises. The only keys dropped on purpose are listed in
``_IGNORED_KEYS`` (a BatchNorm step counter and, for the 9B, the LM head the
encoder never runs).

Key remaps (checkpoint -> native):

    transformer  time_guidance_embed.timestep_embedder.linear_{1,2} -> time_embed.linear_{in,out}
                 double_stream_modulation_{img,txt}.linear           -> mod_double_{img,txt}
                 single_stream_modulation.linear                     -> mod_single
                 x_embedder / context_embedder                       -> img_in / txt_in
                 transformer_blocks.N.attn.{to_q,to_k,to_v}          -> double_blocks.N.attn.img_qkv   (shards q,k,v)
                 transformer_blocks.N.attn.add_{q,k,v}_proj          -> double_blocks.N.attn.txt_qkv   (shards q,k,v)
                 transformer_blocks.N.attn.norm_{q,k} / norm_added_{q,k} -> ...img_{q,k}_norm / txt_{q,k}_norm
                 transformer_blocks.N.attn.to_out.0 / to_add_out     -> double_blocks.N.attn.img_out / txt_out
                 transformer_blocks.N.{ff,ff_context}                -> double_blocks.N.{img_ff,txt_ff}
                 single_transformer_blocks.N.attn.to_qkv_mlp_proj    -> single_blocks.N.qkv_mlp
                 single_transformer_blocks.N.attn.{norm_q,norm_k,to_out} -> single_blocks.N.{q_norm,k_norm,out}
                 norm_out.linear / proj_out                          -> norm_out_mod / proj_out
    vae          {encoder,decoder}.{down,up}_blocks.N.resnets.M      -> ..._stages.N.resnets.M
                 {encoder,decoder}.conv_norm_out                     -> {encoder,decoder}.norm_out
                 ...downsamplers.0.conv / upsamplers.0.conv          -> ...downsample / upsample
                 mid_block.resnets.{0,1} / attentions.0              -> mid.resnet{1,2} / mid.attn (to_out.0 -> to_out)
                 bn.running_{mean,var}                               -> latent_{mean,var}
    text encoder model.embed_tokens / model.layers.N.*               -> embed_tokens / layers.N.* with
                 self_attn.{q,k,v}_proj -> self_attn.qkv_proj (shards q,k,v); layers beyond the deepest tap dropped
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path

import torch

from mstar.model.components.diffusion.lora import LoraSpec, apply_loras
from mstar.model.components.diffusion.text_encoder import Qwen3HiddenStateEncoder
from mstar.model.flux2_klein.components.transformer import Flux2DiT
from mstar.model.flux2_klein.components.vae import Flux2VAE, remap_flux2_vae_key
from mstar.model.flux2_klein.config import Flux2KleinConfig, Qwen3EncoderConfig
from mstar.model.loader.base import StackedParamRule, load_weights_into
from mstar.model.loader.iterators import iter_safetensors_file

logger = logging.getLogger(__name__)

_QKV_RULES_DIT = [
    StackedParamRule(".attn.img_qkv", ".attn.to_q", "q"),
    StackedParamRule(".attn.img_qkv", ".attn.to_k", "k"),
    StackedParamRule(".attn.img_qkv", ".attn.to_v", "v"),
    StackedParamRule(".attn.txt_qkv", ".attn.add_q_proj", "q"),
    StackedParamRule(".attn.txt_qkv", ".attn.add_k_proj", "k"),
    StackedParamRule(".attn.txt_qkv", ".attn.add_v_proj", "v"),
]
_QKV_RULES_LM = [
    StackedParamRule(".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
    StackedParamRule(".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
    StackedParamRule(".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
]

_TRANSFORMER_MAP = [
    (re.compile(r"^time_guidance_embed\.timestep_embedder\.linear_1\."), "time_embed.linear_in."),
    (re.compile(r"^time_guidance_embed\.timestep_embedder\.linear_2\."), "time_embed.linear_out."),
    (re.compile(r"^time_guidance_embed\.guidance_embedder\.linear_1\."), "guidance_embed.linear_in."),
    (re.compile(r"^time_guidance_embed\.guidance_embedder\.linear_2\."), "guidance_embed.linear_out."),
    (re.compile(r"^double_stream_modulation_img\.linear\."), "mod_double_img."),
    (re.compile(r"^double_stream_modulation_txt\.linear\."), "mod_double_txt."),
    (re.compile(r"^single_stream_modulation\.linear\."), "mod_single."),
    (re.compile(r"^x_embedder\."), "img_in."),
    (re.compile(r"^context_embedder\."), "txt_in."),
    (re.compile(r"^norm_out\.linear\."), "norm_out_mod."),
    (re.compile(r"^transformer_blocks\."), "double_blocks."),
    (re.compile(r"^single_transformer_blocks\."), "single_blocks."),
]
_DOUBLE_ATTN_MAP = {
    ".attn.norm_q.": ".attn.img_q_norm.",
    ".attn.norm_k.": ".attn.img_k_norm.",
    ".attn.norm_added_q.": ".attn.txt_q_norm.",
    ".attn.norm_added_k.": ".attn.txt_k_norm.",
    ".attn.to_out.0.": ".attn.img_out.",
    ".attn.to_add_out.": ".attn.txt_out.",
    ".ff.": ".img_ff.",
    ".ff_context.": ".txt_ff.",
}
_SINGLE_ATTN_MAP = {
    ".attn.to_qkv_mlp_proj.": ".qkv_mlp.",
    ".attn.norm_q.": ".q_norm.",
    ".attn.norm_k.": ".k_norm.",
    ".attn.to_out.": ".out.",
}


def remap_transformer_key(name: str) -> str:
    for pattern, repl in _TRANSFORMER_MAP:
        name, n = pattern.subn(repl, name)
        if n:
            break
    if name.startswith("double_blocks."):
        for src, dst in _DOUBLE_ATTN_MAP.items():
            name = name.replace(src, dst)
    elif name.startswith("single_blocks."):
        for src, dst in _SINGLE_ATTN_MAP.items():
            name = name.replace(src, dst)
    return name


remap_vae_key = remap_flux2_vae_key


def remap_text_encoder_key(name: str) -> str:
    return name[len("model."):] if name.startswith("model.") else name


# Checkpoint keys that intentionally load nothing.
_IGNORED_KEYS = {"bn.num_batches_tracked"}


def _iter_component(
    component_dir: Path, device, index_name: str, single_name: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream the safetensors of one pipeline component (sharded or single file)."""
    index_path = component_dir / index_name
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        for shard in sorted(set(index["weight_map"].values())):
            yield from iter_safetensors_file(component_dir / shard, device=device)
        return
    single = component_dir / single_name
    if single.exists():
        yield from iter_safetensors_file(single, device=device)
        return
    raise FileNotFoundError(f"no safetensors checkpoint in {component_dir}")


def iter_diffusers_component(component_dir: Path, device) -> Iterator[tuple[str, torch.Tensor]]:
    return _iter_component(
        component_dir, device, "diffusion_pytorch_model.safetensors.index.json", "diffusion_pytorch_model.safetensors",
    )


def iter_transformers_component(component_dir: Path, device) -> Iterator[tuple[str, torch.Tensor]]:
    return _iter_component(component_dir, device, "model.safetensors.index.json", "model.safetensors")


def load_native(
    module: torch.nn.Module,
    weights: Iterator[tuple[str, torch.Tensor]],
    remap,
    what: str,
    stacked_params: list[StackedParamRule] | None = None,
    skip=None,
) -> torch.nn.Module:
    """Stream ``weights`` into ``module`` through ``remap`` and enforce the completeness contract."""
    targets = dict(module.named_parameters())
    targets.update(dict(module.named_buffers()))
    unexpected: list[str] = []
    skipped: list[str] = []

    def remapper(name: str) -> str | None:
        if name in _IGNORED_KEYS or (skip is not None and skip(name)):
            skipped.append(name)
            return None
        mapped = remap(name)
        # a stacked rule's source name maps to the fused target; check that instead
        for rule in stacked_params or ():
            if rule.source_suffix in mapped:
                mapped_target = mapped.replace(rule.source_suffix, rule.target_suffix)
                if mapped_target in targets:
                    return mapped
                unexpected.append(name)
                return None
        if mapped not in targets:
            unexpected.append(name)
            return None
        return mapped

    loaded = load_weights_into(module, weights, stacked_params=stacked_params, name_remapper=remapper)
    missing = sorted(set(targets) - loaded)
    if unexpected or missing:
        raise RuntimeError(
            f"{what}: checkpoint/module mismatch — {len(unexpected)} unexpected checkpoint keys "
            f"{unexpected[:5]}, {len(missing)} unloaded parameters/buffers {missing[:5]}; refusing to serve a "
            "partially loaded module."
        )
    if skipped:
        logger.info("%s: skipped %d checkpoint keys by design (%s...)", what, len(skipped), skipped[:2])
    return module.eval()


def _materialize(module: torch.nn.Module, dtype: torch.dtype, device) -> torch.nn.Module:
    module.to(dtype)  # on meta: storage is allocated directly in the serving dtype below
    return module.to_empty(device=device)


def build_transformer(config: Flux2KleinConfig, snapshot: Path, device, dtype=torch.bfloat16) -> Flux2DiT:
    with torch.device("meta"):
        dit = Flux2DiT(config.transformer)
    _materialize(dit, dtype, device)
    return load_native(
        dit, iter_diffusers_component(snapshot / "transformer", device), remap_transformer_key,
        "FLUX.2 transformer", stacked_params=_QKV_RULES_DIT,
    )


def build_vae(config: Flux2KleinConfig, snapshot: Path, device, dtype=torch.bfloat16) -> Flux2VAE:
    with torch.device("meta"):
        vae = Flux2VAE(config.vae)
    _materialize(vae, dtype, device)
    return load_native(vae, iter_diffusers_component(snapshot / "vae", device), remap_vae_key, "FLUX.2 VAE")


def build_text_encoder(
    text_config: Qwen3EncoderConfig, snapshot: Path, device, dtype=torch.bfloat16,
) -> Qwen3HiddenStateEncoder:
    with torch.device("meta"):
        encoder = make_text_encoder(text_config)
    _materialize(encoder, dtype, device)
    return load_native(
        encoder, iter_transformers_component(snapshot / "text_encoder", device), remap_text_encoder_key,
        "Qwen3 text encoder", stacked_params=_QKV_RULES_LM, skip=text_encoder_skip(text_config),
    )


def text_encoder_skip(text_config: Qwen3EncoderConfig):
    """Predicate for the LM checkpoint keys the encoder never runs: the LM head, the
    final norm, and every decoder layer past the deepest tapped one."""
    needed = text_config.num_layers_needed
    layer_re = re.compile(r"^model\.layers\.(\d+)\.")

    def skip(name: str) -> bool:
        if name.startswith("lm_head.") or name == "model.norm.weight":
            return True
        m = layer_re.match(name)
        return m is not None and int(m.group(1)) >= needed

    return skip


def make_text_encoder(text_config: Qwen3EncoderConfig) -> Qwen3HiddenStateEncoder:
    return Qwen3HiddenStateEncoder(
        vocab_size=text_config.vocab_size,
        hidden_size=text_config.hidden_size,
        intermediate_size=text_config.intermediate_size,
        num_heads=text_config.num_attention_heads,
        num_kv_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        rms_norm_eps=text_config.rms_norm_eps,
        rope_theta=text_config.rope_theta,
        tap_layers=text_config.hidden_state_layers,
    )


# --------------------------------------------------------------------------- LoRA
_BFL_EXTRA = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    "final_layer.linear": "proj_out",
    "final_layer.adaLN_modulation.1": "norm_out.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
}
_BFL_DOUBLE = {
    "img_attn.proj": "attn.to_out.0", "txt_attn.proj": "attn.to_add_out",
    "img_mlp.0": "ff.linear_in", "img_mlp.2": "ff.linear_out",
    "txt_mlp.0": "ff_context.linear_in", "txt_mlp.2": "ff_context.linear_out",
}
_BFL_SINGLE = {"linear1": "attn.to_qkv_mlp_proj", "linear2": "attn.to_out"}
_BFL_QKV = {"img_attn.qkv": ("to_q", "to_k", "to_v"), "txt_attn.qkv": ("add_q_proj", "add_k_proj", "add_v_proj")}
_BLOCK_KEY = re.compile(r"^(double|single)_blocks\.(\d+)\.(.+)\.lora_([AB])\.weight$")


def convert_bfl_lora_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite FLUX.2's native (BFL / ai-toolkit) LoRA keys to diffusers module paths; the
    fused ``qkv`` adapters split their B matrix into q/k/v (A is shared). Keys already in
    diffusers naming pass through."""
    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        m = _BLOCK_KEY.match(key)
        if m is None:
            for src, dst in _BFL_EXTRA.items():
                if key.startswith(src + "."):
                    key = dst + key[len(src):]
                    break
            out[key] = value
            continue
        kind, idx, inner, ab = m.groups()
        if kind == "single":
            out[f"single_transformer_blocks.{idx}.{_BFL_SINGLE[inner]}.lora_{ab}.weight"] = value
            continue
        block = f"transformer_blocks.{idx}"
        if inner in _BFL_QKV:
            names = _BFL_QKV[inner]
            parts = [value] * 3 if ab == "A" else list(value.chunk(3, dim=0))
            for name, part in zip(names, parts, strict=True):
                out[f"{block}.attn.{name}.lora_{ab}.weight"] = part
        elif inner in _BFL_DOUBLE:
            out[f"{block}.{_BFL_DOUBLE[inner]}.lora_{ab}.weight"] = value
        else:
            raise ValueError(f"unsupported FLUX.2 LoRA key {key!r}")
    return out


def apply_transformer_loras(dit: Flux2DiT, specs: list[LoraSpec]) -> None:
    """Fold the configured adapters into the transformer weights (static merge)."""
    apply_loras(dit, specs, remap_transformer_key, _QKV_RULES_DIT, convert_keys=convert_bfl_lora_keys)
