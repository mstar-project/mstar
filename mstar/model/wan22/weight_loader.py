"""Weight loading for the native Wan2.2 DiT (mstar loader pattern).

``build_wan22_dit`` constructs ``Wan22DiT`` on meta, casts it to the serving dtypes
while still on meta (so ``to_empty`` allocates storage in the final dtypes), moves
it to the device, then streams the safetensors shards through ``load_weights_into``
with a name remapper.

The checkpoint is fp32 throughout. The copy applies the same round-to-nearest bf16
downcast that a ``torch_dtype=bfloat16`` load would, and copies the fp32 islands
exactly, so the loaded bits match the diffusers module parameter for parameter.

Key remap (checkpoint -> native), bijective (825 keys on TI2V-5B, 1095 on each A14B expert):

    condition_embedder.{time,text}_embedder.linear_{1,2}  -> .linear_{in,out}
    blocks.N.ffn.net.0.proj / .net.2                      -> .ffn.linear_{in,out}
    blocks.N.attn1 / attn2                                -> .self_attn / .cross_attn
    ...to_{q,k,v} / to_out.0 / norm_{q,k}                 -> {q,k,v}_proj / o_proj / {q,k}_norm
    patch_embedding, proj_out, scale_shift_table, norm2, time_proj: unchanged

Completeness is a hard contract: a checkpoint key that reaches no parameter, or a
parameter no key reached, raises. A silently skipped weight is a wrong-output bug,
not a warning. So is a transformer whose config.json is not the variant's: a
TI2V-5B config pointed at the A14B checkpoint (or the reverse) raises before any
weight loads.

T2V-A14B's two experts load the same way, from ``transformer`` and ``transformer_2``.
"""

import json
import re
from pathlib import Path

import torch

from mstar.engine.resources.lora.checkpoint import LoraFile
from mstar.model.components.diffusion.lora import apply_loras
from mstar.model.loader.base import load_weights_into
from mstar.model.loader.iterators import iter_safetensors_file
from mstar.model.wan22.components.dit import Wan22DiT
from mstar.model.wan22.config import Wan22Config

# Attention-projection renames applied after the attn1/attn2 module renames.
_ATTN_KEY_MAP = {
    ".to_q.": ".q_proj.",
    ".to_k.": ".k_proj.",
    ".to_v.": ".v_proj.",
    ".to_out.0.": ".o_proj.",
    ".norm_q.": ".q_norm.",
    ".norm_k.": ".k_norm.",
}


def remap_checkpoint_key(name: str) -> str:
    """Map one diffusers checkpoint key to the native parameter path."""
    name = name.replace(".linear_1.", ".linear_in.").replace(".linear_2.", ".linear_out.")
    name = name.replace(".ffn.net.0.proj.", ".ffn.linear_in.").replace(".ffn.net.2.", ".ffn.linear_out.")
    name = name.replace(".attn1.", ".self_attn.").replace(".attn2.", ".cross_attn.")
    for src, dst in _ATTN_KEY_MAP.items():
        name = name.replace(src, dst)
    return name


# transformer/config.json key -> the Wan22Config field it must equal
_CONFIG_FIELDS = {
    "num_attention_heads": "num_attention_heads",
    "attention_head_dim": "attention_head_dim",
    "num_layers": "num_layers",
    "ffn_dim": "ffn_dim",
    "in_channels": "in_channels",
    "out_channels": "out_channels",
    "text_dim": "text_dim",
    "freq_dim": "freq_dim",
}


def resolve_transformer_dir(
    model_path_hf: str, cache_dir: str | None = None, subfolder: str = "transformer",
) -> Path:
    """Local directory holding the transformer's safetensors shards. A local
    checkpoint path is used as-is; a hub id resolves through the HF cache
    (the Wan2.2 snapshot already lives there — same resolution the wrapped
    components' ``from_pretrained`` used)."""
    local = Path(model_path_hf)
    if local.is_dir():
        return local / subfolder
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        model_path_hf, allow_patterns=[f"{subfolder}/*"], cache_dir=cache_dir
    )
    return Path(snapshot) / subfolder


def check_transformer_config(config: Wan22Config, transformer_dir: Path) -> None:
    """Raise if the checkpoint's transformer config.json disagrees with ``config``
    on a dimension the native DiT is built from (a variant pointed at another
    variant's checkpoint). A checkpoint without a config.json is not checked."""
    path = transformer_dir / "config.json"
    if not path.exists():
        return
    with open(path) as f:
        checkpoint = json.load(f)
    drift = {
        key: (checkpoint[key], getattr(config, field))
        for key, field in _CONFIG_FIELDS.items()
        if key in checkpoint and checkpoint[key] != getattr(config, field)
    }
    if drift:
        raise ValueError(
            f"Wan2.2 variant {config.variant!r} does not match the checkpoint at {transformer_dir}: "
            + ", ".join(f"{key}={ckpt} (variant has {ours})" for key, (ckpt, ours) in drift.items())
            + ". Set model_kwargs variant to the checkpoint's, or model_path_hf to the variant's checkpoint."
        )


def _iter_transformer_shards(transformer_dir: Path, device: torch.device | str):
    """Yield ``(key, tensor)`` from the diffusers-named safetensors shards.

    The generic ``iter_safetensors_shards`` expects transformers' index name
    (``model.safetensors.index.json``); diffusers models ship
    ``diffusion_pytorch_model.*`` — same format, different basename — so the
    index walk lives here instead of widening the shared iterator's API.
    """
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        for shard in sorted(set(index["weight_map"].values())):
            yield from iter_safetensors_file(transformer_dir / shard, device=device)
        return
    single = transformer_dir / "diffusion_pytorch_model.safetensors"
    if single.exists():
        yield from iter_safetensors_file(single, device=device)
        return
    raise FileNotFoundError(
        f"No diffusers safetensors checkpoint in {transformer_dir} (looked for "
        "diffusion_pytorch_model.safetensors.index.json and diffusion_pytorch_model.safetensors)"
    )


def build_wan22_dit(
    config: Wan22Config,
    model_path_hf: str,
    device: torch.device | str = "cpu",
    cache_dir: str | None = None,
    subfolder: str = "transformer",
) -> Wan22DiT:
    """Meta-build, materialize on ``device``, and load the checkpoint's
    ``subfolder`` (A14B's low-noise expert is ``transformer_2``) into a
    ready-to-serve (eval-mode) native DiT."""
    transformer_dir = resolve_transformer_dir(model_path_hf, cache_dir, subfolder)
    check_transformer_config(config, transformer_dir)

    with torch.device("meta"):
        dit = Wan22DiT(config)
    dit.cast_serving_dtypes()
    dit.to_empty(device=device)

    params = dict(dit.named_parameters())
    unexpected: list[str] = []

    def remap(name: str) -> str | None:
        mapped = remap_checkpoint_key(name)
        if mapped not in params:
            unexpected.append(name)
            return None
        return mapped

    # load_weights_into directly (not the load_hf_weights wrapper): the
    # wrapper's "rotary_emb" skip fragment would drop such a key BEFORE the
    # remapper's unexpected-key accounting, silently weakening the
    # completeness contract below. This checkpoint derives RoPE at init and
    # ships no such keys; any that appear must land in ``unexpected``.
    loaded = load_weights_into(
        dit, _iter_transformer_shards(transformer_dir, device), name_remapper=remap
    )
    missing = sorted(set(params) - loaded)
    if unexpected or missing:
        raise RuntimeError(
            f"Wan2.2 DiT checkpoint mismatch at {transformer_dir}: "
            f"{len(unexpected)} unexpected checkpoint keys {unexpected[:5]}, "
            f"{len(missing)} unloaded parameters {missing[:5]} — refusing to "
            "serve a partially loaded transformer."
        )
    return dit.fuse_qkv().eval()


# The original Wan layout's module names (Wan-AI's own repos, LightX2V's distill LoRAs) -> diffusers'.
_WAN_LORA_MODULES = {
    "self_attn.q": "attn1.to_q", "self_attn.k": "attn1.to_k", "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "cross_attn.q": "attn2.to_q", "cross_attn.k": "attn2.to_k", "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "ffn.0": "ffn.net.0.proj", "ffn.2": "ffn.net.2",
}
_WAN_LORA_KEY = re.compile(r"^blocks\.(\d+)\.(\w+\.\w+)\.(lora_A\.weight|lora_B\.weight|alpha)$")


def convert_wan_lora_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite original-Wan-layout LoRA keys (``blocks.N.self_attn.q``, ``ffn.0``; the
    ``diffusion_model.`` prefix already stripped) to diffusers module paths. Keys already in
    diffusers naming pass through; anything else is left for the reader to refuse."""
    out = {}
    for key, value in sd.items():
        m = _WAN_LORA_KEY.match(key)
        if m is not None and m.group(2) in _WAN_LORA_MODULES:
            key = f"blocks.{m.group(1)}.{_WAN_LORA_MODULES[m.group(2)]}.{m.group(3)}"
        out[key] = value
    return out


def apply_dit_loras(dit: Wan22DiT, files: list[LoraFile]) -> None:
    """Fold the configured adapters into one DiT's weights (static merge). In place, so the fused q / k / v weight
    the loader left (``Wan22DiT.fuse_qkv``) takes them through the Linears that view it."""
    apply_loras(dit, files, remap_checkpoint_key, convert_keys=convert_wan_lora_keys)
