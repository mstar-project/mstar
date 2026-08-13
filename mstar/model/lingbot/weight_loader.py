from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from mstar.model.lingbot.components.text_encoder import LingBotTextEncoderModel
from mstar.model.lingbot.components.transformer import LingBotVideoTransformer3DModel
from mstar.model.lingbot.components.vae_decoder import LingBotWanVaeDecoder
from mstar.model.loader.base import load_weights_into
from mstar.model.loader.iterators import iter_safetensors_file, iter_safetensors_shards


def resolve_transformer_dir(model_path_hf: str, cache_dir: str | None = None) -> Path:
    local = Path(model_path_hf)
    if local.is_dir():
        return local / "transformer"

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        model_path_hf,
        allow_patterns=["transformer/*"],
        cache_dir=cache_dir,
    )
    return Path(snapshot) / "transformer"


def resolve_subfolder_dir(model_path_hf: str, subfolder: str, cache_dir: str | None = None) -> Path:
    local = Path(model_path_hf)
    if local.is_dir():
        return local / subfolder

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        model_path_hf,
        allow_patterns=[f"{subfolder}/*"],
        cache_dir=cache_dir,
    )
    return Path(snapshot) / subfolder


def _iter_transformer_shards(transformer_dir: Path, device: torch.device | str):
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

    raise FileNotFoundError(f"No LingBot transformer safetensors checkpoint found in {transformer_dir}.")


def _iter_diffusers_safetensors(module_dir: Path, device: torch.device | str):
    index_path = module_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        for shard in sorted(set(index["weight_map"].values())):
            yield from iter_safetensors_file(module_dir / shard, device=device)
        return

    single = module_dir / "diffusion_pytorch_model.safetensors"
    if single.exists():
        yield from iter_safetensors_file(single, device=device)
        return

    raise FileNotFoundError(f"No diffusers-style safetensors checkpoint found in {module_dir}.")


def _load_exact(
    module: nn.Module,
    weights,
    source: Path,
    ignore_keys: set[str] | None = None,
    ignore_prefixes: tuple[str, ...] = (),
) -> None:
    params = dict(module.named_parameters())
    unexpected: list[str] = []
    ignored = ignore_keys or set()

    def remap(name: str) -> str | None:
        if name in ignored or any(name.startswith(prefix) for prefix in ignore_prefixes):
            return None
        if name not in params:
            unexpected.append(name)
            return None
        return name

    loaded = load_weights_into(module, weights, name_remapper=remap)
    missing = sorted(set(params) - loaded)
    if unexpected or missing:
        raise RuntimeError(
            f"LingBot checkpoint mismatch at {source}: "
            f"{len(unexpected)} unexpected checkpoint keys {unexpected[:5]}, "
            f"{len(missing)} unloaded parameters {missing[:5]}."
        )


def build_lingbot_transformer(
    model_path_hf: str,
    device: torch.device | str = "cpu",
    cache_dir: str | None = None,
) -> LingBotVideoTransformer3DModel:
    transformer_dir = resolve_transformer_dir(model_path_hf, cache_dir)

    with torch.device("meta"):
        transformer = LingBotVideoTransformer3DModel()
    transformer.to_empty(device=device)
    transformer.to(dtype=torch.bfloat16)

    _load_exact(
        transformer,
        _iter_transformer_shards(transformer_dir, device),
        transformer_dir,
    )
    return transformer.eval()


def build_lingbot_text_encoder(
    model_path_hf: str,
    device: torch.device | str = "cpu",
    cache_dir: str | None = None,
) -> nn.Module:
    text_encoder_dir = resolve_subfolder_dir(model_path_hf, "text_encoder", cache_dir)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        text_encoder_dir,
        trust_remote_code=True,
    )
    with torch.device("meta"):
        text_encoder = LingBotTextEncoderModel(config)
    text_encoder.to_empty(device=device)
    text_encoder.to(dtype=torch.bfloat16)
    text_encoder.reset_non_persistent_buffers(device=device)
    _load_exact(
        text_encoder,
        iter_safetensors_shards(text_encoder_dir, device=device),
        text_encoder_dir,
        ignore_keys={"lm_head.weight"},
        ignore_prefixes=("model.visual.",),
    )
    return text_encoder.eval()


def build_lingbot_vae_decoder(
    model_path_hf: str,
    device: torch.device | str = "cpu",
    cache_dir: str | None = None,
) -> nn.Module:
    vae_dir = resolve_subfolder_dir(model_path_hf, "vae", cache_dir)

    with open(vae_dir / "config.json") as f:
        config = json.load(f)
    with torch.device("meta"):
        vae = LingBotWanVaeDecoder(config)
    vae.to_empty(device=device)
    vae.to(dtype=torch.bfloat16)
    _load_exact(
        vae,
        _iter_diffusers_safetensors(vae_dir, device),
        vae_dir,
        ignore_prefixes=("encoder.", "quant_conv."),
    )
    return vae.eval()
