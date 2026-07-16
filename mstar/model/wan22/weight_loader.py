"""Weight loading for the native Wan2.2 DiT (mstar loader pattern).

``build_wan22_dit`` constructs ``Wan22DiT`` on meta, casts it to the serving dtypes
while still on meta (so ``to_empty`` allocates storage in the final dtypes), moves
it to the device, then streams the safetensors shards through ``load_weights_into``
with a name remapper.

The checkpoint is fp32 throughout. The copy applies the same round-to-nearest bf16
downcast that a ``torch_dtype=bfloat16`` load would, and copies the fp32 islands
exactly, so the loaded bits match the diffusers module parameter for parameter.

Key remap (checkpoint -> native), 825 keys, bijective:

    condition_embedder.{time,text}_embedder.linear_{1,2}  -> .linear_{in,out}
    blocks.N.ffn.net.0.proj / .net.2                      -> .ffn.linear_{in,out}
    blocks.N.attn1 / attn2                                -> .self_attn / .cross_attn
    ...to_{q,k,v} / to_out.0 / norm_{q,k}                 -> {q,k,v}_proj / o_proj / {q,k}_norm
    patch_embedding, proj_out, scale_shift_table, norm2, time_proj: unchanged

Completeness is a hard contract: a checkpoint key that reaches no parameter, or a
parameter no key reached, raises. A silently skipped weight is a wrong-output bug,
not a warning.
"""

import json
from pathlib import Path

import torch

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


def resolve_transformer_dir(model_path_hf: str, cache_dir: str | None = None) -> Path:
    """Local directory holding the transformer's safetensors shards. A local
    checkpoint path is used as-is; a hub id resolves through the HF cache
    (the Wan2.2 snapshot already lives there — same resolution the wrapped
    components' ``from_pretrained`` used)."""
    local = Path(model_path_hf)
    if local.is_dir():
        return local / "transformer"
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        model_path_hf, allow_patterns=["transformer/*"], cache_dir=cache_dir
    )
    return Path(snapshot) / "transformer"


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
) -> Wan22DiT:
    """Meta-build, materialize on ``device``, and load the checkpoint into a
    ready-to-serve (eval-mode) native DiT."""
    transformer_dir = resolve_transformer_dir(model_path_hf, cache_dir)

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
    return dit.eval()
