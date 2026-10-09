"""Streaming checkpoint -> native module loading for LTX-2.5.

Each ``build_*`` constructs its module on the meta device, casts it to bf16 there
(so ``to_empty`` allocates once, in the serving dtype), materializes it on the
device and streams the diffusers / transformers shards in through
``load_native``, which raises unless every checkpoint key reaches a parameter and
every parameter is reached.

Key remaps (checkpoint -> native):

    transformer  <block>.{attn1,audio_attn1}.to_{q,k,v}                 -> .to_qkv  (shards q, k, v)
                 <block>.{attn2,audio_attn2,audio_to_video_attn,
                          video_to_audio_attn}.to_{k,v}               -> .to_kv   (shards 0, 1)
                 *.to_out.0                                            -> *.to_out
                 *.ff.net.0.proj / *.ff.net.2                          -> *.ff.up / *.ff.down
    connectors   {video,audio}_connector.<block>.attn1.to_{q,k,v}      -> .to_qkv
                 to_out.0, ff.net.{0.proj,2}                           -> as above
    text encoder model.language_model.*                                -> * (vision/audio towers dropped)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.diffusion.weight_loading import (
    iter_diffusers_component,
    iter_transformers_component,
    load_native,
    materialize,
)
from mstar.model.loader.base import StackedParamRule
from mstar.model.ltx2_5.components.connectors import LTX2TextConnectors
from mstar.model.ltx2_5.components.gemma4 import Gemma4TextEncoder
from mstar.model.ltx2_5.components.transformer import LTX2DiT
from mstar.model.ltx2_5.config import LTX25Config

logger = logging.getLogger(__name__)

_SELF_ATTNS = ("attn1", "audio_attn1")
_CROSS_ATTNS = ("attn2", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn")


def _self_attn_rules(names) -> list[StackedParamRule]:
    return [
        StackedParamRule(f".{name}.to_qkv", f".{name}.to_{part}", part)
        for name in names for part in ("q", "k", "v")
    ]


def _cross_attn_rules(names) -> list[StackedParamRule]:
    return [
        StackedParamRule(f".{name}.to_kv", f".{name}.to_{part}", shard)
        for name in names for shard, part in enumerate(("k", "v"))
    ]


def remap_ff_and_out(name: str) -> str:
    return (
        name.replace(".to_out.0.", ".to_out.")
        .replace("ff.net.0.proj.", "ff.up.")
        .replace("ff.net.2.", "ff.down.")
    )


def build_transformer(
    config: LTX25Config, snapshot: Path, device, comm_group: CommGroup | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> LTX2DiT:
    with torch.device("meta"):
        module = LTX2DiT(config.transformer, comm_group)
    materialize(module, dtype, device)
    return load_native(
        module, iter_diffusers_component(snapshot / "transformer", device), remap_ff_and_out, "LTX-2.5 transformer",
        stacked_params=_self_attn_rules(_SELF_ATTNS) + _cross_attn_rules(_CROSS_ATTNS),
    )


def build_connectors(config: LTX25Config, snapshot: Path, device, dtype: torch.dtype = torch.bfloat16):
    with torch.device("meta"):
        module = LTX2TextConnectors(config.connectors)
    materialize(module, dtype, device)
    return load_native(
        module, iter_diffusers_component(snapshot / "connectors", device), remap_ff_and_out, "LTX-2.5 connectors",
        stacked_params=_self_attn_rules(("attn1",)),
    )


_GEMMA_PREFIX = "model.language_model."


def build_text_encoder(config: LTX25Config, snapshot: Path, device, dtype: torch.dtype = torch.bfloat16):
    with torch.device("meta"):
        module = Gemma4TextEncoder(config.text)
    materialize(module, dtype, device)

    def skip(name: str) -> bool:
        return not name.startswith(_GEMMA_PREFIX)

    weights = iter_transformers_component(snapshot / "text_encoder", device, skip=skip)
    return load_native(
        module, weights, lambda name: name[len(_GEMMA_PREFIX):], "LTX-2.5 Gemma-4 text encoder",
    )
