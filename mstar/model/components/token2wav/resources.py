"""The resources a token2wav node declares: its attention caches and its per-request state."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from mstar.engine.resources import NodeResourceSpec
from mstar.engine.resources.kv.bounded import BoundedKVConfig, BoundedKVSpec
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.model.components.token2wav.keys import Token2WavKeys
from mstar.model.components.token2wav.model import (
    CACHE_FAMILIES,
    DIT_COMPUTE_DTYPE,
    CacheCapacity,
    slot_layout,
)

# frames of the s3 tokenizer per second of voice clip
_TOKENS_PER_SECOND = 25


def token2wav_resources(
    keys: Token2WavKeys, node: str, voice_seconds: tuple[float, float], max_slots: int,
) -> list[NodeResourceSpec]:
    """The state pool and the bounded KV caches of ``node``, sized for voices of
    ``voice_seconds`` (shortest, longest). The state pool has ``max_slots``; each cache
    one fewer, since the state pool also keeps its sink slot."""
    # one token of slack for rounding, each way
    capacity = CacheCapacity(int(voice_seconds[1] * _TOKENS_PER_SECOND) + 1)
    min_tokens = max(0, int(voice_seconds[0] * _TOKENS_PER_SECOND) - 1)
    state = RecurrentStateSpec(
        resource_key=keys.state, nodes={node},
        config=RecurrentStateConfig(
            num_layers=1,
            blocks={
                name: RecurrentBlockConfig(shape=block.shape, dtype=block.dtype)
                for name, block in slot_layout(capacity, dit_dtype=DIT_COMPUTE_DTYPE).items()
            },
            max_slots=max_slots,
        ),
    )
    caches = [
        BoundedKVSpec(
            resource_key=keys.kv[name], nodes={node},
            config=BoundedKVConfig(
                num_layers=family.num_layers,
                num_heads=family.num_heads,
                head_dim=family.head_dim,
                rows_per_request=family.rows,
                max_source_len=family.rate * capacity.prompt_tokens,
                min_source_len=family.rate * min_tokens,
                retention=family.retention,
                max_slots=max_slots - 1,
                reverse_step_order=family.reverse_step_order,
                dtype=DIT_COMPUTE_DTYPE if name == "dit" else torch.float32,
            ),
        )
        for name, family in CACHE_FAMILIES.items()
    ]
    return [state, *caches]


def check_token2wav_slots(keys: Token2WavKeys, resources: Mapping, default_slots: int, config_path: str) -> None:
    """Refuse a serving config whose ``resources:`` give the state pool anything but one
    more slot than each cache (``token2wav_resources``)."""

    def slots(key: str, default: int) -> int:
        return int((resources.get(key) or {}).get("max_slots", default))

    state = slots(keys.state, default_slots)
    kv = {key: slots(key, default_slots - 1) for key in keys.kv.values()}
    if any(n != state - 1 for n in kv.values()):
        raise ValueError(
            f"{config_path}: resources.{keys.state}.max_slots ({state}) must be one more than "
            f"each of {', '.join(f'{k}.max_slots ({n})' for k, n in kv.items())}"
        )
