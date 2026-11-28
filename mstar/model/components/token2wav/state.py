"""The state a token2wav node declares: its attention caches, its per-request slots and its host fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.engine.state import Array, Attend, Bidirectional, Entry, Read, TensorType, Value
from mstar.model.components.token2wav.keys import Token2WavKeys
from mstar.model.components.token2wav.model import (
    CACHE_FAMILIES,
    DIT_COMPUTE_DTYPE,
    SLOT_PARTS,
    CacheCapacity,
    slot_layout,
)

# frames of the s3 tokenizer per second of voice clip
_TOKENS_PER_SECOND = 25


def voice_tokens(voice_seconds: tuple[float, float]) -> range:
    """The prompt lengths of voices of ``voice_seconds`` (shortest, longest), in tokens, with a token of slack for
    rounding each way."""
    shortest, longest = voice_seconds
    return range(max(0, int(shortest * _TOKENS_PER_SECOND) - 1), int(longest * _TOKENS_PER_SECOND) + 2)


def token2wav_state(keys: Token2WavKeys, node: str, voice_seconds: tuple[float, float]) -> list[Array]:
    """``node``'s state for voices of ``voice_seconds``: a Value per slot part, a Value of host fields, and an
    unbounded array per cache family, whose rows begin from their voice's cache."""
    tokens = voice_tokens(voice_seconds)
    blocks = slot_layout(CacheCapacity(tokens[-1]), dit_dtype=DIT_COMPUTE_DTYPE)
    slots = [
        Value(keys.state[part], nodes={node}, layers={0}, entry=Entry({
            name: TensorType(blocks[name].shape, dtype=blocks[name].dtype) for name in names
        }))
        for part, names in SLOT_PARTS.items()
    ]
    request = Value(keys.request, nodes={node}, layers={0}, entry=Entry(
        {}, host={"ended": bool, "last_window": int},
    ))
    caches = []
    for name, family in CACHE_FAMILIES.items():
        heads = family.rows * family.num_heads
        kv = TensorType((heads, family.head_dim), dtype=family.dtype, shard=0)
        caches.append(Array(
            keys.kv[name], nodes={node}, layers=frozenset(range(family.num_layers)), max_len=None,
            entry=Entry({"k": kv, "v": kv}),
            reads={keys.attn[name]: Read(
                mask=Bidirectional(kept_window=True, last_first=family.reverse_step_order), op=Attend(q_heads=heads),
                packed=False,
            )},
            start_lengths=range(family.rate * tokens[0], family.rate * tokens[-1] + 1),
            kept_windows={keys.attn[name]: family.kept_window},
        ))
    return [*slots, request, *caches]


def token2wav_kernel_settings(keys: Token2WavKeys, slots: int) -> dict[str, dict[str, Any]]:
    """A request's slot in each Value, which has ``slots`` with its sink, and in each cache, which has one fewer. The
    float32 encoder caches attend in TF32."""
    return {
        **{key: {"max_slots": slots} for key in keys.state.values()},
        **{key: {"max_slots": slots - 1} for key in keys.kv.values()},
        **{keys.attn[name]: {"ieee": False} for name, f in CACHE_FAMILIES.items() if f.dtype == torch.float32},
    }


def check_token2wav_slots(keys: Token2WavKeys, resources: Mapping, default_slots: int, config_path: str) -> None:
    """Refuse a serving config whose ``resources:`` give the slot Values counts that differ, or anything but one more
    slot than each cache (``token2wav_kernel_settings``)."""

    def slots(key: str, default: int) -> int:
        return int((resources.get(key) or {}).get("max_slots", default))

    state = {key: slots(key, default_slots) for key in keys.state.values()}
    kv = {key: slots(key, default_slots - 1) for key in keys.kv.values()}
    if len(set(state.values())) != 1 or any(n != s - 1 for s in state.values() for n in kv.values()):
        raise ValueError(
            f"{config_path}: resources {', '.join(f'{k}.max_slots ({n})' for k, n in state.items())} must agree and "
            f"be one more than each of {', '.join(f'{k}.max_slots ({n})' for k, n in kv.items())}"
        )
