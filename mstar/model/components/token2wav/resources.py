"""What a token2wav node keeps per request: its attention caches, and its conv caches, HiFT tails and host fields.

The caches are bounded KV resources, written by hand. The rest is declared state (`token2wav_state`): a plain Value per
region that writes it, so each region's step writes every field it lists, and the host fields the node's own step
writes.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from mstar.engine.resources import NodeResourceSpec
from mstar.engine.resources.kv.bounded import BoundedKVConfig, BoundedKVSpec
from mstar.engine.state import Array, Entry, TensorType, Value
from mstar.model.components.token2wav.flow import PRE_LOOKAHEAD, REL_POS_MAX_LEN, band_fits
from mstar.model.components.token2wav.keys import Token2WavKeys
from mstar.model.components.token2wav.model import (
    DIT_COMPUTE_DTYPE,
    MAX_LAST_TOKENS,
    WINDOW,
    CacheCapacity,
    Token2WavStreaming,
    slot_layout,
)

# frames of the s3 tokenizer per second of voice clip
_TOKENS_PER_SECOND = 25
# the most 25 Hz tokens a window adds to the conformer's caches: a full window less its
# lookahead, or a last window, whose lookahead is padding
_STEP_TOKENS = max(WINDOW - PRE_LOOKAHEAD, MAX_LAST_TOKENS)
# each Value's fields, by the region that writes them: the flow's conv caches, and whether its request has run a
# window (a fresh row starts from its voice's conv caches, not the Value's zeros); HiFT's held-back mel frames and
# excitation; and the samples the next window cross-fades
_VALUE_FIELDS = {"flow": ("enc_cnn", "dit_cnn"), "hift": ("hift_mel", "hift_source"), "tail": ("hift_speech",)}


def token2wav_state(keys: Token2WavKeys, node: str) -> list[Array]:
    """``node``'s declared state: three plain Values of one entry per request, and a Value of its host field."""
    # the conv caches and tails don't grow with the voice
    blocks = slot_layout(CacheCapacity(1), dit_dtype=DIT_COMPUTE_DTYPE)
    names = {"flow": keys.flow, "hift": keys.hift, "tail": keys.tail}
    values = []
    for kind, fields in _VALUE_FIELDS.items():
        tensors = {name: TensorType(blocks[name].shape, dtype=blocks[name].dtype) for name in fields}
        if kind == "flow":
            tensors["started"] = TensorType((), dtype=torch.bool)
        values.append(Value(names[kind], nodes={node}, layers=frozenset({0}), entry=Entry(tensors)))
    # a request whose last window ran; a later chunk of its stream only repeats that window's tail
    values.append(Value(keys.request, nodes={node}, layers=frozenset({0}), entry=Entry({}, host={"ended": bool})))
    return values


def token2wav_kernel_settings(keys: Token2WavKeys, max_slots: int) -> dict[str, dict[str, int]]:
    """Each Value's default slots: ``max_slots``, one more than each cache's (`token2wav_resources`), its sink."""
    return {name: {"max_slots": max_slots} for name in keys.values}


def token2wav_resources(
    keys: Token2WavKeys, node: str, voice_seconds: tuple[float, float], max_slots: int,
    streaming: Token2WavStreaming | None = None,
) -> list[NodeResourceSpec]:
    """The bounded KV caches of ``node``, sized for voices of ``voice_seconds`` (shortest, longest) and retained as
    ``streaming`` (the model's ``Token2Wav.streaming``; MiniCPM-o's by default). Each has ``max_slots - 1``: one fewer
    than each Value (`token2wav_kernel_settings`), which also keeps its sink slot."""
    streaming = streaming or Token2WavStreaming()
    # one token of slack for rounding, each way
    capacity = streaming.capacity(int(voice_seconds[1] * _TOKENS_PER_SECOND) + 1)
    min_tokens = max(0, int(voice_seconds[0] * _TOKENS_PER_SECOND) - 1)
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
        for name, family in streaming.cache_families().items()
    ]
    for spec, (name, family) in zip(caches, streaming.cache_families().items(), strict=True):
        if family.positional:
            _check_positions(name, spec.config.max_retained, family.rate * _STEP_TOKENS, voice_seconds[1], family.rate)
    return caches


def _check_positions(family: str, max_keys: int, step: int, voice_seconds: float, rate: int) -> None:
    """Refuse a voice so long that a step of the ``family`` conformer cache, retaining
    ``max_keys`` and adding ``step``, reads past the flow's relative position table. A
    cache never cut (``Token2WavStreaming.conformer_reply_tokens``) reaches it soonest."""
    if band_fits(max_keys, step):
        return
    over = -(-(max_keys + step - REL_POS_MAX_LEN) // rate)  # voice tokens too many
    longest = max(0.0, voice_seconds - over / _TOKENS_PER_SECOND)
    raise ValueError(
        f"Token2Wav: a {voice_seconds:.2f} s voice is too long for its {family} conformer cache: a step "
        f"attends {max_keys} retained keys and {step} of its own, past the {REL_POS_MAX_LEN} relative "
        f"positions of the flow's table; keep the deployment's voices to about {longest:.2f} s or less"
    )


def check_token2wav_slots(keys: Token2WavKeys, resources: Mapping, default_slots: int, config_path: str) -> None:
    """Refuse a serving config whose ``resources:`` give a Value anything but one more slot than each cache
    (``token2wav_resources``), or the Values different counts."""

    def slots(key: str, default: int) -> int:
        return int((resources.get(key) or {}).get("max_slots", default))

    values = {key: slots(key, default_slots) for key in keys.values}
    kv = {key: slots(key, default_slots - 1) for key in keys.kv.values()}
    counts = set(values.values())
    if len(counts) > 1 or any(n != next(iter(counts)) - 1 for n in kv.values()):
        raise ValueError(
            f"{config_path}: {', '.join(f'resources.{k}.max_slots ({n})' for k, n in values.items())} must be one "
            f"count, one more than each of {', '.join(f'{k}.max_slots ({n})' for k, n in kv.items())}"
        )
