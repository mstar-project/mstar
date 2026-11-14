"""What a token2wav node keeps per request, as declared state: its attention caches (`token2wav_caches`), an array per
family whose read keeps a sink and a window of a stream starting with the voice's own caches, and its conv caches,
HiFT tails and host fields (`token2wav_state`), a plain Value per region that writes it, so each region's step writes
every field it lists, and the host fields the node's own step writes.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from mstar.engine.state import Array, Attend, Entry, Read, Source, TensorType, Value
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
    """Each Value's default slots, ``max_slots``, and each cache's, one fewer: a Value also keeps its sink."""
    return {
        **{name: {"max_slots": max_slots} for name in keys.values},
        **{name: {"max_slots": max_slots - 1} for name in keys.kv.values()},
    }


def token2wav_caches(
    keys: Token2WavKeys, node: str, voice_seconds: tuple[float, float], reply_tokens: int,
    streaming: Token2WavStreaming | None = None,
) -> list[Array]:
    """The attention caches of ``node`` (`cache_arrays`), for voices of ``voice_seconds`` (shortest, longest) and
    replies of up to ``reply_tokens`` codes; retained as ``streaming`` (the model's ``Token2Wav.streaming``; MiniCPM-o's
    by default) says. Refuses a voice too long for the conformer's position table."""
    streaming = streaming or Token2WavStreaming()
    # one token of slack for rounding, each way
    tokens = (max(0, int(voice_seconds[0] * _TOKENS_PER_SECOND) - 1), int(voice_seconds[1] * _TOKENS_PER_SECOND) + 1)
    arrays = cache_arrays(keys, node, tokens, reply_tokens, streaming)
    for array, (name, family) in zip(arrays, streaming.cache_families().items(), strict=True):
        if family.positional:
            _check_positions(name, family.max_keys(array.source.max_len), family.rate * _STEP_TOKENS,
                             voice_seconds[1], family.rate)
    return arrays


def cache_arrays(
    keys: Token2WavKeys, node: str, voice_tokens: tuple[int, int], reply_tokens: int,
    streaming: Token2WavStreaming | None = None, dit_dtype: torch.dtype = DIT_COMPUTE_DTYPE,
) -> list[Array]:
    """An array per cache family, for voices of ``voice_tokens`` (shortest, longest) codes, each voice's own caches
    the source a row starts from, and replies of up to ``reply_tokens`` codes. The DiT's entries hold its two guidance
    rows (a rows axis) in ``dit_dtype``; the conformer's are float32."""
    streaming = streaming or Token2WavStreaming()
    longest = streaming.capacity(voice_tokens[1]).prompt_tokens
    arrays = []
    for name, family in streaming.cache_families().items():
        heads = family.num_heads
        # the DiT's guidance pair is a rows axis: each row attends its own keys under its own queries
        shape = (heads, family.head_dim) if family.rows == 1 else (family.rows, heads, family.head_dim)
        kv = TensorType(shape, dtype=dit_dtype if name == "dit" else torch.float32)
        source = Source(min_len=family.rate * voice_tokens[0], max_len=family.rate * longest)
        arrays.append(Array(
            keys.kv[name], nodes={node}, layers=frozenset(range(family.num_layers)), entry=Entry({"k": kv, "v": kv}),
            max_len=source.max_len + family.rate * reply_tokens, source=source,
            reads={keys.read(name): Read(
                mask=family.mask, op=Attend(q_heads=heads, distance_bias=family.positional), packed=False,
            )},
        ))
    return arrays


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
    """Refuse a serving config whose ``resources:`` give a Value anything but one more slot than each cache, or the
    Values different counts (``token2wav_kernel_settings``)."""

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
