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
from mstar.model.components.token2wav.flow import PRE_LOOKAHEAD, REL_POS_MAX_LEN, band_fits
from mstar.model.components.token2wav.keys import Token2WavKeys
from mstar.model.components.token2wav.model import (
    DIT_COMPUTE_DTYPE,
    MAX_LAST_TOKENS,
    WINDOW,
    Token2WavStreaming,
    slot_layout,
)

# frames of the s3 tokenizer per second of voice clip
_TOKENS_PER_SECOND = 25
# the most 25 Hz tokens a window adds to the conformer's caches: a full window less its
# lookahead, or a last window, whose lookahead is padding
_STEP_TOKENS = max(WINDOW - PRE_LOOKAHEAD, MAX_LAST_TOKENS)


def token2wav_resources(
    keys: Token2WavKeys, node: str, voice_seconds: tuple[float, float], max_slots: int,
    streaming: Token2WavStreaming | None = None,
) -> list[NodeResourceSpec]:
    """The state pool and the bounded KV caches of ``node``, sized for voices of
    ``voice_seconds`` (shortest, longest) and retained as ``streaming`` (the model's
    ``Token2Wav.streaming``; MiniCPM-o's by default) says. The state pool has
    ``max_slots``; each cache one fewer, since the state pool also keeps its sink slot."""
    streaming = streaming or Token2WavStreaming()
    # one token of slack for rounding, each way
    capacity = streaming.capacity(int(voice_seconds[1] * _TOKENS_PER_SECOND) + 1)
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
        for name, family in streaming.cache_families().items()
    ]
    for spec, (name, family) in zip(caches, streaming.cache_families().items(), strict=True):
        if family.positional:
            _check_positions(name, spec.config.max_retained, family.rate * _STEP_TOKENS, voice_seconds[1], family.rate)
    return [state, *caches]


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
