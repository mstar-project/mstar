"""Whisper timestamp decoding rules, split into a host-side state and a
vectorized logits transform.

When a request asks for timestamps (no ``<|notimestamps|>`` in its prompt),
Whisper's decoding is constrained the way openai-whisper / HF's
``WhisperTimeStampLogitsProcessor`` constrain it:

  * the first generated token is a timestamp no later than
    ``max_initial_timestamp_index``;
  * timestamps come in pairs — after ``<|t|>`` that closed a segment only a
    timestamp (>= the same value) or end-of-text may follow, and after the
    pair only text may follow;
  * timestamps never decrease;
  * when the total probability of all timestamp tokens beats the best text
    token, a timestamp is sampled.

The decision "which rule applies" depends on the request's generated
history, which lives on the host; ``rule_state`` distils it into four
integers per request. ``apply_rules`` turns a batch of those integers into
masks on the ``(bs, vocab)`` logits with no data-dependent control flow, so
it runs inside the captured decode graph: the state is just another staged
input row.
"""
from __future__ import annotations

import torch

from mstar.model.whisper.config import WhisperModelConfig

# state vector layout: [active, phase, ts_floor, ts_ceil]
STATE_SIZE = 4
# phases
FIRST_TOKEN = 0        # nothing generated yet: force a timestamp <= max initial
AFTER_SEGMENT_END = 1  # last token was a timestamp that closed a segment: no text
AFTER_PAIR = 2         # two timestamps in a row: text only
IN_TEXT = 3            # last token was text: anything monotonic


def inactive_state() -> list[int]:
    return [0, IN_TEXT, 0, 0]


def rule_state(generated: list[int], config: WhisperModelConfig) -> list[int]:
    """The rule to apply to the next token, from what the request generated
    since its prompt (its ``<|sot|>...`` prompt excluded).

    ``ts_floor`` is the smallest timestamp token still allowed, ``ts_ceil``
    the exclusive bound (``vocab_size`` except on the first token).
    """
    tb = config.timestamp_begin
    is_ts = [t >= tb for t in generated]
    if not generated:
        ceil = tb + config.max_initial_timestamp_index + 1
        return [1, FIRST_TOKEN, tb, min(ceil, config.vocab_size)]
    last_was_ts = is_ts[-1]
    penultimate_was_ts = len(generated) < 2 or is_ts[-2]
    if last_was_ts and penultimate_was_ts:
        phase = AFTER_PAIR
    elif last_was_ts:
        phase = AFTER_SEGMENT_END
    else:
        phase = IN_TEXT
    timestamps = [t for t, ts in zip(generated, is_ts, strict=True) if ts]
    if not timestamps:
        floor = tb
    elif last_was_ts and not penultimate_was_ts:
        floor = timestamps[-1]  # the pair may repeat the segment end
    else:
        floor = timestamps[-1] + 1  # never emit <|0.00|> (or any earlier stamp) again
    return [1, phase, floor, config.vocab_size]


class TimestampRules:
    """Vectorized application of :func:`rule_state` rows to logits."""

    def __init__(self, config: WhisperModelConfig):
        self.timestamp_begin = config.timestamp_begin
        self.eos_token_id = config.eos_token_id
        self.no_timestamps_token_id = config.no_timestamps_token_id
        self.vocab_size = config.vocab_size
        self._ids: torch.Tensor | None = None

    def _token_ids(self, device: torch.device) -> torch.Tensor:
        if self._ids is None or self._ids.device != device:
            self._ids = torch.arange(self.vocab_size, device=device)
        return self._ids

    def apply(self, logits: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """``logits: (bs, vocab)``, ``state: (bs, 4)`` int64 -> masked logits.

        Rows with ``active == 0`` pass through untouched.
        """
        ids = self._token_ids(logits.device)[None, :]
        active = state[:, 0:1].bool()
        phase = state[:, 1:2]
        floor = state[:, 2:3]
        ceil = state[:, 3:4]

        is_ts = ids >= self.timestamp_begin
        is_text = ids < self.timestamp_begin
        mask = torch.zeros_like(logits, dtype=torch.bool)
        # never emit <|notimestamps|> once timestamps are on
        mask |= ids == self.no_timestamps_token_id
        # first token: a timestamp no later than the initial bound
        first = phase == FIRST_TOKEN
        mask |= first & (is_text | (ids >= ceil))
        # a segment just closed: no text (timestamp or end-of-text only)
        mask |= (phase == AFTER_SEGMENT_END) & (ids < self.eos_token_id)
        # a pair just completed: text only
        mask |= (phase == AFTER_PAIR) & is_ts
        # monotonic timestamps
        mask |= is_ts & (ids < floor)
        mask &= active
        masked = logits.masked_fill(mask, float("-inf"))

        # if timestamps as a whole are likelier than any single text token,
        # sample a timestamp (HF's `_detect_timestamp_from_logprob`)
        logprobs = torch.log_softmax(masked.float(), dim=-1)
        ts_logprob = torch.logsumexp(logprobs.masked_fill(is_text, float("-inf")), dim=-1, keepdim=True)
        text_logprob = logprobs.masked_fill(is_ts, float("-inf")).amax(dim=-1, keepdim=True)
        force_ts = active & (ts_logprob > text_logprob)
        return masked.masked_fill(force_ts & is_text, float("-inf"))
