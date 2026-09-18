"""Whisper timestamp decoding rules as a per-request state that travels with
the token.

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

Which rule applies depends on the tokens generated so far. The worker
launches the next decode step before it has read the previous step's token
on the host, so that history cannot live in Python: the four integers that
summarize it (``[active, phase, last_ts, ceil]``) are a tensor row that is
routed along with the sampled token, updated by :meth:`TimestampRules.advance`
inside the (captured) forward, and turned into masks on the ``(bs, vocab)``
logits by :meth:`TimestampRules.apply` with no data-dependent control flow.
:func:`rule_state` builds the same row from a token history on the host —
the prefill's first row, and the tests' oracle for ``advance``.
"""
from __future__ import annotations

import torch

from mstar.model.whisper.config import WhisperModelConfig

# state vector layout: [active, phase, last_ts, ceil]
STATE_SIZE = 4
# phases
FIRST_TOKEN = 0        # nothing generated yet: force a timestamp <= max initial
AFTER_SEGMENT_END = 1  # last token was a timestamp that closed a segment: no text
AFTER_PAIR = 2         # two timestamps in a row (or the very first one): text only
IN_TEXT = 3            # last token was text: anything monotonic


def inactive_state() -> list[int]:
    return [0, IN_TEXT, 0, 0]


def rule_state(generated: list[int], config: WhisperModelConfig) -> list[int]:
    """The rule for the next token, from what the request generated since
    its prompt. ``last_ts`` is the last timestamp token (``timestamp_begin -
    1`` when none yet); ``ceil`` the exclusive timestamp bound."""
    tb = config.timestamp_begin
    if not generated:
        ceil = tb + config.max_initial_timestamp_index + 1
        return [1, FIRST_TOKEN, tb - 1, min(ceil, config.vocab_size)]
    is_ts = [t >= tb for t in generated]
    last_was_ts = is_ts[-1]
    penultimate_was_ts = len(generated) < 2 or is_ts[-2]
    if last_was_ts and penultimate_was_ts:
        phase = AFTER_PAIR
    elif last_was_ts:
        phase = AFTER_SEGMENT_END
    else:
        phase = IN_TEXT
    timestamps = [t for t, ts in zip(generated, is_ts, strict=True) if ts]
    last_ts = timestamps[-1] if timestamps else tb - 1
    return [1, phase, last_ts, config.vocab_size]


class TimestampRules:
    """Vectorized rule application and state transition."""

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
        last_ts = state[:, 2:3]
        ceil = state[:, 3:4]
        # the pair may repeat the segment end; otherwise never emit an
        # earlier (or the same) timestamp again
        floor = torch.where(phase == AFTER_SEGMENT_END, last_ts, last_ts + 1)

        is_ts = ids >= self.timestamp_begin
        is_text = ids < self.timestamp_begin
        mask = torch.zeros_like(logits, dtype=torch.bool)
        # never emit <|notimestamps|> once timestamps are on
        mask |= ids == self.no_timestamps_token_id
        # first token: a timestamp no later than the initial bound
        mask |= (phase == FIRST_TOKEN) & (is_text | (ids >= ceil))
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

    def advance(self, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """The state after sampling ``tokens: (bs,)``; inactive rows are
        returned unchanged. Pure tensor ops, so it runs inside the graph."""
        tokens = tokens.reshape(-1).to(state.dtype)
        active = state[:, 0].bool()
        phase = state[:, 1]
        is_ts = tokens >= self.timestamp_begin
        # after the first token, a segment end or a pair, the last token was
        # (or counts as) a timestamp: one more makes a pair
        last_was_ts = (phase == FIRST_TOKEN) | (phase == AFTER_SEGMENT_END) | (phase == AFTER_PAIR)
        next_phase = torch.where(
            is_ts,
            torch.where(last_was_ts, torch.full_like(phase, AFTER_PAIR), torch.full_like(phase, AFTER_SEGMENT_END)),
            torch.full_like(phase, IN_TEXT),
        )
        next_last_ts = torch.where(is_ts, tokens, state[:, 2])
        next_ceil = torch.full_like(phase, self.vocab_size)
        advanced = torch.stack([state[:, 0], next_phase, next_last_ts, next_ceil], dim=1)
        return torch.where(active[:, None], advanced, state)
