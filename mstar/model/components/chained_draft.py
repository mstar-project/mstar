"""The chain a drafter that drafts one token a pass runs over a verify block: each pass reads the
last pass's output and draft, so a drafter supplies only its head and how its output becomes a draft."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mstar.engine.resources.draft.drafter import MixedRows


def chain_drafts(
    out: torch.Tensor, k: int,
    step: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    to_draft: Callable[[torch.Tensor], torch.Tensor],
    mixed: "MixedRows | None" = None,
) -> torch.Tensor:
    """``[n, k]`` drafts: the first from ``out`` (each row's carry), then ``k - 1`` passes of
    ``step(hidden, ids, keep)`` over the block's ``n * (k + 1)`` rows, each kept at its pass's row. Over a
    mixed step (``mixed``), the passes run over all its tokens, each row's block at ``mixed.at``."""
    n = out.shape[0]
    draft = to_draft(out)
    drafts = [draft]
    if k == 1:
        return torch.stack(drafts, dim=1)
    if mixed is not None:
        return _chain_mixed(out, draft, k, step, to_draft, mixed, drafts)
    rows = torch.arange(n, device=draft.device) * (k + 1)
    # per row of the block: the hidden vector it reads and its following token; a row past the chain repeats
    # the last real one, and its K/V lands in a slot rewritten before it is read
    hidden = out[:, None, :].expand(n, k + 1, out.shape[-1]).contiguous()
    tokens = draft[:, None].expand(n, k + 1).contiguous()
    for m in range(1, k):
        out = step(hidden.view(n * (k + 1), -1), tokens.view(-1), rows + (m - 1))
        draft = to_draft(out)
        drafts.append(draft)
        if m + 1 < k:
            hidden[:, m:] = out[:, None, :]
            tokens[:, m:] = draft[:, None]
    return torch.stack(drafts, dim=1)


def _chain_mixed(out, draft, k, step, to_draft, mixed, drafts):
    """`chain_drafts` over a mixed step's tokens: each verify row's block places read its carry and draft as
    a verify step's do; a prompt row's places are unread, and its writes rewritten after the target."""
    at, width, n = mixed.at, k + 1, mixed.tokens
    # a prompt or padding row writes to a sink past the step's tokens: its places are another row's or unread
    places = torch.where(mixed.prompt[:, None], n, at)
    hidden = out.new_zeros(n + 1, out.shape[-1]).index_copy(0, places.reshape(-1), out.repeat_interleave(width, 0))
    tokens = draft.new_zeros(n + 1).index_copy(0, places.reshape(-1), draft.repeat_interleave(width))
    for m in range(1, k):
        out = step(hidden[:n], tokens[:n], at[:, m - 1])
        draft = to_draft(out)
        drafts.append(draft)
        if m + 1 < k:
            later = places[:, m:].reshape(-1)
            hidden = hidden.index_copy(0, later, out.repeat_interleave(width - m, 0))
            tokens = tokens.index_copy(0, later, draft.repeat_interleave(width - m))
    return torch.stack(drafts, dim=1)


def mixed_next_ids(mixed: "MixedRows", following: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """``[tokens]``: each token's next id in a mixed step, for a drafter whose state at a row reads it
    (EAGLE-3's observe and prefill in one pass). A verify row's block places read the tokens its verify
    gave (``tokens`` ``[n, k + 1]``); a prompt token its following one (``following``, `DRAFT_FOLLOWING`),
    and its prompt's last (-1 there) the token its row sampled. An empty row (a replay's padding) writes
    nothing: its places are a real row's."""
    n = mixed.tokens
    sink = torch.full_like(mixed.at, n)
    following = following.long()
    last = torch.where(mixed.prompt & ~mixed.empty, mixed.at[:, 0], n)
    sampled = following.new_zeros(n + 1).index_copy(0, last, tokens[:, 0])[:n]
    nxt = torch.where(following < 0, sampled, following)
    places = torch.where(mixed.prompt[:, None], sink, mixed.at).reshape(-1)
    return torch.cat([nxt, nxt.new_zeros(1)]).index_copy(0, places, tokens.reshape(-1))[:n]


def kept_rows(block: torch.Tensor, accepted: torch.Tensor) -> torch.Tensor:
    """Each row's last kept position in the flattened ``[n, k + 1]`` block, where ``observe`` keeps its output."""
    n, width = block.shape
    return torch.arange(n, device=block.device) * width + accepted


def at_last_rows(values: torch.Tensor, at: torch.Tensor, tokens: int) -> torch.Tensor:
    """``[tokens]``: each prompt's ``values`` entry at its last row (`AttentionManager.prompt_last_rows`'s
    ``at``; a padding row's goes to the sink, then dropped), zeros elsewhere."""
    return values.new_zeros(tokens + 1).index_copy(0, at, values)[:tokens]
