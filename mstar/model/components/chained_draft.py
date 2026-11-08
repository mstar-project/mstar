"""The chain a drafter that drafts one token a pass runs over a verify block: each pass reads the
last pass's output and draft, so a drafter supplies only its head and how its output becomes a draft."""
from __future__ import annotations

from collections.abc import Callable

import torch


def chain_drafts(
    out: torch.Tensor, k: int,
    step: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    to_draft: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """``[n, k]`` drafts: the first from ``out`` (each row's carry), then ``k - 1`` passes of
    ``step(hidden, ids, keep)`` over the block's ``n * (k + 1)`` rows, each kept at its pass's row."""
    n = out.shape[0]
    draft = to_draft(out)
    drafts = [draft]
    if k == 1:
        return torch.stack(drafts, dim=1)
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


def kept_rows(block: torch.Tensor, accepted: torch.Tensor) -> torch.Tensor:
    """Each row's last kept position in the flattened ``[n, k + 1]`` block, where ``observe`` keeps its output."""
    n, width = block.shape
    return torch.arange(n, device=block.device) * width + accepted


def at_last_rows(values: torch.Tensor, at: torch.Tensor, tokens: int) -> torch.Tensor:
    """``[tokens]``: each prompt's ``values`` entry at its last row (`AttentionManager.prompt_last_rows`'s
    ``at``; a padding row's goes to the sink, then dropped), zeros elsewhere."""
    return values.new_zeros(tokens + 1).index_copy(0, at, values)[:tokens]
