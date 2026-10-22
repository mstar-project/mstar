"""DSA's selection on the host or in torch: the indexer's scores over a request's keys and each
query row's causal top-k, the CPU path and the reference the kernels are held to."""

import torch


def select_topk_causal(
    scores: torch.Tensor, positions: torch.Tensor, topk: int
) -> torch.Tensor:
    """Per-row top-``topk`` key positions of causally masked ``scores``.

    ``scores`` is ``(T, num_keys)`` with every key past a row's position
    already ``-inf``; the result is ``(T, topk)`` int32, -1 padded where a
    row sees fewer than ``topk`` keys. One stable sort for the whole batch:
    no per-token topk launch, no host sync. Ties break toward the earlier
    key position, so a row's selection does not depend on how the batch
    is packed or on which device ran it (torch.topk's tie order is neither).
    """
    num_tokens, num_keys = scores.shape
    k_eff = min(topk, num_keys)
    order = scores.sort(dim=1, descending=True, stable=True).indices[:, :k_eff]
    # the -inf tail sorts last (stable: by ascending index), so rank >= p_t+1
    # is exactly the invisible remainder
    rank = torch.arange(k_eff, device=scores.device)
    invisible = rank.unsqueeze(0) >= positions.unsqueeze(1) + 1
    selection = torch.full(
        (num_tokens, topk), -1, dtype=torch.int32, device=scores.device)
    selection[:, :k_eff] = order.to(torch.int32).masked_fill(invisible, -1)
    return selection


def index_topk(
    q: torch.Tensor, weights: torch.Tensor, keys: torch.Tensor,
    positions: torch.Tensor, topk: int,
) -> torch.Tensor:
    """Top-``topk`` key positions per query row, ``(T, topk)`` int32, -1 padded.

    ``q`` ``(T, H, D)`` and ``weights`` ``(T, H)`` are the indexer's queries and head weights
    (any scales folded in), ``keys`` ``(S, D)`` a request's keys by position and ``positions``
    ``(T,)`` the rows' own: a row sees keys ``0 .. position``, itself included."""
    # score[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s]), in fp32: per-head ReLU BEFORE the
    # weighted sum; the raw weights get no softmax/sigmoid. The engine's bf16 autocast would run
    # both einsums in bf16.
    with torch.autocast(q.device.type, enabled=False):
        dots = torch.einsum("thd,sd->ths", q.float(), keys.float()).relu()
        scores = torch.einsum("th,ths->ts", weights.float(), dots)
    key_pos = torch.arange(keys.shape[0], device=scores.device)
    scores = scores.masked_fill(key_pos.unsqueeze(0) > positions.unsqueeze(1), float("-inf"))
    return select_topk_causal(scores, positions, topk)
