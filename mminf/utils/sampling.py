"""Generic token sampling utilities.

Uses FlashInfer's fused top-k/top-p sampling kernel for GPU efficiency
and CUDA graph compatibility. Model-agnostic — any AR model returns logits,
this module selects the next token.

Supports per-request sampling parameters (different temperature/top_k/top_p
for each request in a batch) via tensor parameters.

CUDA graph compatible: no Python control flow branches — uses masking
to handle greedy vs sampled requests in the same batch.

Usage:
    from mminf.utils.sampling import sample_tokens
    tokens = sample_tokens(logits, temperature=0.7, top_p=0.9)
"""

import torch


@torch.compiler.disable
def _apply_repetition_penalty(
    logits: torch.Tensor,
    seen_token_ids: list[int] | torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply vLLM-style sign-aware repetition penalty in-place.

    For each seen token id: divide logit by penalty if positive,
    multiply by penalty if negative. Applied before temperature scaling.
    """
    if isinstance(seen_token_ids, list):
        if not seen_token_ids:
            return logits
        ids = torch.tensor(seen_token_ids, device=logits.device, dtype=torch.long)
    else:
        ids = seen_token_ids
        if ids.numel() == 0:
            return logits

    # logits shape: [batch_size, vocab_size] — typically [1, V] here
    selected = logits[:, ids]
    penalized = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits[:, ids] = penalized
    return logits


@torch.compiler.disable
def sample_tokens(
    logits: torch.Tensor,
    temperature: float | torch.Tensor = 0.6,
    top_k: int | torch.Tensor = 0,
    top_p: float | torch.Tensor = 1.0,
    repetition_penalty: float = 1.0,
    seen_token_ids: list[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample tokens from logits with temperature, top-k, top-p, and repetition penalty.

    CUDA-graph safe: no Python if/else on tensor values. Uses masking
    to blend greedy argmax with FlashInfer sampled results.

    Args:
        logits: [batch_size, vocab_size] raw logits from lm_head.
        temperature: Scalar or per-request tensor [batch_size] or [batch_size, 1].
            0 = greedy (argmax) for that request. >0 = scaled sampling.
        top_k: Scalar or per-request tensor [batch_size].
            0 = disabled (uses vocab_size).
        top_p: Scalar or per-request tensor [batch_size].
            1.0 = disabled.
        repetition_penalty: vLLM-style sign-aware penalty (1.0 = disabled).
        seen_token_ids: Token IDs to penalize (prompt + previously generated).

    Returns:
        tokens: [batch_size] sampled token IDs.
    """
    batch_size, vocab_size = logits.shape

    # Step 0: Repetition penalty (before temperature scaling)
    if repetition_penalty != 1.0 and seen_token_ids is not None:
        logits = _apply_repetition_penalty(logits, seen_token_ids, repetition_penalty)

    # Normalize params to tensors [batch_size] for uniform handling
    temperature = _to_tensor(temperature, batch_size, logits.device)
    top_k = _to_tensor(top_k, batch_size, logits.device, dtype=torch.int32)
    top_p = _to_tensor(top_p, batch_size, logits.device)

    # Greedy result (always computed — cheap relative to attention/MLP)
    greedy_tokens = torch.argmax(logits, dim=-1)
    greedy_mask = (temperature == 0).squeeze(-1)  # [batch_size]

    # For greedy requests, set temperature=1 to avoid division by zero
    # (their results will be masked out by torch.where at the end)
    safe_temperature = temperature.masked_fill(temperature == 0, 1.0).unsqueeze(-1)
    scaled_logits = logits / safe_temperature

    # Disabled top_k (0) → use vocab_size (keep all)
    safe_top_k = top_k.masked_fill(top_k == 0, vocab_size)

    # FlashInfer fused sampling kernel (return arity varies by version)
    import flashinfer
    result = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        scaled_logits, safe_top_k, top_p, filter_apply_order="joint",
    )
    sampled_tokens = result[0] if isinstance(result, tuple) else result

    # Blend: greedy where temperature==0, sampled otherwise
    return torch.where(greedy_mask, greedy_tokens, sampled_tokens)


def _to_tensor(
    value: float | int | torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert scalar or tensor to [batch_size] tensor."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype).reshape(-1)
    return torch.full((batch_size,), value, device=device, dtype=dtype)
