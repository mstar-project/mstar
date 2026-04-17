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

from dataclasses import asdict, dataclass, field

import torch


@dataclass
class SamplingConfig:
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 1
    repetition_penalty: float = 1


# TODO: add a method for adding prefill tokens to the _seen_token_mask,
# if applying the repetition penalty to tokens from the prompt is desired
@dataclass
class Sampler:
    # per request
    _sampling_config: dict[str, SamplingConfig] = field(default_factory=dict)
    _seen_token_mask: dict[str, torch.Tensor]= field(default_factory=dict)

    def add_request(self, request_id: str):
        self._sampling_config[request_id] = SamplingConfig()
        # lazy init _seen_token_mask, taking vocab size from logits

    def remove_request(self, request_id: str):
        if request_id in self._sampling_config:
            del self._sampling_config[request_id]
        if request_id in self._seen_token_mask:
            del self._seen_token_mask[request_id]

    def set_config(self, request_id: str, **kwargs):
        curr_config = asdict(self._sampling_config[request_id])
        kwargs = {k: arg for k, arg in kwargs.items() if k in curr_config.keys()}
        self._sampling_config[request_id] = SamplingConfig(**{
            **curr_config, **kwargs
        })

    def sample(
        self, request_ids: list[str], logits: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        for rid in request_ids:
            if rid not in self._seen_token_mask:
                self._seen_token_mask[rid] = torch.zeros(
                    logits.shape[1], dtype=torch.bool, device=logits.device
                )
        configs = [
            self._sampling_config[rid] for rid in request_ids
        ]
        temperature = torch.tensor([c.temperature for c in configs], device=logits.device)
        top_k = torch.tensor([c.top_k for c in configs], device=logits.device, dtype=torch.int32)
        top_p = torch.tensor([c.top_p for c in configs], device=logits.device)
        r_pen = torch.tensor([c.repetition_penalty for c in configs], device=logits.device)
        seen_mask = torch.stack(
            [self._seen_token_mask[rid] for rid in request_ids],
            dim=0,
        )
        tokens = sample_tokens(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=r_pen,
            seen_token_mask=seen_mask
        )

        res = {}
        for i, rid in enumerate(request_ids):
            token = tokens[i:i+1]
            res[rid] = token
            self._seen_token_mask[rid][token] = True
        return res


@torch.compiler.disable
def _apply_repetition_penalty(
    logits: torch.Tensor,          # [B, V]
    seen_mask: torch.Tensor,       # [B, V] (bool or 0/1)
    penalty: torch.Tensor,         # [B]
) -> torch.Tensor:
    """
    Apply vLLM-style sign-aware repetition penalty in-place.

    seen_mask[b, v] = 1 if token v has been seen in batch element b.
    penalty is per-batch.
    """

    if seen_mask is None or seen_mask.sum() == 0:
        return logits

    # Expand penalty to [B, 1] so it broadcasts over vocab
    penalty = penalty.view(-1, 1)

    # Only touch seen positions
    selected = logits

    penalized = torch.where(
        selected > 0,
        selected / penalty,
        selected * penalty,
    )

    # Apply only where seen_mask == 1
    logits = torch.where(seen_mask, penalized, logits)

    return logits


@torch.compiler.disable
def sample_tokens(
    logits: torch.Tensor,
    temperature: float | torch.Tensor = 0.6,
    top_k: int | torch.Tensor = 0,
    top_p: float | torch.Tensor = 1.0,
    repetition_penalty: float | torch.Tensor= 1.0,
    seen_token_mask: torch.Tensor | None = None,
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

    repetition_penalty = _to_tensor(repetition_penalty, batch_size, logits.device)

    # Step 0: Repetition penalty (before temperature scaling)
    if (repetition_penalty != 1.0).any() and seen_token_mask is not None:
        logits = _apply_repetition_penalty(logits, seen_token_mask, repetition_penalty)

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
