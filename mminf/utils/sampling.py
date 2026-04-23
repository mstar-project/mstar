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
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=32, num_stages=2),
    ],
    key=["V", "APPLY_PENALTY", "INCLUDE_GREEDY"],
)
@triton.jit
def _fused_sampling_prep_kernel(
    logits_ptr,        # [B, V] input
    temperature_ptr,   # [B]
    penalty_ptr,       # [B] (only read when APPLY_PENALTY=True)
    seen_mask_ptr,     # [B, V] bool (only read when APPLY_PENALTY=True)
    probs_ptr,         # [B, V] float32 output
    V,
    stride_b, stride_v,
    out_stride_b, out_stride_v,
    mask_stride_b, mask_stride_v,
    BLOCK_SIZE: tl.constexpr,
    APPLY_PENALTY: tl.constexpr,
    INCLUDE_GREEDY: tl.constexpr,
):
    """Fused (optional rep penalty) + (logits/temperature) + softmax.

    When INCLUDE_GREEDY is True and a row's temperature == 0, the kernel
    emits a one-hot distribution at the argmax instead of a temperature-scaled
    softmax — so a downstream multinomial sampler deterministically returns
    the argmax token (replaces the separate torch.argmax + torch.where pair).

    Both constexprs specialize at compile time; the unused branches compile out.
    """
    row = tl.program_id(0)
    temp = tl.load(temperature_ptr + row)
    if INCLUDE_GREEDY:
        is_greedy = temp == 0
        # Safe inv_temp so the softmax branch doesn't produce NaN for greedy
        # rows (their output is overwritten by the one-hot anyway).
        inv_temp = tl.where(is_greedy, 1.0, 1.0 / tl.maximum(temp, 1e-30))
    else:
        inv_temp = 1.0 / temp

    if APPLY_PENALTY:
        penalty = tl.load(penalty_ptr + row)

    # Pass 1: scan over V, compute max of raw vals (post-penalty) + argmax.
    # argmax is only used by the greedy one-hot path; still tracked when
    # INCLUDE_GREEDY is True regardless of per-row temp.
    max_raw = -float("inf")
    max_idx = tl.zeros([], dtype=tl.int32)
    for v_start in range(0, V, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=-float("inf"),
        )
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            penalized = tl.where(vals > 0, vals / penalty, vals * penalty)
            vals = tl.where(seen, penalized, vals)
        masked_vals = tl.where(mask, vals, -float("inf"))
        block_max = tl.max(masked_vals)
        if INCLUDE_GREEDY:
            block_argmax = tl.argmax(masked_vals, axis=0)
            is_new = block_max > max_raw
            max_idx = tl.where(is_new, v_start + block_argmax.to(tl.int32), max_idx)
        max_raw = tl.maximum(max_raw, block_max)

    max_scaled = max_raw * inv_temp

    # Pass 2: exp(scaled - max_scaled), accumulate sum
    sum_exp = tl.zeros([], dtype=tl.float32)
    for v_start in range(0, V, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=0.0,
        )
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            penalized = tl.where(vals > 0, vals / penalty, vals * penalty)
            vals = tl.where(seen, penalized, vals)
        scaled = vals * inv_temp
        exp_val = tl.exp(scaled - max_scaled)
        exp_val = tl.where(mask, exp_val, 0.0)
        sum_exp += tl.sum(exp_val)

    # Avoid div-by-zero in the greedy rows (their output is overwritten).
    inv_sum = 1.0 / tl.maximum(sum_exp, 1e-30)

    # Pass 3: write the output — softmax probs for non-greedy rows,
    # one-hot at argmax for greedy rows.
    for v_start in range(0, V, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=0.0,
        )
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            penalized = tl.where(vals > 0, vals / penalty, vals * penalty)
            vals = tl.where(seen, penalized, vals)
        scaled = vals * inv_temp
        softmax_val = tl.exp(scaled - max_scaled) * inv_sum
        if INCLUDE_GREEDY:
            is_max = offs == max_idx
            one_hot = tl.where(is_max, 1.0, 0.0)
            probs = tl.where(is_greedy, one_hot, softmax_val)
        else:
            probs = softmax_val
        tl.store(
            probs_ptr + row * out_stride_b + offs * out_stride_v,
            probs, mask=mask,
        )


def fused_temperature_softmax(
    logits: torch.Tensor,       # [B, V]
    temperature: torch.Tensor,  # [B]
    penalty: torch.Tensor | None = None,    # [B]
    seen_mask: torch.Tensor | None = None,  # [B, V] bool
    include_greedy: bool = False,
) -> torch.Tensor:
    """softmax(apply_penalty(logits) / temperature) fused, returns [B, V] float32.

    When include_greedy=True, rows with temperature == 0 produce a one-hot
    distribution at argmax (equivalent to argmax sampling via multinomial).
    """
    B, V = logits.shape
    probs = torch.empty_like(logits, dtype=torch.float32)
    apply_penalty = penalty is not None and seen_mask is not None
    pen_ptr = penalty if apply_penalty else logits
    mask_ptr = seen_mask if apply_penalty else logits
    mask_stride_b = seen_mask.stride(0) if apply_penalty else 0
    mask_stride_v = seen_mask.stride(1) if apply_penalty else 0
    grid = (B,)
    # BLOCK_SIZE is picked by @triton.autotune (not passed here).
    _fused_sampling_prep_kernel[grid](
        logits, temperature, pen_ptr, mask_ptr, probs,
        V,
        logits.stride(0), logits.stride(1),
        probs.stride(0), probs.stride(1),
        mask_stride_b, mask_stride_v,
        APPLY_PENALTY=apply_penalty,
        INCLUDE_GREEDY=include_greedy,
    )
    return probs


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
    ) -> torch.Tensor:
        """Return the sampled tokens as a single [B] int tensor.

        Callers that want a per-rid mapping can slice `tokens[i:i+1]` using
        the rid order in `request_ids`. We return the raw tensor (instead of
        a dict of views) because constructing the dict adds Python overhead
        the hot path doesn't need.
        """
        configs = [self._sampling_config[rid] for rid in request_ids]
        temperature = torch.tensor([c.temperature for c in configs], device=logits.device)
        top_k = torch.tensor([c.top_k for c in configs], device=logits.device, dtype=torch.int32)
        top_p = torch.tensor([c.top_p for c in configs], device=logits.device)
        r_pen = torch.tensor([c.repetition_penalty for c in configs], device=logits.device)
        any_rep_pen = any(c.repetition_penalty != 1.0 for c in configs)
        any_greedy = any(c.temperature == 0 for c in configs)
        any_top_k_zero = any(c.top_k == 0 for c in configs)
        all_top_k_zero = all(c.top_k == 0 for c in configs)

        # Only materialize the seen-token mask when at least one request has
        # repetition penalty active — otherwise sample_tokens ignores it.
        seen_mask = None
        if any_rep_pen:
            for rid in request_ids:
                if rid not in self._seen_token_mask:
                    self._seen_token_mask[rid] = torch.zeros(
                        logits.shape[1], dtype=torch.bool, device=logits.device
                    )
            seen_mask = torch.stack(
                [self._seen_token_mask[rid] for rid in request_ids], dim=0,
            )

        tokens = sample_tokens(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=r_pen,
            seen_token_mask=seen_mask,
            any_greedy=any_greedy,
            any_top_k_zero=any_top_k_zero,
            all_top_k_zero=all_top_k_zero,
        )

        # TODO: make this scatter async. Currently runs 2 kernels per rid
        # (broadcast-True + index_put) on the default stream, serializing N=bs
        # small launches that add up (~500 µs at bs=8 for Orpheus with
        # repetition_penalty=1.3). Two options to fix:
        #   (a) Shared [max_concurrent, V] buffer with rid→slot mapping; replace
        #       the loop with a single batched `buf[slots, tokens] = True`
        #       scatter — one launch instead of N.
        #   (b) Issue the updates on a side CUDA stream so the main stream
        #       (next prefill/decode) doesn't wait. The next sample() for the
        #       same rid would need to sync, but amortized over a full
        #       generation this is cheap.
        if any_rep_pen:
            for i, rid in enumerate(request_ids):
                self._seen_token_mask[rid][tokens[i:i+1]] = True

        return tokens


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
    any_greedy: bool | None = None,
    any_top_k_zero: bool | None = None,
    all_top_k_zero: bool | None = None,
) -> torch.Tensor:
    """Sample tokens from logits with temperature, top-k, top-p, and repetition penalty.

    Args:
        logits: [batch_size, vocab_size] raw logits from lm_head.
        temperature: Scalar or per-request tensor [batch_size].
            0 = greedy (argmax) for that request. >0 = scaled sampling.
        top_k: Scalar or per-request tensor [batch_size]. 0 = disabled.
        top_p: Scalar or per-request tensor [batch_size]. 1.0 = disabled.
        repetition_penalty: vLLM-style sign-aware penalty (1.0 = disabled).
        seen_token_mask: [batch_size, vocab_size] bool. None = penalty skipped.
        any_greedy: CPU-side hint. When False, skips the argmax/masked_fill/where
            branch entirely. None = unknown → run the full path.
        any_top_k_zero: CPU-side hint. When False, skips the `top_k == 0 → vocab`
            masked_fill. None = unknown → run the full path.

    Returns:
        tokens: [batch_size] sampled token IDs.
    """
    batch_size, vocab_size = logits.shape

    # Normalize params to tensors [batch_size] for uniform handling
    temperature = _to_tensor(temperature, batch_size, logits.device)
    top_k = _to_tensor(top_k, batch_size, logits.device, dtype=torch.int32)
    top_p = _to_tensor(top_p, batch_size, logits.device)
    if seen_token_mask is not None:
        repetition_penalty = _to_tensor(repetition_penalty, batch_size, logits.device)

    # Default to the conservative "unknown → do the work" path.
    run_greedy = True if any_greedy is None else any_greedy
    run_top_k_zero_fix = True if any_top_k_zero is None else any_top_k_zero

    import flashinfer

    # Fast path: top_k is disabled for every request in the batch. One Triton
    # kernel fuses (optional rep-penalty) + (temperature-scaled softmax) +
    # (argmax → one-hot for greedy rows). FlashInfer's sample-from-probs then
    # deterministically picks argmax on one-hot rows, matching greedy semantics.
    if all_top_k_zero is True:
        probs = fused_temperature_softmax(
            logits, temperature,
            penalty=repetition_penalty if seen_token_mask is not None else None,
            seen_mask=seen_token_mask,
            include_greedy=run_greedy,
        )
        torch.cuda.current_stream().synchronize()
        result = flashinfer.sampling.top_p_sampling_from_probs(probs, top_p)
        return result[0] if isinstance(result, tuple) else result

    # Slow path: apply rep-penalty the old way (short-circuit on mask=None to
    # avoid the `(rep != 1.0).any()` CPU sync when penalty is inactive).
    if seen_token_mask is not None and (repetition_penalty != 1.0).any():
        logits = _apply_repetition_penalty(logits, seen_token_mask, repetition_penalty)

    if run_greedy:
        greedy_tokens = torch.argmax(logits, dim=-1)
        greedy_mask = (temperature == 0).squeeze(-1)
        # For greedy requests, set temperature=1 to avoid division by zero
        # (their results are masked out by torch.where at the end).
        safe_temperature = temperature.masked_fill(temperature == 0, 1.0).unsqueeze(-1)
    else:
        safe_temperature = temperature.unsqueeze(-1)

    scaled_logits = logits / safe_temperature

    safe_top_k = top_k.masked_fill(top_k == 0, vocab_size) if run_top_k_zero_fix else top_k

    result = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        scaled_logits, safe_top_k, top_p, filter_apply_order="joint",
    )
    sampled_tokens = result[0] if isinstance(result, tuple) else result

    if run_greedy:
        return torch.where(greedy_mask, greedy_tokens, sampled_tokens)
    return sampled_tokens


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


# ---------------------------------------------------------------------------
# Graph-safe depth sampler
# ---------------------------------------------------------------------------
#
# Reads top_k / top_p / temperature from preallocated device tensors so the
# call can sit inside a CUDA graph capture region without allocating, syncing,
# or branching on CPU-side values. The full ``Sampler`` class is *not* graph
# capturable (repetition-penalty state, ``@torch.compiler.disable``, the CPU
# stream sync inside ``sample_tokens``), so the unrolled MTP loop uses this
# narrower path. ``deterministic=True`` disables the CPU-RNG-seeded path that
# FlashInfer would otherwise take.


def sample_depth_gpu(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Deterministic per-batch top-k/top-p sampling for graph-captured code.

    Uses ``flashinfer.sampling.top_k_top_p_sampling_from_probs`` with
    ``deterministic=True`` -- the graph-safe variant that avoids CPU-seeded
    RNG paths (those require a CPU sync to pull a random offset). Callers
    encode greedy requests as ``(temperature=1.0, top_k=1)`` so this
    function never needs to branch on CPU values.

    Args:
        logits: ``[batch_size, vocab_size]`` raw logits from the codebook head.
        temperature: ``[batch_size]`` float tensor.
        top_k: ``[batch_size]`` int32 tensor. Use ``vocab_size`` to disable.
        top_p: ``[batch_size]`` float tensor. Use ``1.0`` to disable.

    Returns:
        ``[batch_size]`` int64 sampled token IDs. FlashInfer's default
        output is int32; we cast to int64 so the caller can index
        ``nn.Embedding`` modules (which require int64 indices) directly.
    """
    import flashinfer

    scaled = logits / temperature.unsqueeze(-1).to(logits.dtype)
    probs = torch.softmax(scaled, dim=-1)
    top_k = torch.where(top_k > 0, top_k, logits.shape[1])
    samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probs, top_k, top_p, deterministic=True,
        seed=seed, offset=offset
    )
    return samples.to(torch.int64)
