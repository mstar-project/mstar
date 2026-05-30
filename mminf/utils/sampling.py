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

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

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
    _seed: int = 0 # set by the conductor

    def set_seed(self, seed: int):
        self._seed = seed
    
    @property
    def seed(self):
        return self._seed


@dataclass
class BaseSampler(ABC):
    def _broadcast_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """In-place broadcast of ``tokens`` from rank 0 to all TP ranks.

        No-op for ``tp_group`` of size 1 (trivial group / non-TP) or
        unset. Subclasses set ``self.tp_group`` so all TP ranks agree
        on the sampled token (otherwise per-rank RNG diverges →
        mid-sequence garbage, hangs on EOS, KV drift).
        """
        tp_group = getattr(self, "tp_group", None)
        if tp_group is None or tp_group.world_size == 1:
            return tokens
        return tp_group.broadcast(tokens, src=0)

    @abstractmethod
    def sample(
        self, request_ids: list[str], logits: torch.Tensor
    ) -> torch.Tensor:
        pass

    @torch.compiler.disable
    def sample_with_config(
        self, logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float = 1.0,
    ):
        import flashinfer
        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)
        samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, top_k, top_p, deterministic=True,
        )
        return samples.to(torch.int64)

# TODO: add a method for adding prefill tokens to the _seen_token_mask,
# if applying the repetition penalty to tokens from the prompt is desired
@dataclass
class Sampler(BaseSampler):
    # per request
    _sampling_config: dict[str, SamplingConfig] = field(default_factory=dict)
    _seen_token_mask: dict[str, torch.Tensor]= field(default_factory=dict)
    _autotune_sync_budget_remaining: int = 64
    tp_group: "TPCommGroup | None" = None  # noqa: F821

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
        seed = torch.tensor([c.seed for c in configs], device=logits.device, dtype=torch.long)
        rand_offset = torch.zeros_like(seed)
    
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
            seed=seed,
            rand_offset=rand_offset,
            consume_autotune_sync_budget=self._consume_autotune_sync_budget,
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
        tokens = self._broadcast_tokens(tokens)

        if any_rep_pen:
            for i, rid in enumerate(request_ids):
                self._seen_token_mask[rid][tokens[i:i+1]] = True

        return tokens

    def _consume_autotune_sync_budget(self) -> None:
        # Sync between Triton's fused_temperature_softmax (writes probs)
        # and FlashInfer's sampling kernel (reads probs). They live on
        # different CUDA streams in some configurations — verified by
        # speculation-duplication regressions on Qwen3-Omni Thinker that
        # vanish only when the sync is unconditional. The earlier
        # "autotune budget" hack that limited sync to the first 64 calls
        # was based on the assumption that after Triton autotune both
        # kernels land on the default stream; that assumption holds for
        # Orpheus (single shape bucket) but not for Q3-Omni (multiple
        # bs/num_tokens × Thinker/Talker combinations triggering distinct
        # autotunes), which causes mid-sequence token doubling once the
        # budget runs out. Keep the sync unconditional — its cost is
        # ~10 µs per sample, well below the iter budget (3-5 ms), so it
        # does not regress the speculative/CUDA-graph perf wins.
        torch.cuda.current_stream().synchronize()


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
    seed: torch.Tensor | None = None,
    rand_offset: torch.Tensor | None = None,
    consume_autotune_sync_budget: Callable[[], None] | None = None,
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
        # Budgeted current-stream sync between fused-softmax and FlashInfer
        # top-p sampling. The original unconditional sync was added to fix
        # a Qwen sampling issue caused by Triton-autotune timing on first
        # calls (see git history of this file). After autotune warms up
        # both kernels run on the same default stream and stream ordering
        # already serializes them — at that point the sync is pure overhead
        # AND it blocks the same-thread async path from overlapping
        # post-processing with the just-submitted GPU(N+1) work (see
        # worker.run). The budget is owned by each
        # Sampler instance so co-located Thinker/Talker/code-predictor
        # samplers cannot consume one another's warmup protection.
        if consume_autotune_sync_budget is not None:
            consume_autotune_sync_budget()
        result = flashinfer.sampling.top_p_sampling_from_probs(probs, top_p)
        return result[0] if isinstance(result, tuple) else result
    
    probs = fused_temperature_softmax(
        logits, temperature,
        penalty=repetition_penalty if seen_token_mask is not None else None,
        seen_mask=seen_token_mask,
        include_greedy=run_greedy,
    )
    if consume_autotune_sync_budget is not None:
        consume_autotune_sync_budget()
    result = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probs, top_k, top_p,
        deterministic=True,
        seed=seed, offset=rand_offset
    )
    return result[0] if isinstance(result, tuple) else result


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
# Graph-safe sampler
# ---------------------------------------------------------------------------
#
# Reads top_k / top_p / temperature from preallocated device tensors so the
# call can sit inside a CUDA graph capture region without allocating, syncing,
# or branching on CPU-side values. The full ``Sampler`` class is *not* graph
# capturable (repetition-penalty state, ``@torch.compiler.disable``, the CPU
# stream sync inside ``sample_tokens``), so the unrolled MTP loop uses this
# narrower path. ``deterministic=True`` disables the CPU-RNG-seeded path that
# FlashInfer would otherwise take.

def sample_cuda_graphable_gpu(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Deterministic per-batch top-k/top-p sampling for graph-captured code.

    Uses ``flashinfer.sampling.top_k_top_p_sampling_from_logits`` with
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
    top_k = torch.where(top_k > 0, top_k, logits.shape[1])
    samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        scaled, top_k, top_p, deterministic=True,
        seed=seed, offset=offset
    )
    return samples.to(torch.int64)


@dataclass
class CudaGraphableSampler(BaseSampler):
    temperature_buf: torch.Tensor
    top_k_buf: torch.Tensor
    top_p_buf: torch.Tensor
    seed_buf: torch.Tensor
    offset_buf: torch.Tensor
    tp_group: "TPCommGroup | None" = None  # noqa: F821

    @torch.compiler.disable
    def sample(self, request_ids: list[str], logits: torch.Tensor):
        codes = sample_cuda_graphable_gpu(
            logits, self.temperature_buf,
            self.top_k_buf, self.top_p_buf,
            self.seed_buf, self.offset_buf
        )
        self.offset_buf += 1
        return self._broadcast_tokens(codes)
    
    @torch.compiler.disable
    def sample_with_config(
        self, logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float = 1.0,
    ):
        import flashinfer
        scaled = logits / temperature
        samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            scaled, top_k, top_p, deterministic=True,
            seed=self.seed_buf, offset=self.offset_buf
        )
        self.offset_buf += 1
        tokens = samples.to(torch.int64)
        # Defensive broadcast for callers that run this sampler on every TP
        # rank with replicated logits (Qwen3-Omni CodePredictor's unrolled
        # depth loop). ``deterministic=True`` should already produce
        # bit-equal output, but tied-probability sorts can still resolve
        # differently across GPUs in edge cases — one diverging code
        # cascades into garbled audio with no recovery, so we pay the
        # ~5µs in-place broadcast (no-op for trivial groups) to guarantee
        # agreement. Mirrors ``CudaGraphableSampler.sample``.
        return self._broadcast_tokens(tokens)


@dataclass
class SamplerBuffers:
    """Pre-allocated static buffers for graph-safe MTP sampling.

    Owns two tiers of GPU state:

    1. **Per-step buffers** (``temperature_buf``, ``top_k_buf``, ``top_p_buf``,
       ``seed_buf``, ``offset_buf``) — sized ``[max_batch_size]``, sliced to
       ``padded_bs`` on each call and consumed by ``CudaGraphableSampler``.
    2. **Master buffers** (``_master_temperature``, ``_master_top_k``,
       ``_master_top_p``) — sized ``[max_batch_size]`` and indexed by a stable
       per-request slot. Each active request occupies one slot.

    The per-step path (``gather_for_request_ids``) builds a small slot-index
    tensor on a pinned CPU buffer, async-copies it to GPU, and runs three
    ``torch.index_select`` calls into the per-step buffers. This replaces the
    old per-step Python loop that issued ``temps[i] = float(...)``-style
    item-assignments (one tiny H2D + kernel launch per element), which
    competed with the async-engine pipeline for host-side throughput.

    Seeds are kept GPU-local: ``seed_buf`` is refilled per step via
    ``torch.randint`` on device, preserving step-to-step randomization
    without any CPU sync.
    """
    max_batch_size: int
    temperature_buf: torch.Tensor   # [max_bs], float32
    top_k_buf: torch.Tensor         # [max_bs], int32
    top_p_buf: torch.Tensor         # [max_bs], float32
    seed_buf: torch.Tensor          # [max_bs], int64
    offset_buf: torch.Tensor        # [max_bs], int64
    # TP communicator for the submodule that owns these buffers. Passed
    # through ``slice_for_bs`` into every per-step ``CudaGraphableSampler``
    # so its ``_broadcast_tokens`` aligns the sampled token across ranks.
    # Without this, ``sample`` / ``sample_with_config`` would build a
    # sampler with ``tp_group=None``, the broadcast would silently no-op,
    # and TP ranks would drift apart on the first tied-logit sample —
    # garbled audio for Talker, premature EOS for Thinker. Defaults to
    # ``None`` for non-TP submodules (trivial broadcast is a cheap no-op).
    tp_group: "TPCommGroup | None" = None  # noqa: F821
    # Master cache: one row per active request, indexed by slot. Grown
    # dynamically (doubling) when more than ``max_batch_size`` requests are
    # in-flight simultaneously — the master is decoupled from the per-step
    # buffer size so registered-but-not-batched requests don't constrain the
    # cuda-graph capture batch sizes.
    _master_temperature: torch.Tensor = field(default=None, repr=False)
    _master_top_k: torch.Tensor = field(default=None, repr=False)
    _master_top_p: torch.Tensor = field(default=None, repr=False)
    _master_seed: torch.Tensor = field(default=None, repr=False)
    _master_capacity: int = field(default=0, repr=False)
    # Per-step slot-index staging. ``_slot_idx_cpu`` is pinned so the H2D
    # copy can be issued non-blocking; ``_slot_idx_gpu`` is the device-side
    # index tensor that ``index_select`` reads from.
    _slot_idx_cpu: torch.Tensor = field(default=None, repr=False)
    _slot_idx_gpu: torch.Tensor = field(default=None, repr=False)
    # Single-row pinned staging used by ``register_request`` /
    # ``update_request_config`` to push (temperature, top_k, top_p) for one
    # slot via a single async H2D per buffer (rather than 3 elementwise
    # item-assignments on the GPU master buffers).
    _row_temp_cpu: torch.Tensor = field(default=None, repr=False)
    _row_top_k_cpu: torch.Tensor = field(default=None, repr=False)
    _row_top_p_cpu: torch.Tensor = field(default=None, repr=False)
    _row_seed_cpu: torch.Tensor = field(default=None, repr=False)
    # Slot bookkeeping (CPU-only).
    _rid_to_slot: dict[str, int] = field(default_factory=dict, repr=False)
    _free_slots: list[int] = field(default_factory=list, repr=False)
    # Last-known config per rid — change-detect for ``update_request_config``
    # so steady-state per-step calls do zero GPU work.
    _cached_config: dict[str, SamplingConfig] = field(default_factory=dict, repr=False)

    @classmethod
    def allocate(
        cls,
        max_batch_size: int,
        device: torch.device,
        tp_group: "TPCommGroup | None" = None,  # noqa: F821
    ) -> "SamplerBuffers":
        """Allocate zero-initialised sampling buffers for ``max_batch_size``.
        """
        temperature_buf = torch.ones(max_batch_size, dtype=torch.float32, device=device)
        top_k_buf = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        top_p_buf = torch.ones(max_batch_size, dtype=torch.float32, device=device)
        seed_buf = torch.zeros(max_batch_size, dtype=torch.long, device=device)
        offset_buf = torch.zeros(max_batch_size, dtype=torch.long, device=device)

        # Master cache initialised to the same defaults as a SamplingConfig()
        # row would produce (temp=1, top_k=0, top_p=1) — these defaults are
        # what an unregistered slot would surface if accidentally indexed.
        master_temperature = torch.ones(max_batch_size, dtype=torch.float32, device=device)
        master_top_k = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        master_top_p = torch.ones(max_batch_size, dtype=torch.float32, device=device)
        master_seed = torch.zeros(max_batch_size, dtype=torch.long, device=device)

        # Pinned CPU staging — small, allocated once, reused every step.
        pinned = torch.cuda.is_available() and device.type == "cuda"
        slot_idx_cpu = torch.zeros(max_batch_size, dtype=torch.long, pin_memory=pinned)
        slot_idx_gpu = torch.zeros(max_batch_size, dtype=torch.long, device=device)
        row_temp_cpu = torch.zeros(1, dtype=torch.float32, pin_memory=pinned)
        row_top_k_cpu = torch.zeros(1, dtype=torch.int32, pin_memory=pinned)
        row_top_p_cpu = torch.zeros(1, dtype=torch.float32, pin_memory=pinned)
        row_seed_cpu = torch.zeros(1, dtype=torch.long, pin_memory=pinned)

        return cls(
            max_batch_size=max_batch_size,
            temperature_buf=temperature_buf,
            top_k_buf=top_k_buf,
            top_p_buf=top_p_buf,
            seed_buf=seed_buf,
            offset_buf=offset_buf,
            tp_group=tp_group,
            _master_temperature=master_temperature,
            _master_top_k=master_top_k,
            _master_top_p=master_top_p,
            _master_seed=master_seed,
            _master_capacity=max_batch_size,
            _slot_idx_cpu=slot_idx_cpu,
            _slot_idx_gpu=slot_idx_gpu,
            _row_temp_cpu=row_temp_cpu,
            _row_top_k_cpu=row_top_k_cpu,
            _row_top_p_cpu=row_top_p_cpu,
            _row_seed_cpu=row_seed_cpu,
            _free_slots=list(range(max_batch_size)),
        )

    def slice_for_bs(self, bs: int) -> dict[str, Any]:
        """Return bs-sized views into each buffer (zero-copy slices) plus
        the owning submodule's ``tp_group`` so the constructed sampler
        broadcasts across TP ranks."""
        return {
            "temperature_buf": self.temperature_buf[:bs],
            "top_k_buf": self.top_k_buf[:bs],
            "top_p_buf": self.top_p_buf[:bs],
            "seed_buf": self.seed_buf[:bs],
            "offset_buf": self.offset_buf[:bs],
            "tp_group": self.tp_group,
        }

    # ------------------------------------------------------------------
    # Master-cache lifecycle: register / unregister / update per request
    # ------------------------------------------------------------------

    def _write_master_row(self, slot: int, cfg: SamplingConfig) -> None:
        """Push one config row into the master GPU buffers via pinned H2D.

        Three async non-blocking copies, one per master tensor. Cheap; only
        runs on register or actual config change (change-detection lives in
        ``update_request_config``).
        """
        s = cfg.seed
        if cfg.temperature > 0:
            t = float(cfg.temperature)
            k = int(cfg.top_k)
            p = float(cfg.top_p) if cfg.top_p else 1.0
        else:
            # Greedy: kernel takes the one-hot/argmax branch regardless of
            # top_k/top_p, so park them at the disabled defaults.
            t, k, p = 1.0, 1, 1.0

        self._row_temp_cpu[0] = t
        self._row_top_k_cpu[0] = k
        self._row_top_p_cpu[0] = p
        self._row_seed_cpu[0] = s
        self._master_temperature[slot:slot + 1].copy_(self._row_temp_cpu, non_blocking=True)
        self._master_top_k[slot:slot + 1].copy_(self._row_top_k_cpu, non_blocking=True)
        self._master_top_p[slot:slot + 1].copy_(self._row_top_p_cpu, non_blocking=True)
        self._row_seed_cpu[slot:slot + 1].copy_(self._row_seed_cpu, non_blocking=True)

    def _grow_master(self, new_capacity: int) -> None:
        """Double-and-copy the master buffers up to at least ``new_capacity``.

        Triggered when the number of concurrently-registered requests exceeds
        the current master capacity. Per-step buffers (sized to the cuda-graph
        max_bs) are NOT resized — the gather only reads ``padded_bs`` rows
        from master, which always fits within the per-step buffer.
        """
        device = self._master_temperature.device
        new_temp = torch.ones(new_capacity, dtype=torch.float32, device=device)
        new_top_k = torch.zeros(new_capacity, dtype=torch.int32, device=device)
        new_top_p = torch.ones(new_capacity, dtype=torch.float32, device=device)
        new_seed = torch.zeros(new_capacity, dtype=torch.long, device=device)
        new_temp[: self._master_capacity].copy_(self._master_temperature)
        new_top_k[: self._master_capacity].copy_(self._master_top_k)
        new_top_p[: self._master_capacity].copy_(self._master_top_p)
        new_seed[: self._master_capacity].copy_(self._master_seed)
        self._master_temperature = new_temp
        self._master_top_k = new_top_k
        self._master_top_p = new_top_p
        self._master_seed = new_seed
        self._free_slots.extend(range(self._master_capacity, new_capacity))
        self._master_capacity = new_capacity

    def register_request(
        self, rid: str, sampling_config: SamplingConfig | None = None,
    ) -> None:
        """Allocate a slot for ``rid`` and seed its master row."""
        if rid in self._rid_to_slot:
            # Re-registration: just refresh the config in place.
            if sampling_config is not None:
                self.update_request_config(rid, sampling_config)
            return
        if not self._free_slots:
            self._grow_master(self._master_capacity * 2)
        slot = self._free_slots.pop()
        self._rid_to_slot[rid] = slot
        cfg = sampling_config if sampling_config is not None else SamplingConfig()
        self._cached_config[rid] = cfg
        self._write_master_row(slot, cfg)

    def unregister_request(self, rid: str) -> None:
        """Free the slot owned by ``rid`` (no GPU writes)."""
        slot = self._rid_to_slot.pop(rid, None)
        if slot is None:
            return
        self._cached_config.pop(rid, None)
        self._free_slots.append(slot)

    def update_request_config(
        self, rid: str, sampling_config: SamplingConfig,
    ) -> None:
        """Update the master row for ``rid`` only when its config changed.

        AR engine calls this every step (mirroring the existing
        ``Sampler.set_config`` per-step pattern). Steady-state requests have
        identical configs across steps, so the change-check skips the H2D
        path entirely.
        """
        slot = self._rid_to_slot.get(rid)
        if slot is None:
            # Request not yet registered for this submodule (e.g. ar_engine
            # may invoke set_config for a node that doesn't own a runner /
            # SamplerBuffers). Silently no-op.
            return
        prev = self._cached_config.get(rid)
        if prev == sampling_config:
            return
        self._cached_config[rid] = sampling_config
        self._write_master_row(slot, sampling_config)

    # ------------------------------------------------------------------
    # Per-step gather: pinned-H2D slot-index → index_select into per-step bufs
    # ------------------------------------------------------------------

    def gather_for_request_ids(
        self, request_ids: list[str], padded_bs: int,
    ) -> "CudaGraphableSampler":
        """Materialise the per-step sampling tensors for ``request_ids``.

        Padding slots (``i >= len(request_ids)``) reuse slot 0's row — the
        captured graph forwards them through the same kernels as real slots,
        but their outputs are discarded by the runner's dummy-rid remap, so
        the row contents don't matter as long as they're well-formed.
        """
        assert padded_bs <= self.max_batch_size, (
            f"padded_bs={padded_bs} exceeds SamplerBuffers.max_batch_size="
            f"{self.max_batch_size}"
        )

        # CPU-only fill of the pinned slot-index buffer. Unregistered rids
        # fall back to slot 0 (matches the old code's defaults — temp=1,
        # top_k=0, top_p=1 — for any rid the AR engine forgot to register).
        for i, rid in enumerate(request_ids):
            self._slot_idx_cpu[i] = self._rid_to_slot.get(rid, 0)
        for i in range(len(request_ids), padded_bs):
            self._slot_idx_cpu[i] = 0

        # Single async H2D (pinned) of the slot indices.
        idx_view = self._slot_idx_gpu[:padded_bs]
        idx_view.copy_(self._slot_idx_cpu[:padded_bs], non_blocking=True)

        # Three GPU index_select kernels, writing directly into the
        # cuda-graph-friendly per-step buffers.
        torch.index_select(
            self._master_temperature, 0, idx_view,
            out=self.temperature_buf[:padded_bs],
        )
        torch.index_select(
            self._master_top_k, 0, idx_view, out=self.top_k_buf[:padded_bs],
        )
        torch.index_select(
            self._master_top_p, 0, idx_view, out=self.top_p_buf[:padded_bs],
        )
        torch.index_select(
            self._master_seed, 0, idx_view, out=self.seed_buf[:padded_bs],
        )

        # offset_buf is NOT reset here. With per-request fixed seed and
        # ``deterministic=True`` sampling, resetting offset every call
        # would make every iteration sample with (same seed, offset=0)
        # — identical RNG draws. Once the logits also stabilise (e.g.,
        # Talker decode after the producer stream ends and inputs become
        # the static TTS_EOS/pad embed), the sampler returns the same
        # token forever and the loop never reaches its natural EOS.
        # Letting offset accumulate from the in-graph ``offset_buf += 1``
        # advances the RNG step per iteration so identical-logit
        # iterations still produce different samples.

        slices = self.slice_for_bs(padded_bs)
        return CudaGraphableSampler(**slices)


def make_sampler_from_buffers(
    bufs: SamplerBuffers,
    request_ids: list[str],
    sampling_configs: dict[str, SamplingConfig],
    padded_bs: int,
) -> CudaGraphableSampler:
    """Compatibility shim. Prefer ``bufs.gather_for_request_ids`` directly.

    ``sampling_configs`` is no longer consulted — per-request configs live
    on ``bufs`` (set via ``register_request`` / ``update_request_config``).
    The argument is kept for source-level compatibility with older callers.
    """
    del sampling_configs
    return bufs.gather_for_request_ids(request_ids, padded_bs)