"""Generic token sampling utilities.

Uses FlashInfer's fused top-k/top-p sampling kernel for GPU efficiency
and CUDA graph compatibility. Model-agnostic — any AR model returns logits,
this module selects the next token.

Supports per-request sampling parameters (different temperature/top_k/top_p
for each request in a batch) via tensor parameters.

CUDA graph compatible: no Python control flow branches — uses masking
to handle greedy vs sampled requests in the same batch.

Usage:
    from mstar.utils.sampling import sample_tokens
    tokens = sample_tokens(logits, temperature=0.7, top_p=0.9)
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


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
    with torch.cuda.device(logits.device):
        # BLOCK_SIZE is picked by @triton.autotune (not passed here). The first
        # launch for a given key benchmarks every config (do_bench), which can
        # leave probs in a state not ordered on the current stream relative to
        # the downstream FlashInfer read -> garbage. We can't gate the sync on
        # autotune alone (that wasn't enough on its own); pairing it with the
        # device context above is what fixes it. Detect the autotune call by the
        # config cache growing, and sync only then -- steady state stays sync-free.
        cache = getattr(_fused_sampling_prep_kernel, "cache", None)
        cache_size_before = len(cache) if cache is not None else 0
        _fused_sampling_prep_kernel[grid](
            logits, temperature, pen_ptr, mask_ptr, probs,
            V,
            logits.stride(0), logits.stride(1),
            probs.stride(0), probs.stride(1),
            mask_stride_b, mask_stride_v,
            APPLY_PENALTY=apply_penalty,
            INCLUDE_GREEDY=include_greedy,
        )
        if cache is not None and len(cache) > cache_size_before:
            torch.cuda.current_stream().synchronize()
    return probs


@dataclass
class SamplingConfig:
    # Sizes the per-request seen-token mask for the repetition penalty. When set,
    # it MUST equal the model's logit width (lm_head/codec_head output dim): the
    # mask is indexed as ``[B, vocab_size]`` against ``logits[B, V]``, and on the
    # CUDA-graph path it also gates allocation of the in-graph penalty buffers.
    vocab_size: int | None = None
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 1
    ignore_eos: bool = False # used for benchmark parity
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
        self, request_ids: list[str], logits: torch.Tensor, **kwargs
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


@dataclass
class SeenTokenMask:
    request_id: str
    _seen_token_mask: torch.Tensor | None

    @classmethod
    def new(cls, request_id: str, vocab_size: int | None, device):
        return cls(
            request_id=request_id,
            _seen_token_mask=torch.zeros(
                vocab_size, dtype=torch.bool, device=device
            ) if vocab_size is not None else None,
            
        )

    def add_tokens(self, tokens: torch.Tensor | int):
        if self._seen_token_mask is None:
            logger.warning(
                f"Calling add_tokens on an uninitialized SeenTokenMask, i.e., "
                "one where the vocab_size was provided in the SamplingConfig or "
                "the SamplingConfig has not yet been registered with the Sampler.s"
            )
            return
        idx = torch.as_tensor(
            tokens, dtype=torch.long, device=self._seen_token_mask.device,
        ).reshape(-1)
        self._seen_token_mask.scatter_(0, idx, True)


@dataclass
class Sampler(BaseSampler):
    # per request
    device: torch.device
    _sampling_config: dict[str, SamplingConfig] = field(default_factory=dict)
    _seen_token_mask: dict[str, SeenTokenMask]= field(default_factory=dict)
    # Per-request RNG offset, advanced once per sampled step. Paired with the
    # request's fixed seed, this steps the philox stream so deterministic
    # (seeded) sampling draws a fresh number each step — otherwise identical
    # (seed, offset=0) draws repeat forever and stable logits never reach EOS.
    _step_offset: dict[str, int] = field(default_factory=dict)
    tp_group: "TPCommGroup | None" = None  # noqa: F821

    def add_request(self, request_id: str):
        self._sampling_config[request_id] = SamplingConfig()
        self._seen_token_mask[request_id] =  SeenTokenMask.new(
            request_id,
            vocab_size=None,
            device=self.device
        )
        self._step_offset[request_id] = 0
        # lazy init _seen_token_mask, taking vocab size from logits or cfg

    def get_token_mask(self, request_id: str):
        return self._seen_token_mask[request_id]

    def remove_request(self, request_id: str):
        if request_id in self._sampling_config:
            del self._sampling_config[request_id]
        if request_id in self._seen_token_mask:
            del self._seen_token_mask[request_id]
        self._step_offset.pop(request_id, None)

    def set_config(self, request_id: str, **kwargs):
        old_vocab_size = self._sampling_config[request_id].vocab_size
        curr_config = asdict(self._sampling_config[request_id])
        kwargs = {k: arg for k, arg in kwargs.items() if k in curr_config.keys()}
        self._sampling_config[request_id] = SamplingConfig(**{
            **curr_config, **kwargs
        })

        new_vocab_size = self._sampling_config[request_id].vocab_size
        if old_vocab_size != new_vocab_size:
            self._seen_token_mask[request_id] = SeenTokenMask.new(
                request_id=request_id,
                vocab_size=new_vocab_size,
                device=self.device
            )

    def sample(
        self, request_ids: list[str], logits: torch.Tensor, **kwargs
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
        rand_offset = torch.tensor(
            [self._step_offset.get(rid, 0) for rid in request_ids],
            device=logits.device, dtype=torch.long,
        )
    
        any_rep_pen = any(c.repetition_penalty != 1.0 for c in configs)
        any_greedy = any(c.temperature == 0 for c in configs)
        any_top_k_zero = any(c.top_k == 0 for c in configs)
        all_top_k_zero = all(c.top_k == 0 for c in configs)

        for rid in request_ids:
            if self._seen_token_mask[rid]._seen_token_mask is None:
                self._seen_token_mask[rid] = SeenTokenMask.new(
                    rid, vocab_size=logits.shape[1],
                    device=self.device
                )
    
        seen_mask = None
        if any_rep_pen:
            seen_mask = torch.stack(
                [self._seen_token_mask[rid]._seen_token_mask for rid in request_ids], dim=0,
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
                self._seen_token_mask[rid].add_tokens(tokens[i:i+1])

        # Advance the per-request RNG offset so the next step draws fresh.
        for rid in request_ids:
            self._step_offset[rid] = self._step_offset.get(rid, 0) + 1

        return tokens


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

    import flashinfer

    # Pin the Triton prep kernel (writes probs) and the FlashInfer sampler
    # (reads probs) to the same device/stream so the write-before-read is
    # ordered without an explicit sync. Otherwise FlashInfer runs on the
    # worker's current-device stream while probs lives off-device (e.g. BAGEL
    # LLM on rank 1) — a cross-stream race that yields garbage.
    with torch.cuda.device(logits.device):
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
            result = flashinfer.sampling.top_p_sampling_from_probs(
                probs, top_p,
                deterministic=True,
                seed=seed, offset=rand_offset,
            )
            return result[0] if isinstance(result, tuple) else result

        probs = fused_temperature_softmax(
            logits, temperature,
            penalty=repetition_penalty if seen_token_mask is not None else None,
            seen_mask=seen_token_mask,
            include_greedy=run_greedy,
        )
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
# capturable (repetition-penalty state, ``@torch.compiler.disable``, the
# device-context switch inside ``sample_tokens``), so the unrolled MTP loop uses
# this narrower path. ``deterministic=True`` disables the CPU-RNG-seeded path that
# FlashInfer would otherwise take.

def sample_cuda_graphable_gpu(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    apply_penalty: bool = False,
    rep_penalty: torch.Tensor | None = None,
    seen_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deterministic per-batch top-k/top-p sampling for graph-captured code.

    Routes through the fused Triton prep kernel (``fused_temperature_softmax``)
    so the CUDA-graph path can apply the same vLLM-style repetition penalty as
    the regular ``Sampler``, then samples with
    ``flashinfer.sampling.top_k_top_p_sampling_from_probs`` (``deterministic=True``
    — the graph-safe variant that avoids CPU-seeded RNG paths). Greedy requests
    are encoded as ``(temperature=1.0, top_k=1)`` so ``from_probs`` returns the
    argmax and this function never branches on CPU values (``include_greedy`` is
    therefore left off).

    The autotune sync inside ``fused_temperature_softmax`` only fires the first
    time a kernel key is seen, which happens during eager warmup — by capture
    time the config is cached, so the captured launch is sync-free.

    Args:
        logits: ``[batch_size, vocab_size]`` raw logits from the codebook head.
        temperature: ``[batch_size]`` float tensor.
        top_k: ``[batch_size]`` int32 tensor. Use ``vocab_size`` to disable.
        top_p: ``[batch_size]`` float tensor. Use ``1.0`` to disable.
        apply_penalty: when True, ``rep_penalty`` + ``seen_tokens`` are applied.
        rep_penalty: ``[batch_size]`` float tensor (1.0 = disabled per row).
        seen_tokens: ``[batch_size, vocab_size]`` bool mask of seen tokens.

    Returns:
        ``[batch_size]`` int64 sampled token IDs. FlashInfer's default
        output is int32; we cast to int64 so the caller can index
        ``nn.Embedding`` modules (which require int64 indices) directly.
    """
    import flashinfer

    probs = fused_temperature_softmax(
        logits, temperature,
        penalty=rep_penalty if apply_penalty else None,
        seen_mask=seen_tokens if apply_penalty else None,
        include_greedy=False,
    )
    top_k = torch.where(top_k > 0, top_k, logits.shape[1])
    samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probs, top_k, top_p, deterministic=True,
        seed=seed, offset=offset,
    )
    return samples.to(torch.int64)


@dataclass
class CudaGraphableSampler(BaseSampler):
    temperature_buf: torch.Tensor
    top_k_buf: torch.Tensor
    top_p_buf: torch.Tensor
    seed_buf: torch.Tensor
    offset_buf: torch.Tensor
    # Repetition-penalty state for the CUDA-graph path. ``None`` for submodules
    # that don't opt into seen-token tracking (then ``apply_penalty`` is a no-op).
    rep_penalty_buf: torch.Tensor | None = None
    seen_tokens_buf: torch.Tensor | None = None  # [bs, V] bool
    tp_group: "TPCommGroup | None" = None  # noqa: F821

    # Set during graph capture, and used by the cuda graph runner to determine
    # whether requests' seen token buffers should be synced post-replay
    applied_penalty_in_graph: bool = False

    @torch.compiler.disable
    def sample(
        self, request_ids: list[str], logits: torch.Tensor,
        apply_penalty: bool = False,
    ):
        codes = sample_cuda_graphable_gpu(
            logits, self.temperature_buf,
            self.top_k_buf, self.top_p_buf,
            self.seed_buf, self.offset_buf,
            apply_penalty=apply_penalty,
            rep_penalty=self.rep_penalty_buf,
            seen_tokens=self.seen_tokens_buf,
        )
        self.offset_buf += 1
        codes = self._broadcast_tokens(codes)
        if apply_penalty and self.seen_tokens_buf is not None:
            self.applied_penalty_in_graph = True
            # Record the (broadcast, TP-agreed) token in the seen-token buffer so
            # the next step penalises it. ``scatter_`` with a scalar value is
            # CUDA-graph capturable; advanced-index assignment
            # (``buf[rows, codes] = True``) is not — it trips "operation not
            # permitted when stream is capturing".
            self.seen_tokens_buf.scatter_(1, codes.unsqueeze(1), True)
        return codes

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

    @torch.compiler.disable
    def sync_seen_token_masks(
        self, seen_masks: "Iterable[SeenTokenMask]",
    ) -> None:
        """Copy the in-graph seen-token rows back into canonical ``SeenTokenMask``s.

        Called eagerly after graph replay (the captured ``sample`` scattered the
        newly sampled token into ``seen_tokens_buf``). ``seen_masks`` are in
        request order; padding rows beyond ``len(seen_masks)`` are ignored, and
        not-yet-sized masks (``_seen_token_mask is None``) are skipped.
        """
        if self.seen_tokens_buf is None:
            return
        for i, m in enumerate(seen_masks):
            mask = m._seen_token_mask
            if mask is not None:
                mask.copy_(self.seen_tokens_buf[i])


@dataclass
class Buffer:
    """Three-tier storage for one per-request scalar sampling parameter.

    - ``buf``     ``[max_bs]``   per-step tensor, sliced to ``padded_bs`` and read
      by ``CudaGraphableSampler`` (its address must stay stable across replays).
    - ``master``  ``[capacity]`` slot-indexed cache, one row per active request.
    - ``row_cpu`` ``[1]`` pinned staging for a single async H2D master-row write.
    """
    buf: torch.Tensor
    master: torch.Tensor
    row_cpu: torch.Tensor
    default: float
    dtype: torch.dtype

    @classmethod
    def allocate(
        cls, max_bs: int, capacity: int, device: torch.device,
        dtype: torch.dtype, default: float, pinned: bool,
    ) -> "Buffer":
        return cls(
            buf=torch.full((max_bs,), default, dtype=dtype, device=device),
            master=torch.full((capacity,), default, dtype=dtype, device=device),
            row_cpu=torch.zeros(1, dtype=dtype, pin_memory=pinned),
            default=default,
            dtype=dtype,
        )

    def write_master_row(self, slot: int, value) -> None:
        self.row_cpu[0] = value
        self.master[slot:slot + 1].copy_(self.row_cpu, non_blocking=True)

    def grow_master(self, new_capacity: int) -> None:
        new = torch.full(
            (new_capacity,), self.default, dtype=self.dtype, device=self.master.device,
        )
        new[: self.master.shape[0]].copy_(self.master)
        self.master = new

    def gather(self, idx_view: torch.Tensor, padded_bs: int) -> None:
        torch.index_select(self.master, 0, idx_view, out=self.buf[:padded_bs])


@dataclass
class MaskBuffer:
    """Three-tier storage for the per-request seen-token mask ``[*, V]`` (bool).

    Mirrors ``Buffer`` but 2-D and sourced from on-device ``SeenTokenMask``
    tensors, so the master-row write is a GPU->GPU copy (no pinned staging).
    """
    buf: torch.Tensor       # [max_bs, V] bool
    master: torch.Tensor    # [capacity, V] bool
    vocab_size: int

    @classmethod
    def allocate(
        cls, max_bs: int, capacity: int, vocab_size: int, device: torch.device,
    ) -> "MaskBuffer":
        return cls(
            buf=torch.zeros(max_bs, vocab_size, dtype=torch.bool, device=device),
            master=torch.zeros(capacity, vocab_size, dtype=torch.bool, device=device),
            vocab_size=vocab_size,
        )

    def write_master_row(self, slot: int, mask: torch.Tensor) -> None:
        # ``mask`` is the [V] bool tensor owned by a SeenTokenMask (on device).
        self.master[slot].copy_(mask)

    def clear_master_row(self, slot: int) -> None:
        self.master[slot].zero_()

    def grow_master(self, new_capacity: int) -> None:
        new = torch.zeros(
            new_capacity, self.vocab_size, dtype=torch.bool, device=self.master.device,
        )
        new[: self.master.shape[0]].copy_(self.master)
        self.master = new

    def gather(self, idx_view: torch.Tensor, padded_bs: int) -> None:
        torch.index_select(self.master, 0, idx_view, out=self.buf[:padded_bs])


@dataclass
class SamplerBuffers:
    """Pre-allocated static buffers for graph-safe sampling.

    Each per-request scalar parameter (temperature, top_k, top_p, seed,
    repetition_penalty) is a ``Buffer`` owning a per-step slice, a slot-indexed
    master cache, and pinned row staging. ``offset_buf`` is special-cased (no
    master; it accumulates in-graph via ``offset_buf += 1``). The optional
    ``seen_tokens`` ``MaskBuffer`` (allocated only when ``vocab_size`` is given)
    carries the per-request repetition-penalty mask for the CUDA-graph path.

    ``gather_for_request_ids`` builds a pinned slot-index tensor, async-copies it
    to GPU, and ``index_select``s each master into the per-step buffers — one
    cheap gather per buffer instead of the old per-element item-assignments.
    """
    max_batch_size: int
    temperature: Buffer
    top_k: Buffer
    top_p: Buffer
    seed: Buffer
    rep_penalty: Buffer
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
    # Per-request seen-token mask buffer for the repetition penalty. Present
    # only for submodules that opt in by declaring a vocab size (e.g. the
    # Qwen3-Omni Talker). ``None`` => the CUDA-graph path applies no penalty.
    seen_tokens: "MaskBuffer | None" = None
    # Master cache capacity (grown by doubling when more requests are
    # concurrently registered than the per-step buffer holds).
    _master_capacity: int = field(default=0, repr=False)
    # Per-step slot-index staging. ``_slot_idx_cpu`` is pinned so the H2D
    # copy can be issued non-blocking; ``_slot_idx_gpu`` is the device-side
    # index tensor that ``index_select`` reads from.
    _slot_idx_cpu: torch.Tensor = field(default=None, repr=False)
    _slot_idx_gpu: torch.Tensor = field(default=None, repr=False)
    # Slot bookkeeping (CPU-only).
    _rid_to_slot: dict[str, int] = field(default_factory=dict, repr=False)
    _free_slots: list[int] = field(default_factory=list, repr=False)
    # Last-known config per rid — change-detect for ``update_request_config``
    # so steady-state per-step calls do zero GPU work (for the scalar rows).
    _cached_config: dict[str, SamplingConfig] = field(default_factory=dict, repr=False)

    @property
    def tracks_seen_tokens(self) -> bool:
        return self.seen_tokens is not None

    def _scalar_buffers(self) -> list[Buffer]:
        return [self.temperature, self.top_k, self.top_p, self.seed, self.rep_penalty]

    @classmethod
    def allocate(
        cls,
        max_batch_size: int,
        device: torch.device,
        tp_group: "TPCommGroup | None" = None,  # noqa: F821
        vocab_size: int | None = None,
    ) -> "SamplerBuffers":
        """Allocate sampling buffers for ``max_batch_size``.

        ``vocab_size`` (when not None) enables the seen-token mask buffer for the
        repetition penalty. The master rows default to a ``SamplingConfig()`` row
        (temp=1, top_k=0, top_p=1, rep_penalty=1) — what an unregistered slot
        would surface if accidentally indexed.
        """
        pinned = torch.cuda.is_available() and device.type == "cuda"
        cap = max_batch_size

        def mk(dtype: torch.dtype, default: float) -> Buffer:
            return Buffer.allocate(max_batch_size, cap, device, dtype, default, pinned)

        seen_tokens = (
            MaskBuffer.allocate(max_batch_size, cap, vocab_size, device)
            if vocab_size is not None else None
        )
        return cls(
            max_batch_size=max_batch_size,
            temperature=mk(torch.float32, 1.0),
            top_k=mk(torch.int32, 0),
            top_p=mk(torch.float32, 1.0),
            seed=mk(torch.long, 0),
            rep_penalty=mk(torch.float32, 1.0),
            offset_buf=torch.zeros(max_batch_size, dtype=torch.long, device=device),
            tp_group=tp_group,
            seen_tokens=seen_tokens,
            _master_capacity=cap,
            _slot_idx_cpu=torch.zeros(max_batch_size, dtype=torch.long, pin_memory=pinned),
            _slot_idx_gpu=torch.zeros(max_batch_size, dtype=torch.long, device=device),
            _free_slots=list(range(cap)),
        )

    def slice_for_bs(self, bs: int) -> dict[str, Any]:
        """Return bs-sized views into each buffer (zero-copy slices) plus
        the owning submodule's ``tp_group`` so the constructed sampler
        broadcasts across TP ranks."""
        return {
            "temperature_buf": self.temperature.buf[:bs],
            "top_k_buf": self.top_k.buf[:bs],
            "top_p_buf": self.top_p.buf[:bs],
            "seed_buf": self.seed.buf[:bs],
            "offset_buf": self.offset_buf[:bs],
            "rep_penalty_buf": self.rep_penalty.buf[:bs],
            "seen_tokens_buf": self.seen_tokens.buf[:bs] if self.seen_tokens is not None else None,
            "tp_group": self.tp_group,
        }

    # ------------------------------------------------------------------
    # Master-cache lifecycle: register / unregister / update per request
    # ------------------------------------------------------------------

    def _write_master_row(self, slot: int, cfg: SamplingConfig) -> None:
        """Push one config row into each scalar master buffer via pinned H2D.

        Cheap async copies; only runs on register or actual config change
        (change-detection lives in ``update_request_config``). The seen-token
        mask is NOT written here (it changes every step — see
        ``update_request_config``).
        """
        if cfg.temperature > 0:
            t = float(cfg.temperature)
            k = int(cfg.top_k)
            p = float(cfg.top_p) if cfg.top_p else 1.0
        else:
            # Greedy: encoded as (temp=1, top_k=1) so from_probs returns argmax.
            t, k, p = 1.0, 1, 1.0
        self.temperature.write_master_row(slot, t)
        self.top_k.write_master_row(slot, k)
        self.top_p.write_master_row(slot, p)
        self.seed.write_master_row(slot, cfg.seed)
        self.rep_penalty.write_master_row(slot, float(cfg.repetition_penalty))

    def _grow_master(self, new_capacity: int) -> None:
        """Double-and-copy the master buffers up to at least ``new_capacity``.

        Triggered when concurrently-registered requests exceed the current
        master capacity. Per-step buffers (sized to the cuda-graph max_bs) are
        NOT resized — the gather only reads ``padded_bs`` rows from master.
        """
        for buf in self._scalar_buffers():
            buf.grow_master(new_capacity)
        if self.seen_tokens is not None:
            self.seen_tokens.grow_master(new_capacity)
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
        # Clear any stale seen-token history from a previously-freed slot. The
        # first per-step ``update_request_config`` overwrites it with the live
        # mask before the slot is gathered, but clearing is cheap insurance.
        if self.seen_tokens is not None:
            self.seen_tokens.clear_master_row(slot)

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

        AR engine calls this every step (mirroring ``Sampler.set_config``).
        Steady-state requests have identical configs across steps, so the
        change-check skips the H2D path entirely. The seen-token mask is staged
        separately (see ``stage_seen_token_masks``) because it grows every step.
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

    def stage_seen_token_masks(
        self, request_ids: list[str], seen_masks: "Iterable[SeenTokenMask]",
    ) -> None:
        """Copy each request's current seen-token mask into its master row.

        Called every step (before ``gather_for_request_ids``) for submodules that
        sample with a penalty in-graph, so the gathered per-step buffer reflects
        the live prompt + generated tokens. No-op when seen-token tracking is off.
        """
        if self.seen_tokens is None:
            return
        for rid, m in zip(request_ids, seen_masks, strict=False):
            slot = self._rid_to_slot.get(rid)
            if slot is None:
                continue
            mask = m._seen_token_mask
            if mask is not None:
                self.seen_tokens.write_master_row(slot, mask)

    # ------------------------------------------------------------------
    # Per-step gather: pinned-H2D slot-index → index_select into per-step bufs
    # ------------------------------------------------------------------

    def gather_for_request_ids(
        self, request_ids: list[str], padded_bs: int,
        gather_seen_tokens: bool = True,
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
        # fall back to slot 0 (matches the defaults — temp=1, top_k=0, top_p=1,
        # rep_penalty=1 — for any rid the AR engine forgot to register).
        for i, rid in enumerate(request_ids):
            self._slot_idx_cpu[i] = self._rid_to_slot.get(rid, 0)
        for i in range(len(request_ids), padded_bs):
            self._slot_idx_cpu[i] = 0

        # Single async H2D (pinned) of the slot indices.
        idx_view = self._slot_idx_gpu[:padded_bs]
        idx_view.copy_(self._slot_idx_cpu[:padded_bs], non_blocking=True)

        # One index_select per buffer, writing directly into the
        # cuda-graph-friendly per-step buffers.
        for buf in self._scalar_buffers():
            buf.gather(idx_view, padded_bs)
        # The seen-token mask is large ([bs, V] bool); only gather it when the
        # caller's graph actually applies the penalty in-graph (the Talker), so
        # non-penalty graphs that happen to allocate the buffer don't pay for it.
        if self.seen_tokens is not None and gather_seen_tokens:
            self.seen_tokens.gather(idx_view, padded_bs)

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