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


# MSTAR_SAMPLER_CFG_CACHE (default ON): cache the per-batch device config
# tensors in Sampler.sample keyed by batch membership. Each uncached call
# does SIX pageable H2D copies, each forcing a cudaStreamSynchronize that
# drains the in-flight decode pipeline.
# Off-switch kept for A/B only; outputs are byte-identical either way.
import os as _os  # noqa: E402  (feature-flag import kept beside its rationale)

# DEFAULT OFF after evaluation: killing the syncs regressed end-to-end
# performance despite being faster in isolation and token-identical. The
# blocked GPU-thread RELEASED THE GIL during those pipeline-drain waits,
# and the main thread's per-step postprocess Python ran in that window;
# without the waits the two threads contend and wall time gets worse.
# Lesson: on this two-thread GIL architecture, removing GPU-thread waits
# only pays off if main-thread Python work is removed or moved off-GIL
# FIRST. Keep for re-test after postprocess work shrinks.
_SAMPLER_CFG_CACHE = _os.environ.get(
    "MSTAR_SAMPLER_CFG_CACHE", "0"
).strip().lower() in ("1", "true", "yes", "on")

# MSTAR_SAMPLER_CFG_CACHE_V2 (default OFF): membership-CHURN-proof successor.
# The V1 cache keys on tuple(request_ids) — exact membership — so under
# closed-loop admission churn the cache misses nearly every step and the six
# pageable H2D syncs come back (profiling showed this cost a significant
# share of per-step wall time). V2 holds each config field in a PERSISTENT
# per-request-SLOT device tensor (written once at admission/set_config via
# pinned non_blocking, no per-step H2D) and assembles the batch by
# index_select on a slot-index tensor — the
# slot-index is the only per-step device object, rebuilt sync-free (pinned +
# non_blocking) on a membership change, else cached. rand_offset is an
# on-device per-slot counter (index_add each step). No pageable sync on any
# path. Byte-identical outputs to V1/off (same per-rid values, same
# u(T)=T-1 rand_offset). Supersedes _SAMPLER_CFG_CACHE when on.
_SAMPLER_CFG_CACHE_V2 = _os.environ.get(
    "MSTAR_SAMPLER_CFG_CACHE_V2", "0"
).strip().lower() in ("1", "true", "yes", "on")

# MSTAR_ARGMAX_FAST (default OFF): when EVERY request in the batch is greedy
# (temperature == 0) and no repetition penalty is active, the next token is just
# argmax(logits) per row. Skip the per-batch config-tensor assembly (the six
# pageable H2D copies / their syncs), the fused temperature+softmax Triton
# kernel, and the FlashInfer top-k/top-p sampler entirely (vLLM does this —
# sampler.py:239). Byte-identical token VALUES to the current greedy path, which
# builds a one-hot at argmax and samples it deterministically: torch.argmax and
# the kernel's tl.argmax both break ties to the lowest index, and dtype is int32
# to match FlashInfer's output. Only the eager ``Sampler`` takes this branch —
# the CUDA-graph ``CudaGraphableSampler`` encodes greedy as (temp=1, top_k=1)
# and cannot branch on CPU values, so it is deliberately left untouched (the
# flag is read once in ``Sampler.sample``, never inside a captured region).
_ARGMAX_FAST = _os.environ.get(
    "MSTAR_ARGMAX_FAST", "0"
).strip().lower() in ("1", "true", "yes", "on")


# MSTAR_SLIM_SAMPLE (default OFF): per-step Python-body reduction around
# sampling. Two things, both output-preserving:
#
#  1. ``Sampler.sample`` fuses the up-to-six separate ``any()``/``all()``
#     generator scans over the per-request config list (the ARGMAX_FAST
#     eligibility check, plus any_rep_pen/any_greedy/any_top_k_zero/
#     all_top_k_zero) into one pass — see ``_scan_sampling_configs``.
#  2. The engine's two "sample the batched logits, then build a per-rid
#     new_token map" bodies — ``kv_cache_engine._execute_batched`` (pure
#     eager forward) and ``cuda_graph_runner._sample_and_remap`` (CUDA-graph
#     post-replay) — had drifted into two independently-maintained copies of
#     the same slice/index_select + sample + clone + split logic (down to an
#     identical FlashInfer-buffer-aliasing comment in both files). They now
#     both call ``sample_batched_and_unpack`` in this module.
#
# Read per-call (not cached) so a runtime flag flip applies immediately,
# matching the ``_envflag`` convention in qwen3_omni_model.py — no
# register_cache_clear needed since nothing here is cached across calls.
# Flag off takes the untouched original code path at every call site: same
# operations, same order, so outputs are byte-identical either way.
def _slim_sample_enabled() -> bool:
    return _os.environ.get(
        "MSTAR_SLIM_SAMPLE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")


def _scan_sampling_configs(
    configs: list["SamplingConfig"],
) -> tuple[bool, bool, bool, bool, bool]:
    """One pass over ``configs`` computing every per-batch predicate the
    ``Sampler.sample`` hot path needs, replacing up to six separate
    ``any()``/``all()`` scans (each re-walking the same B-sized list) with
    one. B is small (<=32 in practice) so the per-call saving is a handful
    of microseconds, but this runs on every decode step — pure interpreter
    overhead, not device work — so it composes with the rest of the
    per-step CPU cuts.

    Returns:
        ``(all_greedy, any_greedy, any_rep_pen, any_top_k_zero, all_top_k_zero)``
    """
    all_greedy = True
    any_greedy = False
    any_rep_pen = False
    any_top_k_zero = False
    all_top_k_zero = True
    for c in configs:
        if c.temperature == 0:
            any_greedy = True
        else:
            all_greedy = False
        if c.repetition_penalty != 1.0:
            any_rep_pen = True
        if c.top_k == 0:
            any_top_k_zero = True
        else:
            all_top_k_zero = False
    return all_greedy, any_greedy, any_rep_pen, any_top_k_zero, all_top_k_zero


def sample_batched_and_unpack(
    sampler: "BaseSampler",
    request_ids: list[str],
    batched_logits: torch.Tensor,
    slot_map: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, dict[str, list[torch.Tensor]]]]:
    """Shared sample+unpack body for the two engine call sites that sample a
    stacked ``[padded_bs, V]`` batched-logits tensor and build a per-rid
    ``new_token`` map: ``kv_cache_engine._execute_batched`` (pure eager
    forward) and ``cuda_graph_runner._sample_and_remap`` (CUDA-graph
    post-replay, ``__batched_logits__`` fast path). Both independently
    duplicated: slice/index_select to the real request rows, call
    ``sampler.sample()``, ``.clone()`` to break FlashInfer's reused
    sampling-output-buffer alias (the same tokens-doubling bug is
    reachable from either call site — see the clone comments this
    replaces), ``.split(1)`` into per-rid views, and zip into a
    ``{rid: {"new_token": [view]}}`` map. One copy removes that drift risk
    and cuts one per-wave Python body.

    Args:
        sampler: anything with a ``.sample(request_ids, logits)`` method —
            in practice the eager ``Sampler`` (sampling itself is always
            eager at both call sites; ``cuda_graph_runner`` only replays the
            *forward* under CUDA graph, then samples afterwards in Python).
        request_ids: real (non-dummy) request ids for this wave.
        batched_logits: ``[padded_bs, V]`` stacked logits from one batched
            forward.
        slot_map: optional per-request source row into ``batched_logits``
            (MSTAR_MIXED_SPLIT_ATTN chunk rows); ``None`` uses the first
            ``len(request_ids)`` rows in order.

    Returns:
        ``(sampled, new_token_map)`` — ``sampled`` is the cloned ``[B]``
        token tensor (callers that also need the whole-batch tensor, e.g.
        MSTAR_DIRECT_FEED, get it without a second sample/clone);
        ``new_token_map`` is ``{rid: {"new_token": [view]}}`` where each
        view is a row-slice of ``sampled`` (no extra copy).
    """
    if slot_map is not None:
        idx = torch.tensor(
            slot_map, dtype=torch.long, device=batched_logits.device
        )
        stacked_logits = batched_logits.index_select(0, idx)
    else:
        stacked_logits = batched_logits[: len(request_ids)]
    sampled = sampler.sample(request_ids, stacked_logits).clone()
    new_token_map = {
        rid: {"new_token": [view]}
        for rid, view in zip(request_ids, sampled.split(1), strict=True)
    }
    return sampled, new_token_map


def _refresh_sampler_flags() -> None:
    """Re-read the cache flags at runtime (safe — both
    caches are semantics-free; flipping only changes assembly, not values.
    MSTAR_ARGMAX_FAST is likewise output-preserving: it only fires on all-greedy
    batches, where its argmax equals the sampled token either way)."""
    global _SAMPLER_CFG_CACHE, _SAMPLER_CFG_CACHE_V2, _ARGMAX_FAST
    _SAMPLER_CFG_CACHE = _os.environ.get(
        "MSTAR_SAMPLER_CFG_CACHE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    _SAMPLER_CFG_CACHE_V2 = _os.environ.get(
        "MSTAR_SAMPLER_CFG_CACHE_V2", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    _ARGMAX_FAST = _os.environ.get(
        "MSTAR_ARGMAX_FAST", "0"
    ).strip().lower() in ("1", "true", "yes", "on")




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
                "Calling add_tokens on an uninitialized SeenTokenMask, i.e., "
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
    # Per-batch-membership cache of the six device config tensors (see
    # sample()). Invalidated on set_config; bounded in sample().
    _batch_cfg_cache: dict = field(default_factory=dict)
    # upstream renamed TPCommGroup -> CommGroup (#154 SP generalization).
    tp_group: "CommGroup | None" = None  # noqa: F821
    # --- MSTAR_SAMPLER_CFG_CACHE_V2 slot machinery (lazy; unused when off) ---
    _v2_rid_to_slot: dict = field(default_factory=dict)   # rid -> slot int
    _v2_free_slots: list = field(default_factory=list)    # reusable slot ids
    _v2_n_slots: int = 0
    _v2_dev: dict = field(default_factory=dict)           # field -> device [n_slots]
    _v2_cpu: dict = field(default_factory=dict)           # field -> pinned CPU [n_slots]
    _v2_slot_index_cache: dict = field(default_factory=dict)  # membership -> device idx
    _v2_pinned_idx: "torch.Tensor | None" = None          # [RING, n] pinned staging
    _v2_idx_ring: int = 0                                 # rotates pinned staging rows
    _v2_ones: "torch.Tensor | None" = None                # device ones for rand advance
    _v2_stats: dict = field(default_factory=dict)         # periodic-logging counters
    _v2_was_on: bool = False                              # last-seen V2 flag (flip detect)

    # Static per-request config fields carried in per-slot device tensors, plus
    # the on-device rand_offset counter. (name, dtype, SamplingConfig attr|None)
    _V2_FIELDS = (
        ("temperature", "float32", "temperature"),
        ("top_k", "int32", "top_k"),
        ("top_p", "float32", "top_p"),
        ("r_pen", "float32", "repetition_penalty"),
        ("seed", "int64", "seed"),
        ("rand", "int64", None),  # advanced on-device each step
    )
    # Pinned slot-index staging depth: a fresh row per membership-miss build so
    # the CPU never overwrites a row whose non_blocking H2D is still in flight.
    # 4 covers the current spec pipeline (~1-deep speculation + double-buffer, so
    # at most ~3 sample() H2Ds can be outstanding). If the engine ever deepens
    # the pipeline past ~3, GROW this (power of two — the ring mask assumes it)
    # OR gate each row's reuse on a recorded CUDA event.
    _V2_IDX_RING = 4

    def _v2_ensure_capacity(self, need: int) -> None:
        """Grow the per-slot device+pinned tensors to hold >= ``need`` slots."""
        if need <= self._v2_n_slots:
            return
        new_n = max(need, 64, self._v2_n_slots * 2)
        pin = torch.cuda.is_available()
        for name, dt, _ in self._V2_FIELDS:
            dtype = getattr(torch, dt)
            dev = torch.zeros(new_n, dtype=dtype, device=self.device)
            cpu = torch.zeros(new_n, dtype=dtype, device="cpu", pin_memory=pin)
            if name in self._v2_dev:
                dev[: self._v2_n_slots] = self._v2_dev[name]
                cpu[: self._v2_n_slots] = self._v2_cpu[name]
            self._v2_dev[name] = dev
            self._v2_cpu[name] = cpu
        # Ring of pinned staging rows so the CPU never overwrites a row whose
        # non_blocking H2D (into a fresh cached device tensor) may still be
        # in flight — the membership-miss build runs ~every step at churn.
        self._v2_pinned_idx = torch.zeros(
            (self._V2_IDX_RING, new_n), dtype=torch.int64,
            device="cpu", pin_memory=pin,
        )
        self._v2_ones = torch.ones(new_n, dtype=torch.int64, device=self.device)
        # New slot ids become free (reused LIFO).
        self._v2_free_slots.extend(range(self._v2_n_slots, new_n))
        self._v2_n_slots = new_n

    def _v2_slot_for(self, rid: str) -> int:
        """Slot for ``rid``, assigning + initializing one on first use."""
        slot = self._v2_rid_to_slot.get(rid)
        if slot is not None:
            return slot
        if not self._v2_free_slots:
            self._v2_ensure_capacity(self._v2_n_slots + 1)
        slot = self._v2_free_slots.pop()
        self._v2_rid_to_slot[rid] = slot
        self._v2_write_slot(rid, slot)
        self._v2_inc("cfgv2_slot_miss")
        return slot

    def _v2_write_slot(self, rid: str, slot: int | None = None) -> None:
        """Write ``rid``'s current SamplingConfig into its slot, sync-free
        (pinned CPU write + single-element non_blocking H2D). rand_offset is
        seeded from the Python _step_offset so it matches V1/off exactly."""
        if slot is None:
            slot = self._v2_rid_to_slot.get(rid)
            if slot is None:
                return
        cfg = self._sampling_config[rid]
        vals = {
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "r_pen": cfg.repetition_penalty,
            "seed": cfg.seed,
            "rand": self._step_offset.get(rid, 0),
        }
        for name, v in vals.items():
            self._v2_cpu[name][slot] = v
            self._v2_dev[name][slot : slot + 1].copy_(
                self._v2_cpu[name][slot : slot + 1], non_blocking=True
            )
        # A slot's config changed → any cached slot-index for a membership
        # containing rid is still valid (slot id unchanged); only the field
        # tensors moved, which are read fresh each step. No cache invalidation
        # needed (unlike V1, whose cached VALUES would go stale).

    def _v2_free_slot(self, rid: str) -> None:
        slot = self._v2_rid_to_slot.pop(rid, None)
        if slot is not None:
            self._v2_free_slots.append(slot)
        # Drop cached slot-index tensors whose membership included rid.
        if self._v2_slot_index_cache:
            self._v2_slot_index_cache = {
                k: t for k, t in self._v2_slot_index_cache.items() if rid not in k
            }

    def _v2_resync_rand(self) -> None:
        """OFF→ON flip repair: while V2 is off the per-slot device rand counter
        freezes (index_add runs only on the V2 path) but _step_offset keeps
        advancing (its loop is unconditional), so a rid that survived the off
        interval would read a stale rand. Re-seed every live slot's rand from
        _step_offset on the transition (rids admitted while off have no slot yet
        and get seeded correctly at their first _v2_slot_for). Sync-free."""
        for rid, slot in self._v2_rid_to_slot.items():
            self._v2_cpu["rand"][slot] = self._step_offset.get(rid, 0)
            self._v2_dev["rand"][slot : slot + 1].copy_(
                self._v2_cpu["rand"][slot : slot + 1], non_blocking=True
            )

    def _v2_inc(self, key: str) -> None:
        self._v2_stats[key] = self._v2_stats.get(key, 0) + 1

    def _assemble_cfg_v2(self, request_ids: list[str], device):
        """Assemble the six batch config tensors by gathering per-slot device
        tensors — no pageable H2D on any path. Returns
        (temperature, top_k, top_p, r_pen, seed, rand_offset)."""
        slots = [self._v2_slot_for(rid) for rid in request_ids]
        key = tuple(request_ids)
        slot_index = self._v2_slot_index_cache.get(key)
        if slot_index is None:
            self._v2_inc("cfgv2_rebuilds")
            B = len(slots)
            ring = self._v2_idx_ring & (self._V2_IDX_RING - 1)
            self._v2_idx_ring += 1
            stage = self._v2_pinned_idx[ring, :B]
            stage.copy_(torch.tensor(slots, dtype=torch.int64))  # CPU->pinned
            slot_index = torch.empty(B, dtype=torch.int64, device=device)
            slot_index.copy_(stage, non_blocking=True)  # sync-free H2D (fresh dst)
            if len(self._v2_slot_index_cache) > 64:
                self._v2_slot_index_cache.clear()
            self._v2_slot_index_cache[key] = slot_index
        # Direct index_select gathers (no per-call lambda closure — profiling
        # showed it as a hot line at small batch sizes: sample() runs per step
        # so the closure allocated repeatedly per request).
        dev = self._v2_dev
        temperature = dev["temperature"].index_select(0, slot_index)
        top_k = dev["top_k"].index_select(0, slot_index)
        top_p = dev["top_p"].index_select(0, slot_index)
        r_pen = dev["r_pen"].index_select(0, slot_index)
        seed = dev["seed"].index_select(0, slot_index)
        # rand_offset: gather the PRE-advance value (u(T)=T-1, matches V1/off),
        # then advance the per-slot counter on-device (+1 per rid this step).
        rand_offset = dev["rand"].index_select(0, slot_index)
        self._v2_dev["rand"].index_add_(0, slot_index, self._v2_ones[: len(slots)])
        self._v2_inc("cfgv2_calls")
        if self._v2_stats["cfgv2_calls"] % 2000 == 0:
            # Surface the churn counters in server.log periodically:
            # cfgv2_rebuilds/cfgv2_calls ~ the miss rate the V1 cache suffered;
            # a healthy V2 keeps this high (churn) but sync-free (no regression).
            logger.warning(
                "CFGV2 %s slots=%d free=%d",
                dict(sorted(self._v2_stats.items())),
                self._v2_n_slots,
                len(self._v2_free_slots),
            )
        return temperature, top_k, top_p, r_pen, seed, rand_offset

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
        # V2: return the slot to the pool + drop its cached slot-indices.
        if self._v2_rid_to_slot:
            self._v2_free_slot(request_id)

    def set_config(self, request_id: str, **kwargs):
        # Config change with unchanged batch membership must not serve stale
        # cached tensors (see _batch_cfg_cache in sample()). Scoped to
        # memberships containing this rid — a global clear() combined with
        # prepare_batch's per-step per-rid set_config calls kept the cache
        # permanently empty (fixed in tandem with the change-detect there).
        if self._batch_cfg_cache:
            self._batch_cfg_cache = {
                k: v for k, v in self._batch_cfg_cache.items()
                if request_id not in k
            }
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
        # V2: the config VALUES for this rid changed → refresh its slot's
        # device tensors (sync-free). Slot id is unchanged, so cached
        # slot-index tensors stay valid; only the field values are updated.
        if request_id in self._v2_rid_to_slot:
            self._v2_write_slot(request_id)

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
        # MSTAR_SLIM_SAMPLE: fuse the boolean scans over `configs` below (this
        # ARGMAX_FAST check plus the four any()/all() calls further down) into
        # one pass over the batch instead of up to six. See
        # _scan_sampling_configs / MSTAR_SLIM_SAMPLE's module docstring.
        # Flag off leaves every original any()/all() call untouched below —
        # same statements, same order, byte-identical output either way.
        slim_sample = _slim_sample_enabled()
        if slim_sample:
            all_greedy, any_greedy, any_rep_pen, any_top_k_zero, all_top_k_zero = (
                _scan_sampling_configs(configs)
            )
        # MSTAR_ARGMAX_FAST: whole-batch greedy shortcut. When every request is
        # greedy (temperature == 0) and none carries a repetition penalty (which
        # would shift the argmax), the sampled token is exactly argmax(logits)
        # per row — no config tensors, no softmax, no FlashInfer. int32 matches
        # FlashInfer's output dtype so the return is drop-in for the full path.
        if _ARGMAX_FAST and (
            all_greedy if slim_sample
            else all(c.temperature == 0 for c in configs)
        ) and not (
            any_rep_pen if slim_sample
            else any(c.repetition_penalty != 1.0 for c in configs)
        ):
            tokens = logits.argmax(dim=-1).to(torch.int32)
            tokens = self._broadcast_tokens(tokens)
            # Keep the per-request RNG offset advancing in lockstep with the full
            # path: greedy never reads it, but if a request's temperature later
            # changes the cached/V2 rand is re-seeded from _step_offset, so it
            # must not fall behind while the fast path is active.
            for rid in request_ids:
                self._step_offset[rid] = self._step_offset.get(rid, 0) + 1
            return tokens
        # Per-batch config tensors. Building these from Python lists with
        # torch.tensor(..., device=...) does a PAGEABLE H2D copy each — torch
        # issues cudaStreamSynchronize per copy, and on the decode hot path
        # each such sync drains the whole in-flight pipeline (confirmed with
        # set_sync_debug_mode). Configs are static per request, so cache
        # the FIVE static tensors keyed by batch membership; rand_offset
        # advances by exactly 1 for every rid on every sample() call (the
        # loop at the bottom), so the cached device tensor is add_(1)'d
        # in-place — no H2D at steady state. Any membership change → new key
        # → one rebuild (its syncs are amortized to churn events).
        if _SAMPLER_CFG_CACHE_V2:
            # Churn-proof path: gather per-slot device tensors (no pageable H2D
            # on any path). Byte-identical values to the branches below.
            if not self._v2_was_on:
                # OFF→ON transition (incl. first-ever use, when the slot map is
                # empty and this is a no-op): re-sync frozen per-slot rand.
                self._v2_was_on = True
                self._v2_resync_rand()
            temperature, top_k, top_p, r_pen, seed, rand_offset = (
                self._assemble_cfg_v2(request_ids, logits.device)
            )
        else:
            self._v2_was_on = False
            key = tuple(request_ids)
            cached = (
                self._batch_cfg_cache.get(key) if _SAMPLER_CFG_CACHE else None
            )
            if cached is None:
                temperature = torch.tensor([c.temperature for c in configs], device=logits.device)
                top_k = torch.tensor([c.top_k for c in configs], device=logits.device, dtype=torch.int32)
                top_p = torch.tensor([c.top_p for c in configs], device=logits.device)
                r_pen = torch.tensor([c.repetition_penalty for c in configs], device=logits.device)
                seed = torch.tensor([c.seed for c in configs], device=logits.device, dtype=torch.long)
                rand_offset = torch.tensor(
                    [self._step_offset.get(rid, 0) for rid in request_ids],
                    device=logits.device, dtype=torch.long,
                )
                # Keep the cache from growing over a long server life: batch
                # membership churn creates a new key per admission wave. Bound it.
                if len(self._batch_cfg_cache) > 64:
                    self._batch_cfg_cache.clear()
                self._batch_cfg_cache[key] = (
                    temperature, top_k, top_p, r_pen, seed, rand_offset,
                )
            else:
                temperature, top_k, top_p, r_pen, seed, rand_offset = cached
                rand_offset.add_(1)

        if not slim_sample:
            any_rep_pen = any(c.repetition_penalty != 1.0 for c in configs)
            any_greedy = any(c.temperature == 0 for c in configs)
            any_top_k_zero = any(c.top_k == 0 for c in configs)
            all_top_k_zero = all(c.top_k == 0 for c in configs)
        # else: already computed by the fused _scan_sampling_configs() above.

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
        # small launches that add up measurably for larger batches with
        # repetition penalty enabled. Two options to fix:
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
    tp_group: "CommGroup | None" = None  # noqa: F821

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
        # cascades into garbled audio with no recovery, so we pay the small
        # in-place broadcast cost (no-op for trivial groups) to guarantee
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
    tp_group: "CommGroup | None" = None  # noqa: F821
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
        tp_group: "CommGroup | None" = None,  # noqa: F821
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
