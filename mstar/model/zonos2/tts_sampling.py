"""Multi-codebook TTS sampling for Zonos2.

This is a port of ``../ZONOS2/python/zonos2/tts/sampler.py``. The reference
``sample_tts`` returns Python lists, which forces a device sync.
:func:`sample_frame` returns tensors instead. The forward of the LLM submodule
calls it inside the CUDA graph, with no ``.tolist()`` sync on the GPU thread.
It maps per-codebook logits ``(B, C, V)`` to frames ``(B, C + 1)``. Each frame
holds the sampled audio codes and a text placeholder. One call handles ``B``
requests.

A stateless RNG keeps the result reproducible under batching. The last draw is
a Gumbel-max over noise. :func:`_deterministic_uniform` keys that noise only on
``(seed, step, codebook, vocab)``, not on the batch position of the request. A
request therefore draws the same frame at a given step, whatever other requests
share its batch. A stateful ``torch.Generator`` for each request cannot do
this, because it becomes position-dependent once the code vectorises it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from mstar.engine.resources.spec import ResourceReqConfig

# The request kwargs that override sampling, as the reference names them.
REQUEST_KNOBS = (
    "temperature", "topk", "top_p", "min_p", "ignore_eos",
    "repetition_window", "repetition_penalty", "repetition_codebooks",
)


@dataclass
class TTSSamplingParams(ResourceReqConfig):
    """Sampling parameters for one request. The defaults match the reference.

    One instance holds the deployment defaults; :meth:`for_request` derives each
    request's copy, which travels on ``request_info.resource_configs``.
    """

    temperature: float = 1.15
    topk: int = 106
    top_p: float = 0.0
    min_p: float = 0.18
    max_tokens: int = 1024
    ignore_eos: bool = False
    repetition_window: int = 50
    repetition_penalty: float = 1.2
    # The repetition penalty applies to codebooks 0 to repetition_codebooks - 1.
    # A negative value applies it to all codebooks.
    repetition_codebooks: int = 8

    def for_request(self, model_kwargs: dict, max_window: int) -> "TTSSamplingParams":
        """This request's params: the defaults with its kwargs applied.

        Normalizes as the reference does (``engine/sample.py``): ``topk < 1``
        disables top-k, ``top_p`` clamps to [0, 1], ``min_p`` and the window to
        >= 0, the penalty to >= 1, and a negative codebook count means all.
        ``temperature <= 0`` is greedy. Raises ``ValueError`` on a bad value.
        """
        given = {k: model_kwargs[k] for k in REQUEST_KNOBS if model_kwargs.get(k) is not None}
        out = replace(self, **{k: _coerce(k, v, type(getattr(self, k))) for k, v in given.items()})
        out.topk = max(out.topk, 0)
        out.top_p = min(max(out.top_p, 0.0), 1.0)
        out.min_p = max(out.min_p, 0.0)
        out.repetition_window = max(out.repetition_window, 0)
        out.repetition_penalty = max(out.repetition_penalty, 1.0)
        out.repetition_codebooks = max(out.repetition_codebooks, -1)
        if out.repetition_window > max_window:
            raise ValueError(
                f"repetition_window is {out.repetition_window}; this server allows at "
                f"most {max_window}."
            )
        return out


_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


def _coerce(name: str, value, kind: type):
    """Convert a request kwarg to its field's type, or raise ``ValueError``."""
    if kind is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"{name} must be true or false, got {value!r}.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    if kind is int:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{name} must be an integer, got {value!r}.")
        value = int(value)
        # The sampler stores these in int64 tensors; a wider value fails its whole batch.
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError(f"{name} must fit in a signed 64-bit integer, got {value}.")
        return value
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return float(value)


@dataclass
class SamplingRows:
    """Per-row sampling knobs ``(B,)`` for :func:`sample_frame`.

    Tensors, not Python scalars, so one captured graph serves every request's
    settings. A disabled filter is encoded in-band: ``topk <= 0``,
    ``top_p`` outside (0, 1), ``min_p <= 0``, a penalty of 1.
    """

    temperature: torch.Tensor         # float32
    topk: torch.Tensor                # int64
    top_p: torch.Tensor               # float32
    min_p: torch.Tensor               # float32
    repetition_penalty: torch.Tensor  # float32

    @classmethod
    def uniform(cls, params: TTSSamplingParams, B: int, device) -> "SamplingRows":
        def full(v, dtype=torch.float32):
            return torch.full((B,), v, dtype=dtype, device=device)
        return cls(
            temperature=full(params.temperature),
            topk=full(params.topk, torch.int64),
            top_p=full(params.top_p),
            min_p=full(params.min_p),
            repetition_penalty=full(params.repetition_penalty),
        )


def _rowwise(v: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """View a ``(B,)`` tensor so it broadcasts against ``like``'s rows."""
    return v.view(-1, *([1] * (like.ndim - 1)))


def apply_top_p(probs: torch.Tensor, p: float | torch.Tensor) -> torch.Tensor:
    """Apply a nucleus (top-p) filter. ``p`` is a float or one per row ``(B,)``.

    ``p`` outside (0, 1) leaves that row unchanged.
    """
    if not isinstance(p, torch.Tensor):
        if p <= 0.0 or p >= 1.0:
            return probs
    else:
        p = _rowwise(p, probs)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort = probs_sort.masked_fill(mask, 0.0)
    filtered = probs.scatter(-1, probs_idx, probs_sort)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    if not isinstance(p, torch.Tensor):
        return filtered
    return torch.where((p > 0.0) & (p < 1.0), filtered, probs)


def apply_min_p(probs: torch.Tensor, min_p: float | torch.Tensor) -> torch.Tensor:
    """Drop the tokens below ``min_p * max_prob``. ``min_p`` may be per row.

    ``min_p <= 0`` leaves that row unchanged.
    """
    if not isinstance(min_p, torch.Tensor):
        if min_p <= 0.0:
            return probs
    else:
        min_p = _rowwise(min_p, probs)
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    filtered = probs.masked_fill(probs < (min_p * top_probs), 0.0)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    if not isinstance(min_p, torch.Tensor):
        return filtered
    return torch.where(min_p > 0.0, filtered, probs)


def apply_repetition_penalty(
    logits: torch.Tensor,
    repetition_token_ids: torch.Tensor | None,
    repetition_penalty: float | torch.Tensor,
) -> torch.Tensor:
    """Apply the repetition penalty to each codebook.

    ``repetition_token_ids`` is ``(B, C, W)``: the recent token ids of each
    codebook. The function ignores a token id of ``-1`` or one out of range. To
    exclude a codebook from the penalty, set its ids to ``-1``. The penalty is a
    float or one per row ``(B,)``; 1 leaves a row unchanged.
    """
    if repetition_token_ids is None:
        return logits
    if not isinstance(repetition_penalty, torch.Tensor) and repetition_penalty == 1.0:
        return logits
    if repetition_token_ids.numel() == 0:
        return logits

    B, C, V = logits.shape
    safe_ids = repetition_token_ids.clamp(min=0, max=V - 1).long()
    valid = (repetition_token_ids >= 0) & (repetition_token_ids < V)

    counts = torch.zeros((B, C, V), dtype=torch.int32, device=logits.device)
    counts.scatter_add_(-1, safe_ids, valid.to(torch.int32))
    repeated = counts > 0

    if isinstance(repetition_penalty, torch.Tensor):
        penalty = _rowwise(repetition_penalty, logits).clamp(min=1.0)
    else:
        penalty = max(repetition_penalty, 1.0)
    adjusted = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(repeated, adjusted, logits)


_M32 = 0xFFFFFFFF


def _i32(k: int) -> int:
    """``k`` as a signed int32; Triton rejects a larger literal times an index.

    The low 32 bits of ``x * k`` are unchanged, so the hash after ``& _M32`` is too.
    """
    return k - (1 << 32) if k >= 1 << 31 else k


def _fmix32(h: torch.Tensor) -> torch.Tensor:
    """Apply the MurmurHash3 ``fmix32`` finalizer to uint32 values in int64.

    Every value stays non-negative and less than ``2**32``. The only exception
    is the transient multiply. Its overflow past int64 wraps two's-complement,
    and the code masks it back to 32 bits immediately. The result therefore
    agrees with the uint32 reference, and the ``>>`` shifts act as logical
    shifts.
    """
    h = h & _M32
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & _M32
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & _M32
    h = h ^ (h >> 15)
    return h & _M32


def _deterministic_uniform(
    B: int, C: int, V: int,
    seed: int | torch.Tensor, steps: torch.Tensor,
    device: torch.device, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return reproducible ``U[0, 1)`` noise of shape ``(B, C, V)``.

    A counter-based hash keys the noise only on ``(seed, step, codebook,
    vocab)``. It does not use the batch position. The noise for request ``b`` at
    ``steps[b]`` is therefore the same alone or in any batch. ``steps`` is the
    step index of each request, of shape ``(B,)``. ``seed`` is one int for the
    batch, or one per request ``(B,)``.
    """
    v = torch.arange(V, device=device, dtype=torch.int64).view(1, 1, V)
    c = torch.arange(C, device=device, dtype=torch.int64).view(1, C, 1)
    s = steps.to(device=device, dtype=torch.int64).view(B, 1, 1)
    if isinstance(seed, torch.Tensor):
        base = seed.to(device=device, dtype=torch.int64).view(B, 1, 1) & _M32
    else:
        base = int(seed) & _M32
    # The chained fmix32 rounds mix every field into the result.
    h = (v * _i32(0x27D4EB2F)) & _M32
    h = _fmix32(h ^ ((c * _i32(0x85EBCA77)) & _M32))
    h = _fmix32(h ^ ((s * _i32(0xC2B2AE3D)) & _M32))
    h = _fmix32(h ^ base)
    return (h.to(torch.float64) / 4294967296.0).to(dtype)


def sample_frame(
    logits: torch.Tensor,
    params: TTSSamplingParams | SamplingRows,
    repetition_token_ids: torch.Tensor | None = None,
    text_placeholder: int = 0,
    seed: int | torch.Tensor | None = None,
    steps: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Sample one frame for each request from the per-codebook logits.

    Args:
        logits: the per-codebook logits ``(B, C, V)`` of the current step.
        params: one set of knobs for the whole batch, or :class:`SamplingRows`
            with one per row.
        repetition_token_ids: the recent tokens ``(B, C, W)``, or None. A ``-1``
            marks a padded or ignored slot.
        text_placeholder: the value to write into the appended text column.
        seed: the base RNG seed, one int for the batch or one per request
            ``(B,)``. ``None`` uses the global CUDA generator, which is not
            reproducible; the server never passes it, because a failed graph
            capture on torch 2.9 can leave that generator unusable.
        steps: the step index of each request, of shape ``(B,)``. An int or
            ``None`` maps to 0. With ``seed`` set, ``(seed, step)`` fully
            determines the draw of a request, whatever its batch position.
            Batched sampling is therefore reproducible for each request.

    Returns:
        The int64 frames ``(B, C + 1)``: ``[cb0, ..., cb_{C-1},
        text_placeholder]``.
    """
    B, C, V = logits.shape
    device = logits.device

    rows = (
        params if isinstance(params, SamplingRows)
        else SamplingRows.uniform(params, B, device)
    )

    # Every filter runs on every row, with in-band "disabled" values, so there
    # is no Python branch on a knob and one captured graph serves any mix.
    logits = apply_repetition_penalty(
        logits, repetition_token_ids, rows.repetition_penalty
    )
    greedy_ids = torch.argmax(logits, dim=-1)  # (B, C), for temperature <= 0

    temperature = _rowwise(rows.temperature, logits)
    logits = logits / temperature.clamp(min=1e-8)

    # Top-k as a per-row threshold: the k-th largest logit. A disabled row
    # (k <= 0 or k >= V) takes the smallest logit, which masks nothing.
    k = _rowwise(rows.topk.to(torch.int64), logits).expand(B, C, 1)
    k = torch.where((k > 0) & (k < V), k, V)
    kth = torch.sort(logits, dim=-1, descending=True).values.gather(-1, k - 1)
    logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    probs = apply_top_p(probs, rows.top_p)
    probs = apply_min_p(probs, rows.min_p)

    # Reproducible Gumbel-max. ``argmax(log p + Gumbel)`` samples in
    # proportion to ``probs``, as ``multinomial`` does. The noise comes from
    # the stateless RNG above, so this vectorises across the batch without a
    # Generator for each request.
    if steps is None:
        steps_t = torch.zeros(B, dtype=torch.int64, device=device)
    elif isinstance(steps, int):
        steps_t = torch.full((B,), steps, dtype=torch.int64, device=device)
    else:
        steps_t = steps.to(device=device, dtype=torch.int64).reshape(-1)

    if seed is None:
        u = torch.rand((B, C, V), device=device, dtype=probs.dtype)
    else:
        u = _deterministic_uniform(B, C, V, seed, steps_t, device, probs.dtype)

    eps = 1e-20
    gumbel = -torch.log(-torch.log(u.clamp(eps, 1.0 - eps)))
    # log(0) is -inf on a filtered token, and -inf plus a finite Gumbel
    # stays -inf. The argmax never selects it, and no NaN appears.
    next_ids = torch.argmax(probs.clamp_min(0).log() + gumbel, dim=-1)  # (B, C)

    # A strong filter can set a whole row to zero. The code then falls back
    # to greedy: the argmax of the filtered logits. It applies the fallback
    # unconditionally, so there is no ``bool(invalid.any())`` host sync.
    # Where no row is invalid, ``torch.where`` returns ``next_ids``
    # unchanged. The result is identical to the guarded form, and it is safe
    # for graph capture.
    invalid = probs.sum(dim=-1) <= 0  # (B, C)
    next_ids = torch.where(invalid, logits.argmax(dim=-1), next_ids)
    next_ids = torch.where(temperature.view(B, 1) <= 0, greedy_ids, next_ids)

    text_col = torch.full(
        (B, 1), text_placeholder, dtype=next_ids.dtype, device=device
    )
    return torch.cat([next_ids, text_col], dim=-1)  # (B, C + 1)
