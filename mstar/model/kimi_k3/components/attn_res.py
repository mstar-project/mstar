"""Block attention residuals (AttnRes), Kimi K3's residual stream.

Instead of one running sum, a layer reads its input as a softmax mixture of the residual
stack (the stream's value at each block start, ``attn_res_block_size`` layers apart) and the
running partial sum since the last block start. A candidate ``v`` scores
``<rmsnorm_noweight(v), res_norm.weight * res_proj.weight>`` (no 1/sqrt(d)); the mixture is
of the unnormalized candidates, in fp32. Two reads per layer (before attention, before the
MLP) and one before the final norm.

The stack is per token across depth, so it lives only inside one forward: nothing persists
across steps and no resource holds it. It is one token-major ``(T, num_blocks, D)`` tensor a
forward allocates and fills in place (``stack[:, m] = stream`` at a block start), as FlashInfer's
``kimi_k3_attn_res`` reads it; ``attn_res_read`` takes its block-major view
(``stack.transpose(0, 1)``, the kernels reading by stride), so a read takes ``stack[:, :m]`` where
it lies. A read site with its norm and block write is ``attn_res_read_norm``: on SM100/SM103
FlashInfer's one launch (the residual add, the block write, the mix and the output RMSNorm),
elsewhere the read below, the write and the norm. On a GPU the read is ``attn_res_kernels`` (two small kernels at
decode sizes, one past ``FUSED_MIN_ROWS`` rows), with the residual add ahead of it fused
(``addend``); elsewhere the torch chain below. Follows ``_apply_attn_res`` in the checkpoint's
``modeling_kimi_linear.py`` and merceod's #257.
"""
from __future__ import annotations

import logging

import torch

from mstar.model.kimi_k3.components import attn_res_kernels as _k

logger = logging.getLogger(__name__)

# tests switch this off to run the torch chain on a GPU
KERNELS_ENABLED = True

# FlashInfer's kimi_k3_attn_res: its programs are for K3's hidden size and up to 8 stack blocks, bf16
FI_HIDDEN = 7168
FI_MAX_BLOCKS = 8
# launched with programmatic dependent launch: the program waits on the grid before it in-kernel. Its own successors
# (the next projection's GEMM) launch without it, so none can read the output before the program finishes
FI_PDL = True
_FI: dict[torch.device, object] = {}
# call shapes FlashInfer refused on the host (no program, a precondition), each logged once: the Triton read serves
_FI_REFUSED: set[tuple] = set()


def score_weight(norm_weight: torch.Tensor, proj_weight: torch.Tensor) -> torch.Tensor:
    """A read site's query, ``<site>_norm.weight [D] * <site>_proj.weight [1, D]``, in fp32."""
    return norm_weight.float() * proj_weight.reshape(-1).float()


def attn_res_read_torch(
    stack: torch.Tensor, m: int, stream: torch.Tensor, norm_w: torch.Tensor,
    proj_w: torch.Tensor, eps: float,
) -> torch.Tensor:
    """The reference chain: ``stream (T, D)`` mixed with ``stack[:m]`` (``(m, T, D)``)."""
    # fp32 in the reference; keep it so under the engine's bf16 autocast
    with torch.autocast(device_type=stream.device.type, enabled=False):
        v = torch.cat((stack[:m], stream.unsqueeze(0)), dim=0).float()  # (m + 1, T, D)
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        probs = (k * score_weight(norm_w, proj_w)).sum(-1).softmax(0)  # (m + 1, T)
        mixed = torch.einsum("mt,mtd->td", probs, v)
    return mixed.to(stream.dtype)


def _use_kernels(t: torch.Tensor) -> bool:
    return KERNELS_ENABLED and _k.available() and (t.is_cuda or _k.INTERPRETING)


@torch.library.custom_op("mstar::kimi_attn_res_read", mutates_args=())
def _read_op(stack: torch.Tensor, m: int, prefix: torch.Tensor, norm_w: torch.Tensor,
             proj_w: torch.Tensor, eps: float) -> torch.Tensor:
    return _k.attn_res_read_triton(stack, m, prefix, norm_w, proj_w, eps)[0]


@_read_op.register_fake
def _(stack, m, prefix, norm_w, proj_w, eps):
    return torch.empty_like(prefix, memory_format=torch.contiguous_format)


@torch.library.custom_op("mstar::kimi_attn_res_add_read", mutates_args=())
def _add_read_op(stack: torch.Tensor, m: int, prefix: torch.Tensor, addend: torch.Tensor,
                 norm_w: torch.Tensor, proj_w: torch.Tensor, eps: float,
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    return _k.attn_res_read_triton(stack, m, prefix, norm_w, proj_w, eps, addend)


@_add_read_op.register_fake
def _(stack, m, prefix, addend, norm_w, proj_w, eps):
    dtype = torch.result_type(prefix, addend)
    return prefix.new_empty(prefix.shape, dtype=dtype), prefix.new_empty(prefix.shape, dtype=dtype)


def attn_res_read(
    stack: torch.Tensor | None, m: int, prefix: torch.Tensor, norm_w: torch.Tensor,
    proj_w: torch.Tensor, eps: float, addend: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A read site: ``(mixed, stream)``. The stream is ``prefix (T, D)``, or ``prefix +
    addend`` (the residual add, rounded to its dtype as torch's add rounds it); ``mixed`` is
    the stream mixed with ``stack[:m]`` (block-major, ``(>= m, T, D)``), the stream itself
    when ``m == 0`` (a softmax over one candidate). ``norm_w [D]``, ``proj_w [1, D]``: the
    site's ``*_res_norm`` and ``*_res_proj`` weights."""
    if m == 0:
        stream = prefix if addend is None else prefix + addend
        return stream, stream
    if _use_kernels(prefix):
        if torch.compiler.is_compiling():
            if addend is None:
                return _read_op(stack, m, prefix, norm_w, proj_w, eps), prefix
            return _add_read_op(stack, m, prefix, addend, norm_w, proj_w, eps)
        return _k.attn_res_read_triton(stack, m, prefix, norm_w, proj_w, eps, addend)
    stream = prefix if addend is None else prefix + addend
    return attn_res_read_torch(stack, m, stream, norm_w, proj_w, eps), stream


def _fi(device: torch.device):
    """FlashInfer's ``kimi_k3_attn_res`` module where it has programs for ``device`` (SM100 / SM103), else None."""
    if device not in _FI:
        mod = None
        if device.type == "cuda" and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3)):
            try:
                from flashinfer import kimi_k3_attn_res as mod

                if not mod.generated_program_available(device):
                    mod = None
            except Exception:  # noqa: BLE001 - no FlashInfer, or one without the programs: the Triton read
                mod = None
        _FI[device] = mod
    return _FI[device]


def _device_fault(exc: Exception) -> bool:
    """A CUDA error (sticky: nothing after it runs), as against a program that failed to build or load."""
    fault = getattr(torch, "AcceleratorError", None)
    return (fault is not None and isinstance(exc, fault)) or "CUDA error" in str(exc)


def fi_serves(stack: torch.Tensor, prefix: torch.Tensor, weights: tuple[torch.Tensor, ...],
              addend: torch.Tensor | None = None) -> bool:
    """FlashInfer's one-launch read serves this site: bf16 at K3's hidden size on SM100 / SM103, at most 8 stack
    blocks, unit stride on the hidden dim (its host checks: anything else raises there)."""
    return (prefix.is_cuda and prefix.dtype == torch.bfloat16 and prefix.shape[-1] == FI_HIDDEN
            and stack.shape[1] <= FI_MAX_BLOCKS and stack.dtype == torch.bfloat16 and stack.stride(2) == 1
            and prefix.stride(1) == 1 and all(w.dtype == torch.bfloat16 and w.is_contiguous() for w in weights)
            and (addend is None or (addend.dtype == torch.bfloat16 and addend.shape == prefix.shape
                                    and addend.stride(1) == 1))
            and _fi(prefix.device) is not None)


@torch.compiler.disable
def _fi_read(stack, m, prefix, addend, norm_w, proj_w, eps, out_w, out_eps, write) -> torch.Tensor:
    """FlashInfer's read: ``prefix += addend`` in place, ``stack[:, write] = prefix`` (``write >= 0``), the mix of
    ``stack[:, :m]`` and ``prefix``, RMS-normed by ``out_w``. Eager inside a compiled region (it mutates its
    inputs)."""
    out = torch.empty(prefix.shape, dtype=prefix.dtype, device=prefix.device)
    _fi(prefix.device).kimi_k3_attn_res(
        prefix, addend, stack, norm_w, proj_w, out_w, out, num_blocks=m, block_write_idx=write, eps=eps,
        output_norm_eps=out_eps, enable_pdl=FI_PDL)
    return out


def attn_res_read_norm(
    stack: torch.Tensor, m: int, prefix: torch.Tensor, norm_w: torch.Tensor, proj_w: torch.Tensor, eps: float,
    out_norm, addend: torch.Tensor | None = None, write: int = -1, flashinfer: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A read site and its norm: ``(out_norm(mixed), stream)`` with ``mixed`` as ``attn_res_read`` gives it over
    the token-major ``stack (T, >= m, D)``, and with ``write >= 0`` the stream written into ``stack[:, write]``
    (a block start: ``write == m``). ``out_norm``: the RMSNorm after the read (its weight and epsilon).

    With ``flashinfer`` where it serves (``fi_serves``) it is one launch, and with ``addend`` the stream is
    ``prefix`` itself, the addend added in place: the caller passes a ``prefix`` this forward owns. The norm
    runs on the fp32 mix there (bf16 rounding before the norm here): within a bf16 ulp of each other."""
    proj = proj_w.reshape(-1)
    weights = (norm_w, proj, out_norm.weight)
    if flashinfer and getattr(out_norm, "_weight_bias", 0.0) == 0.0 and fi_serves(stack, prefix, weights, addend):
        key = (tuple(prefix.shape), stack.shape[1], m, write, addend is not None)
        if key not in _FI_REFUSED:
            try:
                out = _fi_read(stack, m, prefix, addend, norm_w, proj, eps, out_norm.weight,
                               out_norm.variance_epsilon, write)
                return out, prefix
            except (ValueError, NotImplementedError) as exc:  # refused on the host, before any launch
                _FI_REFUSED.add(key)
                logger.warning("FlashInfer's kimi_k3_attn_res refused %s (%s: %s); the Triton read serves it",
                               key, type(exc).__name__, str(exc)[:200])
            except Exception as exc:  # noqa: BLE001 - its program didn't build or load: off for the process
                if _device_fault(exc):
                    raise
                # B300 slot 10 (flashinfer 0.7.2): the wheel ships no generated sources, so the JIT build fails
                # ("Ninja build failed ... missing") before any launch, for most call shapes
                _FI[prefix.device] = None
                logger.warning("FlashInfer's kimi_k3_attn_res can't run here (%s: %s); the Triton read serves every "
                               "AttnRes site from now on", type(exc).__name__, str(exc)[:300])
    mixed, stream = attn_res_read(stack.transpose(0, 1), m, prefix, norm_w, proj_w, eps, addend=addend)
    if write >= 0:
        stack[:, write].copy_(stream)
    return out_norm(mixed), stream
