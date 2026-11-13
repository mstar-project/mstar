"""Which kernel a cacheless self-attention runs, kept as data per GPU.

``ragged_attention`` is keyed by one rank's head geometry, its rows by the
longest query segment of a layout (the size class). A ``RaggedAttnBackend.AUTO``
resource reads its entry once when it is built. An eager plan then runs the row
its layout falls in, and a captured bucket the row of the layout that first
plans it (``_RaggedSelfManagerBase``). Where no table has the geometry it keeps
FlashInfer's ragged prefill and FlashInfer's own ``auto`` choice, as before; an
entry gives sizes no served model has run that with a ``"*"`` row naming ``flashinfer``.

Rows name ``flashinfer``, with ``{"backend": ...}`` for one of FlashInfer's
ragged backends, or ``cudnn`` (torch's cuDNN SDPA) or ``fa4`` (FlashAttention 4's
dense forward) over equal-length segments (``cudnn_wrappers``), which a layout of
unequal ones leaves for FlashInfer.
``python -m benchmark.kernels.sweep ragged_attention`` writes them.
"""

from __future__ import annotations

import logging

import torch

from mstar.engine.resources.attn.ragged.wrappers import padded_head_dim
from mstar.kernels import Choice, Impl, KernelOp, Target, Tolerance, register
from mstar.kernels.table import Rows
from mstar.platforms import current_platform

logger = logging.getLogger(__name__)


def _flashinfer_unavailable(target: Target) -> str | None:
    if target.device.type != "cuda" or not current_platform.is_cuda():
        return "FlashInfer's ragged prefill needs an NVIDIA GPU"
    try:
        import flashinfer  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def _cudnn_unavailable(target: Target) -> str | None:
    if target.device.type != "cuda" or not current_platform.is_cuda():
        return "torch's cuDNN attention needs an NVIDIA GPU"
    if not torch.backends.cudnn.is_available():
        return "torch has no cuDNN"
    if target.shape["num_qo_heads"] != target.shape["num_kv_heads"]:
        return "the cuDNN wrapper runs MHA only"
    if target.dtype not in (torch.bfloat16, torch.float16):
        return f"cuDNN attention takes bf16 / fp16, not {target.dtype}"
    head_dim = target.shape["head_dim"]
    try:
        padded = padded_head_dim(head_dim)
    except ValueError as exc:
        return str(exc)
    if padded != head_dim:
        # a compiled caller pads q / k / v to FlashInfer's head dim before the op
        # (`compile_op`), and the cuDNN wrapper views them at the unpadded one
        return f"head dim {head_dim} is padded to {padded} for FlashInfer, which the cuDNN wrapper doesn't take"
    return None


def _fa4_unavailable(target: Target) -> str | None:
    if target.device.type != "cuda" or not current_platform.is_cuda():
        return "FlashAttention 4 needs an NVIDIA GPU"
    capability = current_platform.cuda_capability(target.device)
    if capability is None or capability[0] != 10:
        return f"FlashAttention 4 runs on Blackwell (sm_100 / sm_103), not compute capability {capability}"
    if target.shape["num_qo_heads"] != target.shape["num_kv_heads"]:
        return "the FA4 wrapper runs MHA only"
    if target.dtype not in (torch.bfloat16, torch.float16):
        return f"FA4 takes bf16 / fp16, not {target.dtype}"
    head_dim = target.shape["head_dim"]
    padded = padded_head_dim(head_dim)
    if padded != head_dim:
        return f"head dim {head_dim} is padded to {padded} for FlashInfer, which the FA4 wrapper doesn't take"
    return _fa4_import_error()


def _fa4_import_error() -> str | None:
    try:
        import flash_attn.cute  # noqa: F401
    except Exception as exc:  # noqa: BLE001  # a broken install raises more than ImportError
        return f"flash-attn-4 (flash_attn.cute) is not importable: {exc!r}"
    return None


RAGGED_ATTENTION = register(KernelOp(
    name="ragged_attention",
    shape=("num_qo_heads", "num_kv_heads", "head_dim"),
    bucket="seq_len",
    impls=(
        # every kernel here sits ~0.2% (rel L2) from an fp32 reference at these sizes, bf16's own rounding
        Impl("flashinfer", check=_flashinfer_unavailable, tolerance=Tolerance(2e-2),
             doc="FlashInfer's ragged prefill; config {'backend': 'fa2' | 'cudnn' | 'auto'}"),
        Impl("cudnn", check=_cudnn_unavailable, tolerance=Tolerance(2e-2),
             doc="torch's cuDNN SDPA over equal-length segments"),
        Impl("fa4", check=_fa4_unavailable, tolerance=Tolerance(2e-2),
             doc="FlashAttention 4's dense forward over equal-length segments (Blackwell)"),
    ),
    doc="cacheless self-attention's kernel by one rank's head geometry and the longest query segment",
))


def kernel_rows(
    num_qo_heads: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device,
) -> Rows | None:
    """The table's rows for this geometry on ``device``; None where none applies,
    or where a row names an implementation that can't run here."""
    shape = {"num_qo_heads": num_qo_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim}
    entry = RAGGED_ATTENTION.lookup(dtype, shape, device)
    if entry is None:
        return None
    target = RAGGED_ATTENTION.target(dtype, shape, device)
    refused = {name: why for name, why in RAGGED_ATTENTION.unavailable(target).items() if name in entry.impls()}
    if refused:
        logger.warning("kernel table: ragged_attention %s rows name %s, which can't run here (%s); "
                       "keeping FlashInfer", RAGGED_ATTENTION.format_shape(shape), sorted(refused), refused)
        return None
    return entry.rows


def cudnn_rows(
    key: str, num_qo_heads: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device,
) -> Rows:
    """``backend: cudnn``'s rows, cuDNN at every size; raises where it can't run, so a
    config that names it fails at build rather than at its first request."""
    shape = {"num_qo_heads": num_qo_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim}
    why = RAGGED_ATTENTION.unavailable(RAGGED_ATTENTION.target(dtype, shape, device)).get("cudnn")
    if why is not None:
        raise ValueError(f"ragged attention {key!r} names backend 'cudnn', which can't run "
                         f"{RAGGED_ATTENTION.format_shape(shape)} {str(dtype).removeprefix('torch.')} here: {why}")
    return ((None, Choice("cudnn")),)

