"""Which kernel a cacheless self-attention runs, kept as data per GPU.

``ragged_attention`` is keyed by one rank's head geometry, its rows by the
longest query segment of a layout (the size class). A ``RaggedAttnBackend.AUTO``
resource reads its entry once when it is built: each wrapper then runs the row
its first layout falls in. Where no table has the geometry it keeps FlashInfer's
ragged prefill and FlashInfer's own ``auto`` choice, as before.

Rows name ``flashinfer``, with ``{"backend": ...}`` for one of FlashInfer's
ragged backends, or ``cudnn``, torch's cuDNN SDPA over equal-length segments
(``cudnn_wrappers``), which a layout of unequal ones leaves for FlashInfer.
``python -m benchmark.kernels.sweep ragged_attention`` writes them.
"""

from __future__ import annotations

import logging

import torch

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


def choice_for(rows: Rows, seq_len: int) -> Choice:
    """The row a layout whose longest query segment is ``seq_len`` tokens runs."""
    return RAGGED_ATTENTION.row_for(rows, seq_len)
